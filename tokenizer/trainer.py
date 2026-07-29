"""Streaming vocabulary trainer for large multi-file UTF-8 corpora.

This module trains a word-level vocabulary without ever holding the corpus
in memory. It complements ``tokenizer.vocabulary.Vocabulary``, which trains
from an in-memory ``list[str]`` and is the right choice for small datasets;
``VocabularyTrainer`` is the right choice when the corpus is larger than RAM
or split across many files.

The exported ``vocab.json`` uses the same schema that
``Vocabulary.load()`` reads, so a streamed vocabulary is a drop-in
replacement for an in-memory one.

No external tokenizer libraries are used. Splitting is delegated to
``algorithms.word_level.WordLevelTokenizer``; ``tqdm`` is used only to draw
a progress bar.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from tqdm import tqdm

from algorithms.word_level import WordLevelTokenizer
from configs.config_loader import TokenizerConfig
from preprocessing.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)

# Corpus encoding. UTF-8 is assumed throughout; the streamer decodes manually
# rather than relying on the platform default, which varies by OS and locale.
_DEFAULT_ENCODING = "utf-8"

# UTF-8 byte-order mark. Some Windows-authored .txt files start with it, and
# left in place it would corrupt the very first token of the file.
_BOM = "﻿"

# Unicode replacement character, inserted by the non-strict decode policies.
_REPLACEMENT_CHAR = "�"

# Fraction of ``max_tracked_tokens`` retained by a pruning sweep. The gap
# between this and 1.0 is the headroom that keeps sweeps from firing on every
# subsequent insertion; at 0.8 a sweep is separated from the next by roughly
# a fifth of the ceiling in newly seen tokens.
_PRUNE_RETENTION = 0.8


@dataclass(frozen=True)
class TrainingStats:
    """Summary of a single streaming training run.

    Attributes:
        files_processed: Number of corpus files successfully read.
        lines_processed: Number of non-empty lines consumed across all files.
        bytes_read: Total bytes streamed from disk.
        total_tokens: Total token occurrences counted (corpus length).
        unique_tokens: Distinct tokens still tracked when counting finished.
        tokens_above_min_frequency: Distinct tokens meeting ``min_frequency``.
        vocab_size: Final vocabulary size, including special tokens.
        truncated: True if ``max_vocab_size`` cut off eligible tokens.
        pruned_tokens: Rare tokens discarded by memory-guard pruning. Non-zero
            means the reported counts are approximate; see
            ``VocabularyTrainer`` for the exact caveat.
        decode_errors: Lines containing bytes that were not valid UTF-8.
    """

    files_processed: int
    lines_processed: int
    bytes_read: int
    total_tokens: int
    unique_tokens: int
    tokens_above_min_frequency: int
    vocab_size: int
    truncated: bool
    pruned_tokens: int
    decode_errors: int


class CorpusStreamer:
    """Discover corpus files and stream them line by line as UTF-8 text.

    Called by ``VocabularyTrainer`` to isolate all filesystem and decoding
    concerns from the counting logic. Files are opened in binary mode and
    decoded manually so that byte-accurate progress can be reported and the
    decode-error policy stays explicit.

    Splitting binary input on newline is safe for UTF-8: every byte of a
    multi-byte sequence has its high bit set, so 0x0A can never appear
    inside one. A line therefore never splits a character in half.
    """

    def __init__(
        self,
        sources: str | Path | Iterable[str | Path],
        *,
        pattern: str = "*.txt",
        recursive: bool = True,
        encoding: str = _DEFAULT_ENCODING,
        errors: str = "strict",
    ) -> None:
        """Initialize the streamer.

        Args:
            sources: A file path, a directory path, or an iterable of either.
                Directories are expanded using ``pattern``.
            pattern: Glob pattern used when expanding directories.
            recursive: Whether directory expansion descends into subdirectories.
            encoding: Text encoding of the corpus files.
            errors: Decode error policy — ``"strict"`` raises on malformed
                bytes, ``"replace"`` substitutes U+FFFD, ``"ignore"`` drops them.

        Raises:
            ValueError: If ``sources`` is empty.
        """
        # Accept a bare path as a convenience, but treat str/Path as scalar
        # rather than as an iterable of characters.
        if isinstance(sources, (str, Path)):
            raw_sources: list[str | Path] = [sources]
        else:
            raw_sources = list(sources)
        if not raw_sources:
            raise ValueError("At least one corpus source must be provided")

        self._sources = raw_sources
        self._pattern = pattern
        self._recursive = recursive
        self._encoding = encoding
        self._errors = errors
        self._decode_errors = 0

    @property
    def decode_errors(self) -> int:
        """Return the number of lines that contained undecodable bytes.

        Returns:
            Count of lines where the decoder had to replace or drop bytes.
            Always 0 when ``errors="strict"``, since those runs raise instead.
        """
        return self._decode_errors

    def discover(self) -> list[Path]:
        """Resolve the configured sources into a sorted list of files.

        Returns:
            Absolute file paths, sorted for run-to-run determinism. Sorting
            matters because tie-breaking during vocabulary selection depends
            on counts, and counts must not vary with filesystem ordering.

        Raises:
            FileNotFoundError: If a source does not exist, or if a directory
                source contains no file matching ``pattern``.
            ValueError: If discovery yields no files at all.
        """
        discovered: list[Path] = []
        for source in self._sources:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Corpus source not found: {path}")

            if path.is_dir():
                globber = path.rglob if self._recursive else path.glob
                matches = sorted(p for p in globber(self._pattern) if p.is_file())
                if not matches:
                    raise FileNotFoundError(
                        f"No files matching '{self._pattern}' under directory: {path}"
                    )
                discovered.extend(matches)
            else:
                discovered.append(path)

        # De-duplicate while keeping determinism: the same file may be named
        # both directly and via a parent directory.
        unique = sorted({p.resolve() for p in discovered})
        if not unique:
            raise ValueError("Corpus discovery produced no files")

        logger.info("Discovered %d corpus file(s)", len(unique))
        return unique

    def total_bytes(self, files: Sequence[Path]) -> int:
        """Sum the on-disk size of the given files.

        Used to size the progress bar up front, which is cheap (one ``stat``
        per file) and avoids a full pre-pass over the corpus.

        Args:
            files: Files that will be streamed.

        Returns:
            Total size in bytes. Files that cannot be stat'ed count as 0.
        """
        total = 0
        for path in files:
            try:
                total += path.stat().st_size
            except OSError:
                logger.warning("Could not stat %s; progress may be imprecise", path)
        return total

    def stream(self, path: Path) -> Iterator[tuple[str, int]]:
        """Stream one file as decoded lines paired with their byte length.

        The file handle is opened in binary mode and iterated lazily, so peak
        memory is one line, not one file.

        Args:
            path: File to read.

        Yields:
            ``(text, byte_length)`` for each line. ``text`` is stripped of
            surrounding whitespace (which also normalizes CRLF line endings);
            ``byte_length`` reflects the raw bytes consumed, so progress stays
            accurate even for lines that strip down to nothing.

        Raises:
            OSError: If the file cannot be opened or read.
            UnicodeDecodeError: If ``errors="strict"`` and a line is malformed.
        """
        with path.open("rb") as handle:
            first_line = True
            for raw_line in handle:
                byte_length = len(raw_line)
                try:
                    text = raw_line.decode(self._encoding, errors=self._errors)
                except UnicodeDecodeError:
                    logger.error("Malformed %s in %s", self._encoding, path)
                    raise

                # Count lines whose bytes were not clean UTF-8. Detected by
                # the presence of the replacement character, which only the
                # non-strict policies can introduce.
                if self._errors != "strict" and _REPLACEMENT_CHAR in text:
                    self._decode_errors += 1

                # A BOM is only meaningful at offset 0; strip it there so it
                # does not become part of the first token.
                if first_line:
                    text = text.lstrip(_BOM)
                    first_line = False

                yield text.strip(), byte_length


class VocabularyTrainer:
    """Train a frequency-ranked vocabulary by streaming a corpus from disk.

    Called instead of ``Vocabulary.train()`` when the corpus is too large to
    load into memory or is spread across multiple files. The pipeline is:

        discover files -> stream lines -> clean -> tokenize -> count
                       -> filter by min frequency -> cap at max size
                       -> export vocab.json

    Only the frequency table is held in memory, and its size is bounded by
    the number of *distinct* tokens rather than by corpus length. For corpora
    whose distinct-token count is itself too large, ``max_tracked_tokens``
    enables periodic pruning of rare tokens.

    Pruning caveat: a pruned token restarts from zero if it reappears later,
    so its final count may be understated and it may be missed by the
    ``min_frequency`` filter. Counts are exact whenever pruning never fires
    (``pruned_tokens == 0`` in the returned ``TrainingStats``). Leave
    ``max_tracked_tokens`` at ``None`` — the default — for exact counting.
    """

    def __init__(
        self,
        config: TokenizerConfig,
        *,
        cleaner: TextCleaner | None = None,
        min_token_length: int = 1,
        max_tracked_tokens: int | None = None,
        encoding: str = _DEFAULT_ENCODING,
        errors: str = "strict",
        show_progress: bool = True,
    ) -> None:
        """Initialize the trainer.

        Args:
            config: Tokenizer configuration supplying ``vocab_size``,
                ``min_frequency``, ``special_tokens``, and ``unknown_token``.
            cleaner: Optional text cleaner applied to every line before
                tokenization. When ``None``, lines are tokenized as-is.
            min_token_length: Tokens shorter than this are never counted.
            max_tracked_tokens: Soft ceiling on distinct tracked tokens. When
                exceeded, the rarest tokens are pruned. ``None`` disables
                pruning and keeps counts exact.
            encoding: Corpus text encoding.
            errors: Decode error policy passed to the streamer.
            show_progress: Whether to draw a tqdm progress bar.

        Raises:
            ValueError: If ``min_token_length`` or ``max_tracked_tokens`` is
                out of range.
        """
        if min_token_length < 1:
            raise ValueError("min_token_length must be >= 1")
        if max_tracked_tokens is not None and max_tracked_tokens < config.vocab_size:
            # Pruning below the target vocabulary size would discard tokens
            # the vocabulary still needs, which is never what the caller wants.
            raise ValueError(
                f"max_tracked_tokens ({max_tracked_tokens}) must be >= "
                f"vocab_size ({config.vocab_size})"
            )

        self._config = config
        self._cleaner = cleaner
        self._min_token_length = min_token_length
        self._max_tracked_tokens = max_tracked_tokens
        self._encoding = encoding
        self._errors = errors
        self._show_progress = show_progress

        self._algorithm = WordLevelTokenizer()
        self._frequencies: Counter[str] = Counter()
        self._token_to_id: dict[str, int] = {}
        self._stats: TrainingStats | None = None

        # Running totals kept separately from the Counter, because pruning
        # removes entries and would otherwise corrupt the corpus-length total.
        self._total_tokens = 0
        self._pruned_tokens = 0

    @property
    def size(self) -> int:
        """Return the number of tokens in the trained vocabulary.

        Returns:
            Vocabulary size including special tokens; 0 before training.
        """
        return len(self._token_to_id)

    @property
    def token_to_id(self) -> dict[str, int]:
        """Return a copy of the token-to-id mapping.

        Returns:
            Mapping from token string to integer id.
        """
        return dict(self._token_to_id)

    @property
    def frequencies(self) -> dict[str, int]:
        """Return a copy of the tracked corpus frequencies.

        Returns:
            Mapping from token to occurrence count. Includes tokens that did
            not make the final vocabulary.
        """
        return dict(self._frequencies)

    @property
    def stats(self) -> TrainingStats:
        """Return statistics from the last training run.

        Returns:
            The ``TrainingStats`` produced by ``train()``.

        Raises:
            RuntimeError: If called before ``train()``.
        """
        if self._stats is None:
            raise RuntimeError("Trainer has not been run yet")
        return self._stats

    def train(
        self,
        sources: str | Path | Iterable[str | Path],
        *,
        pattern: str = "*.txt",
        recursive: bool = True,
    ) -> TrainingStats:
        """Stream the corpus and build the vocabulary.

        Args:
            sources: File path, directory path, or iterable of either.
            pattern: Glob used to expand directory sources.
            recursive: Whether directory expansion is recursive.

        Returns:
            A ``TrainingStats`` summary of the run.

        Raises:
            RuntimeError: If the trainer has already been run.
            FileNotFoundError: If a source path does not exist.
            ValueError: If the corpus yields no tokens, or if no token meets
                the minimum frequency threshold.
            OSError: If a corpus file cannot be read.
        """
        if self._token_to_id:
            raise RuntimeError("Trainer has already been run; create a new instance")

        streamer = CorpusStreamer(
            sources,
            pattern=pattern,
            recursive=recursive,
            encoding=self._encoding,
            errors=self._errors,
        )
        files = streamer.discover()
        total_bytes = streamer.total_bytes(files)

        logger.info(
            "Streaming %d file(s) (%.2f MiB) for vocabulary training",
            len(files),
            total_bytes / (1024 * 1024),
        )

        lines_processed, bytes_read = self._count_corpus(streamer, files, total_bytes)

        if not self._frequencies:
            raise ValueError("Corpus produced no tokens; cannot train a vocabulary")

        eligible = self._build_vocabulary()

        self._stats = TrainingStats(
            files_processed=len(files),
            lines_processed=lines_processed,
            bytes_read=bytes_read,
            total_tokens=self._total_tokens,
            unique_tokens=len(self._frequencies),
            tokens_above_min_frequency=eligible,
            vocab_size=self.size,
            # Truncation happened if more tokens qualified than there were
            # free slots after reserving space for the special tokens.
            truncated=eligible > self.size - len(self._config.special_tokens),
            pruned_tokens=self._pruned_tokens,
            decode_errors=streamer.decode_errors,
        )
        logger.info(
            "Training complete: %d tokens from %d occurrences across %d file(s)",
            self._stats.vocab_size,
            self._stats.total_tokens,
            self._stats.files_processed,
        )
        return self._stats

    def save(self, path: Path) -> None:
        """Export the trained vocabulary to a JSON file.

        The payload matches the schema read by ``Vocabulary.load()``, so the
        exported file can be consumed by the existing encoder and decoder.
        Only frequencies for tokens *in* the vocabulary are written, which
        keeps the file size bounded by ``vocab_size`` rather than by the
        number of distinct tokens in the corpus.

        The write is atomic: content goes to a temporary file in the same
        directory and is then renamed over the destination, so an interrupted
        run cannot leave a half-written vocabulary behind.

        Args:
            path: Destination JSON path. Parent directories are created.

        Raises:
            RuntimeError: If called before ``train()``.
            OSError: If the file cannot be written.
        """
        if not self._token_to_id:
            raise RuntimeError("Trainer has not been run yet")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "token_to_id": self._token_to_id,
            "frequencies": {
                token: int(self._frequencies.get(token, 0))
                for token in self._token_to_id
            },
            "special_tokens": list(self._config.special_tokens),
            "unknown_token": self._config.unknown_token,
            "size": self.size,
            # Extra provenance for reproducibility. Vocabulary.load() ignores
            # unknown keys, so this stays backward compatible.
            "training": asdict(self.stats),
            "config": {
                "vocab_size": self._config.vocab_size,
                "min_frequency": self._config.min_frequency,
                "min_token_length": self._min_token_length,
            },
        }

        temp_path = path.with_name(f"{path.name}.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                # ensure_ascii=False keeps non-Latin scripts readable in the
                # exported file instead of expanding them to \uXXXX escapes.
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            os.replace(temp_path, path)
        except OSError:
            logger.exception("Failed to write vocabulary to %s", path)
            temp_path.unlink(missing_ok=True)
            raise

        logger.info("Saved vocabulary (%d tokens) to %s", self.size, path)

    def _count_corpus(
        self,
        streamer: CorpusStreamer,
        files: Sequence[Path],
        total_bytes: int,
    ) -> tuple[int, int]:
        """Stream every file and accumulate token frequencies.

        Args:
            streamer: Configured corpus streamer.
            files: Files to read, in order.
            total_bytes: Pre-computed total size, used to size the progress bar.

        Returns:
            ``(lines_processed, bytes_read)``.

        Raises:
            OSError: If a file cannot be read.
        """
        lines_processed = 0
        bytes_read = 0

        # A single bar spanning the whole corpus, measured in bytes. Bytes are
        # the only unit known in advance without pre-reading the corpus, and
        # they stay meaningful across files of very different line lengths.
        progress = tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Training vocabulary",
            disable=not self._show_progress,
        )
        try:
            for path in files:
                progress.set_postfix_str(path.name, refresh=False)
                for text, byte_length in streamer.stream(path):
                    bytes_read += byte_length
                    progress.update(byte_length)

                    if not text:
                        continue
                    lines_processed += 1
                    self._consume_line(text)
        finally:
            # Close in a finally block so an I/O error mid-corpus does not
            # leave a stale bar rendered over the traceback.
            progress.close()

        return lines_processed, bytes_read

    def _consume_line(self, text: str) -> None:
        """Clean, tokenize, and count a single line of corpus text.

        Args:
            text: One stripped line of decoded corpus text.
        """
        if self._cleaner is not None:
            text = self._cleaner.clean(text)
            if not text:
                return

        tokens = self._algorithm.tokenize(text)
        if self._min_token_length > 1:
            tokens = [t for t in tokens if len(t) >= self._min_token_length]
        if not tokens:
            return

        self._total_tokens += len(tokens)
        self._frequencies.update(tokens)

        # Check the memory guard once per line rather than once per token;
        # len() on a Counter is O(1), so this stays cheap.
        if self._max_tracked_tokens is not None:
            if len(self._frequencies) > self._max_tracked_tokens:
                self._prune_rare_tokens()

    def _prune_rare_tokens(self) -> None:
        """Drop the rarest tracked tokens to bound memory use.

        Keeps the highest-count tokens and discards the tail. Ranking by
        count (ties broken lexicographically) rather than by a count
        threshold matters for two reasons: it removes exactly as many tokens
        as needed instead of everything below some cutoff, and it can never
        discard a token that outranks one it keeps.

        The table is trimmed below the ceiling rather than exactly to it, so
        that the next sweep is separated by a whole batch of insertions.
        Without that headroom, every subsequent token would trigger another
        full ranking pass.
        """
        cap = self._max_tracked_tokens
        if cap is None:
            return

        # Never trim below the target vocabulary size — those tokens are the
        # whole point of the run.
        keep = max(self._config.vocab_size, int(cap * _PRUNE_RETENTION))
        if keep >= len(self._frequencies):
            return

        before = len(self._frequencies)
        ranked = sorted(self._frequencies.items(), key=lambda item: (-item[1], item[0]))
        survivors = ranked[:keep]
        cutoff = survivors[-1][1]

        self._frequencies = Counter(dict(survivors))
        removed = before - len(self._frequencies)
        self._pruned_tokens += removed
        logger.warning(
            "Pruned %d token(s) ranked below the top %d (lowest surviving "
            "count: %d) to respect max_tracked_tokens; counts for rare tokens "
            "are now approximate",
            removed,
            len(self._frequencies),
            cutoff,
        )

    def _build_vocabulary(self) -> int:
        """Select final vocabulary entries from the frequency table.

        Special tokens are assigned the lowest ids so they keep stable
        positions, then remaining slots are filled by descending frequency.
        Ties break lexicographically so repeated runs over the same corpus
        produce byte-identical vocabularies.

        Returns:
            The number of distinct tokens that met ``min_frequency``, before
            the ``vocab_size`` cap was applied. Comparing this to the final
            size tells the caller whether truncation occurred.

        Raises:
            ValueError: If no token met the minimum frequency threshold.
        """
        for token in self._config.special_tokens:
            self._add_token(token)

        candidates = [
            (token, count)
            for token, count in self._frequencies.items()
            if count >= self._config.min_frequency and token not in self._token_to_id
        ]
        eligible = len(candidates)

        if not candidates:
            raise ValueError(
                f"No tokens met min_frequency={self._config.min_frequency}; "
                "lower the threshold or supply a larger corpus"
            )

        remaining_slots = self._config.vocab_size - len(self._token_to_id)
        if remaining_slots <= 0:
            logger.warning(
                "vocab_size (%d) is fully consumed by %d special tokens; "
                "no corpus tokens were added",
                self._config.vocab_size,
                len(self._config.special_tokens),
            )
            return eligible

        candidates.sort(key=lambda item: (-item[1], item[0]))
        for token, _count in candidates[:remaining_slots]:
            self._add_token(token)

        if eligible > remaining_slots:
            logger.info(
                "Truncated vocabulary: %d tokens met min_frequency but only "
                "%d slot(s) were available",
                eligible,
                remaining_slots,
            )
        return eligible

    def _add_token(self, token: str) -> None:
        """Register a token, assigning it the next sequential id.

        Args:
            token: Token string to insert. Re-inserting is a no-op, which
                keeps ids stable when a special token also occurs in the
                corpus.
        """
        if token in self._token_to_id:
            return
        self._token_to_id[token] = len(self._token_to_id)
