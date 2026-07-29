"""Tokenizer statistics computation and display.

Two reporters live here, for two different questions:

* ``StatisticsReporter`` answers "how good is this vocabulary?" — coverage,
  special-token ids, size against the corpus. It needs a trained
  ``Vocabulary``.
* ``CorpusStatisticsReporter`` answers "what does this corpus look like?" —
  token and sentence lengths, unique words, the frequency leaderboard. It
  needs only plain documents, so it can run before a vocabulary exists.

Both return frozen dataclasses that serialize cleanly to JSON, and both
render fixed-width tables that line up in a terminal or a log file.

Note on the module name: this file lives inside the ``visualization``
package on purpose. A top-level ``statistics.py`` would shadow the standard
library module of the same name and break every import of it in the project
— including ``utils.describe_numbers``.
"""

from __future__ import annotations

import logging
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import utils
from tokenizer.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

# Table drawing characters. Box-drawing glyphs render in every modern
# terminal and in monospaced log viewers, and unlike ASCII pipes and dashes
# they produce unbroken rules.
_BOX = {
    "tl": "┌", "tm": "┬", "tr": "┐",
    "ml": "├", "mm": "┼", "mr": "┤",
    "bl": "└", "bm": "┴", "br": "┘",
    "h": "─", "v": "│",
}

# Filled block for the inline proportion bars in the frequency table.
_BAR_GLYPH = "█"
_BAR_WIDTH = 22

# Longest token rendered in a table cell before it is elided. Keeps one
# pathological token from stretching the whole table off-screen.
_MAX_CELL_WIDTH = 28


@dataclass(frozen=True)
class TokenizerStatistics:
    """Aggregate statistics for a trained tokenizer.

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        corpus_documents: Number of documents used for training.
        total_tokens: Total token occurrences in the corpus.
        unique_corpus_tokens: Unique tokens observed before vocab clipping.
        coverage_ratio: Fraction of corpus token mass covered by the vocab
            (excluding special tokens).
        top_tokens: Most frequent tokens and their counts.
        special_token_ids: Mapping of special token -> id.
        unknown_token: Configured unknown token string.
        avg_document_length: Mean tokens per document.
    """

    vocab_size: int
    corpus_documents: int
    total_tokens: int
    unique_corpus_tokens: int
    coverage_ratio: float
    top_tokens: list[dict[str, Any]]
    special_token_ids: dict[str, int]
    unknown_token: str
    avg_document_length: float

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of all fields.
        """
        return asdict(self)


class StatisticsReporter:
    """Compute and display tokenizer training statistics.

    Called after vocabulary training to surface coverage, size, and
    frequency insights before artifacts are saved.
    """

    def __init__(self, vocabulary: Vocabulary, top_k: int = 10) -> None:
        """Initialize the reporter.

        Args:
            vocabulary: Trained vocabulary with frequency data.
            top_k: Number of top-frequency tokens to include in reports.
        """
        self._vocabulary = vocabulary
        self._top_k = top_k

    def compute(
        self,
        documents: list[str],
        special_tokens: list[str],
        unknown_token: str,
    ) -> TokenizerStatistics:
        """Compute tokenizer statistics from the trained vocabulary.

        Args:
            documents: Cleaned corpus documents used for training.
            special_tokens: List of reserved special tokens.
            unknown_token: Unknown-token string.

        Returns:
            Populated ``TokenizerStatistics`` dataclass.

        Raises:
            ValueError: If ``documents`` is empty.
            RuntimeError: If the vocabulary is not trained.
        """
        if not documents:
            raise ValueError("Cannot compute statistics on an empty corpus")

        frequencies = self._vocabulary.frequencies
        total_tokens = sum(frequencies.values())
        vocab_tokens = set(self._vocabulary.token_to_id) - set(special_tokens)
        covered = sum(
            count for token, count in frequencies.items() if token in vocab_tokens
        )
        coverage = (covered / total_tokens) if total_tokens else 0.0

        counter = Counter(frequencies)
        top_tokens = [
            {"token": token, "count": count}
            for token, count in counter.most_common(self._top_k)
        ]
        special_token_ids = {
            token: self._vocabulary.token_id(token) for token in special_tokens
        }
        avg_length = total_tokens / len(documents) if documents else 0.0

        stats = TokenizerStatistics(
            vocab_size=self._vocabulary.size,
            corpus_documents=len(documents),
            total_tokens=total_tokens,
            unique_corpus_tokens=len(frequencies),
            coverage_ratio=round(coverage, 6),
            top_tokens=top_tokens,
            special_token_ids=special_token_ids,
            unknown_token=unknown_token,
            avg_document_length=round(avg_length, 4),
        )
        logger.info(
            "Statistics: vocab=%d docs=%d coverage=%.2f%%",
            stats.vocab_size,
            stats.corpus_documents,
            stats.coverage_ratio * 100,
        )
        return stats

    def display(self, stats: TokenizerStatistics) -> None:
        """Print a human-readable statistics summary to the log.

        Args:
            stats: Computed tokenizer statistics.
        """
        lines = [
            "=" * 60,
            "Tokenizer Statistics",
            "=" * 60,
            f"Vocabulary size        : {stats.vocab_size}",
            f"Corpus documents       : {stats.corpus_documents}",
            f"Total tokens           : {stats.total_tokens}",
            f"Unique corpus tokens   : {stats.unique_corpus_tokens}",
            f"Vocabulary coverage    : {stats.coverage_ratio:.2%}",
            f"Avg document length    : {stats.avg_document_length:.2f} tokens",
            f"Unknown token          : {stats.unknown_token}",
            "Special token ids      :",
        ]
        for token, token_id in stats.special_token_ids.items():
            lines.append(f"  {token:>8} -> {token_id}")
        lines.append("Top tokens:")
        for item in stats.top_tokens:
            lines.append(f"  {item['token']!r:>20} : {item['count']}")
        lines.append("=" * 60)

        summary = "\n".join(lines)
        # Print once for operators; avoid also logging the full block
        # (logging handlers already mirror stdout in this project).
        print(summary)
        logger.debug("Displayed statistics summary (%d lines)", len(lines))


@dataclass(frozen=True)
class WordCount:
    """One row of the frequency leaderboard.

    Attributes:
        rank: 1-based position, most frequent first.
        token: The token string.
        count: Occurrences in the corpus.
        share: Percentage of all token occurrences, to two decimals.
    """

    rank: int
    token: str
    count: int
    share: float


@dataclass(frozen=True)
class CorpusStatistics:
    """Descriptive statistics for a tokenized corpus.

    Attributes:
        vocabulary_size: Size of the trained vocabulary including special
            tokens, or 0 when no vocabulary was supplied. Deliberately
            separate from ``unique_words``: the two diverge whenever a
            maximum size or minimum frequency filter is in play, and the
            gap between them is exactly what those settings cost.
        unique_words: Distinct tokens observed in the corpus.
        total_tokens: Total token occurrences.
        avg_token_length: Mean characters per token, weighted by frequency
            — the corpus as read.
        avg_unique_token_length: Mean characters per *distinct* token — the
            vocabulary as stored. Runs longer than the weighted mean,
            because short function words are common but few.
        max_token_length: Longest token in characters.
        min_token_length: Shortest token in characters.
        longest_token: An example of the longest token.
        shortest_token: An example of the shortest token.
        sentence_count: Number of documents.
        avg_sentence_length: Mean tokens per document.
        max_sentence_length: Longest document in tokens.
        min_sentence_length: Shortest document in tokens.
        median_sentence_length: Median tokens per document.
        top_words: The most frequent tokens, longest list first.
    """

    vocabulary_size: int
    unique_words: int
    total_tokens: int
    avg_token_length: float
    avg_unique_token_length: float
    max_token_length: int
    min_token_length: int
    longest_token: str
    shortest_token: str
    sentence_count: int
    avg_sentence_length: float
    max_sentence_length: int
    min_sentence_length: int
    median_sentence_length: float
    top_words: list[WordCount] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary.

        ``asdict`` recurses into the nested ``WordCount`` entries, so the
        result contains no dataclass instances and can be handed straight to
        ``utils.write_json``.

        Returns:
            Dictionary representation of every field.
        """
        return asdict(self)


class CorpusStatisticsReporter:
    """Compute and render descriptive statistics for a corpus.

    Takes plain documents rather than a ``Vocabulary``, so it can profile a
    corpus before any training has happened — which is when the numbers are
    most useful for choosing a vocabulary size and frequency threshold.
    """

    def __init__(self, top_k: int = 100) -> None:
        """Initialize the reporter.

        Args:
            top_k: How many entries to keep in the frequency leaderboard.

        Raises:
            ValueError: If ``top_k`` is less than 1.
        """
        self._top_k = utils.validate_positive_int(top_k, "top_k", minimum=1)

    def compute(
        self,
        documents: Iterable[str],
        *,
        frequencies: dict[str, int] | None = None,
        vocabulary_size: int = 0,
    ) -> CorpusStatistics:
        """Compute every statistic in one pass over the corpus.

        Args:
            documents: Cleaned document strings, one per sentence or line.
            frequencies: Optional precomputed token counts. Supply these when
                a trainer already counted them, to avoid a second pass over
                a large corpus; otherwise they are derived from
                ``documents``.
            vocabulary_size: Size of the trained vocabulary, if one exists.

        Returns:
            A populated ``CorpusStatistics``.

        Raises:
            ValueError: If the corpus is empty or contains no tokens.
        """
        sentence_lengths: list[int] = []
        derived: Counter[str] = Counter()

        for document in documents:
            tokens = document.split()
            if not tokens:
                # Blank lines are not sentences; counting them would drag the
                # mean sentence length toward zero for no reason.
                continue
            sentence_lengths.append(len(tokens))
            if frequencies is None:
                derived.update(tokens)

        utils.validate_not_empty(sentence_lengths, "documents")
        counts = frequencies if frequencies is not None else dict(derived)
        utils.validate_not_empty(counts, "token frequencies")

        # Token lengths, once weighted by occurrence and once per distinct
        # token. Both are reported because they answer different questions
        # and are routinely confused for one another.
        token_lengths = [len(token) for token in counts]
        weighted_total = sum(len(token) * count for token, count in counts.items())
        total_tokens = sum(counts.values())

        sentence_summary = utils.describe_numbers(sentence_lengths)

        # Ties broken lexicographically so the reported example is stable
        # across runs rather than whichever key the dict happened to yield.
        longest = min(
            (t for t in counts if len(t) == max(token_lengths)), key=lambda t: (t,)
        )
        shortest = min(
            (t for t in counts if len(t) == min(token_lengths)), key=lambda t: (t,)
        )

        stats = CorpusStatistics(
            vocabulary_size=vocabulary_size,
            unique_words=len(counts),
            total_tokens=total_tokens,
            avg_token_length=round(weighted_total / total_tokens, 4),
            avg_unique_token_length=round(sum(token_lengths) / len(token_lengths), 4),
            max_token_length=max(token_lengths),
            min_token_length=min(token_lengths),
            longest_token=longest,
            shortest_token=shortest,
            sentence_count=len(sentence_lengths),
            avg_sentence_length=round(sentence_summary["mean"], 4),
            max_sentence_length=int(sentence_summary["maximum"]),
            min_sentence_length=int(sentence_summary["minimum"]),
            median_sentence_length=sentence_summary["median"],
            top_words=self._rank_words(counts, total_tokens),
        )

        logger.info(
            "Corpus statistics: %d sentences, %d unique words, %d occurrences",
            stats.sentence_count,
            stats.unique_words,
            stats.total_tokens,
        )
        return stats

    def _rank_words(
        self, counts: dict[str, int], total_tokens: int
    ) -> list[WordCount]:
        """Build the frequency leaderboard.

        Args:
            counts: Token frequency mapping.
            total_tokens: Total occurrences, used for the share column.

        Returns:
            Up to ``top_k`` ``WordCount`` entries, most frequent first.
        """
        # Sort by descending count, then alphabetically. The secondary key
        # keeps the leaderboard stable when counts tie, which they do
        # constantly in the tail.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [
            WordCount(
                rank=index,
                token=token,
                count=count,
                share=utils.percentage(count, total_tokens),
            )
            for index, (token, count) in enumerate(ranked[: self._top_k], start=1)
        ]

    def render(self, stats: CorpusStatistics, *, top_display: int = 20) -> str:
        """Render the statistics as aligned tables.

        Args:
            stats: Computed statistics.
            top_display: How many leaderboard rows to include. The structured
                result still carries all ``top_k`` entries; this only bounds
                what is printed, since a 100-row table scrolls a terminal
                past the summary that gives it context.

        Returns:
            A multi-table string ready to print or log.
        """
        summary_rows = [
            ["Vocabulary size", _number(stats.vocabulary_size) if stats.vocabulary_size else "—"],
            ["Unique words", _number(stats.unique_words)],
            ["Total tokens", _number(stats.total_tokens)],
            ["Average token length", f"{stats.avg_token_length:.2f} chars"],
            ["Average unique token length", f"{stats.avg_unique_token_length:.2f} chars"],
            ["Maximum token length", f"{_plural(stats.max_token_length, 'char')} ({stats.longest_token!r})"],
            ["Minimum token length", f"{_plural(stats.min_token_length, 'char')} ({stats.shortest_token!r})"],
            ["Sentences", _number(stats.sentence_count)],
            ["Average sentence length", f"{stats.avg_sentence_length:.2f} tokens"],
            ["Median sentence length", _plural(int(stats.median_sentence_length), "token")],
            ["Longest sentence", _plural(stats.max_sentence_length, "token")],
            ["Shortest sentence", _plural(stats.min_sentence_length, "token")],
        ]

        shown = stats.top_words[:top_display]
        peak = shown[0].count if shown else 0
        word_rows = [
            [
                str(entry.rank),
                _elide(entry.token),
                _number(entry.count),
                f"{entry.share:.2f}%",
                _bar(entry.count, peak),
            ]
            for entry in shown
        ]

        blocks = [
            render_table(
                headers=["Metric", "Value"],
                rows=summary_rows,
                alignments=["left", "right"],
                title="Corpus statistics",
            ),
            render_table(
                headers=["#", "Token", "Count", "Share", ""],
                rows=word_rows,
                alignments=["right", "left", "right", "right", "left"],
                title=f"Top words — showing {len(shown)} of {len(stats.top_words)}",
            ),
        ]
        return "\n\n".join(blocks)

    def display(self, stats: CorpusStatistics, *, top_display: int = 20) -> None:
        """Print the rendered tables to stdout.

        Printed rather than logged: the tables are wide, and log handlers in
        this project prefix every line with a timestamp and level, which
        would break the column alignment that makes them readable.

        Args:
            stats: Computed statistics.
            top_display: How many leaderboard rows to print.
        """
        print(self.render(stats, top_display=top_display))
        logger.debug("Displayed corpus statistics tables")


# =============================================================================
# Table rendering
# =============================================================================


def render_table(
    *,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    alignments: Sequence[str] | None = None,
    title: str | None = None,
) -> str:
    """Render rows as a box-drawn, column-aligned table.

    Column widths are measured in *display* cells rather than characters, so
    a table containing CJK or emoji tokens still lines up — those glyphs
    occupy two terminal columns each while counting as one character.

    Args:
        headers: Column headers.
        rows: Row values. Every row must match the header count.
        alignments: Per-column ``"left"`` or ``"right"``. Defaults to all
            left. Numeric columns should be right-aligned so digits line up.
        title: Optional caption printed above the table.

    Returns:
        The rendered table as a string, without a trailing newline.

    Raises:
        ValueError: If a row's length does not match the header count.
    """
    aligns = list(alignments) if alignments else ["left"] * len(headers)
    if len(aligns) != len(headers):
        raise ValueError("alignments must match the number of headers")

    for index, row in enumerate(rows):
        if len(row) != len(headers):
            raise ValueError(
                f"Row {index} has {len(row)} cells, expected {len(headers)}"
            )

    widths = [_display_width(header) for header in headers]
    for row in rows:
        for column, cell in enumerate(row):
            widths[column] = max(widths[column], _display_width(cell))

    def rule(left: str, middle: str, right: str) -> str:
        return left + middle.join(_BOX["h"] * (w + 2) for w in widths) + right

    def line(cells: Sequence[str]) -> str:
        padded = [
            _pad(cell, widths[column], aligns[column])
            for column, cell in enumerate(cells)
        ]
        return _BOX["v"] + " " + f' {_BOX["v"]} '.join(padded) + " " + _BOX["v"]

    parts: list[str] = []
    if title:
        parts.append(title)
    parts.append(rule(_BOX["tl"], _BOX["tm"], _BOX["tr"]))
    parts.append(line(headers))
    parts.append(rule(_BOX["ml"], _BOX["mm"], _BOX["mr"]))
    parts.extend(line(row) for row in rows)
    parts.append(rule(_BOX["bl"], _BOX["bm"], _BOX["br"]))
    return "\n".join(parts)


def _display_width(text: str) -> int:
    """Measure how many terminal columns a string occupies.

    Args:
        text: String to measure.

    Returns:
        Column count. East Asian wide and fullwidth characters count as two;
        zero-width combining marks count as none.
    """
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def _pad(text: str, width: int, alignment: str) -> str:
    """Pad a cell to a target display width.

    Uses the display width rather than ``str.ljust``, which pads by character
    count and therefore misaligns any row containing a wide glyph.

    Args:
        text: Cell contents.
        width: Target display width.
        alignment: ``"left"`` or ``"right"``.

    Returns:
        The padded string.
    """
    padding = " " * max(0, width - _display_width(text))
    return padding + text if alignment == "right" else text + padding


def _elide(text: str, limit: int = _MAX_CELL_WIDTH) -> str:
    """Shorten an over-long cell value with an ellipsis.

    Args:
        text: Cell contents.
        limit: Maximum display width.

    Returns:
        The original string, or a truncated one ending in a single-character
        ellipsis so the result never exceeds ``limit``.
    """
    if _display_width(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _number(value: int) -> str:
    """Format an integer with thousands separators.

    Args:
        value: Integer to format.

    Returns:
        The formatted string, e.g. ``"12,345"``.
    """
    return f"{value:,}"


def _plural(value: int, noun: str) -> str:
    """Format a count with a correctly pluralized noun.

    Args:
        value: The count.
        noun: Singular form of the noun.

    Returns:
        A string such as ``"1 char"`` or ``"16 chars"``. Only regular
        ``-s`` plurals are needed here; the nouns are fixed and known.
    """
    return f"{_number(value)} {noun}" if value == 1 else f"{_number(value)} {noun}s"


def _bar(value: int, peak: int, width: int = _BAR_WIDTH) -> str:
    """Render a proportional bar for a table cell.

    Gives the frequency table the shape of the distribution alongside the
    exact counts, so the long tail is visible without reading every number.

    Args:
        value: This row's magnitude.
        peak: The largest magnitude in the table.
        width: Bar width in characters at full scale.

    Returns:
        A string of block characters, at least one wide for any non-zero
        value so a small-but-present count never renders as blank.
    """
    if peak <= 0 or value <= 0:
        return ""
    filled = max(1, round(width * value / peak))
    return _BAR_GLYPH * filled
