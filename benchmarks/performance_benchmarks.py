"""Performance benchmark suite for tokenizer evaluation.

This module measures:

- Encoding speed (tokens/sec)
- Peak memory usage during encoding
- OOV rate and vocabulary coverage
- Compression ratio before/after BPE-style encoding
- Markdown comparison table across tokenizer variants
"""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from analysis.tokenizer_analysis import (
    build_tokenizer_comparison_table,
    compression_ratio_before_after,
    compute_oov_rate,
)
from configs.config_loader import ConfigLoader
from preprocessing.dataset_loader import DatasetLoader
from preprocessing.text_cleaner import TextCleaner
from tokenizer import Tokenizer


@dataclass(frozen=True)
class BenchmarkResult:
    """Holds benchmark metrics for one tokenizer implementation.

    Attributes:
        name: Display name of tokenizer.
        vocab_size: Number of tokens in vocabulary (0 for stateless tokenizers).
        oov_rate: Out-of-vocabulary token ratio.
        tokens_per_sec: Encoding throughput.
        memory_kb: Peak memory during encoding in kilobytes.
        compression_ratio: Ratio of tokens-before to tokens-after.
    """

    name: str
    vocab_size: int
    oov_rate: float
    tokens_per_sec: float
    memory_kb: float
    compression_ratio: float


class CharacterTokenizer:
    """Simple character-level tokenizer for baseline comparisons."""

    def encode(self, text: str) -> list[str]:
        """Encode text into characters.

        Args:
            text: Input text string.

        Returns:
            List of character tokens excluding spaces.
        """
        return [char for char in text if not char.isspace()]


class GreedySubwordTokenizer:
    """Simple BPE-style tokenizer using greedy longest-substring matching.

    This is not a full BPE trainer; it approximates post-merge behavior by
    greedily segmenting words using an existing vocabulary.
    """

    def __init__(self, vocabulary: set[str]) -> None:
        """Initialize with an existing token vocabulary.

        Args:
            vocabulary: Known tokens for segmentation.
        """
        self._vocabulary = vocabulary

    def encode(self, text: str) -> list[str]:
        """Encode text into greedy subword segments.

        Args:
            text: Input text.

        Returns:
            Subword token list.
        """
        output: list[str] = []
        for word in text.split():
            output.extend(self._segment_word(word))
        return output

    def _segment_word(self, word: str) -> list[str]:
        """Segment one word with longest-prefix matching.

        Args:
            word: Input word string.

        Returns:
            Segmented subword pieces.
        """
        pieces: list[str] = []
        cursor = 0
        while cursor < len(word):
            matched = None
            for end in range(len(word), cursor, -1):
                candidate = word[cursor:end]
                if candidate in self._vocabulary:
                    matched = candidate
                    break
            if matched is None:
                matched = word[cursor : cursor + 1]
            pieces.append(matched)
            cursor += len(matched)
        return pieces


def _measure_encoding_speed_and_memory(
    encoded_token_count: int,
    fn: Callable[[], object],
) -> tuple[float, float]:
    """Measure encoding speed and peak memory for a callable.

    Args:
        encoded_token_count: Total number of tokens produced.
        fn: Callable that performs encoding work.

    Returns:
        Tuple of ``(tokens_per_sec, peak_memory_kb)``.
    """
    tracemalloc.start()
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tokens_per_sec = encoded_token_count / elapsed if elapsed > 0 else 0.0
    return round(tokens_per_sec, 2), round(peak / 1024, 2)


def run_benchmarks(
    *,
    config_path: Path = Path("configs/default.yaml"),
    output_json: Path = Path("outputs/performance_benchmarks.json"),
    output_markdown: Path = Path("outputs/tokenizer_comparison_table.md"),
) -> list[BenchmarkResult]:
    """Run tokenizer performance and quality benchmarks.

    Args:
        config_path: Path to project YAML config.
        output_json: JSON destination for benchmark metrics.
        output_markdown: Markdown destination for tokenizer table.

    Returns:
        List of ``BenchmarkResult`` entries.
    """
    config = ConfigLoader(config_path).load()
    documents = DatasetLoader(config.paths.dataset).load()
    cleaner = TextCleaner(config.preprocessing)
    cleaned = cleaner.clean_corpus(documents)
    corpus_text = " ".join(cleaned)

    tokenizer = Tokenizer(config.tokenizer)
    tokenizer.train(cleaned)
    vocab_tokens = set(tokenizer.vocabulary.token_to_id.keys())

    # Word-level (project tokenizer)
    word_sequences = [tokenizer.encode(doc, add_special_tokens=False) for doc in cleaned]
    word_token_count = sum(len(seq) for seq in word_sequences)
    word_speed, word_memory = _measure_encoding_speed_and_memory(
        word_token_count,
        lambda: [tokenizer.encode(doc, add_special_tokens=False) for doc in cleaned],
    )
    word_oov = compute_oov_rate(cleaned, vocab_tokens)

    # Character baseline
    char_tokenizer = CharacterTokenizer()
    char_sequences = [char_tokenizer.encode(doc) for doc in cleaned]
    char_token_count = sum(len(seq) for seq in char_sequences)
    char_speed, char_memory = _measure_encoding_speed_and_memory(
        char_token_count,
        lambda: [char_tokenizer.encode(doc) for doc in cleaned],
    )

    # BPE-like greedy subword over the existing vocab.
    bpe_like = GreedySubwordTokenizer(vocab_tokens)
    bpe_sequences = [bpe_like.encode(doc) for doc in cleaned]
    bpe_token_count = sum(len(seq) for seq in bpe_sequences)
    bpe_speed, bpe_memory = _measure_encoding_speed_and_memory(
        bpe_token_count,
        lambda: [bpe_like.encode(doc) for doc in cleaned],
    )
    bpe_oov = compute_oov_rate(cleaned, vocab_tokens)

    word_compression = compression_ratio_before_after(char_token_count, word_token_count)
    bpe_compression = compression_ratio_before_after(char_token_count, bpe_token_count)

    results = [
        BenchmarkResult(
            name="Word-Level",
            vocab_size=tokenizer.vocabulary.size,
            oov_rate=word_oov,
            tokens_per_sec=word_speed,
            memory_kb=word_memory,
            compression_ratio=word_compression,
        ),
        BenchmarkResult(
            name="Character-Level (Baseline)",
            vocab_size=0,
            oov_rate=0.0,
            tokens_per_sec=char_speed,
            memory_kb=char_memory,
            compression_ratio=1.0,
        ),
        BenchmarkResult(
            name="Greedy Subword (BPE-like)",
            vocab_size=tokenizer.vocabulary.size,
            oov_rate=bpe_oov,
            tokens_per_sec=bpe_speed,
            memory_kb=bpe_memory,
            compression_ratio=bpe_compression,
        ),
    ]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "config_path": str(config_path),
            "document_count": len(cleaned),
            "corpus_characters": len(corpus_text),
        },
        "results": [result.__dict__ for result in results],
    }
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    markdown_rows = [
        {
            "name": result.name,
            "vocab_size": result.vocab_size,
            "oov_rate": result.oov_rate,
            "tokens_per_sec": result.tokens_per_sec,
            "memory_kb": result.memory_kb,
            "compression_ratio": result.compression_ratio,
        }
        for result in results
    ]
    table = build_tokenizer_comparison_table(markdown_rows)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(table + "\n", encoding="utf-8")

    return results


if __name__ == "__main__":
    run_benchmarks()
