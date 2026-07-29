"""Analysis helpers for tokenizer quality and efficiency metrics.

This module centralizes production-facing tokenizer diagnostics:

- vocabulary coverage analysis
- out-of-vocabulary (OOV) rate
- compression ratio before/after tokenization
- token-length distribution summaries
- tokenizer comparison tables
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Protocol

from algorithms.word_level import WordLevelTokenizer


class EncodableTokenizer(Protocol):
    """Protocol describing the minimal tokenizer API for analysis."""

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode one text sample into integer ids."""


@dataclass(frozen=True)
class VocabularyCoverage:
    """Coverage statistics over a corpus.

    Attributes:
        total_tokens: Total token occurrences in the corpus.
        in_vocabulary_tokens: Token occurrences present in vocabulary.
        out_of_vocabulary_tokens: Token occurrences missing from vocabulary.
        coverage_rate: Fraction of in-vocabulary token occurrences.
    """

    total_tokens: int
    in_vocabulary_tokens: int
    out_of_vocabulary_tokens: int
    coverage_rate: float


@dataclass(frozen=True)
class TokenLengthDistribution:
    """Summary statistics of token lengths in characters.

    Attributes:
        count: Number of token samples.
        minimum: Shortest token length.
        maximum: Longest token length.
        mean: Mean token length.
        median: Median token length.
    """

    count: int
    minimum: int
    maximum: int
    mean: float
    median: float


def vocabulary_coverage_analysis(
    documents: list[str],
    vocabulary_tokens: set[str],
) -> VocabularyCoverage:
    """Compute vocabulary coverage against a document corpus.

    Args:
        documents: Cleaned documents to evaluate.
        vocabulary_tokens: Set of known vocabulary tokens.

    Returns:
        Coverage metrics as a ``VocabularyCoverage`` dataclass.

    Raises:
        ValueError: If ``documents`` is empty.
    """
    if not documents:
        raise ValueError("documents must not be empty")

    tokenizer = WordLevelTokenizer()
    total = 0
    in_vocab = 0
    for document in documents:
        for token in tokenizer.tokenize(document):
            total += 1
            if token in vocabulary_tokens:
                in_vocab += 1

    oov = total - in_vocab
    rate = (in_vocab / total) if total else 0.0
    return VocabularyCoverage(
        total_tokens=total,
        in_vocabulary_tokens=in_vocab,
        out_of_vocabulary_tokens=oov,
        coverage_rate=round(rate, 6),
    )


def compute_oov_rate(
    documents: list[str],
    vocabulary_tokens: set[str],
) -> float:
    """Compute the out-of-vocabulary rate for a corpus.

    Args:
        documents: Cleaned documents to evaluate.
        vocabulary_tokens: Set of known vocabulary tokens.

    Returns:
        OOV rate between 0 and 1.
    """
    coverage = vocabulary_coverage_analysis(documents, vocabulary_tokens)
    if coverage.total_tokens == 0:
        return 0.0
    return round(coverage.out_of_vocabulary_tokens / coverage.total_tokens, 6)


def compression_ratio_before_after(
    before_token_count: int,
    after_token_count: int,
) -> float:
    """Compute compression ratio between two tokenization stages.

    For BPE reporting, pass:
    - ``before_token_count``: token count before BPE (for example character
      tokens or whitespace tokens)
    - ``after_token_count``: token count after BPE merges

    Args:
        before_token_count: Number of tokens before compression.
        after_token_count: Number of tokens after compression.

    Returns:
        Compression ratio ``before / after``.

    Raises:
        ValueError: If any input is not positive.
    """
    if before_token_count <= 0:
        raise ValueError("before_token_count must be > 0")
    if after_token_count <= 0:
        raise ValueError("after_token_count must be > 0")
    return round(before_token_count / after_token_count, 6)


def token_length_distribution(tokens: list[str]) -> TokenLengthDistribution:
    """Summarize token length distribution.

    Args:
        tokens: Token list to summarize.

    Returns:
        ``TokenLengthDistribution`` with count/min/max/mean/median.

    Raises:
        ValueError: If ``tokens`` is empty.
    """
    if not tokens:
        raise ValueError("tokens must not be empty")
    lengths = [len(token) for token in tokens]
    return TokenLengthDistribution(
        count=len(lengths),
        minimum=min(lengths),
        maximum=max(lengths),
        mean=round(mean(lengths), 6),
        median=round(median(lengths), 6),
    )


def build_tokenizer_comparison_table(rows: list[dict[str, str | int | float]]) -> str:
    """Build a GitHub Markdown comparison table for tokenizers.

    Expected keys per row:
    ``name``, ``vocab_size``, ``oov_rate``, ``tokens_per_sec``, ``memory_kb``,
    ``compression_ratio``.

    Args:
        rows: List of row dictionaries.

    Returns:
        Markdown table as a string.
    """
    header = (
        "| Tokenizer | Vocab Size | OOV Rate | Encoding Speed (tokens/sec) | "
        "Peak Memory (KB) | Compression Ratio |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {name} | {vocab_size} | {oov_rate} | {tokens_per_sec} | "
            "{memory_kb} | {compression_ratio} |".format(
                name=row["name"],
                vocab_size=row["vocab_size"],
                oov_rate=row["oov_rate"],
                tokens_per_sec=row["tokens_per_sec"],
                memory_kb=row["memory_kb"],
                compression_ratio=row["compression_ratio"],
            )
        )
    return "\n".join(lines)
