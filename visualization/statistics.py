"""Tokenizer statistics computation and display."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from tokenizer.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


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
