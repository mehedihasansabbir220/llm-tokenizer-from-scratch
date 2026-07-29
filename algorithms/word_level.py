"""Word-level tokenization algorithm implemented from scratch."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Iterable

logger = logging.getLogger(__name__)


class WordLevelTokenizer:
    """Split text into whitespace-separated word tokens.

    Called by the vocabulary builder and encoder/decoder to keep the
    splitting algorithm isolated from I/O and configuration concerns.
    Does not depend on NLTK, spaCy, or HuggingFace.
    """

    _TOKEN_RE = re.compile(r"\S+")

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a single string into word tokens.

        Args:
            text: Cleaned input text.

        Returns:
            List of word tokens. Empty list if ``text`` is blank.

        Raises:
            TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        return self._TOKEN_RE.findall(text)

    def tokenize_corpus(self, documents: Iterable[str]) -> list[list[str]]:
        """Tokenize a corpus of documents.

        Args:
            documents: Iterable of cleaned document strings.

        Returns:
            Nested list of tokens per document.
        """
        tokenized = [self.tokenize(document) for document in documents]
        logger.debug("Tokenized %d documents", len(tokenized))
        return tokenized

    def count_frequencies(self, documents: Iterable[str]) -> Counter[str]:
        """Count token frequencies across a corpus.

        Args:
            documents: Iterable of cleaned document strings.

        Returns:
            ``Counter`` mapping token -> occurrence count.
        """
        frequencies: Counter[str] = Counter()
        for document in documents:
            frequencies.update(self.tokenize(document))
        logger.info("Counted frequencies for %d unique tokens", len(frequencies))
        return frequencies
