"""Text cleaning utilities for corpus preprocessing."""

from __future__ import annotations

import logging
import re
import string
from typing import Iterable

from configs.config_loader import PreprocessingConfig

logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean raw text according to preprocessing configuration.

    Called after dataset loading to normalize noisy corpus text before
    vocabulary training. Keeps cleaning rules explicit and reusable.
    """

    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, config: PreprocessingConfig) -> None:
        """Initialize the cleaner with preprocessing options.

        Args:
            config: Preprocessing configuration dataclass.
        """
        self._config = config
        self._punctuation_table = str.maketrans("", "", string.punctuation)

    def clean(self, text: str) -> str:
        """Clean a single text string.

        Args:
            text: Raw input text.

        Returns:
            Normalized text string.

        Raises:
            TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        cleaned = text
        if self._config.lowercase:
            cleaned = cleaned.lower()
        if self._config.remove_punctuation:
            cleaned = cleaned.translate(self._punctuation_table)
        if self._config.collapse_whitespace:
            cleaned = self._WHITESPACE_RE.sub(" ", cleaned).strip()
        return cleaned

    def clean_corpus(self, documents: Iterable[str]) -> list[str]:
        """Clean a collection of documents.

        Args:
            documents: Iterable of raw document strings.

        Returns:
            List of cleaned, non-empty documents.

        Raises:
            ValueError: If no usable documents remain after cleaning.
        """
        cleaned_docs: list[str] = []
        for index, document in enumerate(documents):
            try:
                cleaned = self.clean(document)
            except TypeError:
                logger.warning("Skipping non-string document at index %d", index)
                continue
            if cleaned:
                cleaned_docs.append(cleaned)

        if not cleaned_docs:
            raise ValueError("No usable documents remain after cleaning")

        logger.info("Cleaned %d documents", len(cleaned_docs))
        return cleaned_docs
