"""Dataset loading utilities for plain-text corpora."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load text corpora from disk for tokenizer training.

    Called after configuration loading to bring raw training data into
    memory as a list of documents (one document per non-empty line).
    """

    def __init__(self, dataset_path: Path) -> None:
        """Initialize the loader with a corpus path.

        Args:
            dataset_path: Absolute path to a UTF-8 text corpus file.
        """
        self._dataset_path = Path(dataset_path)

    def load(self) -> list[str]:
        """Load documents from the configured corpus file.

        Returns:
            A list of non-empty document strings.

        Raises:
            FileNotFoundError: If the corpus file does not exist.
            ValueError: If the corpus is empty after filtering blanks.
            OSError: If the file cannot be read.
        """
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self._dataset_path}")

        logger.info("Loading dataset from %s", self._dataset_path)
        try:
            with self._dataset_path.open("r", encoding="utf-8") as handle:
                documents = [line.strip() for line in handle if line.strip()]
        except OSError:
            logger.exception("Failed to read dataset file")
            raise

        if not documents:
            raise ValueError(f"Dataset is empty: {self._dataset_path}")

        logger.info("Loaded %d documents from dataset", len(documents))
        return documents

    @property
    def path(self) -> Path:
        """Return the configured dataset path.

        Returns:
            Absolute path to the corpus file.
        """
        return self._dataset_path
