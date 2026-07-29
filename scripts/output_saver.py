"""Helpers for persisting pipeline outputs to disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OutputSaver:
    """Save tokenizer pipeline artifacts as JSON files.

    Called at the end of the pipeline to persist encoded/decoded samples
    and statistics so results are reproducible and inspectable offline.
    """

    def save_json(self, data: Any, path: Path) -> None:
        """Serialize ``data`` to a UTF-8 JSON file.

        Args:
            data: JSON-serializable Python object.
            path: Destination file path. Parent directories are created.

        Raises:
            TypeError: If ``data`` is not JSON-serializable.
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
        except TypeError:
            logger.exception("Data is not JSON-serializable for %s", path)
            raise
        except OSError:
            logger.exception("Failed to write output file %s", path)
            raise

        logger.info("Saved output to %s", path)

    def save_encoded(
        self,
        sentence: str,
        token_ids: list[int],
        path: Path,
    ) -> None:
        """Save an encoded sentence payload.

        Args:
            sentence: Original (cleaned) sentence.
            token_ids: Encoded token ids.
            path: Destination JSON path.
        """
        payload = {
            "sentence": sentence,
            "token_ids": token_ids,
            "length": len(token_ids),
        }
        self.save_json(payload, path)

    def save_decoded(
        self,
        token_ids: list[int],
        decoded_text: str,
        path: Path,
    ) -> None:
        """Save a decoded sentence payload.

        Args:
            token_ids: Source token ids.
            decoded_text: Reconstructed text.
            path: Destination JSON path.
        """
        payload = {
            "token_ids": token_ids,
            "decoded_text": decoded_text,
        }
        self.save_json(payload, path)
