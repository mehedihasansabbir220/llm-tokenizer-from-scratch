"""Vocabulary training and persistence for the tokenizer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from algorithms.word_level import WordLevelTokenizer
from configs.config_loader import TokenizerConfig

logger = logging.getLogger(__name__)


class Vocabulary:
    """Train, store, and look up a word-level vocabulary.

    Called after text cleaning to learn a frequency-ranked token set with
    reserved special tokens. Provides bidirectional token <-> id maps used
    by the encoder and decoder.
    """

    def __init__(self, config: TokenizerConfig) -> None:
        """Initialize an empty vocabulary.

        Args:
            config: Tokenizer configuration including vocab size and
                special tokens.
        """
        self._config = config
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._frequencies: dict[str, int] = {}
        self._algorithm = WordLevelTokenizer()

    @property
    def size(self) -> int:
        """Return the number of tokens in the vocabulary.

        Returns:
            Vocabulary size including special tokens.
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
    def id_to_token(self) -> dict[int, str]:
        """Return a copy of the id-to-token mapping.

        Returns:
            Mapping from integer id to token string.
        """
        return dict(self._id_to_token)

    @property
    def frequencies(self) -> dict[str, int]:
        """Return a copy of corpus token frequencies.

        Returns:
            Mapping from token to occurrence count in the training corpus.
        """
        return dict(self._frequencies)

    def train(self, documents: list[str]) -> None:
        """Train the vocabulary from cleaned documents.

        Special tokens are inserted first, then remaining slots are filled
        by corpus tokens sorted by descending frequency (ties broken by
        lexicographic order for determinism).

        Args:
            documents: Cleaned corpus documents.

        Raises:
            ValueError: If ``documents`` is empty or no tokens remain after
                frequency filtering.
            RuntimeError: If the vocabulary was already trained.
        """
        if self._token_to_id:
            raise RuntimeError("Vocabulary has already been trained")
        if not documents:
            raise ValueError("Cannot train vocabulary on an empty corpus")

        logger.info("Training vocabulary (max size=%d)", self._config.vocab_size)
        frequencies = self._algorithm.count_frequencies(documents)
        self._frequencies = dict(frequencies)

        for token in self._config.special_tokens:
            self._add_token(token)

        remaining_slots = self._config.vocab_size - len(self._token_to_id)
        if remaining_slots <= 0:
            logger.warning(
                "vocab_size (%d) is fully consumed by special tokens",
                self._config.vocab_size,
            )
            return

        candidates = [
            (token, count)
            for token, count in frequencies.items()
            if count >= self._config.min_frequency
            and token not in self._token_to_id
            and len(token) >= 1
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))

        if not candidates and len(self._token_to_id) == len(self._config.special_tokens):
            raise ValueError(
                "No tokens met the minimum frequency threshold; "
                "cannot build a usable vocabulary"
            )

        for token, _count in candidates[:remaining_slots]:
            self._add_token(token)

        logger.info("Vocabulary trained with %d tokens", self.size)

    def token_id(self, token: str) -> int:
        """Look up the integer id for a token.

        Args:
            token: Token string.

        Returns:
            Integer id, or the unknown-token id if missing.

        Raises:
            RuntimeError: If the vocabulary has not been trained.
        """
        self._ensure_trained()
        unk_id = self._token_to_id[self._config.unknown_token]
        return self._token_to_id.get(token, unk_id)

    def token_string(self, token_id: int) -> str:
        """Look up the token string for an integer id.

        Args:
            token_id: Integer token id.

        Returns:
            Token string, or the unknown token if the id is unknown.

        Raises:
            RuntimeError: If the vocabulary has not been trained.
        """
        self._ensure_trained()
        return self._id_to_token.get(token_id, self._config.unknown_token)

    def save(self, path: Path) -> None:
        """Persist the vocabulary to a JSON file.

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If the vocabulary has not been trained.
            OSError: If the file cannot be written.
        """
        self._ensure_trained()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "token_to_id": self._token_to_id,
            "frequencies": self._frequencies,
            "special_tokens": self._config.special_tokens,
            "unknown_token": self._config.unknown_token,
            "size": self.size,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        logger.info("Saved vocabulary (%d tokens) to %s", self.size, path)

    @classmethod
    def load(cls, path: Path, config: TokenizerConfig) -> Vocabulary:
        """Load a previously saved vocabulary from disk.

        Args:
            path: Path to a vocabulary JSON file.
            config: Tokenizer configuration for special-token settings.

        Returns:
            A populated ``Vocabulary`` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the JSON payload is malformed.
        """
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        try:
            token_to_id = {str(k): int(v) for k, v in payload["token_to_id"].items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed vocabulary file: {path}") from exc

        vocabulary = cls(config)
        vocabulary._token_to_id = token_to_id
        vocabulary._id_to_token = {idx: tok for tok, idx in token_to_id.items()}
        vocabulary._frequencies = {
            str(k): int(v) for k, v in payload.get("frequencies", {}).items()
        }
        logger.info("Loaded vocabulary (%d tokens) from %s", vocabulary.size, path)
        return vocabulary

    def _add_token(self, token: str) -> None:
        """Insert a token into both lookup maps.

        Args:
            token: Token string to register.
        """
        if token in self._token_to_id:
            return
        token_id = len(self._token_to_id)
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token

    def _ensure_trained(self) -> None:
        """Raise if the vocabulary has not been trained or loaded.

        Raises:
            RuntimeError: When lookup maps are empty.
        """
        if not self._token_to_id:
            raise RuntimeError("Vocabulary has not been trained or loaded")
