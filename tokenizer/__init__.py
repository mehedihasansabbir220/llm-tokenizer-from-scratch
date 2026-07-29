"""High-level tokenizer facade composing vocabulary, encode, and decode."""

from __future__ import annotations

import logging
from pathlib import Path

from configs.config_loader import TokenizerConfig
from tokenizer.decoder import Decoder
from tokenizer.encoder import Encoder
from tokenizer.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class Tokenizer:
    """Production facade for vocabulary training, encoding, and decoding.

    Called by ``app.py`` as the single public tokenizer API so the
    orchestrator does not need to wire Encoder/Decoder/Vocabulary
    separately for common operations.
    """

    def __init__(self, config: TokenizerConfig) -> None:
        """Initialize an untrained tokenizer.

        Args:
            config: Tokenizer configuration.
        """
        self._config = config
        self._vocabulary = Vocabulary(config)
        self._encoder = Encoder(self._vocabulary, config)
        self._decoder = Decoder(self._vocabulary, config)

    @property
    def vocabulary(self) -> Vocabulary:
        """Return the underlying vocabulary object.

        Returns:
            The ``Vocabulary`` instance owned by this tokenizer.
        """
        return self._vocabulary

    def train(self, documents: list[str]) -> None:
        """Train the vocabulary from cleaned documents.

        Args:
            documents: Cleaned corpus documents.
        """
        logger.info("Training tokenizer vocabulary")
        self._vocabulary.train(documents)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode text into token ids.

        Args:
            text: Input text.
            add_special_tokens: Whether to add BOS/EOS.

        Returns:
            List of token ids.
        """
        return self._encoder.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token ids into text.

        Args:
            token_ids: Sequence of token ids.
            skip_special_tokens: Whether to drop special tokens.

        Returns:
            Reconstructed text string.
        """
        return self._decoder.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save_vocabulary(self, path: Path) -> None:
        """Save the trained vocabulary to disk.

        Args:
            path: Destination JSON path.
        """
        self._vocabulary.save(path)

    @classmethod
    def from_vocabulary_file(cls, path: Path, config: TokenizerConfig) -> Tokenizer:
        """Construct a tokenizer from a saved vocabulary file.

        Args:
            path: Path to vocabulary JSON.
            config: Tokenizer configuration.

        Returns:
            A ready-to-use ``Tokenizer`` instance.
        """
        instance = cls(config)
        instance._vocabulary = Vocabulary.load(path, config)
        instance._encoder = Encoder(instance._vocabulary, config)
        instance._decoder = Decoder(instance._vocabulary, config)
        return instance
