"""Encoding utilities that convert text to token ids."""

from __future__ import annotations

import logging

from algorithms.word_level import WordLevelTokenizer
from configs.config_loader import TokenizerConfig
from tokenizer.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class Encoder:
    """Encode cleaned text into sequences of vocabulary ids.

    Called after vocabulary training to convert sentences into integer
    sequences suitable for model input or persistence.
    """

    def __init__(self, vocabulary: Vocabulary, config: TokenizerConfig) -> None:
        """Initialize the encoder.

        Args:
            vocabulary: Trained vocabulary providing token -> id lookup.
            config: Tokenizer configuration for special-token handling.
        """
        self._vocabulary = vocabulary
        self._config = config
        self._algorithm = WordLevelTokenizer()

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode a single text string into token ids.

        Args:
            text: Cleaned (or raw) input text.
            add_special_tokens: If True, wrap the sequence with BOS/EOS.

        Returns:
            List of integer token ids.

        Raises:
            TypeError: If ``text`` is not a string.
            RuntimeError: If the vocabulary is not ready.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        tokens = self._algorithm.tokenize(text)
        ids = [self._vocabulary.token_id(token) for token in tokens]

        if add_special_tokens:
            bos_id = self._vocabulary.token_id(self._config.bos_token)
            eos_id = self._vocabulary.token_id(self._config.eos_token)
            ids = [bos_id, *ids, eos_id]

        logger.debug("Encoded %d tokens into %d ids", len(tokens), len(ids))
        return ids

    def encode_batch(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        """Encode a batch of text strings.

        Args:
            texts: List of input strings.
            add_special_tokens: If True, wrap each sequence with BOS/EOS.

        Returns:
            Nested list of token-id sequences.
        """
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
