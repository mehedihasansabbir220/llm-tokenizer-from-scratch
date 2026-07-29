"""Decoding utilities that convert token ids back to text."""

from __future__ import annotations

import logging

from configs.config_loader import TokenizerConfig
from tokenizer.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class Decoder:
    """Decode sequences of vocabulary ids back into readable text.

    Called after encoding to reconstruct sentences and verify that the
    tokenizer round-trip is correct (modulo unknown tokens).
    """

    def __init__(self, vocabulary: Vocabulary, config: TokenizerConfig) -> None:
        """Initialize the decoder.

        Args:
            vocabulary: Trained vocabulary providing id -> token lookup.
            config: Tokenizer configuration for special-token filtering.
        """
        self._vocabulary = vocabulary
        self._config = config
        self._special_set = set(config.special_tokens)

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a list of token ids into a text string.

        Args:
            token_ids: Sequence of integer token ids.
            skip_special_tokens: If True, omit PAD/UNK/BOS/EOS tokens
                from the reconstructed string.

        Returns:
            Space-joined reconstructed text.

        Raises:
            TypeError: If ``token_ids`` is not a list.
            RuntimeError: If the vocabulary is not ready.
        """
        if not isinstance(token_ids, list):
            raise TypeError(f"Expected list of ints, got {type(token_ids).__name__}")

        tokens: list[str] = []
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError(f"Token ids must be ints, got {type(token_id).__name__}")
            token = self._vocabulary.token_string(token_id)
            if skip_special_tokens and token in self._special_set:
                continue
            tokens.append(token)

        text = " ".join(tokens)
        logger.debug("Decoded %d ids into text of length %d", len(token_ids), len(text))
        return text

    def decode_batch(
        self,
        batch: list[list[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode a batch of token-id sequences.

        Args:
            batch: Nested list of token-id sequences.
            skip_special_tokens: If True, omit special tokens.

        Returns:
            List of reconstructed text strings.
        """
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in batch
        ]
