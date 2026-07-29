"""Smoke tests for the tokenizer pipeline modules."""

from __future__ import annotations

from pathlib import Path

from configs.config_loader import ConfigLoader
from preprocessing.dataset_loader import DatasetLoader
from preprocessing.text_cleaner import TextCleaner
from tokenizer import Tokenizer


def test_round_trip_encode_decode() -> None:
    """Train on the sample corpus and verify encode/decode round-trip."""
    root = Path(__file__).resolve().parent.parent
    config = ConfigLoader(root / "configs" / "default.yaml", root_dir=root).load()
    documents = DatasetLoader(config.paths.dataset).load()
    cleaned = TextCleaner(config.preprocessing).clean_corpus(documents)

    tokenizer = Tokenizer(config.tokenizer)
    tokenizer.train(cleaned)

    sentence = TextCleaner(config.preprocessing).clean(config.runtime.demo_sentence)
    token_ids = tokenizer.encode(sentence)
    decoded = tokenizer.decode(token_ids)

    assert token_ids[0] == tokenizer.vocabulary.token_id(config.tokenizer.bos_token)
    assert token_ids[-1] == tokenizer.vocabulary.token_id(config.tokenizer.eos_token)
    assert decoded == sentence
