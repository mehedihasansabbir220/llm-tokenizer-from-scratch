"""Unit tests for tokenizer analysis and benchmarking helpers."""

from __future__ import annotations

from pathlib import Path

from analysis.tokenizer_analysis import (
    build_tokenizer_comparison_table,
    compression_ratio_before_after,
    compute_oov_rate,
    token_length_distribution,
    vocabulary_coverage_analysis,
)
from benchmarks.performance_benchmarks import run_benchmarks
from configs.config_loader import ConfigLoader
from preprocessing.dataset_loader import DatasetLoader
from preprocessing.text_cleaner import TextCleaner
from tokenizer import Tokenizer


def _train_tokenizer() -> tuple[list[str], Tokenizer]:
    """Train a tokenizer on the sample dataset for tests.

    Returns:
        Tuple of ``(cleaned_documents, trained_tokenizer)``.
    """
    root = Path(__file__).resolve().parent.parent
    config = ConfigLoader(root / "configs" / "default.yaml", root_dir=root).load()
    documents = DatasetLoader(config.paths.dataset).load()
    cleaned = TextCleaner(config.preprocessing).clean_corpus(documents)
    tokenizer = Tokenizer(config.tokenizer)
    tokenizer.train(cleaned)
    return cleaned, tokenizer


def test_vocabulary_coverage_and_oov_rate() -> None:
    """Coverage should be complete on the training corpus."""
    cleaned, tokenizer = _train_tokenizer()
    vocab = set(tokenizer.vocabulary.token_to_id.keys())
    coverage = vocabulary_coverage_analysis(cleaned, vocab)
    oov_rate = compute_oov_rate(cleaned, vocab)

    assert coverage.total_tokens > 0
    assert coverage.out_of_vocabulary_tokens == 0
    assert coverage.coverage_rate == 1.0
    assert oov_rate == 0.0


def test_compression_ratio_and_token_length_distribution() -> None:
    """Compression and token-length metrics should produce sane values."""
    ratio = compression_ratio_before_after(100, 40)
    distribution = token_length_distribution(["a", "test", "token", "nlp"])

    assert ratio == 2.5
    assert distribution.count == 4
    assert distribution.minimum == 1
    assert distribution.maximum == 5


def test_comparison_table_markdown() -> None:
    """Comparison helper should return a valid GitHub Markdown table."""
    table = build_tokenizer_comparison_table(
        [
            {
                "name": "Word-Level",
                "vocab_size": 1200,
                "oov_rate": 0.03,
                "tokens_per_sec": 60000,
                "memory_kb": 512.0,
                "compression_ratio": 2.1,
            }
        ]
    )
    assert table.startswith("| Tokenizer | Vocab Size |")
    assert "Word-Level" in table


def test_benchmark_runner_generates_outputs(tmp_path: Path) -> None:
    """Benchmark runner should emit JSON metrics and Markdown table."""
    root = Path(__file__).resolve().parent.parent
    output_json = tmp_path / "benchmarks.json"
    output_md = tmp_path / "comparison.md"

    results = run_benchmarks(
        config_path=root / "configs" / "default.yaml",
        output_json=output_json,
        output_markdown=output_md,
    )

    assert len(results) >= 3
    assert output_json.exists()
    assert output_md.exists()
