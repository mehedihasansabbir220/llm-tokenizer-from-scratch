"""Production entrypoint for the from-scratch NLP tokenizer pipeline.

This module orchestrates the full lifecycle:

1. Load configuration
2. Load dataset
3. Clean text
4. Train vocabulary
5. Encode sentence
6. Decode sentence
7. Display statistics
8. Save outputs

Each step delegates to a dedicated module so responsibilities stay
isolated, testable, and free of NLTK / spaCy / HuggingFace dependencies.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from algorithms.word_level import WordLevelTokenizer
from configs.config_loader import AppConfig, ConfigLoader
from preprocessing.dataset_loader import DatasetLoader
from preprocessing.text_cleaner import TextCleaner
from scripts.output_saver import OutputSaver
from tokenizer import Tokenizer
from visualization.statistics import StatisticsReporter

logger = logging.getLogger(__name__)


class TokenizerApplication:
    """Orchestrate the end-to-end tokenizer training and demo pipeline.

    Instantiated by ``main`` after CLI argument parsing. Holds shared
    configuration and wires modular components in a fixed, logged order.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize the application with a loaded configuration.

        Args:
            config: Fully validated application configuration.
        """
        self._config = config
        self._tokenizer: Tokenizer | None = None
        self._cleaned_documents: list[str] = []

    def run(self) -> int:
        """Execute the full pipeline.

        Returns:
            Process exit code: ``0`` on success, ``1`` on failure.
        """
        try:
            self._ensure_output_directories()
            documents = self._load_dataset()
            cleaned = self._clean_text(documents)
            self._cleaned_documents = cleaned
            self._train_vocabulary(cleaned)
            cleaned_demo = self._prepare_demo_sentence()
            token_ids = self._encode_sentence(cleaned_demo)
            decoded = self._decode_sentence(token_ids)
            stats = self._display_statistics()
            self._save_outputs(cleaned_demo, token_ids, decoded, stats)
        except (FileNotFoundError, ValueError, RuntimeError, OSError, TypeError) as exc:
            logger.error("Pipeline failed: %s", exc)
            return 1
        except Exception:
            logger.exception("Unexpected pipeline failure")
            return 1

        logger.info("Pipeline completed successfully")
        return 0

    def _ensure_output_directories(self) -> None:
        """Create directories required for models, outputs, and logs.

        Why this is called:
            Guarantees that later save steps do not fail because parent
            directories are missing on a fresh checkout.
        """
        for path in (
            self._config.paths.vocabulary.parent,
            self._config.paths.encoded_output.parent,
            self._config.paths.decoded_output.parent,
            self._config.paths.statistics.parent,
            self._config.paths.log_file.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)
        logger.debug("Output directories ensured")

    def _load_dataset(self) -> list[str]:
        """Load the raw training corpus from disk.

        Why ``DatasetLoader`` is called:
            Isolates filesystem I/O and empty-corpus validation from the
            rest of the pipeline so training never silently starts on
            missing or blank data.

        Returns:
            List of raw document strings.
        """
        logger.info("Step 1/8 skipped (config already loaded); Step 2/8: Load dataset")
        loader = DatasetLoader(self._config.paths.dataset)
        return loader.load()

    def _clean_text(self, documents: list[str]) -> list[str]:
        """Normalize raw documents before tokenization.

        Why ``TextCleaner`` is called:
            Applies configurable lowercasing, punctuation removal, and
            whitespace collapsing so vocabulary training sees consistent
            tokens rather than noisy surface forms.

        Args:
            documents: Raw corpus documents.

        Returns:
            Cleaned, non-empty documents.
        """
        logger.info("Step 3/8: Clean text")
        cleaner = TextCleaner(self._config.preprocessing)
        return cleaner.clean_corpus(documents)

    def _train_vocabulary(self, documents: list[str]) -> Tokenizer:
        """Train the word-level vocabulary from cleaned documents.

        Why ``Tokenizer`` / ``Vocabulary`` / ``WordLevelTokenizer`` are called:
            ``Tokenizer`` is the facade used by the app.
            ``Vocabulary`` ranks tokens by frequency and reserves special
            tokens.
            ``WordLevelTokenizer`` (algorithms package) performs the actual
            whitespace split without third-party NLP libraries.

        Args:
            documents: Cleaned corpus documents.

        Returns:
            The trained ``Tokenizer`` instance.
        """
        logger.info("Step 4/8: Train vocabulary")
        # WordLevelTokenizer is exercised inside Vocabulary.train via
        # frequency counting; constructing it here documents the algorithm
        # dependency explicitly for operators reading the orchestrator.
        algorithm = WordLevelTokenizer()
        sample_tokens = algorithm.tokenize(documents[0]) if documents else []
        logger.debug("Algorithm sample tokens from first doc: %s", sample_tokens[:10])

        tokenizer = Tokenizer(self._config.tokenizer)
        tokenizer.train(documents)
        self._tokenizer = tokenizer
        return tokenizer

    def _prepare_demo_sentence(self) -> str:
        """Clean the configured demonstration sentence.

        Why ``TextCleaner`` is reused:
            Encode/decode demos must use the same normalization rules as
            training, otherwise demo tokens would not match the vocabulary.

        Returns:
            Cleaned demo sentence.
        """
        cleaner = TextCleaner(self._config.preprocessing)
        return cleaner.clean(self._config.runtime.demo_sentence)

    def _encode_sentence(self, sentence: str) -> list[int]:
        """Encode the demo sentence into token ids.

        Why ``Encoder`` (via ``Tokenizer.encode``) is called:
            Converts human text into the integer sequence format consumed
            by downstream ML models and persisted as an artifact.

        Args:
            sentence: Cleaned demo sentence.

        Returns:
            List of token ids including BOS/EOS.

        Raises:
            RuntimeError: If the tokenizer has not been trained.
        """
        logger.info("Step 5/8: Encode sentence")
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer must be trained before encoding")
        token_ids = self._tokenizer.encode(sentence, add_special_tokens=True)
        logger.info("Encoded ids: %s", token_ids)
        return token_ids

    def _decode_sentence(self, token_ids: list[int]) -> str:
        """Decode token ids back into readable text.

        Why ``Decoder`` (via ``Tokenizer.decode``) is called:
            Reconstructs text from ids to validate the round-trip and to
            produce a human-inspectable output artifact.

        Args:
            token_ids: Encoded token ids.

        Returns:
            Decoded text string.

        Raises:
            RuntimeError: If the tokenizer has not been trained.
        """
        logger.info("Step 6/8: Decode sentence")
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer must be trained before decoding")
        decoded = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        logger.info("Decoded text: %r", decoded)
        return decoded

    def _display_statistics(self) -> dict:
        """Compute and print tokenizer statistics.

        Why ``StatisticsReporter`` is called:
            Surfaces vocabulary size, coverage, and top tokens so operators
            can judge data quality before shipping artifacts.

        Returns:
            Statistics dictionary suitable for JSON serialization.

        Raises:
            RuntimeError: If the tokenizer has not been trained.
        """
        logger.info("Step 7/8: Display statistics")
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer must be trained before statistics")
        reporter = StatisticsReporter(self._tokenizer.vocabulary)
        stats = reporter.compute(
            documents=self._cleaned_documents,
            special_tokens=self._config.tokenizer.special_tokens,
            unknown_token=self._config.tokenizer.unknown_token,
        )
        reporter.display(stats)
        return stats.to_dict()

    def _save_outputs(
        self,
        sentence: str,
        token_ids: list[int],
        decoded: str,
        stats: dict,
    ) -> None:
        """Persist vocabulary, encodings, decodings, and statistics.

        Why ``OutputSaver`` and ``Tokenizer.save_vocabulary`` are called:
            Writes reproducible artifacts under ``models/`` and ``outputs/``
            for later evaluation, debugging, and model integration.

        Args:
            sentence: Cleaned demo sentence.
            token_ids: Encoded ids.
            decoded: Decoded text.
            stats: Statistics dictionary.
        """
        logger.info("Step 8/8: Save outputs")
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer must be trained before saving")

        saver = OutputSaver()
        self._tokenizer.save_vocabulary(self._config.paths.vocabulary)
        saver.save_encoded(sentence, token_ids, self._config.paths.encoded_output)
        saver.save_decoded(token_ids, decoded, self._config.paths.decoded_output)
        saver.save_json(stats, self._config.paths.statistics)


def configure_logging(level: str, log_file: Path | None = None) -> None:
    """Configure root logging for console and optional file output.

    Why this is called before the pipeline:
        Ensures every module emits structured logs to stdout (and optionally
        a file) with a consistent format for production diagnostics.

    Args:
        level: Logging level name (e.g. ``INFO``, ``DEBUG``).
        log_file: Optional path for a rotating-style file handler (plain
            ``FileHandler``; parent dirs must already exist or be creatable).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the application.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="From-scratch NLP tokenizer training and demo pipeline",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file",
    )
    return parser.parse_args(argv)


def load_configuration(config_path: Path) -> AppConfig:
    """Load and validate application configuration.

    Why ``ConfigLoader`` is called first:
        Centralizes path resolution, type coercion, and validation so every
        later step receives a single immutable ``AppConfig`` instead of
        scattered magic strings.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Validated ``AppConfig``.
    """
    logger.info("Step 1/8: Load configuration")
    loader = ConfigLoader(config_path)
    return loader.load()


def main(argv: list[str] | None = None) -> int:
    """Application entrypoint.

    Args:
        argv: Optional CLI argument list for testing.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)

    # Bootstrap logging at INFO until config supplies the preferred level.
    configure_logging("INFO")

    try:
        config = load_configuration(args.config)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Failed to load configuration: %s", exc)
        return 1

    configure_logging(config.runtime.log_level, config.paths.log_file)
    logger.info(
        "Starting %s v%s",
        config.project.name,
        config.project.version,
    )

    application = TokenizerApplication(config)
    return application.run()


if __name__ == "__main__":
    sys.exit(main())
