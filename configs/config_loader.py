"""Configuration loading utilities for the tokenizer application."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectConfig:
    """High-level project metadata.

    Attributes:
        name: Human-readable project name.
        version: Semantic version string.
        seed: Random seed for reproducibility.
    """

    name: str
    version: str
    seed: int


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths used by the pipeline.

    Attributes:
        dataset: Path to the input corpus file.
        vocabulary: Path where the trained vocabulary is saved.
        encoded_output: Path for encoded token-id sequences.
        decoded_output: Path for decoded text output.
        statistics: Path for tokenizer statistics JSON.
        log_file: Path for application log output.
    """

    dataset: Path
    vocabulary: Path
    encoded_output: Path
    decoded_output: Path
    statistics: Path
    log_file: Path


@dataclass(frozen=True)
class PreprocessingConfig:
    """Text-cleaning options.

    Attributes:
        lowercase: Whether to lowercase all text.
        remove_punctuation: Whether to strip punctuation characters.
        collapse_whitespace: Whether to normalize runs of whitespace.
        min_token_length: Minimum allowed token length after cleaning.
    """

    lowercase: bool
    remove_punctuation: bool
    collapse_whitespace: bool
    min_token_length: int


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer training and encoding options.

    Attributes:
        vocab_size: Maximum vocabulary size including special tokens.
        min_frequency: Minimum corpus frequency required for inclusion.
        special_tokens: Ordered list of reserved special tokens.
        unknown_token: Token used for out-of-vocabulary items.
        pad_token: Padding token.
        bos_token: Beginning-of-sequence token.
        eos_token: End-of-sequence token.
    """

    vocab_size: int
    min_frequency: int
    special_tokens: list[str]
    unknown_token: str
    pad_token: str
    bos_token: str
    eos_token: str


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime and demonstration settings.

    Attributes:
        demo_sentence: Example sentence used for encode/decode demo.
        log_level: Logging verbosity level name.
    """

    demo_sentence: str
    log_level: str


@dataclass(frozen=True)
class AppConfig:
    """Root application configuration aggregate.

    Attributes:
        project: Project metadata.
        paths: Input/output filesystem paths.
        preprocessing: Cleaning options.
        tokenizer: Vocabulary and encoding options.
        runtime: Runtime/demo settings.
        root_dir: Absolute project root directory.
    """

    project: ProjectConfig
    paths: PathsConfig
    preprocessing: PreprocessingConfig
    tokenizer: TokenizerConfig
    runtime: RuntimeConfig
    root_dir: Path = field(repr=False)


class ConfigLoader:
    """Load and validate YAML configuration for the tokenizer pipeline.

    This module is called first so every downstream component receives a
    single, typed, immutable configuration object instead of ad-hoc dicts.
    """

    def __init__(self, config_path: str | Path, root_dir: str | Path | None = None) -> None:
        """Initialize the configuration loader.

        Args:
            config_path: Path to a YAML configuration file.
            root_dir: Optional project root. Defaults to the parent of
                the ``configs`` directory.
        """
        self._config_path = Path(config_path)
        if root_dir is None:
            self._root_dir = self._config_path.resolve().parent.parent
        else:
            self._root_dir = Path(root_dir).resolve()

    def load(self) -> AppConfig:
        """Load, parse, and validate the YAML configuration file.

        Returns:
            A fully populated ``AppConfig`` instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If required keys are missing or invalid.
            yaml.YAMLError: If the YAML syntax is invalid.
        """
        if not self._config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self._config_path}")

        logger.info("Loading configuration from %s", self._config_path)
        try:
            with self._config_path.open("r", encoding="utf-8") as handle:
                raw: dict[str, Any] = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            logger.exception("Failed to parse YAML configuration")
            raise ValueError(f"Invalid YAML in {self._config_path}: {exc}") from exc

        try:
            config = self._build_config(raw)
        except KeyError as exc:
            raise ValueError(f"Missing required configuration key: {exc}") from exc

        logger.info(
            "Configuration loaded for project '%s' v%s",
            config.project.name,
            config.project.version,
        )
        return config

    def _resolve(self, relative_path: str) -> Path:
        """Resolve a path relative to the project root.

        Args:
            relative_path: Path string from the config file.

        Returns:
            Absolute ``Path`` under the project root.
        """
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return (self._root_dir / path).resolve()

    def _build_config(self, raw: dict[str, Any]) -> AppConfig:
        """Map a raw YAML dictionary onto typed dataclasses.

        Args:
            raw: Parsed YAML content.

        Returns:
            Validated ``AppConfig``.

        Raises:
            KeyError: If a required section or field is absent.
            ValueError: If numeric constraints are violated.
        """
        project = ProjectConfig(
            name=str(raw["project"]["name"]),
            version=str(raw["project"]["version"]),
            seed=int(raw["project"]["seed"]),
        )
        paths_raw = raw["paths"]
        paths = PathsConfig(
            dataset=self._resolve(paths_raw["dataset"]),
            vocabulary=self._resolve(paths_raw["vocabulary"]),
            encoded_output=self._resolve(paths_raw["encoded_output"]),
            decoded_output=self._resolve(paths_raw["decoded_output"]),
            statistics=self._resolve(paths_raw["statistics"]),
            log_file=self._resolve(paths_raw["log_file"]),
        )
        prep_raw = raw["preprocessing"]
        preprocessing = PreprocessingConfig(
            lowercase=bool(prep_raw["lowercase"]),
            remove_punctuation=bool(prep_raw["remove_punctuation"]),
            collapse_whitespace=bool(prep_raw["collapse_whitespace"]),
            min_token_length=int(prep_raw["min_token_length"]),
        )
        tok_raw = raw["tokenizer"]
        vocab_size = int(tok_raw["vocab_size"])
        min_frequency = int(tok_raw["min_frequency"])
        if vocab_size < 1:
            raise ValueError("tokenizer.vocab_size must be >= 1")
        if min_frequency < 1:
            raise ValueError("tokenizer.min_frequency must be >= 1")

        tokenizer = TokenizerConfig(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(tok_raw["special_tokens"]),
            unknown_token=str(tok_raw["unknown_token"]),
            pad_token=str(tok_raw["pad_token"]),
            bos_token=str(tok_raw["bos_token"]),
            eos_token=str(tok_raw["eos_token"]),
        )
        runtime = RuntimeConfig(
            demo_sentence=str(raw["runtime"]["demo_sentence"]),
            log_level=str(raw["runtime"]["log_level"]).upper(),
        )
        return AppConfig(
            project=project,
            paths=paths,
            preprocessing=preprocessing,
            tokenizer=tokenizer,
            runtime=runtime,
            root_dir=self._root_dir,
        )
