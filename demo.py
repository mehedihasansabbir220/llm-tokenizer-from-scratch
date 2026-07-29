"""Interactive terminal demo for the from-scratch NLP tokenizer.

Type a sentence and watch it travel through every stage of the pipeline:

    raw text -> cleaning -> tokenization -> vocabulary lookup -> decoding

Each stage prints what it produced *and* a one-line explanation of what it
did, so the demo doubles as a walkthrough of how the tokenizer works. The
most instructive column is ``In vocab?`` in the token table: it shows
exactly which words the vocabulary knows and which collapse to the unknown
token, which is the single thing that most often surprises people about a
fixed-vocabulary tokenizer.

Usage:

    python demo.py                       # interactive session
    python demo.py --text "Hello there"  # analyze one string and exit
    python demo.py --vocabulary models/vocabulary.json
    echo "some text" | python demo.py    # piped input, one analysis per line

The demo trains a fresh vocabulary from the configured corpus by default,
so what you see always matches the dataset on disk. Pass ``--vocabulary``
to load a previously saved one instead.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import utils
from algorithms.word_level import WordLevelTokenizer
from configs.config_loader import AppConfig, ConfigLoader
from preprocessing.dataset_loader import DatasetLoader
from preprocessing.text_cleaner import TextCleaner
from tokenizer import Tokenizer
from visualization.statistics import CorpusStatisticsReporter, render_table

logger = logging.getLogger(__name__)

# Prompt shown when waiting for input. A single glyph keeps the left edge
# quiet next to the boxed tables.
_PROMPT = "› "

# Commands the session understands, in the order :help lists them.
_COMMANDS: dict[str, str] = {
    ":help": "Show this command list",
    ":stats": "Show statistics for the whole training corpus",
    ":vocab": "Show the first entries of the vocabulary",
    ":examples": "Analyze a few illustrative sample sentences",
    ":quit": "Leave the demo (Ctrl-D and Ctrl-C also work)",
}

# Sentences chosen to demonstrate specific behaviors rather than to look
# good: one fully in-vocabulary, one with guaranteed unknowns, one that is
# pure punctuation and therefore cleans away to nothing.
_EXAMPLE_SENTENCES = [
    "The vocabulary maps text to token ids.",
    "Zygomorphic bryophytes flabbergasted the quokka!",
    "!!! ??? ...",
]

# Per-stage explanations. Kept as data so the wording can be revised without
# touching the rendering code.
_EXPLANATIONS = {
    "original": "Exactly what you typed, before anything touched it.",
    "cleaned": "TextCleaner applied the configured rules: {rules}.",
    "tokens": "WordLevelTokenizer split the cleaned text on whitespace. "
    "No NLTK, spaCy, or HuggingFace involved.",
    "ids": "Each token was looked up in the vocabulary. Unknown words fall "
    "back to {unk}, and {bos}/{eos} mark the sequence boundaries.",
    "decoded": "Ids mapped back to strings and rejoined with spaces. "
    "Special tokens are dropped on the way out.",
    "stats": "What the round-trip cost: how much survived, and what did not.",
}


class Palette:
    """ANSI color codes, with a single switch to turn them all off.

    Colors are disabled automatically when output is redirected to a file or
    a pipe, and whenever the ``NO_COLOR`` environment variable is set. Escape
    codes written into a log file are noise, and the tables are designed to
    read fine without them.
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the palette.

        Args:
            enabled: Whether to emit escape codes at all.
        """
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        """Wrap text in an escape sequence when colors are enabled.

        Args:
            code: SGR parameter string.
            text: Text to wrap.

        Returns:
            The text, decorated or untouched.
        """
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def title(self, text: str) -> str:
        """Style a section heading.

        Args:
            text: Heading text.

        Returns:
            Bold text.
        """
        return self._wrap("1", text)

    def muted(self, text: str) -> str:
        """Style secondary explanatory text.

        Args:
            text: Text to mute.

        Returns:
            Dimmed text.
        """
        return self._wrap("2", text)

    def good(self, text: str) -> str:
        """Style a positive result.

        Args:
            text: Text to style.

        Returns:
            Green text.
        """
        return self._wrap("32", text)

    def warn(self, text: str) -> str:
        """Style a cautionary result.

        Args:
            text: Text to style.

        Returns:
            Yellow text.
        """
        return self._wrap("33", text)

    def accent(self, text: str) -> str:
        """Style an emphasized value.

        Args:
            text: Text to style.

        Returns:
            Cyan text.
        """
        return self._wrap("36", text)


@dataclass(frozen=True)
class TokenView:
    """One token as it appears in the analysis table.

    Attributes:
        position: 1-based index within the sentence.
        token: The token string after cleaning.
        token_id: Vocabulary id it resolved to.
        known: False when the token is absent from the vocabulary and was
            therefore mapped to the unknown-token id.
    """

    position: int
    token: str
    token_id: int
    known: bool


@dataclass(frozen=True)
class Analysis:
    """The complete result of pushing one input through the pipeline.

    Every displayed value is captured here rather than printed as it is
    computed, so the same analysis can be rendered to a terminal, asserted
    on in a test, or serialized without rerunning the pipeline.

    Attributes:
        original: The raw input string.
        cleaned: The string after cleaning.
        tokens: Token views, in order.
        token_ids: Full id sequence including BOS/EOS.
        decoded: Text reconstructed from the ids.
        round_trip: True when the decoded text matches the cleaned text.
        unknown_count: How many tokens were out of vocabulary.
    """

    original: str
    cleaned: str
    tokens: list[TokenView] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    decoded: str = ""
    round_trip: bool = False
    unknown_count: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether cleaning removed everything.

        Returns:
            True when no tokens survived cleaning.
        """
        return not self.tokens


class TokenizerDemo:
    """Drive the tokenizer interactively and render each stage.

    Holds the trained tokenizer and the cleaner so a session can analyze
    many inputs without repeating the expensive setup.
    """

    def __init__(
        self,
        config: AppConfig,
        tokenizer: Tokenizer,
        documents: Sequence[str],
        *,
        palette: Palette | None = None,
    ) -> None:
        """Initialize the demo session.

        Args:
            config: Loaded application configuration.
            tokenizer: A trained tokenizer.
            documents: Cleaned training documents, used by ``:stats``.
            palette: Optional color palette. Defaults to auto-detection.
        """
        self._config = config
        self._tokenizer = tokenizer
        self._documents = list(documents)
        self._cleaner = TextCleaner(config.preprocessing)
        self._algorithm = WordLevelTokenizer()
        self._palette = palette or Palette(_supports_color())
        self._analyzed = 0

    # -------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------

    def analyze(self, text: str) -> Analysis:
        """Run one input through every pipeline stage.

        Args:
            text: Raw user input.

        Returns:
            A fully populated ``Analysis``.

        Raises:
            TypeError: If ``text`` is not a string.
        """
        utils.validate_type(text, str, "text")

        cleaned = self._cleaner.clean(text)
        tokens = self._algorithm.tokenize(cleaned)

        vocabulary = self._tokenizer.vocabulary
        known_tokens = vocabulary.token_to_id
        views = [
            TokenView(
                position=index,
                token=token,
                token_id=vocabulary.token_id(token),
                known=token in known_tokens,
            )
            for index, token in enumerate(tokens, start=1)
        ]

        token_ids = self._tokenizer.encode(cleaned, add_special_tokens=True)
        decoded = self._tokenizer.decode(token_ids, skip_special_tokens=True)

        analysis = Analysis(
            original=text,
            cleaned=cleaned,
            tokens=views,
            token_ids=token_ids,
            decoded=decoded,
            round_trip=decoded == cleaned,
            unknown_count=sum(1 for view in views if not view.known),
        )
        self._analyzed += 1
        logger.debug(
            "Analyzed input: %d token(s), %d unknown",
            len(views),
            analysis.unknown_count,
        )
        return analysis

    # -------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------

    def render(self, analysis: Analysis) -> str:
        """Render an analysis as a printable report.

        Args:
            analysis: A completed analysis.

        Returns:
            The full multi-section report.
        """
        color = self._palette
        parts: list[str] = []

        parts.append(self._section("1. Original text", _EXPLANATIONS["original"]))
        parts.append(f"   {color.accent(analysis.original)}")

        rules = ", ".join(self._active_cleaning_rules()) or "none enabled"
        parts.append(
            self._section("2. Cleaned text", _EXPLANATIONS["cleaned"].format(rules=rules))
        )
        parts.append(f"   {color.accent(analysis.cleaned or '(empty)')}")

        if analysis.is_empty:
            # Nothing survived cleaning, so the remaining stages have no
            # input. Say so plainly instead of printing four empty tables.
            parts.append("")
            parts.append(
                color.warn(
                    "   Cleaning removed every character, so there is nothing "
                    "left to tokenize."
                )
            )
            parts.append(
                color.muted(
                    "   That is expected for input made entirely of punctuation "
                    "when remove_punctuation is on."
                )
            )
            return "\n".join(parts)

        parts.append(self._section("3. Tokens", _EXPLANATIONS["tokens"]))
        parts.append(self._indent(self._render_token_table(analysis)))

        parts.append(
            self._section(
                "4. Vocabulary ids",
                _EXPLANATIONS["ids"].format(
                    unk=self._config.tokenizer.unknown_token,
                    bos=self._config.tokenizer.bos_token,
                    eos=self._config.tokenizer.eos_token,
                ),
            )
        )
        parts.append(self._indent(self._render_id_sequence(analysis)))

        parts.append(self._section("5. Decoded sentence", _EXPLANATIONS["decoded"]))
        parts.append(f"   {color.accent(analysis.decoded or '(empty)')}")
        parts.append(self._indent(self._render_round_trip(analysis)))

        parts.append(self._section("6. Statistics", _EXPLANATIONS["stats"]))
        parts.append(self._indent(self._render_statistics(analysis)))

        return "\n".join(parts)

    def _render_token_table(self, analysis: Analysis) -> str:
        """Build the token table.

        Combines the token list and its vocabulary ids into one table, since
        reading them side by side is the whole point — a separate list of
        bare ids would force the reader to count positions to match them up.

        Args:
            analysis: A completed analysis.

        Returns:
            The rendered table.
        """
        color = self._palette
        rows = [
            [
                str(view.position),
                view.token,
                str(view.token_id),
                color.good("yes") if view.known else color.warn("no  -> UNK"),
            ]
            for view in analysis.tokens
        ]
        return render_table(
            headers=["#", "Token", "Id", "In vocab?"],
            rows=rows,
            alignments=["right", "left", "right", "left"],
        )

    def _render_id_sequence(self, analysis: Analysis) -> str:
        """Render the id sequence with its special-token boundaries marked.

        Args:
            analysis: A completed analysis.

        Returns:
            Two lines: the raw sequence, and the same sequence as tokens.
        """
        color = self._palette
        vocabulary = self._tokenizer.vocabulary
        ids = ", ".join(str(token_id) for token_id in analysis.token_ids)
        names = " ".join(
            vocabulary.token_string(token_id) for token_id in analysis.token_ids
        )
        return "\n".join(
            [
                f"[{color.accent(ids)}]",
                color.muted(f"as tokens: {names}"),
                color.muted(
                    f"{len(analysis.token_ids)} ids for {len(analysis.tokens)} "
                    f"word(s) — the two extra are the boundary markers."
                ),
            ]
        )

    def _render_round_trip(self, analysis: Analysis) -> str:
        """Explain whether the decoded text matches the cleaned text.

        Args:
            analysis: A completed analysis.

        Returns:
            A short verdict line, with the reason when they differ.
        """
        color = self._palette
        if analysis.round_trip:
            return color.good("Round-trip exact: decoded text matches the cleaned text.")

        # The only way a word-level round-trip loses information is an
        # out-of-vocabulary token, which decodes to UNK and is then dropped.
        # Naming the cause turns a confusing mismatch into the lesson.
        if analysis.unknown_count:
            return color.warn(
                f"Round-trip lossy: {analysis.unknown_count} unknown word(s) "
                f"became {self._config.tokenizer.unknown_token} and were "
                "dropped when decoding."
            )
        return color.warn("Round-trip differs from the cleaned text.")

    def _render_statistics(self, analysis: Analysis) -> str:
        """Build the per-input statistics table.

        Args:
            analysis: A completed analysis.

        Returns:
            The rendered table.
        """
        token_count = len(analysis.tokens)
        unique = len({view.token for view in analysis.tokens})
        known = token_count - analysis.unknown_count
        lengths = [len(view.token) for view in analysis.tokens]

        rows = [
            ["Characters in", str(len(analysis.original))],
            ["Characters after cleaning", str(len(analysis.cleaned))],
            ["Tokens", str(token_count)],
            ["Unique tokens", str(unique)],
            ["Known tokens", f"{known} ({utils.percentage(known, token_count)}%)"],
            [
                "Unknown tokens",
                f"{analysis.unknown_count} "
                f"({utils.percentage(analysis.unknown_count, token_count)}%)",
            ],
            ["Average token length", f"{sum(lengths) / len(lengths):.2f} chars"],
            ["Longest token", f"{max(lengths)} chars"],
            ["Ids emitted", str(len(analysis.token_ids))],
            ["Vocabulary size", f"{self._tokenizer.vocabulary.size:,}"],
        ]
        return render_table(
            headers=["Metric", "Value"], rows=rows, alignments=["left", "right"]
        )

    def _active_cleaning_rules(self) -> list[str]:
        """List the cleaning rules currently switched on.

        Reads the live configuration rather than hardcoding a description,
        so the explanation stays true if someone edits ``default.yaml``.

        Returns:
            Human-readable rule names.
        """
        preprocessing = self._config.preprocessing
        rules = []
        if preprocessing.lowercase:
            rules.append("lowercase")
        if preprocessing.remove_punctuation:
            rules.append("strip punctuation")
        if preprocessing.collapse_whitespace:
            rules.append("collapse whitespace")
        return rules

    def _section(self, title: str, explanation: str) -> str:
        """Format a section heading with its explanation.

        Args:
            title: Section title.
            explanation: One-line description of what the stage did.

        Returns:
            The formatted heading block.
        """
        color = self._palette
        return f"\n{color.title(title)}\n   {color.muted(explanation)}"

    @staticmethod
    def _indent(block: str, spaces: int = 3) -> str:
        """Indent every line of a multi-line block.

        Args:
            block: Text to indent.
            spaces: Indent width.

        Returns:
            The indented block.
        """
        pad = " " * spaces
        return "\n".join(pad + line for line in block.splitlines())

    # -------------------------------------------------------------------
    # Session commands
    # -------------------------------------------------------------------

    def handle_command(self, command: str) -> bool:
        """Execute a colon command.

        Args:
            command: The raw command string, including the leading colon.

        Returns:
            True to continue the session, False to exit.
        """
        color = self._palette
        name = command.split()[0].lower()

        if name in (":quit", ":exit", ":q"):
            return False

        if name == ":help":
            rows = [[cmd, description] for cmd, description in _COMMANDS.items()]
            print()
            print(render_table(headers=["Command", "Effect"], rows=rows))
        elif name == ":stats":
            print()
            reporter = CorpusStatisticsReporter(top_k=100)
            stats = reporter.compute(
                self._documents, vocabulary_size=self._tokenizer.vocabulary.size
            )
            reporter.display(stats, top_display=15)
        elif name == ":vocab":
            self._show_vocabulary()
        elif name == ":examples":
            for sentence in _EXAMPLE_SENTENCES:
                print()
                print(color.muted(f"{_PROMPT}{sentence}"))
                print(self.render(self.analyze(sentence)))
        else:
            print(color.warn(f"Unknown command {name!r}. Try :help."))

        return True

    def _show_vocabulary(self, limit: int = 20) -> None:
        """Print the lowest-id vocabulary entries.

        Shows ids in ascending order because that ordering is meaningful:
        special tokens are assigned first, then corpus tokens by descending
        frequency, so the table doubles as a view of the ranking.

        Args:
            limit: How many entries to show.
        """
        vocabulary = self._tokenizer.vocabulary
        frequencies = vocabulary.frequencies
        specials = set(self._config.tokenizer.special_tokens)

        entries = sorted(vocabulary.token_to_id.items(), key=lambda kv: kv[1])[:limit]
        rows = [
            [
                str(token_id),
                token,
                "special" if token in specials else f"{frequencies.get(token, 0):,}",
            ]
            for token, token_id in entries
        ]
        print()
        print(
            render_table(
                headers=["Id", "Token", "Corpus count"],
                rows=rows,
                alignments=["right", "left", "right"],
                title=f"Vocabulary — first {len(rows)} of {vocabulary.size:,} entries",
            )
        )

    # -------------------------------------------------------------------
    # Session loop
    # -------------------------------------------------------------------

    def run(self, stream: Iterator[str] | None = None) -> int:
        """Run the session until the user leaves or input is exhausted.

        Args:
            stream: Optional iterator of input lines. When None, the session
                reads from stdin — prompting if it is a terminal, and
                consuming lines silently if it is a pipe.

        Returns:
            Process exit code.
        """
        interactive = stream is None and sys.stdin.isatty()
        if interactive:
            self._print_banner()

        lines = stream if stream is not None else self._read_stdin(interactive)

        try:
            for line in lines:
                text = line.strip()
                if not text:
                    continue
                if text.startswith(":"):
                    if not self.handle_command(text):
                        break
                    continue
                print(self.render(self.analyze(text)))
        except KeyboardInterrupt:
            # Ctrl-C during a session is a normal way to leave, not a crash.
            # Print a newline so the shell prompt does not land mid-line.
            print()
            logger.debug("Session interrupted by user")

        if interactive:
            print(
                self._palette.muted(
                    f"\nAnalyzed {self._analyzed} input(s). Goodbye."
                )
            )
        return 0

    def _read_stdin(self, interactive: bool) -> Iterator[str]:
        """Yield input lines from stdin.

        Args:
            interactive: Whether to show a prompt before each read.

        Yields:
            One input line at a time. Stops cleanly on EOF (Ctrl-D).
        """
        while True:
            try:
                yield input(self._palette.muted(_PROMPT) if interactive else "")
            except EOFError:
                if interactive:
                    print()
                return

    def _print_banner(self) -> None:
        """Print the welcome banner and setup summary."""
        color = self._palette
        vocabulary = self._tokenizer.vocabulary
        print()
        print(
            render_table(
                headers=["Setting", "Value"],
                rows=[
                    ["Project", self._config.project.name],
                    ["Corpus", str(self._config.paths.dataset.name)],
                    ["Documents", f"{len(self._documents):,}"],
                    ["Vocabulary size", f"{vocabulary.size:,}"],
                    ["Unknown token", self._config.tokenizer.unknown_token],
                ],
                alignments=["left", "right"],
                title=color.title("Tokenizer demo — ready"),
            )
        )
        print(
            color.muted(
                "\nType a sentence and press Enter to see it move through the "
                "pipeline.\nCommands start with a colon — :help lists them."
            )
        )


def _supports_color() -> bool:
    """Decide whether ANSI colors should be emitted.

    Returns:
        False when output is redirected, when ``NO_COLOR`` is set, or when
        ``TERM`` says the terminal is not capable. Writing escape codes into
        a redirected file corrupts it, so the default is conservative.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


def build_demo(
    config_path: Path,
    *,
    vocabulary_path: Path | None = None,
    palette: Palette | None = None,
) -> TokenizerDemo:
    """Load configuration and prepare a ready-to-run demo session.

    Args:
        config_path: Path to the YAML configuration file.
        vocabulary_path: Optional saved vocabulary to load instead of
            training a new one.
        palette: Optional color palette override.

    Returns:
        A ``TokenizerDemo`` with a trained or loaded tokenizer.

    Raises:
        FileNotFoundError: If the config, corpus, or vocabulary is missing.
        ValueError: If the corpus is empty or the configuration is invalid.
    """
    config = ConfigLoader(config_path).load()

    with utils.timed("load and clean corpus", level=logging.DEBUG):
        documents = DatasetLoader(config.paths.dataset).load()
        cleaned = TextCleaner(config.preprocessing).clean_corpus(documents)

    if vocabulary_path is not None:
        logger.info("Loading vocabulary from %s", vocabulary_path)
        tokenizer = Tokenizer.from_vocabulary_file(
            utils.validate_file_exists(vocabulary_path, "vocabulary"),
            config.tokenizer,
        )
    else:
        with utils.timed("train vocabulary", level=logging.DEBUG):
            tokenizer = Tokenizer(config.tokenizer)
            tokenizer.train(cleaned)

    return TokenizerDemo(config, tokenizer, cleaned, palette=palette)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Interactive terminal demo for the from-scratch tokenizer",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=None,
        help="Load a saved vocabulary instead of training a fresh one",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Analyze a single string and exit, instead of starting a session",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors (also honors the NO_COLOR variable)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for the demo.

    Args:
        argv: Optional CLI argument list, for testing.

    Returns:
        Process exit code: 0 on success, 1 on a handled failure.
    """
    args = parse_args(argv)

    # The demo's output *is* the interface, so logs stay at WARNING to keep
    # them from interleaving with the tables. Real failures still surface.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    palette = Palette(False) if args.no_color else Palette(_supports_color())

    try:
        demo = build_demo(
            args.config, vocabulary_path=args.vocabulary, palette=palette
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"Could not start the demo: {exc}", file=sys.stderr)
        return 1

    if args.text is not None:
        print(demo.render(demo.analyze(args.text)))
        return 0

    return demo.run()


if __name__ == "__main__":
    sys.exit(main())
