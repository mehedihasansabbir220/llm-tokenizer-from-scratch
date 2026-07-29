"""Publication-quality tokenizer visualizations built on matplotlib alone.

Produces five figures, saved into the project ``images/`` directory:

    1. vocabulary_frequency      -- rank/frequency (Zipf) curve, log-log
    2. sentence_length           -- histogram of tokens per document
    3. token_length              -- histogram of characters per token
    4. most_common_words         -- horizontal bar chart, top-K
    5. least_common_words        -- horizontal bar chart, bottom-K

Design rules applied throughout, so every figure reads as one system:

* **One hue, not a value ramp.** Words are nominal categories — coloring each
  bar darker-where-bigger would re-encode bar length as hue and spend the
  identity channel on information the chart already shows. Every mark takes
  the same blue.
* **Thin marks, recessive chrome.** Bars are capped in thickness so the band's
  leftover becomes air; gridlines and axes are solid hairlines one step off
  the surface, never dashed.
* **Rounded data-ends.** Bars grow from a square baseline to a rounded tip,
  which reads as a measured quantity rather than a block of ink.
* **Selective direct labels.** Bar charts label every tip (few enough bars
  that this is the clearest option, and it lets the value axis go away
  entirely); distribution charts label only the mean.
* **No legend.** Every figure plots a single series, so the title names it —
  a one-swatch legend box would only restate the title.
* **Both themes.** Light and dark are separately chosen steps validated
  against their own surface, not an automatic inversion.

Colors are the validated default palette: slot-1 blue ``#2a78d6`` on light and
``#3987e5`` on dark, each of which clears the lightness band, chroma floor, and
the 3:1 contrast gate against its own surface.

matplotlib only — no seaborn, and no plotting helpers from pandas.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

# Select the non-interactive Agg backend before pyplot is imported. This
# module's entire job is writing image files, and Agg is the backend that
# works without a display server — so this runs identically on a laptop, in
# CI, and inside a container.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

import utils  # noqa: E402

logger = logging.getLogger(__name__)

# Mark geometry, expressed in typographic points so the result is identical at
# any DPI. The design specs are given in CSS pixels at 96 dpi; 1 px = 0.75 pt.
_CORNER_RADIUS_PT = 3.0  # 4 px rounded data-end
_MAX_BAR_THICKNESS_PT = 18.0  # 24 px thickness cap
_SURFACE_GAP_PT = 1.5  # 2 px gap between adjacent bars
_LINE_WIDTH_PT = 2.0
_HAIRLINE_PT = 0.8

# Font stack. DejaVu Sans is last because it ships with matplotlib and is
# therefore the one entry guaranteed to resolve on any machine, which keeps
# the "findfont" warning out of the logs on systems without the others.
_FONT_STACK = [
    "SF Pro Text",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "Segoe UI",
    "DejaVu Sans",
]


@dataclass(frozen=True)
class ChartTheme:
    """Color and chrome tokens for one rendering mode.

    Each field is a role rather than a raw color, so a figure is written
    against roles and a whole theme swaps in one place.

    Attributes:
        name: Theme identifier, used in log messages and filename suffixes.
        surface: Chart surface (the plot background).
        page: Figure background behind the surface.
        ink_primary: Titles and emphasized text.
        ink_secondary: Subtitles, direct labels, annotations.
        ink_muted: Axis tick labels and captions.
        grid: Hairline gridlines.
        axis: Baseline and axis rules.
        series: The single series color carried by every mark.
    """

    name: str
    surface: str
    page: str
    ink_primary: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    series: str


# Light and dark are separately selected steps, each validated against its own
# surface — not one palette with the lightness flipped.
LIGHT_THEME = ChartTheme(
    name="light",
    surface="#fcfcfb",
    page="#f9f9f7",
    ink_primary="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series="#2a78d6",
)

DARK_THEME = ChartTheme(
    name="dark",
    surface="#1a1a19",
    page="#0d0d0d",
    ink_primary="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series="#3987e5",
)

THEMES: dict[str, ChartTheme] = {"light": LIGHT_THEME, "dark": DARK_THEME}


class TokenizerPlots:
    """Render and save the tokenizer figure set.

    Every method takes plain data — a frequency mapping or a list of
    documents — rather than a ``Vocabulary`` object, so the plots can be
    driven from a saved ``vocab.json``, from a live training run, from a
    notebook, or from a test with hand-written input.

    Each plotting method returns the path it wrote, so callers can log or
    assert on the result without guessing the filename.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        theme: ChartTheme | str = LIGHT_THEME,
        dpi: int = 200,
        figure_size: tuple[float, float] = (9.0, 5.5),
        file_format: str = "png",
    ) -> None:
        """Initialize the plotter.

        Args:
            output_dir: Directory to save figures into. Defaults to
                ``<project root>/images``, created if missing.
            theme: A ``ChartTheme`` or the name of one (``"light"`` /
                ``"dark"``).
            dpi: Output resolution. 200 is a good default for documents and
                slides; use 300 for print submission.
            figure_size: Figure size in inches (width, height).
            file_format: Output format — ``"png"`` for raster, or a vector
                format such as ``"pdf"`` or ``"svg"`` for print.

        Raises:
            ValueError: If ``theme`` names an unknown theme, or if ``dpi`` is
                not a positive integer.
        """
        if isinstance(theme, str):
            theme = THEMES[utils.validate_choice(theme, list(THEMES), "theme")]
        self._theme = theme
        self._dpi = utils.validate_positive_int(dpi, "dpi", minimum=1)
        self._figure_size = figure_size
        self._file_format = utils.validate_choice(
            file_format, ["png", "pdf", "svg"], "file_format"
        )

        root = Path(output_dir) if output_dir is not None else utils.project_root() / "images"
        self._output_dir = utils.ensure_directory(root)
        logger.info(
            "Plotter ready: theme=%s dpi=%d output=%s",
            self._theme.name,
            self._dpi,
            self._output_dir,
        )

    @property
    def output_dir(self) -> Path:
        """Return the directory figures are written to.

        Returns:
            Absolute path to the output directory.
        """
        return self._output_dir

    @property
    def theme(self) -> ChartTheme:
        """Return the active theme.

        Returns:
            The ``ChartTheme`` in use.
        """
        return self._theme

    # -------------------------------------------------------------------
    # Public plotting methods
    # -------------------------------------------------------------------

    def plot_vocabulary_frequency(
        self,
        frequencies: Mapping[str, int],
        *,
        filename: str = "vocabulary_frequency",
    ) -> Path:
        """Plot the rank/frequency (Zipf) curve of the vocabulary.

        Why log-log:
            Natural-language frequencies span several orders of magnitude —
            the top token can be thousands of times more common than the
            median. On linear axes the curve collapses onto the axes and
            shows nothing. On log-log, Zipf's law appears as a near-straight
            line, so a glance tells you whether the corpus behaves like
            natural text or is skewed by boilerplate.

        Args:
            frequencies: Mapping of token to corpus count.
            filename: Output filename stem, without extension.

        Returns:
            Path to the written figure.

        Raises:
            ValueError: If ``frequencies`` is empty.
        """
        utils.validate_not_empty(frequencies, "frequencies")

        counts = sorted(frequencies.values(), reverse=True)
        ranks = range(1, len(counts) + 1)

        fig, ax = self._new_figure()
        ax.plot(
            list(ranks),
            counts,
            color=self._theme.series,
            linewidth=_LINE_WIDTH_PT,
            solid_joinstyle="round",
            solid_capstyle="round",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Breathing room on all four sides. Without it the curve starts flush
        # against the left spine and the annotation on the first point gets
        # clipped by it. Multiplicative padding, since the scale is log.
        ax.set_xlim(0.85, len(counts) * 1.2)
        ax.set_ylim(min(counts) * 0.8, counts[0] * 1.45)

        self._style_axes(
            ax,
            title="Vocabulary frequency distribution",
            subtitle="Token rank against corpus frequency, log-log. A straight "
            "line indicates Zipfian behavior.",
            xlabel="Token rank (most frequent first)",
            ylabel="Occurrences",
            grid_axis="both",
        )

        # Direct-label only the single most frequent token. Labeling more
        # would crowd the steep left shoulder where points sit closest.
        top_token, top_count = max(frequencies.items(), key=lambda kv: (kv[1], kv[0]))
        ax.annotate(
            f"{top_token!r} — {top_count:,}",
            xy=(1, counts[0]),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=9.5,
            color=self._theme.ink_secondary,
            va="bottom",
            ha="left",
        )

        return self._save(
            fig,
            filename,
            caption=f"{len(counts):,} unique tokens · "
            f"{sum(counts):,} total occurrences",
        )

    def plot_sentence_length(
        self,
        documents: Iterable[str],
        *,
        max_bins: int = 40,
        filename: str = "sentence_length",
    ) -> Path:
        """Plot a histogram of sentence (document) lengths in tokens.

        Why this matters:
            Sequence-length distribution drives padding and truncation
            choices downstream. A long right tail means a fixed max length
            will either truncate real content or waste compute on padding.

        Args:
            documents: Iterable of cleaned document strings.
            max_bins: Upper bound on histogram bin count. Bins are always
                integer-aligned, since a document length is a whole number of
                tokens; this only caps how many of them there are.
            filename: Output filename stem, without extension.

        Returns:
            Path to the written figure.

        Raises:
            ValueError: If no document contains any token.
        """
        lengths = [len(doc.split()) for doc in documents]
        lengths = [n for n in lengths if n > 0]
        utils.validate_not_empty(lengths, "document lengths")

        fig, ax = self._new_figure()
        self._draw_histogram(ax, lengths, max_bins=max_bins)

        summary = utils.describe_numbers(lengths)
        self._style_axes(
            ax,
            title="Sentence length distribution",
            subtitle="Tokens per document after cleaning.",
            xlabel="Tokens per document",
            ylabel="Documents",
            grid_axis="y",
        )
        self._annotate_mean(ax, summary["mean"], unit="tokens")

        return self._save(
            fig,
            filename,
            caption=f"n = {summary['count']:,} documents · "
            f"median {summary['median']:.0f} · "
            f"range {summary['minimum']:.0f}–{summary['maximum']:.0f} tokens",
        )

    def plot_token_length(
        self,
        frequencies: Mapping[str, int],
        *,
        weighted: bool = True,
        filename: str = "token_length",
    ) -> Path:
        """Plot a histogram of token lengths in characters.

        Args:
            frequencies: Mapping of token to corpus count.
            weighted: When True, each token contributes once per occurrence,
                describing the text as read. When False, each distinct token
                contributes once, describing the vocabulary as stored. The
                two differ sharply: short function words dominate the
                weighted view and barely register in the unweighted one.
            filename: Output filename stem, without extension.

        Returns:
            Path to the written figure.

        Raises:
            ValueError: If ``frequencies`` is empty.
        """
        utils.validate_not_empty(frequencies, "frequencies")

        if weighted:
            lengths = [
                len(token) for token, count in frequencies.items() for _ in range(count)
            ]
            basis = "Weighted by corpus frequency — every occurrence counts."
        else:
            lengths = [len(token) for token in frequencies]
            basis = "One count per distinct token, regardless of frequency."

        fig, ax = self._new_figure()
        self._draw_histogram(ax, lengths)

        summary = utils.describe_numbers(lengths)
        self._style_axes(
            ax,
            title="Token length distribution",
            subtitle=basis,
            xlabel="Characters per token",
            ylabel="Occurrences" if weighted else "Distinct tokens",
            grid_axis="y",
        )
        self._annotate_mean(ax, summary["mean"], unit="chars")

        return self._save(
            fig,
            filename,
            caption=f"n = {summary['count']:,} · "
            f"median {summary['median']:.0f} · "
            f"longest {summary['maximum']:.0f} characters",
        )

    def plot_most_common_words(
        self,
        frequencies: Mapping[str, int],
        *,
        top_k: int = 15,
        filename: str = "most_common_words",
    ) -> Path:
        """Plot the most frequent tokens as a horizontal bar chart.

        Why horizontal:
            Word labels are long and of uneven length. On a vertical column
            chart they would need rotating, which is markedly slower to read;
            horizontal bars let every label sit on its own baseline.

        Args:
            frequencies: Mapping of token to corpus count.
            top_k: How many tokens to show.
            filename: Output filename stem, without extension.

        Returns:
            Path to the written figure.

        Raises:
            ValueError: If ``frequencies`` is empty or ``top_k`` < 1.
        """
        return self._plot_word_ranking(
            frequencies,
            count=top_k,
            rarest=False,
            title="Most common words",
            subtitle=f"Top {top_k} tokens by corpus frequency.",
            filename=filename,
        )

    def plot_least_common_words(
        self,
        frequencies: Mapping[str, int],
        *,
        bottom_k: int = 15,
        filename: str = "least_common_words",
    ) -> Path:
        """Plot the least frequent tokens as a horizontal bar chart.

        Why this is worth plotting:
            The rare tail is where vocabulary quality problems live —
            typos, OCR noise, and tokenization artifacts all surface here.
            A tail that is entirely count-1 also tells you the minimum
            frequency threshold is doing nothing.

        Args:
            frequencies: Mapping of token to corpus count.
            bottom_k: How many tokens to show.
            filename: Output filename stem, without extension.

        Returns:
            Path to the written figure.

        Raises:
            ValueError: If ``frequencies`` is empty or ``bottom_k`` < 1.
        """
        return self._plot_word_ranking(
            frequencies,
            count=bottom_k,
            rarest=True,
            title="Least common words",
            subtitle=f"Rarest {bottom_k} tokens by corpus frequency; "
            "ties broken alphabetically.",
            filename=filename,
        )

    def plot_all(
        self,
        frequencies: Mapping[str, int],
        documents: Sequence[str] | None = None,
        *,
        top_k: int = 15,
    ) -> dict[str, Path]:
        """Render the complete figure set in one call.

        Args:
            frequencies: Mapping of token to corpus count.
            documents: Optional cleaned documents. Without them the sentence
                length figure is skipped, since document boundaries cannot be
                recovered from a frequency table alone.
            top_k: Bar count for the two ranking charts.

        Returns:
            Mapping of figure name to the path written. Absent keys indicate
            skipped figures.
        """
        with utils.timed("render all tokenizer figures"):
            written: dict[str, Path] = {
                "vocabulary_frequency": self.plot_vocabulary_frequency(frequencies),
                "token_length": self.plot_token_length(frequencies),
                "most_common_words": self.plot_most_common_words(
                    frequencies, top_k=top_k
                ),
                "least_common_words": self.plot_least_common_words(
                    frequencies, bottom_k=top_k
                ),
            }
            if documents:
                written["sentence_length"] = self.plot_sentence_length(documents)
            else:
                logger.warning(
                    "No documents supplied; skipping the sentence length figure"
                )

        logger.info("Wrote %d figure(s) to %s", len(written), self._output_dir)
        return written

    @classmethod
    def from_vocabulary_file(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> tuple[TokenizerPlots, dict[str, int]]:
        """Build a plotter and load frequencies from a saved vocabulary.

        Reads the ``frequencies`` block written by ``Vocabulary.save()`` and
        ``VocabularyTrainer.save()``.

        Args:
            path: Path to a vocabulary JSON file.
            **kwargs: Forwarded to ``__init__``.

        Returns:
            Tuple of ``(plotter, frequencies)``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file has no usable ``frequencies`` block.
        """
        payload = utils.read_json(utils.validate_file_exists(path, "vocabulary"))
        raw = payload.get("frequencies") if isinstance(payload, dict) else None
        if not raw:
            raise ValueError(f"No 'frequencies' block found in {path}")

        frequencies = {str(token): int(count) for token, count in raw.items()}
        logger.info("Loaded %d token frequencies from %s", len(frequencies), path)
        return cls(**kwargs), frequencies

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _plot_word_ranking(
        self,
        frequencies: Mapping[str, int],
        *,
        count: int,
        rarest: bool,
        title: str,
        subtitle: str,
        filename: str,
    ) -> Path:
        """Render a horizontal bar chart of ranked tokens.

        Shared by the most-common and least-common charts, which differ only
        in sort direction and copy.

        Args:
            frequencies: Mapping of token to corpus count.
            count: Number of bars.
            rarest: When True, select the lowest counts instead of highest.
            title: Chart title.
            subtitle: Chart subtitle.
            filename: Output filename stem.

        Returns:
            Path to the written figure.

        Raises:
            ValueError: If ``frequencies`` is empty or ``count`` < 1.
        """
        utils.validate_not_empty(frequencies, "frequencies")
        utils.validate_positive_int(count, "count", minimum=1)

        # Sort by count, then alphabetically. The secondary key is what makes
        # the output reproducible: without it, tokens sharing a count (very
        # common in the rare tail, where nearly everything appears once)
        # would come out in dict order and shuffle between runs.
        ordered = sorted(frequencies.items(), key=lambda kv: (kv[1], kv[0]))
        # Both charts put their headline entry at the top, where the eye lands
        # first: the most frequent token for one, the rarest for the other.
        # matplotlib's y-axis grows upward, so the headline entry has to be
        # last in the list to be drawn topmost — hence the reversal for the
        # rarest case, whose slice already starts at the extreme.
        selected = list(reversed(ordered[:count])) if rarest else ordered[-count:]
        tokens = [token for token, _ in selected]
        values = [value for _, value in selected]

        # Give each bar a fixed vertical allowance so a 15-bar chart and a
        # 30-bar chart have the same bar rhythm rather than the same height.
        height = max(3.0, 0.34 * len(tokens) + 1.9)
        fig, ax = self._new_figure(size=(self._figure_size[0], height))

        positions = list(range(len(tokens)))
        ax.set_yticks(positions)
        ax.set_yticklabels(tokens)
        ax.set_ylim(-0.6, len(tokens) - 0.4)
        # Headroom so the value labels at the tips are not pressed against
        # the right edge of the plot.
        ax.set_xlim(0, max(values) * 1.14)

        # Limits must be final before the bars are drawn: bar thickness and
        # corner radius are specified in points and converted through the
        # axes transform, which is only meaningful once the data range is set.
        thickness = self._bar_thickness(ax, desired=0.62, orientation="horizontal")
        self._draw_rounded_bars(ax, positions, values, thickness=thickness)

        self._style_axes(
            ax,
            title=title,
            subtitle=subtitle,
            xlabel=None,
            ylabel=None,
            grid_axis="none",
        )
        # Every bar is directly labeled, so the value axis is redundant ink.
        ax.get_xaxis().set_visible(False)
        ax.spines["bottom"].set_visible(False)

        for y, value in zip(positions, values):
            ax.annotate(
                f"{value:,}",
                xy=(value, y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9.5,
                color=self._theme.ink_secondary,
            )

        total = sum(frequencies.values())
        shown = sum(values)
        return self._save(
            fig,
            filename,
            caption=f"{len(tokens)} of {len(frequencies):,} tokens · "
            f"{utils.percentage(shown, total)}% of all occurrences",
        )

    def _new_figure(
        self, size: tuple[float, float] | None = None
    ) -> tuple[Figure, Any]:
        """Create a themed figure and axes.

        Args:
            size: Optional figure size override in inches.

        Returns:
            Tuple of ``(figure, axes)``.
        """
        # constrained_layout sizes the axes to fit titles, labels, and the
        # caption, so nothing is clipped and no manual margin tuning is
        # needed when label lengths change with the data.
        fig, ax = plt.subplots(
            figsize=size or self._figure_size,
            dpi=self._dpi,
            layout="constrained",
        )
        fig.patch.set_facecolor(self._theme.page)
        ax.set_facecolor(self._theme.surface)
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = _FONT_STACK
        return fig, ax

    def _style_axes(
        self,
        ax: Any,
        *,
        title: str,
        subtitle: str,
        xlabel: str | None,
        ylabel: str | None,
        grid_axis: str,
    ) -> None:
        """Apply shared chrome: titles, labels, grid, spines, and ticks.

        Centralizing this is what makes the figures look like one family
        rather than five charts that happen to share a color.

        Args:
            ax: Axes to style.
            title: Main title, left-aligned above the plot.
            subtitle: Supporting line beneath the title.
            xlabel: X-axis label, or None to omit.
            ylabel: Y-axis label, or None to omit.
            grid_axis: ``"x"``, ``"y"``, ``"both"``, or ``"none"``.
        """
        theme = self._theme

        # Title sits above the subtitle; the generous pad is what reserves
        # room for the subtitle drawn just above the axes.
        ax.set_title(
            title,
            loc="left",
            fontsize=14,
            # "bold" rather than "semibold": DejaVu Sans (matplotlib's
            # guaranteed fallback) ships only normal and bold, and asking for
            # a weight the face lacks emits a findfont warning on every call.
            fontweight="bold",
            color=theme.ink_primary,
            pad=30,
        )
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            fontsize=10,
            color=theme.ink_secondary,
            va="bottom",
            ha="left",
        )

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, color=theme.ink_secondary, labelpad=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, color=theme.ink_secondary, labelpad=8)

        if grid_axis != "none":
            # Solid hairlines, never dashed: dashing reads as "projection" or
            # "threshold" when it is only a grid.
            ax.grid(
                axis=grid_axis,
                color=theme.grid,
                linewidth=_HAIRLINE_PT,
                linestyle="-",
            )
            ax.set_axisbelow(True)

        # Drop the top and right spines; keep the two that carry the scale.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(theme.axis)
            ax.spines[side].set_linewidth(_HAIRLINE_PT)

        ax.tick_params(
            colors=theme.ink_muted,
            labelsize=9.5,
            length=0,  # Tick marks add ink without adding information; the
            pad=6,     # gridline already says where the value sits.
        )

    def _draw_histogram(
        self,
        ax: Any,
        values: Sequence[int],
        *,
        max_bins: int = 40,
    ) -> None:
        """Draw a themed histogram with rounded column caps.

        Both quantities plotted by this module — tokens per document and
        characters per token — are *counts*, so the bins are integer-aligned
        and centered on the value they represent. Letting a generic binning
        strategy choose the edges would place boundaries at fractional values
        like 7.5 characters, which is meaningless, and would leave alternating
        empty bins whenever the chosen width fell below 1.

        Args:
            ax: Axes to draw on.
            values: Raw integer observations.
            max_bins: Upper bound on bin count. When the data range exceeds
                it, bins widen to whole numbers greater than 1.
        """
        edges = _integer_bin_edges(values, max_bins=max_bins)
        counts, edges = _histogram(values, edges)

        width = edges[1] - edges[0]
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]

        # Limits before marks — the point-to-data conversion inside
        # _bar_thickness and _draw_rounded_bars depends on the final range.
        ax.set_xlim(centers[0] - width, centers[-1] + width)
        ax.set_ylim(0, max(counts) * 1.12 if max(counts) else 1)

        # A 2px surface gap separates touching columns; white does the
        # separating, never a stroke drawn around each bar.
        #
        # Note the absence of the thickness cap that categorical bars get. In
        # a histogram the bar's width *is* the bin interval — it carries
        # meaning — so capping it would draw bins narrower than they are and
        # imply gaps in coverage that the binning does not have.
        gap, _ = _radius_in_data_units(ax, _SURFACE_GAP_PT)
        self._draw_rounded_bars(
            ax,
            centers,
            list(counts),
            thickness=max(width - gap, width * 0.5),
            orientation="vertical",
        )

    def _draw_rounded_bars(
        self,
        ax: Any,
        positions: Sequence[float],
        values: Sequence[float],
        *,
        thickness: float,
        orientation: str = "horizontal",
    ) -> None:
        """Draw bars with a square baseline and a rounded data-end.

        matplotlib's ``bar``/``barh`` draw plain rectangles, so the rounded
        tip is built as an explicit path. The corner radius is specified in
        points and converted into data units separately for each axis, so the
        corner stays visually circular no matter how the two scales differ.

        Args:
            ax: Axes to draw on.
            positions: Category positions along the non-value axis.
            values: Bar magnitudes.
            thickness: Bar thickness in data units of the category axis.
            orientation: ``"horizontal"`` (grows right) or ``"vertical"``
                (grows up).
        """
        radius_x, radius_y = _radius_in_data_units(ax, _CORNER_RADIUS_PT)

        for position, value in zip(positions, values):
            if value <= 0:
                # A zero-height bar has no tip to round, and a negative one
                # would invert the path. Skip rather than draw a glitch.
                continue
            path = _rounded_bar_path(
                position=position,
                value=value,
                thickness=thickness,
                radius_x=radius_x,
                radius_y=radius_y,
                orientation=orientation,
            )
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor=self._theme.series,
                    edgecolor="none",
                    linewidth=0,
                )
            )

        # Deliberately no set_xlim/set_ylim here: add_patch does not grow the
        # data limits, but the caller has already set them — it had to, since
        # thickness and radius were computed from that range.

    def _bar_thickness(
        self, ax: Any, *, desired: float, orientation: str
    ) -> float:
        """Compute bar thickness in data units, capped at the design maximum.

        Why cap it:
            Letting bars fill their whole band produces a solid block of ink.
            Capping the thickness leaves the leftover as air, which is what
            keeps a dense chart reading as light. The cap is absolute (a
            length in points), so a chart with three bars does not get bars
            five times fatter than one with fifteen.

        The axes limits must already be final when this is called — the
        point-to-data conversion is meaningless against a default 0-1 range.

        Args:
            ax: Axes the bars will be drawn on.
            desired: Preferred thickness in data units of the category axis.
            orientation: ``"horizontal"`` or ``"vertical"``, selecting which
                axis the thickness is measured along.

        Returns:
            Thickness in data units, never above the point-based cap.
        """
        cap_x, cap_y = _radius_in_data_units(ax, _MAX_BAR_THICKNESS_PT)
        cap = cap_y if orientation == "horizontal" else cap_x
        thickness = min(desired, cap) if cap > 0 else desired
        logger.debug(
            "Bar thickness: desired=%.3f cap=%.3f -> %.3f data units",
            desired,
            cap,
            thickness,
        )
        return thickness

    def _annotate_mean(self, ax: Any, mean: float, *, unit: str) -> None:
        """Draw a reference line at the mean with a single direct label.

        Distribution charts get exactly one direct label — the mean. Labeling
        every bar would flood the chart and go unread; the axis carries the
        rest.

        Args:
            ax: Axes to annotate.
            mean: Mean value along the x-axis.
            unit: Unit name for the label, e.g. ``"tokens"``.
        """
        ax.axvline(
            mean,
            color=self._theme.ink_secondary,
            linewidth=1.2,
            linestyle="-",
            zorder=3,
        )
        ax.annotate(
            f"mean {mean:.1f} {unit}",
            xy=(mean, ax.get_ylim()[1]),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=9.5,
            color=self._theme.ink_secondary,
            va="top",
            ha="left",
        )

    def _save(self, fig: Figure, stem: str, *, caption: str = "") -> Path:
        """Write a figure to the output directory and close it.

        Args:
            fig: Figure to save.
            stem: Filename stem, without extension or theme suffix.
            caption: Optional footnote describing the sample.

        Returns:
            Path to the written file.

        Raises:
            OSError: If the file cannot be written.
        """
        if caption:
            # Reserve a strip at the bottom before placing the caption.
            # constrained_layout fits the axes to the rect it is given and
            # knows nothing about loose figure text, so without this the
            # caption would sit underneath the lowest axis label.
            engine = fig.get_layout_engine()
            if engine is not None:
                engine.set(rect=(0.0, 0.045, 1.0, 0.955))
            fig.text(
                0.008,
                0.012,
                caption,
                fontsize=8.5,
                color=self._theme.ink_muted,
                ha="left",
                va="bottom",
            )

        # The theme is part of the filename so light and dark renders of the
        # same chart can sit side by side without overwriting each other.
        suffix = "" if self._theme.name == "light" else f"_{self._theme.name}"
        destination = self._output_dir / f"{stem}{suffix}.{self._file_format}"

        try:
            fig.savefig(
                destination,
                dpi=self._dpi,
                facecolor=fig.get_facecolor(),
                format=self._file_format,
            )
        except OSError:
            logger.exception("Failed to save figure to %s", destination)
            raise
        finally:
            # Always close: pyplot keeps a global reference to every open
            # figure, so a loop that renders many charts without closing will
            # grow until the process runs out of memory.
            plt.close(fig)

        logger.info(
            "Saved %s (%s)",
            destination,
            utils.human_readable_size(destination.stat().st_size),
        )
        return destination


# =============================================================================
# Module-level geometry helpers
# =============================================================================


def _integer_bin_edges(values: Sequence[int], *, max_bins: int = 40) -> list[float]:
    """Build integer-aligned histogram edges centered on whole values.

    Edges are offset by half a unit (…, 6.5, 7.5, 8.5, …) so that each bin's
    midpoint — where the bar is drawn — lands exactly on an integer and
    therefore exactly on its axis tick.

    Args:
        values: Integer observations.
        max_bins: Upper bound on bin count. If the span exceeds it, the bin
            width grows to the smallest whole number that fits.

    Returns:
        List of bin edges, always at least two.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError("Cannot bin an empty series")

    low, high = int(min(values)), int(max(values))
    span = high - low + 1
    # Ceiling division without importing math, and never below 1 — a
    # fractional width would reintroduce the meaningless-boundary problem.
    width = max(1, -(-span // max_bins))
    bin_count = -(-span // width)
    start = low - 0.5
    return [start + index * width for index in range(bin_count + 1)]


def _histogram(
    values: Sequence[float], edges: Sequence[float]
) -> tuple[list[int], list[float]]:
    """Bin values into counts against explicit edges.

    Args:
        values: Raw observations.
        edges: Bin edges, as produced by ``_integer_bin_edges``.

    Returns:
        Tuple of ``(counts, edges)``.
    """
    # numpy arrives as a matplotlib dependency, so using it here costs nothing
    # extra at install time.
    import numpy as np

    counts, computed_edges = np.histogram(values, bins=edges)
    return counts.tolist(), computed_edges.tolist()


def _radius_in_data_units(ax: Any, points: float) -> tuple[float, float]:
    """Convert a length in typographic points into x and y data units.

    Args:
        ax: Axes whose transform defines the mapping.
        points: Length in points (1/72 inch).

    Returns:
        Tuple of ``(x_units, y_units)``. Returns ``(0.0, 0.0)`` if the axes
        transform is degenerate, which makes callers fall back to square
        corners rather than raising.
    """
    pixels = points * ax.figure.dpi / 72.0
    try:
        # The transform is only correct once a layout pass has run, since the
        # axes box is not final before that.
        ax.figure.canvas.draw()
        inverse = ax.transData.inverted()
        origin_x, origin_y = inverse.transform((0.0, 0.0))
        offset_x, offset_y = inverse.transform((pixels, pixels))
    except Exception:  # pragma: no cover - only on a degenerate transform
        logger.debug("Could not invert axes transform; using square corners")
        return 0.0, 0.0
    return abs(offset_x - origin_x), abs(offset_y - origin_y)


def _rounded_bar_path(
    *,
    position: float,
    value: float,
    thickness: float,
    radius_x: float,
    radius_y: float,
    orientation: str,
) -> MplPath:
    """Build the outline of a bar with a square base and a rounded tip.

    The rounding uses quadratic Bezier corners with the control point at the
    sharp corner — visually indistinguishable from a true quarter-circle at
    a 3pt radius, and far cheaper to construct than an arc approximation.

    Args:
        position: Center of the bar along the category axis.
        value: Bar magnitude along the value axis.
        thickness: Bar thickness in category-axis data units.
        radius_x: Corner radius in x data units.
        radius_y: Corner radius in y data units.
        orientation: ``"horizontal"`` or ``"vertical"``.

    Returns:
        A closed ``matplotlib.path.Path`` outlining the bar.
    """
    half = thickness / 2.0

    if orientation == "horizontal":
        low, high = position - half, position + half
        # Never let the radius exceed half the bar's thickness or its whole
        # length, or the corners would overlap and turn the tip inside out.
        r_x = min(radius_x, value)
        r_y = min(radius_y, half)
        vertices = [
            (0.0, low),
            (value - r_x, low),
            (value, low),           # control
            (value, low + r_y),
            (value, high - r_y),
            (value, high),          # control
            (value - r_x, high),
            (0.0, high),
            (0.0, low),
        ]
    else:
        low, high = position - half, position + half
        r_x = min(radius_x, half)
        r_y = min(radius_y, value)
        vertices = [
            (low, 0.0),
            (low, value - r_y),
            (low, value),           # control
            (low + r_x, value),
            (high - r_x, value),
            (high, value),          # control
            (high, value - r_y),
            (high, 0.0),
            (low, 0.0),
        ]

    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(vertices, codes)


def token_frequencies_from_documents(documents: Iterable[str]) -> dict[str, int]:
    """Count token frequencies from raw documents.

    A convenience for driving the plots directly from a corpus when no
    trained vocabulary is at hand. Splits on whitespace, matching the
    project's word-level tokenizer.

    Args:
        documents: Iterable of cleaned document strings.

    Returns:
        Mapping of token to occurrence count.

    Raises:
        ValueError: If the documents contain no tokens.
    """
    counter: Counter[str] = Counter()
    for document in documents:
        counter.update(document.split())
    utils.validate_not_empty(counter, "token frequencies")
    logger.debug("Counted %d unique token(s) from documents", len(counter))
    return dict(counter)
