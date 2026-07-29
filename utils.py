"""Reusable helper functions shared across the tokenizer pipeline.

This module is deliberately dependency-free with respect to the rest of the
project: nothing here imports ``configs``, ``tokenizer``, ``preprocessing``,
or any other package. That one-way rule is what keeps these helpers reusable
— any module may import ``utils``, and ``utils`` will never import back and
create a cycle.

Helpers are grouped into six sections:

1. Path utilities        -- locating, resolving, and creating paths
2. File I/O              -- reading and writing text and JSON safely
3. Timing                -- measuring how long an operation takes
4. Statistics            -- summarizing numeric series and frequency tables
5. Validation            -- failing fast with clear, consistent messages
6. Formatting            -- human-readable sizes and durations

Every function is a plain module-level function with no hidden state, so it
can be called from any module, a notebook, or a test without setup.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import statistics as _stats
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# Default text encoding for the whole project. Centralized here so that a
# future change (say, to UTF-8 with BOM tolerance) happens in exactly one
# place rather than in every open() call across the codebase.
DEFAULT_ENCODING = "utf-8"

# Unit ladder for human_readable_size. Binary units (1024-based) are used
# because they match what `ls -lh` and most tooling report.
_SIZE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")

# Generic type variable so validators can return the value they checked
# without erasing its static type at the call site.
T = TypeVar("T")


# =============================================================================
# 1. Path utilities
# =============================================================================


def project_root() -> Path:
    """Return the absolute path of the project root directory.

    Why this exists:
        Code that builds paths from the current working directory breaks the
        moment it is run from a subdirectory, a notebook, or a test runner.
        Anchoring to this file's own location makes paths independent of
        where the process was started.

    Returns:
        Absolute path to the directory containing this file.
    """
    return Path(__file__).resolve().parent


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a possibly-relative path against a base directory.

    Absolute paths are returned unchanged, which lets callers accept either
    form from configuration files or CLI arguments without branching.

    Args:
        path: Path to resolve. May be relative or absolute.
        base: Directory to resolve against. Defaults to the project root.

    Returns:
        An absolute, normalized ``Path``.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    anchor = Path(base) if base is not None else project_root()
    resolved = (anchor / candidate).resolve()
    logger.debug("Resolved %s -> %s", path, resolved)
    return resolved


def ensure_directory(path: str | Path) -> Path:
    """Create a directory (and any missing parents) if it does not exist.

    Why this exists:
        Every save step in the pipeline needs its destination directory to
        exist. Doing this in one helper keeps the ``parents=True,
        exist_ok=True`` pair from being retyped — and occasionally
        mistyped — at each call site.

    Args:
        path: Directory path to create.

    Returns:
        The directory path, for convenient chaining.

    Raises:
        OSError: If the directory cannot be created.
    """
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Failed to create directory %s", directory)
        raise
    logger.debug("Ensured directory exists: %s", directory)
    return directory


def ensure_parent_directory(path: str | Path) -> Path:
    """Create the parent directory of a *file* path.

    The file-path counterpart of ``ensure_directory``. Call this before
    writing a file whose containing folder may not exist yet.

    Args:
        path: File path whose parent should be created.

    Returns:
        The original file path, for convenient chaining.

    Raises:
        OSError: If the parent directory cannot be created.
    """
    file_path = Path(path)
    ensure_directory(file_path.parent)
    return file_path


def list_files(
    directory: str | Path,
    pattern: str = "*",
    *,
    recursive: bool = False,
) -> list[Path]:
    """List files in a directory matching a glob pattern.

    Results are sorted. Sorting matters more than it looks: filesystem
    iteration order is not guaranteed, so unsorted discovery makes any
    downstream result that depends on file order irreproducible.

    Args:
        directory: Directory to search.
        pattern: Glob pattern, e.g. ``"*.txt"``.
        recursive: Whether to descend into subdirectories.

    Returns:
        Sorted list of matching file paths. Directories are excluded.

    Raises:
        FileNotFoundError: If ``directory`` does not exist.
        NotADirectoryError: If ``directory`` is not a directory.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    globber = root.rglob if recursive else root.glob
    matches = sorted(item for item in globber(pattern) if item.is_file())
    logger.debug("Found %d file(s) matching %r in %s", len(matches), pattern, root)
    return matches


def file_size(path: str | Path) -> int:
    """Return the size of a file in bytes.

    Args:
        path: File to measure.

    Returns:
        Size in bytes.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be stat'ed.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path.stat().st_size


# =============================================================================
# 2. File I/O
# =============================================================================


def read_text_file(
    path: str | Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    errors: str = "strict",
) -> str:
    """Read an entire text file into a single string.

    Use this for small-to-moderate files such as configs and templates. For
    corpus-sized files, prefer ``stream_lines`` so the whole file never has
    to fit in memory at once.

    Args:
        path: File to read.
        encoding: Text encoding. Defaults to UTF-8.
        errors: Decode error policy — ``"strict"``, ``"replace"``, or
            ``"ignore"``.

    Returns:
        The file contents as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnicodeDecodeError: If ``errors="strict"`` and the bytes are invalid.
        OSError: If the file cannot be read.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        content = file_path.read_text(encoding=encoding, errors=errors)
    except (OSError, UnicodeDecodeError):
        logger.exception("Failed to read text file %s", file_path)
        raise

    logger.info("Read %d character(s) from %s", len(content), file_path)
    return content


def read_lines(
    path: str | Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    errors: str = "strict",
    skip_blank: bool = True,
    strip: bool = True,
) -> list[str]:
    """Read a text file into a list of lines.

    Args:
        path: File to read.
        encoding: Text encoding.
        errors: Decode error policy.
        skip_blank: Whether to drop lines that are empty after stripping.
        strip: Whether to strip surrounding whitespace from each line. This
            also normalizes Windows CRLF endings.

    Returns:
        List of lines.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
    """
    lines = list(
        stream_lines(
            path,
            encoding=encoding,
            errors=errors,
            skip_blank=skip_blank,
            strip=strip,
        )
    )
    logger.info("Read %d line(s) from %s", len(lines), Path(path))
    return lines


def stream_lines(
    path: str | Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    errors: str = "strict",
    skip_blank: bool = True,
    strip: bool = True,
) -> Iterator[str]:
    """Yield lines from a text file one at a time.

    Why this exists alongside ``read_lines``:
        This is the memory-safe variant. Peak memory is a single line rather
        than the whole file, so it works on corpora larger than RAM. The
        file handle stays open until the iterator is exhausted or closed, so
        consume it promptly rather than storing it for later.

    Args:
        path: File to read.
        encoding: Text encoding.
        errors: Decode error policy.
        skip_blank: Whether to skip lines that are empty after stripping.
        strip: Whether to strip surrounding whitespace from each line.

    Yields:
        One line of text at a time.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.debug("Streaming lines from %s", file_path)
    with file_path.open("r", encoding=encoding, errors=errors) as handle:
        for line in handle:
            text = line.strip() if strip else line
            if skip_blank and not text.strip():
                continue
            yield text


def write_text_file(
    content: str,
    path: str | Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    atomic: bool = True,
) -> Path:
    """Write a string to a text file, creating parent directories.

    Args:
        content: Text to write.
        path: Destination file path.
        encoding: Text encoding.
        atomic: When True, write to a temporary file in the same directory
            and rename it into place, so an interrupted write cannot leave a
            truncated file behind. Set to False only if the destination is a
            special file (a pipe or device) that cannot be renamed over.

    Returns:
        The destination path.

    Raises:
        OSError: If the file cannot be written.
    """
    destination = ensure_parent_directory(path)
    _write_atomically_or_directly(
        destination,
        lambda handle: handle.write(content),
        encoding=encoding,
        atomic=atomic,
    )
    logger.info("Wrote %d character(s) to %s", len(content), destination)
    return destination


def read_json(path: str | Path, *, encoding: str = DEFAULT_ENCODING) -> Any:
    """Read and parse a JSON file.

    Args:
        path: JSON file to read.
        encoding: Text encoding.

    Returns:
        The parsed Python object — usually a dict or list.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain valid JSON. The underlying
            ``json.JSONDecodeError`` is a subclass of ``ValueError``; it is
            re-raised with the offending path in the message, because the
            default error text names neither the file nor its location.
        OSError: If the file cannot be read.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        with file_path.open("r", encoding=encoding) as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", file_path, exc)
        raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc
    except OSError:
        logger.exception("Failed to read JSON file %s", file_path)
        raise

    logger.info("Loaded JSON from %s", file_path)
    return data


def write_json(
    data: Any,
    path: str | Path,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    encoding: str = DEFAULT_ENCODING,
    atomic: bool = True,
) -> Path:
    """Serialize a Python object to a JSON file.

    Args:
        data: JSON-serializable object.
        path: Destination file path. Parent directories are created.
        indent: Indentation width for pretty-printing. Pass ``None`` for the
            most compact output.
        sort_keys: Whether to sort object keys. Useful when the output is
            committed to version control, since it keeps diffs stable.
        encoding: Text encoding.
        atomic: Whether to write via a temporary file and rename.

    Returns:
        The destination path.

    Raises:
        TypeError: If ``data`` is not JSON-serializable.
        OSError: If the file cannot be written.
    """
    destination = ensure_parent_directory(path)

    def _dump(handle: Any) -> None:
        # ensure_ascii=False keeps non-Latin scripts and emoji readable in
        # the output instead of expanding them into \uXXXX escapes.
        json.dump(data, handle, indent=indent, ensure_ascii=False, sort_keys=sort_keys)

    try:
        _write_atomically_or_directly(
            destination, _dump, encoding=encoding, atomic=atomic
        )
    except TypeError:
        logger.exception("Object is not JSON-serializable for %s", destination)
        raise

    logger.info("Wrote JSON to %s", destination)
    return destination


def _write_atomically_or_directly(
    destination: Path,
    writer: Callable[[Any], None],
    *,
    encoding: str,
    atomic: bool,
) -> None:
    """Run a write callback against a file handle, optionally atomically.

    Shared by ``write_text_file`` and ``write_json`` so the temporary-file
    dance and its cleanup exist in exactly one place.

    Args:
        destination: Final path the content should end up at.
        writer: Callback that receives an open text-mode handle and writes.
        encoding: Text encoding.
        atomic: Whether to stage the write through a temporary file.

    Raises:
        OSError: If the file cannot be written or renamed.
        TypeError: Propagated from ``writer`` for non-serializable data.
    """
    if not atomic:
        with destination.open("w", encoding=encoding) as handle:
            writer(handle)
        return

    # The temporary file sits in the destination directory rather than the
    # system temp dir, because os.replace is only atomic within a single
    # filesystem and /tmp is frequently a different mount.
    temp_path = destination.with_name(f"{destination.name}.tmp")
    try:
        with temp_path.open("w", encoding=encoding) as handle:
            writer(handle)
        os.replace(temp_path, destination)
    except (OSError, TypeError):
        # Remove the partial file so a failed write leaves no debris. The
        # original destination is untouched, since the rename never ran.
        temp_path.unlink(missing_ok=True)
        raise


# =============================================================================
# 3. Timing
# =============================================================================


@contextmanager
def timed(label: str, *, level: int = logging.INFO) -> Iterator[None]:
    """Measure and log how long a block of code takes.

    Timing is emitted even when the block raises, so a failure that takes 40
    seconds is still visible in the log rather than vanishing with the
    exception.

    Args:
        label: Human-readable name for the operation being timed.
        level: Logging level for the timing message.

    Yields:
        Nothing; the value is unused. Use as ``with timed("step"): ...``.

    Example:
        >>> with timed("train vocabulary"):
        ...     train()
    """
    # perf_counter, not time(): it is monotonic, so a clock adjustment or
    # NTP correction mid-operation cannot produce a negative duration.
    start = time.perf_counter()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        elapsed = time.perf_counter() - start
        outcome = "failed after" if failed else "took"
        logger.log(level, "%s %s %s", label, outcome, format_duration(elapsed))


def measure_time(func: Callable[..., T]) -> Callable[..., T]:
    """Decorate a function so each call logs its execution time.

    Why this exists alongside ``timed``:
        ``timed`` wraps an arbitrary block; this wraps a whole function
        permanently, which is the right tool when a function is slow every
        time it is called and you always want that recorded.

    Args:
        func: Function to wrap.

    Returns:
        The wrapped function, with ``functools.wraps`` preserving its name,
        docstring, and signature metadata for introspection and Sphinx.

    Example:
        >>> @measure_time
        ... def slow(): ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        with timed(f"{func.__name__}()", level=logging.DEBUG):
            return func(*args, **kwargs)

    return wrapper


# =============================================================================
# 4. Statistics
# =============================================================================


def describe_numbers(values: Iterable[float]) -> dict[str, float]:
    """Compute summary statistics for a numeric series.

    Args:
        values: Any iterable of numbers. Consumed once, so generators are
            fine.

    Returns:
        Dictionary with ``count``, ``total``, ``minimum``, ``maximum``,
        ``mean``, ``median``, and ``std_dev``. All values are rounded to six
        decimal places so the result serializes to stable JSON rather than
        to float noise like ``0.30000000000000004``.

    Raises:
        ValueError: If ``values`` is empty; there is no meaningful mean or
            median of nothing, and returning zeros would silently hide the
            fact that the input was empty.
    """
    series = [float(value) for value in values]
    if not series:
        raise ValueError("Cannot describe an empty series")

    # A single sample has no spread. stdev() raises on n<2, so this is
    # special-cased rather than left to blow up.
    std_dev = _stats.stdev(series) if len(series) > 1 else 0.0

    summary = {
        "count": len(series),
        "total": round(sum(series), 6),
        "minimum": round(min(series), 6),
        "maximum": round(max(series), 6),
        "mean": round(_stats.fmean(series), 6),
        "median": round(_stats.median(series), 6),
        "std_dev": round(std_dev, 6),
    }
    logger.debug("Described series of %d value(s)", len(series))
    return summary


def frequency_summary(
    counts: Mapping[str, int],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Summarize a token-frequency table.

    Complements ``describe_numbers``: that one summarizes a list of numbers,
    this one summarizes a name-to-count mapping.

    Args:
        counts: Mapping of item to occurrence count, e.g. a ``Counter``.
        top_k: How many of the most frequent items to include.

    Returns:
        Dictionary with ``unique_items``, ``total_occurrences``,
        ``singleton_items`` (items seen exactly once — a useful signal of
        how much of a vocabulary is noise), ``singleton_ratio``, and
        ``top_items`` as a list of ``{"item": ..., "count": ...}`` entries
        sorted by descending count with lexicographic tie-breaks.

    Raises:
        ValueError: If ``counts`` is empty or ``top_k`` is negative.
    """
    if not counts:
        raise ValueError("Cannot summarize an empty frequency table")
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")

    total = sum(counts.values())
    singletons = sum(1 for count in counts.values() if count == 1)

    # Sorting by (-count, item) rather than using Counter.most_common gives
    # a deterministic order for equal counts, so repeated runs on the same
    # data produce identical output.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    summary = {
        "unique_items": len(counts),
        "total_occurrences": total,
        "singleton_items": singletons,
        "singleton_ratio": round(singletons / len(counts), 6),
        "top_items": [
            {"item": item, "count": count} for item, count in ranked[:top_k]
        ],
    }
    logger.debug(
        "Summarized frequency table: %d unique, %d total",
        summary["unique_items"],
        total,
    )
    return summary


def percentage(part: float, whole: float, *, digits: int = 2) -> float:
    """Express ``part`` as a percentage of ``whole``.

    Why this exists:
        Percentage calculations are where division-by-zero bugs hide — an
        empty corpus or an empty vocabulary makes the denominator zero. This
        returns 0.0 for that case instead of raising, since "0% of nothing"
        is the sensible reading in a report.

    Args:
        part: The subset amount.
        whole: The total amount.
        digits: Decimal places to round to.

    Returns:
        Percentage value between 0 and 100 (not clamped — a ``part`` larger
        than ``whole`` returns a value above 100, which is a real signal
        worth surfacing rather than hiding).
    """
    if whole == 0:
        logger.debug("percentage() called with zero denominator; returning 0.0")
        return 0.0
    return round((part / whole) * 100, digits)


# =============================================================================
# 5. Validation
# =============================================================================


def validate_not_empty(value: T, name: str) -> T:
    """Validate that a value is present and non-empty.

    Accepts any object with a length (strings, lists, dicts). Note that 0
    and False are *not* treated as empty, unlike a bare truthiness check —
    a count of zero is a legitimate value, not a missing one.

    Args:
        value: Value to check.
        name: Parameter name, used in the error message so the caller knows
            which argument was wrong.

    Returns:
        The value unchanged, so validation can be inlined:
        ``self._docs = validate_not_empty(docs, "docs")``.

    Raises:
        ValueError: If the value is None or has zero length.
    """
    if value is None:
        raise ValueError(f"{name} must not be None")
    if hasattr(value, "__len__") and len(value) == 0:  # type: ignore[arg-type]
        raise ValueError(f"{name} must not be empty")
    return value


def validate_type(value: Any, expected: type | tuple[type, ...], name: str) -> Any:
    """Validate that a value is of the expected type.

    Args:
        value: Value to check.
        expected: A type or tuple of acceptable types.
        name: Parameter name for the error message.

    Returns:
        The value unchanged.

    Raises:
        TypeError: If the value is not an instance of ``expected``.
    """
    if not isinstance(value, expected):
        if isinstance(expected, tuple):
            names = " or ".join(t.__name__ for t in expected)
        else:
            names = expected.__name__
        raise TypeError(
            f"{name} must be {names}, got {type(value).__name__}"
        )
    return value


def validate_positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    """Validate that a value is an integer at or above a minimum.

    Args:
        value: Value to check.
        name: Parameter name for the error message.
        minimum: Smallest acceptable value. Pass 0 to allow zero.

    Returns:
        The value as an ``int``.

    Raises:
        TypeError: If the value is not an integer. ``bool`` is rejected
            explicitly, because ``isinstance(True, int)`` is True in Python
            and a stray flag passed as a size would otherwise be read as 1.
        ValueError: If the value is below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def validate_choice(value: T, allowed: Sequence[T], name: str) -> T:
    """Validate that a value is one of an allowed set.

    Args:
        value: Value to check.
        allowed: Acceptable values.
        name: Parameter name for the error message.

    Returns:
        The value unchanged.

    Raises:
        ValueError: If the value is not in ``allowed``. The message lists
            every valid option, so the caller does not have to go read the
            source to find out what was expected.
    """
    if value not in allowed:
        options = ", ".join(repr(item) for item in allowed)
        raise ValueError(f"{name} must be one of [{options}], got {value!r}")
    return value


def validate_file_exists(path: str | Path, name: str = "path") -> Path:
    """Validate that a path exists and is a regular file.

    Args:
        path: Path to check.
        name: Parameter name for the error message.

    Returns:
        The path as a ``Path`` object.

    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path exists but is a directory. This is
            kept distinct from "not found" because the two have completely
            different fixes.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{name} does not exist: {file_path}")
    if file_path.is_dir():
        raise IsADirectoryError(f"{name} is a directory, expected a file: {file_path}")
    return file_path


def validate_directory_exists(path: str | Path, name: str = "path") -> Path:
    """Validate that a path exists and is a directory.

    Args:
        path: Path to check.
        name: Parameter name for the error message.

    Returns:
        The path as a ``Path`` object.

    Raises:
        FileNotFoundError: If the path does not exist.
        NotADirectoryError: If the path exists but is not a directory.
    """
    directory = Path(path)
    if not directory.exists():
        raise FileNotFoundError(f"{name} does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {directory}")
    return directory


# =============================================================================
# 6. Formatting
# =============================================================================


def human_readable_size(num_bytes: float) -> str:
    """Format a byte count as a human-readable string.

    Args:
        num_bytes: Size in bytes. Negative values are formatted with a
            leading minus rather than rejected, so byte *deltas* can be
            formatted with the same helper.

    Returns:
        A string such as ``"1.5 MiB"``. Byte counts below 1024 are shown as
        whole numbers, since ``"512.00 B"`` reads worse than ``"512 B"``.
    """
    sign = "-" if num_bytes < 0 else ""
    size = float(abs(num_bytes))

    for unit in _SIZE_UNITS:
        if size < 1024 or unit == _SIZE_UNITS[-1]:
            if unit == "B":
                return f"{sign}{int(size)} {unit}"
            return f"{sign}{size:.2f} {unit}"
        size /= 1024

    # Unreachable: the loop always returns at the last unit. Present so
    # static analyzers see a definite return on every path.
    return f"{sign}{size:.2f} {_SIZE_UNITS[-1]}"


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    The unit adapts to the magnitude, because a single format cannot serve
    both a 0.4 ms function call and a 90 minute training run legibly.

    Args:
        seconds: Duration in seconds.

    Returns:
        A string such as ``"412.30 ms"``, ``"2.51 s"``, or ``"1h 03m 12s"``.
    """
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"

    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"
