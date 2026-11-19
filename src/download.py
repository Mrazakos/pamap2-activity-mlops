"""Download the PAMAP2 archive, verify it, and convert the ``.dat`` files to Parquet.

Output layout (``data/raw``, versioned with ``dvc add``)::

    data/raw/protocol/subject101.parquet ...
    data/raw/optional/subject101.parquet ...
    data/raw/docs/*.pdf                  (dataset documentation)
    data/raw/manifest.json               (rows per file, archive SHA-256)
"""

from __future__ import annotations

import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.constants import DAT_COLUMNS, RAW_SENSOR_COLUMNS
from src.utils import get_logger, kv, sha256_file, write_json

logger = get_logger(__name__)

_SUBJECT_RE = re.compile(r"subject(\d{3})\.dat$", re.IGNORECASE)


class ChecksumError(RuntimeError):
    """Raised when a downloaded file does not match its pinned SHA-256."""


def download_file(url: str, dest: Path, timeout: float = 120.0) -> Path:
    """Stream ``url`` to ``dest`` (atomically, via a ``.part`` file).

    Args:
        url: HTTP(S) URL.
        dest: Destination path; parent folders are created.
        timeout: Socket timeout in seconds.

    Returns:
        ``dest``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("downloading", extra=kv(url=url, dest=str(dest)))
    with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as fh:
        written, next_report = 0, 100 << 20
        while chunk := response.read(1 << 20):
            fh.write(chunk)
            written += len(chunk)
            if written >= next_report:
                logger.info("download progress", extra=kv(mb=written >> 20))
                next_report += 100 << 20
    tmp.replace(dest)
    logger.info("download finished", extra=kv(mb=dest.stat().st_size >> 20))
    return dest


def verify_sha256(path: Path, expected: str) -> str:
    """Compute a file's SHA-256 and compare it with the pinned value.

    Args:
        path: File to check.
        expected: Pinned hex digest; an empty string skips the comparison.

    Returns:
        The actual digest.

    Raises:
        ChecksumError: If ``expected`` is set and differs.
    """
    actual = sha256_file(path)
    if expected and actual != expected.lower():
        raise ChecksumError(f"{path}: sha256 {actual} != pinned {expected}")
    if not expected:
        logger.warning(
            "dataset.sha256 is not pinned; add it to configs/params.yaml",
            extra=kv(sha256=actual),
        )
    return actual


def _extract_checked(zip_path: Path, out_dir: Path) -> None:
    """Extract one archive after checking that no member escapes ``out_dir`` (zip slip)."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not (out_dir / member).resolve().is_relative_to(out_dir):
                raise ValueError(f"unsafe path in archive: {member}")
        archive.extractall(out_dir)


def safe_extract(zip_path: Path, out_dir: Path, max_rounds: int = 5) -> list[Path]:
    """Extract a zip archive, refusing paths that escape ``out_dir``.

    Zip files found inside the archive are extracted next to themselves and
    then deleted, round by round (the UCI download wraps ``PAMAP2_Dataset.zip``).

    Args:
        zip_path: Archive to extract.
        out_dir: Target folder.
        max_rounds: Maximum nesting depth to unpack.

    Returns:
        Sorted list of all extracted non-zip files.

    Raises:
        ValueError: If an archive member would be written outside ``out_dir``,
            or archives are nested deeper than ``max_rounds``.
    """
    out_dir = Path(out_dir).resolve()
    _extract_checked(zip_path, out_dir)
    for _ in range(max_rounds):
        inner = sorted(out_dir.rglob("*.zip"))
        if not inner:
            return sorted(p for p in out_dir.rglob("*") if p.is_file())
        for nested in inner:
            _extract_checked(nested, nested.parent)
            nested.unlink()
    raise ValueError(f"archives nested deeper than {max_rounds} levels in {zip_path}")


def find_dat_files(root: Path) -> list[tuple[str, int, Path]]:
    """Locate the recordings below an extracted dataset folder.

    Args:
        root: Folder containing ``Protocol/`` and ``Optional/`` (at any depth).

    Returns:
        Sorted ``(session, subject_id, path)`` tuples.
    """
    found: list[tuple[str, int, Path]] = []
    for path in sorted(Path(root).rglob("*.dat")):
        match = _SUBJECT_RE.search(path.name)
        session = path.parent.name.lower()
        if match and session in {"protocol", "optional"}:
            found.append((session, int(match.group(1)), path))
    return sorted(found)


def read_dat(path: Path) -> pd.DataFrame:
    """Read one whitespace-separated PAMAP2 recording with documented column names.

    Args:
        path: ``subjectXXX.dat`` file.

    Returns:
        DataFrame with the 54 documented columns (``NaN`` for missing values).
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",  # special-cased to the fast C parser; tolerant to trailing spaces
        header=None,
        names=DAT_COLUMNS,
        dtype={c: "float64" for c in ["timestamp", *RAW_SENSOR_COLUMNS]} | {"activity_id": "int64"},
        engine="c",
    )
    return df


def convert_recording(path: Path, session: str, subject_id: int, raw_dir: Path) -> tuple[Path, int]:
    """Convert one ``.dat`` file to Parquet with ``subject_id``/``session`` columns.

    Args:
        path: Source ``.dat`` file.
        session: ``protocol`` or ``optional``.
        subject_id: Subject number (101-109).
        raw_dir: Root of ``data/raw``.

    Returns:
        The written Parquet path and its row count.
    """
    from src.schemas import validate_raw

    df = read_dat(path)
    df.insert(0, "session", session)
    df.insert(0, "subject_id", subject_id)
    df = validate_raw(df)
    out = Path(raw_dir) / session / f"subject{subject_id}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd")
    logger.info("converted", extra=kv(subject=subject_id, session=session, rows=len(df)))
    return out, len(df)


def run_download(
    url: str,
    expected_sha256: str,
    zip_path: Path,
    extract_dir: Path,
    raw_dir: Path,
    keep_extracted: bool = False,
) -> dict[str, object]:
    """Full download step: fetch (if needed), verify, extract, convert.

    Args:
        url: Archive URL.
        expected_sha256: Pinned digest ('' to only report it).
        zip_path: Where the archive is stored.
        extract_dir: Scratch folder for extraction.
        raw_dir: Output folder (``data/raw``).
        keep_extracted: Keep the extracted ``.dat`` files (they are large).

    Returns:
        The manifest written to ``raw_dir/manifest.json``.
    """
    zip_path, extract_dir, raw_dir = Path(zip_path), Path(extract_dir), Path(raw_dir)
    if not zip_path.exists():
        download_file(url, zip_path)
    digest = verify_sha256(zip_path, expected_sha256)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    files = safe_extract(zip_path, extract_dir)

    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    docs_dir = raw_dir / "docs"
    docs_dir.mkdir(parents=True)
    for pdf in (p for p in files if p.suffix.lower() == ".pdf"):
        shutil.copy2(pdf, docs_dir / pdf.name)

    recordings: dict[str, int] = {}
    for session, subject_id, path in find_dat_files(extract_dir):
        out, rows = convert_recording(path, session, subject_id, raw_dir)
        recordings[out.relative_to(raw_dir).as_posix()] = rows
    if not recordings:
        raise FileNotFoundError(f"no subjectXXX.dat files found below {extract_dir}")

    manifest: dict[str, object] = {
        "source_url": url,
        "zip_sha256": digest,
        "recordings": recordings,
        "docs": sorted(p.name for p in docs_dir.iterdir()),
    }
    write_json(raw_dir / "manifest.json", manifest)
    if not keep_extracted:
        shutil.rmtree(extract_dir)
    logger.info("raw data ready", extra=kv(files=len(recordings), sha256=digest))
    return manifest
