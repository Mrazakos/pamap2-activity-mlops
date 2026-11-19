"""Tests for download, checksum verification, safe extraction and conversion (synthetic archive)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.constants import DAT_COLUMNS
from src.download import ChecksumError, find_dat_files, read_dat, run_download, safe_extract, verify_sha256
from src.utils import sha256_file
from tests.conftest import make_recording


def _dat_text(subject: int) -> str:
    """A synthetic recording in the original whitespace-separated .dat format."""
    df = make_recording(subject, seconds_per_activity=1.0, transient_seconds=0.2, defects=False)
    return df[DAT_COLUMNS].to_csv(sep=" ", header=False, index=False, na_rep="NaN")


def _archive(path: Path) -> Path:
    """Build a UCI-like archive: outer zip containing PAMAP2_Dataset.zip + a PDF."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("PAMAP2_Dataset/Protocol/subject101.dat", _dat_text(101))
        z.writestr("PAMAP2_Dataset/Protocol/subject102.dat", _dat_text(102))
        z.writestr("PAMAP2_Dataset/Optional/subject101.dat", _dat_text(101))
        z.writestr("PAMAP2_Dataset/readme.pdf", b"%PDF-1.4 synthetic")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("PAMAP2_Dataset.zip", inner.getvalue())
    return path


def test_run_download_end_to_end(tmp_path: Path) -> None:
    source = _archive(tmp_path / "source.zip")
    digest = sha256_file(source)
    manifest = run_download(
        source.as_uri(), digest, tmp_path / "ext" / "pamap2.zip", tmp_path / "extract", tmp_path / "raw"
    )
    assert manifest["zip_sha256"] == digest
    assert sorted(manifest["recordings"]) == [  # type: ignore[call-overload]
        "optional/subject101.parquet",
        "protocol/subject101.parquet",
        "protocol/subject102.parquet",
    ]
    raw = pd.read_parquet(tmp_path / "raw" / "protocol" / "subject102.parquet")
    assert list(raw.columns[:2]) == ["subject_id", "session"] and raw["subject_id"].unique().tolist() == [102]
    assert (tmp_path / "raw" / "docs" / "readme.pdf").exists()
    assert not (tmp_path / "extract").exists()  # large extracted files are cleaned up


def test_checksum_mismatch_is_fatal(tmp_path: Path) -> None:
    source = _archive(tmp_path / "source.zip")
    with pytest.raises(ChecksumError):
        verify_sha256(source, "0" * 64)


def test_zip_slip_is_refused(tmp_path: Path) -> None:
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        safe_extract(evil, tmp_path / "out")


def test_find_and_read_dat(tmp_path: Path) -> None:
    folder = tmp_path / "Protocol"
    folder.mkdir()
    (folder / "subject105.dat").write_text(_dat_text(105))
    (folder / "notes.txt").write_text("ignore me")
    found = find_dat_files(tmp_path)
    assert [(s, n) for s, n, _ in found] == [("protocol", 105)]
    df = read_dat(found[0][2])
    assert list(df.columns) == DAT_COLUMNS
    assert df["heart_rate"].isna().any()  # 'NaN' tokens are parsed as missing
