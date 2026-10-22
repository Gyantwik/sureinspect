from __future__ import annotations

import zipfile
from pathlib import Path

from eval.verify_submission_zip import main


def test_missing_explicit_archive_fails(capsys, tmp_path: Path):
    missing = tmp_path / "missing.zip"

    assert main([str(missing)]) == 1
    output = capsys.readouterr().out
    assert str(missing.resolve()) in output
    assert "does not exist" in output


def test_explicit_archive_argument_is_honored(capsys, tmp_path: Path):
    archive = tmp_path / "wrong-layout.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("proof-that-this-archive-was-opened.txt", "fixture")

    assert main([str(archive)]) == 1
    output = capsys.readouterr().out
    assert str(archive.resolve()) in output
    assert "Missing required artifact" in output
