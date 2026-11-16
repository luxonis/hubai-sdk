"""Tests for safe extraction of NN archives."""

import io
import tarfile
from pathlib import Path

import pytest

from hubai_sdk.utils.nn_archive import _extract_nn_archive


def _add_file(tf: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    tf.addfile(member, io.BytesIO(content))


def test_extract_nn_archive_accepts_regular_files(tmp_path: Path) -> None:
    archive = tmp_path / "model.tar"
    destination = tmp_path / "extracted"
    with tarfile.open(archive, "w") as tf:
        _add_file(tf, "model/config.json", b"{}")

    _extract_nn_archive(archive, destination)

    assert (destination / "model" / "config.json").read_bytes() == b"{}"


def test_extract_nn_archive_rejects_symlink_escape(tmp_path: Path) -> None:
    archive = tmp_path / "model.tar"
    destination = tmp_path / "extracted"
    outside = tmp_path / "outside"
    outside.mkdir()
    with tarfile.open(archive, "w") as tf:
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = str(outside)
        tf.addfile(link)
        _add_file(tf, "link/probe.txt", b"escaped")

    with pytest.raises(tarfile.FilterError):
        _extract_nn_archive(archive, destination)

    assert not (outside / "probe.txt").exists()


def test_extract_nn_archive_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "model.tar"
    destination = tmp_path / "extracted"
    with tarfile.open(archive, "w") as tf:
        _add_file(tf, "../probe.txt", b"escaped")

    with pytest.raises(tarfile.FilterError):
        _extract_nn_archive(archive, destination)

    assert not (tmp_path / "probe.txt").exists()


def test_extract_nn_archive_fails_closed_without_data_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "model.tar"
    destination = tmp_path / "extracted"
    with tarfile.open(archive, "w") as tf:
        _add_file(tf, "config.json", b"{}")
    monkeypatch.delattr(tarfile, "data_filter")

    with pytest.raises(RuntimeError, match=r"tarfile\.data_filter"):
        _extract_nn_archive(archive, destination)

    assert not destination.exists()
