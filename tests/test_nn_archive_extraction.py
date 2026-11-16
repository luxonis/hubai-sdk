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


@pytest.mark.parametrize("legacy", [False, True])
def test_extract_nn_archive_accepts_regular_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    if legacy:
        monkeypatch.delattr(tarfile, "data_filter", raising=False)
    archive = tmp_path / "model.tar"
    destination = tmp_path / "extracted"
    with tarfile.open(archive, "w") as tf:
        _add_file(tf, "model/config.json", b"{}")

    _extract_nn_archive(archive, destination)

    assert (destination / "model" / "config.json").read_bytes() == b"{}"


@pytest.mark.parametrize("legacy", [False, True])
def test_extract_nn_archive_rejects_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    if legacy:
        monkeypatch.delattr(tarfile, "data_filter", raising=False)
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

    with pytest.raises((tarfile.TarError, ValueError)):
        _extract_nn_archive(archive, destination)

    assert not (outside / "probe.txt").exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_extract_nn_archive_rejects_hardlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    if legacy:
        monkeypatch.delattr(tarfile, "data_filter", raising=False)
    archive = tmp_path / "model.tar"
    destination = tmp_path / "extracted"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"original")
    with tarfile.open(archive, "w") as tf:
        link = tarfile.TarInfo("link.txt")
        link.type = tarfile.LNKTYPE
        link.linkname = str(outside)
        tf.addfile(link)

    with pytest.raises((tarfile.TarError, ValueError)):
        _extract_nn_archive(archive, destination)

    assert outside.read_bytes() == b"original"


@pytest.mark.parametrize("legacy", [False, True])
def test_extract_nn_archive_rejects_parent_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    if legacy:
        monkeypatch.delattr(tarfile, "data_filter", raising=False)
    archive = tmp_path / "model.tar"
    destination = tmp_path / "extracted"
    with tarfile.open(archive, "w") as tf:
        _add_file(tf, "../probe.txt", b"escaped")

    with pytest.raises((tarfile.TarError, ValueError)):
        _extract_nn_archive(archive, destination)

    assert not (tmp_path / "probe.txt").exists()


def test_extract_nn_archive_rejects_existing_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    archive = tmp_path / "model.tar"
    destination = tmp_path / "extracted"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination.mkdir()
    (destination / "link").symlink_to(outside, target_is_directory=True)
    with tarfile.open(archive, "w") as tf:
        _add_file(tf, "link/probe.txt", b"escaped")

    with pytest.raises(ValueError, match="Unsafe NN archive member"):
        _extract_nn_archive(archive, destination)

    assert not (outside / "probe.txt").exists()
