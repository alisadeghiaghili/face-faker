"""End-to-end CLI tests using a local source directory."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from face_faker.interfaces import cli


def _write_faces(directory: Path, count: int = 3) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    for i in range(count):
        Image.new("RGB", (32, 32), color=colors[i % len(colors)]).save(
            directory / f"src_{i:02d}.png"
        )


def test_cli_generate_from_local_dir(tmp_path: Path, capsys) -> None:
    inbox = tmp_path / "inbox"
    out = tmp_path / "out"
    _write_faces(inbox, count=3)

    code = cli.main(
        [
            "generate",
            "--count",
            "2",
            "--source-dir",
            str(inbox),
            "--no-gender",
            "--output-dir",
            str(out),
            "--no-metadata",
        ]
    )
    assert code == 0
    assert (out / "face_0001.png").exists()
    assert (out / "face_0002.png").exists()
    assert not (out / "metadata.json").exists()
    assert "Produced 2/2" in capsys.readouterr().out


def test_cli_local_dir_writes_metadata_with_source_ref(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    out = tmp_path / "out"
    _write_faces(inbox, count=2)

    code = cli.main(
        [
            "generate",
            "--count",
            "2",
            "--source-dir",
            str(inbox),
            "--no-gender",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    metadata = (out / "metadata.json").read_text(encoding="utf-8")
    assert "source_ref" in metadata
    csv_text = (out / "faces.csv").read_text(encoding="utf-8")
    assert "source_ref" in csv_text.splitlines()[0]


def test_cli_missing_local_source_dir(tmp_path: Path) -> None:
    code = cli.main(
        [
            "generate",
            "--count",
            "1",
            "--source-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "out"),
            "--no-gender",
        ]
    )
    assert code == cli.EXIT_UNEXPECTED
