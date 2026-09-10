"""Tests for CLI wiring and exit codes."""

from __future__ import annotations

import pytest

from face_faker.domain.entities import GenerationConfig, GenerationResult, GenerationStats
from face_faker.domain.errors import MissingModelError, SourceUnavailableError
from face_faker.interfaces import cli


def test_parser_defaults_remove_bg_off() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["generate"])
    assert args.remove_bg is False
    assert args.yaw_left == 15.0
    assert args.yaw_right == 15.0
    assert args.pitch_up == 15.0
    assert args.pitch_down == 15.0
    assert args.roll_threshold == 15.0
    assert args.face_x_min == 0.0
    assert args.face_x_max == 1.0
    assert args.face_y_min == 0.0
    assert args.face_y_max == 1.0


def test_parser_accepts_directional_pose_and_region() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--yaw-left",
            "12",
            "--yaw-right",
            "8",
            "--pitch-up",
            "6",
            "--pitch-down",
            "10",
            "--roll-threshold",
            "5",
            "--face-x-min",
            "0.25",
            "--face-x-max",
            "0.75",
            "--face-y-min",
            "0.15",
            "--face-y-max",
            "0.85",
        ]
    )
    assert args.yaw_left == 12.0
    assert args.yaw_right == 8.0
    assert args.pitch_up == 6.0
    assert args.pitch_down == 10.0
    assert args.face_x_min == 0.25
    assert args.face_y_max == 0.85


def test_info_exits_zero(capsys) -> None:
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "face-faker" in out
    assert "Synthetic face dataset" in out


def test_no_command_prints_help(capsys) -> None:
    assert cli.main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_generate_missing_model_exit_code(monkeypatch, tmp_path) -> None:
    def boom(config: GenerationConfig):
        raise MissingModelError("no model")

    monkeypatch.setattr(cli, "generate_faces", boom)
    code = cli.main(["generate", "--count", "1", "--frontal-only", "-o", str(tmp_path)])
    assert code == cli.EXIT_MISSING_MODEL


def test_generate_source_error_exit_code(monkeypatch, tmp_path) -> None:
    def boom(config: GenerationConfig):
        raise SourceUnavailableError("dead source")

    monkeypatch.setattr(cli, "generate_faces", boom)
    code = cli.main(["generate", "--count", "1", "-o", str(tmp_path)])
    assert code == cli.EXIT_SOURCE


def test_generate_success_exit_code(monkeypatch, tmp_path, capsys) -> None:
    stats = GenerationStats(
        requested=1,
        produced=1,
        attempts=1,
        failed_fetches=0,
        filtered_out=0,
        gender_male=0,
        gender_female=1,
        gender_unknown=0,
        elapsed_seconds=0.1,
    )
    result = GenerationResult(records=(), stats=stats, output_dir=tmp_path, paths={})

    def ok(config: GenerationConfig):
        assert isinstance(config, GenerationConfig)
        assert config.remove_bg is False
        return result

    monkeypatch.setattr(cli, "generate_faces", ok)
    code = cli.main(["generate", "--count", "1", "-o", str(tmp_path)])
    assert code == 0
    assert "Produced 1/1" in capsys.readouterr().out
