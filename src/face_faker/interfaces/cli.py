"""Command-line interface for Face Faker."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from face_faker._version import __version__
from face_faker.domain.entities import FaceRegion, GenerationConfig, PoseLimits
from face_faker.domain.errors import FaceFakerError
from face_faker.interfaces.api import generate_faces
from face_faker.logging_config import configure_cli_logging, get_logger

logger = get_logger("cli")

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_MISSING_MODEL = 2
EXIT_DEPENDENCY = 3
EXIT_SOURCE = 4
EXIT_INCOMPLETE = 5
EXIT_UNEXPECTED = 10


def _make_progress_callback(total: int) -> Callable[[int, int], None]:
    """Build a progress callback for the CLI.

    Uses tqdm when installed; otherwise prints a simple counter.

    Args:
        total: Requested image count.

    Returns:
        Callback ``(produced, requested) -> None``.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm is a core dep

        def _plain(produced: int, requested: int) -> None:
            print(f"\r{produced}/{requested}", end="", flush=True)

        return _plain

    bar = tqdm(total=total, unit="image", desc="Generating")

    def _tqdm(produced: int, requested: int) -> None:
        bar.n = produced
        bar.refresh()
        if produced >= requested:
            bar.close()

    return _tqdm


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.

    Example:
        >>> build_parser().prog
        'face-faker'
    """
    parser = argparse.ArgumentParser(
        prog="face-faker",
        description="Synthetic face dataset toolkit",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    sub = parser.add_subparsers(dest="command", help="Command to run")

    gen = sub.add_parser("generate", help="Generate synthetic face images")
    gen.add_argument("--count", type=int, default=100, help="Images to produce (default: 100)")
    gen.add_argument(
        "--output-dir",
        "-o",
        default="id_faces",
        help="Output directory (default: id_faces)",
    )
    gen.add_argument(
        "--remove-bg",
        action="store_true",
        help="Remove background (requires rembg; default: off)",
    )
    gen.add_argument(
        "--frontal-only",
        action="store_true",
        help="Keep only frontal poses (requires dlib + OpenCV)",
    )
    gen.add_argument(
        "--yaw-left",
        type=float,
        default=15.0,
        help="Max yaw toward subject's left, degrees (default: 15)",
    )
    gen.add_argument(
        "--yaw-right",
        type=float,
        default=15.0,
        help="Max |yaw| toward subject's right, degrees (default: 15)",
    )
    gen.add_argument(
        "--pitch-up",
        type=float,
        default=15.0,
        help="Max looking-up pitch, degrees (default: 15)",
    )
    gen.add_argument(
        "--pitch-down",
        type=float,
        default=15.0,
        help="Max |looking-down| pitch, degrees (default: 15)",
    )
    gen.add_argument(
        "--roll-threshold",
        type=float,
        default=15.0,
        help="Max absolute head tilt, degrees (default: 15)",
    )
    gen.add_argument(
        "--face-x-min",
        type=float,
        default=0.0,
        help="Min face-center X in [0,1] — left bound (default: 0)",
    )
    gen.add_argument(
        "--face-x-max",
        type=float,
        default=1.0,
        help="Max face-center X in [0,1] — right bound (default: 1)",
    )
    gen.add_argument(
        "--face-y-min",
        type=float,
        default=0.0,
        help="Min face-center Y in [0,1] — top bound (default: 0)",
    )
    gen.add_argument(
        "--face-y-max",
        type=float,
        default=1.0,
        help="Max face-center Y in [0,1] — bottom bound (default: 1)",
    )
    gen.add_argument(
        "--source-dir",
        default=None,
        help="Read faces from a local image directory instead of TPNDE",
    )
    gen.add_argument(
        "--source-retries",
        type=int,
        default=2,
        help="Extra TPNDE attempts after failure (default: 2)",
    )
    gen.add_argument(
        "--source-backoff",
        type=float,
        default=0.25,
        help="Base seconds between TPNDE retries (default: 0.25)",
    )
    gen.add_argument(
        "--source-shuffle",
        action="store_true",
        help="Shuffle --source-dir listing once at start",
    )
    gen.add_argument(
        "--gender-max-share",
        type=float,
        default=None,
        help="Cap share of male/female outputs in (0,1], e.g. 0.55",
    )
    gen.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar while generating",
    )
    gen.add_argument(
        "--no-gender",
        action="store_true",
        help="Skip gender classification",
    )
    gen.add_argument(
        "--require-gender",
        action="store_true",
        help="Fail if gender classification is unavailable",
    )
    gen.add_argument(
        "--models-dir",
        default=None,
        help="Directory containing shape_predictor_68_face_landmarks.dat",
    )
    gen.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not write metadata.json / stats.json / faces.csv",
    )
    gen.add_argument(
        "--color",
        action="store_true",
        help="Keep RGB output (default is grayscale)",
    )
    gen.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if fewer images than requested are produced",
    )

    sub.add_parser("info", help="Show package information")
    return parser


def cmd_generate(args: argparse.Namespace) -> int:
    """Run the generate subcommand.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    config = GenerationConfig(
        output_dir=args.output_dir,
        count=args.count,
        save_metadata=not args.no_metadata,
        remove_bg=args.remove_bg,
        frontal_only=args.frontal_only,
        pose_limits=PoseLimits(
            yaw_left=args.yaw_left,
            yaw_right=args.yaw_right,
            pitch_up=args.pitch_up,
            pitch_down=args.pitch_down,
            roll=args.roll_threshold,
        ),
        face_region=FaceRegion(
            center_x_min=args.face_x_min,
            center_x_max=args.face_x_max,
            center_y_min=args.face_y_min,
            center_y_max=args.face_y_max,
        ),
        classify_gender=not args.no_gender,
        require_gender=args.require_gender,
        grayscale=not args.color,
        models_dir=args.models_dir,
        strict_completion=args.strict,
        source_dir=args.source_dir,
        source_retries=args.source_retries,
        source_backoff_s=args.source_backoff,
        source_shuffle=args.source_shuffle,
        gender_max_share=args.gender_max_share,
    )

    progress_cb = _make_progress_callback(config.count) if args.progress else None

    try:
        result = generate_faces(config=config, progress=progress_cb)
    except FaceFakerError as exc:
        logger.error("%s", exc)
        name = type(exc).__name__
        if name == "MissingModelError":
            return EXIT_MISSING_MODEL
        if name == "DependencyError":
            return EXIT_DEPENDENCY
        if name == "SourceUnavailableError":
            return EXIT_SOURCE
        if name == "GenerationIncompleteError":
            return EXIT_INCOMPLETE
        return EXIT_UNEXPECTED

    stats = result.stats
    print(
        f"Produced {stats.produced}/{stats.requested} faces in "
        f"'{result.output_dir}' (attempts={stats.attempts}, "
        f"failed={stats.failed_fetches}, filtered={stats.filtered_out})"
    )
    print(
        f"Gender counts: male={stats.gender_male} "
        f"female={stats.gender_female} unknown={stats.gender_unknown}"
    )
    for name, path in result.paths.items():
        print(f"  {name}: {path}")
    return EXIT_OK


def cmd_info(_args: argparse.Namespace) -> int:
    """Print package information.

    Args:
        _args: Unused parsed arguments.

    Returns:
        Always ``0``.
    """
    print(f"face-faker {__version__}")
    print("Synthetic face dataset toolkit for CV testing and research.")
    print()
    print("Commands:")
    print("  face-faker generate --count 20 --output-dir ./faces")
    print("  face-faker generate --frontal-only --yaw-threshold 12")
    print("  face-faker info")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.

    Example:
        >>> main(["info"])
        0
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_cli_logging(verbose=getattr(args, "verbose", False))

    if args.command is None:
        parser.print_help()
        return EXIT_USAGE
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "info":
        return cmd_info(args)
    parser.error(f"Unknown command: {args.command}")
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
