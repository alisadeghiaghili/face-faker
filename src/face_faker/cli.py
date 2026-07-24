"""
CLI Interface for Face Faker

Usage:
    face-faker generate [--count N] [--output-dir DIR]
    face-faker generate --frontal-only [--threshold N]
    face-faker generate --remove-bg
    face-faker info
"""

import argparse
import sys

from ._version import __version__
from .face import generate_id_faces


def cmd_generate(args):
    """Generate face images."""
    metadata = generate_id_faces(
        output_dir=args.output_dir,
        num_images=args.count,
        remove_bg=args.remove_bg,
        frontal_only=args.frontal_only,
        frontal_threshold=args.threshold,
    )
    print(f"\nGenerated {len(metadata)} face images in '{args.output_dir}'")


def cmd_info(args):
    """Show Face Faker information."""
    print(f"Face Faker v{__version__}")
    print("=" * 40)
    print("AI-powered face image generator")
    print()
    print("Features:")
    print("  - Fetch AI-generated faces from thispersondoesnotexist.com")
    print("  - Gender detection using DeepFace")
    print("  - Background removal with rembg")
    print("  - Frontal face filtering with dlib")
    print()
    print("Usage:")
    print("  face-faker generate --count 100 --output-dir ./faces")
    print("  face-faker generate --frontal-only --remove-bg --count 50")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='face-faker',
        description='AI face image generator',
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate face images')
    gen_parser.add_argument('--count', type=int, default=100, help='Number of images to generate (default: 100)')
    gen_parser.add_argument('--output-dir', '-o', default='id_faces', help='Output directory (default: id_faces)')
    gen_parser.add_argument('--remove-bg', action='store_true', help='Remove background (transparent PNG)')
    gen_parser.add_argument('--frontal-only', action='store_true', help='Only save frontal faces')
    gen_parser.add_argument('--threshold', type=int, default=15, help='Frontal threshold in degrees (default: 15)')

    # Info command
    subparsers.add_parser('info', help='Show Face Faker information')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        'generate': cmd_generate,
        'info': cmd_info,
    }

    commands[args.command](args)


if __name__ == '__main__':
    main()
