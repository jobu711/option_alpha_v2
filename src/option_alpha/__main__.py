"""Entry point for python -m option_alpha."""

import sys


def main() -> None:
    """Run the Option Alpha scanner."""
    print(f"Option Alpha v1.0.0")
    print("Usage: python -m option_alpha [scan|serve|config]")
    print()
    print("Commands:")
    print("  scan    Run the full scanning pipeline")
    print("  serve   Start the dashboard web server")
    print("  config  Show current configuration")

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "config":
            from option_alpha.config import get_settings

            settings = get_settings()
            print()
            print(settings.model_dump_json(indent=2))
        else:
            print(f"\nCommand '{command}' not yet implemented.")


if __name__ == "__main__":
    main()
