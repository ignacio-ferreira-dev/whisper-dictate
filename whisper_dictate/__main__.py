"""
whisper-dictate CLI entry point.

Usage:
    whisper-dictate [options]
    python -m whisper_dictate [options]
"""

import argparse
import asyncio
import sys


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the whisper-dictate CLI."""
    parser = argparse.ArgumentParser(
        prog="whisper-dictate",
        description=(
            "Global voice-to-keyboard: press a hotkey to record, "
            "speak, press again to transcribe and type at cursor."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hotkey",
        default="f9",
        metavar="KEY",
        help="Pynput key name for the record toggle",
    )
    parser.add_argument(
        "--language",
        default="auto",
        metavar="CODE",
        help="Whisper language code ('auto', 'es', 'en', ...)",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=0.8,
        metavar="FLOAT",
        help="Alert sound volume in [0.0, 1.0]",
    )
    parser.add_argument(
        "--no-alerts",
        action="store_true",
        help="Disable all audio alerts",
    )
    parser.add_argument(
        "--add-space",
        action="store_true",
        help="Prepend a space before each typed transcription",
    )
    parser.add_argument(
        "--char-delay",
        type=float,
        default=0.01,
        metavar="FLOAT",
        help="Delay (seconds) between typed characters",
    )
    parser.add_argument(
        "--backend",
        default="openai",
        choices=["openai"],
        help="Transcription backend to use",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    """Async entry point: builds and runs the WhisperDictateClient."""
    from whisper_dictate.audio.alerts import AudioAlertsManager
    from whisper_dictate.client import WhisperDictateClient
    from whisper_dictate.config import get_settings
    from whisper_dictate.transcription.openai_backend import OpenAIWhisperBackend
    from whisper_dictate.typing.text_typer import TextTyper

    settings = get_settings()
    try:
        settings.validate()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}")
        return 1

    alerts = AudioAlertsManager(volume=args.volume, enabled=not args.no_alerts)
    typer = TextTyper(char_delay=args.char_delay, add_space_before=args.add_space)
    backend = OpenAIWhisperBackend(api_key=settings.openai_api_key, model=settings.whisper_model)

    client = WhisperDictateClient(
        backend=backend,
        hotkey=args.hotkey,
        language=args.language,
        alerts=alerts,
        typer=typer,
    )

    try:
        return await client.run()
    except KeyboardInterrupt:
        print("\nInterrupted - goodbye!")
        return 0


def main() -> None:
    """Synchronous entry point registered as the 'whisper-dictate' console script."""
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
