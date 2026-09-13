"""
whisper-dictate CLI entry point.

Usage:
    whisper-dictate [options]
    python -m whisper_dictate [options]
"""

import argparse
import asyncio
import sys

from whisper_dictate.config import Settings, get_settings


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    """
    Return the argument parser for the whisper-dictate CLI.

    Defaults come from Settings so that .env values are honoured and a flag
    passed on the command line still wins over them.
    """
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
        default=settings.hotkey,
        metavar="KEY",
        help="Pynput key name for the record toggle",
    )
    parser.add_argument(
        "--quit-key",
        default=settings.quit_key,
        metavar="KEY",
        help="Pynput key name that quits the application",
    )
    parser.add_argument(
        "--language",
        default=settings.default_language,
        metavar="CODE",
        help="Whisper language code ('auto', 'es', 'en', ...)",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=settings.alert_volume,
        metavar="FLOAT",
        help="Alert sound volume in [0.0, 1.0]",
    )
    parser.add_argument(
        "--no-alerts",
        action="store_true",
        default=not settings.alerts_enabled,
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


def build_backend(settings: Settings):
    """
    Return the transcription backend: Whisper, wrapped so that recordings too
    long for one request are split and transcribed in parallel.
    """
    from whisper_dictate.transcription.chunked import ChunkedTranscriptionBackend
    from whisper_dictate.transcription.openai_backend import OpenAIWhisperBackend

    return ChunkedTranscriptionBackend(
        OpenAIWhisperBackend(
            api_key=settings.openai_api_key,
            model=settings.whisper_model,
        ),
        max_segment_seconds=settings.transcription_chunk_seconds,
        max_parallel=settings.transcription_max_parallel,
    )


async def async_main(args: argparse.Namespace, settings: Settings) -> int:
    """Async entry point: builds and runs the WhisperDictateClient."""
    # Imported lazily: pulling in PyAudio/pynput costs noticeable startup time
    # and must not happen for `--help` or a failed config check.
    from whisper_dictate.audio.alerts import AudioAlertsManager
    from whisper_dictate.client import WhisperDictateClient
    from whisper_dictate.typing.text_typer import TextTyper

    try:
        settings.validate()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}")
        return 1

    alerts = AudioAlertsManager(volume=args.volume, enabled=not args.no_alerts)
    typer = TextTyper(char_delay=args.char_delay, add_space_before=args.add_space)
    client = WhisperDictateClient(
        backend=build_backend(settings),
        hotkey=args.hotkey,
        quit_key=args.quit_key,
        language=args.language,
        alerts=alerts,
        typer=typer,
        max_recording_seconds=settings.max_recording_seconds,
    )

    try:
        return await client.run()
    except KeyboardInterrupt:
        print("\nInterrupted - goodbye!")
        return 0


def main() -> None:
    """Synchronous entry point registered as the 'whisper-dictate' console script."""
    settings = get_settings()
    args = build_parser(settings).parse_args()
    sys.exit(asyncio.run(async_main(args, settings)))


if __name__ == "__main__":
    main()
