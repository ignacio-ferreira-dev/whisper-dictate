# whisper_dictate

Global voice-to-keyboard for Linux. Press a hotkey, speak, press again — your words are typed at the cursor in any application.

Powered by [OpenAI Whisper](https://openai.com/research/whisper). No local model download required.

> **Platform support:** Tested on Linux (PulseAudio). macOS is untested but may work. Windows is not supported (PortAudio setup is complex without conda).

---

## How it works

```
F9 pressed  →  audio alert (start)  →  microphone captures audio
F9 pressed  →  audio alert (stop)   →  audio sent to Whisper API
                                    →  transcription typed at cursor
```

Recordings auto-stop after 10 minutes and transcribe automatically.

---

## Installation

### Prerequisites

- An [OpenAI API key](https://platform.openai.com/api-keys)

### Option A — Conda (recommended)

Conda handles the PortAudio system dependency automatically.

```bash
git clone https://github.com/ignacio-ferreira-dev/whisper-dictate.git
cd whisper-dictate

conda env create -f environment.yml
conda activate whisper-dictate
pip install -e .
```

### Option B — pip (requires PortAudio on the system)

```bash
# Linux
sudo apt install portaudio19-dev
pip install whisper-dictate

# macOS (untested)
brew install portaudio
pip install whisper-dictate
```

### Configure your API key

```bash
cp .env.example .env
```

Open `.env` and set your key:

```
OPENAI_API_KEY=sk-your_actual_key_here
```

> **Never commit `.env` to version control.** It is listed in `.gitignore`.

### Updating the conda environment

If `environment.yml` changes after a `git pull`:

```bash
conda env update -f environment.yml --prune
```

---

## Usage

```bash
conda activate whisper-dictate
whisper-dictate
```

| Key | Action |
|-----|--------|
| `F9` | Start recording (plays a start beep) |
| `F9` again | Stop recording, transcribe, type at cursor |
| `ESC` | Quit |

Recordings stop automatically after 10 minutes and are transcribed immediately — no need to press F9 again.

### Options

```
whisper-dictate --help

  --hotkey KEY        Pynput key name for the record toggle  [default: f9]
  --language CODE     Language code: 'auto', 'es', 'en', 'fr', ...  [default: auto]
  --volume FLOAT      Alert volume 0.0–1.0  [default: 0.8]
  --no-alerts         Disable audio alerts
  --add-space         Prepend a space before each typed transcription
  --char-delay FLOAT  Delay (seconds) between typed characters  [default: 0.01]
  --backend {openai}  Transcription backend  [default: openai]
```

### Examples

```bash
# Spanish, no alerts, use F10 instead of F9
whisper-dictate --language es --no-alerts --hotkey f10

# Slower typing (useful for apps that drop characters)
whisper-dictate --char-delay 0.03

# Always prepend a space (useful mid-sentence)
whisper-dictate --add-space
```

---

## Architecture

```
whisper_dictate/
├── __main__.py           # CLI entry point (whisper-dictate)
├── client.py             # WhisperDictateClient — orchestrates all components
├── config.py             # Settings — loads .env, validates API key
├── audio/
│   ├── alerts.py         # AudioAlertsManager — plays MP3 sounds on state changes
│   └── recorder.py       # AudioRecorder — PyAudio microphone capture
├── transcription/
│   ├── base.py           # TranscriptionBackend — abstract interface (ABC)
│   └── openai_backend.py # OpenAIWhisperBackend — implementation via OpenAI API
├── typing/
│   └── text_typer.py     # TextTyper — types text into the active window (pynput)
└── sounds/               # Audio alert files (MP3)
    ├── recording_start.mp3
    ├── recording_end.mp3
    ├── transcription_end_success.mp3
    └── transcription_end_error.mp3
```

### Adding a new transcription backend

1. Create `whisper_dictate/transcription/my_backend.py`
2. Subclass `TranscriptionBackend` and implement `transcribe()`:

```python
from whisper_dictate.transcription.base import TranscriptionBackend

class MyBackend(TranscriptionBackend):
    async def transcribe(self, frames, sample_rate, language=None) -> str:
        # Convert frames to audio, call your service, return text
        ...
```

3. Add your backend name to the `--backend` choices in `whisper_dictate/__main__.py`
4. Instantiate it in `async_main()` when `args.backend == "my_backend"`

---

## Configuration reference

All settings can be set in `.env` (copy from `.env.example`) or as environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `WHISPER_MODEL` | `whisper-1` | OpenAI Whisper model |
| `DEFAULT_LANGUAGE` | `auto` | ISO 639-1 code or `auto` |
| `SAMPLE_RATE` | `16000` | Microphone sample rate in Hz |
| `ALERT_VOLUME` | `0.8` | Alert sound volume (0.0–1.0) |
| `ALERTS_ENABLED` | `true` | Enable/disable audio alerts |

---

## Publishing to PyPI

```bash
conda activate whisper-dictate
pip install build twine
python -m build
twine upload dist/*
```

Requires a [PyPI](https://pypi.org) account and an API token.

---

## Development

```bash
# Run unit tests (no microphone or API needed)
pytest tests/unit/

# Run integration tests (requires microphone)
pytest tests/integration/

# Run all tests
pytest

# Format code
black whisper_dictate/ tests/

# Lint
pylint whisper_dictate/
```

### Project layout

```
whisper-dictate/
├── whisper_dictate/     # Main package
│   ├── audio/           # Microphone capture and audio alerts
│   ├── transcription/   # Backend interface + OpenAI implementation
│   └── typing/          # Keyboard injection (pynput)
├── tests/
│   ├── unit/            # Fast, no external dependencies
│   └── integration/     # Require microphone / audio hardware
├── environment.yml      # Conda environment definition (start here)
├── pyproject.toml       # Package metadata and pip dependencies
├── .env.example         # Environment variable template
└── README.md
```

---

## Supported languages

Any language supported by OpenAI Whisper. Common codes: `en`, `es`, `fr`, `de`, `pt`, `it`, `ru`, `ja`, `ko`, `zh`, `ar`, `hi`. Use `auto` for automatic detection.

---

## License

MIT
