# whisper_dictate

Global voice-to-keyboard for Linux. Press a hotkey, speak, press again — your words are typed at the cursor in any application.

Powered by [OpenAI Whisper](https://openai.com/research/whisper). No local model download required.

> **Platform support:** Tested on Linux (PulseAudio). macOS is untested but may work. Windows is not supported (PortAudio setup is complex without conda).

---

## How it works

```
F9 pressed    →  audio alert (start)  →  microphone captures audio
F9 pressed    →  audio alert (stop)   →  audio sent to Whisper API
                                      →  transcription typed at cursor
HOME pressed  →  two overlapping beeps →  app quits
```

Long recordings are split into segments that are transcribed in parallel and joined back in order, so they are no longer limited by Whisper's 25 MB upload cap; recordings auto-stop after 60 minutes (configurable) as a safety net.

---

## Installation

### Prerequisites

- An [OpenAI API key](https://platform.openai.com/api-keys)
- `ffmpeg` (for `ffplay`), used to play the MP3 alert sounds:
  `sudo apt install ffmpeg` — `mpg123` works too. Without one of them the app
  runs fine but the alerts are silent, and it says so on the first attempt.
  `paplay`/`aplay` are **not** enough: neither decodes MP3.

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
| `HOME` | Quit (plays two overlapping beeps) |

Recordings stop automatically after 60 minutes (`MAX_RECORDING_MINUTES`) and are transcribed immediately — no need to press F9 again.

### Long recordings

Whisper accepts at most 25 MB per request — about 13 minutes of audio. Anything longer than `TRANSCRIPTION_CHUNK_SECONDS` (5 minutes by default) is split into segments, which are sent in parallel (`TRANSCRIPTION_MAX_PARALLEL` at a time) and joined back in order before typing. Each cut is placed at the quietest moment in the 3 seconds before the boundary, so words are not sliced in half. Short dictations are unaffected: they still go out as a single request.

If a segment still fails after the API client's automatic retries, its place in the text is filled with `[error processing this extract of voice command]` and the rest of the dictation is typed normally. Only when every segment fails (or a short, single-request dictation fails) is nothing typed and the error beep played.

The quit sound is the stop beep played twice, the second starting half a second into the first so they overlap. That stutter is what tells you the app closed rather than just finishing a recording.

### Changing the keys

Both keys are configurable, in `.env` or on the command line:

```bash
# .env
HOTKEY=f9
QUIT_KEY=home
```

```bash
# or per run — the flag wins over .env, which wins over the defaults
whisper-dictate --hotkey f10 --quit-key end
```

Any [pynput key name](https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key) works — `f1`–`f20`, `home`, `end`, `insert`, `delete`, `page_up`, `page_down`, `pause`, `scroll_lock` — as does a single character such as `q`. An unknown name falls back to the default and says so on startup; the banner always shows the key that is actually bound.

`ESC` deliberately is **not** the default quit key: it is pressed far too often while working and kept killing the app in the middle of a dictation. Set `QUIT_KEY=esc` if you want the old behaviour back.

Pick a quit key you will not hit by accident — it is captured globally, in every application.

### Options

```
whisper-dictate --help

  --hotkey KEY        Pynput key name for the record toggle  [default: f9]
  --quit-key KEY      Pynput key name that quits the app  [default: home]
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

# Quit with F12 instead of HOME
whisper-dictate --quit-key f12
```

Command-line flags override the `.env` values; the `.env` values override the built-in defaults.

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
│   ├── chunked.py        # ChunkedTranscriptionBackend — splits long audio, parallel requests
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

3. If the service caps the upload size, set the `max_upload_bytes` class attribute — long recordings are then split to fit
4. Add your backend name to the `--backend` choices in `whisper_dictate/__main__.py`
5. Instantiate it in `build_backend()` in `whisper_dictate/__main__.py`, choosing it from the selected backend name

---

## Configuration reference

All settings can be set in `.env` (copy from `.env.example`) or as environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `WHISPER_MODEL` | `whisper-1` | OpenAI Whisper model |
| `DEFAULT_LANGUAGE` | `auto` | ISO 639-1 code or `auto` |
| `HOTKEY` | `f9` | Key that starts/stops recording |
| `QUIT_KEY` | `home` | Key that quits the application |
| `SAMPLE_RATE` | `16000` | Microphone sample rate in Hz |
| `MAX_RECORDING_MINUTES` | `60` | Recordings auto-stop and transcribe at this length |
| `TRANSCRIPTION_CHUNK_SECONDS` | `300` | Longer recordings are split into segments of at most this length |
| `TRANSCRIPTION_MAX_PARALLEL` | `4` | Most segments transcribed at the same time |
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
