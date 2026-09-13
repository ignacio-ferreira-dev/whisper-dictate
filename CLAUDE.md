# whisper_dictate — repo index

Global voice-to-keyboard for Linux: press F9, speak, press F9 again — the transcription is typed
at the cursor. HOME quits. Whisper API (no local model). See `README.md` for the user-facing
flow and platform notes.

## Layout

```
whisper_dictate/
  __main__.py      entry point (`whisper-dictate` console script)
  config.py        settings: hotkeys, timeouts, sample rate, API config (env-driven)
  audio/           recorder.py (microphone capture, 10-minute auto-stop) · alerts.py (start/stop beeps)
  transcription/   base.py (backend interface) · openai_backend.py (Whisper API)
  typing/          text_typer.py (types the text at the cursor)
  client.py        API client
  sounds/          alert assets
tests/unit · tests/integration · tests/conftest.py (shared fixtures)
run.sh             conda activate whisper-flow && whisper-dictate
```

## Working rules

- **Env**: conda env `whisper-flow` (`environment.yml`). Never install into base.
- **Tests**: `pytest tests/unit` is the default suite; `tests/integration` calls the Whisper
  API and **spends money** — run it only when explicitly asked.
- **Hardware**: the app grabs the microphone and global hotkeys; never run two instances, and
  never run it from an agent session without the owner present.
- **Branches**: work on a feature branch, PR to `master`; squash merges; English everywhere.
- **Keep `README.md` true**: any change to hotkeys, timeouts or platform support updates it in
  the same PR.

## Agent working contract

The machine-local contract in `.claude/` (gitignored) governs how agent sessions run here —
bootstrap first, gates, evidence, the PR body shape. `.claude/scripts/print-agent-index.sh`
prints the read manifest; if it reports the contract missing, say so instead of working from
memory.
