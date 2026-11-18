#!/usr/bin/env bash
# Install (or remove with --uninstall) a systemd *user* service that starts whisper-dictate when
# the graphical session starts. It must be a user service, not a system one: the app needs the
# X11 display (global hotkeys) and the user's audio server (microphone).
set -euo pipefail

ENV_NAME="whisper-flow"
UNIT="whisper-dictate.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--uninstall" ]]; then
    systemctl --user disable --now "$UNIT" || true
    rm -f "$UNIT_DIR/$UNIT"
    systemctl --user daemon-reload
    echo "Removed $UNIT"
    exit 0
fi

CONDA_BASE="$(conda info --base)"
ENV_BIN="$CONDA_BASE/envs/$ENV_NAME/bin"
[[ -x "$ENV_BIN/whisper-dictate" ]] || { echo "whisper-dictate not found in conda env '$ENV_NAME' ($ENV_BIN)" >&2; exit 1; }

mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/$UNIT" <<UNITEOF
[Unit]
Description=whisper-dictate (F9 voice-to-keyboard)
After=graphical-session.target
PartOf=graphical-session.target

[Service]
WorkingDirectory=$REPO
Environment=PATH=$ENV_BIN:/usr/local/bin:/usr/bin:/bin
Environment=DISPLAY=:0
ExecStart=$ENV_BIN/whisper-dictate
# on-failure, not always: pressing the quit key (HOME) exits cleanly and must stay quit.
Restart=on-failure
RestartSec=10

[Install]
WantedBy=graphical-session.target
UNITEOF

systemctl --user daemon-reload
systemctl --user enable "$UNIT"
echo "Installed. Starts at next login; start it now with: systemctl --user start $UNIT"
