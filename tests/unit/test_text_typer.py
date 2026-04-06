"""
Unit tests for TextTyper.

The pynput Controller is patched so no real keystrokes are sent to any
window. Tests verify character sequences, Unicode handling, space
prepending, empty-text no-ops, and Enter injection.
"""

import pytest
from unittest.mock import MagicMock, patch

from whisper_dictate.typing.text_typer import TextTyper
from pynput.keyboard import Key


def _make_typer():
    """Return a TextTyper with a mocked Controller."""
    mock_ctrl = MagicMock()
    with patch("whisper_dictate.typing.text_typer.Controller", return_value=mock_ctrl):
        typer = TextTyper(char_delay=0.0, add_space_before=False)
        typer._controller = mock_ctrl
    return typer, mock_ctrl


def test_instantiation_defaults():
    typer = TextTyper()
    assert typer.char_delay == TextTyper.DEFAULT_CHAR_DELAY
    assert typer.add_space_before == TextTyper.DEFAULT_SPACE_BEFORE


def test_type_ascii_string():
    typer, ctrl = _make_typer()
    assert typer.type_text("hello", delay_before_typing=0.0) is True
    typed = [c.args[0] for c in ctrl.type.call_args_list]
    assert typed == list("hello")


def test_type_unicode_accented_chars():
    typer, ctrl = _make_typer()
    text = "hola niño"
    assert typer.type_text(text, delay_before_typing=0.0) is True
    typed = [c.args[0] for c in ctrl.type.call_args_list]
    assert typed == list(text)


def test_add_space_before_prepends_space():
    typer, ctrl = _make_typer()
    assert typer.type_text("hi", add_space=True, delay_before_typing=0.0) is True
    typed = [c.args[0] for c in ctrl.type.call_args_list]
    assert typed[0] == " "
    assert typed[1:] == list("hi")


def test_empty_string_returns_false_and_no_keystrokes():
    typer, ctrl = _make_typer()
    assert typer.type_text("", delay_before_typing=0.0) is False
    assert typer.type_text("   ", delay_before_typing=0.0) is False
    assert ctrl.type.call_count == 0


def test_type_with_newline_sends_enter():
    typer, ctrl = _make_typer()
    assert typer.type_text_with_newline("done") is True
    typed = [c.args[0] for c in ctrl.type.call_args_list]
    assert typed == list("done")
    pressed = [c.args[0] for c in ctrl.press.call_args_list]
    assert Key.enter in pressed
