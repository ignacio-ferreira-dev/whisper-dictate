"""
Unit tests for whisper_dictate.typing.text_typer.TextTyper.

pynput's Controller is patched throughout so no real keystrokes are
sent to any window. Tests verify character sequences, Unicode handling,
space prepending, empty-text no-ops, and Enter injection.
"""

import pytest
from unittest.mock import MagicMock, call, patch

from pynput.keyboard import Key

from whisper_dictate.typing.text_typer import TextTyper


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_controller():
    """A MagicMock replacing pynput.keyboard.Controller."""
    return MagicMock()


@pytest.fixture
def typer(mock_controller):
    """TextTyper wired to a mock Controller (no real keystrokes)."""
    with patch("whisper_dictate.typing.text_typer.Controller", return_value=mock_controller):
        t = TextTyper(char_delay=0.0, add_space_before=False)
        t._controller = mock_controller
    return t


def typed_chars(mock_controller) -> list:
    """Return the list of characters passed to Controller.type()."""
    return [c.args[0] for c in mock_controller.type.call_args_list]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestTextTyperDefaults:
    """TextTyper is created with sensible defaults."""

    def test_default_char_delay(self):
        t = TextTyper()
        assert t.char_delay == TextTyper.DEFAULT_CHAR_DELAY

    def test_default_no_space_before(self):
        t = TextTyper()
        assert t.add_space_before is False

    def test_custom_values_are_stored(self):
        t = TextTyper(char_delay=0.05, add_space_before=True)
        assert t.char_delay == 0.05
        assert t.add_space_before is True


# ---------------------------------------------------------------------------
# type_text — happy path
# ---------------------------------------------------------------------------


class TestTypeText:
    """type_text() sends the correct character sequence to the Controller."""

    def test_simple_ascii_string(self, typer, mock_controller):
        result = typer.type_text("hello", delay_before_typing=0.0)
        assert result is True
        assert typed_chars(mock_controller) == list("hello")

    def test_unicode_accented_characters(self, typer, mock_controller):
        text = "hola niño"
        typer.type_text(text, delay_before_typing=0.0)
        assert typed_chars(mock_controller) == list(text)

    def test_mixed_case_string(self, typer, mock_controller):
        typer.type_text("Hello World", delay_before_typing=0.0)
        assert typed_chars(mock_controller) == list("Hello World")

    def test_string_with_punctuation(self, typer, mock_controller):
        text = "Hi, how are you?"
        typer.type_text(text, delay_before_typing=0.0)
        assert typed_chars(mock_controller) == list(text)

    def test_leading_trailing_whitespace_is_stripped(self, typer, mock_controller):
        typer.type_text("  hi  ", delay_before_typing=0.0)
        assert typed_chars(mock_controller) == list("hi")

    def test_returns_true_on_success(self, typer):
        result = typer.type_text("ok", delay_before_typing=0.0)
        assert result is True


# ---------------------------------------------------------------------------
# type_text — space prepending
# ---------------------------------------------------------------------------


class TestSpacePrepending:
    """A leading space can be injected to avoid merging with existing text."""

    def test_add_space_kwarg_prepends_space(self, typer, mock_controller):
        typer.type_text("hi", add_space=True, delay_before_typing=0.0)
        chars = typed_chars(mock_controller)
        assert chars[0] == " "
        assert chars[1:] == list("hi")

    def test_instance_default_add_space_before(self, mock_controller):
        with patch("whisper_dictate.typing.text_typer.Controller", return_value=mock_controller):
            t = TextTyper(char_delay=0.0, add_space_before=True)
            t._controller = mock_controller
        t.type_text("hi", delay_before_typing=0.0)
        chars = typed_chars(mock_controller)
        assert chars[0] == " "

    def test_kwarg_overrides_instance_default(self, mock_controller):
        """add_space=False overrides an instance default of True."""
        with patch("whisper_dictate.typing.text_typer.Controller", return_value=mock_controller):
            t = TextTyper(char_delay=0.0, add_space_before=True)
            t._controller = mock_controller
        t.type_text("hi", add_space=False, delay_before_typing=0.0)
        chars = typed_chars(mock_controller)
        assert chars[0] != " "


# ---------------------------------------------------------------------------
# type_text — empty / whitespace-only input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Empty or whitespace-only text is a safe no-op."""

    @pytest.mark.parametrize("text", ["", "   ", "\t", "\n"])
    def test_returns_false_for_blank_text(self, typer, text):
        assert typer.type_text(text, delay_before_typing=0.0) is False

    def test_no_keystrokes_sent_for_blank_text(self, typer, mock_controller):
        typer.type_text("", delay_before_typing=0.0)
        mock_controller.type.assert_not_called()


# ---------------------------------------------------------------------------
# type_text_with_newline
# ---------------------------------------------------------------------------


class TestTypeTextWithNewline:
    """type_text_with_newline() appends an Enter keystroke after the text."""

    def test_sends_enter_after_text(self, typer, mock_controller):
        typer.type_text_with_newline("done")
        chars = typed_chars(mock_controller)
        assert chars == list("done")
        pressed = [c.args[0] for c in mock_controller.press.call_args_list]
        assert Key.enter in pressed

    def test_returns_true_on_success(self, typer):
        assert typer.type_text_with_newline("ok") is True

    def test_empty_input_does_not_press_enter(self, typer, mock_controller):
        typer.type_text_with_newline("")
        mock_controller.press.assert_not_called()
