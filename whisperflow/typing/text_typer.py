"""
Text Typer
==========

Types transcribed text into whichever application currently has focus,
simulating keyboard input via pynput.

Key behaviors:
  - Small delay between characters to avoid dropped keystrokes
  - Handles Unicode (accents, special chars) transparently via pynput
  - Optional leading space to avoid merging with existing text
  - Configurable typing speed

Usage:
    typer = TextTyper()
    typer.type_text("Hello world")        # types the string at the cursor
    typer.type_text("hola", add_space=True)  # prepends a space
"""

import time
from typing import Optional

from pynput.keyboard import Controller, Key


class TextTyper:
    """
    Simulates keyboard input to type text into the active application window.

    Uses pynput's Controller which works on X11 (Linux), macOS, and Windows
    without requiring the target application to have special support.
    """

    DEFAULT_CHAR_DELAY: float = 0.01   # seconds between characters
    DEFAULT_SPACE_BEFORE: bool = False

    def __init__(
        self,
        char_delay: float = DEFAULT_CHAR_DELAY,
        add_space_before: bool = DEFAULT_SPACE_BEFORE,
    ):
        """
        Args:
            char_delay:       Pause (seconds) between each typed character.
                              Increase if characters are dropped by the target app.
            add_space_before: When True, a space is typed before the text so it
                              doesn't merge with existing content.
        """
        self.char_delay = char_delay
        self.add_space_before = add_space_before
        self._controller = Controller()

    def type_text(
        self,
        text: str,
        add_space: Optional[bool] = None,
        delay_before_typing: float = 0.15,
    ) -> bool:
        """
        Type the given text at the current cursor position.

        A brief pause before typing ensures that any hotkey release events
        have been processed by the OS before we inject characters.

        Args:
            text:               The string to type.
            add_space:          Override the instance default for prepending a space.
                                Pass None to use the instance default.
            delay_before_typing: Seconds to wait before the first keystroke.

        Returns:
            True if the text was typed without error; False otherwise.
        """
        text = text.strip()
        if not text:
            return False

        use_space = add_space if add_space is not None else self.add_space_before

        try:
            time.sleep(delay_before_typing)

            if use_space:
                self._controller.type(" ")
                time.sleep(self.char_delay)

            self._type_string(text)
            return True

        except Exception:
            return False

    def type_text_with_newline(self, text: str) -> bool:
        """
        Type text followed by Enter.

        Useful when the target field requires confirmation (e.g. search bars).

        Args:
            text: The string to type.

        Returns:
            True on success.
        """
        result = self.type_text(text)
        if result:
            time.sleep(self.char_delay)
            self._controller.press(Key.enter)
            self._controller.release(Key.enter)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _type_string(self, text: str) -> None:
        """
        Type each character individually with a configurable delay.

        pynput's Controller.type() is used for Unicode safety (it handles
        accented characters and emoji without manual key mapping).
        """
        for char in text:
            self._controller.type(char)
            if self.char_delay > 0:
                time.sleep(self.char_delay)
