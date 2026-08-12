"""Wake-word gate: Korean spellings must wake the assistant.

Korean STT never returns the ASCII "jarvis", so an ASCII-only gate rejects
every spoken wake-up and the assistant appears dead while merely asleep.
"""
from __future__ import annotations

import queue
import unittest

from unified_copilot.ui_server import HologramUIServer


class WakeMatchTest(unittest.TestCase):
    def setUp(self):
        self.ui = HologramUIServer(queue.Queue(), open_browser=False)

    def test_ascii_wake_word(self):
        self.assertEqual(self.ui.wake_match("jarvis"), (True, ""))
        self.assertEqual(self.ui.wake_match("JARVIS 해머 가져와"),
                         (True, "해머 가져와"))

    def test_korean_spellings_wake(self):
        for text in ("자비스", "자비스야", "쟈비스", "재비스", "저비스", "자비쓰"):
            matched, remainder = self.ui.wake_match(text)
            self.assertTrue(matched, text)
            self.assertEqual(remainder, "", text)

    def test_korean_wake_keeps_the_request(self):
        self.assertEqual(self.ui.wake_match("자비스 해머 가져와줘"),
                         (True, "해머 가져와줘"))
        # Agglutinative: no separator after the wake word.
        self.assertEqual(self.ui.wake_match("자비스해머 가져와줘"),
                         (True, "해머 가져와줘"))

    def test_longest_alias_wins(self):
        # "자비스야" must not match bare "자비스" and leave "야 ..." behind.
        self.assertEqual(self.ui.wake_match("자비스야 해머 가져와"),
                         (True, "해머 가져와"))

    def test_ascii_needs_a_separator(self):
        matched, _ = self.ui.wake_match("jarvistest 해머")
        self.assertFalse(matched)

    def test_non_wake_text_is_not_a_wake_up(self):
        for text in ("해머 가져와줘", "안녕", "서비스 좋네"):
            matched, remainder = self.ui.wake_match(text)
            self.assertFalse(matched, text)
            self.assertEqual(remainder, text)


class VoiceGateTest(unittest.TestCase):
    def setUp(self):
        self.ui = HologramUIServer(queue.Queue(), open_browser=False)

    def test_dormant_ignores_unprefixed_speech(self):
        accepted, text = self.ui.gate_voice("해머 가져와줘")
        self.assertFalse(accepted)
        self.assertEqual(text, "")

    def test_korean_wake_awakens_and_forwards(self):
        accepted, text = self.ui.gate_voice("자비스 해머 가져와줘")
        self.assertTrue(accepted)
        self.assertEqual(text, "해머 가져와줘")
        # Once awake, plain speech passes through unchanged.
        accepted, text = self.ui.gate_voice("이제 정리해줘")
        self.assertTrue(accepted)
        self.assertEqual(text, "이제 정리해줘")


if __name__ == "__main__":
    unittest.main()
