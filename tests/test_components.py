import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from unittest.mock import MagicMock, patch
from core.llm_engine import SuggestionEngine, LLMProvider
from core.audio_capture import AudioCapture
import queue


class TestComponents(unittest.TestCase):
    def test_audio_capture_init(self):
        ac = AudioCapture()
        self.assertIsInstance(ac.audio_queue, queue.Queue)
        print("AudioCapture initialized successfully.")

    @patch("core.llm_engine.OllamaProvider")
    def test_suggestion_engine(self, MockOllama):
        # Mock provider to avoid network calls
        mock_provider = MockOllama.return_value
        mock_provider.generate.return_value = "Mock suggestion"

        engine = SuggestionEngine()
        engine.update_transcript("Hello, I need help.")
        sugg = engine.generate_suggestions()

        self.assertEqual(sugg, "Mock suggestion")
        print("SuggestionEngine generated mock suggestion successfully.")


if __name__ == "__main__":
    unittest.main()
