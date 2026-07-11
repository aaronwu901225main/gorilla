import json
import tempfile
import unittest
from pathlib import Path

from bfcl_eval.model_handler.local_inference.gemma_4 import Gemma4Handler
from bfcl_eval.scripts.check_vllm_gemma4_compat import (
    is_gemma4_unified,
    validate,
)


class Gemma4UnifiedCompatibilityTest(unittest.TestCase):
    def _model_dir(self, config: dict):
        temporary_directory = tempfile.TemporaryDirectory()
        model_path = Path(temporary_directory.name)
        (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return temporary_directory, model_path

    def test_unified_architecture_requires_new_vllm(self):
        config = {
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified",
        }
        temporary_directory, model_path = self._model_dir(config)
        self.addCleanup(temporary_directory.cleanup)

        compatible, message = validate(model_path, "0.19.1")

        self.assertFalse(compatible)
        self.assertIn("requires vLLM >= 0.23.0", message)

    def test_unified_architecture_accepts_supported_vllm(self):
        config = {
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified",
        }
        temporary_directory, model_path = self._model_dir(config)
        self.addCleanup(temporary_directory.cleanup)

        compatible, _ = validate(model_path, "0.24.0")

        self.assertTrue(compatible)
        self.assertTrue(is_gemma4_unified(config))

    def test_existing_gemma4_architecture_keeps_working(self):
        config = {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
        }
        temporary_directory, model_path = self._model_dir(config)
        self.addCleanup(temporary_directory.cleanup)

        compatible, _ = validate(model_path, "0.19.1")

        self.assertTrue(compatible)

    def test_unified_tool_call_format_is_decoded(self):
        result = '<|tool_call>call:get_weather{city:<|"|>Taipei<|"|>}<tool_call|>'

        decoded = Gemma4Handler._parse_call_notation(result)

        self.assertEqual(decoded, [{"get_weather": {"city": "Taipei"}}])


if __name__ == "__main__":
    unittest.main()
