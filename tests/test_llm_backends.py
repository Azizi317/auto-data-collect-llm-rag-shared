from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from esgreveal.llm import (
    LLMCallError,
    LLMConfig,
    call_claude_cli,
    call_codex_cli,
    call_gemini_cli,
    call_llm,
    cli_is_available,
    default_reasoning_effort_for_model,
    load_llm_config_from_env,
    suggested_models_for_backend,
)


class LLMBackendTests(unittest.TestCase):
    def test_load_codex_cli_config_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_BACKEND": "codex_cli",
                "LLM_MODEL": "gpt-5.4-mini",
                "LLM_TIMEOUT": "240",
                "CODEX_EXECUTABLE": "codex",
                "CODEX_WORKDIR": "C:/workspace/Auto_Data collect_LLM_RAG",
                "CODEX_REASONING_EFFORT": "medium",
            },
            clear=False,
        ):
            config = load_llm_config_from_env()

        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.backend, "codex_cli")
        self.assertEqual(config.model, "gpt-5.4-mini")
        self.assertEqual(config.timeout, 240)
        self.assertEqual(config.reasoning_effort, "medium")

    def test_load_gemini_cli_config_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_BACKEND": "gemini_cli",
                "LLM_MODEL": "gemini-3-flash-preview",
                "LLM_TIMEOUT": "120",
                "GEMINI_EXECUTABLE": "gemini.cmd",
                "GEMINI_WORKDIR": "C:/workspace/Auto_Data collect_LLM_RAG",
            },
            clear=False,
        ):
            config = load_llm_config_from_env()

        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.backend, "gemini_cli")
        self.assertEqual(config.model, "gemini-3-flash-preview")
        self.assertEqual(config.gemini_executable, "gemini.cmd")

    def test_load_llm_config_from_env_ignores_removed_local_api_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_BACKEND": "",
                "LLM_MODEL": "gpt-5.4-mini",
                "LLM_BASE_URL": "http://127.0.0.1:11434/v1",
                "LLM_API_KEY": "dummy",
            },
            clear=False,
        ):
            config = load_llm_config_from_env()

        self.assertIsNone(config)

    def test_call_codex_cli_reads_schema_output_file(self) -> None:
        tmpdir = Path.cwd() / ".tmp" / f"test-codex-{uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=False)
        try:
            output_payload = (
                '{"Analysis":null,"disclosure":1,"data":{"KPI":"에너지 사용량","Topic":"직접 및 간접 에너지 소비",'
                '"Value":"101,588","Unit":"MWh","Target":null,"Action":null,"Evidence":"근거",'
                '"SourcePage":12,"MatchedTerms":["에너지 사용량"]}}'
            )

            def _fake_run(command: list[str], **_: object):
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text(output_payload, encoding="utf-8")

                class _Completed:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return _Completed()

            config = LLMConfig(
                backend="codex_cli",
                model="gpt-5.4-mini",
                codex_executable="codex",
                codex_workdir=str(tmpdir),
                timeout=60,
            )

            with patch("esgreveal.llm.subprocess.run", side_effect=_fake_run):
                payload = call_codex_cli(config, "JSON만 반환하라")

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertTrue(payload["disclosed"])
        self.assertEqual(payload["disclosure_kpi"], "에너지 사용량")
        self.assertEqual(payload["value"], "101,588")
        self.assertEqual(payload["unit"], "MWh")

    def test_call_gemini_cli_reads_stdout_json(self) -> None:
        def _fake_run(_: list[str], **__: object):
            class _Completed:
                returncode = 0
                stdout = json.dumps(
                    {
                        "response": json.dumps(
                            {
                                "Analysis": None,
                                "disclosure": 1,
                                "data": {
                                    "KPI": "테스트 KPI",
                                    "Topic": "테스트 토픽",
                                    "Value": "123",
                                    "Unit": "kg",
                                    "Target": None,
                                    "Action": None,
                                    "Evidence": "근거",
                                    "SourcePage": 1,
                                    "MatchedTerms": ["테스트"],
                                },
                            },
                            ensure_ascii=False,
                        )
                    },
                    ensure_ascii=False,
                )
                stderr = "Loaded cached credentials."

            return _Completed()

        config = LLMConfig(
            backend="gemini_cli",
            model="gemini-3-flash-preview",
            gemini_executable="gemini.cmd",
            timeout=60,
        )

        with patch("esgreveal.llm.subprocess.run", side_effect=_fake_run):
            payload = call_gemini_cli(config, "JSON만 반환하라")

        self.assertTrue(payload["disclosed"])
        self.assertEqual(payload["disclosure_kpi"], "테스트 KPI")
        self.assertEqual(payload["value"], "123")

    def test_call_llm_dispatches_to_gemini_cli_backend(self) -> None:
        config = LLMConfig(backend="gemini_cli", model="gemini-3-flash-preview")
        with patch("esgreveal.llm.call_gemini_cli", return_value={"disclosed": True, "value": "1"}) as mocked:
            payload = call_llm(config, "prompt")

        mocked.assert_called_once_with(config, "prompt")
        self.assertEqual(payload["value"], "1")
        self.assertEqual(payload["_llm_backend_used"], "gemini_cli")

    def test_call_claude_cli_reads_stdout_json(self) -> None:
        def _fake_run(_: list[str], **__: object):
            class _Completed:
                returncode = 0
                stdout = (
                    '{"disclosure":1,"data":{"KPI":"테스트 KPI","Topic":"테스트 토픽","Value":"123",'
                    '"Unit":"kg","Target":null,"Action":null,"Evidence":"근거","SourcePage":1,"MatchedTerms":["테스트"]}}'
                )
                stderr = ""

            return _Completed()

        config = LLMConfig(
            backend="claude_cli",
            model="sonnet",
            claude_executable="claude",
            timeout=60,
        )

        with patch("esgreveal.llm.subprocess.run", side_effect=_fake_run):
            payload = call_claude_cli(config, "JSON만 반환하라")

        self.assertTrue(payload["disclosed"])
        self.assertEqual(payload["disclosure_kpi"], "테스트 KPI")
        self.assertEqual(payload["value"], "123")

    def test_call_llm_dispatches_to_claude_cli_backend(self) -> None:
        config = LLMConfig(backend="claude_cli", model="sonnet")
        with patch("esgreveal.llm.call_claude_cli", return_value={"disclosed": True, "value": "1"}) as mocked:
            payload = call_llm(config, "prompt")

        mocked.assert_called_once_with(config, "prompt")
        self.assertEqual(payload["value"], "1")

    def test_suggested_models_include_requested_presets(self) -> None:
        self.assertEqual(suggested_models_for_backend("codex_cli")[0], "gpt-5.4-mini")
        self.assertIn("gpt-5.4", suggested_models_for_backend("codex_cli"))
        self.assertIn("gpt-5.3", suggested_models_for_backend("codex_cli"))
        self.assertNotIn("gemini-3.1-fast", suggested_models_for_backend("gemini_cli"))
        self.assertNotIn("gemini-3.1-pro", suggested_models_for_backend("gemini_cli"))
        self.assertIn("gemini-3-flash-preview", suggested_models_for_backend("gemini_cli"))
        self.assertIn("haiku", suggested_models_for_backend("claude_cli"))
        self.assertIn("sonnet", suggested_models_for_backend("claude_cli"))
        self.assertIn("opus", suggested_models_for_backend("claude_cli"))

    def test_codex_default_reasoning_effort_matches_requested_rules(self) -> None:
        self.assertEqual(default_reasoning_effort_for_model("codex_cli", "gpt-5.4-mini"), "xhigh")
        self.assertEqual(default_reasoning_effort_for_model("codex_cli", "gpt-5.4"), "medium")

    def test_call_llm_raises_when_selected_backend_fails(self) -> None:
        config = LLMConfig(backend="gemini_cli", model="gemini-3-flash-preview")
        with (
            patch("esgreveal.llm.call_gemini_cli", side_effect=RuntimeError("gemini failed")),
        ):
            with self.assertRaises(LLMCallError) as ctx:
                call_llm(config, "prompt")

        self.assertEqual(ctx.exception.backend, "gemini_cli")
        self.assertEqual(ctx.exception.model, "gemini-3-flash-preview")

    def test_invalid_schema_raises(self) -> None:
        def _fake_run(_: list[str], **__: object):
            class _Completed:
                returncode = 0
                stdout = '{"disclosure":1,"data":{"KPI":"bad"}}'
                stderr = ""
            return _Completed()

        config = LLMConfig(backend="claude_cli", model="sonnet", claude_executable="claude", timeout=60)
        with patch("esgreveal.llm.subprocess.run", side_effect=_fake_run):
            with self.assertRaises(ValueError):
                call_claude_cli(config, "JSON만 반환")


if __name__ == "__main__":
    unittest.main()
