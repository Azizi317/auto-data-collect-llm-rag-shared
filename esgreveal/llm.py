from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


LLMBackend = Literal["codex_cli", "gemini_cli", "claude_cli"]

# Prevent extra console windows when subprocesses run on Windows.
_NO_WINDOW: dict[str, int] = {"creationflags": 0x08000000} if os.name == "nt" else {}


@dataclass(frozen=True)
class LLMConfig:
    backend: LLMBackend
    model: str = ""
    timeout: int = 180
    reasoning_effort: str = "medium"
    codex_executable: str = "codex"
    codex_workdir: str = ""
    codex_skip_git_repo_check: bool = True
    gemini_executable: str = "gemini.cmd"
    gemini_workdir: str = ""
    claude_executable: str = "claude"
    claude_workdir: str = ""


class LLMCallError(RuntimeError):
    def __init__(self, backend: str, model: str, message: str) -> None:
        self.backend = backend
        self.model = model
        self.message = message
        super().__init__(f"{backend}:{model or '-'} {message}")


def load_llm_config_from_env() -> LLMConfig | None:
    backend = os.environ.get("LLM_BACKEND", "").strip().lower()
    timeout = int(os.environ.get("LLM_TIMEOUT", "180") or "180")

    if backend == "codex_cli":
        model = os.environ.get("LLM_MODEL", "").strip() or os.environ.get("CODEX_MODEL", "").strip() or "gpt-5.4-mini"
        return LLMConfig(
            backend="codex_cli",
            model=model,
            timeout=timeout,
            reasoning_effort=os.environ.get("CODEX_REASONING_EFFORT", "").strip()
            or default_reasoning_effort_for_model("codex_cli", model),
            codex_executable=os.environ.get("CODEX_EXECUTABLE", "codex").strip() or "codex",
            codex_workdir=os.environ.get("CODEX_WORKDIR", "").strip(),
            codex_skip_git_repo_check=_env_truthy(os.environ.get("CODEX_SKIP_GIT_REPO_CHECK", "1")),
        )

    if backend == "gemini_cli":
        return LLMConfig(
            backend="gemini_cli",
            model=os.environ.get("LLM_MODEL", "").strip() or os.environ.get("GEMINI_MODEL", "").strip(),
            timeout=timeout,
            gemini_executable=os.environ.get("GEMINI_EXECUTABLE", _default_gemini_executable()).strip()
            or _default_gemini_executable(),
            gemini_workdir=os.environ.get("GEMINI_WORKDIR", "").strip(),
        )

    if backend == "claude_cli":
        return LLMConfig(
            backend="claude_cli",
            model=os.environ.get("LLM_MODEL", "").strip() or os.environ.get("CLAUDE_MODEL", "").strip(),
            timeout=timeout,
            claude_executable=os.environ.get("CLAUDE_EXECUTABLE", "claude").strip() or "claude",
            claude_workdir=os.environ.get("CLAUDE_WORKDIR", "").strip(),
        )

    return None


def _default_gemini_executable() -> str:
    return "gemini.cmd" if os.name == "nt" else "gemini"


def cli_is_available(executable: str) -> bool:
    executable = (executable or "").strip()
    if not executable:
        return False
    if Path(executable).exists():
        return True
    return shutil.which(executable) is not None


def _env_truthy(value: str) -> bool:
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _scratch_root() -> Path:
    root = Path.cwd() / ".tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _scratch_dir(prefix: str) -> Path:
    path = _scratch_root() / f"{prefix}{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def suggested_models_for_backend(backend: str) -> list[str]:
    if backend == "codex_cli":
        return ["gpt-5.4-mini", "gpt-5.4", "gpt-5.3", "gpt-5.3-codex"]
    if backend == "gemini_cli":
        return ["gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
    if backend == "claude_cli":
        return ["haiku", "sonnet", "opus", "claude-sonnet-4-20250514", "claude-opus-4-20250514"]
    return []


def default_reasoning_effort_for_model(backend: str, model: str) -> str:
    normalized = (model or "").strip().lower()
    if backend == "codex_cli":
        if normalized == "gpt-5.4-mini":
            return "xhigh"
        if normalized == "gpt-5.4":
            return "medium"
        if normalized:
            return "medium"
    if backend == "claude_cli":
        return "medium"
    return "medium"


def parse_json_like(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"\{.*\}", content, re.S)
    if fenced:
        candidate = fenced.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("LLM response did not contain valid JSON.")


def _validate_raw_llm_payload(payload: dict[str, Any]) -> None:
    candidate = payload.get("Response") if isinstance(payload.get("Response"), dict) else payload
    if not isinstance(candidate, dict):
        raise ValueError("LLM response payload must be a JSON object.")
    if "disclosure" not in candidate:
        raise ValueError("LLM response is missing 'disclosure'.")
    if "data" not in candidate or not isinstance(candidate.get("data"), dict):
        raise ValueError("LLM response is missing object field 'data'.")
    data = candidate["data"]
    required = {"KPI", "Topic", "Value", "Unit", "Target", "Action", "Evidence", "SourcePage", "MatchedTerms"}
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"LLM response data is missing required fields: {', '.join(sorted(missing))}")


def _coerce_disclosure(value: Any) -> bool | None:
    if value in (1, "1", True, "true", "True", "TRUE"):
        return True
    if value in (0, "0", False, "false", "False", "FALSE"):
        return False
    return None


def normalize_llm_output(payload: dict[str, Any]) -> dict[str, Any]:
    response = payload.get("Response")
    if isinstance(response, dict):
        payload = response

    data = payload.get("data")
    if isinstance(data, dict):
        normalized = dict(data)
        normalized["disclosed"] = _coerce_disclosure(payload.get("disclosure"))
        if "KPI" in normalized and "disclosure_kpi" not in normalized:
            normalized["disclosure_kpi"] = normalized.get("KPI")
        if "Topic" in normalized and "topic" not in normalized:
            normalized["topic"] = normalized.get("Topic")
        if "Value" in normalized and "value" not in normalized:
            normalized["value"] = normalized.get("Value")
        if "Unit" in normalized and "unit" not in normalized:
            normalized["unit"] = normalized.get("Unit")
        if "Target" in normalized and "target" not in normalized:
            normalized["target"] = normalized.get("Target")
        if "Action" in normalized and "action" not in normalized:
            normalized["action"] = normalized.get("Action")
        if "Evidence" in normalized and "evidence" not in normalized:
            normalized["evidence"] = normalized.get("Evidence")
        if "SourcePage" in normalized and "source_page" not in normalized:
            normalized["source_page"] = normalized.get("SourcePage")
        if "MatchedTerms" in normalized and "matched_terms" not in normalized:
            normalized["matched_terms"] = normalized.get("MatchedTerms")
        if "analysis" not in normalized and payload.get("Analysis") is not None:
            normalized["analysis"] = payload.get("Analysis")
        return normalized

    payload = dict(payload)
    if "disclosed" not in payload:
        payload["disclosed"] = _coerce_disclosure(payload.get("disclosure"))
    return payload


def _validate_normalized_output(payload: dict[str, Any]) -> None:
    if payload.get("disclosed") not in {True, False, None}:
        raise ValueError("Normalized LLM output has invalid 'disclosed' value.")
    matched_terms = payload.get("matched_terms")
    if matched_terms is not None and not isinstance(matched_terms, list):
        raise ValueError("Normalized LLM output has invalid 'matched_terms' type.")


def _finalize_llm_response(content: str) -> dict[str, Any]:
    raw = parse_json_like(content)
    _validate_raw_llm_payload(raw)
    normalized = normalize_llm_output(raw)
    _validate_normalized_output(normalized)
    return normalized


def call_llm(config: LLMConfig, prompt: str) -> dict[str, Any]:
    warnings: list[str] = []
    last_error: Exception | None = None
    candidate = config
    for attempt in range(2):
        try:
            if candidate.backend == "codex_cli":
                output = call_codex_cli(candidate, prompt)
            elif candidate.backend == "gemini_cli":
                output = call_gemini_cli(candidate, prompt)
            elif candidate.backend == "claude_cli":
                output = call_claude_cli(candidate, prompt)
            else:
                raise ValueError(f"Unsupported LLM backend: {candidate.backend}")
            output["_llm_backend_used"] = candidate.backend
            output["_llm_model_used"] = candidate.model
            output["_llm_warnings"] = warnings
            return output
        except Exception as exc:
            last_error = exc
            warnings.append(f"{candidate.backend}:{candidate.model or '-'} attempt {attempt + 1} failed: {exc}")
    raise LLMCallError(
        backend=candidate.backend,
        model=candidate.model,
        message="LLM call failed.\n" + "\n".join(warnings),
    ) from last_error


def call_codex_cli(config: LLMConfig, prompt: str) -> dict[str, Any]:
    workdir = Path(config.codex_workdir).resolve() if config.codex_workdir else Path.cwd()
    schema = _codex_response_schema()
    tmp = _scratch_dir("esgreveal-codex-")
    try:
        schema_path = tmp / "schema.json"
        output_path = tmp / "response.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

        command = [
            config.codex_executable or "codex",
            "exec",
            "-C",
            str(workdir),
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
        ]
        if config.codex_skip_git_repo_check:
            command.append("--skip-git-repo-check")
        if config.reasoning_effort:
            command.extend(["-c", f'reasoning_effort="{config.reasoning_effort}"'])
        if config.model:
            command.extend(["-m", config.model])
        command.append(_json_only_instruction(prompt))

        completed = subprocess.run(
            command,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=config.timeout,
            **_NO_WINDOW,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Codex CLI call failed.\n"
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )
        if not output_path.exists():
            raise RuntimeError("Codex CLI did not create the response file.")

        return _finalize_llm_response(output_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def call_gemini_cli(config: LLMConfig, prompt: str) -> dict[str, Any]:
    workdir = Path(config.gemini_workdir).resolve() if config.gemini_workdir else Path.cwd()

    # Feed the prompt through stdin because Windows subprocess handling is more
    # reliable this way than passing a long inline prompt argument.
    tmp_dir = _scratch_dir("esgreveal-gemini-")
    tmp_path = tmp_dir / "prompt.txt"
    tmp_path.write_text(_json_only_instruction(prompt), encoding="utf-8")

    try:
        command = [
            config.gemini_executable or _default_gemini_executable(),
            "-o",
            "json",
            "--approval-mode",
            "yolo",
        ]
        if config.model:
            command.extend(["-m", config.model])

        with tmp_path.open(encoding="utf-8") as stdin_file:
            completed = subprocess.run(
                command,
                cwd=str(workdir),
                stdin=stdin_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=config.timeout,
                **_NO_WINDOW,
            )
        stdout = (
            completed.stdout.decode("utf-8", errors="replace")
            if isinstance(completed.stdout, (bytes, bytearray))
            else str(completed.stdout)
        )
        stderr = (
            completed.stderr.decode("utf-8", errors="replace")
            if isinstance(completed.stderr, (bytes, bytearray))
            else str(completed.stderr)
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if completed.returncode != 0:
        raise RuntimeError(
            "Gemini CLI call failed.\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )
    try:
        outer = json.loads(stdout)
        content = outer.get("response")
        if content is None:
            raise ValueError("Gemini CLI JSON response did not contain 'response'.")
    except json.JSONDecodeError:
        content = stdout
    return _finalize_llm_response(content)


def call_claude_cli(config: LLMConfig, prompt: str) -> dict[str, Any]:
    workdir = Path(config.claude_workdir).resolve() if config.claude_workdir else Path.cwd()
    command = [
        config.claude_executable or "claude",
        "--print",
        "--output-format",
        "text",
        "--permission-mode",
        "plan",
    ]
    if config.reasoning_effort:
        command.extend(["--effort", config.reasoning_effort])
    if config.model:
        command.extend(["--model", config.model])
    command.append(_json_only_instruction(prompt))

    completed = subprocess.run(
        command,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=config.timeout,
        **_NO_WINDOW,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Claude Code call failed.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    return _finalize_llm_response(completed.stdout)


def _json_only_instruction(prompt: str) -> str:
    return (
        "Read the prompt below and return exactly one JSON object. "
        "Do not return markdown, explanation, code fences, or any extra text.\n\n"
        f"{prompt}"
    )


def _codex_response_schema() -> dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    nullable_integer = {"type": ["integer", "null"]}
    action_field = {
        "anyOf": [
            {"type": "null"},
            {"type": "string"},
            {"type": "array", "items": {"type": "string"}},
        ]
    }
    return {
        "type": "object",
        "properties": {
            "Analysis": nullable_string,
            "disclosure": {"type": ["integer", "null"], "enum": [0, 1, None]},
            "data": {
                "type": "object",
                "properties": {
                    "KPI": nullable_string,
                    "Topic": nullable_string,
                    "Value": {"type": ["string", "number", "null"]},
                    "Unit": nullable_string,
                    "Target": nullable_string,
                    "Action": action_field,
                    "Evidence": nullable_string,
                    "SourcePage": nullable_integer,
                    "MatchedTerms": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "KPI",
                    "Topic",
                    "Value",
                    "Unit",
                    "Target",
                    "Action",
                    "Evidence",
                    "SourcePage",
                    "MatchedTerms",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["Analysis", "disclosure", "data"],
        "additionalProperties": False,
    }
