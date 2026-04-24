from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .knowledge_base import build_knowledge_base, knowledge_base_to_json
from .llm import LLMConfig, default_reasoning_effort_for_model
from .pipeline import batch_results_to_json, results_to_json, results_to_markdown, run_batch, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="esgreveal",
        description="한국 ESG 보고서에서 KPI를 추출하고 knowledge base를 구축하는 도구",
    )
    parser.add_argument("--input", help="ESG PDF 파일 또는 폴더 경로")
    parser.add_argument("--pdf", help="단일 ESG PDF 파일 경로")
    parser.add_argument("--metadata", help="KPI 메타데이터 JSON 경로")
    parser.add_argument("--top-k", type=int, default=3, help="KPI마다 가져올 최종 문맥 수")
    parser.add_argument("--initial-k", type=int, default=20, help="reranking 전에 먼저 가져올 후보 수")
    parser.add_argument("--recursive", action="store_true", help="폴더 입력 시 하위 폴더까지 PDF를 찾습니다.")
    parser.add_argument("--out", help="JSON 결과 저장 경로")
    parser.add_argument("--markdown-out", help="Markdown 결과 저장 경로")
    parser.add_argument("--out-dir", help="폴더 입력 시 JSON 결과를 저장할 폴더")
    parser.add_argument("--markdown-dir", help="폴더 입력 시 Markdown 결과를 저장할 폴더")
    parser.add_argument("--build-kb", action="store_true", help="추출 대신 structured knowledge base를 구축합니다.")
    parser.add_argument("--kb-dir", help="Knowledge Base folder. Used as build output in build-kb and retrieval input in extraction.")
    parser.add_argument("--summary-model", default=None, help="mt5 요약 모델 경로 또는 허브 이름")
    parser.add_argument("--embedding-model", default=None, help="m3e 임베딩 모델 경로 또는 허브 이름")
    parser.add_argument("--skip-faiss", action="store_true", help="FAISS 인덱스를 생성하지 않습니다.")
    parser.add_argument("--no-resume-checkpoints", action="store_true", help="Do not resume Knowledge Base checkpoints.")
    parser.add_argument("--milvus-uri", default=None, help="Milvus 또는 Milvus Lite URI")
    parser.add_argument("--milvus-collection", default=None, help="Milvus collection 이름")

    parser.add_argument(
        "--llm-backend",
        choices=["none", "codex_cli", "gemini_cli", "claude_cli"],
        default="none",
        help="최종 추출에 사용할 LLM backend",
    )
    parser.add_argument("--llm-model", default="", help="LLM 모델 이름")
    parser.add_argument("--llm-timeout", type=int, default=180, help="LLM 호출 timeout 초")
    parser.add_argument("--reasoning-effort", choices=["", "low", "medium", "high", "xhigh", "max"], default="", help="CLI LLM 추론 강도")
    parser.add_argument("--codex-executable", default="codex", help="Codex CLI 실행 파일 경로")
    parser.add_argument("--codex-workdir", default="", help="Codex CLI가 실행될 작업 폴더")
    parser.add_argument("--gemini-executable", default="gemini.cmd", help="Gemini CLI 실행 파일 경로")
    parser.add_argument("--gemini-workdir", default="", help="Gemini CLI가 실행될 작업 폴더")
    parser.add_argument("--claude-executable", default="claude", help="Claude Code 실행 파일 경로")
    parser.add_argument("--claude-workdir", default="", help="Claude Code가 실행될 작업 폴더")
    return parser


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _resolve_input_path(args: argparse.Namespace) -> Path:
    raw = args.input or args.pdf
    if not raw:
        raise SystemExit("--input 또는 --pdf 중 하나는 반드시 지정해야 합니다.")
    return Path(raw)


def _build_llm_config(args: argparse.Namespace) -> LLMConfig | None:
    if args.llm_backend == "none":
        return None
    if args.llm_backend == "codex_cli":
        model = args.llm_model or "gpt-5.4-mini"
        return LLMConfig(
            backend="codex_cli",
            model=model,
            timeout=args.llm_timeout,
            reasoning_effort=args.reasoning_effort or default_reasoning_effort_for_model("codex_cli", model),
            codex_executable=args.codex_executable,
            codex_workdir=args.codex_workdir,
        )
    if args.llm_backend == "gemini_cli":
        return LLMConfig(
            backend="gemini_cli",
            model=args.llm_model,
            timeout=args.llm_timeout,
            reasoning_effort=args.reasoning_effort or default_reasoning_effort_for_model("gemini_cli", args.llm_model),
            gemini_executable=args.gemini_executable,
            gemini_workdir=args.gemini_workdir,
        )
    if args.llm_backend == "claude_cli":
        return LLMConfig(
            backend="claude_cli",
            model=args.llm_model,
            timeout=args.llm_timeout,
            reasoning_effort=args.reasoning_effort or default_reasoning_effort_for_model("claude_cli", args.llm_model),
            claude_executable=args.claude_executable,
            claude_workdir=args.claude_workdir,
        )
    raise ValueError(f"Unsupported LLM backend: {args.llm_backend}")


def _write_single_outputs(out_path: str | None, markdown_path: str | None, json_payload: str, markdown_payload: str) -> None:
    if out_path:
        Path(out_path).write_text(json_payload, encoding="utf-8")
    else:
        sys.stdout.write(json_payload)
        sys.stdout.write("\n")
    if markdown_path:
        Path(markdown_path).write_text(markdown_payload, encoding="utf-8")


def _run_single(args: argparse.Namespace, input_path: Path) -> int:
    run = run_pipeline(
        pdf_path=input_path,
        metadata_path=Path(args.metadata) if args.metadata else None,
        top_k=args.top_k,
        llm_config=_build_llm_config(args),
        kb_dir=Path(args.kb_dir) if args.kb_dir else None,
        retrieval_initial_k=args.initial_k,
    )
    _write_single_outputs(
        args.out,
        args.markdown_out,
        results_to_json(run),
        results_to_markdown(run),
    )
    return 0


def _run_batch(args: argparse.Namespace, input_path: Path) -> int:
    runs = run_batch(
        input_path=input_path,
        metadata_path=Path(args.metadata) if args.metadata else None,
        top_k=args.top_k,
        recursive=args.recursive,
        llm_config=_build_llm_config(args),
        kb_dir=Path(args.kb_dir) if args.kb_dir else None,
        retrieval_initial_k=args.initial_k,
    )
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / "esgreveal_output"
    md_dir = Path(args.markdown_dir) if args.markdown_dir else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    for run in runs:
        source = Path(run.source_path)
        json_path = out_dir / f"{source.stem}.json"
        md_path = md_dir / f"{source.stem}.md"
        json_path.write_text(results_to_json(run), encoding="utf-8")
        md_path.write_text(results_to_markdown(run), encoding="utf-8")

    sys.stdout.write(batch_results_to_json(runs))
    sys.stdout.write("\n")
    return 0


def _run_kb(args: argparse.Namespace, input_path: Path) -> int:
    kb_dir = Path(args.kb_dir) if args.kb_dir else Path.cwd() / "esgreveal_knowledge_base"
    build = build_knowledge_base(
        input_path=input_path,
        output_dir=kb_dir,
        recursive=args.recursive,
        summary_model=args.summary_model or None,
        embedding_model=args.embedding_model or None,
        use_faiss=not args.skip_faiss,
        milvus_uri=args.milvus_uri,
        milvus_collection=args.milvus_collection,
        resume_checkpoints=not args.no_resume_checkpoints,
    )
    payload = knowledge_base_to_json(build)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
        sys.stdout.write("\n")
    return 0


def main() -> int:
    _configure_stdout()
    args = build_parser().parse_args()
    input_path = _resolve_input_path(args)
    if args.build_kb:
        return _run_kb(args, input_path)
    if input_path.is_dir():
        return _run_batch(args, input_path)
    return _run_single(args, input_path)


if __name__ == "__main__":
    raise SystemExit(main())
