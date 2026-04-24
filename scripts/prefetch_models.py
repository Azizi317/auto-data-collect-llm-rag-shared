from __future__ import annotations

from pathlib import Path
import argparse
import os

from huggingface_hub import snapshot_download

from esgreveal.runtime_env import configure_runtime_environment, project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESGReveal 실행에 필요한 모델을 repo local cache로 미리 받습니다.")
    parser.add_argument("--skip-summary", action="store_true", help="mt5 summary 모델 다운로드를 건너뜁니다.")
    parser.add_argument("--skip-embedding", action="store_true", help="m3e embedding 모델 다운로드를 건너뜁니다.")
    parser.add_argument("--skip-table", action="store_true", help="Table Transformer 다운로드를 건너뜁니다.")
    parser.add_argument("--skip-lore", action="store_true", help="LORE-TSR 사전 다운로드를 건너뜁니다.")
    return parser.parse_args()


def download_hf_model(model_name: str) -> Path:
    path = snapshot_download(repo_id=model_name)
    return Path(path)


def main() -> int:
    configure_runtime_environment()
    args = parse_args()

    print(f"Project root: {project_root()}")
    print(f"HF cache: {os.environ.get('HF_HOME')}")
    print(f"ModelScope cache: {os.environ.get('MODELSCOPE_CACHE')}")

    if not args.skip_summary:
        model_name = os.environ["ESGREVEAL_SUMMARY_MODEL"]
        path = download_hf_model(model_name)
        print(f"[summary] {model_name} -> {path}")

    if not args.skip_embedding:
        model_name = os.environ["ESGREVEAL_EMBEDDING_MODEL"]
        path = download_hf_model(model_name)
        print(f"[embedding] {model_name} -> {path}")

    if not args.skip_table:
        model_name = os.environ["ESGREVEAL_TABLE_TRANSFORMER_MODEL"]
        path = download_hf_model(model_name)
        print(f"[table] {model_name} -> {path}")

    if not args.skip_lore:
        try:
            from modelscope.hub.snapshot_download import snapshot_download as ms_download

            model_name = os.environ["ESGREVEAL_LORE_MODEL"]
            path = ms_download(model_name)
            print(f"[lore] {model_name} -> {path}")
        except Exception as exc:
            print(f"[lore] download skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
