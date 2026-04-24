from __future__ import annotations

from pathlib import Path
import os


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cache_root() -> Path:
    return project_root() / ".cache"


def configure_runtime_environment() -> None:
    root = cache_root()
    hf_root = root / "huggingface"
    st_root = root / "sentence_transformers"
    ms_root = root / "modelscope"
    torch_root = root / "torch"

    for directory in (root, hf_root, st_root, ms_root, torch_root):
        directory.mkdir(parents=True, exist_ok=True)

    defaults = {
        "ESGREVEAL_CACHE_ROOT": str(root),
        "HF_HOME": str(hf_root),
        "HUGGINGFACE_HUB_CACHE": str(hf_root / "hub"),
        "TRANSFORMERS_CACHE": str(hf_root / "transformers"),
        "SENTENCE_TRANSFORMERS_HOME": str(st_root),
        "MODELSCOPE_CACHE": str(ms_root),
        "TORCH_HOME": str(torch_root),
        "ESGREVEAL_SUMMARY_MODEL": "google/mt5-small",
        "ESGREVEAL_EMBEDDING_MODEL": "dragonkue/BGE-m3-ko",
        "ESGREVEAL_COROM_MODEL": "dragonkue/bge-reranker-v2-m3-ko",
        "ESGREVEAL_RERANKER_MODEL": "dragonkue/bge-reranker-v2-m3-ko",
        "ESGREVEAL_TABLE_TRANSFORMER_MODEL": "microsoft/table-transformer-structure-recognition-v1.1-all",
        "ESGREVEAL_LORE_MODEL": "damo/cv_resnet-transformer_table-structure-recognition_lore",
        "ESGREVEAL_LAYOUT_BACKEND": "geolayoutlm,layoutlmv3,layoutparser",
        "ESGREVEAL_TABLE_BACKEND": "table_transformer,lore_tsr,fitz",
        "LLM_BACKEND": "gemini_cli",
        "LLM_MODEL": "gemini-3-flash-preview",
        "GEMINI_EXECUTABLE": "gemini.cmd",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
