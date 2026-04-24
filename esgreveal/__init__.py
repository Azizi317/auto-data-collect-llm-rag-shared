"""ESGReveal-style utilities for extracting structured data from ESG reports."""

from .runtime_env import configure_runtime_environment

configure_runtime_environment()

from .knowledge_base import KnowledgeBaseBuild, build_knowledge_base, knowledge_base_to_json
from .pipeline import PipelineRun, batch_results_to_json, results_to_json, results_to_markdown, run_batch, run_pipeline
from .gui import launch_gui

__all__ = [
    "KnowledgeBaseBuild",
    "PipelineRun",
    "batch_results_to_json",
    "build_knowledge_base",
    "knowledge_base_to_json",
    "launch_gui",
    "results_to_json",
    "results_to_markdown",
    "run_batch",
    "run_pipeline",
]
