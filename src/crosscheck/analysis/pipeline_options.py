"""Shared CLI flags for retrieve + NLI scripts."""

from __future__ import annotations

import argparse
import os
from typing import Literal

RerankerChoice = Literal["pinecone", "mps"]
NliModelChoice = Literal["gemini"]


def add_retrieve_nli_arguments(parser: argparse.ArgumentParser) -> None:
    """Register ``--reranker``, ``--nli-model``, and ``--no-rerank``."""
    parser.add_argument(
        "--reranker",
        choices=("pinecone", "mps"),
        default="pinecone",
        help="Rerank backend: pinecone (default, Inference API) or mps (local GPU).",
    )
    parser.add_argument(
        "--nli-model",
        choices=("gemini",),
        default="gemini",
        help="NLI provider (default: gemini family from config rank).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking.",
    )


def add_numeric_check_arguments(parser: argparse.ArgumentParser) -> None:
    """Register ``--numeric-check`` and ``--numeric-override-threshold`` (opt-in)."""
    from crosscheck.analysis.numeric_check import DEFAULT_OVERRIDE_THRESHOLD

    parser.add_argument(
        "--numeric-check",
        action="store_true",
        help=(
            "Run the deterministic numeric checker before NLI: its verdict goes "
            "into the prompt as a hint and may override a low-confidence label."
        ),
    )
    parser.add_argument(
        "--numeric-override-threshold",
        type=float,
        default=DEFAULT_OVERRIDE_THRESHOLD,
        help=(
            "Override the LLM only when its confidence is below this "
            f"(default: {DEFAULT_OVERRIDE_THRESHOLD})."
        ),
    )


def apply_retrieve_nli_options(
    *,
    reranker: RerankerChoice,
    nli_model: NliModelChoice,
) -> None:
    """Set process env / overrides consumed by rerank + LLM loaders."""
    if reranker == "mps":
        os.environ["CROSSCHECK_RERANK_BACKEND"] = "torch"
        os.environ["CROSSCHECK_EMBEDDING_DEVICE"] = "mps"
    else:
        os.environ["CROSSCHECK_RERANK_BACKEND"] = "pinecone"

    if nli_model == "gemini":
        from crosscheck.config import clear_llm_model_override

        clear_llm_model_override()
    else:
        raise ValueError(f"unsupported nli_model={nli_model!r}")
