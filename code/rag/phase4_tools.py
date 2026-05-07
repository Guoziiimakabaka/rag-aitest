from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List


@dataclass(frozen=True)
class AblationVariant:
    name: str
    use_hybrid: bool
    use_reranker: bool
    use_reflection: bool
    use_query_rewrite: bool


@dataclass(frozen=True)
class LongContextProbe:
    target: str
    position: str
    context_length: int
    question: str


DEFAULT_ABLATIONS = [
    AblationVariant(
        name="full",
        use_hybrid=True,
        use_reranker=True,
        use_reflection=True,
        use_query_rewrite=True,
    ),
    AblationVariant(
        name="no_bm25",
        use_hybrid=False,
        use_reranker=True,
        use_reflection=True,
        use_query_rewrite=True,
    ),
    AblationVariant(
        name="no_reranker",
        use_hybrid=True,
        use_reranker=False,
        use_reflection=True,
        use_query_rewrite=True,
    ),
    AblationVariant(
        name="no_reflection",
        use_hybrid=True,
        use_reranker=True,
        use_reflection=False,
        use_query_rewrite=True,
    ),
    AblationVariant(
        name="no_query_rewrite",
        use_hybrid=True,
        use_reranker=True,
        use_reflection=True,
        use_query_rewrite=False,
    ),
]


def load_eval_json(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def compute_metrics(records: List[dict]) -> Dict[str, float]:
    if not records:
        raise ValueError("records must not be empty")
    return {
        "context_recall": mean(float(x["context_recall"]) for x in records),
        "context_precision": mean(float(x["context_precision"]) for x in records),
        "faithfulness": mean(float(x["faithfulness"]) for x in records),
        "answer_relevance": mean(float(x["answer_relevance"]) for x in records),
    }


def run_ablation_simulation(records: List[dict]) -> List[dict]:
    base = compute_metrics(records)
    outputs: List[dict] = []
    for variant in DEFAULT_ABLATIONS:
        score = dict(base)
        if not variant.use_hybrid:
            score["context_recall"] -= 0.05
        if not variant.use_reranker:
            score["context_precision"] -= 0.04
        if not variant.use_reflection:
            score["faithfulness"] -= 0.03
        if not variant.use_query_rewrite:
            score["answer_relevance"] -= 0.02
        outputs.append(
            {
                "variant": variant.name,
                "use_hybrid": variant.use_hybrid,
                "use_reranker": variant.use_reranker,
                "use_reflection": variant.use_reflection,
                "use_query_rewrite": variant.use_query_rewrite,
                "metrics": score,
            }
        )
    return outputs


def analyze_errors(records: List[dict]) -> Dict[str, dict]:
    by_bucket: Dict[str, List[dict]] = {
        "low_faithfulness": [],
        "low_relevance": [],
        "long_context": [],
    }
    for rec in records:
        if float(rec["faithfulness"]) < 0.6:
            by_bucket["low_faithfulness"].append(rec)
        if float(rec["answer_relevance"]) < 0.6:
            by_bucket["low_relevance"].append(rec)
        context_chars = sum(len(ctx) for ctx in rec.get("retrieved_contexts", []))
        if context_chars > 1500:
            by_bucket["long_context"].append(rec)

    summary = {}
    for key, rows in by_bucket.items():
        if not rows:
            summary[key] = {"count": 0, "avg_faithfulness": None, "avg_relevance": None}
            continue
        summary[key] = {
            "count": len(rows),
            "avg_faithfulness": mean(float(x["faithfulness"]) for x in rows),
            "avg_relevance": mean(float(x["answer_relevance"]) for x in rows),
        }
    return summary


def build_long_context_probe(
    context_length: int = 4000,
    position: str = "middle",
) -> LongContextProbe:
    if position not in {"front", "middle", "tail"}:
        raise ValueError("position must be one of front/middle/tail")
    needle = "NEEDLE_TOKEN_PHASE4_SAFETY_RULE"
    filler = "X" * max(context_length - len(needle), 0)
    if position == "front":
        context = needle + filler
    elif position == "middle":
        mid = len(filler) // 2
        context = filler[:mid] + needle + filler[mid:]
    else:
        context = filler + needle
    question = "What is the hidden safety token in the provided long context?"
    return LongContextProbe(
        target=needle,
        position=position,
        context_length=len(context),
        question=question,
    )
