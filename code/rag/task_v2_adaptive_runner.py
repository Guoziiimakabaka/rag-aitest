from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from task_v2 import build_adaptive_eval_records, estimate_latency_cost_tradeoff
from task_v2.utils import ensure_output_dir, load_yaml_config, require_field


def _repo_relative(path: Path, root: Path) -> str:
    return Path(path.relative_to(root)).as_posix()


def _find_variant(variants: List[dict], name: str) -> dict:
    for item in variants:
        if str(item.get("name")) == name:
            return item
    raise KeyError(f"Variant not found: {name}")


def build_adaptive_config(
    config_path: Path,
    output_dir: Path,
    output_config_path: Path,
    question_type_path: Path,
    policy_path: Path,
    repeat_runs: int,
    seed: int,
    source_variant_name: str,
    adaptive_variant_name: str,
) -> Dict[str, Path]:
    if repeat_runs < 1:
        raise ValueError("repeat_runs must be >= 1")

    root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(config_path)
    variants = require_field(config, "variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Config field 'variants' must be a non-empty list.")

    source_variant = _find_variant(variants=variants, name=source_variant_name)
    if "eval_json" not in source_variant:
        raise KeyError(f"Source variant {source_variant_name} missing eval_json.")

    source_eval_json = (root / str(source_variant["eval_json"])).resolve()
    run_dir = ensure_output_dir(output_dir.resolve())
    adaptive_eval_json = (run_dir / f"{adaptive_variant_name}.json").resolve()
    tradeoff_csv = (run_dir / "latency_cost_tradeoff.csv").resolve()
    cost_summary_csv = (run_dir / "latency_cost_summary.csv").resolve()
    repeated_dir = ensure_output_dir(run_dir / f"{adaptive_variant_name}_runs")
    repeated_paths: List[Path] = []

    for run_idx in range(repeat_runs):
        run_seed = seed + run_idx
        adaptive_records = build_adaptive_eval_records(
            base_eval_path=source_eval_json,
            question_type_path=question_type_path,
            policy_path=policy_path,
            seed=run_seed,
        )
        target = (repeated_dir / f"run_{run_idx + 1}.json").resolve()
        target.write_text(
            json.dumps(adaptive_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        repeated_paths.append(target)

    # Keep backward-compatible single eval_json path for existing pipeline logic.
    adaptive_eval_json.write_text(repeated_paths[0].read_text(encoding="utf-8"), encoding="utf-8")

    tradeoff_df = estimate_latency_cost_tradeoff(
        baseline_eval_path=source_eval_json,
        adaptive_eval_path=adaptive_eval_json,
        question_type_path=question_type_path,
        policy_path=policy_path,
    )
    tradeoff_df.to_csv(tradeoff_csv, index=False, encoding="utf-8-sig")
    cost_summary = pd.DataFrame(
        [
            {
                "variant": adaptive_variant_name,
                "repeat_runs": repeat_runs,
                "avg_latency_multiplier": float(tradeoff_df["estimated_latency_multiplier"].mean()),
                "max_latency_multiplier": float(tradeoff_df["estimated_latency_multiplier"].max()),
                "mean_context_recall_gain": float(tradeoff_df["context_recall_gain"].mean()),
                "mean_context_precision_gain": float(tradeoff_df["context_precision_gain"].mean()),
                "mean_faithfulness_gain": float(tradeoff_df["faithfulness_gain"].mean()),
                "mean_answer_relevance_gain": float(tradeoff_df["answer_relevance_gain"].mean()),
            }
        ]
    )
    cost_summary.to_csv(cost_summary_csv, index=False, encoding="utf-8-sig")

    adaptive_variant = {
        "name": adaptive_variant_name,
        "eval_json": _repo_relative(adaptive_eval_json, root),
        "eval_json_runs": [_repo_relative(path, root) for path in repeated_paths],
        "switches": {
            "use_hybrid": True,
            "use_reranker": True,
            "use_reflection": True,
            "use_query_rewrite": True,
            "use_adaptive_retrieval": True,
        },
    }

    new_variants: List[dict] = []
    has_adaptive = False
    for item in variants:
        if str(item.get("name")) == adaptive_variant_name:
            new_variants.append(adaptive_variant)
            has_adaptive = True
        else:
            new_variants.append(dict(item))
    if not has_adaptive:
        new_variants.append(adaptive_variant)

    updated_config = dict(config)
    updated_config["variants"] = new_variants

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(
        yaml.safe_dump(updated_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    return {
        "adaptive_eval_json": adaptive_eval_json,
        "adaptive_eval_runs_dir": repeated_dir.resolve(),
        "tradeoff_csv": tradeoff_csv,
        "cost_summary_csv": cost_summary_csv,
        "output_config": output_config_path.resolve(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate adaptive retrieval variant artifacts and config for task-v2."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/task_v2.yaml",
        help="Base task-v2 config path.",
    )
    parser.add_argument(
        "--output-dir",
        default="code/rag/outputs/adaptive/latest",
        help="Directory for adaptive eval json and tradeoff csv.",
    )
    parser.add_argument(
        "--output-config",
        default="configs/task_v2.adaptive.generated.yaml",
        help="Generated task-v2 config path with adaptive variant.",
    )
    parser.add_argument(
        "--question-type-path",
        default="code/rag/outputs/generated_testset.json",
        help="Question type mapping path.",
    )
    parser.add_argument(
        "--policy-path",
        default="configs/adaptive_retrieval.yaml",
        help="Adaptive retrieval policy yaml path.",
    )
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=3,
        help="How many repeated adaptive run files to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for adaptive noise injection across repeated runs.",
    )
    parser.add_argument(
        "--source-variant",
        default="full",
        help="Source variant name used to synthesize adaptive variant.",
    )
    parser.add_argument(
        "--adaptive-variant-name",
        default="adaptive_variant",
        help="Name of generated adaptive variant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    result = build_adaptive_config(
        config_path=(root / args.config).resolve(),
        output_dir=(root / args.output_dir).resolve(),
        output_config_path=(root / args.output_config).resolve(),
        question_type_path=(root / args.question_type_path).resolve(),
        policy_path=(root / args.policy_path).resolve(),
        repeat_runs=int(args.repeat_runs),
        seed=int(args.seed),
        source_variant_name=str(args.source_variant),
        adaptive_variant_name=str(args.adaptive_variant_name),
    )
    print("adaptive_config_built=true")
    print(f"adaptive_eval_json={result['adaptive_eval_json']}")
    print(f"adaptive_eval_runs_dir={result['adaptive_eval_runs_dir']}")
    print(f"tradeoff_csv={result['tradeoff_csv']}")
    print(f"cost_summary_csv={result['cost_summary_csv']}")
    print(f"output_config={result['output_config']}")


if __name__ == "__main__":
    main()
