from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from phase4_tools import compute_metrics
from task_v2.utils import METRIC_KEYS


@dataclass(frozen=True)
class VariantRun:
    name: str
    eval_json_paths: List[Path]
    switches: dict
    records_by_run: List[List[dict]]


def _load_eval_records(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval json for variant: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Invalid or empty eval records: {path}")
    for rec in records:
        for key in ["question", *METRIC_KEYS]:
            if key not in rec:
                raise KeyError(f"Missing key '{key}' in eval record from {path}")
    return records


def _records_to_dataframe(
    records: List[dict],
    variant_name: str,
    run_id: int,
) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = df[["question", *METRIC_KEYS]].copy()
    for key in METRIC_KEYS:
        df[key] = pd.to_numeric(df[key], errors="raise")
    df["variant"] = variant_name
    df["run_id"] = run_id
    return df


def _build_variant_runs(config: dict, root: Path) -> List[VariantRun]:
    variants = config.get("variants", [])
    if not variants:
        raise ValueError("Config must contain non-empty variants list.")

    outputs: List[VariantRun] = []
    for item in variants:
        if "name" not in item:
            raise KeyError("Each variant must contain name field.")
        name = str(item["name"])
        raw_paths = item.get("eval_json_runs")
        if raw_paths is None:
            if "eval_json" not in item:
                raise KeyError(
                    "Each variant must contain eval_json or eval_json_runs."
                )
            raw_paths = [item["eval_json"]]
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError("eval_json_runs must be a non-empty list.")

        eval_json_paths = [(root / str(path)).resolve() for path in raw_paths]
        records_by_run = [_load_eval_records(path) for path in eval_json_paths]

        outputs.append(
            VariantRun(
                name=name,
                eval_json_paths=eval_json_paths,
                switches=dict(item.get("switches", {})),
                records_by_run=records_by_run,
            )
        )
    return outputs


def _aggregate_runs_to_frame(records_frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not records_frames:
        raise ValueError("records_frames must not be empty")

    question_set = set(records_frames[0]["question"].astype(str).tolist())
    for frame in records_frames[1:]:
        current_set = set(frame["question"].astype(str).tolist())
        if current_set != question_set:
            raise ValueError(
                "All repeated runs for a variant must contain identical question sets."
            )

    merged = pd.concat(records_frames, ignore_index=True)
    agg = (
        merged.groupby("question", as_index=False)[METRIC_KEYS]
        .mean(numeric_only=True)
        .copy()
    )
    return agg


def _build_stability_rows(
    variant_name: str,
    per_run_metrics: List[Dict[str, float]],
) -> List[dict]:
    rows: List[dict] = []
    for metric in METRIC_KEYS:
        values = np.array([float(item[metric]) for item in per_run_metrics], dtype=float)
        run_count = int(values.size)
        if run_count == 0:
            raise ValueError(f"No run metrics found for variant {variant_name}.")
        std = float(np.std(values, ddof=1)) if run_count > 1 else 0.0
        metric_mean = float(np.mean(values))
        cv = 0.0 if metric_mean == 0.0 else std / abs(metric_mean)
        rows.append(
            {
                "variant": variant_name,
                "metric": metric,
                "runs": run_count,
                "mean": metric_mean,
                "std": std,
                "cv": cv,
            }
        )
    return rows


def run_real_ablation(
    config: dict,
    root: Path,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, dict], pd.DataFrame]:
    runs = _build_variant_runs(config=config, root=root)

    variant_frames: Dict[str, pd.DataFrame] = {}
    row_payloads: List[dict] = []
    summary_payload: Dict[str, dict] = {}
    stability_rows: List[dict] = []

    for run in runs:
        run_frames = [
            _records_to_dataframe(
                records=records,
                variant_name=run.name,
                run_id=idx,
            )
            for idx, records in enumerate(run.records_by_run)
        ]
        per_run_metrics = [compute_metrics(records) for records in run.records_by_run]
        agg_metrics = {
            key: float(np.mean([item[key] for item in per_run_metrics]))
            for key in METRIC_KEYS
        }

        agg_df = _aggregate_runs_to_frame(records_frames=run_frames)
        agg_df["variant"] = run.name
        variant_frames[run.name] = agg_df
        stability_rows.extend(
            _build_stability_rows(
                variant_name=run.name,
                per_run_metrics=per_run_metrics,
            )
        )

        eval_counts = {len(records) for records in run.records_by_run}
        if len(eval_counts) != 1:
            raise ValueError(
                f"All repeated runs must share identical eval counts for variant {run.name}."
            )
        eval_count = int(next(iter(eval_counts)))

        summary_payload[run.name] = {
            "eval_json_paths": [str(path) for path in run.eval_json_paths],
            "run_count": len(run.eval_json_paths),
            "eval_count": eval_count,
            "switches": run.switches,
            "metrics": agg_metrics,
            "per_run_metrics": per_run_metrics,
        }

        row_payload = {
            "variant": run.name,
            "eval_json_path": str(run.eval_json_paths[0]),
            "run_count": len(run.eval_json_paths),
            "eval_count": eval_count,
            "use_hybrid": bool(run.switches.get("use_hybrid", True)),
            "use_reranker": bool(run.switches.get("use_reranker", True)),
            "use_reflection": bool(run.switches.get("use_reflection", True)),
            "use_query_rewrite": bool(run.switches.get("use_query_rewrite", True)),
        }
        row_payload.update(agg_metrics)
        row_payloads.append(row_payload)

    result_df = pd.DataFrame(row_payloads)
    stability_df = pd.DataFrame(stability_rows)
    return result_df, variant_frames, summary_payload, stability_df
