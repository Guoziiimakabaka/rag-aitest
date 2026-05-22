from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from phase4_tools import compute_metrics
from evaluation.utils import METRIC_KEYS


@dataclass(frozen=True)
class VariantRun:
    name: str
    eval_json_path: Path
    switches: dict
    records: List[dict]


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


def _records_to_dataframe(records: List[dict], variant_name: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = df[["question", *METRIC_KEYS]].copy()
    for key in METRIC_KEYS:
        df[key] = pd.to_numeric(df[key], errors="raise")
    df["variant"] = variant_name
    return df


def _build_variant_runs(config: dict, root: Path) -> List[VariantRun]:
    variants = config.get("variants", [])
    if not variants:
        raise ValueError("Config must contain non-empty variants list.")

    outputs: List[VariantRun] = []
    for item in variants:
        if "name" not in item or "eval_json" not in item:
            raise KeyError("Each variant must contain name and eval_json fields.")
        name = str(item["name"])
        eval_json_path = (root / str(item["eval_json"])).resolve()
        records = _load_eval_records(eval_json_path)
        outputs.append(
            VariantRun(
                name=name,
                eval_json_path=eval_json_path,
                switches=dict(item.get("switches", {})),
                records=records,
            )
        )
    return outputs


def run_real_ablation(config: dict, root: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, dict]]:
    runs = _build_variant_runs(config=config, root=root)

    variant_frames: Dict[str, pd.DataFrame] = {}
    row_payloads: List[dict] = []
    summary_payload: Dict[str, dict] = {}

    for run in runs:
        metrics = compute_metrics(run.records)
        df = _records_to_dataframe(records=run.records, variant_name=run.name)
        variant_frames[run.name] = df

        summary_payload[run.name] = {
            "eval_json_path": str(run.eval_json_path),
            "eval_count": len(run.records),
            "switches": run.switches,
            "metrics": metrics,
        }

        row_payload = {
            "variant": run.name,
            "eval_json_path": str(run.eval_json_path),
            "eval_count": len(run.records),
            "use_hybrid": bool(run.switches.get("use_hybrid", True)),
            "use_reranker": bool(run.switches.get("use_reranker", True)),
            "use_reflection": bool(run.switches.get("use_reflection", True)),
            "use_query_rewrite": bool(run.switches.get("use_query_rewrite", True)),
        }
        row_payload.update(metrics)
        row_payloads.append(row_payload)

    result_df = pd.DataFrame(row_payloads)
    return result_df, variant_frames, summary_payload

