from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from phase4_tools import (
    analyze_errors,
    compute_metrics,
    load_eval_json,
    run_ablation_simulation,
)


@dataclass(frozen=True)
class ReportPaths:
    output_dir: Path
    summary_json: Path
    ablation_csv: Path
    error_csv: Path
    markdown: Path


def build_report_paths(output_dir: Path) -> ReportPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    return ReportPaths(
        output_dir=output_dir,
        summary_json=output_dir / "summary.json",
        ablation_csv=output_dir / "ablation.csv",
        error_csv=output_dir / "error_summary.csv",
        markdown=output_dir / "report.md",
    )


def flatten_ablation(ablation_items: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for item in ablation_items:
        row = {
            "variant": item["variant"],
            "use_hybrid": item["use_hybrid"],
            "use_reranker": item["use_reranker"],
            "use_reflection": item["use_reflection"],
            "use_query_rewrite": item["use_query_rewrite"],
            "context_recall": item["metrics"]["context_recall"],
            "context_precision": item["metrics"]["context_precision"],
            "faithfulness": item["metrics"]["faithfulness"],
            "answer_relevance": item["metrics"]["answer_relevance"],
        }
        rows.append(row)
    return rows


def flatten_error_summary(summary: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for bucket, values in summary.items():
        rows.append(
            {
                "bucket": bucket,
                "count": values["count"],
                "avg_faithfulness": values["avg_faithfulness"],
                "avg_relevance": values["avg_relevance"],
            }
        )
    return rows


def render_markdown(
    base_metrics: Dict[str, float],
    ablation_rows: List[dict],
    error_rows: List[dict],
    eval_count: int,
) -> str:
    lines: List[str] = []
    lines.append("# RAG-Eye Experiment Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Evaluated samples: {eval_count}")
    lines.append("")
    lines.append("## Baseline Metrics")
    lines.append("")
    lines.append(
        f"- context_recall: {base_metrics['context_recall']:.4f}"
    )
    lines.append(
        f"- context_precision: {base_metrics['context_precision']:.4f}"
    )
    lines.append(f"- faithfulness: {base_metrics['faithfulness']:.4f}")
    lines.append(f"- answer_relevance: {base_metrics['answer_relevance']:.4f}")
    lines.append("")
    lines.append("## Ablation Summary")
    lines.append("")
    lines.append(
        "| variant | hybrid | reranker | reflection | rewrite | recall | precision | faithfulness | relevance |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in ablation_rows:
        lines.append(
            "| "
            f"{row['variant']} | "
            f"{int(bool(row['use_hybrid']))} | "
            f"{int(bool(row['use_reranker']))} | "
            f"{int(bool(row['use_reflection']))} | "
            f"{int(bool(row['use_query_rewrite']))} | "
            f"{row['context_recall']:.4f} | "
            f"{row['context_precision']:.4f} | "
            f"{row['faithfulness']:.4f} | "
            f"{row['answer_relevance']:.4f} |"
        )
    lines.append("")
    lines.append("## Error Bucket Summary")
    lines.append("")
    lines.append("| bucket | count | avg_faithfulness | avg_relevance |")
    lines.append("|---|---:|---:|---:|")
    for row in error_rows:
        faith = (
            f"{row['avg_faithfulness']:.4f}"
            if row["avg_faithfulness"] is not None
            else "NA"
        )
        rel = (
            f"{row['avg_relevance']:.4f}"
            if row["avg_relevance"] is not None
            else "NA"
        )
        lines.append(
            f"| {row['bucket']} | {row['count']} | {faith} | {rel} |"
        )
    lines.append("")
    return "\n".join(lines)


def export_report(eval_json_path: Path, output_dir: Path) -> ReportPaths:
    records = load_eval_json(eval_json_path)
    base_metrics = compute_metrics(records)
    ablation_items = run_ablation_simulation(records)
    error_summary = analyze_errors(records)

    paths = build_report_paths(output_dir)
    ablation_rows = flatten_ablation(ablation_items)
    error_rows = flatten_error_summary(error_summary)

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_json_path": str(eval_json_path),
        "eval_count": len(records),
        "baseline_metrics": base_metrics,
        "ablation": ablation_items,
        "error_summary": error_summary,
    }
    paths.summary_json.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(ablation_rows).to_csv(paths.ablation_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(error_rows).to_csv(paths.error_csv, index=False, encoding="utf-8-sig")

    markdown = render_markdown(
        base_metrics=base_metrics,
        ablation_rows=ablation_rows,
        error_rows=error_rows,
        eval_count=len(records),
    )
    paths.markdown.write_text(markdown, encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RAG evaluation report")
    parser.add_argument(
        "--eval-json",
        default="code/rag/outputs/large/rag_eval_results.json",
        help="Path to evaluation json file.",
    )
    parser.add_argument(
        "--output-dir",
        default="code/rag/reports/latest",
        help="Directory to write report files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_json_path = Path(args.eval_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    paths = export_report(eval_json_path=eval_json_path, output_dir=output_dir)
    print(f"summary_json={paths.summary_json}")
    print(f"ablation_csv={paths.ablation_csv}")
    print(f"error_csv={paths.error_csv}")
    print(f"markdown={paths.markdown}")


if __name__ == "__main__":
    main()
