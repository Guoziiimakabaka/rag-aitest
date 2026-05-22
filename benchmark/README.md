# Benchmark Protocol (Task-v2)

## Goal
Provide a reproducible benchmark contract for RAG-Eye experiments.

## Input Sources
- Evaluation records: `code/rag/outputs/*/rag_eval_results.json`
- Question type labels: `code/rag/outputs/generated_testset.json`

## Required Record Schema
Each evaluation record must include:
- `question`
- `ground_truth`
- `answer`
- `context_recall`
- `context_precision`
- `faithfulness`
- `answer_relevance`

## Split Strategy
- `dev`: early iteration and smoke checks.
- `test`: reported metric benchmark.
- `challenge`: stress and long-context regression.

Current repository keeps a unified set and metadata; split files can be added under `benchmark/splits/` with stable ids.

## Output Contract
The task-v2 pipeline writes:
- `summary.json`
- `ablation_real.csv`
- `stats_significance.csv`
- `gain_by_query_type.csv`
- `error_dashboard.csv`
- `error_cases_topk.json`
- `report.md`

## Reproducibility Rules
- Fixed seed via config and CLI override.
- Config-driven variants.
- Pairwise significance tests against baseline variant.

## Repeated-Run Stability Protocol (Week-2)
- Use `eval_json_runs` for each variant to provide repeated run files.
- If `eval_json_runs` is absent, pipeline falls back to single `eval_json`.
- Stability outputs include:
  - `runs`: repeated run count
  - `mean`: repeated-run metric mean
  - `std`: sample standard deviation (`ddof=1` when runs > 1)
  - `cv`: coefficient of variation (`std / abs(mean)`)
- Suggested alert thresholds:
  - `std > 0.015`, or
  - `cv > 0.05`
