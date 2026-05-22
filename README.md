# RAG-Eye: Vertical-Domain RAG Evaluation and Optimization System

## 1. Project Overview
RAG-Eye is a domain-focused Retrieval-Augmented Generation (RAG) project for the EV maintenance standard document `GBT+44510-2024.pdf`.

This repository upgrades a basic chunk-size comparison demo into an algorithm-oriented RAG workflow with:

- Hybrid retrieval (`Dense + BM25`)
- Query enhancement (`Multi-Query + HyDE`)
- Cross-Encoder reranking
- Reflection-based retry (Self-RAG style)
- Layered evaluation with multi-role judges
- Ablation simulation and error-bucket analysis
- One-command report export pipeline

The project emphasizes measurable improvements and interview-ready experiment deliverables.

## 2. Technical Architecture
The system is organized into four layers.

1. Data and Retrieval Layer
- Source document: `code/rag/data/GBT+44510-2024.pdf`
- Vector store: Chroma (`code/rag/vectorstores/`)
- Embedding model: `BAAI/bge-m3`
- Sparse retrieval: BM25
- Fusion retrieval: `EnsembleRetriever`

2. Generation and Reflection Layer
- LLM backend: OpenAI-compatible endpoint (default DeepSeek-compatible API)
- Initial answer generation from reranked evidence
- Reflection judge returns structured confidence and retry signal
- Retry loop triggered by confidence threshold and max retry count

3. Evaluation and Analysis Layer
- Phase 3 evaluator:
  - Hard negative mining
  - Layer classification (`fact`, `multi-hop`, `negative`, `long_context`)
  - Multi-role judges: retriever/generator/safety/meta
- Phase 4 tools:
  - Ablation simulation
  - Error bucket analysis
  - Long-context probe sample generator

4. Reporting Layer
- JSON summary export
- CSV tables for ablation and error buckets
- Markdown report generation
- One-command pipeline script for reproducible output

## 3. Repository Structure
Key files:

- `code/rag/api.py`: FastAPI service, hybrid retrieval chain, reflection endpoint, phase3/phase4 endpoints
- `code/rag/env_utils.py`: `.env` loading and required env validation
- `code/rag/pull_models.py`: model pre-pull from mirror/cache
- `code/rag/phase3_eval.py`: layered evaluation and multi-judge implementation
- `code/rag/phase4_tools.py`: ablation, error analysis, long-context probe helpers
- `code/rag/export_report.py`: report export (`summary.json`, `ablation.csv`, `error_summary.csv`, `report.md`)
- `code/rag/run_pipeline.py`: one-command pipeline entry
- `code/rag/rag.py`: baseline chunking/retrieval pipeline
- `code/rag/eval_engine.py`: metric scoring workflow
- `code/rag/generate_testset.py`: LLM-driven testset generation

## 4. Core Algorithm Pipeline
For each question, the enhanced pipeline runs:

1. Query expansion
- Multi-query rewriting generates diverse retrieval queries
- HyDE creates a hypothetical answer passage for better recall

2. Hybrid retrieval
- Dense retrieval from Chroma
- BM25 sparse retrieval from loaded documents
- Weighted fusion (`0.6` dense, `0.4` BM25 by default)

3. Deduplication and reranking
- Remove duplicated candidate documents
- Cross-Encoder (`BAAI/bge-reranker-base`) reranks retrieved candidates
- Keep top-N passages for final generation

4. Controlled generation
- Prompt constrains model to answer only from evidence
- If evidence is insufficient, model should explicitly state insufficiency

5. Reflection retry (optional endpoint)
- Judge returns strict JSON:
  - `confidence` (0~1)
  - `needs_retry` (bool)
  - `reason` (string)
- Retry loop re-runs retrieval/generation when threshold not satisfied

## 5. API Endpoints
Main online endpoints in `api.py`:

- `POST /retrieve`
  - Returns expanded queries and reranked documents
- `POST /ask`
  - Returns answer plus evidence documents
- `POST /ask_with_reflection`
  - Adds confidence-based retry and judge reason
- `POST /phase3/evaluate_sample`
  - Runs layered evaluation + hard negatives + multi-judge output
- `POST /phase4/ablation`
  - Runs ablation simulation from evaluation JSON
- `POST /phase4/error_analysis`
  - Outputs error-bucket statistics
- `POST /phase4/long_context_probe`
  - Generates long-context needle probe sample

## 6. Environment Configuration
Create `.env` in project root (do not commit it). Example:

```env
OPENAI_API_KEY=replace_with_your_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=.cache/huggingface
RERANKER_MODEL=BAAI/bge-reranker-base
```

Notes:
- `.env` is ignored by Git via `.gitignore`
- `env_utils.py` enforces fast-fail on missing required variables
- `HF_ENDPOINT` supports mirror acceleration (recommended `https://hf-mirror.com`)

## 7. Quick Start
From project root:

1. Install dependencies
```bash
pip install -U fastapi uvicorn langchain langchain-openai langchain-community sentence-transformers chromadb pandas pydantic tqdm
```

2. (Optional) Pre-pull models from mirror
```bash
python code/rag/pull_models.py
```

3. Start API service
```bash
uvicorn code.rag.api:app --host 0.0.0.0 --port 8000
```

4. Run one-command reporting pipeline
```bash
python code/rag/run_pipeline.py --eval-json code/rag/outputs/large/rag_eval_results.json --output-dir code/rag/reports/latest
```

5. Pipeline with model pull
```bash
python code/rag/run_pipeline.py --pull-models --eval-json code/rag/outputs/large/rag_eval_results.json --output-dir code/rag/reports/latest
```

6. Run task-v2 pipeline (real ablation + significance + query gain + error dashboard)
```bash
python code/rag/run_pipeline.py --task-v2 --task-v2-config configs/task_v2.yaml --output-dir code/rag/reports/task_v2/latest
```

7. (Optional) Generate repeated-run config for stability tests
```bash
python code/rag/run_pipeline.py --task-v2-repeat-setup --task-v2-config configs/task_v2.yaml --task-v2-repeat-count 3 --task-v2-repeat-output-dir code/rag/outputs/repeat_runs/latest --task-v2-repeat-output-config configs/task_v2.repeated.generated.yaml --task-v2-repeat-overwrite
```

8. (Optional) Generate adaptive variant with policy + repeated noisy runs
```bash
python code/rag/run_pipeline.py --task-v2-adaptive-setup --task-v2-config configs/task_v2.yaml --task-v2-adaptive-policy-path configs/adaptive_retrieval.yaml --task-v2-adaptive-repeat-runs 3 --task-v2-adaptive-seed 42 --task-v2-adaptive-output-dir code/rag/outputs/adaptive/latest --task-v2-adaptive-output-config configs/task_v2.adaptive.generated.yaml
```

## 8. Current Baseline Metrics
From `code/rag/reports/latest/summary.json` (`eval_count=50`):

- Context Recall: `0.758`
- Context Precision: `0.624`
- Faithfulness: `0.956`
- Answer Relevance: `0.922`

Phase4 simulation examples:
- Removing BM25 lowers recall by about `0.05`
- Removing reranker lowers precision by about `0.04`
- Removing reflection lowers faithfulness by about `0.03`
- Removing query rewrite lowers relevance by about `0.02`

## 9. Engineering Principles
This project follows:

- Fast-fail strategy
  - Required env vars are validated early
  - Structured parsing failures raise exceptions directly
- Security baseline
  - API key moved out of source code into `.env`
  - `.env` tracked by ignore rules
- Reuse mature libraries
  - LangChain ecosystem for retrieval/orchestration
  - sentence-transformers for reranking
  - pandas for report export
  - FastAPI for service interface

## 10. Suggested Next Iterations
For stronger algorithm-depth presentation:

1. Replace simulated ablation with real switchable inference-time ablation runs
2. Add statistical significance testing for metric differences
3. Migrate deprecated LangChain imports to latest split packages
4. Add reproducible experiment configs (YAML) and seeded runs
5. Add CI checks for lint, type hints, and smoke API tests

## 11. Task-v2 Deliverables
Task-v2 implementation adds:

- Real ablation runner (config-driven variants)
- Statistical significance tests (Wilcoxon + bootstrap CI + Cohen's d)
- Query-type gain analysis (`fact/multi-hop/negative`)
- Retrieval error taxonomy dashboard and top-risk cases
- Benchmark protocol docs and schema
- CI smoke workflow

Main files:

- `configs/task_v2.yaml`
- `benchmark/README.md`
- `benchmark/benchmark_schema.json`
- `code/rag/task_v2_pipeline.py`
- `code/rag/task_v2/*.py`
- `tests/smoke/test_task_v2_pipeline.py`
- `.github/workflows/ci.yml`
