from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evaluation.utils import METRIC_KEYS


@dataclass(frozen=True)
class TestConfig:
    bootstrap_iters: int = 2000


def _cohen_d_paired(diff: np.ndarray) -> float:
    std = float(np.std(diff, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(diff) / std)


def _bootstrap_ci_mean_diff(
    diff: np.ndarray,
    seed: int,
    iters: int,
) -> tuple[float, float]:
    if diff.size == 0:
        raise ValueError("diff must not be empty")
    rng = np.random.default_rng(seed)
    means = np.empty(iters, dtype=float)
    n = diff.size
    for idx in range(iters):
        sample = diff[rng.integers(0, n, n)]
        means[idx] = float(np.mean(sample))
    low = float(np.quantile(means, 0.025))
    high = float(np.quantile(means, 0.975))
    return low, high


def _paired_metric_arrays(
    baseline_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    left = baseline_df[["question", metric]].rename(columns={metric: "baseline"})
    right = variant_df[["question", metric]].rename(columns={metric: "variant"})
    merged = left.merge(right, on="question", how="inner")
    if merged.empty:
        raise ValueError("No overlapping questions for paired significance test.")
    return (
        merged["baseline"].to_numpy(dtype=float),
        merged["variant"].to_numpy(dtype=float),
    )


def _wilcoxon_p_value(baseline: np.ndarray, variant: np.ndarray) -> tuple[float, float]:
    diff = variant - baseline
    if np.allclose(diff, 0.0):
        return 0.0, 1.0
    stat, p_value = wilcoxon(diff, zero_method="pratt", alternative="two-sided")
    return float(stat), float(p_value)


def run_significance_tests(
    variant_frames: Dict[str, pd.DataFrame],
    baseline_variant: str,
    seed: int,
    test_config: TestConfig | None = None,
) -> pd.DataFrame:
    if baseline_variant not in variant_frames:
        raise KeyError(f"Missing baseline variant: {baseline_variant}")

    cfg = test_config or TestConfig()
    baseline_df = variant_frames[baseline_variant]

    rows: List[dict] = []
    for variant_name, variant_df in variant_frames.items():
        if variant_name == baseline_variant:
            continue

        for metric in METRIC_KEYS:
            baseline_arr, variant_arr = _paired_metric_arrays(
                baseline_df=baseline_df,
                variant_df=variant_df,
                metric=metric,
            )
            diff = variant_arr - baseline_arr
            stat, p_value = _wilcoxon_p_value(baseline_arr, variant_arr)
            ci_low, ci_high = _bootstrap_ci_mean_diff(
                diff=diff,
                seed=seed,
                iters=cfg.bootstrap_iters,
            )

            rows.append(
                {
                    "baseline_variant": baseline_variant,
                    "variant": variant_name,
                    "metric": metric,
                    "n_pairs": int(diff.size),
                    "baseline_mean": float(np.mean(baseline_arr)),
                    "variant_mean": float(np.mean(variant_arr)),
                    "mean_diff": float(np.mean(diff)),
                    "effect_size_cohen_d": _cohen_d_paired(diff),
                    "wilcoxon_stat": stat,
                    "p_value": p_value,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "significant_p_lt_0_05": bool(p_value < 0.05),
                }
            )

    return pd.DataFrame(rows)

