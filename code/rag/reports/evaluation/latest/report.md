# Evaluation Experiment Report

- Generated at: 2026-05-22T06:29:02.951076+00:00
- Baseline variant: full

## Real Ablation Summary

| variant | recall | precision | faithfulness | relevance |
|---|---:|---:|---:|---:|
| full | 0.4600 | 0.5100 | 0.9700 | 0.8800 |
| chunk200_baseline | 0.5800 | 0.6100 | 0.8500 | 0.9700 |

## Significance Tests (vs baseline)

| variant | metric | mean_diff | cohen_d | p_value | ci95 | significant |
|---|---|---:|---:|---:|---:|---|
| chunk200_baseline | context_recall | 0.1200 | 0.4140 | 0.500000 | [0.0000, 0.3157] | False |
| chunk200_baseline | context_precision | 0.1000 | 0.3873 | 0.375000 | [-0.0100, 0.2800] | False |
| chunk200_baseline | faithfulness | -0.1200 | -0.4140 | 0.500000 | [-0.3300, 0.0000] | False |
| chunk200_baseline | answer_relevance | 0.0900 | 0.3162 | 1.000000 | [0.0000, 0.2700] | False |

## Gain by Query Type

| variant | q_type | metric | gain | count |
|---|---|---|---:|---:|
| full | fact | context_recall | 0.0000 | 4 |
| full | fact | context_precision | 0.0000 | 4 |
| full | fact | faithfulness | 0.0000 | 4 |
| full | fact | answer_relevance | 0.0000 | 4 |
| full | multi-hop | context_recall | 0.0000 | 3 |
| full | multi-hop | context_precision | 0.0000 | 3 |
| full | multi-hop | faithfulness | 0.0000 | 3 |
| full | multi-hop | answer_relevance | 0.0000 | 3 |
| full | negative | context_recall | 0.0000 | 3 |
| full | negative | context_precision | 0.0000 | 3 |
| full | negative | faithfulness | 0.0000 | 3 |
| full | negative | answer_relevance | 0.0000 | 3 |
| chunk200_baseline | fact | context_recall | 0.3000 | 4 |
| chunk200_baseline | fact | context_precision | 0.2250 | 4 |
| chunk200_baseline | fact | faithfulness | -0.2250 | 4 |
| chunk200_baseline | fact | answer_relevance | 0.2250 | 4 |
| chunk200_baseline | multi-hop | context_recall | 0.0000 | 3 |
| chunk200_baseline | multi-hop | context_precision | 0.0333 | 3 |
| chunk200_baseline | multi-hop | faithfulness | -0.1000 | 3 |
| chunk200_baseline | multi-hop | answer_relevance | 0.0000 | 3 |
| chunk200_baseline | negative | context_recall | 0.0000 | 3 |
| chunk200_baseline | negative | context_precision | 0.0000 | 3 |
| chunk200_baseline | negative | faithfulness | 0.0000 | 3 |
| chunk200_baseline | negative | answer_relevance | 0.0000 | 3 |

## Error Dashboard

| variant | error_label | count | avg_faithfulness | avg_relevance |
|---|---|---:|---:|---:|
| full | no_recall | 5 | 1.0000 | 0.8200 |
| full | ok | 5 | 0.9400 | 0.9400 |
| chunk200_baseline | no_recall | 4 | 0.7750 | 1.0000 |
| chunk200_baseline | ok | 6 | 0.9000 | 0.9500 |

## Answer Calibration & Refusal

| variant | correctness_mode | ece | brier | refusal_rate | accepted_accuracy_proxy | error_capture_rate_by_refusal |
|---|---|---:|---:|---:|---:|---:|
| full | proxy | 0.2320 | 0.0952 | 0.1000 | 0.8889 | 0.5000 |
| chunk200_baseline | proxy | 0.1290 | 0.1519 | 0.1000 | 0.7778 | 0.3333 |

## Refusal Threshold Sweep (Top Utility)

| variant | threshold | utility_score | refusal_rate | accepted_accuracy_proxy | error_leakage_rate_after_accept |
|---|---:|---:|---:|---:|---:|
| full | 0.75 | 0.9200 | 0.4000 | 1.0000 | 0.0000 |
| chunk200_baseline | 0.80 | 0.8800 | 0.6000 | 1.0000 | 0.0000 |

## Decision Gate

| variant | gate_passed | rules_passed | rules_total | failed_rules |
|---|---|---:|---:|---|
| full | True | 7 | 7 |  |
| chunk200_baseline | False | 4 | 8 | metric_min:faithfulness;calibration:min_accepted_accuracy_proxy;calibration:max_error_leakage_rate_after_accept;significant_improve:faithfulness |

## Competition Scorecard

| variant | weighted_pass_rate | competition_readiness |
|---|---:|---|
| full | 1.0000 | READY |
| chunk200_baseline | 0.4500 | NOT_READY |

