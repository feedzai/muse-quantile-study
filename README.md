# Quantile Mapping — Model Update Stability Study

A self-contained study and reference implementation of **quantile
mapping** for keeping fixed-threshold alerting rules stable across model
updates, demonstrated end to end on the public [Bank Account Fraud
(BAF)](https://github.com/feedzai/bank-account-fraud) dataset.

## The problem

Downstream systems often convert a model's score into an action using a
fixed, manually chosen threshold (e.g. "flag anything scoring above 0.7").
These thresholds, in fraud detection, are chosen to maintain a desired
alert rate (percentage of flagged events), based on analyst capacity.
Whenever the model behind that score is retrained or replaced, its raw
score distribution usually shifts, even if predictive quality stays the
same or improves — so a threshold tuned for the old model can silently
start flagging a very different fraction of events under the new one.
Re-tuning every downstream threshold by hand every time a model changes
does not scale, especially when many independent consumers each rely on
their own threshold.

## The idea

Fit a **quantile mapping**: a set of score thresholds, one per fixed
alert-rate edge, chosen so that a model's raw scores clear each threshold
at a fixed, pre-specified target rate. Because the mapping is refit per
model version (using only the new model's own scores, no labels
required), the fraction of events above any given threshold — the alert
rate — stays approximately constant across model updates, without any
downstream threshold ever needing to change.

```
mapped_score = quantile_mapping(posterior_correction(model(x)))
```

`posterior_correction` is an optional first step that corrects for score
inflation caused by training on an undersampled majority class — a common
practice in imbalanced classification (see `posterior_correction.py`).

## What this study demonstrates

An "old" and a "new" model are trained **once**, up front, on the **same**
training window, with a **wide hyperparameter gap** — a shallow,
under-tuned "legacy" model vs. a properly tuned, higher-capacity one
(different `max_depth`/`num_leaves`/`learning_rate`/`n_estimators`/
undersampling ratio). The gap needs to be wide: a mild difference
(e.g. `n_estimators` alone) leaves the two models' raw scores highly
correlated (~0.8, measured on this dataset) even on identical training
data, understating how badly a mapping calibrated for one model fits the
other. Using the *same* training window for both also means neither model
has a home-turf recency advantage over the other when their initial
mappings are fit. The two models are then frozen and never retrained
again. `p1`/`p1_5`'s quantile mapping is **also** fit once, on the single
month right after training, and never revisited — a production mapping
that stops being maintained once the old model is retired.

Every later month — strictly after the training window and that initial
mapping-fit month, so no evaluation step scores a model on data it or its
mapping has already seen — becomes one walk-forward **evaluation step**:
`p2`'s mapping is refit fresh at each step on just the month immediately
before it; `custom`'s mapping is refit fresh at each step on an expanding
window starting at the last training month; `p1`'s model+mapping and
`p1_5`'s mapping never change again.

Six predictor variants are evaluated at every step, and none of them
numerically coincide with any other, at any evaluation step including
the first one (each uses either a different model, a different mapping,
or a mapping fit on a different month than every other predictor's):

| Predictor | Model | Quantile mapping |
|---|---|---|
| `p1` | an "old" model, trained once | `mapping_old`, fit once on the month right after training, never refit |
| `p1_5` | the "new" model, trained once on the SAME window with different hyperparameters | **not refit** — paired with `p1`'s `mapping_old`, fit for a different model and never revisited: the worst-case combination |
| `p2` | the same "new" model | `mapping_new`, **refit every step**, on the nearest prior month's data |
| `custom` | new model | `mapping_custom`, **refit every step**, on an expanding window starting at the last training month (always ≥2 months wide) |
| `default` | new model | `mapping_default`, its own dedicated mapping fit once (on a month distinct from `p1`/`p1_5`'s, `p2`'s, and `custom`'s), never revisited |
| `raw` | new model | no mapping at all |

**Result** (3 BAF variants — `Base`, `Variant II`, `Variant IV` — each
run with 3 seeds (0, 1, 2); both models trained on months 0–1 — old:
`max_depth=1, num_leaves=2, learning_rate=0.5, n_estimators=20,
beta=0.05`; new: `max_depth=10, num_leaves=128, learning_rate=0.03,
n_estimators=500, beta=0.50` — `mapping_old`/the first `mapping_new` fit
on month 2, 5 walk-forward evaluation steps across months 3–7 per run):
median absolute **alert-rate** relative error — i.e.
`(observed_alert_rate - target_alert_rate) / target_alert_rate * 100` at
each of the 4 fixed threshold edges (0.2, 0.5, 0.6, 0.8) — across all 4
edges, every evaluation step, and every seed within a variant (median /
range across the 3 seeds):

| Predictor | `Base` | `Variant II` | `Variant IV` |
|---|---|---|---|
| `p2` | **14.4%** [14.4, 15.0] | **8.7%** [8.6, 9.7] | **9.5%** [9.3, 10.7] |
| `custom` | **15.0%** [13.9, 15.2] | **7.0%** [6.6, 7.2] | **7.9%** [7.0, 8.3] |
| `default` | 16.6% [15.2, 16.9] | 7.6% [6.4, 9.1] | 7.5% [6.0, 8.0] |
| `p1` | 16.5% [15.0, 18.1] | 18.8% [18.1, 19.1] | 16.3% [13.6, 16.7] |
| `p1_5` | 47.0% [45.5, 47.1] | 39.5% [37.9, 40.9] | 55.5% [52.8, 56.1] |
| `raw` | 99.7% [99.7, 99.8] | 99.8% [99.8, 99.8] | 99.8% [99.8, 99.8] |

`p2` and `custom` — the two predictors whose mapping is refit every
step — are consistently the best across all 3 variants and all 3 seeds,
with tight seed-to-seed spread. `p1_5` (new model scored through `p1`'s
stale, never-refit mapping) is consistently the worst *calibrated*
predictor, at roughly 2-7x every other calibrated predictor's error
(closest to `p1` — the other never-refit mapping — and furthest from
`p2`/`custom`), but `raw` (no mapping at all) is far worse still, at
~100% — roughly 1.8-2.5x `p1_5`'s error and 7-14x `p2`/`custom`'s. This
is not a single-seed or single-variant artifact: see
`results/summary_all_variants.json` for the full per-seed breakdown, and
`scripts/run_experiment.py --variant ... --seeds ...` to reproduce or
extend the sweep to more BAF variants/seeds.

Alert-rate error (measured at the 4 actual operating thresholds a
downstream fixed-threshold rule would use) is deliberately a more
conservative metric than raw per-bin population density: errors in
adjacent bins can partially offset when you look at the *cumulative*
alert rate at a threshold rather than each bin's density in isolation, so
this number tends to be smaller (and more representative of real
operational impact) than a bin-density-based metric would show for the
same predictors. The per-edge breakdown for `Base` makes the pattern
clear (median relative error per edge, %):

| Predictor | `>= 0.2` | `>= 0.5` | `>= 0.6` | `>= 0.8` |
|---|---|---|---|---|
| `p1` | 0.2 | 11.5 | 42.3 | 54.6 |
| `p1_5` | -63.5 | -56.7 | -25.3 | -0.1 |
| `p2` | -0.3 | 2.2 | 4.2 | 2.3 |
| `custom` | -0.5 | -1.1 | 0.3 | -0.8 |
| `default` | -0.7 | 4.4 | 16.6 | 23.4 |
| `raw` | -99.2 | -99.8 | -99.7 | -99.9 |

`p2` and `custom` (mappings refit every step) stay within a few percent
of target at every edge. `p1` and `default` (mappings fit once, never
refit) drift increasingly far from target as the edge moves up the score
range — expected, since higher edges are more sensitive to distributional
shift in the tail. `p1_5` (new model scored through `p1`'s mapping, fit
for a very different, much shallower model) is badly wrong at the low
edges (-64%, -57%) and only converges toward target at the highest edge —
exactly what "pairing a model with a mapping calibrated for a very
different model" should look like: not a uniform shift, but a
threshold-dependent distortion. `raw` (no mapping at all) collapses to
roughly -100% at every edge, since the new model's raw probability output
is nowhere near the target's alert-rate scale — a stark illustration of
why a fixed threshold on an uncalibrated score is meaningless on its own.
See `notebooks/results.ipynb` for a single-variant, single-seed
walkthrough with narrative and figures, and `results/<variant>/` (after
running `scripts/run_experiment.py`) for the full sweep's plots (a
model-update box plot, a calibration-only box plot, and a fixed-threshold
alert-rate bar chart, per variant).

## Baseline: label-based recalibration

`scripts/run_label_baseline.py` compares quantile mapping (no labels)
against recalibrating the model with fraud labels — Platt scaling
(logistic regression on the logit of the score) or isotonic regression —
which is what a client with labels could do instead. Same models, training
window and months as above. The client fixes its thresholds once, in
probability space, on month 2 (old model, calibrated on month-2 labels, so
that they hit the target alert rates). After the update, the new model is
recalibrated on the most recent labeled month available given a label delay
`d` (month `t-1-d`), and the client's thresholds are kept. `p2` refits its
mapping on the month before, from scores alone.

All numbers below are medians over the 3 variants × 3 seeds × eval
months 5–7 (the months every delay can evaluate; 27 evaluations per cell).
`p1` is the old model before the update; every other column serves the new
model.

**Alert-rate relative error (%)**: median absolute error, with the median
signed error in parentheses (+ alerts more, − alerts less than the
reference distribution prescribes). The two differ because the median of
the absolute errors is not the absolute value of the median error.

| Threshold | `p1` (old) | `p1_5` (no refit) | `p2` (quantile mapping) | Platt, d=1 | Platt, d=2 | Isotonic, d=1 | Isotonic, d=2 |
|---|---|---|---|---|---|---|---|
| `>= 0.2` | 3.1 (+1.6) | 65.1 (−65.1) | **2.4** (−2.4) | 12.6 (−0.5) | 13.7 (−4.0) | 9.5 (−8.4) | 13.0 (−3.9) |
| `>= 0.5` | 8.4 (+8.4) | 59.8 (−59.8) | **8.6** (−8.6) | 20.2 (+8.1) | 20.8 (+5.7) | 26.7 (+22.0) | 23.6 (+6.0) |
| `>= 0.6` | 23.7 (+23.7) | 37.8 (−37.8) | **17.7** (−10.7) | 21.8 (+12.5) | 19.6 (+15.4) | 24.4 (+14.3) | 28.1 (+9.4) |
| `>= 0.8` | 31.2 (+31.2) | 18.9 (−18.9) | 23.2 (−12.7) | 24.8 (+16.2) | **18.2** (+12.0) | 35.6 (+35.6) | 24.5 (+20.2) |

Among the predictors that serve the new model, quantile mapping has the
lowest median absolute error at `>= 0.2`, `>= 0.5` and `>= 0.6`; at
`>= 0.8`, Platt with a two-month delay (18.2%) and `p1_5` (18.9%) are
lower than `p2` (23.2%). Recalibration with no label delay (`d=0`, not
available in production) is no better (median 11–45% for Platt, 11–70%
for isotonic). Recalibration keeps scores interpretable as fraud
probabilities, but a fixed probability threshold does not fix the alert
volume when the model changes: a better-separating model puts a different
share of events above the same probability, and the recalibrated models
alert more than the reference from `>= 0.5` up. Platt is monotone, so
`recal_platt_old` and `recal_platt_stale` reproduce `p1` and `p1_5`
exactly — a check on the setup.

These months (5–7) are a subset of the walk-forward evaluation above, and
the models and mappings are the same: over all eval months (3–7), `Base`
alone, the signed errors of `p1`, `p1_5` and `p2` match the per-edge
table above. Months 5–7 are harder for `p2` than months 3–4 (for `Base`,
`p2` under-alerts by 17–36% at `>= 0.5`–`>= 0.8` in months 5–7), so its
error here is larger than in the first table.

**Precision and recall at the client's fixed thresholds** (recall /
precision):

| Threshold | `p1` (old) | `p1_5` (no refit) | `p2` (quantile mapping) | Platt, d=1 | Platt, d=2 | Isotonic, d=1 | Isotonic, d=2 |
|---|---|---|---|---|---|---|---|
| `>= 0.2` | 0.984 / 0.017 | 0.828 / 0.039 | 0.980 / 0.017 | 0.982 / 0.019 | 0.980 / 0.017 | 0.971 / 0.020 | 0.979 / 0.016 |
| `>= 0.5` | 0.906 / 0.030 | 0.707 / 0.059 | 0.867 / 0.034 | 0.890 / 0.032 | 0.899 / 0.029 | 0.901 / 0.030 | 0.909 / 0.030 |
| `>= 0.6` | 0.666 / 0.068 | 0.528 / 0.098 | 0.558 / 0.077 | 0.650 / 0.077 | 0.634 / 0.069 | 0.638 / 0.078 | 0.640 / 0.076 |
| `>= 0.8` | 0.512 / 0.100 | 0.432 / 0.124 | 0.388 / 0.113 | 0.486 / 0.114 | 0.468 / 0.105 | 0.512 / 0.109 | 0.511 / 0.104 |

At a fixed threshold, recall and precision follow the alert volume, not
only the model. `p1` and the recalibrated models alert more than the
reference from `>= 0.5` up (at `>= 0.6` and `>= 0.8`, `p1` alerts more
than `p2` in every evaluation), so they catch more fraud at lower
precision; `p2`'s lower recall there comes from alerting less, not from a
worse model. `p1_5` and `p2` share the same model: evaluation by
evaluation, `p1_5` only has higher recall than `p2` when it alerts more
(about a third of the evaluations at `>= 0.8`); its higher median there is
an artifact of taking medians over different evaluations. To compare the
models themselves, fix the false-positive rate instead (e.g., recall at
1% FPR): each model then gets its own threshold, and a monotone mapping
such as quantile mapping or Platt does not change the result.

Full results: `results/label_baseline/summary_months5-7.csv`,
`summary_all_months.csv` and `per_step.csv`. Every number in this section
is reproduced by `scripts/run_label_baseline.py`.

## Repository layout

```
src/muse/
    quantile_mapping.py       percentile-fit quantile mapping against a target alert-rate curve
    posterior_correction.py   undersampling-bias correction
    recalibration.py          label-based recalibration baseline (Platt / isotonic)
    data.py                   BAF loading (via kagglehub)
    models.py                 LightGBM training with majority-class undersampling
    experiment.py             walk-forward driver + p1/p1_5/p2/custom/default/raw orchestration
    plotting.py               colorblind-safe, hatched, symlog figures
scripts/
    download_data.py            one-time BAF download
    run_experiment.py           end-to-end CLI run across variants/seeds, writes results/<variant>/ + figures
    check_model_divergence.py   reproduces the old/new score correlation numbers quoted in this README
    run_label_baseline.py       label-based recalibration baseline vs quantile mapping, writes results/label_baseline/
notebooks/
    results.ipynb             a single-variant, single-seed walkthrough, with narrative and plots
tests/
    test_quantile_mapping.py  unit tests for the transform itself
    test_experiment.py        correctness properties on synthetic data
    test_recalibration.py     unit tests for the recalibration baseline
```

## Running it

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest tests/ -v                      # fast, no data download needed

uv run python scripts/download_data.py       # downloads BAF via kagglehub (~530MB)
uv run python scripts/run_experiment.py --variant Base --seeds 0
uv run python scripts/run_experiment.py --variant Base "Variant II" "Variant IV" --seeds 0 1 2
uv run python scripts/check_model_divergence.py   # reproduce the old/new score correlation numbers
uv run python scripts/run_label_baseline.py --data-dir <dir with the BAF CSVs>   # label-based recalibration baseline
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb
```

`download_data.py` uses `kagglehub`, which can download the dataset even
without an explicit Kaggle API token in most environments; if it fails,
create an API token at kaggle.com → Settings → "Create New Token" and save
it to `~/.kaggle/kaggle.json`, or export `KAGGLE_USERNAME`/`KAGGLE_KEY`.

---

## About the dataset

This study runs on the **Bank Account Fraud (BAF)** dataset suite.

**The public dataset suite is available for download through
[Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022).**

> The paper describing this dataset suite, *"Turning the Tables: Biased,
> Imbalanced, Dynamic Tabular Datasets for ML Evaluation,"* was accepted at
> **NeurIPS 2022**.
> [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html) ·
> [Datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf) ·
> [Source repository](https://github.com/feedzai/bank-account-fraud)
