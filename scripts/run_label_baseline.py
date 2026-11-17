"""Label-based recalibration baseline vs. quantile mapping, on BAF.

Same models, training window and walk-forward months as run_experiment.py
(both models trained on months 0-1, initial fits on month 2, eval months 3-7).

Quantile mapping (no labels):
  p1    old model + mapping fit on month 2
  p1_5  new model + that same old mapping (not refit)
  p2    new model + mapping refit on the month before each eval month

Label-based recalibration (Platt / isotonic), with a client that keeps fixed
thresholds in probability space:
  - on month 2, the old model is recalibrated with month-2 labels and the
    client picks the probability thresholds that give the target alert rates;
  - after the update, the new model is recalibrated on the most recent labeled
    month available given a label delay d (month t-1-d), and the client's
    thresholds are kept.
  recal_<m>_old    old model + month-2 calibrator (pre-update reference)
  recal_<m>_stale  new model + old model's calibrator (not recalibrated)
  recal_<m>_d<d>   new model recalibrated with delay d in {0, 1, 2}

Reports alert-rate relative error, precision and recall at each threshold.

    uv run python scripts/run_label_baseline.py --data-dir <dir with BAF CSVs>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx  # noqa: E402  reuse the study's model hyperparameters

from muse.data import load_variant  # noqa: E402
from muse.experiment import INTERIOR_EDGES  # noqa: E402
from muse.models import train_model  # noqa: E402
from muse.quantile_mapping import DEFAULT_TARGET_ALERT_RATES, QuantileMapping  # noqa: E402
from muse.recalibration import Recalibrator, calibration_month  # noqa: E402

METHODS = ("platt", "isotonic")
DELAYS = (0, 1, 2)
TARGET = np.array([DEFAULT_TARGET_ALERT_RATES[e] for e in INTERIOR_EDGES])


def _metrics(scores: np.ndarray, thresholds: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for edge, thr, target in zip(INTERIOR_EDGES, thresholds, TARGET):
        alert = scores >= thr
        tp = int((alert & y).sum())
        rate = float(alert.mean())
        rows.append({
            "edge": edge,
            "rel_err_pct": 100 * (rate - target) / target,
            "precision": tp / alert.sum() if alert.any() else np.nan,
            "recall": tp / y.sum(),
        })
    return rows


def run_variant(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    month = df["month"].to_numpy()
    by_month = {m: df[month == m].reset_index(drop=True) for m in sorted(set(month))}
    train_df = pd.concat([by_month[0], by_month[1]], ignore_index=True)
    old = train_model(train_df, beta=rx.BETA_OLD, seed=seed, n_estimators=rx.N_ESTIMATORS_OLD,
                      max_depth=rx.MAX_DEPTH_OLD, num_leaves=rx.NUM_LEAVES_OLD, learning_rate=rx.LEARNING_RATE_OLD)
    new = train_model(train_df, beta=rx.BETA_NEW, seed=seed, n_estimators=rx.N_ESTIMATORS_NEW,
                      max_depth=rx.MAX_DEPTH_NEW, num_leaves=rx.NUM_LEAVES_NEW, learning_rate=rx.LEARNING_RATE_NEW)

    s_old = {m: old.corrected_scores(d) for m, d in by_month.items() if m >= 2}
    s_new = {m: new.corrected_scores(d) for m, d in by_month.items() if m >= 2}
    y = {m: d["fraud_bool"].to_numpy().astype(bool) for m, d in by_month.items()}

    mapping_old = QuantileMapping.fit(s_old[2])
    cal_old = {m: Recalibrator(m).fit(s_old[2], y[2]) for m in METHODS}
    # client thresholds in probability space, picked on month 2 to hit the targets
    client_thr = {m: np.quantile(cal_old[m].predict(s_old[2]), 1 - TARGET) for m in METHODS}

    rows = []
    for t in range(3, max(by_month) + 1):
        preds = {
            "p1": (s_old[t], mapping_old.thresholds),
            "p1_5": (s_new[t], mapping_old.thresholds),
            "p2": (s_new[t], QuantileMapping.fit(s_new[t - 1]).thresholds),
        }
        for m in METHODS:
            preds[f"recal_{m}_old"] = (cal_old[m].predict(s_old[t]), client_thr[m])
            preds[f"recal_{m}_stale"] = (cal_old[m].predict(s_new[t]), client_thr[m])
            for d in DELAYS:
                c = calibration_month(t, d)
                if c < 2:  # never calibrate on the models' training months
                    continue
                cal = Recalibrator(m).fit(s_new[c], y[c])
                preds[f"recal_{m}_d{d}"] = (cal.predict(s_new[t]), client_thr[m])
        for name, (scores, thr) in preds.items():
            for r in _metrics(scores, thr, y[t]):
                rows.append({"seed": seed, "eval_month": t, "predictor": name, **r})
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    res = res.assign(abs_rel_err_pct=res.rel_err_pct.abs())
    return (res.groupby(["predictor", "edge"])
            .agg(n=("rel_err_pct", "size"), median_rel_err_pct=("rel_err_pct", "median"),
                 median_abs_rel_err_pct=("abs_rel_err_pct", "median"), max_abs_rel_err_pct=("abs_rel_err_pct", "max"),
                 median_precision=("precision", "median"), median_recall=("recall", "median"))
            .reset_index())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--variants", nargs="+", default=["Base", "Variant II", "Variant IV"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="results/label_baseline")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    parts = []
    for v in args.variants:
        df = None
        for s in args.seeds:
            part_file = out / "parts" / f"{v.replace(' ', '_')}_seed{s}.csv"
            if not part_file.exists():  # resumable: each (variant, seed) is saved as it finishes
                df = load_variant(v, args.data_dir) if df is None else df
                print(f"{v} seed {s}", flush=True)
                part_file.parent.mkdir(parents=True, exist_ok=True)
                run_variant(df, s).assign(variant=v).to_csv(part_file, index=False)
            parts.append(pd.read_csv(part_file))
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(out / "per_step.csv", index=False)
    common = res[res.eval_month >= 5]  # months every delay can evaluate
    summarize(common).to_csv(out / "summary_months5-7.csv", index=False)
    summarize(res).to_csv(out / "summary_all_months.csv", index=False)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print(summarize(common).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
