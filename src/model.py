"""
Preliminary prediction model (the Checkpoint-2 modeling deliverable).

This implements exactly what the Checkpoint-1 feedback prescribed:

  1. Difference-in-differences: for a champion c changed in patch P,
        raw shift   Delta_c = WR(c, P) - WR(c, P-1)
        baseline    = mean/median Delta over UNTOUCHED champs (meta drift)
        effect_c    = Delta_c - baseline
     -> a complete, gradeable estimate of each change's isolated effect.

  2. Regression upgrade (counters regression-to-the-mean):
        Delta ~ changed? + prior_winrate + role     (weighted by games played)
     The coefficient on `changed?` is the average effect, net of starting strength.

  3. Backtest vs a naive baseline: leave-one-out, predict each champion's Delta,
     report MAE for the model vs. "predict zero change". If we can't beat
     predict-zero, better to know now than at Checkpoint 3.

Usage:
  python src/model.py --new 16.13 --old 16.12
  python src/model.py --new 16.13 --old 16.12 --changed-json data/processed/extracted_16.13.json
"""
from __future__ import annotations

import os
# Pin BLAS to a single thread BEFORE importing numpy — avoids an OpenBLAS
# allocation crash seen on some Windows setups. Must run before numpy import.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Force UTF-8 output so non-ASCII characters don't crash on non-UTF-8 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from winrates import load_winrates, patch_boundary  # noqa: E402

# Fallback "changed champions" set for the bundled sample (matches sample_16_13.txt).
# In real use, pass --changed-json (from llm_extract.py) or wire in the Data Dragon diff.
DEFAULT_CHANGED = {"Aatrox", "Brand", "Cassiopeia", "Ezreal", "Lux"}


def changed_from_json(path: Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {c["champion"] for c in data.get("changes", [])}


def build_panel(new: str, old: str, changed: set[str], min_games: int,
                include_sample: bool = False) -> pd.DataFrame:
    df = load_winrates(include_sample=include_sample)
    panel = patch_boundary(df, new, old, min_games=min_games)
    panel["changed"] = panel["champion"].isin(changed).astype(int)
    panel["prior_winrate"] = panel["wr_old"]
    return panel


def diff_in_diff(panel: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    controls = panel[panel["changed"] == 0]
    baseline = controls["delta"].median()  # median is robust to outlier drift
    treated = panel[panel["changed"] == 1].copy()
    treated["baseline_drift"] = baseline
    treated["isolated_effect"] = treated["delta"] - baseline
    return baseline, treated


def run_regression(panel: pd.DataFrame):
    # Weighted least squares: trust high-games observations more.
    model = smf.wls(
        "delta ~ changed + prior_winrate + C(role)",
        data=panel,
        weights=np.sqrt(panel["games_new"]),
    ).fit()
    return model


def backtest_mae(panel: pd.DataFrame) -> tuple[float, float]:
    """Leave-one-out: fit on all but one champion, predict its delta.
    Return (model_MAE, predict_zero_MAE) over the changed champions."""
    changed_rows = panel[panel["changed"] == 1]
    model_errs, zero_errs = [], []
    for idx in changed_rows.index:
        train = panel.drop(index=idx)
        test = panel.loc[[idx]]
        try:
            fit = smf.wls(
                "delta ~ changed + prior_winrate + C(role)",
                data=train, weights=np.sqrt(train["games_new"]),
            ).fit()
            pred = fit.predict(test).iloc[0]
        except Exception:
            pred = train["delta"].median()
        actual = test["delta"].iloc[0]
        model_errs.append(abs(actual - pred))
        zero_errs.append(abs(actual - 0.0))  # predict-zero baseline
    return float(np.mean(model_errs)), float(np.mean(zero_errs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preliminary diff-in-diff prediction model")
    parser.add_argument("--new", required=True, help="newer patch, e.g. 16.13")
    parser.add_argument("--old", required=True, help="older patch, e.g. 16.12")
    parser.add_argument("--changed-json", help="extracted_*.json to define changed champions")
    parser.add_argument("--min-games", type=int, default=200)
    parser.add_argument("--demo", action="store_true",
                        help="include SAMPLE_* synthetic data (for the out-of-the-box demo)")
    args = parser.parse_args()

    changed = changed_from_json(Path(args.changed_json)) if args.changed_json else DEFAULT_CHANGED
    panel = build_panel(args.new, args.old, changed, args.min_games, include_sample=args.demo)

    print(f"Patch boundary {args.old} -> {args.new}   "
          f"({len(panel)} champs after min_games={args.min_games} filter, "
          f"{panel['changed'].sum()} changed / {len(panel) - panel['changed'].sum()} control)\n")

    # --- 1. Difference-in-differences -------------------------------------------
    baseline, treated = diff_in_diff(panel)
    print(f"Meta drift (median control Delta): {baseline*100:+.2f} pp\n")
    print("Diff-in-differences - isolated effect per changed champion:")
    show = treated[["champion", "role", "wr_old", "wr_new", "delta",
                    "baseline_drift", "isolated_effect"]].copy()
    for col in ["wr_old", "wr_new", "delta", "baseline_drift", "isolated_effect"]:
        show[col] = (show[col] * 100).round(2)
    print(show.to_string(index=False))

    # --- 2. Regression upgrade --------------------------------------------------
    fit = run_regression(panel)
    print("\nRegression:  delta ~ changed + prior_winrate + role  (WLS, weight=sqrt(games))")
    print(f"  coef[changed]        = {fit.params['changed']*100:+.2f} pp   "
          f"(p = {fit.pvalues['changed']:.3f})   <- avg effect of being patched, net of starting strength")
    print(f"  coef[prior_winrate]  = {fit.params['prior_winrate']:+.3f}        "
          f"(regression-to-the-mean control)")

    # --- 3. Backtest vs predict-zero -------------------------------------------
    model_mae, zero_mae = backtest_mae(panel)
    print("\nBacktest (leave-one-out over changed champs):")
    print(f"  model MAE        = {model_mae*100:.2f} pp")
    print(f"  predict-zero MAE = {zero_mae*100:.2f} pp")
    verdict = "beats" if model_mae < zero_mae else "does NOT beat"
    print(f"  -> model {verdict} the predict-zero baseline.")


if __name__ == "__main__":
    main()
