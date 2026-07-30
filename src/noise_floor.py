"""
Why win-rate prediction is closed — computed from the data, not asserted.

The project's central negative result is that a champion's patch-to-patch win-rate movement
is almost entirely binomial sampling noise, so no feature can predict it. That claim was
previously carried only as prose in docstrings. This script derives every number in it, so
the claim is reproducible and re-checks itself against current data.

What it reports, and what each one means:

  1. VARIANCE DECOMPOSITION. If a champion's win-rate moves from one patch to the next,
     how much of that movement is real change and how much is coin-flipping? A win-rate
     measured over n games has binomial variance p(1-p)/n. Summed over both patches, that
     is the movement we'd see even from a champion whose true strength never changed.
     Compare it to the movement actually observed.

  2. RELIABILITY of a single champion-role win-rate: the share of the spread ACROSS
     champions that reflects real differences rather than measurement error. 1.0 = perfect
     measurement, 0 = pure noise. Reported for win-rate and pick share side by side.

  3. LAG-1 AUTOCORRELATION: does a champion's win-rate this patch tell you its win-rate
     next patch? A reliable measurement persists; noise doesn't.

  4. THE MAE FLOOR: the error an ORACLE would post — a model that knew every champion's
     true strength exactly, and still had to predict a win-rate measured over finitely many
     games. This is the best score any model can achieve. For a proportion measured over n
     games, E|observed - true| = sqrt(2/pi) * sqrt(p(1-p)/n).

  5. BAKE-OFF: simple predictors scored against that floor, so you can see how much room
     is actually left.

Usage:
  python src/noise_floor.py                      # min_games 20, 100, 200
  python src/noise_floor.py --min-games 20
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from winrates import load_winrates  # noqa: E402
from pickrate import _step_width, MIN_TOTAL_GAMES  # noqa: E402
from magnitude_by_type import _pk  # noqa: E402

# E|X - E X| for a normal is sqrt(2/pi) * sigma. A win-rate over n games is close enough to
# normal at these sample sizes for this to be the right conversion from variance to MAE.
MAD_OVER_SD = float(np.sqrt(2 / np.pi))


def build_pairs(min_games: int) -> pd.DataFrame:
    """Consecutive-patch pairs of the same champion-role, both sides above min_games.

    Uses the same adjacency rule as the pick-rate panel: a step must span exactly one real
    patch, so a gap in the data can't masquerade as one patch of movement."""
    wr = load_winrates()
    tot = wr.groupby("patch")["games"].sum().rename("tot")
    wr = wr.join(tot, on="patch")
    wr = wr[wr["tot"] >= MIN_TOTAL_GAMES].copy()
    wr["pick"] = wr["games"] / wr["tot"]

    patches = sorted(wr["patch"].unique(), key=_pk)
    tix = {p: i for i, p in enumerate(patches)}
    wr["t"] = wr["patch"].map(tix)

    prev = wr[["champion", "role", "t", "patch", "winrate", "games", "pick"]].copy()
    prev["t"] += 1
    prev = prev.rename(columns={"patch": "patch_l1", "winrate": "wr_l1",
                                "games": "g_l1", "pick": "pick_l1"})
    p = wr.merge(prev, on=["champion", "role", "t"], how="inner")
    p = p[(p["games"] >= min_games) & (p["g_l1"] >= min_games)]
    p = p[[_step_width(a, b) == 1 for a, b in zip(p["patch_l1"], p["patch"])]].copy()
    p["delta"] = p["winrate"] - p["wr_l1"]
    return p


def analyse(min_games: int) -> dict:
    p = build_pairs(min_games)
    if len(p) < 50:
        return {"min_games": min_games, "n": len(p)}

    # --- 1. variance decomposition of the patch-to-patch win-rate CHANGE ----------------
    # Pool the two patches for the best estimate of the champion's true rate, then ask how
    # much variance two independent binomial draws of that rate would produce on their own.
    pooled = ((p["winrate"] * p["games"] + p["wr_l1"] * p["g_l1"])
              / (p["games"] + p["g_l1"]))
    var_binom = float(np.mean(pooled * (1 - pooled) * (1 / p["games"] + 1 / p["g_l1"])))
    var_obs = float(np.var(p["delta"], ddof=1))
    noise_share = var_binom / var_obs

    # --- 2. reliability of a single measurement (levels, not changes) -------------------
    # reliability = true-score variance / observed variance, where observed = true + error.
    def reliability(values: pd.Series, err_var: np.ndarray) -> float:
        v_obs = float(np.var(values, ddof=1))
        return max(0.0, (v_obs - float(np.mean(err_var))) / v_obs)

    rel_wr = reliability(p["winrate"], (p["winrate"] * (1 - p["winrate"]) / p["games"]).values)
    # Pick share is measured against EVERY game sampled that patch, not the champion's own
    # handful, so its error variance uses the patch total as the denominator.
    tot_games = p["games"] / p["pick"]
    rel_pick = reliability(p["pick"], (p["pick"] * (1 - p["pick"]) / tot_games).values)

    # --- 3. lag-1 autocorrelation of the LEVEL ------------------------------------------
    ac_wr = float(p["winrate"].corr(p["wr_l1"]))
    ac_pick = float(p["pick"].corr(p["pick_l1"]))

    # --- 4. the oracle's MAE: knowing true strength exactly, sampling noise remains ------
    floor = float(np.mean(MAD_OVER_SD * np.sqrt(p["winrate"] * (1 - p["winrate"]) / p["games"])))

    # --- 5. bake-off: simple predictors of the NEXT patch's observed win-rate ------------
    grand = float(p["wr_l1"].mean())
    mae = {
        "persistence (predict last patch)": float(np.mean(np.abs(p["winrate"] - p["wr_l1"]))),
        "grand mean (predict the average)": float(np.mean(np.abs(p["winrate"] - grand))),
    }
    # Linear reversion: regress this patch's rate on last patch's, fit on the same data.
    # Generous to the baseline (it sees the answers), which only strengthens the argument.
    b, a = np.polyfit(p["wr_l1"], p["winrate"], 1)
    mae["linear reversion (fit in-sample)"] = float(np.mean(np.abs(p["winrate"] - (a + b * p["wr_l1"]))))

    return {"min_games": min_games, "n": len(p),
            "var_obs_pp2": var_obs * 1e4, "var_binom_pp2": var_binom * 1e4,
            "noise_share": noise_share, "rel_wr": rel_wr, "rel_pick": rel_pick,
            "ac_wr": ac_wr, "ac_pick": ac_pick, "floor_pp": floor * 100,
            "mae": {k: v * 100 for k, v in mae.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive the win-rate noise-floor result")
    ap.add_argument("--min-games", type=int, nargs="+", default=[20, 100, 200])
    args = ap.parse_args()

    for mg in args.min_games:
        r = analyse(mg)
        print("=" * 68)
        print(f"min_games = {mg}")
        print("=" * 68)
        if r.get("n", 0) < 50:
            print(f"  only {r.get('n', 0)} paired observations — not enough to analyse.\n")
            continue

        print(f"Paired champion-role observations: {r['n']:,}\n")

        print("1. VARIANCE DECOMPOSITION of the patch-to-patch win-rate change")
        print(f"     observed variance of the change : {r['var_obs_pp2']:8.2f} pp^2")
        print(f"     variance from sampling alone    : {r['var_binom_pp2']:8.2f} pp^2")
        print(f"     -> SAMPLING NOISE EXPLAINS {r['noise_share'] * 100:.1f}% OF ALL MOVEMENT\n")

        print("2. RELIABILITY of one measurement (1.0 = perfect, 0 = pure noise)")
        print(f"     win-rate  : {r['rel_wr']:.3f}")
        print(f"     pick share: {r['rel_pick']:.3f}\n")

        print("3. LAG-1 AUTOCORRELATION (does this patch predict the next?)")
        print(f"     win-rate  : {r['ac_wr']:.3f}")
        print(f"     pick share: {r['ac_pick']:.3f}\n")

        print("4. THE FLOOR — MAE of an ORACLE that knows every champion's true strength")
        print(f"     {r['floor_pp']:.3f} pp   <- no model can beat this\n")

        print("5. BAKE-OFF (MAE predicting the next patch's win-rate)")
        for k, v in sorted(r["mae"].items(), key=lambda kv: kv[1]):
            gap = v - r["floor_pp"]
            print(f"     {k:36} {v:6.3f} pp   ({gap:+.3f} vs floor)")
        print()


if __name__ == "__main__":
    main()
