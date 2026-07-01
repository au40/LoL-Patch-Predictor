"""
Multi-patch PREDICTOR (the actual project goal).

model.py estimates effects on ONE patch boundary. That measures; it can't predict.
This pools MANY patch boundaries into one training set where each row is a
(champion-role, patch-boundary) observation with:

  LABEL:    delta = WR(new) - WR(old)
  FEATURES (from the LLM extraction):
    net_buff       = (# buff changes) - (# nerf changes)   <- SIGNED, the key feature
    has_base_stat  = did any change touch base stats?
    n_changes      = how many changes the champ got
    prior_winrate  = WR(old)   (regression-to-the-mean control)
    role

Then it does a TEMPORAL backtest: train on the earlier boundaries, predict the most
recent one it has never seen, and compare MAE to a naive "predict zero change" baseline.
Training on past patches to predict a future patch is what makes this a predictor.

Usage:
  python src/predict.py --min-games 10
  python src/predict.py --boundaries 16.11:16.10 16.12:16.11 16.13:16.12 --min-games 10
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED  # noqa: E402
from winrates import load_winrates, patch_boundary  # noqa: E402

# (new, old) patch pairs, OLDEST boundary first. Each new patch needs extracted_<new>.json.
DEFAULT_BOUNDARIES = [("16.9", "16.8"), ("16.10", "16.9"), ("16.11", "16.10"),
                      ("16.12", "16.11"), ("16.13", "16.12")]

FORMULA = "delta ~ net_buff + has_base_stat + prior_winrate + C(role)"


def champ_features(changes: list[dict]) -> dict:
    buffs = sum(1 for c in changes if c.get("change_type") == "buff")
    nerfs = sum(1 for c in changes if c.get("change_type") == "nerf")
    has_base = any("base" in c.get("target", "").lower() for c in changes)
    return {"n_changes": len(changes), "net_buff": buffs - nerfs,
            "has_base_stat": int(has_base)}


def load_changes(new_patch: str) -> dict[str, dict]:
    """champion -> feature dict, from extracted_<new>.json (empty if file missing)."""
    path = DATA_PROCESSED / f"extracted_{new_patch}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8")).get("changes", [])
    by_champ: dict[str, list] = {}
    for c in data:
        by_champ.setdefault(c["champion"], []).append(c)
    return {champ: champ_features(chs) for champ, chs in by_champ.items()}


def build_pool(boundaries, min_games: int) -> pd.DataFrame:
    wr = load_winrates()
    rows = []
    for new, old in boundaries:
        panel = patch_boundary(wr, new, old, min_games=min_games)
        if panel.empty:
            continue
        feats = load_changes(new)
        for r in panel.itertuples():
            f = feats.get(r.champion, {"n_changes": 0, "net_buff": 0, "has_base_stat": 0})
            rows.append({
                "boundary": f"{old}->{new}",
                "champion": r.champion, "role": r.role,
                "prior_winrate": r.wr_old, "delta": r.delta, "games_new": r.games_new,
                "changed": int(f["n_changes"] > 0), **f,
            })
    return pd.DataFrame(rows)


def fit(df: pd.DataFrame):
    return smf.wls(FORMULA, data=df, weights=np.sqrt(df["games_new"])).fit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-patch win-rate-delta predictor")
    parser.add_argument("--boundaries", nargs="+", help="new:old pairs, oldest first")
    parser.add_argument("--min-games", type=int, default=10)
    args = parser.parse_args()

    boundaries = ([tuple(b.split(":")) for b in args.boundaries]
                  if args.boundaries else DEFAULT_BOUNDARIES)
    pool = build_pool(boundaries, args.min_games)
    if pool.empty:
        print("No data. Ingest win-rates and extract patches first.")
        return

    order = [f"{o}->{n}" for n, o in boundaries]
    present = [b for b in order if b in set(pool["boundary"])]
    print(f"Pooled {len(pool)} champion-role observations across {len(present)} boundaries: {present}")
    print(f"  changed champ-observations: {int(pool['changed'].sum())}  "
          f"(buffs net_buff>0: {(pool['net_buff']>0).sum()}, nerfs net_buff<0: {(pool['net_buff']<0).sum()})\n")

    # --- What the pooled model learns (fit on everything) ----------------------
    full = fit(pool)
    print("Pooled fit (all boundaries) - what the model learned:")
    print(f"  coef[net_buff]      = {full.params['net_buff']*100:+.2f} pp per net buff  "
          f"(p={full.pvalues['net_buff']:.3f})  <- SIGNED effect; the whole point")
    print(f"  coef[has_base_stat] = {full.params['has_base_stat']*100:+.2f} pp   "
          f"(p={full.pvalues['has_base_stat']:.3f})")
    print(f"  coef[prior_winrate] = {full.params['prior_winrate']:+.3f}   (reversion control)")

    # --- Temporal backtest: train on earlier boundaries, predict the newest ----
    if len(present) < 2:
        print("\nNeed >=2 boundaries for a temporal backtest. Ingest/extract more patches.")
        return
    test_b = present[-1]
    train = pool[pool["boundary"] != test_b]
    test = pool[pool["boundary"] == test_b]
    model = fit(train)
    pred = model.predict(test)

    mae_model = float(np.mean(np.abs(test["delta"] - pred)))
    mae_zero = float(np.mean(np.abs(test["delta"] - 0.0)))
    print(f"\nTemporal backtest: train on {present[:-1]} -> predict {test_b} "
          f"({len(test)} champ-roles, unseen)")
    print(f"  model MAE        = {mae_model*100:.2f} pp")
    print(f"  predict-zero MAE = {mae_zero*100:.2f} pp")
    verdict = "BEATS" if mae_model < mae_zero else "does NOT beat"
    print(f"  -> predictor {verdict} predict-zero on the held-out future patch.")


if __name__ == "__main__":
    main()
