"""
Magnitude-aware model — does the SIZE of a change predict win-rate, where the count didn't?

`predict.py` used net_buff = (#buffs - #nerfs), a count: it treats a +5 HP tweak and a
full ability rework identically. Here we parse each change's `old -> new` into a signed
percent magnitude (using the LLM's buff/nerf label for direction, the numbers for size),
sum it per champion, and compare a COUNT model against a MAGNITUDE model head-to-head:

    count model:      delta ~ net_buff         + prior_winrate + role
    magnitude model:  delta ~ total_magnitude  + prior_winrate + role

If magnitude has a real effect where the count didn't, that's evidence the project's
original vision (predict the effect of a *specific-sized* change) is the right direction.

Usage:  python src/magnitude.py --min-games 30
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED  # noqa: E402
from winrates import load_winrates, patch_boundary  # noqa: E402

DEFAULT_BOUNDARIES = [("16.5", "16.4"), ("16.6", "16.5"), ("16.7", "16.6"), ("16.8", "16.7"),
                      ("16.9", "16.8"), ("16.10", "16.9"), ("16.11", "16.10"),
                      ("16.12", "16.11"), ("16.13", "16.12")]
_DIR = {"buff": 1.0, "nerf": -1.0}   # adjust/new/removed -> 0 (direction unknown)
MAG_CAP = 100.0                      # cap one change's |%| so a rework can't dominate the sum


def _primary_nums(s: str) -> list[float]:
    """Numbers from the main value, dropping ratio parentheticals like '(+130% AD)'."""
    s = str(s).split("(")[0]
    return [float(x) for x in re.findall(r"-?\d+\.?\d+|-?\d+", s)]


def _signed_magnitude(change: dict) -> float:
    """|percent change of the numbers|, signed by the LLM's buff/nerf label.

    Using the label for direction handles 'lower is better' fields (a cooldown drop is a
    buff) without hand-coding which stats invert. Non-numeric changes contribute 0 size."""
    old, new = _primary_nums(change.get("old", "")), _primary_nums(change.get("new", ""))
    if not old or not new or mean(old) == 0:
        return 0.0
    pct = abs((mean(new) - mean(old)) / abs(mean(old)) * 100.0)
    return _DIR.get(change.get("change_type"), 0.0) * min(pct, MAG_CAP)


def champ_features(changes: list[dict]) -> dict:
    buffs = sum(1 for c in changes if c.get("change_type") == "buff")
    nerfs = sum(1 for c in changes if c.get("change_type") == "nerf")
    return {
        "n_changes": len(changes),
        "net_buff": buffs - nerfs,
        "total_magnitude": sum(_signed_magnitude(c) for c in changes),
        "has_base_stat": int(any("base" in c.get("target", "").lower() for c in changes)),
    }


def load_changes(new_patch: str) -> dict[str, dict]:
    path = DATA_PROCESSED / f"extracted_{new_patch}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8")).get("changes", [])
    by_champ: dict[str, list] = {}
    for c in data:
        by_champ.setdefault(c["champion"], []).append(c)
    return {champ: champ_features(chs) for champ, chs in by_champ.items()}


def build_pool(min_games: int) -> pd.DataFrame:
    wr = load_winrates()
    zero = {"n_changes": 0, "net_buff": 0, "total_magnitude": 0.0, "has_base_stat": 0}
    rows = []
    for new, old in DEFAULT_BOUNDARIES:
        panel = patch_boundary(wr, new, old, min_games=min_games)
        if panel.empty:
            continue
        feats = load_changes(new)
        for r in panel.itertuples():
            f = feats.get(r.champion, zero)
            rows.append({"boundary": f"{old}->{new}", "champion": r.champion, "role": r.role,
                         "prior_winrate": r.wr_old, "delta": r.delta, "games_new": r.games_new, **f})
    return pd.DataFrame(rows)


def fit_and_backtest(pool: pd.DataFrame, feature: str, present: list[str]):
    formula = f"delta ~ {feature} + prior_winrate + C(role)"
    full = smf.wls(formula, data=pool, weights=np.sqrt(pool["games_new"])).fit()
    test_b = present[-1]
    train, test = pool[pool["boundary"] != test_b], pool[pool["boundary"] == test_b]
    model = smf.wls(formula, data=train, weights=np.sqrt(train["games_new"])).fit()
    pred = model.predict(test)
    mae = float(np.mean(np.abs(test["delta"] - pred)))
    return full.params[feature], full.pvalues[feature], mae


def main() -> None:
    ap = argparse.ArgumentParser(description="Count vs magnitude feature comparison")
    ap.add_argument("--min-games", type=int, default=30)
    args = ap.parse_args()

    pool = build_pool(args.min_games)
    order = [f"{o}->{n}" for n, o in DEFAULT_BOUNDARIES]
    present = [b for b in order if b in set(pool["boundary"])]
    if len(present) < 2:
        print("Not enough boundaries with data.")
        return

    changed = pool[pool["net_buff"] != 0]
    mae_zero = float(np.mean(np.abs(pool[pool["boundary"] == present[-1]]["delta"])))
    print(f"Pooled {len(pool)} obs, {len(changed)} changed, across {present}")
    print(f"Predict-zero MAE on held-out {present[-1]}: {mae_zero*100:.2f} pp\n")

    print(f"{'feature':16} {'coef (pp)':>11} {'p-value':>9} {'backtest MAE':>13}")
    print("-" * 52)
    for feat, unit in [("net_buff", "per net buff"), ("total_magnitude", "per +1% mag")]:
        coef, p, mae = fit_and_backtest(pool, feat, present)
        star = " *" if p < 0.05 else ""
        print(f"{feat:16} {coef*100:>+8.3f}    {p:>9.3f} {mae*100:>10.2f} pp{star}")

    print("\n* = significant at p<0.05. If total_magnitude is significant where net_buff is "
          "not,\n  the SIZE of a change carries signal the count throws away.")
    # quick sanity: show the biggest-magnitude changed champs on the latest boundary
    latest = changed[changed["boundary"] == present[-1]].copy()
    if not latest.empty:
        latest["total_magnitude"] = latest["total_magnitude"].round(1)
        latest["delta_pp"] = (latest["delta"] * 100).round(1)
        print(f"\nLargest-magnitude changes in {present[-1]} (magnitude vs actual win-rate move):")
        show = latest.reindex(latest["total_magnitude"].abs().sort_values(ascending=False).index)
        print(show[["champion", "role", "net_buff", "total_magnitude", "delta_pp"]]
              .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
