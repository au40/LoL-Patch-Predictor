"""
Per-stat-type model — does a specific KIND of change predict win-rate?

`magnitude.py` summed every change into one number, mixing a cooldown cut with a base-HP
bump. But those plausibly move win-rate differently. Here we bucket each change into a
stat category and give each its own signed-magnitude feature and coefficient:

    delta ~ mag_base_stat + mag_damage + mag_utility + mag_other + prior_winrate + role

If, say, base-stat changes show an effect where the pooled magnitude didn't, that's the
"different stats have different sensitivities" idea paying off. Watch the per-category
COUNTS though — with few changed champs, categories get sparse and a lone p<0.05 is more
likely noise than signal (we learned that lesson already).

Usage:  python src/magnitude_by_type.py --min-games 30
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED  # noqa: E402
from winrates import load_winrates, patch_boundary  # noqa: E402
from magnitude import _signed_magnitude, DEFAULT_BOUNDARIES  # noqa: E402

CATEGORIES = ["base_stat", "damage", "utility", "other"]


def categorize(change: dict) -> str:
    """Bucket a change by what it touches (kept to 4 buckets to preserve power)."""
    tgt = change.get("target", "").lower()
    fld = change.get("field", "").lower()
    if "base stat" in tgt:
        return "base_stat"
    if "damage" in fld:
        return "damage"
    if "cooldown" in fld or "cost" in fld or ("mana" in fld and "regen" not in fld):
        return "utility"   # tempo/resource: cooldowns and costs
    return "other"          # ratios, shields, heals, slows, ranges, durations, ...


def champ_features(changes: list[dict]) -> dict:
    feats = {f"mag_{c}": 0.0 for c in CATEGORIES}
    buffs = nerfs = 0
    for ch in changes:
        feats[f"mag_{categorize(ch)}"] += _signed_magnitude(ch)
        buffs += ch.get("change_type") == "buff"
        nerfs += ch.get("change_type") == "nerf"
    feats["net_buff"] = buffs - nerfs
    return feats


def load_changes(new_patch: str) -> dict[str, dict]:
    path = DATA_PROCESSED / f"extracted_{new_patch}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8")).get("changes", [])
    by_champ: dict[str, list] = {}
    for c in data:
        by_champ.setdefault(c["champion"], []).append(c)
    return {champ: champ_features(chs) for champ, chs in by_champ.items()}


def _pk(patch: str) -> tuple[int, int]:
    a, b = patch.split(".")
    return (int(a), int(b))


def build_pool(min_games: int, boundaries=DEFAULT_BOUNDARIES) -> pd.DataFrame:
    wr = load_winrates()
    zero = {f"mag_{c}": 0.0 for c in CATEGORIES} | {"net_buff": 0}
    rows = []
    for new, old in boundaries:
        panel = patch_boundary(wr, new, old, min_games=min_games)
        if panel.empty:
            continue
        feats = load_changes(new)
        for r in panel.itertuples():
            f = feats.get(r.champion, zero)
            rows.append({"boundary": f"{old}->{new}", "champion": r.champion, "role": r.role,
                         "prior_winrate": r.wr_old, "delta": r.delta, "games_new": r.games_new, **f})
    return pd.DataFrame(rows)


MONSTER_HINTS = ("monster", "minion", "jungle", "clear")


def _is_monster_dmg(change: dict) -> bool:
    """True if a damage change hits monsters/minions (jungle clear) rather than champion combat."""
    blob = (change.get("field", "") + " " + change.get("target", "")).lower()
    return any(h in blob for h in MONSTER_HINTS)


def damage_scatter_table(min_games: int = 20, boundaries=DEFAULT_BOUNDARIES) -> pd.DataFrame:
    """One row per champion-role-boundary that got a (nonzero) damage change, annotated with the
    specific ability(ies) and whether it is monster/jungle-clear vs champion-combat damage.
    Feeds the dashboard's damage-effect scatter (buffs trend toward higher win-rate)."""
    wr = load_winrates()
    rows = []
    for new, old in boundaries:
        panel = patch_boundary(wr, new, old, min_games=min_games)
        path = DATA_PROCESSED / f"extracted_{new}.json"
        if panel.empty or not path.exists():
            continue
        by_champ: dict[str, list] = {}
        for c in json.loads(path.read_text(encoding="utf-8")).get("changes", []):
            if categorize(c) == "damage":
                by_champ.setdefault(c["champion"], []).append(c)
        for r in panel.itertuples():
            chs = by_champ.get(r.champion)
            if not chs:
                continue
            mag = sum(_signed_magnitude(c) for c in chs)
            if round(mag, 1) == 0:
                continue
            rows.append({
                "patch": new, "champion": r.champion, "role": r.role,
                "mag_damage": round(mag, 1),
                "winrate_change_pp": round(r.delta * 100, 1),
                "games": int(r.games_new),
                "direction": "buff" if mag > 0 else "nerf",
                "damage_type": "monster / jungle-clear" if any(_is_monster_dmg(c) for c in chs)
                else "combat",
                "abilities": "; ".join(f"{c.get('target', '?')} ({c.get('change_type', '')})"
                                       for c in chs),
            })
    return pd.DataFrame(rows)


def damage_split_fit(min_games: int = 20, boundaries=DEFAULT_BOUNDARIES) -> pd.DataFrame:
    """Refit the per-stat-type model with `damage` split into combat vs monster buckets, and
    return a small table of (coef_pp, p_value, n_champs) for each. Tests whether the damage
    effect is champion combat power or mostly jungle-clear speed."""
    cats = ["base_stat", "damage_combat", "damage_monster", "utility", "other"]

    def cat2(change: dict) -> str:
        base = categorize(change)
        if base != "damage":
            return base
        return "damage_monster" if _is_monster_dmg(change) else "damage_combat"

    wr = load_winrates()
    zero = {f"mag_{c}": 0.0 for c in cats}
    rows = []
    for new, old in boundaries:
        panel = patch_boundary(wr, new, old, min_games=min_games)
        if panel.empty:
            continue
        path = DATA_PROCESSED / f"extracted_{new}.json"
        feats: dict[str, dict] = {}
        if path.exists():
            by_champ: dict[str, list] = {}
            for c in json.loads(path.read_text(encoding="utf-8")).get("changes", []):
                by_champ.setdefault(c["champion"], []).append(c)
            for champ, chs in by_champ.items():
                f = dict(zero)
                for c in chs:
                    f[f"mag_{cat2(c)}"] += _signed_magnitude(c)
                feats[champ] = f
        for r in panel.itertuples():
            rows.append({"role": r.role, "prior_winrate": r.wr_old, "delta": r.delta,
                         "games_new": r.games_new, **feats.get(r.champion, zero)})
    pool = pd.DataFrame(rows)
    cols = [f"mag_{c}" for c in cats]
    formula = "delta ~ " + " + ".join(cols) + " + prior_winrate + C(role)"
    fit = smf.wls(formula, data=pool, weights=np.sqrt(pool["games_new"])).fit()
    return pd.DataFrame([
        {"bucket": c.replace("damage_", ""), "coef_pp": round(fit.params[f"mag_{c}"] * 100, 3),
         "p_value": round(fit.pvalues[f"mag_{c}"], 3), "n_champs": int((pool[f"mag_{c}"] != 0).sum())}
        for c in ("damage_combat", "damage_monster")
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-stat-type magnitude model")
    ap.add_argument("--min-games", type=int, default=30)
    ap.add_argument("--min-patch", default=None,
                    help="only use boundaries where BOTH patches are >= this (e.g. 16.10 to "
                         "drop 16.9, which the Master+ soft-reset may have contaminated)")
    args = ap.parse_args()

    boundaries = DEFAULT_BOUNDARIES
    if args.min_patch:
        mp = _pk(args.min_patch)
        boundaries = [(n, o) for (n, o) in DEFAULT_BOUNDARIES if _pk(o) >= mp]
        print(f"Restricting to boundaries with both patches >= {args.min_patch}\n")
    pool = build_pool(args.min_games, boundaries)
    order = [f"{o}->{n}" for n, o in DEFAULT_BOUNDARIES]
    present = [b for b in order if b in set(pool["boundary"])]
    if len(present) < 2:
        print("Not enough boundaries with data.")
        return

    mag_cols = [f"mag_{c}" for c in CATEGORIES]
    formula = "delta ~ " + " + ".join(mag_cols) + " + prior_winrate + C(role)"
    full = smf.wls(formula, data=pool, weights=np.sqrt(pool["games_new"])).fit()

    # temporal backtest
    test_b = present[-1]
    train, test = pool[pool["boundary"] != test_b], pool[pool["boundary"] == test_b]
    pred = smf.wls(formula, data=train, weights=np.sqrt(train["games_new"])).fit().predict(test)
    mae_model = float(np.mean(np.abs(test["delta"] - pred)))
    mae_zero = float(np.mean(np.abs(test["delta"])))

    print(f"Pooled {len(pool)} obs across {present}\n")
    print(f"{'stat category':14} {'coef (pp)':>11} {'p-value':>9} {'# champs w/ change':>19}")
    print("-" * 56)
    for c in CATEGORIES:
        col = f"mag_{c}"
        n = int((pool[col] != 0).sum())
        coef, p = full.params[col], full.pvalues[col]
        star = " *" if p < 0.05 else ""
        print(f"{c:14} {coef*100:>+8.3f}    {p:>9.3f} {n:>19}{star}")
    print(f"{'prior_winrate':14} {full.params['prior_winrate']:>+8.3f}    "
          f"{full.pvalues['prior_winrate']:>9.3f}")

    print(f"\nTemporal backtest (predict {test_b}):  model MAE {mae_model*100:.2f} pp  vs  "
          f"predict-zero {mae_zero*100:.2f} pp  -> "
          f"{'BEATS' if mae_model < mae_zero else 'does not beat'}")
    print("\n* = p<0.05. Cross-check any hit against its champ count: a significant coef backed "
          "by\n  only a handful of champs is likely noise, not a real per-stat effect.")


if __name__ == "__main__":
    main()
