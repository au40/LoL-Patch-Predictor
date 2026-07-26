"""
Pick-rate model — the one outcome patch notes actually predict.

Win-rate turned out to be a dead end and we can now say why with a number: 97.8% of a
champion's patch-to-patch win-rate movement is binomial sampling noise, and the best
possible model (one that knew every champion's true strength) would still miss by 4.43pp
at min_games=20 — which is where our model already sits. Nothing is left to extract.

PICK-rate is the opposite. A champion's share of all games played is measured against the
whole sample rather than its own handful of games, so it is ~98% signal (lag-1
autocorrelation 0.93 vs win-rate's 0.03). And it responds to patch notes:

    buffed champions   +24% pick share the next patch
    nerfed champions   -12%

Out-of-sample (walk-forward, train on patches < t and predict t), adding buff/nerf flags
cuts MAE on changed champions from 31.4% to 27.9% and improves 15 of 16 folds.

What drives it is the ANNOUNCEMENT, not the size: given that a champion was announced as
changed, the magnitude adds nothing (mag_damage p=0.80) and including magnitudes makes
out-of-sample accuracy worse. So this predicts what people PLAY, not how well they perform.

But it is NOT a fad -- see persistence(). Tracking champions that were not touched again,
a buff still shows +15% three patches later, and a nerf DEEPENS from -13% to -27%. Players
rush to a buffed champion and mostly stay; they abandon a nerfed one gradually. The
buff>nerf asymmetry at the patch of the change therefore REVERSES by +3 patches.

No leakage: patch notes are public when the patch ships, so the features for patch t are
genuinely known before patch t's pick rates exist.

Usage:
  python src/pickrate.py                      # effect sizes + walk-forward backtest
  python src/pickrate.py --min-games 50
  python src/pickrate.py --save               # write the per-fold backtest to data/processed
"""
from __future__ import annotations

import os
# Pin BLAS to one thread BEFORE numpy is imported (avoids an OpenBLAS crash on Windows).
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
from winrates import load_winrates  # noqa: E402
from magnitude import _signed_magnitude  # noqa: E402
from magnitude_by_type import categorize, _pk  # noqa: E402
from datadragon import champion_id_map, normalize_champion  # noqa: E402

# Patches with very few sampled games give a meaningless share denominator.
MIN_TOTAL_GAMES = 5000

# The winning specification. `lp1` is the reversion control (log pick-share last patch),
# `mom1` the prior patch's change (kills the "Riot buffs already-declining champs" story).
# Magnitudes are deliberately ABSENT — they test null and hurt out-of-sample.
FORMULA = "y ~ lp1 + C(role) + buffed + nerfed + mom1"
FORMULA_BASE = "y ~ lp1 + C(role)"          # reversion-only baseline


def build_panel(min_games: int = 20) -> pd.DataFrame:
    """One row per (champion, role, patch) that also appears in the previous patch.

    Columns: y (Δlog pick-share, the outcome), lp1, mom1, buffed, nerfed, plus the
    magnitude features so callers can re-test them. Champion names are normalized to
    Data Dragon ids so this joins to every other table in the project."""
    idm = champion_id_map()
    wr = load_winrates()
    wr["champion"] = [normalize_champion(c, idm) for c in wr["champion"]]

    tot = wr.groupby("patch")["games"].sum().rename("tot")
    wr = wr.join(tot, on="patch")
    wr = wr[wr["tot"] >= MIN_TOTAL_GAMES].copy()
    wr["pick"] = wr["games"] / wr["tot"]

    patches = sorted(wr["patch"].unique(), key=_pk)
    tix = {p: i for i, p in enumerate(patches)}
    wr["t"] = wr["patch"].map(tix)

    base = wr[["champion", "role", "t", "patch", "pick", "games", "winrate"]]
    prev = base[["champion", "role", "t", "pick", "games", "winrate"]].copy()
    prev["t"] += 1
    prev = prev.rename(columns={"pick": "pick_l1", "games": "g_l1", "winrate": "wr_l1"})
    p = base.merge(prev, on=["champion", "role", "t"], how="inner")
    p = p[(p["games"] >= min_games) & (p["g_l1"] >= min_games)].copy()
    p["y"] = np.log(p["pick"]) - np.log(p["pick_l1"])
    p["lp1"] = np.log(p["pick_l1"])

    p = _attach_changes(p, patches, idm)

    # momentum: last patch's own pick-rate change (0 when we have no prior observation)
    mom = p[["champion", "role", "t", "y"]].copy()
    mom["t"] += 1
    p = p.merge(mom.rename(columns={"y": "mom1"}), on=["champion", "role", "t"], how="left")
    p["mom1"] = p["mom1"].fillna(0.0)
    return p


def _attach_changes(p: pd.DataFrame, patches: list[str], idm: dict) -> pd.DataFrame:
    """Join each row to its champion's changes in THAT patch's notes."""
    feats: dict[str, dict[str, list]] = {}
    for patch in patches:
        f = DATA_PROCESSED / f"extracted_{patch}.json"
        if not f.exists():
            continue
        by: dict[str, list] = {}
        for c in json.loads(f.read_text(encoding="utf-8")).get("changes", []):
            by.setdefault(normalize_champion(c["champion"], idm), []).append(c)
        feats[patch] = by

    cols = ["mag_damage", "mag_base_stat", "mag_utility", "mag_other"]
    rows = []
    for r in p.itertuples():
        f = dict.fromkeys(cols, 0.0)
        f["n_changes"] = 0
        f["net_buff"] = 0
        for c in feats.get(r.patch, {}).get(r.champion, []):
            f["mag_" + categorize(c)] += _signed_magnitude(c)
            f["n_changes"] += 1
            f["net_buff"] += (1 if c.get("change_type") == "buff"
                              else -1 if c.get("change_type") == "nerf" else 0)
        rows.append(f)
    out = pd.concat([p.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["changed"] = (out["n_changes"] > 0).astype(int)
    out["buffed"] = (out["net_buff"] > 0).astype(int)
    out["nerfed"] = (out["net_buff"] < 0).astype(int)
    return out


def fit(panel: pd.DataFrame, formula: str = FORMULA):
    """Fit the pick-rate model, clustering standard errors by champion (each champion
    contributes many patches, so plain SEs would be over-confident)."""
    return smf.ols(formula, data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["champion"]})


def effects(panel: pd.DataFrame) -> pd.DataFrame:
    """Buff/nerf effect sizes as percent change in pick-share, with 95% CIs."""
    f = fit(panel)
    ci = f.conf_int()
    return pd.DataFrame([
        {"direction": k,
         "pick_share_change_pct": round(f.params[k] * 100, 2),
         "ci_low": round(ci.loc[k, 0] * 100, 2),
         "ci_high": round(ci.loc[k, 1] * 100, 2),
         "p_value": round(f.pvalues[k], 5),
         "n_obs": int(panel[k].sum())}
        for k in ("buffed", "nerfed")])


def backtest(panel: pd.DataFrame, first_test_frac: float = 0.5) -> pd.DataFrame:
    """Walk-forward: for each patch in the back half, train on every earlier patch and
    predict it. Reports MAE on the CHANGED champions, where the feature can matter."""
    ts = sorted(panel["t"].unique())
    start = ts[int(len(ts) * first_test_frac)]
    rows = []
    for t in [x for x in ts if x >= start]:
        tr, te = panel[panel["t"] < t], panel[panel["t"] == t]
        if len(tr) < 300 or te.empty:
            continue
        ch = (te["changed"] == 1).values
        if ch.sum() < 5:
            continue
        base = np.abs(te["y"].values - smf.ols(FORMULA_BASE, data=tr).fit()
                      .predict(te).fillna(0).values)[ch]
        full = np.abs(te["y"].values - smf.ols(FORMULA, data=tr).fit()
                      .predict(te).fillna(0).values)[ch]
        rows.append({"patch": te["patch"].iloc[0], "n_changed": int(ch.sum()),
                     "mae_reversion_pct": round(base.mean() * 100, 2),
                     "mae_model_pct": round(full.mean() * 100, 2),
                     "gain_pp": round((base.mean() - full.mean()) * 100, 2)})
    return pd.DataFrame(rows)


def persistence(panel: pd.DataFrame, horizons: int = 4) -> pd.DataFrame:
    """How long does the pick-rate shift last? Effect at k patches after the change,
    measured against the patch BEFORE it.

    Only counts champions that were NOT touched again in the interim, so this is the decay
    of one change rather than the sum of several. The answer is not what you'd guess from
    the magnitude-null: buffs spike then partly settle, while nerfs COMPOUND — by +3 patches
    a nerf outweighs a buff. So the shift is durable, not a fad."""
    pick = {(r.champion, r.role, r.t): r.pick for r in panel.itertuples()}
    chg = {(r.champion, r.role, r.t): (r.buffed, r.nerfed) for r in panel.itertuples()}
    rows = []
    for r in panel.itertuples():
        if not (r.buffed or r.nerfed):
            continue
        base = pick.get((r.champion, r.role, r.t - 1))
        if not base:
            continue
        for k in range(horizons):
            nxt = pick.get((r.champion, r.role, r.t + k))
            if not nxt:
                continue
            if any(sum(chg.get((r.champion, r.role, r.t + j), (0, 0))) > 0
                   for j in range(1, k + 1)):
                continue   # touched again -- not a clean read on the original change
            rows.append({"champion": r.champion, "k": k,
                         "direction": "buff" if r.buffed else "nerf",
                         "y": np.log(nxt) - np.log(base)})
    ev = pd.DataFrame(rows)
    out = []
    for k in range(horizons):
        for d in ("buff", "nerf"):
            s = ev[(ev["k"] == k) & (ev["direction"] == d)]
            if len(s) < 10:
                continue
            f = smf.ols("y ~ 1", data=s).fit(cov_type="cluster",
                                             cov_kwds={"groups": s["champion"]})
            m = f.params["Intercept"]
            ci = f.conf_int().loc["Intercept"]
            out.append({"patches_after": k, "direction": d,
                        "pick_share_change_pct": round((np.exp(m) - 1) * 100, 1),
                        "ci_low": round((np.exp(ci[0]) - 1) * 100, 1),
                        "ci_high": round((np.exp(ci[1]) - 1) * 100, 1),
                        "n_obs": len(s)})
    return pd.DataFrame(out)


def predict_change(fitted, pick_prev: float, role: str, direction: str,
                   momentum: float = 0.0) -> dict:
    """Predicted pick-share change for one champion-role under a hypothetical patch note.

    `direction` is 'buff', 'nerf' or 'none'; `pick_prev` is its current pick share (0..1).
    Returns the multiplicative change and the resulting share, with a 95% interval."""
    row = pd.DataFrame([{
        "lp1": np.log(max(pick_prev, 1e-9)), "role": role, "mom1": momentum,
        "buffed": int(direction == "buff"), "nerfed": int(direction == "nerf"),
    }])
    pred = fitted.get_prediction(row)
    y = float(pred.predicted_mean[0])
    lo, hi = (float(v) for v in pred.conf_int()[0])
    return {"pct_change": (np.exp(y) - 1) * 100,
            "pct_low": (np.exp(lo) - 1) * 100,
            "pct_high": (np.exp(hi) - 1) * 100,
            "new_pick": pick_prev * np.exp(y)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick-rate model (patch notes -> what people play)")
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument("--save", action="store_true", help="write the per-fold backtest to CSV")
    args = ap.parse_args()

    panel = build_panel(args.min_games)
    print(f"Panel: {len(panel)} champion-role-patch rows, {panel['t'].nunique()} patches, "
          f"{int(panel['changed'].sum())} with patch-note changes "
          f"({int(panel['buffed'].sum())} buffed / {int(panel['nerfed'].sum())} nerfed)\n")

    print("Effect on pick share the patch AFTER a change (clustered by champion):")
    print(effects(panel).to_string(index=False))

    bt = backtest(panel)
    if bt.empty:
        print("\nNot enough patches for a walk-forward backtest.")
        return
    wins = int((bt["gain_pp"] > 0).sum())
    print(f"\nWalk-forward backtest ({len(bt)} folds, changed champions only):")
    print(bt.to_string(index=False))
    print(f"\n  reversion-only MAE {bt['mae_reversion_pct'].mean():.2f}%  ->  "
          f"model MAE {bt['mae_model_pct'].mean():.2f}%")
    print(f"  improves {wins}/{len(bt)} folds, mean gain {bt['gain_pp'].mean():.2f} pp")

    f = fit(panel, FORMULA + " + mag_damage + mag_base_stat + mag_utility + mag_other")
    print("\n  Does the SIZE of the change matter, given it was announced? (all null by design)")
    for k in ("mag_damage", "mag_base_stat", "mag_utility", "mag_other"):
        print(f"    {k:16}{f.params[k] * 100:+8.3f}%  p={f.pvalues[k]:.3f}")

    if args.save:
        out = DATA_PROCESSED / f"pickrate_backtest_mg{args.min_games}.csv"
        bt.to_csv(out, index=False)
        print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
