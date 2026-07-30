"""
Pick-rate model — the one outcome patch notes actually predict.

Win-rate turned out to be a dead end and we can now say why with a number: 98.7% of a
champion's patch-to-patch win-rate movement is binomial sampling noise, and the best
possible model (one that knew every champion's true strength) would still miss by 4.45pp
at min_games=20 — which is where our baselines already sit. Nothing is left to extract.
Every figure in this paragraph is derived by `src/noise_floor.py`; don't edit them by hand.

PICK-rate is the opposite. A champion's share of all games played is measured against the
whole sample rather than its own handful of games, so it is ~98% signal (reliability 0.979
and lag-1 autocorrelation 0.920, vs win-rate's 0.040 and 0.036). And it responds to patch
notes:

    buffed champions   +24% pick share the next patch
    nerfed champions   -12%

Out-of-sample (walk-forward, train on patches < t and predict t), adding buff/nerf flags
cuts MAE on changed champions from 31.4% to 27.9% and improves 15 of 16 folds.

What drives it is HOW MUCH OF THE CHAMPION GOT TOUCHED, not how hard. The magnitude of a
change adds nothing (mag_damage p=0.80) and including magnitudes makes out-of-sample
accuracy worse -- but the COUNT of buff/nerf lines does carry signal (net_buff p=0.03, and
it beats the bare flags out-of-sample 28.22% vs 28.68% MAE). A champion with five buff
bullets moves more than one with a single line, regardless of the numbers in them. Read
that as prominence in the notes rather than power: five bullets look like a bigger deal.
So this predicts what people PLAY, not how well they perform.

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
FORMULA = "y ~ lp1 + C(role) + buffed + nerfed + mom1 + net_buff"
FORMULA_BASE = "y ~ lp1 + C(role)"          # reversion-only baseline

# Effect sizes are reported WITHOUT net_buff. With the count in the model, `buffed` becomes
# "the jump from unchanged to buffed, holding the number of buff lines fixed" (+15%), which
# is not the question a reader is asking. Dropping it gives the total marginal effect of
# appearing in the notes as buffed (+24%). Same data, different question -- prediction wants
# the count, interpretation wants the flag alone.
FORMULA_EFFECTS = "y ~ lp1 + C(role) + buffed + nerfed + mom1"


def _step_width(prev_patch: str, patch: str) -> int:
    """How many real patches separate these two, as an integer (1 = truly consecutive).

    Season rollover counts as 1: 15.24 -> 16.1 is one patch apart even though the numbers
    jump. Anything we can't reason about (a multi-season jump) returns a large number so
    the adjacency guard rejects it rather than guessing."""
    pmaj, pmin = _pk(prev_patch)
    nmaj, nmin = _pk(patch)
    if pmaj == nmaj:
        return nmin - pmin
    if nmaj == pmaj + 1 and nmin == 1:
        return 1
    return 99


def build_panel(min_games: int = 20, require_adjacent: bool = True) -> pd.DataFrame:
    """One row per (champion, role, patch) that also appears in the previous patch.

    ADJACENCY GUARD (`require_adjacent`, on by default): only keep steps that span exactly
    one real patch. Without it a gap in the data gets modelled as a normal step -- the
    outcome then covers two patches of player behaviour while the features cover one, so
    champions changed in the skipped patch are mislabelled `changed=0` and pollute the
    control group, while champions changed in the visible patch get credited with two
    patches of movement. Both errors at once. Pass False to inspect what's being excluded;
    the `step_width` column is kept either way.

    Columns: y (Δlog pick-share, the outcome), lp1, mom1, buffed, nerfed, step_width, plus
    the magnitude features so callers can re-test them. Champion names are normalized to
    Data Dragon ids so this joins to every other table in the project."""
    idm = champion_id_map()
    wr = load_winrates()
    wr["champion"] = [normalize_champion(c, idm) for c in wr["champion"]]

    tot = wr.groupby("patch")["games"].sum().rename("tot")
    wr = wr.join(tot, on="patch")

    # Index t over EVERY patch we have data for, BEFORE the volume filter. Indexing the
    # survivors instead would renumber them, closing the hole a dropped patch leaves and
    # silently gluing its neighbours into one "consecutive" step.
    all_patches = sorted(wr["patch"].unique(), key=_pk)
    tix = {p: i for i, p in enumerate(all_patches)}

    wr = wr[wr["tot"] >= MIN_TOTAL_GAMES].copy()
    wr["pick"] = wr["games"] / wr["tot"]
    wr["t"] = wr["patch"].map(tix)
    patches = sorted(wr["patch"].unique(), key=_pk)

    base = wr[["champion", "role", "t", "patch", "pick", "games", "winrate"]]
    prev = base[["champion", "role", "t", "patch", "pick", "games", "winrate"]].copy()
    prev["t"] += 1
    prev = prev.rename(columns={"patch": "patch_l1", "pick": "pick_l1",
                                "games": "g_l1", "winrate": "wr_l1"})
    p = base.merge(prev, on=["champion", "role", "t"], how="inner")
    p = p[(p["games"] >= min_games) & (p["g_l1"] >= min_games)].copy()

    # How many real patches does this "one step" actually span? Holes left by patches
    # missing from the data entirely (not merely thin) survive the re-indexing above,
    # so check the patch numbers themselves.
    p["step_width"] = [_step_width(a, b) for a, b in zip(p["patch_l1"], p["patch"])]
    if require_adjacent:
        p = p[p["step_width"] == 1].copy()

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
    f = fit(panel, FORMULA_EFFECTS)
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


def persistence(panel: pd.DataFrame, horizons: int = 4,
                split_lines: bool = False) -> pd.DataFrame:
    """How long does the pick-rate shift last? Effect at k patches after the change,
    measured against the patch BEFORE it.

    Only counts champions that were NOT touched again in the interim, so this is the decay
    of one change rather than the sum of several. The answer is not what you'd guess from
    the magnitude-null: buffs spike then partly settle, while nerfs COMPOUND — by +3 patches
    a nerf outweighs a buff. So the shift is durable, not a fad.

    `split_lines=True` splits each direction by how many buff/nerf lines the champion got
    (1 vs 2+) and adds a `bucket` column. This is measurable -- 72-157 observations per cell
    -- and the split says something the pooled curve hides: the line COUNT shapes the whole
    trajectory, not just the size of the first jump. A multi-line buff starts far higher
    (+37% vs +15%) but decays toward the single-line case by +3 patches, while a multi-line
    nerf is deeper at every horizon and keeps compounding."""
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
                         "bucket": "1 line" if abs(r.net_buff) <= 1 else "2+ lines",
                         "y": np.log(nxt) - np.log(base)})
    ev = pd.DataFrame(rows)
    buckets = ("1 line", "2+ lines") if split_lines else (None,)
    out = []
    for k in range(horizons):
        for d in ("buff", "nerf"):
            for b in buckets:
                s = ev[(ev["k"] == k) & (ev["direction"] == d)]
                if b is not None:
                    s = s[s["bucket"] == b]
                if len(s) < 10:
                    continue
                f = smf.ols("y ~ 1", data=s).fit(cov_type="cluster",
                                                 cov_kwds={"groups": s["champion"]})
                m = f.params["Intercept"]
                ci = f.conf_int().loc["Intercept"]
                row = {"patches_after": k, "direction": d,
                       "pick_share_change_pct": round((np.exp(m) - 1) * 100, 1),
                       "ci_low": round((np.exp(ci[0]) - 1) * 100, 1),
                       "ci_high": round((np.exp(ci[1]) - 1) * 100, 1),
                       "n_obs": len(s)}
                if b is not None:
                    row["bucket"] = b
                out.append(row)
    return pd.DataFrame(out)


def scoreboard(panel: pd.DataFrame, patch: str | None = None,
               changed_only: bool = True) -> pd.DataFrame:
    """Named per-champion predictions for ONE patch, scored against what actually happened.

    Same protocol as `backtest` -- train on every patch strictly before this one, then
    predict it -- but keeps the per-champion rows instead of collapsing them to an MAE.
    That turns the aggregate result ("buffed champions gain ~24%") into a checkable claim
    ("the model said this champion would gain 31%, it gained 27%").

    `patch` defaults to the newest one in the panel. Returns predicted vs actual percent
    change in pick share, the reversion-only baseline for comparison, and the signed error,
    sorted by predicted movement. Empty if there isn't enough history to train on."""
    ts = sorted(panel["t"].unique())
    if patch is None:
        t = ts[-1]
    else:
        m = panel[panel["patch"] == str(patch)]
        if m.empty:
            return pd.DataFrame()
        t = int(m["t"].iloc[0])

    tr, te = panel[panel["t"] < t], panel[panel["t"] == t]
    if len(tr) < 300 or te.empty:
        return pd.DataFrame()

    # .fillna(0) mirrors backtest(): a role level absent from the training half yields NaN
    # from patsy, and "no predicted movement" is the honest fallback there.
    pred = smf.ols(FORMULA, data=tr).fit().predict(te).fillna(0).values
    base = smf.ols(FORMULA_BASE, data=tr).fit().predict(te).fillna(0).values

    out = te.copy()
    out["predicted_pct"] = (np.exp(pred) - 1) * 100
    out["reversion_pct"] = (np.exp(base) - 1) * 100
    out["actual_pct"] = (np.exp(out["y"].values) - 1) * 100
    out["error_pp"] = out["predicted_pct"] - out["actual_pct"]
    # 'mixed' matters: net_buff is a NET, so a champion with one buff line and one nerf
    # line lands at 0 and is neither buffed nor nerfed -- but it was still touched, and
    # calling it 'untouched' in a changed-only table is simply wrong.
    out["direction"] = np.where(out["buffed"] == 1, "buff",
                       np.where(out["nerfed"] == 1, "nerf",
                       np.where(out["n_changes"] > 0, "mixed", "untouched")))
    out["lines"] = out["n_changes"].astype(int)          # how many change lines in total
    out["net_lines"] = out["net_buff"].astype(int)       # buffs minus nerfs -- the feature
    out["pick_prev_pct"] = out["pick_l1"] * 100
    out["pick_now_pct"] = out["pick"] * 100

    if changed_only:
        out = out[out["changed"] == 1]
    cols = ["champion", "role", "direction", "lines", "net_lines", "pick_prev_pct",
            "pick_now_pct", "predicted_pct", "actual_pct", "reversion_pct", "error_pp",
            "games"]
    return out[cols].sort_values("predicted_pct", ascending=False).reset_index(drop=True)


def forecast(fitted, pick_prev: float, role: str, direction: str, momentum: float = 0.0,
             n_changes: int = 1, horizons: int = 4) -> list[float]:
    """Roll the model forward: the change lands at k=0, then nothing else happens.

    Each step feeds its own output back in as the next step's momentum and pick-share level,
    so this is what the fitted model alone implies over several patches. Returns cumulative
    percent change versus the patch BEFORE the change, which makes it directly comparable to
    `persistence()`.

    READ WITH CARE, and compare against persistence() rather than trusting this alone. The
    model carries ONE symmetric momentum term, so it mean-reverts buffs and nerfs at the same
    rate. Real behaviour is asymmetric -- players abandon a nerfed champion progressively but
    largely STAY on a buffed one -- so the iterated path tracks the measured nerf trajectory
    closely and decays far too fast on buffs (predicting a buffed champion ends up slightly
    NEGATIVE by +3 patches, where the data says roughly +10%). That gap is a real limitation
    of a lag-1 specification, not a bug in this function."""
    p, p0, mom, out = pick_prev, pick_prev, momentum, []
    for k in range(horizons):
        d, lines = (direction, n_changes) if k == 0 else ("none", 0)
        step = predict_change(fitted, p, role, d, mom, lines)
        new_p = step["new_pick"]
        mom = float(np.log(max(new_p, 1e-12)) - np.log(max(p, 1e-12)))
        p = new_p
        out.append((p / p0 - 1) * 100)
    return out


def forecast_path(panel: pd.DataFrame, fitted, pick_prev: float, role: str, direction: str,
                  momentum: float = 0.0, n_changes: int = 1) -> pd.DataFrame:
    """Champion-specific multi-patch forecast: where THIS champion's pick share goes.

    Built from the two things each does best, rather than from one of them doing both:

      * the LEVEL at k=0 comes from the fitted model, which knows this champion's current
        pick share, its own momentum, its role, and how many lines it got;
      * the SHAPE from k=0 onward comes from the measured persistence profile for this
        direction and line-count bucket.

    That matters because iterating the model forward instead (see `forecast()`) gets buffs
    badly wrong -- a single symmetric momentum term mean-reverts buffs and nerfs at the same
    rate, so it decays a buff to slightly negative by +3 patches where the data says roughly
    +10%. The measured shape has the real asymmetry built in: nerfs compound, buffs settle
    but hold.

    The one assumption is that this champion decays like the average champion in its bucket.
    That is far milder than assuming lag-1 dynamics describe multi-patch behaviour.

    Returns columns: patches_after, pct (cumulative % vs the patch before the change),
    lo, hi (the bucket's measured spread, shifted to this champion's anchor), n_obs."""
    bucket = "1 line" if n_changes <= 1 else "2+ lines"
    pers = persistence(panel, split_lines=True)
    src = pers[(pers["direction"] == direction) & (pers["bucket"] == bucket)]
    if src.empty:
        return pd.DataFrame()
    src = src.sort_values("patches_after")

    anchor = predict_change(fitted, pick_prev, role, direction, momentum, n_changes)
    lvl0 = 1 + anchor["pct_change"] / 100          # model's level at k=0, as a multiplier
    base0 = 1 + float(src["pick_share_change_pct"].iloc[0]) / 100   # measured level at k=0

    rows = []
    for r in src.itertuples():
        # measured level at k, expressed RELATIVE to the measured level at k=0, then applied
        # to the model's champion-specific anchor.
        ratio = (1 + r.pick_share_change_pct / 100) / base0
        rows.append({
            "patches_after": int(r.patches_after),
            "pct": (lvl0 * ratio - 1) * 100,
            "lo": (lvl0 * (1 + r.ci_low / 100) / base0 - 1) * 100,
            "hi": (lvl0 * (1 + r.ci_high / 100) / base0 - 1) * 100,
            "n_obs": int(r.n_obs),
        })
    return pd.DataFrame(rows)


def role_splits(panel: pd.DataFrame, patch: str | None = None) -> pd.DataFrame:
    """Champions whose roles moved in OPPOSITE directions after the same patch note.

    The model scores each champion-role independently, so it cannot represent a champion
    *moving between roles* -- it can only predict "more picked" or "less picked". But a buff
    aimed at one role does exactly that: the pick share doesn't grow, it migrates. Qiyana in
    16.13 is the clean case (jungle +93%, mid -27% off one buff line), and the model
    necessarily got one of the two badly wrong.

    Returns one row per (patch, champion) that appears in 2+ roles with actual movement in
    both directions, with the spread between its best and worst role. Empty if none."""
    ts = sorted(panel["t"].unique())
    patches = [str(patch)] if patch else [panel[panel["t"] == t]["patch"].iloc[0] for t in ts]
    rows = []
    for p in patches:
        sb = scoreboard(panel, p)
        if sb.empty:
            continue
        for champ, g in sb.groupby("champion"):
            if len(g) < 2 or not (g["actual_pct"] > 0).any() or not (g["actual_pct"] < 0).any():
                continue
            up, dn = g.loc[g["actual_pct"].idxmax()], g.loc[g["actual_pct"].idxmin()]
            rows.append({
                "patch": p, "champion": champ, "direction": up["direction"],
                "gained_role": up["role"], "gained_pct": round(up["actual_pct"], 1),
                "lost_role": dn["role"], "lost_pct": round(dn["actual_pct"], 1),
                "spread_pp": round(up["actual_pct"] - dn["actual_pct"], 1),
                "worst_error_pp": round(g["error_pp"].abs().max(), 1),
            })
    out = pd.DataFrame(rows)
    return out.sort_values("spread_pp", ascending=False).reset_index(drop=True) if not out.empty else out


def predict_change(fitted, pick_prev: float, role: str, direction: str,
                   momentum: float = 0.0, n_changes: int = 1) -> dict:
    """Predicted pick-share change for one champion-role under a hypothetical patch note.

    `direction` is 'buff', 'nerf' or 'none'; `pick_prev` is its current pick share (0..1).
    `n_changes` is how many separate buff/nerf lines the champion gets -- the count carries
    real signal (a champion with five buff bullets moves more than one with a single line)
    even though the SIZE of each change does not. Returns the multiplicative change and the
    resulting share, with a 95% interval."""
    sign = 1 if direction == "buff" else -1 if direction == "nerf" else 0
    row = pd.DataFrame([{
        "lp1": np.log(max(pick_prev, 1e-9)), "role": role, "mom1": momentum,
        "buffed": int(direction == "buff"), "nerfed": int(direction == "nerf"),
        "net_buff": sign * max(n_changes, 0),
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

    # Show what the adjacency guard is holding back -- these rows come back automatically
    # once the missing/thin patches are ingested.
    unguarded = build_panel(args.min_games, require_adjacent=False)
    gaps = unguarded[unguarded["step_width"] != 1]
    if not gaps.empty:
        print("Adjacency guard is EXCLUDING these non-consecutive steps "
              "(ingest the missing patch to recover them):")
        g = (gaps.groupby(["patch_l1", "patch"])
             .agg(rows=("y", "size"), changed=("changed", "sum"),
                  spans=("step_width", "first")).reset_index())
        for r in g.itertuples():
            print(f"    {r.patch_l1:>6} -> {r.patch:<6} spans {r.spans} patches | "
                  f"{r.rows:>4} rows, {int(r.changed):>3} changed")
        print(f"    total held back: {len(gaps)} rows, {int(gaps['changed'].sum())} changed\n")

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
