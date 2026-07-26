"""
Streamlit dashboard - a live, at-a-glance view of the whole pipeline's current state:
data ingestion, LLM patch-note extraction, and the prediction model. It reads the real
project files and recomputes the model on the fly, so it always reflects current data.

Run it:
  .venv/Scripts/streamlit.exe run src/dashboard.py      # Windows
  streamlit run src/dashboard.py                         # if streamlit is on PATH
"""
from __future__ import annotations

import os
# Pin BLAS to one thread BEFORE numpy is imported (avoids an OpenBLAS crash on Windows).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import altair as alt  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from winrates import load_winrates  # noqa: E402
import predict as P  # noqa: E402
import magnitude_by_type as MBT  # noqa: E402
import pickrate as PR  # noqa: E402
from magnitude import _signed_magnitude  # noqa: E402
from datadragon import (champion_spell_cooldowns, change_cooldown,  # noqa: E402
                        champion_icons, champion_names, champion_id_map, normalize_champion)

_CD_MAP = champion_spell_cooldowns()

DATA_PROCESSED = ROOT / "data" / "processed"

st.set_page_config(page_title="LoL Patch Predictor", layout="wide")
st.title("LoL Patch Predictor - pipeline dashboard")
st.caption("Live view of the three Checkpoint-2 phases: data ingestion, LLM patch-note "
           "extraction, and the preliminary prediction model. Reads real project data.")

# ------------------------------------------------------------------ load data
try:
    wr = load_winrates()
except Exception as exc:  # noqa: BLE001
    st.error(f"No win-rate data on disk yet ({exc}). Run src/riot_ingest.py first.")
    st.stop()

extracted = sorted(DATA_PROCESSED.glob("extracted_*.json"))
total_matches = int(wr["games"].sum() / 10)

# ------------------------------------------------------------- pipeline status
st.subheader("Pipeline status")
s1, s2, s3 = st.columns(3)
s1.metric("① Ingestion — real matches", f"{total_matches:,}")
s2.metric("② Extraction — patches", len(extracted))
s3.metric("③ Model — patches with win-rates", wr["patch"].nunique())

# ---------------------------------------------------------------- 1. ingestion
st.subheader("① Data ingestion (Riot API + Data Dragon)")
c1, c2 = st.columns([2, 1])
with c1:
    st.write("**Champion-games per patch**")
    st.bar_chart(wr.groupby("patch")["games"].sum().sort_index())
with c2:
    st.write("**Games by role**")
    st.dataframe(wr.groupby("role")["games"].sum().sort_values(ascending=False),
                 width="stretch")

# --------------------------------------------------------------- 2. extraction
st.subheader("② LLM patch-note extraction")
rows = []
for f in extracted:
    d = json.loads(f.read_text(encoding="utf-8"))
    ch = d.get("changes", [])
    rows.append({
        "patch": str(d.get("patch", f.stem)),
        "champions": len({c["champion"] for c in ch}),
        "changes": len(ch),
        "buffs": sum(1 for c in ch if c.get("change_type") == "buff"),
        "nerfs": sum(1 for c in ch if c.get("change_type") == "nerf"),
    })
if rows:
    st.dataframe(pd.DataFrame(rows).sort_values("patch"), width="stretch", hide_index=True)
st.info(
    "Base-stat changes are auto-validated against **Data Dragon ground truth** "
    "(~89% recall / ~80% precision across 3 patches). The cross-check even caught an "
    "**undocumented micropatch** — Smolder's base AD 60→58, documented in patch 26.10 "
    "but missing from the 26.11 notes — which a patch-notes-only analysis would miss."
)

# -------------------------------------------------------------------- 3. model
st.subheader("③ Preliminary prediction model")
min_games = st.slider("Minimum games per champion-role (data-hygiene filter)", 5, 40, 20, 5)

pool = P.build_pool(P.DEFAULT_BOUNDARIES, min_games)
if pool.empty or pool["boundary"].nunique() < 2:
    st.warning("Not enough data at this threshold for a temporal backtest. Lower the slider "
               "or ingest more matches.")
else:
    order = [f"{o}->{n}" for n, o in P.DEFAULT_BOUNDARIES]
    present = [b for b in order if b in set(pool["boundary"])]
    full = P.fit(pool)

    # temporal backtest: train on earlier boundaries, predict the newest unseen one
    test_b = present[-1]
    train, test = pool[pool["boundary"] != test_b], pool[pool["boundary"] == test_b]
    pred = P.fit(train).predict(test)
    mae_model = float(np.mean(np.abs(test["delta"] - pred)))
    mae_zero = float(np.mean(np.abs(test["delta"])))
    beats = mae_model < mae_zero

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pooled observations", f"{len(pool)}", f"{int(pool['changed'].sum())} changed")
    m2.metric("net_buff effect", f"{full.params['net_buff'] * 100:+.2f} pp/buff",
              f"p = {full.pvalues['net_buff']:.2f}")
    m3.metric("Model MAE (held-out patch)", f"{mae_model * 100:.2f} pp")
    m4.metric("vs predict-zero", f"{mae_zero * 100:.2f} pp",
              "model beats it" if beats else "does not beat", delta_color="off")

    st.caption(f"Temporal backtest: train on {present[:-1]} → predict **{test_b}** (never seen). "
               f"Signed buff/nerf feature comes from the LLM extraction.")

    # diff-in-differences on the most recent boundary
    latest = pool[(pool["boundary"] == test_b) & (pool["changed"] == 1)].copy()
    if not latest.empty:
        latest["isolated_effect_pp"] = (latest["delta"] - pool[pool["boundary"] == test_b]
                                        .loc[pool["changed"] == 0, "delta"].median()) * 100
        st.write(f"**Diff-in-differences — isolated effect of each changed champ in {test_b}:**")
        st.dataframe(
            latest[["champion", "role", "net_buff", "prior_winrate", "delta", "isolated_effect_pp"]]
            .assign(prior_winrate=lambda d: (d["prior_winrate"] * 100).round(1),
                    delta=lambda d: (d["delta"] * 100).round(1),
                    isolated_effect_pp=lambda d: d["isolated_effect_pp"].round(1))
            .sort_values("isolated_effect_pp"),
            width="stretch", hide_index=True,
        )

# --------------------------------------------- 4. damage-change effect (Checkpoint-3 lead)
st.subheader("④ Damage-change effect — the strongest change signal")
st.caption(
    "Each point is a champion-role in a patch where its **damage** changed: x = size of the "
    "damage change (buffs → right), y = its win-rate change the next patch. Hover for the exact "
    "ability. ● = champion-combat damage, ▲ = monster (jungle-clear) damage. "
    "Re-uses the games filter above, so it re-renders as you move the slider or ingest more data."
)

dmg = MBT.damage_scatter_table(min_games)
if dmg.empty:
    st.info("No damage changes clear the games filter at this threshold. Lower the slider.")
else:
    pts = alt.Chart(dmg).mark_point(filled=True, opacity=0.78).encode(
        x=alt.X("mag_damage:Q", title="damage change size (signed %, buffs → right)"),
        y=alt.Y("winrate_change_pp:Q", title="win-rate change next patch (pp)"),
        color=alt.Color("direction:N", title="direction",
                        scale=alt.Scale(domain=["buff", "nerf"], range=["#2a78d6", "#e34948"])),
        shape=alt.Shape("damage_type:N", title="damage type",
                        scale=alt.Scale(domain=["combat", "monster / jungle-clear"],
                                        range=["circle", "triangle-up"])),
        size=alt.Size("games:Q", title="games", scale=alt.Scale(range=[40, 320])),
        tooltip=[alt.Tooltip("champion:N"), alt.Tooltip("role:N"), alt.Tooltip("patch:N"),
                 alt.Tooltip("mag_damage:Q", title="damage change %"),
                 alt.Tooltip("winrate_change_pp:Q", title="win-rate Δ (pp)"),
                 alt.Tooltip("games:Q"), alt.Tooltip("abilities:N", title="ability(ies)")],
    )
    trend = pts.transform_regression("mag_damage", "winrate_change_pp").mark_line(
        color="#4a3aa7", size=2.5)
    xrule = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(strokeDash=[4, 4], color="#999").encode(x="v:Q")
    yrule = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(strokeDash=[4, 4], color="#999").encode(y="v:Q")
    st.altair_chart((xrule + yrule + pts + trend).interactive(), width="stretch")

    st.write("**Is it combat power or jungle-clear speed?** Refit with the damage feature split in two:")
    try:
        split = MBT.damage_split_fit(min_games)
        d1, d2 = st.columns(2)
        for col, (_, row) in zip((d1, d2), split.iterrows()):
            sig = "significant" if row.p_value < 0.05 else "not significant"
            col.metric(f"{row.bucket} damage", f"{row.coef_pp:+.3f} pp",
                       f"p={row.p_value:.3f} · {int(row.n_champs)} champs · {sig}", delta_color="off")
        st.caption("Monster/jungle-clear damage usually carries the stronger, steadier coefficient; "
                   "pure combat damage is weaker and threshold-fragile — so the 'damage helps' signal "
                   "is partly a faster-clear effect. (Monster rests on few champs; read with care.)")
    except Exception as exc:  # noqa: BLE001
        st.info(f"Split-fit needs more data at this threshold ({exc}). Lower the slider.")

    with st.expander("See the underlying damage changes (one row per point)"):
        st.dataframe(dmg.sort_values("mag_damage"), width="stretch", hide_index=True)

# -------------------------------- ⑤ damage x cooldown (does cast frequency matter?)
st.subheader("⑤ Damage effect by ability cooldown — does cast frequency matter?")
st.caption(
    "The hypothesis: a damage buff on a low-cooldown (spammed) spell should move win-rate more than "
    "the same buff on a long-cooldown ult. If true, the **low-cooldown** trend would be steeper than "
    "the **high-cooldown** one. Each point is a champion-role's damage change; colour = the fastest-cast "
    "damaged ability's cooldown. Uses the games filter above — slide it up to test the low-data theory."
)


def _damage_cooldown_table(min_games: int) -> pd.DataFrame:
    rows = []
    for new, old in MBT.DEFAULT_BOUNDARIES:
        panel = MBT.patch_boundary(wr, new, old, min_games=min_games)
        path = DATA_PROCESSED / f"extracted_{new}.json"
        if panel.empty or not path.exists():
            continue
        by_champ: dict[str, list] = {}
        for c in json.loads(path.read_text(encoding="utf-8")).get("changes", []):
            if MBT.categorize(c) == "damage":
                by_champ.setdefault(c["champion"], []).append(c)
        for r in panel.itertuples():
            chs = by_champ.get(r.champion)
            if not chs:
                continue
            mag = sum(_signed_magnitude(c) for c in chs)
            if round(mag, 1) == 0:
                continue
            real = [cd for cd in (change_cooldown(c, _CD_MAP) for c in chs) if cd not in (None, 0)]
            cd = min(real) if real else None  # fastest-cast damaged ability
            tier = "no cooldown" if cd is None else ("low (<=10s)" if cd <= 10 else "high (>10s)")
            abilities = "; ".join(f"{c.get('target', '?')} ({c.get('change_type', '')})" for c in chs)
            rows.append({"champion": r.champion, "role": r.role, "patch": new,
                         "mag_damage": round(mag, 1), "winrate_change_pp": round(r.delta * 100, 1),
                         "cooldown": cd, "tier": tier, "games": int(r.games_new),
                         "abilities": abilities})
    return pd.DataFrame(rows)


dct = _damage_cooldown_table(min_games)
if dct.empty:
    st.info("No damage changes clear the games filter at this threshold. Lower the slider.")
else:
    base = alt.Chart(dct)
    pts = base.mark_point(filled=True, opacity=0.7).encode(
        x=alt.X("mag_damage:Q", title="damage change size (signed %, buffs -> right)"),
        y=alt.Y("winrate_change_pp:Q", title="win-rate change next patch (pp)"),
        color=alt.Color("tier:N",
                        legend=alt.Legend(title="ability cooldown", orient="top"),
                        scale=alt.Scale(domain=["low (<=10s)", "high (>10s)", "no cooldown"],
                                        range=["#eb6834", "#2a78d6", "#b4b2a9"])),
        size=alt.Size("games:Q", title="games", legend=None,
                      scale=alt.Scale(range=[40, 320])),
        tooltip=[alt.Tooltip("champion:N"), alt.Tooltip("role:N"), alt.Tooltip("patch:N"),
                 alt.Tooltip("cooldown:Q", title="cooldown (s)"),
                 alt.Tooltip("mag_damage:Q", title="damage change %"),
                 alt.Tooltip("winrate_change_pp:Q", title="win-rate change (pp)"),
                 alt.Tooltip("games:Q"),
                 alt.Tooltip("abilities:N", title="ability(ies)")],
    )
    trends = base.transform_filter(alt.datum.tier != "no cooldown").transform_regression(
        "mag_damage", "winrate_change_pp", groupby=["tier"]).mark_line(size=2.5).encode(
        x="mag_damage:Q", y="winrate_change_pp:Q",
        color=alt.Color("tier:N", legend=None,
                        scale=alt.Scale(domain=["low (<=10s)", "high (>10s)", "no cooldown"],
                                        range=["#eb6834", "#2a78d6", "#b4b2a9"])))
    xr = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(strokeDash=[4, 4], color="#999").encode(x="v:Q")
    yr = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(strokeDash=[4, 4], color="#999").encode(y="v:Q")
    st.altair_chart((xr + yr + pts + trends).interactive(), width="stretch")
    st.caption(
        "The two trend lines come out **nearly parallel** — a damage buff moves win-rate about the same "
        "whether the ability is spammed or on a long cooldown. That's why weighting damage by cooldown "
        "didn't help the model: cast frequency doesn't organise the win-rate response. Grey points are "
        "passives / zero-cooldown abilities (no cooldown to weight by)."
    )

# ------------------------- ⑥ champion picker — predict a hypothetical damage change
st.subheader("⑥ Try it — pick a champion, size a damage change, see the predicted shift")
st.caption(
    "Pick a champion below, then dial in a damage change. The model is the per-stat-type fit "
    "from ③/④ (`delta ~ mag_base_stat + mag_damage + mag_utility + mag_other + prior_winrate "
    "+ role`), asked what it expects for a champion whose **only** change this patch is damage."
)


@st.cache_data(show_spinner=False)
def _art() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Champion square icons + display names, from the champion.json already on disk."""
    return champion_icons(), champion_names(), champion_id_map()


ICONS, NAMES, _ID_MAP = _art()


@st.cache_data(show_spinner="Building the pick-rate panel…")
def _pick_panel(min_games: int) -> pd.DataFrame:
    return PR.build_panel(min_games)


@st.cache_resource(show_spinner=False)
def _pick_fit(min_games: int):
    """Fitted pick-rate model. cache_resource, not cache_data — a statsmodels results
    object is a live object, not a serialisable value."""
    return PR.fit(_pick_panel(min_games))


def _icon(champ: str) -> str | None:
    """Icon URL for a win-rate row's champion, or None if we have no art for it.

    The win-rate CSVs come from third-party aggregators whose casing drifts from Data
    Dragon's ('FiddleSticks' vs 'Fiddlesticks'), so go through normalize_champion rather
    than indexing directly — a raw lookup KeyErrors and takes the whole grid down."""
    return ICONS.get(normalize_champion(champ, _ID_MAP))


def _label(champ: str) -> str:
    """Display name ('MonkeyKing' -> 'Wukong'), falling back to the id we were given."""
    return NAMES.get(normalize_champion(champ, _ID_MAP), champ)

# Only offer champions we can actually anchor a prediction on: they need a current win-rate
# at the newest patch, above the same games filter the rest of the page uses.
latest_patch = max(wr["patch"].unique(), key=MBT._pk)
current = wr[(wr["patch"] == latest_patch) & (wr["games"] >= min_games)]

pool6 = MBT.build_pool(min_games)
if current.empty or pool6.empty:
    st.info("No champions clear the games filter at the newest patch. Lower the slider.")
else:
    formula6 = ("delta ~ " + " + ".join(f"mag_{c}" for c in MBT.CATEGORIES)
                + " + prior_winrate + C(role)")
    fit6 = smf.wls(formula6, data=pool6, weights=np.sqrt(pool6["games_new"])).fit()
    coef_dmg = float(fit6.params["mag_damage"])

    roster = sorted(current["champion"].unique(), key=_label)
    st.session_state.setdefault("picked_champ", roster[0])

    query = st.text_input("Search champions", placeholder="type a name, e.g. Jinx",
                          key="champ_search").strip().lower()
    shown = [c for c in roster if query in _label(c).lower()] if query else roster
    if not shown:
        st.warning(f"No champion matching '{query}' has data at patch {latest_patch}.")
    else:
        PER_ROW, MAX_TILES = 10, 60
        tiles = shown[:MAX_TILES]

        # Streamlit buttons can't hold an image, but any keyed widget gets an `st-key-{key}`
        # class on its container — so these are ordinary st.buttons wearing the champion icon
        # as a CSS background. That keeps normal click/session-state behaviour (a query-param
        # link would full-reload the page and wipe the games slider and search box).
        # The label stays the champion name but is rendered at font-size 0: invisible on screen,
        # still there for screen readers. Champion ids are alphanumeric, so they're safe as
        # CSS class names (checked against both Data Dragon and the win-rate CSVs).
        st.html("<style>" + """
        [class*="st-key-pick_"] button {
            height: 62px; padding: 0; border-radius: 10px;
            background-size: cover; background-position: center;
            background-color: transparent; border: 2px solid transparent;
            transition: transform .08s ease, border-color .08s ease;
        }
        [class*="st-key-pick_"] button p { font-size: 0; line-height: 0; }
        [class*="st-key-pick_"] button:hover {
            transform: translateY(-2px); border-color: #2a78d6;
        }
        """ + "".join(
            f'.st-key-pick_{cid} button {{ background-image: url("{_icon(cid)}"); }}'
            for cid in tiles if _icon(cid)
        ) + f"""
        .st-key-pick_{st.session_state.picked_champ} button {{
            border-color: #eb6834; box-shadow: 0 0 0 2px rgba(235,104,52,.35);
        }}
        </style>""")

        for i in range(0, len(tiles), PER_ROW):
            for col, cid in zip(st.columns(PER_ROW, gap="small"), tiles[i:i + PER_ROW]):
                if col.button(_label(cid), key=f"pick_{cid}", help=_label(cid),
                              width="stretch"):
                    st.session_state.picked_champ = cid
        if len(shown) > MAX_TILES:
            st.caption(f"Showing {MAX_TILES} of {len(shown)} champions — search to narrow.")

    champ = st.session_state.picked_champ
    if champ not in set(current["champion"]):
        st.info(f"{_label(champ)} has no role above {min_games} games at patch "
                f"{latest_patch}. Pick another champion or lower the slider.")
    else:
        st.divider()
        hero, controls = st.columns([1, 3])
        with hero:
            art = _icon(champ)
            if art:
                st.image(art, width=110)
            st.markdown(f"### {_label(champ)}")
        with controls:
            roles = sorted(current.loc[current["champion"] == champ, "role"].unique())
            c_role, c_mag = st.columns([1, 2])
            role = c_role.selectbox("Role", roles, key="pick_role")
            mag = c_mag.slider("Damage change (signed %, buff → right)",
                               -30.0, 30.0, 5.0, 0.5, key="pick_mag")

            here = current[(current["champion"] == champ) & (current["role"] == role)]
            prior = float(here["winrate"].iloc[0])
            games = int(here["games"].iloc[0])

            row = pd.DataFrame([{**{f"mag_{c}": 0.0 for c in MBT.CATEGORIES},
                                 "mag_damage": mag, "prior_winrate": prior, "role": role}])
            pr = fit6.get_prediction(row)
            delta = float(pr.predicted_mean[0])
            lo, hi = (float(x) for x in pr.conf_int()[0])

            k1, k2, k3 = st.columns(3)
            k1.metric(f"Win-rate now ({latest_patch})", f"{prior * 100:.1f}%",
                      f"{games:,} games", delta_color="off")
            k2.metric("Predicted next patch", f"{(prior + delta) * 100:.1f}%",
                      f"{delta * 100:+.2f} pp")
            k3.metric("Of which the damage change", f"{coef_dmg * mag * 100:+.2f} pp",
                      f"{coef_dmg * 100:+.3f} pp per 1%", delta_color="off")
            st.caption(f"95% CI on the predicted shift: {lo * 100:+.2f} to {hi * 100:+.2f} pp. "
                       f"The gap between the middle and right numbers is **reversion to the mean** — "
                       f"at {prior * 100:.1f}% the model expects this champion to drift toward average "
                       f"whether or not Riot touches it. Damage is the part you're actually buying.")

            # --- the same change, scored against the outcome that IS predictable -------
            st.markdown("**…and what it does to how often the champion gets *played*:**")
            ppanel = _pick_panel(min_games)
            mine = ppanel[(ppanel["champion"] == champ) & (ppanel["role"] == role)]
            if mine.empty:
                st.info(f"{_label(champ)} {role} isn't in the pick-rate panel (needs a pick share "
                        f"in two consecutive patches above the games filter).")
            else:
                cur = mine.sort_values("t").iloc[-1]
                # Direction comes from the slider's SIGN only — magnitude is deliberately
                # ignored, because it tests null and hurts out-of-sample accuracy.
                direction = "buff" if mag > 0 else "nerf" if mag < 0 else "none"
                pk = PR.predict_change(_pick_fit(min_games), float(cur["pick"]), role,
                                       direction, float(cur["y"]))
                q1, q2, q3 = st.columns(3)
                q1.metric("Pick share now", f"{cur['pick'] * 100:.2f}%",
                          f"patch {cur['patch']}", delta_color="off")
                q2.metric("Pick share next patch", f"{pk['new_pick'] * 100:.2f}%",
                          f"{pk['pct_change']:+.1f}%")
                q3.metric("Change treated as", direction,
                          "magnitude ignored — by design", delta_color="off")
                st.caption(
                    f"95% CI: {pk['pct_low']:+.1f}% to {pk['pct_high']:+.1f}%. Move the damage "
                    f"slider and watch this number ignore everything except which side of zero "
                    f"you're on — that's the finding. Buffed champions gain ~24% pick share and "
                    f"nerfed ones lose ~11%, but **how big** the change is predicts nothing.")

        # sweep the same prediction across the whole slider range
        sweep = pd.DataFrame([{**{f"mag_{c}": 0.0 for c in MBT.CATEGORIES}, "mag_damage": m,
                               "prior_winrate": prior, "role": role}
                              for m in np.arange(-30, 30.5, 1.0)])
        sp = fit6.get_prediction(sweep)
        ci = sp.conf_int()
        curve = pd.DataFrame({
            "mag_damage": sweep["mag_damage"],
            "predicted_wr": (prior + sp.predicted_mean) * 100,
            "lo": (prior + ci[:, 0]) * 100,
            "hi": (prior + ci[:, 1]) * 100,
        })
        band = alt.Chart(curve).mark_area(opacity=0.18, color="#2a78d6").encode(
            x=alt.X("mag_damage:Q", title="damage change size (signed %)"),
            y=alt.Y("lo:Q", title="predicted win-rate next patch (%)", scale=alt.Scale(zero=False)),
            y2="hi:Q")
        line = alt.Chart(curve).mark_line(color="#2a78d6", size=2.5).encode(
            x="mag_damage:Q", y="predicted_wr:Q",
            tooltip=[alt.Tooltip("mag_damage:Q", title="damage change %"),
                     alt.Tooltip("predicted_wr:Q", title="predicted win-rate", format=".2f")])
        now = alt.Chart(pd.DataFrame({"v": [prior * 100]})).mark_rule(
            strokeDash=[4, 4], color="#999").encode(y="v:Q")
        pick = alt.Chart(pd.DataFrame({"v": [mag]})).mark_rule(color="#eb6834", size=2).encode(x="v:Q")
        st.altair_chart(band + line + now + pick, width="stretch")
        st.caption("Blue = predicted win-rate across every damage change size, with its 95% band; "
                   "grey dashed = where the champion sits today; orange = your slider. The band is "
                   "wide because the damage coefficient is fit on relatively few changed champions "
                   "— read the direction, not the third decimal.")

# ------------------------------- ⑦ pick rate — the outcome patch notes actually predict
st.subheader("⑦ Pick rate — the outcome patch notes actually predict")
st.caption(
    "Win-rate is a dead end and we can now say why with a number: **97.8% of a champion's "
    "patch-to-patch win-rate movement is binomial sampling noise**, and a model that knew every "
    "champion's true strength would still miss by 4.43pp at min_games=20 — which is where the "
    "model above already sits. Pick share is the opposite: measured against every game played "
    "rather than one champion's handful, it is ~98% signal (lag-1 autocorrelation 0.93 vs "
    "win-rate's 0.03) — and patch notes move it."
)

ppanel = _pick_panel(min_games)
if ppanel.empty or ppanel["changed"].sum() < 20:
    st.info("Not enough changed champions in the pick-rate panel at this games filter.")
else:
    eff = PR.effects(ppanel)
    e1, e2, e3 = st.columns(3)
    for col, (_, r) in zip((e1, e2), eff.iterrows()):
        col.metric(f"{r['direction']} champions", f"{r['pick_share_change_pct']:+.1f}% pick share",
                   f"95% CI [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}] · n={int(r['n_obs'])}",
                   delta_color="off")
    bt = PR.backtest(ppanel)
    if not bt.empty:
        wins = int((bt["gain_pp"] > 0).sum())
        e3.metric("Walk-forward folds improved", f"{wins}/{len(bt)}",
                  f"MAE {bt['mae_reversion_pct'].mean():.1f}% → "
                  f"{bt['mae_model_pct'].mean():.1f}%")

        st.write("**Held-out accuracy per patch** — train on every earlier patch, predict this "
                 "one. Bars below zero are folds where the patch notes made things worse.")
        ch = alt.Chart(bt).mark_bar().encode(
            x=alt.X("patch:N", title="predicted patch", sort=list(bt["patch"])),
            y=alt.Y("gain_pp:Q", title="MAE improvement vs reversion (pp of log pick-share)"),
            color=alt.condition(alt.datum.gain_pp > 0, alt.value("#2a78d6"), alt.value("#e34948")),
            tooltip=[alt.Tooltip("patch:N"), alt.Tooltip("n_changed:Q", title="changed champs"),
                     alt.Tooltip("mae_reversion_pct:Q", title="reversion MAE %"),
                     alt.Tooltip("mae_model_pct:Q", title="model MAE %"),
                     alt.Tooltip("gain_pp:Q", title="gain (pp)")],
        )
        rule = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color="#999").encode(y="v:Q")
        st.altair_chart(ch + rule, width="stretch")

    st.info(
        "**The mechanism is attention, not power.** Given a champion was announced as changed, "
        "the SIZE of the change adds nothing (`mag_damage` p=0.80) and putting magnitudes back in "
        "makes out-of-sample accuracy *worse*. Players read the notes and pick what got buffed, "
        "roughly regardless of how much — and buffs move picks about twice as hard as nerfs. "
        "So the deliverable predicts what people **play**, not how well they perform."
    )
    with st.expander("Per-fold backtest numbers"):
        st.dataframe(bt, width="stretch", hide_index=True)

st.divider()
st.caption("Win-rate prediction is closed — it sits at the sampling-noise floor. The live "
           "Checkpoint-3 result is the pick-rate model in ⑦: patch notes predict what people play.")
