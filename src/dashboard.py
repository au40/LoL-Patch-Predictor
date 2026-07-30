"""
Streamlit dashboard - an interactive walkthrough of the project's result.

Organised as an ARGUMENT, not as a build log:

  1. The finding      - patch notes predict PICK rate. This is the live result.
  2. This patch       - the model's named calls for one patch, scored against what happened.
  3. Try it           - pick a champion, announce a buff/nerf, see where its pick share goes.
  4. Why not winrate - the negative result, with the proof, plus the leads that died.
  5. Data & method    - ingestion, extraction, and how complete the patch coverage is.

Everything recomputes from the real project files on the fly, so it always reflects
current data. The games filter lives in the sidebar and drives every tab.

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
import noise_floor as NF  # noqa: E402
import annotations as ANN  # noqa: E402
from magnitude import _signed_magnitude  # noqa: E402
from datadragon import (champion_spell_cooldowns, change_cooldown,  # noqa: E402
                        champion_icons, champion_names, champion_id_map, normalize_champion)

_CD_MAP = champion_spell_cooldowns()

DATA_PROCESSED = ROOT / "data" / "processed"

BUFF_BLUE, NERF_RED, ACCENT = "#2a78d6", "#e34948", "#eb6834"

st.set_page_config(page_title="LoL Patch Predictor", layout="wide")
st.title("LoL Patch Predictor")
st.caption("""Do League patch notes predict anything? Winrate: no, and we can prove it. **Pick rate: yes.**""")

# ------------------------------------------------------------------ load data
try:
    wr = load_winrates()
except Exception as exc:  # noqa: BLE001
    st.error(f"No winrate data on disk yet ({exc}). Run src/riot_ingest.py first.")
    st.stop()

extracted = sorted(DATA_PROCESSED.glob("extracted_*.json"))
total_matches = int(wr["games"].sum() / 10)

# ------------------------------------------------------- global controls (sidebar)
# One filter drives every tab, so a reader can stress-test the whole argument at once
# instead of wondering whether two sections used different thresholds.
with st.sidebar:
    st.header("Controls")
    min_games = st.slider("Minimum games per champion-role", 5, 40, 20, 5,
                          help="Data-hygiene filter. Applies to every tab.")
    st.divider()
    st.caption("**Data on disk**")
    st.metric("Real matches ingested", f"{total_matches:,}")
    st.metric("Patches with winrates", wr["patch"].nunique())
    st.metric("Patches extracted from notes", len(extracted))


# ------------------------------------------------------------------- shared helpers
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
    """Icon URL for a winrate row's champion, or None if we have no art for it.

    The winrate CSVs come from third-party aggregators whose casing drifts from Data
    Dragon's ('FiddleSticks' vs 'Fiddlesticks'), so go through normalize_champion rather
    than indexing directly — a raw lookup KeyErrors and takes the whole grid down."""
    return ICONS.get(normalize_champion(champ, _ID_MAP))


def _label(champ: str) -> str:
    """Display name ('MonkeyKing' -> 'Wukong'), falling back to the id we were given."""
    return NAMES.get(normalize_champion(champ, _ID_MAP), champ)


latest_patch = max(wr["patch"].unique(), key=MBT._pk)
current = wr[(wr["patch"] == latest_patch) & (wr["games"] >= min_games)]
ppanel = _pick_panel(min_games)


@st.cache_data(show_spinner=False)
def _noise(min_games: int) -> dict:
    """The winrate noise-floor decomposition, computed live.

    These figures used to be typed into the captions by hand, which silently went stale every
    time more matches were ingested — at one point two captions quoted the same statistic as
    97.8% and 98.7%. Deriving them here means the prose cannot drift from the data, and
    rewording a caption can't break a number."""
    try:
        return NF.analyse(min_games)
    except Exception:  # noqa: BLE001
        return {}


NOISE = _noise(min_games)

# Short aliases for the live figures, so the captions below read as prose instead of as
# dictionary lookups. Rewording a caption then only means editing words around a name like
# {NOISE_PCT:.1f} -- there is no long expression to accidentally break. NaN if the
# decomposition couldn't run, which formats harmlessly as "nan".
_nan = float("nan")
NOISE_PCT = NOISE.get("noise_share", _nan) * 100      # % of WR movement that is pure noise
VAR_OBS = NOISE.get("var_obs_pp2", _nan)              # observed variance of the WR change
VAR_NOISE = NOISE.get("var_binom_pp2", _nan)          # variance from sampling alone
REL_WR = NOISE.get("rel_wr", _nan)                    # reliability of one winrate
REL_PICK = NOISE.get("rel_pick", _nan)                # reliability of one pick share
AC_WR = NOISE.get("ac_wr", _nan)                      # lag-1 autocorrelation, winrate
AC_PICK = NOISE.get("ac_pick", _nan)                  # lag-1 autocorrelation, pick share
FLOOR_PP = NOISE.get("floor_pp", _nan)                # MAE an oracle would still post
BEST_PP = min(NOISE.get("mae", {"—": _nan}).values())  # best baseline we actually built
GAP_PP = BEST_PP - FLOOR_PP                           # how far off the floor that leaves us

tab_find, tab_score, tab_try, tab_wr, tab_data = st.tabs([
    "① The finding", "② This patch", "③ Try it", "④ Why not winrate", "⑤ Data & method"])

# ==================================================================== ① THE FINDING
with tab_find:
    st.subheader("Patch notes predict what people **play**, not how well they perform")
    st.caption(
        f"""Winrate is dead: **{NOISE_PCT:.1f}% of a champion's patch-to-patch winrate movement
is binomial sampling noise** (see tab 4). Pick share is the opposite: measured against every
game played rather than one champion's handful, it is ~{REL_PICK * 100:.0f}% signal."""
    )

    if ppanel.empty or ppanel["changed"].sum() < 20:
        st.info("""Not enough changed champions in the pick-rate panel at this games filter. Lower the sidebar
slider.""")
    else:
        eff = PR.effects(ppanel)
        e1, e2, e3 = st.columns(3)
        for col, (_, r) in zip((e1, e2), eff.iterrows()):
            col.metric(f"{r['direction']} champions",
                       f"{r['pick_share_change_pct']:+.1f}% pick share",
                       f"95% CI [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}] · n={int(r['n_obs'])}",
                       delta_color="off")
        bt = PR.backtest(ppanel)
        if not bt.empty:
            wins = int((bt["gain_pp"] > 0).sum())
            e3.metric("Walk-forward folds improved", f"{wins}/{len(bt)}",
                      f"""MAE {bt['mae_reversion_pct'].mean():.1f}% → {bt['mae_model_pct'].mean():.1f}%""")

            st.write("""**Held-out accuracy per patch** — train on every earlier patch, predict this one. Bars below
zero are where the patch notes made it worse. Going from x.24->x.1 is a special case: these are
year-end massive patches where Riot changes fundamental mechanics of the game.""")
            ch = alt.Chart(bt).mark_bar().encode(
                x=alt.X("patch:N", title="predicted patch", sort=list(bt["patch"])),
                y=alt.Y("gain_pp:Q", title="MAE improvement vs reversion (pp of log pick-share)"),
                color=alt.condition(alt.datum.gain_pp > 0,
                                    alt.value(BUFF_BLUE), alt.value(NERF_RED)),
                tooltip=[alt.Tooltip("patch:N"),
                         alt.Tooltip("n_changed:Q", title="changed champs"),
                         alt.Tooltip("mae_reversion_pct:Q", title="reversion MAE %"),
                         alt.Tooltip("mae_model_pct:Q", title="model MAE %"),
                         alt.Tooltip("gain_pp:Q", title="gain (pp)")],
            )
            rule = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color="#999").encode(y="v:Q")
            st.altair_chart(ch + rule, width="stretch")

        # p-values derived, not typed: the magnitude terms are fitted on top of the effects
        # formula (the same specification results_summary.py section 4 reports), and the
        # count term comes from the shipped model. Hand-written values here went stale every
        # time more matches landed.
        _mag_f = PR.fit(panel_or_none := ppanel,
                        PR.FORMULA_EFFECTS + " + "
                        + " + ".join(f"mag_{c}" for c in MBT.CATEGORIES))
        _cnt_f = PR.fit(panel_or_none, PR.FORMULA)
        _p_dmg = float(_mag_f.pvalues["mag_damage"])
        _p_cnt = float(_cnt_f.pvalues["net_buff"])
        st.info(
            f"""Interestingly, it matters how many buffs/nerfs there are, not how hard they actually got hit. The magnitude of a change adds nothing
(`mag_damage` p={_p_dmg:.2f}) and putting magnitudes back in makes out-of-sample accuracy
worse — but the actual count of buff/nerf lines does carry signal (`net_buff` p={_p_cnt:.3f}, and
it beats the bare flags out-of-sample). A champion with four buff bullets moves more than one
with a single line, regardless of the numbers in them."""
        )

        # --- how long does it last? (tests whether the above is just a fad) --------------
        pers = PR.persistence(ppanel)
        if not pers.empty:
            st.write("**The effects persist! It's not a one-off change. And the way they persist is different to buff vs. nerf.**")
            band = alt.Chart(pers).mark_area(opacity=0.18).encode(
                x=alt.X("patches_after:O", title="patches after the change"),
                y=alt.Y("ci_low:Q", title="pick-share change vs the patch before (%)"),
                y2="ci_high:Q",
                color=alt.Color("direction:N", scale=alt.Scale(domain=["buff", "nerf"],
                                                               range=[BUFF_BLUE, NERF_RED])))
            line = alt.Chart(pers).mark_line(point=True, size=3).encode(
                x="patches_after:O", y=alt.Y("pick_share_change_pct:Q"),
                color=alt.Color("direction:N", title="direction",
                                scale=alt.Scale(domain=["buff", "nerf"],
                                                range=[BUFF_BLUE, NERF_RED])),
                tooltip=[alt.Tooltip("direction:N"), alt.Tooltip("patches_after:O"),
                         alt.Tooltip("pick_share_change_pct:Q", title="pick-share change %"),
                         alt.Tooltip("ci_low:Q"), alt.Tooltip("ci_high:Q"),
                         alt.Tooltip("n_obs:Q", title="champions")])
            zero = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color="#999").encode(y="v:Q")
            st.altair_chart(band + zero + line, width="stretch")
            # Pull the endpoints out of the fitted persistence table rather than typing them.
            def _pers(direction: str, k: int) -> float:
                row = pers[(pers["direction"] == direction) & (pers["patches_after"] == k)]
                return float(row["pick_share_change_pct"].iloc[0]) if not row.empty else float("nan")

            _kmax = int(pers["patches_after"].max())
            _b0, _bk = _pers("buff", 0), _pers("buff", _kmax)
            _n0, _nk = _pers("nerf", 0), _pers("nerf", _kmax)
            _flipped = abs(_nk) > abs(_bk)

        with st.expander("Per-fold backtest numbers"):
            st.dataframe(bt, width="stretch", hide_index=True)

# ================================================================== ② THIS PATCH
with tab_score:
    st.subheader("The model made these predictions: here's how it went")
    st.caption(
        """A visual representation of a single patch's pick rate changes against the pick rate predicted off of the patch notes."""
    )

    if ppanel.empty:
        st.info("The pick-rate panel is empty at this games filter. Lower the sidebar slider.")
    else:
        scorable = sorted(ppanel["patch"].unique(), key=PR._pk)[-10:]
        sel = st.selectbox("Patch to score", list(reversed(scorable)), index=0,
                           key="score_patch",
                           help="Trained only on patches before this one — no leakage.")
        sb = PR.scoreboard(ppanel, sel)

        if sb.empty:
            st.info(f"""Not enough training history before patch {sel} to score it. Pick a later patch.""")
        else:
            hit = float((np.sign(sb["predicted_pct"]) == np.sign(sb["actual_pct"])).mean())
            mae_m = float(sb["error_pp"].abs().mean())
            mae_r = float((sb["reversion_pct"] - sb["actual_pct"]).abs().mean())
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Champions scored", f"{len(sb)}", f"patch {sel}", delta_color="off")
            n2.metric("Got the direction right", f"{hit:.0%}",
                      "up vs down", delta_color="off")
            n3.metric("Model error", f"{mae_m:.1f} pp")
            n4.metric("vs reversion baseline", f"{mae_r:.1f} pp",
                      "model is closer" if mae_m < mae_r else "baseline is closer",
                      delta_color="off")

            sb = sb.copy()
            sb["label"] = sb["champion"].map(_label) + "  (" + sb["role"] + ")"
            order = list(sb.sort_values("predicted_pct", ascending=False)["label"])
            long = sb.melt(id_vars=["label", "direction", "lines", "games"],
                           value_vars=["predicted_pct", "actual_pct"],
                           var_name="kind", value_name="pct")
            long["kind"] = long["kind"].map({"predicted_pct": "predicted",
                                             "actual_pct": "actual"})
            rules = alt.Chart(sb).mark_rule(color="#b4b2a9", size=2).encode(
                y=alt.Y("label:N", sort=order, title=None),
                x=alt.X("predicted_pct:Q", title="change in pick share (%)"),
                x2="actual_pct:Q")
            dots = alt.Chart(long).mark_point(filled=True, size=120, opacity=0.95).encode(
                y=alt.Y("label:N", sort=order, title=None),
                x=alt.X("pct:Q", title="change in pick share (%)"),
                color=alt.Color("kind:N", title="",
                                scale=alt.Scale(domain=["predicted", "actual"],
                                                range=[BUFF_BLUE, ACCENT])),
                shape=alt.Shape("kind:N", title="",
                                scale=alt.Scale(domain=["predicted", "actual"],
                                                range=["circle", "diamond"])),
                tooltip=[alt.Tooltip("label:N", title="champion"),
                         alt.Tooltip("direction:N", title="notes said"),
                         alt.Tooltip("lines:Q", title="change lines"),
                         alt.Tooltip("kind:N", title=""),
                         alt.Tooltip("pct:Q", title="pick-share change %", format=".1f"),
                         alt.Tooltip("games:Q")])
            zero = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(
                strokeDash=[4, 4], color="#999").encode(x="v:Q")
            st.altair_chart((zero + rules + dots).properties(
                height=max(280, 26 * len(sb))), width="stretch")
            # --- hand-written domain notes on the misses --------------------------------
            # The one input to this project that isn't derived from the data: what the model
            # could not see. See src/annotations.py.
            sb["note"] = [ANN.note(sel, c, r)
                          for c, r in zip(sb["champion"], sb["role"])]
            annotated = sb[sb["note"] != ""]
            if not annotated.empty:
                written, slots = ANN.coverage()
                st.markdown("#### Why the model was wrong: domain notes")
                st.caption(
                    f"""The model has a limitation: it only reads off of patch notes, it can't take context into account. Examples of context: buffing an unplayable build or the 
                    release of an extremely popular skin. I've handwritten a couple of context notes here for some of the patches as examples of the limitations of the model.""")
                for r in annotated.sort_values("error_pp", key=abs, ascending=False).itertuples():
                    st.markdown(
                        f"""**{_label(r.champion)} ({r.role})** — notes said *{r.direction}*, model predicted
**{r.predicted_pct:+.0f}%**, actual **{r.actual_pct:+.0f}%**  \n{r.note}""")

            with st.expander("Full scoreboard table"):
                st.dataframe(
                    sb[["champion", "role", "direction", "lines", "net_lines",
                        "pick_prev_pct", "pick_now_pct", "predicted_pct", "actual_pct",
                        "reversion_pct", "error_pp", "games", "note"]].round(2),
                    width="stretch", hide_index=True)
                st.caption(
                    """`lines` = how many change bullets the champion got; `net_lines` = buffs minus nerfs, which is
the feature the model actually sees (a champion with one buff and one nerf nets to 0 and is
labelled *mixed*). `reversion_pct` is the no-patch-notes baseline. `error_pp` is predicted −
actual, so positive means the model over-promised."""
                )

# ======================================================================= ③ TRY IT
with tab_try:
    st.subheader("Announce a change: see where the pick share goes")
    st.caption(
        """Pick a champion, then scroll down and see what pick rate is predicted."""
    )

    if current.empty or ppanel.empty:
        st.info("No champions clear the games filter at the newest patch. Lower the slider.")
    else:
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

            # Streamlit buttons can't hold an image, but any keyed widget gets an
            # `st-key-{key}` class on its container — so these are ordinary st.buttons
            # wearing the champion icon as a CSS background. That keeps normal click /
            # session-state behaviour (a query-param link would full-reload the page and
            # wipe the sidebar slider and search box). The label stays the champion name but
            # is rendered at font-size 0: invisible on screen, still there for screen
            # readers. Champion ids are alphanumeric, so they're safe as CSS class names
            # (checked against both Data Dragon and the winrate CSVs).
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
                border-color: {ACCENT}; box-shadow: 0 0 0 2px rgba(235,104,52,.35);
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
            st.info(f"""{_label(champ)} has no role above {min_games} games at patch {latest_patch}. Pick another
champion or lower the slider.""")
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
                c_role, c_dir, c_n = st.columns([1, 1, 2])
                role = c_role.selectbox("Role", roles, key="try_role")
                direction = c_dir.radio("The notes say", ["buff", "nerf"], key="try_dir",
                                        horizontal=True)
                # Capped at 4 on purpose: 99% of announced changes in the panel have 4 or
                # fewer buff/nerf lines, 5 never occurs at all, and 6-7 have three
                # observations between them. A wider slider would extrapolate past the data
                # and print a confident number with nothing behind it.
                n_changes = c_n.slider("How many buff/nerf lines", 1, 4, 1, key="try_n",
                                       help="""The count carries signal; the size does not. Capped at 4 — that covers 99% of real changes.""")

                mine = ppanel[(ppanel["champion"] == champ) & (ppanel["role"] == role)]
                if mine.empty:
                    st.info(f"""{_label(champ)} {role} isn't in the pick-rate panel — it needs a pick share in two consecutive
patches above the games filter.""")
                else:
                    cur = mine.sort_values("t").iloc[-1]
                    pk = PR.predict_change(_pick_fit(min_games), float(cur["pick"]), role,
                                           direction, float(cur["y"]), n_changes)
                    q1, q2, q3 = st.columns(3)
                    q1.metric("Pick share now", f"{cur['pick'] * 100:.2f}%",
                              f"patch {cur['patch']}", delta_color="off")
                    q2.metric("Predicted next patch", f"{pk['new_pick'] * 100:.2f}%",
                              f"{pk['pct_change']:+.1f}%")
                    q3.metric("Lines announced", f"{n_changes}",
                              "size ignored", delta_color="off")
                    st.caption(
                        f"""95% CI: {pk['pct_low']:+.1f}% to {pk['pct_high']:+.1f}%. Move the line count and the prediction
changes; there is deliberately no place to type how big the change is, because that tests null
and makes out-of-sample accuracy worse.""")

            # --- where this champion goes over the next few patches -------------------
            mine = ppanel[(ppanel["champion"] == champ) & (ppanel["role"] == role)]
            if not mine.empty:
                cur = mine.sort_values("t").iloc[-1]
                p0 = float(cur["pick"])
                # Same bucketing forecast_path() uses internally -- named here so the caption
                # can tell the reader which measured decay shape it's looking at.
                _bucket = "1 line" if n_changes <= 1 else "2+ lines"
                fp = PR.forecast_path(ppanel, _pick_fit(min_games), p0, role,
                                      direction, float(cur["y"]), n_changes)
                if not fp.empty:
                    traj = fp.assign(
                        mid=lambda d: p0 * (1 + d["pct"] / 100) * 100,
                        band_lo=lambda d: p0 * (1 + d["lo"] / 100) * 100,
                        band_hi=lambda d: p0 * (1 + d["hi"] / 100) * 100)
                    colour = BUFF_BLUE if direction == "buff" else NERF_RED
                    st.write(f"""**Forecast — the next {int(traj['patches_after'].max())} patches.** Where {_label(champ)} ({role})
lands if it gets a {n_changes}-line {direction} now and nothing else changes:""")
                    band = alt.Chart(traj).mark_area(opacity=0.16, color=colour).encode(
                        x=alt.X("patches_after:O", title="patches after the change"),
                        y=alt.Y("band_lo:Q", title="pick share (%)",
                                scale=alt.Scale(zero=False)),
                        y2="band_hi:Q")
                    line = alt.Chart(traj).mark_line(
                        color=colour, size=3,
                        point=dict(filled=True, size=110, color=colour)).encode(
                        x="patches_after:O", y="mid:Q",
                        tooltip=[alt.Tooltip("patches_after:O", title="patches after"),
                                 alt.Tooltip("mid:Q", title="pick share %", format=".3f"),
                                 alt.Tooltip("pct:Q", title="change vs today %",
                                             format="+.1f"),
                                 alt.Tooltip("n_obs:Q", title="champions behind the shape")])
                    now = alt.Chart(pd.DataFrame({"v": [p0 * 100]})).mark_rule(
                        strokeDash=[4, 4], color="#999").encode(y="v:Q")
                    st.altair_chart(band + now + line, width="stretch")
                    _k = int(traj["patches_after"].max())
                    _end = float(traj.loc[traj["patches_after"] == _k, "pct"].iloc[0])
                    st.caption(
                        f"""Grey dashed = today's pick share. Shaded region is the spread across champions that
got the same kind of change, not a guarantee for this one."""
                    )

# ============================================================== ④ WHY NOT WIN-RATE
with tab_wr:
    st.subheader("Why not winrate?")
    st.info(
        f"""When a champion's winrate changes from one patch to the next, {NOISE_PCT:.1f}% of that
change is luck. Comparing champions to each other, only **{REL_WR * 100:.1f}%** of the
difference between their winrates is real; for pick rate it's **{REL_PICK * 100:.0f}%**.

Interestingly, a model that somehow knew every champion's *true* strength would still be wrong by
{FLOOR_PP:.2f} points. My best baseline gets
{BEST_PP:.2f}. That {GAP_PP:.2f}-point gap is all the room any model could have. This is why I chose to model pickrate."""
    )

    st.markdown("#### The winrate model")
    pool = P.build_pool(P.DEFAULT_BOUNDARIES, min_games)
    if pool.empty or pool["boundary"].nunique() < 2:
        st.warning("""Not enough data at this threshold for a temporal backtest. Lower the slider or ingest more
matches.""")
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

        st.caption(f"""Temporal backtest: train on {present[:-1]} → predict **{test_b}** (never seen). It does beat
predict-zero — but the signal is *regression to the mean* (`prior_winrate` coef ≈ −0.95), not
the patch changes.""")

        with st.expander(f"Diff-in-differences — isolated effect of each changed champ in {test_b}"):
            latest = pool[(pool["boundary"] == test_b) & (pool["changed"] == 1)].copy()
            if latest.empty:
                st.info("No changed champions clear the filter on this boundary.")
            else:
                latest["isolated_effect_pp"] = (latest["delta"] - pool[pool["boundary"] == test_b]
                                                .loc[pool["changed"] == 0, "delta"].median()) * 100
                st.dataframe(
                    latest[["champion", "role", "net_buff", "prior_winrate", "delta",
                            "isolated_effect_pp"]]
                    .assign(prior_winrate=lambda d: (d["prior_winrate"] * 100).round(1),
                            delta=lambda d: (d["delta"] * 100).round(1),
                            isolated_effect_pp=lambda d: d["isolated_effect_pp"].round(1))
                    .sort_values("isolated_effect_pp"),
                    width="stretch", hide_index=True,
                )

    st.divider()
    st.markdown("#### Retired lead: the damage effect")
    st.warning(
        """These are the original graphs from the damage lead in checkpoint 2. Consider the below deprecated."""
    )

    dmg = MBT.damage_scatter_table(min_games)
    if dmg.empty:
        st.info("No damage changes clear the games filter at this threshold. Lower the slider.")
    else:
        st.caption(
            """Each point is a champion-role in a patch where its **damage** changed: x = size of the damage
change (buffs → right), y = its winrate change the next patch. ● = champion-combat damage, ▲ =
monster (jungle-clear) damage."""
        )
        pts = alt.Chart(dmg).mark_point(filled=True, opacity=0.78).encode(
            x=alt.X("mag_damage:Q", title="damage change size (signed %, buffs → right)"),
            y=alt.Y("winrate_change_pp:Q", title="winrate change next patch (pp)"),
            color=alt.Color("direction:N", title="direction",
                            scale=alt.Scale(domain=["buff", "nerf"],
                                            range=[BUFF_BLUE, NERF_RED])),
            shape=alt.Shape("damage_type:N", title="damage type",
                            scale=alt.Scale(domain=["combat", "monster / jungle-clear"],
                                            range=["circle", "triangle-up"])),
            size=alt.Size("games:Q", title="games", scale=alt.Scale(range=[40, 320])),
            tooltip=[alt.Tooltip("champion:N"), alt.Tooltip("role:N"), alt.Tooltip("patch:N"),
                     alt.Tooltip("mag_damage:Q", title="damage change %"),
                     alt.Tooltip("winrate_change_pp:Q", title="winrate Δ (pp)"),
                     alt.Tooltip("games:Q"), alt.Tooltip("abilities:N", title="ability(ies)")],
        )
        trend = pts.transform_regression("mag_damage", "winrate_change_pp").mark_line(
            color="#4a3aa7", size=2.5)
        xrule = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(
            strokeDash=[4, 4], color="#999").encode(x="v:Q")
        yrule = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(
            strokeDash=[4, 4], color="#999").encode(y="v:Q")
        st.altair_chart((xrule + yrule + pts + trend).interactive(), width="stretch")

        st.write("**Was it combat power or jungle-clear speed?** Refit with damage split in two:")
        try:
            split = MBT.damage_split_fit(min_games)
            d1, d2 = st.columns(2)
            for col, (_, row) in zip((d1, d2), split.iterrows()):
                sig = "significant" if row.p_value < 0.05 else "not significant"
                col.metric(f"{row.bucket} damage", f"{row.coef_pp:+.3f} pp",
                           f"p={row.p_value:.3f} · {int(row.n_champs)} champs · {sig}",
                           delta_color="off")
            st.caption("""Monster/jungle-clear damage carried the stronger coefficient on the discovery window, which
suggested a faster-clear effect rather than a combat one — but it rests on few champions and
slipped below significance once the 15.x data arrived. Read as a hypothesis that ran out of
evidence, not a result.""")
        except Exception as exc:  # noqa: BLE001
            st.info(f"Split-fit needs more data at this threshold ({exc}). Lower the slider.")

        with st.expander("See the underlying damage changes (one row per point)"):
            st.dataframe(dmg.sort_values("mag_damage"), width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### Retired lead: does ability cooldown organise the damage response?")
    st.caption(
        """The hypothesis: a damage buff on a low-cooldown (spammed) spell should move winrate more than
the same buff on a long-cooldown ult. Colour = the fastest-cast damaged ability's cooldown.
**The two trend lines come out nearly parallel** — cast frequency does not organise the
response, which is why weighting damage by cooldown never helped the model. Grey points are
passives / zero-cooldown abilities."""
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
                real = [cd for cd in (change_cooldown(c, _CD_MAP) for c in chs)
                        if cd not in (None, 0)]
                cd = min(real) if real else None  # fastest-cast damaged ability
                tier = "no cooldown" if cd is None else ("low (<=10s)" if cd <= 10 else "high (>10s)")
                abilities = "; ".join(f"{c.get('target', '?')} ({c.get('change_type', '')})"
                                      for c in chs)
                rows.append({"champion": r.champion, "role": r.role, "patch": new,
                             "mag_damage": round(mag, 1),
                             "winrate_change_pp": round(r.delta * 100, 1),
                             "cooldown": cd, "tier": tier, "games": int(r.games_new),
                             "abilities": abilities})
        return pd.DataFrame(rows)


    dct = _damage_cooldown_table(min_games)
    if dct.empty:
        st.info("No damage changes clear the games filter at this threshold. Lower the slider.")
    else:
        base = alt.Chart(dct)
        cd_scale = alt.Scale(domain=["low (<=10s)", "high (>10s)", "no cooldown"],
                             range=[ACCENT, BUFF_BLUE, "#b4b2a9"])
        pts = base.mark_point(filled=True, opacity=0.7).encode(
            x=alt.X("mag_damage:Q", title="damage change size (signed %, buffs -> right)"),
            y=alt.Y("winrate_change_pp:Q", title="winrate change next patch (pp)"),
            color=alt.Color("tier:N", legend=alt.Legend(title="ability cooldown", orient="top"),
                            scale=cd_scale),
            size=alt.Size("games:Q", title="games", legend=None,
                          scale=alt.Scale(range=[40, 320])),
            tooltip=[alt.Tooltip("champion:N"), alt.Tooltip("role:N"), alt.Tooltip("patch:N"),
                     alt.Tooltip("cooldown:Q", title="cooldown (s)"),
                     alt.Tooltip("mag_damage:Q", title="damage change %"),
                     alt.Tooltip("winrate_change_pp:Q", title="winrate change (pp)"),
                     alt.Tooltip("games:Q"),
                     alt.Tooltip("abilities:N", title="ability(ies)")],
        )
        trends = base.transform_filter(alt.datum.tier != "no cooldown").transform_regression(
            "mag_damage", "winrate_change_pp", groupby=["tier"]).mark_line(size=2.5).encode(
            x="mag_damage:Q", y="winrate_change_pp:Q",
            color=alt.Color("tier:N", legend=None, scale=cd_scale))
        xr = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(
            strokeDash=[4, 4], color="#999").encode(x="v:Q")
        yr = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(
            strokeDash=[4, 4], color="#999").encode(y="v:Q")
        st.altair_chart((xr + yr + pts + trends).interactive(), width="stretch")

    st.divider()
    st.markdown("#### Retired interactive: the winrate predictor")
    st.caption(
        f"""The original Checkpoint-2 demo, kept for comparison. It uses the champion selected in the **Try
it** tab and asks the per-stat-type winrate fit what it expects from a damage change alone.
Compare its confidence band to the pick-rate prediction next door — this is what a model sitting
on the noise floor looks like."""
    )

    pool6 = MBT.build_pool(min_games)
    champ = st.session_state.get("picked_champ")
    if pool6.empty or current.empty or not champ:
        st.info("Not enough data at this threshold, or no champion picked yet in tab ③.")
    elif champ not in set(current["champion"]):
        st.info(f"{_label(champ)} has no role above {min_games} games at patch {latest_patch}.")
    else:
        formula6 = ("delta ~ " + " + ".join(f"mag_{c}" for c in MBT.CATEGORIES)
                    + " + prior_winrate + C(role)")
        fit6 = smf.wls(formula6, data=pool6, weights=np.sqrt(pool6["games_new"])).fit()
        coef_dmg = float(fit6.params["mag_damage"])

        roles = sorted(current.loc[current["champion"] == champ, "role"].unique())
        r1, r2 = st.columns([1, 2])
        role = r1.selectbox("Role", roles, key="retired_role")
        mag = r2.slider("Damage change (signed %, buff → right)", -30.0, 30.0, 5.0, 0.5,
                        key="retired_mag")

        here = current[(current["champion"] == champ) & (current["role"] == role)]
        prior = float(here["winrate"].iloc[0])
        games = int(here["games"].iloc[0])

        row = pd.DataFrame([{**{f"mag_{c}": 0.0 for c in MBT.CATEGORIES},
                             "mag_damage": mag, "prior_winrate": prior, "role": role}])
        pr = fit6.get_prediction(row)
        delta = float(pr.predicted_mean[0])
        lo, hi = (float(x) for x in pr.conf_int()[0])

        k1, k2, k3 = st.columns(3)
        k1.metric(f"Winrate now ({latest_patch})", f"{prior * 100:.1f}%",
                  f"{games:,} games", delta_color="off")
        k2.metric("Predicted next patch", f"{(prior + delta) * 100:.1f}%",
                  f"{delta * 100:+.2f} pp")
        k3.metric("Of which the damage change", f"{coef_dmg * mag * 100:+.2f} pp",
                  f"{coef_dmg * 100:+.3f} pp per 1%", delta_color="off")
        st.caption(f"""95% CI on the predicted shift: {lo * 100:+.2f} to {hi * 100:+.2f} pp. The gap between the middle
and right numbers is **reversion to the mean** — at {prior * 100:.1f}% the model expects this
champion to drift toward average whether or not Riot touches it. That reversion is doing nearly
all the work.""")

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
        band = alt.Chart(curve).mark_area(opacity=0.18, color=BUFF_BLUE).encode(
            x=alt.X("mag_damage:Q", title="damage change size (signed %)"),
            y=alt.Y("lo:Q", title="predicted winrate next patch (%)",
                    scale=alt.Scale(zero=False)),
            y2="hi:Q")
        line = alt.Chart(curve).mark_line(color=BUFF_BLUE, size=2.5).encode(
            x="mag_damage:Q", y="predicted_wr:Q",
            tooltip=[alt.Tooltip("mag_damage:Q", title="damage change %"),
                     alt.Tooltip("predicted_wr:Q", title="predicted winrate", format=".2f")])
        now = alt.Chart(pd.DataFrame({"v": [prior * 100]})).mark_rule(
            strokeDash=[4, 4], color="#999").encode(y="v:Q")
        pick = alt.Chart(pd.DataFrame({"v": [mag]})).mark_rule(
            color=ACCENT, size=2).encode(x="v:Q")
        st.altair_chart(band + line + now + pick, width="stretch")
        st.caption("""Blue = predicted winrate across every damage change size, with its 95% band; grey dashed =
where the champion sits today; orange = your slider. The band is wide because the damage
coefficient is fit on relatively few changed champions — read the direction, not the third
decimal.""")

# ================================================================ ⑤ DATA & METHOD
with tab_data:
    st.subheader("What the result rests on")

    tot = wr.groupby("patch")["games"].sum()
    allp = sorted(tot.index, key=PR._pk)
    cov = pd.DataFrame({"patch": allp, "games": [int(tot[p]) for p in allp]})
    cov["status"] = np.where(cov["games"] >= PR.MIN_TOTAL_GAMES, "usable", "too thin")

    # Which consecutive steps actually made it into the panel? A patch below the volume
    # threshold is dropped BEFORE the lag merge, so the steps into and out of it are never
    # built — the loss is real but invisible unless you check for it here.
    present_steps = set(zip(ppanel["patch_l1"], ppanel["patch"])) if not ppanel.empty else set()
    lost = []
    for a, b in zip(allp, allp[1:]):
        if (a, b) in present_steps:
            continue
        if tot[a] < PR.MIN_TOTAL_GAMES and tot[b] < PR.MIN_TOTAL_GAMES:
            why = f"both below {PR.MIN_TOTAL_GAMES:,} games"
        elif tot[a] < PR.MIN_TOTAL_GAMES:
            why = f"{a} too thin ({int(tot[a]):,})"
        elif tot[b] < PR.MIN_TOTAL_GAMES:
            why = f"{b} too thin ({int(tot[b]):,})"
        elif PR._step_width(a, b) != 1:
            why = f"not consecutive (spans {PR._step_width(a, b)} patches)"
        else:
            why = "no champion clears the games filter on both sides"
        lost.append({"step": f"{a} → {b}", "why": why})

    d1, d2, d3 = st.columns(3)
    d1.metric("Real matches ingested", f"{total_matches:,}")
    d2.metric("Patch steps in the model", f"{len(present_steps)}",
              f"{len(lost)} unusable", delta_color="off")
    d3.metric("Champion-patches used", f"{len(ppanel):,}",
              f"{int(ppanel['changed'].sum())} with patch notes" if not ppanel.empty else "",
              delta_color="off")

    st.write("""**Sample depth per patch.** The dashed line is the 5,000-game floor below which a patch's
pick-share denominator is too small to trust — patches under it are dropped, and they take the
steps on *both* sides of them with them.""")
    bars = alt.Chart(cov).mark_bar().encode(
        x=alt.X("patch:N", title="patch", sort=allp),
        y=alt.Y("games:Q", title="champion-games sampled"),
        color=alt.Color("status:N", title="",
                        scale=alt.Scale(domain=["usable", "too thin"],
                                        range=[BUFF_BLUE, NERF_RED])),
        tooltip=[alt.Tooltip("patch:N"), alt.Tooltip("games:Q", format=","),
                 alt.Tooltip("status:N")])
    floor = alt.Chart(pd.DataFrame({"v": [PR.MIN_TOTAL_GAMES]})).mark_rule(
        strokeDash=[5, 5], color="#999").encode(y="v:Q")
    st.altair_chart(bars + floor, width="stretch")

    if lost:
        with st.expander(f"The {len(lost)} patch steps the model can't use, and why"):
            st.dataframe(pd.DataFrame(lost), width="stretch", hide_index=True)
            st.caption("""Each of these is recoverable by ingesting more matches on the thin patch — roughly 10
champion-games per match.""")

    st.divider()
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("**Games by role**")
        st.dataframe(wr.groupby("role")["games"].sum().sort_values(ascending=False),
                     width="stretch")
    with c2:
        st.write("**Adjacency guard**")
        st.caption("""Steps are kept only when they span exactly one real patch. Without it, a missing patch gets
modelled as a normal step: the outcome covers two patches of player behaviour while the notes
cover one, so champions changed in the skipped patch are mislabelled as unchanged *and*
champions changed in the visible one get credited with two patches of movement.""")

    st.divider()
    st.write("**LLM patch-note extraction**")
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
        """Base-stat changes are auto-validated against **Data Dragon ground truth** (~89% recall / ~80%
precision across 3 patches). The cross-check even caught an **undocumented micropatch** —
Smolder's base AD 60→58, documented in patch 26.10 but missing from the 26.11 notes — which a
patch-notes-only analysis would miss."""
    )
