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
from winrates import load_winrates  # noqa: E402
import predict as P  # noqa: E402

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

st.divider()
st.caption("Preliminary model - the signed change feature strengthens as patches/volume grow. "
           "Full magnitude-aware model + user-facing prediction UI are Checkpoint 3.")
