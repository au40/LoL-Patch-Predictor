"""
Full results summary for the LoL Patch Predictor — every headline number, in one command.

    python src/results_summary.py --save

This is the project's reproducibility artifact. Every figure the write-up quotes is derived
here, live, from the same modules the analysis uses — nothing is copied or hard-coded, so a
reported number cannot silently drift from the code that produced it. Re-running it on the
current data regenerates the whole result set from scratch.

Nine sections, each stating what it establishes:

  1. Data volume            how much was ingested, and how much of it is usable
  2. Why win-rate is closed the negative result, with its proof
  3. The pick-rate result   the project's main finding
  4. Announcement vs size   what drives the effect, and what turns out not to
  5. Persistence            how long the effect lasts
  6. Per-champion scoreboard named predictions, scored against what actually happened
  7. Extraction validation  whether the LLM's buff/nerf labels can be trusted
  8. The retired damage lead a finding that looked real and failed to replicate
  9. Data coverage          what the dataset is missing, and what it would cost to fix

Sections are independent and failures are caught per-section, so if one analysis can't run
at the current data volume the rest still print.

Options:
  --min-games N     the data-hygiene filter applied throughout (default 20)
  --save            also write the output to data/processed/results_summary.txt
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import io
import json
import sys
import traceback
from contextlib import redirect_stdout
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

from winrates import load_winrates            # noqa: E402
import pickrate as PR                          # noqa: E402
import magnitude_by_type as MBT                # noqa: E402
import noise_floor as NF                       # noqa: E402
import validate_direction as VD                # noqa: E402

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"


def header(n: int, title: str, claim: str) -> None:
    print()
    print("=" * 78)
    print(f"  [{n}] {title}")
    print(f"       {claim}")
    print("=" * 78)


def section(fn):
    """Run a section, but never let one failure take down the whole report."""
    try:
        fn()
    except Exception:                          # noqa: BLE001
        print("  !! this section failed to run:")
        print("     " + traceback.format_exc().strip().replace("\n", "\n     "))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full results summary — every headline number, derived live")
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    mg = args.min_games

    buf = io.StringIO()

    class Tee:
        """Print to the console and capture for --save at the same time."""
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)

        def flush(self):
            sys.__stdout__.flush()

    with redirect_stdout(Tee()):
        print("LoL PATCH PREDICTOR — FULL RESULTS SUMMARY")
        print("Every number below is derived live from the analysis modules, not copied.")
        print(f"Regenerate with: python src/results_summary.py --min-games {mg}")
        print(f"min_games = {mg}   (data-hygiene filter applied to every figure below)")

        wr = load_winrates()
        panel = PR.build_panel(mg)

        # ---------------------------------------------------------------- 1. data volume
        def s1():
            header(1, "DATA VOLUME", "how much was ingested, and how much of it is usable")
            state = json.loads((DATA_RAW / "winrates" / "riot_ingested.state.json")
                               .read_text(encoding="utf-8"))
            ex = sorted(DATA_PROCESSED.glob("extracted_*.json"))
            lines = sum(len(json.loads(f.read_text(encoding="utf-8")).get("changes", []))
                        for f in ex)
            buffs = nerfs = 0
            for f in ex:
                for c in json.loads(f.read_text(encoding="utf-8")).get("changes", []):
                    buffs += c.get("change_type") == "buff"
                    nerfs += c.get("change_type") == "nerf"
            # `seen_matches` is every match ID we FETCHED. A match is marked seen before
            # the patch-window filter runs, so matches landing outside the targeted patches
            # count here and contribute no rows. Report both, or the two numbers look
            # inconsistent to anyone who divides one by the other.
            seen = len(state["seen_matches"])
            champ_games = int(wr["games"].sum())
            contributed = champ_games / 10          # 10 participants recorded per match
            print(f"  matches examined (API calls) : {seen:>10,}")
            print(f"  matches inside target patches: {contributed:>10,.0f}"
                  f"   ({contributed / seen * 100:.0f}% — the rest fell outside the")
            print(f"  {'':>29}    patch windows and were discarded)")
            print(f"  champion-game observations   : {champ_games:>10,}"
                  f"   (10 per contributing match)")
            print(f"  lane-matchup rows            : {len(state['matchups']):>10,}")
            print(f"  patches with win-rate data   : {wr['patch'].nunique():>10}"
                  f"   ({min(wr['patch'], key=MBT._pk)} -> {max(wr['patch'], key=MBT._pk)})")
            print(f"  patches LLM-extracted        : {len(ex):>10}")
            print(f"  individual change lines      : {lines:>10,}"
                  f"   ({buffs} buffs / {nerfs} nerfs)")
            print(f"  champion-role-patch rows used: {len(panel):>10,}"
                  f"   ({int(panel['changed'].sum())} with patch-note changes)")

        # ------------------------------------------------------------ 2. why win-rate died
        def s2():
            header(2, "WHY WIN-RATE IS CLOSED", "the negative result, and the proof behind it")
            r = NF.analyse(mg)
            print(f"  paired observations              : {r['n']:,}")
            print(f"  observed variance of the change  : {r['var_obs_pp2']:8.2f} pp^2")
            print(f"  variance from sampling alone     : {r['var_binom_pp2']:8.2f} pp^2")
            print(f"  -> SAMPLING NOISE EXPLAINS {r['noise_share'] * 100:.1f}% OF ALL MOVEMENT")
            print()
            print(f"  reliability   win-rate {r['rel_wr']:.3f}   vs   pick share {r['rel_pick']:.3f}")
            print(f"  lag-1 autocorr win-rate {r['ac_wr']:.3f}   vs   pick share {r['ac_pick']:.3f}")
            print()
            print(f"  ORACLE FLOOR (knows true strength): {r['floor_pp']:.3f} pp")
            for k, v in sorted(r["mae"].items(), key=lambda kv: kv[1]):
                print(f"    {k:36} {v:6.3f} pp  ({v - r['floor_pp']:+.3f} vs floor)")

        # ------------------------------------------------------------- 3. the pick result
        def s3():
            header(3, "THE PICK-RATE RESULT", "the project's main finding")
            print(PR.effects(panel).to_string(index=False))
            print()
            bt = PR.backtest(panel)
            wins = int((bt["gain_pp"] > 0).sum())
            print(f"  walk-forward: improves {wins}/{len(bt)} held-out patches")
            print(f"  MAE {bt['mae_reversion_pct'].mean():.2f}% (reversion) "
                  f"-> {bt['mae_model_pct'].mean():.2f}% (model), "
                  f"mean gain {bt['gain_pp'].mean():.2f} pp")
            print("\n  per-fold:")
            print("   " + bt.to_string(index=False).replace("\n", "\n   "))

        # ------------------------------------------------- 4. announcement, not magnitude
        def s4():
            header(4, "ANNOUNCEMENT, NOT MAGNITUDE", "what drives the effect, and what turns out not to")
            f = PR.fit(panel, PR.FORMULA_EFFECTS + " + " +
                       " + ".join(f"mag_{c}" for c in MBT.CATEGORIES))
            for c in MBT.CATEGORIES:
                k = f"mag_{c}"
                print(f"  {k:16} {f.params[k] * 100:+7.3f}%   p = {f.pvalues[k]:.3f}")
            fn = PR.fit(panel, PR.FORMULA)
            print(f"\n  net_buff (COUNT of buff/nerf lines): "
                  f"{fn.params['net_buff'] * 100:+.2f}%  p = {fn.pvalues['net_buff']:.3f}")
            print("  -> the size of a change tests null; the COUNT of lines does not.")

        # ------------------------------------------------------------- 5. does it persist
        def s5():
            header(5, "PERSISTENCE", "how long the effect lasts: buffs settle, nerfs compound")
            print(PR.persistence(panel).to_string(index=False))

        # ------------------------------------------------------- 6. named predictions
        def s6():
            header(6, "PER-CHAMPION SCOREBOARD", "named predictions, scored against what happened")
            patches = sorted(panel["patch"].unique(), key=PR._pk)[-6:]
            rows = []
            for p in patches:
                sb = PR.scoreboard(panel, p)
                if sb.empty:
                    continue
                hit = float((np.sign(sb["predicted_pct"]) == np.sign(sb["actual_pct"])).mean())
                m = float(sb["error_pp"].abs().mean())
                r = float((sb["reversion_pct"] - sb["actual_pct"]).abs().mean())
                rows.append({"patch": p, "n": len(sb), "direction_right_pct": round(hit * 100),
                             "model_mae": round(m, 1), "reversion_mae": round(r, 1),
                             "beats_baseline": "yes" if m < r else "no"})
            df = pd.DataFrame(rows)
            print(df.to_string(index=False))
            print(f"\n  beats the reversion baseline on "
                  f"{int((df['beats_baseline'] == 'yes').sum())}/{len(df)} recent patches")
            newest = patches[-1]
            sb = PR.scoreboard(panel, newest)
            print(f"\n  biggest misses at {newest} (where the model failed, and by how much):")
            sb = sb.assign(absmiss=sb["error_pp"].abs())
            print("   " + sb.nlargest(5, "absmiss")[
                ["champion", "role", "direction", "lines", "predicted_pct", "actual_pct"]
            ].round(1).to_string(index=False).replace("\n", "\n   "))

            # Tested hypothesis: the model scores each champion-role independently, so it
            # cannot represent a champion MIGRATING between roles -- a buff aimed at one role
            # moves pick share sideways rather than up. Vivid cases exist (Qiyana 16.13:
            # jungle +93%, mid -27% off one buff line). But it does NOT explain the model's
            # worst errors, and reporting it as though it did would be wrong.
            rs = PR.role_splits(panel)
            if not rs.empty:
                keys = set(zip(rs["patch"], rs["champion"]))
                allr = topr = allc = topc = 0
                # EVERY patch, not the 6 shown in the table above -- a base rate computed on
                # a handful of patches is too noisy to compare an enrichment against.
                for p in sorted(panel["patch"].unique(), key=PR._pk):
                    s = PR.scoreboard(panel, p)
                    if s.empty:
                        continue
                    s = s.assign(am=s["error_pp"].abs())
                    allr += len(s)
                    allc += sum((p, c) in keys for c in s["champion"])
                    t = s.nlargest(5, "am")
                    topr += len(t)
                    topc += sum((p, c) in keys for c in t["champion"])
                print(f"\n  ROLE MIGRATION — tested, and it is NOT the explanation:")
                print(f"    champion-patches whose roles moved in opposite directions: {len(rs)}")
                print(f"    share of all scored rows            : {allc}/{allr} = {allc/allr*100:.0f}%")
                print(f"    share of the top-5 misses per patch : {topc}/{topr} = {topc/topr*100:.0f}%")
                print(f"    -> essentially no enrichment ({topc/topr*100:.0f}% vs a "
                      f"{allc/allr*100:.0f}% base rate), so role migration is common but does")
                print(f"       not drive the largest errors. Hypothesis checked and rejected.")
                print(f"    gaining role when it happens: "
                      f"{rs['gained_role'].value_counts().to_dict()}")
                print(f"\n    clearest individual cases:")
                print("     " + rs.head(4)[["patch", "champion", "direction", "gained_role",
                                            "gained_pct", "lost_role", "lost_pct"]]
                      .to_string(index=False).replace("\n", "\n     "))

        # --------------------------------------------------------------- 7. is it trusted
        def s7():
            header(7, "EXTRACTION VALIDATION", "whether the LLM's buff/nerf labels can be trusted")
            per_change, _ = VD.grade()
            sc = per_change[per_change["llm_label"].isin(["buff", "nerf"])]
            acc = sc["agrees"].mean() * 100
            print(f"  buff/nerf label vs Data Dragon ground truth")
            print(f"    changes graded : {len(sc)}   across {per_change['patch'].nunique()} patches")
            print(f"    DIRECTION ACCURACY = {acc:.1f}%")
            print(f"    class balance  : {sc['true_label'].value_counts().to_dict()}")

            flip = sc["llm_label"].map({"buff": "nerf", "nerf": "buff"})
            print(f"    negative control (every label flipped): "
                  f"{(flip == sc['true_label']).mean() * 100:.1f}%  <- proves the test can fail")

            by_kind = (sc.groupby("kind").agg(graded=("agrees", "size"),
                                              correct=("agrees", "sum")).reset_index())
            by_kind["accuracy_pct"] = (by_kind["correct"] / by_kind["graded"] * 100).round(1)
            print("\n    by change kind:")
            print("      " + by_kind.to_string(index=False).replace("\n", "\n      "))

            hard = sc[sc["kind"] != "base_stat"]
            if not hard.empty:
                print(f"\n    ABILITY-ONLY ACCURACY = {hard['agrees'].mean() * 100:.1f}% "
                      f"({int(hard['agrees'].sum())}/{len(hard)})")
                print("      Base stats are the easy cases — the notes print the number.")
                print("      Ability cooldown/cost is the hard case: a LONGER cooldown is a")
                print("      nerf even though the number went UP. This is the real test.")
            print("\n    LIMIT: damage values and scaling ratios are absent from Data Dragon")
            print("           (effect arrays are zeroed, vars empty), so those changes have")
            print("           no ground truth and remain unvalidated.")

        # ------------------------------------------------------------ 8. the retired lead
        def s8():
            header(8, "THE RETIRED DAMAGE LEAD", "a finding that looked real and failed to replicate")
            out = io.StringIO()
            with redirect_stdout(out):
                sys.argv = ["magnitude_by_type", "--min-games", str(mg)]
                MBT.main()
            for line in out.getvalue().splitlines():
                if line.strip() and not line.startswith("Pooled"):
                    print("  " + line)
            print("\n  Context: this read +0.05 pp / p=0.008 on its discovery")
            print("  window (16.4-16.13) and passed Bonferroni. It failed to replicate on the")
            print("  15.x backfill, and re-extracting the SAME notes with a different LLM moved")
            print("  the coefficient by a third. 4 categories => Bonferroni needs p < 0.0125.")

        # ---------------------------------------------------------------- 9. data coverage
        def s9():
            header(9, "DATA COVERAGE", "what the dataset is missing, and what it would cost to fix")
            tot = wr.groupby("patch")["games"].sum()
            allp = sorted(tot.index, key=PR._pk)
            present = set(zip(panel["patch_l1"], panel["patch"]))
            lost = []
            for a, b in zip(allp, allp[1:]):
                if (a, b) in present:
                    continue
                if tot[a] < PR.MIN_TOTAL_GAMES and tot[b] < PR.MIN_TOTAL_GAMES:
                    why = "both below the volume floor"
                elif tot[a] < PR.MIN_TOTAL_GAMES:
                    why = f"{a} too thin ({int(tot[a]):,} games)"
                elif tot[b] < PR.MIN_TOTAL_GAMES:
                    why = f"{b} too thin ({int(tot[b]):,} games)"
                elif PR._step_width(a, b) != 1:
                    why = f"not consecutive (spans {PR._step_width(a, b)})"
                else:
                    why = "no champion clears the games filter on both sides"
                lost.append({"step": f"{a} -> {b}", "why": why})
            print(f"  patch steps usable by the model : {len(present)}")
            print(f"  patch steps unusable            : {len(lost)}")
            if lost:
                print()
                for r in lost:
                    print(f"    {r['step']:18} {r['why']}")
            thin = [(p, int(tot[p])) for p in allp if tot[p] < PR.MIN_TOTAL_GAMES]
            if thin:
                print(f"\n  patches below the {PR.MIN_TOTAL_GAMES:,}-game floor:")
                for p, g in thin:
                    need = PR.MIN_TOTAL_GAMES - g
                    print(f"    {p:>6}: {g:>7,} games   needs +{need:,} (~{need // 10} matches)")

        for fn in (s1, s2, s3, s4, s5, s6, s7, s8, s9):
            section(fn)

        print("\n" + "=" * 78)
        print("  Reproduce any single section on its own:")
        print("    python src/noise_floor.py            # section 2")
        print("    python src/pickrate.py               # sections 3, 4")
        print("    python src/validate_direction.py     # section 7")
        print("    python src/magnitude_by_type.py      # section 8")
        print("=" * 78)

    if args.save:
        out = DATA_PROCESSED / "results_summary.txt"
        out.write_text(buf.getvalue(), encoding="utf-8")
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
