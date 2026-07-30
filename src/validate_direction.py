"""
Validate the LLM's BUFF/NERF LABEL against Data Dragon ground truth.

`validate_extraction.py` grades whether the extraction found the right base-stat *changes*.
It never checks whether the extraction got the *direction* right — and the direction is the
only feature the pick-rate model actually consumes (`buffed`, `nerfed`, `net_buff`). So the
headline result has been resting on an unvalidated label. This closes that.

The grading is fully automatic, not hand-labelled. Data Dragon publishes real per-patch base
stats, and every base stat it carries is positive-polarity — more health, armour, attack
damage, move speed, regen or growth is unambiguously better for the champion. So for any
base-stat change we can match to Data Dragon, the true direction is just the sign of the
delta, and we can score the LLM's `change_type` against it across every patch at once.

Scope and honesty about it: only BASE-STAT changes are gradable, because ability changes
aren't in Data Dragon. Base stats are ~19% of all extracted changes, so this is a sample of
the label's accuracy, not a census — but it is a real accuracy number on real ground truth,
and it is the same population the model reads.

Usage:
  python src/validate_direction.py                 # every patch we have an extraction for
  python src/validate_direction.py --patches 16.11 16.12 16.13
  python src/validate_direction.py --show-disagreements
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from datadragon import (diff_stats, diff_spells, spell_slot,  # noqa: E402
                        champion_id_map, normalize_champion)
from validate_extraction import STAT_ALIASES  # noqa: E402
from magnitude_by_type import _pk  # noqa: E402

# Which Data Dragon ability field an extracted change is talking about. Damage and scaling
# ratios are absent on purpose — Riot zeroes the `effect` arrays and leaves `vars` empty, so
# those changes have no ground truth and are simply not gradable here.
SPELL_FIELD_WORDS = {
    "cooldown": ["cooldown", "cd", "recharge"],
    "cost": ["cost", "mana", "energy"],
    "range": ["range"],
}


def _spell_field(text: str) -> str | None:
    """Which gradable ability field does this change's `field` refer to, if any."""
    t = text.lower()
    best, best_len = None, 0
    for field, words in SPELL_FIELD_WORDS.items():
        for w in words:
            if w in t and len(w) > best_len:
                best, best_len = field, len(w)
    return best

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"

# Every base stat Data Dragon carries is "higher is better" for the champion, so the sign of
# the delta IS the ground-truth direction. (There is no cost/cooldown among base stats --
# those live on abilities, which Data Dragon doesn't diff and this script doesn't grade.)
POSITIVE_POLARITY = True


def _best_stat_match(field: str, stats: list[str]) -> str | None:
    """Which Data Dragon stat is this extracted `field` talking about?

    Aliases overlap -- 'magic resist growth' contains the alias for both `spellblock` and
    `spellblockperlevel` -- so take the stat whose LONGEST matching alias is longest, which
    prefers the more specific reading ('magic resist growth' over bare 'magic resist')."""
    field = field.lower()
    best, best_len = None, 0
    for stat in stats:
        for alias in STAT_ALIASES.get(stat, [stat]):
            if alias in field and len(alias) > best_len:
                best, best_len = stat, len(alias)
    return best


def _dd_versions() -> list[str]:
    """Data Dragon versions we have champion files for, oldest first."""
    vs = [p.stem.replace("champion_", "") for p in DATA_RAW.glob("champion_*.json")]
    return sorted(vs, key=lambda v: _pk(".".join(v.split(".")[:2])))


def grade(patches: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every gradable base-stat change. Returns (per-change rows, per-patch summary)."""
    versions = _dd_versions()
    vindex = {v: i for i, v in enumerate(versions)}

    files = sorted(DATA_PROCESSED.glob("extracted_*.json"),
                   key=lambda f: _pk(f.stem.replace("extracted_", "")))
    rows = []
    for f in files:
        patch = f.stem.replace("extracted_", "")
        if patches and patch not in patches:
            continue
        new_v = f"{patch}.1"
        i = vindex.get(new_v)
        if i is None or i == 0:
            continue                      # no Data Dragon file, or nothing before it
        old_v = versions[i - 1]

        try:
            truth = diff_stats(new_v, old_v)
        except Exception:                 # noqa: BLE001 - a missing/corrupt version file
            continue
        if truth.empty:
            continue

        by_champ: dict[str, list] = {}
        for r in truth.itertuples():
            by_champ.setdefault(r.champion, []).append(r)

        # Ability-level ground truth, when we have detail files for both versions. Data
        # Dragon publishes per-rank cooldown / cost / range; damage and ratios it does not.
        spells = pd.DataFrame()
        try:
            spells = diff_spells(new_v, old_v)
        except Exception:                 # noqa: BLE001 - detail files not fetched for this pair
            pass
        by_spell: dict[tuple, object] = {}
        for r in spells.itertuples():
            by_spell[(r.champion, r.slot, r.field)] = r

        idm = champion_id_map(new_v)
        for c in json.loads(f.read_text(encoding="utf-8")).get("changes", []):
            champ = normalize_champion(c["champion"], idm)
            target = str(c.get("target", ""))
            field_txt = str(c.get("field", ""))

            if "base" in target.lower():
                # --- base stat: graded against the champion.json stat diff ---------------
                cands = by_champ.get(champ)
                if not cands:
                    continue              # claimed a change Data Dragon doesn't show
                stat = _best_stat_match(field_txt, [r.stat for r in cands])
                if stat is None:
                    continue
                t = next(r for r in cands if r.stat == stat)
                truth_dir = "buff" if t.delta > 0 else "nerf"
                kind, what, old_v_, new_v_, delta = "base_stat", stat, t.old, t.new, t.delta
            else:
                # --- ability: graded against the per-rank spell diff ---------------------
                slot = spell_slot(target)
                sfield = _spell_field(field_txt)
                if slot in (None, "PASSIVE") or sfield is None:
                    continue              # damage/ratio/mechanic change — no ground truth
                t = by_spell.get((champ, slot, sfield))
                if t is None:
                    continue              # Data Dragon shows no change to that field
                truth_dir = "buff" if t.buffed else "nerf"
                kind = f"ability_{sfield}"
                what, old_v_, new_v_, delta = f"{slot} {sfield}", t.old, t.new, t.delta

            rows.append({
                "patch": patch, "kind": kind, "champion": champ, "field": c.get("field"),
                "stat": what, "old": old_v_, "new": new_v_, "delta": delta,
                "llm_label": c.get("change_type"), "true_label": truth_dir,
                "agrees": c.get("change_type") == truth_dir,
            })

    per_change = pd.DataFrame(rows)
    if per_change.empty:
        return per_change, pd.DataFrame()

    # Only buff/nerf claims are scoreable. 'adjust' / 'new' / 'removed' are the LLM
    # declining to take a side; count them separately rather than marking them wrong.
    scoreable = per_change[per_change["llm_label"].isin(["buff", "nerf"])]
    summary = (scoreable.groupby("patch")
               .agg(graded=("agrees", "size"), correct=("agrees", "sum"))
               .reset_index())
    summary["accuracy_pct"] = (summary["correct"] / summary["graded"] * 100).round(1)
    return per_change, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade the LLM's buff/nerf label vs Data Dragon")
    ap.add_argument("--patches", nargs="+", default=None, help="limit to these patches")
    ap.add_argument("--show-disagreements", action="store_true",
                    help="print every change where the LLM and Data Dragon disagree")
    ap.add_argument("--save", action="store_true",
                    help="write the per-change rows to data/processed/direction_validation.csv")
    args = ap.parse_args()

    per_change, summary = grade(args.patches)
    if per_change.empty:
        print("Nothing gradable — no base-stat changes matched Data Dragon.")
        return

    scoreable = per_change[per_change["llm_label"].isin(["buff", "nerf"])]
    abstained = len(per_change) - len(scoreable)
    correct = int(scoreable["agrees"].sum())
    total = len(scoreable)
    acc = correct / total * 100 if total else float("nan")

    print("Buff/nerf LABEL validation vs Data Dragon ground truth")
    print("=" * 62)
    print(f"Patches covered:                    {per_change['patch'].nunique()}")
    print(f"Base-stat changes matched to truth: {len(per_change)}")
    print(f"  scoreable (labelled buff/nerf):   {total}")
    print(f"  abstained (adjust/new/removed):   {abstained}")
    print()
    print(f"  DIRECTION ACCURACY = {acc:.1f}%   ({correct}/{total})")
    print()

    print("Confusion matrix (rows = LLM said, cols = Data Dragon truth):")
    cm = pd.crosstab(scoreable["llm_label"], scoreable["true_label"])
    print(cm.to_string())

    # Base stats are the EASY cases — the notes print the number and the direction is
    # obvious. Ability cooldown/cost changes are the hard ones (a longer cooldown is a nerf
    # even though the number went up), so report them separately or the headline flatters.
    print("\nAccuracy by change kind (base stats are the easy cases):")
    by_kind = (scoreable.groupby("kind")
               .agg(graded=("agrees", "size"), correct=("agrees", "sum"))
               .reset_index())
    by_kind["accuracy_pct"] = (by_kind["correct"] / by_kind["graded"] * 100).round(1)
    print(by_kind.to_string(index=False))

    hard = scoreable[scoreable["kind"] != "base_stat"]
    if not hard.empty:
        print(f"\n  ABILITY-ONLY ACCURACY = {hard['agrees'].mean() * 100:.1f}%  "
              f"({int(hard['agrees'].sum())}/{len(hard)})  <- the cases that can fool an LLM")

    if not summary.empty:
        print("\nPer-patch accuracy:")
        print(summary.to_string(index=False))

    bad = scoreable[~scoreable["agrees"]]
    if args.show_disagreements and not bad.empty:
        print(f"\nAll {len(bad)} disagreements (LLM label vs the real stat move):")
        print(bad[["patch", "champion", "field", "stat", "old", "new",
                   "llm_label", "true_label"]].to_string(index=False))
    elif not bad.empty:
        print(f"\n{len(bad)} disagreements — rerun with --show-disagreements to inspect them.")

    if args.save:
        out = DATA_PROCESSED / "direction_validation.csv"
        per_change.to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
