"""
Validate LLM patch-note extraction against Data Dragon ground truth.

Data Dragon publishes real per-patch base stats, so for BASE-STAT changes we know the
true numbers. Comparing the LLM's extracted base-stat changes against the Data Dragon
diff gives a measurable accuracy bar — BUT the naive "recall" number conflates two very
different things, so this script reports them separately:

  1. EXTRACTION PRECISION  — of the base-stat changes the LLM claimed, how many are
     real (match Data Dragon)? This measures the LLM's job: parsing the notes correctly.

  2. GROUND-TRUTH COVERAGE — of the real base-stat changes in the game files, how many
     did the extraction report? The gap here is NOT necessarily an LLM error: a real
     change missing from the notes is an UNDOCUMENTED change (e.g. a micropatch). We
     flag those explicitly — detecting them is a feature of the two-source design, not
     a failure of extraction.

Ability changes aren't in Data Dragon and can't be graded here — the LLM is the only
source for those.

Usage:
  python src/validate_extraction.py --extracted data/processed/extracted_16.13.json \
      --new 16.13.1 --old 16.12.1 --notes data/raw/patch_notes/26_13.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 output so non-ASCII characters don't crash on non-UTF-8 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datadragon import diff_stats  # noqa: E402

# Map Data Dragon stat keys to the words the LLM is likely to emit in `field`.
STAT_ALIASES = {
    "hp": ["hp", "health"],
    "attackdamage": ["attack damage", "ad", "base ad"],
    "attackdamageperlevel": ["ad growth", "attack damage growth", "ad per level", "attack damage per level"],
    "armor": ["armor"],
    "armorperlevel": ["armor growth", "armor per level"],
    "attackspeed": ["attack speed"],
    "movespeed": ["move speed", "movement speed", "ms"],
    "mp": ["mana"],
    "mpregen": ["mana regen", "mana regeneration"],
    "hpregen": ["health regen", "hp regen", "health regeneration"],
    "hpperlevel": ["hp per level", "health per level", "hp growth"],
    "spellblock": ["magic resist", "mr", "spell block"],
}


def field_matches_stat(field: str, stat: str) -> bool:
    field = field.lower()
    return any(alias in field for alias in STAT_ALIASES.get(stat, [stat]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade LLM extraction vs Data Dragon")
    parser.add_argument("--extracted", required=True, help="extracted_*.json from llm_extract.py")
    parser.add_argument("--new", required=True, help="Data Dragon NEW version, e.g. 16.13.1")
    parser.add_argument("--old", required=True, help="Data Dragon OLD version, e.g. 16.12.1")
    parser.add_argument("--notes", help="raw patch-note .txt (lets us classify unreported changes "
                                        "as undocumented vs. missed)")
    args = parser.parse_args()

    truth = diff_stats(args.new, args.old)
    truth_pairs = {(row.champion, row.stat) for row in truth.itertuples()}
    truth_name = {row.champion: row.name for row in truth.itertuples()}

    extracted = json.loads(Path(args.extracted).read_text(encoding="utf-8")).get("changes", [])
    notes_text = Path(args.notes).read_text(encoding="utf-8").lower() if args.notes else None

    matched, claimed_base = set(), 0
    for c in extracted:
        if "base" not in c["target"].lower():
            continue  # only grading base-stat changes here
        claimed_base += 1
        for champ, stat in truth_pairs:
            if c["champion"] == champ and field_matches_stat(c["field"], stat):
                matched.add((champ, stat))

    tp = len(matched)
    precision = tp / claimed_base if claimed_base else float("nan")
    coverage = tp / len(truth_pairs) if truth_pairs else float("nan")

    print(f"Base-stat validation  {args.old} -> {args.new}")
    print("=" * 60)
    print(f"Real base-stat changes in game files (Data Dragon): {len(truth_pairs)}")
    print(f"Base-stat changes claimed by extraction:            {claimed_base}")
    print(f"Correct (claimed AND real):                         {tp}")
    print()
    print(f"  EXTRACTION PRECISION = {precision:.0%}  "
          f"(of the extraction's base-stat claims, how many are real)")
    print(f"  GROUND-TRUTH COVERAGE = {coverage:.0%}  "
          f"(of real base-stat changes, how many the extraction reported)")

    # Classify real changes the extraction did NOT report.
    unreported = truth_pairs - matched
    if unreported:
        print("\nReal base-stat changes NOT in the extraction "
              "(the coverage gap - investigate, don't just count as errors):")
        for champ, stat in sorted(unreported):
            name = truth_name.get(champ, champ)
            if notes_text is not None:
                mentioned = name.lower() in notes_text or champ.lower() in notes_text
                if not mentioned:
                    tag = "UNDOCUMENTED - champion not mentioned in patch notes at all"
                else:
                    tag = "champion IS in notes but this stat line isn't - undocumented tweak or extraction gap (review)"
            else:
                tag = "pass --notes to classify as undocumented vs missed"
            print(f"  {name} - {stat}: {tag}")


if __name__ == "__main__":
    main()
