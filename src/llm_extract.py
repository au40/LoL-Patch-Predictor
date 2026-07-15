"""
LLM patch-note extraction.

Riot's patch notes are written prose, not structured data ("Q damage increased to
80/120/160", "Base health lowered", "now grants bonus movement speed"). This script
uses Claude with a fixed JSON schema (structured outputs) to turn a patch note into
machine-readable change records:

    {champion, target, field, change_type, old, new, magnitude_pct}

Base-stat changes extracted here can be cross-checked against Data Dragon (see
validate_extraction.py) to get a real, measurable accuracy number for the write-up.

Usage:
  python src/llm_extract.py --file data/raw/patch_notes/sample_16_13.txt --patch 16.13
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ANTHROPIC_API_KEY, DATA_PROCESSED  # noqa: E402
from datadragon import champion_id_map, normalize_champion  # noqa: E402

# Default model. Opus is the most capable; for high-volume extraction across many
# patches you can switch to a cheaper model to save cost (your call):
#   "claude-sonnet-5"   -> strong + cheaper
#   "claude-haiku-4-5"  -> cheapest, fine for clean structured extraction
MODEL = "claude-opus-4-8"

CHANGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "patch": {"type": "string"},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "champion": {"type": "string", "description": "Champion name, e.g. 'Aatrox'"},
                    "target": {"type": "string", "description": "What was changed: 'Base Stats', an ability name ('Q - Dark Flight'), or 'Passive'"},
                    "field": {"type": "string", "description": "The specific attribute, e.g. 'HP', 'Attack Damage', 'Cooldown', 'Q damage'"},
                    "change_type": {"type": "string", "enum": ["buff", "nerf", "adjust", "new", "removed"]},
                    "old": {"type": "string", "description": "Previous value as written, or '' if not stated"},
                    "new": {"type": "string", "description": "New value as written, or '' if not stated"},
                    "magnitude_pct": {"type": "number", "description": "Percent change if computable from old/new numbers, else 0"},
                },
                "required": ["champion", "target", "field", "change_type", "old", "new", "magnitude_pct"],
            },
        },
    },
    "required": ["patch", "changes"],
}

SYSTEM = (
    "You extract League of Legends champion balance changes from patch notes into a "
    "strict schema. Only include champion changes (skip items, runes, systems, bugfixes "
    "unless they are champion ability changes). One record per distinct attribute changed. "
    "For per-rank values written like '80/120/160', keep them verbatim in old/new. "
    "Set magnitude_pct only when both old and new are single numbers you can compare; "
    "otherwise 0. change_type: 'buff' if strictly better for the champion, 'nerf' if worse, "
    "'adjust' if mixed/unclear, 'new'/'removed' for added or deleted effects."
)


def extract(patch_note_text: str, patch: str, model: str = MODEL) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY or None)
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": CHANGE_SCHEMA}},
        messages=[{
            "role": "user",
            "content": f"Patch {patch} notes:\n\n{patch_note_text}",
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM patch-note extraction")
    parser.add_argument("--file", required=True, help="path to a patch-note .txt file")
    parser.add_argument("--patch", required=True, help="patch label, e.g. 16.13")
    parser.add_argument("--model", default=MODEL, help=f"Anthropic model id (default {MODEL})")
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    text = Path(args.file).read_text(encoding="utf-8")
    result = extract(text, args.patch, model=args.model)

    # Normalize champion names to Data Dragon ids so the extraction joins to win-rate data
    # ('Cho'Gath' -> 'Chogath', 'Xin Zhao' -> 'XinZhao', 'Kai'Sa' -> 'Kaisa').
    id_map = champion_id_map()
    for c in result.get("changes", []):
        c["champion"] = normalize_champion(c["champion"], id_map)

    changes = result.get("changes", [])
    print(f"Extracted {len(changes)} champion changes from patch {args.patch}:\n")
    for c in changes:
        arrow = f"{c['old']} -> {c['new']}" if c["old"] or c["new"] else "(no numeric value)"
        print(f"  [{c['change_type']:6}] {c['champion']:14} {c['target']:22} {c['field']:22} {arrow}")

    out = DATA_PROCESSED / f"extracted_{args.patch}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
