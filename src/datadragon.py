"""
Data Dragon ingestion (NO API KEY REQUIRED).

Data Dragon is Riot's static CDN. It publishes champion *base stats* for every
patch version. We use it two ways:

  1. As a data-ingestion source (champion base stats per patch).
  2. As GROUND TRUTH: diffing base stats between two patches gives us the *actual*
     numeric changes Riot made (HP, AD, attack speed, ...). We later check the
     LLM patch-note extractor against this, giving us a measurable accuracy bar.

Endpoints:
  versions:   https://ddragon.leagueoflegends.com/api/versions.json
  champions:  https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json

Usage:
  python src/datadragon.py --list 5              # show 5 most recent patch versions
  python src/datadragon.py --diff 15.13.1 15.12.1  # base-stat changes between two patches
  python src/datadragon.py --diff-latest         # diff the two most recent patches
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROCESSED  # noqa: E402

BASE = "https://ddragon.leagueoflegends.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "LoL-Patch-Predictor (educational project)"})

# Base stats we care about for change detection.
STAT_COLS = [
    "hp", "hpperlevel", "mp", "mpperlevel", "movespeed",
    "armor", "armorperlevel", "spellblock", "spellblockperlevel",
    "attackrange", "hpregen", "hpregenperlevel", "mpregen", "mpregenperlevel",
    "crit", "attackdamage", "attackdamageperlevel", "attackspeed", "attackspeedperlevel",
]


def _get_json(url: str, cache_name: str | None = None) -> dict:
    """GET JSON with a simple on-disk cache under data/raw."""
    if cache_name:
        cache_path = DATA_RAW / cache_name
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if cache_name:
        (DATA_RAW / cache_name).write_text(json.dumps(data), encoding="utf-8")
    return data


def get_versions() -> list[str]:
    """All Data Dragon versions, newest first, filtered to numeric patch versions."""
    versions = _get_json(f"{BASE}/api/versions.json", "ddragon_versions.json")
    return [v for v in versions if re.match(r"^\d+\.\d+\.\d+$", v)]


def get_champion_stats(version: str) -> pd.DataFrame:
    """Return a DataFrame of base stats for every champion at a given patch version."""
    url = f"{BASE}/cdn/{version}/data/en_US/champion.json"
    data = _get_json(url, f"champion_{version}.json")
    rows = []
    for champ_id, champ in data["data"].items():
        stats = champ.get("stats", {})
        row = {"champion": champ["id"], "name": champ["name"],
               "tags": "/".join(champ.get("tags", []))}
        for col in STAT_COLS:
            row[col] = stats.get(col)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("champion").reset_index(drop=True)
    df.insert(0, "version", version)
    return df


def champion_id_map(version: str = "16.13.1") -> dict[str, str]:
    """Map any spelling of a champion -> its Data Dragon id (the win-rate join key).

    Riot's patch notes say 'Cho'Gath' / 'Xin Zhao' / 'Kai'Sa'; the model joins on the
    Data Dragon id 'Chogath' / 'XinZhao' / 'Kaisa'. We index both the id and the display
    name, plus an alphanumeric-stripped form of each, so 'Kai'Sa', 'kaisa', and 'KaiSa'
    all resolve. Built from the (cached) champion.json, so no network hit if it's local."""
    df = get_champion_stats(version)
    m: dict[str, str] = {}
    for r in df.itertuples():
        cid, name = r.champion, r.name
        for key in (cid, name):
            m[key.lower()] = cid
            m[re.sub(r"[^a-z0-9]", "", key.lower())] = cid
    return m


def normalize_champion(name: str, id_map: dict[str, str]) -> str:
    """Resolve a champion name to its Data Dragon id; return the original if unknown."""
    key = name.lower()
    if key in id_map:
        return id_map[key]
    return id_map.get(re.sub(r"[^a-z0-9]", "", key), name)


def diff_stats(version_new: str, version_old: str) -> pd.DataFrame:
    """
    Base-stat changes from version_old -> version_new.
    Returns one row per (champion, stat) that changed, with old/new/delta/pct.
    This is our GROUND TRUTH for base-stat patch changes.
    """
    new = get_champion_stats(version_new).set_index("champion")
    old = get_champion_stats(version_old).set_index("champion")
    champs = new.index.intersection(old.index)

    changes = []
    for champ in champs:
        for stat in STAT_COLS:
            a, b = old.at[champ, stat], new.at[champ, stat]
            if a is None or b is None:
                continue
            if a != b:
                pct = (b - a) / a * 100 if a not in (0, None) else float("nan")
                changes.append({
                    "champion": champ,
                    "name": new.at[champ, "name"],
                    "stat": stat,
                    "old": a,
                    "new": b,
                    "delta": round(b - a, 4),
                    "pct_change": round(pct, 2) if pct == pct else None,
                    "from_patch": version_old,
                    "to_patch": version_new,
                })
    return pd.DataFrame(changes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Dragon ingestion / stat diffing")
    parser.add_argument("--list", type=int, metavar="N", help="show N most recent patch versions")
    parser.add_argument("--diff", nargs=2, metavar=("NEW", "OLD"), help="diff two patch versions")
    parser.add_argument("--diff-latest", action="store_true", help="diff the two most recent patches")
    args = parser.parse_args()

    if args.list:
        versions = get_versions()
        print(f"Most recent {args.list} patch versions:")
        for v in versions[: args.list]:
            print(f"  {v}")
        return

    if args.diff_latest:
        versions = get_versions()
        args.diff = [versions[0], versions[1]]

    if args.diff:
        new, old = args.diff
        print(f"Base-stat changes  {old}  ->  {new}\n" + "=" * 60)
        changes = diff_stats(new, old)
        if changes.empty:
            print("No base-stat changes detected between these patches.")
            return
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", 120)
        print(changes.to_string(index=False))
        out = DATA_PROCESSED / f"ddragon_diff_{old}_to_{new}.csv"
        changes.to_csv(out, index=False)
        print(f"\n{len(changes)} stat changes across "
              f"{changes['champion'].nunique()} champions.")
        print(f"Saved -> {out.relative_to(out.parent.parent.parent)}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
