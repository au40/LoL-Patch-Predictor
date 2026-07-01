"""
Win-rate data layer (the swappable seam).

The model consumes ONE fixed schema, regardless of where the numbers came from:

    columns: patch, champion, role, games, winrate
      patch     str    e.g. "16.13"
      champion  str    Data Dragon id, e.g. "Aatrox"
      role      str    "TOP" | "JUNGLE" | "MID" | "BOT" | "SUPPORT"
      games     int    sample size (used for weighting / min-games filter)
      winrate   float  0..1

Backends drop CSVs into data/raw/winrates/ following that schema:
  - NOW (hybrid): third-party aggregates (op.gg / u.gg / lolalytics) exported to CSV.
  - LATER (raw Riot): riot_ingest.py will aggregate match data and WRITE the same CSV.

Because the model only knows this schema, switching data sources never touches the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW  # noqa: E402

WINRATE_DIR = DATA_RAW / "winrates"
REQUIRED_COLS = ["patch", "champion", "role", "games", "winrate"]


def load_winrates(directory: Path = WINRATE_DIR, include_sample: bool = False) -> pd.DataFrame:
    """Load and concatenate every winrate CSV in the directory.

    Files named SAMPLE_*.csv (synthetic placeholder data) are skipped by default so
    real ingested data is never silently blended with the demo data. Pass
    include_sample=True to load them anyway (e.g. for the out-of-the-box demo)."""
    directory.mkdir(parents=True, exist_ok=True)
    csvs = [c for c in sorted(directory.glob("*.csv"))
            if include_sample or not c.name.startswith("SAMPLE_")]
    if not csvs:
        raise FileNotFoundError(
            f"No winrate CSVs in {directory} (SAMPLE_* files are skipped unless "
            f"include_sample=True). Drop in files with columns: {REQUIRED_COLS}"
        )
    # Force patch + champion to string on read: otherwise pandas infers the patch
    # column as float and collapses "16.10" -> 16.1 (== "16.1"), silently merging patches.
    frames = [pd.read_csv(c, dtype={"patch": str, "champion": str}) for c in csvs]
    df = pd.concat(frames, ignore_index=True)
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Winrate data missing columns: {missing}")
    df["patch"] = df["patch"].astype(str)
    return df[REQUIRED_COLS]


def patch_boundary(df: pd.DataFrame, patch_new: str, patch_old: str,
                   min_games: int = 200, role: str | None = None) -> pd.DataFrame:
    """
    Join a champion's win-rate across a patch boundary and compute the raw shift.
    Applies the data-hygiene rules the feedback asked us to lock in early:
    a minimum games-played threshold and (optionally) a single role.

    Returns columns: champion, role, wr_old, wr_new, games_old, games_new, delta
    """
    old = df[df["patch"] == patch_old].copy()
    new = df[df["patch"] == patch_new].copy()
    if role:
        old = old[old["role"] == role]
        new = new[new["role"] == role]

    old = old[old["games"] >= min_games]
    new = new[new["games"] >= min_games]

    merged = old.merge(new, on=["champion", "role"], suffixes=("_old", "_new"))
    merged["delta"] = merged["winrate_new"] - merged["winrate_old"]
    return merged.rename(columns={"winrate_old": "wr_old", "winrate_new": "wr_new"})[
        ["champion", "role", "wr_old", "wr_new", "games_old", "games_new", "delta"]
    ]
