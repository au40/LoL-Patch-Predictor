"""
Skill-order ingestion: which ability each champion MAXES FIRST, from the Riot match
TIMELINE endpoint (its SKILL_LEVEL_UP events are the only place leveling order lives).

Why a short pull is enough: skill order is very low-variance — for a given champion-role
almost everyone maxes the same basic ability first — so a few hundred matches already give
a clear modal max-first. That's why this doesn't need the multi-hour grind the win-rates did.

Cost: TWO calls per match (match() for champion+role, match_timeline() for the skill-ups),
so ~half the throughput of the win-rate ingester — but you need far fewer matches. Reuses
RiotClient (rate limiter, 429/5xx retry) and ROLE_MAP from riot_ingest.py, and normalizes
champion names to Data Dragon ids so the output joins to your win-rate / cooldown data.

"Maxed first" = the first basic ability (Q/W/E) to reach 5 points; if a game ends early,
fall back to whichever basic has the most points. Champions are pooled across patches
(skill order barely moves patch-to-patch), which keeps per-champ samples up.

Output: data/processed/skill_order.csv
  columns: champion, role, games, max_first, max_first_share, q, w, e

Usage:
  python src/skill_order.py --tier DIAMOND --division I --max-players 60 --matches-per-player 15
  python src/skill_order.py --resume --tier DIAMOND --divisions I II --max-players 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RIOT_API_KEY, RIOT_REGION, RIOT_PLATFORM, DATA_PROCESSED  # noqa: E402
from riot_ingest import RiotClient, ROLE_MAP, patch_of  # noqa: E402
from datadragon import champion_id_map, normalize_champion  # noqa: E402

SLOT = {1: "Q", 2: "W", 3: "E", 4: "R"}   # Riot skillSlot -> ability letter
BASICS = ("Q", "W", "E")
OUT_CSV = DATA_PROCESSED / "skill_order.csv"
STATE = DATA_PROCESSED / "skill_order.state.json"


def maxed_first(slots: list[int]) -> str | None:
    """First basic ability (Q/W/E) to reach 5 points, from one player's ordered skill-ups.
    Falls back to the most-pointed basic if the game ended before any basic maxed."""
    counts = {"Q": 0, "W": 0, "E": 0, "R": 0}
    for s in slots:
        letter = SLOT.get(s)
        if not letter:
            continue
        counts[letter] += 1
        if letter in BASICS and counts[letter] == 5:
            return letter
    best = max(BASICS, key=lambda b: counts[b])
    return best if counts[best] > 0 else None


def timeline_max_first(timeline: dict) -> dict[int, str]:
    """participantId (1-10) -> maxed-first ability, read from SKILL_LEVEL_UP events."""
    seq: dict[int, list[int]] = defaultdict(list)
    for frame in timeline.get("info", {}).get("frames", []):
        for ev in frame.get("events", []):
            if ev.get("type") == "SKILL_LEVEL_UP":
                seq[ev["participantId"]].append(ev.get("skillSlot"))
    out = {}
    for pid, slots in seq.items():
        mf = maxed_first(slots)
        if mf:
            out[pid] = mf
    return out


def write_csv(stats: dict, id_map: dict):
    rows = []
    for (champ, role), s in stats.items():
        games = s["games"]
        if games == 0:
            continue
        mf = max(BASICS, key=lambda b: s[b])
        rows.append({
            "champion": normalize_champion(champ, id_map), "role": role, "games": games,
            "max_first": mf, "max_first_share": round(s[mf] / games, 3),
            "q": s["Q"], "w": s["W"], "e": s["E"],
        })
    df = pd.DataFrame(rows).sort_values(["role", "games"], ascending=[True, False])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df


def save_state(stats: dict, seen: set):
    STATE.write_text(json.dumps({
        "stats": [[c, r, s["Q"], s["W"], s["E"], s["games"]] for (c, r), s in stats.items()],
        "seen_matches": list(seen),
    }), encoding="utf-8")


def load_state():
    stats = defaultdict(lambda: {"Q": 0, "W": 0, "E": 0, "games": 0})
    seen: set[str] = set()
    if STATE.exists():
        st = json.loads(STATE.read_text(encoding="utf-8"))
        for c, r, q, w, e, g in st.get("stats", []):
            stats[(c, r)] = {"Q": q, "W": w, "E": e, "games": g}
        seen = set(st.get("seen_matches", []))
    return stats, seen


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest champion skill-max order from match timelines")
    ap.add_argument("--queue", default="RANKED_SOLO_5x5")
    ap.add_argument("--tier", default="DIAMOND")
    ap.add_argument("--division", default="I")
    ap.add_argument("--divisions", nargs="+", default=None, help="sweep several divisions")
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--matches-per-player", type=int, default=15)
    ap.add_argument("--max-players", type=int, default=60)
    ap.add_argument("--days", type=int, default=45, help="only pull matches from the last N days")
    ap.add_argument("--min-games", type=int, default=5, help="drop champ-roles below this in the CSV")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not RIOT_API_KEY:
        print("ERROR: RIOT_API_KEY is not set (see .env).")
        sys.exit(1)

    client = RiotClient(RIOT_API_KEY, RIOT_PLATFORM, RIOT_REGION)
    id_map = champion_id_map()
    start_time = int(time.time()) - args.days * 86400

    # 1. Sample players from the ladder.
    divisions = args.divisions if args.divisions else [args.division]
    puuids: list[str] = []
    for division in divisions:
        div: list[str] = []
        for page in range(1, args.pages + 1):
            for e in client.league_entries(args.queue, args.tier, division, page):
                if "puuid" in e:
                    div.append(e["puuid"])
            if len(div) >= args.max_players:
                break
        puuids.extend(div[: args.max_players])
        print(f"Sampled {len(div[:args.max_players])} players from {args.tier} {division}.")
    seen_p: set[str] = set()
    puuids = [p for p in puuids if not (p in seen_p or seen_p.add(p))]
    print(f"Total {len(puuids)} players across {divisions}.")

    stats, seen_matches = (load_state() if args.resume
                           else (defaultdict(lambda: {"Q": 0, "W": 0, "E": 0, "games": 0}), set()))
    if args.resume:
        print(f"Resuming: {sum(s['games'] for s in stats.values())} player-games, "
              f"{len(seen_matches)} matches seen.")
    patches_seen: set[str] = set()

    # 2. Per player -> match ids -> (match + timeline) -> skill order per participant.
    for i, puuid in enumerate(puuids, 1):
        try:
            ids = client.match_ids(puuid, args.matches_per_player, start_time=start_time)
        except requests.RequestException as exc:
            print(f"  [{i}/{len(puuids)}] match_ids failed: {exc}")
            continue
        for mid in ids:
            if mid in seen_matches:
                continue
            seen_matches.add(mid)
            try:
                match = client.match(mid)
                info = match.get("info", {})
                parts = info.get("participants", [])
                # participantId is 1..10 in participant-list order.
                pid_info = {}
                for idx, p in enumerate(parts, 1):
                    role = ROLE_MAP.get(p.get("teamPosition", ""))
                    if role:
                        pid_info[idx] = (p["championName"], role)
                if not pid_info:
                    continue
                timeline = client.match_timeline(mid)
            except requests.RequestException:
                continue
            patches_seen.add(patch_of(info.get("gameVersion", "")))
            mf_by_pid = timeline_max_first(timeline)
            for pid, (champ, role) in pid_info.items():
                mf = mf_by_pid.get(pid)
                if not mf:
                    continue
                s = stats[(champ, role)]
                s[mf] += 1
                s["games"] += 1
        print(f"  [{i}/{len(puuids)}] {len(seen_matches)} matches, "
              f"{sum(s['games'] for s in stats.values())} player-games")

        if args.checkpoint_every and i % args.checkpoint_every == 0:
            write_csv(stats, id_map)
            save_state(stats, seen_matches)
            print("      [checkpoint] saved")

    save_state(stats, seen_matches)
    df = write_csv(stats, id_map)
    df = df[df["games"] >= args.min_games]
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} champion-role skill orders (min {args.min_games} games) -> {OUT_CSV}")
    print(f"Patches pooled: {sorted(patches_seen)}")
    if not df.empty:
        strong = df[df["max_first_share"] >= 0.6].sort_values("games", ascending=False)
        print(f"\nSample (clear max-first, most games first):")
        print(strong[["champion", "role", "games", "max_first", "max_first_share"]]
              .head(15).to_string(index=False))


if __name__ == "__main__":
    main()
