"""
Riot API ingestion (REQUIRES a dev key in .env).

Riot's match endpoints are per-player / per-match — you can't query "all games on
patch X for champion Y" directly. The real pipeline is:

    ranked ladder  ->  sample players (puuids)
                   ->  each player's recent match IDs
                   ->  each match's details (gameVersion + participants)
                   ->  filter to the target patch
                   ->  aggregate win-rate by (champion, role)
                   ->  write data/raw/winrates/riot_<patch>.csv   (model-ready)

The output CSV matches the schema in winrates.py exactly, so once this runs it
replaces the SAMPLE aggregate data with your own Riot-derived numbers — no change
to model.py needed.

Dev keys are rate-limited (~20 req/s, 100 req/2min) and expire every 24h. Start small.

Usage (after adding RIOT_API_KEY to .env):
  python src/riot_ingest.py --patch 16.13 --tier DIAMOND --division I --pages 1 --matches-per-player 15
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
from config import RIOT_API_KEY, RIOT_REGION, RIOT_PLATFORM, DATA_RAW  # noqa: E402
from datadragon import get_versions  # noqa: E402

ROLE_MAP = {"TOP": "TOP", "JUNGLE": "JUNGLE", "MIDDLE": "MID", "BOTTOM": "BOT", "UTILITY": "SUPPORT"}


class RiotClient:
    """Thin Riot API client with a courteous rate limiter for dev keys."""

    def __init__(self, key: str, platform: str, region: str, min_interval: float = 0.06):
        self.key = key
        self.platform = f"https://{platform}.api.riotgames.com"
        self.region = f"https://{region}.api.riotgames.com"
        self.min_interval = min_interval
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers.update({"X-Riot-Token": key})

    def _get(self, url: str, **params):
        # Spacing + honor 429 Retry-After + retry transient network / 5xx errors,
        # so a dropped connection over a long run doesn't kill the whole ingestion.
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        attempts = 0
        while True:
            try:
                resp = self.session.get(url, params=params, timeout=30)
                self._last = time.time()
            except (requests.ConnectionError, requests.Timeout) as exc:
                attempts += 1
                if attempts > 6:
                    raise
                back = min(2 ** attempts, 60)
                print(f"  network error ({type(exc).__name__}); retry {attempts}/6 in {back}s")
                time.sleep(back)
                continue
            if resp.status_code == 429:
                retry = int(resp.headers.get("Retry-After", "5"))
                print(f"  rate limited; sleeping {retry}s")
                time.sleep(retry)
                continue
            if resp.status_code >= 500:  # transient Riot server error
                attempts += 1
                if attempts > 6:
                    resp.raise_for_status()
                back = min(2 ** attempts, 60)
                print(f"  server {resp.status_code}; retry {attempts}/6 in {back}s")
                time.sleep(back)
                continue
            resp.raise_for_status()
            return resp.json()

    def league_entries(self, queue: str, tier: str, division: str, page: int):
        return self._get(f"{self.platform}/lol/league/v4/entries/{queue}/{tier}/{division}", page=page)

    def match_ids(self, puuid: str, count: int, start_time: int | None = None,
                  end_time: int | None = None):
        params = {"start": 0, "count": count, "type": "ranked"}
        if start_time is not None:
            params["startTime"] = start_time  # epoch seconds; only matches after this
        if end_time is not None:
            params["endTime"] = end_time      # and before this (for older-patch windows)
        return self._get(f"{self.region}/lol/match/v5/matches/by-puuid/{puuid}/ids", **params)

    def match(self, match_id: str):
        return self._get(f"{self.region}/lol/match/v5/matches/{match_id}")


def patch_of(game_version: str) -> str:
    # gameVersion looks like "16.13.598.1234" -> "16.13"
    parts = game_version.split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else game_version


def write_winrates(stats: dict, tag: str):
    """Write the current accumulated stats to the winrate CSV.
    Called periodically (checkpoint) and at the end, so a crash / key expiry / usage
    cutoff mid-run leaves the latest progress on disk instead of losing everything."""
    if not stats:
        return None
    rows = [
        {"patch": patch, "champion": champ, "role": role,
         "games": s["games"], "winrate": round(s["wins"] / s["games"], 4)}
        for (patch, champ, role), s in stats.items()
    ]
    df = pd.DataFrame(rows).sort_values(["patch", "role", "games"], ascending=[True, True, False])
    out = DATA_RAW / "winrates" / f"riot_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out, df


def write_matchups(matchup_stats: dict, tag: str):
    """Write lane-matchup win-rates (champion vs its lane opponent) to a SEPARATE file
    under data/raw/matchups/ so it never collides with the model's win-rate schema in
    data/raw/winrates/. Schema: patch, champion, role, opponent, games, wins, winrate."""
    if not matchup_stats:
        return None
    rows = [
        {"patch": p, "champion": c, "role": r, "opponent": o,
         "games": s["games"], "wins": s["wins"], "winrate": round(s["wins"] / s["games"], 4)}
        for (p, c, r, o), s in matchup_stats.items()
    ]
    df = pd.DataFrame(rows).sort_values(["patch", "role", "champion", "games"],
                                        ascending=[True, True, True, False])
    out = DATA_RAW / "matchups" / f"riot_matchups_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out, df


def _state_path(tag: str) -> Path:
    return DATA_RAW / "winrates" / f"riot_{tag}.state.json"


def save_state(stats: dict, seen_matches: set, tag: str, matchup_stats: dict | None = None) -> None:
    """Persist raw counts + seen match IDs (+ lane matchups) alongside each checkpoint so
    --resume can continue exactly (no re-fetching, no double-counting)."""
    state = {
        "stats": [[p, c, r, s["games"], s["wins"]] for (p, c, r), s in stats.items()],
        "seen_matches": list(seen_matches),
        "matchups": [[p, c, r, o, s["games"], s["wins"]]
                     for (p, c, r, o), s in (matchup_stats or {}).items()],
    }
    _state_path(tag).write_text(json.dumps(state), encoding="utf-8")


def load_prior(tag: str):
    """Load prior progress for --resume. Prefer the exact state sidecar; otherwise
    reconstruct counts from the CSV (seen-match list unknown -> sample a disjoint
    ladder slice). Returns (stats, seen_matches, matchup_stats, source).
    Old sidecars without a 'matchups' key load empty matchups (i.e. capture-forward)."""
    stats = defaultdict(lambda: {"games": 0, "wins": 0})
    matchups = defaultdict(lambda: {"games": 0, "wins": 0})
    seen: set[str] = set()
    sp = _state_path(tag)
    if sp.exists():
        state = json.loads(sp.read_text(encoding="utf-8"))
        for p, c, r, g, w in state.get("stats", []):
            stats[(p, c, r)] = {"games": g, "wins": w}
        for p, c, r, o, g, w in state.get("matchups", []):
            matchups[(p, c, r, o)] = {"games": g, "wins": w}
        return stats, set(state.get("seen_matches", [])), matchups, "state sidecar (exact)"
    csv = DATA_RAW / "winrates" / f"riot_{tag}.csv"
    if csv.exists():
        df = pd.read_csv(csv, dtype={"patch": str, "champion": str})
        for r in df.itertuples():
            stats[(r.patch, r.champion, r.role)] = {
                "games": int(r.games), "wins": int(round(r.winrate * r.games))}
        return stats, seen, matchups, "CSV (no seen-match list -> use a disjoint ladder slice)"
    return stats, seen, matchups, "nothing found (starting fresh)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Riot match data -> winrate CSV")
    parser.add_argument("--patch", default=None,
                        help="keep only this single patch (e.g. 16.13). Usually leave unset and use "
                             "--recent-patches instead.")
    parser.add_argument("--recent-patches", type=int, default=5,
                        help="keep only the N most-recent patches (auto-detected from Data Dragon) "
                             "and window the pull to ~the last N patches so no requests are wasted "
                             "on ancient matches. Default 5.")
    parser.add_argument("--all-patches", action="store_true",
                        help="disable recent-patch targeting; keep every patch found (noisy old tail)")
    parser.add_argument("--patches", nargs="+", default=None,
                        help="target SPECIFIC (incl. OLDER) patches, e.g. --patches 16.8 16.7 16.6 "
                             "16.5 16.4. Auto-windows the pull to roughly when they were live "
                             "(~2 weeks each). Overrides --recent-patches.")
    parser.add_argument("--queue", default="RANKED_SOLO_5x5")
    parser.add_argument("--tier", default="DIAMOND")
    parser.add_argument("--division", default="I")
    parser.add_argument("--divisions", nargs="+", default=None,
                        help="sweep several divisions in one run, e.g. --divisions III IV. "
                             "Overrides --division; --max-players applies PER division.")
    parser.add_argument("--pages", type=int, default=1, help="ladder pages of players to sample")
    parser.add_argument("--matches-per-player", type=int, default=15)
    parser.add_argument("--max-players", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="save the winrate CSV every N players so a crash mid-run keeps "
                             "progress. 0 disables. Default 10.")
    parser.add_argument("--resume", action="store_true",
                        help="load prior progress (CSV / state sidecar) and keep ADDING to it "
                             "instead of starting over. Sample a disjoint ladder slice (e.g. a "
                             "different --division) so new matches don't double-count old ones.")
    args = parser.parse_args()

    if not RIOT_API_KEY:
        print("ERROR: RIOT_API_KEY is not set. Get one at https://developer.riotgames.com/")
        print("Then copy .env.example to .env and paste it in.")
        sys.exit(1)

    client = RiotClient(RIOT_API_KEY, RIOT_PLATFORM, RIOT_REGION)

    # Decide which patches to keep and how far back to pull.
    target_patches: set[str] | None = None
    start_time: int | None = None
    end_time: int | None = None
    if args.patches:
        # Explicit patch list (may be OLDER patches). Window the match-id pull to roughly
        # when those patches were live, estimated from the ~2-week cadence with a generous
        # buffer; the gameVersion filter below still does the precise selection.
        target_patches = set(args.patches)
        minors: list[str] = []
        for v in get_versions():                 # newest first
            mm = v.rsplit(".", 1)[0]
            if mm not in minors:
                minors.append(mm)
        idx = {mm: k for k, mm in enumerate(minors)}   # 0 = newest patch
        tgt = [idx[p] for p in args.patches if p in idx]
        if not tgt:
            print(f"ERROR: none of --patches {args.patches} match Data Dragon versions "
                  f"(newest are {minors[:6]}).")
            sys.exit(1)
        now, day = int(time.time()), 86400
        start_time = now - int((max(tgt) + 2) * 14 * day)             # buffer before oldest
        end_time = min(now, now - int(max(0, min(tgt) - 1) * 14 * day))  # buffer after newest
        print(f"Targeting patches {sorted(target_patches)} via date window "
              f"{time.strftime('%Y-%m-%d', time.gmtime(start_time))} .. "
              f"{time.strftime('%Y-%m-%d', time.gmtime(end_time))}")
    elif args.patch:
        target_patches = {args.patch}
    elif not args.all_patches:
        recent: list[str] = []
        for v in get_versions():                 # newest first, e.g. "16.13.1"
            mm = v.rsplit(".", 1)[0]              # -> "16.13"
            if mm not in recent:
                recent.append(mm)
            if len(recent) >= args.recent_patches:
                break
        target_patches = set(recent)
        # Window the pull to ~the last N patches (~2 weeks each, +1 buffer) so match-id
        # requests skip ancient games entirely.
        start_time = int(time.time()) - (args.recent_patches + 1) * 15 * 86400
        print(f"Targeting last {args.recent_patches} patches: {sorted(target_patches)}")

    # 1. Sample players from the ladder (optionally sweeping several divisions).
    divisions = args.divisions if args.divisions else [args.division]
    puuids: list[str] = []
    for division in divisions:
        div_puuids: list[str] = []
        for page in range(1, args.pages + 1):
            entries = client.league_entries(args.queue, args.tier, division, page)
            for e in entries:
                if "puuid" in e:
                    div_puuids.append(e["puuid"])
            if len(div_puuids) >= args.max_players:
                break
        div_puuids = div_puuids[: args.max_players]
        puuids.extend(div_puuids)
        print(f"Sampled {len(div_puuids)} players from {args.tier} {division}.")
    # De-dup players across divisions, preserving order (safe even though a player
    # normally sits in only one division).
    _seen_p: set[str] = set()
    puuids = [p for p in puuids if not (p in _seen_p or _seen_p.add(p))]
    print(f"Total {len(puuids)} players across division(s): {divisions}")

    tag = args.patch if args.patch else "ingested"

    # 2-4. Pull matches, filter to patch, aggregate win-rate by (champion, role).
    if args.resume:
        stats, seen_matches, matchup_stats, source = load_prior(tag)
        print(f"Resuming from {source}: {sum(s['games'] for s in stats.values())} "
              f"champ-games, {len(seen_matches)} seen matches, "
              f"{len(matchup_stats)} matchup cells loaded.")
    else:
        seen_matches = set()
        stats = defaultdict(lambda: {"games": 0, "wins": 0})
        matchup_stats = defaultdict(lambda: {"games": 0, "wins": 0})
    for i, puuid in enumerate(puuids, 1):
        try:
            ids = client.match_ids(puuid, args.matches_per_player,
                                   start_time=start_time, end_time=end_time)
        except requests.RequestException as exc:
            print(f"  [{i}/{len(puuids)}] match_ids failed, skipping player: {exc}")
            continue
        for mid in ids:
            if mid in seen_matches:
                continue
            seen_matches.add(mid)
            try:
                match = client.match(mid)
            except requests.RequestException:
                continue  # skip this one match, keep going
            info = match.get("info", {})
            mpatch = patch_of(info.get("gameVersion", ""))
            if target_patches is not None and mpatch not in target_patches:
                continue
            parts = info.get("participants", [])
            # (teamId, position) -> champion, so we can find each player's lane opponent.
            by_slot = {(p.get("teamId"), p.get("teamPosition", "")): p["championName"]
                       for p in parts if p.get("teamPosition")}
            for p in parts:
                role = ROLE_MAP.get(p.get("teamPosition", ""), None)
                if role is None:
                    continue
                win = int(p["win"])
                key = (mpatch, p["championName"], role)  # bucket by patch too
                stats[key]["games"] += 1
                stats[key]["wins"] += win
                # NEW: record the lane matchup (enemy in the same position).
                opp = by_slot.get((200 if p.get("teamId") == 100 else 100,
                                   p.get("teamPosition", "")))
                if opp:
                    mkey = (mpatch, p["championName"], role, opp)
                    matchup_stats[mkey]["games"] += 1
                    matchup_stats[mkey]["wins"] += win
        print(f"  [{i}/{len(puuids)}] processed; {len(seen_matches)} matches, "
              f"{sum(s['games'] for s in stats.values())} champ-games kept")

        # Checkpoint: save progress so far every N players.
        if args.checkpoint_every and i % args.checkpoint_every == 0:
            res = write_winrates(stats, tag)
            write_matchups(matchup_stats, tag)
            save_state(stats, seen_matches, tag, matchup_stats)
            if res:
                print(f"      [checkpoint] saved {len(res[1])} rows + "
                      f"{len(matchup_stats)} matchup cells")

    save_state(stats, seen_matches, tag, matchup_stats)
    mres = write_matchups(matchup_stats, tag)
    res = write_winrates(stats, tag)
    if res is None:
        print("No matches captured. Try more players / matches, or a more recent run.")
        return
    out, df = res
    print(f"\nWrote {len(df)} champion-role win-rates -> {out}")
    if mres:
        print(f"Wrote {len(mres[1])} lane-matchup rows -> {mres[0]}")
    print("Patches captured (patch -> total games):",
          df.groupby("patch")["games"].sum().to_dict())
    print("Model-ready: run  python src/model.py --new <newer> --old <older> --min-games <N>")


if __name__ == "__main__":
    main()
