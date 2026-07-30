"""
Hand-written domain notes on the model's biggest misses.

The model knows only THAT a champion was announced as changed and how many lines it got.
It cannot know whether a buff mattered in context — the recurring "Irelia problem": a
numerically large damage buff to a build nobody plays reads as a big change to the model and
as nothing at all to a player. This file is where that missing context lives.

These notes are the one input to the project that could not be derived from the data. They
come from playing the game, not from the pipeline, and they are the honest answer to "why is
your model wrong here?" — a specific reason per case rather than a shrug.

Keyed by (patch, champion_id, role). Champion ids are Data Dragon ids (Kaisa, Leblanc,
KSante, MonkeyKing), which is what the panel joins on. Role is TOP/JUNGLE/MID/BOT/SUPPORT.

Leave an entry as "" and nothing renders for it — partial coverage is fine.

Writing guidance, so these read as analysis rather than excuses:
  - Say what the model could not see, not that it was wrong.
  - One or two sentences. This is a tooltip, not an essay.
  - Prefer a mechanism ("the buff pushed it into a different role") over a judgement
    ("this buff was bad").
  - If a miss is genuinely unexplained, write that. "I don't know why this moved" is a
    legitimate and more credible entry than an invented reason.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------------------
# The model's largest errors, worst first per patch. Fill in the strings.
#
# The role-split pairs below are the most interesting cases and the data already tells you
# something: the SAME buff sent one role up and the other down. The model treats every
# champion-role as independent, so it cannot represent a champion MOVING between roles —
# pick share didn't just grow, it migrated. If that's what happened, say so.
# ---------------------------------------------------------------------------------------

NOTES: dict[tuple[str, str, str], str] = {
    # --- 16.13 -------------------------------------------------------------------------
    ("16.13", "Qiyana", "JUNGLE"): "",   # buff x1: predicted +31%, actual +93%
    ("16.13", "Qiyana", "MID"): "This buff was only to jungle, people started playing her in jungle.",      # buff x1: predicted +26%, actual -27%   <- pair
    ("16.13", "Senna", "SUPPORT"): "They nerfed her adc role, not her support.",   # nerf x4: predicted -3%,  actual +48%
    ("16.13", "Leblanc", "MID"): "Three buffs, but they can't be used together, they buff each of one mode of a spell.",     # buff x3: predicted +39%, actual +4%
    ("16.13", "Draven", "BOT"): "",      # buff x2: predicted +21%, actual -12%

    # --- 16.12 -------------------------------------------------------------------------
    ("16.12", "Sylas", "JUNGLE"): "",    # buff x2: predicted +27%, actual +101%
    ("16.12", "Sylas", "TOP"): "Very unpopular role for the champion, numbers move easily.",       # buff x2: predicted +35%, actual -36%   <- pair
    ("16.12", "Syndra", "MID"): "",      # buff x2: predicted +18%, actual +89%
    ("16.12", "Hwei", "BOT"): "",        # buff x4: predicted +46%, actual -25%
    ("16.12", "Gwen", "TOP"): "",        # buff x2: predicted +30%, actual -20%

    # --- 16.11 -------------------------------------------------------------------------
    ("16.11", "Diana", "JUNGLE"): "She got a new very expensive skin line in this patch.",    # buff x2: predicted +25%, actual +196%
    ("16.11", "Smolder", "BOT"): "I think these changes are just incredibly hard to parse and read as nerfs.",     # mixed x1: predicted +2%, actual -43%
    ("16.11", "Quinn", "JUNGLE"): "",    # buff x2: predicted +33%, actual +74%
    ("16.11", "Ekko", "JUNGLE"): "",     # buff x1: predicted +24%, actual +62%
    ("16.11", "Heimerdinger", "TOP"): "This is a very strange champion who will only see play from a small subset of people in the world.",  # buff x3: predicted +44%, actual +9%
}


def note(patch: str, champion: str, role: str) -> str:
    """The hand-written note for one champion-role-patch, or '' if there isn't one."""
    return NOTES.get((str(patch), str(champion), str(role)), "")


def coverage() -> tuple[int, int]:
    """(how many notes are written, how many slots exist) — for the dashboard caption."""
    return sum(1 for v in NOTES.values() if v.strip()), len(NOTES)
