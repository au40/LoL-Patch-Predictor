# LoL Patch Predictor

Pipeline for measuring what League of Legends patch notes do to a champion. Ingests two years
of Riot match data, extracts patch-note changes with an LLM, and models the effect on champion
winrate and pick rate.

This readme is to help run the program, and actual explanations of findings live in checkpoints/the app.

## Quick start

Data is committed, so nothing here needs an API key.

```bash
.venv/Scripts/streamlit.exe run src/dashboard.py
```
Opens the webapp.


```bash
.venv/Scripts/python.exe src/results_summary.py --save
```
Full report of numbers from the project.


## Setup

Only tested on Windows.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Keys go in `.env` (never committed) and are needed **only** to ingest new data:
- `RIOT_API_KEY` — https://developer.riotgames.com/
- `ANTHROPIC_API_KEY` — https://console.anthropic.com/

## Architecture

Two independent sources come in, meet on a common key, and feed the models.

```
  RIOT MATCH API                          LEAGUE PATCH NOTES
        |                                         |
  riot_ingest.py                            llm_extract.py
   ladder -> players                     notes -> structured changes
   -> matches -> aggregate                (champion, target, field,
        |                                  change_type, old, new)
        v                                         |
  data/raw/winrates/*.csv                         v
  patch,champion,role,games,winrate      data/processed/extracted_*.json
        |                                         |
        |            DATA DRAGON                  |
        |         datadragon.py                   |
        |     base stats + per-rank spell         |
        |     fields, diffed between patches      |
        |                  |                      |
        |                  +--> validate_*.py <---+   (grades the extraction
        |                                             against ground truth)
        v                                         v
   winrates.py  ------------ join on --------> (champion, role, patch)
                                 |
                                 v
                    +------------+------------+
                    |                         |
              pickrate.py               noise_floor.py
          pick share as outcome     shows winrate is unpredictable
                    |                         |
                    +------------+------------+
                                 |
                    results_summary.py / dashboard.py
```

**Three sources** Riot match API = Player data; Data Dragon = actual game stats; LLM Extraction = grabbing what data dragon can't

**Winrate data all goes through winrates.py** `winrates.py` defines `patch, champion, role, games, winrate` and
the models know only that. Where the numbers came from can change without touching any
analysis.


### Data scope

Diamond I–IV, NA, ranked solo queue. Winrate data spans patches 14.18 → 16.13; 39 patches have
extracted notes. `results_summary.py` section 9 reports exactly which patch steps are usable and
which aren't.

### Design decisions worth knowing before you edit

**The panel is built on consecutive patch pairs only** Nothing compares between patch 15.4 -> 15.6, for example (if 15.5 has insufficient data) - just skip that boundary.

**Patches below 5,000 total sampled games are dropped** 

**Effect sizes and predictions use different specifications.** `pickrate.FORMULA` includes
`net_buff` (the count of buff/nerf lines) because it helps prediction; `FORMULA_EFFECTS` drops it
because with the count in the model the `buffed` coefficient stops meaning "the effect of being
announced as buffed."

**Statsmodels results objects are cached with `@st.cache_resource`, not `@st.cache_data`** in the
dashboard — they're live objects, not serialisable values.

## Running each piece

### Analysis (no keys needed)

```bash
.venv/Scripts/python.exe src/pickrate.py --min-games 20
```
Builds the panel, reports buff/nerf effect sizes with standard errors clustered by champion, and
runs a walk-forward backtest (train on every patch before *t*, predict *t*). Also exposes
`persistence()`, `scoreboard()`, `forecast_path()` and `role_splits()` as importable functions.

```bash
.venv/Scripts/python.exe src/noise_floor.py
```
Variance decomposition of winrate movement, reliability, lag-1 autocorrelation, the theoretical
MAE floor, and a bake-off of simple predictors against it. Runs at min_games 20 / 100 / 200.

```bash
.venv/Scripts/python.exe src/magnitude_by_type.py --min-games 20
```
The retired winrate models. Also `model.py` (diff-in-differences + regression), `predict.py`
(multi-patch temporal), `magnitude.py`, `damage_to_cooldown.py`.

### Validation (no keys needed)

```bash
.venv/Scripts/python.exe src/validate_direction.py --show-disagreements
```
Grades the LLM's buff/nerf label against Data Dragon, reporting base-stat and ability changes
separately. Data Dragon publishes per-patch base stats and per-rank cooldown / cost / range, so
the true direction is the sign of the change. Damage values and scaling ratios aren't published,
so those changes can't be graded.

```bash
.venv/Scripts/python.exe src/validate_extraction.py --extracted data/processed/extracted_16.13.json --new 16.13.1 --old 16.12.1 --notes data/raw/patch_notes/26_13.txt
```
Whether the extraction found the right base-stat changes. Reports precision and ground-truth
coverage separately, and flags real changes missing from the notes.

### Data Dragon (no key needed)

```bash
.venv/Scripts/python.exe src/datadragon.py --diff 16.13.1 16.12.1
```
Also `--list N`, `--diff-latest`, `--cooldowns [VERSION]`. Everything is cached under
`data/raw/`, so re-runs are free. `champion_spells()` / `diff_spells()` pull per-rank ability
fields; fetching a new version costs ~170 CDN requests and takes about a minute.

### Ingestion (needs keys)

```bash
.venv/Scripts/python.exe src/riot_ingest.py --resume --tier DIAMOND --divisions I II III IV --patches 15.4 --max-players 400 --matches-per-player 100 --checkpoint-every 10
```

```bash
.venv/Scripts/python.exe src/llm_extract.py --file data/raw/patch_notes/26_13.txt --patch 16.13
```

## Layout

```
config.py                       # loads .env, defines data paths

# ingestion
src/riot_ingest.py              # ladder -> matches -> winrate + matchup CSVs (Riot key)
src/datadragon.py               # patch versions, base stats, per-rank spell fields, diffs
src/llm_extract.py              # patch notes -> structured changes (Anthropic key)
src/skill_order.py              # which spell each champion maxes first
src/winrates.py                 # the one winrate schema everything reads

# analysis
src/pickrate.py                 # pick-share model: panel, effects, backtest, persistence,
                                #   scoreboard, forecast_path, role_splits
src/noise_floor.py              # winrate variance decomposition and MAE floor
src/model.py                    # retired: diff-in-diff + regression on winrate
src/predict.py                  # retired: multi-patch temporal winrate predictor
src/magnitude.py                # retired: change magnitude
src/magnitude_by_type.py        # retired: magnitude split by stat type
src/damage_to_cooldown.py       # retired: damage weighted by ability cooldown

# validation and output
src/validate_direction.py       # grades the buff/nerf label vs Data Dragon
src/validate_extraction.py      # grades base-stat extraction vs Data Dragon
src/results_summary.py          # every reported number, in one command
src/annotations.py              # hand-written domain notes on the model's misses
src/dashboard.py                # Streamlit app (5 tabs)

data/raw/                       # API caches, patch notes, winrate + matchup CSVs
data/processed/                 # DDragon diffs, LLM extractions, results_summary.txt
```
