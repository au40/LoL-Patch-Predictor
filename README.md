# LoL Patch Predictor

Predict how League of Legends champion patch changes affect win-rate, using two
years of Riot data plus LLM-parsed patch notes.

Readme might be slightly out of date at any given moment.

## Setup

I've only tested on windows, but this is the setup:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Keys go in `.env` (never committed):
- `RIOT_API_KEY` — https://developer.riotgames.com/ (dev keys expire every 24h)
- `ANTHROPIC_API_KEY` — https://console.anthropic.com/

## Run each piece

**1. Data ingestion — Data Dragon (no key needed):**
```bash
python src/datadragon.py --list 6                 # recent patch versions
python src/datadragon.py --diff 16.13.1 16.12.1   # real base-stat changes Riot made
```
Data Dragon gives champion base stats per patch. Diffing two patches yields the
base-stat changes — this is best computation for champion base stats. If you need ability changes, you extract the patch notes with the LLM.

**2. LLM patch-note extraction (needs Anthropic key):**
```bash
python src/llm_extract.py --file data/raw/patch_notes/sample_16_13.txt --patch 16.13
```
Completely untested by me currently. Currently just working through LLM directly, but will test this before final submission.

**3. Validate the extraction against ground truth (bonus, needs Riot version data):**
```bash
python src/validate_extraction.py --extracted data/processed/extracted_16.13.json \
    --new 16.13.1 --old 16.12.1
```
Judges LLM extraction to data_dragon base stat changes.

**4. Preliminary model (works now on sample data):**
```bash
python src/model.py --new 16.13 --old 16.12
```
This is just one of the models - you can replace this with any of the other model (magnitude, magnitude_by_type, etc - these are just for viewing different views on the data though. nothing crazy at the moment.)

**5. Real Riot win-rate ingestion (needs Riot key):**
```bash
python src/riot_ingest.py --patch 16.13 --tier DIAMOND --division I --pages 1 --matches-per-player 15
```
Writes `data/raw/winrates/riot_16.13.csv` in the same schema the model reads, so real
data drops straight into step 4. There is data included in the github though, so not necessary to run this and have a Riot API key.

## The base model as of now (per Checkpoint-1 feedback) 

For a champion `c` changed in patch `P`:
- **Diff-in-differences:** `effect_c = [WR(c,P) - WR(c,P-1)] - baseline`, where `baseline`
  is the median win-rate shift of **untouched** champions (controls for meta drift).
- **Regression upgrade:** `delta ~ changed? + prior_winrate + role` (weighted by games).
  `prior_winrate` controls for regression-to-the-mean (Riot buffs weak champs / nerfs strong ones).
- **Backtest:** leave-one-out prediction, reported as MAE vs. a predict-zero baseline.

This will all quickly be outdated.

## Data layer note

`src/winrates.py` defines ONE win-rate schema (`patch, champion, role, games, winrate`).
The model only knows that schema — so third-party aggregates today and raw Riot data
later are interchangeable without touching the model.

## Layout
```
config.py                     # loads .env, defines data paths
src/datadragon.py             # Data Dragon ingestion
src/riot_ingest.py            # Riot match ingestion -> winrate CSV (Riot key needed)
src/llm_extract.py            # patch notes -> structured changes (Anthropic API needed) (untested!)
src/validate_extraction.py    # grade LLM extraction vs Data Dragon
src/winrates.py               # win-rate data layer (the swappable seam)
src/model.py                  # diff-in-diff + regression + backtest - not final model at all
src/magnitude                 # the preliminary model to the by_type version
src/magnitude_by_type         # models specific changes by magnitude, another modeling try
data/raw/                     # API caches, patch notes, winrate CSVs
data/processed/               # diffs, extractions, model outputs
```
