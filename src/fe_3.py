import pandas as pd
import nfl_data_py as nfl
import numpy as np

# ── Load schedules ──────────────────────────────────────────────────────────
schedules = pd.read_csv('../data/raw/schedules_2015_2025.csv')
schedules = schedules[schedules['game_type'] == 'REG'].copy()

# ── Primetime indicator ─────────────────────────────────────────────────────
primetime_days = ['Thursday', 'Sunday', 'Monday']
schedules['gametime'] = pd.to_datetime(schedules['gametime'], format='%H:%M', errors='coerce')
schedules['is_primetime'] = (
    schedules['weekday'].isin(primetime_days) &
    (schedules['gametime'].dt.hour >= 19)
).astype(int)

# ── Roof encoding → is_dome ─────────────────────────────────────────────────
dome_values = ['closed', 'retractable']
schedules['is_dome'] = schedules['roof'].str.lower().isin(dome_values).astype(int)

# ── Surface encoding → is_grass ────────────────────────────────────────────
grass_values = ['grass', 'dessograss']
schedules['is_grass'] = schedules['surface'].str.lower().isin(grass_values).astype(int)

# ── Temp and wind — fill nulls with 0 (irrelevant for dome games) ───────────
schedules['temp'] = schedules['temp'].fillna(0)
schedules['wind'] = schedules['wind'].fillna(0)

# ── div_game, home_rest, away_rest — already clean, just keep them ──────────
schedules['div_game'] = schedules['div_game'].fillna(0).astype(int)

# ── Injury features ─────────────────────────────────────────────────────────
injuries = nfl.import_injuries(list(range(2015, 2026)))
injuries = injuries[injuries['game_type'] == 'REG'].copy()

skill_positions = ['QB', 'RB', 'WR', 'TE']
out_doubtful = ['Out', 'Doubtful']

skill_injuries = (
    injuries[
        injuries['position'].isin(skill_positions) &
        injuries['report_status'].isin(out_doubtful)
    ]
    .groupby(['season', 'week', 'team'])
    .size()
    .reset_index(name='skill_injuries')
)

qb_out = (
    injuries[
        (injuries['position'] == 'QB') &
        injuries['report_status'].isin(out_doubtful)
    ]
    .groupby(['season', 'week', 'team'])
    .size()
    .reset_index(name='qb_out')
)
qb_out['qb_out'] = (qb_out['qb_out'] > 0).astype(int)

# ── Merge injury data for home team ────────────────────────────────────────
schedules = schedules.merge(
    skill_injuries.rename(columns={'team': 'home_team', 'skill_injuries': 'home_skill_injuries'}),
    on=['season', 'week', 'home_team'], how='left'
)
schedules = schedules.merge(
    qb_out.rename(columns={'team': 'home_team', 'qb_out': 'home_qb_out'}),
    on=['season', 'week', 'home_team'], how='left'
)

# ── Merge injury data for away team ────────────────────────────────────────
schedules = schedules.merge(
    skill_injuries.rename(columns={'team': 'away_team', 'skill_injuries': 'away_skill_injuries'}),
    on=['season', 'week', 'away_team'], how='left'
)
schedules = schedules.merge(
    qb_out.rename(columns={'team': 'away_team', 'qb_out': 'away_qb_out'}),
    on=['season', 'week', 'away_team'], how='left'
)

# ── Fill missing injury values with 0 (no injuries reported = 0) ────────────
injury_cols = ['home_skill_injuries', 'away_skill_injuries', 'home_qb_out', 'away_qb_out']
schedules[injury_cols] = schedules[injury_cols].fillna(0).astype(int)

# ── Select final columns ────────────────────────────────────────────────────
context_features = schedules[[
    'game_id', 'season', 'week',
    'is_primetime', 'div_game', 'is_dome', 'is_grass',
    'temp', 'wind', 'home_rest', 'away_rest',
    'home_skill_injuries', 'away_skill_injuries',
    'home_qb_out', 'away_qb_out'
]]

context_features.to_csv('../data/processed/game_context_features.csv', index=False)
print(f"Saved {len(context_features)} games to game_context_features.csv")
print(context_features.head())
