import pandas as pd
import os

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR       = '/Users/anthonyjaramillo/PycharmProjects/4900/data/raw'
PROCESSED_DIR = '/Users/anthonyjaramillo/PycharmProjects/4900/data/processed'
SCHEDULE_PATH = os.path.join(RAW_DIR, 'schedules_2015_2025.csv')

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── 1. Load & filter schedules ────────────────────────────────────────────────
print("Loading schedules...")
schedules = pd.read_csv(SCHEDULE_PATH)
schedules = schedules[schedules['game_type'] == 'REG']
schedules = schedules[[
    'game_id', 'season', 'home_team', 'away_team',
    'spread_line', 'home_spread_odds', 'away_spread_odds'
]]
reg_game_ids = set(schedules['game_id'])
print(f"  REG season games: {len(reg_game_ids)}")

# ── 2. Load all PBP parquet files ─────────────────────────────────────────────
print("Loading PBP data...")
pbp_frames = []
for year in range(2015, 2026):
    path = os.path.join(RAW_DIR, f'pbp_{year}.parquet')
    if os.path.exists(path):
        df = pd.read_parquet(path, columns=[
            'game_id', 'posteam', 'defteam',
            'pass', 'rush', 'yards_gained',
            'fixed_drive', 'fixed_drive_result'
        ])
        pbp_frames.append(df)
        print(f"  Loaded {year}")
    else:
        print(f"  Skipping {year} — file not found")

pbp = pd.concat(pbp_frames, ignore_index=True)

# Filter to REG season only
pbp = pbp[pbp['game_id'].isin(reg_game_ids)]
print(f"  PBP rows after REG filter: {len(pbp):,}")

# ── 3. Passing yards (offense) ────────────────────────────────────────────────
pass_off = (
    pbp[pbp['pass'] == 1]
    .groupby(['game_id', 'posteam'])['yards_gained']
    .sum()
    .reset_index()
    .rename(columns={'posteam': 'team', 'yards_gained': 'pass_yds_off'})
)

# ── 4. Passing yards (defense) ────────────────────────────────────────────────
pass_def = (
    pbp[pbp['pass'] == 1]
    .groupby(['game_id', 'defteam'])['yards_gained']
    .sum()
    .reset_index()
    .rename(columns={'defteam': 'team', 'yards_gained': 'pass_yds_def'})
)

# ── 5. Rush yards (offense) ───────────────────────────────────────────────────
rush_off = (
    pbp[pbp['rush'] == 1]
    .groupby(['game_id', 'posteam'])['yards_gained']
    .sum()
    .reset_index()
    .rename(columns={'posteam': 'team', 'yards_gained': 'rush_yds_off'})
)

# ── 6. Rush yards (defense) ───────────────────────────────────────────────────
rush_def = (
    pbp[pbp['rush'] == 1]
    .groupby(['game_id', 'defteam'])['yards_gained']
    .sum()
    .reset_index()
    .rename(columns={'defteam': 'team', 'yards_gained': 'rush_yds_def'})
)

# ── 7. Drive scoring ──────────────────────────────────────────────────────────
# One row per drive (dedup by game_id + posteam + fixed_drive)
drives = (
    pbp
    .dropna(subset=['fixed_drive', 'posteam', 'defteam'])
    .drop_duplicates(subset=['game_id', 'posteam', 'fixed_drive'])
)

SCORING = ['Touchdown', 'Field goal']

# Scoring drives — offense perspective
drives_scored_off = (
    drives[drives['fixed_drive_result'].isin(SCORING)]
    .groupby(['game_id', 'posteam'])
    .size()
    .reset_index(name='drives_scored_off')
    .rename(columns={'posteam': 'team'})
)

# Total drives — offense (used to compute ratio later if needed)
total_drives_off = (
    drives
    .groupby(['game_id', 'posteam'])
    .size()
    .reset_index(name='total_drives_off')
    .rename(columns={'posteam': 'team'})
)

# Scoring drives allowed — defense perspective
drives_scored_def = (
    drives[drives['fixed_drive_result'].isin(SCORING)]
    .groupby(['game_id', 'defteam'])
    .size()
    .reset_index(name='drives_scored_def')
    .rename(columns={'defteam': 'team'})
)

# Total drives faced — defense
total_drives_def = (
    drives
    .groupby(['game_id', 'defteam'])
    .size()
    .reset_index(name='total_drives_def')
    .rename(columns={'defteam': 'team'})
)

# ── 8. Merge all stats into one table ─────────────────────────────────────────
print("Merging stats...")
game_stats = pass_off.copy()

for df in [pass_def, rush_off, rush_def,
           drives_scored_off, total_drives_off,
           drives_scored_def, total_drives_def]:
    game_stats = game_stats.merge(df, on=['game_id', 'team'], how='left')

# Fill 0 for teams that had no scoring drives
game_stats[[
    'drives_scored_off', 'total_drives_off',
    'drives_scored_def', 'total_drives_def'
]] = game_stats[[
    'drives_scored_off', 'total_drives_off',
    'drives_scored_def', 'total_drives_def'
]].fillna(0)

# ── 9. Add season from schedule ───────────────────────────────────────────────
game_stats = game_stats.merge(
    schedules[['game_id', 'season']], on='game_id', how='left'
)

game_stats = game_stats.sort_values(
    ['season', 'game_id', 'team']
).reset_index(drop=True)

# ── 10. Save ──────────────────────────────────────────────────────────────────
out_path = os.path.join(PROCESSED_DIR, 'game_stats_raw.csv')
game_stats.to_csv(out_path, index=False)

print(f"\nDone! Shape: {game_stats.shape}")
print(f"Saved to {out_path}")
print(game_stats.head(10))
