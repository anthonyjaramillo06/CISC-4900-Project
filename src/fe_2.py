import pandas as pd

RAW_DIR       = '/Users/anthonyjaramillo/PycharmProjects/4900/data/raw'
PROCESSED_DIR = '/Users/anthonyjaramillo/PycharmProjects/4900/data/processed'

print("Loading files...")
game_stats = pd.read_csv(f'{PROCESSED_DIR}/game_stats_raw.csv')
schedules  = pd.read_csv(f'{RAW_DIR}/schedules_2015_2025.csv')
schedules  = schedules[schedules['game_type'] == 'REG']

# Normalize relocated team names to match PBP naming
TEAM_MAP = {'OAK': 'LV', 'STL': 'LA', 'SD': 'LAC'}
game_stats['home_team'] = game_stats['home_team'].replace(TEAM_MAP)
game_stats['away_team'] = game_stats['away_team'].replace(TEAM_MAP)


STAT_COLS = [
    'points_scored',    'points_conceded',
    'pass_yds_off',     'pass_yds_def',
    'rush_yds_off',     'rush_yds_def',
    'drives_scored_off','drives_scored_def'
]

print("Computing rolling averages...")
for col in STAT_COLS:
    game_stats[f'avg_{col}'] = (
        game_stats.groupby(['team', 'season'])[col]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

# Drop first game of each season
avg_cols   = [f'avg_{col}' for col in STAT_COLS]
game_stats = game_stats.dropna(subset=avg_cols)
print(f"  Rows after dropping week 1 NaNs: {len(game_stats):,}")

# ── Pivot to one row per game ─────────────────────────────────────────────────
print("Pivoting to game-level feature matrix...")

home_df = game_stats[game_stats['team'] == game_stats['home_team']][
    ['game_id'] + avg_cols
].rename(columns={col: f'home_{col}' for col in avg_cols})

away_df = game_stats[game_stats['team'] == game_stats['away_team']][
    ['game_id'] + avg_cols
].rename(columns={col: f'away_{col}' for col in avg_cols})

model_df = home_df.merge(away_df, on='game_id', how='inner')

# ── Add context + target variable ─────────────────────────────────────────────
TEAM_MAP_SCHED = {'OAK': 'LV', 'STL': 'LA', 'SD': 'LAC'}
schedules['home_team'] = schedules['home_team'].replace(TEAM_MAP_SCHED)
schedules['away_team'] = schedules['away_team'].replace(TEAM_MAP_SCHED)
schedules['covered']   = ((schedules['result'] + schedules['spread_line']) > 0).astype(int)

model_df = model_df.merge(
    schedules[[
        'game_id', 'season', 'week', 'gameday',
        'home_team', 'away_team',
        'spread_line', 'covered'
    ]],
    on='game_id', how='inner'
)

model_df = model_df.dropna(subset=['spread_line', 'covered'])

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = f'{PROCESSED_DIR}/model_features.csv'
model_df = model_df.round(2)
model_df.to_csv(out_path, index=False)

print(f"\nDone! Shape: {model_df.shape}")
print(f"Cover rate: {model_df['covered'].mean():.3f}  ← should be ~0.50")
print(f"Seasons: {sorted(model_df['season'].unique())}")
print(f"\n{model_df.head()}")
