import pandas as pd
import numpy as np

RAW_DIR       = '/Users/anthonyjaramillo/PycharmProjects/4900/data/raw'
PROCESSED_DIR = '/Users/anthonyjaramillo/PycharmProjects/4900/data/processed'

# Load files
game_stats = pd.read_csv(f'{PROCESSED_DIR}/game_stats_raw.csv')
schedules  = pd.read_csv(f'{RAW_DIR}/schedules_2015_2025.csv')
schedules  = schedules[schedules['game_type'] == 'REG']

# Bring over the schedule columns we need
sched_cols = schedules[[
    'game_id', 'home_team', 'away_team',
    'home_score', 'away_score',
    'spread_line', 'home_spread_odds', 'away_spread_odds'
]]

game_stats = game_stats.merge(sched_cols, on='game_id', how='left')

# Derive points_scored and points_conceded based on home/away
game_stats['points_scored']   = np.where(
    game_stats['team'] == game_stats['home_team'],
    game_stats['home_score'],
    game_stats['away_score']
)
game_stats['points_conceded'] = np.where(
    game_stats['team'] == game_stats['home_team'],
    game_stats['away_score'],
    game_stats['home_score']
)

# Drop redundant columns
game_stats = game_stats.drop(columns=['home_score', 'away_score'])

# Save back
game_stats.to_csv(f'{PROCESSED_DIR}/game_stats_raw.csv', index=False)
print(f"Done! Shape: {game_stats.shape}")
print(game_stats.head())
