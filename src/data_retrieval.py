import nfl_data_py as nfl
import pandas as pd

# Import schedule data for 2015-2025
years = list(range(2015, 2026))
schedules = nfl.import_schedules(years)

print(f"Shape: {schedules.shape}")
print(f"Seasons: {sorted(schedules['season'].unique())}")

# Save to raw data folder
schedules.to_csv('/Users/anthonyjaramillo/PycharmProjects/4900/data/raw/schedules_2015_2025.csv', index=False)
print("Saved to data/raw/schedules_2015_2025.csv")

