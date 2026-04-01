import nfl_data_py as nfl

years = list(range(2015, 2024)) + [2025]  # skip 2024, already have it

for year in years:
    print(f"Downloading {year}...")
    pbp = nfl.import_pbp_data([year])
    pbp.to_parquet(f'/Users/anthonyjaramillo/PycharmProjects/4900/data/raw/pbp_{year}.parquet', index=False)
    print(f"Saved pbp_{year}.parquet")

print("All done!")

