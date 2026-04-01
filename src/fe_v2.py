import pandas as pd

# ── Load feature set 1 (rolling averages) ───────────────────────────────────
model_features = pd.read_csv('../data/processed/model_features.csv')

# ── Load new context features ────────────────────────────────────────────────
context_features = pd.read_csv('../data/processed/game_context_features.csv')

# ── Drop season/week from context since model_features already has them ──────
context_features = context_features.drop(columns=['season', 'week'])

# ── Merge on game_id ─────────────────────────────────────────────────────────
model_features_v2 = model_features.merge(context_features, on='game_id', how='left')

# ── Check for any nulls after merge ─────────────────────────────────────────
null_counts = model_features_v2.isnull().sum()
null_counts = null_counts[null_counts > 0]
if len(null_counts) > 0:
    print("Columns with nulls after merge:")
    print(null_counts)
else:
    print("No nulls found.")

print(f"\nFinal shape: {model_features_v2.shape}")
print(f"Columns: {model_features_v2.columns.tolist()}")

model_features_v2.to_csv('../data/processed/model_features_v2.csv', index=False)
print(f"\nSaved {len(model_features_v2)} games to model_features_v2.csv")
