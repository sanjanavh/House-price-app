import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import json

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("bengaluru_house_prices.csv")

# ── 2. Clean ─────────────────────────────────────────────────────────────────
df['location'] = df['location'].apply(lambda x: str(x).strip().title())
df['bhk'] = df['size'].str.extract(r'(\d+)').astype(float)

df = df[['location', 'total_sqft', 'bath', 'bhk', 'price']].dropna()

def convert_sqft(x):
    x = str(x).strip()
    if '-' in x:
        parts = x.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return None
    try:
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna()

# ── 3. Feature engineering ───────────────────────────────────────────────────
df['price_per_sqft'] = (df['price'] * 1e5) / df['total_sqft']
df['bath_per_bhk']   = df['bath'] / df['bhk']
df['sqft_per_bhk']   = df['total_sqft'] / df['bhk']

# ── 4. Outlier removal ───────────────────────────────────────────────────────
# Remove impossible room combos
df = df[df['bath'] < df['bhk'] + 2]
df = df[df['bhk'] <= 10]

# Remove per-sqft price outliers per location
def remove_pps_outliers(df):
    out = []
    for loc, sub in df.groupby('location'):
        m, s = sub['price_per_sqft'].mean(), sub['price_per_sqft'].std()
        kept = sub[(sub['price_per_sqft'] > m - 2*s) & (sub['price_per_sqft'] < m + 2*s)]
        out.append(kept)
    return pd.concat(out, ignore_index=True)

df = remove_pps_outliers(df)

# ── 5. Location stats (saved for confidence scoring) ─────────────────────────
loc_stats = (
    df.groupby('location')['price_per_sqft']
    .agg(['mean', 'std', 'count'])
    .rename(columns={'mean': 'pps_mean', 'std': 'pps_std', 'count': 'n'})
)
loc_stats['pps_std'] = loc_stats['pps_std'].fillna(loc_stats['pps_mean'] * 0.15)
loc_stats_dict = loc_stats.to_dict(orient='index')
with open("location_stats.json", "w") as f:
    json.dump(loc_stats_dict, f)
print(f"Saved location stats for {len(loc_stats_dict)} locations.")

# ── 6. Encode locations ───────────────────────────────────────────────────────
loc_counts = df['location'].value_counts()
df['location'] = df['location'].apply(
    lambda x: x if loc_counts[x] > 10 else 'Other'
)
df = pd.get_dummies(df, columns=['location'])

# Drop helper columns before training
df = df.drop(columns=['price_per_sqft', 'bath_per_bhk', 'sqft_per_bhk'], errors='ignore')

X = df.drop('price', axis=1)
y = df['price']

# ── 7. Model selection ────────────────────────────────────────────────────────
candidates = {
    'LinearRegression': LinearRegression(),
    'Lasso':            Lasso(alpha=1.0),
    'Ridge':            Ridge(alpha=1.0),
    'GradientBoosting': GradientBoostingRegressor(
                            n_estimators=200, max_depth=4,
                            learning_rate=0.1, random_state=42),
}

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
best_name, best_score, best_model = None, -np.inf, None
for name, m in candidates.items():
    scores = cross_val_score(m, X, y, cv=cv, scoring='r2')
    mean_r2 = scores.mean()
    print(f"  {name}: R² = {mean_r2:.4f} ± {scores.std():.4f}")
    if mean_r2 > best_score:
        best_score, best_name, best_model = mean_r2, name, m

print(f"\n✔ Best model: {best_name} (R² = {best_score:.4f})")

best_model.fit(X, y)

# ── 8. Save ───────────────────────────────────────────────────────────────────
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))
print("model.pkl and columns.pkl saved.")