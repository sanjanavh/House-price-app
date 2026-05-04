import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("bengaluru_house_prices.csv")

# Clean location
df['location'] = df['location'].apply(lambda x: str(x).strip())

# Extract BHK from size
df['bhk'] = df['size'].str.extract('(\d+)')
df['bhk'] = pd.to_numeric(df['bhk'], errors='coerce')

# Keep needed columns
df = df[['location', 'total_sqft', 'bath', 'bhk', 'price']]

# Drop missing values
df = df.dropna()

# Convert sqft
def convert_sqft(x):
    try:
        if '-' in str(x):
            a, b = x.split('-')
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna()

# Reduce locations
location_stats = df['location'].value_counts()
df['location'] = df['location'].apply(
    lambda x: x if location_stats[x] > 10 else 'other'
)

# One-hot encoding
df = pd.get_dummies(df, columns=['location'])

# Features & target
X = df.drop('price', axis=1)
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model & columns
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("Model trained and saved!")