import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load rental dataset
df = pd.read_csv("bengaluru_rent.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("Columns found:", list(df.columns))

# Rename columns
df = df.rename(columns={
    "Locality": "location",
    "AvgRent": "rent",
    "HouseType": "house_type"
})

# Keep required columns
df = df[["location", "house_type", "rent"]]

# Extract BHK from HouseType
# Examples: "1 BHK", "2 BHK", "3 BHK"
df["bhk"] = df["house_type"].astype(str).str.extract(r"(\d+)")
df["bhk"] = pd.to_numeric(df["bhk"], errors="coerce")

# Remove rows with missing values
df = df.dropna()

# Clean location names
df["location"] = df["location"].astype(str).str.strip()

# Convert rent to numeric
df["rent"] = pd.to_numeric(df["rent"], errors="coerce")
df = df.dropna()

# Add default values because dataset lacks these columns
# These are placeholders to keep compatibility with your Flask app
df["total_sqft"] = df["bhk"] * 500   # Approximate area per bedroom
df["bath"] = df["bhk"]               # Assume bathrooms ≈ bedrooms

# Keep locations with at least 2 records
location_counts = df["location"].value_counts()
df["location"] = df["location"].apply(
    lambda x: x if location_counts[x] >= 2 else "other"
)

# One-hot encode location
df = pd.get_dummies(df, columns=["location"])

# Features and target
X = df.drop(["rent", "house_type"], axis=1)
y = df["rent"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and columns
with open("rent_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("rent_columns.pkl", "wb") as f:
    pickle.dump(X.columns, f)

print("\nRental model trained successfully!")
print("Files created:")
print(" - rent_model.pkl")
print(" - rent_columns.pkl")
print(f"Training samples: {len(df)}")