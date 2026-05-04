from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model + columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Extract locations from columns
locations = sorted([
    col.replace("location_", "")
    for col in columns
    if col.startswith("location_")
])

# ---------------- VALIDATION ----------------
def validate_inputs(sqft, bhk, bath):
    if sqft < 200:
        return False, "Area too small (min 200 sq ft)."
    if sqft > 50000:
        return False, "Area too large."
    if bhk < 1 or bhk > 10:
        return False, "BHK must be between 1–10."
    if bath < 1 or bath > 10:
        return False, "Bathrooms must be between 1–10."
    return True, ""

# ---------------- CONFIDENCE (SIMPLIFIED) ----------------
def compute_confidence(price, sqft):
    pps = (price * 1e5) / sqft

    # simple heuristic confidence
    if 3000 < pps < 10000:
        return 85
    elif 2000 < pps < 12000:
        return 70
    else:
        return 55

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template("index.html", locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sqft = float(request.form['sqft'])
        bath = float(request.form['bath'])
        bhk = float(request.form['bhk'])
        location = request.form['location'].strip()

        # Validation
        ok, msg = validate_inputs(sqft, bhk, bath)
        if not ok:
            return render_template("index.html", locations=locations, error=msg)

        # Build input
        input_data = pd.DataFrame([{c: 0 for c in columns}])
        input_data['total_sqft'] = sqft
        input_data['bath'] = bath
        input_data['bhk'] = bhk

        loc_col = "location_" + location
        if loc_col in input_data.columns:
            input_data[loc_col] = 1

        # Prediction
        prediction = model.predict(input_data)[0]
        price = round(float(prediction), 2)

        if price <= 0:
            return render_template("index.html", locations=locations,
                                   error="Invalid prediction for given inputs.")

        # Extra outputs
        low = round(price * 0.9, 2)
        high = round(price * 1.1, 2)
        pps = round((price * 1e5) / sqft, 2)

        if price < 50:
            category = "Affordable"
        elif price < 150:
            category = "Mid-range"
        else:
            category = "Premium"

        confidence = compute_confidence(price, sqft)

        return render_template("index.html",
                               locations=locations,
                               price=price,
                               low=low,
                               high=high,
                               pps=pps,
                               category=category,
                               confidence=confidence)

    except Exception as e:
        return render_template("index.html",
                               locations=locations,
                               error=f"Error: {str(e)}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)