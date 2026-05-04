from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)

# ── Load artifacts ────────────────────────────────────────────────────────────
model   = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

with open("location_stats.json") as f:
    location_stats = json.load(f)

locations = sorted([
    col.replace("location_", "")
    for col in columns
    if col.startswith("location_")
])

# ── Helpers ───────────────────────────────────────────────────────────────────
def validate_inputs(sqft, bhk, bath):
    """Return (ok: bool, message: str)."""
    if sqft < 200:
        return False, "Area seems too small. Please enter at least 200 sq ft."
    if sqft > 50_000:
        return False, "Area seems unrealistically large. Please check your input."
    if bhk < 1 or bhk > 10:
        return False, "BHK must be between 1 and 10."
    if bath < 1 or bath > 10:
        return False, "Bathrooms must be between 1 and 10."
    if bath > bhk + 2:
        return False, f"A {bhk} BHK unit with {bath} bathrooms is unusual. Please verify."
    if sqft / bhk < 200:
        return False, f"Only {sqft/bhk:.0f} sq ft per BHK seems too low. Please check your inputs."
    return True, ""


def compute_confidence(price, sqft, location):
    """
    Derive a 0–100 confidence score based on:
    - How close the predicted price-per-sqft is to the location average
    - How many training samples that location had
    """
    if location not in location_stats:
        location = "Other"

    stats = location_stats.get(location, {})
    pps_mean = stats.get("pps_mean", 5000)
    pps_std  = stats.get("pps_std",  pps_mean * 0.15)
    n        = stats.get("n", 10)

    pred_pps = (price * 1e5) / sqft

    # How many std-devs away from mean?  0 → highest conf.
    z = abs(pred_pps - pps_mean) / max(pps_std, 1)
    pps_conf = max(0, 1 - z / 3)          # z=0 → 1.0, z≥3 → 0

    # Sample size confidence: saturates at ~200 samples
    n_conf = min(n / 200, 1.0)

    raw = 0.65 * pps_conf + 0.35 * n_conf
    return int(round(raw * 100))


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template("index.html", locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        sqft     = float(request.form['sqft'])
        bath     = float(request.form['bath'])
        bhk      = float(request.form['bhk'])
        location = request.form['location'].strip()

        # ── Validation ──
        ok, msg = validate_inputs(sqft, bhk, bath)
        if not ok:
            return render_template("index.html", locations=locations,
                                   error=msg,
                                   form_sqft=sqft, form_bhk=int(bhk),
                                   form_bath=int(bath), form_location=location)

        # ── Build feature vector ──
        input_data = pd.DataFrame([{c: 0 for c in columns}])
        input_data['total_sqft'] = sqft
        input_data['bath']       = bath
        input_data['bhk']        = bhk

        loc_col = "location_" + location
        if loc_col in input_data.columns:
            input_data[loc_col] = 1

        # ── Predict ──
        raw_price = model.predict(input_data)[0]
        price     = round(float(raw_price), 2)

        if price <= 0:
            return render_template("index.html", locations=locations,
                                   error="Prediction came out negative — "
                                         "the combination of inputs is outside our training range.",
                                   form_sqft=sqft, form_bhk=int(bhk),
                                   form_bath=int(bath), form_location=location)

        low  = round(price * 0.90, 2)
        high = round(price * 1.10, 2)
        pps  = round((price * 1e5) / sqft, 2)

        if price < 50:
            category = "Affordable"
        elif price < 150:
            category = "Mid-range"
        else:
            category = "Premium"

        confidence = compute_confidence(price, sqft, location)

        # Price per sqft vs location average (for the UI insight line)
        loc_info  = location_stats.get(location, location_stats.get("Other", {}))
        avg_pps   = round(loc_info.get("pps_mean", pps))
        pps_delta = round(((pps - avg_pps) / avg_pps) * 100, 1) if avg_pps else 0

        return render_template("index.html",
                               locations=locations,
                               price=price,
                               low=low, high=high,
                               pps=pps,
                               avg_pps=avg_pps,
                               pps_delta=pps_delta,
                               category=category,
                               confidence=confidence,
                               form_sqft=int(sqft),
                               form_bhk=int(bhk),
                               form_bath=int(bath),
                               form_location=location)

    except ValueError:
        return render_template("index.html", locations=locations,
                               error="Please make sure all fields contain valid numbers.")
    except Exception as e:
        return render_template("index.html", locations=locations,
                               error=f"Something went wrong: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)