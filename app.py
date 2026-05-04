from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Extract location names from columns for the dropdown
locations = sorted([
    col.replace("location_", "")
    for col in columns
    if col.startswith("location_")
])


@app.route('/')
def home():
    return render_template("index.html", locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        sqft = float(request.form['sqft'])
        bath = float(request.form['bath'])
        bhk = float(request.form['bhk'])
        location = request.form['location']

        input_data = pd.DataFrame(columns=columns)
        input_data.loc[0] = 0

        input_data['total_sqft'] = sqft
        input_data['bath'] = bath
        input_data['bhk'] = bhk

        loc_col = "location_" + location
        if loc_col in input_data.columns:
            input_data[loc_col] = 1

        prediction = model.predict(input_data)[0]
        price = round(prediction, 2)

        low = round(price * 0.9, 2)
        high = round(price * 1.1, 2)
        pps = round((price * 100000) / sqft, 2)

        if price < 50:
            category = "Affordable"
        elif price < 150:
            category = "Mid-range"
        else:
            category = "Premium"

        confidence = 85

        return render_template("index.html",
                               locations=locations,
                               price=price,
                               low=low,
                               high=high,
                               pps=pps,
                               category=category,
                               confidence=confidence)

    except Exception as e:
        return render_template("index.html", locations=locations, error=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)