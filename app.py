from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")  # Required for server-side chart generation
import matplotlib.pyplot as plt

app = Flask(__name__)

# -------------------------------
# Load Purchase Model
# -------------------------------
buy_model = pickle.load(open("model.pkl", "rb"))
buy_columns = pickle.load(open("columns.pkl", "rb"))

# -------------------------------
# Load Rental Model
# -------------------------------
rent_model = pickle.load(open("rent_model.pkl", "rb"))
rent_columns = pickle.load(open("rent_columns.pkl", "rb"))


# -------------------------------
# Extract Locations
# -------------------------------
def extract_locations(columns):
    return {
        col.replace("location_", "")
        for col in columns
        if col.startswith("location_")
    }


locations = sorted(
    list(
        extract_locations(buy_columns).union(
            extract_locations(rent_columns)
        )
    )
)


# -------------------------------
# Helper Function
# -------------------------------
def prepare_input(columns, sqft, bath, bhk, location):
    df = pd.DataFrame(columns=columns)
    df.loc[0] = 0

    # Fill common numeric features only if present
    if "total_sqft" in df.columns:
        df["total_sqft"] = sqft
    if "bath" in df.columns:
        df["bath"] = bath
    if "bhk" in df.columns:
        df["bhk"] = bhk

    # Set location column if it exists
    loc_col = "location_" + location
    if loc_col in df.columns:
        df[loc_col] = 1

    return df


# -------------------------------
# Predict Single Value
# -------------------------------
def predict_price(model, columns, sqft, bath, bhk, location):
    input_data = prepare_input(columns, sqft, bath, bhk, location)
    prediction = model.predict(input_data)[0]
    return round(float(prediction), 2)


# -------------------------------
# Generate Comparison Line Chart
# -------------------------------
def generate_comparison_chart(results, prediction_type):
    localities = [r["location"] for r in results]
    values = [r["value"] for r in results]

    # Color palette for each point
    marker_colors = [
        "#e8c06d",  # gold
        "#4a5eff",  # blue
        "#4fd1c5",  # teal
        "#f87171",  # coral
        "#a78bfa",  # purple
        "#34d399",  # green
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    # Dark theme background
    fig.patch.set_facecolor("#12151f")
    ax.set_facecolor("#12151f")

    # X positions
    x = list(range(len(localities)))

    # Plot the line
    ax.plot(
        x,
        values,
        color="#e8c06d",
        linewidth=3,
        marker=None,
        zorder=1
    )

    # Plot each point with different colors
    for i, value in enumerate(values):
        color = marker_colors[i % len(marker_colors)]

        ax.scatter(
            i,
            value,
            s=120,
            color=color,
            edgecolors="white",
            linewidths=1.5,
            zorder=3
        )

        # Data labels
        if prediction_type == "buy":
            label = f"{value:.1f} L"
        else:
            label = f"₹{value:,.0f}"

        ax.annotate(
            label,
            (i, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#f0f0f0",
            fontweight="bold"
        )

    # IMPORTANT: Start Y-axis at zero to avoid exaggerated slopes
    ymax = max(values) * 1.15
    ax.set_ylim(0, ymax)

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        localities,
        rotation=0,
        ha="center",
        fontsize=10,
        color="#e0e0e0"
    )

    # Titles and labels
    if prediction_type == "buy":
        ax.set_ylabel("Price (Lakhs)", color="#e0e0e0", fontsize=10)
        title = "Purchase Price Comparison"
    else:
        ax.set_ylabel("Monthly Rent (₹)", color="#e0e0e0", fontsize=10)
        title = "Rental Price Comparison"

    ax.set_title(
        title,
        fontsize=14,
        color="#f0f0f0",
        fontweight="bold",
        pad=12
    )

    # Grid styling
    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.8,
        color="#2a3045",
        alpha=0.8
    )

    # Axis styling
    ax.tick_params(axis="y", colors="#e0e0e0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2a3045")
    ax.spines["bottom"].set_color("#2a3045")

    # Layout
    plt.tight_layout()

    # Save image
    plt.savefig(
        "static/plot.png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        transparent=False
    )

    plt.close(fig)

# -------------------------------
# Home Page (Single Prediction)
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html", locations=locations)


# -------------------------------
# Comparison Page
# -------------------------------
@app.route("/compare-page")
def compare_page():
    return render_template("compare.html", locations=locations)


# -------------------------------
# Single Prediction Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        sqft = float(request.form["sqft"])
        bath = float(request.form["bath"])
        bhk = float(request.form["bhk"])
        location = request.form["location"]
        prediction_type = request.form["prediction_type"]

        if prediction_type == "rent":
            value = predict_price(
                rent_model,
                rent_columns,
                sqft,
                bath,
                bhk,
                location
            )
            low = round(value * 0.9, 2)
            high = round(value * 1.1, 2)
            pps = round(value / sqft, 2)
            category = "Rental Estimate"
            confidence = 87
        else:
            value = predict_price(
                buy_model,
                buy_columns,
                sqft,
                bath,
                bhk,
                location
            )
            low = round(value * 0.9, 2)
            high = round(value * 1.1, 2)
            pps = round((value * 100000) / sqft, 2)

            if value < 50:
                category = "Affordable"
            elif value < 150:
                category = "Mid-range"
            else:
                category = "Premium"

            confidence = 85

        return render_template(
            "index.html",
            locations=locations,
            price=value,
            low=low,
            high=high,
            pps=pps,
            category=category,
            confidence=confidence,
            prediction_type=prediction_type,
        )

    except Exception as e:
        return render_template(
            "index.html",
            locations=locations,
            error=str(e)
        )


# -------------------------------
# Compare Multiple Locations
# -------------------------------
@app.route("/compare", methods=["POST"])
def compare():
    try:
        sqft = float(request.form["compare_sqft"])
        bath = float(request.form["compare_bath"])
        bhk = float(request.form["compare_bhk"])
        prediction_type = request.form["compare_prediction_type"]

        # Collect selected locations
        selected_locations = [
            request.form.get("location1"),
            request.form.get("location2"),
            request.form.get("location3"),
            request.form.get("location4"),
        ]

        # Remove empty selections
        selected_locations = [
            loc for loc in selected_locations
            if loc and loc.strip()
        ]

        # Remove duplicates while preserving order
        seen = set()
        selected_locations = [
            loc for loc in selected_locations
            if not (loc in seen or seen.add(loc))
        ]

        # Validate at least two locations
        if len(selected_locations) < 2:
            raise ValueError("Please select at least two locations.")

        results = []

        # Predict for each location
        for location in selected_locations:
            if prediction_type == "rent":
                value = predict_price(
                    rent_model,
                    rent_columns,
                    sqft,
                    bath,
                    bhk,
                    location
                )
            else:
                value = predict_price(
                    buy_model,
                    buy_columns,
                    sqft,
                    bath,
                    bhk,
                    location
                )

            results.append({
                "location": location,
                "value": value
            })

        # Sort descending
        results = sorted(
            results,
            key=lambda x: x["value"],
            reverse=True
        )

        # Generate line chart
        generate_comparison_chart(results, prediction_type)

        return render_template(
            "compare.html",
            locations=locations,
            comparison_results=results,
            compare_prediction_type=prediction_type,
        )

    except Exception as e:
        return render_template(
            "compare.html",
            locations=locations,
            error=str(e)
        )


# -------------------------------
# Run Application
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )