import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os

# -----------------------------
# Step 1: Load model and columns
# -----------------------------
model_file = "best_model_compressed.pkl"
columns_file = "model_training_columns.pkl"

if not os.path.exists(model_file) or not os.path.exists(columns_file):
    st.error("Model file or columns file is missing!")
    st.stop()

try:
    model = joblib.load(model_file)
    model_columns = joblib.load(columns_file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Mapping for stops and times
# -----------------------------
stops_mapping = {"0": 0, "1": 1, "2": 2}
time_mapping = {
    "Early_Morning": 5,
    "Morning": 9,
    "Afternoon": 14,
    "Evening": 18,
    "Night": 21,
    "Late_Night": 0
}

# -----------------------------
# Prediction function
# -----------------------------
def predict_price(flight_details):
    df = pd.DataFrame([flight_details])
    df.drop(columns=["flight", "airline"], inplace=True, errors="ignore")
    df["stops"] = df["stops"].map(stops_mapping)
    
    # Cap unrealistic durations (2–8 hrs only)
    if "duration" in df.columns:
        df["duration"] = df["duration"].clip(lower=2.0, upper=8.0)

    df_encoded = pd.get_dummies(df)
    df_aligned = df_encoded.reindex(columns=model_columns, fill_value=0)
    pred_log = model.predict(df_aligned)
    pred_actual = np.expm1(pred_log)
    return int(round(pred_actual[0]))

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Indigo Flight Price Predictor", page_icon="✈️", layout="wide")

st.markdown(
    """
    <style>
    .main {background: linear-gradient(to right, #e0f7fa, #ffffff); padding: 2rem;}
    .stButton button {background-color: #1976d2; color: white; font-weight: bold; border-radius: 10px; padding: 0.6rem 1.2rem;}
    .stButton button:hover {background-color: #1565c0;}
    .price-box {background-color: #e3f2fd; padding: 20px; border-radius: 15px; text-align: center; font-size: 1.3rem; font-weight: bold; color: #0d47a1; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);}
    input, select {font-size: 1rem; color: #0d47a1; padding: 0.5rem; border-radius: 4px; border: 1px solid #ccc; width: 100%;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='color:#0d47a1;'>✈️ Indigo Flight Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("### Plan Better. Travel Easier. 💡")

# -----------------------------
# Input section
# -----------------------------
with st.container():
    st.subheader("Enter Flight Details")
    col1, col2 = st.columns(2)

    with col1:
        source_city = st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
        destination_city = st.selectbox(
            "Destination City",
            [c for c in ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"] if c != source_city]
        )
        travel_date = st.date_input(
            "Travel Date",
            min_value=datetime.date.today(),
            max_value=datetime.date.today() + datetime.timedelta(days=180),
            value=datetime.date.today() + datetime.timedelta(days=30)
        )
        days_left = (travel_date - datetime.date.today()).days
        # Display Days Left dynamically
        st.metric(label="Days Left until Travel", value=days_left)

    with col2:
        departure_time = st.selectbox("Departure Time", list(time_mapping.keys()))
        arrival_time = st.selectbox("Arrival Time", list(time_mapping.keys()))
        stops = st.selectbox("Stops (0=Non-stop)", ["0", "1", "2"])
        duration = st.number_input("Flight Duration (hrs)", min_value=2.0, max_value=8.0, step=0.1, value=2.5)

# -----------------------------
# Predict button and output
# -----------------------------
st.markdown("---")
if st.button("🚀 Predict Price"):
    details = {
        "source_city": source_city,
        "destination_city": destination_city,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "stops": stops,
        "duration": duration,
        "days_left": days_left
    }

    price = predict_price(details)
    lower = int(round(price * 0.95))
    upper = int(round(price * 1.05))

    st.markdown(
        f"""
        <div class="price-box">
            ✈️ Estimated Indigo Economy Price: ₹{price:,} <br>
            🔹 Price Range: ₹{lower:,} – ₹{upper:,}
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("Fill in the details and click **Predict Price** to see the fare.")

st.markdown("---")
st.caption("💡 Powered by RandomForest model trained on Indigo flight dataset")
