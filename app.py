import streamlit as st
from model import train_model

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.markdown(
    """
    <style>
    /* Hide top-right deploy & menu */
    [data-testid="stToolbar"] {
        display: none;
    }

    /* Hide footer */
    footer {
        visibility: hidden;
    }

    /* Optional: hide Streamlit header spacing */
    header {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

model, features , encoders= train_model()

st.title("📉 Customer Churn Prediction")
st.write("Inputs exactly match dataset column types")

st.divider()

# ---------- INPUTS (MATCHING CSV TYPES) ----------

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure (months)", min_value=0, step=1)

phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox("Online Security", ["Yes", "No"])
online_backup = st.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.selectbox("Device Protection", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)
raw_input = [
    gender,
    senior,
    partner,
    dependents,
    tenure,
    phone,
    multiple_lines,
    internet,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless,
    payment,
    monthly,
    total
]

# Convert strings to numbers using encoders
for i, feature in enumerate(features):
    if feature in encoders:
        if isinstance(raw_input[i], str):
            raw_input[i] = encoders[feature].transform([raw_input[i]])[0]

# Predict
if st.button("🚀 Predict Churn"):
    prediction = model.predict([raw_input])[0]

    if prediction == 1:
        st.error("⚠️ Customer will CHURN")
    else:
        st.success("✅ Customer will NOT churn")