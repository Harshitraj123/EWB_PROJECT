"""
Streamlit interface for the telecom customer churn model.

Run with:
    streamlit run app.py
"""

import streamlit as st

from src.predict import ChurnPredictor


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
)

st.title("📉 Customer Churn Predictor")
st.caption(
    "Predict whether a telecom customer is likely to churn "
    "using a trained scikit-learn pipeline."
)


@st.cache_resource
def get_predictor() -> ChurnPredictor:
    """Load and cache the trained prediction pipeline."""
    return ChurnPredictor()


try:
    predictor = get_predictor()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Run `python main.py` first to train and save the model.")
    st.stop()


with st.form("churn_form"):
    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12,
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0,
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0,
            step=1.0,
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
        )

        partner = st.selectbox(
            "Has Partner",
            ["Yes", "No"],
        )

        dependents = st.selectbox(
            "Has Dependents",
            ["Yes", "No"],
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"],
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )

    with col2:
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"],
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True,
    )


if submitted:
    if tenure == 0 and total_charges != 0:
        st.warning(
            "Customers with zero tenure normally have no recorded "
            "Total Charges in the source dataset."
        )

    features = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
    }

    try:
        result = predictor.predict(features)
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
    else:
        st.subheader("Prediction")

        probability = result["churn_probability"]

        if result["churn_prediction"] == "Yes":
            st.error(
                f"⚠️ Likely to churn — "
                f"churn probability: {probability:.1%}"
            )
        else:
            st.success(
                f"✅ Likely to stay — "
                f"churn probability: {probability:.1%}"
            )