import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
random_forest_loan = joblib.load("random_forest_loan.pkl")
scaler = joblib.load("scaling_chd.pkl")

def predict_credit_approval(features):
    scaled_features = scaler.transform(features)
    prediction = random_forest_loan.predict(scaled_features)[0]
    return prediction

def main():
    st.title("Loan Approval Prediction")

    # Credit Type selection
    credit_type = st.selectbox("Credit Type", ["Individual", "Joint"])

    # Mapping Credit Type to numeric value
    credit_type_mapping = {"Individual": 0, "Joint": 1}
    credit_type_numeric = credit_type_mapping[credit_type]

    # Other input features
    job_hours = st.number_input("Job Hours", value=0)
    income = st.number_input("Income", value=0)
    ltv = st.number_input("Loan-to-Value (LTV)", value=0)
    term = st.number_input("Term", value=0)
    price = st.number_input("Price", value=0)
    downpayment = st.number_input("Downpayment", value=0)
    book_value = st.number_input("Book Value", value=0)
    amount_financed = st.number_input("Amount Financed", value=0)
    apr = st.number_input("APR", value=0)
    monthly_payment = st.number_input("Monthly Payment", value=0)
    monthly_dedt = st.number_input("Monthly Dedt", value=0)
    num_tradelines = st.number_input("Number of Tradelines", value=0)
    fico = st.number_input("FICO Score", value=0)
    clearfraud_score = st.number_input("ClearFraud Score", value=0)
    vehicle_year = st.number_input("Vehicle Year", value=0)
    vehicle_age = st.number_input("Vehicle Age", value=0)
    vehicle_miles = st.number_input("Vehicle Miles", value=0)
    dti = st.number_input("Debt-to-Income Ratio (DTI)", value=0)
    ltv_dp = st.number_input("LTV Downpayment", value=0)

    # Prepare feature array for prediction
    features = np.array([
        credit_type_numeric,
        job_hours,
        income,
        ltv,
        term,
        price,
        downpayment,
        book_value,
        amount_financed,
        apr,
        monthly_payment,
        monthly_dedt,
        num_tradelines,
        fico,
        clearfraud_score,
        vehicle_year,
        vehicle_age,
        vehicle_miles,
        dti,
        ltv_dp
    ]).reshape(1, -1)

    # Predict loan approval
    if st.button("Predict"):
        prediction = predict_credit_approval(features)
        if prediction == 1:
            st.success("Loan Approved!")
        else:
            st.error("Loan Rejected.")

if __name__ == "__main__":
    main()
