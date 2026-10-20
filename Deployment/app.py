import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("best_loan_model.joblib")


model = load_model()

st.title("🏦 Loan Approval Predictor")
st.write("Fill in the applicant's details to predict whether the loan will be approved.")

with st.form("loan_form"):
    st.subheader("Applicant Information")
    col1, col2 = st.columns(2)

    with col1:
        no_of_dependents = st.number_input("Number of dependents", min_value=0, max_value=10, value=2, step=1)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self employed", ["No", "Yes"])
        cibil_score = st.slider("CIBIL score", min_value=300, max_value=900, value=700)

    with col2:
        income_annum = st.number_input("Annual income", min_value=0, value=5_000_000, step=100_000)
        loan_amount = st.number_input("Loan amount", min_value=0, value=15_000_000, step=100_000)
        loan_term = st.slider("Loan term (years)", min_value=2, max_value=20, value=10)

    st.subheader("Assets")
    col3, col4 = st.columns(2)
    with col3:
        residential_assets_value = st.number_input("Residential assets value", min_value=0, value=5_000_000, step=100_000)
        commercial_assets_value = st.number_input("Commercial assets value", min_value=0, value=5_000_000, step=100_000)
    with col4:
        luxury_assets_value = st.number_input("Luxury assets value", min_value=0, value=10_000_000, step=100_000)
        bank_asset_value = st.number_input("Bank asset value", min_value=0, value=5_000_000, step=100_000)

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    applicant = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }])

    pred = model.predict(applicant)[0]
    proba = model.predict_proba(applicant)[0, 1]

    st.divider()
    if pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.metric("Approval probability", f"{proba:.1%}")
    st.progress(float(proba))

    with st.expander("Submitted data"):
        st.dataframe(applicant, use_container_width=True)
