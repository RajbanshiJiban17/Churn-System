import joblib
import pandas as pd
import numpy as np
from pathlib import Path

import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ------------------------- PATHS & LOADING -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_churn_model_GradientBoosting.pkl"
DATA_PATH = BASE_DIR / "telco.csv"  

@st.cache_resource
def load_model():
    loaded = joblib.load(MODEL_PATH)
    # If saved as dict
    if isinstance(loaded, dict):
        model = loaded["model"]
        feature_names = loaded.get("features")
    else:
        model = loaded
        feature_names = None
    return model, feature_names


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    # basic cleaning (TotalCharges numeric)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"])
    return df


model, feature_names = load_model()
df_raw = load_data()

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(
    page_title="Telecom Churn – Streamlit",
    page_icon="📉",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- SIDEBAR -------------------------
st.sidebar.title("📊 Churn Employment Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "EDA & Churn Rate", "Model Performance", "Predict Churn", "Business Recommendations"],
    index=0,
)

st.sidebar.info("Using existing Gradient Boosting `.pkl` model.")

# ------------------------- HELPER: ENCODING -------------------------


def encode_single_customer(raw_dict: dict) -> pd.DataFrame:
    """
    raw_dict: values taken from Streamlit widgets.
    Output: single-row DataFrame with SAME columns that the model expects.
    Adjust column names to match your df_encoded.columns.
    """
    tenure = raw_dict["tenure"]
    monthly = raw_dict["MonthlyCharges"]
    total = raw_dict["TotalCharges"]

    contract = raw_dict["Contract"]          # "Month-to-month", "One year", "Two year"
    internet = raw_dict["InternetService"]   # "No", "DSL", "Fiber optic"
    payment = raw_dict["PaymentMethod"]      # "Electronic check", ...

    # One-hot columns – names must match training columns exactly
    row = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,

        "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
        "Contract_One year":       1 if contract == "One year"       else 0,
        "Contract_Two year":       1 if contract == "Two year"       else 0,

        "InternetService_DSL":         1 if internet == "DSL"         else 0,
        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No":          1 if internet == "No"          else 0,

        "PaymentMethod_Electronic check":          1 if payment == "Electronic check"          else 0,
        "PaymentMethod_Mailed check":              1 if payment == "Mailed check"              else 0,
        "PaymentMethod_Bank transfer (automatic)": 1 if payment == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)":   1 if payment == "Credit card (automatic)"   else 0,
    }

    # Optionally: add other binary features used in training
    for col in [
        "SeniorCitizen",
        "Partner_Yes",
        "Dependents_Yes",
        "PhoneService_Yes",
        "PaperlessBilling_Yes",
        "OnlineSecurity_Yes",
        "OnlineBackup_Yes",
        "DeviceProtection_Yes",
        "TechSupport_Yes",
        "StreamingTV_Yes",
        "StreamingMovies_Yes",
    ]:
        if col not in row:
            row[col] = 0

    df = pd.DataFrame([row])

    # If feature_names were saved during training, enforce same ordering
    if feature_names is not None:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]
    return df


# ------------------------- PAGES -------------------------

if page == "Overview":
    st.title("📉 Telecom Customer Churn – Employment Use Case")
    st.write(
        """
        This Streamlit app reuses your previously trained Gradient Boosting churn model
        to provide a complete interactive dashboard.
        """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model type", type(model).__name__)
    with col2:
        if feature_names is not None:
            st.metric("Input features", len(feature_names))

    if df_raw is not None:
        st.write("Sample of raw data:")
        st.dataframe(df_raw.head())

elif page == "EDA & Churn Rate":
    st.title("🔎 EDA & Churn Rate")

    if df_raw is None or "Churn" not in df_raw.columns:
        st.warning("`telco.csv` file or `Churn` column not found.")
    else:
        churn_rate = (df_raw["Churn"] == "Yes").mean()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total customers", len(df_raw))
        with col2:
            st.metric("Churn customers", int((df_raw["Churn"] == "Yes").sum()))
        with col3:
            st.metric("Churn rate", f"{churn_rate*100:.2f}%")

        st.subheader("Churn by Tenure bucket")
        bins = [0, 6, 12, 24, 72]
        labels = ["0–6", "6–12", "12–24", "24+"]
        tmp = df_raw.copy()
        tmp["tenure_bucket"] = pd.cut(tmp["tenure"], bins=bins, labels=labels, right=True)
        bucket_churn = (
            tmp.groupby("tenure_bucket")["Churn"]
            .apply(lambda x: (x == "Yes").mean())
            .reset_index(name="churn_rate")
        )
        st.bar_chart(bucket_churn.set_index("tenure_bucket"))

elif page == "Model Performance":
    st.title("📈 Model Performance (offline)")

    st.write(
        "If you saved test-set metrics during training you can load them here. "
        "For now we use manual inputs."
    )

    with st.form("metrics_form"):
        acc = st.number_input("Accuracy", value=0.7953, min_value=0.0, max_value=1.0, step=0.0001)
        prec = st.number_input("Precision", value=0.6378, min_value=0.0, max_value=1.0, step=0.0001)
        rec = st.number_input("Recall", value=0.5321, min_value=0.0, max_value=1.0, step=0.0001)
        f1 = st.number_input("F1-score", value=0.5802, min_value=0.0, max_value=1.0, step=0.0001)
        auc = st.number_input("AUC", value=0.8407, min_value=0.0, max_value=1.0, step=0.0001)
        submitted = st.form_submit_button("Update view")

    cols = st.columns(5)
    metrics = [("Accuracy", acc), ("Precision", prec), ("Recall", rec), ("F1", f1), ("AUC", auc)]
    for c, (name, val) in zip(cols, metrics):
        with c:
            if name in ("F1", "AUC"):
                st.metric(name, f"{val:.4f}")
            else:
                st.metric(name, f"{val*100:.2f}%")

elif page == "Predict Churn":
    st.title("🧮 Predict Individual Customer Churn")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        with c2:
            monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        with c3:
            total = st.number_input("Total Charges", min_value=0.0, value=800.0)

        c4, c5, c6 = st.columns(3)
        with c4:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c5:
            internet = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        with c6:
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        raw = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "InternetService": internet,
            "PaymentMethod": payment,
        }

        X = encode_single_customer(raw)
        proba = model.predict_proba(X)[:, 1][0]
        label = int(proba >= 0.5)

        st.subheader("Result")
        st.write(f"**Churn probability:** `{proba*100:.2f}%`")
        if label == 1:
            st.error("Model prediction: **Likely to CHURN**")
        else:
            st.success("Model prediction: **Not likely to churn**")

        st.write("Feature vector (first 10 columns):")
        st.dataframe(X.iloc[:, :10])

elif page == "Business Recommendations":
    st.title("💡 Business Recommendations")

    st.markdown(
        """
        - **New customer onboarding:** Focus on high-risk customers in the first 0–6 months.
        - **Contract length incentives:** Move month‑to‑month customers to 1–2 year contracts with discounts.
        - **Payment method shift:** Encourage a shift from electronic check to automatic payments.
        - **Fiber optic quality program:** Improve QoS and reduce complaints for fiber customers.
        """
    )
