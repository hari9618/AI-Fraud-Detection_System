import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlxtend.frequent_patterns import apriori, association_rules


# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="AI Fraud Intelligence System",
    page_icon="💳",
    layout="wide"
)

st.title("💳 AI-Powered Fraud Intelligence System")
st.markdown("### Hybrid Model: Association Rule Mining + Machine Learning")


# ----------------------------------------------------------
# FILE UPLOADER
# ----------------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Credit Card Dataset (CSV File)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the credit card dataset to continue.")
    st.stop()


# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df


df = load_data(uploaded_file)

# Validate required columns
required_columns = ["Amount", "Class"]

for col in required_columns:
    if col not in df.columns:
        st.error(f"Dataset must contain '{col}' column.")
        st.stop()


# Remove missing values if any
df = df.dropna()


# ----------------------------------------------------------
# TRAIN MODEL (Optimized & Safe)
# ----------------------------------------------------------
@st.cache_resource
def train_model(data):

    # Sample data (prevents heavy training)
    sample_size = min(40000, len(data))
    data_sample = data.sample(sample_size, random_state=42)

    X = data_sample.drop("Class", axis=1)
    y = data_sample["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # keeps fraud ratio balanced
    )

    model = RandomForestClassifier(
        n_estimators=25,
        max_depth=8,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


model, X_test, y_test = train_model(df)


# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["📊 Dashboard", "📈 Fraud Pattern Discovery", "🤖 ML Prediction"]
)


# ==========================================================
# DASHBOARD
# ==========================================================
if page == "📊 Dashboard":

    total_tx = len(df)
    fraud_tx = int(df["Class"].sum())
    fraud_rate = (fraud_tx / total_tx) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", total_tx)
    col2.metric("Fraud Cases", fraud_tx)
    col3.metric("Fraud Rate (%)", f"{fraud_rate:.4f}%")

    st.markdown("---")

    fig = px.histogram(
        df,
        x="Amount",
        nbins=50,
        title="Transaction Amount Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# FRAUD PATTERN DISCOVERY
# ==========================================================
elif page == "📈 Fraud Pattern Discovery":

    st.subheader("Association Rule Mining (Apriori)")

    df_copy = df.copy()

    df_copy["Amount_Category"] = pd.cut(
        df_copy["Amount"],
        bins=[0, 50, 200, 1000, df_copy["Amount"].max()],
        labels=["Low", "Medium", "High", "Very_High"]
    )

    df_copy["Fraud_Label"] = df_copy["Class"].map({0: "No", 1: "Yes"})

    basket = df_copy[["Amount_Category", "Fraud_Label"]]
    basket = pd.get_dummies(basket)

    frequent_items = apriori(
        basket,
        min_support=0.02,
        use_colnames=True
    )

    if frequent_items.empty:
        st.warning("No frequent patterns detected.")
    else:
        rules = association_rules(
            frequent_items,
            metric="lift",
            min_threshold=1
        )

        if rules.empty:
            st.warning("No strong association rules found.")
        else:
            rules_sorted = rules.sort_values(
                by="lift",
                ascending=False
            )

            st.dataframe(rules_sorted.head(10))

            fig = px.bar(
                rules_sorted.head(10),
                x="confidence",
                y="lift",
                title="Top Fraud Rules"
            )
            st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# ML PREDICTION
# ==========================================================
elif page == "🤖 ML Prediction":

    st.subheader("Random Forest Fraud Detection")

    accuracy = model.score(X_test, y_test)
    st.success(f"Model Accuracy: {accuracy:.4f}")

    st.markdown("---")
    st.subheader("🔍 Test Custom Transaction")

    amount = st.slider(
        "Transaction Amount",
        min_value=0.0,
        max_value=float(df["Amount"].max()),
        value=100.0
    )

    if st.button("Analyze Transaction"):

        sample = X_test.iloc[0].copy()
        sample["Amount"] = amount

        prediction = model.predict([sample])[0]
        probability = model.predict_proba([sample])[0][1]
        risk_score = probability * 100

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Fraud Risk Score (%)"},
            gauge={'axis': {'range': [0, 100]}}
        ))

        st.plotly_chart(fig)

        if prediction == 1:
            st.error("⚠ High Fraud Risk Detected!")
        else:
            st.success("✅ Transaction Appears Legitimate")


st.markdown("---")
st.markdown("🚀 Hybrid AI-Based Credit Card Fraud Detection System")