import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from xgboost import XGBClassifier

# === Streamlit page config ===
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection (XGBoost)")

# === PDF Generation ===
def create_pdf(model_name, cm):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, height - 40, "Credit Card Fraud Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(30, height - 70, f"Model Used: {model_name}")

    c.setFont("Helvetica", 10)
    c.drawString(30, height - 100, "Confusion Matrix:")
    text_cm = c.beginText(30, height - 120)
    for row in cm:
        text_cm.textLine("  ".join(map(str, row)))
    c.drawText(text_cm)

    c.save()
    buf.seek(0)
    return buf

# train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_train.columns, y_test, y_pred

# === Main app ===
uploaded_file = st.file_uploader("📁 Upload credit card transactions CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Class' not in df.columns:
        st.error("❌ 'Class' column not found in dataset.")
        st.stop()

    # Dataset overview
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df))
    col2.metric("Fraud Cases", int(df['Class'].sum()))
    col3.metric("Normal Cases", int(len(df) - df['Class'].sum()))

    st.write("### Fraud Distribution")
    fig_dist, ax_dist = plt.subplots()
    sns.countplot(data=df, x='Class', ax=ax_dist)
    ax_dist.set_xticklabels(['Normal (0)', 'Fraud (1)'])
    st.pyplot(fig_dist)

    # Train model
    X = df.drop(columns=['Class'])
    y = df['Class']
    model, feature_names, y_test, y_pred = train_model(X, y)

    # Show Confusion Matrix only
    st.subheader("📈 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # Download report PDF
    model_name = "XGBoost Classifier"
    pdf_data = create_pdf(model_name, cm)
    st.download_button("📄 Download Report as PDF", data=pdf_data, file_name="xgboost_report.pdf", mime="application/pdf")

    # Live prediction UI
    st.subheader("🧪 Live Transaction Prediction")
    st.write("🔢 Enter values separated by commas in this order:")
    st.code(", ".join(feature_names), language="plaintext")

    user_input_str = st.text_input("📥 Paste transaction values (comma-separated):")

    if st.button("🔍 Predict"):
        try:
            input_values = [float(x.strip()) for x in user_input_str.split(",")]
            if len(input_values) != len(feature_names):
                st.error(f"❌ Expected {len(feature_names)} values, but got {len(input_values)}.")
            else:
                user_df = pd.DataFrame([input_values], columns=feature_names)
                prediction = model.predict(user_df)[0]
                proba = model.predict_proba(user_df)[0][1]

                if prediction == 1:
                    st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {proba:.2%})")
                else:
                    st.success(f"✅ Normal Transaction. (Fraud Probability: {proba:.2%})")
        except ValueError:
            st.error("❌ Invalid input. Please enter numeric values separated by commas.")
else:
    st.info("👆 Please upload a dataset to continue.")
