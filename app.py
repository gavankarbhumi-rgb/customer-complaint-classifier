
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Complaint Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Complaint Intelligence")
st.write(
    "AI-powered customer complaint classification "
    "and analytics dashboard."
)

st.info(
    "Welcome! Your dashboard interface is ready. "
    "The trained model will be connected in the next step."
)

# Model performance
col1, col2, col3 = st.columns(3)

col1.metric("Best Model", "CNN")
col2.metric("Test Accuracy", "88.34%")
col3.metric("Complaint Categories", "5")

# Model comparison
comparison = pd.DataFrame({
    "Model": ["CNN", "Baseline", "LSTM"],
    "Test Accuracy (%)": [88.34, 87.92, 86.94]
})

st.subheader("📈 Model Performance Comparison")

fig = px.bar(
    comparison,
    x="Model",
    y="Test Accuracy (%)",
    color="Model",
    text="Test Accuracy (%)",
    title="Test Accuracy by Model",
    template="plotly_dark"
)

fig.update_layout(
    yaxis_range=[80, 100],
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# Complaint categories
st.subheader("🗂️ Supported Complaint Categories")

categories = [
    "Credit Card",
    "Credit Reporting",
    "Debt Collection",
    "Mortgages and Loans",
    "Retail Banking"
]

for category in categories:
    st.write("•", category)

st.caption(
    "Academic project prototype | Customer Complaint Classification"
)
