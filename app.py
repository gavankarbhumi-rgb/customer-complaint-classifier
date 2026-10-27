
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pickle
import re

from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Complaint Intelligence",
    page_icon="📊",
    layout="wide"
)

MODEL_REPO = "Bhumiiiiii26/customer-complaint-cnn"
MODEL_FILE = "customer_complaint_cnn (2).keras"

@st.cache_resource
def load_ai_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )
    return load_model(model_path, compile=False)

@st.cache_resource
def load_assets():
    vocab_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="vocab.pkl"
    )
    encoder_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="label_encoder.pkl"
    )

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    return vocab, label_encoder

st.title("📊 Customer Complaint Intelligence")
st.markdown(
    "### AI-powered complaint analysis and classification"
)
st.write(
    "Enter a customer complaint below to predict its category "
    "using our trained CNN deep learning model."
)

try:
    model = load_ai_model()
    vocab, label_encoder = load_assets()
    model_ready = True
    st.success("AI model loaded successfully!")
except Exception as e:
    model_ready = False
    st.error(
        "The AI model could not be loaded. "
        "Please check the model repository and file names."
    )
    st.code(str(e))

# Dashboard metrics
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
    template="plotly_dark",
    title="Test Accuracy by Model"
)
fig.update_layout(
    yaxis_range=[0, 100],
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

# Complaint prediction
st.divider()
st.subheader("🔍 Analyze a Customer Complaint")

complaint = st.text_area(
    "Enter complaint details",
    placeholder=(
        "Example: I was charged a fee on my credit card "
        "that I do not recognize."
    ),
    height=150
)

if st.button("🚀 Classify Complaint", type="primary"):
    if not complaint.strip():
        st.warning("Please enter a complaint first.")
    elif not model_ready:
        st.error("The AI model is not ready yet.")
    else:
        try:
            cleaned = re.sub(
                r"\s+", " ", complaint.lower()
            ).strip()

            tokens = cleaned.split()
            oov_id = vocab.get("<OOV>")

            if oov_id is None:
                st.error(
                    "The vocabulary is missing its <OOV> token."
                )
            else:
                sequence = [
                    vocab.get(word, oov_id)
                    for word in tokens
                ]

                padded = pad_sequences(
                    [sequence],
                    maxlen=240,
                    padding="post",
                    truncating="post",
                    value=0
                )

                probabilities = model.predict(
                    padded, verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(probabilities)
                )

                category = label_encoder.inverse_transform(
                    [predicted_index]
                )[0]

                confidence = float(
                    probabilities[predicted_index]
                ) * 100

                st.subheader("✅ Prediction Result")
                st.success(
                    f"Predicted Category: "
                    f"{category.replace('_', ' ').title()}"
                )

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

                if confidence < 60:
                    st.warning(
                        "The model is less confident about "
                        "this prediction. Please review it."
                    )

                st.subheader("Category Probability Scores")

                score_data = pd.DataFrame({
                    "Category": [
                        name.replace("_", " ").title()
                        for name in label_encoder.classes_
                    ],
                    "Probability (%)": probabilities * 100
                }).sort_values(
                    "Probability (%)", ascending=True
                )

                score_fig = px.bar(
                    score_data,
                    x="Probability (%)",
                    y="Category",
                    orientation="h",
                    template="plotly_dark",
                    title="Prediction Scores by Category"
                )

                st.plotly_chart(
                    score_fig, use_container_width=True
                )

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))

st.divider()
st.subheader("🗂️ Supported Complaint Categories")

categories = [
    "Credit Card",
    "Credit Reporting",
    "Debt Collection",
    "Mortgages and Loans",
    "Retail Banking"
]

st.write(" • ".join(categories))

st.caption(
    "Academic project prototype | "
    "CNN test accuracy: 88.34%"
)
