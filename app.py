
import json
import streamlit.components.v1 as components
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

# ===== PREMIUM DARK PURPLE UI =====

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at top left, #25205a 0%, transparent 40%),
        linear-gradient(135deg, #080b18 0%, #10152b 55%, #17112e 100%);
    color: #f5f3ff;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11152b 0%, #17112e 100%);
    border-right: 1px solid rgba(167, 139, 250, 0.22);
}

[data-testid="stSidebar"] h1 {
    color: #c4b5fd;
    font-weight: 800;
}

h1, h2, h3 {
    color: #f5f3ff !important;
    letter-spacing: -0.5px;
}

p, label, .stMarkdown {
    color: #d6d3e8;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1b2140, #25204b);
    border: 1px solid rgba(167, 139, 250, 0.28);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.18);
}

div[data-testid="stMetricLabel"] {
    color: #c4b5fd;
}

div[data-testid="stMetricValue"] {
    color: #ffffff;
    font-weight: 800;
}

.stButton > button {
    background: linear-gradient(100deg, #7c3aed, #a855f7);
    color: #ffffff;
    border: 1px solid rgba(216, 180, 254, 0.35);
    border-radius: 12px;
    padding: 0.65rem 1.15rem;
    font-weight: 700;
    min-height: 45px;
    transition: all 0.2s ease;
    box-shadow: 0 5px 18px rgba(124, 58, 237, 0.22);
}

.stButton > button:hover {
    background: linear-gradient(100deg, #8b5cf6, #c084fc);
    border-color: #ddd6fe;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(168, 85, 247, 0.35);
    color: #ffffff;
}

.stTextArea textarea {
    background: #131a31;
    color: #ffffff;
    border: 1px solid #49416f;
    border-radius: 14px;
}

.stTextArea textarea:focus {
    border-color: #a78bfa;
    box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.18);
}

div[data-baseweb="select"] > div {
    background-color: #171d36;
    border-color: #49416f;
    border-radius: 10px;
}

hr {
    border-color: rgba(167, 139, 250, 0.2);
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

[data-testid="stPlotlyChart"] {
    background: rgba(24, 29, 55, 0.65);
    border: 1px solid rgba(167, 139, 250, 0.16);
    border-radius: 16px;
    padding: 8px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ===== SIDEBAR NAVIGATION =====

st.sidebar.markdown("# 💜 Complaint AI")
st.sidebar.caption("Intelligent Complaint Management")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Home Dashboard",
        "🔍 Complaint Analyzer",
        "📈 Analytics & Insights",
        "🎙️ Voice Assistant & Help"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### 🤖 Model Status")
st.sidebar.caption("CNN Deep Learning Model")
st.sidebar.caption("Test Accuracy: 88.34%")

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

if complaint.strip():
    components.html(
        f"""
        <div style="font-family: sans-serif;">
            <button
                onclick="speakComplaint()"
                style="
                    background: #6C63FF;
                    color: white;
                    border: none;
                    padding: 12px 20px;
                    border-radius: 10px;
                    font-size: 15px;
                    cursor: pointer;
                "
            >
                🔊 Listen to Complaint
            </button>

            <button
                onclick="stopSpeaking()"
                style="
                    background: #444;
                    color: white;
                    border: none;
                    padding: 12px 20px;
                    border-radius: 10px;
                    font-size: 15px;
                    cursor: pointer;
                    margin-left: 8px;
                "
            >
                ⏹ Stop
            </button>

            <script>
                const complaintText = {json.dumps(complaint)};

                function speakComplaint() {{
                    window.speechSynthesis.cancel();

                    const message = new SpeechSynthesisUtterance(
                        complaintText
                    );

                    message.lang = "en-IN";
                    message.rate = 0.95;
                    message.pitch = 1;

                    window.speechSynthesis.speak(message);
                }}

                function stopSpeaking() {{
                    window.speechSynthesis.cancel();
                }}
            </script>
        </div>
        """,
        height=65
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
