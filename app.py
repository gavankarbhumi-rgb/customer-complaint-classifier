
import json
import os
import re
import pickle
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Complaint Intelligence",
    page_icon="💜",
    layout="wide"
)

# ---------------- PREMIUM UI ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
      radial-gradient(ellipse at top left,#25205a 0%,transparent 42%),
      linear-gradient(135deg,#080b18,#10152b 55%,#17112e);
    color: #f5f3ff;
    font-family: Inter, sans-serif;
}
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(ellipse at top left, #30205c 0%, transparent 55%),
        linear-gradient(180deg, #11152b 0%, #0b1020 100%);
    border-right: 1px solid rgba(167, 139, 250, 0.35);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    color: #c4b5fd !important;
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -0.7px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: #c4b5fd;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 8px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 10px 12px;
    border-radius: 12px;
    transition: background 0.2s ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(167, 139, 250, 0.12);
}

[data-testid="stSidebar"] hr {
    border-color: rgba(167, 139, 250, 0.25);
}

h1,h2,h3 { color:#f5f3ff !important; }
p,label { color:#d6d3e8; }
div[data-testid="stMetric"] {
    background:linear-gradient(135deg,#1b2140,#25204b);
    border:1px solid #49416f;
    padding:20px;
    border-radius:18px;
}
.stButton>button {
    background:linear-gradient(100deg,#7c3aed,#a855f7);
    color:white;
    border:1px solid #a78bfa;
    border-radius:12px;
    min-height:44px;
    font-weight:700;
}
.stTextArea textarea, .stTextInput input {
    background:#131a31;
    color:white;
    border:1px solid #49416f;
    border-radius:12px;
}
hr { border-color:#39305d; }
footer,#MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------- NAVIGATION ----------------

st.sidebar.markdown("# 💜 Complaint AI")
st.sidebar.caption("Intelligent Complaint Management")


pages = [
    "🏠 Home Dashboard",
    "🔍 Complaint Analyzer",
    "📈 Analytics & Insights",
    "🗂️ Complaint History",
    "🎙️ Voice Assistant & Help"
]

if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Home Dashboard"

page = st.sidebar.radio(
    "✦ MAIN MENU",
    pages,
    key="page",
    label_visibility="visible"
)


st.sidebar.divider()
st.sidebar.markdown("### 🤖 AI Model")
st.sidebar.caption("CNN Deep Learning")
st.sidebar.caption("Test accuracy: 88.34%")
st.sidebar.caption("5 complaint categories")


# ---------------- MODEL SETTINGS ----------------

MODEL_REPO = "Bhumiiiiii26/customer-complaint-cnn"
MODEL_FILE = "customer_complaint_cnn (2).keras"

DB_PATH = "complaint_history.db"

CATEGORY_NAMES = {
    "credit_card": "Credit Card",
    "credit_reporting": "Credit Reporting",
    "debt_collection": "Debt Collection",
    "mortgages_and_loans": "Mortgages and Loans",
    "retail_banking": "Retail Banking"
}


@st.cache_resource
def load_ai_model():
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )
    return load_model(path, compile=False)


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
        encoder = pickle.load(f)

    return vocab, encoder


# ---------------- LOCAL HISTORY DATABASE ----------------

def init_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_complaint(complaint, category, confidence):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO complaints
            (complaint, category, confidence, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            complaint,
            category,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))


def get_history():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM complaints ORDER BY id DESC",
            conn
        )


init_database()


# ---------------- VOICE WELCOME ----------------

WELCOME_TEXT = (
    "Welcome to Complaint Intelligence! "
    "Your AI powered customer complaint assistant. "
    "You can analyze complaints, explore analytics, "
    "view your complaint history, and learn how the "
    "website works. Select Start Exploring to begin."
)


def voice_player(text_to_speak, button_label="🔊 Play Voice"):
    safe_text = json.dumps(text_to_speak)

    components.html(
        f"""
        <div style="font-family:Arial;color:white;">
          <button onclick="speakText()"
            style="background:linear-gradient(90deg,#7c3aed,#a855f7);
            color:white;border:0;border-radius:10px;
            padding:12px 18px;cursor:pointer;font-weight:bold;">
            {button_label}
          </button>
          <button onclick="stopVoice()"
            style="background:#29243f;color:white;border:1px solid #62558a;
            border-radius:10px;padding:12px 18px;cursor:pointer;">
            ⏹ Stop
          </button>
          <script>
            const spokenText = {safe_text};
            function speakText() {{
              window.speechSynthesis.cancel();
              const message = new SpeechSynthesisUtterance(spokenText);
              message.lang = "en-IN";
              message.rate = 0.92;
              window.speechSynthesis.speak(message);
            }}
            function stopVoice() {{
              window.speechSynthesis.cancel();
            }}
          </script>
        </div>
        """,
        height=60
    )


# ---------------- SHARED MODEL METRICS ----------------

comparison = pd.DataFrame({
    "Model": ["CNN", "Baseline", "LSTM"],
    "Test Accuracy (%)": [88.34, 87.92, 86.94]
})

category_metrics = pd.DataFrame({
    "Category": [
        "Credit Card",
        "Credit Reporting",
        "Debt Collection",
        "Mortgages and Loans",
        "Retail Banking"
    ],
    "Precision (%)": [77.59, 91.35, 83.51, 88.07, 88.00],
    "Recall (%)": [81.21, 94.77, 74.90, 81.81, 85.37],
    "F1 Score (%)": [79.36, 93.03, 78.97, 84.82, 86.67]
})


# ====================================================
# PAGE 1: HOME DASHBOARD
# ====================================================

if page == "🏠 Home Dashboard":

    st.title("💜 Welcome to Complaint Intelligence")
    st.markdown(
        "### Understand customer complaints with AI."
    )

    st.write(
        "An intelligent complaint classification system "
        "powered by your trained CNN deep learning model."
    )

    voice_player(WELCOME_TEXT, "🔊 Start Voice Welcome")

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", "CNN")
    c2.metric("Test Accuracy", "88.34%")
    c3.metric("Categories", "5")

    st.subheader("✨ What can you do here?")

    a, b = st.columns(2)

    with a:
        st.markdown("#### 🔍 Complaint Analyzer")
        st.write("Enter a complaint and predict its category.")

        st.markdown("#### 📈 Analytics & Insights")
        st.write("Explore model performance and evaluation metrics.")

    with b:
        st.markdown("#### 🗂️ Complaint History")
        st.write("Review previously classified complaints.")

        st.markdown("#### 🎙️ Voice Assistant")
        st.write("Listen to the welcome message and instructions.")

    st.subheader("📊 Model Performance")

    fig = px.bar(
        comparison,
        x="Model",
        y="Test Accuracy (%)",
        color="Model",
        text="Test Accuracy (%)",
        template="plotly_dark",
        title="Model Accuracy Comparison"
    )
    fig.update_layout(yaxis_range=[0, 100], showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    


    st.button(
        "🚀 Start Exploring",
        key="start_exploring",
        on_click=lambda: st.session_state.update(
            {"page": "🔍 Complaint Analyzer"}
        )
    )





# ====================================================
# PAGE 2: COMPLAINT ANALYZER
# ====================================================


elif page == "🔍 Complaint Analyzer":

    st.title("🔍 Complaint Analyzer")
    st.write("Enter the complaint below to classify it using CNN.")

    # Instructions yahan add hongi

    instruction_text = """
    Welcome to the Complaint Analyzer.

    Step 1: Enter your customer complaint in the text box.
    Step 2: Describe the main problem clearly.
    Step 3: Click the Classify Complaint button.
    Step 4: Wait while the AI model analyzes your complaint.
    Step 5: View the predicted category and confidence score.

    For better results, write your complaint in clear English.
    Include the main issue and relevant details.
    Do not enter passwords or sensitive personal information.
    """

    voice_player(instruction_text, "🔊 Listen to Instructions")

   
    with st.expander("📖 How to Use the Complaint Analyzer", expanded=True):
        st.markdown("""
        **Follow these simple steps:**

        1. Enter your customer complaint in the text box below.
        2. Describe the main problem clearly.
        3. Click the **🚀 Classify Complaint** button.
        4. Wait while the AI model analyzes your complaint.
        5. View the predicted category and confidence score.

        **💡 Tips for better results**
        - Write your complaint in clear English.
        - Include the main issue and relevant details.
        - Do not enter passwords or sensitive personal information.
        """)

    complaint = st.text_area(
        "Customer complaint",
        placeholder=(
            "Example: I noticed a fee on my credit card "
            "that I do not recognize."
        ),
        height=180,
        key="complaint_input"
    )

    if complaint.strip():
        voice_player(
            complaint,
            "🔊 Listen to Complaint"
        )

    if st.button("🚀 Classify Complaint", type="primary"):

        if not complaint.strip():
            st.warning("Please enter a complaint first.")

        else:
            try:
                with st.spinner("Loading AI model and analyzing..."):
                    model = load_ai_model()
                    vocab, encoder = load_assets()

                    cleaned = re.sub(
                        r"\s+", " ", complaint.lower()
                    ).strip()

                    tokens = cleaned.split()
                    oov_id = vocab.get("<OOV>")

                    if oov_id is None:
                        st.error(
                            "Vocabulary is missing the <OOV> token."
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

                        index = int(np.argmax(probabilities))
                        raw_category = str(
                            encoder.inverse_transform([index])[0]
                        )

                        category = CATEGORY_NAMES.get(
                            raw_category,
                            raw_category.replace("_", " ").title()
                        )

                        confidence = float(
                            probabilities[index] * 100
                        )

                        save_complaint(
                            complaint,
                            category,
                            confidence
                        )

                        st.session_state["last_result"] = {
                            "complaint": complaint,
                            "category": category,
                            "confidence": confidence,
                            "probabilities": probabilities.tolist(),
                            "labels": [
                                CATEGORY_NAMES.get(
                                    str(x),
                                    str(x).replace("_", " ").title()
                                )
                                for x in encoder.classes_
                            ]
                        }

                st.success("Complaint classified and saved to history!")

            except Exception as e:
                st.error("Could not classify the complaint.")
                st.code(str(e))

    result = st.session_state.get("last_result")

    if result:
        st.divider()
        st.subheader("✅ Prediction Result")
        st.success(f"Predicted Category: {result['category']}")

        st.metric(
            "Model Confidence",
            f"{result['confidence']:.2f}%"
        )

        if result["confidence"] < 60:
            st.warning(
                "The model is less confident. Please review "
                "the prediction before acting on it."
            )

        score_data = pd.DataFrame({
            "Category": result["labels"],
            "Probability (%)": (
                np.array(result["probabilities"]) * 100
            )
        }).sort_values("Probability (%)")

        fig = px.bar(
            score_data,
            x="Probability (%)",
            y="Category",
            orientation="h",
            template="plotly_dark",
            title="Category Probability Scores"
        )
        st.plotly_chart(fig, use_container_width=True)


# ====================================================
# PAGE 3: ANALYTICS & INSIGHTS
# ====================================================

elif page == "📈 Analytics & Insights":

    st.title("📈 Analytics & Insights")
    st.write("Explore the evaluation results of the trained models.")

    st.subheader("Model Accuracy")

    fig = px.bar(
        comparison,
        x="Model",
        y="Test Accuracy (%)",
        color="Model",
        text="Test Accuracy (%)",
        template="plotly_dark"
    )
    fig.update_layout(yaxis_range=[0, 100], showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("CNN Classification Metrics")

    st.caption(
        "These are the saved test-set evaluation results "
        "from the trained CNN model."
    )

    st.dataframe(
        category_metrics,
        use_container_width=True,
        hide_index=True
    )

    chart_data = category_metrics.melt(
        id_vars="Category",
        value_vars=["Precision (%)", "Recall (%)", "F1 Score (%)"],
        var_name="Metric",
        value_name="Score (%)"
    )

    fig2 = px.bar(
        chart_data,
        x="Category",
        y="Score (%)",
        color="Metric",
        barmode="group",
        template="plotly_dark",
        title="CNN Performance by Category"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "The CNN achieved 88.34% test accuracy in the "
        "recorded model evaluation."
    )


# ====================================================
# PAGE 4: COMPLAINT HISTORY
# ====================================================

elif page == "🗂️ Complaint History":

    st.title("🗂️ Complaint History")
    st.write("Previously classified complaints are listed here.")

    history = get_history()

    if history.empty:
        st.info(
            "No saved complaints yet. Visit Complaint Analyzer "
            "and classify your first complaint."
        )

    else:
        c1, c2 = st.columns(2)
        c1.metric("Saved Complaints", len(history))
        c2.metric(
            "Categories Found",
            history["category"].nunique()
        )

        search_text = st.text_input(
            "🔎 Search complaint text"
        )

        all_categories = sorted(history["category"].unique())
        selected_categories = st.multiselect(
            "Filter by category",
            options=all_categories,
            default=all_categories
        )

        filtered = history[
            history["category"].isin(selected_categories)
        ]

        if search_text.strip():
            filtered = filtered[
                filtered["complaint"].str.contains(
                    search_text,
                    case=False,
                    na=False,
                    regex=False
                )
            ]

        st.dataframe(
            filtered[
                ["id", "created_at", "complaint",
                 "category", "confidence"]
            ],
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Export History as CSV",
            data=csv_data,
            file_name="complaint_history.csv",
            mime="text/csv"
        )

        if not filtered.empty:
            st.subheader("View a Saved Complaint")

            selected_id = st.selectbox(
                "Choose complaint ID",
                filtered["id"].tolist()
            )

            selected = filtered[
                filtered["id"] == selected_id
            ].iloc[0]

            st.markdown("#### Complaint")
            st.write(selected["complaint"])

            st.markdown("#### Predicted Category")
            st.success(selected["category"])

            st.metric(
                "Saved Confidence",
                f"{selected['confidence']:.2f}%"
            )

            st.caption(f"Saved at: {selected['created_at']}")


# ====================================================
# PAGE 5: VOICE ASSISTANT & HELP
# ====================================================

elif page == "🎙️ Voice Assistant & Help":

    st.title("🎙️ Voice Assistant & Help")

    st.write(
        "Your guide to using the Complaint Intelligence website."
    )

    voice_player(
        WELCOME_TEXT,
        "🔊 Listen to Welcome Message"
    )

    instructions = (
        "First, open the Complaint Analyzer page. "
        "Enter the customer complaint in the text box. "
        "Click Classify Complaint. Read the predicted category "
        "and model confidence. Your result will be saved in "
        "Complaint History. Open the Analytics page to explore "
        "model performance. Please avoid entering sensitive "
        "personal information."
    )

    st.subheader("📖 How to use this website")

    steps = [
        "Open Complaint Analyzer.",
        "Enter the customer's complaint.",
        "Click Classify Complaint.",
        "Review the predicted category and confidence.",
        "Open Complaint History to find saved results.",
        "Use Analytics & Insights to review model performance."
    ]

    for i, step in enumerate(steps, start=1):
        st.markdown(f"**Step {i}.** {step}")

    voice_player(
        instructions,
        "🔊 Listen to Instructions"
    )

    st.divider()
    st.subheader("📚 Supported Categories")

    for name in CATEGORY_NAMES.values():
        st.markdown(f"- {name}")

    st.warning(
        "This is an academic AI prototype. Predictions may be "
        "incorrect and should be reviewed before making decisions."
    )


# ---------------- FOOTER ----------------

st.divider()
st.caption(
    "Complaint Intelligence | Academic Project Prototype | "
    "CNN Test Accuracy: 88.34%"
)
