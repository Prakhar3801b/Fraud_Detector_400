# app.py
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils import load_and_clean

from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
import io
import os

# -----------------------
# CONFIG / KEYS
# -----------------------
# Put your Gemini key here (or load from env)
genai.configure(api_key="")

GEMINI_CHAT_MODEL = "gemini-2.0-flash"
GEMINI_ANALYSIS_MODEL = "gemini-2.0-pro"

# -----------------------
# STREAMLIT PAGE CONFIG
# -----------------------
st.set_page_config(page_title="FD-400 Fraud Detection", layout="wide", initial_sidebar_state="collapsed")

# -----------------------
# UI THEME CSS (safe)
# -----------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg,#0A0F1F,#121C36,#0B2942); color:white; }
.title-center { text-align:center; font-size:78px; font-weight:800; color:#F8D574; margin-top:-10px; letter-spacing:2px; text-shadow:0 0 12px rgba(255,215,130,0.6); }
.glass-box { background:rgba(255,255,255,0.07); border-radius:18px; padding:18px; margin-top:16px; backdrop-filter:blur(10px); border:1px solid rgba(255,255,255,0.12); }
.fade-in { animation: fadeIn 0.75s ease; } @keyframes fadeIn { from {opacity:0;} to {opacity:1;} }
[data-testid="stSidebar"] { background:rgba(0,0,0,0.40); backdrop-filter:blur(14px); }
.stTextInput>div>div>input { background-color:#121827; color:white; border-radius:10px; padding:10px; }
.stButton>button { background:linear-gradient(90deg,#ff9966,#ff5e62); border:none; padding:10px 16px; color:white; font-size:15px; border-radius:10px; transition:0.12s; } .stButton>button:hover { transform:scale(1.03); }
.chatbox-container { transition:max-height .65s ease, opacity .55s ease; overflow:hidden; max-height:0; opacity:0; } .chatbox-container.open { max-height:900px; opacity:1; }
.top-right { display:flex; justify-content:flex-end; align-items:center; } .small-note { font-size:12px; color:#cfcfcf; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# CACHING HELPERS
# -----------------------
@st.cache_data
def load_base_data(path="synthetic_dataset_balanced.csv", max_rows=500_000):
    """Load base csv used for EDA. Downsample if extremely large."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # If extremely large, downsample for UI/EDA
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    return df

@st.cache_resource
def load_trained_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

# -----------------------
# UTILITY: retrain_from_file
# Accepts a file path string OR a Streamlit UploadedFile
# -----------------------
def retrain_from_file(filelike, sample_limit=300_000):
    """
    filelike: file path (str) or uploaded file-like object.
    returns: (model, df_used_for_training, classification_report_string)
    """
    # load df
    if isinstance(filelike, str):
        df_up = pd.read_csv(filelike)
    else:
        # streamlit uploaded file (io.BytesIO)
        filelike.seek(0)
        df_up = pd.read_csv(filelike)

    # downsample huge datasets for training
    if len(df_up) > sample_limit:
        df_used = df_up.sample(sample_limit, random_state=42)
    else:
        df_used = df_up

    # get X,y via provided load_and_clean - assumes it accepts path or filelike
    X_new, y_new = load_and_clean(filelike)
    # if X_new huge, sample with same indices to keep shape consistent
    if hasattr(X_new, "shape") and X_new.shape[0] > sample_limit:
        X_new = X_new.sample(sample_limit, random_state=42)
        y_new = y_new.loc[X_new.index]

    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_new, y_new)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    preds = model.predict(X_new)
    rpt = classification_report(y_new, preds, digits=4)
    return model, df_used, rpt

# -----------------------
# PDF Generation (P2)
# -----------------------
def generate_pdf_report(df_pred, max_table_rows=20):
    """Takes df_pred (must include column 'fraud_pred') and returns a BytesIO buffer with the PDF."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Title page + summary
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111); ax.axis("off")
        ax.text(0.5, 0.9, "FD-400 — Fraud Report", ha="center", fontsize=20, weight="bold")
        ax.text(0.5, 0.86, f"Generated: {timestamp}", ha="center", fontsize=10)
        total = len(df_pred)
        fraud_count = int(df_pred["fraud_pred"].sum()) if "fraud_pred" in df_pred.columns else 0
        fraud_pct = 100.0 * fraud_count / total if total > 0 else 0.0
        ax.text(0.5, 0.78, f"Total transactions: {total}", ha="center")
        ax.text(0.5, 0.75, f"Predicted frauds: {fraud_count} ({fraud_pct:.2f}%)", ha="center")
        pdf.savefig(fig); plt.close(fig)

        # Pie chart
        fig2 = plt.figure(figsize=(8, 5)); ax2 = fig2.add_subplot(111)
        if "fraud_pred" in df_pred.columns:
            counts = df_pred["fraud_pred"].value_counts().sort_index()
            counts.plot(kind="pie", autopct="%1.1f%%", labels=[str(i) for i in counts.index], colors=["#36A2EB", "#FF6384"], ax=ax2)
            ax2.set_ylabel("")
            ax2.set_title("Prediction Distribution")
        else:
            ax2.text(0.5, 0.5, "No predictions available", ha="center")
        pdf.savefig(fig2); plt.close(fig2)

        # Bar chart
        fig3 = plt.figure(figsize=(8,5)); ax3 = fig3.add_subplot(111)
        if "fraud_pred" in df_pred.columns:
            df_pred["fraud_pred"].value_counts().sort_index().plot(kind="bar", color=["#36A2EB", "#FF6384"], ax=ax3)
            ax3.set_title("Prediction Counts")
            ax3.set_xlabel("fraud_pred")
            ax3.set_ylabel("count")
        pdf.savefig(fig3); plt.close(fig3)

        # Table excerpt
        fig4 = plt.figure(figsize=(8.27, 3.5)); ax4 = fig4.add_subplot(111); ax4.axis("off")
        if len(df_pred) > 0:
            excerpt = df_pred.head(max_table_rows).astype(str)
            table = ax4.table(cellText=excerpt.values, colLabels=excerpt.columns, loc="center", cellLoc="left")
            table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1, 1.2)
        else:
            ax4.text(0.5, 0.5, "No data", ha="center")
        pdf.savefig(fig4); plt.close(fig4)
    buf.seek(0)
    return buf

# -----------------------
# Load base EDA data (cached) - downsample if extremely large
# -----------------------
base_df = load_base_data()

# -----------------------
# HEADER + CHATBOX + AUTORETRAIN UI
# -----------------------
st.markdown("<h1 class='title-center fade-in'>FD-400</h1>", unsafe_allow_html=True)

col_chat, col_controls = st.columns([3, 1])

# Chat column
with col_chat:
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = True
    toggle = st.button("💬 Hide Chatbox" if st.session_state.chat_open else "💬 Show Chatbox")
    if toggle:
        st.session_state.chat_open = not st.session_state.chat_open

    chat_class = "chatbox-container open" if st.session_state.chat_open else "chatbox-container"
    st.markdown(f"<div class='{chat_class} glass-box fade-in'>", unsafe_allow_html=True)
    st.subheader("🤖 FraudGuard Chat Assistant (Gemini)")

    # create chat model -- wrap in try/except to avoid app-breaking if key or model is invalid
    try:
        chat_model = genai.GenerativeModel(GEMINI_CHAT_MODEL)
    except Exception as e:
        chat_model = None
        st.warning("Gemini client init failed (check API key).")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_query = st.text_input("Ask anything about fraud, ML, patterns or banking:", key="chat_input")
    if user_query:
        if chat_model:
            try:
                resp = chat_model.generate_content(user_query)
                ai_text = resp.text
            except Exception as e:
                ai_text = f"[Gemini error: {e}]"
        else:
            ai_text = "[Gemini not available]"
        st.session_state["chat_history"].append(("You", user_query))
        st.session_state["chat_history"].append(("FD-400 AI", ai_text))

    for who, msg in st.session_state["chat_history"]:
        st.write(f"**{who}:** {msg}")
    st.markdown("</div>", unsafe_allow_html=True)

# Controls column (auto retrain)
with col_controls:
    st.markdown("<div class='glass-box fade-in'>", unsafe_allow_html=True)
    st.write("### ⏱ Auto-Retrain")
    if "auto_enabled" not in st.session_state:
        st.session_state.auto_enabled = False
    if "auto_freq" not in st.session_state:
        st.session_state.auto_freq = "On upload"
    if "auto_next" not in st.session_state:
        st.session_state.auto_next = None

    st.session_state.auto_enabled = st.checkbox("Enable Auto-Retrain", value=st.session_state.auto_enabled)
    freq = st.selectbox("Frequency", ["On refresh", "On upload", "Daily", "Weekly"],
                        index=["On refresh", "On upload", "Daily", "Weekly"].index(st.session_state.auto_freq) if st.session_state.auto_freq in ["On refresh","On upload","Daily","Weekly"] else 1)
    st.session_state.auto_freq = freq
    st.markdown("<div class='small-note'>Auto-retrain triggers when conditions meet (checked on page load).</div>", unsafe_allow_html=True)

    if st.button("Retrain Now"):
        if os.path.exists("uploaded_for_retrain.csv"):
            try:
                model_obj, df_up, rpt = retrain_from_file("uploaded_for_retrain.csv")
                st.success("Manual retrain finished.")
                st.text(rpt)
            except Exception as e:
                st.error(f"Retrain failed: {e}")
        else:
            st.error("No uploaded_for_retrain.csv found. Upload in Retrain Model section first.")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Sidebar menu (unchanged)
# -----------------------
st.sidebar.title("FD-400 Menu")
choice = st.sidebar.radio("Choose an option:", ["📊 View Graphs", "🔁 Retrain Model", "🧮 Predict Fraud"])

# -----------------------
# VIEW GRAPHS
# -----------------------
if choice == "📊 View Graphs":
    df = base_df  # cached and downsampled if huge
    st.markdown("<div class='glass-box fade-in'>", unsafe_allow_html=True)
    st.subheader("📊 Dataset Graphs")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Fraud Distribution")
        fig1, ax1 = plt.subplots()
        if not df.empty and "isFraud" in df.columns:
            df["isFraud"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["#FF6384", "#36A2EB"], ax=ax1)
            ax1.set_ylabel("")
        else:
            ax1.text(0.5, 0.5, "No data", ha="center")
        st.pyplot(fig1)

    with col2:
        st.write("### Transaction Amount Distribution")
        fig2, ax2 = plt.subplots()
        if not df.empty and "amount" in df.columns:
            # sample for plotting if too many rows
            sample_rows = 50_000
            plot_series = df["amount"] if len(df) <= sample_rows else df["amount"].sample(sample_rows, random_state=42)
            sns.histplot(plot_series, bins=50, kde=True, color="#4FC3F7", ax=ax2)
        else:
            ax2.text(0.5, 0.5, "No data", ha="center")
        st.pyplot(fig2)

    st.write("### Correlation Heatmap ")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    if not df.empty:
        numeric = df.select_dtypes(include=["number"])
        if numeric.shape[1] == 0:
            ax3.text(0.5, 0.5, "No numeric columns", ha="center")
        else:
            top_cols = numeric.columns[:20]
            sample_rows = 50_000
            safe_numeric = numeric[top_cols] if len(numeric) <= sample_rows else numeric[top_cols].sample(sample_rows, random_state=42)
            sns.heatmap(safe_numeric.corr(numeric_only=True), cmap="RdBu_r", annot=False, ax=ax3)
    else:
        ax3.text(0.5, 0.5, "No data", ha="center")
    st.pyplot(fig3)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# RETRAIN MODEL
# -----------------------
elif choice == "🔁 Retrain Model":
    st.markdown("<div class='glass-box fade-in'>", unsafe_allow_html=True)
    st.subheader("🔁 Retrain the Random Forest Model")

    uploaded = st.file_uploader("Upload training CSV", type=["csv"], key="retrain_upload")
    if uploaded is not None:
        # save copy for auto retrain
        with open("uploaded_for_retrain.csv", "wb") as f:
            f.write(uploaded.getbuffer())

    if st.button("Train Model"):
        if uploaded is None:
            st.error("Please upload a CSV first.")
        else:
            try:
                model_obj, df_used, rpt = retrain_from_file(uploaded)
                st.success("Model retrained and saved to model.pkl")
                st.write("### Training metrics")
                st.text(rpt)
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# PREDICT FRAUD
# -----------------------
elif choice == "🧮 Predict Fraud":
    st.markdown("<div class='glass-box fade-in'>", unsafe_allow_html=True)
    st.subheader("🧮 Predict Fraudulent Transactions")

    file_pred = st.file_uploader("Upload CSV for prediction", type=["csv"], key="predict_upload")

    if file_pred and st.button("Run Prediction"):
        try:
            # Load original (but do not process entire massive file for plotting)
            df_pred = pd.read_csv(file_pred)
            # downsample for UI if huge (we still predict on entire file below if memory allows)
            ui_preview = df_pred.head(10) if len(df_pred) > 10000 else df_pred
            st.dataframe(ui_preview)

            # load trained model (cached)
            model = load_trained_model()
            if model is None:
                st.error("Train a model first!")
            else:
                # run prediction using your load_and_clean pipeline
                # load_and_clean should accept file-like or path as before
                X_pred, _ = load_and_clean(file_pred)
                # if the feature matrix is extremely large, sample for speed - but predictions are row-wise and should be done on real X_pred
                # We will run predictions on entire X_pred if possible; if memory error occurs, sample and warn.
                try:
                    preds = model.predict(X_pred)
                except MemoryError:
                    st.warning("MemoryError during prediction on full file — sampling and predicting a subset for UI.")
                    sample_idx = X_pred.sample(100000, random_state=42).index if X_pred.shape[0] > 100000 else X_pred.index
                    preds = model.predict(X_pred.loc[sample_idx])
                    # map back to a smaller df for UI
                    df_pred = df_pred.loc[sample_idx]

                df_pred["fraud_pred"] = preds
                # save last prediction for download
                df_pred.to_csv("last_prediction.csv", index=False)
                st.write("### Results")
                st.dataframe(df_pred.head(50))

                # small distribution plot (safe sampling)
                fig4, ax4 = plt.subplots()
                counts = df_pred["fraud_pred"].value_counts().sort_index()
                counts.plot(kind="bar", color=["#36A2EB", "#FF6384"], ax=ax4)
                ax4.set_xlabel("fraud_pred")
                st.pyplot(fig4)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# AUTO-RETRAIN TRIGGER (checked on page load)
# -----------------------
try:
    if st.session_state.get("auto_enabled", False):
        now = datetime.now()
        freq = st.session_state.get("auto_freq", "On upload")
        next_run = st.session_state.get("auto_next", None)

        # On refresh: attempt retrain once per refresh (if uploaded file exists)
        if freq == "On refresh" and os.path.exists("uploaded_for_retrain.csv"):
            try:
                retrain_from_file("uploaded_for_retrain.csv")
                st.success("Auto-retrain (on refresh) executed.")
            except Exception as e:
                st.warning(f"Auto-retrain failed: {e}")

        # On upload: retrain when file modification changes
        if freq == "On upload" and os.path.exists("uploaded_for_retrain.csv"):
            mtime = os.path.getmtime("uploaded_for_retrain.csv")
            last = st.session_state.get("auto_last_mtime", None)
            if last is None or mtime != last:
                try:
                    retrain_from_file("uploaded_for_retrain.csv")
                    st.success("Auto-retrain (on upload) executed.")
                    st.session_state["auto_last_mtime"] = mtime
                except Exception as e:
                    st.warning(f"Auto-retrain failed: {e}")

        # Daily/Weekly: schedule based on next_run
        if freq in ["Daily", "Weekly"]:
            if st.session_state.get("auto_next") is None:
                if freq == "Daily":
                    st.session_state["auto_next"] = now + timedelta(days=1)
                else:
                    st.session_state["auto_next"] = now + timedelta(weeks=1)
            elif now >= st.session_state["auto_next"]:
                if os.path.exists("uploaded_for_retrain.csv"):
                    try:
                        retrain_from_file("uploaded_for_retrain.csv")
                        st.success(f"Auto-retrain ({freq}) executed.")
                        # push next
                        st.session_state["auto_next"] = now + (timedelta(days=1) if freq == "Daily" else timedelta(weeks=1))
                    except Exception as e:
                        st.warning(f"Auto-retrain failed: {e}")
except Exception:
    # swallow auto-retrain exceptions so app doesn't crash
    pass

# -----------------------
# DOWNLOADS (PDF + CSV) in sidebar
# -----------------------
if os.path.exists("last_prediction.csv"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Download Reports")
    # CSV download
    with open("last_prediction.csv", "rb") as f:
        csv_bytes = f.read()
    st.sidebar.download_button(label="Download CSV (predictions)", data=csv_bytes, file_name="fraud_predictions.csv", mime="text/csv")

    # PDF generation + download
    if st.sidebar.button("Generate PDF Report (P2)"):
        try:
            df_for_pdf = pd.read_csv("last_prediction.csv")
            pdf_buf = generate_pdf_report(df_for_pdf)
            st.sidebar.download_button(label="Download PDF Report", data=pdf_buf, file_name="fraud_report.pdf", mime="application/pdf")
            st.sidebar.success("PDF ready — click download.")
        except Exception as e:
            st.sidebar.error(f"PDF generation failed: {e}")

# -----------------------
# END
# -----------------------
