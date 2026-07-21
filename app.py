

import streamlit as st                              # web app framework
import pandas as pd                                   # loads CSV data + builds the history table for export
import numpy as np                                    # used for the softmax confidence calculation
import joblib                                         # loads .pkl model/vectorizer files
from datetime import datetime                         # timestamps each prediction in the history log
 
from utils import predict_sentiment                  # our function: comment -> (prediction, raw_scores)
from charts import (                                  # our chart-drawing functions
    render_dataset_stats,
    render_model_performance,
    render_top_words,
    LABEL_MAP,
    COLOR_MAP,
    ORDER,
)
 
# ==========================
# Page Settings
# ==========================
st.set_page_config(
    page_title="Gun Control Discourse Analysis",
    page_icon="🔫",
    layout="wide"
)
 
# ==========================
# Custom CSS
# (styling now lives in assets/style.css instead of inline in this file —
# this function reads that file's text and injects it as a <style> block)
# ==========================
def load_css(css_path):
    with open(css_path) as f:                             # open the CSS file in read mode
        css_text = f.read()                                 # read its entire contents as one string
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)
    # ^ wrap the raw CSS text in <style> tags and render it; unsafe_allow_html
    #   is required because Streamlit escapes HTML/CSS by default
 
load_css("assets/style.css") 

                           # actually load and apply our stylesheet
 
# ==========================
# Load Data + Model (cached so this only runs once, not on every rerun)
# ==========================
@st.cache_data
def load_data():
    return pd.read_csv("data/Reddit_Data.csv").dropna(subset=["clean_comment"])
 
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("models/svm_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer
 
df = load_data()
model, vectorizer = load_model_and_vectorizer()
 
# ==========================
# Session State Setup (prediction history persists across reruns)
# ==========================
if "history" not in st.session_state:
    st.session_state.history = []
# ==========================
 

# ==========================
# Hero Header
# ==========================
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🔫 Gun Control Discourse Analysis</p>
    <p class="hero-subtitle">NLP pipeline · PRAW scraping → TF-IDF + Linear SVM → EMPATH tone analysis</p>
</div>
""", unsafe_allow_html=True)
 
# ==========================
# Tabbed Navigation
# (splits the long single-page scroll into focused sections —
# reads as a proper app rather than a stacked report)
# ==========================
tab_overview, tab_model, tab_words, tab_predict, tab_history = st.tabs(
    ["📊 Overview", "🎯 Model performance", "🔑 Word insights", "💬 Predict", "📝 History"]
)
 
with tab_overview:
    st.markdown('<p class="section-header">Dataset statistics</p>', unsafe_allow_html=True)
    render_dataset_stats(df)
 
with tab_model:
    st.markdown('<p class="section-header">Model performance</p>', unsafe_allow_html=True)
    render_model_performance(model, vectorizer, df)
 
with tab_words:
    st.markdown('<p class="section-header">Top words per sentiment</p>', unsafe_allow_html=True)
    render_top_words(model, vectorizer)
 
with tab_predict:
    st.markdown('<p class="section-header">Sentiment prediction</p>', unsafe_allow_html=True)
 
    comment = st.text_area(
        "Enter a Reddit comment",
        placeholder="Type or paste a comment about gun control...",
        height=100,
    )
 
    if st.button("Analyze sentiment", type="primary"):
 
        if comment.strip() == "":
            st.warning("⚠ Please enter a comment.")
 
        else:
            prediction, raw_scores = predict_sentiment(comment)
 
            scores = np.array(raw_scores).flatten()
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / exp_scores.sum()
            prob_by_class = dict(zip(model.classes_, probabilities))
 
            label = LABEL_MAP[prediction]
            color = COLOR_MAP[prediction]
 
            st.markdown(
                f"""
                <div style="border-left:6px solid {color}; padding:12px 18px;
                            border-radius:0 10px 10px 0; background:rgba(255,255,255,0.05);
                            margin:14px 0; box-shadow:0 4px 12px rgba(0,0,0,0.25);">
                    <span style="font-weight:700; font-size:1.05rem; color:{color};">{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
            st.markdown("**Confidence breakdown**")
            for k in ORDER:
                pct = prob_by_class[k] * 100
                bar_col, label_col = st.columns([6, 1])
                with bar_col:
                    st.markdown(
                        f"""
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="width:60px; font-size:13px; color:#94a3b8;">{LABEL_MAP[k]}</span>
                            <div style="flex:1; background:#2a2a2a; border-radius:6px; height:16px;">
                                <div style="width:{pct:.1f}%; background:{COLOR_MAP[k]};
                                            height:16px; border-radius:6px;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with label_col:
                    st.write(f"{pct:.0f}%")
 
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "comment": comment,
                "prediction": label,
                "positive_%": round(prob_by_class[1] * 100, 1),
                "neutral_%": round(prob_by_class[0] * 100, 1),
                "negative_%": round(prob_by_class[-1] * 100, 1),
            })
 
with tab_history:
    st.markdown('<p class="section-header">Prediction history</p>', unsafe_allow_html=True)
 
    if len(st.session_state.history) == 0:
        st.info("No predictions yet — analyze a comment in the Predict tab to start building your history.")
 
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
 
        csv_data = history_df.to_csv(index=False).encode("utf-8")
 
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.download_button(
                label="⬇ Download predictions as CSV",
                data=csv_data,
                file_name="sentiment_predictions.csv",
                mime="text/csv",
            )
        with col_b:
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()
 
st.divider()
st.caption("Built with Streamlit, scikit-learn, and Plotly · Data scraped via PRAW")
 
