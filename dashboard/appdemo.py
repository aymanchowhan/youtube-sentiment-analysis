import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from dashboard_styles import inject_css, sec_hdr
from dashboard_data import load
import dashboard_tabs as tabs
from dashboard_live import render_tab_live

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Sentiment Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_css()

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df, topic_df, model_df, summary_df = load()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 18px'>
        <div style='font-size:22px;font-weight:700;color:#D4FF3D;font-family:"Space Grotesk",sans-serif'>⚡ YT Sentiment</div>
        <div style='font-size:11px;color:#9494B8;letter-spacing:.04em'>AI ANALYTICS DASHBOARD</div>
    </div>
    <hr style='border-top:1px solid rgba(255,255,255,0.08);margin:0 0 16px'>
    """, unsafe_allow_html=True)

    st.markdown("**Filters**")
    sel_video = st.selectbox("Video", ["All Videos"] + sorted(df["video"].unique().tolist()))
    sel_sent  = st.selectbox("Sentiment", ["All"] + sorted(df["vader_sentiment"].dropna().unique().tolist()))
    sel_emo   = st.selectbox("Emotion",   ["All"] + sorted(df["emotion"].dropna().unique().tolist()))

    st.markdown("<hr style='border-top:1px solid rgba(255,255,255,0.08);margin:16px 0'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:12px;color:#9494B8;line-height:2.2;font-family:"JetBrains Mono",monospace'>
        Comments &nbsp;<span style='color:#D4FF3D;font-weight:600'>{len(df):,}</span><br>
        Videos &nbsp;<span style='color:#D4FF3D;font-weight:600'>{df['video'].nunique()}</span><br>
        Models &nbsp;<span style='color:#D4FF3D;font-weight:600'>3</span><br>
        Emotions &nbsp;<span style='color:#D4FF3D;font-weight:600'>7</span>
    </div>
    <hr style='border-top:1px solid rgba(255,255,255,0.08);margin:16px 0'>
    <div style='font-size:11px;color:#9494B8;line-height:1.8'>
        <span style='color:#F1F1FA;font-weight:600;font-size:12px'>Ayman Chowhan</span><br>
        M.Sc. Information Technology<br>Semester IV · YouTube NLP Pipeline
    </div>
    """, unsafe_allow_html=True)

# Apply filters
fdf = df.copy()
if sel_video != "All Videos": fdf = fdf[fdf["video"] == sel_video]
if sel_sent  != "All":        fdf = fdf[fdf["vader_sentiment"] == sel_sent]
if sel_emo   != "All":        fdf = fdf[fdf["emotion"] == sel_emo]
total = len(fdf)

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>⚡ YouTube Comment Sentiment Analysis</h1>
    <p>
        <span class='hero-pill'>RoBERTa</span>
        <span class='hero-pill'>Emotion Detection</span>
        <span class='hero-pill'>Topic Modeling</span>
        <span class='hero-pill'>Spam Detection</span>
        <span class='hero-pill'>Sarcasm Detection</span>
        <span class='hero-pill'>10,000 Comments</span>
        <span class='hero-pill'>10 Videos</span>
        <span class='hero-pill'>Real-Time Analytics</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab_overview, tab_sentiment, tab_emotion, tab_models, tab_video, tab_topics, tab_wordcloud, tab_quality, tab_explorer, tab_live = st.tabs([
    "📊 Overview", "😊 Sentiment", "❤️ Emotion", "🤖 Models",
    "🎬 Per-Video", "🧠 Topics", "☁️ Word Cloud",
    "🛡️ Quality", "🔍 Explorer", "⚡ Live Analysis"
])

with tab_overview:  tabs.render_tab_overview(fdf, total)
with tab_sentiment: tabs.render_tab_sentiment(fdf, total)
with tab_emotion:   tabs.render_tab_emotion(fdf)
with tab_models:    tabs.render_tab_models(model_df)
with tab_video:     tabs.render_tab_video(df)
with tab_topics:    tabs.render_tab_topics(summary_df, sel_video)
with tab_wordcloud: tabs.render_tab_wordcloud(fdf)
with tab_quality:   tabs.render_tab_quality(fdf)
with tab_explorer:  tabs.render_tab_explorer(fdf)
with tab_live:      render_tab_live(df)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class='footer'>
    YouTube Comment Sentiment Analysis &nbsp;·&nbsp;
    M.Sc. Information Technology &nbsp;·&nbsp;
    VADER · BERT · RoBERTa · Emotion · BERTopic · Spam · Sarcasm · Flask · MLflow · Docker
</div>
""", unsafe_allow_html=True)