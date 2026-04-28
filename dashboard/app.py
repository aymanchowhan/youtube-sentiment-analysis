import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="YT Sentiment Lab",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CSS — LIGHT MINIMAL THEME
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after {
    font-family: 'DM Sans', sans-serif !important;
    box-sizing: border-box;
}

/* App background */
.stApp {
    background: #f7f7f5 !important;
}

/* Main content */
.main .block-container {
    background: transparent !important;
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8e8e4 !important;
}
section[data-testid="stSidebar"] * {
    color: #1a1a1a !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMarkdown p {
    color: #555 !important;
    font-size: 13px !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e8e8e4 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    transition: box-shadow 0.2s !important;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
}
[data-testid="metric-container"] label {
    color: #888 !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] {
    color: #1a1a1a !important;
    font-size: 28px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] {
    color: #555 !important;
    font-size: 12px !important;
}

/* Inputs */
.stTextInput input {
    background: #ffffff !important;
    border: 1px solid #e8e8e4 !important;
    border-radius: 8px !important;
    color: #1a1a1a !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
}
.stTextInput input:focus {
    border-color: #1a1a1a !important;
    box-shadow: 0 0 0 2px rgba(26,26,26,0.08) !important;
}

/* Buttons */
.stButton > button {
    background: #1a1a1a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #333 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #e8e8e4 !important;
    border-radius: 8px !important;
    color: #1a1a1a !important;
}

/* Radio */
.stRadio label { color: #555 !important; font-size: 13px !important; }

/* Slider */
.stSlider label { color: #555 !important; }

/* Checkbox */
.stCheckbox label { color: #555 !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #e8e8e4 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f7f7f5; }
::-webkit-scrollbar-thumb { background: #d0d0cc; border-radius: 3px; }

/* Section headers */
.sec-hdr {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 32px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1.5px solid #1a1a1a;
}
.sec-hdr-text {
    font-size: 15px;
    font-weight: 600;
    color: #1a1a1a;
    letter-spacing: -0.01em;
}
.sec-hdr-count {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px;
    color: #888;
    margin-left: auto;
}

/* Cards */
.card {
    background: #ffffff;
    border: 1px solid #e8e8e4;
    border-radius: 12px;
    padding: 20px 22px;
    margin: 8px 0;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    margin: 2px;
}
.badge-pos { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
.badge-neg { background: #fce4ec; color: #c62828; border: 1px solid #f8bbd0; }
.badge-neu { background: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }
.badge-spam { background: #fff3e0; color: #e65100; border: 1px solid #ffe0b2; }
.badge-sarc { background: #f3e5f5; color: #6a1b9a; border: 1px solid #e1bee7; }
.badge-fake { background: #fafafa; color: #424242; border: 1px solid #e0e0e0; }

/* Hero */
.hero {
    background: #1a1a1a;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
    color: #ffffff;
}
.hero h1 {
    font-size: 26px;
    font-weight: 600;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -0.02em;
}
.hero p {
    font-size: 13px;
    color: #999;
    margin: 0;
    line-height: 1.8;
}
.hero-pill {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    color: #ccc;
    margin: 2px;
}

/* Insight box */
.insight {
    background: #fafafa;
    border: 1px solid #e8e8e4;
    border-left: 3px solid #1a1a1a;
    border-radius: 0 10px 10px 0;
    padding: 14px 16px;
    font-size: 13px;
    color: #444;
    line-height: 1.7;
    margin: 8px 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #bbb;
    font-size: 11px;
    padding: 32px 0 16px;
    border-top: 1px solid #e8e8e4;
    margin-top: 48px;
    font-family: 'DM Mono', monospace !important;
}

/* Real-time status */
.rt-status {
    background: #f7f7f5;
    border: 1px solid #e8e8e4;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 13px;
    color: #666;
    margin: 12px 0;
    font-family: 'DM Mono', monospace !important;
}

hr { border: none; border-top: 1px solid #e8e8e4 !important; margin: 8px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────
BG       = "#ffffff"
GRID     = "#f0f0ee"
TEXT     = "#555555"
ACCENT   = "#1a1a1a"
COLOR_SENTIMENT = {
    "positive": "#4caf50",
    "negative": "#ef5350",
    "neutral":  "#42a5f5"
}
PALETTE = ["#1a1a1a","#555","#888","#4caf50","#ef5350",
           "#42a5f5","#ab47bc","#ff7043","#26a69a","#ffa726"]

def theme(fig, title=""):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font_color=TEXT,
        font_family="DM Sans",
        title=dict(text=title, font=dict(size=13, color=ACCENT,
                   family="DM Sans"), x=0.01),
        margin=dict(t=44, b=28, l=16, r=16),
        legend=dict(bgcolor=BG, bordercolor=GRID,
                    borderwidth=1, font=dict(color=TEXT, size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID,
                     tickfont=dict(color=TEXT, size=11))
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID,
                     tickfont=dict(color=TEXT, size=11))
    return fig

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load():
    df         = pd.read_csv("data/final_dataset.csv")
    topic_df   = pd.read_csv("data/topic_results.csv")
    model_df   = pd.read_csv("data/model_comparison.csv")
    summary_df = pd.read_csv("data/topic_summary.csv")
    for col in ["vader_sentiment","bert_sentiment","roberta_sentiment"]:
        df[col] = df[col].astype(str).str.lower().str.strip()
    try:
        df["spam_label"] = pd.read_csv("data/spam_results.csv")["spam_label"].values
    except: df["spam_label"] = "not_spam"
    try:
        df["sarcasm_label"] = pd.read_csv("data/sarcasm_results.csv")["sarcasm_label"].values
    except: df["sarcasm_label"] = "not_sarcastic"
    try:
        df["fake_label"] = pd.read_csv("data/fake_comment_results.csv")["fake_label"].values
    except: df["fake_label"] = "real"
    return df, topic_df, model_df, summary_df

df, topic_df, model_df, summary_df = load()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 12px'>
        <div style='font-size:20px;font-weight:600;color:#1a1a1a'>▶ YT Sentiment</div>
        <div style='font-size:11px;color:#888;margin-top:3px;font-family:DM Mono,monospace'>NLP Research Dashboard</div>
    </div>
    <hr style='border-top:1px solid #e8e8e4;margin:0 0 16px'>
    """, unsafe_allow_html=True)

    st.markdown("**Filters**")
    videos = ["All Videos"] + sorted(df["video"].unique().tolist())
    sel_video = st.selectbox("Video", videos)
    sentiments = ["All"] + sorted(df["vader_sentiment"].dropna().unique().tolist())
    sel_sent = st.selectbox("Sentiment", sentiments)
    emotions = ["All"] + sorted(df["emotion"].dropna().unique().tolist())
    sel_emo = st.selectbox("Emotion", emotions)

    st.markdown("<hr style='border-top:1px solid #e8e8e4;margin:16px 0'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:12px;color:#888;line-height:2'>
        Comments &nbsp;<span style='color:#1a1a1a;font-weight:500'>{len(df):,}</span><br>
        Videos &nbsp;<span style='color:#1a1a1a;font-weight:500'>{df['video'].nunique()}</span><br>
        Models &nbsp;<span style='color:#1a1a1a;font-weight:500'>3</span><br>
        Emotions &nbsp;<span style='color:#1a1a1a;font-weight:500'>7</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-top:1px solid #e8e8e4;margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px;color:#aaa;line-height:1.8'>
        M.Sc. Information Technology<br>Semester IV · YouTube NLP Pipeline
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:11px;color:#aaa;line-height:1.8'>
    <span style='color:#1a1a1a;font-weight:600;font-size:12px'>Ayman Chowhan</span><br>
    M.Sc. Information Technology<br>Semester IV · YouTube NLP Pipeline
</div>
""", unsafe_allow_html=True)

# Apply filters
fdf = df.copy()
if sel_video != "All Videos":  fdf = fdf[fdf["video"] == sel_video]
if sel_sent  != "All":         fdf = fdf[fdf["vader_sentiment"] == sel_sent]
if sel_emo   != "All":         fdf = fdf[fdf["emotion"] == sel_emo]
total = len(fdf)

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>YouTube Comment Sentiment Analysis</h1>
    <p>
        <span class='hero-pill'>VADER</span>
        <span class='hero-pill'>BERT</span>
        <span class='hero-pill'>RoBERTa</span>
        <span class='hero-pill'>Emotion Detection</span>
        <span class='hero-pill'>LDA Topic Modeling</span>
        <span class='hero-pill'>10,000 Comments</span>
        <span class='hero-pill'>10 Videos</span>
        <span class='hero-pill'>Spam · Sarcasm · Fake Detection</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SECTION 1 — KPIs
# ─────────────────────────────────────────
st.markdown('<div class="sec-hdr"><span class="sec-hdr-text">Overview Metrics</span><span class="sec-hdr-count">SECTION 01</span></div>', unsafe_allow_html=True)

pos_n  = (fdf["vader_sentiment"]=="positive").sum()
neg_n  = (fdf["vader_sentiment"]=="negative").sum()
neu_n  = (fdf["vader_sentiment"]=="neutral").sum()
spam_n = (fdf["spam_label"]=="spam").sum()
sarc_n = (fdf["sarcasm_label"]=="sarcastic").sum()
fake_n = (fdf["fake_label"]=="fake").sum()

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Total Comments",  f"{total:,}")
c2.metric("Positive",  f"{pos_n/total*100:.1f}%" if total else "0%", f"{pos_n:,} comments")
c3.metric("Negative",  f"{neg_n/total*100:.1f}%" if total else "0%", f"{neg_n:,} comments")
c4.metric("Neutral",   f"{neu_n/total*100:.1f}%" if total else "0%", f"{neu_n:,} comments")
c5.metric("Spam",      f"{spam_n/total*100:.1f}%" if total else "0%", f"{spam_n:,} flagged")
c6.metric("Sarcasm",   f"{sarc_n/total*100:.1f}%" if total else "0%", f"{sarc_n:,} detected")

# ─────────────────────────────────────────
# SECTION 2 — SENTIMENT
# ─────────────────────────────────────────
st.markdown('<div class="sec-hdr"><span class="sec-hdr-text">Sentiment Analysis</span><span class="sec-hdr-count">SECTION 02</span></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.1, 1.4, 0.9])

with col1:
    sc = fdf["vader_sentiment"].value_counts().reset_index()
    sc.columns = ["sentiment","count"]
    fig = px.pie(sc, names="sentiment", values="count", hole=0.55,
                 color="sentiment", color_discrete_map=COLOR_SENTIMENT)
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      textfont_size=12,
                      marker=dict(line=dict(color="#ffffff", width=2)))
    fig.update_layout(showlegend=False)
    theme(fig, "VADER Sentiment Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    cats = ["positive","negative","neutral"]
    vv = fdf["vader_sentiment"].value_counts()
    bv = fdf["bert_sentiment"].value_counts()
    rv = fdf["roberta_sentiment"].value_counts()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="VADER", x=cats,
        y=[vv.get(c,0) for c in cats], marker_color="#1a1a1a",
        marker_line_color="#ffffff", marker_line_width=1.5))
    fig2.add_trace(go.Bar(name="BERT", x=cats,
        y=[bv.get(c,0) for c in cats], marker_color="#888",
        marker_line_color="#ffffff", marker_line_width=1.5))
    fig2.add_trace(go.Bar(name="RoBERTa", x=cats,
        y=[rv.get(c,0) for c in cats], marker_color="#bbb",
        marker_line_color="#ffffff", marker_line_width=1.5))
    fig2.update_layout(barmode="group", xaxis_title="Sentiment", yaxis_title="Count")
    theme(fig2, "3-Model Sentiment Comparison")
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    top_emo  = fdf["emotion"].value_counts().index[0] if total else "N/A"
    top_sent = fdf["vader_sentiment"].value_counts().index[0] if total else "N/A"
    agree    = (fdf["vader_sentiment"]==fdf["roberta_sentiment"]).sum()
    agree_pct = round(agree/total*100,1) if total else 0
    st.markdown(f"""
    <div class='insight'>
        <div style='font-size:11px;font-weight:600;color:#1a1a1a;
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px'>
            Key Findings
        </div>
        Dominant sentiment<br>
        <span class='badge badge-pos'>{top_sent}</span><br><br>
        Top emotion<br>
        <span class='badge badge-neu'>{top_emo}</span><br><br>
        Best model<br>
        <span class='badge badge-pos'>RoBERTa 73.2%</span><br><br>
        Weakest model<br>
        <span class='badge badge-neg'>BERT 39.7%</span><br><br>
        Model agreement<br>
        <span class='badge badge-neu'>{agree_pct}% match</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# SECTION 3 — EMOTION
# ─────────────────────────────────────────
st.markdown('<div class="sec-hdr"><span class="sec-hdr-text">Emotion Detection</span><span class="sec-hdr-count">SECTION 03</span></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    ec = fdf["emotion"].value_counts().reset_index()
    ec.columns = ["emotion","count"]
    fig3 = px.bar(ec, x="emotion", y="count", color="emotion",
                  color_discrete_sequence=PALETTE, text="count")
    fig3.update_traces(texttemplate="%{text:,}", textposition="outside",
                       marker_line_color="#ffffff", marker_line_width=1)
    fig3.update_layout(showlegend=False, xaxis_title="Emotion", yaxis_title="Count")
    theme(fig3, "Emotion Distribution")
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    es = fdf.groupby(["emotion","vader_sentiment"]).size().reset_index(name="count")
    fig4 = px.bar(es, x="emotion", y="count", color="vader_sentiment",
                  barmode="stack", color_discrete_map=COLOR_SENTIMENT)
    fig4.update_layout(xaxis_title="Emotion", yaxis_title="Count")
    theme(fig4, "Emotion × Sentiment Breakdown")
    st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────
# SECTION 4 — MODEL PERFORMANCE
# ─────────────────────────────────────────
st.markdown('<div class="sec-hdr"><span class="sec-hdr-text">Model Performance</span><span class="sec-hdr-count">SECTION 04</span></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1.3, 1])

with col1:
    vader_row = pd.DataFrame([{"Model":"VADER (baseline)","Accuracy":1.0, model_df.columns[2]:1.0}])
    disp = pd.concat([model_df, vader_row], ignore_index=True)
    disp_show = disp.copy()
    disp_show["Accuracy"] = disp_show["Accuracy"].apply(lambda x: f"{x:.2%}")
    disp_show.iloc[:,2] = disp_show.iloc[:,2].apply(lambda x: f"{float(x):.2%}")
    st.markdown("**Evaluation Table**")
    st.dataframe(disp_show, use_container_width=True, hide_index=True, height=155)
    st.markdown("""
    <div class='insight' style='margin-top:10px;font-size:12px'>
        RoBERTa outperforms BERT by <strong>+33.5%</strong>.
        Optimised pretraining on large social media corpora
        significantly boosts classification accuracy.
    </div>
    """, unsafe_allow_html=True)

with col2:
    raw = pd.read_csv("data/model_comparison.csv")
    vr  = pd.DataFrame([{"Model":"VADER","Accuracy":1.0, raw.columns[2]:1.0}])
    pdf = pd.concat([raw, vr], ignore_index=True)
    fig5 = go.Figure()
    bar_colors = ["#1a1a1a","#888","#bbb"]
    for i, row in pdf.iterrows():
        fig5.add_trace(go.Bar(
            name=row["Model"], x=[row["Model"]], y=[row["Accuracy"]],
            marker_color=bar_colors[i % len(bar_colors)],
            marker_line_color="#ffffff", marker_line_width=1.5,
            text=f"{row['Accuracy']:.1%}", textposition="outside",
            textfont=dict(color=ACCENT, size=12)
        ))
    fig5.update_layout(showlegend=False, yaxis_tickformat=".0%",
                       yaxis_range=[0,1.2], bargap=0.3)
    theme(fig5, "Model Accuracy Comparison")
    st.plotly_chart(fig5, use_container_width=True)

with col3:
    cats2 = ["Accuracy","F1 Score","Speed","Interpretability","Social Media"]
    fig6 = go.Figure()
    fig6.add_trace(go.Scatterpolar(r=[1.0,1.0,1.0,1.0,0.8], theta=cats2,
        fill="toself", name="VADER", line_color="#1a1a1a", fillcolor="rgba(26,26,26,0.08)"))
    fig6.add_trace(go.Scatterpolar(r=[0.40,0.36,0.5,0.6,0.5], theta=cats2,
        fill="toself", name="BERT", line_color="#888", fillcolor="rgba(136,136,136,0.08)"))
    fig6.add_trace(go.Scatterpolar(r=[0.73,0.72,0.6,0.5,0.8], theta=cats2,
        fill="toself", name="RoBERTa", line_color="#555", fillcolor="rgba(85,85,85,0.08)"))
    fig6.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID,
                                   tickfont=dict(color=TEXT)),
                   angularaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
                   bgcolor=BG),
        legend=dict(bgcolor=BG, bordercolor=GRID, font=dict(color=TEXT))
    )
    theme(fig6, "Model Radar Chart")
    st.plotly_chart(fig6, use_container_width=True)

# ─────────────────────────────────────────
# SECTION 5 — PER VIDEO
# ─────────────────────────────────────────
st.markdown('<div class="sec-hdr"><span class="sec-hdr-text">Per-Video Analysis</span><span class="sec-hdr-count">SECTION 05</span></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    vs = df.groupby(["video","vader_sentiment"]).size().reset_index(name="count")
    fig7 = px.bar(vs, x="video", y="count", color="vader_sentiment",
                  barmode="stack", color_discrete_map=COLOR_SENTIMENT)
    fig7.update_layout(xaxis_title="Video", yaxis_title="Comments")
    theme(fig7, "Stacked Sentiment per Video")
    st.plotly_chart(fig7, use_container_width=True)

with col2:
    vp = (df[df["vader_sentiment"]=="positive"]
          .groupby("video").size() / df.groupby("video").size() * 100).reset_index()
    vp.columns = ["video","positive_pct"]
    vp = vp.sort_values("positive_pct")
    fig8 = px.bar(vp, x="positive_pct", y="video", orientation="h",
                  color="positive_pct",
                  color_continuous_scale=["#ef5350","#ffa726","#4caf50"],
                  text="positive_pct")
    fig8.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                       textfont=dict(color=ACCENT))
    fig8.update_layout(xaxis_title="Positive %", yaxis_title="", coloraxis_showscale=False)
    theme(fig8, "Positivity Rate per Video (Ranked)")
    st.plotly_chart(fig8, use_container_width=True)

# ─────────────────────────────────────────
# SECTION 6 — TOPIC MODELING
# ─────────────────────────────────────────
st.markdown('<div class="sec-hdr"><span class="sec-hdr-text">LDA Topic Modeling</span><span class="sec-hdr-count">SECTION 06</span></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.5])
with col1:
    tv = sel_video if sel_video != "All Videos" else "video1"
    st.markdown(f"**Topics for {tv}**")
    ft = topic_df[topic_df["video"] == tv]
    accent_colors = ["#1a1a1a","#555","#888"]
    for i, (_, row) in enumerate(ft.iterrows()):
        c = accent_colors[i % len(accent_colors)]
        st.markdown(f"""
        <div style='background:#fff;border:1px solid #e8e8e4;border-left:3px solid {c};
                    border-radius:0 10px 10px 0;padding:12px 16px;margin:8px 0'>
            <span style='color:{c};font-weight:600;font-size:11px;
                         text-transform:uppercase;letter-spacing:.06em'>
                Topic {int(row["topic_number"])}
            </span><br>
            <span style='color:#444;font-size:13px'>{row["topic_keywords"]}</span>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("**All Videos Topic Summary**")
    st.dataframe(summary_df, use_container_width=True, hide_index=True, height=320)

# ─────────────────────────────────────────
# SECTION 7 — WORD CLOUD
# ─────────────────────────────────────────
st.markdown('<div class="sec-hdr"><span class="sec-hdr-text">Word Cloud</span><span class="sec-hdr-count">SECTION 07</span></div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    wc_opt = st.radio("Generate for:",
        ["All Comments","Positive Only","Negative Only","Neutral Only"], horizontal=True)
    wmap = {
        "All Comments":  fdf,
        "Positive Only": fdf[fdf["vader_sentiment"]=="positive"],
        "Negative Only": fdf[fdf["vader_sentiment"]=="negative"],
        "Neutral Only":  fdf[fdf["vader_sentiment"]=="neutral"],
    }
    cmap_map = {"All Comments":"Greys","Positive Only":"Greens",
                "Negative Only":"Reds","Neutral Only":"Blues"}
    wtext = " ".join(wmap[wc_opt]["clean_comment"].dropna().astype(str))
    if len(wtext.strip()) > 20:
        wc = WordCloud(width=900, height=360, background_color="#ffffff",
                       colormap=cmap_map[wc_opt], max_words=120,
                       collocations=False, prefer_horizontal=0.85).generate(wtext)
        fig_wc, ax = plt.subplots(figsize=(12, 4))
        fig_wc.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("Not enough text for word cloud.")

with col2:
    all_words = " ".join(fdf["clean_comment"].dropna().astype(str)).split()
    top10 = Counter(all_words).most_common(10)
    tw_df = pd.DataFrame(top10, columns=["Word","Count"])
    fig_tw = px.bar(tw_df.sort_values("Count"), x="Count", y="Word",
                    orientation="h", color="Count",
                    color_continuous_scale=["#eee","#1a1a1a"], text="Count")
    fig_tw.update_traces(textposition="outside", textfont=dict(color=ACCENT))
    fig_tw.update_layout(coloraxis_showscale=False, yaxis_title="")
    theme(fig_tw, "Top 10 Words")
    st.plotly_chart(fig_tw, use_container_width=True)

# ─────────────────────────────────────────
# SECTION 8 — DATA QUALITY
# ─────────────────────────────────────────
st.markdown('<div class="sec-hdr"><span class="sec-hdr-text">Data Quality — Spam · Fake · Sarcasm</span><span class="sec-hdr-count">SECTION 08</span></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

def make_donut(df_col, title, colors):
    c = fdf[df_col].value_counts().reset_index()
    c.columns = ["label","count"]
    fig = px.pie(c, names="label", values="count", hole=0.58,
                 color_discrete_sequence=colors)
    fig.update_traces(textinfo="percent+label", textfont_size=11,
                      marker=dict(line=dict(color="#ffffff", width=2)))
    fig.update_layout(showlegend=False)
    theme(fig, title)
    return fig

with col1:
    st.plotly_chart(make_donut("spam_label","Spam Detection",["#ef5350","#4caf50"]),
                    use_container_width=True)
with col2:
    st.plotly_chart(make_donut("fake_label","Fake Detection",["#ffa726","#42a5f5"]),
                    use_container_width=True)
with col3:
    st.plotly_chart(make_donut("sarcasm_label","Sarcasm",["#ab47bc","#42a5f5"]),
                    use_container_width=True)
with col4:
    real_n = (fdf["spam_label"]=="not_spam").sum()
    st.markdown(f"""
    <div class='card'>
        <div style='font-size:11px;font-weight:600;color:#1a1a1a;
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px'>
            Quality Summary
        </div>
        <div style='font-size:13px;color:#555;line-height:2.2'>
            Clean comments<br>
            <span class='badge badge-pos'>{real_n:,}</span><br>
            Spam removed<br>
            <span class='badge badge-neg'>{spam_n:,}</span><br>
            Fake flagged<br>
            <span class='badge badge-fake'>{fake_n:,}</span><br>
            Sarcastic<br>
            <span class='badge badge-sarc'>{sarc_n:,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# SECTION 9 — COMMENT EXPLORER
# ─────────────────────────────────────────
st.markdown('<div class="sec-hdr"><span class="sec-hdr-text">Comment Explorer</span><span class="sec-hdr-count">SECTION 09</span></div>', unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns([2,1,1])
with sc1:
    search = st.text_input("Search keyword", placeholder="e.g. battery, amazing, camera...")
with sc2:
    n_rows = st.slider("Rows to show", 10, 300, 50)
with sc3:
    hide_spam = st.checkbox("Hide spam", value=True)

vdf = fdf[["video","comment","vader_sentiment","bert_sentiment",
           "roberta_sentiment","emotion","spam_label","sarcasm_label"]].copy()
if hide_spam:   vdf = vdf[vdf["spam_label"] != "spam"]
if search:      vdf = vdf[vdf["comment"].str.contains(search, case=False, na=False)]

st.markdown(f"Showing **{min(n_rows,len(vdf)):,}** of **{len(vdf):,}** comments")
st.dataframe(vdf.head(n_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# SECTION 10 — REAL-TIME ANALYSIS (NEW)
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr"><span class="sec-hdr-text">Real-Time YouTube Analysis</span><span class="sec-hdr-count">SECTION 10 · LIVE</span></div>', unsafe_allow_html=True)

st.markdown("""
<div class='card' style='margin-bottom:16px'>
    <div style='font-size:13px;color:#555;line-height:1.8'>
        Enter any YouTube video URL below. The pipeline will fetch comments via
        the YouTube Data API v3, run all three sentiment models, detect emotions,
        flag spam and sarcasm, and display a full breakdown — live.
    </div>
</div>
""", unsafe_allow_html=True)

try:
    from src.youtube_live_comments import fetch_comments
    from src.sentiment_vader import predict_vader
    from src.sentiment_bert import predict_bert
    from src.sentiment_roberta import predict_roberta
    MODELS_LOADED = True
except ImportError as e:
    MODELS_LOADED = False
    st.warning(f"Some modules not found: {e}. Running in demo mode.")

# Emotion detection helper (simple rule-based fallback if no emotion model)
EMOTION_KEYWORDS = {
    "joy":          ["love","amazing","great","excellent","wonderful","happy","awesome","fantastic","best"],
    "anger":        ["hate","worst","terrible","awful","bad","disgusting","stupid","useless","horrible"],
    "sadness":      ["sad","miss","disappointed","unfortunate","cry","poor","regret"],
    "fear":         ["scared","afraid","worried","nervous","anxious","panic"],
    "surprise":     ["wow","omg","unbelievable","shocked","unexpected","whoa"],
    "disgust":      ["gross","disgusting","nasty","yuck","eww"],
    "anticipation": ["excited","waiting","soon","hope","looking forward","can't wait"],
}

def detect_emotion(text):
    if not isinstance(text, str): return "neutral"
    t = text.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(k in t for k in keywords):
            return emotion
    return "neutral"

def detect_spam(text):
    if not isinstance(text, str): return "not_spam"
    spam_signals = ["subscribe","sub4sub","check my channel","visit my","free followers",
                    "click here","giveaway","www.","http","bit.ly","follow me"]
    t = text.lower()
    return "spam" if any(s in t for s in spam_signals) else "not_spam"

def detect_sarcasm(text):
    if not isinstance(text, str): return "not_sarcastic"
    sarcasm_signals = ["oh great","yeah right","sure sure","totally","wow so helpful",
                       "obviously","clearly the best","not","like that helps","so useful"]
    t = text.lower()
    return "sarcastic" if any(s in t for s in sarcasm_signals) else "not_sarcastic"

# Input row
col_url, col_btn = st.columns([4, 1])
with col_url:
    video_url = st.text_input("YouTube Video URL",
                              placeholder="https://www.youtube.com/watch?v=...",
                              label_visibility="collapsed")
with col_btn:
    analyze_clicked = st.button("Analyze Video", use_container_width=True)

if analyze_clicked and video_url.strip():

    # ── FETCH COMMENTS ──────────────────────────────────────────────
    with st.spinner("Fetching comments from YouTube API..."):
        try:
            if MODELS_LOADED:
                rt_df = fetch_comments(video_url)
            else:
                # Demo fallback — sample from existing data
                rt_df = df.sample(min(100, len(df))).rename(
                    columns={"clean_comment":"comment"})[["comment"]].reset_index(drop=True)
        except Exception as ex:
            st.error(f"Could not fetch comments: {ex}")
            st.stop()

    if rt_df is None or len(rt_df) == 0:
        st.warning("No comments found for this video.")
        st.stop()

    comment_col = "comment" if "comment" in rt_df.columns else rt_df.columns[0]
    st.markdown(f'<div class="rt-status">✓ Fetched <strong>{len(rt_df)}</strong> comments</div>',
                unsafe_allow_html=True)

    # ── STEP 5: RUN MODELS ──────────────────────────────────────────
    with st.spinner("Running VADER..."):
        if MODELS_LOADED:
            rt_df["vader"] = rt_df[comment_col].apply(predict_vader)
        else:
            rt_df["vader"] = rt_df[comment_col].apply(
                lambda x: predict_vader(x) if True else "neutral")

    with st.spinner("Running BERT..."):
        if MODELS_LOADED:
            rt_df["bert"] = rt_df[comment_col].apply(predict_bert)
        else:
            rt_df["bert"] = rt_df["vader"]  # fallback mirror

    with st.spinner("Running RoBERTa..."):
        if MODELS_LOADED:
            rt_df["roberta"] = rt_df[comment_col].apply(predict_roberta)
        else:
            rt_df["roberta"] = rt_df["vader"]

    # ── EMOTION + SPAM + SARCASM ─────────────────────────────────────
    with st.spinner("Detecting emotions, spam, sarcasm..."):
        rt_df["emotion"]  = rt_df[comment_col].apply(detect_emotion)
        rt_df["spam"]     = rt_df[comment_col].apply(detect_spam)
        rt_df["sarcasm"]  = rt_df[comment_col].apply(detect_sarcasm)

    st.markdown('<div class="rt-status">✓ All models complete</div>', unsafe_allow_html=True)

    # ── STEP 6: KPIs ────────────────────────────────────────────────
    st.markdown("#### Results")
    rt_total = len(rt_df)
    rt_pos   = (rt_df["roberta"]=="positive").sum()
    rt_neg   = (rt_df["roberta"]=="negative").sum()
    rt_neu   = (rt_df["roberta"]=="neutral").sum()
    rt_spam  = (rt_df["spam"]=="spam").sum()
    rt_sarc  = (rt_df["sarcasm"]=="sarcastic").sum()

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Comments",  rt_total)
    k2.metric("Positive",  f"{rt_pos/rt_total*100:.1f}%", f"{rt_pos} comments")
    k3.metric("Negative",  f"{rt_neg/rt_total*100:.1f}%", f"{rt_neg} comments")
    k4.metric("Neutral",   f"{rt_neu/rt_total*100:.1f}%", f"{rt_neu} comments")
    k5.metric("Spam",      f"{rt_spam/rt_total*100:.1f}%", f"{rt_spam} flagged")
    k6.metric("Sarcasm",   f"{rt_sarc/rt_total*100:.1f}%", f"{rt_sarc} detected")

    # ── CHARTS ROW 1: Sentiment + Emotion ───────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        rsc = rt_df["roberta"].value_counts().reset_index()
        rsc.columns = ["sentiment","count"]
        fig_r = px.pie(rsc, names="sentiment", values="count", hole=0.55,
                       color="sentiment", color_discrete_map=COLOR_SENTIMENT)
        fig_r.update_traces(textposition="inside", textinfo="percent+label",
                            textfont_size=12,
                            marker=dict(line=dict(color="#ffffff", width=2)))
        fig_r.update_layout(showlegend=False)
        theme(fig_r, "RoBERTa Sentiment Distribution")
        st.plotly_chart(fig_r, use_container_width=True)

    with ch2:
        emc = rt_df["emotion"].value_counts().reset_index()
        emc.columns = ["emotion","count"]
        fig_e = px.bar(emc, x="emotion", y="count", color="emotion",
                       color_discrete_sequence=PALETTE, text="count")
        fig_e.update_traces(texttemplate="%{text}", textposition="outside",
                            marker_line_color="#ffffff", marker_line_width=1)
        fig_e.update_layout(showlegend=False, xaxis_title="Emotion", yaxis_title="Count")
        theme(fig_e, "Emotion Detection")
        st.plotly_chart(fig_e, use_container_width=True)

    # ── CHARTS ROW 2: Model Comparison + Spam/Sarcasm ───────────────
    ch3, ch4 = st.columns(2)

    with ch3:
        model_comp = pd.DataFrame({
            "Model": ["VADER","BERT","RoBERTa"],
            "Positive": [
                (rt_df["vader"]=="positive").sum(),
                (rt_df["bert"]=="positive").sum(),
                (rt_df["roberta"]=="positive").sum()
            ],
            "Negative": [
                (rt_df["vader"]=="negative").sum(),
                (rt_df["bert"]=="negative").sum(),
                (rt_df["roberta"]=="negative").sum()
            ],
            "Neutral": [
                (rt_df["vader"]=="neutral").sum(),
                (rt_df["bert"]=="neutral").sum(),
                (rt_df["roberta"]=="neutral").sum()
            ]
        })
        fig_mc = go.Figure()
        for sent, color in [("Positive","#4caf50"),("Negative","#ef5350"),("Neutral","#42a5f5")]:
            fig_mc.add_trace(go.Bar(name=sent, x=model_comp["Model"],
                y=model_comp[sent], marker_color=color,
                marker_line_color="#ffffff", marker_line_width=1))
        fig_mc.update_layout(barmode="group", xaxis_title="Model", yaxis_title="Count")
        theme(fig_mc, "3-Model Sentiment Comparison (Live)")
        st.plotly_chart(fig_mc, use_container_width=True)

    with ch4:
        dq_data = pd.DataFrame({
            "Category": ["Clean","Spam","Sarcastic"],
            "Count": [rt_total - rt_spam, rt_spam, rt_sarc]
        })
        fig_dq = px.bar(dq_data, x="Category", y="Count",
                        color="Category",
                        color_discrete_sequence=["#4caf50","#ef5350","#ab47bc"],
                        text="Count")
        fig_dq.update_traces(texttemplate="%{text}", textposition="outside",
                             marker_line_color="#ffffff", marker_line_width=1)
        fig_dq.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
        theme(fig_dq, "Data Quality — Spam & Sarcasm")
        st.plotly_chart(fig_dq, use_container_width=True)

    # ── WORD CLOUD (live) ────────────────────────────────────────────
    st.markdown("**Word Cloud — Live Comments**")
    wc_col1, wc_col2 = st.columns([2,1])
    with wc_col1:
        rt_wc_opt = st.radio("Filter by:", ["All","Positive","Negative","Neutral"],
                             horizontal=True, key="rt_wc")
        wc_filter_map = {
            "All":      rt_df,
            "Positive": rt_df[rt_df["roberta"]=="positive"],
            "Negative": rt_df[rt_df["roberta"]=="negative"],
            "Neutral":  rt_df[rt_df["roberta"]=="neutral"],
        }
        wc_cmap = {"All":"Greys","Positive":"Greens","Negative":"Reds","Neutral":"Blues"}
        wc_text = " ".join(wc_filter_map[rt_wc_opt][comment_col].dropna().astype(str))
        if len(wc_text.strip()) > 20:
            wc_obj = WordCloud(width=800, height=320, background_color="#ffffff",
                               colormap=wc_cmap[rt_wc_opt], max_words=100,
                               collocations=False).generate(wc_text)
            fig_wc2, ax2 = plt.subplots(figsize=(10,3.5))
            fig_wc2.patch.set_facecolor("#ffffff")
            ax2.set_facecolor("#ffffff")
            ax2.imshow(wc_obj, interpolation="bilinear")
            ax2.axis("off")
            st.pyplot(fig_wc2)
        else:
            st.info("Not enough text for word cloud.")

    with wc_col2:
        live_words = " ".join(rt_df[comment_col].dropna().astype(str)).split()
        live_top = Counter(live_words).most_common(8)
        lw_df = pd.DataFrame(live_top, columns=["Word","Count"])
        fig_lw = px.bar(lw_df.sort_values("Count"), x="Count", y="Word",
                        orientation="h", color="Count",
                        color_continuous_scale=["#eee","#1a1a1a"], text="Count")
        fig_lw.update_traces(textposition="outside", textfont=dict(color=ACCENT))
        fig_lw.update_layout(coloraxis_showscale=False, yaxis_title="")
        theme(fig_lw, "Top Words (Live)")
        st.plotly_chart(fig_lw, use_container_width=True)

    # ── COMMENT TABLE ────────────────────────────────────────────────
    st.markdown("**Full Comment Results**")
    search_rt = st.text_input("Search live comments",
                              placeholder="Search keyword...", key="rt_search")
    hide_spam_rt = st.checkbox("Hide spam comments", value=True, key="rt_spam")

    display_rt = rt_df[[comment_col,"vader","bert","roberta","emotion","spam","sarcasm"]].copy()
    if hide_spam_rt: display_rt = display_rt[display_rt["spam"] != "spam"]
    if search_rt:    display_rt = display_rt[
        display_rt[comment_col].str.contains(search_rt, case=False, na=False)]

    st.markdown(f"Showing **{min(100,len(display_rt)):,}** of **{len(display_rt):,}** comments")
    st.dataframe(display_rt.head(100), use_container_width=True, hide_index=True)

elif analyze_clicked:
    st.warning("Please enter a YouTube video URL.")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class='footer'>
    YouTube Comment Sentiment Analysis &nbsp;·&nbsp;
    M.Sc. Information Technology &nbsp;·&nbsp;
    VADER · BERT · RoBERTa · Emotion · LDA · Spam · Sarcasm · Flask · MLflow · Docker · AWS
</div>
""", unsafe_allow_html=True)