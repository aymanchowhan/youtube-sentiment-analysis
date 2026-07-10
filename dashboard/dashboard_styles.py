"""
Theme constants, global CSS, and reusable chart-building helpers.

Nothing in this file is tab-specific — every tab (offline or live) imports
from here so the look stays consistent and the CSS only lives in one place.

THEME: "Voltage" — dark, high-contrast, electric/lightning accents.
"""
import streamlit as st
import plotly.express as px

# ─────────────────────────────────────────
# THEME / CONSTANTS
# ─────────────────────────────────────────
BG      = "#0A0A12"   # page + chart background — near-black indigo
CARD    = "#12121C"   # card / panel surface
TEXT    = "#F1F1FA"   # primary text — near white
MUTED   = "#9494B8"   # secondary text
ACCENT  = "#D4FF3D"   # electric lime — primary accent
ACCENT2 = "#B14EFF"   # electric purple
ACCENT3 = "#22D3EE"   # electric cyan
ACCENT4 = "#FF2D6D"   # electric pink/red
GRID    = "#25253A"   # chart gridlines

COLOR_SENTIMENT = {
    "positive": "#39FF14",   # neon green
    "negative": "#FF2D6D",   # neon pink-red
    "neutral":  "#22D3EE",   # neon cyan
}

PALETTE = [
    "#D4FF3D", "#B14EFF", "#22D3EE", "#FF2D6D",
    "#FFD23F", "#4D7CFE", "#FF6EC7", "#39FF88",
]

# The dashboard's signature gradient — electric lime → purple → pink,
# used for the hero, the live-analysis banner, and every section underline.
SENTIMENT_GRADIENT = "linear-gradient(135deg, #D4FF3D, #B14EFF, #FF2D6D)"


def inject_css():
    """Injects the dashboard's global stylesheet. Call once, right after
    st.set_page_config()."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

    *, *::before, *::after {
        font-family: 'Inter', sans-serif !important;
        box-sizing: border-box;
    }

    [data-testid="stIconMaterial"],
    [data-testid="stExpanderIcon"],
    .material-icons,
    .material-symbols-outlined,
    [class*="material-symbols"] {
        font-family: 'Material Symbols Outlined', 'Material Icons' !important;
    }

    html, body, .stApp {
        background:
            radial-gradient(circle at 12% 18%, rgba(212,255,61,0.07), transparent 42%),
            radial-gradient(circle at 88% 12%, rgba(177,78,255,0.10), transparent 45%),
            radial-gradient(circle at 90% 85%, rgba(34,211,238,0.07), transparent 40%),
            #0A0A12 !important;
    }

    .main .block-container {
        background: transparent !important;
        padding: 2rem 2.5rem !important;
        max-width: 1400px !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0D0D18 !important;
        width: 230px !important;
        min-width: 230px !important;
        border-right: 1px solid rgba(212,255,61,0.14) !important;
        box-shadow: 6px 0 30px rgba(0,0,0,0.5) !important;
    }
    section[data-testid="stSidebar"] * { color: #E4E4F5 !important; }
    section[data-testid="stSidebar"] label { color: #9494B8 !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #F1F1FA !important; }

    [data-testid="collapsedControl"] { display: none !important; }
    button[kind="header"] { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #12121C !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-top: 3px solid #D4FF3D !important;
        border-radius: 18px !important;
        padding: 20px !important;
        box-shadow: 0 8px 28px rgba(0,0,0,0.45) !important;
        transition: transform .25s ease, box-shadow .25s ease !important;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 0 0 1px rgba(212,255,61,0.35), 0 16px 36px rgba(0,0,0,0.55) !important;
    }
    [data-testid="metric-container"] label {
        color: #9494B8 !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    [data-testid="stMetricValue"] {
        color: #F1F1FA !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.01em !important;
    }
    [data-testid="stMetricDelta"] {
        color: #9494B8 !important;
        font-size: 12px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    div[data-testid="column"]:nth-of-type(6n+1) [data-testid="metric-container"] { border-top-color: #D4FF3D !important; }
    div[data-testid="column"]:nth-of-type(6n+2) [data-testid="metric-container"] { border-top-color: #B14EFF !important; }
    div[data-testid="column"]:nth-of-type(6n+3) [data-testid="metric-container"] { border-top-color: #FF2D6D !important; }
    div[data-testid="column"]:nth-of-type(6n+4) [data-testid="metric-container"] { border-top-color: #22D3EE !important; }
    div[data-testid="column"]:nth-of-type(6n+5) [data-testid="metric-container"] { border-top-color: #FFD23F !important; }
    div[data-testid="column"]:nth-of-type(6n+6) [data-testid="metric-container"] { border-top-color: #FF6EC7 !important; }

    /* ── Inputs ── */
    .stTextInput input {
        background: #12121C !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 18px !important;
        color: #F1F1FA !important;
        font-size: 14px !important;
        padding: 10px 14px !important;
    }
    .stTextInput input::placeholder { color: #6B6B90 !important; }
    .stTextInput input:focus {
        border-color: #D4FF3D !important;
        box-shadow: 0 0 0 3px rgba(212,255,61,0.18) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #D4FF3D, #22D3EE) !important;
        color: #0A0A12 !important;
        border: none !important;
        border-radius: 18px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.01em !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 20px rgba(212,255,61,0.25) !important;
        transition: transform 0.15s ease, box-shadow 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(212,255,61,0.4) !important;
    }
    .stButton > button p { color: #0A0A12 !important; font-weight: 700 !important; }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: #12121C !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 18px !important;
        color: #F1F1FA !important;
    }
    [data-baseweb="popover"] { background: #12121C !important; }
    [data-baseweb="menu"] { background: #12121C !important; }
    [data-baseweb="menu"] li { color: #E4E4F5 !important; }

    body, p, span, div, label { color: #F1F1FA; }

    .stRadio label   { color: #C6C6E0 !important; font-size: 13px !important; }
    .stSlider label  { color: #C6C6E0 !important; }
    .stCheckbox label{ color: #C6C6E0 !important; }
    [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: #6B6B90 !important; }
    div[data-baseweb="slider"] div[role="slider"] { background: #D4FF3D !important; }

    /* ── DataFrame ── */
    [data-testid="stDataFrame"] {
        border-radius: 18px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0D0D18; }
    ::-webkit-scrollbar-thumb { background: #3A3A55; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #D4FF3D; }

    /* ── Section headers ── */
    .sec-hdr {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 32px 0 16px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid transparent;
        border-image: linear-gradient(90deg, #D4FF3D, #B14EFF, #FF2D6D) 1;
        filter: drop-shadow(0 0 8px rgba(212,255,61,0.15));
    }
    .sec-hdr::before {
        content: "⚡";
        font-size: 15px;
    }
    .sec-hdr-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 15px; font-weight: 700; color: #F1F1FA; letter-spacing: -0.01em;
    }
    .sec-hdr-count {
        font-size: 11px;
        color: #9494B8;
        margin-left: auto;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── Cards ── */
    .card {
        background: #12121C;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 20px 22px;
        margin: 8px 0;
        box-shadow: 0 8px 28px rgba(0,0,0,0.45);
        transition: all .25s ease;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 0 0 1px rgba(212,255,61,0.3), 0 16px 36px rgba(0,0,0,0.55);
    }

    /* ── Badges — neon outline chips ── */
    .badge {
        display: inline-block;
        padding: 3px 11px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        margin: 2px;
        border: 1px solid transparent;
        background: rgba(255,255,255,0.04);
    }
    .badge-pos  { color: #39FF14; border-color: rgba(57,255,20,0.4);  box-shadow: 0 0 10px rgba(57,255,20,0.12); }
    .badge-neg  { color: #FF2D6D; border-color: rgba(255,45,109,0.4); box-shadow: 0 0 10px rgba(255,45,109,0.12); }
    .badge-neu  { color: #22D3EE; border-color: rgba(34,211,238,0.4); box-shadow: 0 0 10px rgba(34,211,238,0.12); }
    .badge-spam { color: #FFD23F; border-color: rgba(255,210,63,0.4); box-shadow: 0 0 10px rgba(255,210,63,0.12); }
    .badge-sarc { color: #B14EFF; border-color: rgba(177,78,255,0.4); box-shadow: 0 0 10px rgba(177,78,255,0.12); }
    .badge-fake { color: #9494B8; border-color: rgba(148,148,184,0.4); }

    /* ── Hero ── */
    .hero {
        position: relative;
        background: #101018;
        border: 1px solid rgba(212,255,61,0.18);
        border-radius: 22px;
        padding: 40px;
        margin-bottom: 28px;
        overflow: hidden;
        box-shadow: 0 0 60px rgba(177,78,255,0.08), 0 12px 40px rgba(0,0,0,0.5);
    }
    .hero::before {
        content: "";
        position: absolute; inset: 0;
        background: linear-gradient(120deg, rgba(212,255,61,0.10), rgba(177,78,255,0.10), rgba(255,45,109,0.10));
        pointer-events: none;
    }
    .hero h1 {
        position: relative;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 10px 0;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #D4FF3D, #22D3EE 45%, #B14EFF 75%, #FF2D6D);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p { position: relative; font-size: 13px; color: #C6C6E0; margin: 0; line-height: 1.9; }
    .hero-pill {
        display: inline-block;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 4px 11px;
        font-size: 11px;
        color: #F1F1FA !important;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        margin: 2px;
    }

    /* ── Insight box ── */
    .insight {
        background: rgba(212,255,61,0.05);
        border-left: 3px solid #D4FF3D;
        border-radius: 0 18px 18px 0;
        padding: 14px 16px;
        font-size: 13px;
        color: #E4E4F5;
        line-height: 1.7;
        margin: 8px 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #6B6B90;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.02em;
        padding: 32px 0 16px;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 48px;
    }

    /* ── Live / RT ── */
    .rt-status {
        background: rgba(57,255,20,0.06);
        border: 1px solid rgba(57,255,20,0.3);
        border-radius: 18px;
        padding: 10px 16px;
        font-size: 13px;
        color: #39FF14;
        font-family: 'JetBrains Mono', monospace;
        margin: 12px 0;
        animation: pulse-glow 2.4s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 6px rgba(57,255,20,0.05); }
        50%      { box-shadow: 0 0 18px rgba(57,255,20,0.25); }
    }

    .live-section {
        position: relative;
        background: #101018;
        border: 1px solid rgba(255,45,109,0.25);
        border-radius: 22px;
        padding: 24px;
        margin-bottom: 16px;
        overflow: hidden;
        box-shadow: 0 0 40px rgba(255,45,109,0.08);
    }
    .live-section::before {
        content: "";
        position: absolute; inset: 0;
        background: linear-gradient(120deg, rgba(255,45,109,0.10), rgba(177,78,255,0.10), rgba(34,211,238,0.10));
        pointer-events: none;
    }
    .live-section > div { position: relative; }

    hr { border: none; border-top: 1px solid rgba(255,255,255,0.08) !important; margin: 8px 0 !important; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────
def theme(fig, title=""):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font_color=TEXT, font_family="Inter",
        title=dict(text=title, font=dict(size=13, color=ACCENT, family="Space Grotesk"), x=0.01),
        margin=dict(t=44, b=28, l=16, r=16),
        legend=dict(bgcolor=BG, bordercolor=GRID, borderwidth=1,
                    font=dict(color=TEXT, size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=TEXT, size=11))
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=TEXT, size=11))
    return fig


def make_pie(df_col, title, color_map=None, colors=None):
    """Reusable donut chart."""
    c = df_col.value_counts().reset_index()
    c.columns = ["label", "count"]
    kwargs = dict(color="label", color_discrete_map=color_map) if color_map \
             else dict(color_discrete_sequence=colors or PALETTE)
    fig = px.pie(c, names="label", values="count", hole=0.72, **kwargs)
    fig.update_traces(
        textposition="inside", textinfo="percent+label", textfont_size=11,
        marker=dict(line=dict(color=BG, width=2))
    )
    fig.update_layout(showlegend=False)
    return theme(fig, title)


def make_bar_h(df_data, x, y, title, color_col=None):
    """Reusable horizontal bar chart (used by Top Words / Top Topics everywhere)."""
    fig = px.bar(df_data, x=x, y=y, orientation="h",
                 color=color_col or x,
                 color_continuous_scale=["#25253A", ACCENT], text=x)
    fig.update_traces(textposition="outside", textfont=dict(color=ACCENT),
                      marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    return theme(fig, title)


def make_bar_v(count_series, x_title, title, texttemplate="%{text}"):
    """Reusable vertical count bar chart (e.g. emotion distribution) — used by
    both the offline Emotion tab and the Live Analysis emotion chart."""
    c = count_series.value_counts().reset_index()
    c.columns = [x_title, "count"]
    fig = px.bar(c, x=x_title, y="count", color=x_title,
                 color_discrete_sequence=PALETTE, text="count")
    fig.update_traces(texttemplate=texttemplate, textposition="outside", marker_line_width=0)
    fig.update_layout(showlegend=False, xaxis_title=x_title.capitalize(), yaxis_title="Count")
    return theme(fig, title)


def sec_hdr(label, number):
    st.markdown(
        f'<div class="sec-hdr">'
        f'<span class="sec-hdr-text">{label}</span>'
        f'<span class="sec-hdr-count">{number}</span>'
        f'</div>',
        unsafe_allow_html=True
    )