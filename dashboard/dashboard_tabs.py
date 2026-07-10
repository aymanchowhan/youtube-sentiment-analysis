"""
Offline (trained-data) tabs: Overview, Sentiment, Emotion, Models,
Per-Video, Topics, Word Cloud, Quality, Explorer.

Each render_tab_* function is called from completeappdemo.py inside its
matching `with tab_x:` block. Nothing here touches the YouTube API or
live models — that's all in dashboard_live.py.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_styles import theme, make_pie, make_bar_h, make_bar_v, sec_hdr, PALETTE, COLOR_SENTIMENT, ACCENT, GRID, BG, TEXT
from dashboard_components import render_badge_card, render_topic_card, render_wordcloud_block, clean_topic_name, render_sarcasm_correction_block


# ─────────────────────────────────────────
# TAB 1 — OVERVIEW (KPIs)
# ─────────────────────────────────────────
def render_tab_overview(fdf, total):
    sec_hdr("Overview Metrics", "SECTION 01")

    pos_n  = (fdf["vader_sentiment"] == "positive").sum()
    neg_n  = (fdf["vader_sentiment"] == "negative").sum()
    neu_n  = (fdf["vader_sentiment"] == "neutral").sum()
    spam_n = (fdf["spam_label"]      == "spam").sum()
    sarc_n = (fdf["sarcasm_label"]   == "sarcastic").sum()

    def pct(n): return f"{n/total*100:.1f}%" if total else "0%"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Comments", f"{total:,}")
    c2.metric("Positive",  pct(pos_n),  f"{pos_n:,} comments")
    c3.metric("Negative",  pct(neg_n),  f"{neg_n:,} comments")
    c4.metric("Neutral",   pct(neu_n),  f"{neu_n:,} comments")
    c5.metric("Spam",      pct(spam_n), f"{spam_n:,} flagged")
    c6.metric("Sarcasm",   pct(sarc_n), f"{sarc_n:,} detected")


# ─────────────────────────────────────────
# TAB 2 — SENTIMENT
# ─────────────────────────────────────────
def render_tab_sentiment(fdf, total):
    sec_hdr("Sentiment Analysis", "SECTION 02")
    col1, col2, col3 = st.columns([1.1, 1.4, 0.9])

    with col1:
        st.plotly_chart(
make_pie(
    fdf["roberta_sentiment"],"Final Sentiment Distribution (RoBERTa)",
                     color_map=COLOR_SENTIMENT),
            use_container_width=True
        )

    with col2:
        cats = ["positive", "negative", "neutral"]
        fig2 = go.Figure()
        for model, col_name, color in [
            ("VADER",   "vader_sentiment",   "#FFD23F"),
            ("BERT",    "bert_sentiment",    "#B14EFF"),
            ("RoBERTa", "roberta_sentiment", "#39FF14"),
        ]:
            vc = fdf[col_name].value_counts()
            fig2.add_trace(go.Bar(
                name=model, x=cats, y=[vc.get(c, 0) for c in cats],
                marker_color=color, marker_line_width=0
            ))
        fig2.update_layout(barmode="group", xaxis_title="Sentiment", yaxis_title="Count")
        theme(fig2, "3-Model Sentiment Comparison")
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        top_emo   = fdf["emotion"].value_counts().index[0] if total else "N/A"
        top_sent  = fdf["roberta_sentiment"].value_counts().index[0] if total else "N/A"
        agree_pct = round((fdf["vader_sentiment"] == fdf["roberta_sentiment"]).sum() / total * 100, 1) if total else 0

        render_badge_card(
            "Key Findings",
            rows=[
                ("Dominant sentiment", f"<span class='badge badge-pos'>{top_sent}</span>"),
                ("Top emotion",        f"<span class='badge badge-neu'>{top_emo}</span>"),
                ("Best model",         "<span class='badge badge-pos'>RoBERTa 73.2%</span>"),
                ("Weakest model",      "<span class='badge badge-neg'>BERT 39.7%</span>"),
                ("Model agreement",    f"<span class='badge badge-neu'>{agree_pct}% match</span>"),
            ],
            margin_bottom=10, wrap_content=False, entry_sep="<br><br>",
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    sec_hdr("Sarcasm-Corrected Sentiment", "BEYOND FACE VALUE")
    render_sarcasm_correction_block(fdf["roberta_sentiment"], fdf["sarcasm_label"])


# ─────────────────────────────────────────
# TAB 3 — EMOTION
# ─────────────────────────────────────────
def render_tab_emotion(fdf):
    sec_hdr("Emotion Detection", "SECTION 03")
    col1, col2 = st.columns(2)

    with col1:
        fig3 = make_bar_v(fdf["emotion"], "emotion", "Emotion Distribution", texttemplate="%{text:,}")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        es = fdf.groupby(["emotion", "vader_sentiment"]).size().reset_index(name="count")
        fig4 = px.bar(es, x="emotion", y="count", color="vader_sentiment",
                      barmode="stack", color_discrete_map=COLOR_SENTIMENT)
        fig4.update_traces(marker_line_width=0)
        fig4.update_layout(xaxis_title="Emotion", yaxis_title="Count")
        theme(fig4, "Emotion × Sentiment Breakdown")
        st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────
# TAB 4 — MODEL PERFORMANCE
# ─────────────────────────────────────────
def render_tab_models(model_df):
    sec_hdr("Model Performance", "SECTION 04")
    col1, col2, col3 = st.columns([1, 1.3, 1])

    with col1:
        vader_row = pd.DataFrame([{"Model": "VADER (baseline)", "Accuracy": 1.0,
                                   model_df.columns[2]: 1.0}])
        disp = pd.concat([model_df, vader_row], ignore_index=True)
        disp_show = disp.copy()
        disp_show["Accuracy"] = disp_show["Accuracy"].apply(lambda x: f"{x:.2%}")
        metric_col = disp_show.columns[2]
        disp_show[metric_col] = disp_show[metric_col].apply(lambda x: f"{float(x):.2%}")
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
        raw  = pd.read_csv("data/model_comparison.csv")
        pdf  = pd.concat([raw, pd.DataFrame([{"Model": "VADER", "Accuracy": 1.0,
                                              raw.columns[2]: 1.0}])], ignore_index=True)
        fig5 = go.Figure()
        for i, row in pdf.iterrows():
            fig5.add_trace(go.Bar(
                name=row["Model"], x=[row["Model"]], y=[row["Accuracy"]],
                marker_color=PALETTE[i % len(PALETTE)], marker_line_width=0,
                text=f"{row['Accuracy']:.1%}", textposition="outside",
                textfont=dict(color=ACCENT, size=12)
            ))
        fig5.update_layout(showlegend=False, yaxis_tickformat=".0%",
                           yaxis_range=[0, 1.2], bargap=0.3)
        theme(fig5, "Model Accuracy Comparison")
        st.plotly_chart(fig5, use_container_width=True)

    with col3:
        cats2 = ["Accuracy", "F1 Score", "Speed", "Interpretability", "Social Media"]
        radar_models = [
            ("VADER",   [1.0, 1.0, 1.0, 1.0, 0.8], "#FFD23F", "rgba(255,210,63,0.12)"),
            ("BERT",    [0.40, 0.36, 0.5, 0.6, 0.5], "#B14EFF", "rgba(177,78,255,0.12)"),
            ("RoBERTa", [0.73, 0.72, 0.6, 0.5, 0.8], "#39FF14", "rgba(57,255,20,0.12)"),
        ]
        fig6 = go.Figure()
        for name, r, lc, fc in radar_models:
            fig6.add_trace(go.Scatterpolar(
                r=r, theta=cats2, fill="toself",
                name=name, line_color=lc, fillcolor=fc
            ))
        fig6.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID,
                                tickfont=dict(color=TEXT)),
                angularaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
                bgcolor=BG,
            ),
            legend=dict(bgcolor=BG, bordercolor=GRID, font=dict(color=TEXT))
        )
        theme(fig6, "Model Radar Chart")
        st.plotly_chart(fig6, use_container_width=True)


# ─────────────────────────────────────────
# TAB 5 — PER-VIDEO
# ─────────────────────────────────────────
def render_tab_video(df):
    sec_hdr("Per-Video Analysis", "SECTION 05")
    col1, col2 = st.columns(2)

    with col1:
        vs = df.groupby(["video", "vader_sentiment"]).size().reset_index(name="count")
        fig7 = px.bar(vs, x="video", y="count", color="vader_sentiment",
                      barmode="stack", color_discrete_map=COLOR_SENTIMENT)
        fig7.update_traces(marker_line_width=0)
        fig7.update_layout(xaxis_title="Video", yaxis_title="Comments")
        theme(fig7, "Stacked Sentiment per Video")
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        vp = (df[df["vader_sentiment"] == "positive"]
              .groupby("video").size() / df.groupby("video").size() * 100).reset_index()
        vp.columns = ["video", "positive_pct"]
        vp = vp.sort_values("positive_pct")
        fig8 = px.bar(vp, x="positive_pct", y="video", orientation="h",
                      color="positive_pct",
                      color_continuous_scale=["#FF2D6D", "#FFD23F", "#39FF14"],
                      text="positive_pct")
        fig8.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                           textfont=dict(color=ACCENT), marker_line_width=0)
        fig8.update_layout(xaxis_title="Positive %", yaxis_title="", coloraxis_showscale=False)
        theme(fig8, "Positivity Rate per Video (Ranked)")
        st.plotly_chart(fig8, use_container_width=True)


# ─────────────────────────────────────────
# TAB 6 — TOPIC MODELING (per video — each video has its own BERTopic model)
# ─────────────────────────────────────────
def render_tab_topics(summary_df, sel_video):
    sec_hdr("BERTopic Topic Modeling — Per Video", "SECTION 06")

    if "video" not in summary_df.columns:
        st.warning("topic_summary.csv is missing a 'video' column — rerun topic_modeling.py "
                   "to regenerate it with the per-video schema.")
        return

    all_videos = sorted(summary_df["video"].unique())
    if not all_videos:
        st.info("No topic data found.")
        return

    default_video = sel_video if sel_video in all_videos else all_videos[0]
    tv = st.selectbox("Video", all_videos, index=all_videos.index(default_video), key="topics_video_select")

    video_summary = (
        summary_df[(summary_df["video"] == tv) & (summary_df["Topic"] != -1)]
        .sort_values("Count", ascending=False)
    )

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown(f"**Top Topics — {tv}**")
        if video_summary.empty:
            st.info("No clear topics found for this video.")
        else:
            for i, (_, row) in enumerate(video_summary.head(6).iterrows()):
                render_topic_card(
                    row["Topic"], clean_topic_name(row["Name"]),
                    PALETTE[i % len(PALETTE)], count=row["Count"]
                )
            remaining = video_summary.iloc[6:]
            if not remaining.empty:
                show_key = f"show_more_topics_{tv}"
                if show_key not in st.session_state:
                    st.session_state[show_key] = False
                btn_label = ("Hide extra topics" if st.session_state[show_key]
                             else f"Show {len(remaining)} more topics for {tv}")
                if st.button(btn_label, key=f"btn_{show_key}"):
                    st.session_state[show_key] = not st.session_state[show_key]
                if st.session_state[show_key]:
                    for _, row in remaining.iterrows():
                        st.markdown(f"**Topic {int(row['Topic'])}** · {int(row['Count'])} comments "
                                    f"— {clean_topic_name(row['Name'])}")

    with col2:
        st.markdown(f"**Topic Distribution — {tv}**")
        if video_summary.empty:
            st.info("Nothing to chart yet.")
        else:
            chart_df = video_summary.copy()
            chart_df["Keywords"] = chart_df["Name"].apply(clean_topic_name)
            fig = make_bar_h(chart_df.sort_values("Count"), "Count", "Keywords", f"Topics for {tv}")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**All Videos — Top Topic Overview**")
    overview_rows = []
    for v in all_videos:
        v_summary = (
            summary_df[(summary_df["video"] == v) & (summary_df["Topic"] != -1)]
            .sort_values("Count", ascending=False)
        )
        if v_summary.empty:
            overview_rows.append({"Video": v, "Top Topic": "No clear topics found",
                                   "Comments": 0, "Topics Found": 0})
        else:
            top_row = v_summary.iloc[0]
            overview_rows.append({
                "Video": v,
                "Top Topic": clean_topic_name(top_row["Name"]),
                "Comments": int(top_row["Count"]),
                "Topics Found": len(v_summary),
            })
    st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────
# TAB 7 — WORD CLOUD
# ─────────────────────────────────────────
def render_tab_wordcloud(fdf):
    sec_hdr("Word Cloud Analysis", "SECTION 07")
    render_wordcloud_block(
        full_df=fdf, text_col="clean_comment", sentiment_col="vader_sentiment",
        filter_options={
            "All Comments": None, "Positive Only": "positive",
            "Negative Only": "negative", "Neutral Only": "neutral",
        },
        cmap_map={"All Comments": "cool", "Positive Only": "spring",
                  "Negative Only": "Wistia", "Neutral Only": "winter"},
        radio_label="Generate for:", key_prefix="offline_wc",
        wc_size=(900, 360), max_words=120, prefer_horizontal=0.85,
        figsize=(12, 4), top_n=10, chart_title="Top 10 Words",
    )


# ─────────────────────────────────────────
# TAB 8 — DATA QUALITY
# ─────────────────────────────────────────
def render_tab_quality(fdf):
    sec_hdr("Data Quality — Spam · Fake · Sarcasm", "SECTION 08")
    col1, col2, col3, col4 = st.columns(4)

    real_n = (fdf["spam_label"] == "not_spam").sum()
    spam_n = (fdf["spam_label"] == "spam").sum()
    fake_n = (fdf["fake_label"] == "fake").sum()
    sarc_n = (fdf["sarcasm_label"] == "sarcastic").sum()

    with col1:
        st.plotly_chart(
            make_pie(fdf["spam_label"], "Spam Detection", colors=["#FF2D6D", "#39FF14"]),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            make_pie(fdf["fake_label"], "Fake Detection", colors=["#FFD23F", "#22D3EE"]),
            use_container_width=True
        )
    with col3:
        st.plotly_chart(
            make_pie(fdf["sarcasm_label"], "Sarcasm", colors=["#B14EFF", "#22D3EE"]),
            use_container_width=True
        )
    with col4:
        render_badge_card(
            "Quality Summary",
            rows=[
                ("Clean comments", f"<span class='badge badge-pos'>{real_n:,}</span>"),
                ("Spam removed",   f"<span class='badge badge-neg'>{spam_n:,}</span>"),
                ("Fake flagged",   f"<span class='badge badge-fake'>{fake_n:,}</span>"),
                ("Sarcastic",      f"<span class='badge badge-sarc'>{sarc_n:,}</span>"),
            ],
            css_class="card",
        )


# ─────────────────────────────────────────
# TAB 9 — COMMENT EXPLORER
# ─────────────────────────────────────────
def render_tab_explorer(fdf):
    sec_hdr("Comment Explorer", "SECTION 09")
    sc1, sc2, sc3 = st.columns([2, 1, 1])
    with sc1: search    = st.text_input("Search keyword", placeholder="e.g. battery, amazing, camera...")
    with sc2: n_rows    = st.slider("Rows to show", 10, 300, 50)
    with sc3: hide_spam = st.checkbox("Hide spam", value=True)

    vdf = fdf[["video", "comment", "vader_sentiment", "bert_sentiment",
               "roberta_sentiment", "emotion", "spam_label", "sarcasm_label"]].copy()
    if hide_spam: vdf = vdf[vdf["spam_label"] != "spam"]
    if search:    vdf = vdf[vdf["comment"].str.contains(search, case=False, na=False)]

    st.markdown(f"Showing **{min(n_rows, len(vdf)):,}** of **{len(vdf):,}** comments")
    st.dataframe(vdf.head(n_rows), use_container_width=True, hide_index=True)