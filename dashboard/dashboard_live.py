"""
Live Analysis tab (Tab 10) — fetches real YouTube comments via the Data API
and runs RoBERTa, emotion/spam/sarcasm/fake detection, and BERTopic against
them. Mirrors the same architecture as the offline pipeline, just live.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import dashboard_data as dd
from dashboard_styles import theme, make_pie, make_bar_v, sec_hdr, PALETTE, COLOR_SENTIMENT
from dashboard_components import render_badge_card, render_topic_card, render_wordcloud_block, render_sarcasm_correction_block


def render_tab_live(df):
    sec_hdr("Real-Time YouTube Analysis", "SECTION 10 · LIVE")

    st.markdown("""
    <div class='live-section'>
        <div style='font-size:13px;color:rgba(255,255,255,0.9);line-height:1.8'>
            Enter any YouTube video URL below. The pipeline will fetch comments via
            the YouTube Data API v3, run the best model RoBERTA, detect emotions,
            flag spam and sarcasm, and display a full breakdown — live.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_url, col_btn = st.columns([4, 1])
    with col_url:
        video_url = st.text_input("YouTube Video URL",
                                  placeholder="https://www.youtube.com/watch?v=...",
                                  label_visibility="collapsed")
    with col_btn:
        analyze_clicked = st.button("Analyze Video", use_container_width=True)

    live_max_comments = st.slider(
        "Comments to analyze", min_value=20, max_value=300, value=100, step=20,
        help="Fewer comments = faster analysis. RoBERTa and BERTopic both scale with this number."
    )

    if not (analyze_clicked and video_url.strip()):
        if analyze_clicked:
            st.warning("Please enter a YouTube video URL.")
        return

    rt_df, video_info = _fetch_video_and_comments(df, video_url, live_max_comments)
    if rt_df is None:
        return  # error/warning already shown inside the helper

    if video_info:
        _render_video_details_card(video_info)

    rt_df, rt_top_topic = _run_models(rt_df)
    st.markdown('<div class="rt-status">✓ All models complete</div>', unsafe_allow_html=True)

    _render_results(rt_df, rt_top_topic)


# ─────────────────────────────────────────
# STEP 1 — fetch video details + comments
# ─────────────────────────────────────────
def _fetch_video_and_comments(df, video_url, live_max_comments):
    with st.spinner("Fetching video details..."):
        video_info = None
        if dd.MODELS_LOADED:
            try:
                video_info = dd.get_video_details(video_url)
            except Exception as ex:
                st.info(f"Could not fetch video details: {ex}")

    with st.spinner("Fetching comments from YouTube API..."):
        try:
            rt_df = dd.fetch_comments(video_url, max_comments=live_max_comments) if dd.MODELS_LOADED else (
                df.sample(min(live_max_comments, len(df)))[["clean_comment"]]
                  .rename(columns={"clean_comment": "comment"})
                  .reset_index(drop=True)
            )
        except Exception as ex:
            st.error(f"Could not fetch comments: {ex}")
            return None, None

    if rt_df is None or len(rt_df) == 0:
        st.warning("No comments found for this video.")
        return None, None

    raw_col = "comment" if "comment" in rt_df.columns else rt_df.columns[0]
    rt_df["original_comment"] = rt_df[raw_col]
    rt_df["clean_comment"]    = rt_df[raw_col].apply(dd.clean_comment)

    st.markdown(f'<div class="rt-status">✓ Fetched <strong>{len(rt_df)}</strong> comments</div>',
                unsafe_allow_html=True)
    return rt_df, video_info


def _render_video_details_card(video_info):
    vcol1, vcol2 = st.columns([1, 3])
    with vcol1:
        if video_info["thumbnail"]:
            st.image(video_info["thumbnail"], use_container_width=True)
    with vcol2:
        published = video_info["published_at"][:10] if video_info["published_at"] else "N/A"
        likes_txt = f'{video_info["likes"]:,}' if video_info["likes"] is not None else "Hidden"
        yt_comments_txt = f'{video_info["comment_count"]:,}' if video_info["comment_count"] is not None else "Hidden"
        subs_txt = f'{video_info["subscriber_count"]:,}' if video_info["subscriber_count"] is not None else "Hidden"
        st.markdown(f"""
        <div class='card'>
            <div style='font-size:16px;font-weight:700;color:#F1F1FA;
                        font-family:"Space Grotesk",sans-serif;margin-bottom:6px'>
                {video_info["title"]}
            </div>
            <div style='font-size:13px;color:#9494B8;margin-bottom:12px'>
                {video_info["channel"]} &nbsp;·&nbsp; Published {published}
            </div>
            <span class='badge badge-neu'>👁 {video_info["views"]:,} views</span>
            <span class='badge badge-pos'>👍 {likes_txt} likes</span>
            <span class='badge badge-spam'>💬 {yt_comments_txt} comments (YouTube total)</span>
            <span class='badge badge-sarc'>👥 {subs_txt} subscribers</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# STEP 2 — run the model pipeline
# ─────────────────────────────────────────
def _run_models(rt_df):
    with st.spinner("Running RoBERTa sentiment analysis..."):
        if dd.MODELS_LOADED:
            rt_df["roberta"] = dd.predict_roberta_batch(rt_df["original_comment"].tolist())
        else:
            rt_df["roberta"] = rt_df["original_comment"].apply(dd.detect_emotion).map(
                lambda x: "positive" if x in ["joy","anticipation"] else
                          "negative" if x in ["anger","sadness","disgust","fear"] else "neutral"
            )

    with st.spinner("Detecting emotions, spam, sarcasm, fake comments..."):
        rt_df["emotion"]  = rt_df["original_comment"].apply(dd.detect_emotion)
        rt_df["spam"]     = rt_df["original_comment"].apply(dd.detect_spam)
        rt_df["sarcasm"]  = rt_df["original_comment"].apply(dd.detect_sarcasm)

        rt_df = dd.detect_fake_comments(rt_df, comment_col="original_comment")
        rt_df["fake"] = rt_df["fake_label"]

    rt_top_topic = "N/A"
    with st.spinner("Discovering topics with BERTopic..."):
        try:
            if dd.TOPIC_MODEL_LOADED and len(rt_df) >= 5:
                embed_model = dd.get_live_embedding_model()
                topics_df = dd.get_bertopic_topics(
                    rt_df["clean_comment"].astype(str).tolist(),
                    embedding_model=embed_model,
                    min_topic_size=3
                )
                rt_df["topic"]          = topics_df["topic_number"].values
                rt_df["topic_keywords"] = topics_df["topic_keywords"].values
                non_outlier = topics_df[topics_df["topic_number"] != -1]
                if not non_outlier.empty:
                    rt_top_topic = non_outlier["topic_keywords"].mode().iloc[0]
            else:
                rt_df["topic"] = -1
        except Exception as ex:
            rt_df["topic"] = -1
            st.info(f"Topic modeling skipped: {ex}")

    return rt_df, rt_top_topic


# ─────────────────────────────────────────
# STEP 3 — KPIs, charts, tables
# ─────────────────────────────────────────
def _render_results(rt_df, rt_top_topic):
    st.markdown("#### Results")
    rt_total = len(rt_df)
    rt_pos   = (rt_df["roberta"] == "positive").sum()
    rt_neg   = (rt_df["roberta"] == "negative").sum()
    rt_neu   = (rt_df["roberta"] == "neutral").sum()
    rt_spam  = (rt_df["spam"]    == "spam").sum()
    rt_sarc  = (rt_df["sarcasm"] == "sarcastic").sum()
    rt_fake  = (rt_df["fake"]    == "fake").sum()
    rt_dominant_emotion = rt_df["emotion"].value_counts().idxmax() if rt_total else "N/A"

    def rt_pct(n): return f"{n/rt_total*100:.1f}%"

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Comments", rt_total)
    k2.metric("Positive", rt_pct(rt_pos), f"{rt_pos} comments")
    k3.metric("Negative", rt_pct(rt_neg), f"{rt_neg} comments")
    k4.metric("Neutral",  rt_pct(rt_neu), f"{rt_neu} comments")
    k5.metric("Spam",     rt_pct(rt_spam), f"{rt_spam} flagged")
    k6.metric("Sarcasm",  rt_pct(rt_sarc), f"{rt_sarc} detected")

    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(
            make_pie(rt_df["roberta"], "RoBERTa Sentiment Distribution",
                     color_map=COLOR_SENTIMENT),
            use_container_width=True
        )
    with ch2:
        st.plotly_chart(make_bar_v(rt_df["emotion"], "emotion", "Emotion Detection"),
                         use_container_width=True)

    ch3, ch4 = st.columns(2)
    with ch3:
        render_badge_card(
            "Executive Summary",
            rows=[
                ("Total comments analysed", f"<span class='badge badge-neu'>{rt_total:,}</span>"),
                ("Positive sentiment",      f"<span class='badge badge-pos'>{rt_pct(rt_pos)}</span>"),
                ("Negative sentiment",      f"<span class='badge badge-neg'>{rt_pct(rt_neg)}</span>"),
                ("Spam detected",           f"<span class='badge badge-spam'>{rt_pct(rt_spam)}</span>"),
                ("Sarcasm detected",        f"<span class='badge badge-sarc'>{rt_pct(rt_sarc)}</span>"),
                ("Dominant emotion",        f"<span class='badge badge-neu'>{rt_dominant_emotion}</span>"),
                ("Top topic",               f"<span class='badge badge-neu'>{rt_top_topic}</span>"),
                ("Fake comments",           f"<span class='badge badge-fake'>{rt_pct(rt_fake)}</span>"),
            ],
        )

    with ch4:
        dq_data = pd.DataFrame({
            "Category": ["Clean", "Spam", "Sarcastic"],
            "Count":    [rt_total - rt_spam, rt_spam, rt_sarc]
        })
        fig_dq = px.bar(dq_data, x="Category", y="Count", color="Category",
                        color_discrete_sequence=["#39FF14", "#FF2D6D", "#B14EFF"],
                        text="Count")
        fig_dq.update_traces(texttemplate="%{text}", textposition="outside",
                             marker_line_width=0)
        fig_dq.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
        theme(fig_dq, "Data Quality — Spam & Sarcasm")
        st.plotly_chart(fig_dq, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    sec_hdr("Sarcasm-Corrected Sentiment", "BEYOND FACE VALUE")
    render_sarcasm_correction_block(rt_df["roberta"], rt_df["sarcasm"])

    render_wordcloud_block(
        full_df=rt_df, text_col="clean_comment", sentiment_col="roberta",
        filter_options={"All": None, "Positive": "positive", "Negative": "negative"},
        cmap_map={"All": "cool", "Positive": "spring", "Negative": "Wistia"},
        radio_label="Filter by:", key_prefix="live_wc",
        wc_size=(800, 320), max_words=100, prefer_horizontal=None,
        figsize=(10, 3.5), top_n=8, chart_title="Top Words (Live)",
        header="Word Cloud — Live Comments",
    )

    st.markdown("**Live Topic Modeling (BERTopic)**")
    if rt_top_topic != "N/A":
        topic_cards = rt_df[rt_df["topic"] != -1].drop_duplicates("topic")[["topic", "topic_keywords"]]
        for i, (_, row) in enumerate(topic_cards.iterrows()):
            render_topic_card(row["topic"], row["topic_keywords"], PALETTE[i % len(PALETTE)])
    else:
        st.info("Not enough comments for topic modeling (need at least 5, and sentence-transformers/bertopic installed).")

    st.markdown("**Full Comment Results**")
    sc1, sc2 = st.columns([3, 1])
    with sc1: search_rt    = st.text_input("Search live comments", placeholder="Search keyword...", key="rt_search")
    with sc2: hide_spam_rt = st.checkbox("Hide spam comments", value=True, key="rt_spam")

    display_rt = rt_df[["original_comment", "clean_comment", "roberta",
                         "emotion", "spam", "fake", "sarcasm"]].copy()
    display_rt.columns = ["Original Comment", "Clean Comment", "RoBERTa Sentiment",
                           "Emotion", "Spam", "Fake", "Sarcasm"]
    if hide_spam_rt: display_rt = display_rt[display_rt["Spam"] != "spam"]
    if search_rt:    display_rt = display_rt[
        display_rt["Original Comment"].str.contains(search_rt, case=False, na=False)]

    st.markdown(f"Showing **{min(100, len(display_rt)):,}** of **{len(display_rt):,}** comments")
    st.dataframe(display_rt.head(100), use_container_width=True, hide_index=True)