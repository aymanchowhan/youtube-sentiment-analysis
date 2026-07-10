"""
Reusable UI building blocks shared across tabs.

These three patterns previously appeared duplicated, near-identically,
across multiple tabs in the original single-file dashboard:
  - a titled "badge list" card       (Key Findings / Quality Summary / Executive Summary)
  - a colored left-border topic card (offline Topics tab / Live topic cards)
  - a wordcloud + top-words block    (offline Word Cloud tab / Live word cloud)
Factoring them out here means a style tweak only needs to happen once.
"""
from collections import Counter

import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from dashboard_styles import make_bar_h, make_pie, COLOR_SENTIMENT


def render_badge_card(title, rows, css_class="insight", margin_bottom=12,
                       wrap_content=True, entry_sep="<br>"):
    """
    A card with a title and a list of (label, badge_html) rows.

    rows: list of (label_text, badge_html) tuples, e.g.
          [("Positive sentiment", "<span class='badge badge-pos'>62%</span>")]
    css_class: "insight" (indigo left border) or "card" (plain white card).
    entry_sep: "<br>" for single spacing, "<br><br>" for extra breathing room.
    """
    rows_html = entry_sep.join(f"{label}<br>{badge}" for label, badge in rows)
    content = (
        f"<div style='font-size:13px;color:#C6C6E0;line-height:2.2'>{rows_html}</div>"
        if wrap_content else rows_html
    )
    st.markdown(f"""
    <div class='{css_class}'>
        <div style='font-size:11px;font-weight:700;color:#D4FF3D;font-family:"Space Grotesk",sans-serif;
                    text-transform:uppercase;letter-spacing:.08em;margin-bottom:{margin_bottom}px'>
            ⚡ {title}
        </div>
        {content}
    </div>
    """, unsafe_allow_html=True)


def clean_topic_name(name):
    """Turn BERTopic's raw offline 'Name' field ('0_camera_phone_battery')
    into a readable keyword string ('Camera, Phone, Battery'). Only needed
    for the offline topic_summary.csv — Live Analysis topics are already
    cleaned at the source by topic_modeling.get_bertopic_topics()."""
    if not isinstance(name, str):
        return "N/A"
    parts = name.split("_")
    if parts and parts[0].lstrip("-").isdigit():
        parts = parts[1:]
    return ", ".join(p.capitalize() for p in parts if p) or "N/A"


def render_sarcasm_correction_block(sentiment_series, sarcasm_series,
                                     sarcastic_value="sarcastic",
                                     positive_value="positive", negative_value="negative"):
    """
    Compares NAIVE sentiment (what the model says at face value) against a
    SARCASM-CORRECTED version, where comments labeled positive but flagged
    as sarcastic get flipped to negative.

    Rationale: sarcasm on YouTube typically disguises criticism as fake
    praise ("wow, brilliant decision" said mockingly) — a sentiment model
    reading only the words will call that positive. This is a simplifying
    assumption (sarcasm doesn't ALWAYS mean disguised negativity), stated
    here as an explicit, visible limitation rather than hidden.

    sentiment_series / sarcasm_series must share the same index (i.e. come
    from the same dataframe / same set of rows).

    Returns the corrected sentiment Series, in case the caller wants it.
    """
    corrected = sentiment_series.copy()
    flip_mask = (sentiment_series == positive_value) & (sarcasm_series == sarcastic_value)
    corrected.loc[flip_mask] = negative_value

    total = len(sentiment_series)
    naive_pos_pct     = (sentiment_series == positive_value).mean() * 100 if total else 0
    corrected_pos_pct = (corrected == positive_value).mean() * 100 if total else 0
    n_flipped = int(flip_mask.sum())
    gap = naive_pos_pct - corrected_pos_pct

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.plotly_chart(make_pie(sentiment_series, "Naive Sentiment (face value)",
                                  color_map=COLOR_SENTIMENT), use_container_width=True)
    with col2:
        st.plotly_chart(make_pie(corrected, "Sarcasm-Corrected Sentiment",
                                  color_map=COLOR_SENTIMENT), use_container_width=True)
    with col3:
        render_badge_card(
            "Sarcasm Correction Impact",
            rows=[
                ("Naive positive rate",      f"<span class='badge badge-pos'>{naive_pos_pct:.1f}%</span>"),
                ("Corrected positive rate",  f"<span class='badge badge-neg'>{corrected_pos_pct:.1f}%</span>"),
                ("Comments flipped",         f"<span class='badge badge-sarc'>{n_flipped}</span>"),
                ("Positivity overstated by", f"<span class='badge badge-fake'>{gap:.1f} pp</span>"),
            ],
        )

    st.markdown(f"""
    <div class='insight' style='margin-top:8px'>
        Taken at face value, <strong>{naive_pos_pct:.1f}%</strong> of comments read as positive.
        But <strong>{n_flipped}</strong> of those were also flagged as sarcastic — positive
        wording used to disguise criticism. Correcting for this, real positive sentiment is
        <strong>{corrected_pos_pct:.1f}%</strong> — naive sentiment analysis overstated
        positivity by <strong>{gap:.1f} percentage points</strong>.
        <em>Limitation: this assumes sarcasm always disguises negativity, which is the common
        pattern but not universal.</em>
    </div>
    """, unsafe_allow_html=True)

    return corrected


def render_topic_card(topic_id, keywords, color, count=None):
    """A single colored-left-border topic card. `count`, if given, shows
    '· N comments' next to the topic number."""
    label = f"Topic {int(topic_id)}"
    if count is not None:
        label += f" &nbsp;·&nbsp; {int(count)} comments"
    st.markdown(f"""
    <div style='background:#12121C;border:1px solid rgba(255,255,255,0.06);
                border-radius:0 18px 18px 0;border-left:3px solid {color};
                padding:12px 16px;margin:8px 0;box-shadow:0 4px 18px {color}26'>
        <span style='color:{color};font-weight:700;font-size:11px;font-family:"Space Grotesk",sans-serif;
                     text-transform:uppercase;letter-spacing:.06em'>
            ⚡ {label}
        </span><br>
        <span style='color:#E4E4F5;font-size:13px'>{keywords}</span>
    </div>
    """, unsafe_allow_html=True)


def render_wordcloud_block(full_df, text_col, sentiment_col, filter_options, cmap_map,
                            radio_label, key_prefix, wc_size=(900, 360), max_words=120,
                            prefer_horizontal=None, figsize=(12, 4), top_n=10,
                            chart_title="Top Words", header=None):
    """
    Renders a [word cloud | top-words bar chart] two-column block.

    filter_options: dict {radio_label: value_to_match_in_sentiment_col}.
                     Use value=None for "no filter" (e.g. "All Comments").
    The top-words chart always uses the FULL `full_df`, independent of the
    word cloud's radio filter — matches the original dashboard's behavior.
    """
    if header:
        st.markdown(f"**{header}**")

    col1, col2 = st.columns([2, 1])

    with col1:
        wc_opt = st.radio(radio_label, list(filter_options.keys()),
                           horizontal=True, key=f"{key_prefix}_radio")
        match_value = filter_options[wc_opt]
        filtered = full_df if match_value is None else full_df[full_df[sentiment_col] == match_value]

        wtext = " ".join(filtered[text_col].dropna().astype(str))
        if len(wtext.strip()) > 20:
            wc_kwargs = dict(width=wc_size[0], height=wc_size[1], background_color="#0A0A12",
                              colormap=cmap_map[wc_opt], max_words=max_words, collocations=False)
            if prefer_horizontal is not None:
                wc_kwargs["prefer_horizontal"] = prefer_horizontal
            wc_obj = WordCloud(**wc_kwargs).generate(wtext)
            fig_wc, ax = plt.subplots(figsize=figsize)
            fig_wc.patch.set_facecolor("#0A0A12")
            ax.set_facecolor("#0A0A12")
            ax.imshow(wc_obj, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info("Not enough text for word cloud.")

    with col2:
        all_words = " ".join(full_df[text_col].dropna().astype(str)).split()
        tw_df = pd.DataFrame(Counter(all_words).most_common(top_n), columns=["Word", "Count"])
        fig_tw = make_bar_h(tw_df.sort_values("Count"), "Count", "Word", chart_title)
        st.plotly_chart(fig_tw, use_container_width=True)