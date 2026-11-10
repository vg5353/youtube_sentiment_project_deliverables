import streamlit as st
import sqlite3
import pandas as pd

# Page Configuration
st.set_page_config(page_title="YouTube Sentiment Analytics", page_icon="📊", layout="wide")

st.title("🎬 AI-Agentic YouTube Sentiment Analysis Dashboard")
st.markdown("Automated sentiment analysis pipeline powered by **smolagents**, **YouTube Data API v3**, **DistilBERT**, and **SQLite**.")

# Connect to the SQLite Database
@st.cache_data
def load_data():
    conn = sqlite3.connect("youtube_sentiment.db")
    channels = pd.read_sql_query("SELECT * FROM channels", conn)
    videos = pd.read_sql_query("SELECT * FROM videos", conn)
    comments = pd.read_sql_query("SELECT * FROM comments", conn)
    
    video_summary = pd.read_sql_query('''
        SELECT 
            v.video_title,
            ch.channel_title,
            COUNT(c.comment_id) AS total_comments,
            SUM(CASE WHEN c.sentiment_label = 'POSITIVE' THEN 1 ELSE 0 END) AS positive_count,
            SUM(CASE WHEN c.sentiment_label = 'NEGATIVE' THEN 1 ELSE 0 END) AS negative_count,
            ROUND(AVG(c.sentiment_score), 4) AS avg_confidence_score
        FROM videos v
        JOIN channels ch ON v.channel_id = ch.channel_id
        LEFT JOIN comments c ON v.video_id = c.video_id
        GROUP BY v.video_id, v.video_title, ch.channel_title
    ''', conn)
    
    video_summary['positive_pct'] = (video_summary['positive_count'] / video_summary['total_comments'].replace(0, 1) * 100).round(2)
    video_summary['negative_pct'] = (video_summary['negative_count'] / video_summary['total_comments'].replace(0, 1) * 100).round(2)
    conn.close()
    return channels, videos, comments, video_summary

try:
    channels_df, videos_df, comments_df, video_summary_df = load_data()

    # Top KPI Metrics
    total_vids = len(videos_df)
    total_comms = len(comments_df)
    pos_comms = len(comments_df[comments_df['sentiment_label'] == 'POSITIVE'])
    neg_comms = len(comments_df[comments_df['sentiment_label'] == 'NEGATIVE'])
    pos_rate = round((pos_comms / total_comms * 100), 1) if total_comms > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Videos Analyzed", total_vids)
    col2.metric("Total Comments", total_comms)
    col3.metric("Positive Sentiment Rate", f"{pos_rate}%")
    col4.metric("Channels Tracked", len(channels_df))

    st.divider()

    # Video-Wise Sentiment Chart & Table
    st.subheader("📈 Video-Wise Sentiment Breakdown")
    chart_data = video_summary_df.set_index('video_title')[['positive_count', 'negative_count']]
    st.bar_chart(chart_data, color=["#2ecc71", "#e74c3c"])

    st.dataframe(video_summary_df, use_container_width=True)

    st.divider()

    # Interactive Comment Explorer
    st.subheader("💬 Raw Comment Explorer")
    selected_video = st.selectbox("Select a Video to Inspect Comments:", ["All Videos"] + list(videos_df['video_title'].unique()))
    sentiment_filter = st.radio("Filter by Sentiment:", ["ALL", "POSITIVE", "NEGATIVE"], horizontal=True)

    merged_comments = comments_df.merge(videos_df[['video_id', 'video_title']], on='video_id', how='left')

    if selected_video != "All Videos":
        merged_comments = merged_comments[merged_comments['video_title'] == selected_video]
    if sentiment_filter != "ALL":
        merged_comments = merged_comments[merged_comments['sentiment_label'] == sentiment_filter]

    st.dataframe(
        merged_comments[['video_title', 'author_name', 'comment_text', 'like_count', 'sentiment_label', 'sentiment_score']],
        use_container_width=True
    )

except Exception as e:
    st.error(f"Could not load 'youtube_sentiment.db'. Make sure 'youtube_sentiment.db' is in the same folder as app.py! Error: {e}")