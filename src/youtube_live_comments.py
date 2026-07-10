from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get YouTube API Key
API_KEY = os.environ["YOUTUBE_API_KEY"]

# Build YouTube API client
youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)


def get_video_id(url):
    parsed_url = urlparse(url)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    return None


def get_video_details(video_url):
    """
    Fetch video + channel stats for the Live Analysis header card:
    title, channel, thumbnail, publish date, views, likes, comment count,
    subscriber count.
    """
    video_id = get_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    video_resp = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()

    items = video_resp.get("items", [])
    if not items:
        raise ValueError("Video not found or unavailable.")

    snippet = items[0]["snippet"]
    stats   = items[0]["statistics"]

    channel_stats = {}
    try:
        channel_resp = youtube.channels().list(
            part="statistics",
            id=snippet["channelId"]
        ).execute()
        channel_items = channel_resp.get("items", [])
        if channel_items:
            channel_stats = channel_items[0]["statistics"]
    except Exception:
        pass

    subscriber_count = None
    if channel_stats and not channel_stats.get("hiddenSubscriberCount", False):
        subscriber_count = int(channel_stats.get("subscriberCount", 0))

    return {
        "title":            snippet.get("title", "N/A"),
        "channel":          snippet.get("channelTitle", "N/A"),
        "published_at":     snippet.get("publishedAt", ""),
        "thumbnail":        snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
        "views":            int(stats.get("viewCount", 0)),
        "likes":            int(stats.get("likeCount", 0)) if "likeCount" in stats else None,
        "comment_count":    int(stats.get("commentCount", 0)) if "commentCount" in stats else None,
        "subscriber_count": subscriber_count,
    }


def fetch_comments(video_url, max_comments=500):
    """
    Fetch up to 500 top-level comments.
    """

    video_id = get_video_id(video_url)

    if not video_id:
        raise ValueError("Invalid YouTube URL")

    comments = []

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            order="relevance"
        )

        while request and len(comments) < max_comments:

            response = request.execute()

            for item in response.get("items", []):

                comment = (
                    item["snippet"]
                    ["topLevelComment"]
                    ["snippet"]
                    ["textDisplay"]
                )

                comments.append(comment)

                if len(comments) >= max_comments:
                    break

            request = youtube.commentThreads().list_next(
                request,
                response
            )

    except Exception as e:
        raise Exception(
            f"Failed to fetch comments: {str(e)}"
        )

    comments = list(dict.fromkeys(comments))

    return pd.DataFrame(
        comments,
        columns=["comment"]
    )