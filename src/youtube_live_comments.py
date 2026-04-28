from googleapiclient.discovery import build
import pandas as pd

API_KEY = "AIzaSyCbwwd8iTHATcxX5Yv-D0NrxuCq9ZGYyd8"

youtube = build("youtube", "v3", developerKey=API_KEY)


def get_video_id(url):
    return url.split("v=")[-1]


def fetch_comments(video_url, max_comments=500):

    video_id = get_video_id(video_url)

    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request and len(comments) < max_comments:

        response = request.execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        request = youtube.commentThreads().list_next(request, response)

    df = pd.DataFrame(comments, columns=["comment"])

    return df