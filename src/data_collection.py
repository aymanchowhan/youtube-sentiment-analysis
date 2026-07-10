from googleapiclient.discovery import build
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()

# SECURITY: never hardcode API keys in source. Put this in a .env file instead:
#   YOUTUBE_API_KEY=your-key-here
API_KEY = os.environ["YOUTUBE_API_KEY"]

youtube = build("youtube", "v3", developerKey=API_KEY)

VIDEO_IDS = {
    "video1": "cBpGq-vDr2Y",
    "video2": "0e3GPea1Tyg",
    "video3": "R1cVsyUXxYM",
    "video4": "g3JUbgOHgdw",
    "video5": "BddP6PYo2gs",
    "video6": "af3Dmfsotbc",
    "video7": "h6fcK_fRYaI",
    "video8": "LhpZJwUboeI",
    "video9": "zzwRbKI2pn4",
    "video10": "dcBXmj1nMTQ"
}

comments = []

for video_name, video_id in VIDEO_IDS.items():

    print(f"\nFetching comments for {video_name}")

    next_page_token = None
    comment_count = 0

    while comment_count < 1000:

        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            )

            response = request.execute()

        except Exception as e:
            print("Error occurred:", e)
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue

        for item in response["items"]:

            if comment_count >= 1000:
                break

            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

            comments.append({
                "video": video_name,
                "comment": comment
            })

            comment_count += 1

        print(f"Collected {comment_count} comments...")

        next_page_token = response.get("nextPageToken")

        if not next_page_token or comment_count >= 1000:
            break

        time.sleep(1)

df = pd.DataFrame(comments)

df.to_csv("data/comments.csv", index=False)

print("\nComments saved successfully in data/comments.csv")