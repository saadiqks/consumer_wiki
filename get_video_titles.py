import os
from googleapiclient.discovery import build
import polars as pl
from datetime import datetime

def get_all_video_ids(youtube, channel_id):
    video_ids = []
    request = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    )
    response = request.execute()
    playlist_id = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    next_page_token = None
    while True:
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        video_ids.extend([item["contentDetails"]["videoId"] for item in response["items"]])
        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return video_ids


def get_video_details(youtube, video_ids):
    videos = []
    # Process in batches of 50 (API limit)
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        request = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(batch)
        )
        response = request.execute()

        for item in response["items"]:
            video = {
                "title": item["snippet"]["title"],
                "published_at": item["snippet"]["publishedAt"],
                "url": f"https://youtube.com/watch?v={item["id"]}",
            }
            videos.append(video)

    return videos


def get_video_list(api_key, channel_id):
    youtube = build("youtube", "v3", developerKey=api_key)

    video_ids = get_all_video_ids(youtube, channel_id)
    videos = get_video_details(youtube, video_ids)

    df = pl.DataFrame(videos)
    df = df.with_columns([
        pl.col("published_at").str.strptime(pl.Datetime)
    ])
    df = df.sort("published_at", descending=True)
    df = df.drop("published_at")
    df.write_csv("rossmann_wiki_tracker.csv")

    print(f"Retrieved {len(videos)} videos")


def add_columns():
    df = pl.read_csv("rossmann_wiki_tracker.csv")

    df = df.with_columns(
        needs_wiki_article = None,
        wiki_url = None,
    )
    df = df.select("video_title", "needs_wiki_article", "wiki_url", "video_url")

    df.write_csv("new_rossmann_wiki_tracker.csv")


def main():
    # CHANNEL_ID = "UCl2mFZoRqjw_ELax4Yisf6w"
    # API_KEY = os.getenv("YOUTUBE_API")

    # get_video_list(API_KEY, CHANNEL_ID)

    add_columns()


if __name__ == "__main__":
    main()
