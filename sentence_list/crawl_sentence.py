from googleapiclient.discovery import build
import csv
from dotenv import load_dotenv
import os
import time

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Hàm để crawl comments từ một video YouTube
def crawl_youtube_comments(video_id, max_results=200):
    youtube = build("youtube", "v3", developerKey=API_KEY)

    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )

    while request and len(comments) < max_results:
        response = request.execute()

        for item in response["items"]:
            text = item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
            comments.append({
                "video_id": video_id,
                "comment": text
            })

        request = youtube.commentThreads().list_next(request, response)

    return comments[:max_results]

# Hàm để đọc video IDs từ file
def read_video_ids(file_path):
    path = os.path.abspath(file_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video ID file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# Hàm để lưu dữ liệu vào CSV
def save_to_csv(data, output_path=None):
    """Write a list of dicts to CSV. If output_path is missing, a default
    location under the current working directory will be used."""
    if output_path is None:
        output_path = "raw_data/raw_comment.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["video_id", "comment"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)

script_dir = os.path.dirname(os.path.abspath(__file__))
video_id_file = os.path.join(script_dir, "youtube_id.txt")
video_ids = read_video_ids(video_id_file)

all_comments = []

for vid in video_ids:
    print(f"Đang crawl video: {vid}")
    comments = crawl_youtube_comments(vid, max_results=200)
    all_comments.extend(comments)
    time.sleep(1)  # tránh bị quota limit

output_csv = os.path.join(script_dir, "raw_data", "raw_comment.csv")
save_to_csv(all_comments, output_csv)

print(f"Đã lưu tổng cộng {len(all_comments)} comments vào raw_data/raw_comment.csv")