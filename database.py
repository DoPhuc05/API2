import requests
import os
from pymongo import MongoClient

# ✅ Kết nối MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
try:
    client = MongoClient(MONGO_URI)
    db = client["DACN2"]
    print("✅ Kết nối MongoDB thành công!")

    # ✅ Kiểm tra & tạo collection nếu chưa có
    if "predictions" not in db.list_collection_names():
        db.create_collection("predictions")
        print("✅ Collection `predictions` đã được tạo!")

except Exception as e:
    print(f"❌ Lỗi kết nối MongoDB: {e}")

# ✅ Cấu hình API của ImgBB
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")

def upload_to_imgbb(file_path):
    """Tải file lên ImgBB và trả về URL công khai"""
    with open(file_path, "rb") as file:
        response = requests.post(
            f"https://api.imgbb.com/1/upload?key={IMGBB_API_KEY}",
            files={"image": file}
        )

    # ✅ Kiểm tra kết quả upload
    if response.status_code == 200:
        return response.json()["data"]["url"]
    else:
        print(f"❌ Lỗi upload ImgBB: {response.json()}")
        return None

# ✅ Cấu hình API của Streamable
STREAMABLE_USERNAME = os.getenv("STREAMABLE_USERNAME")
STREAMABLE_PASSWORD = os.getenv("STREAMABLE_PASSWORD")

def upload_to_streamable(file_path):
    """Upload video lên Streamable và trả về URL"""
    with open(file_path, "rb") as video_file:
        response = requests.post(
            "https://api.streamable.com/upload",
            auth=(STREAMABLE_USERNAME, STREAMABLE_PASSWORD),
            files={"file": video_file}
        )

    # ✅ Kiểm tra kết quả upload
    if response.status_code == 200:
        return f"https://streamable.com/{response.json().get('shortcode', '')}"
    else:
        print(f"❌ Lỗi upload Streamable: {response.json()}")
        return None
