import os
import requests
import urllib.parse
from pymongo import MongoClient

# ✅ Mã hóa username & password cho MongoDB URI
username = urllib.parse.quote_plus("phuc201005")
password = urllib.parse.quote_plus("hj9gg4lWz5Hp7w83")

# ✅ Kết nối MongoDB (Fix lỗi "Username and password must be escaped")
MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.ujsoo.mongodb.net/?retryWrites=true&w=majority"

try:
    client = MongoClient(MONGO_URI)
    db = client["DACN2"]  # ✅ Đổi tên DB nếu cần
    print("✅ Kết nối MongoDB thành công!")

    # ✅ Kiểm tra & tạo collection nếu chưa có
    if "predictions" not in db.list_collection_names():
        db.create_collection("predictions")
        print("✅ Collection `predictions` đã được tạo!")

except Exception as e:
    print(f"❌ Lỗi kết nối MongoDB: {e}")

# ✅ API Key của ImgBB (Lưu ảnh)
IMGBB_API_KEY = "13e5fc0624e61eb94fc4e9ab0d6b8c4e"

def upload_to_imgbb(file_path):
    """Tải file lên ImgBB và trả về URL công khai"""
    with open(file_path, "rb") as file:
        response = requests.post(
            f"https://api.imgbb.com/1/upload?key={IMGBB_API_KEY}",
            files={"image": file}
        )
    if response.status_code == 200:
        return response.json()["data"]["url"]
    else:
        print(f"❌ Lỗi upload ImgBB: {response.json()}")
        return None

# ✅ API của Streamable (Lưu video)
STREAMABLE_USERNAME = "phuc201005@gmail.com"
STREAMABLE_PASSWORD = "dphc052010"

def upload_to_streamable(file_path):
    """Upload video lên Streamable và trả về URL"""
    with open(file_path, "rb") as video_file:
        response = requests.post(
            "https://api.streamable.com/upload",
            auth=(STREAMABLE_USERNAME, STREAMABLE_PASSWORD),
            files={"file": video_file}
        )
    if response.status_code == 200:
        return f"https://streamable.com/{response.json().get('shortcode', '')}"
    else:
        print(f"❌ Lỗi upload Streamable: {response.json()}")
        return None
