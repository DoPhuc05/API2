import os
import requests
import urllib.parse
from pymongo import MongoClient

# ✅ Lấy MongoDB username & password từ biến môi trường
MONGO_USERNAME = os.getenv("MONGO_USERNAME", "phuc201005")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "hj9gg4lWz5Hp7w83")

# ✅ Mã hóa username & password theo chuẩn RFC 3986
encoded_username = urllib.parse.quote_plus(MONGO_USERNAME)
encoded_password = urllib.parse.quote_plus(MONGO_PASSWORD)

# ✅ Xây dựng URI MongoDB chuẩn
MONGO_URI = f"mongodb+srv://{encoded_username}:{encoded_password}@cluster0.ujsoo.mongodb.net/?retryWrites=true&w=majority"

# ✅ Kết nối MongoDB
try:
    client = MongoClient(MONGO_URI)
    db = client["DACN2"]
    print("✅ Kết nối MongoDB thành công!")
except Exception as e:
    print(f"❌ Lỗi kết nối MongoDB: {e}")

# ✅ Lấy API Key của ImgBB từ biến môi trường
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")

def upload_to_imgbb(file_path):
    """Tải file lên ImgBB và trả về URL công khai"""
    if not IMGBB_API_KEY:
        print("❌ Lỗi: Chưa cấu hình API Key cho ImgBB!")
        return None

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

# ✅ Lấy tài khoản Streamable từ biến môi trường
STREAMABLE_USERNAME = os.getenv("STREAMABLE_USERNAME")
STREAMABLE_PASSWORD = os.getenv("STREAMABLE_PASSWORD")

def upload_to_streamable(file_path):
    """Upload video lên Streamable và trả về URL"""
    if not STREAMABLE_USERNAME or not STREAMABLE_PASSWORD:
        print("❌ Lỗi: Chưa cấu hình tài khoản Streamable!")
        return None

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
