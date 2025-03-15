import os
import cv2
import numpy as np
import aiofiles
from fastapi import FastAPI, UploadFile, File
import uvicorn
from ultralytics import YOLO
from database import db, upload_to_imgbb, upload_to_streamable  # 🔥 Sửa import
from collections import deque  # 🔥 Lưu lịch sử số lượng swimmer
import gdown

# ✅ Load mô hình YOLOv8
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "best (3).pt")

GDRIVE_FILE_ID = "1Ay0CueS1oS4AxD_u8igoUtj_Z4fwfqT6"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# ✅ Tải file nếu chưa có
if not os.path.exists(MODEL_PATH):
    print("🔄 Đang tải mô hình YOLO từ Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    print("✅ Mô hình đã tải xong!")

# ✅ Khởi tạo model
print(f"🔄 Đang tải mô hình YOLOv8 từ {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Mô hình YOLOv8 đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi khi load mô hình: {e}")
    model = None

# ✅ Khởi tạo FastAPI
app = FastAPI()

# ✅ XỬ LÝ ẢNH & LƯU VÀO MONGODB + IMGBB
@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    """Nhận ảnh, chạy YOLOv8, lưu MongoDB & ImgBB"""
    image_bytes = await file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    results = model(image)
    predictions = []
    person_count = 0

    for result in results:
        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].tolist())
            score = round(result.boxes.conf[i].item(), 2)
            label = model.names[int(result.boxes.cls[i].item())]  # 🔥 Sửa lỗi cls

            if label == "swimmer":
                person_count += 1

            predictions.append({
                "label": label,
                "confidence": score,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

    # ✅ Lưu ảnh có bounding box
    image_with_boxes = results[0].plot()
    output_path = "output.jpg"
    cv2.imwrite(output_path, image_with_boxes)

    # ✅ Upload lên ImgBB
    imgbb_url = upload_to_imgbb(output_path)

    # ✅ Lưu link ảnh & số người vào MongoDB
    db.predictions.insert_one({
        "image_url": imgbb_url,
        "predictions": predictions,
        "person_count": person_count
    })

    return {"person_count": person_count, "image_url": imgbb_url}

# ✅ XỬ LÝ VIDEO & LƯU VÀO MONGODB + STREAMABLE
@app.post("/predict-video/")
async def predict_video(file: UploadFile = File(...)):
    """Nhận video, xử lý bằng YOLOv8, lưu MongoDB & Streamable"""
    input_video_path = "temp_input.mp4"
    async with aiofiles.open(input_video_path, "wb") as f:
        await f.write(await file.read())

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return {"error": "Không thể mở video!"}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_swimmer_count = 0
    prev_counts = deque(maxlen=3)  # 🔥 Lưu lịch sử 3 frame gần nhất

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 == 0:
            results = model(frame)

            if results and len(results) > 0:
                frame = results[0].plot()
                current_swimmer_count = sum(
                    1 for box in results[0].boxes if model.names[int(box.cls[i].item())] == "swimmer"
                )

                prev_counts.append(current_swimmer_count)  # Lưu số swimmer của frame hiện tại

                # 🔥 Cập nhật số swimmer nếu có thay đổi trong 3 frame liên tiếp
                if len(prev_counts) == 3:
                    avg_swimmer_count = round(sum(prev_counts) / 3)
                    if avg_swimmer_count != total_swimmer_count:
                        total_swimmer_count = avg_swimmer_count  # Cập nhật số swimmer

        out.write(frame)

    cap.release()
    out.release()

    # ✅ Upload video lên Streamable
    streamable_url = upload_to_streamable(output_video_path)

    # ✅ Lưu vào MongoDB
    db.predictions.insert_one({
        "video_url": streamable_url,
        "total_swimmer_count": total_swimmer_count
    })

    return {"total_swimmer_count": total_swimmer_count, "video_url": streamable_url}

# ✅ Chạy FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)  # 🔥 Đổi port 10000
