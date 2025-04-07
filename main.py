import os
import cv2
import numpy as np
import aiofiles
import requests
from fastapi import FastAPI, UploadFile, File
import uvicorn
from ultralytics import YOLO
from database import db, upload_to_imgbb, upload_to_streamable
import threading

app = FastAPI()

# ✅ Tải mô hình từ Google Drive
def download_model():
    file_url = 'https://drive.google.com/uc?id=1GNc8GNxEhlU4f2gOHvVpIWLqbDIjgU2R&export=download'
    response = requests.get(file_url, stream=True)

    # Kiểm tra nếu tải thành công
    if response.status_code == 200:
        with open('best_model.pt', 'wb') as f:
            for chunk in response.iter_content(chunk_size=128):
                f.write(chunk)
        print("Model downloaded successfully!")
    else:
        print("Failed to download the model.")

# ✅ Kiểm tra nếu file mô hình đã tồn tại, nếu không thì tải xuống
MODEL_PATH = r"best_model.pt"
if not os.path.exists(MODEL_PATH):
    download_model()

# ✅ Load YOLOv8 model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy file {MODEL_PATH}")
model = YOLO(MODEL_PATH)

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Không thể giải mã hình ảnh."}

    results = model(image)
    predictions = []
    person_count = 0

    for result in results:
        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].tolist())
            score = round(result.boxes.conf[i].item(), 2)
            label = model.names[int(result.boxes.cls[i].item())]

            if label.lower() == "swimmer":
                person_count += 1

            predictions.append({
                "label": label,
                "confidence": score,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

    image_with_boxes = results[0].plot()
    output_path = "output.jpg"
    cv2.imwrite(output_path, image_with_boxes)
    imgbb_url = upload_to_imgbb(output_path)

    db.predictions.insert_one({
        "image_url": imgbb_url,
        "predictions": predictions,
        "person_count": person_count
    })

    return {"person_count": person_count, "image_url": imgbb_url}


@app.post("/predict-video/")
async def predict_video(file: UploadFile = File(...)):
    input_video_path = "temp_input.mp4"
    async with aiofiles.open(input_video_path, "wb") as f:
        await f.write(await file.read())

    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = "output_video.mp4"
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    total_swimmer_count = 0
    prev_swimmer_count = 0
    prev_counts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 == 0:
            results = model(frame)
            frame = results[0].plot()

            current_swimmer_count = sum(
                1 for box in results[0].boxes if model.names[int(box.cls.item())] == "swimmer"
            )
            prev_counts.append(current_swimmer_count)

            if len(prev_counts) == 3:
                avg_swimmer_count = round(sum(prev_counts) / 3)
                if avg_swimmer_count != prev_swimmer_count:
                    total_swimmer_count = avg_swimmer_count
                    prev_swimmer_count = avg_swimmer_count
                prev_counts = []

        out.write(frame)

    cap.release()
    out.release()

    streamable_url = upload_to_streamable(output_video_path)
    db.predictions.insert_one({
        "video_url": streamable_url,
        "total_swimmer_count": total_swimmer_count
    })

    return {
        "total_swimmer_count": total_swimmer_count,
        "video_url": streamable_url
    }


@app.get("/realtime/")
def realtime_detection():
    def detect():
        cap = cv2.VideoCapture(0)  # webcam, hoặc thay bằng đường dẫn video
        if not cap.isOpened():
            print("❌ Không thể mở webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            swimmer_count = sum(
                1 for box in results[0].boxes if model.names[int(box.cls.item())] == "swimmer"
            )

            frame = results[0].plot()
            cv2.putText(frame, f"Swimmers: {swimmer_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Realtime Detection", frame)

            db.predictions.insert_one({
                "source": "realtime",
                "swimmer_count": swimmer_count
            })

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    threading.Thread(target=detect).start()
    return {"message": "Realtime detection started. Press Q to stop."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
