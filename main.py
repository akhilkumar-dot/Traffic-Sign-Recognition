import asyncio
import json
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
import uvicorn

model = load_model("traffic_sign_model.h5")

labels = [
    "Speed Limit (20)", "Speed Limit (30)", "Speed Limit (50)", "Speed Limit (60)", "Speed Limit (70)",
    "Speed Limit (80)", "End of Speed Limit (80)", "Speed Limit (100)", "Speed Limit (120)", "No Overtaking",
    "Priority Road", "Yield", "Stop", "No Vehicles", "Vehicles over 3.5t prohibited",
    "No Entry", "General caution", "Dangerous curve left", "Dangerous curve right", "Double curve",
    "Bumpy road", "Slippery road", "Road narrows on right", "Roadwork", "Traffic signals",
    "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead", "Ahead only",
    "Go straight or right", "Go straight or left", "Keep right", "Keep left", "Roundabout", "End of priority road",
    "Caution, dangerous intersection", "Caution, no overtaking", "Caution, slippery road"
]

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)

    try:
        while True:
            # Check for stop message
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                if msg.lower() == "stop":
                    print("Stop signal received.")
                    break
            except asyncio.TimeoutError:
                pass

            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            img = cv2.resize(frame, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            prediction = model.predict(img)
            predicted_class = int(np.argmax(prediction))
            sign_name = labels[predicted_class]
            confidence_score = float(np.max(prediction))

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Send frame and prediction
            await websocket.send_bytes(frame_bytes)
            await websocket.send_text(json.dumps({
                "label": sign_name,
                "confidence": confidence_score
            }))

    except WebSocketDisconnect:
        print("Client disconnected.")
    finally:
        cap.release()
