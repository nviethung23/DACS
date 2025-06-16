import time
import numpy as np
import joblib
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from attention_layer import AttentionLayer
from database import init_db, insert_log

# --- Khởi tạo database (tạo bảng nếu chưa có) ---
init_db()

# --- Đọc danh sách tên feature từ file features.txt ---
with open("features.txt", "r", encoding="utf-8") as f:
    features_names = [line.strip() for line in f if line.strip()]

# --- Load model và scaler/encoder ---
ddos_model = load_model("CNN_ddos_model.h5", custom_objects={"AttentionLayer": AttentionLayer})
attack_model = load_model("attack_classifier.h5", custom_objects={"AttentionLayer": AttentionLayer})
scaler_ddos = joblib.load("scaler_2019.pkl")
scaler_attack = joblib.load("scaler_attack.pkl")
attack_encoder = joblib.load("attack_label_encoder.pkl")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Lưu tối đa 100 log gần nhất trong memory để broadcast qua WebSocket
history = []

rate_limit = {}  # { ip: { "window_start": int, "count": int } }
WINDOW_SIZE = 10  # giây
MAX_REQUESTS_PER_WINDOW = 3000

# Quản lý kết nối WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        living = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
                living.append(connection)
            except WebSocketDisconnect:
                pass
        self.active_connections = living

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_api(request: Request):
    form = await request.form()
    ip = form.get("ip", "unknown")


    now_s = int(time.time())
    window_id = now_s // WINDOW_SIZE
    entry = rate_limit.get(ip)

    if entry is None or entry["window_id"] != window_id:
        # Bắt đầu window mới cho IP này
        rate_limit[ip] = {"window_id": window_id, "count": 1}
    else:
        # Trong cùng window
        entry["count"] += 1
        if entry["count"] > MAX_REQUESTS_PER_WINDOW:
            # Quá giới hạn → trả 429 Too Many Requests
            raise HTTPException(status_code=429, detail="Too Many Requests")

  
    X = np.array([[float(form.get(f, 0)) for f in features_names]])

    # --- 1. Tính thời gian phát hiện DDoS ---
    start_detect = time.time()
    X_ddos = scaler_ddos.transform(X)
    X_ddos = X_ddos.reshape((X_ddos.shape[0], X_ddos.shape[1], 1))
    y_pred = ddos_model.predict(X_ddos)
    is_ddos = (y_pred[0][0] > 0.5)
    end_detect = time.time()
    detection_time = round((end_detect - start_detect) * 1000, 3)  # ms

    timestamp = int(time.time())

    if not is_ddos:
        entry = {
            "ip": ip,
            "result": "Benign",
            "detection_time": detection_time,
            "block_time": 0.0,
            "status_code": 200,
            "sec": timestamp
        }
        history.append(entry)
        if len(history) > 100:
            history.pop(0)
        # Gửi log mới qua WebSocket
        await manager.broadcast(entry)
        # Lưu vào database
        insert_log(ip, "Benign", detection_time, 0.0, 200, timestamp)
        return JSONResponse({
            "ip": ip,
            "result": "Benign",
            "detection_time": detection_time,
            "block_time": 0.0
        })

    # --- 2. Nếu là DDoS: tính thời gian ngăn chặn (phân loại loại attack) ---
    start_block = time.time()
    X_attack = scaler_attack.transform(X)
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
    y_attack = attack_model.predict(X_attack)
    attack_class = np.argmax(y_attack, axis=1)[0]
    attack_type = attack_encoder.inverse_transform([attack_class])[0]
    end_block = time.time()
    block_time = round((end_block - start_block) * 1000, 3)  # ms

    entry = {
        "ip": ip,
        "result": f"DDoS - {attack_type}",
        "detection_time": detection_time,
        "block_time": block_time,
        "status_code": 403,
        "sec": timestamp
    }
    history.append(entry)
    if len(history) > 100:
        history.pop(0)
    await manager.broadcast(entry)
    insert_log(ip, f"DDoS - {attack_type}", detection_time, block_time, 403, timestamp)

    raise HTTPException(
        status_code=403,
        detail={
            "ip": ip,
            "result": f"DDoS - {attack_type}",
            "detection_time": detection_time,
            "block_time": block_time
        }
    )

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await manager.connect(websocket)
    # Khi client mới connect, gửi 20 log gần nhất (nếu có)
    initial = history[-20:]
    for entry in initial:
        await websocket.send_json(entry)
    try:
        while True:
            # Giữ kết nối mở, nhận nhưng không dùng message từ client
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
