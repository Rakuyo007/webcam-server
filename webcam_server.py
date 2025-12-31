#!/usr/bin/env python3
"""
webcam_server.py
只处理最新帧（实时视觉标准做法）
"""

import asyncio
import json
import traceback
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from paddleocr import PaddleOCR
from ultralytics import YOLO

from ocr import perform_ocr_on_image
from yolo import detect_objects

app = FastAPI()

# -----------------------------
# 模型加载（全局只加载一次）
# -----------------------------
yolo_model = YOLO("xiangjiaba_best.pt")

ocr = PaddleOCR(
    text_recognition_model_name="PP-OCRv5_server_rec",
    text_recognition_model_dir="./PP-OCRv5_server_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# -----------------------------
# 全局“最新帧”容器
# -----------------------------
latest_frame = None
latest_frame_time = None
frame_lock = asyncio.Lock()


@app.websocket("/ws/webcam")
async def ws_webcam_endpoint(websocket: WebSocket):
    await websocket.accept()
    client = websocket.client
    print(f"客户端已连接: {client.host}:{client.port}")

    # 启动后台处理任务
    processor_task = asyncio.create_task(process_latest_frame(websocket))

    try:
        while True:
            # 1. 接收一帧
            data: bytes = await websocket.receive_bytes()

            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 2. 覆盖“最新帧”（旧帧自动被丢弃）
            async with frame_lock:
                global latest_frame, latest_frame_time
                latest_frame = frame
                latest_frame_time = now

    except WebSocketDisconnect:
        print(f"客户端断开: {client.host}:{client.port}")
    except Exception as e:
        print(f"异常: {e}")
        traceback.print_exc()
    finally:
        processor_task.cancel()
        print("WebSocket 会话结束")


# ------------------------------------------------
# 后台任务：只处理“最新帧”
# ------------------------------------------------
async def process_latest_frame(websocket: WebSocket):
    global latest_frame, latest_frame_time

    loop = asyncio.get_running_loop()

    while True:
        await asyncio.sleep(0.01)  # 防止空转占满 CPU

        async with frame_lock:
            if latest_frame is None:
                continue

            frame = latest_frame
            frame_time = latest_frame_time

            # 取走后立刻清空，保证只处理最新一帧
            latest_frame = None

        try:
            # 3. YOLO（同步函数 → executor）
            yolo_result = await loop.run_in_executor(
                None, detect_objects, yolo_model, frame
            )

            # 4. OCR（同步函数 → executor）
            ocr_result = await loop.run_in_executor(
                None, perform_ocr_on_image, ocr, frame, yolo_result
            )
            print(f"处理帧时间: {frame_time}, 结果: {ocr_result}")
            # 5. 返回结果（一定是“最新画面”的结果）
            await websocket.send_text(json.dumps({
                "type": "result",
                "frame_time": frame_time,
                "data": ocr_result
            }, ensure_ascii=False))

        except Exception as e:
            print("处理帧时异常:", e)
            traceback.print_exc()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7999,
        log_level="info"
    )