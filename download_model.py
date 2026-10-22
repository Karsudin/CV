"""
download_model.py

Mengunduh model MediaPipe Hand Landmarker (~10 MB) ke folder assets/.
Cukup dijalankan SEKALI:   python download_model.py
"""

import os
import urllib.request

from config.settings import MODEL_PATH

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def main():
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Model sudah ada: {MODEL_PATH}")
        return

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("[INFO] Mengunduh model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[INFO] Selesai: {MODEL_PATH}")


if __name__ == "__main__":
    main()
