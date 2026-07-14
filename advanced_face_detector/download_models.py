"""
Download face detection model files from OpenCV's model zoo.
Run this once before starting the web app.
"""

import urllib.request
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        28104
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        10666211
    ),
}


def download_file(url, dest, expected_size=None):
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if expected_size and abs(size - expected_size) < 1000:
            print(f"[SKIP] {os.path.basename(dest)} already exists ({size:,} bytes)")
            return True
    
    print(f"[DOWNLOADING] {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        print(f"[OK] {os.path.basename(dest)} ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {os.path.basename(dest)}: {e}")
        print(f"  Please download manually from: {url}")
        return False


def main():
    print("Downloading face detection models...")
    os.makedirs(str(BASE_DIR), exist_ok=True)
    
    all_ok = True
    for filename, (url, size) in FILES.items():
        dest = str(BASE_DIR / filename)
        if not download_file(url, dest, size):
            all_ok = False
    
    if all_ok:
        print("\n[OK] All models downloaded successfully!")
        print("You can now run the web app.")
    else:
        print("\n[WARN] Some models failed to download.")
        print("Check your internet connection and try again.")


if __name__ == "__main__":
    main()
