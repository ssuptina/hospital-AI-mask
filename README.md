# Hospital AI Mask Compliance System

An advanced, real-time computer vision system built for monitoring mask compliance in critical healthcare environments (e.g., ICU, ER, operation theaters). Leveraging a lightweight MobileNetV2 architecture and OpenCV, it provides high-accuracy, 3-class face mask detection (Correct, Incorrect, No Mask).

## Features

- **Real-Time Detection**: Processes live webcam feeds to instantly detect faces and analyze mask-wearing compliance.
- **Three-Class Categorization**: Differentiates between:
  - 🟢 **Correct Mask** (mask covering nose and mouth)
  - 🟡 **Incorrect Mask** (mask under nose or chin)
  - 🔴 **No Mask** (no face covering)
- **High Accuracy & Speed**: Built on top of MobileNetV2 for the classifier and an SSD-based Caffe model for rapid face localization.
- **Interactive Web UI**: A professional dashboard built with HTML, CSS, and vanilla JS. It features live statistics, detection history, and manual image uploads.
- **Screenshot & Export**: Allows taking snapshots of the live camera feed with drawn bounding boxes and downloading them instantly.
- **Automated Logging**: Keeps a persistent CSV log of all non-compliance events (or generic detections) for auditing and compliance tracking.

## Technology Stack

- **Backend / API**: Python 3, Flask, Flask-CORS, Flask-Limiter
- **Deep Learning Model**: TensorFlow / Keras (MobileNetV2 fine-tuned)
- **Computer Vision**: OpenCV (dnn module for face detection)
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (with Lucide icons)

## Project Structure

```text
mask_detection/
├── advanced_face_detector/     # Caffe face detection models (.prototxt and .caffemodel)
├── dataset/                    # Training images separated by class
├── logs/                       # Auto-generated CSV logs of detections
├── src/                        # Core ML code and models
│   ├── best_model.h5           # The trained Keras 3-class classification model
│   ├── train_mask_classifier.py# Script used to train the model
│   └── config.py               # Centralized configuration variables
├── web_app/                    # Web Application Server (Flask API + HTML/CSS/JS)
│   ├── app_enhanced.py         # Main Flask server
│   ├── static/                 # CSS styles and uploaded images
│   └── templates/              # HTML frontend files
├── setup_and_train_enhanced.bat# Utility script to set up environment and run training
├── requirements_enhanced.txt   # Python dependencies
└── README.md                   # This file
```

## Setup & Installation

1. **Clone the repository** and navigate to the project root.
2. **Set up the Virtual Environment**:
   ```powershell
   python -m venv mask_venv
   .\mask_venv\Scripts\Activate.ps1
   ```
3. **Install Dependencies**:
   ```powershell
   pip install -r requirements_enhanced.txt
   ```
4. **Ensure Models are Present**:
   - Make sure `advanced_face_detector/res10_300x300_ssd_iter_140000.caffemodel` and `deploy.prototxt` exist.
   - Make sure `src/best_model.h5` exists. (If you want to train it from scratch, you can run `setup_and_train_enhanced.bat`).

## Running the Application

1. Activate your virtual environment: `.\mask_venv\Scripts\Activate.ps1`
2. Change directory to the web app: `cd web_app`
3. Run the Flask server:
   ```powershell
   python app_enhanced.py
   ```
4. Open your web browser and navigate to: `http://localhost:5000`

## Using the Dashboard

- **Start Monitoring**: Click the "Start Monitoring" button to turn on your webcam. Ensure your browser has permission to access the camera.
- **Enable Realtime**: Click this to automatically scan the camera feed continuously and update the HUD stats.
- **Screenshot**: Press the "Screenshot" button (or the Spacebar on your keyboard) to capture a manual frame, analyze it, and download the annotated image to your PC.
- **Export CSV**: Download a summary report of all detected faces and their compliance status during the session.
- **Static Analysis**: Drag and drop an image into the upload box on the right side to analyze pre-existing photos.

## Authors & Acknowledgments
Built for rigorous infection prevention protocols. This project demonstrates end-to-end integration of deep learning models into a user-friendly, responsive web dashboard.
