<<<<<<< HEAD
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
=======
# Face Mask Detection (Multi-Class + Real-Time)

A deep learning-based face mask detection system supporting:
- **With Mask**
- **Without Mask**
- **Incorrect Mask Wearing**

Trained with **MobileNetV2 + Squeeze-and-Excitation**, real-time webcam support, and Grad-CAM for explainability.

---

## 📂 Project Structure

```
mask-detection/
│
├── dataset/                     # Dataset folder (with_mask, without_mask, mask_incorrect)
├── saved_model/                 # Saved models & TFLite exports
├── src/
│   ├── models.py                 # Model architecture
│   ├── train.py                  # Training script
│   ├── predict.py                # Image prediction
│   ├── realtime.py               # Real-time webcam detection
│   ├── gradcam.py                # Grad-CAM visualization
│   ├── utils.py                  # Helper functions
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

---

## 🧠 Research Basis

| **Feature in Code** | **Paper(s) / Dataset(s)** | **Year** | **Why Included (Benefit)** | **Drawback Avoided** |
|---------------------|---------------------------|----------|----------------------------|----------------------|
| **Transfer Learning with MobileNetV2** | *Kaggle Mask Detection Baseline*, *MaskedFace-Net original paper*, ViDMASK real-time study | 2020–2024 | Lightweight, fast, accurate with small datasets | Heavy architectures (VGG16, ResNet101) are too slow for real-time |
| **Data Augmentation (rotation, zoom, flip, brightness)** | ViDMASK Dataset Study, Mendeley Mask Dataset Paper | 2023–2024 | Improves generalization, handles lighting & angle variations | Overfitting on small datasets |
| **Multi-class Support (With Mask / Without Mask / Incorrect Mask)** | *MaskedFace-Net Extended* (3-class), *BAFMD Bias-Aware Face Mask Detection* | 2023 | Detects improper mask usage for better public safety | 2-class models miss partial violations |
| **Grad-CAM Visualization (`gradcam.py`)** | BAFMD Explainability Section, Fair AI in Mask Detection studies | 2023–2024 | Shows *where* the model looks, improves transparency | Black-box predictions with no interpretability |
| **Real-time Detection (OpenCV + Webcam)** | ViDMASK Real-time Deployment Study | 2024 | Enables immediate deployment on laptop webcams or CCTV | Offline-only or slow detection systems |
| **Web-based Interface (Flask + Upload/Webcam)** | Real-time Mask Detection Deployment Papers, Edge AI Deployment Reports | 2023–2024 | Interactive, easy-to-use interface for non-technical users | CLI-only tools are not user-friendly |

---

## 🚀 How to Run

### 1️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 2️⃣ Prepare Dataset
```
dataset/
├── with_mask/
├── without_mask/
└── mask_incorrect/
```
You can use the **MaskedFace-Net** dataset:
🔗 [Download Link](https://github.com/cabani/MaskedFace-Net)

### 3️⃣ Train the Model
```bash
cd src
python train.py
```

### 4️⃣ Run Real-Time Detection
```bash
python realtime.py
```

### 5️⃣ Run Grad-CAM for Explainability
```bash
python gradcam.py --image_path ../dataset/with_mask/sample.jpg
```

---

## 📜 License
MIT License.
>>>>>>> 777bf114792935706a7be8159fbdd100724ee755
