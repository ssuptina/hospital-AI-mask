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
