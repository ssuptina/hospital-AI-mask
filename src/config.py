from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Model training configuration
MODEL_CONFIG = {
    "DATASET_DIR": str(PROJECT_ROOT / "dataset"),
    "IMAGE_SIZE": 224,
    "BATCH_SIZE": 32,
    "EPOCHS": 30,
    "LEARNING_RATE": 1e-4,
    "VALIDATION_SPLIT": 0.2,
}

# Detection configuration
DETECTION_CONFIG = {
    "FACE_PROTOTXT": str(PROJECT_ROOT / "advanced_face_detector" / "deploy.prototxt"),
    "FACE_MODEL": str(PROJECT_ROOT / "advanced_face_detector" / "res10_300x300_ssd_iter_140000.caffemodel"),
    "MASK_MODEL": str(PROJECT_ROOT / "src" / "best_model.h5"),
    "FACE_CONFIDENCE": 0.9,
    "MASK_CONFIDENCE": 0.9,
}

# Web app configuration
WEB_CONFIG = {
    "UPLOAD_FOLDER": "static/uploads",
    "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,  # 16MB
    "ALLOWED_EXTENSIONS": {"png", "jpg", "jpeg", "gif"},
    "HOST": "0.0.0.0",
    "PORT": 5000,
    "DEBUG": True,
}

# Class labels and colors (3-class setup)
CLASS_LABELS = ["Mask Correct", "Mask Incorrect", "No Mask"]
CLASS_COLORS = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]  # BGR format: Green, Orange, Red

# Logging configuration
LOG_CONFIG = {
    "LOG_DIR": "../logs",
    "LOG_FILE": "non_compliance_log.csv",
    "LOG_LEVEL": "INFO",
}
