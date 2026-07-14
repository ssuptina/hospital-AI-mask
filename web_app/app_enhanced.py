"""
Enhanced Flask API for Mask Detection with Image Storage & Logging
Features:
- Image storage in static/uploads/
- Compliance violation logging to CSV
- 3-class classification (Correct, Incorrect, No Mask)
- Request validation & error handling
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import csv
from werkzeug.utils import secure_filename
from datetime import datetime
import logging
import sys
import socket
import traceback
from pathlib import Path

# Ensure source package is resolvable regardless of current working directory
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from config import WEB_CONFIG, CLASS_LABELS, DETECTION_CONFIG

# Initialize Flask app
app = Flask(__name__)
app.config.update(WEB_CONFIG)
CORS(app)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Global variables
face_net = None
mask_model = None
model_loaded = False


class ModelLoader:
    """Load detection models"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.face_net = None
            cls._instance.mask_model = None
        return cls._instance
    
    def load_models(self):
        """Load face detector and mask classifier"""
        try:
            logger.info("[LOADING] Face detector...")
            self.face_net = cv2.dnn.readNetFromCaffe(
                DETECTION_CONFIG["FACE_PROTOTXT"],
                DETECTION_CONFIG["FACE_MODEL"]
            )
            logger.info("[OK] Face detector loaded")
        except Exception as e:
            logger.error(f"[ERROR] Error loading face detector: {e}")
            return False
        
        try:
            logger.info("[LOADING] Mask classifier...")
            model_path = DETECTION_CONFIG.get("MASK_MODEL")
            if not model_path or not os.path.exists(model_path):
                logger.warning(f"[WARN] Model not found at {model_path}")
                return False
            
            self.mask_model = load_model(model_path)
            logger.info(f"[OK] Mask classifier loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Error loading mask classifier: {e}")
            return False


class MaskDetectionAPI:
    """API for mask detection with logging"""
    
    def __init__(self, face_net, mask_model):
        self.face_net = face_net
        self.mask_model = mask_model
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > DETECTION_CONFIG["FACE_CONFIDENCE"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Add padding
                startX = max(0, startX - 10)
                startY = max(0, startY - 10)
                endX = min(w, endX + 10)
                endY = min(h, endY + 10)
                
                faces.append({
                    'box': (int(startX), int(startY), int(endX), int(endY)),
                    'confidence': float(confidence)
                })
        
        return faces
    
    def predict_mask(self, face_roi):
        """Predict mask status for face ROI"""
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = face_roi.astype("float") / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        # Removed preprocess_input because the model was trained with rescale=1./255
        
        prediction = self.mask_model.predict(face_roi, verbose=0)
        confidence = float(np.max(prediction))
        
        if confidence < DETECTION_CONFIG.get("MASK_CONFIDENCE", 0.9):
            class_idx = -1
            class_label = "Low Confidence"
        else:
            class_idx = int(np.argmax(prediction[0]))
            class_label = CLASS_LABELS[class_idx]
        
        return {
            'class': class_label,
            'class_id': class_idx,
            'confidence': confidence,
            'probabilities': prediction[0].tolist()
        }
    
    def log_detection(self, filename, detections_data):
        """Log detection results to CSV"""
        log_dir = "../logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "non_compliance_log.csv")
        
        file_exists = os.path.exists(log_file)
        
        try:
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Timestamp', 'Filename', 'Total_Faces', 'Correct_Mask', 'Incorrect_Mask', 'No_Mask', 'Compliance_Rate'])
                
                summary = detections_data.get('summary', {})
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    filename,
                    detections_data.get('faces_detected', 0),
                    summary.get('mask_correct', 0),
                    summary.get('mask_incorrect', 0),
                    summary.get('no_mask', 0),
                    f"{summary.get('compliance_rate', 0):.2f}%"
                ])
            logger.info(f"Detection logged: {filename}")
        except Exception as e:
            logger.error(f"Error logging detection: {e}")
    
    def process_image(self, image_path):
        """Process image and detect masks"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None, "Failed to load image"
            
            faces = self.detect_faces(image)
            if not faces:
                return {
                    'faces_detected': 0,
                    'detections': [],
                    'summary': {
                        'mask_correct': 0,
                        'mask_incorrect': 0,
                        'no_mask': 0,
                        'compliance_rate': 0.0
                    }
                }, None
            
            detections = []
            mask_count = {'mask_correct': 0, 'mask_incorrect': 0, 'no_mask': 0}
            
            for face in faces:
                (startX, startY, endX, endY) = face['box']
                face_roi = image[startY:endY, startX:endX]
                
                if face_roi.size == 0:
                    continue
                
                mask_pred = self.predict_mask(face_roi)
                class_label = mask_pred['class'].lower().replace(' ', '_')
                mask_count[class_label] = mask_count.get(class_label, 0) + 1
                
                detections.append({
                    'box': {
                        'x1': startX, 'y1': startY,
                        'x2': endX, 'y2': endY
                    },
                    'face_confidence': face['confidence'],
                    'mask_prediction': mask_pred
                })
            
            compliance_rate = (mask_count['mask_correct'] / len(detections) * 100) if detections else 0
            
            return {
                'faces_detected': len(detections),
                'detections': detections,
                'summary': {
                    'mask_correct': mask_count.get('mask_correct', 0),
                    'mask_incorrect': mask_count.get('mask_incorrect', 0),
                    'no_mask': mask_count.get('no_mask', 0),
                    'compliance_rate': round(compliance_rate, 2)
                }
            }, None
        
        except Exception as e:
            logger.error(f"Error processing image: {e}\n{traceback.format_exc()}")
            return None, f"Processing error: {str(e)}"


def allowed_file(filename):
    """Validate file extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in WEB_CONFIG['ALLOWED_EXTENSIONS']


def validate_request():
    """Validate incoming request"""
    if 'file' not in request.files:
        return False, "No file provided"
    
    file = request.files['file']
    if file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, f"Invalid file type. Allowed: {', '.join(WEB_CONFIG['ALLOWED_EXTENSIONS'])}"
    
    return True, file


# Initialize models
loader = ModelLoader()
if loader.load_models():
    model_loaded = True
    detector_api = MaskDetectionAPI(loader.face_net, loader.mask_model)
else:
    logger.error("Failed to load models")


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', model_loaded=model_loaded, class_labels=CLASS_LABELS)


@app.route('/health')
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/v1/detect', methods=['POST', 'OPTIONS'])
@limiter.limit("20 per minute")
def detect_mask_v1():
    """
    Enhanced API endpoint for 3-class mask detection
    POST /api/v1/detect
    File: multipart image file
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    if not model_loaded:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 503
    
    start_time = datetime.now()
    
    valid, result = validate_request()
    if not valid:
        logger.warning(f"Invalid request: {result}")
        return jsonify({'error': result}), 400
    
    try:
        file = result
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(filepath)
        logger.info(f"Image saved: {filename}")
        
        results, error = detector_api.process_image(filepath)
        
        if error:
            return jsonify({'error': error}), 500
        
        detector_api.log_detection(filename, results)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return jsonify({
            'success': True,
            'data': results,
            'metadata': {
                'filename': filename,
                'processing_time_ms': round(processing_time * 1000, 2),
                'timestamp': datetime.now().isoformat(),
                'api_version': 'v1',
                'classes': CLASS_LABELS
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/v1/status', methods=['GET'])
def api_status():
    """Get API status"""
    return jsonify({
        'status': 'operational' if model_loaded else 'model_missing',
        'model': 'MobileNetV2 - 3-Class',
        'classes': CLASS_LABELS,
        'max_file_size_mb': WEB_CONFIG['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'api_version': 'v1'
    }), 200


@app.route('/detect', methods=['POST'])
@limiter.limit("20 per minute")
def detect_image():
    """Web UI endpoint for image upload detection"""
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 503
    
    valid, result = validate_request()
    if not valid:
        return jsonify({'success': False, 'error': result}), 400
    
    try:
        file = result
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(filepath)
        
        results, error = detector_api.process_image(filepath)
        
        if error:
            return jsonify({'success': False, 'error': error}), 500
        
        detector_api.log_detection(filename, results)
        
        detections = []
        for det in results.get('detections', []):
            mask_pred = det['mask_prediction']
            detections.append({
                'class_id': mask_pred['class_id'],
                'label': mask_pred['class'],
                'confidence': round(mask_pred['confidence'] * 100, 2)
            })
        
        return jsonify({
            'success': True,
            'results': detections,
            'image_path': f'/static/uploads/{filename}',
            'summary': results.get('summary', {})
        }), 200
    
    except Exception as e:
        logger.error(f"Detection error: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': 'Processing failed'}), 500


@app.route('/webcam_detect', methods=['POST'])
@limiter.limit("600 per minute")
def webcam_detect():
    """Webcam frame detection endpoint"""
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        import base64
        import io
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        
        image_file = io.BytesIO(image_bytes)
        image_array = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        if image_array is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'}), 400
        
        faces = detector_api.detect_faces(image_array)
        
        detections = []
        for face in faces:
            startX, startY, endX, endY = face['box']
            face_roi = image_array[startY:endY, startX:endX]
            
            if face_roi.size == 0:
                continue
            
            mask_pred = detector_api.predict_mask(face_roi)
            detections.append({
                'class_id': mask_pred['class_id'],
                'label': mask_pred['class'],
                'confidence': round(mask_pred['confidence'] * 100, 2),
                'box': {'x1': startX, 'y1': startY, 'x2': endX, 'y2': endY}
            })
        
        return jsonify({
            'success': True,
            'detections': detections,
            'faces_detected': len(detections)
        }), 200
    
    except Exception as e:
        logger.error(f"Webcam detection error: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': 'Detection failed'}), 500


@app.route('/logs', methods=['GET'])
def get_logs():
    """Get compliance logs"""
    try:
        log_file = "../logs/non_compliance_log.csv"
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs_content = f.read()
            return jsonify({'success': True, 'logs': logs_content}), 200
        else:
            return jsonify({'success': True, 'logs': 'No logs available'}), 200
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return jsonify({'success': False, 'error': 'Failed to read logs'}), 500


@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    host = WEB_CONFIG['HOST']
    port = WEB_CONFIG['PORT']
    local_url = f"http://127.0.0.1:{port}"
    try:
        network_host = socket.gethostbyname(socket.gethostname())
        network_url = f"http://{network_host}:{port}"
    except Exception:
        network_url = 'Unavailable'

    print("\n" + "="*60)
    print("MASK DETECTION API - 3-CLASS MODEL")
    print("="*60)
    print(f"Model Loaded : {model_loaded}")
    print(f"Uploads      : web_app/static/uploads/")
    print(f"Logs         : ../logs/non_compliance_log.csv")
    print(f"Classes      : {', '.join(CLASS_LABELS)}")
    print(f"Local        : {local_url}")
    print(f"Network      : {network_url}")
    print("="*60 + "\n")
    
    app.run(host=host, port=port, debug=WEB_CONFIG['DEBUG'])
