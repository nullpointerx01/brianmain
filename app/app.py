"""
Flask Web Application for Brain Tumor Detection
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import BrainTumorPredictor, get_tumor_info
from src.config import MODEL_PATH, CLASS_NAMES

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = 'brain_tumor_detection_secret_key_drdo_2024'

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize predictor
predictor = None


def get_predictor():
    """Get or initialize the predictor"""
    global predictor
    if predictor is None:
        predictor = BrainTumorPredictor(MODEL_PATH)
        try:
            predictor.load_model()
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            return None
    return predictor


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, tiff'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get predictor
        pred = get_predictor()
        if pred is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.',
                'instruction': 'Run: python src/train.py'
            }), 500
        
        # Make prediction
        results = pred.predict(filepath)
        
        # Get tumor information
        tumor_info = get_tumor_info(results['predicted_class'])
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'class': results['predicted_class'],
                'confidence': round(results['confidence'] * 100, 2),
                'is_tumor_detected': results['is_tumor_detected'],
                'probabilities': {
                    k: round(v * 100, 2) 
                    for k, v in results['all_probabilities'].items()
                }
            },
            'tumor_info': tumor_info,
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (same as /predict but clearly marked as API)"""
    return predict()


@app.route('/health')
def health_check():
    """Health check endpoint"""
    pred = get_predictor()
    model_loaded = pred is not None and pred.model is not None
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'classes': CLASS_NAMES,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/info')
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'Brain Tumor Detection API',
        'version': '1.0.0',
        'organization': 'DRDO',
        'endpoints': {
            '/': 'Web interface',
            '/predict': 'POST - Upload MRI image for prediction',
            '/api/predict': 'POST - API endpoint for predictions',
            '/health': 'GET - Health check',
            '/api/info': 'GET - API information'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '16MB',
        'classes': CLASS_NAMES
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(500)
def server_error(e):
    """Handle server errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("="*60)
    print("BRAIN TUMOR DETECTION - WEB APPLICATION")
    print("="*60)
    print(f"Starting server...")
    print(f"URL: http://localhost:5000")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
