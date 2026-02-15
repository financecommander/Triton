#!/usr/bin/env python3
"""
Triton DSL Inference Server

A Flask-based REST API for serving Triton DSL ternary neural network models.
Provides endpoints for health checks, model inference, and metrics.

Features:
- REST API for model inference
- Image classification endpoint
- Batch processing support
- Health checks and monitoring
- Prometheus metrics
- Error handling and logging
- CORS support for web applications

Usage:
    python app.py
    # Or with Gunicorn
    gunicorn -w 4 -b 0.0.0.0:5000 app:app
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import io

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, InternalServerError

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', '/app/saved_models/model.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '4'))

# Global variables
model = None
class_names = None

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        'inference_requests_total',
        'Total inference requests',
        ['endpoint', 'status']
    )
    REQUEST_LATENCY = Histogram(
        'inference_request_duration_seconds',
        'Inference request latency',
        ['endpoint']
    )
    MODEL_LOAD_TIME = Gauge(
        'model_load_time_seconds',
        'Time taken to load the model'
    )
    ACTIVE_REQUESTS = Gauge(
        'active_inference_requests',
        'Number of active inference requests'
    )


# ============================================================================
# Image Preprocessing
# ============================================================================

# Standard transforms for CIFAR-10 style models
TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Alternative transforms for ImageNet style models
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image(image_bytes: bytes, transform_type: str = 'cifar') -> torch.Tensor:
    """
    Preprocess image bytes for model inference.
    
    Args:
        image_bytes: Raw image bytes
        transform_type: Type of transform ('cifar' or 'imagenet')
        
    Returns:
        Preprocessed tensor ready for inference
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        if transform_type == 'imagenet':
            tensor = IMAGENET_TRANSFORM(image)
        else:
            tensor = TRANSFORM(image)
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise BadRequest(f"Failed to preprocess image: {str(e)}")


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path: str) -> nn.Module:
    """
    Load PyTorch model from file.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded model in eval mode
    """
    start_time = time.time()
    
    try:
        logger.info(f"Loading model from: {model_path}")
        
        model = torch.load(model_path, map_location=DEVICE)
        model.eval()
        model.to(DEVICE)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Parameters: {total_params:,}")
        logger.info(f"  Device: {DEVICE}")
        
        load_time = time.time() - start_time
        logger.info(f"  Load time: {load_time:.2f}s")
        
        if PROMETHEUS_AVAILABLE:
            MODEL_LOAD_TIME.set(load_time)
        
        return model
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_class_names(class_names_path: str = None) -> Optional[List[str]]:
    """
    Load class names from file.
    
    Args:
        class_names_path: Path to class names JSON file
        
    Returns:
        List of class names or None
    """
    if class_names_path and Path(class_names_path).exists():
        try:
            with open(class_names_path, 'r') as f:
                names = json.load(f)
            logger.info(f"✓ Loaded {len(names)} class names")
            return names
        except Exception as e:
            logger.warning(f"Failed to load class names: {e}")
    
    # Default CIFAR-10 class names
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


# ============================================================================
# Inference Functions
# ============================================================================

def run_inference(image_tensor: torch.Tensor, top_k: int = 5) -> Dict[str, Any]:
    """
    Run inference on preprocessed image tensor.
    
    Args:
        image_tensor: Preprocessed image tensor
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with predictions and metadata
    """
    if model is None:
        raise InternalServerError("Model not loaded")
    
    try:
        start_time = time.time()
        
        # Run inference
        image_tensor = image_tensor.to(DEVICE)
        with torch.no_grad():
            output = model(image_tensor)
        
        # Get predictions
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, min(top_k, output.size(1)))
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Format results
        predictions = []
        for i in range(top_probs.size(1)):
            class_idx = top_indices[0, i].item()
            prob = top_probs[0, i].item()
            
            predictions.append({
                'class_id': class_idx,
                'class_name': class_names[class_idx] if class_names and class_idx < len(class_names) else f"class_{class_idx}",
                'probability': float(prob),
                'confidence': float(prob * 100)
            })
        
        return {
            'predictions': predictions,
            'inference_time_ms': inference_time,
            'device': str(DEVICE)
        }
    
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise InternalServerError(f"Inference failed: {str(e)}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(DEVICE)
    }), 200


@app.route('/info', methods=['GET'])
def model_info():
    """Get model information."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return jsonify({
            'model_loaded': True,
            'parameters': {
                'total': total_params,
                'trainable': trainable_params
            },
            'device': str(DEVICE),
            'num_classes': len(class_names) if class_names else 'unknown',
            'class_names': class_names[:10] if class_names else None  # First 10 classes
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Accepts:
        - image file in multipart/form-data
        - Optional: top_k parameter for number of predictions
    
    Returns:
        JSON with predictions and metadata
    """
    if PROMETHEUS_AVAILABLE:
        ACTIVE_REQUESTS.inc()
        REQUEST_COUNT.labels(endpoint='predict', status='processing').inc()
    
    try:
        start_time = time.time()
        
        # Check if image is in request
        if 'image' not in request.files:
            if PROMETHEUS_AVAILABLE:
                REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
                ACTIVE_REQUESTS.dec()
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            if PROMETHEUS_AVAILABLE:
                REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
                ACTIVE_REQUESTS.dec()
            return jsonify({'error': 'Empty filename'}), 400
        
        # Get parameters
        top_k = int(request.form.get('top_k', 5))
        transform_type = request.form.get('transform', 'cifar')
        
        # Read and preprocess image
        image_bytes = file.read()
        image_tensor = preprocess_image(image_bytes, transform_type)
        
        # Run inference
        results = run_inference(image_tensor, top_k)
        
        total_time = (time.time() - start_time) * 1000
        results['total_time_ms'] = total_time
        
        if PROMETHEUS_AVAILABLE:
            REQUEST_COUNT.labels(endpoint='predict', status='success').inc()
            REQUEST_LATENCY.labels(endpoint='predict').observe(time.time() - start_time)
            ACTIVE_REQUESTS.dec()
        
        return jsonify(results), 200
    
    except BadRequest as e:
        if PROMETHEUS_AVAILABLE:
            REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
            ACTIVE_REQUESTS.dec()
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        if PROMETHEUS_AVAILABLE:
            REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
            ACTIVE_REQUESTS.dec()
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint.
    
    Accepts:
        - Multiple image files in multipart/form-data
        - Optional: top_k parameter
    
    Returns:
        JSON with predictions for each image
    """
    if PROMETHEUS_AVAILABLE:
        ACTIVE_REQUESTS.inc()
        REQUEST_COUNT.labels(endpoint='predict_batch', status='processing').inc()
    
    try:
        start_time = time.time()
        
        # Get all files
        files = request.files.getlist('images')
        if not files or len(files) == 0:
            if PROMETHEUS_AVAILABLE:
                REQUEST_COUNT.labels(endpoint='predict_batch', status='error').inc()
                ACTIVE_REQUESTS.dec()
            return jsonify({'error': 'No images provided'}), 400
        
        top_k = int(request.form.get('top_k', 5))
        transform_type = request.form.get('transform', 'cifar')
        
        # Process all images
        batch_results = []
        for idx, file in enumerate(files):
            try:
                image_bytes = file.read()
                image_tensor = preprocess_image(image_bytes, transform_type)
                results = run_inference(image_tensor, top_k)
                results['image_index'] = idx
                results['filename'] = file.filename
                batch_results.append(results)
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                batch_results.append({
                    'image_index': idx,
                    'filename': file.filename,
                    'error': str(e)
                })
        
        total_time = (time.time() - start_time) * 1000
        
        response = {
            'batch_size': len(files),
            'results': batch_results,
            'total_time_ms': total_time
        }
        
        if PROMETHEUS_AVAILABLE:
            REQUEST_COUNT.labels(endpoint='predict_batch', status='success').inc()
            REQUEST_LATENCY.labels(endpoint='predict_batch').observe(time.time() - start_time)
            ACTIVE_REQUESTS.dec()
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        if PROMETHEUS_AVAILABLE:
            REQUEST_COUNT.labels(endpoint='predict_batch', status='error').inc()
            ACTIVE_REQUESTS.dec()
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return jsonify({'error': 'Prometheus not available'}), 503
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# Initialization
# ============================================================================

def initialize_app():
    """Initialize the application."""
    global model, class_names
    
    logger.info("="*70)
    logger.info("Triton DSL Inference Server")
    logger.info("="*70)
    
    # Load model
    if Path(MODEL_PATH).exists():
        model = load_model(MODEL_PATH)
    else:
        logger.warning(f"Model not found at: {MODEL_PATH}")
        logger.warning("Server will start but /predict endpoints will fail")
    
    # Load class names
    class_names_path = os.getenv('CLASS_NAMES_PATH', '/app/saved_models/class_names.json')
    class_names = load_class_names(class_names_path)
    
    logger.info("="*70)
    logger.info("✓ Server initialized successfully")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Workers: {NUM_WORKERS}")
    logger.info("="*70)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    initialize_app()
    
    # Run Flask development server
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
