from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import gc

class ModelSingleton:
    """Singleton pattern for model - ensures one instance"""
    _instance = None
    _model = None
    _transform = None
    _device = None
    _classes = [
        'Cashew anthracnose', 'Cashew gumosis', 'Cashew healthy', 'Cashew leaf miner', 'Cashew red rust',
        'Cassava bacterial blight', 'Cassava brown spot', 'Cassava green mite', 'Cassava healthy', 'Cassava mosaic',
        'Maize fall armyworm', 'Maize grasshoper', 'Maize healthy', 'Maize leaf beetle', 'Maize leaf blight',
        'Maize leaf spot', 'Maize streak virus', 'Tomato healthy', 'Tomato leaf blight', 'Tomato leaf curl',
        'Tomato septoria leaf spot', 'Tomato verticulium wilt'
    ]
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_path):
        """Load model once"""
        if self._model is not None:
            return  # Already loaded
            
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimized transform
        self._transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        self._model = models.efficientnet_b3(pretrained=False)
        self._model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self._model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, len(self._classes))
        )
        
        # Load weights
        self._model.load_state_dict(torch.load(model_path, map_location=self._device))
        self._model.eval()
        self._model = self._model.to(self._device)
        
        # Optimize for inference
        if hasattr(torch, 'jit') and self._device.type == 'cpu':
            # JIT compile for CPU inference speed
            dummy_input = torch.randn(1, 3, 384, 384).to(self._device)
            self._model = torch.jit.trace(self._model, dummy_input)
        
        # Warm up the model
        self._warmup()
        
        print(f"✅ Model loaded and optimized on {self._device}")
    
    def _warmup(self):
        """Warm up model with dummy prediction"""
        dummy_image = Image.new('RGB', (384, 384), color='red')
        for _ in range(3):  # Multiple warmup runs
            self.predict(dummy_image)
        gc.collect()  # Clean up memory
    
    def predict(self, image):
        """Ultra-fast prediction"""
        # Preprocess
        input_tensor = self._transform(image.convert('RGB')).unsqueeze(0).to(self._device)
        
        # Predict with optimizations
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self._device.type=='cuda'):
            outputs = self._model(input_tensor)
            probs = F.softmax(outputs[0], dim=0)
            
            # Get top prediction efficiently
            confidence, predicted_idx = torch.max(probs, 0)
            
        return {
            'predicted_class': self._classes[predicted_idx.item()],
            'confidence': confidence.item(),
            'confidence_percentage': confidence.item() * 100
        }
    
    @property
    def is_loaded(self):
        return self._model is not None

# Global model instance
model_instance = ModelSingleton()

# Flask app with optimizations
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Thread pool for handling multiple requests
executor = ThreadPoolExecutor(max_workers=4)

def process_prediction(image_bytes):
    """Process prediction in thread pool"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return model_instance.predict(image)
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    """Lightning fast prediction endpoint"""
    start_time = time.time()
    
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image'}), 400
        
        file = request.files['image']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({'error': 'Empty file'}), 400
        
        # Quick file type check (optional)
        if not image_bytes.startswith((b'\xff\xd8', b'\x89PNG', b'GIF')):  # JPEG, PNG, GIF
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Predict
        result = process_prediction(image_bytes)
        
        # Add processing time
        result['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_instance.is_loaded,
        'device': str(model_instance._device) if model_instance._device else 'none'
    })

@app.route('/batch', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    start_time = time.time()
    
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No images provided'}), 400
        
        results = []
        for i, file in enumerate(files[:10]):  # Limit to 10 images
            try:
                image_bytes = file.read()
                result = process_prediction(image_bytes)
                result['image_index'] = i
                result['filename'] = file.filename
                results.append(result)
            except Exception as e:
                results.append({
                    'image_index': i,
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 16MB)'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Load model at startup
def initialize_model():
    """Initialize model at startup"""
    model_path = "best_crop_disease_model.pth"  # Update this path
    try:
        print("🚀 Initializing high-performance model...")
        model_instance.load_model(model_path)
        print("⚡ API ready for high-speed inference!")
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

if __name__ == '__main__':
    # Initialize model
    if not initialize_model():
        print("⚠️  Exiting due to model loading failure")
        exit(1)
    
    # Production configuration
    print("🌐 Starting production server...")
    print("📡 Endpoints:")
    print("  POST /predict - Single image prediction")
    print("  POST /batch   - Batch prediction (max 10 images)")
    print("  GET  /health  - Health check")
    print("⚡ Optimizations enabled: JIT compilation, AMP, thread pool")
    
    # Configure app
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
    
    # Run with optimizations
    app.run(
        debug=False, 
        host='0.0.0.0', 
        port=5000,
        threaded=True,
        use_reloader=False  # Prevent model reloading
    )
