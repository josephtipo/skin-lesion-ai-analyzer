import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import base64
import io
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from torchvision.models import efficientnet_b0

# Class names
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# Image preprocessing (same as training, but without augmentation)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Mapping from short code to full disease name
CLASS_NAME_MAP = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AKIEC": "Actinic keratosis",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion"
}

def load_model(model_path):
    """Load the trained model"""
    model = efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(image, model):
    """Make prediction on the input image"""
    img = preprocess(image).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, 1)
    short_code = CLASS_NAMES[pred.item()]
    return CLASS_NAME_MAP[short_code], confidence.item()

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Global model variable
model = None
MODEL_PATH = "model.pth"

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image"""
    global model
    
    try:
        # Load model if not already loaded
        if model is None:
            if not os.path.exists(MODEL_PATH):
                return jsonify({'error': 'Model file not found'}), 500
            model = load_model(MODEL_PATH)
        
        # Handle both form file upload and base64 JSON
        image = None
        
        # Check if it's a form file upload
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename == '':
                return jsonify({'error': 'No image selected'}), 400
            image = Image.open(image_file.stream)
        
        # Check if it's a JSON request with base64 image
        elif request.is_json and 'image' in request.json:
            try:
                import base64
                base64_data = request.json['image']
                image_data = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_data))
            except Exception as e:
                return jsonify({'error': 'Invalid base64 image data'}), 400
        
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Process the image
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Make prediction
            prediction, confidence = predict_image(image, model)
            
            return jsonify({
                'prediction': prediction,
                'confidence': float(confidence),
                'class_name': prediction,
                'status': 'success'
            })
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
    except Exception as e:
        print(f"General error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH)
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Skin Lesion AI Analyzer...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    
    # Get port from environment variable (for production)
    port = int(os.environ.get('PORT', 5000))
    
    # Try to load model at startup for faster first prediction
    try:
        if os.path.exists(MODEL_PATH):
            print("Found model file, loading...")
            model = load_model(MODEL_PATH)
            print("Model pre-loaded successfully!")
        else:
            print("Warning: Model file not found. App will start but predictions will fail.")
            print("Please upload model.pth to the root directory.")
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")
        print("App will start but predictions may fail.")
    
    print(f"Starting Flask server on port {port}")
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"Failed to start server: {e}")
        print("App is in recovery mode")