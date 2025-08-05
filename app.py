import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import base64
import io
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Register HEIF plugin for HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Warning: pillow-heif not available. HEIC format not supported.")

from torchvision.models import efficientnet_b0
import numpy as np
import cv2

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

def validate_lesion_image(image):
    """Validate if the image is likely a skin lesion image"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Check if image is valid
        if img_array.size == 0:
            return False, "Empty image"
            
        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            rgb_img = img_array
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            rgb_img = img_array[:, :, :3]  # Remove alpha channel
        else:
            return False, "Invalid image format"
        
        # 1. Check for solid colors or very low complexity
        if is_solid_color(rgb_img):
            return False, "Image appears to be a solid color or screenshot"
        
        # 2. Check image quality and content
        if not is_photographic_quality(rgb_img):
            return False, "Image appears to be low quality or non-photographic"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def is_solid_color(img_array):
    """Check if image is mostly solid color or very low complexity"""
    # Calculate color variance
    variance = np.var(img_array)
    
    # If variance is very low, it's likely a solid color
    if variance < 100:
        return True
    
    # Check for text-like patterns (high contrast edges)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.sum(edges > 0) / edges.size
    
    # If too many edges, might be text or diagram
    if edge_ratio > 0.15:
        return True
        
    return False

def has_skin_like_colors(img_array):
    """Check if image contains skin-like colors"""
    # Convert to HSV for better skin detection
    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Define skin color ranges in HSV
    # Range 1: Light skin tones
    lower_skin1 = np.array([0, 20, 70])
    upper_skin1 = np.array([20, 255, 255])
    
    # Range 2: Medium skin tones  
    lower_skin2 = np.array([0, 40, 50])
    upper_skin2 = np.array([25, 180, 230])
    
    # Create masks for skin color ranges
    mask1 = cv2.inRange(hsv_img, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv_img, lower_skin2, upper_skin2)
    
    # Combine masks
    skin_mask = cv2.bitwise_or(mask1, mask2)
    
    # Calculate percentage of skin-like pixels
    skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
    
    # Require at least 15% skin-like colors
    return skin_ratio > 0.15

def is_photographic_quality(img_array):
    """Check if image appears to be a photograph"""
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Check image size (too small might not be a proper photo)
    height, width = gray.shape
    if height < 50 or width < 50:
        return False
    
    # Check for blur (too blurry images)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 10:  # Very blurry
        return False
    
    # Check for excessive noise or compression artifacts
    # Calculate local standard deviation
    kernel = np.ones((5,5), np.float32) / 25
    smooth = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    noise = np.abs(gray.astype(np.float32) - smooth)
    noise_level = np.mean(noise)
    
    if noise_level > 50:  # Too noisy
        return False
        
    return True

def predict_image(image, model):
    """Make prediction on the input image"""
    img = preprocess(image).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, 1)
    short_code = CLASS_NAMES[pred.item()]
    return CLASS_NAME_MAP[short_code], confidence.item(), probabilities[0]

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
            
            # Validate if image is lesion-related
            is_valid, validation_message = validate_lesion_image(image)
            if not is_valid:
                return jsonify({
                    'error': 'Invalid image for lesion analysis',
                    'message': validation_message,
                    'type': 'invalid_image'
                }), 400
            
            # Make prediction
            prediction, confidence, probabilities = predict_image(image, model)
            
            # Additional confidence-based validation
            max_confidence = float(confidence)
            if max_confidence < 0.3:  # Very low confidence across all classes
                return jsonify({
                    'error': 'Image does not appear to contain a recognizable skin lesion',
                    'message': 'Please upload a clear image of a skin lesion',
                    'type': 'invalid_image'
                }), 400
            
            return jsonify({
                'prediction': prediction,
                'confidence': max_confidence,
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