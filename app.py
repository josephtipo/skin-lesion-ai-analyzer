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
        
        # 2. Check if it's a medical-style close-up photo
        if not is_medical_photo_composition(rgb_img):
            return False, "Image doesn't appear to be a medical close-up photo"
        
        # 3. Check for skin-like texture patterns
        if not has_skin_texture(rgb_img):
            return False, "Image doesn't contain skin-like texture patterns"
        
        # 4. Check image quality and content
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

def is_medical_photo_composition(img_array):
    """Check if image has composition typical of medical photos"""
    height, width = img_array.shape[:2]
    
    # Check if image is reasonably sized for medical photo
    if height < 100 or width < 100:
        return False
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Check for centered subject (medical photos usually have centered lesions)
    # Create center region (middle 60% of image)
    center_h_start = int(height * 0.2)
    center_h_end = int(height * 0.8)
    center_w_start = int(width * 0.2)
    center_w_end = int(width * 0.8)
    
    center_region = gray[center_h_start:center_h_end, center_w_start:center_w_end]
    edge_regions = np.concatenate([
        gray[:center_h_start, :].flatten(),
        gray[center_h_end:, :].flatten(),
        gray[:, :center_w_start].flatten(),
        gray[:, center_w_end:].flatten()
    ])
    
    # Medical photos usually have more detail/variance in center than edges
    center_variance = np.var(center_region)
    edge_variance = np.var(edge_regions) if len(edge_regions) > 0 else 0
    
    # Check if center has more detail (typical of medical close-ups)
    if center_variance < edge_variance * 0.5:
        return False
    
    # Check for excessive geometric patterns (suggests non-medical content)
    # Use Hough line detection to find straight lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(min(height, width) * 0.3))
    
    # Too many straight lines suggests architectural/geometric content
    if lines is not None and len(lines) > 10:
        return False
    
    return True

def has_skin_texture(img_array):
    """Check if image contains skin-like texture patterns using texture analysis"""
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Calculate Local Binary Pattern-like texture measure
    # Use gradient-based texture analysis instead of full LBP to avoid new dependencies
    
    # Calculate gradients in different directions
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Skin has relatively smooth gradients with some texture
    # Too smooth = solid color, too rough = artificial texture
    texture_variance = np.var(grad_mag)
    
    # Check for organic texture patterns
    # Skin should have moderate texture variance (not too smooth, not too chaotic)
    if texture_variance < 50:  # Too smooth (solid color, very blurred)
        return False
    if texture_variance > 5000:  # Too chaotic (text, artificial patterns)
        return False
    
    # Check for repetitive patterns (suggests artificial content)
    # Use frequency domain analysis
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    
    # Check for dominant frequency peaks (suggests repetitive patterns)
    # Flatten and find peaks
    spectrum_flat = magnitude_spectrum.flatten()
    spectrum_sorted = np.sort(spectrum_flat)[::-1]
    
    # If there are very dominant frequencies, it might be artificial
    if len(spectrum_sorted) > 10:
        # Check if top frequencies are much higher than average
        top_freq_ratio = spectrum_sorted[0] / np.mean(spectrum_sorted[10:])
        if top_freq_ratio > 3.0:  # Too dominant = artificial pattern
            return False
    
    # Check color distribution - skin should have some color variation
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Check if image has reasonable color diversity (not monochrome)
    color_variance = np.var(hsv[:, :, 1])  # Saturation variance
    if color_variance < 10:  # Too little color variation
        return False
    
    return True

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
            
            # Check if image is HEIC format and convert to JPG
            if hasattr(image, 'format') and image.format in ['HEIF', 'HEIC']:
                # Convert HEIC to JPG format
                jpg_buffer = io.BytesIO()
                # Convert to RGB first to ensure compatibility
                rgb_image = image.convert('RGB')
                rgb_image.save(jpg_buffer, format='JPEG', quality=95)
                jpg_buffer.seek(0)
                # Reload as JPG
                image = Image.open(jpg_buffer)
                print(f"Converted HEIC file upload to JPG format")
        
        # Check if it's a JSON request with base64 image
        elif request.is_json and 'image' in request.json:
            try:
                import base64
                base64_data = request.json['image']
                image_data = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_data))
                
                # Check if image is HEIC format and convert to JPG
                if hasattr(image, 'format') and image.format in ['HEIF', 'HEIC']:
                    # Convert HEIC to JPG format
                    jpg_buffer = io.BytesIO()
                    # Convert to RGB first to ensure compatibility
                    rgb_image = image.convert('RGB')
                    rgb_image.save(jpg_buffer, format='JPEG', quality=95)
                    jpg_buffer.seek(0)
                    # Reload as JPG
                    image = Image.open(jpg_buffer)
                    print(f"Converted HEIC base64 image to JPG format")
                    
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


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Handle feedback form submissions"""
    try:
        feedback_data = request.get_json()
        
        # Validate required fields
        if not feedback_data or not feedback_data.get('message') or not feedback_data.get('category'):
            return jsonify({'error': 'Message and category are required'}), 400
        
        # Extract feedback information
        name = feedback_data.get('name', 'Anonymous')
        email = feedback_data.get('email', 'Not provided')
        category = feedback_data.get('category')
        message = feedback_data.get('message')
        timestamp = feedback_data.get('timestamp', '')
        user_agent = feedback_data.get('user_agent', '')
        page_url = feedback_data.get('page_url', '')
        
        # Log feedback to console
        print(f"=== FEEDBACK FORM SUBMISSION ===")
        print(f"Name: {name}")
        print(f"Email: {email}")
        print(f"Category: {category}")
        print(f"Message: {message}")
        print(f"Timestamp: {timestamp}")
        print(f"User Agent: {user_agent}")
        print(f"Page URL: {page_url}")
        print("================================")
        
        # Here you could save to database, send email, etc.
        # For now, we'll just log and return success
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback submitted successfully'
        })
        
    except Exception as e:
        print(f"Error processing feedback: {e}")
        return jsonify({'error': 'Failed to process feedback'}), 500

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
            print("Please upload the trained model file to the root directory.")
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")
        print("App will start but predictions may fail.")
    
    print(f"Starting Flask server on port {port}")
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"Failed to start server: {e}")
        print("App is in recovery mode")