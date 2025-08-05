# Skin Lesion AI Analyzer - Production Deployment

A web application for analyzing skin lesions using deep learning with EfficientNet.

## Features

- Upload and analyze skin lesion images
- Real-time AI-powered classification
- Support for 7 different lesion types:
  - Melanoma (MEL)
  - Melanocytic nevus (NV)
  - Basal cell carcinoma (BCC)
  - Actinic keratosis (AKIEC)
  - Benign keratosis (BKL)
  - Dermatofibroma (DF)
  - Vascular lesion (VASC)

## Quick Start on Replit

1. Upload this entire folder to Replit
2. Replit will automatically install dependencies from `requirements.txt`
3. Click "Run" to start the application
4. The app will be available at your Replit URL

## Manual Deployment

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and go to `http://localhost:5000`

## Production Deployment

For production environments, use a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Environment Variables

- `PORT`: Port to run the application (default: 5000)

## File Structure

```
.
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── EfficientNet_EfficientNet-FocalLoss-AdamW-LRScheduler-M1Mac_big_data_20_20250805_1428_best.pth  # Trained AI model
├── templates/            # HTML templates
│   └── index.html
├── static/               # Static assets (CSS, JS)
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── .replit              # Replit configuration
├── replit.nix           # Nix environment for Replit
└── README.md            # This file
```

## API Endpoints

- `GET /`: Main web interface
- `POST /analyze`: Upload and analyze an image
- `GET /health`: Health check endpoint

## Model Information

The application uses an EfficientNet-B0 model trained on skin lesion data. The model achieves high accuracy in classifying various types of skin lesions.

## Security Notes

- This application is for educational/research purposes
- Medical decisions should not be based solely on AI predictions
- Always consult healthcare professionals for medical advice

## License

This project is for educational purposes only.