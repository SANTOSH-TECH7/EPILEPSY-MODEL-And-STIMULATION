from flask import Flask, request, render_template, jsonify
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

app = Flask(__name__)

# Global variable to store the loaded model
model_package = None

# Define the correct paths
BASE_DIR = r"D:\EPILEPSY_PROJECT_DETAILS\EpilepsyTest"
MODEL_PATH = os.path.join(BASE_DIR, "epilepsy_seizure_model_joblib.pkl")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# Update Flask app to use the correct template directory
app = Flask(__name__, template_folder=TEMPLATE_DIR)

def load_model():
    """Load the trained model package"""
    global model_package
    try:
        # First try to load the joblib file
        if os.path.exists(MODEL_PATH):
            model_package = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from: {MODEL_PATH}")
            return True
        else:
            # Fallback to pickle file
            pickle_path = os.path.join(BASE_DIR, "epilepsy_seizure_model.pkl")
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    model_package = pickle.load(f)
                print(f"Model loaded successfully from: {pickle_path}")
                return True
            else:
                print(f"Model files not found at:")
                print(f"- {MODEL_PATH}")
                print(f"- {pickle_path}")
                return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_input(data):
    """Preprocess input data for prediction"""
    try:
        # Convert to numpy array if it's a list
        if isinstance(data, list):
            data = np.array(data).reshape(1, -1)
        elif isinstance(data, dict):
            # If data is a dictionary, extract values in the correct order
            feature_names = model_package['feature_names']
            data = np.array([data.get(feature, 0) for feature in feature_names]).reshape(1, -1)
        
        # Scale the data
        scaled_data = model_package['scaler'].transform(data)
        return scaled_data
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def make_prediction(processed_data):
    """Make prediction using the loaded model"""
    try:
        # Get prediction and probabilities
        prediction = model_package['model'].predict(processed_data)
        prediction_proba = model_package['model'].predict_proba(processed_data)
        
        # Decode prediction to original class name
        predicted_class = model_package['label_encoder'].inverse_transform(prediction)[0]
        
        # Get class probabilities
        class_probabilities = {}
        for i, class_name in enumerate(model_package['class_names']):
            class_probabilities[str(class_name)] = float(prediction_proba[0][i])
        
        return {
            'predicted_class': int(predicted_class),
            'confidence': float(max(prediction_proba[0])),
            'probabilities': class_probabilities,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

@app.route('/')
def home():
    """Render the main page"""
    try:
        return render_template('html_template.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        return f"Template error: {e}. Please ensure html_template.html exists in {TEMPLATE_DIR}"

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Check if model is loaded
        if model_package is None:
            return jsonify({'error': 'Model not loaded. Please restart the application.'}), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features from the request
        if 'features' in data:
            features = data['features']
        else:
            # Assume the entire data object contains features
            features = data
        
        # Validate input length
        expected_features = len(model_package['feature_names'])
        if isinstance(features, list) and len(features) != expected_features:
            return jsonify({
                'error': f'Expected {expected_features} features, got {len(features)}'
            }), 400
        
        # Preprocess input
        processed_data = preprocess_input(features)
        if processed_data is None:
            return jsonify({'error': 'Error preprocessing input data'}), 400
        
        # Make prediction
        result = make_prediction(processed_data)
        if result is None:
            return jsonify({'error': 'Error making prediction'}), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Get information about the loaded model"""
    if model_package is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'feature_count': len(model_package['feature_names']),
        'feature_names': model_package['feature_names'],
        'classes': model_package['class_names'].tolist(),
        'model_metrics': model_package['model_metrics'],
        'model_path': MODEL_PATH,
        'base_directory': BASE_DIR
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_package is not None,
        'model_path': MODEL_PATH,
        'template_directory': TEMPLATE_DIR,
        'base_directory': BASE_DIR,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/debug')
def debug_info():
    """Debug endpoint to check file paths and existence"""
    debug_info = {
        'base_directory': BASE_DIR,
        'model_path': MODEL_PATH,
        'template_directory': TEMPLATE_DIR,
        'files_exist': {
            'model_joblib': os.path.exists(MODEL_PATH),
            'model_pickle': os.path.exists(os.path.join(BASE_DIR, "epilepsy_seizure_model.pkl")),
            'template_html': os.path.exists(os.path.join(TEMPLATE_DIR, "html_template.html")),
            'base_dir': os.path.exists(BASE_DIR),
            'template_dir': os.path.exists(TEMPLATE_DIR)
        },
        'model_loaded': model_package is not None
    }
    
    # List files in directories
    try:
        if os.path.exists(BASE_DIR):
            debug_info['files_in_base'] = os.listdir(BASE_DIR)
        if os.path.exists(TEMPLATE_DIR):
            debug_info['files_in_templates'] = os.listdir(TEMPLATE_DIR)
    except Exception as e:
        debug_info['error_listing_files'] = str(e)
    
    return jsonify(debug_info)

if __name__ == '__main__':
    print("=" * 60)
    print("EPILEPSY SEIZURE DETECTION - Flask Application")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Template Directory: {TEMPLATE_DIR}")
    print("-" * 60)
    
    # Check if directories exist
    if not os.path.exists(BASE_DIR):
        print(f"❌ Base directory does not exist: {BASE_DIR}")
        exit(1)
    
    if not os.path.exists(TEMPLATE_DIR):
        print(f"❌ Template directory does not exist: {TEMPLATE_DIR}")
        print("Creating template directory...")
        try:
            os.makedirs(TEMPLATE_DIR, exist_ok=True)
            print(f"✅ Template directory created: {TEMPLATE_DIR}")
        except Exception as e:
            print(f"❌ Failed to create template directory: {e}")
            exit(1)
    
    # Check if template file exists
    template_file = os.path.join(TEMPLATE_DIR, "html_template.html")
    if not os.path.exists(template_file):
        print(f"❌ Template file not found: {template_file}")
        print("Please ensure html_template.html is in the templates directory")
        exit(1)
    
    # Load the model when starting the application
    print("Loading model...")
    if load_model():
        print("✅ Model loaded successfully!")
        print(f"✅ Template directory: {TEMPLATE_DIR}")
        print(f"✅ Template file: html_template.html")
        print("-" * 60)
        print("🚀 Starting Flask application...")
        print("🌐 Access the application at: http://localhost:5000")
        print("🔍 Debug info available at: http://localhost:5000/debug")
        print("❤️  Health check at: http://localhost:5000/health")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model.")
        print(f"Please ensure the model file exists at: {MODEL_PATH}")
        print("or run model_training.py first to create the model.")
        exit(1)