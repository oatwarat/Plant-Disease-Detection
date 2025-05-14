"""
Plant Disease Detection API Service
This script implements a REST API for the plant disease detection model
"""

import os
import io
import json
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
from werkzeug.utils import secure_filename
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variables
MODEL = None
CLASS_NAMES = []
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (224, 224)

# Disease information for providing context with predictions
DISEASE_INFO = {
    "Healthy": {
        "description": "The plant appears healthy with no visible signs of disease.",
        "treatment": "Continue regular maintenance and monitoring.",
        "prevention": ["Regular watering", "Proper nutrition", "Adequate sunlight"]
    },
    "Bacterial_spot": {
        "description": "Bacterial spot causes small, water-soaked spots on leaves that later turn brown.",
        "treatment": "Apply copper-based bactericides. Remove and destroy infected plant parts.",
        "prevention": ["Use disease-free seeds", "Rotate crops", "Avoid overhead irrigation"]
    },
    "Early_blight": {
        "description": "Early blight causes dark brown spots with concentric rings, often on lower leaves.",
        "treatment": "Apply appropriate fungicides. Remove infected leaves.",
        "prevention": ["Crop rotation", "Proper spacing", "Keep leaves dry"]
    },
    "Late_blight": {
        "description": "Late blight causes dark, water-soaked lesions on leaves that can spread rapidly.",
        "treatment": "Apply fungicides at first signs. Remove infected plants in severe cases.",
        "prevention": ["Plant resistant varieties", "Improve air circulation", "Avoid wet conditions"]
    },
    "Leaf_rust": {
        "description": "Leaf rust appears as orange-brown pustules on leaf surfaces.",
        "treatment": "Apply fungicides specifically designed for rust diseases.",
        "prevention": ["Plant resistant varieties", "Crop rotation", "Proper field sanitation"]
    },
    "Powdery_mildew": {
        "description": "Powdery mildew appears as white powdery spots on leaf surfaces.",
        "treatment": "Apply sulfur-based fungicides or neem oil.",
        "prevention": ["Increase air circulation", "Avoid overhead watering", "Plant resistant varieties"]
    },
    "Leaf_spot": {
        "description": "Leaf spot diseases cause circular spots with defined edges on leaves.",
        "treatment": "Apply appropriate fungicides. Remove and destroy severely infected leaves.",
        "prevention": ["Crop rotation", "Proper spacing", "Avoid wetting leaves"]
    }
}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_and_classes():
    """Load the trained model and class names"""
    global MODEL, CLASS_NAMES
    
    # Load model
    model_path = os.environ.get('MODEL_PATH', 'best_model.h5')
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    MODEL = load_model(model_path)
    
    # Load class names
    class_names_path = os.environ.get('CLASS_NAMES_PATH', 'class_names.txt')
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            CLASS_NAMES = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(CLASS_NAMES)} class names")
    else:
        logger.warning(f"Class names file not found at {class_names_path}")
        # Infer class names from model output shape
        num_classes = MODEL.output_shape[1]
        CLASS_NAMES = [f"Class_{i}" for i in range(num_classes)]
        logger.info(f"Using generated class names for {num_classes} classes")
    
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image):
    """Preprocess the image for model input"""
    # Resize image
    image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    
    # Convert to array and preprocess
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    
    # Create batch dimension
    image_batch = np.expand_dims(image_array, axis=0)
    
    return image_batch

def generate_grad_cam(image_array, predicted_class_idx):
    """Generate Grad-CAM heatmap for visualization"""
    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(MODEL.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
        elif hasattr(layer, 'layers'):
            # Look in nested model if present
            for nested_layer in reversed(layer.layers):
                if isinstance(nested_layer, tf.keras.layers.Conv2D):
                    last_conv_layer = nested_layer.name
                    break
            if last_conv_layer:
                break
    
    if not last_conv_layer:
        logger.warning("Could not find a convolutional layer for Grad-CAM")
        return None
    
    # Create Grad-CAM model
    try:
        grad_model = tf.keras.models.Model(
            [MODEL.inputs], 
            [MODEL.get_layer(last_conv_layer).output, MODEL.output]
        )
    except:
        # Try to find layer in base model if present
        base_model = None
        for layer in MODEL.layers:
            if hasattr(layer, 'layers'):
                try:
                    base_model = layer
                    break
                except:
                    pass
        
        if base_model is None:
            logger.warning("Could not create Grad-CAM model")
            return None
        
        try:
            grad_model = tf.keras.models.Model(
                [MODEL.inputs], 
                [base_model.get_layer(last_conv_layer).output, MODEL.output]
            )
        except:
            logger.warning("Could not create Grad-CAM model from base model")
            return None
    
    # Original image
    original_img = cv2.resize(image_array, IMG_SIZE)
    
    # Preprocess for model
    img_preprocessed = preprocess_input(original_img.copy())
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    # Generate class activation heatmap
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_batch)
        class_channel = predictions[:, predicted_class_idx]
    
    # Extract gradients and convolutional outputs
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the gradient importance
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on original image
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    # Convert the images to base64 for API response
    _, heatmap_encoded = cv2.imencode('.png', heatmap)
    heatmap_base64 = base64.b64encode(heatmap_encoded).decode('utf-8')
    
    _, superimposed_encoded = cv2.imencode('.png', superimposed_img)
    superimposed_base64 = base64.b64encode(superimposed_encoded).decode('utf-8')
    
    return {
        'heatmap': heatmap_base64,
        'superimposed': superimposed_base64
    }

def predict_disease(image):
    """
    Predict plant disease from image
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with prediction results
    """
    # Keep a copy of the original image
    img_array = np.array(image.convert('RGB'))
    
    # Preprocess the image
    preprocessed = preprocess_image(image)
    
    # Make prediction
    predictions = MODEL.predict(preprocessed)[0]
    
    # Get top 3 predictions
    top_indices = predictions.argsort()[-3:][::-1]
    top_predictions = [
        {
            "class": CLASS_NAMES[i],
            "probability": float(predictions[i]),
            "confidence": f"{float(predictions[i] * 100):.2f}%"
        }
        for i in top_indices
    ]
    
    # Generate Grad-CAM for visualization
    grad_cam_result = generate_grad_cam(img_array, top_indices[0])
    
    # Get information about the predicted disease
    top_disease = CLASS_NAMES[top_indices[0]]
    disease_info = DISEASE_INFO.get(top_disease, {
        "description": f"Information for {top_disease} is not available.",
        "treatment": "Consult with an agricultural expert for treatment options.",
        "prevention": ["Regular monitoring", "Good agricultural practices"]
    })
    
    # Return results
    result = {
        "prediction": top_predictions,
        "disease_info": disease_info,
        "grad_cam": grad_cam_result
    }
    
    return result

@app.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    
    return jsonify({"status": "ok", "message": "API is running"}), 200

@app.route('/predict', methods=['POST'])
def predict_api():
    """Prediction API endpoint accepting image file"""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "status": "error", 
            "message": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    try:
        # Load and process the image
        image = Image.open(file.stream)
        
        # Optional: Save the image
        if request.args.get('save', 'false').lower() == 'true':
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            image.save(filepath)
            logger.info(f"Saved uploaded image to {filepath}")
        
        # Make prediction
        result = predict_disease(image)
        
        # Return the results
        return jsonify({"status": "success", "result": result}), 200
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/predict/base64', methods=['POST'])
def predict_base64_api():
    """Prediction API endpoint accepting base64 encoded image"""
    if not request.json or 'image' not in request.json:
        return jsonify({"status": "error", "message": "No image data provided"}), 400
    
    try:
        # Decode the base64 image
        image_data = request.json['image']
        # Remove data URL prefix if present
        if 'data:image' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        result = predict_disease(image)
        
        # Return the results
        return jsonify({"status": "success", "result": result}), 200
    
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info_api():
    """API endpoint to get information about the model"""
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    
    try:
        info = {
            "name": "Plant Disease Detection Model",
            "version": os.environ.get('MODEL_VERSION', '1.0.0'),
            "classes": CLASS_NAMES,
            "input_shape": [dim if dim is not None else -1 for dim in MODEL.input_shape],
            "image_size": IMG_SIZE
        }
        
        return jsonify({"status": "success", "info": info}), 200
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/disease/info/<disease_name>', methods=['GET'])
def disease_info_api(disease_name):
    """API endpoint to get information about a specific disease"""
    # Check if the disease exists in our database
    if disease_name not in DISEASE_INFO and disease_name not in CLASS_NAMES:
        return jsonify({
            "status": "error", 
            "message": f"Disease '{disease_name}' not found"
        }), 404
    
    # Get the disease key that matches either the direct name or a class name
    disease_key = disease_name if disease_name in DISEASE_INFO else None
    
    # If we didn't find it directly, try to clean up the name
    if disease_key is None:
        # Try some name cleanup (replace underscores with spaces, etc.)
        cleaned_name = disease_name.replace('_', ' ').title()
        for key in DISEASE_INFO:
            if key.replace('_', ' ').title() == cleaned_name:
                disease_key = key
                break
    
    # If we still don't have info, return a default response
    if disease_key is None or disease_key not in DISEASE_INFO:
        return jsonify({
            "status": "success",
            "info": {
                "name": disease_name,
                "description": f"Information for {disease_name} is not available.",
                "treatment": "Consult with an agricultural expert for treatment options.",
                "prevention": ["Regular monitoring", "Good agricultural practices"]
            }
        }), 200
    
    # Return the information
    info = DISEASE_INFO[disease_key]
    info["name"] = disease_name  # Add the name to the response
    
    return jsonify({"status": "success", "info": info}), 200

@app.route('/feedback', methods=['POST'])
def feedback_api():
    """API endpoint to collect user feedback on predictions"""
    if not request.json:
        return jsonify({"status": "error", "message": "No data provided"}), 400
    
    required_fields = ['image_id', 'prediction', 'correct', 'user_correction']
    for field in required_fields:
        if field not in request.json:
            return jsonify({
                "status": "error", 
                "message": f"Missing required field: {field}"
            }), 400
    
    try:
        # Save the feedback
        feedback = {
            "image_id": request.json['image_id'],
            "prediction": request.json['prediction'],
            "correct": request.json['correct'],
            "user_correction": request.json['user_correction'],
            "comments": request.json.get('comments', '')
        }
        
        # Save to a JSON file
        feedback_dir = 'feedback'
        os.makedirs(feedback_dir, exist_ok=True)
        
        feedback_file = os.path.join(feedback_dir, 'feedback.json')
        
        # Load existing feedback if any
        all_feedback = []
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                all_feedback = json.load(f)
        
        # Append new feedback
        all_feedback.append(feedback)
        
        # Save the updated feedback
        with open(feedback_file, 'w') as f:
            json.dump(all_feedback, f, indent=2)
        
        logger.info(f"Saved feedback for image {feedback['image_id']}")
        
        return jsonify({"status": "success", "message": "Feedback saved successfully"}), 200
    
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

def start_server():
    """Start the Flask server"""
    try:
        # Load the model and class names
        load_model_and_classes()
        
        # Run the server
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
    
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    start_server()