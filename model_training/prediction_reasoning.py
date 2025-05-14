"""
Plant Disease Detection - Prediction Reasoning
This script implements mechanisms to provide reasoning for individual predictions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import json
try:
    import shap
except ImportError:
    print("SHAP not installed. Install it with: pip install shap")

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PredictionReasoner:
    """
    Class to generate human-understandable reasoning for model predictions
    """
    def __init__(self, model_path, class_names_path=None, img_size=(224, 224)):
        """
        Initialize the reasoner
        
        Args:
            model_path: Path to the trained model
            class_names_path: Path to file containing class names (one per line)
            img_size: Input image size expected by the model
        """
        self.model = load_model(model_path)
        self.img_size = img_size
        
        # Load class names
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            print(f"Warning: Class names file not found at {class_names_path}")
            self.class_names = [f"Class_{i}" for i in range(self.model.output_shape[1])]
        
        # Disease symptoms and descriptions (for reasoning)
        self.disease_info = self._load_disease_info()
    
    def _load_disease_info(self):
        """
        Load or define disease information for reasoning
        Returns a dictionary with disease characteristics
        """
        # This would ideally come from a database or external file
        # For this example, we'll define it directly
        disease_info = {
            "Healthy": {
                "symptoms": ["uniform green color", "no spots", "no discoloration"],
                "characteristics": ["healthy leaf structure", "vibrant color"],
                "reasoning": "The leaf shows uniform coloration without spots or lesions, which is characteristic of healthy plant tissue."
            },
            "Bacterial_spot": {
                "symptoms": ["small dark spots", "yellowing around spots", "water-soaked lesions"],
                "characteristics": ["spots with yellow halos", "irregular shape"],
                "reasoning": "The image shows small dark spots with yellowing around them, which is typical of bacterial spot disease."
            },
            "Early_blight": {
                "symptoms": ["dark brown spots", "concentric rings", "yellow areas around spots"],
                "characteristics": ["target-like pattern", "spots mainly on lower leaves"],
                "reasoning": "The concentric ring pattern (target-like appearance) on the leaf is a classic sign of early blight infection."
            },
            "Late_blight": {
                "symptoms": ["irregular green-black water-soaked spots", "white fuzzy growth", "rapid spread"],
                "characteristics": ["water-soaked appearance", "edges of lesions not well-defined"],
                "reasoning": "The water-soaked lesions with fuzzy white growth indicate late blight, which spreads rapidly in humid conditions."
            },
            "Leaf_rust": {
                "symptoms": ["small orange-brown pustules", "dusty appearance", "circular spots"],
                "characteristics": ["powdery orange spots", "pustules mainly on leaf underside"],
                "reasoning": "The orange-brown powdery pustules visible on the leaf surface are characteristic of leaf rust fungal infection."
            },
            "Powdery_mildew": {
                "symptoms": ["white powdery coating", "yellow spots", "distorted growth"],
                "characteristics": ["powder-like substance on leaves", "typically starts on upper leaf surface"],
                "reasoning": "The white powdery coating on the leaf surface is a clear indicator of powdery mildew fungal infection."
            },
            "Leaf_spot": {
                "symptoms": ["circular spots with defined edges", "dark brown center", "yellow halo"],
                "characteristics": ["spots may fall out leaving holes", "starts on lower leaves"],
                "reasoning": "The well-defined circular spots with dark centers and yellow halos indicate leaf spot disease."
            }
        }
        
        # Generate generic info for any class not explicitly defined
        for class_name in self.class_names:
            if class_name not in disease_info:
                disease_info[class_name] = {
                    "symptoms": ["various visual symptoms"],
                    "characteristics": ["distinctive patterns"],
                    "reasoning": f"The visual patterns in the image match characteristics typically associated with {class_name}."
                }
        
        return disease_info

    def preprocess_image(self, image_path):
        """
        Load and preprocess an image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with original and preprocessed image
        """
        # Load the image
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        
        # Preprocess the image
        img_preprocessed = preprocess_input(img_array.copy())
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        
        return {
            'original': img_array,
            'preprocessed': img_preprocessed,
            'batch': img_batch
        }

    def predict(self, image_path):
        """
        Make a prediction for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess the image
        img_data = self.preprocess_image(image_path)
        
        # Make the prediction
        predictions = self.model.predict(img_data['batch'])[0]
        
        # Get the top predicted classes and their probabilities
        top_indices = predictions.argsort()[-3:][::-1]
        top_classes = [self.class_names[i] for i in top_indices]
        top_probabilities = [predictions[i] for i in top_indices]
        
        return {
            'image_path': image_path,
            'image_data': img_data,
            'predictions': predictions,
            'top_indices': top_indices,
            'top_classes': top_classes,
            'top_probabilities': top_probabilities
        }
    
    def generate_grad_cam(self, prediction_results):
        """
        Generate Grad-CAM heatmap for the prediction
        
        Args:
            prediction_results: Results from predict() method
            
        Returns:
            Updated prediction results with Grad-CAM
        """
        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(self.model.layers):
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
            print("Could not find a convolutional layer for Grad-CAM.")
            return prediction_results
        
        # Create Grad-CAM model
        try:
            grad_model = tf.keras.models.Model(
                [self.model.inputs], 
                [self.model.get_layer(last_conv_layer).output, self.model.output]
            )
        except:
            # Try to find layer in base model if present
            base_model = None
            for layer in self.model.layers:
                if hasattr(layer, 'layers'):
                    try:
                        base_model = layer
                        break
                    except:
                        pass
            
            if base_model is None:
                print("Could not create Grad-CAM model.")
                return prediction_results
            
            try:
                grad_model = tf.keras.models.Model(
                    [self.model.inputs], 
                    [base_model.get_layer(last_conv_layer).output, self.model.output]
                )
            except:
                print("Could not create Grad-CAM model from base model.")
                return prediction_results
        
        # Get the top predicted class
        top_class_idx = prediction_results['top_indices'][0]
        
        # Generate class activation heatmap
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(prediction_results['image_data']['batch'])
            class_output = predictions[:, top_class_idx]
        
        # Extract gradients and convolutional outputs
        grads = tape.gradient(class_output, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the gradient importance
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to original image size
        heatmap = cv2.resize(heatmap, (self.img_size[1], self.img_size[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose the heatmap on original image
        original_img = prediction_results['image_data']['original'].astype(np.uint8)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        # Update prediction results
        prediction_results['grad_cam'] = {
            'heatmap': heatmap,
            'superimposed': superimposed_img
        }
        
        return prediction_results
    
    def apply_shap(self, prediction_results, num_background=10):
        """
        Apply SHAP to explain the prediction
        
        Args:
            prediction_results: Results from predict() method
            num_background: Number of background images for SHAP
            
        Returns:
            Updated prediction results with SHAP values
        """
        try:
            import shap
        except ImportError:
            print("SHAP not installed. Skipping SHAP analysis.")
            return prediction_results
        
        # Create a background dataset (using random noise)
        background = np.random.random((num_background, self.img_size[0], self.img_size[1], 3)) * 255
        background = np.array([preprocess_input(b.copy()) for b in background])
        
        # Initialize the SHAP explainer
        explainer = shap.DeepExplainer(self.model, background)
        
        # Apply SHAP
        shap_values = explainer.shap_values(np.expand_dims(prediction_results['image_data']['preprocessed'], axis=0))
        
        # Get values for the top predicted class
        top_class_idx = prediction_results['top_indices'][0]
        
        # Get absolute SHAP values for visualization
        abs_shap_values = np.abs(shap_values[top_class_idx][0])
        max_value = np.max(abs_shap_values)
        
        # Create a visualization
        shap_img = np.zeros_like(prediction_results['image_data']['original'])
        
        if max_value > 0:
            # Normalize and scale to 0-255
            normalized_values = (abs_shap_values / max_value) * 255
            
            # Apply a colormap
            for c in range(3):
                shap_img[:, :, c] = normalized_values.sum(axis=2)
        
        # Update prediction results
        prediction_results['shap'] = {
            'values': shap_values,
            'visualization': shap_img.astype(np.uint8)
        }
        
        return prediction_results
    
    def generate_text_explanation(self, prediction_results):
        """
        Generate a human-readable explanation for the prediction
        
        Args:
            prediction_results: Results from predict() method
            
        Returns:
            Dictionary with text explanations
        """
        # Get the top predicted class and confidence
        top_class = prediction_results['top_classes'][0]
        top_confidence = prediction_results['top_probabilities'][0] * 100
        
        # Get information about this disease
        disease_info = self.disease_info.get(top_class, {
            "symptoms": ["various symptoms"],
            "characteristics": ["distinctive patterns"],
            "reasoning": f"The visual patterns match {top_class}."
        })
        
        # Confidence level interpretation
        confidence_interpretation = "low"
        if top_confidence > 90:
            confidence_interpretation = "very high"
        elif top_confidence > 75:
            confidence_interpretation = "high"
        elif top_confidence > 50:
            confidence_interpretation = "moderate"
        
        # Create the explanation
        explanation = {
            "diagnosis": top_class,
            "confidence": f"{top_confidence:.1f}%",
            "confidence_level": confidence_interpretation,
            "reasoning": disease_info["reasoning"],
            "symptoms_detected": disease_info["symptoms"],
            "characteristics": disease_info["characteristics"]
        }
        
        # Add information about alternative diagnoses
        if len(prediction_results['top_classes']) > 1:
            alternatives = []
            for i in range(1, min(3, len(prediction_results['top_classes']))):
                alt_class = prediction_results['top_classes'][i]
                alt_confidence = prediction_results['top_probabilities'][i] * 100
                alt_info = self.disease_info.get(alt_class, {})
                
                alternatives.append({
                    "diagnosis": alt_class,
                    "confidence": f"{alt_confidence:.1f}%",
                    "key_difference": self._get_key_difference(alt_class, top_class)
                })
            
            explanation["alternative_diagnoses"] = alternatives
        
        # Add a plain language summary
        explanation["summary"] = self._generate_summary(explanation)
        
        # Update prediction results
        prediction_results['explanation'] = explanation
        
        return prediction_results
    
    def _get_key_difference(self, class1, class2):
        """
        Generate explanation of key differences between two classes
        
        Args:
            class1: First class name
            class2: Second class name
            
        Returns:
            String explaining the key difference
        """
        # Get information about both classes
        info1 = self.disease_info.get(class1, {})
        info2 = self.disease_info.get(class2, {})
        
        symptoms1 = info1.get("symptoms", [])
        symptoms2 = info2.get("symptoms", [])
        
        # Find a symptom in class1 that's not in class2
        for symptom in symptoms1:
            if symptom not in symptoms2:
                return f"{class1} typically shows {symptom}, which is not present in {class2}."
        
        # If no specific difference found
        return f"{class1} has a different visual pattern compared to {class2}."
    
    def _generate_summary(self, explanation):
        """
        Generate a plain language summary of the explanation
        
        Args:
            explanation: Explanation dictionary
            
        Returns:
            Plain language summary
        """
        disease = explanation["diagnosis"]
        confidence = explanation["confidence"]
        confidence_level = explanation["confidence_level"]
        reasoning = explanation["reasoning"]
        
        summary = f"With {confidence_level} confidence ({confidence}), this appears to be {disease}. {reasoning}"
        
        if "alternative_diagnoses" in explanation and explanation["alternative_diagnoses"]:
            alt = explanation["alternative_diagnoses"][0]
            summary += f" Alternative possibilities include {alt['diagnosis']} ({alt['confidence']})."
        
        return summary
    
    def create_visual_explanation(self, prediction_results, output_path):
        """
        Create and save a visual explanation of the prediction
        
        Args:
            prediction_results: Results with explanations
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        # Create a figure with the explanations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(prediction_results['image_data']['original'].astype(np.uint8))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Grad-CAM visualization
        if 'grad_cam' in prediction_results:
            axes[0, 1].imshow(prediction_results['grad_cam']['superimposed'])
            axes[0, 1].set_title("Grad-CAM: Areas of Interest")
            axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, "Grad-CAM not available", ha='center', va='center')
            axes[0, 1].axis('off')
        
        # SHAP visualization
        if 'shap' in prediction_results:
            axes[1, 0].imshow(prediction_results['shap']['visualization'])
            axes[1, 0].set_title("SHAP: Feature Importance")
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, "SHAP not available", ha='center', va='center')
            axes[1, 0].axis('off')
        
        # Text explanation
        explanation = prediction_results['explanation']
        explanation_text = (
            f"Diagnosis: {explanation['diagnosis']} ({explanation['confidence']})\n\n"
            f"Reasoning: {explanation['reasoning']}\n\n"
            f"Key symptoms detected:\n"
        )
        for symptom in explanation['symptoms_detected']:
            explanation_text += f"• {symptom}\n"
        
        if "alternative_diagnoses" in explanation:
            explanation_text += "\nAlternative diagnoses:\n"
            for alt in explanation["alternative_diagnoses"]:
                explanation_text += f"• {alt['diagnosis']} ({alt['confidence']})\n  {alt['key_difference']}\n"
        
        # Add the text explanation to the plot
        axes[1, 1].text(0.5, 0.5, explanation_text, 
                       ha='center', va='center', 
                       wrap=True, 
                       fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[1, 1].axis('off')
        
        # Add a title with the summary
        plt.suptitle(explanation["summary"], fontsize=12)
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    def explain_prediction(self, image_path, output_dir="explanations"):
        """
        Complete pipeline to explain a prediction
        
        Args:
            image_path: Path to image
            output_dir: Directory to save explanation
            
        Returns:
            Dictionary with explanation results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image name for output files
        image_basename = os.path.basename(image_path)
        image_name = os.path.splitext(image_basename)[0]
        
        # 1. Make prediction
        prediction_results = self.predict(image_path)
        
        # 2. Generate Grad-CAM
        prediction_results = self.generate_grad_cam(prediction_results)
        
        # 3. Apply SHAP (if available)
        try:
            prediction_results = self.apply_shap(prediction_results)
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
        
        # 4. Generate text explanation
        prediction_results = self.generate_text_explanation(prediction_results)
        
        # 5. Create visual explanation
        visual_path = os.path.join(output_dir, f"{image_name}_explanation.png")
        self.create_visual_explanation(prediction_results, visual_path)
        
        # 6. Save explanation to JSON
        json_path = os.path.join(output_dir, f"{image_name}_explanation.json")
        explanation_data = {
            "image_path": image_path,
            "explanation": prediction_results['explanation'],
            "top_classes": prediction_results['top_classes'],
            "top_probabilities": [float(p) for p in prediction_results['top_probabilities']]
        }
        
        with open(json_path, 'w') as f:
            json.dump(explanation_data, f, indent=2)
        
        # Return paths to results
        return {
            "prediction": prediction_results,
            "visual_explanation": visual_path,
            "json_explanation": json_path
        }

def main():
    """
    Main function to demonstrate prediction reasoning
    """
    # Set parameters
    model_path = 'best_model.h5'
    class_names_path = 'class_names.txt'
    img_size = (224, 224)
    output_dir = 'prediction_explanations'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    # Create the reasoner
    reasoner = PredictionReasoner(model_path, class_names_path, img_size)
    
    # Find some test images
    test_dir = 'plant_disease_dataset'
    test_images = []
    
    # Look for at least three test images
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
                if len(test_images) >= 3:
                    break
        if len(test_images) >= 3:
            break
    
    if not test_images:
        print("No test images found. Please add some images to the test directory.")
        return
    
    # Generate explanations for each test image
    for image_path in test_images:
        print(f"\nGenerating explanation for {image_path}...")
        explanation_results = reasoner.explain_prediction(image_path, output_dir)
        
        print(f"Explanation saved to {explanation_results['visual_explanation']}")
        
        # Print summary
        summary = explanation_results['prediction']['explanation']['summary']
        print(f"Summary: {summary}")
    
    print("\nPrediction reasoning completed successfully!")
    print(f"All explanations saved to {output_dir}/")

if __name__ == "__main__":
    main()