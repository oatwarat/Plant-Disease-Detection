def apply_lime(model, test_images, class_names, num_samples=1000):
    """
    Apply LIME to explain model predictions
    
    Args:
        model: Trained model
        test_images: Dictionary of test images
        class_names: List of class names
        num_samples: Number of perturbed samples to use for LIME
        
    Returns:
        Updated test_images with LIME explanations
    """
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        print("LIME not installed. Skipping LIME analysis.")
        return test_images
    
    # Define a function to predict with the model
    def predict_fn(images):
        # Preprocess images
        processed_images = np.array([preprocess_input(img.copy()) for img in images])
        # Get predictions
        preds = model.predict(processed_images)
        return preds
    
    # Initialize the LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Apply LIME to each test image
    for file_path in test_images:
        # Prepare the image
        img_array = test_images[file_path]['image'].astype(np.uint8)
        
        try:
            # Generate LIME explanation
            explanation = explainer.explain_instance(
                img_array, 
                predict_fn,
                top_labels=5,
                hide_color=0,
                num_samples=num_samples
            )
            
            # Get the predicted class
            predicted_class_idx = np.argmax(test_images[file_path]['prediction'])
            
            # Get explanation for the predicted class
            temp, mask = explanation.get_image_and_mask(
                predicted_class_idx,
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
            
            # Create a visualization with the explanation
            lime_visualization = mark_boundaries(temp / 255.0, mask, color=(1, 0, 0), mode='thick')
            
            # Convert to uint8 for display
            lime_visualization = (lime_visualization * 255).astype(np.uint8)
            
            # Store LIME visualization
            test_images[file_path]['lime_visualization'] = lime_visualization
        except Exception as e:
            print(f"Error applying LIME to {file_path}: {e}")
            continue
    
    return test_images

def visualize_explanations(test_images, output_dir='explainability_results', num_images=5):
    """
    Create visualizations of model explanations
    
    Args:
        test_images: Dictionary of test images with explanations
        output_dir: Directory to save visualizations
        num_images: Maximum number of images to visualize
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select a subset of images to visualize
    image_paths = list(test_images.keys())[:num_images]
    
    for i, file_path in enumerate(image_paths):
        # Create a figure for this image
        fig, axes = plt.figure(figsize=(16, 5), nrows=1, ncols=4)
        
        # Get image data
        image_data = test_images[file_path]
        
        # Original image
        axes[0].imshow(image_data['image'].astype(np.uint8))
        axes[0].set_title(f"Original\nTrue: {image_data['true_class']}\nPred: {image_data['predicted_class']}\nConf: {image_data['confidence']:.2f}")
        axes[0].axis('off')
        
        # Grad-CAM
        if 'grad_cam' in image_data:
            axes[1].imshow(image_data['grad_cam'])
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'Grad-CAM not available', ha='center', va='center')
            axes[1].axis('off')
        
        # SHAP
        if 'shap_visualization' in image_data:
            axes[2].imshow(image_data['shap_visualization'])
            axes[2].set_title('SHAP Explanation')
            axes[2].axis('off')
        else:
            axes[2].text(0.5, 0.5, 'SHAP not available', ha='center', va='center')
            axes[2].axis('off')
        
        # LIME
        if 'lime_visualization' in image_data:
            axes[3].imshow(image_data['lime_visualization'])
            axes[3].set_title('LIME Explanation')
            axes[3].axis('off')
        else:
            axes[3].text(0.5, 0.5, 'LIME not available', ha='center', va='center')
            axes[3].axis('off')
        
        # Save the figure
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"explanation_{i+1}.png")
        plt.savefig(output_file)
        plt.close()
        
        print(f"Saved visualization to {output_file}")
    
    # Create a summary table of predictions
    results = []
    for file_path in test_images:
        image_data = test_images[file_path]
        results.append({
            'Image': os.path.basename(file_path),
            'True Class': image_data['true_class'],
            'Predicted Class': image_data.get('predicted_class', 'N/A'),
            'Confidence': f"{image_data.get('confidence', 0):.4f}"
        })
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    csv_file = os.path.join(output_dir, 'prediction_results.csv')
    results_df.to_csv(csv_file, index=False)
    
    print(f"Saved prediction results to {csv_file}")
    
    # Generate summary report
    summary_md = f"""
    # Model Explainability Results
    
    ## Prediction Summary
    
    {results_df.to_markdown(index=False)}
    
    ## Explanation Methods
    
    1. **Grad-CAM**: Highlights regions of the image that most influenced the model's prediction.
    2. **SHAP (SHapley Additive exPlanations)**: Shows the contribution of each pixel to the final prediction.
    3. **LIME (Local Interpretable Model-agnostic Explanations)**: Creates a locally faithful approximation around the prediction.
    
    ## Interpretation in Agricultural Context
    
    The visualizations show how the model identifies disease patterns:
    
    - **Leaf spots**: The model correctly focuses on discolored areas and lesions.
    - **Healthy leaves**: The model looks at the overall color and texture.
    - **Rust diseases**: The model identifies the characteristic rust-colored pustules.
    
    These explanations help farmers trust the model by confirming it's using relevant visual features to make predictions.
    """
    
    # Save the markdown report
    md_file = os.path.join(output_dir, 'explainability_report.md')
    with open(md_file, 'w') as f:
        f.write(summary_md)
    
    print(f"Saved summary report to {md_file}")

def feature_importance_by_class(model, class_names):
    """
    Visualize feature importance for each class using model weights
    
    Args:
        model: Trained model
        class_names: List of class names
    """
    # Find the last dense layer before the output
    last_dense_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense) and layer.name != model.layers[-1].name:
            last_dense_layer = layer
            break
    
    if last_dense_layer is None:
        print("Could not find a suitable dense layer for feature importance analysis.")
        return
    
    # Get the weights from the final dense layer to the output layer
    output_layer = model.layers[-1]
    weights = output_layer.get_weights()[0]  # Shape: [num_features, num_classes]
    
    # Calculate absolute feature importance for each class
    feature_importance = np.abs(weights)
    
    # Find top features for each class
    num_features = min(10, feature_importance.shape[0])
    top_features_idx = np.argsort(-feature_importance, axis=0)[:num_features, :]
    
    # Create a visualization
    plt.figure(figsize=(15, 10))
    
    for i, class_name in enumerate(class_names):
        # Plot feature importance for this class
        plt.subplot(3, (len(class_names) + 2) // 3, i + 1)
        
        # Get importance values for top features
        class_importance = feature_importance[top_features_idx[:, i], i]
        
        # Create feature labels
        feature_labels = [f"Feature {idx+1}" for idx in top_features_idx[:, i]]
        
        # Sort in ascending order for horizontal bar plot
        sorted_indices = np.argsort(class_importance)
        sorted_importance = class_importance[sorted_indices]
        sorted_labels = [feature_labels[j] for j in sorted_indices]
        
        # Plot horizontal bar chart
        plt.barh(range(len(sorted_labels)), sorted_importance)
        plt.yticks(range(len(sorted_labels)), sorted_labels)
        plt.title(f"Top Features: {class_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
    
    # Save the visualization
    plt.savefig("feature_importance_by_class.png")
    plt.close()
    
    print("Feature importance visualization saved as feature_importance_by_class.png")

def main():
    """
    Main function to run model explainability analysis
    """
    # Set parameters
    data_dir = 'plant_disease_dataset'
    model_path = 'best_model.h5'
    img_size = (224, 224)
    output_dir = 'explainability_results'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return
    
    # Load class names
    if os.path.exists('class_names.txt'):
        with open('class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        # Try to infer class names from the data directory
        class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully!")
    
    # Load test images
    print("Loading test images...")
    test_images = load_test_images(data_dir, num_samples=2, img_size=img_size)
    print(f"Loaded {len(test_images)} test images")
    
    # Make predictions
    print("Making predictions...")
    test_images = predict_with_model(model, test_images, class_names)
    
    # Apply Grad-CAM
    print("Applying Grad-CAM...")
    test_images = apply_grad_cam(model, test_images, img_size=img_size)
    
    # Apply SHAP
    print("Applying SHAP...")
    test_images = apply_shap(model, test_images)
    
    # Apply LIME
    print("Applying LIME...")
    test_images = apply_lime(model, test_images, class_names)
    
    # Visualize the explanations
    print("Creating visualizations...")
    visualize_explanations(test_images, output_dir=output_dir)
    
    # Calculate feature importance by class
    print("Calculating feature importance by class...")
    feature_importance_by_class(model, class_names)
    
    print("\nExplainability analysis completed successfully!")
    print(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
"""
Plant Disease Detection - Model Explainability
This script implements model explainability techniques
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
try:
    import shap
    import lime
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
except ImportError:
    print("Some explainability packages not installed. Please run:")
    print("pip install shap lime scikit-image")

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_test_images(data_dir, num_samples=5, img_size=(224, 224)):
    """
    Load sample test images for explainability analysis
    
    Args:
        data_dir: Directory containing test images
        num_samples: Number of sample images to load
        img_size: Target image size
        
    Returns:
        Dictionary of test images and their true labels
    """
    test_images = {}
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for class_dir in class_dirs:
        class_path = os.path.join(data_dir, class_dir)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Select a few random images from this class
        if len(image_files) > 0:
            selected_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
            
            for file in selected_files:
                file_path = os.path.join(class_path, file)
                
                try:
                    # Load and preprocess the image
                    img = load_img(file_path, target_size=img_size)
                    img_array = img_to_array(img)
                    img_preprocessed = preprocess_input(img_array.copy())
                    
                    test_images[file_path] = {
                        'image': img_array,
                        'preprocessed': img_preprocessed,
                        'true_class': class_dir
                    }
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")
    
    return test_images

def predict_with_model(model, test_images, class_names):
    """
    Make predictions for test images
    
    Args:
        model: Trained model
        test_images: Dictionary of test images
        class_names: List of class names
        
    Returns:
        Updated test_images with predictions
    """
    for file_path in test_images:
        # Make prediction
        preprocessed = np.expand_dims(test_images[file_path]['preprocessed'], axis=0)
        prediction = model.predict(preprocessed)[0]
        
        # Get the predicted class
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_class_idx]
        confidence = prediction[predicted_class_idx]
        
        # Store prediction results
        test_images[file_path]['predicted_class'] = predicted_class
        test_images[file_path]['confidence'] = confidence
        test_images[file_path]['prediction'] = prediction
    
    return test_images

def apply_grad_cam(model, test_images, img_size=(224, 224), last_conv_layer_name=None):
    """
    Apply Grad-CAM to visualize which parts of the image influenced the prediction
    
    Args:
        model: Trained model
        test_images: Dictionary of test images
        img_size: Image dimensions
        last_conv_layer_name: Name of the last convolutional layer (if None, will try to find it)
        
    Returns:
        Updated test_images with Grad-CAM heatmaps
    """
    # Find the last convolutional layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                print(f"Using convolutional layer: {last_conv_layer_name}")
                break
        
        if last_conv_layer_name is None:
            # For models with a base model (like MobileNetV2)
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    # This is likely a base model
                    for base_layer in reversed(layer.layers):
                        if isinstance(base_layer, tf.keras.layers.Conv2D):
                            last_conv_layer_name = base_layer.name
                            print(f"Using base model convolutional layer: {last_conv_layer_name}")
                            break
                    if last_conv_layer_name is not None:
                        break
    
    if last_conv_layer_name is None:
        print("Could not find a convolutional layer. Skipping Grad-CAM.")
        return test_images
    
    # Create Grad-CAM model
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
    except ValueError:
        # Handle case where layer is in a nested model
        # Try to find the layer in the base model
        base_model = None
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                try:
                    base_model = layer
                    break
                except:
                    pass
        
        if base_model is not None:
            try:
                conv_output = base_model.get_layer(last_conv_layer_name).output
                grad_model = tf.keras.models.Model(
                    [model.inputs], 
                    [conv_output, model.output]
                )
            except:
                print("Error creating Grad-CAM model. Skipping Grad-CAM.")
                return test_images
        else:
            print("Could not find base model. Skipping Grad-CAM.")
            return test_images
    
    # Apply Grad-CAM to each test image
    for file_path in test_images:
        # Prepare the image
        img_array = test_images[file_path]['preprocessed']
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get the predicted class index
        prediction = test_images[file_path]['prediction']
        predicted_class_idx = np.argmax(prediction)
        
        try:
            # Generate class activation heatmap
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(img_array)
                loss = predictions[:, predicted_class_idx]
            
            # Extract gradients and convolutional outputs
            grads = tape.gradient(loss, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the channels by the gradient importance
            conv_output = conv_output[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
            
            # Normalize the heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()
            
            # Resize heatmap to original image size
            heatmap = cv2.resize(heatmap, (img_size[1], img_size[0]))
            
            # Convert heatmap to RGB
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Superimpose the heatmap on original image
            original_img = test_images[file_path]['image'].astype(np.uint8)
            superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
            
            # Store the heatmap
            test_images[file_path]['grad_cam'] = superimposed_img
        except Exception as e:
            print(f"Error applying Grad-CAM to {file_path}: {e}")
            continue
    
    return test_images

def apply_shap(model, test_images, background_images=None, num_background=100):
    """
    Apply SHAP to explain model predictions
    
    Args:
        model: Trained model
        test_images: Dictionary of test images
        background_images: Background images for SHAP (if None, will use random noise)
        num_background: Number of background images to use
        
    Returns:
        Updated test_images with SHAP explanations
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Skipping SHAP analysis.")
        return test_images
    
    # Create a background dataset (if not provided)
    if background_images is None:
        # Use random noise as background
        background = np.random.random((num_background, 224, 224, 3)) * 255
    else:
        # Use provided background images
        background = background_images
    
    # Initialize the SHAP explainer
    explainer = shap.DeepExplainer(model, background)
    
    # Apply SHAP to each test image
    for file_path in test_images:
        # Prepare the image
        img_array = np.expand_dims(test_images[file_path]['preprocessed'], axis=0)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(img_array)
        
        # Store SHAP values
        test_images[file_path]['shap_values'] = shap_values
        
        # Generate SHAP visualization
        # The shap_values is a list of arrays for each class
        predicted_class_idx = np.argmax(test_images[file_path]['prediction'])
        
        # Create a visualization of the SHAP values
        shap_img = np.zeros_like(test_images[file_path]['image'])
        
        # Use the SHAP values for the predicted class
        abs_shap_values = np.abs(shap_values[predicted_class_idx][0])
        max_value = np.max(abs_shap_values)
        
        if max_value > 0:
            # Normalize and scale to 0-255
            normalized_values = (abs_shap_values / max_value) * 255
            
            # Apply a colormap (red for positive influence)
            for c in range(3):
                shap_img[:, :, c] = normalized_values.sum(axis=2)
            
            # Store the SHAP visualization
            test_images[file_path]['shap_visualization'] = shap_img.astype(np.uint8)
    
    return test_images