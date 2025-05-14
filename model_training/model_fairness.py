"""
Plant Disease Detection Model - Fairness Analysis
This script analyzes potential biases in the plant disease detection model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Fairness metrics
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_metadata_from_directory(data_dir, protected_attribute='image_brightness'):
    """
    Create metadata for images with a protected attribute
    In this example, we use image brightness as a protected attribute
    
    Args:
        data_dir: Directory containing the dataset
        protected_attribute: Name of the protected attribute
        
    Returns:
        DataFrame with metadata including protected attributes
    """
    from PIL import Image
    import numpy as np
    
    metadata = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(data_dir):
        if files:
            class_name = os.path.basename(root)
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(root, file)
                    
                    # Open the image
                    try:
                        img = Image.open(file_path).convert('RGB')
                        
                        # Calculate brightness
                        img_array = np.array(img)
                        brightness = np.mean(img_array)
                        
                        # Determine if image is "dark" or "bright"
                        is_bright = 1 if brightness > 127 else 0
                        
                        metadata.append({
                            'file_path': file_path,
                            'class': class_name,
                            'brightness': brightness,
                            'is_bright': is_bright
                        })
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
    
    return pd.DataFrame(metadata)

def analyze_dataset_bias(metadata):
    """
    Analyze the dataset for potential biases related to protected attributes
    
    Args:
        metadata: DataFrame with metadata
        
    Returns:
        DataFrame with bias analysis results
    """
    print("Analyzing dataset for potential biases...")
    
    # Check class distribution by protected attribute
    class_by_brightness = pd.crosstab(
        metadata['class'], 
        metadata['is_bright'], 
        normalize='index'
    )
    
    # Rename columns for clarity
    class_by_brightness.columns = ['Dark Images', 'Bright Images']
    
    print("\nClass distribution by image brightness:")
    print(class_by_brightness)
    
    # Plot the distribution
    plt.figure(figsize=(12, 6))
    class_by_brightness.plot(kind='bar', stacked=True)
    plt.title('Class Distribution by Image Brightness')
    plt.xlabel('Disease Class')
    plt.ylabel('Proportion')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('class_distribution_by_brightness.png')
    
    # Calculate statistical disparity
    disparity = []
    for class_name in metadata['class'].unique():
        class_data = metadata[metadata['class'] == class_name]
        
        # Calculate proportion of bright images
        bright_prop = (class_data['is_bright'] == 1).mean()
        
        # Calculate proportion of dark images
        dark_prop = (class_data['is_bright'] == 0).mean()
        
        # Calculate disparity
        disparity.append({
            'class': class_name,
            'bright_proportion': bright_prop,
            'dark_proportion': dark_prop,
            'disparity': abs(bright_prop - dark_prop)
        })
    
    disparity_df = pd.DataFrame(disparity)
    disparity_df = disparity_df.sort_values(by='disparity', ascending=False)
    
    print("\nStatistical disparity by class:")
    print(disparity_df)
    
    # Plot disparity
    plt.figure(figsize=(12, 6))
    sns.barplot(x='class', y='disparity', data=disparity_df)
    plt.title('Statistical Disparity by Class')
    plt.xlabel('Disease Class')
    plt.ylabel('Disparity (|bright_prop - dark_prop|)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('statistical_disparity.png')
    
    return disparity_df

def evaluate_model_fairness(model, data_dir, metadata, img_size=(224, 224), batch_size=32):
    """
    Evaluate the model for fairness across protected attributes
    
    Args:
        model: Trained model
        data_dir: Directory containing the test dataset
        metadata: DataFrame with metadata
        img_size: Image dimensions
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with fairness metrics
    """
    print("Evaluating model fairness...")
    
    # Create data generator for evaluation
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load the test data
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class indices
    class_indices = test_generator.class_indices
    class_names = list(class_indices.keys())
    
    # Make predictions
    y_pred_prob = model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_generator.classes
    
    # Get filenames from the generator
    filenames = [os.path.join(data_dir, fn) for fn in test_generator.filenames]
    
    # Create a dataframe with predictions
    results_df = pd.DataFrame({
        'file_path': filenames,
        'true_class': [class_names[i] for i in y_true],
        'predicted_class': [class_names[i] for i in y_pred],
        'correct': y_true == y_pred
    })
    
    # Merge with metadata to include protected attributes
    merged_df = pd.merge(results_df, metadata, on='file_path', how='left')
    
    # Calculate accuracy by protected attribute
    accuracy_by_brightness = merged_df.groupby('is_bright')['correct'].mean()
    
    print("\nAccuracy by image brightness:")
    print(accuracy_by_brightness)
    
    # Calculate accuracy by class and protected attribute
    accuracy_by_class_brightness = merged_df.groupby(['class', 'is_bright'])['correct'].mean().unstack()
    
    # Rename columns for clarity
    if 0 in accuracy_by_class_brightness.columns and 1 in accuracy_by_class_brightness.columns:
        accuracy_by_class_brightness.columns = ['Dark Images', 'Bright Images']
    
    print("\nAccuracy by class and image brightness:")
    print(accuracy_by_class_brightness)
    
    # Calculate fairness metrics - Disparate Impact
    if 'Dark Images' in accuracy_by_class_brightness.columns and 'Bright Images' in accuracy_by_class_brightness.columns:
        disparate_impact = accuracy_by_class_brightness['Dark Images'] / accuracy_by_class_brightness['Bright Images']
        
        # Avoid division by zero
        disparate_impact = disparate_impact.replace([np.inf, -np.inf], np.nan)
        
        print("\nDisparate Impact (Dark/Bright accuracy ratio):")
        print(disparate_impact)
        
        # Calculate fairness metrics - Equal Opportunity Difference
        equal_opp_diff = accuracy_by_class_brightness['Dark Images'] - accuracy_by_class_brightness['Bright Images']
        
        print("\nEqual Opportunity Difference (Dark - Bright accuracy):")
        print(equal_opp_diff)
        
        # Plot fairness metrics
        plt.figure(figsize=(12, 6))
        equal_opp_diff.plot(kind='bar')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Equal Opportunity Difference by Class')
        plt.xlabel('Disease Class')
        plt.ylabel('Dark - Bright Accuracy')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('equal_opportunity_difference.png')
    
    return {
        'accuracy_by_brightness': accuracy_by_brightness,
        'accuracy_by_class_brightness': accuracy_by_class_brightness
    }

def suggest_mitigation_strategies(fairness_metrics):
    """
    Suggest strategies to mitigate detected bias
    
    Args:
        fairness_metrics: Dictionary with fairness metrics
    """
    print("\nBias Mitigation Strategies:")
    
    # Check if there is significant disparity in accuracy
    if 'accuracy_by_brightness' in fairness_metrics:
        accuracy_diff = abs(fairness_metrics['accuracy_by_brightness'].iloc[1] - 
                            fairness_metrics['accuracy_by_brightness'].iloc[0])
        
        if accuracy_diff > 0.05:  # 5% difference threshold
            print("\n1. Data Augmentation Strategies:")
            print("   - Implement targeted brightness augmentation to balance exposure conditions")
            print("   - Apply random brightness adjustments during training")
            print("   - Add contrast normalization to reduce the impact of lighting conditions")
            
            print("\n2. Model Improvement Strategies:")
            print("   - Implement adaptive histogram equalization as a preprocessing step")
            print("   - Add a brightness normalization layer to the model")
            print("   - Train separate models for different lighting conditions and ensemble them")
            
            print("\n3. Fairness-aware Training:")
            print("   - Use a fairness-aware loss function that penalizes disparity")
            print("   - Implement class weighting based on lighting conditions")
            print("   - Apply adversarial debiasing techniques to make the model invariant to brightness")
    
    print("\n4. General Recommendations:")
    print("   - Collect more diverse data across different lighting conditions")
    print("   - Implement a preprocessing pipeline that standardizes image brightness")
    print("   - Add explainability features to help users understand model confidence")
    print("   - Monitor fairness metrics in production and retrain as needed")

def main():
    """
    Main function to perform fairness analysis
    """
    # Set parameters
    data_dir = 'plant_disease_dataset'
    model_path = 'plant_disease_model.h5'
    img_size = (224, 224)
    batch_size = 32
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return
    
    # 1. Load the trained model
    model = load_model(model_path)
    
    # 2. Create metadata for the dataset with protected attributes
    metadata = create_metadata_from_directory(data_dir)
    
    # 3. Analyze dataset bias
    disparity_df = analyze_dataset_bias(metadata)
    
    # 4. Evaluate model fairness
    fairness_metrics = evaluate_model_fairness(model, data_dir, metadata, img_size, batch_size)
    
    # 5. Suggest mitigation strategies
    suggest_mitigation_strategies(fairness_metrics)
    
    print("\nFairness analysis completed successfully!")

if __name__ == "__main__":
    main()