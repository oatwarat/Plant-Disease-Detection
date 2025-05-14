"""
Plant Disease Detection Model Training
This script trains a CNN model for plant disease detection using transfer learning
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --------------------------------------
# 1. Data Exploration
# --------------------------------------

def explore_dataset(data_dir):
    """
    Explore the dataset structure and characteristics
    
    Args:
        data_dir: Root directory of the dataset
    """
    print("Dataset Exploration:")
    
    # Count classes and samples
    classes = os.listdir(data_dir)
    class_counts = {}
    
    for cls in classes:
        class_path = os.path.join(data_dir, cls)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            class_counts[cls] = count
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    total_images = df['Count'].sum()
    
    print(f"Total number of classes: {len(df)}")
    print(f"Total number of images: {total_images}")
    print("\nClass distribution:")
    print(df)
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Count', data=df)
    plt.xticks(rotation=90)
    plt.title('Number of Images per Class')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    
    # Sample images from each class for visual inspection
    sample_images(data_dir, classes, 'sample_images.png')
    
    return df

def sample_images(data_dir, classes, output_file, samples_per_class=3):
    """
    Display sample images from each class
    
    Args:
        data_dir: Root directory of the dataset
        classes: List of class names
        output_file: File to save the visualization
        samples_per_class: Number of samples to show per class
    """
    import matplotlib.image as mpimg
    
    # Filter only directories
    classes = [cls for cls in classes if os.path.isdir(os.path.join(data_dir, cls))]
    
    # Set up the figure
    fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(12, 2*len(classes)))
    
    # For each class, show sample images
    for i, cls in enumerate(classes):
        class_path = os.path.join(data_dir, cls)
        image_files = os.listdir(class_path)[:samples_per_class]
        
        for j, img_file in enumerate(image_files):
            img_path = os.path.join(class_path, img_file)
            img = mpimg.imread(img_path)
            if len(classes) > 1:
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(f"{cls}")
            else:
                axes[j].imshow(img)
                axes[j].axis('off')
                axes[j].set_title(f"{cls}")
    
    plt.tight_layout()
    plt.savefig(output_file)

# --------------------------------------
# 2. Data Preprocessing
# --------------------------------------

def create_data_generators(data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Create training and validation data generators
    
    Args:
        data_dir: Root directory of the dataset
        img_size: Target image size (height, width)
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        
    Returns:
        train_generator, validation_generator, class_names
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = valid_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Get class names
    class_names = list(train_generator.class_indices.keys())
    
    return train_generator, validation_generator, class_names

# --------------------------------------
# 3. Model Architecture
# --------------------------------------

def build_model(num_classes, img_size=(224, 224)):
    """
    Build the CNN model using transfer learning with MobileNetV2
    
    Args:
        num_classes: Number of classes to predict
        img_size: Input image size
        
    Returns:
        Compiled model
    """
    # Load the pretrained model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3)
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# --------------------------------------
# 4. Training Loop
# --------------------------------------

def train_model(model, train_generator, validation_generator, epochs=20):
    """
    Train the model with appropriate callbacks
    
    Args:
        model: Compiled model
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs: Number of epochs to train
        
    Returns:
        Training history
    """
    # Callbacks
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return history

# --------------------------------------
# 5. Validation Methodology
# --------------------------------------

def evaluate_model(model, validation_generator, class_names):
    """
    Evaluate the model and generate classification report
    
    Args:
        model: Trained model
        validation_generator: Validation data generator
        class_names: List of class names
        
    Returns:
        Evaluation metrics
    """
    # Reset the generator
    validation_generator.reset()
    
    # Get the true labels
    y_true = validation_generator.classes
    
    # Predict with the model
    y_pred_prob = model.predict(validation_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Generate classification report
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    return report

# --------------------------------------
# 6. Results Analysis
# --------------------------------------

def plot_training_history(history):
    """
    Plot the training history
    
    Args:
        history: Training history object
    """
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')

def analyze_results(report, class_names):
    """
    Analyze the model performance in detail
    
    Args:
        report: Classification report dictionary
        class_names: List of class names
    """
    # Extract metrics
    df_metrics = pd.DataFrame({
        'Class': class_names,
        'Precision': [report[cls]['precision'] for cls in class_names],
        'Recall': [report[cls]['recall'] for cls in class_names],
        'F1-Score': [report[cls]['f1-score'] for cls in class_names],
        'Support': [report[cls]['support'] for cls in class_names]
    })
    
    # Print metrics table
    print("\nDetailed Class Metrics:")
    print(df_metrics)
    
    # Plot metrics by class
    plt.figure(figsize=(12, 6))
    df_metrics_melt = pd.melt(
        df_metrics, 
        id_vars=['Class'], 
        value_vars=['Precision', 'Recall', 'F1-Score'],
        var_name='Metric',
        value_name='Value'
    )
    
    sns.barplot(x='Class', y='Value', hue='Metric', data=df_metrics_melt)
    plt.title('Model Performance by Class')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('performance_by_class.png')
    
    # Return the metrics dataframe
    return df_metrics

# --------------------------------------
# 7. Main Function
# --------------------------------------

def main():
    """
    Main function to run the training pipeline
    """
    # Set parameters
    data_dir = 'plant_disease_dataset'
    img_size = (224, 224)
    batch_size = 32
    epochs = 20
    
    # 1. Explore dataset
    class_df = explore_dataset(data_dir)
    
    # 2. Create data generators
    train_generator, validation_generator, class_names = create_data_generators(
        data_dir, 
        img_size=img_size, 
        batch_size=batch_size
    )
    print(f"Class names: {class_names}")
    
    # 3. Build model
    num_classes = len(class_names)
    model = build_model(num_classes, img_size=img_size)
    print(model.summary())
    
    # 4. Train model
    history = train_model(
        model,
        train_generator,
        validation_generator,
        epochs=epochs
    )
    
    # 5. Evaluate model
    report = evaluate_model(model, validation_generator, class_names)
    
    # 6. Analyze results
    plot_training_history(history)
    metrics_df = analyze_results(report, class_names)
    
    # 7. Save model and class names
    model.save('plant_disease_model.h5')
    
    # Save class names for inference
    with open('class_names.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print("\nTraining completed successfully!")
    print("Model saved as 'plant_disease_model.h5'")
    print("Class names saved as 'class_names.txt'")

if __name__ == "__main__":
    main()