"""
Plant Disease Detection - Model Versioning and Experimentation
This script implements model versioning and experimentation using MLflow
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Import MLflow for experiment tracking
import mlflow
import mlflow.keras

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

def build_model_v1(num_classes, img_size=(224, 224)):
    """
    Version 1: MobileNetV2 with default hyperparameters
    
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

def build_model_v2(num_classes, img_size=(224, 224)):
    """
    Version 2: ResNet50 with fine-tuning
    
    Args:
        num_classes: Number of classes to predict
        img_size: Input image size
        
    Returns:
        Compiled model
    """
    # Load the pretrained model
    base_model = ResNet50(
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
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=SGD(learning_rate=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_model_v3(num_classes, img_size=(224, 224)):
    """
    Version 3: EfficientNetB0 with deeper architecture
    
    Args:
        num_classes: Number of classes to predict
        img_size: Input image size
        
    Returns:
        Compiled model
    """
    # Load the pretrained model
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3)
    )
    
    # Freeze most of the base model layers but unfreeze the last few
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    
    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=RMSprop(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_track_experiment(model, model_name, train_generator, validation_generator, 
                              epochs=20, patience=5, experiment_name="plant_disease_detection"):
    """
    Train the model and track the experiment with MLflow
    
    Args:
        model: Compiled model
        model_name: Name of the model version
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs: Number of epochs to train
        patience: Patience for early stopping
        experiment_name: Name of the MLflow experiment
        
    Returns:
        Trained model and history
    """
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Start a new MLflow run
    with mlflow.start_run(run_name=model_name) as run:
        print(f"\nTraining model: {model_name}")
        
        # Log model parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", train_generator.batch_size)
        mlflow.log_param("learning_rate", model.optimizer.lr.numpy())
        mlflow.log_param("optimizer", model.optimizer.__class__.__name__)
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            f'best_{model_name}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
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
        
        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(validation_generator)
        
        # Log metrics
        mlflow.log_metric("final_val_loss", val_loss)
        mlflow.log_metric("final_val_accuracy", val_accuracy)
        
        # Log the training history
        for epoch in range(len(history.history['accuracy'])):
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
        
        # Generate and log training history plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'{model_name} - Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{model_name} - Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        history_plot_path = f"{model_name}_training_history.png"
        plt.savefig(history_plot_path)
        
        # Log the plot
        mlflow.log_artifact(history_plot_path)
        
        # Generate detailed metrics by class
        y_true = validation_generator.classes
        y_pred_prob = model.predict(validation_generator)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Get class names
        class_names = list(validation_generator.class_indices.keys())
        
        # Calculate F1 scores
        f1 = f1_score(y_true, y_pred, average='weighted')
        mlflow.log_metric("f1_score", f1)
        
        # Log confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{model_name} - Confusion Matrix')
        plt.tight_layout()
        cm_plot_path = f"{model_name}_confusion_matrix.png"
        plt.savefig(cm_plot_path)
        
        # Log the confusion matrix
        mlflow.log_artifact(cm_plot_path)
        
        # Log the best model
        mlflow.keras.log_model(model, "model")
        
        print(f"Model {model_name} training completed. Validation Accuracy: {val_accuracy:.4f}, F1 Score: {f1:.4f}")
        
        return model, history, run.info.run_id

def compare_models(run_ids, experiment_name="plant_disease_detection"):
    """
    Compare models from different runs
    
    Args:
        run_ids: List of run IDs to compare
        experiment_name: Name of the MLflow experiment
    """
    # Set up the experiment
    mlflow.set_experiment(experiment_name)
    
    # Get the experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    
    # Get the runs
    runs = []
    for run_id in run_ids:
        run = mlflow.get_run(run_id)
        runs.append(run)
    
    # Create comparison table
    comparison_data = []
    for run in runs:
        model_name = run.data.params.get("model_name", "Unknown")
        val_accuracy = run.data.metrics.get("final_val_accuracy", 0)
        val_loss = run.data.metrics.get("final_val_loss", 0)
        f1 = run.data.metrics.get("f1_score", 0)
        
        comparison_data.append({
            "Model": model_name,
            "Validation Accuracy": val_accuracy,
            "Validation Loss": val_loss,
            "F1 Score": f1
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Create comparison plots
    plt.figure(figsize=(15, 5))
    
    # Accuracy comparison
    plt.subplot(1, 3, 1)
    sns.barplot(x="Model", y="Validation Accuracy", data=comparison_df)
    plt.title("Validation Accuracy")
    plt.ylim(0.7, 1.0)  # Adjust as needed
    plt.xticks(rotation=45)
    
    # Loss comparison
    plt.subplot(1, 3, 2)
    sns.barplot(x="Model", y="Validation Loss", data=comparison_df)
    plt.title("Validation Loss")
    plt.xticks(rotation=45)
    
    # F1 score comparison
    plt.subplot(1, 3, 3)
    sns.barplot(x="Model", y="F1 Score", data=comparison_df)
    plt.title("F1 Score")
    plt.ylim(0.7, 1.0)  # Adjust as needed
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    comparison_plot_path = "model_comparison.png"
    plt.savefig(comparison_plot_path)
    
    print(f"\nComparison plot saved as: {comparison_plot_path}")
    
    # Get learning curves for each run
    plt.figure(figsize=(12, 10))
    
    # Training accuracy
    plt.subplot(2, 2, 1)
    for run in runs:
        model_name = run.data.params.get("model_name", "Unknown")
        client = mlflow.tracking.MlflowClient()
        metrics = client.get_metric_history(run.info.run_id, "train_accuracy")
        epochs = [metric.step for metric in metrics]
        values = [metric.value for metric in metrics]
        plt.plot(epochs, values, label=model_name)
    
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Validation accuracy
    plt.subplot(2, 2, 2)
    for run in runs:
        model_name = run.data.params.get("model_name", "Unknown")
        client = mlflow.tracking.MlflowClient()
        metrics = client.get_metric_history(run.info.run_id, "val_accuracy")
        epochs = [metric.step for metric in metrics]
        values = [metric.value for metric in metrics]
        plt.plot(epochs, values, label=model_name)
    
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Training loss
    plt.subplot(2, 2, 3)
    for run in runs:
        model_name = run.data.params.get("model_name", "Unknown")
        client = mlflow.tracking.MlflowClient()
        metrics = client.get_metric_history(run.info.run_id, "train_loss")
        epochs = [metric.step for metric in metrics]
        values = [metric.value for metric in metrics]
        plt.plot(epochs, values, label=model_name)
    
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Validation loss
    plt.subplot(2, 2, 4)
    for run in runs:
        model_name = run.data.params.get("model_name", "Unknown")
        client = mlflow.tracking.MlflowClient()
        metrics = client.get_metric_history(run.info.run_id, "val_loss")
        epochs = [metric.step for metric in metrics]
        values = [metric.value for metric in metrics]
        plt.plot(epochs, values, label=model_name)
    
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    learning_curves_path = "learning_curves_comparison.png"
    plt.savefig(learning_curves_path)
    
    print(f"Learning curves comparison saved as: {learning_curves_path}")
    
    # Identify the best model
    best_idx = comparison_df["F1 Score"].idxmax()
    best_model = comparison_df.iloc[best_idx]
    
    print(f"\nBest Model: {best_model['Model']}")
    print(f"Validation Accuracy: {best_model['Validation Accuracy']:.4f}")
    print(f"Validation Loss: {best_model['Validation Loss']:.4f}")
    print(f"F1 Score: {best_model['F1 Score']:.4f}")
    
    return comparison_df, best_model['Model']

def main():
    """
    Main function to run model versioning and experimentation
    """
    # Set parameters
    data_dir = 'plant_disease_dataset'
    img_size = (224, 224)
    batch_size = 32
    epochs = 20
    patience = 5
    experiment_name = "plant_disease_detection"
    
    # Create data generators
    train_generator, validation_generator, class_names = create_data_generators(
        data_dir, 
        img_size=img_size, 
        batch_size=batch_size
    )
    
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    
    # Build and train model version 1: MobileNetV2
    model_v1 = build_model_v1(len(class_names), img_size=img_size)
    _, _, run_id_v1 = train_and_track_experiment(
        model_v1, 
        "MobileNetV2_baseline", 
        train_generator, 
        validation_generator, 
        epochs=epochs,
        patience=patience,
        experiment_name=experiment_name
    )
    
    # Build and train model version 2: ResNet50
    model_v2 = build_model_v2(len(class_names), img_size=img_size)
    _, _, run_id_v2 = train_and_track_experiment(
        model_v2, 
        "ResNet50_deeper", 
        train_generator, 
        validation_generator, 
        epochs=epochs,
        patience=patience,
        experiment_name=experiment_name
    )
    
    # Build and train model version 3: EfficientNetB0
    model_v3 = build_model_v3(len(class_names), img_size=img_size)
    _, _, run_id_v3 = train_and_track_experiment(
        model_v3, 
        "EfficientNetB0_finetuned", 
        train_generator, 
        validation_generator, 
        epochs=epochs,
        patience=patience,
        experiment_name=experiment_name
    )
    
    # Compare the models
    run_ids = [run_id_v1, run_id_v2, run_id_v3]
    comparison_df, best_model_name = compare_models(run_ids, experiment_name=experiment_name)
    
    # Generate comparison report
    report = f"""
    # Model Versioning and Experimentation Report
    
    ## Models Compared
    1. MobileNetV2_baseline: MobileNetV2 architecture with default hyperparameters
    2. ResNet50_deeper: ResNet50 architecture with deeper classification layers
    3. EfficientNetB0_finetuned: EfficientNetB0 with fine-tuning of the last 10 layers
    
    ## Comparison Results
    {comparison_df.to_markdown()}
    
    ## Best Model: {best_model_name}
    
    ## Justification
    The {best_model_name} model was selected based on the highest F1 score, which balances precision and recall. 
    This model demonstrated the best overall performance on the validation dataset, with good accuracy and
    low loss. The F1 score is particularly important for plant disease detection as it ensures we're 
    correctly identifying diseases without too many false positives or false negatives.
    
    ## Recommendations for Production
    1. Perform additional testing on real-world images
    2. Implement confidence thresholds for predictions
    3. Consider model quantization for mobile deployment
    4. Implement periodic retraining with new data
    """
    
    # Save the report
    with open("model_comparison_report.md", "w") as f:
        f.write(report)
    
    print("\nExperimentation completed!")
    print(f"Comparison report saved as: model_comparison_report.md")
    print(f"Best model: {best_model_name}")

if __name__ == "__main__":
    main()