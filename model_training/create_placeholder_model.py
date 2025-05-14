import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

print("Creating placeholder model...")

# Create a basic MobileNetV2 model with random weights
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)  # 7 classes for common plant diseases

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model
model.save('best_model.h5')
print("Model saved as 'best_model.h5'")

# Create a simple class names file
with open('class_names.txt', 'w') as f:
    f.write("Healthy\nBacterial_spot\nEarly_blight\nLate_blight\nLeaf_rust\nPowdery_mildew\nLeaf_spot")
print("Class names saved as 'class_names.txt'")

# Move files to backend directory
backend_dir = os.path.join('..', 'backend')
os.system(f"cp best_model.h5 {backend_dir}/")
os.system(f"cp class_names.txt {backend_dir}/")
print(f"Files copied to backend directory: {backend_dir}")

print("Placeholder model creation completed successfully!")