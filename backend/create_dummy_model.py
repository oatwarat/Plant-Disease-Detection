import tensorflow as tf
from tensorflow.keras import layers, models

# Create a simple model for testing
print("Creating dummy model for testing...")
model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 3)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 disease classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save the model
print("Saving model to best_model.h5...")
model.save('best_model.h5')

# Create class_names.txt file
print("Creating class_names.txt...")
classes = [
    "Healthy",
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_rust",
    "Powdery_mildew", 
    "Leaf_spot"
]

with open('class_names.txt', 'w') as f:
    f.write('\n'.join(classes))

print("Done! Files created successfully.")