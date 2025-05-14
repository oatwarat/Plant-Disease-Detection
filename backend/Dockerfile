# Use TensorFlow GPU image as base
FROM tensorflow/tensorflow:2.10.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directories for uploads and feedback
RUN mkdir -p uploads feedback

# Environment variables
ENV MODEL_PATH=best_model.h5
ENV CLASS_NAMES_PATH=class_names.txt
ENV PORT=5000

# Expose the port
EXPOSE 5000

# Command to run the application
CMD ["python", "model_deployment.py"]