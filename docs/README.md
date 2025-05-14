# Plant Disease Detection System

A proof-of-concept AI-enabled system for plant disease detection using computer vision and deep learning.

## Overview

This system identifies plant diseases from leaf images, providing treatment recommendations and visual explanations. It consists of a Flask backend API, React frontend interface, and a machine learning model implemented with TensorFlow.

## Project Structure
plant-disease-detection/
├── backend/              # Flask API server
├── frontend/             # React application
├── model_training/       # ML model implementation
└── docs/                 # Documentation

## Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

## Quick Start

### 1. Setup Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### 2. Generate Placeholder Model
```bash
# Navigate to model_training directory
cd ../model_training

# Create placeholder model
python create_placeholder_model.py

# Ensure model files are copied to backend
cp best_model.h5 ../backend/
cp class_names.txt ../backend/

```
### 3. Setup Frontend

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

###4. Run the Application
##Start Backend Server
# Navigate to backend directory
cd ../backend

# Run the server
PORT=5001 python model_deployment.py

##Start Frontend Development Server
In a new terminal:
# Navigate to frontend directory
cd frontend

# Start development server
npm start

The application will open in your browser at http://localhost:3000
```