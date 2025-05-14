# Plant Disease Detection API Documentation

This document provides information about the Plant Disease Detection API service, which offers plant disease diagnosis through image analysis using machine learning.

## Base URL

`http://localhost:5001`

## Authentication

Currently, the API does not require authentication.

## API Endpoints

### Health Check

**Request**
- **URL**: `/health`
- **Method**: `GET`

**Response**
- **Success Response**:
  - **Code**: 200
  - **Content**: 
    ```json
    {
      "status": "ok",
      "message": "API is running"
    }
    ```
- **Error Response**:
  - **Code**: 503
  - **Content**: 
    ```json
    {
      "status": "error",
      "message": "Model not loaded"
    }
    ```

### Predict (File Upload)

**Request**
- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Image file to analyze (JPG, JPEG, or PNG)
- **Query Parameters**:
  - `save` (optional): Set to "true" to save the uploaded image on the server

**Response**
- **Success Response**:
  - **Code**: 200
  - **Content**: 
    ```json
    {
      "status": "success",
      "result": {
        "prediction": [
          {
            "class": "Healthy",
            "probability": 0.95,
            "confidence": "95.00%"
          },
          {
            "class": "Bacterial_spot",
            "probability": 0.03,
            "confidence": "3.00%"
          },
          {
            "class": "Early_blight",
            "probability": 0.02,
            "confidence": "2.00%"
          }
        ],
        "disease_info": {
          "description": "The plant appears healthy with no visible signs of disease.",
          "treatment": "Continue regular maintenance and monitoring.",
          "prevention": [
            "Regular watering",
            "Proper nutrition",
            "Adequate sunlight"
          ]
        },
        "grad_cam": {
          "heatmap": "base64_encoded_image",
          "superimposed": "base64_encoded_image"
        }
      }
    }
    ```
- **Error Responses**:
  - **Code**: 400
    - No file part
    - No selected file
    - File type not allowed
  - **Code**: 500
    - Error processing image

### Predict (Base64 Image)

**Request**
- **URL**: `/predict/base64`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "image": "base64_encoded_image_data"
  }