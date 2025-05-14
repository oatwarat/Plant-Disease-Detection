// services/api.js
/**
 * API service for communicating with the Plant Disease Detection backend
 */

// Base API URL - change this to your deployed API endpoint
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';


/**
 * Analyze plant image for disease detection
 * 
 * @param {File} imageFile - The image file to analyze
 * @returns {Promise} - Promise with the analysis results
 */
export const analyzeImage = async (imageFile) => {
  try {
    // Create form data
    const formData = new FormData();
    formData.append('file', imageFile);
    
    // Make API request
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });
    
    // Parse response
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Error analyzing image');
    }
    
    return data;
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw error;
  }
};

/**
 * Analyze plant image using base64 encoding
 * 
 * @param {string} base64Image - Base64 encoded image data
 * @returns {Promise} - Promise with the analysis results
 */
export const analyzeBase64Image = async (base64Image) => {
  try {
    // Make API request
    const response = await fetch(`${API_BASE_URL}/predict/base64`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: base64Image }),
    });
    
    // Parse response
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Error analyzing image');
    }
    
    return data;
  } catch (error) {
    console.error('Error analyzing base64 image:', error);
    throw error;
  }
};

/**
 * Get information about a specific disease
 * 
 * @param {string} diseaseName - Name of the disease
 * @returns {Promise} - Promise with disease information
 */
export const getDiseaseInfo = async (diseaseName) => {
  try {
    // Make API request
    const response = await fetch(`${API_BASE_URL}/disease/info/${encodeURIComponent(diseaseName)}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    // Parse response
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Error fetching disease information');
    }
    
    return data;
  } catch (error) {
    console.error('Error fetching disease info:', error);
    throw error;
  }
};

/**
 * Get information about the model
 * 
 * @returns {Promise} - Promise with model information
 */
export const getModelInfo = async () => {
  try {
    // Make API request
    const response = await fetch(`${API_BASE_URL}/model/info`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    // Parse response
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Error fetching model information');
    }
    
    return data;
  } catch (error) {
    console.error('Error fetching model info:', error);
    throw error;
  }
};

/**
 * Submit user feedback
 * 
 * @param {Object} feedback - Feedback data
 * @param {string} feedback.image_id - Unique identifier for the image
 * @param {string} feedback.prediction - The model's prediction
 * @param {boolean} feedback.correct - Whether the prediction was correct
 * @param {string} feedback.user_correction - User's correction if prediction was wrong
 * @param {string} feedback.comments - Additional comments from the user
 * @param {number} feedback.helpfulness_rating - Rating of how helpful the results were
 * @returns {Promise} - Promise with submission result
 */
export const submitFeedback = async (feedback) => {
  try {
    // Make API request
    const response = await fetch(`${API_BASE_URL}/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(feedback),
    });
    
    // Parse response
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Error submitting feedback');
    }
    
    return data;
  } catch (error) {
    console.error('Error submitting feedback:', error);
    throw error;
  }
};

/**
 * Check API health
 * 
 * @returns {Promise} - Promise with health status
 */
export const checkApiHealth = async () => {
  try {
    // Make API request
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    // Parse response
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'API is not healthy');
    }
    
    return data;
  } catch (error) {
    console.error('Error checking API health:', error);
    throw error;
  }
};