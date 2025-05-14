// pages/UploadImage.js
import React, { useState, useRef } from 'react';
import { useHistory } from 'react-router-dom';
import { 
  Container, 
  Paper, 
  Typography, 
  Button, 
  Box, 
  CircularProgress,
  Snackbar,
  Alert,
  Grid
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CameraAltIcon from '@mui/icons-material/CameraAlt';
import CropIcon from '@mui/icons-material/Crop';
import SendIcon from '@mui/icons-material/Send';
import ImageCropper from '../components/ImageCropper';
import { analyzeImage } from '../services/Api';

const UploadImage = ({ onAnalysisComplete }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showCropper, setShowCropper] = useState(false);
  const [croppedImage, setCroppedImage] = useState(null);
  
  const fileInputRef = useRef(null);
  const history = useHistory();

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type.match('image.*')) {
        setSelectedFile(file);
        const reader = new FileReader();
        reader.onload = () => {
          setPreview(reader.result);
        };
        reader.readAsDataURL(file);
        setError('');
      } else {
        setError('Please select an image file (png, jpg, jpeg)');
      }
    }
  };

  // Handle drag and drop
  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      if (file.type.match('image.*')) {
        setSelectedFile(file);
        const reader = new FileReader();
        reader.onload = () => {
          setPreview(reader.result);
        };
        reader.readAsDataURL(file);
        setError('');
      } else {
        setError('Please select an image file (png, jpg, jpeg)');
      }
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  // Handle camera capture
  const handleCameraCapture = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.capture = 'environment';
    input.onchange = (e) => handleFileChange(e);
    input.click();
  };

  // Open file dialog
  const handleBrowseClick = () => {
    fileInputRef.current.click();
  };

  // Submit image for analysis
  const handleSubmit = async () => {
    setLoading(true);
    setError('');
    
    try {
      const imageToAnalyze = croppedImage || selectedFile;
      const results = await analyzeImage(imageToAnalyze);
      
      if (results.status === 'success') {
        onAnalysisComplete(results.result, preview);
        history.push('/results');
      } else {
        setError(results.message || 'Analysis failed. Please try again.');
      }
    } catch (err) {
      setError('Error analyzing image: ' + (err.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  // Toggle cropper view
  const handleCropClick = () => {
    setShowCropper(!showCropper);
  };

  // Handle cropped image
  const handleCropComplete = (croppedImageData) => {
    setCroppedImage(croppedImageData);
    setShowCropper(false);
    
    // Create preview for cropped image
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(croppedImageData);
  };

  // Cancel cropping
  const handleCropCancel = () => {
    setShowCropper(false);
  };

  return (
    <Container maxWidth="md" className="upload-container">
      <Paper elevation={3} className="upload-paper">
        <Typography variant="h4" component="h1" gutterBottom>
          Upload Plant Image
        </Typography>
        
        <Typography variant="body1" gutterBottom>
          Upload a clear image of the plant leaf to detect diseases. For best results, ensure good lighting and focus on the affected area.
        </Typography>
        
        {!showCropper ? (
          <>
            <Box 
              className="dropzone"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={handleBrowseClick}
              sx={{
                border: '2px dashed #ccc',
                borderRadius: 2,
                padding: 4,
                textAlign: 'center',
                cursor: 'pointer',
                marginY: 3,
                backgroundColor: '#f9f9f9',
                minHeight: 200,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center'
              }}
            >
              {preview ? (
                <img 
                  src={preview} 
                  alt="Preview" 
                  style={{ maxWidth: '100%', maxHeight: 300, marginTop: 16 }} 
                />
              ) : (
                <>
                  <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
                  <Typography variant="h6">
                    Drag & Drop an image here or click to browse
                  </Typography>
                </>
              )}
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
                style={{ display: 'none' }}
              />
            </Box>
            
            <Grid container spacing={2} justifyContent="center">
              <Grid item>
                <Button
                  variant="contained"
                  startIcon={<CameraAltIcon />}
                  onClick={handleCameraCapture}
                >
                  Take Photo
                </Button>
              </Grid>
              
              {preview && (
                <>
                  <Grid item>
                    <Button
                      variant="outlined"
                      startIcon={<CropIcon />}
                      onClick={handleCropClick}
                    >
                      Crop Image
                    </Button>
                  </Grid>
                  
                  <Grid item>
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={loading ? <CircularProgress size={24} color="inherit" /> : <SendIcon />}
                      onClick={handleSubmit}
                      disabled={loading}
                    >
                      {loading ? 'Analyzing...' : 'Analyze Disease'}
                    </Button>
                  </Grid>
                </>
              )}
            </Grid>
          </>
        ) : (
          <ImageCropper 
            image={preview} 
            onCropComplete={handleCropComplete} 
            onCancel={handleCropCancel}
          />
        )}
      </Paper>
      
      <Snackbar open={!!error} autoHideDuration={6000} onClose={() => setError('')}>
        <Alert onClose={() => setError('')} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default UploadImage;