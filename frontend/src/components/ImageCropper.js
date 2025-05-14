import React, { useState, useCallback } from 'react';
import Cropper from 'react-easy-crop';
import { Button, Box, Slider, Typography, Paper } from '@mui/material';

const ImageCropper = ({ image, onCropComplete, onCancel }) => {
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [croppedAreaPixels, setCroppedAreaPixels] = useState(null);

  const onCropChange = (newCrop) => {
    setCrop(newCrop);
  };

  const onZoomChange = (newZoom) => {
    setZoom(newZoom);
  };

  const onCropAreaChange = useCallback((croppedArea, croppedAreaPixels) => {
    setCroppedAreaPixels(croppedAreaPixels);
  }, []);

  const handleCropComplete = useCallback(async () => {
    if (!croppedAreaPixels) return;

    // Create a temporary canvas to extract the cropped image
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const image = new Image();
    image.src = image;

    try {
      // Wait for the image to load
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = reject;
      });

      // Set canvas dimensions to the cropped size
      canvas.width = croppedAreaPixels.width;
      canvas.height = croppedAreaPixels.height;

      // Draw the cropped image
      ctx.drawImage(
        image,
        croppedAreaPixels.x,
        croppedAreaPixels.y,
        croppedAreaPixels.width,
        croppedAreaPixels.height,
        0,
        0,
        croppedAreaPixels.width,
        croppedAreaPixels.height
      );

      // Convert to blob and then to File object
      canvas.toBlob((blob) => {
        const croppedFile = new File([blob], 'cropped-image.jpg', {
          type: 'image/jpeg',
        });
        onCropComplete(croppedFile);
      }, 'image/jpeg');
    } catch (e) {
      console.error('Error cropping image:', e);
    }
  }, [croppedAreaPixels, image, onCropComplete]);

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Crop Image
      </Typography>
      <Box sx={{ position: 'relative', height: 300, mb: 3 }}>
        <Cropper
          image={image}
          crop={crop}
          zoom={zoom}
          aspect={1}
          onCropChange={onCropChange}
          onZoomChange={onZoomChange}
          onCropComplete={onCropAreaChange}
        />
      </Box>
      <Box sx={{ mb: 2 }}>
        <Typography>Zoom</Typography>
        <Slider
          value={zoom}
          min={1}
          max={3}
          step={0.1}
          onChange={(e, zoom) => setZoom(zoom)}
        />
      </Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Button variant="outlined" onClick={onCancel}>
          Cancel
        </Button>
        <Button variant="contained" color="primary" onClick={handleCropComplete}>
          Crop
        </Button>
      </Box>
    </Paper>
  );
};

export default ImageCropper;