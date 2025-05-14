import React from 'react';
import { Container, Typography, Button, Paper, Box, Grid } from '@mui/material';
import { useHistory } from 'react-router-dom';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const Home = () => {
  const history = useHistory();

  const handleUploadClick = () => {
    history.push('/upload');
  };

  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ p: 4, mt: 4, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Plant Disease Detection
        </Typography>
        
        <Typography variant="h6" color="textSecondary" paragraph>
          Identify plant diseases instantly with AI-powered analysis
        </Typography>
        
        <Box sx={{ my: 4 }}>
          <Button
            variant="contained"
            color="primary"
            size="large"
            startIcon={<CloudUploadIcon />}
            onClick={handleUploadClick}
            sx={{ py: 1.5, px: 4 }}
          >
            Upload Plant Image
          </Button>
        </Box>
        
        <Grid container spacing={3} sx={{ mt: 2, textAlign: 'left' }}>
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="h6" gutterBottom>
                Quick & Accurate
              </Typography>
              <Typography variant="body2">
                Get instant analysis of plant diseases with our advanced AI model trained on thousands of plant images.
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="h6" gutterBottom>
                Visual Explanations
              </Typography>
              <Typography variant="body2">
                See what our AI is looking at with heat maps that highlight the affected areas of your plant.
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="h6" gutterBottom>
                Treatment Advice
              </Typography>
              <Typography variant="body2">
                Receive tailored recommendations for treating identified diseases and preventing future occurrences.
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
};

export default Home;