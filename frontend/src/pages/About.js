import React from 'react';
import { Container, Typography, Paper, Box, Divider, List, ListItem, ListItemIcon, ListItemText } from '@mui/material';
import ScienceIcon from '@mui/icons-material/Science';
import CodeIcon from '@mui/icons-material/Code';
import DataObjectIcon from '@mui/icons-material/DataObject';
import VerifiedIcon from '@mui/icons-material/Verified';

const About = () => {
  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          About This Project
        </Typography>
        
        <Typography variant="body1" paragraph>
          The Plant Disease Detection System is a proof-of-concept AI-enabled application designed to help farmers and gardeners identify plant diseases quickly and accurately through image analysis.
        </Typography>
        
        <Divider sx={{ my: 3 }} />
        
        <Typography variant="h5" gutterBottom>
          How It Works
        </Typography>
        
        <List>
          <ListItem>
            <ListItemIcon>
              <ScienceIcon color="primary" />
            </ListItemIcon>
            <ListItemText 
              primary="Deep Learning Model" 
              secondary="We utilize a Convolutional Neural Network based on MobileNetV2 architecture, trained on thousands of plant disease images to achieve high accuracy in disease identification."
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <CodeIcon color="primary" />
            </ListItemIcon>
            <ListItemText 
              primary="Advanced Explainability" 
              secondary="Our system doesn't just tell you what disease your plant has - it shows you why it made that diagnosis using techniques like Grad-CAM and SHAP to highlight relevant areas."
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <DataObjectIcon color="primary" />
            </ListItemIcon>
            <ListItemText 
              primary="Comprehensive Recommendations" 
              secondary="For each detected disease, we provide detailed information about symptoms, treatment options, and prevention strategies based on agricultural best practices."
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <VerifiedIcon color="primary" />
            </ListItemIcon>
            <ListItemText 
              primary="Continuous Improvement" 
              secondary="Your feedback helps improve our system. Each time you confirm or correct a diagnosis, our model learns and gets better at identifying plant diseases."
            />
          </ListItem>
        </List>
        
        <Divider sx={{ my: 3 }} />
        
        <Typography variant="h5" gutterBottom>
          Technology Stack
        </Typography>
        
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mb: 3 }}>
          <Typography variant="body1">
            <strong>Frontend:</strong> React, Material-UI
          </Typography>
          <Typography variant="body1">
            <strong>Backend:</strong> Flask, TensorFlow
          </Typography>
          <Typography variant="body1">
            <strong>ML Model:</strong> MobileNetV2 (Transfer Learning)
          </Typography>
          <Typography variant="body1">
            <strong>Deployment:</strong> Docker, RESTful API
          </Typography>
        </Box>
        
        <Typography variant="body2" color="textSecondary" align="center" sx={{ mt: 4 }}>
          Created as a proof-of-concept for Software Engineering for AI-Enabled Systems
          <br />
          © 2025 Plant Disease Detection System
        </Typography>
      </Paper>
    </Container>
  );
};

export default About;