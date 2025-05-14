// pages/Feedback.js
import React, { useState } from 'react';
import { useHistory } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  Button,
  Grid,
  Box,
  FormControl,
  FormControlLabel,
  FormLabel,
  Radio,
  RadioGroup,
  TextField,
  MenuItem,
  Snackbar,
  Alert,
  Rating,
  Stack
} from '@mui/material';
import ThumbUpIcon from '@mui/icons-material/ThumbUp';
import ThumbDownIcon from '@mui/icons-material/ThumbDown';
import SendIcon from '@mui/icons-material/Send';
import { submitFeedback } from '../services/Api';

const Feedback = ({ results, image }) => {
  const history = useHistory();
  const [isCorrect, setIsCorrect] = useState(null);
  const [userCorrection, setUserCorrection] = useState('');
  const [comments, setComments] = useState('');
  const [helpfulRating, setHelpfulRating] = useState(4);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');

  // If no results, redirect to upload page
  React.useEffect(() => {
    if (!results) {
      history.push('/upload');
    }
  }, [results, history]);

  // Handle diagnostic accuracy feedback
  const handleAccuracyChange = (event) => {
    setIsCorrect(event.target.value === 'true');
  };

  // Handle user correction selection
  const handleCorrectionChange = (event) => {
    setUserCorrection(event.target.value);
  };

  // Handle comments change
  const handleCommentsChange = (event) => {
    setComments(event.target.value);
  };

  // Handle helpfulness rating change
  const handleRatingChange = (event, newValue) => {
    setHelpfulRating(newValue);
  };

  // Submit feedback
  const handleSubmit = async () => {
    if (isCorrect === null) {
      setError('Please indicate if the diagnosis was correct');
      return;
    }

    if (!isCorrect && !userCorrection) {
      setError('Please select the correct diagnosis');
      return;
    }

    setLoading(true);
    setError('');

    // Prepare feedback data
    const feedbackData = {
      image_id: new Date().getTime().toString(), // Using timestamp as a unique ID
      prediction: results.prediction[0].class,
      correct: isCorrect,
      user_correction: isCorrect ? '' : userCorrection,
      comments: comments,
      helpfulness_rating: helpfulRating
    };

    try {
      // Submit feedback to API
      const response = await submitFeedback(feedbackData);
      
      if (response.status === 'success') {
        setSuccess(true);
        // Reset form fields
        setIsCorrect(null);
        setUserCorrection('');
        setComments('');
        setHelpfulRating(4);
        
        // Navigate back to home after a delay
        setTimeout(() => {
          history.push('/');
        }, 3000);
      } else {
        setError(response.message || 'Failed to submit feedback');
      }
    } catch (err) {
      setError('Error submitting feedback: ' + (err.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  // Close error alert
  const handleCloseError = () => {
    setError('');
  };

  // Close success alert
  const handleCloseSuccess = () => {
    setSuccess(false);
  };

  // If no results, show loading placeholder
  if (!results) {
    return (
      <Container maxWidth="md">
        <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h5">Loading...</Typography>
        </Paper>
      </Container>
    );
  }

  // Get the prediction information
  const prediction = results.prediction[0];
  const predictionClass = prediction.class.replace('_', ' ');

  // Get list of all possible classes for the correction dropdown
  const allClasses = Object.keys(
    results.prediction.reduce((acc, p) => {
      acc[p.class] = true;
      return acc;
    }, {})
  );

  // Add some additional common disease classes that might not be in the top predictions
  const additionalClasses = [
    'Healthy',
    'Early_blight',
    'Late_blight',
    'Bacterial_spot',
    'Leaf_rust',
    'Powdery_mildew',
    'Leaf_spot',
    'Mosaic_virus',
    'Anthracnose',
    'Downy_mildew'
  ];

  // Combine and deduplicate classes
  const allPossibleClasses = [...new Set([...allClasses, ...additionalClasses])];

  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ p: 4, mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Provide Feedback
        </Typography>
        
        <Typography variant="body1" paragraph>
          Your feedback helps improve our plant disease detection system. Please let us know if the diagnosis was accurate and provide any additional information.
        </Typography>
        
        <Grid container spacing={4}>
          {/* Left column: Image and prediction */}
          <Grid item xs={12} md={4}>
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Your Plant Image
              </Typography>
              <Box sx={{ border: '1px solid #eee', borderRadius: 1, overflow: 'hidden' }}>
                <img 
                  src={image} 
                  alt="Uploaded plant" 
                  style={{ width: '100%', maxHeight: '200px', objectFit: 'contain' }} 
                />
              </Box>
            </Box>
            
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                AI Diagnosis
              </Typography>
              <Typography variant="body1" fontWeight="medium">
                {predictionClass}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Confidence: {prediction.confidence}
              </Typography>
            </Box>
          </Grid>
          
          {/* Right column: Feedback form */}
          <Grid item xs={12} md={8}>
            <Box component="form" sx={{ mt: 1 }}>
              {/* Was the diagnosis correct? */}
              <FormControl component="fieldset" sx={{ mb: 3 }}>
                <FormLabel component="legend">Was this diagnosis correct?</FormLabel>
                <RadioGroup
                  row
                  name="diagnosis-accuracy"
                  value={isCorrect === null ? '' : isCorrect.toString()}
                  onChange={handleAccuracyChange}
                >
                  <FormControlLabel 
                    value="true" 
                    control={<Radio />} 
                    label="Yes" 
                    icon={<ThumbUpIcon />}
                  />
                  <FormControlLabel 
                    value="false" 
                    control={<Radio />} 
                    label="No" 
                  />
                </RadioGroup>
              </FormControl>
              
              {/* If incorrect, what is the correct diagnosis? */}
              {isCorrect === false && (
                <FormControl fullWidth sx={{ mb: 3 }}>
                  <FormLabel component="legend">What is the correct diagnosis?</FormLabel>
                  <TextField
                    select
                    value={userCorrection}
                    onChange={handleCorrectionChange}
                    variant="outlined"
                    fullWidth
                    margin="normal"
                  >
                    <MenuItem value="">Select the correct disease</MenuItem>
                    {allPossibleClasses.map((cls) => (
                      <MenuItem key={cls} value={cls}>
                        {cls.replace('_', ' ')}
                      </MenuItem>
                    ))}
                    <MenuItem value="Other">
                      Other (please specify in comments)
                    </MenuItem>
                  </TextField>
                </FormControl>
              )}
              
              {/* How helpful was the information? */}
              <FormControl component="fieldset" sx={{ mb: 3 }}>
                <FormLabel component="legend">How helpful was the information provided?</FormLabel>
                <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 1 }}>
                  <Typography variant="body2">Not helpful</Typography>
                  <Rating
                    name="helpfulness"
                    value={helpfulRating}
                    onChange={handleRatingChange}
                  />
                  <Typography variant="body2">Very helpful</Typography>
                </Stack>
              </FormControl>
              
              {/* Additional comments */}
              <FormControl fullWidth sx={{ mb: 4 }}>
                <FormLabel component="legend">Additional Comments (Optional)</FormLabel>
                <TextField
                  value={comments}
                  onChange={handleCommentsChange}
                  multiline
                  rows={4}
                  variant="outlined"
                  fullWidth
                  margin="normal"
                  placeholder="Share any additional observations or suggestions..."
                />
              </FormControl>
              
              {/* Submit button */}
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Button 
                  variant="outlined" 
                  onClick={() => history.push('/results')}
                >
                  Back to Results
                </Button>
                <Button 
                  variant="contained" 
                  color="primary"
                  startIcon={<SendIcon />}
                  onClick={handleSubmit}
                  disabled={loading}
                >
                  {loading ? 'Submitting...' : 'Submit Feedback'}
                </Button>
              </Box>
            </Box>
          </Grid>
        </Grid>
      </Paper>
      
      {/* Success message */}
      <Snackbar open={success} autoHideDuration={6000} onClose={handleCloseSuccess}>
        <Alert onClose={handleCloseSuccess} severity="success" sx={{ width: '100%' }}>
          Thank you for your feedback! It will help improve our system.
        </Alert>
      </Snackbar>
      
      {/* Error message */}
      <Snackbar open={!!error} autoHideDuration={6000} onClose={handleCloseError}>
        <Alert onClose={handleCloseError} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default Feedback;