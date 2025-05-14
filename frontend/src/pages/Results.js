// pages/Results.js
import React from 'react';
import { useHistory } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  Button,
  Grid,
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Tab,
  Tabs
} from '@mui/material';
import NatureIcon from '@mui/icons-material/Nature';
import WarningIcon from '@mui/icons-material/Warning';
import HealingIcon from '@mui/icons-material/Healing';
import VerifiedIcon from '@mui/icons-material/Verified';
import FeedbackIcon from '@mui/icons-material/Feedback';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ScienceIcon from '@mui/icons-material/Science';
import BugReportIcon from '@mui/icons-material/BugReport';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

const Results = ({ results, image }) => {
  const [tabValue, setTabValue] = React.useState(0);
  const history = useHistory();

  // If no results, redirect to upload page
  React.useEffect(() => {
    if (!results) {
      history.push('/upload');
    }
  }, [results, history]);

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Navigate to feedback page
  const handleProvideFeedback = () => {
    history.push('/feedback');
  };

  // Return to upload page
  const handleBackToUpload = () => {
    history.push('/upload');
  };

  // If no results, show loading or placeholder
  if (!results) {
    return (
      <Container maxWidth="md" className="results-container">
        <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h5">Loading results...</Typography>
        </Paper>
      </Container>
    );
  }

  // Get the top prediction
  const topPrediction = results.prediction[0];
  const diseaseInfo = results.disease_info;
  const gradCam = results.grad_cam;

  // Calculate confidence level styling
  const getConfidenceColor = (probability) => {
    if (probability > 0.8) return 'success';
    if (probability > 0.5) return 'warning';
    return 'error';
  };

  return (
    <Container maxWidth="lg" className="results-container">
      <Paper elevation={3} sx={{ p: 4, mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Analysis Results
        </Typography>

        <Grid container spacing={4}>
          {/* Left column: Image and visualization */}
          <Grid item xs={12} md={6}>
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Your Plant Image
              </Typography>
              <Box sx={{ border: '1px solid #eee', borderRadius: 1, overflow: 'hidden' }}>
                <img 
                  src={image} 
                  alt="Uploaded plant" 
                  style={{ width: '100%', maxHeight: '300px', objectFit: 'contain' }} 
                />
              </Box>
            </Box>

            {gradCam && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Areas of Interest
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  The highlighted areas show regions the AI focused on for diagnosis.
                </Typography>
                <Box sx={{ border: '1px solid #eee', borderRadius: 1, overflow: 'hidden' }}>
                  <img 
                    src={`data:image/png;base64,${gradCam.superimposed}`} 
                    alt="Analysis visualization" 
                    style={{ width: '100%', maxHeight: '300px', objectFit: 'contain' }} 
                  />
                </Box>
              </Box>
            )}

            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
              <Button 
                variant="outlined" 
                startIcon={<ArrowBackIcon />}
                onClick={handleBackToUpload}
              >
                Analyze Another Image
              </Button>
              <Button 
                variant="contained" 
                color="primary"
                startIcon={<FeedbackIcon />}
                onClick={handleProvideFeedback}
              >
                Provide Feedback
              </Button>
            </Box>
          </Grid>

          {/* Right column: Diagnosis and details */}
          <Grid item xs={12} md={6}>
            <Card 
              variant="outlined" 
              sx={{ 
                mb: 3, 
                borderColor: getConfidenceColor(topPrediction.probability) === 'success' ? 'success.main' : 'warning.main',
                boxShadow: 2
              }}
            >
              <CardContent>
                <Typography variant="h5" component="h2" gutterBottom>
                  Diagnosis
                </Typography>
                
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ mr: 2 }}>
                    {topPrediction.class.replace('_', ' ')}
                  </Typography>
                  <Chip 
                    label={topPrediction.confidence} 
                    color={getConfidenceColor(topPrediction.probability)}
                    icon={<VerifiedIcon />}
                  />
                </Box>
                
                <Typography variant="body1" paragraph>
                  {diseaseInfo.description}
                </Typography>

                {/* Alternative diagnoses */}
                {results.prediction.length > 1 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Alternative possibilities:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                      {results.prediction.slice(1, 3).map((pred, idx) => (
                        <Chip 
                          key={idx}
                          label={`${pred.class.replace('_', ' ')} (${pred.confidence})`}
                          variant="outlined"
                          size="small"
                        />
                      ))}
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>

            <Box sx={{ width: '100%', mb: 3 }}>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={tabValue} onChange={handleTabChange} aria-label="disease information tabs">
                  <Tab icon={<HealingIcon />} label="Treatment" id="tab-0" />
                  <Tab icon={<WarningIcon />} label="Prevention" id="tab-1" />
                  <Tab icon={<ScienceIcon />} label="More Info" id="tab-2" />
                </Tabs>
              </Box>
              
              {/* Treatment Tab */}
              <TabPanel value={tabValue} index={0}>
                <Typography variant="h6" gutterBottom>
                  Recommended Treatment
                </Typography>
                <Typography variant="body1" paragraph>
                  {diseaseInfo.treatment}
                </Typography>
                
                <Card variant="outlined" sx={{ mt: 2, bgcolor: 'background.default' }}>
                  <CardContent>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Important Note:
                    </Typography>
                    <Typography variant="body2">
                      These recommendations are for informational purposes only. 
                      For serious plant health issues, consult with a professional 
                      agriculturist or plant pathologist.
                    </Typography>
                  </CardContent>
                </Card>
              </TabPanel>
              
              {/* Prevention Tab */}
              <TabPanel value={tabValue} index={1}>
                <Typography variant="h6" gutterBottom>
                  Prevention Tips
                </Typography>
                <List>
                  {diseaseInfo.prevention.map((tip, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                      <NatureIcon color="primary" />
                      </ListItemIcon>
                      <ListItemText primary={tip} />
                    </ListItem>
                  ))}
                </List>
              </TabPanel>
              
              {/* More Info Tab */}
              <TabPanel value={tabValue} index={2}>
                <Typography variant="h6" gutterBottom>
                  Disease Characteristics
                </Typography>
                <Typography variant="body1" paragraph>
                  {topPrediction.class.replace('_', ' ')} typically affects plants in the following ways:
                </Typography>
                
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <BugReportIcon color="warning" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Symptoms" 
                      secondary={getSymptoms(topPrediction.class)} 
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <ErrorOutlineIcon color="error" />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Risk Factors" 
                      secondary={getRiskFactors(topPrediction.class)} 
                    />
                  </ListItem>
                </List>
              </TabPanel>
            </Box>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
};

// Tab Panel component
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`disease-tabpanel-${index}`}
      aria-labelledby={`disease-tab-${index}`}
      {...other}
      style={{ padding: '16px 0' }}
    >
      {value === index && (
        <Box>
          {children}
        </Box>
      )}
    </div>
  );
}

// Helper functions for additional disease information
function getSymptoms(diseaseName) {
  const symptoms = {
    'Healthy': 'No visible symptoms. Leaves have normal coloration and structure.',
    'Bacterial_spot': 'Small, water-soaked spots that enlarge and turn dark brown with a yellow halo.',
    'Early_blight': 'Dark brown spots with concentric rings, often starting on older leaves.',
    'Late_blight': 'Water-soaked pale green to brown spots with white fungal growth in humid conditions.',
    'Leaf_rust': 'Orange-brown pustules primarily on the undersides of leaves.',
    'Powdery_mildew': 'White powdery spots on upper leaf surfaces that spread to cover the entire leaf.',
    'Leaf_spot': 'Circular spots with defined margins that may have gray centers and dark borders.'
  };
  
  return symptoms[diseaseName] || 'Specific symptoms information not available.';
}

function getRiskFactors(diseaseName) {
  const risks = {
    'Healthy': 'No risks identified.',
    'Bacterial_spot': 'High humidity, warm temperatures, and overhead irrigation promote bacterial spread.',
    'Early_blight': 'Warm, humid conditions and poor air circulation increase risk.',
    'Late_blight': 'Cool, wet weather (especially night temperatures of 50-60°F and high humidity).',
    'Leaf_rust': 'High humidity, moderate temperatures, and extended leaf wetness.',
    'Powdery_mildew': 'Dry conditions with high humidity, moderate temperatures, and shade.',
    'Leaf_spot': 'Warm, wet conditions and poor sanitation of plant debris.'
  };
  
  return risks[diseaseName] || 'Risk factor information not available.';
}

export default Results;