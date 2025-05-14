// App.js - Main Application Component
import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import NavBar from './components/NavBar';
import Home from './pages/Home';
import UploadImage from './pages/UploadImage';
import Results from './pages/Results';
import Feedback from './pages/Feedback';
import About from './pages/About';
import './App.css';

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#2e7d32', // Green color for agriculture theme
    },
    secondary: {
      main: '#ff8f00', // Orange accent
    },
    background: {
      default: '#f8f8f8',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 500,
    },
  },
});

function App() {
  const [analysisResults, setAnalysisResults] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);

  // Handle the image analysis results
  const handleAnalysisComplete = (results, image) => {
    setAnalysisResults(results);
    setSelectedImage(image);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <div className="App">
          <NavBar />
          <main className="main-content">
            <Switch>
              <Route exact path="/" component={Home} />
              <Route 
                path="/upload" 
                render={(props) => (
                  <UploadImage 
                    {...props} 
                    onAnalysisComplete={handleAnalysisComplete} 
                  />
                )} 
              />
              <Route 
                path="/results" 
                render={(props) => (
                  <Results 
                    {...props} 
                    results={analysisResults}
                    image={selectedImage}
                  />
                )} 
              />
              <Route 
                path="/feedback" 
                render={(props) => (
                  <Feedback 
                    {...props} 
                    results={analysisResults}
                    image={selectedImage}
                  />
                )} 
              />
              <Route path="/about" component={About} />
            </Switch>
          </main>
          <footer className="footer">
            <p>© 2025 Plant Disease Detection System</p>
          </footer>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;