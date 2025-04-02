/**
 * Configuration file for Finance RAG Frontend
 * This file manages environment-specific configuration 
 */

// Base API URL - Will be replaced during build for production
const API_URL = process.env.REACT_APP_API_URL || process.env.API_URL || 'http://localhost:8000';

// Feature flags
const FEATURES = {
  enableFeedback: true,
  enableFileUpload: true,
  enableAuthentication: false,
  enableVisualization: true,
  enableCaching: true,
};

// Timeouts and limits
const TIMEOUTS = {
  apiRequestTimeout: 30000, // 30 seconds
  queryThrottleMs: 500,     // Debounce time for queries
  maxFileSize: 5 * 1024 * 1024, // 5MB in bytes
};

// UI Configuration
const UI_CONFIG = {
  theme: 'light',           // 'light' or 'dark'
  primaryColor: '#2563eb',  // Primary brand color
  defaultView: 'chat',      // Default view on app load
  maxQueryHistory: 50,      // Maximum query history to store
};

// Export configuration
export default {
  API_URL,
  FEATURES,
  TIMEOUTS,
  UI_CONFIG,
}; 