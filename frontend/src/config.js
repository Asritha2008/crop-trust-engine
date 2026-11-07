// Single place for API address and runtime configurations
// Rule: Never hard-code the API address; a missing environment variable should fail loudly, not fall back to localhost in production.

const rawApiUrl = import.meta.env.VITE_API_URL;

if (import.meta.env.PROD && (!rawApiUrl || rawApiUrl.trim() === '')) {
  throw new Error(
    'CRITICAL CONFIGURATION ERROR: VITE_API_URL is missing in production build. ' +
    'Set VITE_API_URL in your hosting provider environment variables and rebuild.'
  );
}

export const API_BASE_URL = (rawApiUrl && rawApiUrl.trim() !== '')
  ? rawApiUrl.replace(/\/+$/, '')
  : 'http://127.0.0.1:8000';

export const IS_PRODUCTION = import.meta.env.PROD;
export const DEFAULT_USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true';

// Frozen physical boundaries from the API contract (Section 7.5)
export const PHYSICAL_BOUNDS = {
  N: { min: 0, max: 200, step: 1, unit: 'kg/ha', label: 'Nitrogen (N)', typical: '60 – 120' },
  P: { min: 0, max: 200, step: 1, unit: 'kg/ha', label: 'Phosphorus (P)', typical: '35 – 60' },
  K: { min: 0, max: 300, step: 1, unit: 'kg/ha', label: 'Potassium (K)', typical: '35 – 50' },
  temperature: { min: -10, max: 60, step: 0.5, unit: '°C', label: 'Temperature', typical: '20 – 30' },
  humidity: { min: 0, max: 100, step: 1, unit: '%', label: 'Humidity', typical: '60 – 85' },
  ph: { min: 0, max: 14, step: 0.1, unit: '', label: 'Soil pH', typical: '5.5 – 7.5' },
  rainfall: { min: 0, max: 1000, step: 1, unit: 'mm', label: 'Rainfall', typical: '150 – 300' }
};
