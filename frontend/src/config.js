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
  // Typical ranges and units:
  // PLACEHOLDER: units unconfirmed, ask Role 1 (class_stats.json / data plots)
  N: { min: 0, max: 200, step: 1, unit: '', label: 'Nitrogen (N)', typical: '60 – 120' }, // PLACEHOLDER: units unconfirmed, ask Role 1
  P: { min: 0, max: 200, step: 1, unit: '', label: 'Phosphorus (P)', typical: '35 – 60' }, // PLACEHOLDER: units unconfirmed, ask Role 1
  K: { min: 0, max: 300, step: 1, unit: '', label: 'Potassium (K)', typical: '35 – 50' }, // PLACEHOLDER: units unconfirmed, ask Role 1
  temperature: { min: -10, max: 60, step: 0.5, unit: '°C', label: 'Temperature', typical: '20 – 30' },
  humidity: { min: 0, max: 100, step: 1, unit: '%', label: 'Humidity', typical: '60 – 85' },
  ph: { min: 0, max: 14, step: 0.1, unit: '', label: 'Soil pH', typical: '5.5 – 7.5' },
  rainfall: { min: 0, max: 1000, step: 1, unit: '', label: 'Rainfall', typical: '150 – 300' } // PLACEHOLDER: units unconfirmed, ask Role 1
};

// Demo preset inputs for evaluation and viva demo
// PLACEHOLDER: replace with value from Role 1 (class_stats.json / data plots)
export const DEMO_PRESET_INPUTS = {
  // PLACEHOLDER: replace with value from Role 1 (class_stats.json / data plots)
  recommend: {
    N: 90, P: 42, K: 43, temperature: 21.0, humidity: 82.0, ph: 6.5, rainfall: 203.0
  },
  // PLACEHOLDER: replace with value from Role 1 (class_stats.json / data plots)
  caution: {
    N: 80, P: 40, K: 35, temperature: 24.5, humidity: 80.0, ph: 6.7, rainfall: 175.0
  },
  // PLACEHOLDER: replace with value from Role 1 (class_stats.json / data plots)
  refuse: {
    N: 90, P: 42, K: 43, temperature: 25.0, humidity: 92.0, ph: 6.5, rainfall: 25.0
  }
};

/**
 * Clamps numeric values strictly inside physical bounds
 */
export function clampValue(field, value) {
  const bounds = PHYSICAL_BOUNDS[field];
  if (!bounds) return Number(value);
  const num = Number(value);
  if (isNaN(num)) return bounds.min;
  return Math.min(bounds.max, Math.max(bounds.min, num));
}

// Climate stress test scenario shifts
// PLACEHOLDER: replace with value from Role 1 (sensible shifts should be confirmed with Role 1's data / EDA plots)
export const CLIMATE_SCENARIOS = [
  {
    id: 'warmer',
    name: 'Warmer (+2 °C)',
    // PLACEHOLDER: replace with value from Role 1 (confirm temperature delta with data)
    deltaTemp: 2.0,
    rainfallFactor: 1.0,
    description: '+2 °C temperature increase'
  },
  {
    id: 'drier',
    name: 'Drier (rainfall -20%)',
    // PLACEHOLDER: replace with value from Role 1 (confirm rainfall reduction with data)
    deltaTemp: 0.0,
    rainfallFactor: 0.8,
    description: '-20% rainfall reduction'
  },
  {
    id: 'hotter_drier',
    name: 'Hotter and drier (+2 °C, rainfall -20%)',
    // PLACEHOLDER: replace with value from Role 1 (confirm combined stress shifts with data)
    deltaTemp: 2.0,
    rainfallFactor: 0.8,
    description: '+2 °C warming and -20% rainfall reduction'
  }
];
