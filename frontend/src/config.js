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

// Batch assessment limits (frozen by API contract / specification)
export const BATCH_MAX_ROWS = 50;
export const BATCH_MAX_BYTES = 1000000;

import classStatsData from './data/class_stats.json';

const classStats = classStatsData.class_stats || classStatsData;

/**
 * Computes overall typical range [min(p05), max(p95)] across all crops from class_stats.json
 */
export function computeTypicalRange(feature) {
  let minP05 = Infinity;
  let maxP95 = -Infinity;
  for (const crop of Object.keys(classStats)) {
    const s = classStats[crop]?.[feature];
    if (s) {
      if (typeof s.p05 === 'number' && s.p05 < minP05) minP05 = s.p05;
      if (typeof s.p95 === 'number' && s.p95 > maxP95) maxP95 = s.p95;
    }
  }
  if (!isFinite(minP05) || !isFinite(maxP95)) return '';
  const fmt = (v) => (Number.isInteger(v) ? `${v}` : (Math.round(v * 10) / 10).toString());
  return `${fmt(minP05)} – ${fmt(maxP95)}`;
}

// Frozen physical boundaries from the API contract (Section 7.5)
export const PHYSICAL_BOUNDS = {
  N: { min: 0, max: 200, step: 1, unit: '', label: 'Nitrogen (N)', typical: computeTypicalRange('N') },
  P: { min: 0, max: 200, step: 1, unit: '', label: 'Phosphorus (P)', typical: computeTypicalRange('P') },
  K: { min: 0, max: 300, step: 1, unit: '', label: 'Potassium (K)', typical: computeTypicalRange('K') },
  temperature: { min: -10, max: 60, step: 0.5, unit: '°C', label: 'Temperature', typical: computeTypicalRange('temperature') },
  humidity: { min: 0, max: 100, step: 1, unit: '%', label: 'Humidity', typical: computeTypicalRange('humidity') },
  ph: { min: 0, max: 14, step: 0.1, unit: '', label: 'Soil pH', typical: computeTypicalRange('ph') },
  rainfall: { min: 0, max: 1000, step: 1, unit: 'mm', label: 'Rainfall', typical: computeTypicalRange('rainfall') }
};

// Demo preset inputs for evaluation and viva demo
// PLACEHOLDER: replace with value from Role 1 (class_stats.json / data plots)
export const DEMO_PRESET_INPUTS = {
  // Validation-set row 4 (rice) from Role 2 (Varshith), engine-verified: recommend, stability 0.995, trust 100. A validation example, not an agronomic recommendation.
  recommend: {
    N: 78, P: 42, K: 42, temperature: 20.13017482, humidity: 81.60487287, ph: 7.628472891, rainfall: 262.7173405
  },
  // Row 2049 from Role 1 (Abhishek): true crop jute, model rice 0.5269 / jute 0.4731. NOT confirmed as Caution until the real Role 2 engine runs it.
  caution: {
    N: 100, P: 56, K: 40, temperature: 26.38905406, humidity: 83.31240346, ph: 7.433313409, rainfall: 176.1516409
  },
  // PLACEHOLDER: replace with a real impossible pair from Role 1 (candidate N=112 with K=185; full row not confirmed yet)
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
