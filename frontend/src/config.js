// Single place for API address (baked in at build time by Vite)
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

