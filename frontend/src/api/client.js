import { API_BASE_URL, DEFAULT_USE_MOCK, PHYSICAL_BOUNDS } from '../config';
import mockExamples from '../../mock/predict_examples.json';

// In-memory mock toggle, defaults to environment variable or false
let mockModeEnabled = DEFAULT_USE_MOCK;

export function isMockEnabled() {
  return mockModeEnabled;
}

export function setMockEnabled(val) {
  mockModeEnabled = Boolean(val);
}

export function getMockPresets() {
  return mockExamples;
}

/**
 * Validate input values against frozen physical boundaries.
 * Returns an object with field-level error messages if invalid.
 */
export function validatePhysicalBounds(values) {
  const errors = {};
  for (const [field, bounds] of Object.entries(PHYSICAL_BOUNDS)) {
    const val = values[field];
    if (val === undefined || val === null || val === '') {
      errors[field] = `${bounds.label || field} is required`;
    } else {
      const num = Number(val);
      if (isNaN(num)) {
        errors[field] = `${bounds.label || field} must be a number`;
      } else if (num < bounds.min || num > bounds.max) {
        errors[field] = `${bounds.label || field} must be between ${bounds.min} and ${bounds.max} ${bounds.unit}`.trim();
      }
    }
  }
  return errors;
}

/**
 * Fetch health status from live API or return mock health
 */
export async function fetchHealth(signal) {
  if (mockModeEnabled) {
    await new Promise((r) => setTimeout(r, 150));
    return {
      status: 'ok',
      mode: 'mock',
      message: 'Running against local contract mock',
      version: '2026-11-07-mock'
    };
  }

  const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET', signal });
  if (!res.ok) {
    throw new Error(`Health check failed: HTTP ${res.status} (${res.statusText})`);
  }
  return res.json();
}

/**
 * Send prediction request to either live backend or mock engine.
 * Supports AbortSignal for debouncing and request cancellation.
 */
export async function predictCrop(values, risk = 'balanced', signal = null) {
  // Validate physical bounds on client first to detect invalid values
  const clientValidation = validatePhysicalBounds(values);
  if (Object.keys(clientValidation).length > 0) {
    const err = new Error('Physical range validation failed');
    err.status = 422;
    err.fieldErrors = clientValidation;
    throw err;
  }

  if (mockModeEnabled) {
    return simulateMockPrediction(values, risk, signal);
  }

  try {
    const res = await fetch(`${API_BASE_URL}/predict?risk=${encodeURIComponent(risk)}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(values),
      signal
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      const err = new Error(errorData.detail?.toString() || `Request failed with HTTP ${res.status}`);
      err.status = res.status;

      // Extract field-level errors from FastAPI 422 format
      if (res.status === 422 && Array.isArray(errorData.detail)) {
        err.fieldErrors = {};
        errorData.detail.forEach((item) => {
          const field = item.loc ? item.loc[item.loc.length - 1] : 'general';
          err.fieldErrors[field] = item.msg;
        });
      }
      throw err;
    }

    return await res.json();
  } catch (e) {
    if (e.name === 'AbortError') throw e;

    // If live backend is not reachable and not in strict production, allow automatic fallback to mock with notice
    if (e.message.includes('Failed to fetch') || e.message.includes('NetworkError')) {
      const fallbackErr = new Error(
        `Cannot reach backend at ${API_BASE_URL}. Enable Mock Mode to test with contract fixtures.`
      );
      fallbackErr.status = 0;
      fallbackErr.canFallbackToMock = true;
      throw fallbackErr;
    }
    throw e;
  }
}

/**
 * Mock predictor implementing the exact three contract cases from Sections 6.2 and 7.5
 */
async function simulateMockPrediction(values, risk, signal) {
  // Simulate 200ms realistic network latency
  await new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, 200);
    if (signal) {
      signal.addEventListener('abort', () => {
        clearTimeout(timer);
        reject(new DOMException('Aborted', 'AbortError'));
      });
    }
  });

  const { humidity, rainfall, K } = values;

  // Case 1: Refusal (impossible combination: high humidity with very low rainfall)
  if (humidity >= 88 && rainfall <= 50) {
    const refusal = JSON.parse(JSON.stringify(mockExamples.refuse.response));
    refusal.explanation = `Humidity ${humidity}% and rainfall ${rainfall} mm are each common, but this combination does not appear in the training data. No recommendation is made.`;
    return refusal;
  }

  // Case 2: Caution (ambiguous / close competitors or low K)
  if ((rainfall < 185 && rainfall > 100) || K < 38 || risk === 'conservative') {
    const caution = JSON.parse(JSON.stringify(mockExamples.caution.response));
    if (risk === 'bold') {
      // Bold risk profile is more lenient, promotes top crop
      caution.verdict = 'recommend';
      caution.trust_score = 68;
    }
    return caution;
  }

  // Case 3: Recommend
  const recommend = JSON.parse(JSON.stringify(mockExamples.recommend.response));
  if (risk === 'bold') {
    recommend.trust_score = Math.min(100, recommend.trust_score + 5);
  }
  return recommend;
}

