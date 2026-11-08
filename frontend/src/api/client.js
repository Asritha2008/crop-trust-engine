import { API_BASE_URL, DEFAULT_USE_MOCK, PHYSICAL_BOUNDS, DEMO_PRESET_INPUTS } from '../config';
import mockExamples from '../../mock/predict_examples.json';

// In-memory mock toggle, defaults to environment variable or false
let mockModeEnabled = DEFAULT_USE_MOCK;

export function isMockEnabled() {
  return mockModeEnabled;
}

export function setMockEnabled(val) {
  mockModeEnabled = Boolean(val);
}

// Build presets from DEMO_PRESET_INPUTS and mockExamples response objects
// PLACEHOLDER: replace with value from Role 1 (class_stats.json / data plots)
export function getMockPresets() {
  const presets = {};
  ['recommend', 'caution', 'refuse'].forEach((key) => {
    presets[key] = {
      input: DEMO_PRESET_INPUTS[key],
      response: mockExamples[key]
    };
  });
  return presets;
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
        const unitSuffix = bounds.unit ? ` ${bounds.unit}` : '';
        errors[field] = `${bounds.label || field} must be between ${bounds.min} and ${bounds.max}${unitSuffix}`;
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

      let message = `Request failed with HTTP ${res.status}`;
      if (typeof errorData.error === 'string') {
        message = errorData.error;
      } else if (typeof errorData.detail === 'string') {
        message = errorData.detail;
      }

      const err = new Error(message);
      err.status = res.status;

      if (res.status === 422) {
        err.fieldErrors = {};
        if (Array.isArray(errorData.errors)) {
          errorData.errors.forEach((e) => {
            if (e && e.field) {
              err.fieldErrors[e.field] = e.message;
            }
          });
        } else if (Array.isArray(errorData.detail)) {
          errorData.detail.forEach((item) => {
            const field = item.loc ? item.loc[item.loc.length - 1] : 'general';
            err.fieldErrors[field] = item.msg;
          });
        }

        if (Object.keys(err.fieldErrors).length > 0) {
          err.message = 'Please fix the highlighted fields';
        }
      }

      throw err;
    }

    return await res.json();
  } catch (e) {
    if (e.name === 'AbortError') throw e;

    // Network failure (TypeError in fetch): wrap with clear message and mock fallback capability
    if (
      e instanceof TypeError ||
      e.name === 'TypeError' ||
      e.message?.includes('Failed to fetch') ||
      e.message?.includes('NetworkError')
    ) {
      const fallbackErr = new Error(
        `Cannot reach backend at ${API_BASE_URL}. Enable Mock Mode to test with contract fixtures.`
      );
      fallbackErr.status = 0;
      fallbackErr.isNetwork = true;
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
    const refusal = JSON.parse(JSON.stringify(mockExamples.refuse));
    refusal.explanation = `Humidity ${humidity}% and rainfall ${rainfall} are each common, but this combination does not appear in the training data. No recommendation is made.`;
    return refusal;
  }

  // Case 2: Caution (ambiguous / close competitors or low K)
  if ((rainfall < 185 && rainfall > 100) || K < 38 || risk === 'conservative') {
    const caution = JSON.parse(JSON.stringify(mockExamples.caution));
    if (risk === 'bold') {
      // Bold risk profile is more lenient, promotes top crop
      caution.verdict = 'recommend';
      caution.trust_score = 68;
    }
    return caution;
  }

  // Case 3: Recommend
  const recommend = JSON.parse(JSON.stringify(mockExamples.recommend));
  if (risk === 'bold') {
    recommend.trust_score = Math.min(100, recommend.trust_score + 5);
  }
  return recommend;
}

/**
 * Send batch CSV assessment request to live backend or mock engine.
 * Supports AbortSignal for request cancellation.
 */
export async function predictBatch(file, risk = 'balanced', signal = null) {
  if (mockModeEnabled) {
    return simulateMockBatch(signal);
  }

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE_URL}/predict/batch?risk=${encodeURIComponent(risk)}`, {
      method: 'POST',
      body: formData,
      signal
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));

      let message = `Request failed with HTTP ${res.status}`;
      if (typeof errorData.error === 'string') {
        message = errorData.error;
      } else if (typeof errorData.detail === 'string') {
        message = errorData.detail;
      }

      const err = new Error(message);
      err.status = res.status;
      throw err;
    }

    return await res.json();
  } catch (e) {
    if (e.name === 'AbortError') throw e;

    if (
      e instanceof TypeError ||
      e.name === 'TypeError' ||
      e.message?.includes('Failed to fetch') ||
      e.message?.includes('NetworkError')
    ) {
      const fallbackErr = new Error(
        `Cannot reach backend at ${API_BASE_URL}. Enable Mock Mode to test with contract fixtures.`
      );
      fallbackErr.status = 0;
      fallbackErr.isNetwork = true;
      fallbackErr.canFallbackToMock = true;
      throw fallbackErr;
    }
    throw e;
  }
}

/**
 * Mock batch processor returning a simulated 4-row result (3 valid verdicts + 1 error row)
 */
async function simulateMockBatch(signal) {
  // Simulate 350ms processing latency
  await new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, 350);
    if (signal) {
      signal.addEventListener('abort', () => {
        clearTimeout(timer);
        reject(new DOMException('Aborted', 'AbortError'));
      });
    }
  });

  const results = [
    { ...JSON.parse(JSON.stringify(mockExamples.recommend)), row_index: 0 },
    { ...JSON.parse(JSON.stringify(mockExamples.caution)), row_index: 1 },
    { ...JSON.parse(JSON.stringify(mockExamples.refuse)), row_index: 2 },
    { row_index: 3, error: 'Invalid physical bounds: ph must be between 0 and 14' }
  ];

  return {
    model_version: 'mock-batch-v1',
    count: results.length,
    results
  };
}


