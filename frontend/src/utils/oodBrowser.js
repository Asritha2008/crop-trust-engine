import oodBrowserData from '../data/ood_browser.json' with { type: 'json' };

const FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'];

/**
 * Checks whether inputs are complete and valid numbers.
 * @param {Object} inputs
 * @returns {boolean}
 */
export function areInputsComplete(inputs) {
  if (!inputs || typeof inputs !== 'object') return false;
  for (const f of FEATURES) {
    const val = inputs[f];
    if (val === null || val === undefined || val === '') return false;
    const num = Number(val);
    if (Number.isNaN(num)) return false;
  }
  return true;
}

/**
 * In-browser Mahalanobis distance calculation.
 * Formula from ood_browser.json note:
 * z = (x - scaler_mean) / scaler_scale
 * d2 = min over crops of (z - mean)^T P (z - mean)
 *
 * @param {Object} inputs - { N, P, K, temperature, humidity, ph, rainfall }
 * @param {Object} [data] - ood_browser.json data structure (defaults to imported JSON)
 * @returns {{ minD2: number, nearestCrop: string, cropDistances: Object }}
 */
export function computeMahalanobisDistance(inputs, data = oodBrowserData) {
  const { scaler_mean, scaler_scale, crops, class_mean, class_precision } = data;

  // 1. Standardize inputs to z
  const z = FEATURES.map((f, i) => {
    const x = Number(inputs[f]);
    return (x - scaler_mean[i]) / scaler_scale[i];
  });

  let minD2 = Infinity;
  let nearestCrop = null;
  const cropDistances = {};

  // 2. Compute (z - mean)^T P (z - mean) for each crop
  for (const crop of crops) {
    const mean = class_mean[crop];
    const P = class_precision[crop];

    // delta = z - mean
    const delta = new Array(7);
    for (let i = 0; i < 7; i++) {
      delta[i] = z[i] - mean[i];
    }

    // d2 = delta^T * P * delta
    let d2 = 0;
    for (let i = 0; i < 7; i++) {
      let pDeltaI = 0;
      const pRow = P[i];
      for (let j = 0; j < 7; j++) {
        pDeltaI += pRow[j] * delta[j];
      }
      d2 += delta[i] * pDeltaI;
    }

    cropDistances[crop] = d2;
    if (d2 < minD2) {
      minD2 = d2;
      nearestCrop = crop;
    }
  }

  return { minD2, nearestCrop, cropDistances };
}

/**
 * In-browser familiarity check.
 * Thresholds in ood_browser.json are in d^2 units.
 *
 * Status mapping:
 * - ok: "Familiar" (d2 <= caution)
 * - caution: "Atypical" (caution < d2 <= refuse)
 * - refuse: "Unusual combination" (d2 > refuse)
 * - incomplete: "Enter all 7 values"
 *
 * @param {Object} inputs
 * @param {string} [riskProfile='balanced']
 * @param {Object} [data]
 * @returns {{
 *   status: 'ok' | 'caution' | 'refuse' | 'incomplete',
 *   label: string,
 *   icon: string,
 *   distance: number | null,
 *   nearestCrop: string | null,
 *   thresholds: { caution: number, refuse: number } | null
 * }}
 */
export function checkFamiliarity(inputs, riskProfile = 'balanced', data = oodBrowserData) {
  if (!areInputsComplete(inputs)) {
    return {
      status: 'incomplete',
      label: 'Enter all 7 values',
      icon: 'ℹ️',
      distance: null,
      nearestCrop: null,
      thresholds: null
    };
  }

  const { minD2, nearestCrop } = computeMahalanobisDistance(inputs, data);
  const thresholds = (data.thresholds && data.thresholds[riskProfile]) || {
    caution: 10.78,
    refuse: 12.92
  };

  let status = 'ok';
  let label = 'Familiar';
  let icon = '✅';

  if (minD2 > thresholds.refuse) {
    status = 'refuse';
    label = 'Unusual combination';
    icon = '🛑';
  } else if (minD2 > thresholds.caution) {
    status = 'caution';
    label = 'Atypical';
    icon = '⚠️';
  } else {
    status = 'ok';
    label = 'Familiar';
    icon = '✅';
  }

  return {
    status,
    label,
    icon,
    distance: minD2,
    nearestCrop,
    thresholds
  };
}

