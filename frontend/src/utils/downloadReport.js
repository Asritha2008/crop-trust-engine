import { PHYSICAL_BOUNDS } from '../config';

/**
 * Generates and triggers download of a formatted crop recommendation report
 */
export function downloadReport(inputs, prediction, riskProfile) {
  if (!prediction) return;

  const timestamp = new Date().toISOString().replace('T', ' ').substring(0, 19);
  const verdictUpper = (prediction.verdict || 'UNKNOWN').toUpperCase();

  const formatWithUnit = (field, val) => {
    const b = PHYSICAL_BOUNDS[field];
    const unitPart = b && b.unit ? ` ${b.unit}` : '';
    return `${val}${unitPart}`;
  };

  const lines = [
    '========================================================================',
    '                   CROP TRUST ENGINE - ASSESSMENT REPORT                ',
    '========================================================================',
    `Generated at: ${timestamp}`,
    `Model Version: ${prediction.model_version || 'N/A'}`,
    `Risk Appetite Profile: ${riskProfile.toUpperCase()}`,
    '',
    '------------------------------------------------------------------------',
    '1. SYSTEM VERDICT',
    '------------------------------------------------------------------------',
    `Verdict:      [ ${verdictUpper} ]`,
    `Trust Score:  ${prediction.trust_score ?? 'N/A'} / 100`,
    `Stability:    ${prediction.stability !== null && prediction.stability !== undefined ? `${Math.round(prediction.stability * 100)}%` : 'N/A (Refused)'}`,
    `Unusual combination: ${prediction.ood?.flagged ? 'YES' : 'NO'}`,
    `How unusual (percentile): ${prediction.ood?.percentile !== undefined ? `${prediction.ood.percentile}%` : 'N/A'}`,
    '',
    '------------------------------------------------------------------------',
    '2. FIELD & SOIL INPUTS',
    '------------------------------------------------------------------------',
    `Nitrogen (N):     ${formatWithUnit('N', inputs.N)}`,
    `Phosphorus (P):   ${formatWithUnit('P', inputs.P)}`,
    `Potassium (K):    ${formatWithUnit('K', inputs.K)}`,
    `Temperature:      ${formatWithUnit('temperature', inputs.temperature)}`,
    `Humidity:         ${formatWithUnit('humidity', inputs.humidity)}`,
    `Soil pH:          ${inputs.ph}`,
    `Rainfall:         ${formatWithUnit('rainfall', inputs.rainfall)}`,
    '',
    '------------------------------------------------------------------------',
    '3. CALIBRATED CROP PROBABILITIES',
    '------------------------------------------------------------------------'
  ];

  if (prediction.top_crops && prediction.top_crops.length > 0) {
    prediction.top_crops.forEach((crop, idx) => {
      const probPct = `${Math.round(crop.probability * 100)}%`.padStart(4, ' ');
      lines.push(`  ${idx + 1}. ${crop.crop.padEnd(15, ' ')} : ${probPct}`);
    });
  } else {
    lines.push('  No crops recommended because the inputs are an unusual combination.');
  }

  if (prediction.conformal_set && prediction.conformal_set.length > 0) {
    lines.push(`  Possible crops: [ ${prediction.conformal_set.join(', ')} ]`);
  }

  lines.push('');
  lines.push('------------------------------------------------------------------------');
  lines.push('4. EXPLANATION & ACTIONABLE GUIDANCE');
  lines.push('------------------------------------------------------------------------');
  lines.push(`Why: ${prediction.verdict === 'refuse' ? 'The engine refuses this input.' : prediction.explanation}`);

  if (prediction.measure_next) {
    lines.push('');
    lines.push(`What to measure next: ${prediction.measure_next}`);
  }



  lines.push('');
  lines.push('========================================================================');
  lines.push('DISCLAIMER:');
  lines.push('This report is for decision support only and does not constitute agronomic');
  lines.push('advice. Predictions reflect Kaggle dataset distributions and selected');
  lines.push('sensor noise models.');
  lines.push('========================================================================');

  const content = lines.join('\n');
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `crop-trust-report-${Date.now()}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
