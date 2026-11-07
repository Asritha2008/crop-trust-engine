import React from 'react';

export default function TrustMeter({ trustScore, stability, ood }) {
  const isRefusal = trustScore === 0 && stability === null;

  function getScoreColor(score) {
    if (score >= 75) return '#15803d'; // Green
    if (score >= 45) return '#d97706'; // Amber
    return '#dc2626'; // Red
  }

  const scoreColor = getScoreColor(trustScore ?? 0);
  const stabilityPct = stability !== null && stability !== undefined ? Math.round(stability * 100) : null;

  return (
    <div className="trust-meter-card">
      <div className="metric-tile">
        <div className="metric-header">
          <span className="metric-title">Trust Score</span>
          <span className="metric-hint">Stage 5 Composite</span>
        </div>
        <div className="metric-gauge-wrapper">
          <div
            className="metric-circle"
            style={{
              borderColor: scoreColor,
              color: scoreColor
            }}
          >
            <span className="metric-circle-val">{trustScore ?? 0}</span>
            <span className="metric-circle-max">/100</span>
          </div>
          <div className="metric-explanation">
            {isRefusal ? (
              <span className="metric-badge-refused">Score 0 (Refusal)</span>
            ) : trustScore >= 75 ? (
              <span className="metric-badge-high">High Confidence</span>
            ) : (
              <span className="metric-badge-medium">Hedging Advised</span>
            )}
          </div>
        </div>
      </div>

      {/* Stability Metric */}
      <div className="metric-tile">
        <div className="metric-header">
          <span className="metric-title">Input Stability</span>
          <span className="metric-hint">Stage 4 Sensor Noise</span>
        </div>
        <div className="metric-bar-wrapper">
          <div className="metric-val-large">
            {stabilityPct !== null ? `${stabilityPct}%` : '—'}
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{
                width: `${stabilityPct ?? 0}%`,
                backgroundColor: scoreColor
              }}
            />
          </div>
          <span className="metric-subtext">
            {stabilityPct !== null
              ? `${stabilityPct}% answers consistent under sensor perturbation`
              : 'Nullified due to OOD refusal'}
          </span>
        </div>
      </div>

      {/* OOD Metric */}
      <div className="metric-tile">
        <div className="metric-header">
          <span className="metric-title">Distribution Gate</span>
          <span className="metric-hint">Stage 2 Mahalanobis</span>
        </div>
        <div className="ood-status-wrapper">
          <div className={`ood-pill ${ood?.flagged ? 'ood-flagged' : 'ood-normal'}`}>
            <span>{ood?.flagged ? '🛑 Gate Triggered' : '✅ Within Distribution'}</span>
          </div>
          <div className="ood-percentile-text">
            Distance Percentile: <strong>{ood?.percentile !== undefined ? `${ood.percentile}%` : 'N/A'}</strong>
          </div>
        </div>
      </div>
    </div>
  );
}
