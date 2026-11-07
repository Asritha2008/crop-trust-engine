import React, { useState } from 'react';

export default function ExplanationCard({
  explanation,
  measureNext,
  ood,
  modelVersion
}) {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <div className="explanation-card">
      <button
        type="button"
        className="explanation-toggle-btn"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
      >
        <div className="toggle-title">
          <span>💡</span>
          <strong>Why did the Trust Engine give this verdict?</strong>
        </div>
        <span className="toggle-arrow">{isOpen ? '▲ Hide' : '▼ Expand'}</span>
      </button>

      {isOpen && (
        <div className="explanation-content">
          <p className="primary-reason">{explanation}</p>

          {/* Actionable Guidance (Shown especially on Caution) */}
          {measureNext && (
            <div className="actionable-box">
              <div className="actionable-header">
                <span className="actionable-icon">🔬</span>
                <strong>What to Measure Next</strong>
              </div>
              <p className="actionable-text">{measureNext}</p>
            </div>
          )}

          {/* Unusual Feature Pair Highlight (Shown on Refusal) */}
          {ood?.unusual_pair && (
            <div className="ood-pair-box">
              <div className="ood-pair-header">
                <span className="ood-pair-icon">⚠️</span>
                <strong>Unusual Feature Combination Detected:</strong>
              </div>
              <p className="ood-pair-text">
                Features <code>{ood.unusual_pair[0]}</code> and <code>{ood.unusual_pair[1]}</code> do
                not co-occur in the 2,200 training samples.
              </p>
              {ood.nearest_known && (
                <p className="ood-nearest-text">
                  Nearest known clusters: <em>{ood.nearest_known.join(', ')}</em>
                </p>
              )}
            </div>
          )}

          <div className="explanation-footer">
            <span className="version-tag">Model Version: {modelVersion || '2026-11-07-a'}</span>
            <span className="disclaimer-text">
              Five-stage trust engine: Validation → OOD Gate → Calibrated Classifier → Stability Check → Verdict
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
