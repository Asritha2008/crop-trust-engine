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
              How it decides: check inputs → check they look realistic → estimate crops → test stability → verdict
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
