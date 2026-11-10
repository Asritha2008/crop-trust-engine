import React, { useState } from 'react';

export default function ExplanationCard({
  explanation,
  verdict,
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
          <p className="primary-reason">
            {verdict === 'refuse' ? 'The engine refuses this input.' : explanation}
          </p>


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
