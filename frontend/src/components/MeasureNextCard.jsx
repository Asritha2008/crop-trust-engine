import React from 'react';

/**
 * Task 6: Dedicated "What to Measure Next" Card
 * Shown only when measure_next is not null (appears on Caution).
 */
export default function MeasureNextCard({ measureNext }) {
  if (!measureNext) return null;

  return (
    <div className="measure-next-card" role="region" aria-label="What to measure next">
      <div className="measure-next-header">
        <span className="measure-next-icon">🔬</span>
        <div>
          <h3 className="measure-next-title">What to Measure Next</h3>
          <span className="measure-next-subtitle">Targeted measurement to resolve decision ambiguity</span>
        </div>
      </div>
      <p className="measure-next-text">{measureNext}</p>
    </div>
  );
}

