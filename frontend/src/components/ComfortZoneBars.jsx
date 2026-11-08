import React from 'react';
import classStatsData from '../data/class_stats.json';
import { PHYSICAL_BOUNDS } from '../config';

const classStats = classStatsData.class_stats || classStatsData;

const FEATURES_ORDER = [
  'N',
  'P',
  'K',
  'temperature',
  'humidity',
  'ph',
  'rainfall'
];

/**
 * Formats a number to 1 decimal place or whole integer
 */
function formatNum(val) {
  if (val === null || val === undefined || isNaN(val)) return '—';
  const num = Number(val);
  return Number.isInteger(num) ? num.toString() : (Math.round(num * 10) / 10).toString();
}

/**
 * Task 6: Comfort-Zone Bars Component
 * Rules:
 * - Shown when top_crops is non-empty, for the top crop
 * - 7 inputs: horizontal bar covering min..max of that crop
 * - p05..p95 band highlighted as "typical"
 * - marker at the current input
 * - text label per row (never colour alone): "Typical", "At the edge", or "Outside what was seen for this crop"
 * - heading: "Comfort zone for <crop>"
 * - note: "Based on the training data for this crop."
 */
export default function ComfortZoneBars({ crop, inputs }) {
  if (!crop || !inputs) return null;

  const cropKey = crop.toLowerCase();
  const cropData = classStats[cropKey];
  if (!cropData) return null;

  const displayCrop = crop.charAt(0).toUpperCase() + crop.slice(1);

  return (
    <div className="comfort-zone-card" role="region" aria-label={`Comfort zone for ${cropKey}`}>
      <div className="comfort-zone-header">
        <div className="comfort-zone-title-wrap">
          <span className="comfort-zone-icon">📊</span>
          <div>
            <h3 className="comfort-zone-title">Comfort zone for {cropKey}</h3>
            <p className="comfort-zone-note">Based on the training data for this crop.</p>
          </div>
        </div>
        <span className="comfort-crop-badge">{displayCrop} Profile</span>
      </div>

      <div className="comfort-bars-list">
        {FEATURES_ORDER.map((feature) => {
          const stats = cropData[feature];
          if (!stats) return null;

          const bounds = PHYSICAL_BOUNDS[feature] || { label: feature, unit: '' };
          const currentVal = Number(inputs[feature]);
          const unitSuffix = bounds.unit ? ` ${bounds.unit}` : '';

          const range = stats.max - stats.min;
          const safeRange = range <= 0 ? 1 : range;

          // Band percentages (relative to crop min..max track)
          const p05Pct = Math.max(0, Math.min(100, ((stats.p05 - stats.min) / safeRange) * 100));
          const p95Pct = Math.max(0, Math.min(100, ((stats.p95 - stats.min) / safeRange) * 100));
          const bandWidthPct = Math.max(0, p95Pct - p05Pct);

          // Marker percentage (clamped to 0..100% on track)
          const rawMarkerPct = ((currentVal - stats.min) / safeRange) * 100;
          const clampedMarkerPct = Math.max(0, Math.min(100, rawMarkerPct));

          // Text label evaluation (never colour alone)
          let labelText = '';
          let statusType = '';
          let statusIcon = '';

          if (currentVal >= stats.p05 && currentVal <= stats.p95) {
            labelText = 'Typical';
            statusType = 'typical';
            statusIcon = '✅';
          } else if ((currentVal >= stats.min && currentVal < stats.p05) || (currentVal > stats.p95 && currentVal <= stats.max)) {
            labelText = 'At the edge';
            statusType = 'edge';
            statusIcon = '⚠️';
          } else {
            labelText = 'Outside what was seen for this crop';
            statusType = 'outside';
            statusIcon = '🛑';
          }

          const isBelowMin = currentVal < stats.min;
          const isAboveMax = currentVal > stats.max;

          return (
            <div key={feature} className={`comfort-bar-row status-${statusType}`}>
              <div className="comfort-row-top">
                <div className="comfort-feature-meta">
                  <span className="comfort-feature-name">{bounds.label}</span>
                  <span className="comfort-current-val">
                    Current: <strong>{formatNum(currentVal)}{unitSuffix}</strong>
                  </span>
                </div>

                <div className={`comfort-status-badge badge-${statusType}`}>
                  <span className="badge-icon">{statusIcon}</span>
                  <span className="badge-text">{labelText}</span>
                </div>
              </div>

              {/* Horizontal Bar Track */}
              <div className="comfort-bar-track-wrap">
                <div className="comfort-bar-track">
                  {/* Highlighted p05..p95 "Typical" Band */}
                  <div
                    className="comfort-typical-band"
                    style={{
                      left: `${p05Pct}%`,
                      width: `${bandWidthPct}%`
                    }}
                    title={`Typical range: ${formatNum(stats.p05)} – ${formatNum(stats.p95)}${unitSuffix}`}
                  >
                    <span className="band-inline-label">Typical (p05–p95)</span>
                  </div>

                  {/* Marker for Current Input */}
                  <div
                    className={`comfort-marker marker-${statusType} ${isBelowMin ? 'marker-below' : ''} ${isAboveMax ? 'marker-above' : ''}`}
                    style={{ left: `${clampedMarkerPct}%` }}
                    title={`Current value: ${formatNum(currentVal)}${unitSuffix} (${labelText})`}
                  >
                    <div className="marker-pin"></div>
                    <div className="marker-label">
                      {isBelowMin ? '◀ ' : isAboveMax ? '▶ ' : ''}{formatNum(currentVal)}
                    </div>
                  </div>
                </div>

                {/* Range Limits beneath the bar */}
                <div className="comfort-limits-bar">
                  <span className="limit-min">Min: {formatNum(stats.min)}</span>
                  <span className="limit-typical">
                    Typical: {formatNum(stats.p05)} – {formatNum(stats.p95)}{unitSuffix}
                  </span>
                  <span className="limit-max">Max: {formatNum(stats.max)}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

