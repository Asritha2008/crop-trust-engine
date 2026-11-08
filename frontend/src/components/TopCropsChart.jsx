import React from 'react';

export default function TopCropsChart({ topCrops, conformalSet, isRefusal }) {
  if (isRefusal || !topCrops || topCrops.length === 0) {
    return (
      <div className="top-crops-card empty-crops">
        <h3 className="section-title">Calibrated Crop Probabilities</h3>
        <div className="empty-crops-notice">
          <span className="notice-icon">🚫</span>
          <div>
            <strong>No Crop Recommendations Made</strong>
            <p className="notice-desc">
              When an input is gated as Out-of-Distribution, the model abstains from outputting misleading probabilities.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const inConformal = (cropName) => {
    return conformalSet && conformalSet.includes(cropName.toLowerCase());
  };

  return (
    <div className="top-crops-card">
      <div className="top-crops-header">
        <h3 className="section-title">Top Recommended Crops</h3>
        <span className="section-subtitle">Calibrated probabilities</span>
      </div>

      <div className="crops-bars-list">
        {topCrops.map((item, index) => {
          const pct = Math.round(item.probability * 100);
          const isTop = index === 0;
          const isCovered = inConformal(item.crop);

          return (
            <div key={item.crop} className={`crop-bar-row ${isTop ? 'top-crop-winner' : ''}`}>
              <div className="crop-meta">
                <span className="crop-rank">#{index + 1}</span>
                <div className="crop-name-group">
                  <span className="crop-label">{item.crop}</span>
                  {isCovered && (
                    <span className="conformal-badge" title="Included in conformal prediction set">
                      In set
                    </span>
                  )}
                </div>
              </div>

              <div className="bar-track">
                <div
                  className="bar-fill"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: isTop ? '#15803d' : '#475569'
                  }}
                />
              </div>

              <span className="prob-value">{pct}%</span>
            </div>
          );
        })}
      </div>

      {conformalSet && conformalSet.length > 0 && (
        <div className="conformal-footer">
          <span className="footer-tag">Possible crops (conformal set):</span>
          <span>[{conformalSet.join(', ')}]</span>
        </div>
      )}
    </div>
  );
}
