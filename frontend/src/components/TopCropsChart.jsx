import React from 'react';

export default function TopCropsChart({ topCrops, conformalSet, isRefusal }) {
  if (isRefusal || !topCrops || topCrops.length === 0) {
    return (
      <div className="top-crops-card empty-crops">
        <h3 className="section-title">Chance for each crop</h3>
        <div className="empty-crops-notice">
          <span className="notice-icon">🚫</span>
          <div>
            <strong>No Crop Recommendations Made</strong>
            <p className="notice-desc">
              The inputs are an unusual combination, so the model refuses to guess and shows no probabilities.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const inConformal = (cropName) => {
    return conformalSet && conformalSet.includes(cropName.toLowerCase());
  };
  // Hide extra bars that would show 0%. Always keep the first (top) crop.
  const visibleCrops = topCrops.filter(
    (item, index) => index === 0 || Math.round(item.probability * 100) > 0
  );

  return (
    <div className="top-crops-card">
      <div className="top-crops-header">
        <h3 className="section-title">Top Recommended Crops</h3>
        <span className="section-subtitle">How sure the model is</span>
      </div>

      <div className="crops-bars-list">
        {visibleCrops.map((item, index) => {
          const pct = Math.round(item.probability * 100);
          const pctLabel = item.probability >= 0.995 ? '99%+' : `${pct}%`;
          const isTop = index === 0;
          const isCovered = inConformal(item.crop);

          return (
            <div key={item.crop} className={`crop-bar-row ${isTop ? 'top-crop-winner' : ''}`}>
              <div className="crop-meta">
                <span className="crop-rank">#{index + 1}</span>
                <div className="crop-name-group">
                  <span className="crop-label">{item.crop}</span>
                  {isCovered && (
                    <span className="conformal-badge" title="One of the crops the model considers possible">
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

              <span className="prob-value">{pctLabel}</span>
            </div>
          );
        })}
      </div>

      {conformalSet && conformalSet.length > 0 && (
        <div className="conformal-footer">
          <span className="footer-tag">Possible crops:</span>
          <span>[{conformalSet.join(', ')}]</span>
        </div>
      )}
    </div>
  );
}
