import React from 'react';
import classStatsData from '../data/class_stats.json';

const classStats = classStatsData.class_stats || classStatsData;

const NUTRIENT_NAMES = {
  N: 'Nitrogen',
  P: 'Phosphorus',
  K: 'Potassium'
};

/**
 * Task 6: Nutrient Gap View (N, P, K only)
 * Rules:
 * - One plain-language line each
 * - Describe the gap only: "below the typical range for <crop>" / "within" / "above"
 * - NO fertiliser quantities, NO advice on amounts, NO units for N, P, K
 * - Hidden when refused / no top crop
 */
export default function NutrientGapCard({ crop, inputs }) {
  if (!crop || !inputs) return null;

  const cropKey = crop.toLowerCase();
  const cropData = classStats[cropKey];
  if (!cropData) return null;

  const displayCrop = crop.charAt(0).toUpperCase() + crop.slice(1);

  const nutrientKeys = ['N', 'P', 'K'];

  const rows = nutrientKeys.map((key) => {
    const stats = cropData[key];
    const val = Number(inputs[key]);
    const name = NUTRIENT_NAMES[key] || key;

    let gapType = 'within';
    let text = '';
    let badgeText = '';
    let icon = '';

    if (val < stats.p05) {
      gapType = 'below';
      text = `${name} is below the typical range for ${cropKey}.`;
      badgeText = 'Below typical';
      icon = '⬇️';
    } else if (val > stats.p95) {
      gapType = 'above';
      text = `${name} is above the typical range for ${cropKey}.`;
      badgeText = 'Above typical';
      icon = '⬆️';
    } else {
      gapType = 'within';
      text = `${name} is within the typical range for ${cropKey}.`;
      badgeText = 'Within typical';
      icon = '✅';
    }

    return {
      key,
      name,
      val,
      stats,
      gapType,
      text,
      badgeText,
      icon
    };
  });

  return (
    <div className="nutrient-gap-card" role="region" aria-label={`Nutrient gap view for ${cropKey}`}>
      <div className="nutrient-gap-header">
        <span className="nutrient-gap-icon">🌱</span>
        <div>
          <h3 className="nutrient-gap-title">Nutrient Gap View for {displayCrop}</h3>
          <span className="nutrient-gap-subtitle">
            Soil macronutrients (N, P, K) compared with the typical range for {displayCrop} in the training data.
          </span>
        </div>
      </div>

      <div className="nutrient-gap-list">
        {rows.map((row) => (
          <div key={row.key} className={`nutrient-gap-row gap-${row.gapType}`}>
            <div className="nutrient-status-meta">
              <span className="nutrient-status-icon">{row.icon}</span>
              <span className="nutrient-gap-text">{row.text}</span>
            </div>
            <span className={`nutrient-badge badge-${row.gapType}`}>
              {row.badgeText}
            </span>
          </div>
        ))}
      </div>

      <p className="nutrient-gap-disclaimer">
        ℹ️ Describes statistical gap only. No fertiliser quantities or application amounts are provided.
      </p>
    </div>
  );
}

