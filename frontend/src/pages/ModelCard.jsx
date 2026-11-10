import React, { useState, useEffect } from 'react';

export default function ModelCard() {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const baseUrl = (import.meta.env.VITE_API_URL || 'https://crop-trust-engine.onrender.com').replace(/\/+$/, '');

  async function fetchModelInfo(signal) {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${baseUrl}/model-info`, { signal });
      if (!res.ok) {
        throw new Error(`Failed to load model specifications (HTTP ${res.status})`);
      }
      const data = await res.json();
      setModelInfo(data);
    } catch (err) {
      if (err.name === 'AbortError') return;
      let msg = err.message || 'Could not connect to model information service';
      if (err instanceof TypeError || msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
        msg = `Could not reach the server at ${baseUrl}. If using a free-tier hosting instance, it may take up to 60 seconds to wake from sleep.`;
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    fetchModelInfo(controller.signal);
    return () => controller.abort();
  }, [baseUrl]);

  // Frozen physical bounds according to the API contract
  const physicalBounds = [
    { feature: 'Nitrogen (N)', range: '0 .. 200', unit: 'ratio / kg/ha', desc: 'Soil nitrogen ratio; essential for leaf and vegetative growth.' },
    { feature: 'Phosphorus (P)', range: '0 .. 200', unit: 'ratio / kg/ha', desc: 'Soil phosphorus ratio; critical for root establishment and energy transfer.' },
    { feature: 'Potassium (K)', range: '0 .. 300', unit: 'ratio / kg/ha', desc: 'Soil potassium ratio; regulates drought tolerance and water retention.' },
    { feature: 'Temperature', range: '-10 .. 60', unit: '°C', desc: 'Ambient growing temperature.' },
    { feature: 'Relative Humidity', range: '0 .. 100', unit: '%', desc: 'Relative atmospheric humidity during the cultivation season.' },
    { feature: 'Soil pH', range: '0 .. 14', unit: 'pH scale', desc: 'Soil acidity / alkalinity scale (neutral = 7.0).' },
    { feature: 'Rainfall', range: '0 .. 1000', unit: 'mm', desc: 'Annual or seasonal precipitation volume.' }
  ];

  return (
    <div className="model-card-page">
      {/* Page Header */}
      <div className="page-header">
        <div className="page-header-text">
          <span className="page-category-tag">System Documentation</span>
          <h2 className="page-title">📄 Model Card & System Specifications</h2>
          <p className="page-description">
            Transparent model provenance, validation protocols, and agronomic operating boundaries.
          </p>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="loading-state-card" role="status">
          <div className="loading-spinner" aria-hidden="true" />
          <p className="loading-title">Waking server / loading model info...</p>
          <p className="loading-subtext">Free-tier instances may take up to 60 seconds to wake from sleep.</p>
        </div>
      )}

      {/* Error State */}
      {!loading && error && (
        <div className="error-state-card" role="alert">
          <div className="error-content">
            <span className="error-icon" aria-hidden="true">⚠️</span>
            <div className="error-text-wrap">
              <h3 className="error-heading">Unable to Load Model Specifications</h3>
              <p className="error-message">{error}</p>
            </div>
          </div>
          <button
            type="button"
            className="btn-retry"
            onClick={() => {
              const controller = new AbortController();
              fetchModelInfo(controller.signal);
            }}
          >
            🔄 Try Again
          </button>
        </div>
      )}

      {/* Model Metadata Section */}
      {!loading && modelInfo && (
        <section className="model-card-section" aria-labelledby="metadata-heading">
          <h3 id="metadata-heading" className="section-title">Model Metadata & Provenance</h3>
          <div className="metadata-grid">
            <div className="metadata-item">
              <span className="metadata-label">Model Version</span>
              <span className="metadata-value highlight">{modelInfo.model_version || '—'}</span>
            </div>

            <div className="metadata-item">
              <span className="metadata-label">Trained On</span>
              <span className="metadata-value">{modelInfo.trained_on || '—'}</span>
            </div>

            <div className="metadata-item">
              <span className="metadata-label">scikit-learn Version</span>
              <span className="metadata-value code-font">{modelInfo.sklearn_version || '—'}</span>
            </div>

            {modelInfo.final_model && (
              <div className="metadata-item">
                <span className="metadata-label">Selected Architecture</span>
                <span className="metadata-value">{modelInfo.final_model}</span>
              </div>
            )}

            {modelInfo.calibration_method && (
              <div className="metadata-item">
                <span className="metadata-label">Probability Calibration</span>
                <span className="metadata-value">{modelInfo.calibration_method}</span>
              </div>
            )}

            {modelInfo.date && (
              <div className="metadata-item">
                <span className="metadata-label">Release Date</span>
                <span className="metadata-value">{modelInfo.date}</span>
              </div>
            )}
          </div>

          {modelInfo.selection_rule && (
            <div className="selection-rule-box">
              <span className="selection-rule-title">Selection Protocol:</span>
              <p className="selection-rule-text">{modelInfo.selection_rule}</p>
            </div>
          )}
        </section>
      )}

      {/* Intended Use & Scope */}
      <section className="model-card-section" aria-labelledby="intended-use-heading">
        <h3 id="intended-use-heading" className="section-title">Intended Use & Agronomic Scope</h3>
        <div className="intended-use-cards">
          <div className="use-card primary">
            <h4 className="use-card-title">🎯 Primary Purpose</h4>
            <p className="use-card-text">
              Provides pre-planting decision support by evaluating 7 soil and climate parameters against 22 crop profiles
              to recommend the most viable agricultural options.
            </p>
          </div>

          <div className="use-card warning">
            <h4 className="use-card-title">🛡️ Selective Prediction Architecture</h4>
            <p className="use-card-text">
              Equipped with a 5-stage Trust Engine (validation, out-of-distribution detection, probability calibration,
              stability testing, and verdict generation). The system will deliberately refuse to guess or warn the user
              when input combinations are unfamiliar or physically implausible.
            </p>
          </div>

          <div className="use-card neutral">
            <h4 className="use-card-title">⚠️ Non-Prescriptive Guidance</h4>
            <p className="use-card-text">
              This system is an advisory tool and does not replace professional agronomic consultation or local extension services.
              It does not prescribe specific chemical fertilizer dosages, seed treatment formulations, or financial yield forecasts.
            </p>
          </div>
        </div>
      </section>

      {/* Physical Bounds */}
      <section className="model-card-section" aria-labelledby="physical-bounds-heading">
        <h3 id="physical-bounds-heading" className="section-title">Physical Bounds & Validation Rules</h3>
        <p className="section-subtext">
          The API strictly validates inputs against these physiological bounds. Any input falling outside these ranges
          is rejected immediately with HTTP 422 before reaching the classification models.
        </p>

        <div className="bounds-table-container">
          <table className="bounds-table">
            <thead>
              <tr>
                <th scope="col">Feature</th>
                <th scope="col">Valid Range</th>
                <th scope="col">Unit</th>
                <th scope="col">Agronomic Significance</th>
              </tr>
            </thead>
            <tbody>
              {physicalBounds.map((b) => (
                <tr key={b.feature}>
                  <td className="feature-name"><strong>{b.feature}</strong></td>
                  <td className="feature-range"><code>{b.range}</code></td>
                  <td className="feature-unit">{b.unit}</td>
                  <td className="feature-desc">{b.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
