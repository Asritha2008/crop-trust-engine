import React, { useState, useEffect } from 'react';

export default function Arena() {
  const [arenaData, setArenaData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const baseUrl = (import.meta.env.VITE_API_URL || 'https://crop-trust-engine.onrender.com').replace(/\/+$/, '');

  async function fetchArena(signal) {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${baseUrl}/arena`, { signal });
      if (!res.ok) {
        throw new Error(`Failed to load benchmark arena data (HTTP ${res.status})`);
      }
      const data = await res.json();
      setArenaData(data);
    } catch (err) {
      if (err.name === 'AbortError') return;
      let msg = err.message || 'Could not connect to benchmark arena service';
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
    fetchArena(controller.signal);
    return () => controller.abort();
  }, [baseUrl]);

  const modelsSummary = arenaData?.models_summary || {};
  const championModel = arenaData?.champion_model || '';

  // The 9 classifiers evaluated in the benchmark
  const evaluatedClassifiers = [
    { key: 'Logistic Regression', name: 'Logistic Regression' },
    { key: 'K-Nearest Neighbors', altKeys: ['KNN', 'K-Nearest Neighbors (KNN)'], name: 'KNN (K-Nearest Neighbors)' },
    { key: 'SVM (RBF)', altKeys: ['SVM', 'Support Vector Machine'], name: 'SVM (RBF)' },
    { key: 'Decision Tree', name: 'Decision Tree' },
    { key: 'Random Forest', name: 'Random Forest' },
    { key: 'Extra Trees', name: 'Extra Trees' },
    { key: 'Gradient Boosting', name: 'Gradient Boosting' },
    { key: 'Gaussian Naive Bayes', altKeys: ['Gaussian NB'], name: 'Gaussian Naive Bayes' },
    { key: 'Small MLP', altKeys: ['MLP'], name: 'Small MLP' }
  ];

  // Helper to match data from modelsSummary
  function getModelData(item) {
    if (modelsSummary[item.key]) return modelsSummary[item.key];
    if (item.altKeys) {
      for (const alt of item.altKeys) {
        if (modelsSummary[alt]) return modelsSummary[alt];
      }
    }
    const match = Object.keys(modelsSummary).find(
      (k) => k.toLowerCase() === item.key.toLowerCase()
    );
    return match ? modelsSummary[match] : null;
  }

  // Check if a model is the champion
  function isChampion(item) {
    if (!championModel) return false;
    const champ = championModel.toLowerCase();
    const name = item.name.toLowerCase();
    const key = item.key.toLowerCase();
    return champ === name || champ === key || champ.includes(key) || key.includes(champ);
  }

  function formatAccuracy(metrics) {
    if (!metrics || typeof metrics.accuracy_mean !== 'number') return '—';
    const mean = (metrics.accuracy_mean * 100).toFixed(2);
    const std = typeof metrics.accuracy_std === 'number' ? ` ± ${(metrics.accuracy_std * 100).toFixed(2)}%` : '';
    return `${mean}%${std}`;
  }

  function formatMacroF1(metrics) {
    if (!metrics || typeof metrics.macro_f1_mean !== 'number') return '—';
    const mean = (metrics.macro_f1_mean * 100).toFixed(2);
    const std = typeof metrics.macro_f1_std === 'number' ? ` ± ${(metrics.macro_f1_std * 100).toFixed(2)}%` : '';
    return `${mean}%${std}`;
  }

  function formatEce(metrics) {
    if (!metrics || typeof metrics.ece_mean !== 'number') return '—';
    return metrics.ece_mean.toFixed(4);
  }

  function formatFitTime(metrics) {
    if (!metrics || typeof metrics.fit_time_s_mean !== 'number') return '—';
    return `${metrics.fit_time_s_mean.toFixed(3)}s`;
  }

  return (
    <div className="arena-page">
      {/* Page Header */}
      <div className="page-header">
        <div className="page-header-text">
          <span className="page-category-tag">Empirical Evaluation</span>
          <h2 className="page-title">⚔️ Model Arena & Benchmark Comparison</h2>
          <p className="page-description">
            Rigorous 15-fold cross-validated comparison of 9 multi-class classifiers on identical stratified splits.
          </p>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="loading-state-card" role="status">
          <div className="loading-spinner" aria-hidden="true" />
          <p className="loading-title">Waking server / loading benchmark arena...</p>
          <p className="loading-subtext">Free-tier instances may take up to 60 seconds to wake from sleep.</p>
        </div>
      )}

      {/* Error State */}
      {!loading && error && (
        <div className="error-state-card" role="alert">
          <div className="error-content">
            <span className="error-icon" aria-hidden="true">⚠️</span>
            <div className="error-text-wrap">
              <h3 className="error-heading">Unable to Load Arena Benchmark</h3>
              <p className="error-message">{error}</p>
            </div>
          </div>
          <button
            type="button"
            className="btn-retry"
            onClick={() => {
              const controller = new AbortController();
              fetchArena(controller.signal);
            }}
          >
            🔄 Try Again
          </button>
        </div>
      )}

      {/* Benchmark Content */}
      {!loading && !error && (
        <>
          {/* Protocol Info Cards */}
          {arenaData?.benchmark_metadata && (
            <div className="arena-metadata-banner">
              <div className="arena-meta-item">
                <span className="meta-label">Protocol</span>
                <span className="meta-value">{arenaData.benchmark_metadata.protocol || 'RepeatedStratifiedKFold'}</span>
              </div>
              <div className="arena-meta-item">
                <span className="meta-label">Training Size</span>
                <span className="meta-value">{arenaData.benchmark_metadata.train_samples || 1540} rows (22 classes)</span>
              </div>
              <div className="arena-meta-item">
                <span className="meta-label">Champion Selected</span>
                <span className="meta-value highlight">🏆 {championModel || 'Extra Trees'}</span>
              </div>
            </div>
          )}

          {/* Benchmark Table */}
          <div className="arena-table-card">
            <div className="table-header-group">
              <h3 className="table-title">9-Classifier Performance Arena</h3>
              <span className="table-caption">
                Metrics averaged across 15 folds (5 folds × 3 repeats). Highlighted row indicates the selected production pipeline.
              </span>
            </div>

            <div className="arena-table-container">
              <table className="arena-table">
                <thead>
                  <tr>
                    <th scope="col">Model Name</th>
                    <th scope="col">Accuracy (mean ± std)</th>
                    <th scope="col">Macro F1 (mean ± std)</th>
                    <th scope="col">ECE Mean</th>
                    <th scope="col">Fit Time (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {evaluatedClassifiers.map((model) => {
                    const metrics = getModelData(model);
                    const champion = isChampion(model);

                    return (
                      <tr
                        key={model.key}
                        className={champion ? 'arena-row-champion' : 'arena-row'}
                      >
                        <td className="model-name-cell">
                          <div className="model-name-wrapper">
                            <span className="model-name-text">{model.name}</span>
                            {champion && (
                              <span className="champion-badge" title="Selected champion model">
                                🏆 Champion
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="metric-cell">
                          <strong>{formatAccuracy(metrics)}</strong>
                        </td>
                        <td className="metric-cell">
                          {formatMacroF1(metrics)}
                        </td>
                        <td className="metric-cell code-font">
                          {formatEce(metrics)}
                        </td>
                        <td className="metric-cell muted">
                          {formatFitTime(metrics)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="table-footer-notes">
              <p>
                <strong>ECE (Expected Calibration Error):</strong> Lower is better. Measures how closely predicted probabilities reflect actual empirical frequencies.
              </p>
              <p>
                <strong>Champion Selection:</strong> Extra Trees with isotonic calibration achieved the lowest post-calibration validation loss and optimal stability across sensor noise draws.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
