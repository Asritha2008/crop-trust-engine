import React, { useState } from 'react';
import { API_BASE_URL, IS_PRODUCTION } from './config';
import './App.css';

export default function App() {
  const [healthStatus, setHealthStatus] = useState(null); // null | 'loading' | 'success' | 'error'
  const [healthData, setHealthData] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [isWakingUp, setIsWakingUp] = useState(false);

  async function checkBackend() {
    setHealthStatus('loading');
    setErrorMessage('');
    setHealthData(null);
    setIsWakingUp(false);

    // Free hosts often take 30-60s to wake up from idle.
    // Set a wake-up warning timer after 3.5 seconds.
    const wakeTimer = setTimeout(() => {
      setIsWakingUp(true);
    }, 3500);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 65000); // 65s timeout for cold starts

    try {
      const res = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        signal: controller.signal
      });

      clearTimeout(wakeTimer);
      clearTimeout(timeoutId);

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status} (${res.statusText})`);
      }

      const data = await res.json();
      setHealthData(data);
      setHealthStatus('success');
    } catch (err) {
      clearTimeout(wakeTimer);
      clearTimeout(timeoutId);

      let msg = err.message;
      if (err.name === 'AbortError') {
        msg = 'Connection timed out after 65s. The server may still be spinning up or the address is unreachable.';
      } else if (msg === 'Failed to fetch') {
        msg = 'Failed to connect. Ensure the backend is running and CORS allows this origin.';
      }

      setErrorMessage(msg);
      setHealthStatus('error');
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
          Calibrated selective crop recommendation with out-of-distribution detection.
        </p>
      </header>

      <main>
        <section className="card">
          <h2 className="card-title">Backend Connection Test</h2>
          <p className="card-desc">
            Verifies connectivity with the FastAPI backend across local or deployed hosts.
          </p>

          <div className="meta-grid">
            <div className="meta-item">
              <span className="meta-label">Configured API Base URL</span>
              <span className="meta-value">{API_BASE_URL}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Environment Mode</span>
              <span className="meta-value">
                {IS_PRODUCTION ? 'Production (VITE_API_URL required at build)' : 'Development'}
              </span>
            </div>
          </div>

          <button
            className="btn-primary"
            onClick={checkBackend}
            disabled={healthStatus === 'loading'}
          >
            {healthStatus === 'loading' ? 'Checking /health...' : 'Check Backend Health'}
          </button>

          {healthStatus === 'loading' && (
            <div className="status-box loading">
              <div className="status-header">
                <span>⏳</span>
                <strong>Connecting to backend...</strong>
              </div>
              <p>Sending GET request to {API_BASE_URL}/health</p>
              {isWakingUp && (
                <p className="status-note">
                  ⚠️ <strong>Waking the server:</strong> Free cloud tiers (Render / HF Spaces) sleep when idle and may take up to 60 seconds to spin up. Please wait...
                </p>
              )}
            </div>
          )}

          {healthStatus === 'success' && (
            <div className="status-box success">
              <div className="status-header">
                <span>✅</span>
                <strong>Backend Online (HTTP 200)</strong>
              </div>
              <div className="status-code">
                {JSON.stringify(healthData, null, 2)}
              </div>
            </div>
          )}

          {healthStatus === 'error' && (
            <div className="status-box error">
              <div className="status-header">
                <span>❌</span>
                <strong>Connection Failed</strong>
              </div>
              <p>{errorMessage}</p>
              <p className="status-note">
                💡 <em>Note: If testing against a deployed backend, confirm with Role 3 that your exact origin is permitted in the CORS allow-list.</em>
              </p>
            </div>
          )}
        </section>

        <section className="card checklist-card">
          <h2 className="card-title">Task 1: Week 1 Deliverables</h2>
          <ul className="checklist">
            <li>
              <span className="check-icon">✓</span>
              <span><strong>React + Vite Scaffolding:</strong> Modern component-driven architecture ready for UI pages and charting.</span>
            </li>
            <li>
              <span className="check-icon">✓</span>
              <span><strong>Centralized API Config:</strong> Single source of truth via <code>VITE_API_URL</code>, strictly validated to fail loudly in production if missing.</span>
            </li>
            <li>
              <span className="check-icon">✓</span>
              <span><strong>Mobile-First Viewport:</strong> Responsive, accessible styles and minimum 48px touch targets for phone evaluation.</span>
            </li>
            <li>
              <span className="check-icon">✓</span>
              <span><strong>Deploy-Ready:</strong> Production build testable immediately on Vercel, Netlify, or GitHub Pages.</span>
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

