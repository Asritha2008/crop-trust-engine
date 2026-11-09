import React, { useState, useEffect, useRef } from 'react';
import { PHYSICAL_BOUNDS, CLIMATE_SCENARIOS, clampValue } from '../config';
import { predictCrop } from '../api/client';
import VerdictBadge from './VerdictBadge';

export default function WhatIfPanel({
  baselineInputs,
  baselinePrediction,
  riskProfile
}) {
  // 1. Single Feature What-If State
  const [selectedField, setSelectedField] = useState('rainfall');
  const [whatIfValue, setWhatIfValue] = useState(baselineInputs?.rainfall ?? 150);
  const [whatIfPrediction, setWhatIfPrediction] = useState(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const [whatIfError, setWhatIfError] = useState(null);

  const whatIfControllerRef = useRef(null);
  const whatIfDebounceRef = useRef(null);
  const rateLimitUntilRef = useRef(0);

  // 2. Climate Stress Test State
  const [stressLoading, setStressLoading] = useState(false);
  const [stressResults, setStressResults] = useState(null);
  const [stressError, setStressError] = useState(null);
  const stressControllersRef = useRef([]);

  // Sync whatIfValue when selectedField or baselineInputs change
  useEffect(() => {
    if (baselineInputs && baselineInputs[selectedField] !== undefined) {
      setWhatIfValue(baselineInputs[selectedField]);
    }
  }, [selectedField, baselineInputs]);

  // 3. Cancel in-flight stress calls if baseline inputs change
  useEffect(() => {
    if (stressControllersRef.current.length > 0) {
      stressControllersRef.current.forEach((c) => c && c.abort());
      stressControllersRef.current = [];
    }
    setStressLoading(false);
  }, [baselineInputs]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (whatIfDebounceRef.current) clearTimeout(whatIfDebounceRef.current);
      if (whatIfControllerRef.current) whatIfControllerRef.current.abort();
      if (stressControllersRef.current.length > 0) {
        stressControllersRef.current.forEach((c) => c && c.abort());
      }
    };
  }, []);

  // Run What-If prediction with debounce (200 ms)
  function handleWhatIfValueChange(val) {
    const fieldBounds = PHYSICAL_BOUNDS[selectedField];
    const clamped = clampValue(selectedField, val);
    setWhatIfValue(clamped);

    if (whatIfDebounceRef.current) clearTimeout(whatIfDebounceRef.current);

    whatIfDebounceRef.current = setTimeout(() => {
      // Don't send automatic re-calls from slider moves while rate-limited
      if (Date.now() < rateLimitUntilRef.current) {
        return;
      }
      executeWhatIfPrediction(selectedField, clamped);
    }, 200);
  }

  async function executeWhatIfPrediction(field = selectedField, val = whatIfValue) {
    if (whatIfControllerRef.current) {
      whatIfControllerRef.current.abort();
    }
    whatIfControllerRef.current = new AbortController();
    const signal = whatIfControllerRef.current.signal;

    setWhatIfLoading(true);
    setWhatIfError(null);

    const adjustedInputs = {
      ...baselineInputs,
      [field]: val
    };

    try {
      const res = await predictCrop(adjustedInputs, riskProfile, signal);
      if (signal.aborted) return;
      setWhatIfPrediction(res);
      setWhatIfError(null);
    } catch (err) {
      if (err.name === 'AbortError' || signal.aborted) return;

      let msg = 'Failed to evaluate scenario.';
      if (err.isRateLimited || err.status === 429) {
        const waitText = typeof err.retryAfter === 'number' ? `${err.retryAfter} seconds` : 'a few seconds';
        msg = `You're going a bit fast. Please wait ${waitText} and try again.`;
        rateLimitUntilRef.current = Date.now() + (typeof err.retryAfter === 'number' ? err.retryAfter * 1000 : 3000);
      } else if (err.isNetwork) {
        msg = "Couldn't reach the server";
      } else if (err.status >= 500) {
        msg = 'Something went wrong on our side';
      } else if (typeof err.message === 'string' && err.message !== '[object Object]') {
        msg = err.message;
      }
      setWhatIfError(msg);
    } finally {
      if (!signal.aborted) {
        setWhatIfLoading(false);
      }
    }
  }

  // Initial What-If execution on load
  useEffect(() => {
    if (baselineInputs) {
      executeWhatIfPrediction(selectedField, baselineInputs[selectedField] ?? 150);
    }
  }, [selectedField]);

  // Run Climate Stress Test across 3 scenarios using Promise.all
  async function runClimateStressTest() {
    setStressLoading(true);
    setStressResults(null);
    setStressError(null);

    if (stressControllersRef.current.length > 0) {
      stressControllersRef.current.forEach((c) => c && c.abort());
    }
    stressControllersRef.current = [];

    const promises = CLIMATE_SCENARIOS.map(async (sc, index) => {
      const controller = new AbortController();
      stressControllersRef.current[index] = controller;

      const scInputs = { ...baselineInputs };
      if (sc.deltaTemp) {
        scInputs.temperature = clampValue('temperature', Number(baselineInputs.temperature) + sc.deltaTemp);
      }
      if (sc.rainfallFactor) {
        scInputs.rainfall = clampValue(
          'rainfall',
          Math.round(Number(baselineInputs.rainfall) * sc.rainfallFactor * 10) / 10
        );
      }

      try {
        const res = await predictCrop(scInputs, riskProfile, controller.signal);
        return {
          id: sc.id,
          name: sc.name,
          inputs: scInputs,
          success: true,
          data: res
        };
      } catch (err) {
        if (err.name === 'AbortError' || controller.signal.aborted) {
          return { id: sc.id, name: sc.name, aborted: true };
        }
        let msg = "couldn't run";
        if (err.isRateLimited || err.status === 429) {
          const waitText = typeof err.retryAfter === 'number' ? `${err.retryAfter} seconds` : 'a few seconds';
          msg = `You're going a bit fast. Please wait ${waitText} and try again.`;
        } else if (err.isNetwork) {
          msg = "Couldn't reach the server";
        } else if (err.status >= 500) {
          msg = 'Something went wrong on our side';
        } else if (typeof err.message === 'string' && err.message !== '[object Object]') {
          msg = err.message;
        }
        return {
          id: sc.id,
          name: sc.name,
          inputs: scInputs,
          success: false,
          error: msg
        };
      }
    });

    const results = await Promise.all(promises);

    if (results.some((r) => r.aborted)) {
      return;
    }

    setStressResults(results);
    setStressLoading(false);
  }

  // Compute what-changed summary line
  function getChangeSummary() {
    if (!baselinePrediction || !whatIfPrediction) return null;

    const baseVerdict = (baselinePrediction.verdict || '').toLowerCase();
    const newVerdict = (whatIfPrediction.verdict || '').toLowerCase();

    const baseTop = baselinePrediction.top_crops?.[0];
    const newTop = whatIfPrediction.top_crops?.[0];

    const baseTrust = baselinePrediction.trust_score ?? 0;
    const newTrust = whatIfPrediction.trust_score ?? 0;

    if (baseVerdict !== newVerdict) {
      return `Verdict changed from ${baseVerdict} to ${newVerdict}.`;
    }

    if (baseTop?.crop !== newTop?.crop) {
      if (!newTop) {
        return `Top crop recommendation cleared under this scenario.`;
      }
      const basePct = Math.round((baseTop?.probability ?? 0) * 100);
      const newPct = Math.round((newTop?.probability ?? 0) * 100);
      return `Top crop shifted from ${baseTop?.crop} (${basePct}%) to ${newTop?.crop} (${newPct}%).`;
    }

    if (baseTrust !== newTrust) {
      const diff = newTrust - baseTrust;
      const dir = diff > 0 ? `increased by +${diff}` : `decreased by ${diff}`;
      return `Top crop remained ${baseTop?.crop || 'the same'}, but trust score ${dir} (${baseTrust} → ${newTrust}).`;
    }

    return `Recommendation and trust score remained identical under this adjustment.`;
  }

  const changeSummary = getChangeSummary();
  const currentBounds = PHYSICAL_BOUNDS[selectedField] || { min: 0, max: 100, step: 1, unit: '' };

  return (
    <div className="whatif-container-card">
      <div className="whatif-header">
        <div>
          <h2 className="whatif-title">🔮 What-If Simulator & Stress Testing</h2>
          <span className="whatif-subtitle">
            Simulate parameter adjustments and climate shifts without altering baseline inputs.
          </span>
        </div>
      </div>

      {/* 1. Single Field What-If Controls */}
      <div className="whatif-controls-section">
        <div className="whatif-select-row">
          <div className="whatif-field-picker">
            <label htmlFor="whatif-field-select" className="whatif-label">
              Select Parameter to Adjust:
            </label>
            <select
              id="whatif-field-select"
              className="whatif-select"
              value={selectedField}
              onChange={(e) => setSelectedField(e.target.value)}
            >
              {Object.entries(PHYSICAL_BOUNDS).map(([key, b]) => (
                <option key={key} value={key}>
                  {b.label}{b.unit ? ` (${b.unit})` : ''}
                </option>
              ))}
            </select>
          </div>

          <div className="whatif-value-controls">
            <div className="whatif-slider-box">
              <div className="whatif-slider-header">
                <span className="whatif-val-label">New Simulated Value:</span>
                <span className="whatif-val-display">
                  {whatIfValue}{currentBounds.unit ? ` ${currentBounds.unit}` : ''}
                </span>
              </div>
              <input
                type="range"
                className="input-range-slider"
                min={currentBounds.min}
                max={currentBounds.max}
                step={currentBounds.step}
                value={whatIfValue}
                onChange={(e) => handleWhatIfValueChange(Number(e.target.value))}
              />
              <div className="slider-limits">
                <span>{currentBounds.min}</span>
                <span>{currentBounds.max}</span>
              </div>
            </div>

            <div className="whatif-num-input-wrap">
              <input
                type="number"
                className="input-number-field"
                min={currentBounds.min}
                max={currentBounds.max}
                step={currentBounds.step}
                value={whatIfValue}
                onChange={(e) => handleWhatIfValueChange(e.target.value)}
              />
              {currentBounds.unit ? <span className="number-unit-suffix">{currentBounds.unit}</span> : null}
            </div>
          </div>
        </div>

        {/* Side-by-Side Comparison */}
        <div className="comparison-grid">
          {/* Baseline Card */}
          <div className="comparison-card baseline-card">
            <div className="comparison-card-badge">Baseline Input</div>
            <div className="comparison-metric-group">
              <span className="comparison-sublabel">Verdict</span>
              <div className="comparison-verdict">
                <VerdictBadge verdict={baselinePrediction?.verdict} />
              </div>
            </div>
            <div className="comparison-metric-group">
              <span className="comparison-sublabel">Top Crop</span>
              <strong className="comparison-crop-name">
                {baselinePrediction?.top_crops?.[0]
                  ? `${baselinePrediction.top_crops[0].crop} (${Math.round(
                    baselinePrediction.top_crops[0].probability * 100
                  )}%)`
                  : 'None (Refused)'}
              </strong>
            </div>
            <div className="comparison-metric-group">
              <span className="comparison-sublabel">Trust Score</span>
              <span className="comparison-trust-val">
                {baselinePrediction?.trust_score ?? 0} <small>/ 100</small>
              </span>
            </div>
          </div>

          {/* What-If Card */}
          <div className="comparison-card whatif-card">
            <div className="comparison-card-badge whatif-badge-active">
              What-If ({currentBounds.label}: {whatIfValue}{currentBounds.unit ? ` ${currentBounds.unit}` : ''})
            </div>

            {whatIfLoading ? (
              <div className="comparison-loading">
                <div className="spinner-sm"></div>
                <span>Evaluating scenario...</span>
              </div>
            ) : whatIfError ? (
              <div className="comparison-error">
                <span>⚠️ {whatIfError}</span>
              </div>
            ) : (
              <>
                <div className="comparison-metric-group">
                  <span className="comparison-sublabel">Verdict</span>
                  <div className="comparison-verdict">
                    <VerdictBadge verdict={whatIfPrediction?.verdict} />
                  </div>
                </div>
                <div className="comparison-metric-group">
                  <span className="comparison-sublabel">Top Crop</span>
                  <strong className="comparison-crop-name">
                    {whatIfPrediction?.top_crops?.[0]
                      ? `${whatIfPrediction.top_crops[0].crop} (${Math.round(
                        whatIfPrediction.top_crops[0].probability * 100
                      )}%)`
                      : 'None (Refused)'}
                  </strong>
                </div>
                <div className="comparison-metric-group">
                  <span className="comparison-sublabel">Trust Score</span>
                  <span className="comparison-trust-val">
                    {whatIfPrediction?.trust_score ?? 0} <small>/ 100</small>
                  </span>
                </div>
              </>
            )}
          </div>
        </div>

        {/* One-Line "What Changed" Summary */}
        {changeSummary && !whatIfLoading && !whatIfError && (
          <div className="what-changed-banner">
            <span className="what-changed-icon">🔍</span>
            <span>
              <strong>Outcome:</strong> {changeSummary}
            </span>
          </div>
        )}
      </div>

      {/* 2. Climate Stress Test Section */}
      <div className="climate-stress-section">
        <div className="stress-section-header">
          <div>
            <h3 className="stress-title">⚡ Climate Stress Test</h3>
            <p className="stress-desc">
              Test resilience against 3 environmental scenarios (temperature & rainfall perturbation).
            </p>
          </div>
          <button
            type="button"
            className="btn-run-stress"
            onClick={runClimateStressTest}
            disabled={stressLoading}
          >
            {stressLoading ? 'Running Scenarios...' : 'Run Stress Test'}
          </button>
        </div>

        {/* Scrollable Table Container */}
        <div className="table-scroll-container">
          <table className="stress-table">
            <thead>
              <tr>
                <th>Scenario</th>
                <th>Verdict</th>
                <th>Top Crop</th>
                <th>Trust Score</th>
              </tr>
            </thead>
            <tbody>
              {CLIMATE_SCENARIOS.map((sc) => {
                const result = stressResults?.find((r) => r.id === sc.id);

                return (
                  <tr key={sc.id}>
                    <td className="table-scenario-cell">
                      <span className="table-scenario-name">{sc.name}</span>
                      <span className="table-scenario-desc">{sc.description}</span>
                    </td>
                    <td>
                      {stressLoading ? (
                        <span className="cell-loading">...</span>
                      ) : !result ? (
                        <span className="cell-idle">—</span>
                      ) : result.success ? (
                        <VerdictBadge verdict={result.data.verdict} />
                      ) : (
                        <span className="table-row-error">⚠️ {result.error}</span>
                      )}
                    </td>
                    <td>
                      {stressLoading ? (
                        <span className="cell-loading">...</span>
                      ) : !result ? (
                        <span className="cell-idle">—</span>
                      ) : result.success ? (
                        result.data.top_crops?.[0] ? (
                          <span>
                            <strong>{result.data.top_crops[0].crop}</strong> (
                            {Math.round(result.data.top_crops[0].probability * 100)}%)
                          </span>
                        ) : (
                          <span className="text-muted">None (Refused)</span>
                        )
                      ) : (
                        <span className="table-row-error">—</span>
                      )}
                    </td>
                    <td>
                      {stressLoading ? (
                        <span className="cell-loading">...</span>
                      ) : !result ? (
                        <span className="cell-idle">—</span>
                      ) : result.success ? (
                        <span className="table-trust-badge">{result.data.trust_score ?? 0}</span>
                      ) : (
                        <span className="table-row-error">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Mandatory Note */}
        <p className="climate-forecast-note">
          ℹ️ <em>Scenarios are what-if tests, not climate forecasts.</em>
        </p>
      </div>
    </div>
  );
}

