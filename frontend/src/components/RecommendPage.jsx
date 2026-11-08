import React, { useState, useEffect, useRef } from 'react';
import { PHYSICAL_BOUNDS, DEMO_PRESET_INPUTS } from '../config';
import mockExamples from '../../mock/predict_examples.json';
import { predictCrop, isMockEnabled } from '../api/client';
import InputControl from './InputControl';
import VerdictBanner from './VerdictBanner';
import TrustMeter from './TrustMeter';
import TopCropsChart from './TopCropsChart';
import ExplanationCard from './ExplanationCard';
import MeasureNextCard from './MeasureNextCard';
import ComfortZoneBars from './ComfortZoneBars';
import NutrientGapCard from './NutrientGapCard';
import WhatIfPanel from './WhatIfPanel';
import { downloadReport } from '../utils/downloadReport';
import { checkFamiliarity } from '../utils/oodBrowser';

export default function RecommendPage({ onNavigateToWhatIf }) {
  // Build presets from DEMO_PRESET_INPUTS and mockExamples responses
  // PLACEHOLDER: replace with value from Role 1 (class_stats.json / data plots)
  const presets = {};
  ['recommend', 'caution', 'refuse'].forEach((key) => {
    presets[key] = {
      input: DEMO_PRESET_INPUTS[key],
      response: mockExamples[key]
    };
  });

  // Inputs state
  const [inputs, setInputs] = useState(presets.recommend.input);
  const [riskProfile, setRiskProfile] = useState('balanced');
  const [activePreset, setActivePreset] = useState('recommend');

  // Task 9: In-browser familiarity quick check (evaluates synchronously as inputs/risk change)
  const quickCheck = checkFamiliarity(inputs, riskProfile);

  // Prediction and network state
  const [prediction, setPrediction] = useState(presets.recommend.response);
  const [loading, setLoading] = useState(false);
  const [isWakingUp, setIsWakingUp] = useState(false);
  const [error, setError] = useState(null);
  const [fieldErrors, setFieldErrors] = useState({});

  const abortControllerRef = useRef(null);
  const debounceTimerRef = useRef(null);

  // Clean up timers & controller on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Execute prediction call with debouncing and cancellation
  async function executePrediction(targetInputs = inputs, targetRisk = riskProfile) {
    // 4. Abort any in-flight request with AbortController
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    setLoading(true);
    setError(null);
    setIsWakingUp(false);

    // 3. Waking server note: if live request takes longer than 3 seconds (never in mock mode)
    let wakeTimer = null;
    if (!isMockEnabled()) {
      wakeTimer = setTimeout(() => {
        setIsWakingUp(true);
      }, 3000);
    }

    try {
      const res = await predictCrop(targetInputs, targetRisk, signal);

      // Stale response check: ignore if this request was aborted
      if (signal.aborted) return;

      if (wakeTimer) clearTimeout(wakeTimer);
      setIsWakingUp(false);

      setPrediction(res);
      setError(null);
      setFieldErrors({});
    } catch (err) {
      if (wakeTimer) clearTimeout(wakeTimer);
      setIsWakingUp(false);

      // 4. Ignore AbortError (not a user-facing error)
      if (err.name === 'AbortError' || signal.aborted) {
        return;
      }

      // 2. Inline field errors: pass each message to matching InputControl
      if (err.fieldErrors && Object.keys(err.fieldErrors).length > 0) {
        setFieldErrors(err.fieldErrors);
        setError(null);
      } else {
        // 1. Error banner: friendly wording for network / >= 500 errors
        let friendlyMessage = 'Failed to obtain recommendation.';
        if (err.isNetwork) {
          friendlyMessage = "Couldn't reach the server";
        } else if (err.status >= 500) {
          friendlyMessage = 'Something went wrong on our side';
        } else if (typeof err.message === 'string' && err.message !== '[object Object]') {
          friendlyMessage = err.message;
        }

        setError({
          message: friendlyMessage,
          status: err.status,
          isNetwork: Boolean(err.isNetwork)
        });
      }
    } finally {
      if (!signal.aborted) {
        setLoading(false);
      }
    }
  }

  // Load a preset immediately
  function handleSelectPreset(key) {
    setActivePreset(key);
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    if (presets[key]) {
      const newInputs = { ...presets[key].input };
      const newRisk = presets[key].risk || 'balanced';
      setInputs(newInputs);
      setRiskProfile(newRisk);
      executePrediction(newInputs, newRisk);
    }
  }

  // 4. Debounce: when inputs change, wait 300 ms before calling predictCrop
  function handleInputChange(field, value) {
    setActivePreset(null);

    const updatedInputs = {
      ...inputs,
      [field]: value
    };
    setInputs(updatedInputs);

    // 2. Clear field's error when user edits that field
    if (fieldErrors[field]) {
      setFieldErrors((prev) => {
        const next = { ...prev };
        delete next[field];
        return next;
      });
    }

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      executePrediction(updatedInputs, riskProfile);
    }, 300);
  }

  // Handle risk profile switch
  function handleRiskChange(profile) {
    setRiskProfile(profile);
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    executePrediction(inputs, profile);
  }

  // Manual run / "Try again" handler
  function handleManualRun() {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    executePrediction(inputs, riskProfile);
  }

  // Initial load
  useEffect(() => {
    executePrediction(presets.recommend.input, 'balanced');
  }, []);

  return (
    <div className="recommend-layout">
      {/* Left Column: Form & Controls */}
      <section className="form-column">
        {/* Presets Bar */}
        <div className="presets-container">
          <span className="presets-heading">Demo Presets:</span>
          <div className="presets-buttons">
            <button
              type="button"
              className={`preset-pill ${activePreset === 'recommend' ? 'active' : ''}`}
              onClick={() => handleSelectPreset('recommend')}
            >
              ✅ Typical (Rice)
            </button>
            <button
              type="button"
              className={`preset-pill ${activePreset === 'caution' ? 'active' : ''}`}
              onClick={() => handleSelectPreset('caution')}
            >
              ⚠️ Ambiguous (Caution)
            </button>
            <button
              type="button"
              className={`preset-pill ${activePreset === 'refuse' ? 'active' : ''}`}
              onClick={() => handleSelectPreset('refuse')}
            >
              🛑 Impossible (OOD Refusal)
            </button>
          </div>
        </div>

        {/* 7 Inputs Grid */}
        <div className="inputs-container">
          <div className="inputs-header">
            <h2 className="inputs-title">Soil & Environmental Conditions</h2>
            <span className="inputs-subtitle">7 Physical Parameters</span>
          </div>

          {/* Task 9: In-Browser Familiarity Quick Check Pill (updates as sliders move) */}
          <div
            className="quick-check-panel"
            role="status"
            aria-live="polite"
          >
            <span className="quick-check-tagline">
              Quick check (distance only) - estimate, the server decides
            </span>
            <div className={`quick-check-pill quick-check-${quickCheck.status}`}>
              <span className="quick-check-icon" aria-hidden="true">{quickCheck.icon}</span>
              <span className="quick-check-word">{quickCheck.label}</span>
              {quickCheck.distance !== null && (
                <span className="quick-check-dist">(d² = {quickCheck.distance.toFixed(1)})</span>
              )}
            </div>
          </div>

          <div className="controls-list">
            {Object.entries(PHYSICAL_BOUNDS).map(([field, bounds]) => (
              <InputControl
                key={field}
                field={field}
                bounds={bounds}
                value={inputs[field] ?? ''}
                onChange={handleInputChange}
                error={fieldErrors[field]}
              />
            ))}
          </div>
        </div>

        {/* Risk Profile Switcher */}
        <div className="risk-profile-card">
          <div className="risk-header">
            <label className="risk-title">Risk Appetite Profile</label>
            <span className="risk-hint">Controls OOD & stability sensitivity</span>
          </div>
          <div className="risk-segmented-control" role="group" aria-label="Risk Appetite">
            {[
              { id: 'conservative', label: 'Conservative', desc: 'Readily hedges & refuses' },
              { id: 'balanced', label: 'Balanced', desc: 'Standard thresholds' },
              { id: 'bold', label: 'Bold', desc: 'Maximizes recommendations' }
            ].map((p) => (
              <button
                key={p.id}
                type="button"
                className={`risk-btn ${riskProfile === p.id ? 'active' : ''}`}
                onClick={() => handleRiskChange(p.id)}
              >
                <span className="risk-btn-label">{p.label}</span>
                <span className="risk-btn-desc">{p.desc}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Action Button */}
        <div className="form-actions">
          <button
            type="button"
            className="btn-submit-predict"
            onClick={handleManualRun}
            disabled={loading}
          >
            {loading ? 'Evaluating Trust Engine...' : 'Run Trust Engine Assessment'}
          </button>
        </div>
      </section>

      {/* Right Column: Trust Engine Results */}
      <section className="results-column">
        {/* 1. Error banner: shown above results when predictCrop throws an error with no fieldErrors */}
        {error && (
          <div className="error-state-card" role="alert">
            <div className="error-content">
              <span className="error-icon">⚠️</span>
              <div className="error-text-wrap">
                <strong className="error-heading">Assessment Notice</strong>
                <p className="error-message">{error.message}</p>
              </div>
            </div>
            <button
              type="button"
              className="btn-retry"
              onClick={handleManualRun}
              disabled={loading}
            >
              Try again
            </button>
          </div>
        )}

        {/* 3. Loading state with "Waking server" note */}
        {loading && (
          <div className="loading-state-card" role="status">
            <div className="spinner"></div>
            <strong>Running 5-Stage Assessment...</strong>
            <p className="loading-subtext">Validation → OOD Gate → Calibration → Stability → Verdict</p>
            {isWakingUp && (
              <div className="waking-alert">
                ⏳ <strong>Waking the server up, this can take up to a minute on the free plan...</strong>
              </div>
            )}
          </div>
        )}

        {prediction && !loading && (
          <div className="prediction-results-card">
            {/* 1. Verdict Banner (Verdict first, never colour alone) */}
            <VerdictBanner
              verdict={prediction.verdict}
              topCrop={prediction.top_crops?.[0]?.crop}
            />

            {/* 2. Numbers second: Trust Meter & Stability */}
            <TrustMeter
              trustScore={prediction.trust_score}
              stability={prediction.stability}
              ood={prediction.ood}
            />

            {/* Top Crops Calibrated Bars */}
            <TopCropsChart
              topCrops={prediction.top_crops}
              conformalSet={prediction.conformal_set}
              isRefusal={prediction.verdict === 'refuse'}
            />

            {/* Task 6: "What to measure next" in its own card only when not null */}
            {prediction.measure_next && (
              <MeasureNextCard measureNext={prediction.measure_next} />
            )}

            {/* Task 6: Comfort-Zone Bars and Nutrient Gap View (hidden on refuse / when no top crop) */}
            {prediction.verdict !== 'refuse' && prediction.top_crops && prediction.top_crops.length > 0 && (
              <>
                <ComfortZoneBars
                  crop={prediction.top_crops[0].crop}
                  inputs={inputs}
                />
                <NutrientGapCard
                  crop={prediction.top_crops[0].crop}
                  inputs={inputs}
                />
              </>
            )}

            {/* 3. Reasons third: Expandable Why */}
            <ExplanationCard
              explanation={prediction.explanation}
              ood={prediction.ood}
              modelVersion={prediction.model_version}
            />

            {/* Results Actions */}
            <div className="results-action-row">
              <button
                type="button"
                className="btn-download-report"
                onClick={() => downloadReport(inputs, prediction, riskProfile)}
              >
                📥 Download Full Report (.txt)
              </button>
              <button
                type="button"
                className="btn-climate-stress"
                onClick={() => {
                  const el = document.getElementById('what-if-panel-section');
                  if (el) el.scrollIntoView({ behavior: 'smooth' });
                  else if (onNavigateToWhatIf) onNavigateToWhatIf();
                }}
              >
                ⚡ Climate Stress Test
              </button>
            </div>

            {/* Task 5: What-If Simulator & Climate Stress Test */}
            <div id="what-if-panel-section">
              <WhatIfPanel
                baselineInputs={inputs}
                baselinePrediction={prediction}
                riskProfile={riskProfile}
              />
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
