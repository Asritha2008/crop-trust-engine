import React, { useState, useEffect, useRef } from 'react';
import { PHYSICAL_BOUNDS, DEMO_PRESET_INPUTS } from '../config';
import mockExamples from '../../mock/predict_examples.json';
import { predictCrop } from '../api/client';
import InputControl from './InputControl';
import VerdictBanner from './VerdictBanner';
import TrustMeter from './TrustMeter';
import TopCropsChart from './TopCropsChart';
import ExplanationCard from './ExplanationCard';
import { downloadReport } from '../utils/downloadReport';

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

  // Prediction and network state
  const [prediction, setPrediction] = useState(presets.recommend.response);
  const [loading, setLoading] = useState(false);
  const [isWakingUp, setIsWakingUp] = useState(false);
  const [error, setError] = useState(null);
  const [fieldErrors, setFieldErrors] = useState({});

  const abortControllerRef = useRef(null);

  // Load a preset
  function handleSelectPreset(key) {
    setActivePreset(key);
    if (presets[key]) {
      setInputs({ ...presets[key].input });
      setRiskProfile(presets[key].risk || 'balanced');
      executePrediction(presets[key].input, presets[key].risk || 'balanced');
    }
  }

  // Handle single input change
  function handleInputChange(field, value) {
    setActivePreset(null);
    setInputs((prev) => ({
      ...prev,
      [field]: value
    }));

    // Clear field-level error on change
    if (fieldErrors[field]) {
      setFieldErrors((prev) => {
        const updated = { ...prev };
        delete updated[field];
        return updated;
      });
    }
  }

  // Execute prediction call with debouncing and cancellation
  async function executePrediction(targetInputs = inputs, targetRisk = riskProfile) {
    // Cancel any in-flight request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setLoading(true);
    setError(null);
    setFieldErrors({});
    setIsWakingUp(false);

    const wakeTimer = setTimeout(() => setIsWakingUp(true), 3500);

    try {
      const res = await predictCrop(targetInputs, targetRisk, abortControllerRef.current.signal);
      clearTimeout(wakeTimer);
      setPrediction(res);
    } catch (err) {
      clearTimeout(wakeTimer);
      if (err.name === 'AbortError') return;

      if (err.status === 422 && err.fieldErrors) {
        setFieldErrors(err.fieldErrors);
        setError('Please correct the highlighted input fields outside physical boundaries.');
      } else {
        setError(err.message || 'Failed to obtain recommendation.');
      }
    } finally {
      setLoading(false);
    }
  }

  // Handle risk profile switch
  function handleRiskChange(profile) {
    setRiskProfile(profile);
    executePrediction(inputs, profile);
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
            onClick={() => executePrediction()}
            disabled={loading}
          >
            {loading ? 'Evaluating Trust Engine...' : 'Run Trust Engine Assessment'}
          </button>
        </div>
      </section>

      {/* Right Column: Trust Engine Results */}
      <section className="results-column">
        {loading && (
          <div className="loading-state-card">
            <div className="spinner"></div>
            <strong>Running 5-Stage Assessment...</strong>
            <p className="loading-subtext">Validation → OOD Gate → Calibration → Stability → Verdict</p>
            {isWakingUp && (
              <div className="waking-alert">
                ⏳ <strong>Waking Server:</strong> Free cloud tiers sleep when idle. Please allow up to 60 seconds...
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="error-state-card" role="alert">
            <div className="error-title">
              <span>⚠️</span>
              <strong>Assessment Error</strong>
            </div>
            <p className="error-desc">{error}</p>
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

            {/* 3. Reasons third: Expandable Why */}
            <ExplanationCard
              explanation={prediction.explanation}
              measureNext={prediction.measure_next}
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
              {onNavigateToWhatIf && (
                <button
                  type="button"
                  className="btn-climate-stress"
                  onClick={onNavigateToWhatIf}
                >
                  ⚡ Climate Stress Test
                </button>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
