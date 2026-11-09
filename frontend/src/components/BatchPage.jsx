import React, { useState, useRef, useEffect } from 'react';
import { DEMO_PRESET_INPUTS, BATCH_MAX_BYTES, BATCH_MAX_ROWS } from '../config';
import { predictBatch, isMockEnabled } from '../api/client';
import VerdictBadge from './VerdictBadge';

const RISK_PROFILES = [
  { id: 'conservative', label: 'Conservative', desc: 'Warns and refuses more often' },
  { id: 'balanced', label: 'Balanced', desc: 'Standard thresholds' },
  { id: 'bold', label: 'Bold', desc: 'Maximizes recommendations' }
];

/**
 * Task 8: Batch CSV Assessment Page
 * Handles CSV upload, client-side validation, loading/cancellation states,
 * result tabulation with row-level error reporting, flagging, and CSV export.
 */
export default function BatchPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [riskProfile, setRiskProfile] = useState('balanced');
  const [sizeError, setSizeError] = useState(null);

  // Loading & network state (Piece 4)
  const [loading, setLoading] = useState(false);
  const [isWakingUp, setIsWakingUp] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  // Results state (Piece 5)
  const [batchData, setBatchData] = useState(null);
  const [expandedErrorRows, setExpandedErrorRows] = useState({});

  const abortControllerRef = useRef(null);
  const wakeTimerRef = useRef(null);

  // Clean up timers & abort controller on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (wakeTimerRef.current) {
        clearTimeout(wakeTimerRef.current);
      }
    };
  }, []);

  function handleFileSelect(e) {
    const file = e.target.files?.[0] || null;
    setSizeError(null);
    setUploadError(null);

    if (!file) {
      setSelectedFile(null);
      return;
    }

    // Client-side size check against BATCH_MAX_BYTES
    if (file.size > BATCH_MAX_BYTES) {
      const maxMb = BATCH_MAX_BYTES / 1000000;
      const fileMb = (file.size / 1000000).toFixed(2);
      setSizeError(
        `File too large: maximum allowed size is ${maxMb} MB (selected file is ${fileMb} MB). Please choose a smaller CSV.`
      );
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  }

  function handleDownloadTemplate() {
    const rec = DEMO_PRESET_INPUTS.recommend;
    const csvContent =
      'N,P,K,temperature,humidity,ph,rainfall\n' +
      `${rec.N},${rec.P},${rec.K},${rec.temperature},${rec.humidity},${rec.ph},${rec.rainfall}\n`;

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'crop_batch_template.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  async function handleRunBatch() {
    if (!selectedFile) return;

    // Abort any existing in-flight batch request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    setLoading(true);
    setIsWakingUp(false);
    setUploadError(null);
    setBatchData(null);
    setExpandedErrorRows({});

    // Waking server note after 3 seconds in live mode only (never in mock mode)
    if (wakeTimerRef.current) clearTimeout(wakeTimerRef.current);
    if (!isMockEnabled()) {
      wakeTimerRef.current = setTimeout(() => {
        setIsWakingUp(true);
      }, 3000);
    }

    try {
      const res = await predictBatch(selectedFile, riskProfile, signal);

      // Stale response check: ignore if aborted
      if (signal.aborted) return;

      if (wakeTimerRef.current) clearTimeout(wakeTimerRef.current);
      setIsWakingUp(false);

      setBatchData(res);
      setUploadError(null);
    } catch (err) {
      if (wakeTimerRef.current) clearTimeout(wakeTimerRef.current);
      setIsWakingUp(false);

      // Cancelled request shows no error
      if (err.name === 'AbortError' || signal.aborted) {
        return;
      }

      let friendlyMessage = 'Batch assessment failed';
      if (err.isNetwork) {
        friendlyMessage = "Couldn't reach the server. Enable Mock Mode to test with contract fixtures.";
      } else if (err.status >= 500) {
        friendlyMessage = 'Something went wrong on our side';
      } else if (typeof err.message === 'string' && err.message !== '[object Object]') {
        friendlyMessage = err.message;
      }

      setUploadError({
        message: friendlyMessage,
        status: err.status,
        isNetwork: Boolean(err.isNetwork)
      });
    } finally {
      if (!signal.aborted) {
        setLoading(false);
      }
    }
  }

  function handleCancelBatch() {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (wakeTimerRef.current) {
      clearTimeout(wakeTimerRef.current);
    }
    setLoading(false);
    setIsWakingUp(false);
  }

  // Piece 6: Client-side CSV Download
  function handleDownloadResults() {
    if (!batchData || !Array.isArray(batchData.results)) return;

    function escapeCsvCell(val) {
      if (val === null || val === undefined) return '';
      const str = String(val);
      if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r')) {
        return `"${str.replace(/"/g, '""')}"`;
      }
      return str;
    }

    const headers = ['row', 'verdict', 'top_crop', 'probability', 'trust_score', 'error'];
    const lines = [headers.join(',')];

    batchData.results.forEach((r, idx) => {
      const rowNum = (r.row_index ?? idx) + 1;
      const isFailed = Boolean(r && Object.prototype.hasOwnProperty.call(r, 'error'));

      const verdict = isFailed ? 'failed' : (r.verdict || '');
      let topCrop = '';
      let probability = '';

      if (!isFailed) {
        if (r.top_crops && r.top_crops.length > 0) {
          topCrop = r.top_crops[0].crop;
          probability = (Math.round(r.top_crops[0].probability * 100) / 100).toString();
        } else {
          topCrop = 'No recommendation';
        }
      }

      const trustScore = isFailed
        ? ''
        : (r.trust_score !== undefined && r.trust_score !== null ? r.trust_score : 0);
      const errorText = isFailed ? (r.error || 'Unknown error') : '';

      lines.push(
        [
          escapeCsvCell(rowNum),
          escapeCsvCell(verdict),
          escapeCsvCell(topCrop),
          escapeCsvCell(probability),
          escapeCsvCell(trustScore),
          escapeCsvCell(errorText)
        ].join(',')
      );
    });

    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `crop_batch_results_${Date.now()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  function toggleErrorExpand(rowNum) {
    setExpandedErrorRows((prev) => ({
      ...prev,
      [rowNum]: !prev[rowNum]
    }));
  }

  // Summary counts computed from results
  const results = batchData?.results || [];
  const totalRows = results.length;
  const failedRows = results.filter((r) => r && Object.prototype.hasOwnProperty.call(r, 'error')).length;
  const okRows = totalRows - failedRows;

  /**
   * Shows the received error text exactly without repeating "Row N:" if the row is already identified.
   */
  function getCleanErrorText(error, rowNum) {
    if (error === null || error === undefined) return 'Unknown error';
    const str = typeof error === 'string' ? error : JSON.stringify(error);
    const regex = new RegExp(`^Row\\s*#?\\d+\\s*[:\\-]\\s*`, 'i');
    const cleaned = str.replace(regex, '').trim();
    return cleaned || str;
  }

  return (
    <div className="batch-page-container">
      {/* Header & Template Download */}
      <div className="batch-header-card">
        <div className="batch-title-group">
          <span className="batch-header-icon">📁</span>
          <div>
            <h2 className="batch-title">Batch CSV Assessment</h2>
            <p className="batch-subtitle">
              Upload multiple soil sample readings (up to {BATCH_MAX_ROWS} rows, max {BATCH_MAX_BYTES / 1000000} MB)
              for selective Trust Engine prediction.
            </p>
          </div>
        </div>

        <button
          type="button"
          className="btn-download-template"
          onClick={handleDownloadTemplate}
        >
          📥 Download Template (.csv)
        </button>
      </div>

      {/* Upload Controls Card */}
      <div className="batch-controls-card">
        <div className="batch-file-picker-group">
          <label htmlFor="batch-file-input" className="batch-field-label">
            Select CSV File (.csv)
          </label>
          <input
            id="batch-file-input"
            type="file"
            accept=".csv"
            className="batch-file-input"
            onChange={handleFileSelect}
            disabled={loading}
          />
          {selectedFile && (
            <div className="batch-selected-file-meta">
              <span className="file-name">📄 {selectedFile.name}</span>
              <span className="file-size">({(selectedFile.size / 1024).toFixed(1)} KB)</span>
            </div>
          )}
        </div>

        {/* Client-side Size Validation Error */}
        {sizeError && (
          <div className="batch-error-banner" role="alert">
            <span className="error-icon">⚠️</span>
            <span>{sizeError}</span>
          </div>
        )}

        {/* Risk Appetite Selector */}
        <div className="risk-profile-card">
          <div className="risk-header">
            <label className="risk-title">Risk Appetite Profile</label>
            <span className="risk-hint">Controls how cautious the advice is</span>
          </div>
          <div className="risk-segmented-control" role="group" aria-label="Risk Appetite">
            {RISK_PROFILES.map((p) => (
              <button
                key={p.id}
                type="button"
                className={`risk-btn ${riskProfile === p.id ? 'active' : ''}`}
                onClick={() => setRiskProfile(p.id)}
                disabled={loading}
              >
                <span className="risk-btn-label">{p.label}</span>
                <span className="risk-btn-desc">{p.desc}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Run Button (disabled until file is selected and valid, or when loading) */}
        <div className="batch-actions-row">
          <button
            type="button"
            className="btn-submit-predict btn-run-batch"
            disabled={!selectedFile || Boolean(sizeError) || loading}
            onClick={handleRunBatch}
          >
            {loading ? 'Evaluating Batch...' : 'Run Batch Assessment'}
          </button>
        </div>
      </div>

      {/* Piece 4: Loading State with Waking Note & Cancel Button */}
      {loading && (
        <div className="batch-loading-card" role="status">
          <div className="spinner"></div>
          <p className="loading-headline">
            Assessing rows... this can take about 10 seconds for {BATCH_MAX_ROWS} rows
          </p>
          {isWakingUp && (
            <div className="waking-alert">
              ⏳ <strong>Waking the server up, this can take up to a minute on the free plan...</strong>
            </div>
          )}
          <button
            type="button"
            className="btn-cancel-batch"
            onClick={handleCancelBatch}
          >
            ✕ Cancel Assessment
          </button>
        </div>
      )}

      {/* Piece 5: Whole-Upload Error Banner */}
      {uploadError && !loading && (
        <div className="batch-whole-error-card" role="alert">
          <div className="whole-error-content">
            <span className="error-icon">⚠️</span>
            <div className="whole-error-text-wrap">
              <strong className="whole-error-heading">Upload Assessment Notice</strong>
              <p className="whole-error-msg">{uploadError.message}</p>
            </div>
          </div>
          <button
            type="button"
            className="btn-retry-batch"
            onClick={handleRunBatch}
          >
            Try again
          </button>
        </div>
      )}

      {/* Piece 5 & 6: Batch Results Table */}
      {batchData && !loading && (
        <div className="batch-results-card">
          <div className="batch-results-header">
            <div className="batch-summary-info">
              <h3 className="batch-results-title">Assessment Results</h3>
              <p className="batch-summary-text">
                <strong>{totalRows} rows: {okRows} ok, {failedRows} failed</strong>
                {batchData.model_version && (
                  <span className="batch-version-tag"> ({batchData.model_version})</span>
                )}
              </p>
            </div>

            <button
              type="button"
              className="btn-download-results"
              onClick={handleDownloadResults}
            >
              📥 Download results (.csv)
            </button>
          </div>

          {/* Mobile-First Scrollable Table Container */}
          <div className="batch-table-container">
            <table className="batch-table">
              <thead>
                <tr>
                  <th className="th-row">Row</th>
                  <th className="th-verdict">Verdict</th>
                  <th className="th-crop">Top Crop with Probability</th>
                  <th className="th-trust">Trust Score</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, index) => {
                  const rowNum = (r.row_index ?? index) + 1;
                  const isFailed = Boolean(r && Object.prototype.hasOwnProperty.call(r, 'error'));

                  if (isFailed) {
                    const cleanMsg = getCleanErrorText(r.error, rowNum);
                    const isLong = cleanMsg.length > 120;
                    const isExpanded = Boolean(expandedErrorRows[rowNum]);
                    const displayMsg = isLong && !isExpanded ? `${cleanMsg.slice(0, 120)}...` : cleanMsg;

                    return (
                      <tr key={`row-${rowNum}`} className="batch-row-failed">
                        <td colSpan={4} className="cell-failed-full">
                          <div className="failed-error-wrap">
                            <span className="failed-icon" aria-hidden="true">❌</span>
                            <strong className="failed-row-tag">Row {rowNum}:</strong>
                            <span className="failed-error-text">{displayMsg}</span>
                            {isLong && (
                              <button
                                type="button"
                                className="btn-toggle-error-msg"
                                onClick={() => toggleErrorExpand(rowNum)}
                                aria-expanded={isExpanded}
                              >
                                {isExpanded ? 'Show less' : 'Show full message'}
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  }

                  const vLower = (r.verdict || '').toLowerCase();
                  const isCaution = vLower === 'caution';
                  const isRefusal = vLower === 'refuse';

                  let rowModifierClass = 'batch-row-recommend';
                  if (isCaution) rowModifierClass = 'batch-row-caution';
                  if (isRefusal) rowModifierClass = 'batch-row-refuse';

                  const topCropObj = r.top_crops?.[0];
                  const hasTopCrop = Boolean(topCropObj && topCropObj.crop);

                  return (
                    <tr key={`row-${rowNum}`} className={`batch-row-item ${rowModifierClass}`}>
                      <td className="cell-row-num">
                        <strong>#{rowNum}</strong>
                      </td>
                      <td className="cell-verdict">
                        <div className="verdict-cell-wrap">
                          <VerdictBadge verdict={r.verdict} />
                          {isCaution && (
                            <span className="row-flag-badge flag-caution">
                              ⚠️ Caution (High Ambiguity)
                            </span>
                          )}
                          {isRefusal && (
                            <span className="row-flag-badge flag-refuse">
                              🛑 Refused (unusual combination)
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="cell-crop">
                        {hasTopCrop ? (
                          <span>
                            <strong>{topCropObj.crop}</strong> (
                            {Math.round(topCropObj.probability * 100)}%)
                          </span>
                        ) : (
                          <span className="text-muted">No recommendation</span>
                        )}
                      </td>
                      <td className="cell-trust">
                        <span className="table-trust-badge">
                          {r.trust_score !== undefined && r.trust_score !== null
                            ? r.trust_score
                            : 0}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
