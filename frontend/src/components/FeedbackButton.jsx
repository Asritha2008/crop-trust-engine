import React, { useState, useEffect } from 'react';

/**
 * Task 10: "Was this wrong?" user feedback component.
 * Allows users to report unexpected model outputs and saves a local text report
 * without collecting any personal data.
 */
export default function FeedbackButton({ inputs, prediction, riskProfile }) {
  const [isOpen, setIsOpen] = useState(false);
  const [comment, setComment] = useState('');
  const [submitted, setSubmitted] = useState(false);

  // Close the feedback box on Escape key for keyboard accessibility
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape' && isOpen) {
        handleCancel();
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  function handleCancel() {
    setIsOpen(false);
    setComment('');
    setSubmitted(false);
  }

  function handleReset() {
    setIsOpen(false);
    setComment('');
    setSubmitted(false);
  }

  function handleDownloadFeedback() {
    const timestamp = new Date().toISOString().replace('T', ' ').substring(0, 19);
    const verdictText = prediction?.verdict || 'N/A';
    const trustScoreText = prediction?.trust_score !== null && prediction?.trust_score !== undefined
      ? `${prediction.trust_score} / 100`
      : 'N/A';
    const modelVersionText = prediction?.model_version || 'N/A';

    const lines = [
      '========================================================================',
      '                   CROP TRUST ENGINE - USER FEEDBACK                    ',
      '========================================================================',
      `Time:          ${timestamp}`,
      `Model Version: ${modelVersionText}`,
      `Risk Setting:  ${riskProfile || 'N/A'}`,
      `Verdict:       ${verdictText}`,
      `Trust Score:   ${trustScoreText}`,
      '',
      'Input Values:',
      `  Nitrogen (N):     ${inputs?.N ?? 'N/A'}`,
      `  Phosphorus (P):   ${inputs?.P ?? 'N/A'}`,
      `  Potassium (K):    ${inputs?.K ?? 'N/A'}`,
      `  Temperature:      ${inputs?.temperature ?? 'N/A'}`,
      `  Humidity:         ${inputs?.humidity ?? 'N/A'}`,
      `  Soil pH:          ${inputs?.ph ?? 'N/A'}`,
      `  Rainfall:         ${inputs?.rainfall ?? 'N/A'}`,
      '',
      'Comment:',
      comment.trim() ? comment.trim() : '(No comment provided)',
      '========================================================================'
    ];

    const content = lines.join('\n');
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `crop-trust-feedback-${Date.now()}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    setSubmitted(true);
  }

  return (
    <div className="feedback-section">
      {!isOpen ? (
        <button
          type="button"
          className="btn-feedback-trigger"
          onClick={() => {
            setIsOpen(true);
            setSubmitted(false);
          }}
          aria-expanded={isOpen}
        >
          <span className="feedback-icon" aria-hidden="true">💬</span>
          <span>Was this wrong?</span>
        </button>
      ) : (
        <div
          className="feedback-box"
          role="region"
          aria-label="Feedback form"
        >
          {!submitted ? (
            <>
              <label htmlFor="feedback-comment-input" className="feedback-label">
                What looks wrong?
              </label>
              <textarea
                id="feedback-comment-input"
                className="feedback-textarea"
                value={comment}
                onChange={(e) => setComment(e.target.value.slice(0, 300))}
                maxLength={300}
                rows={3}
                placeholder="Describe what looks incorrect (optional, max 300 characters)..."
              />
              <div className="feedback-char-count">
                {comment.length} / 300
              </div>
              <div className="feedback-actions">
                <button
                  type="button"
                  className="btn-feedback-cancel"
                  onClick={handleCancel}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="btn-feedback-download"
                  onClick={handleDownloadFeedback}
                >
                  Download feedback
                </button>
              </div>
            </>
          ) : (
            <div className="feedback-success-card" role="status">
              <span className="feedback-success-text">
                Thanks, feedback saved to your downloads.
              </span>
              <button
                type="button"
                className="btn-feedback-close"
                onClick={handleReset}
              >
                Close
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

