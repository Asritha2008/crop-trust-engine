import React from 'react';

/**
 * Compact verdict badge following the rule:
 * Verdict first: icon + word + colour, never colour alone.
 */
export default function VerdictBadge({ verdict }) {
  if (!verdict) return <span className="verdict-pill verdict-pill-neutral">Unknown</span>;

  const vLower = verdict.toLowerCase();

  const configs = {
    recommend: {
      icon: '✅',
      label: 'RECOMMEND',
      className: 'verdict-pill-recommend'
    },
    caution: {
      icon: '⚠️',
      label: 'CAUTION',
      className: 'verdict-pill-caution'
    },
    refuse: {
      icon: '🛑',
      label: 'REFUSE',
      className: 'verdict-pill-refuse'
    }
  };

  const current = configs[vLower] || {
    icon: 'ℹ️',
    label: verdict.toUpperCase(),
    className: 'verdict-pill-neutral'
  };

  return (
    <span className={`verdict-pill ${current.className}`} role="status">
      <span className="verdict-pill-icon" aria-hidden="true">{current.icon}</span>
      <span className="verdict-pill-label">{current.label}</span>
    </span>
  );
}

