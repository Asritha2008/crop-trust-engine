import React from 'react';

export default function VerdictBanner({ verdict, topCrop }) {
  if (!verdict) return null;

  const vLower = verdict.toLowerCase();

  const configs = {
    recommend: {
      icon: '✅',
      title: topCrop ? `RECOMMEND: ${topCrop.toUpperCase()}` : 'RECOMMEND',
      subtitle: 'Input conditions are familiar and model prediction is stable.',
      className: 'verdict-banner-recommend',
      tag: 'PASS'
    },
    caution: {
      icon: '⚠️',
      title: 'CAUTION: TWO CROPS ARE CLOSE',
      subtitle: 'Input conditions are near boundary or sensitive to sensor noise.',
      className: 'verdict-banner-caution',
      tag: 'CAUTION'
    },
    refuse: {
      icon: '🛑',
      title: 'REFUSE: UNUSUAL INPUTS',
      subtitle: "These conditions don't match anything in the training data. No recommendation made.",
      className: 'verdict-banner-refuse',
      tag: 'REFUSED'
    }
  };

  const current = configs[vLower] || {
    icon: 'ℹ️',
    title: verdict.toUpperCase(),
    subtitle: 'Verdict issued by trust engine',
    className: 'verdict-banner-neutral',
    tag: 'STATUS'
  };

  return (
    <div className={`verdict-banner-container ${current.className}`} role="status">
      <div className="verdict-banner-main">
        <div className="verdict-banner-icon" aria-hidden="true">
          {current.icon}
        </div>
        <div className="verdict-banner-content">
          <div className="verdict-banner-tag">{current.tag}</div>
          <h2 className="verdict-banner-headline">{current.title}</h2>
          <p className="verdict-banner-desc">{current.subtitle}</p>
        </div>
      </div>
    </div>
  );
}
