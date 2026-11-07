import React from 'react';

export default function Navbar({ activeTab, onSelectTab, isMock, onToggleMock, currentLang, onSelectLang }) {
  const tabs = [
    { id: 'recommend', label: 'Recommend' },
    { id: 'what-if', label: 'What-If' },
    { id: 'arena', label: 'Model Arena' },
    { id: 'batch', label: 'Batch' },
    { id: 'model-card', label: 'Model Card' }
  ];

  const languages = [
    { code: 'en', label: 'EN' },
    { code: 'te', label: 'TE' },
    { code: 'hi', label: 'HI' }
  ];

  return (
    <header className="navbar-container">
      <div className="navbar-top">
        <div className="navbar-brand">
          <span className="navbar-logo">🌱</span>
          <div>
            <h1 className="navbar-title">Crop Trust Engine</h1>
            <span className="navbar-tagline">Selective Prediction & OOD Gate</span>
          </div>
        </div>

        <div className="navbar-actions">
          {/* Mock / Live Toggle */}
          <button
            type="button"
            className={`badge-toggle ${isMock ? 'badge-mock' : 'badge-live'}`}
            onClick={onToggleMock}
            title={isMock ? 'Switch to live API backend' : 'Switch to local mock fixtures'}
          >
            <span className="indicator-dot"></span>
            <span>{isMock ? 'Mock Mode' : 'Live API'}</span>
          </button>

          {/* Language Switch: hidden (English only) until translations are verified by a native speaker */}
          {/*
          <div className="lang-switch">
            {languages.map((l) => (
              <button
                key={l.code}
                className={`lang-btn ${currentLang === l.code ? 'active' : ''}`}
                onClick={() => onSelectLang(l.code)}
              >
                {l.label}
              </button>
            ))}
          </div>
          */}
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav className="navbar-nav">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`nav-tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => onSelectTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
