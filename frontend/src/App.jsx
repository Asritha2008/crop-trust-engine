import React, { useState } from 'react';
import Navbar from './components/Navbar';
import RecommendPage from './components/RecommendPage';
import { isMockEnabled, setMockEnabled } from './api/client';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('recommend');
  const [isMock, setIsMock] = useState(isMockEnabled());
  const [currentLang, setCurrentLang] = useState('en');

  function handleToggleMock() {
    const nextVal = !isMock;
    setIsMock(nextVal);
    setMockEnabled(nextVal);
  }

  return (
    <div className="site-wrapper">
      <Navbar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        isMock={isMock}
        onToggleMock={handleToggleMock}
        currentLang={currentLang}
        onSelectLang={setCurrentLang}
      />

      <main className="content-container">
        {activeTab === 'recommend' && (
          <RecommendPage onNavigateToWhatIf={() => setActiveTab('what-if')} />
        )}

        {activeTab === 'what-if' && (
          <div className="placeholder-card">
            <span className="placeholder-icon">🔮</span>
            <h2>What-If Simulator & Climate Stress Test</h2>
            <p>
              Interactive sensitivity analysis: live sliders that re-call the API, counterfactual hints,
              and climate stress testing (+temp / -rainfall comparison without new routes).
            </p>
            <span className="placeholder-badge">Mapped to Task 5</span>
            <div style={{ marginTop: '1.25rem' }}>
              <button
                type="button"
                className="btn-link-action"
                onClick={() => setActiveTab('recommend')}
              >
                ← Return to Recommend Page
              </button>
            </div>
          </div>
        )}

        {activeTab === 'arena' && (
          <div className="placeholder-card">
            <span className="placeholder-icon">⚔️</span>
            <h2>Model Arena & Evaluation Benchmark</h2>
            <p>
              Classifier benchmark fed by <code>GET /arena</code>: 9-model comparison table (accuracy, macro-F1,
              log-loss, ECE), 22x22 per-crop confusion heatmap, reliability diagrams, and E5 AUROC detector table.
            </p>
            <span className="placeholder-badge">Mapped to Task 7</span>
          </div>
        )}

        {activeTab === 'batch' && (
          <div className="placeholder-card">
            <span className="placeholder-icon">📁</span>
            <h2>Batch CSV Assessment</h2>
            <p>
              Upload farm datasets (up to 1 MB / 500 rows) to <code>POST /predict/batch</code> with tabular verdict
              highlights, OOD flagged rows, and downloadable CSV/text report.
            </p>
            <span className="placeholder-badge">Mapped to Task 8</span>
          </div>
        )}

        {activeTab === 'model-card' && (
          <div className="placeholder-card">
            <span className="placeholder-icon">📄</span>
            <h2>Model Card & System Specifications</h2>
            <p>
              System provenance fed by <code>GET /model-info</code>: Kaggle 2,200-sample dataset boundaries,
              calibration methods, sensor noise thresholds, limitations, and intended agronomic decision-support scope.
            </p>
            <span className="placeholder-badge">Mapped to Task 7</span>
          </div>
        )}
      </main>

      <footer className="site-footer">
        <p>
          Crop Trust Engine • Five-Stage Trust Architecture • Built with React + Vite • Role 4 Frontend
        </p>
      </footer>
    </div>
  );
}
