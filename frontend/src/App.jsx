import React, { useState } from 'react';
import Navbar from './components/Navbar';
import RecommendPage from './components/RecommendPage';
import WhatIfPanel from './components/WhatIfPanel';
import BatchPage from './components/BatchPage';
import { DEMO_PRESET_INPUTS } from './config';
import mockExamples from '../mock/predict_examples.json';
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
      <div className="app-shell-container">
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
            <div className="what-if-tab-view">
              <WhatIfPanel
                baselineInputs={DEMO_PRESET_INPUTS.recommend}
                baselinePrediction={mockExamples.recommend}
                riskProfile="balanced"
              />
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
            <BatchPage />
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
    </div>
  );
}
