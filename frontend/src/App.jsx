import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import RecommendPage from './components/RecommendPage';
import WhatIfPanel from './components/WhatIfPanel';
import BatchPage from './components/BatchPage';
import Arena from './pages/Arena';
import ModelCard from './pages/ModelCard';
import { DEMO_PRESET_INPUTS } from './config';
import mockExamples from '../mock/predict_examples.json';
import { isMockEnabled, setMockEnabled } from './api/client';
import './App.css';

function getTabFromPath(path) {
  const clean = (path || (typeof window !== 'undefined' ? window.location.pathname : ''))
    .toLowerCase()
    .replace(/\/+$/, '');
  if (clean === '/arena') return 'arena';
  if (clean === '/model-info' || clean === '/model-card') return 'model-info';
  if (clean === '/what-if') return 'what-if';
  if (clean === '/batch') return 'batch';
  return 'recommend';
}

export default function App() {
  const [activeTab, setActiveTab] = useState(() => getTabFromPath());
  const [isMock, setIsMock] = useState(isMockEnabled());
  const [currentLang, setCurrentLang] = useState('en');

  useEffect(() => {
    function handlePopState() {
      setActiveTab(getTabFromPath(window.location.pathname));
    }
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  function handleSelectTab(tabId) {
    setActiveTab(tabId);
    const pathMap = {
      'recommend': '/',
      'what-if': '/what-if',
      'arena': '/arena',
      'batch': '/batch',
      'model-info': '/model-info',
      'model-card': '/model-info'
    };
    const targetPath = pathMap[tabId] || '/';
    if (typeof window !== 'undefined' && window.location.pathname !== targetPath) {
      window.history.pushState(null, '', targetPath);
    }
  }

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
          onSelectTab={handleSelectTab}
          isMock={isMock}
          onToggleMock={handleToggleMock}
          currentLang={currentLang}
          onSelectLang={setCurrentLang}
        />

        <main className="content-container">
          {activeTab === 'recommend' && (
            <RecommendPage onNavigateToWhatIf={() => handleSelectTab('what-if')} />
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
            <Arena />
          )}

          {activeTab === 'batch' && (
            <BatchPage />
          )}

          {(activeTab === 'model-info' || activeTab === 'model-card') && (
            <ModelCard />
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
