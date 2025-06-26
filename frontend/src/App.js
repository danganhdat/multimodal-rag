import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function QueryInput({ value, onChange, onClear, placeholder }) {
  return (
    <div className="input-wrapper">
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={e => onChange(e.target.value)}
        className="query-input"
      />
      {value && <button className="clear-btn" onClick={onClear}>×</button>}
    </div>
  );
}

function TopKSelect({ value, onChange, options }) {
  return (
    <select value={value} onChange={e => onChange(Number(e.target.value))} className="select">
      {options.map(opt => <option key={opt} value={opt}>Top {opt}</option>)}
    </select>
  );
}

function SearchButton({ onClick, disabled, loading }) {
  return (
    <button onClick={onClick} disabled={disabled || loading} className="search-btn">
      {loading ? 'Searching...' : 'Search Scenes'}
    </button>
  );
}

function Sidebar({ queries, onQueryChange, onClearQuery, topK, onTopKChange, onSearch, isLoading }) {
  return (
    <aside className="sidebar">
      <div className="header">
        <div className="title-row">
          <div className="icon-wrapper">
            <span className="icon">🎬</span>
          </div>
          <div>
            <div className="title">Video Search</div>
            <div className="subtitle">Multi-query frame finder</div>
          </div>
        </div>
      </div>
      <div className="input-container">
        <div className="query-group">
          {queries.map((q, i) => (
            <QueryInput
              key={i}
              value={q}
              placeholder={`Query ${i + 1}...`}
              onChange={v => onQueryChange(i, v)}
              onClear={() => onClearQuery(i)}
            />
          ))}
        </div>
      </div>
      <div className="settings">
        <TopKSelect value={topK} onChange={onTopKChange} options={[10, 20, 30, 40, 50]} />
      </div>
      <div className="search-section">
        <SearchButton onClick={onSearch} disabled={!queries.some(q => q.trim())} loading={isLoading} />
      </div>
    </aside>
  );
}

function SceneStrip({ frames, index, onSelectFrame }) {
  return (
    <div className="scene-container">
      <div className="scene-header">
        <div className="scene-index">{index + 1}</div>
        <div className="scene-title">Scene {index + 1}</div>
        <div className="scene-count">{frames.length} frames</div>
      </div>
      <div className="strip">
        {frames.map((url, fi) => (
          <div key={fi} className="frame-wrapper" onClick={() => onSelectFrame(index + 1, fi + 1, url)}>
            <img src={url} alt={`Frame ${fi + 1}`} className="frame-img" />
            <div className="frame-index">{fi + 1}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function FrameModal({ frame, onClose }) {
  if (!frame) return null;
  return (
    <div className="modal-backdrop">
      <div className="modal">
        <div className="modal-header">
          <div className="modal-title">Scene {frame.scene}, Frame {frame.frame}</div>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>
        <div className="modal-content">
          <img src={frame.url} alt="Selected frame" />
        </div>
      </div>
    </div>
  );
}

export default function VideoSearchApp() {
  const [queries, setQueries] = useState(['', '', '']);
  const [topK, setTopK] = useState(10);
  const [scenes, setScenes] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selected, setSelected] = useState(null);

  const updateQuery = (i, v) => setQueries(qs => qs.map((q, idx) => (idx === i ? v : q)));
  const clearQuery = i => updateQuery(i, '');

  const fetchScenes = async () => {
    setIsLoading(true);
    try {
      const results = await Promise.all(
        queries.map(async q => {
          if (!q.trim()) return [];
          const { data } = await axios.post('/search/', { query: q, limit: topK });
          return data.urls;
        })
      );
      setScenes(results);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectFrame = (scene, frame, url) => setSelected({ scene, frame, url });

  return (
    <div className="container">
      <Sidebar
        queries={queries}
        onQueryChange={updateQuery}
        onClearQuery={clearQuery}
        topK={topK}
        onTopKChange={setTopK}
        onSearch={fetchScenes}
        isLoading={isLoading}
      />
      <main className="main">
        {scenes.length === 0 && !isLoading ? (
          <div className="empty-state">
            <div className="icon-large">🎬</div>
            <div className="ready">Ready to Search</div>
            <div>Enter queries and click "Search Scenes" to find video frames.</div>
          </div>
        ) : (
          scenes.map((frames, idx) => (
            <SceneStrip key={idx} frames={frames} index={idx} onSelectFrame={handleSelectFrame} />
          ))
        )}
      </main>
      <FrameModal frame={selected} onClose={() => setSelected(null)} />
    </div>
  );
}