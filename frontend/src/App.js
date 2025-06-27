import React, { useState } from "react";
import "./App.css";

const API_SEARCH = "http://localhost:8000/search/";
const API_IMAGE = "http://localhost:8000/image/";

function App() {
  const [query, setQuery] = useState("");
  const [images, setImages] = useState([]);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    setLoading(true);
    setImages([]);
    setPreview(null);
    try {
      const resp = await fetch(API_SEARCH, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit: 30 }), // adjust limit as you wish
      });
      if (!resp.ok) {
        // Handle HTTP errors like 4xx, 5xx
        const errorData = await resp.json().catch(() => ({ detail: "Unknown server error" })); // Try to parse error, fallback
        throw new Error(`Server error: ${resp.status} ${resp.statusText} - ${errorData.detail || 'No details'}`);
      }
      const data = await resp.json();
      // Backend key is now "filepaths"
      const filepaths = data.filepaths || [];
      const fullUrls = filepaths.map(fp => `${API_IMAGE}${encodeURIComponent(fp)}`);
      setImages(fullUrls);
      setPreview(fullUrls[0] || null);
    } catch (err) {
      alert("Search error: " + err);
    }
    setLoading(false);
  };

  return (
    <div className="app-layout">
      {/* --- Search Bar --- */}
      <div style={{ padding: 16, display: "flex", gap: 8, alignItems: "center", background: "#fff", borderBottom: "1px solid #eee" }}>
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Type search text"
          style={{ flex: 1, padding: 8, fontSize: 16, border: "1px solid #ccc", borderRadius: 4 }}
          onKeyDown={e => { if (e.key === "Enter") handleSearch(); }}
        />
        <button
          onClick={handleSearch}
          disabled={loading || !query.trim()}
          style={{
            padding: "8px 16px",
            background: "#2563eb",
            color: "#fff",
            border: "none",
            borderRadius: 4,
            fontSize: 16
          }}
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {/* --- Main Content --- */}
      <div className="right-panel">
        {/* Preview */}
        {preview && (
          <div className="preview-box" style={{ margin: "20px auto" }}>
            <img src={preview} alt="Preview" className="thumb-img" style={{ width: 360, height: 240, objectFit: "contain", borderRadius: 12, border: "2px solid #ddd" }} />
          </div>
        )}

        {/* Image Grid */}
        <div className="image-grid" style={{ margin: "0 auto" }}>
          {images.map((img, idx) => (
            <div
              key={idx}
              className="thumb-box"
              style={{ cursor: "pointer" }}
              onClick={() => setPreview(img)}
            >
              <img
                src={img}
                alt={`Result ${idx}`}
                className="thumb-img"
                style={{ opacity: img === preview ? 0.5 : 1 }}
                onError={e => (e.target.style.opacity = 0.4)}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;