// API configuration
const API_BASE = "http://localhost:8000";

// API functions
const API = {
  // Search for images using text query
  async search(queries, limit = 10) {
    const searchUrl = `${API_BASE}/search`;
    const response = await fetch(searchUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ queries, limit })
    });
    
    if (!response.ok) {
      throw new Error(await response.text());
    }
    
    return await response.json();
  },

  // Search for images using multiple text queries (hybrid search)
  async searchHybrid(queries, limit = 10) {
    const searchUrl = `${API_BASE}/search_hybrid`;
    const response = await fetch(searchUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ queries, limit })
    });
    
    if (!response.ok) {
      throw new Error(await response.text());
    }
    
    return await response.json();
  },

  // Get surrounding frames for given addresses
  async getSurroundings(addresses, window = 10) {
    const response = await fetch(`${API_BASE}/surroundings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ addresses, window })
    });
    
    if (!response.ok) {
      throw new Error(await response.text());
    }
    
    return await response.json();
  },

  // Save sketch image to server
  async saveSketch(imageData) {
    const response = await fetch(`${API_BASE}/save_sketch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_data: imageData
      })
    });
    
    if (!response.ok) {
      throw new Error(await response.text());
    }
    
    return await response.json();
  },

  // Process sketch from file path
  async processSketch(imagePath) {
    const response = await fetch(`${API_BASE}/process_sketch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_path: imagePath
      })
    });
    
    if (!response.ok) {
      throw new Error(await response.text());
    }
    
    return await response.json();
  },

  // Get image URL for display
  getImageUrl(folderName, imageName) {
    return `${API_BASE}/image/${folderName}/${imageName}`;
  },

  // Perform search based on query type and count
  async performSearch(queries, limit = 10) {
    if (queries.length >= 2) {
      return await this.searchHybrid(queries, limit);
    } else {
      return await this.search(queries, limit);
    }
  }
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
  module.exports = API;
}
