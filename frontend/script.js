document.addEventListener("DOMContentLoaded", function () {
  const leftColumn = document.querySelector(".column.left");
  let queryDivs = [leftColumn.querySelector(".query-divs")];
  const addBtn = leftColumn.querySelector(".add-button");
  const removeBtn = leftColumn.querySelector(".remove-button");
  const searchBtn = document.querySelector(".search-button");

  // --- Query div add/remove logic ---
  function createQueryDiv() {
    const div = document.createElement("div");
    div.className = "query-divs";
    div.innerHTML = `
      <button class="text-type">Text</button>
      <button class="object-type">Object</button>
      <button class="image-type">Sketch</button>
      <button class="color-type">Colors</button>
      <textarea class="query-by-text" placeholder="Enter your query..."></textarea>
      `;
    return div;
  }

  addBtn.onclick = function () {
    if (queryDivs.length >= 10) return;
    const newDiv = createQueryDiv();
    leftColumn.insertBefore(newDiv, leftColumn.querySelector(".add-remove-buttons"));
    queryDivs.push(newDiv);
    updateButtons();
  };

  removeBtn.onclick = function () {
    if (queryDivs.length > 1) {
      const lastDiv = queryDivs.pop();
      lastDiv.remove();
      updateButtons();
    }
  };

  function updateButtons() {
    removeBtn.disabled = queryDivs.length <= 1;
    addBtn.disabled = queryDivs.length >= 10;
  }

  updateButtons();


  // --- Show image groups ---

  function showError(msg) {
    const container = document.querySelector('.image-groups');
    container.innerHTML = '';
    container.textContent = msg;
    container.style.color = "#e74c3c";
  }

  // Keep track of the currently selected image (only one allowed)
  let selectedImages = [];

  // Render image groups with single-selection functionality
  function renderImageGroups(listOfImageLists) {
    const container = document.querySelector('.image-groups');
    container.innerHTML = '';
    selectedImages = [];
    container.style.color = "#555";

    if (!listOfImageLists || !listOfImageLists.length) {
      showError("No images found for the given queries");
      return;
    }

    listOfImageLists.forEach((addresses, idx) => {
      const row = document.createElement('div');
      row.className = 'image-row';

      addresses.forEach(addr => {
        const cell = document.createElement('div');
        cell.className = 'image-cell';

        cell.dataset.folderName = addr.folder_name;

        const match = addr.image_name.match(/(\d+)/);
        cell.dataset.frameIdx = match ? match[1] : addr.image_name;

        const img = document.createElement('img');
        img.src = API.getImageUrl(addr.folder_name, addr.image_name);
        img.alt = `${addr.folder_name}/${addr.image_name}`;

        // Click to select only this image
        cell.onclick = function() {
          // Deselect all other images
          document.querySelectorAll('.image-cell.selected').forEach(selectedCell => {
            selectedCell.classList.remove('selected');
          });

          // Select this one
          cell.classList.add('selected');

          // Update selectedImages to only this image
          selectedImages = [{
            folder_name: cell.dataset.folderName,
            frame_idx: cell.dataset.frameIdx,
          }];
        };

        cell.appendChild(img);
        row.appendChild(cell);
      });

      container.appendChild(row);
      if (idx < listOfImageLists.length - 1) {
        container.appendChild(document.createElement('hr'));
      }
    });
  }


  // --- Search logic ---
  function getQueries() {
    return Array.from(document.querySelectorAll('.query-by-text'))
      .map(q => q.value.trim())
      .filter(Boolean);
  }

  searchBtn.onclick = async function () {
    const queries = getQueries();
    const container = document.querySelector('.image-groups');
    if (!queries.length) {
      showError("Enter at least one query");
      return;
    }
    container.textContent = "Loading...";
    container.style.color = "#555";
    try {
      // Use API module for search
      const searchData = await API.performSearch(queries, 10);
      
      // Get surroundings for the search results
      const surroundingsData = await API.getSurroundings(searchData, 10);
      renderImageGroups(surroundingsData);
    } catch (e) {
      console.error('Search error:', e);
      showError("Search error");
    }
  };


// TODO: NOT CLEAN YET


  // --- Add to CSV logic ---
  let imagesToDownload = [];   // List of frames to export
  // ========== CSV Table Rendering ==========
  function renderCSVTable() {
    const container = document.getElementById('show-csv-table');
    if (!imagesToDownload.length) {
      container.innerHTML = '<p style="color: #aaa;">No frames added yet.</p>';
      return;
    }
    let html = '<table style="width:100%;border-collapse:collapse;">';
    html += '<thead><tr><th>File Name</th><th>Frame Idx</th></tr></thead><tbody>';
    imagesToDownload.forEach(item => {
      html += `<tr>
        <td style="border:1px solid #ddd;padding:4px;">${item.folder_name}</td>
        <td style="border:1px solid #ddd;padding:4px;">${item.frame_idx}</td>
      </tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
  }

  function downloadCSV(dataArray) {
    if (!dataArray.length) return;
    const csvRows = [];
    csvRows.push("File Name,Frame Idx"); // Header
    dataArray.forEach(item => {
      csvRows.push(`${item.folder_name},${item.frame_idx}`);
    });
    const csvContent = csvRows.join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);

    // Trigger download
    const a = document.createElement("a");
    a.href = url;
    a.download = "selected_frames.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ========== Button Event Handlers ==========

  document.querySelector('.add-csv-button').onclick = function() {
    if (!selectedImages.length) {
      alert("Please select a frame to add!");
      return;
    }
    const toAdd = selectedImages[0];

    // Prevent duplicates
    const exists = imagesToDownload.some(
      item => item.folder_name === toAdd.folder_name && item.frame_idx === toAdd.frame_idx
    );
    if (exists) {
      alert("Frame already added!");
      return;
    }
    imagesToDownload.push({...toAdd});
    renderCSVTable();
    alert("Frame added!");
  };

  document.querySelector('.download-csv-button').onclick = function() {
    if (!imagesToDownload.length) {
      alert("Please add at least one frame to download!");
      return;
    }
    downloadCSV(imagesToDownload);
  };

  // ========== Initialize ==========
  renderCSVTable(); // Show empty table on first load

  // ========== Drawing Functionality ==========
  initializeDrawing();

});

// Drawing functionality
function initializeDrawing() {
  const drawingPopup = document.getElementById('drawingPopup');
  const closeDrawingBtn = document.getElementById('closeDrawing');
  
  // Handle Sketch button clicks
  document.addEventListener('click', function(e) {
    if (e.target.classList.contains('image-type')) {
      drawingPopup.classList.add('show');
      initializeCanvas();
    }
  });

  // Close popup
  closeDrawingBtn.addEventListener('click', function() {
    drawingPopup.classList.remove('show');
  });

  // Close popup when clicking outside
  drawingPopup.addEventListener('click', function(e) {
    if (e.target === drawingPopup) {
      drawingPopup.classList.remove('show');
    }
  });

  function initializeCanvas() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');

    // State variables
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let currentTool = 'pencil';

    // UI elements
    const colorPicker = document.getElementById('colorPicker');
    const brushSizeSlider = document.getElementById('brushSize');
    const brushSizeValue = document.getElementById('brushSizeValue');
    const pencilBtn = document.getElementById('pencilBtn');
    const eraserBtn = document.getElementById('eraserBtn');
    const clearBtn = document.getElementById('clearBtn');
    const saveBtn = document.getElementById('saveBtn');
    const useSketchBtn = document.getElementById('useSketchBtn');

    // Set canvas size
    function resizeCanvas() {
      const container = canvas.parentElement;
      const containerRect = container.getBoundingClientRect();
      
      canvas.width = containerRect.width - 20;
      canvas.height = containerRect.height - 20;
      
      // Set default styles
      ctx.strokeStyle = colorPicker.value;
      ctx.lineWidth = brushSizeSlider.value;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
    }

    // Initialize canvas
    resizeCanvas();

    // Drawing functions
    function getMousePos(evt) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
      };
    }

    function getTouchPos(evt) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: evt.touches[0].clientX - rect.left,
        y: evt.touches[0].clientY - rect.top
      };
    }

    function startDrawing(e) {
      e.preventDefault();
      isDrawing = true;
      const pos = e.touches ? getTouchPos(e) : getMousePos(e);
      [lastX, lastY] = [pos.x, pos.y];
    }

    function draw(e) {
      if (!isDrawing) return;
      e.preventDefault();

      const pos = e.touches ? getTouchPos(e) : getMousePos(e);
      
      // Set tool properties
      if (currentTool === 'pencil') {
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = colorPicker.value;
      } else if (currentTool === 'eraser') {
        ctx.globalCompositeOperation = 'destination-out';
      }
      ctx.lineWidth = brushSizeSlider.value;

      // Draw line
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();

      [lastX, lastY] = [pos.x, pos.y];
    }

    function stopDrawing() {
      isDrawing = false;
    }

    // Event listeners for drawing
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch events
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);
    canvas.addEventListener('touchcancel', stopDrawing);

    // Toolbar controls
    brushSizeSlider.addEventListener('input', function(e) {
      const size = e.target.value;
      brushSizeValue.textContent = size;
      ctx.lineWidth = size;
    });

    colorPicker.addEventListener('change', function(e) {
      ctx.strokeStyle = e.target.value;
    });

    clearBtn.addEventListener('click', function() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    });

    saveBtn.addEventListener('click', async function() {
      try {
        // Create a temporary canvas with white background
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
        
        // Fill white background
        tempCtx.fillStyle = 'white';
        tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
        
        // Draw the original canvas on top
        tempCtx.drawImage(canvas, 0, 0);
        
        // Get image data
        const dataURL = tempCanvas.toDataURL('image/png');
        
        // Comment out download functionality
        // Set download link for local download
        // saveBtn.href = dataURL;
        
        // Save to server using API module
        const result = await API.saveSketch(dataURL);
        console.log('Sketch saved to server:', result);
        alert(`Ảnh đã được lưu thành công!\nFile: ${result.file_info.file_name}\nSize: ${result.file_info.file_size} bytes`);
        
      } catch (error) {
        console.error('Error saving sketch:', error);
        alert('Có lỗi khi lưu ảnh lên server');
      }
    });

    useSketchBtn.addEventListener('click', function() {
      // Convert canvas to image data
      const dataURL = canvas.toDataURL('image/png');
      
      // Here you can implement logic to use the sketch for search
      // For now, just show an alert
      alert('Sketch sẽ được sử dụng để tìm kiếm!\n(Chức năng này cần được implement trong backend)');
      
      // Close the popup
      drawingPopup.classList.remove('show');
      
      // Optionally clear the canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    });

    pencilBtn.addEventListener('click', function() {
      currentTool = 'pencil';
      pencilBtn.classList.add('active');
      eraserBtn.classList.remove('active');
    });

    eraserBtn.addEventListener('click', function() {
      currentTool = 'eraser';
      eraserBtn.classList.add('active');
      pencilBtn.classList.remove('active');
    });

    // Handle window resize
    window.addEventListener('resize', resizeCanvas);
  }
}
