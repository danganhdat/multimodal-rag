const API_BASE = "http://localhost:8000";

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
        img.src = `${API_BASE}/image/${addr.folder_name}/${addr.image_name}`;
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
      const searchRes = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ queries, limit: 10 })
      });
      if (!searchRes.ok) throw new Error(await searchRes.text());
      const searchData = await searchRes.json();

      const surroundingsRes = await fetch(`${API_BASE}/surroundings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ addresses: searchData, window: 10 }) // adjust window as needed
      });
      if (!surroundingsRes.ok) throw new Error(await surroundingsRes.text());
      const surroundingsData = await surroundingsRes.json();
      renderImageGroups(surroundingsData);
    } catch (e) {
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

});
