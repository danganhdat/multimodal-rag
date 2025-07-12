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

  // --- Search logic ---
  function getQueries() {
    return Array.from(document.querySelectorAll('.query-by-text'))
      .map(q => q.value.trim())
      .filter(Boolean);
  }

  function renderImageGroups(listOfImageLists) {
    const container = document.querySelector('.image-groups');
    container.innerHTML = '';
    container.style.color = "#555";
    if (!listOfImageLists || !listOfImageLists.length) {
      container.textContent = "No results";
      container.style.color = "red";
      return;
    }
    listOfImageLists.forEach((addresses, idx) => {
      const row = document.createElement('div');
      row.className = 'image-row';
      addresses.forEach(addr => {
        const cell = document.createElement('div');
        cell.className = 'image-cell';
        const img = document.createElement('img');
        img.src = `${API_BASE}/image/${addr.folder_name}/${addr.image_name}`;
        img.alt = `${addr.folder_name}/${addr.image_name}`;
        img.tabIndex = 0; // for accessibility
        cell.appendChild(img);
        row.appendChild(cell);
      });
      container.appendChild(row);
      if (idx < listOfImageLists.length - 1) {
        container.appendChild(document.createElement('hr'));
      }
    });
  }

  function showError(msg) {
    const container = document.querySelector('.image-groups');
    container.innerHTML = '';
    container.textContent = msg;
    container.style.color = "#e74c3c";
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
});
