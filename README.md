# Multimodal-RAG

**Step-by-Step: Running a Multimodal-RAG Project (FastAPI + React)**

---

## **Prerequisites**

* **Install Docker Desktop.**
* **Install Windows Subsystem for Linux 2 (WSL 2).**
* **Install Python 3.8+.**
* **Install Node.js**

---

## **Run Milvus with Docker Compose**

From PowerShell or Windows Command Prompt

1. Open Docker Desktop in administrator mode by right-clicking and selecting Run as administrator.

2. Run the following commands in PowerShell or Windows Command Prompt to download the Docker Compose configuration file for Milvus Standalone and start Milvus.

```bash
# Download the configuration file and rename it as docker-compose.yml
C:\>Invoke-WebRequest https://github.com/milvus-io/milvus/releases/download/v2.4.15/milvus-standalone-docker-compose.yml -OutFile docker-compose.yml

# Start Milvus
C:\>docker compose up -d
Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done
# Stop Milvus
C:\>docker compose stop
```
---
## **Install Python Dependencies**

Open **new terminal** (or Command Prompt/PowerShell on Windows):

```bash
python -m pip install --upgrade pip
```

**Create a virtual Python environment:**

```bash
python -m venv venv
```

**Activate the environment:**

* **For Windows:**

  ```bash
  venv\Scripts\activate
  ```
* **For macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

**Install backend dependencies:**

```bash
pip install -r requirements.txt
```

---

## **2. Install Node.js/React Dependencies**

In your project folder:

```bash
npm install
```

This installs all frontend dependencies listed in `package.json`.

---

## **3. Start the React Frontend**

In your project folder (make sure the virtual environment is still active if your frontend requires it):

```bash
npm start
```

* This usually runs at [http://localhost:3000](http://localhost:3000)

---

## **4. Start the FastAPI Backend**

Open **another terminal** (so your frontend keeps running):

* **Activate your virtual environment again** (see step 1 if unsure).

Then run:

```bash
uvicorn app.main:app --reload
```

* The FastAPI server runs by default at [http://localhost:8000](http://localhost:8000)

---

## **5. Testing Everything**

* Visit your frontend at `http://localhost:3000`
* The frontend should communicate with the backend at `http://localhost:8000`

---

### **Extra Tips**

* If you need CORS for frontend-backend communication, make sure to include FastAPI's CORS middleware.
* To stop any server, press `Ctrl+C` in its terminal.
* If you run into errors, double-check paths and environment activation steps.

---