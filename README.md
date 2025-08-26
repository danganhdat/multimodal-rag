<<<<<<< HEAD
# Multimodal-RAG

**Step-by-Step: Running a Multimodal-RAG Project (FastAPI)**

---

## **Prerequisites**

* **Install Windows Subsystem for Linux 2 (WSL 2).**
* **Install Docker Desktop.**

---

## **Run Milvus with Docker Compose**

From PowerShell or Windows Command Prompt

1. Open Docker Desktop.

2. Run the following commands in PowerShell or Windows Command Prompt.

```bash
# Start Milvus
>docker compose up -d
Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done

# Stop Milvus
>docker compose stop
```

## **Start the Backend**

```bash
cd backend
uvicorn python main.py


## **Start the Frontend**


```bash
cd ..
cd frontend

python -m http.server 8080
```

* This usually runs at [http://localhost:8080](http://localhost:8080)

---

## **Test Everything**

* Visit your frontend at `http://localhost:8080`
* The frontend should communicate with the backend at `http://localhost:8000`

=======
# Multimodal-RAG

**Step-by-Step: Running a Multimodal-RAG Project (FastAPI)**

---

## **Prerequisites**

* **Install Windows Subsystem for Linux 2 (WSL 2).**
* **Install Docker Desktop.**

---

## **Run Milvus with Docker Compose**

From PowerShell or Windows Command Prompt

1. Open Docker Desktop.

2. Run the following commands in PowerShell or Windows Command Prompt.

```bash
# Start Milvus
>docker compose up -d
Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done

# Stop Milvus
>docker compose stop
```

## **Start the Backend**

```bash
cd backend
uvicorn python main.py


## **Start the Frontend**


```bash
cd ..
cd frontend

python -m http.server 8080
```

* This usually runs at [http://localhost:8080](http://localhost:8080)

---

## **Test Everything**

* Visit your frontend at `http://localhost:8080`
* The frontend should communicate with the backend at `http://localhost:8000`

>>>>>>> 67e60c9 (Initial commit (reset history))
---