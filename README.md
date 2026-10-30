Start Backend

cd backend
pip install -r ../requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Yêu cầu trước khi chạy:
- File .env ở root project cần có:
  - DATA_ROOT=D:/ (đường dẫn tới thư mục chứa keyframes)
  - GOOGLE_APPLICATION_CREDENTIALS=./gcp-key.json
  - LLM_API_KEY=sk-or-v1-... (OpenRouter API key)
  - LLM_BASE_URL=https://openrouter.ai/api/v1
  - LLM_MODEL=qwen/qwen3.8-27b
  - VLM_MODEL=qwen/qwen3.8-27b
- Milvus server đang chạy (collection aic25_keyframes, 168,749 docs)
- SigLIP2 model đã download offline

Start Frontend

cd frontend
npm install
npm run dev

Frontend thường chạy ở http://localhost:5173

Sử dụng

1. Search (Manual Mode)

- UI: Gõ query vào ô tìm kiếm, hệ thống trả về top-k keyframes với score
- API: POST /api/v1/search với body {"parsed": {...}, "k": 10}

2. Chat (Agentic Mode)

- UI: Tab Chat, gõ câu hỏi tiếng Việt — agent tự search, look (VLM), read_ocr rồi trả lời
- API: POST /api/v1/chat với body {"question": "...", "max_steps": 5, "timeout_s": 60} — trả SSE stream

3. VQA (Direct)

- API: POST /api/v1/vqa/answer với body {"question": "...", "evidence_ids": ["L30_V078/031"]}

4. Dashboard & Traces

- Trang Dashboard xem metrics hệ thống
- Trang Traces xem chi tiết từng request (span tree, latency)

Ingest data (nếu cần rebuild index)

      python backend/scripts/ingest.py

      Cái này cần DATA_ROOT trỏ đúng tới thư mục keyframes của AIC dataset.
