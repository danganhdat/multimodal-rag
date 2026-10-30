# Plan: Video VQA Search + Agent (Đồ án A3) trên repo multimodal-rag

## Context

Repo hiện là công cụ **tìm keyframe**: FastAPI `backend/app`, Milvus `aic25_keyframes`, React `frontend/`. Chưa có LLM/VLM, chưa có câu trả lời, observability và đánh giá thật.

Mục tiêu là hệ thống **VQA** với hai chế độ:
- **Manual:** LLM phân rã câu hỏi, người dùng sửa, tìm, chọn/upvote frame, rồi hỏi.
- **Agentic:** chatbot ReAct tự tìm, xem frame và trả lời kèm bằng chứng, có timeline trên UI.

Plan bám kiến trúc 2 tầng (recall → precision), metric đúng tầng, ablation một thay đổi mỗi lần, API `/api/v1`, và observability gồm span, sequence diagram, dashboard.

**Tài liệu gốc:** "PROMPT CHO AGENT CODE (A3)" của người dùng. Khi mâu thuẫn với bản tổng hợp phản biện thì tài liệu này thắng. Ngoài ra áp dụng các quyết định bổ sung bên dưới.

**Quyết định bổ sung của người dùng:**
- **Xóa hết như đã yêu cầu:** legacy v1, toàn bộ BGE, CLIP text và `/api/switch-encoder`. `/api/search` cũ **được phép đổi contract** (bỏ `rerank`, `rerank_candidates`, `encoder_mode`). Đây là ngoại lệ cho mục 0.3 của tài liệu gốc.
- **Không làm trang gợi ý / recommender.** Thay vào đó, trong manual mode người dùng **chọn hoặc upvote frame** và frame đó trở thành tín hiệu truy vấn bổ sung (mục 5.4).
- **TransNetV2 đã chạy xong**, keyframe là kết quả của nó. Phần offline chỉ mô tả trong báo cáo, **không chạy lại**.
- **Gán qrels/VQA làm ở một plan riêng.** Plan này chỉ làm `metrics.py`, harness và unit test. Harness báo lỗi rõ khi chưa có file nhãn.

**Quy tắc bắt buộc (mục 0 của tài liệu gốc):**
- **Không fake gì cả:** không FakeLLM/mock, không dữ liệu/số liệu/tool giả, không `sleep` mô phỏng.
- Thiếu phụ thuộc thì raise lỗi rõ ràng, hoặc tắt tính năng và hiện lý do trên UI.
- Code đơn giản: hàm thường, file dưới ~200 dòng, không ABC/registry/DI, không cache khi chưa đo được là chậm.
  - **Ngoại lệ do người dùng chốt:** agent dùng **LangGraph** (+ `langchain-openai`, `langgraph-checkpoint-sqlite`). Không dùng LlamaIndex. Không dùng AgentExecutor cũ.
- **Không re-ingest, không đổi schema Milvus** khi chưa hỏi người dùng.

---

## 1. Inventory đã biết (ghi vào `docs/inventory.md` ở bước 0, kiểm tra lại bằng code)

| Mục | Thực tế trong repo |
|---|---|
| Milvus | collection `aic25_keyframes` (docker-compose: etcd, minio, standalone v3.0.2). Khoảng 177k keyframe, 873 video (L21–L30). Frame trắng/đen đã bị lọc lúc ingest |
| Field | `id` = `"L21_V001/001"`, `video_name`, `keyframe_idx` (= n), `frame_idx` (đã map từ CSV), `pts_time`, `fps`, `text_vector` 768, `clip_vector` 512, `objects` (YOLO11 COCO, tối đa 20), `ocr` (Florence-2, hay mất dấu) |
| Encoder | `text_vector` = **SigLIP2 image** (`google/siglip2-base-patch16-512`), nên text encoder phải là **SigLIP2 text** (không dùng OpenAI CLIP như sơ đồ). `clip_vector` = TSBIR/TASK-former image tower, đóng vai **milvus-Sketch** |
| Map keyframe | `map-keyframes/{video}.csv` (`n, pts_time, fps, frame_idx`). n ≠ frame_idx |
| Shot | TransNetV2 đã chạy ngoài repo. **Trong repo chưa có file scene/shot** → `shot_id` để trống (xem mục 3.5) |
| OCR / object / ASR | OCR có (Milvus `ocr`), object có (Milvus `objects`, chỉ nhãn, không có bbox), ASR không có |
| Service | `/api/search`, `/api/switch-encoder`, `/api/translate`, `/api/video/{v}` (Range), `/api/video-meta/{v}`, `/api/object-classes`. Keyframe do Vite middleware phục vụ ở `/keyframes` |
| Lỗi đã thấy | (1) TSBIR dựng bằng GELU thay vì QuickGELU, re-encode chỉ đạt cos≈0.98 so với vector đã index. (2) `_client_merge` search sai field. (3) `ingest.py` có `PKL_PATH` sai và fallback L131 lấy n làm frame_idx. (4) neighbors query N+1. (5) Google v2 không unescape HTML. (6) Dockerfile CMD trỏ vào legacy `main:app`. (7) `requirements.txt` thiếu `open_clip_torch`, `transformers` |

Phần "Câu hỏi" trong `inventory.md` cần ghi:
- đường dẫn file scene của TransNetV2, nếu người dùng muốn đưa vào sau;
- ID model Qwen-VL trên OpenRouter có hỗ trợ ảnh hay không.

---

## 2. Dọn dẹp và sửa nền (bước 0–1)

**Xóa:**
- `backend/main.py`, `backend/encoder.py`, `backend/setup.py`. Sửa `backend/Dockerfile` CMD → `uvicorn app.main:app` trước khi xóa.
- `backend/app/reranker.py`. Bỏ `RERANKER_MODEL`, block reranker trong lifespan, bước BGE trong `scripts/download_models.py`, toggle BGE trong `QueryPanel.tsx`.
- CLIP text: `CLIPEncoder.encode_text` và tokenizer, `/api/switch-encoder`, `EncoderMode`/`encoder_mode` (schemas và frontend), dropdown encoder, `switchEncoder` trong `api/client.ts`, bước open_clip `pretrained="openai"` trong `download_models.py`.
- `sentence-transformers` trong `requirements.txt`. Thêm tường minh `transformers>=4.49`, `safetensors`, `huggingface_hub`, `open_clip_torch`, `pyyaml`, `langgraph`, `langchain-openai`, `langgraph-checkpoint-sqlite`. Version pin theo bản đang có trên PyPI lúc cài.
- `backend/scripts/evaluate.py` (thay bằng `eval/evaluate.py`).
- `/api/search` cũ: bỏ nhánh BGE và `encoder_mode`; text luôn dùng SigLIP2, sketch dùng `clip_vector`. Các script gán nhãn cũ vẫn chạy được vì Pydantic bỏ qua field thừa.

**Sửa:**
- `backend/app/encoder.py`:
  - **`SketchEncoder`**: chỉ dựng vision tower ViT-B-16 với `quick_gelu=True`; load các key `visual.*` từ `tsbir_model_final.pt` (`torch.load(mmap=True)`, `strict=True`); fp16 trên GPU; lazy load.
  - **`SigLIP2Encoder`**: thêm `encode_texts(list)` (encode batch) và `encode_image(img)` cho image-by-example khi upload ảnh.
- `scripts/ingest.py`: sửa `PKL_PATH` thành `siglip-tsbir-features/…`; fallback L131 đổi thành raise. Chỉ sửa code, **không chạy lại ingest**.
- **Mới** `backend/scripts/verify_frame_map.py` (chỉ đọc): so `frame_idx`/`pts_time` trong Milvus với CSV; báo video thiếu CSV và dòng `n>1 ∧ pts_time==0`.
- `translator.py`: `html.unescape` + dịch batch.
- `milvus_client.py`:
  - thêm `search_batch(vectors, field, top_k, filter)`, `get_by_ids`, `get_vectors(ids, field)`;
  - thêm `get_neighbors_batch` để thay vòng lặp N+1;
  - thêm `iter_ocr()` (query_iterator) để build BM25.

---

## 3. Pipeline

### 3.1. Offline

Đã xong, **không chạy lại**. Báo cáo mô tả: Video → TransNetV2 → shot → 3 keyframe/shot → SigLIP2 image (`text_vector`) + TSBIR image (`clip_vector`) + Florence-2 OCR + YOLO11 → Milvus. BM25 được build trong RAM lúc khởi động từ field `ocr`.

### 3.2. Online: các stage theo cờ

| # | Stage | Mặc định | Tầng | File |
|---|---|---|---|---|
| S1 | Query Enhancement | bật | trước tầng 1 | `query/qe.py` |
| S2 | Dense **ANN Milvus** (HNSW, SigLIP2 → `text_vector`) + BM25 (OCR) → RRF phía client | bật | **Tầng 1 – recall** | `retrieve/dense.py`, `bm25.py`, `fusion.py` |
| S3 | Filter object / metadata / sketch | **tắt** | Tầng 1 | `retrieve/filter.py` |
| S4 | **Dynamic temporal search hai chiều** (tự chạy khi có ≥2 event) | bật | Tầng 1 | `retrieve/temporal.py` |
| S5 | Rerank `none \| superglobal \| vlm` | `superglobal` | **Tầng 2 – precision** | `rerank/superglobal.py`, `vlm.py` |
| S6 | Shot validation | **tắt** | Tầng 2 | `rerank/shot_validation.py` |
| S7 | Frame/Shot input vào prompt agent | **tắt** | trả lời | `agent/context.py` |
| S8 | Shot summary | **tắt** | trả lời | `agent/context.py` |
| S9 | Agent ReAct | bật (chat) | trả lời | `agent/react.py` |

**Temporal chạy trước rerank:**
- Temporal cần top-200 cho mỗi event để ghép chuỗi. Tầng 2 không khôi phục được những gì tầng 1 đã cắt.
- Thứ tự này được kiểm chứng bằng ablation E8.

### 3.3. Chi tiết (bản đơn giản nhất)

**S2 – Dense:**
- Giữ nguyên **ANN của Milvus** như hiện tại: HNSW, `ef=512`, COSINE.
- Mỗi truy vấn (text_en + variants nếu bật): `encode_texts` → `search_batch` top-N=200 trên `text_vector`. Mỗi vector cho một list, cùng một lời gọi Milvus (nq = số variant).

**S2 – Gộp:**
- **RRF tự cài phía client**: `Σ w/(k+rank)`, k=60, gộp mọi list (dense variants, BM25, upvote).
- Không dùng `RRFRanker` của Milvus nữa, vì list BM25/upvote không nằm trong Milvus. BM25 sẵn có của Milvus cần thêm field sparse và ingest lại, mà ingest lại thì không được làm.
- Mỗi kết quả ghi `sources = {dense: rank, bm25: rank, upvote: rank, rerank: rank}`.

**BM25 tự cài** (`retrieve/bm25.py`):
- **Search từ khóa thuần, không dùng embedding.** So **chữ OCR người dùng nhập hoặc QE trích ra** (`events[*].ocr`: chuỗi trong ngoặc kép, chữ IN HOA, số…) với **chữ OCR lưu trong field `ocr` của Milvus**.
- Event không có chữ OCR → nhánh BM25 **không chạy** cho event đó, vì đem cả câu hỏi so với chữ trên màn hình chỉ gây nhiễu. Manual mode cho người dùng tự nhập/sửa ô OCR.
- Tokenizer: NFC → lowercase → **bỏ dấu** (`đ→d`, vì Florence hay mất dấu) → `\w+`, áp cho cả hai phía.
- Dùng k1=1.5, b=0.75; inverted index là dict trong RAM, build lúc khởi động từ `iter_ocr()`.
- Milvus không có OCR → **tắt nhánh BM25 và hiện lý do** trong `/api/v1/config` và UI.

**S3 (tắt):**
- **Object:** giữ frame có nhãn COCO yêu cầu.
- **Metadata:** `video_name`, hoặc ngày đăng/author từ `media-info` đã có trong `VIDEO_META`.
- **Sketch:** sắp lại ứng viên theo cos giữa sketch embedding và `clip_vector` của chúng. Chỉ sắp lại, không fusion.

**S4 – Dynamic temporal search hai chiều:**
- Chỉ chạy khi S1 trả ≥2 event.
- **Ứng viên:** top-200 cho mỗi event (chạy dense + BM25 + RRF riêng từng event), gom theo video. Lấy top-40 video theo số event có hit.
- **Vector:** với các video ứng viên, lấy `text_vector` + `pts_time` của mọi keyframe (`query video_name in [...]`), sort theo `frame_idx`. Tính trước `S[e, j] = cos(q_e, v_j)` (numpy).
- **Neo từ mọi event**, không chỉ event 1: top-`K_anchor=50` hit của mỗi event. Từ neo của event k, **tiến** cho các event k+1…m, **lùi** cho các event k−1…1, mỗi event bắt đầu từ vị trí vừa chọn của event kề.
  ```python
  def expand(S_e, start, step, t, t_ref, max_gap_s, base_patience, eps=0.01, tau_min=None):
      best, pos, tol, j = -inf, None, 0, start + step
      while 0 <= j < len(t) and abs(t[j] - t_ref) <= max_gap_s:
          if S_e[j] > best + eps: best, pos, tol = S_e[j], j, 0   # cải thiện -> reset tolerance
          else: tol += 1
          if tol >= patience(j, t, base_patience): break           # patience động
          j += step
      return (pos, best) if pos is not None and (tau_min is None or best >= tau_min) else (None, None)

  def patience(j, t, base):   # keyframe dày (khoảng cách nhỏ) -> cho phép nhiều bước hơn
      return clamp(round(base * median_gap(t) / max(local_gap(t, j), 1e-3)), 2, 8)
  ```
- **Event bị miss** (không vượt `tau_min`): cho phép bỏ tối đa 1 event, trừ `skip_penalty`.
- **Điểm chuỗi:** `mean(S_e(pos_e)) − λ·Σ max(0, Δt − free_gap_s)/max_gap_s − skip_pen + w_sup·log(1+support)`. `support` = số neo khác nhau hội tụ về cùng chuỗi.
- **Dedup:** hai chuỗi có cùng các keyframe là một chuỗi, giữ điểm cao nhất và cộng support.
- **Output:** frame của `answer_event` xếp theo điểm chuỗi, kèm `chain` (kf_id từng event) để UI và agent xem.
- Thứ tự và khoảng cách tính theo `pts_time`/`frame_idx`, **không theo n**.
- **Cờ:** `temporal: {enabled, max_gap_s: 60, free_gap_s: 10, base_patience: 3, k_anchor: 50, tau_min: null, lambda_gap: 0.3, w_sup: 0.1, bidirectional: true}`. Ablation E7 (tắt temporal) và E8 (temporal sau rerank), thêm E7b `bidirectional: false` (chỉ tiến từ event 1).

**S5:**
- **`superglobal`** (mặc định): lấy `text_vector` của top-50; với mỗi frame tính `v̂ = normalize(mean(v_i, m láng giềng gần nhất trong top-50))` (m=3); điểm mới = cos(q, v̂). Không model, không API.
- **`vlm`**: VLM chấm 0–10 mức khớp với **mô tả cảnh** (không đưa câu hỏi vào), top-20, 4 ảnh mỗi request; hòa thì giữ thứ tự tầng 1. Quá `timeout_ms` hoặc lỗi → trả thứ tự tầng 1, ghi `fallback=true`.

**S6, S7, S8** (tắt):
- **Đơn vị "shot":** vì chưa có file scene, shot = keyframe đó cùng ±1 keyframe lân cận trong cùng video (theo `pts_time`). Ghi rõ trong `inventory.md`. Khi người dùng cung cấp file scene thì thay bằng `shot_id` thật.
- **S6:** gửi ảnh của shot + mô tả cảnh cho VLM, hỏi có/không; loại các shot trả lời "không".
- **S7:** nạp sẵn ảnh top-3 vào prompt đầu của agent.
- **S8:** VLM viết 1–2 câu mô tả mỗi shot top-5. Cache theo khóa shot; đây là kết quả thật đã tính.

---

## 4. Cấu hình

**`configs/default.yaml`:**

```yaml
qe: {enabled: true, multi_query: false, max_variants: 3}
search: {dense: true, bm25: true, top_n: 200, rrf_k: 60}
filter: {objects: false, metadata: false, sketch: false}
temporal: {enabled: true, bidirectional: true, max_gap_s: 60, free_gap_s: 10, base_patience: 3, k_anchor: 50, tau_min: null, lambda_gap: 0.3, w_sup: 0.1}
rerank: {method: superglobal, top: 50, sg_m: 3, vlm_top: 20, timeout_ms: 2000}
shot_validation: {enabled: false}
agent: {frame_input: false, shot_summary: false, max_steps: 5, timeout_s: 60}
manual: {upvote_weight: 1.0, downvote_penalty: 0.5}
api: {auth: false, rate_limit: false, cache: false}
online_judge: {enabled: false, sample_rate: 0.1}
llm: {base_url: env, model: env, vlm_model: env}
```

- **`backend/app/config_flags.py`:** `load_flags(overrides) -> dict` (merge sâu một cấp). `config_id = sha1(json.dumps(flags, sort_keys=True))[:8]`.
- **Ablation:** mỗi thí nghiệm là một file `configs/ablation/E*.yaml`, chỉ khác một cờ.
- **Biến môi trường** (`config.py` / `.env.example`): `LLM_BASE_URL` (mặc định `https://openrouter.ai/api/v1`), `LLM_API_KEY`, `LLM_MODEL`, `VLM_MODEL`, `LLM_SSL_VERIFY`, `LLM_CA_BUNDLE`, `INDEX_VERSION`, `TRACES_DB=data/traces.db`. ID model Qwen phải kiểm tra trên openrouter.ai/models, **không hardcode**.

---

## 5. Query Enhancement và manual mode

### 5.1. `llm.py`: một chỗ tạo model

```python
def get_chat(kind="text"|"vlm", **kw) -> ChatOpenAI     # langchain_openai, base_url=OpenRouter, model từ env
def image_message(text, image_paths, max_side=768) -> HumanMessage   # content [text, image_url data:image/jpeg;base64]
```

- `ChatOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=…, timeout=…, max_retries=2, http_client=httpx.Client(verify=LLM_CA_BUNDLE or LLM_SSL_VERIFY))`.
- Dùng chung cho QE (`with_structured_output(ParsedQuery)`), rerank VLM, tool `look` và controller của agent.
- **Span:** `observe.SpanCallback(BaseCallbackHandler)` ghi `llm.call`/`vlm.call`/`tool.<tên>` từ `on_chat_model_start/on_llm_end/on_tool_start/on_tool_end` (model, `prompt_version`, số ảnh, `usage_metadata` tokens, latency). Không cần Langfuse/LangSmith; tắt LangSmith tracing (`LANGCHAIN_TRACING_V2` unset).
- Không có key → raise `LLMUnavailable` với thông báo rõ. Không có đường giả.
- **Kiểm tra trước khi chọn model** trên OpenRouter: model controller phải hỗ trợ **tool calling**, model VLM phải hỗ trợ **ảnh**. Ghi model đã chọn vào `inventory.md`.

### 5.2. `query/qe.py`

- **Một lời gọi LLM, JSON:** `{events[{text_vi, text_en, ocr[], objects[]}], answer_event, question, answer_type, variants[]}` (`variants` chỉ có khi bật `multi_query`).
- **Prompt:** `agent/prompts/qe_v1.txt`, có version. Nội dung: `text_en` là mô tả thị giác (bỏ framing tin tức và suy diễn, giữ màu/số lượng/chữ trên màn hình); tách cảnh khỏi câu hỏi; phân rã event theo thứ tự thời gian; nhãn object thuộc COCO-80 (danh sách lấy từ `object_classes.json`).
- **Regex ràng buộc chạy trước LLM** (`query/regex.py`): chuỗi trong ngoặc kép và chữ số → `ocr`; từ điển nhỏ vi→COCO → `objects`. Manual hiển thị kết quả regex ngay.
- **Fallback thật:** LLM lỗi hoặc timeout → `text_en = translate(question)` (Google), một event, và **cảnh báo hiện trên UI** (`qe.fallback=true`).
- Lưu ParsedQuery (bản LLM và bản người dùng sửa) vào SQLite (bảng `parsed_queries`).

### 5.3. Kỹ thuật theo chế độ

| Kỹ thuật | Manual | Agentic |
|---|---|---|
| Dịch + viết lại thành mô tả thị giác, tách câu hỏi, phân rã event | bật | bật |
| Regex OCR/object (hiện ngay) | bật | bật |
| Multi-query | cờ | cờ |
| Sửa field, bật/tắt event, khóa field đã sửa | **có** | không |
| Agent tự viết lại truy vấn (tool `search`) | không | **có** |

### 5.4. Upvote / chọn frame làm tín hiệu tìm kiếm (thay cho trang gợi ý)

- Người dùng 👍 hoặc chọn frame X (cũng là bằng chứng để hỏi). Lần tìm tiếp theo thêm **một list dense image→image** từ `text_vector` của X (lấy bằng `get_vectors`) vào RRF với trọng số `upvote_weight`. Nhiều frame thì mỗi frame một list.
- 👎 frame Y: ứng viên có cos > 0.92 với `text_vector` của Y bị nhân `downvote_penalty`.
- `/api/v1/search/image` (image-by-example) dùng chung đường này: một kf_id, hoặc ảnh upload encode bằng `SigLIP2Encoder.encode_image`.
- Upvote/downvote nằm trong state của phiên manual (SQLite `selections`), và được ghi vào span `search.upvote`.

---

## 6. Agent ReAct (`agent/`)

- **`react.py`:** LangGraph `StateGraph` tự định nghĩa. Không dùng `create_react_agent` prebuilt, vì cần node guard.
  ```
  State: messages, looked_kf_ids: set, candidates, steps, final: dict|None
  START → agent ──(tool_calls)──→ tools → agent
                └─(gọi final_answer)→ guard → END
  agent : get_chat("text").bind_tools([search, look, neighbors, final_answer]) — Thought là nội dung message trước tool call
  tools : ToolNode; sau mỗi `look` cập nhật looked_kf_ids (Command/state update)
  guard : kiểm tra evidence; không đạt → đổi status=uncertain, guard_changed=true
  ```

  | Tool (`tools.py`, `@tool` + Pydantic args) | Làm gì |
  |---|---|
  | `search(text_en, k)` | chạy S2→S5 thật |
  | `look(kf_ids≤4, question)` | VLM thật trên ảnh trong `KEYFRAMES_DIR` |
  | `neighbors(kf_id, direction, n)` | keyframe trước/sau theo `frame_idx`, cùng video, từ CSV ∩ Milvus |
  | `final_answer(status, answer, evidence)` | kết thúc, `status ∈ {answered, uncertain, not_found, clarify}` |

- **Guard:** `answered` phải có `evidence`, và mỗi `kf_id` trong đó phải nằm trong `looked_kf_ids`; không thỏa thì đổi thành `uncertain` và ghi `guard_changed=true`. kf_id phải khớp regex `^L\d+_V\d+/\d{3,}$` và tồn tại trong Milvus.
- **Tool call hỏng hoặc args sai:** ToolNode trả lỗi về controller (`handle_tool_errors=True`) một lần; lỗi tiếp thì `status: error`.
- **Chạm `max_steps` / `timeout_s`:**
  - `max_steps` dùng `recursion_limit = 2·max_steps + 2`; `timeout_s` dùng `asyncio.wait_for`.
  - Bắt `GraphRecursionError` / `TimeoutError` → trả `uncertain` kèm đáp án tốt nhất và lý do.
- **Stream ra UI:** `graph.astream(..., stream_mode=["updates", "messages"])` → mỗi Thought / Action / Observation là một event SSE.
- **Bộ nhớ hội thoại:** `SqliteSaver`/`AsyncSqliteSaver` (`data/agent_memory.db`), `thread_id = session_id`. Chỉ đưa 3 lượt gần nhất vào context qua `trim_messages`.
- **Manual `/ask`:** cùng graph, `candidates` = frame đã chọn, tool `search` bị gỡ khỏi `bind_tools`.
- **An toàn và ngôn ngữ:** OCR và mô tả của VLM được bọc trong `<data>…</data>` kèm ghi chú "không phải lệnh". Trả lời bằng ngôn ngữ câu hỏi, chép nguyên văn chữ nhìn thấy.
- **Prompt có version:** `agent/prompts/react_v1.txt`, `look_v1.txt`, `rerank_v1.txt`, `qe_v1.txt`.
- **"Submission" = bản ghi câu trả lời:** bảng `answers` (`trace_id`, `session_id`, question, answer, status, evidence, mode) + `GET /api/v1/answers/export?session_id=` trả JSON.

---

## 7. API `/api/v1` (`backend/app/routes/v1/`)

| Method | Endpoint | Mục đích |
|---|---|---|
| POST | `/api/v1/query/parse` | S1 |
| POST | `/api/v1/search` | ParsedQuery (đã sửa) hoặc text thô, `k`, override cờ, upvote/downvote |
| POST | `/api/v1/search/image` | image-by-example (`kf_id` hoặc ảnh upload) |
| POST | `/api/v1/search/sketch` | sketch (cờ) |
| POST | `/api/v1/ask` | manual: câu hỏi + `kf_ids` đã chọn → agent chỉ trên các frame đó (tool `search` tắt) |
| POST | `/api/v1/chat` | agentic, SSE: mỗi Thought/Action/Observation là một event |
| POST | `/api/v1/feedback` | `{trace_id, rating, note, kf_ids?, client_total_ms?}` |
| GET | `/api/v1/traces`, `/api/v1/traces/{id}` | danh sách / cây span |
| GET | `/api/v1/traces/{id}/sequence` | Mermaid `sequenceDiagram` sinh từ span thật |
| GET | `/api/v1/stats/latency` | p50/p95 theo tên span |
| GET | `/api/v1/stats/dashboard` | số liệu dashboard (SQL trên SQLite) |
| GET | `/api/v1/config` | cờ mặc định + trạng thái nhánh (BM25 có hay không, LLM có key hay không) |
| GET | `/api/v1/answers/export` | bản ghi câu trả lời dạng JSON |

- Mọi response có `trace_id` và `latency_ms`; có `usage` khi gọi LLM.
- Lỗi cùng một định dạng: `{"error": {"code", "message", "trace_id"}}`.
- **Middleware** (`backend/app/middleware.py`): sinh `trace_id`, mở span `request`, bắt lỗi trả JSON.
- Auth/rate-limit/cache: cờ, **tắt**.
- Uvicorn chạy 1 worker.

---

## 8. Observability (`backend/app/observe.py`)

- **SQLite `data/traces.db`:** bảng `traces`, `spans` đúng schema ở mục 8.2 tài liệu gốc. Thêm các bảng `parsed_queries`, `selections`, `turns`, `answers`, `feedback`.
- **`span(trace_id, name, **attrs)`:** contextmanager dùng contextvars, như mục 8.2. Không lưu ảnh (chỉ `kf_id`), không lưu API key.
- **Các span:** `request`, `qe.parse`, `search.encode_text`, `search.dense`, `search.bm25`, `search.upvote`, `fusion.rrf`, `filter`, `temporal`, `rerank`, `shot_validation`, `shot_summary`, `agent.step`, `tool.<tên>`, `llm.call`, `vlm.call`, `answer`. Thuộc tính tối thiểu theo bảng 8.3 của tài liệu gốc.
- **Sequence diagram:** sinh từ span. Participant gồm UI, API, LLM (QE), Milvus, BM25, Temporal, Rerank, Agent, VLM; mỗi mũi tên ghi số ms thật. Phần UI lấy từ `client_total_ms`.
- **Latency:** `/stats/latency` tính p50/p95 trên N trace gần nhất. Mục tiêu p95 ≤ 300ms áp cho S2→S5 (`superglobal`), báo cáo số thật kể cả khi vượt.
- **Dashboard** (một trang React đọc `/stats/dashboard`):
  - vận hành: request theo giờ, p50/p95 xếp chồng theo span, tỉ lệ lỗi/timeout/fallback;
  - truy hồi: truy vấn 0 kết quả, phân phối top-1, đóng góp dense/BM25 trong top-10, metric offline đọc từ `eval/results/`;
  - câu trả lời: phân bố status, tỉ lệ có evidence, 👍/👎 theo thời gian và `answer_type`;
  - chi phí: token/ngày, số lời gọi VLM mỗi request;
  - agent: số bước, phân phối tool, tỉ lệ lỗi tool, tỉ lệ chạm `max_steps`;
  - chỉ mục: collection, encoder, `index_version`, BM25 có hay không;
  - drill-down: lọc 👎 / 0 kết quả / `not_found` / latency cao → trace viewer;
  - **cảnh báo tô đỏ:** p95 search > 300ms, tỉ lệ 0 kết quả hoặc 👎 trong ngày cao hơn trung bình 7 ngày, lỗi tool tăng.
- **Trace viewer:** cây span cùng sequence diagram (render bằng mermaid.js qua CDN, hoặc npm `mermaid` có pin version).

---

## 9. Đánh giá (code; dữ liệu nhãn thuộc plan riêng)

- **Định dạng** `data/eval/vqa.jsonl` và `data/eval/qrels.tsv` (grade **0–3**, khoảng theo `frame_idx`, bao cả hai đầu) đúng như mục 9.1 tài liệu gốc. Ghi vào `docs/eval_format.md` để plan gán nhãn dùng.
- **`eval/metrics.py`:** `match`, `recall_at_k`, `mrr`, `ndcg_at_k` theo mục 9.3; thêm `answerable_at_k` (có grade 3 trong top-k), `answer_accuracy` (normalize lỏng: NFC, lowercase, bỏ dấu câu, so với danh sách `answers`), `evidence_accuracy` (grade ≥2), `negative_rejection`.
- **`tests/test_metrics.py`:** các ví dụ trong slide (Recall@2=0, Recall@5=1, MRR=0.25; Recall@5=0.75; MRR 3 truy vấn=0.583; một ví dụ nDCG tính tay). Thêm `tests/test_rrf.py`, `tests/test_bm25.py`, `tests/test_temporal.py` với input nhỏ viết tay.
- **`eval/evaluate.py`:**
  - `--mode retrieval|vqa`, `--config`, `--split`, `--oracle-frames`;
  - chạy đúng code của service (in-process);
  - lưu `eval/results/{config_id}_{ts}.json`, không ghi đè;
  - **thiếu file nhãn → thoát với thông báo rõ**, không tạo dữ liệu.
- **Integration test** (`tests/integration/`): gọi Milvus/LLM thật, `pytest.skip` kèm lý do khi dịch vụ không có.
- **Ma trận ablation E1–E14:** viết sẵn `configs/ablation/E*.yaml` theo mục 10 tài liệu gốc. Riêng E11 (encoder khác) để skip, vì CLIP text đã bị xóa và không re-ingest. Ablation chỉ chạy được khi đã có nhãn.
- **Error analysis:** taxonomy theo mục 11 tài liệu gốc, đưa vào `docs/error_taxonomy.md`. Trace viewer có trường gán loại lỗi (cột `error_label` trong `traces`).

---

## 10. Frontend (`frontend/src`)

- **`Layout.tsx`:** tab **Search (manual) | Chat | Dashboard | Traces**. `VideoModal`/`ImageLightbox` dùng chung (mở video đúng `pts_time`).
- **Search (manual):**
  - `ParsedQueryEditor.tsx`: event (text_vi/text_en), câu hỏi, `answer_type`, chip OCR/object; bật/tắt event; khóa field đã sửa; kết quả regex hiện ngay.
  - Panel cờ (`FlagsPanel.tsx`) map sang override.
  - Kết quả hiển thị `sources` để giải thích vì sao frame lên top.
  - 👍/👎 và "Chọn làm bằng chứng" (upvote).
  - "Tìm frame giống" (image-by-example) và sketch (khi bật cờ).
  - **Hỏi trên frame đã chọn** → `/api/v1/ask` → AnswerCard.
- **Chat:**
  - `api/sse.ts` dùng fetch stream.
  - `ChatTab`, `AgentTimeline` (thought, action + args, observation có thumbnail và chữ VLM đọc, ms và token mỗi bước).
  - `AnswerCard` (đáp án, status, frame bằng chứng, 👍/👎 gắn `trace_id`, link "Xem trace").
- **Dashboard:** `DashboardPage.tsx`. **Traces:** `TracesPage.tsx` + `TraceViewer.tsx` (cây span + Mermaid).
- **Đo phía client:** `client_total_ms` (thời điểm gửi đến byte cuối), gửi qua `/feedback` hoặc một log nhẹ cùng `trace_id`.
- **Bỏ:** dropdown encoder, toggle BGE. `SketchCanvas` chỉ hiện khi cờ sketch bật.

---

## 11. Cấu trúc thêm vào repo

```
backend/app/
  llm.py  observe.py  middleware.py  config_flags.py  pipeline.py   # pipeline chạy S1→S9 theo cờ, mỗi stage một span
  query/{qe.py, regex.py}
  retrieve/{dense.py, bm25.py, fusion.py, filter.py, temporal.py, keyframe_map.py}
  rerank/{superglobal.py, vlm.py, shot_validation.py}
  agent/{react.py (LangGraph), tools.py, context.py, prompts/*.txt}
  routes/v1/{query.py, search.py, ask.py, chat.py, feedback.py, traces.py, stats.py, answers.py}
backend/scripts/verify_frame_map.py
configs/default.yaml, configs/ablation/E*.yaml
eval/{metrics.py, evaluate.py, results/}      data/eval/ (plan riêng)
tests/{test_metrics.py, test_rrf.py, test_bm25.py, test_temporal.py, integration/}
docs/{inventory.md, eval_format.md, error_taxonomy.md}
```

---

## 12. Thứ tự làm và tiêu chí "xong"

| Bước | Việc | Xong khi |
|---|---|---|
| 0 | `docs/inventory.md` (xác nhận bảng mục 1 bằng code + `verify_frame_map.py`) | Có đủ service, collection, encoder, OCR/object, câu hỏi; script không còn lỗi hoặc đã liệt kê đủ |
| 1 | Dọn dẹp + sửa nền (mục 2); `observe.py` + middleware + lỗi JSON + SQLite | `/api/search` chạy với contract mới; `npm run build` không lỗi; mọi request `/api/v1` có trace trong DB |
| 2 | `eval/metrics.py` + unit test theo slide + `evaluate.py` + định dạng eval | `pytest tests` pass; `evaluate` báo lỗi rõ khi chưa có nhãn |
| 3 | S2: dense + BM25 + RRF + `/api/v1/search` + `/search/image` + upvote | Search trả `sources`; span dense/bm25/rrf có trong DB; p95 đo được |
| 4 | `llm.py` + S1 + `/api/v1/query/parse` + UI manual (editor, chọn frame, cờ) | Parse thật qua OpenRouter; fallback hiện cảnh báo |
| 5 | S5 superglobal + vlm; S4 temporal | Có span rerank/temporal; rerank vlm quá giờ thì fallback |
| 6 | Agent ReAct + guard + `/api/v1/ask` + `/api/v1/chat` SSE + timeline + AnswerCard + bảng `answers` | Chat chạy end-to-end với VLM thật; guard đổi status khi thiếu evidence |
| 7 | Trace viewer + sequence diagram + dashboard + feedback + `client_total_ms` | Mở được trace bất kỳ; có bảng p50/p95 |
| 8 | Optional: filter, shot validation, frame input, shot summary, sketch (QuickGELU) | Mỗi cái có cờ, mặc định tắt; có file ablation |
| 9 | (Sau khi plan gán nhãn xong) chạy ablation trên dev/test, error analysis, báo cáo | Bảng kết quả từ `eval/results/` |

Không chuyển sang bước sau khi bước trước chưa đạt tiêu chí "xong".

---

## 13. Verification

- `python -m backend.scripts.verify_frame_map` chạy sạch, hoặc liệt kê đủ các video bị lệch.
- `pytest tests` pass (hàm thuần). `pytest tests/integration` chạy với Milvus + key thật; dịch vụ nào vắng thì skip có lý do.
- **Sketch:** encode 50 keyframe bằng `SketchEncoder` và so với `tsbir-features/tsbir_features.pkl`, cos trung bình ≥ 0.9995.
- **End-to-end:**
  1. `docker compose up` → `uvicorn app.main:app --workers 1` (từ `backend/`) → `npm run dev`.
  2. **Manual:** nhập câu hỏi có chữ trong ngoặc kép → regex hiện chip ngay → Parse → sửa `text_en` → Tìm (thấy `sources` dense/bm25) → 👍 một frame → tìm lại (span `search.upvote` có trong trace) → chọn 2 frame → Hỏi → AnswerCard có evidence.
  3. **Chat:** timeline hiện từng Thought/Action/Observation; câu trả lời có link trace; sequence diagram hiển thị ms thật từng bước.
  4. **Tắt `LLM_API_KEY`:** `/api/v1/config` báo LLM tắt; parse fallback có cảnh báo; chat báo lỗi rõ, không trả lời giả.
  5. **Dashboard:** có số liệu từ các request vừa chạy; 👎 hiện trong drill-down.
