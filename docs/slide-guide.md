# Slide Guide — AIC 2025: Hệ thống truy xuất sự kiện video đa phương thức

> Dùng file này làm source cho NotebookLM để tạo slide thuyết trình.
> Mỗi section = 1 slide. Tổng 12 slides.
> Chi tiết kiến trúc xem [architecture-detail.md](architecture-detail.md)

---

## Slide 1: Title

**Hệ thống truy xuất sự kiện video dựa trên biểu diễn đa phương thức**

- Cuộc thi: AI Challenge 2025 (AIC 2025)
- GVHD: TS. Phạm Đỗ Kim Chi
- Thành viên: Trang Kỳ Anh (250101006), Lê Nguyễn Bảo Hân (250101017), Đặng Anh Đạt (250101009)
- Trường Đại học Công Nghệ Thông Tin — ĐHQG TP.HCM

---

## Slide 2: Bài toán

**Bài toán Known-Item Search (KIS):** Cho một đoạn mô tả bằng tiếng Việt (nhiều câu), tìm đúng keyframe trong tập dữ liệu lớn gồm hàng trăm ngàn keyframes từ video tin tức.

**Các loại truy vấn:**
- **KIS** (18 queries): Mô tả cảnh cụ thể → tìm đúng keyframe
- **QA** (3 queries): Câu hỏi → tìm keyframe chứa câu trả lời (thường cần OCR)
- **TRAKE** (3 queries): Theo dõi sự kiện thời gian — ngoài phạm vi single-frame retrieval

Mỗi truy vấn cho phép nộp tối đa **100 câu trả lời** xếp hạng. Đánh giá trên 21 truy vấn (bỏ TRAKE).

---

## Slide 2b: Metric thi đấu — Mean of Top-k R-Scores

**Bước 1 — R-Score cho từng câu trả lời:**
- **Textual-KIS**: Câu trả lời = (video_name, frame_idx)
  - R-Score(r_i) = 1 nếu video_name khớp **VÀ** frame_idx ∈ [s, e]
- **Visual QA**: Câu trả lời = (video_name, frame_idx, answer_text)
  - R-Score(r_i) = 1 nếu video đúng **VÀ** frame đúng **VÀ** answer_text khớp GT

**Bước 2 — Best-in-top-k:**
- Với mỗi k ∈ {1, 5, 20, 50, 100}: Best@k = max R-Score trong top k kết quả
- Nếu đáp án đúng xuất hiện ở bất kỳ vị trí nào trong top-k → điểm tối đa cho ngưỡng đó

**Bước 3 — Final Score:**
- Final Score = (1/5) × Σ Best@k qua 5 ngưỡng
- Điểm tối đa 1.0 = đáp án đúng luôn ở vị trí 1
- Điểm 0.6 = đáp án đúng từ vị trí 3 trở đi (được điểm cho k≥5 nhưng không cho k=1)

**Metric phụ:** Recall@k (tỷ lệ query có kết quả đúng trong top-k), MRR (trung bình 1/rank của kết quả đúng đầu tiên)

---

## Slide 3: Tập dữ liệu gốc

**Nguồn:** 10 batch video tin tức tiếng Việt (L21–L30) do BTC cung cấp.

| Chỉ số | Giá trị |
|---|---|
| Tổng video | 873 |
| Tổng keyframes (sau trích xuất) | 177.321 |
| Trung bình keyframes/video | ~203 |

**Dữ liệu đa phương thức cho mỗi keyframe:**
- **Ảnh** (visual): file .jpg tại scene change boundaries
- **Text** (OCR): văn bản nhìn thấy trong frame (biển hiệu, phụ đề, logo kênh)
- **Metadata**: video_name, frame_idx, pts_time, fps
- **Object labels**: nhãn đối tượng phát hiện trong frame

→ Bài toán yêu cầu khai thác **cả ảnh, text, và metadata** — do đó cần biểu diễn đa phương thức.

---

## Slide 4: Tiền xử lý & Trích xuất đặc trưng

[Hình: docs/diagrams/data-pipeline.mmd]

**Bước 1 — Trích xuất keyframe:** Scene change detection → .jpg files

**Bước 2 — Lọc frame trống (blank frame filtering):**
- Frame chuyển cảnh (toàn trắng/đen) gây nhiễu tìm kiếm
- Tiêu chí: mean pixel > 220 & std < 30 (trắng) hoặc mean < 15 & std < 10 (đen)
- Loại bỏ trước khi trích xuất đặc trưng

**Bước 3 — 4 bộ trích xuất chạy song song trên mỗi keyframe:**

| Extractor | Output | Mục đích |
|---|---|---|
| **TSBIR ViT-B-16** | clip_vector 512-dim | Embedding cho sketch/image search |
| **SigLIP2** | text_vector 768-dim | Embedding cho text search |
| **Florence-2** | OCR text (string) | Văn bản nhìn thấy trong frame |
| **Faster R-CNN** | Object labels (max 20) | Nhãn đối tượng, conf ≥ 0.3 |

**Bước 4 — Nạp vào Milvus:** Gộp features → pickle → batch insert 1000 records

---

## Slide 5: Lựa chọn mô hình & Tham số

**Tại sao chọn các mô hình này?**

| Mô hình | Lý do chọn | Tham số quan trọng |
|---|---|---|
| **SigLIP2** (`siglip2-base-patch16-512`) | Sigmoid loss → text-image alignment tốt hơn CLIP; patch-level attention | max_length=64 tokens, L2-norm |
| **TSBIR ViT-B-16** (open_clip + checkpoint) | Fine-tuned chuyên cho sketch-based image retrieval | 512-dim, COSINE similarity |
| **BGE-reranker-v2-m3** (568M params) | Cross-encoder đa ngôn ngữ, xử lý Việt-Anh hỗn hợp | FP16 inference, batch scoring |
| **Florence-2** (microsoft/Florence-2-base) | OCR đa ngôn ngữ, nhẹ, chạy offline | Task: OCR |
| **Faster R-CNN** (inception_resnet_v2) | Phát hiện 80 lớp đối tượng COCO-style | conf threshold ≥ 0.3, max 20 labels |

**Thiết kế dual-encoder:**
- Chỉ 1 encoder trên GPU tại một thời điểm (lazy-load pattern)
- SigLIP2 mặc định cho text query → trường `text_vector` (768-dim)
- TSBIR kích hoạt khi sketch query → trường `clip_vector` (512-dim)
- Chuyển đổi qua API `/api/switch-encoder`

---

## Slide 6: Lưu trữ — Milvus Vector DB

**Schema collection `aic25_keyframes`:**

| Trường | Kiểu | Index |
|---|---|---|
| id (PK) | VARCHAR(32) | — |
| video_name | VARCHAR(16) | — |
| keyframe_idx, frame_idx | INT | — |
| **clip_vector** | FLOAT_VECTOR(512) | HNSW (M=32, ef=256, COSINE) |
| **text_vector** | FLOAT_VECTOR(768) | HNSW (M=32, ef=256, COSINE) |
| objects | ARRAY\<VARCHAR\>(20) | — |
| ocr | VARCHAR(4096) | — |

**Tại sao HNSW?**
- Approximate Nearest Neighbor — sub-linear search trên 177K vectors
- COSINE metric phù hợp với L2-normalized embeddings
- ef_search=512 cân bằng accuracy vs latency

**Hai index riêng biệt** cho hai encoder → mỗi loại truy vấn tìm trên không gian phù hợp.

---

## Slide 7: Kiến trúc hệ thống tổng quan

[Hình: docs/diagrams/architecture.mmd]

**3 lớp:**
- **Frontend** (React + Vite + Bulma): Nhập truy vấn text/sketch, hiển thị grid kết quả, lightbox, bộ lọc
- **Backend** (FastAPI): Dịch thuật → Tách câu → Encode → Tìm kiếm → Rerank
- **Storage** (Milvus Docker): Dual-vector HNSW index

**Kiến trúc hai tầng (Two-tier):**
- Tầng 1 — Bi-encoder (SigLIP2): Nhanh, coarse — cosine trên HNSW index
- Tầng 2 — Cross-encoder (BGE): Chậm hơn, precise — cross-attention đọc OCR + objects

---

## Slide 8: Pipeline truy xuất chi tiết

[Hình: docs/diagrams/two-tier.mmd]

**Sentence Splitting + RRF Fusion:**
1. Truy vấn (đoạn văn VI) → Tách thành N câu tại dấu chấm/xuống dòng (min 10 ký tự)
2. Mỗi câu → Google Translate VI→EN → SigLIP2 encode → vector 768-dim
3. N vectors → N ANN search song song trên Milvus
4. **RRF fusion** merge N danh sách: score(d) = Σ 1/(k + rank_i), k=60
5. Top 200 ứng viên → **BGE rerank** (query, objects+OCR) → Top 100

**Tại sao tách câu?**
- Đoạn mô tả chứa nhiều khía cạnh: "4 phi hành gia mặc đen" + "nghiên cứu cực quang"
- Mỗi câu match một phần khác nhau của keyframe → RRF tổng hợp tín hiệu → tăng recall

**Tại sao reranking?**
- Cross-encoder đọc được **OCR text** + **object labels** — thông tin bi-encoder không dùng trực tiếp
- Đặc biệt hữu ích cho QA queries tham chiếu text (biển hiệu "FANA", logo kênh)

---

## Slide 9: Phương pháp đánh giá & Ground Truth

**Cách tính điểm cuộc thi:**
1. Mỗi câu trả lời (video_name, frame_idx): R-Score = 1 nếu đúng video VÀ frame ∈ [s,e]
2. Best@k = max R-Score trong top k (k ∈ {1, 5, 20, 50, 100})
3. Final Score = trung bình 5 giá trị Best@k

**Xây dựng Ground Truth (search-assisted + anti-contamination):**
1. **Search**: Tìm bằng SigLIP2 (tách câu + RRF + rerank) → top-50 ứng viên
2. **Annotate**: Annotator chọn frame đúng từ ứng viên (hoặc browse thủ công)
3. **Frame range**: Tolerance [frame_idx ± 30] — tính sai lệch sampling
4. **Rewrite (anti-contamination)**: Viết `eval_query` — mô tả lại cảnh bằng lời khác **sau khi đã thấy frame**
5. **Evaluate**: Dùng eval_query → model chưa "thấy" query này → không bias

**Công cụ annotation:** Web UI tự phát triển (localhost:8501) — search proxy + manual browse + export JSON

---

## Slide 10: Kết quả thí nghiệm

Đánh giá trên 21 truy vấn (18 KIS + 3 QA), luôn tách câu + RRF:

| Cấu hình | TopK-R ↑ | MRR ↑ | R@1 | R@5 | R@20 | R@50 | R@100 |
|---|---|---|---|---|---|---|---|
| SigLIP2 + Tách câu + Rerank | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| SigLIP2 + Tách câu (không rerank) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

**Nhận xét dự kiến:**
- Reranking cải thiện cho QA queries (cần OCR) nhưng latency tăng (~12s cho 200 candidates trên CPU)
- Tách câu cải thiện recall cho truy vấn nhiều câu (trung bình ~3 câu/truy vấn)

---

## Slide 11: Phân tích lỗi & Hạn chế

**3 loại lỗi chính:**

1. **Semantic gap** — Encoder không hiểu ngữ cảnh phức tạp
   - Bi-encoder nén cả câu thành 1 vector → mất chi tiết
   - Ví dụ: "4 phi hành gia mặc áo đen" match sai vì chỉ nhìn thấy bề ngoài

2. **Chất lượng OCR tiếng Việt** — Florence-2 không optimize cho tiếng Việt
   - Mất dấu, text nhỏ không đọc được → ảnh hưởng QA queries

3. **Cross-encoder thiếu visual** — BGE chỉ đọc text (objects + OCR), không "nhìn" ảnh
   - Frame ít text/objects → cross-encoder có ít tín hiệu để xếp hạng

**Hướng phát triển:**
- Image captioning (BLIP-2/LLaVA) thay object labels → input phong phú hơn cho reranker
- Vietnamese OCR chuyên biệt (VietOCR/PaddleOCR) thay Florence-2
- Multilingual encoder → encode trực tiếp tiếng Việt, bỏ bước dịch
- GPU inference cho BGE → latency < 1s (hiện ~12s trên CPU)
- SigLIP2-large thay base

---

## Slide 12: Kết luận

**Đóng góp chính:**
- Hệ thống **hai tầng** kết hợp bi-encoder (nhanh) + cross-encoder (chính xác)
- **Dual-encoder**: SigLIP2 cho text (768-dim) + TSBIR ViT-B-16 cho sketch (512-dim)
- **Tách câu + RRF**: Decompose đoạn văn nhiều câu → encode riêng → merge thông minh
- **Đa phương thức**: Khai thác cả visual embedding, OCR text (Florence-2), object labels (Faster R-CNN)
- **Lazy-load**: Chỉ 1 encoder trên GPU → chạy được trên máy giới hạn VRAM
- Hỗ trợ tiếng Việt qua dịch tự động (Google Translate)

**Cảm ơn!**

---

## Ghi chú cho NotebookLM

- Mỗi section = 1 slide. Thứ tự: Bài toán → Data → Tiền xử lý → Mô hình → Lưu trữ → Kiến trúc → Pipeline → Đánh giá → Kết quả → Phân tích → Kết luận
- Các bảng [TODO] cần được điền số liệu thực sau khi chạy evaluation
- Slide 4, 7, 8 có hình từ docs/diagrams/*.png (đã render từ Mermaid)
- Giọng thuyết trình: technical nhưng accessible
- Thời lượng ước tính: 10-15 phút
