# Schema dữ liệu Video RAG Eval v1

Bộ eval pilot nằm ở `data/video_rag_eval/benchmark/v1/` (artifact cục bộ, không commit vào Git). Phiên bản hiện tại có 20 QA: 6 `dev`, 14 `test`, và 416 nhãn qrel. Đây là **silver/pseudo-ground-truth** do MLLM sinh và kiểm chứng, chưa phải nhãn chuyên gia.

## Quy ước chung

- `query_id` (ví dụ `Q0001`) là khóa nối giữa các file. Mỗi `queries.jsonl`, `answers.jsonl`, `evidence.jsonl` có đúng một dòng cho mỗi `query_id`.
- JSONL dùng UTF-8; mỗi dòng là một object JSON độc lập. TSV dùng tab làm dấu phân cách.
- `frame_id`/`kf_id` có dạng `L22_V019/054`: phần trước dấu `/` là `video_id`, phần sau là chỉ số keyframe. `source_frame_id` và `frame_ids` dùng cùng định dạng.
- Các file nhãn đóng băng nằm trong `benchmark/v1/`; kết quả chạy mô hình nằm ở `data/video_rag_eval/runs/`, điểm chấm ở `data/video_rag_eval/eval/`.

## `queries.jsonl`

| Trường | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `query_id` | string | Khóa QA, duy nhất. |
| `description_vi` | string | Mô tả cảnh để truy hồi; không chứa đáp án. |
| `question_vi` | string | Câu hỏi VQA bằng tiếng Việt. |
| `question_type` | string | Loại câu hỏi, ví dụ `visible_action`, `ocr_signage`, `unanswerable`. |
| `answerable` | boolean | Có thể trả lời từ ảnh nguồn hay không. |
| `split` | `dev` hoặc `test` | Tập đánh giá; tách theo video/nhóm video trùng lặp. |

```json
{"query_id":"Q0001","description_vi":"Mô tả cảnh không tiết lộ đáp án","question_vi":"Người trong ảnh đang làm gì?","question_type":"visible_action","answerable":true,"split":"test"}
```

## `answers.jsonl`

| Trường | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `query_id` | string | Nối với `queries.jsonl`. |
| `reference_answer` | string | Đáp án tham chiếu; với câu không trả lời được, xem thêm `answerable`. |
| `required_claims` | string[] | Các mệnh đề cần đúng trong câu trả lời. |
| `answerable` | boolean | Cờ khả năng trả lời sau bước xác minh. |
| `label_status` | string | Trạng thái nhãn, hiện là `verified_silver`. |

## `evidence.jsonl`

| Trường | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `query_id` | string | Nối với câu hỏi và đáp án. |
| `source_video_id` | string | Video nguồn của đề xuất QA. |
| `source_frame_id` | string | Keyframe nguồn của đề xuất QA. |
| `temporal_relation` | string hoặc null | Quan hệ thời gian nếu QA cần nhiều frame; hiện dùng `ordered` hoặc `null`. |
| `evidence_sets` | object[] | Một hoặc nhiều tập ảnh đủ làm bằng chứng. |
| `judge_ids` | string[] | Các lượt judge đã xác minh nhãn. |

Mỗi phần tử `evidence_sets` gồm `set_id` (string), `frame_ids` (string[]), `required_all` (boolean) và `crop_paths` (string[]). `required_all=true` nghĩa là cần đủ các frame trong tập đó để kiểm chứng câu trả lời.

## `qrels.tsv` và `accepted_ranges.tsv`

`qrels.tsv` **không có header**. Bốn cột theo thứ tự: `query_id`, `iteration` (luôn `0`), `frame_id`, `grade`. `grade` là số nguyên 0–3; nhãn liên quan tích cực dùng ngưỡng `grade >= 2` theo cấu hình đánh giá.

```text
Q0001\t0\tL21_V001/203\t0
```

`accepted_ranges.tsv` **có header**: `query_id`, `video_id`, `lo_frame_idx`, `hi_frame_idx`, `grade`, `source_frame_id`. Hai cột `lo_frame_idx` và `hi_frame_idx` là biên bao gồm của khoảng frame video gốc; hit chỉ khớp khi cùng video và `frame_idx` nằm trong khoảng. Khoảng này dùng cho truy hồi/citation, không tự chứng minh nội dung ảnh.

## File liên quan

- `splits.json`: danh sách video `dev`, `test` và các nhóm video trùng lặp.
- `benchmark_manifest.json`: phiên bản, số lượng QA/qrel, seed, model và hash để kiểm tra tính tái lập.
- `runs/E1.jsonl` … `runs/E8.jsonl`: dự đoán của từng cấu hình, có `query_id`, `retrieved_hits`, `context_hits`, `context_image_paths`, `generated_answer`, `citations`, `status`.
- `eval/E*_scores.jsonl`: điểm theo từng `query_id`; `eval/E*_summary.json`: tổng hợp điểm.

Khi đọc dữ liệu, nên kiểm tra `query_id` khớp giữa ba file JSONL và mọi `frame_id` trong evidence/qrels đều theo đúng quy ước trên. Xem thêm [hướng dẫn pipeline](../video_rag_eval/README.md).
