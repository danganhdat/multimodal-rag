# Video RAG Eval

Schema các file benchmark 20 QA: [docs/video-rag-eval-schema.md](../docs/video-rag-eval-schema.md).

Pipeline tạo benchmark silver và đánh giá E1–E8 cho corpus keyframe hiện có. Chạy từ thư mục gốc repo bằng `.venv\Scripts\python.exe -m video_rag_eval <stage>`; cấu hình tại [config.yaml](config.yaml). Artifact được lưu ở `data/video_rag_eval/` và không commit vào Git.

## Chạy pilot

1. Cài `video_rag_eval/requirements.txt` trong `.venv`, bật Milvus tại `localhost:19530`, cấu hình OpenRouter **inference** key bằng `LLM_API_KEY` hoặc `OPENROUTER_API_KEY` trong môi trường hay `.env` của repo. Management/provisioning key không gọi được endpoint chat. Không lưu key trong config/artifact.
2. Chạy `inventory --source-only`, `verify-index`, `sample`, `facts`, `generate`, `verify`, `claims`, `pool`, `freeze`, `smoke-metrics` theo thứ tự. `claims` tách đáp án thành các khẳng định cần thiết rồi kiểm chứng trên chuỗi ảnh nguồn; `verify-index` bắt buộc trước khi pool và run. Mỗi stage ghi tiến độ JSONL để chạy lại không gọi trùng phần đã hoàn tất.
3. Chạy `run --experiment E1` đến `E8`, sau đó `adjudicate` để chấm bổ sung hợp các kết quả top context mà không sửa benchmark đã đóng băng.
4. Chạy `score --run data/video_rag_eval/runs/E1.jsonl` đến `E8.jsonl`, rồi `report` và `audit`. `--offline` chỉ dùng để thử phép chấm truy hồi khi không muốn gọi judge/RAGAS.

Mặc định target 20 QA và trần chi phí ước tính 25 USD; sửa `target_qa` và `max_usd` trong config để đổi. CLI dùng giá token được cấu hình cho Qwen3.8 Flash; một số thư viện không trả token usage nên được tính allowance cố định, do đó số USD là ước tính. E8 ghi token usage của controller và VLM qua callback; RAGAS dùng allowance cố định. CLI giữ lại 0,10 USD cuối hạn mức để có khoảng đệm cho các lượt gọi đang chạy.

## Nhãn và phép chấm

`qrels.tsv` là nhãn gốc ở cấp keyframe (`query_id  0  frame_id  grade`). `accepted_ranges.tsv` chiếu từng qrel sang khoảng frame video gốc từ cột `frame_idx` của CSV. Keyframe 351 với hai hàng kề 261 và 411 tạo `[262,410]` đóng. Hit chỉ khớp khi cùng `video_id` và `frame_idx` thuộc khoảng. Khoảng giao nhau được gộp để tránh cộng điểm lặp. Candidate chưa judge không bị coi là grade 0.

Visual Context Recall và Faithfulness được judge trên đường dẫn **ảnh/crop thật trong `context_image_paths`**. Các khoảng chỉ dùng cho phép chấm truy hồi, citation và correctness theo frame; chúng không tự chứng minh nội dung ảnh.

Xem [báo cáo pipeline](../docs/video-rag-eval-data-report.md) và [đối chiếu plan](../docs/video-rag-eval-requirements.md). Hai file này được tạo lại bởi `report` và `audit` từ artifact hiện có.
