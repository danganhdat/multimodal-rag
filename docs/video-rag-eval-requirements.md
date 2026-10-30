# Đối chiếu PLAN_VIDEO_RAG_EVAL.md

SHA-256 plan gốc: `9a10355ce811b0545712e067be577780a9a70e7056a8d2f24f2901b0513ebc66`.

Trạng thái được tạo từ artifact/kiểm tra hiện có. `một phần` không được tính là nghiệm thu.

| Mục trong plan | Trạng thái | Code / artifact / test | Việc còn thiếu |
|---|---|---|---|
| 0 | đạt | video_rag_eval/labels.py · benchmark/v1/*.jsonl; benchmark/v1/qrels.tsv; video_rag_eval/corpus.py · derived/index_verification.json |  |
| 1 | một phần | video_rag_eval/corpus.py · derived/frames.parquet · derived/dataset_report.json; video_rag_eval/corpus.py · tests/test_video_rag_eval.py | 26003 valid frame rows; 128/873 videos scanned; 0 quarantined |
| 2 | đạt | video_rag_eval/labels.py · benchmark/v1/*.jsonl; benchmark/v1/qrels.tsv; video_rag_eval/ranges.py · benchmark/v1/accepted_ranges.tsv · tests/test_video_rag_eval.py |  |
| 3 | đạt | video_rag_eval/corpus.py · derived/sample_report.json; video_rag_eval/labels.py · derived/api_preflight.json; video_rag_eval/labels.py · derived/frame_facts.jsonl; video_rag_eval/labels.py · derived/proposals.jsonl; video_rag_eval/labels.py · derived/judge_votes.jsonl; video_rag_eval/labels.py · derived/pool_judgments.jsonl; video_rag_eval/labels.py · benchmark/v1/splits.json |  |
| 4 | đạt | video_rag_eval/scoring.py · eval/metric_smoke.json · tests/test_video_rag_eval.py; video_rag_eval/scoring.py · eval/E*_summary.json |  |
| 5 | đạt | video_rag_eval/scoring.py · eval/E*_scores.jsonl |  |
| 6 | đạt | derived/pool_judgments.jsonl · benchmark/v1/qrels.tsv; video_rag_eval/labels.py · benchmark/v1/splits.json; video_rag_eval/labels.py · derived/pool_judgments.jsonl; derived/pool_judgments.jsonl |  |
| 7 | đạt | video_rag_eval/labels.py · derived/frame_facts.jsonl; video_rag_eval/labels.py · derived/proposals.jsonl; video_rag_eval/labels.py · derived/judge_votes.jsonl |  |
| 8 | một phần | video_rag_eval/corpus.py · derived/frames.parquet · derived/dataset_report.json; video_rag_eval/labels.py · derived/api_preflight.json; video_rag_eval/labels.py · derived/frame_facts.jsonl; video_rag_eval/labels.py · derived/proposals.jsonl; video_rag_eval/labels.py · derived/judge_votes.jsonl; video_rag_eval/labels.py · derived/pool_judgments.jsonl; video_rag_eval/labels.py · benchmark/v1/*.jsonl; video_rag_eval/runner.py · runs/E1.jsonl; video_rag_eval/scoring.py · eval/E*_summary.json | 26003 valid frame rows; 128/873 videos scanned; 0 quarantined |
| 9 | một phần | benchmark/v1/benchmark_manifest.json; video_rag_eval/scoring.py · eval/E*_summary.json; video_rag_eval/runner.py · runs/E1.jsonl; video_rag_eval/runner.py · runs/E2.jsonl; video_rag_eval/runner.py · runs/E3.jsonl; video_rag_eval/runner.py · runs/E4.jsonl; video_rag_eval/runner.py · runs/E5.jsonl; video_rag_eval/runner.py · runs/E6.jsonl; video_rag_eval/runner.py · runs/E7.jsonl; video_rag_eval/runner.py · runs/E8.jsonl | 14-query run uses per-frame captions plus images; grouped shot-summary-before-agent is not implemented in this pilot |
| 10 | đạt | docs/video-rag-eval-data-report.md; video_rag_eval/scoring.py · eval/E*_summary.json |  |
| 11 | đạt | video_rag_eval/labels.py · derived/judge_votes.jsonl; video_rag_eval/labels.py · derived/pool_judgments.jsonl; derived/pool_judgments.jsonl · benchmark/v1/qrels.tsv |  |
| 12 | đạt | docs/video-rag-eval-data-report.md · source references · eval/metric_smoke.json |  |
| M0 | một phần | video_rag_eval/corpus.py · derived/frames.parquet · derived/dataset_report.json; video_rag_eval/corpus.py · tests/test_video_rag_eval.py | 26003 valid frame rows; 128/873 videos scanned; 0 quarantined |
| M1 | một phần | video_rag_eval/corpus.py · derived/sample_report.json; video_rag_eval/config.yaml · benchmark/v1/queries.jsonl | question-type quota deficits: {"ocr_signage": {"actual": 2, "target": 5}} |
| M2 | đạt | video_rag_eval/labels.py · derived/api_preflight.json; video_rag_eval/labels.py · derived/judge_votes.jsonl; video_rag_eval/labels.py · derived/claim_verification.jsonl |  |
| M3 | đạt | video_rag_eval/labels.py · derived/pool_judgments.jsonl; benchmark/v1/qrels.tsv; derived/pool_judgments.jsonl |  |
| M4 | đạt | video_rag_eval/labels.py · benchmark/v1/*.jsonl; video_rag_eval/labels.py · benchmark/v1/answers.jsonl · tests/test_video_rag_eval.py; video_rag_eval/labels.py · derived/claim_verification.jsonl; benchmark/v1/benchmark_manifest.json; video_rag_eval/labels.py · benchmark/v1/splits.json; benchmark/v1/evidence.jsonl · D:/keyframes |  |
| M5 | đạt | video_rag_eval/scoring.py · eval/metric_smoke.json · tests/test_video_rag_eval.py; video_rag_eval/scoring.py · eval/E*_summary.json; video_rag_eval/scoring.py · eval/E*_scores.jsonl |  |
| M6 | một phần | video_rag_eval/runner.py · runs/E1.jsonl; video_rag_eval/runner.py · runs/E2.jsonl; video_rag_eval/runner.py · runs/E3.jsonl; video_rag_eval/runner.py · runs/E4.jsonl; video_rag_eval/runner.py · runs/E5.jsonl; video_rag_eval/runner.py · runs/E6.jsonl; video_rag_eval/runner.py · runs/E7.jsonl; video_rag_eval/runner.py · runs/E8.jsonl | 14-query run uses per-frame captions plus images; grouped shot-summary-before-agent is not implemented in this pilot |
| E1 | đạt | video_rag_eval/runner.py · runs/E1.jsonl |  |
| E2 | đạt | video_rag_eval/runner.py · runs/E2.jsonl |  |
| E3 | đạt | video_rag_eval/runner.py · runs/E3.jsonl |  |
| E4 | đạt | video_rag_eval/runner.py · runs/E4.jsonl |  |
| E5 | đạt | video_rag_eval/runner.py · runs/E5.jsonl |  |
| E6 | đạt | video_rag_eval/runner.py · runs/E6.jsonl |  |
| E7 | một phần | video_rag_eval/runner.py · runs/E7.jsonl | 14-query run uses per-frame captions plus images; grouped shot-summary-before-agent is not implemented in this pilot |
| E8 | đạt | video_rag_eval/runner.py · runs/E8.jsonl |  |
| QG01 | đạt | video_rag_eval/labels.py · benchmark/v1/*.jsonl; video_rag_eval/labels.py · derived/judge_votes.jsonl |  |
| QG02 | đạt | benchmark/v1/evidence.jsonl · D:/keyframes; video_rag_eval/labels.py · derived/judge_votes.jsonl; video_rag_eval/labels.py · derived/claim_verification.jsonl |  |
| QG03 | đạt | derived/pool_judgments.jsonl; benchmark/v1/qrels.tsv; derived/pool_judgments.jsonl · benchmark/v1/qrels.tsv |  |
| QG04 | đạt | benchmark/v1/qrels.tsv; video_rag_eval/ranges.py · benchmark/v1/accepted_ranges.tsv · tests/test_video_rag_eval.py |  |
| QG05 | đạt | derived/judge_votes.jsonl · derived/crops |  |
| QG06 | đạt | derived/judge_votes.jsonl · D:/map-keyframes; video_rag_eval/corpus.py · derived/index_verification.json |  |
| QG07 | một phần | video_rag_eval/labels.py · benchmark/v1/splits.json | exact image hashes do not cross dev/test; global cross-video near-scene dedup is not covered by the pilot |
| QG08 | đạt | video_rag_eval/scoring.py · eval/E*_summary.json |  |
| QG09 | đạt | video_rag_eval/scoring.py · eval/metric_smoke.json · tests/test_video_rag_eval.py |  |
| QG10 | đạt | docs/video-rag-eval-data-report.md |  |
| QG11 | đạt | .gitignore · derived/api_usage.jsonl |  |

## Khác biệt có chủ ý

- Pilot xác minh vector SigLIP2 trong Milvus trên mẫu; lệnh `reembed` tạo lại toàn bộ embedding khi không khớp.
- Generator và hai judge dùng cùng checkpoint Qwen3.8 Flash với prompt độc lập; đây là giới hạn về tính độc lập của nhãn silver.
- `qrels.tsv` giữ nhãn keyframe; `accepted_ranges.tsv` là phép chiếu sang các frame video được chấp nhận để chấm truy hồi.
- Các metric cần MLLM/RAGAS chỉ được đánh dấu đạt sau khi đã chạy trên ảnh thật.

- E7 pilot dùng caption từng frame kèm ảnh; đây là proxy của Shot Summary. Bản gom shot và cấp summary trước agent trong plan gốc chưa được triển khai.

- Quota loại câu hỏi là mục tiêu lấy mẫu. Các câu không qua gate bị loại; bảng kiểm tra ghi rõ quota thiếu thay vì bù bằng nhãn chưa được kiểm chứng.

- Split chặn trùng video và ảnh có cùng hash. Dedup gần giống hiện chạy theo video; chưa có kiểm tra near-scene toàn cục giữa các video, nên gate tương ứng giữ trạng thái một phần.
