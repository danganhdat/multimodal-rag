# Evaluation Data Format

## `data/eval/qrels.tsv`

Tab-separated, no header. Each line is one relevance judgment:

```
query_id    lo_frame_idx    hi_frame_idx    grade
```

| Column | Type | Description |
|---|---|---|
| `query_id` | str | Matches `query_id` in `vqa.jsonl` |
| `lo_frame_idx` | int | Start of relevant frame_idx range (inclusive) |
| `hi_frame_idx` | int | End of relevant frame_idx range (inclusive) |
| `grade` | int | 0=not relevant, 1=marginally, 2=relevant, 3=answer frame (highly relevant) |

**Important:** Ranges use `frame_idx` (actual video frame), not `n` (keyframe number). Map via `map-keyframes/{video}.csv`.

Example:
```
q001	260	410	3
q001	100	259	1
q002	1500	1600	2
```

## `data/eval/vqa.jsonl`

One JSON object per line:

```json
{
  "query_id": "q001",
  "question": "What text appears on the red banner?",
  "answers": ["SALE 50%", "sale 50%"],
  "is_answerable": true,
  "video": "L21_V001"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `query_id` | str | yes | Unique query identifier |
| `question` | str | yes | The VQA question in Vietnamese or English |
| `answers` | list[str] | yes | Acceptable answer strings (matched with normalization) |
| `is_answerable` | bool | yes | Whether the question can be answered from the video |
| `video` | str | no | Video name hint (for filtering) |

## Metrics Computed

### Retrieval (mode=retrieval)
- **Recall@50, Recall@100** — fraction of relevant items found
- **nDCG@10** — graded relevance ranking quality
- **MRR** — reciprocal rank of first relevant result

### VQA (mode=vqa)
- **Answerable@3** — is there a grade-3 frame in top 3?
- **Answer Accuracy** — normalized string match against ground truth
- **Evidence Accuracy** — fraction of evidence frames in relevant ranges
- **Negative Rejection** — correctly refuses unanswerable questions

## Running

```bash
# Retrieval evaluation
python -m eval.evaluate --mode retrieval --config configs/default.yaml

# VQA evaluation
python -m eval.evaluate --mode vqa --config configs/default.yaml

# With a specific split
python -m eval.evaluate --mode retrieval --split test
```

Results are saved to `eval/results/{config_id}_{timestamp}.json`.
