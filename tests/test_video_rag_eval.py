"""Acceptance semantics for original-video frame intervals."""

from video_rag_eval.ranges import interval, make_ranges, match, positive_units
from video_rag_eval.scoring import context_precision, retrieval_scores
from video_rag_eval.core import CostLedger, config, output, read_jsonl, write_json, write_jsonl
from video_rag_eval.corpus import load_maps
from video_rag_eval.documentation import audit
from video_rag_eval.labels import _answers_equivalent, _resolve_claims, freeze
from video_rag_eval.scoring import load_ranges, score_runs


MAP = [0, 90, 261, 351, 411, 531, 711, 858, 921]


def test_claim_references_become_real_visual_claims():
    import pytest

    qa = {"required_claims": ["f1", "fact_2"]}
    facts = {"visual_facts": [{"fact_id": "f1", "claim": "A red car is visible."},
                              {"fact_id": "f2", "claim": "Two people stand beside it."}]}
    assert _resolve_claims(qa, facts)["required_claims"] == ["A red car is visible.", "Two people stand beside it."]
    assert qa["required_claim_refs"] == ["f1", "fact_2"]
    with pytest.raises(ValueError, match="unresolved"):
        _resolve_claims({"required_claims": ["f9"]}, facts)


def test_answer_comparison_receives_question():
    class FakeClient:
        def ask(self, stage, prompt, images, schema):
            assert "What is her hand doing?" in prompt
            assert not images
            return {"equivalent": True, "reason": "same essential action"}

    result = _answers_equivalent(FakeClient(), "She bends and holds the mouse.",
                                 ["Her hand is on the mouse."], "test", "What is her hand doing?")
    assert result["equivalent"]


def test_cost_ledger_serializes_threads_and_refreshes_other_instances(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    cfg = config()
    cfg["output"] = tmp_path
    first, second = CostLedger(cfg), CostLedger(cfg)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: first.record("test", {"prompt_tokens": 1000}, "qwen"), range(12)))
    second.reserve_estimate("library", 0.05, "qwen")
    rows = read_jsonl(output(cfg, "derived", "api_usage.jsonl"))
    assert len(rows) == 13
    assert abs(second.total - (12 * 0.00015 + 0.05)) < 1e-9


def test_insufficient_pixels_can_verify_a_deliberately_unanswerable_query(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from video_rag_eval import labels

    cfg = config()
    cfg["output"] = tmp_path
    qa = {"description_vi": "An office scene.", "question_vi": "What does the unreadable small print say?",
          "reference_answer": "", "required_claims": [], "question_type": "multiple_choice", "answerable": False}
    vote = {"answerable": False, "description_grounded": True, "question_unambiguous": True,
            "leaks_answer": False, "needs_more_pixels": True, "independent_answer": "Unreadable", "reasons": []}
    positive = {**qa, "answerable": True, "reference_answer": "ABC", "required_claims": ["The text says ABC"],
                "question_type": "object_attribute"}
    write_jsonl(output(cfg, "derived", "judge_votes.jsonl"), [
        {"frame_id": "L21_V001/001", "proposal": qa, "votes": [vote, vote], "accepted": False},
        {"frame_id": "L21_V002/001", "proposal": positive,
         "votes": [{**vote, "answerable": True}, {**vote, "answerable": True}], "accepted": False}])
    monkeypatch.setattr(labels, "VisionClient", lambda _: SimpleNamespace(ledger=SimpleNamespace(total=0)))
    result = labels.verify_qa(cfg)
    rows = read_jsonl(output(cfg, "derived", "judge_votes.jsonl"))
    assert result["accepted"] == 1
    assert rows[0]["proposal"]["question_type"] == "unanswerable"
    assert rows[0]["accepted"] is True
    assert rows[1]["accepted"] is False


def test_user_example_and_video_scope():
    assert interval(351, MAP) == (262, 410)
    ranges = make_ranges([{"query_id": "Q1", "frame_id": "L21_V001/004", "grade": 3}],
                         {"L21_V001": MAP})
    assert [match("Q1", "L21_V001", n, ranges) for n in (261, 262, 351, 410, 411)] == [None, 3, 3, 3, None]
    assert match("Q1", "L21_V002", 351, ranges) is None


def test_first_last_and_duplicate_csv_frame_index():
    assert interval(0, [0, 0, 90]) == (0, 89)
    assert interval(921, MAP) == (859, 921)
    assert interval(921, MAP, video_last_frame=1000) == (859, 1000)


def test_missing_and_invalid_map_are_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="missing map"):
        make_ranges([{"query_id": "Q1", "frame_id": "L21_V001/004", "grade": 3}], {})
    (tmp_path / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0,30,0\n2,3,30,90\n3,8,30,80\n", encoding="utf-8")
    maps, errors = load_maps(tmp_path)
    assert not maps and errors[0]["reason"] == "invalid_map"


def test_audit_detects_split_leakage(tmp_path):
    cfg = config()
    cfg["output"] = tmp_path / "out"
    cfg["docs_root"] = tmp_path / "docs"
    write_json(output(cfg, "benchmark", "v1", "splits.json"),
               {"dev_videos": ["L21_V001"], "test_videos": ["L21_V001"]})
    result = audit(cfg)
    assert result["total"] >= 32
    checks = __import__("json").loads(output(cfg, "audit", "requirements.json").read_text(encoding="utf-8"))["checks"]
    assert checks["split"]["status"] == "chưa đạt"


def test_overlap_gets_one_retrieval_credit():
    ranges = make_ranges([
        {"query_id": "Q1", "frame_id": "L21_V001/003", "grade": 3},
        {"query_id": "Q1", "frame_id": "L21_V001/004", "grade": 3},
    ], {"L21_V001": MAP})
    hits = [{"video_id": "L21_V001", "frame_idx": n} for n in (300, 301, 350)]
    scores = retrieval_scores("Q1", hits, ranges, ks=(10, 50, 300))
    assert scores["recall_at_50"] == 1.0
    assert len(positive_units("Q1", ranges)) == 1
    assert scores["ndcg_at_10"] <= 1.0


def test_unjudged_is_not_negative_precision():
    ranges = make_ranges([{"query_id": "Q1", "frame_id": "L21_V001/004", "grade": 3}],
                         {"L21_V001": MAP})
    contexts = [{"video_id": "L21_V002", "frame_idx": 351},
                {"video_id": "L21_V001", "frame_idx": 351}]
    score = context_precision("Q1", contexts, ranges)
    assert score["qrels_context_precision_ap"] is None
    assert score["context_judged_coverage"] == 0.5


def test_freeze_and_offline_score_keep_original_video_range(tmp_path):
    cfg = config()
    cfg["output"] = tmp_path / "out"
    cfg["target_qa"] = 1
    maps = tmp_path / "maps"
    maps.mkdir()
    (maps / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0,30,0\n2,3,30,90\n3,8.7,30,261\n4,11.7333,30,351\n5,13.7,30,411\n", encoding="utf-8")
    cfg["sources"] = {**cfg["sources"], "maps": str(maps), "keyframes": str(tmp_path / "images")}
    write_jsonl(output(cfg, "derived", "judge_votes.jsonl"), [{
        "frame_id": "L21_V001/004", "accepted": True, "crop_paths": [],
        "claim_verification": {"passed": True},
        "proposal": {"description_vi": "Cảnh vật trên đường.", "question_vi": "Đây là gì?",
                     "reference_answer": "Xe", "question_type": "object_attribute",
                     "required_claims": ["Có xe"], "answerable": True}}])
    write_jsonl(output(cfg, "derived", "pool_judgments.jsonl"), [{
        "frame_id": "L21_V001/004", "source_frame_id": "L21_V001/004", "status": "judged", "grade": 3}])
    freeze(cfg)
    ranges = load_ranges(output(cfg, "benchmark", "v1", "accepted_ranges.tsv"))
    assert [(r["lo_frame_idx"], r["hi_frame_idx"]) for r in ranges] == [(262, 410)]
    run = {"query_id": "Q0001", "config_id": "E1",
           "retrieved_hits": [{"frame_id": "L21_V001/004", "video_id": "L21_V001", "frame_idx": 351}],
           "context_hits": [{"frame_id": "L21_V001/004", "video_id": "L21_V001", "frame_idx": 351}],
           "generated_answer": "Xe", "citations": ["L21_V001/004"], "status": "answered", "latency_ms": 5}
    path = output(cfg, "runs", "E1.jsonl")
    write_jsonl(path, [run])
    summary = score_runs(cfg, path, with_mllm=False)
    assert summary["answer_correctness"] == 1
    assert summary["recall_at_50"] == 1
    wrong = {**run, "retrieved_hits": [{"frame_id": "L21_V001/003", "video_id": "L21_V001", "frame_idx": 261}],
             "context_hits": [{"frame_id": "L21_V001/003", "video_id": "L21_V001", "frame_idx": 261}],
             "citations": ["L21_V001/003"]}
    unsupported = {**run, "citations": []}
    write_jsonl(path, [run, wrong, unsupported])
    score_runs(cfg, path, with_mllm=False)
    rows = read_jsonl(output(cfg, "eval", "E1_scores.jsonl"))
    assert [r["answer_correctness"] for r in rows] == [1, 0, 0]
    assert rows[1]["recall_at_50"] == 0
    assert rows[1]["judged_coverage_at_50"] == 0
    assert rows[2]["answer_text_correctness"] == 1
    assert freeze(cfg)["reused"]
    qrels = output(cfg, "benchmark", "v1", "qrels.tsv")
    with qrels.open("a", encoding="utf-8") as f:
        f.write("Q0001\t0\tL21_V001/003\t0\n")
    import pytest
    with pytest.raises(RuntimeError, match="hash mismatch"):
        score_runs(cfg, path, with_mllm=False)
