from ytrag.evaluate import run_eval
from ytrag.models import Chunk


def test_evaluation_reports_rank_and_timestamp_error(monkeypatch, tmp_path):
    expected = Chunk("v1:100", "v1", "Expected lesson", 100, 150, "answer passage")
    distractor = Chunk("v2:20", "v2", "Other lesson", 20, 70, "other passage")

    monkeypatch.setattr(
        "ytrag.evaluate.retrieve_only",
        lambda question, top_k: [(distractor, 0.1), (expected, 0.2)],
    )
    monkeypatch.setattr("ytrag.evaluate.answer", lambda question, top_k: {"grounded": False})

    report = run_eval(
        path=_write_golden(tmp_path, [{
            "q": "where is the expected concept?",
            "expect_video_id": "v1",
            "expect_around_sec": 90,
            "tolerance_sec": 5,
        }]),
        k=5,
    )

    assert report["video_hit_rate"] == 1.0
    assert report["retrieval_hit_rate"] == 0.0
    assert report["mrr_at_k"] == 0.5
    assert report["mean_expected_video_rank"] == 2.0
    assert report["mean_timestamp_error_sec"] == 10.0
    assert report["misses"][0]["timestamp_error_sec"] == 10


def _write_golden(tmp_path, entries):
    import json

    path = tmp_path / "golden.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path
