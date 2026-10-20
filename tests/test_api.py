import pytest
from fastapi import HTTPException, Request

from api import main


def test_rate_limit_blocks_after_configured_request_count(monkeypatch):
    monkeypatch.setattr(main.config, "RATE_LIMIT_REQUESTS", 2)
    monkeypatch.setattr(main.config, "RATE_LIMIT_WINDOW_SECONDS", 60)
    main._HITS.clear()
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/search",
            "raw_path": b"/search",
            "query_string": b"",
            "headers": [],
            "client": ("192.0.2.10", 1234),
            "server": ("testserver", 80),
        }
    )

    main._rate_limit(request)
    main._rate_limit(request)
    with pytest.raises(HTTPException) as error:
        main._rate_limit(request)

    assert error.value.status_code == 429
    main._HITS.clear()


def test_readiness_fails_when_index_is_empty(monkeypatch):
    monkeypatch.setattr(main, "index_stats", lambda: {"exists": True, "chunks": 0})

    with pytest.raises(HTTPException) as error:
        main.ready()

    assert error.value.status_code == 503


def test_readiness_reports_provisioned_index(monkeypatch):
    monkeypatch.setattr(
        main,
        "index_stats",
        lambda: {"exists": True, "chunks": 42, "collection": "course_384"},
    )

    assert main.ready() == {"status": "ready", "chunks": 42, "collection": "course_384"}
