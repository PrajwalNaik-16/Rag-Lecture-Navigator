"""FastAPI wrapper around answer(), plus the single-page UI.

Phase 4 asks for a rate limit and a question-length cap before this goes
public — both are here, because "add it later" never happens.
"""

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from groq import APIStatusError, RateLimitError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ytrag import config
from ytrag.answer import answer as answer_question
from ytrag.answer import search_only
from ytrag.index import stats as index_stats

STATIC_DIR = Path(__file__).resolve().parent / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model before accepting traffic.

    Without this the model loads lazily on the first question, so the first
    student to use it waits ~14 seconds staring at a spinner while every
    subsequent search takes one. Better to spend that time at startup, in the
    terminal, where a wait is expected and explained.
    """
    from ytrag.embed import get_embedder

    print("Loading embedding model (first run downloads it)...", flush=True)
    embedder = get_embedder()
    print(f"Ready: {embedder.name} ({embedder.dim}-dim)", flush=True)
    yield


app = FastAPI(title="YT Lecture RAG", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=config.MAX_QUESTION_CHARS)
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)


# A dict of deques is enough for one process on a free tier. Behind more than
# one worker this becomes per-worker, so move it to Redis before it matters.
_HITS: dict[str, deque] = defaultdict(deque)
_RATE_LIMIT_CALLS = 0


def _rate_limit(request: Request) -> None:
    global _RATE_LIMIT_CALLS
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _HITS[client]

    while window and now - window[0] > config.RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()

    if len(window) >= config.RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Slow down — max {config.RATE_LIMIT_REQUESTS} questions per "
            f"{config.RATE_LIMIT_WINDOW_SECONDS}s.",
        )
    window.append(now)

    # Forget inactive clients periodically so a public endpoint cannot grow
    # this process-local limiter forever from one-off IP addresses.
    _RATE_LIMIT_CALLS += 1
    if _RATE_LIMIT_CALLS % 256 == 0:
        cutoff = now - config.RATE_LIMIT_WINDOW_SECONDS
        for address in [address for address, hits in _HITS.items() if not hits or hits[-1] < cutoff]:
            _HITS.pop(address, None)


@app.get("/health")
def health():
    """Process liveness; intentionally independent of the external vector store."""
    return {"status": "ok", "embed_model": config.EMBED_MODEL}


@app.get("/ready")
def ready():
    """Readiness check for deployments after the vector collection is provisioned."""
    try:
        info = index_stats()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Vector index is unavailable.") from exc
    if not info.get("exists") or not info.get("chunks"):
        raise HTTPException(status_code=503, detail="Vector index has no searchable chunks.")
    return {"status": "ready", "chunks": info["chunks"], "collection": info["collection"]}


@app.get("/stats")
def stats():
    info = index_stats()
    info.pop("videos", None)
    return info


@app.get("/suggestions")
def suggestions():
    """Return a small set of topics drawn from the currently indexed videos."""
    videos = index_stats().get("videos", {}).values()
    titles = sorted(
        {video.get("title", "").strip() for video in videos if video.get("title", "").strip()},
        key=str.casefold,
    )
    return {"suggestions": titles[:6]}


@app.post("/search")
def search(payload: AskRequest, request: Request):
    """Search without LLM generation; still rate-limited because embedding costs CPU."""
    _rate_limit(request)
    try:
        return search_only(payload.question, top_k=payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ask")
def ask(payload: AskRequest, request: Request):
    _rate_limit(request)
    try:
        return answer_question(payload.question, top_k=payload.top_k)
    except RateLimitError as exc:
        # The LLM provider's own quota, not ours. Surfacing this as a 500 tells
        # the student nothing; they need to know it is temporary and whose
        # limit it is.
        raise HTTPException(
            status_code=429,
            detail="The language model's usage quota is exhausted. "
            "This is a provider limit, not a problem with your question — try again later.",
        )
    except APIStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Language model error: {exc.status_code}")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        # Provider SDK errors vary. Log details server-side, but don't expose
        # request context or secrets in the response.
        import logging

        logging.getLogger(__name__).exception("Answer generation failed")
        raise HTTPException(status_code=502, detail="Answer provider request failed.") from exc


@app.get("/meta")
def meta():
    """Homepage details derived from the index, including hosted deployments."""
    info = index_stats()
    videos = info.get("videos", {})
    titles = sorted(
        {
            video.get("title", "").strip()
            for video in videos.values()
            if video.get("title", "").strip()
        },
        key=str.casefold,
    )
    seconds = sum(video.get("duration_sec", 0) for video in videos.values())
    return {
        "lectures": len(videos),
        "hours": round(seconds / 3600, 1),
        "chunks": info.get("chunks", 0),
        "suggestions": titles[:6],
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
