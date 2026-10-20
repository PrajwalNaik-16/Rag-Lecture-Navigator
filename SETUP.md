# Run this on your own machine

No GPU. No transcribing. No Qdrant account. No embedding 68 hours of lecture.

The working tree includes a sample corpus of 126 transcripts and a prebuilt vector export.
You need one optional LLM API key to enable generated answers; retrieval works without one.

---

## Setup

### 1. Install

```bash
git clone <YOUR_REPOSITORY_URL>
cd <repository-directory>
uv --cache-dir .uv-cache sync
```

No `uv`? `pip install uv` first.

The repository includes GitHub Actions CI for pushes and pull requests. To run the suite
locally after setup:

```bash
uv pip install "pytest>=8.3,<9"
uv run --locked pytest -q
```

### 2. One free API key

```bash
cp .env.example .env
```

Open `.env` and paste in **one** of these (both free, no card):

- **Gemini** — https://aistudio.google.com/apikey → `GEMINI_API_KEY=`
- **Groq** — https://console.groq.com/keys → `GROQ_API_KEY=`

Gemini's free tier is much more generous. You don't need a Qdrant account —
the index is stored in a local folder by default.

### Index your own YouTube videos or playlist

Use this flow when you want your own course instead of the bundled DSA sample. Set a
distinct collection name in `.env`, for example `YTRAG_COLLECTION=my_course`. You can
pass a playlist URL with `--playlist`, or a text file of individual video links with
`--urls-file`. Put one URL on each line; blank lines and lines beginning with `#` are
ignored. The links can be in any order, and duplicate video IDs are only processed once.

For a file called `video_urls.txt`, check one lesson first:

```bash
uv --cache-dir .uv-cache run python -m ytrag.cli langtest "<ONE_VIDEO_URL>"
uv --cache-dir .uv-cache run python -m ytrag.cli ingest --urls-file video_urls.txt --limit 1
```

Check the transcript and timestamps before starting the full playlist:

```bash
uv --cache-dir .uv-cache run python -m ytrag.cli ingest --urls-file video_urls.txt
```

To index a regular YouTube playlist instead, replace `--urls-file video_urls.txt` with
`--playlist "<PLAYLIST_URL>"`.

Do not run `ytrag load` for this workflow; it imports the bundled DSA vectors. After ingest,
start the app with `uv --cache-dir .uv-cache run python -m ytrag.cli serve`. Completed
transcripts are cached, so rerunning `ingest` resumes after an interruption. Skip the
bundled-index step below.

### 3. Load the prebuilt index (bundled sample only)

```bash
uv --cache-dir .uv-cache run python -m ytrag.cli load
```

**About 20 seconds.** It reads `index/vectors.npz` (4 MB, committed to the
repo) and loads it straight into your local vector store. Nothing is embedded,
because the embedding was already done once and shipped.

The one download is the query embedding model — `all-MiniLM-L6-v2`, 87 MB.

### 4. Use it

```bash
uv --cache-dir .uv-cache run python -m ytrag.cli serve
```

Open <http://127.0.0.1:8000>. Ask a question, click a timestamp, the lecture
plays from that exact second.

---

## Every time after that

Setup is once. From then on:

```bash
uv --cache-dir .uv-cache run python -m ytrag.cli serve
```

About **15 seconds** to start — that is the model loading, and it prints
`Ready:` when it is done. Every search after that is roughly one second.

Ctrl-C stops it and `localhost:8000` goes dead until you start it again.
Nothing is lost; the index stays on disk.

No server needed for these:

```bash
uv --cache-dir .uv-cache run python -m ytrag.cli search "sliding window"    # timestamps only, instant, no LLM
uv --cache-dir .uv-cache run python -m ytrag.cli ask "kadane ka intuition"  # with a written answer
uv --cache-dir .uv-cache run python -m ytrag.cli stats
```

> **One at a time.** The local index allows a single process, so stop
> `ytrag serve` before running `ytrag ask` elsewhere. Setting `QDRANT_URL` to a
> hosted Qdrant lifts that.

---

## Rebuilding the index yourself

`ytrag load` is a shortcut. To build it from the transcripts instead:

```bash
uv --cache-dir .uv-cache run python -m ytrag.cli reindex
```

Takes a couple of minutes on the default model. Worth doing if you change the
chunk size, or want to try a different embedding model.

### Which embedding model

Default is `all-MiniLM-L6-v2`. Measured on this corpus of 2,933 chunks:

| model | download | reindex (CPU) | top-1 | top-5 |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** (default) | **87 MB** | **1.6 min** | 11/12 | 12/12 |
| BAAI/bge-m3 | 4.35 GB | 55 min | 12/12 | 12/12 |

bge-m3 is the stronger multilingual model and wins exactly one question out of
twelve — which the smaller model still returns at rank 2. Fifty times the
download for that. The title-boost re-ranking in `index.py` recovers most of
the gap, which is the interesting result: better ranking beat a bigger encoder.

To use it anyway:

```bash
YTRAG_EMBED_MODEL=BAAI/bge-m3 uv --cache-dir .uv-cache run python -m ytrag.cli reindex
```

---

## If something breaks

```bash
uv --cache-dir .uv-cache run python -m ytrag.cli preflight
```

Checks every dependency — keys, vector store, embedding model, LLM — and says
which one is unhappy.

**`Warning: You are sending unauthenticated requests to the HF Hub`** — expected,
and safe to ignore. HuggingFace suggests setting a token for faster downloads;
nothing here needs one. It is printed by a compiled library at a level Python
cannot intercept, which is why it survives.

| symptom | fix |
|---|---|
| `GROQ_API_KEY is not set` | `.env` missing, or you are in the wrong folder |
| `These vectors were built with…` | your `YTRAG_EMBED_MODEL` differs from the shipped index; unset it, or `reindex` |
| `already open in another process` | stop `ytrag serve` first |
| `No prebuilt index found` | run `ytrag load` from the repository root, or use `ytrag reindex` after ingesting your own videos |
| LLM quota errors | free tiers are limited; `ytrag search` needs no LLM at all |

---

## Point it at your own playlist

This is the part that wants a GPU.

```bash
uv --cache-dir .uv-cache run python -m ytrag.cli langtest "<one video URL>"           # pick the Whisper language first
uv --cache-dir .uv-cache run python -m ytrag.cli ingest --playlist "<PLAYLIST_URL>"   # download, transcribe, index
```

Roughly 8x realtime on a laptop GPU, so an hour of video takes about seven
minutes. Start with `--limit 10`, not a whole playlist. See the README for the
failure modes worth knowing before a long run.
