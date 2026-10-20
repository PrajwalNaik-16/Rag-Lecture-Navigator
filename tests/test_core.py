from ytrag.answer import _renumber
from ytrag.chunk import chunk_segments
from ytrag.models import Chunk, Segment, Video


def test_chunking_keeps_segment_timestamps_and_stable_ids():
    video = Video(video_id="abc123", title="Graph Traversal")
    segments = [
        Segment(start=0, end=20, text="BFS explores a graph level by level."),
        Segment(start=20, end=40, text="It uses a queue and visits each neighbor."),
        Segment(start=40, end=60, text="DFS explores one path before backtracking."),
    ]

    chunks = chunk_segments(video, segments, window_seconds=40, overlap_seconds=20, min_words=1)

    assert chunks
    assert all(chunk.video_id == video.video_id for chunk in chunks)
    assert all(chunk.end_sec >= chunk.start_sec for chunk in chunks)
    assert len({chunk.point_id for chunk in chunks}) == len(chunks)
    assert all("Graph Traversal" in chunk.text for chunk in chunks)


def test_citation_renumbering_keeps_only_valid_used_sources():
    chunks = [
        Chunk("a:10", "a", "Lesson A", 10, 20, "first"),
        Chunk("b:30", "b", "Lesson B", 30, 40, "second"),
    ]

    answer, citations = _renumber("Use [2], ignore [9], then [1] and [2].", [(chunks[0], 0.2), (chunks[1], 0.3)])

    assert answer == "Use [1], ignore , then [2] and [1]."
    assert [citation["title"] for citation in citations] == ["Lesson B", "Lesson A"]
