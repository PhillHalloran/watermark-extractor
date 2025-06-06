# tests/test_db_manager.py

import os
import sqlite3
import csv
import pytest
# Adjust import path if module is located elsewhere
from src.db_manager import DBManager


@pytest.fixture
def temp_db_path(tmp_path):
    """
    Fixture to create a temporary SQLite database file path.
    """
    db_file = tmp_path / "test_watermarks.db"
    return str(db_file)


@pytest.fixture
def db_manager(temp_db_path):
    """
    Fixture to instantiate DBManager with a temporary database.
    """
    manager = DBManager(db_path=temp_db_path)
    yield manager
    # Clean up: close connection and remove the file
    manager.conn.close()
    if os.path.isfile(temp_db_path):
        os.remove(temp_db_path)


def test_tables_created(db_manager, temp_db_path):
    """
    Verifies that the expected tables (Videos, Clips, Watermarks) exist after initialization.
    """
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_schema WHERE type='table' AND name IN ('Videos', 'Clips', 'Watermarks')"
    )
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    assert "Videos" in tables
    assert "Clips" in tables
    assert "Watermarks" in tables


@pytest.mark.parametrize(
    "source_type, source_path, original_url, duration, w, h, import_ts, expect_error",
    [
        ("file", "/videos/vid.mp4", None, 120.5,
         1920, 1080, "2025-06-06T12:00:00", False),
        ("url", "http://example.com/vid.mp4", "http://example.com/vid.mp4",
         0.0, 1, 1, "2025-06-06T23:59:59", False),
        ("invalid", "/x", None, 10, 640, 480,
         "2025-06-06T00:00:00", True),  # bad source_type
        ("file", "", None, 10, 640, 480, "2025-06-06T00:00:00",
         True),       # empty source_path
        ("file", "/x", None, -1, 640, 480,
         "2025-06-06T00:00:00", True),    # negative duration
        ("file", "/x", None, 10, 0, 480,
         "2025-06-06T00:00:00", True),     # zero width
        ("file", "/x", None, 10, 640, -5,
         "2025-06-06T00:00:00", True),     # negative height
        # empty import_timestamp
        ("file", "/x", None, 10, 640, 480, "", True),
    ],
)
def test_insert_video_validation(db_manager, source_type, source_path, original_url, duration, w, h, import_ts, expect_error):
    """
    Tests that insert_video accepts valid parameters and rejects invalid ones.
    """
    if expect_error:
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            db_manager.insert_video(
                source_type, source_path, original_url, duration, w, h, import_ts
            )
    else:
        vid_id = db_manager.insert_video(
            source_type, source_path, original_url, duration, w, h, import_ts
        )
        assert isinstance(vid_id, int) and vid_id > 0

        # Confirm row exists in Videos
        cursor = db_manager.conn.cursor()
        cursor.execute("SELECT * FROM Videos WHERE video_id = ?", (vid_id,))
        row = cursor.fetchone()
        assert row is not None
        # Check that source_type and duration match
        assert row[1] == source_type
        assert pytest.approx(row[4], rel=1e-6) == duration


@pytest.mark.parametrize(
    "video_kwargs, clip_kwargs, expect_error",
    [
        # First insert a valid video, then valid clip
        (
            dict(source_type="file", source_path="/v.mp4", original_url=None, duration=10,
                 resolution_w=640, resolution_h=480, import_timestamp="2025-06-06T10:00:00"),
            dict(start_time=1.0, end_time=5.0, file_path="/clips/c1.mp4"),
            False,
        ),
        # Clip with negative start_time
        (
            dict(source_type="file", source_path="/v.mp4", original_url=None, duration=10,
                 resolution_w=640, resolution_h=480, import_timestamp="2025-06-06T10:00:00"),
            dict(start_time=-1.0, end_time=2.0, file_path=None),
            True,
        ),
        # Clip with end_time <= start_time
        (
            dict(source_type="file", source_path="/v.mp4", original_url=None, duration=10,
                 resolution_w=640, resolution_h=480, import_timestamp="2025-06-06T10:00:00"),
            dict(start_time=2.0, end_time=2.0, file_path=None),
            True,
        ),
    ],
)
def test_insert_clip(db_manager, video_kwargs, clip_kwargs, expect_error):
    """
    Tests insert_clip with valid and invalid parameters, including foreign-key checks.
    """
    # Insert a base video first
    vid_id = db_manager.insert_video(**video_kwargs)

    if expect_error:
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            db_manager.insert_clip(video_id=vid_id, **clip_kwargs)
    else:
        clip_id = db_manager.insert_clip(video_id=vid_id, **clip_kwargs)
        assert isinstance(clip_id, int) and clip_id > 0

        # Confirm row exists in Clips
        cursor = db_manager.conn.cursor()
        cursor.execute("SELECT * FROM Clips WHERE clip_id = ?", (clip_id,))
        row = cursor.fetchone()
        assert row is not None
        # Check that video_id, start_time, end_time match
        assert row[1] == vid_id
        assert pytest.approx(row[2], rel=1e-6) == clip_kwargs["start_time"]
        assert pytest.approx(row[3], rel=1e-6) == clip_kwargs["end_time"]


def test_insert_clip_fk_violation(db_manager):
    """
    Inserting a clip for a non-existent video_id should raise an IntegrityError.
    """
    with pytest.raises(sqlite3.IntegrityError):
        db_manager.insert_clip(video_id=9999, start_time=0.0, end_time=1.0)


@pytest.mark.parametrize(
    "watermark_kwargs, expect_error",
    [
        # Valid watermark insertion (after inserting video and clip)
        (
            dict(timestamp=0.5, extracted_text="Hello", confidence=0.9,
                 roi_x=10, roi_y=5, roi_width=100, roi_height=50),
            False,
        ),
        # Negative timestamp
        (
            dict(timestamp=-0.1, extracted_text="Neg", confidence=0.5,
                 roi_x=0, roi_y=0, roi_width=10, roi_height=10),
            True,
        ),
        # Confidence out of range
        (
            dict(timestamp=1.0, extracted_text="BadConf", confidence=1.5,
                 roi_x=0, roi_y=0, roi_width=10, roi_height=10),
            True,
        ),
        # Empty extracted_text
        (
            dict(timestamp=1.0, extracted_text="   ", confidence=0.5,
                 roi_x=0, roi_y=0, roi_width=10, roi_height=10),
            True,
        ),
        # Negative roi_x
        (
            dict(timestamp=1.0, extracted_text="NegROI", confidence=0.5,
                 roi_x=-5, roi_y=0, roi_width=10, roi_height=10),
            True,
        ),
        # Zero roi_width
        (
            dict(timestamp=1.0, extracted_text="ZeroWidth", confidence=0.5,
                 roi_x=0, roi_y=0, roi_width=0, roi_height=10),
            True,
        ),
    ],
)
def test_insert_watermark(db_manager, watermark_kwargs, expect_error):
    """
    Tests insert_watermark for valid insertion and various invalid parameter cases.
    """
    # First insert a video and clip to reference
    vid_id = db_manager.insert_video(
        source_type="file",
        source_path="/v.mp4",
        original_url=None,
        duration=10.0,
        resolution_w=640,
        resolution_h=480,
        import_timestamp="2025-06-06T10:00:00",
    )
    clip_id = db_manager.insert_clip(
        video_id=vid_id, start_time=0.0, end_time=2.0, file_path=None)

    if expect_error:
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            db_manager.insert_watermark(
                video_id=vid_id, clip_id=clip_id, **watermark_kwargs)
    else:
        wm_id = db_manager.insert_watermark(
            video_id=vid_id, clip_id=clip_id, **watermark_kwargs)
        assert isinstance(wm_id, int) and wm_id > 0

        # Confirm row exists in Watermarks
        cursor = db_manager.conn.cursor()
        cursor.execute(
            "SELECT * FROM Watermarks WHERE watermark_id = ?", (wm_id,))
        row = cursor.fetchone()
        assert row is not None
        # Check that extracted_text and confidence match
        assert row[4] == watermark_kwargs["extracted_text"]
        assert pytest.approx(
            row[5], rel=1e-6) == watermark_kwargs["confidence"]


def test_insert_watermark_fk_violation(db_manager):
    """
    Inserting a watermark with non-existent video_id or clip_id should raise IntegrityError.
    """
    # video_id exists, clip_id does not
    vid_id = db_manager.insert_video(
        source_type="file",
        source_path="/v.mp4",
        original_url=None,
        duration=5.0,
        resolution_w=320,
        resolution_h=240,
        import_timestamp="2025-06-06T11:00:00",
    )
    with pytest.raises(sqlite3.IntegrityError):
        db_manager.insert_watermark(
            video_id=vid_id,
            clip_id=9999,
            timestamp=0.0,
            extracted_text="FK",
            confidence=0.5,
            roi_x=0,
            roi_y=0,
            roi_width=10,
            roi_height=10,
        )

    # clip_id exists but video_id does not
    clip_id = db_manager.insert_clip(
        video_id=vid_id, start_time=0.0, end_time=1.0, file_path=None)
    with pytest.raises(sqlite3.IntegrityError):
        db_manager.insert_watermark(
            video_id=9999,
            clip_id=clip_id,
            timestamp=0.0,
            extracted_text="FK2",
            confidence=0.5,
            roi_x=0,
            roi_y=0,
            roi_width=10,
            roi_height=10,
        )


def test_query_watermarks_filters(db_manager):
    """
    Tests query_watermarks with various filtering combinations.
    """
    # Insert video, clip, and multiple watermarks
    vid_id = db_manager.insert_video(
        source_type="file",
        source_path="/movie.mp4",
        original_url=None,
        duration=20.0,
        resolution_w=1280,
        resolution_h=720,
        import_timestamp="2025-06-06T12:00:00",
    )
    clip_id1 = db_manager.insert_clip(
        video_id=vid_id, start_time=0.0, end_time=5.0, file_path=None)
    clip_id2 = db_manager.insert_clip(
        video_id=vid_id, start_time=5.0, end_time=10.0, file_path=None)

    # Watermarks with varying text and confidence
    wm1 = dict(timestamp=0.5, extracted_text="Hello World",
               confidence=0.95, roi_x=0, roi_y=0, roi_width=50, roi_height=20)
    wm2 = dict(timestamp=1.0, extracted_text="Test Case",
               confidence=0.60, roi_x=1, roi_y=1, roi_width=50, roi_height=20)
    wm3 = dict(timestamp=6.0, extracted_text="Another hello",
               confidence=0.80, roi_x=2, roi_y=2, roi_width=40, roi_height=10)

    id1 = db_manager.insert_watermark(video_id=vid_id, clip_id=clip_id1, **wm1)
    id2 = db_manager.insert_watermark(video_id=vid_id, clip_id=clip_id1, **wm2)
    id3 = db_manager.insert_watermark(video_id=vid_id, clip_id=clip_id2, **wm3)

    # No filters: should return all three
    all_rows = db_manager.query_watermarks()
    assert len(all_rows) == 3
    returned_ids = {r["watermark_id"] for r in all_rows}
    assert {id1, id2, id3} == returned_ids

    # Filter by text_filter (case-insensitive substring "hello"): should return wm1 and wm3
    hello_rows = db_manager.query_watermarks(text_filter="hello")
    assert len(hello_rows) == 2
    assert {r["watermark_id"] for r in hello_rows} == {id1, id3}

    # Filter by min_confidence (>= 0.8): should return wm1 and wm3
    conf_rows = db_manager.query_watermarks(min_confidence=0.8)
    assert len(conf_rows) == 2
    assert {r["watermark_id"] for r in conf_rows} == {id1, id3}

    # Filter by clip_id=clip_id1: should return wm1 and wm2
    clip1_rows = db_manager.query_watermarks(clip_id=clip_id1)
    assert len(clip1_rows) == 2
    assert {r["watermark_id"] for r in clip1_rows} == {id1, id2}

    # Combined filter: text_filter="test", min_confidence=0.5, clip_id=clip_id1 => only wm2
    combined = db_manager.query_watermarks(
        text_filter="test", min_confidence=0.5, clip_id=clip_id1)
    assert len(combined) == 1
    assert combined[0]["watermark_id"] == id2


def test_query_watermarks_invalid_filters(db_manager):
    """
    Passing invalid types for filters should raise ValueError.
    """
    with pytest.raises(ValueError):
        db_manager.query_watermarks(min_confidence="high")

    with pytest.raises(ValueError):
        db_manager.query_watermarks(clip_id="not_an_int")


def test_export_to_csv_and_contents(db_manager, tmp_path):
    """
    Verifies export_to_csv writes a CSV file with the expected header and rows.
    """
    # Insert video, clip, watermark
    vid_id = db_manager.insert_video(
        source_type="file",
        source_path="/example.mp4",
        original_url=None,
        duration=15.0,
        resolution_w=800,
        resolution_h=600,
        import_timestamp="2025-06-06T13:00:00",
    )
    clip_id = db_manager.insert_clip(
        video_id=vid_id, start_time=2.0, end_time=4.0, file_path=None)
    wm_kwargs = dict(timestamp=2.5, extracted_text="ExportTest",
                     confidence=0.75, roi_x=5, roi_y=5, roi_width=20, roi_height=10)
    wm_id = db_manager.insert_watermark(
        video_id=vid_id, clip_id=clip_id, **wm_kwargs)

    # Query watermarks
    rows = db_manager.query_watermarks()

    # Export to CSV
    csv_path = tmp_path / "watermarks_export.csv"
    db_manager.export_to_csv(rows, str(csv_path))

    # Verify the CSV file exists
    assert os.path.isfile(str(csv_path))

    # Read the CSV and verify header and single row
    with open(str(csv_path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        expected_fields = [
            "watermark_id", "video_id", "clip_id", "timestamp", "extracted_text",
            "confidence", "roi_x", "roi_y", "roi_width", "roi_height"
        ]
        assert header == expected_fields

        all_rows = list(reader)
        assert len(all_rows) == 1
        row = all_rows[0]

        # Verify column values match inserted watermark
        assert int(row["watermark_id"]) == wm_id
        assert int(row["video_id"]) == vid_id
        assert int(row["clip_id"]) == clip_id
        assert abs(float(row["timestamp"]) -
                   wm_kwargs["timestamp"]) < 1e-6  # type: ignore
        assert row["extracted_text"] == wm_kwargs["extracted_text"]
        assert abs(float(row["confidence"]) -
                   wm_kwargs["confidence"]) < 1e-6  # type: ignore
        assert int(row["roi_x"]) == wm_kwargs["roi_x"]
        assert int(row["roi_y"]) == wm_kwargs["roi_y"]
        assert int(row["roi_width"]) == wm_kwargs["roi_width"]
        assert int(row["roi_height"]) == wm_kwargs["roi_height"]


def test_export_to_csv_io_error(db_manager, tmp_path):
    """
    Attempting to write to a non-writable location should raise IOError.
    """
    # Generate some dummy rows
    rows = [{"watermark_id": 1, "video_id": 1, "clip_id": 1, "timestamp": 0.0,
             "extracted_text": "X", "confidence": 0.5, "roi_x": 0, "roi_y": 0, "roi_width": 1, "roi_height": 1}]

    # Use a directory path instead of a file path to force an IOError
    with pytest.raises(IOError):
        # tmp_path is a directory, not a file
        db_manager.export_to_csv(rows, str(tmp_path))
