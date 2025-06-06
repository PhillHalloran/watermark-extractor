# src/db_manager.py

import sqlite3
import csv
import os
from typing import List, Dict, Optional, Union


class DBManager:
    """
    Manages connection, schema creation, and CRUD operations for a SQLite database
    with tables: Videos, Clips, and Watermarks.
    """

    def __init__(self, db_path: str = "watermarks.db") -> None:
        """
        - Connect to SQLite at db_path.
        - Execute PRAGMA foreign_keys = ON.
        - Call self._create_tables() to ensure schema exists.

        Raises:
          - sqlite3.Error on connection failure.
        """
        # Ensure the directory for db_path exists (if a path is provided)
        parent_dir = os.path.dirname(os.path.abspath(db_path))
        if parent_dir and not os.path.isdir(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            # Enable foreign key support
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self._create_tables()
        except sqlite3.Error:
            raise

    def _create_tables(self) -> None:
        """
        Run the three CREATE TABLE IF NOT EXISTS statements in a single transaction.
        Raises:
          - sqlite3.Error on SQL syntax or I/O errors.
        """
        create_videos = """
        CREATE TABLE IF NOT EXISTS Videos (
            video_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type      TEXT    NOT NULL CHECK (source_type IN ("file", "url")),
            source_path      TEXT    NOT NULL,
            original_url     TEXT    DEFAULT NULL,
            duration         REAL    NOT NULL,
            resolution_w     INTEGER NOT NULL,
            resolution_h     INTEGER NOT NULL,
            import_timestamp DATETIME NOT NULL
        );
        """

        create_clips = """
        CREATE TABLE IF NOT EXISTS Clips (
            clip_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id   INTEGER NOT NULL REFERENCES Videos(video_id) ON DELETE CASCADE,
            start_time REAL    NOT NULL CHECK (start_time >= 0),
            end_time   REAL    NOT NULL CHECK (end_time > start_time),
            file_path  TEXT    DEFAULT NULL
        );
        """

        create_watermarks = """
        CREATE TABLE IF NOT EXISTS Watermarks (
            watermark_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id       INTEGER NOT NULL REFERENCES Videos(video_id) ON DELETE CASCADE,
            clip_id        INTEGER NOT NULL REFERENCES Clips(clip_id) ON DELETE CASCADE,
            timestamp      REAL    NOT NULL CHECK (timestamp >= 0),
            extracted_text TEXT    NOT NULL,
            confidence     REAL    NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
            roi_x          INTEGER NOT NULL CHECK (roi_x >= 0),
            roi_y          INTEGER NOT NULL CHECK (roi_y >= 0),
            roi_width      INTEGER NOT NULL CHECK (roi_width > 0),
            roi_height     INTEGER NOT NULL CHECK (roi_height > 0)
        );
        """

        try:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION;")
            cursor.execute(create_videos)
            cursor.execute(create_clips)
            cursor.execute(create_watermarks)
            cursor.execute("COMMIT;")
        except sqlite3.Error:
            # Roll back if anything fails
            self.conn.rollback()
            raise

    def insert_video(
        self,
        source_type: str,
        source_path: str,
        original_url: Optional[str],
        duration: float,
        resolution_w: int,
        resolution_h: int,
        import_timestamp: str,
    ) -> int:
        """
        Insert a row into Videos. Returns the new video_id (int).
        Preconditions:
          - source_type in {"file", "url"} or raise ValueError.
          - duration >= 0; resolution_w, resolution_h > 0
          - import_timestamp is ISO 8601 string.
        Raises:
          - ValueError on invalid arguments.
          - sqlite3.IntegrityError on constraint violation.
          - sqlite3.Error on other DB errors.
        """
        # Validate arguments
        if source_type not in {"file", "url"}:
            raise ValueError("source_type must be 'file' or 'url'")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("source_path must be a non-empty string")
        if original_url is not None and not isinstance(original_url, str):
            raise ValueError("original_url must be a string or None")
        if not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError("duration must be a non-negative number")
        if not isinstance(resolution_w, int) or resolution_w <= 0:
            raise ValueError("resolution_w must be a positive integer")
        if not isinstance(resolution_h, int) or resolution_h <= 0:
            raise ValueError("resolution_h must be a positive integer")
        if not isinstance(import_timestamp, str) or not import_timestamp.strip():
            raise ValueError(
                "import_timestamp must be a non-empty ISO 8601 string")

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO Videos (
                    source_type, source_path, original_url,
                    duration, resolution_w, resolution_h,
                    import_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_type,
                    source_path,
                    original_url,
                    float(duration),
                    resolution_w,
                    resolution_h,
                    import_timestamp,
                ),
            )
            self.conn.commit()
            return cursor.lastrowid  # type: ignore
        except sqlite3.IntegrityError:
            # Constraint violation (e.g., CHECK fails)
            raise
        except sqlite3.Error:
            raise

    def insert_clip(
        self,
        video_id: int,
        start_time: float,
        end_time: float,
        file_path: Optional[str] = None,
    ) -> int:
        """
        Insert a row into Clips. Returns new clip_id.
        Preconditions:
          - video_id must already exist in Videos; otherwise sqlite3.IntegrityError.
          - start_time >= 0, end_time > start_time.
        Raises:
          - ValueError on invalid argument values.
          - sqlite3.IntegrityError if FK fails or CHECK fails.
          - sqlite3.Error on other errors.
        """
        # Validate arguments
        if not isinstance(video_id, int) or video_id < 1:
            raise ValueError("video_id must be a positive integer")
        if not isinstance(start_time, (int, float)) or start_time < 0:
            raise ValueError("start_time must be a non-negative number")
        if not isinstance(end_time, (int, float)) or end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")
        if file_path is not None and (not isinstance(file_path, str) or not file_path.strip()):
            raise ValueError("file_path must be a non-empty string or None")

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO Clips (
                    video_id, start_time, end_time, file_path
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    video_id,
                    float(start_time),
                    float(end_time),
                    file_path,
                ),
            )
            self.conn.commit()
            return cursor.lastrowid  # type: ignore
        except sqlite3.IntegrityError:
            # FK violation or CHECK fails
            raise
        except sqlite3.Error:
            raise

    def insert_watermark(
        self,
        video_id: int,
        clip_id: int,
        timestamp: float,
        extracted_text: str,
        confidence: float,
        roi_x: int,
        roi_y: int,
        roi_width: int,
        roi_height: int,
    ) -> int:
        """
        Insert a row into Watermarks. Returns new watermark_id.
        Preconditions:
          - video_id and clip_id must exist (FK).
          - timestamp >= 0; 0.0 <= confidence <= 1.0; roi_x, roi_y >= 0; roi_width, roi_height > 0.
          - extracted_text is non-empty string.
        Raises:
          - ValueError on invalid argument values.
          - sqlite3.IntegrityError on constraint/FK violation.
          - sqlite3.Error on other DB errors.
        """
        # Validate arguments
        if not isinstance(video_id, int) or video_id < 1:
            raise ValueError("video_id must be a positive integer")
        if not isinstance(clip_id, int) or clip_id < 1:
            raise ValueError("clip_id must be a positive integer")
        if not isinstance(timestamp, (int, float)) or timestamp < 0:
            raise ValueError("timestamp must be a non-negative number")
        if not isinstance(extracted_text, str) or not extracted_text.strip():
            raise ValueError("extracted_text must be a non-empty string")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not isinstance(roi_x, int) or roi_x < 0:
            raise ValueError("roi_x must be a non-negative integer")
        if not isinstance(roi_y, int) or roi_y < 0:
            raise ValueError("roi_y must be a non-negative integer")
        if not isinstance(roi_width, int) or roi_width <= 0:
            raise ValueError("roi_width must be a positive integer")
        if not isinstance(roi_height, int) or roi_height <= 0:
            raise ValueError("roi_height must be a positive integer")

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO Watermarks (
                    video_id, clip_id, timestamp,
                    extracted_text, confidence,
                    roi_x, roi_y, roi_width, roi_height
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    clip_id,
                    float(timestamp),
                    extracted_text,
                    float(confidence),
                    roi_x,
                    roi_y,
                    roi_width,
                    roi_height,
                ),
            )
            self.conn.commit()
            return cursor.lastrowid  # type: ignore
        except sqlite3.IntegrityError:
            # Foreign key violation or CHECK fails
            raise
        except sqlite3.Error:
            raise

    def query_watermarks(
        self,
        text_filter: Optional[str] = None,
        min_confidence: Optional[float] = None,
        clip_id: Optional[int] = None,
    ) -> List[Dict[str, Union[int, float, str]]]:
        """
        Return a list of dict rows from Watermarks matching filters:
          - text_filter: substring match on extracted_text (case-insensitive). If None, no filter.
          - min_confidence: only include rows with confidence >= min_confidence.
          - clip_id: only include rows with that clip_id.
        Each dict has keys:
          "watermark_id", "video_id", "clip_id", "timestamp", "extracted_text",
          "confidence", "roi_x", "roi_y", "roi_width", "roi_height".
        Raises:
          - sqlite3.Error on query failure.
        """
        base_query = "SELECT watermark_id, video_id, clip_id, timestamp, extracted_text, confidence, roi_x, roi_y, roi_width, roi_height FROM Watermarks WHERE 1=1"
        params: List[Union[str, float, int]] = []

        if text_filter is not None:
            base_query += " AND LOWER(extracted_text) LIKE LOWER(?)"
            params.append(f"%{text_filter}%")

        if min_confidence is not None:
            if not isinstance(min_confidence, (int, float)):
                raise ValueError("min_confidence must be a number")
            base_query += " AND confidence >= ?"
            params.append(float(min_confidence))

        if clip_id is not None:
            if not isinstance(clip_id, int):
                raise ValueError("clip_id must be an integer")
            base_query += " AND clip_id = ?"
            params.append(clip_id)

        try:
            cursor = self.conn.cursor()
            cursor.execute(base_query, params)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

            result: List[Dict[str, Union[int, float, str]]] = []
            for row in rows:
                row_dict: Dict[str, Union[int, float, str]] = {}
                for idx, col in enumerate(columns):
                    row_dict[col] = row[idx]
                result.append(row_dict)

            return result
        except sqlite3.Error:
            raise

    def export_to_csv(
        self,
        rows: List[Dict[str, Union[int, float, str]]],
        output_path: str,
    ) -> None:
        """
        Write `rows` to `output_path` in CSV format. Columns in order:
        ["watermark_id","video_id","clip_id","timestamp","extracted_text",
         "confidence","roi_x","roi_y","roi_width","roi_height"].
        Raises:
          - IOError on failure to write.
        """
        fieldnames = [
            "watermark_id",
            "video_id",
            "clip_id",
            "timestamp",
            "extracted_text",
            "confidence",
            "roi_x",
            "roi_y",
            "roi_width",
            "roi_height",
        ]

        # Ensure parent directory exists
        parent_dir = os.path.dirname(os.path.abspath(output_path))
        if parent_dir and not os.path.isdir(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        try:
            with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    # Only include the expected keys; ignore any extras
                    filtered_row = {key: row[key] for key in fieldnames}
                    writer.writerow(filtered_row)
        except IOError:
            raise
