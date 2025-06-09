import os
import subprocess
import tempfile
from datetime import datetime
from typing import Optional, Tuple

from .config_manager import AppConfig
from .logger import Logger, AppError

# Optional dependency for video downloading
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# Initialize shared configuration and logger
config = AppConfig()
logger = Logger()


class Video:
    """
    Represents a video source and its metadata.

    Attributes:
        video_id: Optional[int]         # Assigned upon DB insert; None until then
        source_type: str                # "file" or "url"
        source_path: str                # Absolute local file path
        original_url: Optional[str]     # Non-null if source_type == "url"
        duration: float                 # Duration in seconds
        resolution: Tuple[int, int]     # (width, height)
        import_timestamp: str           # ISO 8601 UTC string
    """

    def __init__(
        self,
        source_type: str,
        source_path: str,
        original_url: Optional[str],
        duration: float,
        resolution: Tuple[int, int],
        import_timestamp: str
    ) -> None:
        self.video_id: Optional[int] = None
        self.source_type = source_type
        self.source_path = source_path
        self.original_url = original_url
        self.duration = duration
        self.resolution = resolution
        self.import_timestamp = import_timestamp


def probe_video_metadata(file_path: str) -> Tuple[float, Tuple[int, int]]:
    """
    Probe video metadata (duration, width, height) using ffprobe.
    Raises AppError on failure.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) < 3:
            raise ValueError(f"Unexpected ffprobe output: {result.stdout}")
        width = int(lines[0])
        height = int(lines[1])
        duration = float(lines[2])
        return duration, (width, height)
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        raise AppError("Cannot read video metadata.",
                       internal_message=e.stderr)
    except Exception as e:
        logger.error(str(e))
        raise AppError("Cannot read video metadata.", internal_message=str(e))


def import_video_from_file(file_path: str) -> Video:
    """
    Import video from local file, validating format and extracting metadata.

    Raises:
        AppError if file missing or unsupported format, or metadata probe fails.
    """
    if not os.path.isfile(file_path):
        raise AppError("File not found.", internal_message=file_path)

    ext = os.path.splitext(file_path)[1].lstrip('.').lower()
    if ext not in config.get_supported_file_formats():
        raise AppError("Unsupported file format.", internal_message=ext)

    duration, resolution = probe_video_metadata(file_path)
    abs_path = os.path.abspath(file_path)
    import_timestamp = datetime.utcnow().isoformat()
    return Video(
        source_type="file",
        source_path=abs_path,
        original_url=None,
        duration=duration,
        resolution=resolution,
        import_timestamp=import_timestamp
    )


def import_video_from_url(url: str, download_dir: Optional[str] = None) -> Video:
    """
    Download video from URL using yt_dlp and extract metadata.

    Raises:
        AppError if downloader not available, download fails, or metadata probe fails.
    """
    if yt_dlp is None:
        raise AppError(
            "Cannot download video from URL: yt_dlp not installed.",
            internal_message="yt_dlp_import_error"
        )

    # Prepare download directory
    if download_dir is None:
        download_dir = tempfile.mkdtemp(prefix="wm_download_")
    else:
        os.makedirs(download_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best",
        "outtmpl": os.path.join(download_dir, "%(title)s.%(ext)s")
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.error(str(e))
        raise AppError("Cannot download video from URL.",
                       internal_message=str(e))

    # Identify downloaded .mp4 file
    files = [f for f in os.listdir(download_dir) if f.lower().endswith('.mp4')]
    if len(files) != 1:
        msg = f"Expected one mp4 file, found {len(files)}"
        logger.error(msg)
        raise AppError(
            "Downloaded file not found or ambiguous.",
            internal_message=msg
        )

    downloaded_path = os.path.join(download_dir, files[0])
    duration, resolution = probe_video_metadata(downloaded_path)
    abs_path = os.path.abspath(downloaded_path)
    import_timestamp = datetime.utcnow().isoformat()
    return Video(
        source_type="url",
        source_path=abs_path,
        original_url=url,
        duration=duration,
        resolution=resolution,
        import_timestamp=import_timestamp
    )
