import os
import subprocess
import pytest
import tempfile

import src.video_handler as vh
from src.video_handler import Video, probe_video_metadata, import_video_from_file, import_video_from_url, AppError

# ---- Tests for probe_video_metadata ----


def test_probe_video_metadata_success(monkeypatch):
    # Simulate ffprobe returning width, height, duration
    fake_stdout = "1920\n1080\n15.25\n"

    def fake_run(cmd, stdout, stderr, text, check):
        return subprocess.CompletedProcess(cmd, 0, stdout=fake_stdout, stderr="")
    monkeypatch.setattr(subprocess, 'run', fake_run)

    duration, resolution = probe_video_metadata('dummy.mp4')
    assert duration == pytest.approx(15.25)
    assert resolution == (1920, 1080)


def test_probe_video_metadata_failure(monkeypatch):
    # Simulate ffprobe error
    def fake_run(cmd, stdout, stderr, text, check):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, stderr="probe error")
    monkeypatch.setattr(subprocess, 'run', fake_run)

    with pytest.raises(AppError) as exc:
        probe_video_metadata('dummy.mp4')
    assert "Cannot read video metadata" in str(exc.value)


# ---- Tests for import_video_from_file ----

def test_import_video_from_file_success(monkeypatch, tmp_path):
    # Create a dummy file with supported extension
    file_path = tmp_path / "video.MP4"
    file_path.write_text("data")
    # Ensure config supports mp4
    monkeypatch.setattr(
        vh.config, 'get_supported_file_formats', lambda: ['mp4'])
    # Stub out metadata probing
    monkeypatch.setattr(vh, 'probe_video_metadata',
                        lambda fp: (8.0, (640, 480)))

    vid = import_video_from_file(str(file_path))
    assert isinstance(vid, Video)
    assert vid.source_type == "file"
    assert vid.duration == 8.0
    assert vid.resolution == (640, 480)
    assert vid.original_url is None
    assert os.path.isabs(vid.source_path)


def test_import_video_from_file_not_found():
    with pytest.raises(AppError) as exc:
        import_video_from_file('no_exist.mp4')
    assert "File not found" in str(exc.value)


def test_import_video_from_file_unsupported(monkeypatch, tmp_path):
    # Create a dummy .avi file
    file_path = tmp_path / "video.avi"
    file_path.write_text("data")
    monkeypatch.setattr(
        vh.config, 'get_supported_file_formats', lambda: ['mp4'])

    with pytest.raises(AppError) as exc:
        import_video_from_file(str(file_path))
    assert "Unsupported file format" in str(exc.value)


def test_import_video_from_file_probe_error(monkeypatch, tmp_path):
    # Create dummy file
    file_path = tmp_path / "video.mp4"
    file_path.write_text("data")
    monkeypatch.setattr(
        vh.config, 'get_supported_file_formats', lambda: ['mp4'])
    # Probe raises AppError
    monkeypatch.setattr(vh, 'probe_video_metadata', lambda fp: (
        _ for _ in ()).throw(AppError("meta fail", internal_message="err")))

    with pytest.raises(AppError) as exc:
        import_video_from_file(str(file_path))
    assert "meta fail" in str(exc.value)


# ---- Tests for import_video_from_url ----

class DummyYDL:
    def __init__(self, opts):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def download(self, urls):
        # simulate download success
        return None


def test_import_video_from_url_no_yt(monkeypatch):
    monkeypatch.setattr(vh, 'yt_dlp', None)
    with pytest.raises(AppError) as exc:
        import_video_from_url('http://example.com/video')
    assert "yt_dlp not installed" in str(exc.value)


def test_import_video_from_url_download_failure(monkeypatch, tmp_path):
    # YTDLP raises error
    class FailingYDL(DummyYDL):
        def download(self, urls):
            raise Exception("download error")
    monkeypatch.setattr(vh, 'yt_dlp', type('m', (), {'YoutubeDL': FailingYDL}))

    with pytest.raises(AppError) as exc:
        import_video_from_url('http://example.com/video',
                              download_dir=str(tmp_path))
    assert "Cannot download video from URL" in str(exc.value)


def test_import_video_from_url_ambiguous(monkeypatch, tmp_path):
    # Successful download but multiple mp4 files
    monkeypatch.setattr(vh, 'yt_dlp', type('m', (), {'YoutubeDL': DummyYDL}))
    d = tmp_path / "dl"
    d.mkdir()
    # create two mp4 files
    (d / "a.mp4").write_text('x')
    (d / "b.mp4").write_text('x')
    monkeypatch.setattr(os, 'listdir', lambda p: ['a.mp4', 'b.mp4'])
    monkeypatch.setattr(vh, 'probe_video_metadata',
                        lambda fp: (5.0, (320, 240)))

    with pytest.raises(AppError) as exc:
        import_video_from_url('http://example.com/video', download_dir=str(d))
    assert "Expected one mp4 file" in exc.value.internal_message


def test_import_video_from_url_success(monkeypatch, tmp_path):
    monkeypatch.setattr(vh, 'yt_dlp', type('m', (), {'YoutubeDL': DummyYDL}))
    d = tmp_path / "dl"
    d.mkdir()
    (d / "video.mp4").write_text('x')
    monkeypatch.setattr(os, 'listdir', lambda p: ['video.mp4'])
    monkeypatch.setattr(vh, 'probe_video_metadata',
                        lambda fp: (3.5, (128, 72)))

    vid = import_video_from_url(
        'http://example.com/video', download_dir=str(d))
    assert isinstance(vid, Video)
    assert vid.source_type == "url"
    assert vid.original_url == 'http://example.com/video'
    assert vid.duration == pytest.approx(3.5)
    assert vid.resolution == (128, 72)
