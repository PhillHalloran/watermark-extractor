import os
import tempfile
import pytest
import numpy as np
import cv2

from unittest.mock import MagicMock, patch
from src.video_processor import Clip, FrameBatch, detect_clips, merge_clips, split_clip, extract_frames_for_clip
from src.logger import AppError


@pytest.fixture
def dummy_video():
    return MagicMock(
        video_id=1,
        source_path="tests/assets/test_video.mp4",
        duration=10.0
    )


def test_detect_clips_invalid_threshold(dummy_video):
    with pytest.raises(ValueError):
        detect_clips(dummy_video, scene_threshold=0.0)
    with pytest.raises(ValueError):
        detect_clips(dummy_video, scene_threshold=1.0)


@patch("subprocess.run")
def test_detect_clips_valid(mock_run, dummy_video):
    mock_run.return_value = MagicMock(
        stderr="frame:0 pts_time:2.0\nframe:1 pts_time:4.5\n",
        stdout="",
        returncode=0
    )
    clips = detect_clips(dummy_video, scene_threshold=0.4)
    assert len(clips) == 3
    assert clips[0].start_time == 0.0 and clips[0].end_time == 2.0
    assert clips[1].start_time == 2.0 and clips[1].end_time == 4.5
    assert clips[2].end_time == dummy_video.duration


def test_merge_clips_success():
    clips = [Clip(1, 0.0, 5.0), Clip(1, 5.0, 10.0)]
    clips[0].clip_id = 101
    clips[1].clip_id = 102
    merged = merge_clips(clips, [101, 102])
    assert merged.start_time == 0.0
    assert merged.end_time == 10.0
    assert len(clips) == 1


def test_merge_clips_error_noncontiguous():
    clips = [Clip(1, 0.0, 5.0), Clip(1, 5.0, 10.0)]
    clips[0].clip_id = 101
    clips[1].clip_id = 102
    with pytest.raises(ValueError):
        merge_clips(clips, [101, 103])  # 103 doesn't exist


def test_split_clip_success():
    clips = [Clip(1, 0.0, 10.0)]
    clips[0].clip_id = 201
    new1, new2 = split_clip(clips, 201, 5.0)
    assert new1.start_time == 0.0
    assert new1.end_time == 5.0
    assert new2.start_time == 5.0
    assert new2.end_time == 10.0
    assert len(clips) == 2


def test_split_clip_out_of_bounds():
    clips = [Clip(1, 0.0, 10.0)]
    clips[0].clip_id = 301
    with pytest.raises(ValueError):
        split_clip(clips, 301, 10.0)


@patch("subprocess.run")
@patch("cv2.VideoCapture")
def test_extract_frames_for_clip(mock_cv, mock_subproc, dummy_video):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap_mock = MagicMock()
    cap_mock.isOpened.return_value = True
    cap_mock.read.side_effect = [(True, frame)] * 3 + [(False, None)]
    mock_cv.return_value = cap_mock
    clip = Clip(1, 0.0, 0.3)
    clip.clip_id = 1
    batch = extract_frames_for_clip(clip, dummy_video, sampling_rate_fps=10)
    assert isinstance(batch, FrameBatch)
    assert len(batch.frames) > 0
    assert len(batch.frames) == len(batch.timestamps)
    assert all(isinstance(f, np.ndarray) for f in batch.frames)
