import os
import re
import cv2
import numpy as np
import tempfile
import subprocess
from typing import List, Tuple, Optional

from .logger import AppError
from .video_handler import Video


class Clip:
    def __init__(
        self,
        video_id: int,
        start_time: float,
        end_time: float,
        file_path: Optional[str] = None
    ) -> None:
        self.clip_id: Optional[int] = None
        self.video_id = video_id
        self.start_time = start_time
        self.end_time = end_time
        self.file_path = file_path


class FrameBatch:
    def __init__(self, clip_id: int) -> None:
        self.clip_id = clip_id
        self.frames: List[np.ndarray] = []
        self.timestamps: List[float] = []

    def add_frame(self, frame: np.ndarray, timestamp: float) -> None:
        self.frames.append(frame)
        self.timestamps.append(timestamp)


def detect_clips(video: Video, scene_threshold: float = 0.4) -> List[Clip]:
    if not (0.0 < scene_threshold < 1.0):
        raise ValueError("scene_threshold must be between 0.0 and 1.0")

    cmd = [
        "ffmpeg",
        "-i", video.source_path,
        "-filter_complex", f"select='gt(scene,{scene_threshold})',showinfo",
        "-f", "null", "-"
    ]

    try:
        result = subprocess.run(
            cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise AppError("Scene detection failed.", internal_message=e.stderr)

    pts_times = sorted(
        float(m.group(1))
        for m in re.finditer(r"pts_time:(\d+\.\d+)", result.stderr)
    )

    clips = []
    prev = 0.0
    for t in pts_times:
        clips.append(Clip(video_id=video.video_id,  # type: ignore
                     start_time=prev, end_time=t))
        prev = t
    clips.append(Clip(video_id=video.video_id, start_time=prev,  # type: ignore
                 end_time=video.duration))
    return clips


def merge_clips(clips: List[Clip], clip_ids_to_merge: List[int]) -> Clip:
    if not clip_ids_to_merge or sorted(clip_ids_to_merge) != clip_ids_to_merge:
        raise ValueError("clip_ids_to_merge must be a sorted, non-empty list")

    id_map = {clip.clip_id: clip for clip in clips}
    if not all(cid in id_map for cid in clip_ids_to_merge):
        raise ValueError("Some clip IDs not found")

    selected = [id_map[cid] for cid in clip_ids_to_merge]
    video_ids = {clip.video_id for clip in selected}
    if len(video_ids) != 1:
        raise ValueError("Clips must have the same video_id")

    start = min(clip.start_time for clip in selected)
    end = max(clip.end_time for clip in selected)

    for clip in selected:
        clips.remove(clip)

    merged = Clip(video_id=selected[0].video_id,
                  start_time=start, end_time=end)
    clips.append(merged)
    return merged


def split_clip(clips: List[Clip], clip_id_to_split: int, split_time: float) -> Tuple[Clip, Clip]:
    target = next((c for c in clips if c.clip_id == clip_id_to_split), None)
    if not target:
        raise ValueError(f"Clip ID {clip_id_to_split} not found")
    if not (target.start_time < split_time < target.end_time):
        raise ValueError("split_time must be within the bounds of the clip")

    clips.remove(target)

    new1 = Clip(video_id=target.video_id,
                start_time=target.start_time, end_time=split_time)
    new2 = Clip(video_id=target.video_id,
                start_time=split_time, end_time=target.end_time)

    clips.extend([new1, new2])
    return new1, new2


def extract_frames_for_clip(
    clip: Clip,
    video: Video,
    sampling_rate_fps: float,
    temp_dir: Optional[str] = None
) -> FrameBatch:
    if sampling_rate_fps <= 0:
        raise ValueError("sampling_rate_fps must be greater than 0")

    if clip.file_path is None:
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        temp_clip_path = os.path.join(
            temp_dir, f"clip_{clip.clip_id or 'temp'}.mp4")

        cmd = [
            "ffmpeg",
            "-ss", str(clip.start_time),
            "-to", str(clip.end_time),
            "-i", video.source_path,
            "-c", "copy",
            temp_clip_path
        ]

        try:
            subprocess.run(cmd, stderr=subprocess.PIPE,
                           stdout=subprocess.DEVNULL, text=True, check=True)
            clip.file_path = temp_clip_path
        except subprocess.CalledProcessError as e:
            raise AppError("Failed to trim clip.", internal_message=e.stderr)

    capture = cv2.VideoCapture(clip.file_path)
    if not capture.isOpened():
        raise AppError("Cannot open clip file for frame extraction.",
                       internal_message=clip.file_path)

    frame_batch = FrameBatch(clip_id=clip.clip_id or -1)
    frame_interval = 1.0 / sampling_rate_fps
    current_time = 0.0
    duration = clip.end_time - clip.start_time

    while current_time < duration:
        capture.set(cv2.CAP_PROP_POS_MSEC, (current_time * 1000))
        ret, frame = capture.read()
        if not ret:
            break
        frame_batch.add_frame(frame, clip.start_time + current_time)
        current_time += frame_interval

    capture.release()
    return frame_batch
