import json
import os
from typing import List


class ROI:
    x: int
    y: int
    width: int
    height: int

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        """
        Create an ROI rectangle.
        """
        if not all(isinstance(v, int) for v in (x, y, width, height)):
            raise ValueError(
                "ROI coordinates and dimensions must be integers.")
        if width <= 0 or height <= 0:
            raise ValueError("ROI width and height must be positive integers.")
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def to_dict(self) -> dict:
        """
        Return {"x": x, "y": y, "width": width, "height": height}.
        """
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @staticmethod
    def from_dict(d: dict) -> "ROI":
        """
        Construct ROI from a dictionary with keys "x", "y", "width", "height".
        Raises ValueError if any key missing or not int, or width/height <= 0.
        """
        required_keys = ("x", "y", "width", "height")
        for key in required_keys:
            if key not in d:
                raise ValueError(
                    f'ROI dictionary missing required key "{key}".')
            if not isinstance(d[key], int):
                raise ValueError(f'ROI field "{key}" must be an integer.')

        x = d["x"]
        y = d["y"]
        width = d["width"]
        height = d["height"]

        if width <= 0 or height <= 0:
            raise ValueError("ROI width and height must be positive integers.")

        return ROI(x, y, width, height)


class AppConfig:
    """
    Holds all configurable parameters in memory.
    """

    ocr_confidence_threshold: float
    frame_sampling_rate_fps: float
    default_rois: List[ROI]
    supported_file_formats: List[str]

    def __init__(self, config_path: str = "config.json") -> None:
        """
        Load or create default config. If config_path does not exist, write defaults:
            ocr_confidence_threshold = 0.75
            frame_sampling_rate_fps = 1.0
            default_rois = [ROI(10,10,200,50), ROI(10,300,200,50)]
            supported_file_formats = ["mp4","avi","mov","mkv"]
        """
        self.config_path = config_path
        try:
            self.load()
        except FileNotFoundError:
            # Set defaults
            self.ocr_confidence_threshold = 0.75
            self.frame_sampling_rate_fps = 1.0
            self.default_rois = [ROI(10, 10, 200, 50), ROI(10, 300, 200, 50)]
            self.supported_file_formats = ["mp4", "avi", "mov", "mkv"]

            # Attempt to save defaults to disk
            self.save()

    def load(self) -> None:
        """
        Read JSON from file. Update all in-memory fields.
        Raises:
          - FileNotFoundError if file is missing (caller should catch and create defaults).
          - json.JSONDecodeError if file is invalid JSON.
          - ValueError if any field has wrong type/format.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f'Configuration file "{self.config_path}" not found.')

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate ocr_confidence_threshold
        if "ocr_confidence_threshold" not in data:
            raise ValueError(
                'Missing required key "ocr_confidence_threshold".')
        if not isinstance(data["ocr_confidence_threshold"], (int, float)):
            raise ValueError('"ocr_confidence_threshold" must be a float.')
        ocr_val = float(data["ocr_confidence_threshold"])
        if not (0.0 <= ocr_val <= 1.0):
            raise ValueError(
                '"ocr_confidence_threshold" must be between 0.0 and 1.0.')

        # Validate frame_sampling_rate_fps
        if "frame_sampling_rate_fps" not in data:
            raise ValueError('Missing required key "frame_sampling_rate_fps".')
        if not isinstance(data["frame_sampling_rate_fps"], (int, float)):
            raise ValueError('"frame_sampling_rate_fps" must be a float.')
        fps_val = float(data["frame_sampling_rate_fps"])
        if fps_val <= 0:
            raise ValueError(
                '"frame_sampling_rate_fps" must be greater than 0.')

        # Validate default_rois
        if "default_rois" not in data:
            raise ValueError('Missing required key "default_rois".')
        if not isinstance(data["default_rois"], list):
            raise ValueError(
                '"default_rois" must be a list of ROI dictionaries.')
        rois_list = []
        for idx, item in enumerate(data["default_rois"]):
            if not isinstance(item, dict):
                raise ValueError(
                    f'Each item in "default_rois" must be a dict (invalid at index {idx}).')
            roi = ROI.from_dict(item)  # This will validate internally
            rois_list.append(roi)

        # Validate supported_file_formats
        if "supported_file_formats" not in data:
            raise ValueError('Missing required key "supported_file_formats".')
        if not isinstance(data["supported_file_formats"], list):
            raise ValueError(
                '"supported_file_formats" must be a list of strings.')
        formats_list = []
        for idx, fmt in enumerate(data["supported_file_formats"]):
            if not isinstance(fmt, str):
                raise ValueError(
                    f'File format at index {idx} must be a string.')
            if fmt.strip() == "":
                raise ValueError("File formats must be non-empty strings.")
            if fmt.lower() != fmt:
                raise ValueError(f'File format "{fmt}" must be lowercase.')
            formats_list.append(fmt)
            if fmt.startswith("."):
                raise ValueError(
                    f'File format "{fmt}" should not have a leading dot.')

        # All validations passed; assign to in-memory fields
        self.ocr_confidence_threshold = ocr_val
        self.frame_sampling_rate_fps = fps_val
        self.default_rois = rois_list
        self.supported_file_formats = formats_list

    def save(self) -> None:
        """
        Write current in-memory values to config_path as JSON, pretty-printed.
        Raises:
          - IOError if cannot write to disk.
        """
        to_dump = {
            "ocr_confidence_threshold": self.ocr_confidence_threshold,
            "frame_sampling_rate_fps": self.frame_sampling_rate_fps,
            "default_rois": [roi.to_dict() for roi in self.default_rois],
            "supported_file_formats": self.supported_file_formats,
        }
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(to_dump, f, indent=2)
        except OSError as e:
            raise IOError(f"Failed to write configuration to disk: {e}")

    # Getter and setter methods

    def get_ocr_confidence_threshold(self) -> float:
        return self.ocr_confidence_threshold

    def set_ocr_confidence_threshold(self, value: float) -> None:
        """
        Validate 0.0 <= value <= 1.0 before setting.
        """
        if not isinstance(value, (int, float)):
            raise ValueError("ocr_confidence_threshold must be a number.")
        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                "ocr_confidence_threshold must be between 0.0 and 1.0.")
        self.ocr_confidence_threshold = v

    def get_frame_sampling_rate(self) -> float:
        return self.frame_sampling_rate_fps

    def set_frame_sampling_rate(self, fps: float) -> None:
        """
        Validate fps > 0 before setting.
        """
        if not isinstance(fps, (int, float)):
            raise ValueError("frame_sampling_rate_fps must be a number.")
        v = float(fps)
        if v <= 0:
            raise ValueError("frame_sampling_rate_fps must be greater than 0.")
        self.frame_sampling_rate_fps = v

    def get_default_rois(self) -> List[ROI]:
        return self.default_rois

    def set_default_rois(self, rois: List[ROI]) -> None:
        """
        Validate each item is ROI instance.
        """
        if not isinstance(rois, list):
            raise ValueError(
                "default_rois must be provided as a list of ROI instances.")
        for idx, r in enumerate(rois):
            if not isinstance(r, ROI):
                raise ValueError(
                    f"Item at index {idx} in default_rois is not an ROI instance.")
        self.default_rois = rois

    def get_supported_file_formats(self) -> List[str]:
        return self.supported_file_formats

    def set_supported_file_formats(self, formats: list[str]) -> None:
        """
        Validate each element is a lowercase string without leading dots.
        """
        if not isinstance(formats, list):
            raise ValueError(
                "supported_file_formats must be provided as a list of strings.")

        clean_formats = []
        for idx, fmt in enumerate(formats):
            if not isinstance(fmt, str):
                raise ValueError(
                    f"File format at index {idx} must be a string.")

            if fmt.strip() == "":
                raise ValueError("File formats must be non-empty strings.")

            # 1) Enforce lowercase first
            if fmt.lower() != fmt:
                raise ValueError(f'File format "{fmt}" must be lowercase.')

            # 2) Then enforce no leading dot
            if fmt.startswith("."):
                raise ValueError(
                    f'File format "{fmt}" should not have a leading dot.')

            clean_formats.append(fmt)

        self.supported_file_formats = clean_formats
