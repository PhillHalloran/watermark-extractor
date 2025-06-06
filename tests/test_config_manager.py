import json
import os
import pytest
# adjust the import to match your module name
from src.config_manager import ROI, AppConfig


def test_roi_valid():
    roi = ROI(10, 20, 100, 200)
    assert roi.x == 10
    assert roi.y == 20
    assert roi.width == 100
    assert roi.height == 200

    d = roi.to_dict()
    assert d == {"x": 10, "y": 20, "width": 100, "height": 200}

    roi2 = ROI.from_dict(d)
    assert isinstance(roi2, ROI)
    assert roi2.x == 10
    assert roi2.y == 20
    assert roi2.width == 100
    assert roi2.height == 200


@pytest.mark.parametrize(
    "d, error_msg",
    [
        ({}, 'ROI dictionary missing required key "x".'),
        (
            {"x": "a", "y": 0, "width": 10, "height": 10},
            'ROI field "x" must be an integer.',
        ),
        (
            {"x": 0, "y": 0, "width": -5, "height": 10},
            "ROI width and height must be positive integers.",
        ),
        (
            {"x": 0, "y": 0, "width": 5, "height": 0},
            "ROI width and height must be positive integers.",
        ),
    ],
)
def test_roi_from_dict_invalid(d, error_msg):
    with pytest.raises(ValueError) as excinfo:
        ROI.from_dict(d)
    assert error_msg in str(excinfo.value)


def test_roi_init_invalid():
    with pytest.raises(ValueError):
        ROI("a", 0, 10, 10)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ROI(0, 0, -10, 10)


def test_appconfig_defaults(tmp_path):
    config_file = tmp_path / "config.json"
    assert not config_file.exists()

    cfg = AppConfig(str(config_file))

    # After initialization, the file should have been created
    assert config_file.exists()

    # Check default in-memory values
    assert cfg.get_ocr_confidence_threshold() == 0.75
    assert cfg.get_frame_sampling_rate() == 1.0

    rois = cfg.get_default_rois()
    assert isinstance(rois, list)
    assert len(rois) == 2
    assert all(isinstance(r, ROI) for r in rois)

    formats = cfg.get_supported_file_formats()
    assert formats == ["mp4", "avi", "mov", "mkv"]

    # Verify on-disk contents match defaults
    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["ocr_confidence_threshold"] == 0.75
    assert data["frame_sampling_rate_fps"] == 1.0
    assert isinstance(data["default_rois"], list)
    assert data["supported_file_formats"] == ["mp4", "avi", "mov", "mkv"]


def test_appconfig_load_valid(tmp_path):
    config_file = tmp_path / "custom_config.json"
    custom = {
        "ocr_confidence_threshold": 0.5,
        "frame_sampling_rate_fps": 2.0,
        "default_rois": [{"x": 0, "y": 0, "width": 50, "height": 50}],
        "supported_file_formats": ["mp4", "mkv"],
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(custom, f)

    cfg = AppConfig(str(config_file))
    assert cfg.get_ocr_confidence_threshold() == 0.5
    assert cfg.get_frame_sampling_rate() == 2.0

    rois = cfg.get_default_rois()
    assert len(rois) == 1
    assert rois[0].x == 0 and rois[0].y == 0 and rois[0].width == 50 and rois[0].height == 50

    assert cfg.get_supported_file_formats() == ["mp4", "mkv"]


@pytest.mark.parametrize(
    "data, error_substr",
    [
        # Missing ocr_confidence_threshold
        (
            {"frame_sampling_rate_fps": 1.0, "default_rois": [],
                "supported_file_formats": []},
            'Missing required key "ocr_confidence_threshold".',
        ),
        # ocr_confidence_threshold out of range
        (
            {
                "ocr_confidence_threshold": 1.1,
                "frame_sampling_rate_fps": 1.0,
                "default_rois": [],
                "supported_file_formats": [],
            },
            '"ocr_confidence_threshold" must be between 0.0 and 1.0.',
        ),
        # Missing frame_sampling_rate_fps
        (
            {"ocr_confidence_threshold": 0.5, "default_rois": [],
                "supported_file_formats": []},
            'Missing required key "frame_sampling_rate_fps".',
        ),
        # frame_sampling_rate_fps not > 0
        (
            {
                "ocr_confidence_threshold": 0.5,
                "frame_sampling_rate_fps": 0.0,
                "default_rois": [],
                "supported_file_formats": [],
            },
            '"frame_sampling_rate_fps" must be greater than 0.',
        ),
        # Missing default_rois
        (
            {
                "ocr_confidence_threshold": 0.5,
                "frame_sampling_rate_fps": 1.0,
                "supported_file_formats": [],
            },
            'Missing required key "default_rois".',
        ),
        # default_rois not a list
        (
            {
                "ocr_confidence_threshold": 0.5,
                "frame_sampling_rate_fps": 1.0,
                "default_rois": "notalist",
                "supported_file_formats": [],
            },
            '"default_rois" must be a list of ROI dictionaries.',
        ),
        # default_rois contains invalid ROI (width <= 0)
        (
            {
                "ocr_confidence_threshold": 0.5,
                "frame_sampling_rate_fps": 1.0,
                "default_rois": [{"x": 0, "y": 0, "width": 0, "height": 10}],
                "supported_file_formats": [],
            },
            "ROI width and height must be positive integers.",
        ),
        # supported_file_formats not a list
        (
            {
                "ocr_confidence_threshold": 0.5,
                "frame_sampling_rate_fps": 1.0,
                "default_rois": [],
                "supported_file_formats": "notalist",
            },
            '"supported_file_formats" must be a list of strings.',
        ),
        # supported_file_formats has leading dot
        (
            {
                "ocr_confidence_threshold": 0.5,
                "frame_sampling_rate_fps": 1.0,
                "default_rois": [],
                "supported_file_formats": [".mp4"],
            },
            'File format ".mp4" should not have a leading dot.',
        ),
    ],
)
def test_appconfig_load_invalid(tmp_path, data, error_substr):
    config_file = tmp_path / "invalid_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    with pytest.raises(ValueError) as excinfo:
        AppConfig(str(config_file))
    assert error_substr in str(excinfo.value)


def test_appconfig_save(tmp_path):
    config_file = tmp_path / "config.json"
    cfg = AppConfig(str(config_file))

    # Change values
    cfg.set_ocr_confidence_threshold(0.3)
    cfg.set_frame_sampling_rate(5.0)
    cfg.set_default_rois([ROI(1, 1, 10, 10)])
    cfg.set_supported_file_formats(["avi", "mov"])

    # Save and verify on-disk content
    cfg.save()
    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["ocr_confidence_threshold"] == 0.3
    assert data["frame_sampling_rate_fps"] == 5.0
    assert data["default_rois"] == [
        {"x": 1, "y": 1, "width": 10, "height": 10}]
    assert data["supported_file_formats"] == ["avi", "mov"]


@pytest.mark.parametrize(
    "setter, value, error_substr",
    [
        ("set_ocr_confidence_threshold", -0.1, "must be between 0.0 and 1.0"),
        ("set_ocr_confidence_threshold", 2.0, "must be between 0.0 and 1.0"),
        ("set_ocr_confidence_threshold", "notafloat", "must be a number"),
        ("set_frame_sampling_rate", 0.0, "must be greater than 0"),
        ("set_frame_sampling_rate", -1, "must be greater than 0"),
        ("set_frame_sampling_rate", "no", "must be a number"),
        (
            "set_default_rois",
            "notalist",
            "default_rois must be provided as a list",
        ),
        ("set_default_rois", [123], "is not an ROI instance"),
        (
            "set_supported_file_formats",
            "notalist",
            "supported_file_formats must be provided as a list",
        ),
        ("set_supported_file_formats", [".MP4"], "must be lowercase"),
        ("set_supported_file_formats", [".mp4"],
         "should not have a leading dot"),
        ("set_supported_file_formats", [""], "must be non-empty strings"),
    ],
)
def test_appconfig_setters_invalid(tmp_path, setter, value, error_substr):
    config_file = tmp_path / "config.json"
    cfg = AppConfig(str(config_file))

    with pytest.raises(ValueError) as excinfo:
        getattr(cfg, setter)(value)
    assert error_substr in str(excinfo.value)
