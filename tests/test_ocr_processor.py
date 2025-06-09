import pytest
import numpy as np
import pytesseract
from pytesseract import TesseractNotFoundError

from src.config_manager import ROI
from src.video_processor import FrameBatch
from src.ocr_processor import ROIManager, process_frame_for_ocr, process_batch_for_ocr, OCRResult


class DummyTesseract:
    """Helper to simulate pytesseract.image_to_data responses."""
    @staticmethod
    def good_data(img, output_type=None):
        # Two tokens with confidences 85 and 90
        return {'text': ['Hello', 'World'], 'conf': ['85', '90']}

    @staticmethod
    def empty_data(img, output_type=None):
        # No valid tokens
        return {'text': ['', ''], 'conf': ['-1', '-1']}

    @staticmethod
    def missing_tesseract(img, output_type=None):
        raise TesseractNotFoundError()


@pytest.fixture(autouse=True)
def clear_patches(monkeypatch):
    # Ensure no cross-test contamination
    yield


def test_roi_manager_add_list_remove():
    mgr = ROIManager()
    r = ROI(5, 5, 10, 10)
    mgr.add_roi(r)
    listed = mgr.list_rois()
    assert listed == [r]
    assert listed is not mgr.rois

    mgr.remove_roi(0)
    assert mgr.list_rois() == []


@pytest.mark.parametrize("coords", [
    (-1, 0, 10, 10),
    (0, -1, 10, 10),
    (0, 0, 0, 10),
    (0, 0, 10, 0),
])
def test_roi_manager_add_invalid(coords):
    mgr = ROIManager()
    x, y, w, h = coords
    with pytest.raises(ValueError):
        mgr.add_roi(ROI(x, y, w, h))


def test_roi_manager_remove_invalid():
    mgr = ROIManager()
    with pytest.raises(IndexError):
        mgr.remove_roi(0)


def test_process_frame_for_ocr_normal(monkeypatch):
    monkeypatch.setattr(pytesseract, "image_to_data", DummyTesseract.good_data)

    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    roi = ROI(0, 0, 20, 20)

    results = process_frame_for_ocr(
        frame=frame,
        timestamp=1.23,
        video_id=42,
        clip_id=7,
        rois=[roi],
        confidence_threshold=0.5,
    )

    assert isinstance(results, list)
    assert len(results) == 1

    res = results[0]
    assert isinstance(res, OCRResult)
    assert res.video_id == 42
    assert res.clip_id == 7
    assert res.timestamp == 1.23
    assert res.text == "HelloWorld"
    assert pytest.approx(0.875, rel=1e-3) == res.confidence
    assert res.roi == roi


def test_process_frame_for_ocr_filters_by_threshold(monkeypatch):
    monkeypatch.setattr(pytesseract, "image_to_data", DummyTesseract.good_data)

    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    roi = ROI(0, 0, 10, 10)

    empty = process_frame_for_ocr(
        frame, 0.0, 1, 1, [roi], confidence_threshold=0.9)
    assert empty == []


@pytest.mark.parametrize("thr", [-0.1, 1.1])
def test_process_frame_for_ocr_bad_threshold(thr):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        process_frame_for_ocr(frame, 0.0, 1, 1, [], confidence_threshold=thr)


def test_process_frame_for_ocr_tesseract_missing(monkeypatch):
    monkeypatch.setattr(pytesseract, "image_to_data",
                        DummyTesseract.missing_tesseract)

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    roi = ROI(0, 0, 5, 5)

    with pytest.raises(RuntimeError):
        process_frame_for_ocr(
            frame, 0.0, 1, 1, [roi], confidence_threshold=0.0)


def test_process_frame_for_ocr_empty_text(monkeypatch):
    monkeypatch.setattr(pytesseract, "image_to_data",
                        DummyTesseract.empty_data)

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    roi = ROI(0, 0, 10, 10)

    results = process_frame_for_ocr(
        frame, 0.0, 1, 1, [roi], confidence_threshold=0.0)
    assert results == []


def test_process_batch_for_ocr_aggregates(monkeypatch):
    def one_token(img, output_type=None):
        return {'text': ['X'], 'conf': ['100']}

    monkeypatch.setattr(pytesseract, "image_to_data", one_token)

    fb = FrameBatch(clip_id=99)
    fb.add_frame(np.zeros((5, 5, 3), dtype=np.uint8), timestamp=0.1)
    fb.add_frame(np.zeros((5, 5, 3), dtype=np.uint8), timestamp=0.2)

    roi = ROI(0, 0, 3, 3)
    results = process_batch_for_ocr(fb, [roi], confidence_threshold=0.0)

    assert len(results) == 2
    assert [r.timestamp for r in results] == [0.1, 0.2]
    for r in results:
        assert r.text == "X"
        assert pytest.approx(1.0) == r.confidence


@pytest.mark.parametrize("thr", [-0.5, 1.5])
def test_process_batch_for_ocr_bad_threshold(thr):
    fb = FrameBatch(clip_id=1)
    with pytest.raises(ValueError):
        process_batch_for_ocr(fb, [], confidence_threshold=thr)
