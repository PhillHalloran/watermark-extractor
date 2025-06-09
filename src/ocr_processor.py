import logging
from typing import List
import numpy as np
import cv2
import pytesseract
from pytesseract import Output, TesseractNotFoundError

from .config_manager import ROI
from .video_processor import FrameBatch

logger = logging.getLogger(__name__)


class OCRResult:
    """
    Represents a single OCR detection result within a frame ROI.
    """

    def __init__(
        self,
        video_id: int,
        clip_id: int,
        timestamp: float,
        text: str,
        confidence: float,
        roi: ROI
    ) -> None:
        if not text:
            raise ValueError("OCRResult text must be non-empty")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(
                "OCRResult confidence must be between 0.0 and 1.0")
        self.video_id = video_id
        self.clip_id = clip_id
        self.timestamp = timestamp
        self.text = text
        self.confidence = confidence
        self.roi = roi


class ROIManager:
    """
    Manage a list of ROIs where OCR should be run.
    """

    def __init__(self, initial_rois: List[ROI] = None) -> None:  # type: ignore
        if initial_rois:
            # Deep-copy provided ROIs
            self.rois = [ROI(r.x, r.y, r.width, r.height)
                         for r in initial_rois]
        else:
            self.rois = []

    def add_roi(self, roi: ROI) -> None:
        # only check width/height
        if roi.width <= 0 or roi.height <= 0:
            raise ValueError("ROI width and height must be positive integers.")
        # x,y must be non-negative, width/height must be positive
        if roi.x < 0 or roi.y < 0:
            raise ValueError("ROI x and y must be non-negative integers.")
        if roi.width <= 0 or roi.height <= 0:
            raise ValueError("ROI width and height must be positive integers.")
        self.rois.append(roi)

    def remove_roi(self, index: int) -> None:
        """
        Remove ROI at the specified index.
        Raises IndexError if index is invalid.
        """
        del self.rois[index]

    def list_rois(self) -> List[ROI]:
        """
        Return a shallow copy of the current ROI list.
        """
        return self.rois.copy()


def process_frame_for_ocr(
    frame: np.ndarray,
    timestamp: float,
    video_id: int,
    clip_id: int,
    rois: List[ROI],
    confidence_threshold: float
) -> List[OCRResult]:
    """
    Run OCR on each ROI in a frame, filter by confidence, and return OCRResult list.
    """
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")

    results: List[OCRResult] = []
    for roi in rois:
        x, y, w, h = roi.x, roi.y, roi.width, roi.height
        # Validate ROI bounds
        if y < 0 or x < 0 or y + h > frame.shape[0] or x + w > frame.shape[1]:
            logger.warning(
                f"ROI {roi} is out of frame bounds and will be skipped.")
            continue
        sub_img = frame[y: y + h, x: x + w]
        if sub_img.size == 0:
            logger.warning(
                f"ROI {roi} resulted in empty crop and will be skipped.")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        try:
            data = pytesseract.image_to_data(gray, output_type=Output.DICT)
        except TesseractNotFoundError:
            raise RuntimeError("Tesseract not installed or not in PATH.")
        except Exception as e:
            logger.error(f"OCR engine error: {e}")
            raise RuntimeError("Error during OCR processing.")

        texts = data.get("text", [])
        confs = data.get("conf", [])
        numeric_confs: List[float] = []
        tokens: List[str] = []

        # Parse confidences and assemble tokens
        for text, conf_str in zip(texts, confs):
            try:
                conf_val = float(conf_str) / 100.0
            except (ValueError, TypeError):
                continue
            numeric_confs.append(conf_val)
            if conf_val > 0 and text.strip():
                tokens.append(text)

        text_str = "".join(tokens).strip()
        overall_conf = sum(numeric_confs) / \
            len(numeric_confs) if numeric_confs else 0.0

        if text_str and overall_conf >= confidence_threshold:
            result = OCRResult(
                video_id=video_id,
                clip_id=clip_id,
                timestamp=timestamp,
                text=text_str,
                confidence=overall_conf,
                roi=roi
            )
            results.append(result)
    return results


def process_batch_for_ocr(
    frame_batch: FrameBatch,
    rois: List[ROI],
    confidence_threshold: float
) -> List[OCRResult]:
    """
    Process an entire batch of frames for OCR, returning all OCRResult entries.
    """
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")

    all_results: List[OCRResult] = []
    # Note: video_id and clip_id are both taken from frame_batch.clip_id per spec
    for frame, ts in zip(frame_batch.frames, frame_batch.timestamps):
        partial = process_frame_for_ocr(
            frame,
            ts,
            frame_batch.clip_id,
            frame_batch.clip_id,
            rois,
            confidence_threshold
        )
        all_results.extend(partial)
    return all_results
