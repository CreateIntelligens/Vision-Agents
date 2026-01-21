"""Security camera processor with face and package detection."""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar

import aiortc
import av
import cv2
import face_recognition
import numpy as np

from vision_agents.core.events.base import PluginBaseEvent
from vision_agents.core.events.manager import EventManager
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.core.warmup import Warmable

logger = logging.getLogger(__name__)

# Constants
OVERLAY_WIDTH = 200
GRID_COLS = 2
MAX_THUMBNAILS = 12
PICKUP_THRESHOLD_SECONDS = 5.0
PICKUP_MAX_AGE_SECONDS = 300.0
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 150, 150)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (200, 200, 200)
COLOR_BLACK = (0, 0, 0)
COLOR_DARK_GRAY = (40, 40, 40)

# Font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.3
FONT_MEDIUM = 0.4
FONT_LARGE = 0.5

# Detection classes for YOLO
PACKAGE_DETECT_CLASSES = [
    "Box", "Box_broken", "Open_package", "Package",  # Custom model
    "suitcase", "backpack", "handbag", "bottle", "book",  # COCO pretrained
]


@dataclass
class PersonDetectedEvent(PluginBaseEvent):
    """Event emitted when a person/face is detected."""

    type: str = field(default="security.person_detected", init=False)
    face_id: str = ""
    is_new: bool = False
    detection_count: int = 1
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


@dataclass
class PackageDetectedEvent(PluginBaseEvent):
    """Event emitted when a package is detected."""

    type: str = field(default="security.package_detected", init=False)
    package_id: str = ""
    is_new: bool = False
    detection_count: int = 1
    confidence: float = 0.0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


@dataclass
class PackageDisappearedEvent(PluginBaseEvent):
    """Event emitted when a package disappears from the frame."""

    type: str = field(default="security.package_disappeared", init=False)
    package_id: str = ""
    confidence: float = 0.0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    picker_face_id: Optional[str] = None
    picker_name: Optional[str] = None


@dataclass
class PersonDisappearedEvent(PluginBaseEvent):
    """Event emitted when a person disappears from the frame."""

    type: str = field(default="security.person_disappeared", init=False)
    face_id: str = ""
    name: Optional[str] = None
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


@dataclass
class FaceDetection:
    """Represents a detected face with metadata."""

    face_id: str
    face_image: np.ndarray
    face_encoding: np.ndarray
    first_seen: float
    last_seen: float
    bbox: tuple
    detection_count: int = 1
    name: Optional[str] = None
    disappeared_at: Optional[float] = None
    _event_sent: bool = False


@dataclass
class PackageDetection:
    """Represents a detected package with metadata."""

    package_id: str
    package_image: np.ndarray
    first_seen: float
    last_seen: float
    bbox: tuple
    confidence: float
    detection_count: int = 1
    disappeared_at: Optional[float] = None


@dataclass
class KnownFace:
    """Represents a known/registered face."""

    name: str
    face_encoding: np.ndarray
    registered_at: float


@dataclass
class ActivityLogEntry:
    """Represents an entry in the activity log."""

    timestamp: float
    event_type: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


T = TypeVar("T", FaceDetection, PackageDetection)


def format_timestamp(timestamp: float) -> str:
    """Format a Unix timestamp as a human-readable string."""
    return time.strftime(TIMESTAMP_FORMAT, time.localtime(timestamp))


def calculate_iou(bbox1: tuple, bbox2: tuple) -> float:
    """Calculate Intersection over Union between two (x, y, w, h) bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    inter_x_min = max(x1, x2)
    inter_y_min = max(y1, y2)
    inter_x_max = min(x1 + w1, x2 + w2)
    inter_y_max = min(y1 + h1, y2 + h2)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    union_area = w1 * h1 + w2 * h2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def get_bbox_centroid(bbox: tuple) -> tuple[float, float]:
    """Get the centroid of a (x, y, w, h) bounding box."""
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def calculate_centroid_distance(bbox1: tuple, bbox2: tuple) -> float:
    """Calculate Euclidean distance between centroids of two bounding boxes."""
    cx1, cy1 = get_bbox_centroid(bbox1)
    cx2, cy2 = get_bbox_centroid(bbox2)
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def clamp_bbox(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> tuple:
    """Clamp bounding box coordinates to frame bounds."""
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return (x, y, w, h)


def draw_labeled_bbox(
    frame: np.ndarray,
    bbox: tuple,
    label: str,
    color: tuple,
    frame_size: tuple[int, int],
) -> None:
    """Draw a labeled bounding box on the frame."""
    x, y, w, h = [int(v) for v in bbox]
    frame_h, frame_w = frame_size
    x, y, w, h = clamp_bbox(x, y, w, h, frame_w, frame_h)
    x2, y2 = x + w, y + h

    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    cv2.putText(frame, label, (x, max(10, y - 5)), FONT, FONT_MEDIUM, color, 1, cv2.LINE_AA)


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple,
    color: tuple = COLOR_WHITE,
    scale: float = FONT_LARGE,
) -> None:
    """Draw text on the frame."""
    cv2.putText(frame, text, position, FONT, scale, color, 1, cv2.LINE_AA)


class SecurityCameraProcessor(VideoProcessorPublisher, Warmable[Optional[Any]]):
    """
    Security camera processor that detects and recognizes faces and packages.

    Detects faces using face_recognition library and packages using YOLO.
    Maintains a sliding window of unique visitors and packages with thumbnails.
    """

    name = "security_camera"

    def __init__(
        self,
        fps: int = 5,
        max_workers: int = 10,
        time_window: int = 1800,
        thumbnail_size: int = 80,
        detection_interval: float = 2.0,
        bbox_update_interval: float = 0.3,
        face_match_tolerance: float = 0.6,
        person_disappeared_threshold: float = 2.0,
        model_path: str = "weights_custom.pt",
        device: str = "cpu",
        package_detection_interval: float = 0.4,
        package_fps: int = 1,
        package_conf_threshold: float = 0.6,
        package_min_area_ratio: float = 0.01,
        package_max_area_ratio: float = 0.9,
        max_tracked_packages: Optional[int] = None,
    ):
        self.fps = fps
        self.max_workers = max_workers
        self.time_window = time_window
        self.thumbnail_size = thumbnail_size
        self.detection_interval = detection_interval
        self.bbox_update_interval = bbox_update_interval
        self.face_match_tolerance = face_match_tolerance
        self.person_disappeared_threshold = person_disappeared_threshold
        self.package_detection_interval = package_detection_interval
        self.package_fps = package_fps
        self.package_conf_threshold = package_conf_threshold
        self.package_min_area_ratio = package_min_area_ratio
        self.package_max_area_ratio = package_max_area_ratio
        self.max_tracked_packages = max_tracked_packages

        self._detected_faces: Dict[str, FaceDetection] = {}
        self._detected_packages: Dict[str, PackageDetection] = {}
        self._known_faces: Dict[str, KnownFace] = {}
        self._activity_log: List[ActivityLogEntry] = []
        self._max_activity_log_entries = 100

        self._last_detection_time = 0.0
        self._last_bbox_update_time = 0.0
        self._last_package_detection_time = 0.0

        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="security_camera"
        )
        self._shutdown = False

        self._video_track: QueuedVideoTrack = QueuedVideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None
        self._shared_image: Optional[av.VideoFrame] = None
        self._shared_image_until: float = 0.0

        self.model_path = model_path
        self.device = device
        self.yolo_model: Optional[Any] = None

        self.events = EventManager()
        for event_cls in [PersonDetectedEvent, PackageDetectedEvent, PackageDisappearedEvent, PersonDisappearedEvent]:
            self.events.register(event_cls)

        logger.info(f"Security Camera Processor initialized (window: {time_window // 60}min)")

    def _cleanup_old_items(
        self, items: Dict[str, T], current_time: float, item_type: str
    ) -> int:
        """Remove items whose last_seen is older than the time window."""
        cutoff_time = current_time - self.time_window
        to_remove = [k for k, v in items.items() if v.last_seen < cutoff_time]
        for key in to_remove:
            del items[key]
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old {item_type}(s)")
        return len(to_remove)

    def _cleanup_old_faces(self, current_time: float) -> int:
        return self._cleanup_old_items(self._detected_faces, current_time, "face")

    def _cleanup_old_packages(self, current_time: float) -> int:
        return self._cleanup_old_items(self._detected_packages, current_time, "package")

    async def on_warmup(self) -> Optional[Any]:
        """Load YOLO model for package detection."""
        try:
            from ultralytics import YOLO
            loop = asyncio.get_event_loop()
            def load_model():
                model = YOLO(self.model_path)
                model.to(self.device)
                return model
            yolo_model = await loop.run_in_executor(self.executor, load_model)
            logger.info(f"YOLO model loaded: {self.model_path}")
            return yolo_model
        except Exception as e:
            logger.warning(f"YOLO model failed to load: {e} - package detection disabled")
            return None

    def on_warmed_up(self, resource: Optional[Any]) -> None:
        self.yolo_model = resource

    def _log_activity(
        self, event_type: str, description: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an entry to the activity log."""
        self._activity_log.append(ActivityLogEntry(
            timestamp=time.time(),
            event_type=event_type,
            description=description,
            details=details or {},
        ))
        if len(self._activity_log) > self._max_activity_log_entries:
            self._activity_log = self._activity_log[-self._max_activity_log_entries:]

    def _find_person_present_at(self, timestamp: float) -> Optional[FaceDetection]:
        """Find who was present around a given timestamp (within 10 seconds)."""
        candidates = [f for f in self._detected_faces.values() if abs(f.last_seen - timestamp) < 10.0]
        return max(candidates, key=lambda f: f.last_seen) if candidates else None

    def _check_for_picked_up_packages(self, current_time: float) -> None:
        """Check if any packages have disappeared (picked up)."""
        to_remove = []
        for package_id, package in self._detected_packages.items():
            time_since_seen = current_time - package.last_seen
            package_age = current_time - package.first_seen

            if PICKUP_THRESHOLD_SECONDS < time_since_seen < PICKUP_MAX_AGE_SECONDS and package_age < PICKUP_MAX_AGE_SECONDS:
                picker = self._find_person_present_at(package.last_seen)
                picker_name = picker.name if picker and picker.name else (picker.face_id[:8] if picker else "unknown person")

                logger.info(f"Package {package_id[:8]} was picked up by {picker_name}")
                self._log_activity(
                    "package_picked_up",
                    f"Package picked up by {picker_name}",
                    {
                        "package_id": package_id[:8],
                        "picked_up_by": picker_name,
                        "picker_face_id": picker.face_id[:8] if picker else None,
                        "picker_is_known": picker.name is not None if picker else False,
                    },
                )
                to_remove.append(package_id)

        for package_id in to_remove:
            del self._detected_packages[package_id]

    def _find_matching_package(
        self, bbox: tuple, frame_shape: tuple[int, int], iou_threshold: float = 0.3
    ) -> Optional[str]:
        """Find matching package based on IoU overlap, with centroid distance fallback."""
        if not self._detected_packages:
            return None

        if self.max_tracked_packages == 1 and len(self._detected_packages) == 1:
            return next(iter(self._detected_packages.keys()))

        best_match = None
        best_iou = 0.0
        for pkg_id, pkg in self._detected_packages.items():
            iou = calculate_iou(bbox, pkg.bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = pkg_id

        if best_match:
            return best_match

        frame_h, frame_w = frame_shape
        max_distance = ((frame_w ** 2 + frame_h ** 2) ** 0.5) * 0.25
        best_distance = float("inf")

        for pkg_id, pkg in self._detected_packages.items():
            dist = calculate_centroid_distance(bbox, pkg.bbox)
            if dist < best_distance and dist < max_distance:
                best_distance = dist
                best_match = pkg_id

        return best_match

    def _detect_faces_sync(self, frame_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces synchronously, returning bbox and encoding."""
        face_locations = face_recognition.face_locations(frame_rgb, model="hog")
        if not face_locations:
            return []

        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
        return [
            {"bbox": (left, top, right - left, bottom - top), "encoding": enc}
            for (top, right, bottom, left), enc in zip(face_locations, face_encodings)
        ]

    def _find_matching_face(self, face_encoding: np.ndarray) -> Optional[str]:
        """Find existing face that matches the encoding."""
        if not self._detected_faces:
            return None

        face_ids = list(self._detected_faces.keys())
        encodings = [self._detected_faces[fid].face_encoding for fid in face_ids]
        matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=self.face_match_tolerance)

        for i, is_match in enumerate(matches):
            if is_match:
                return face_ids[i]
        return None

    def _find_known_face_name(self, face_encoding: np.ndarray) -> Optional[str]:
        """Check if face matches any known/registered face."""
        if not self._known_faces:
            return None

        names = list(self._known_faces.keys())
        encodings = [self._known_faces[n].face_encoding for n in names]
        matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=self.face_match_tolerance)

        for i, is_match in enumerate(matches):
            if is_match:
                return names[i]
        return None

    def _detect_face_locations_fast_sync(self, frame_rgb: np.ndarray) -> List[tuple]:
        """Fast face location detection without encoding."""
        face_locations = face_recognition.face_locations(frame_rgb, model="hog")
        return [(left, top, right - left, bottom - top) for top, right, bottom, left in face_locations]

    def _match_bbox_to_face(
        self, bbox: tuple, frame_shape: tuple[int, int], max_distance_ratio: float = 0.15
    ) -> Optional[str]:
        """Match a detected bbox to an existing face based on proximity."""
        active_faces = {fid: f for fid, f in self._detected_faces.items() if f.disappeared_at is None}
        if not active_faces:
            return None

        frame_h, frame_w = frame_shape
        max_distance = ((frame_w ** 2 + frame_h ** 2) ** 0.5) * max_distance_ratio

        best_match = None
        best_distance = float("inf")
        for face_id, face in active_faces.items():
            dist = calculate_centroid_distance(bbox, face.bbox)
            if dist < best_distance and dist < max_distance:
                best_distance = dist
                best_match = face_id

        return best_match

    async def _update_face_bboxes_fast(self, frame_bgr: np.ndarray, current_time: float) -> None:
        """Fast bbox update for existing faces."""
        if self._shutdown or current_time - self._last_bbox_update_time < self.bbox_update_interval:
            return

        active_faces = [f for f in self._detected_faces.values() if f.disappeared_at is None]
        if not active_faces:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        loop = asyncio.get_event_loop()
        detected_bboxes = await loop.run_in_executor(self.executor, self._detect_face_locations_fast_sync, frame_rgb)

        frame_shape = frame_bgr.shape[:2]
        for bbox in detected_bboxes:
            face_id = self._match_bbox_to_face(bbox, frame_shape)
            if face_id:
                self._detected_faces[face_id].bbox = bbox

        self._last_bbox_update_time = current_time

    def _extract_thumbnail(self, frame_bgr: np.ndarray, bbox: tuple, padding_ratio: float = 0.0) -> np.ndarray:
        """Extract and resize a thumbnail from the frame."""
        x, y, w, h = [int(v) for v in bbox]
        frame_h, frame_w = frame_bgr.shape[:2]

        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_y)

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros((self.thumbnail_size, self.thumbnail_size, 3), dtype=np.uint8)
        return cv2.resize(roi, (self.thumbnail_size, self.thumbnail_size))

    def _emit_person_event(self, face: FaceDetection, is_new: bool, current_time: float) -> None:
        """Emit a person detected event."""
        display_name = face.name or face.face_id[:8]
        self.events.send(PersonDetectedEvent(
            plugin_name="security_camera",
            face_id=display_name,
            is_new=is_new,
            detection_count=face.detection_count,
            first_seen=format_timestamp(face.first_seen),
            last_seen=format_timestamp(current_time),
        ))

    def _emit_package_event(self, package: PackageDetection, is_new: bool, current_time: float) -> None:
        """Emit a package detected event."""
        self.events.send(PackageDetectedEvent(
            plugin_name="security_camera",
            package_id=package.package_id[:8],
            is_new=is_new,
            detection_count=package.detection_count,
            confidence=package.confidence,
            first_seen=format_timestamp(package.first_seen),
            last_seen=format_timestamp(current_time),
        ))

    async def _detect_and_store_faces(self, frame_bgr: np.ndarray, current_time: float) -> int:
        """Detect faces in frame and store new unique faces or update existing ones."""
        if self._shutdown or current_time - self._last_detection_time < self.detection_interval:
            return 0

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        loop = asyncio.get_event_loop()
        detected_faces = await loop.run_in_executor(self.executor, self._detect_faces_sync, frame_rgb)

        new_faces = 0
        faces_seen: set[str] = set()

        for face_data in detected_faces:
            bbox = face_data["bbox"]
            encoding = face_data["encoding"]
            known_name = self._find_known_face_name(encoding)
            matching_id = self._find_matching_face(encoding)

            if matching_id:
                face = self._detected_faces[matching_id]
                faces_seen.add(matching_id)
                face.last_seen = current_time
                face.bbox = bbox
                face.face_image = self._extract_thumbnail(frame_bgr, bbox, padding_ratio=0.3)

                if known_name and not face.name:
                    face.name = known_name

                # Only emit "return" event if person actually left (event was sent)
                if face.disappeared_at is not None and face._event_sent:
                    face.detection_count += 1
                    display_name = face.name or matching_id[:8]
                    logger.info(f"Returning: {display_name} (visit #{face.detection_count})")
                    self._emit_person_event(face, is_new=False, current_time=current_time)
                    face._event_sent = False
                elif face.disappeared_at is not None:
                    # Person was briefly out of frame but didn't actually "leave"
                    logger.debug(f"Redetected: {face.name or matching_id[:8]} (was briefly occluded)")
                
                # Reset disappeared state when person is seen again
                face.disappeared_at = None
            else:
                face_id = str(uuid.uuid4())
                detection = FaceDetection(
                    face_id=face_id,
                    face_image=self._extract_thumbnail(frame_bgr, bbox, padding_ratio=0.3),
                    face_encoding=encoding,
                    first_seen=current_time,
                    last_seen=current_time,
                    bbox=bbox,
                    name=known_name,
                )
                self._detected_faces[face_id] = detection
                faces_seen.add(face_id)
                new_faces += 1

                display_name = known_name or face_id[:8]
                logger.info(f"New unique visitor detected: {display_name}")
                self._log_activity("person_arrived", f"New person arrived: {display_name}", {
                    "face_id": face_id[:8],
                    "name": known_name,
                    "is_known": known_name is not None,
                })
                self._emit_person_event(detection, is_new=True, current_time=current_time)

        self._handle_disappeared_faces(faces_seen, current_time)

        if new_faces > 0:
            self._last_detection_time = current_time
        return new_faces

    def _handle_disappeared_faces(self, faces_seen: set[str], current_time: float) -> None:
        """Handle faces that weren't seen in the current frame."""
        for face_id, face in self._detected_faces.items():
            if face_id in faces_seen:
                continue

            if face.disappeared_at is None:
                face.disappeared_at = current_time
                logger.debug(f"Person temporarily out of frame: {face.name or face_id[:8]}")
            elif current_time - face.disappeared_at >= self.person_disappeared_threshold and not face._event_sent:
                display_name = face.name or face_id[:8]
                logger.info(f"Person left: {display_name} (not seen for {self.person_disappeared_threshold}s)")
                self._log_activity("person_left", f"Person left: {display_name}", {
                    "face_id": face_id[:8],
                    "name": face.name,
                    "is_known": face.name is not None,
                })
                self.events.send(PersonDisappearedEvent(
                    plugin_name="security_camera",
                    face_id=face_id,
                    name=face.name,
                    first_seen=format_timestamp(face.first_seen),
                    last_seen=format_timestamp(face.last_seen),
                ))
                face._event_sent = True

    def _detect_packages_sync(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO package detection synchronously."""
        if not self.yolo_model:
            return []

        height, width = frame_bgr.shape[:2]
        detections = []

        try:
            results = self.yolo_model(frame_bgr, verbose=False, conf=self.package_conf_threshold, device=self.device)
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                return []

            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = class_names[cls_id].lower()
                if not any(dc.lower() in class_name for dc in PACKAGE_DETECT_CLASSES):
                    continue

                x_min, y_min, x_max, y_max = [int(max(0, min(v, width if i % 2 == 0 else height))) for i, v in enumerate(box)]
                w, h = x_max - x_min, y_max - y_min

                area_ratio = (w * h) / (width * height)
                if area_ratio < self.package_min_area_ratio or area_ratio > self.package_max_area_ratio:
                    continue

                detections.append({"bbox": (x_min, y_min, w, h), "confidence": float(conf), "label": class_names[cls_id]})

        except Exception as e:
            logger.warning(f"Package detection failed: {e}")

        return detections

    async def _detect_and_store_packages(self, frame_bgr: np.ndarray, current_time: float) -> int:
        """Detect packages in frame and store new unique packages or update existing ones."""
        if self._shutdown or not self.yolo_model:
            return 0

        if current_time - self._last_package_detection_time < self.package_detection_interval:
            return 0

        loop = asyncio.get_event_loop()
        detected_packages = await loop.run_in_executor(self.executor, self._detect_packages_sync, frame_bgr)

        new_packages = 0
        packages_seen: set[str] = set()
        frame_shape = frame_bgr.shape[:2]

        for pkg_data in detected_packages:
            bbox = pkg_data["bbox"]
            confidence = pkg_data["confidence"]
            x, y, w, h = clamp_bbox(*[int(v) for v in bbox], frame_bgr.shape[1], frame_bgr.shape[0])

            thumbnail = self._extract_thumbnail(frame_bgr, (x, y, w, h))
            if thumbnail.size == 0:
                continue

            matching_id = self._find_matching_package((x, y, w, h), frame_shape)

            if matching_id:
                pkg = self._detected_packages[matching_id]
                packages_seen.add(matching_id)
                pkg.last_seen = current_time
                pkg.bbox = (x, y, w, h)
                pkg.confidence = max(pkg.confidence, confidence)
                pkg.package_image = thumbnail

                if pkg.disappeared_at is not None:
                    pkg.detection_count += 1
                    logger.info(f"Package returned: {matching_id[:8]}")
                    self._emit_package_event(pkg, is_new=False, current_time=current_time)
                    pkg.disappeared_at = None
            else:
                package_id = str(uuid.uuid4())
                detection = PackageDetection(
                    package_id=package_id,
                    package_image=thumbnail,
                    first_seen=current_time,
                    last_seen=current_time,
                    bbox=(x, y, w, h),
                    confidence=confidence,
                )
                self._detected_packages[package_id] = detection
                packages_seen.add(package_id)
                new_packages += 1

                logger.info(f"New unique package detected: {package_id[:8]}")
                self._log_activity("package_arrived", f"New package detected (confidence: {confidence:.2f})", {
                    "package_id": package_id[:8],
                    "confidence": confidence,
                })
                self._emit_package_event(detection, is_new=True, current_time=current_time)

        self._handle_disappeared_packages(packages_seen, current_time)

        if new_packages > 0:
            self._last_package_detection_time = current_time
        return new_packages

    def _handle_disappeared_packages(self, packages_seen: set[str], current_time: float) -> None:
        """Handle packages that weren't seen in the current frame."""
        for pkg_id, pkg in self._detected_packages.items():
            if pkg_id in packages_seen or pkg.disappeared_at is not None:
                continue

            pkg.disappeared_at = current_time
            picker = self._find_person_present_at(pkg.last_seen)
            picker_display = picker.name if picker and picker.name else (picker.face_id[:8] if picker else "unknown")

            logger.info(f"Package disappeared: {pkg_id[:8]} (confidence: {pkg.confidence:.2f}, picker: {picker_display})")
            self.events.send(PackageDisappearedEvent(
                plugin_name="security_camera",
                package_id=pkg_id[:8],
                confidence=pkg.confidence,
                first_seen=format_timestamp(pkg.first_seen),
                last_seen=format_timestamp(current_time),
                picker_face_id=picker.face_id if picker else None,
                picker_name=picker.name if picker else None,
            ))

    def _create_overlay(self, frame_bgr: np.ndarray, face_count: int, package_count: int) -> np.ndarray:
        """Create video overlay with counts, bounding boxes, and thumbnail grid."""
        height, width = frame_bgr.shape[:2]
        frame = frame_bgr.copy()
        frame_size = (height, width)

        # Draw bounding boxes for visible faces
        for face in self._detected_faces.values():
            if face.disappeared_at is None:
                draw_labeled_bbox(frame, face.bbox, face.name or face.face_id[:8], COLOR_GREEN, frame_size)

        # Draw bounding boxes for visible packages
        for pkg in self._detected_packages.values():
            if pkg.disappeared_at is None:
                draw_labeled_bbox(frame, pkg.bbox, f"Package {pkg.confidence:.2f}", COLOR_BLUE, frame_size)

        # Draw semi-transparent overlay panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (width - OVERLAY_WIDTH, 0), (width, height), COLOR_DARK_GRAY, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw header and counts
        base_x = width - OVERLAY_WIDTH + 10
        draw_text(frame, "SECURITY CAMERA", (base_x, 25), COLOR_WHITE, FONT_LARGE)

        visible_faces = sum(1 for f in self._detected_faces.values() if f.disappeared_at is None)
        visible_packages = sum(1 for p in self._detected_packages.values() if p.disappeared_at is None)

        draw_text(frame, f"Visitors: {visible_faces}/{face_count}", (base_x, 50), COLOR_GREEN, 0.45)
        draw_text(frame, f"Packages: {visible_packages}/{package_count}", (base_x, 70), COLOR_BLUE, 0.45)

        # Draw legend
        legend_y = 90
        cv2.rectangle(frame, (base_x, legend_y - 8), (base_x + 10, legend_y + 2), COLOR_GREEN, -1)
        draw_text(frame, "Person", (base_x + 15, legend_y), COLOR_GRAY, FONT_SMALL)
        cv2.rectangle(frame, (base_x + 70, legend_y - 8), (base_x + 80, legend_y + 2), COLOR_BLUE, -1)
        draw_text(frame, "Package", (base_x + 85, legend_y), COLOR_GRAY, FONT_SMALL)

        # Draw thumbnail grid
        self._draw_thumbnail_grid(frame, width, height)

        # Draw timestamp
        draw_text(frame, format_timestamp(time.time()), (10, height - 10), COLOR_WHITE, FONT_LARGE)

        return frame

    def _draw_thumbnail_grid(self, frame: np.ndarray, width: int, height: int) -> None:
        """Draw the thumbnail grid for detected faces and packages."""
        grid_start_y = 105
        grid_padding = 10
        thumb_size = self.thumbnail_size

        all_detections = []
        for face in self._detected_faces.values():
            all_detections.append({
                "type": "face",
                "image": face.face_image,
                "last_seen": face.last_seen,
                "detection_count": face.detection_count,
            })
        for pkg in self._detected_packages.values():
            all_detections.append({
                "type": "package",
                "image": pkg.package_image,
                "last_seen": pkg.last_seen,
                "detection_count": pkg.detection_count,
            })

        recent = sorted(all_detections, key=lambda d: d["last_seen"], reverse=True)[:MAX_THUMBNAILS]

        for idx, det in enumerate(recent):
            row, col = idx // GRID_COLS, idx % GRID_COLS
            x_pos = width - OVERLAY_WIDTH + 10 + col * (thumb_size + grid_padding)
            y_pos = grid_start_y + row * (thumb_size + grid_padding)

            if y_pos + thumb_size > height:
                break

            try:
                frame[y_pos:y_pos + thumb_size, x_pos:x_pos + thumb_size] = det["image"]
                border_color = COLOR_GREEN if det["type"] == "face" else COLOR_BLUE
                cv2.rectangle(frame, (x_pos, y_pos), (x_pos + thumb_size, y_pos + thumb_size), border_color, 2)

                if det["detection_count"] > 1:
                    badge_text = f"{det['detection_count']}x"
                    badge_size = cv2.getTextSize(badge_text, FONT, FONT_SMALL, 1)[0]
                    badge_x = x_pos + thumb_size - badge_size[0] - 2
                    badge_y = y_pos + thumb_size - 2
                    cv2.rectangle(frame, (badge_x - 2, badge_y - badge_size[1] - 2), (x_pos + thumb_size, y_pos + thumb_size), COLOR_BLACK, -1)
                    cv2.putText(frame, badge_text, (badge_x, badge_y), FONT, FONT_SMALL, COLOR_WHITE, 1, cv2.LINE_AA)
            except Exception as e:
                logger.debug(f"Failed to draw thumbnail: {e}")

    async def _process_and_add_frame(self, frame: av.VideoFrame) -> None:
        """Process a single video frame."""
        try:
            current_time = time.time()

            if self._shared_image is not None and current_time < self._shared_image_until:
                await self._video_track.add_frame(self._shared_image)
                return
            elif self._shared_image is not None:
                self._shared_image = None
                self._shared_image_until = 0.0
                logger.info("Shared image display ended, resuming camera feed")

            frame_rgb = frame.to_ndarray(format="rgb24")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            self._cleanup_old_faces(current_time)
            self._cleanup_old_packages(current_time)
            self._check_for_picked_up_packages(current_time)

            await self._detect_and_store_faces(frame_bgr, current_time)
            await self._update_face_bboxes_fast(frame_bgr, current_time)
            await self._detect_and_store_packages(frame_bgr, current_time)

            frame_with_overlay = self._create_overlay(frame_bgr, len(self._detected_faces), len(self._detected_packages))
            frame_rgb_overlay = cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB)
            processed_frame = av.VideoFrame.from_ndarray(frame_rgb_overlay, format="rgb24")

            await self._video_track.add_frame(processed_frame)

        except Exception as e:
            logger.exception(f"Frame processing failed: {e}")
            await self._video_track.add_frame(frame)

    async def process_video(
        self,
        track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """Set up video processing pipeline."""
        if shared_forwarder is not None:
            self._video_forwarder = shared_forwarder
            self._video_forwarder.add_frame_handler(self._process_and_add_frame, fps=float(self.fps), name="security_camera")
        else:
            self._video_forwarder = VideoForwarder(track, max_buffer=30, fps=self.fps, name="security_camera_forwarder")
            self._video_forwarder.add_frame_handler(self._process_and_add_frame)
        logger.info("Security camera video processing started")

    async def stop_processing(self) -> None:
        """Stop processing video tracks."""
        if self._video_forwarder:
            await self._video_forwarder.stop()

    def publish_video_track(self):
        """Return the video track for publishing."""
        return self._video_track

    def state(self) -> Dict[str, Any]:
        """Return current state for LLM context."""
        current_time = time.time()
        self._cleanup_old_faces(current_time)
        self._cleanup_old_packages(current_time)

        return {
            "unique_visitors": len(self._detected_faces),
            "currently_visible_visitors": sum(1 for f in self._detected_faces.values() if f.disappeared_at is None),
            "total_face_detections": sum(f.detection_count for f in self._detected_faces.values()),
            "unique_packages": len(self._detected_packages),
            "currently_visible_packages": sum(1 for p in self._detected_packages.values() if p.disappeared_at is None),
            "total_package_detections": sum(p.detection_count for p in self._detected_packages.values()),
            "time_window_minutes": self.time_window // 60,
            "last_face_detection_time": format_timestamp(self._last_detection_time) if self._last_detection_time > 0 else "No detections yet",
            "last_package_detection_time": format_timestamp(self._last_package_detection_time) if self._last_package_detection_time > 0 else "No detections yet",
        }

    def get_face_image(self, face_id: str) -> Optional[np.ndarray]:
        """Get the face image for a given face ID."""
        face = self._detected_faces.get(face_id)
        return face.face_image if face else None

    def share_image(self, image: bytes | np.ndarray, duration: float = 5.0) -> None:
        """Temporarily display an image in the video feed."""
        if isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img_bgr = image

        track_w, track_h = self._video_track.width, self._video_track.height
        h, w = img_bgr.shape[:2]
        scale = min(track_w / w, track_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((track_h, track_w, 3), dtype=np.uint8)
        x_offset, y_offset = (track_w - new_w) // 2, (track_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        self._shared_image = av.VideoFrame.from_ndarray(canvas_rgb, format="rgb24")
        self._shared_image_until = time.time() + duration

        logger.info(f"Sharing image in video feed for {duration}s")

    def get_visitor_count(self) -> int:
        """Get the current unique visitor count."""
        self._cleanup_old_faces(time.time())
        return len(self._detected_faces)

    def get_visitor_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all visitors."""
        self._cleanup_old_faces(time.time())
        return [
            {
                "face_id": f.face_id[:8],
                "name": f.name,
                "is_known": f.name is not None,
                "first_seen": format_timestamp(f.first_seen),
                "last_seen": format_timestamp(f.last_seen),
                "detection_count": f.detection_count,
            }
            for f in sorted(self._detected_faces.values(), key=lambda x: x.last_seen, reverse=True)
        ]

    def get_package_count(self) -> int:
        """Get the current unique package count."""
        self._cleanup_old_packages(time.time())
        return len(self._detected_packages)

    def get_package_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all packages."""
        self._cleanup_old_packages(time.time())
        return [
            {
                "package_id": p.package_id[:8],
                "first_seen": format_timestamp(p.first_seen),
                "last_seen": format_timestamp(p.last_seen),
                "detection_count": p.detection_count,
                "confidence": p.confidence,
            }
            for p in sorted(self._detected_packages.values(), key=lambda x: x.last_seen, reverse=True)
        ]

    def get_activity_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent activity log entries."""
        return [
            {
                "timestamp": format_timestamp(e.timestamp),
                "event_type": e.event_type,
                "description": e.description,
                "details": e.details,
            }
            for e in reversed(self._activity_log[-limit:])
        ]

    def register_known_face(self, name: str, face_encoding: np.ndarray) -> bool:
        """Register a face encoding with a name for future recognition."""
        self._known_faces[name] = KnownFace(name=name, face_encoding=face_encoding, registered_at=time.time())
        self._log_activity("face_registered", f"Registered: {name}", {"name": name})
        logger.info(f"Registered face: {name}")
        return True

    def register_current_face_as(self, name: str) -> Dict[str, Any]:
        """Register the most recently detected face with a name."""
        if not self._detected_faces:
            return {"success": False, "message": "No faces currently detected. Please make sure your face is visible."}

        most_recent = max(self._detected_faces.values(), key=lambda f: f.last_seen)
        self.register_known_face(name, most_recent.face_encoding)
        most_recent.name = name

        return {
            "success": True,
            "message": f"I'll remember you as {name}! Next time I see you, I'll recognize you.",
            "face_id": most_recent.face_id[:8],
        }

    def get_known_faces(self) -> List[Dict[str, Any]]:
        """Get list of all registered known faces."""
        return [{"name": f.name, "registered_at": format_timestamp(f.registered_at)} for f in self._known_faces.values()]

    async def close(self) -> None:
        """Clean up resources."""
        self._shutdown = True
        if self._video_forwarder is not None:
            await self._video_forwarder.stop()
        self.executor.shutdown(wait=False)
        self._detected_faces.clear()
        self._detected_packages.clear()
        logger.info("Security camera processor closed")
