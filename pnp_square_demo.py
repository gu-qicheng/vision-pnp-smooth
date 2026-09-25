from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


TARGET_SIDE_MM = 110.0
PREVIEW_SIZE = (1280, 720)
EXPECTED_IMAGE_SIZE = (1920, 1080)
MAX_REPROJECTION_RMS_PX = 5.0
AXIS_LENGTH_MM = 55.0
WINDOW_NAME = "110 mm Square PnP"
POINT_LABELS = ("TL", "TR", "BR", "BL")
# =========================
# 红色门框自动检测参数
# =========================

RED_LOWER_1 = np.array([0, 20, 50], dtype=np.uint8)
RED_UPPER_1 = np.array([15, 255, 255], dtype=np.uint8)

RED_LOWER_2 = np.array([165, 20, 50], dtype=np.uint8)
RED_UPPER_2 = np.array([179, 255, 255], dtype=np.uint8)

# 颜色阈值只负责产生候选；轮廓几何和跨帧约束负责排除皮肤误检。
MIN_RED_EXCESS = 18
RED_CLOSE_KERNEL_SIZE = 5
MIN_FRAME_AREA_PX = 5000
MIN_QUAD_AREA_FRACTION = 0.005
MIN_QUAD_RECTANGULARITY = 0.55
MAX_QUAD_ASPECT_RATIO = 3.5
MAX_QUAD_ANGLE_COSINE = 0.65
MAX_BORDER_FILL_RATIO = 0.35
MIN_BORDER_EDGE_FRACTION = 0.45
MIN_SOLID_RED_FILL_RATIO = 0.35
MIN_SOLID_RED_EXCESS = 70.0
MIN_SOLID_RED_SATURATION = 90.0
CORNER_SMOOTH_ALPHA = 0.35
MIN_REFERENCE_SIDE_PX = 40.0
MAX_CORNER_JUMP_RATIO = 0.45

class CalibrationError(ValueError):
    """Raised when a calibration archive is missing or malformed."""


@dataclass(frozen=True)
class CalibrationData:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]


@dataclass(frozen=True)
class PoseEstimate:
    valid: bool
    rvec: Optional[np.ndarray]
    tvec: Optional[np.ndarray]
    reprojection_rms_px: Optional[float]
    candidate_count: int
    message: str


@dataclass(frozen=True)
class AutoFrameResult:
    """Detection and pose result for one camera frame."""

    corners: Optional[np.ndarray]
    mask: np.ndarray
    pose: PoseEstimate
    camera_position_mm: Optional[np.ndarray]


class CaptureState(Enum):
    LIVE = "live"
    SELECTING = "selecting"
    POSE_VALID = "pose_valid"
    POSE_INVALID = "pose_invalid"


def _invalid_pose(message: str, candidate_count: int = 0) -> PoseEstimate:
    return PoseEstimate(
        valid=False,
        rvec=None,
        tvec=None,
        reprojection_rms_px=None,
        candidate_count=candidate_count,
        message=message,
    )


def load_calibration(path: Path) -> CalibrationData:
    """Load and validate the calibration archive used by the demo."""
    path = Path(path)
    if not path.is_file():
        raise CalibrationError(f"标定文件不存在：{path}")

    required = {"camera_matrix", "dist_coeffs", "image_size"}
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = required.difference(archive.files)
            if missing:
                names = ", ".join(sorted(missing))
                raise CalibrationError(f"标定文件缺少字段：{names}")
            camera_matrix = np.asarray(archive["camera_matrix"], dtype=np.float64)
            dist_coeffs = np.asarray(archive["dist_coeffs"], dtype=np.float64)
            image_size_raw = np.asarray(archive["image_size"])
    except CalibrationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise CalibrationError(f"无法读取标定文件：{exc}") from exc

    if camera_matrix.shape != (3, 3):
        raise CalibrationError(
            f"camera_matrix 必须是 3×3，实际为 {camera_matrix.shape}"
        )
    if dist_coeffs.size < 4:
        raise CalibrationError(
            f"dist_coeffs 至少需要 4 个参数，实际为 {dist_coeffs.size}"
        )
    if not np.all(np.isfinite(camera_matrix)):
        raise CalibrationError("camera_matrix 含有非有限数值")
    if not np.all(np.isfinite(dist_coeffs)):
        raise CalibrationError("dist_coeffs 含有非有限数值")
    if not np.isfinite(camera_matrix[2, 2]) or camera_matrix[2, 2] == 0:
        raise CalibrationError("camera_matrix[2,2] 必须是非零有限数值")

    image_size_values = image_size_raw.reshape(-1)
    if image_size_values.size != 2:
        raise CalibrationError(
            f"image_size 必须包含宽和高，实际为 {image_size_raw.shape}"
        )
    try:
        image_size_float = image_size_values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError("image_size 必须是数值") from exc
    if not np.all(np.isfinite(image_size_float)):
        raise CalibrationError("image_size 含有非有限数值")
    if not np.all(image_size_float == np.rint(image_size_float)):
        raise CalibrationError("image_size 必须是整数宽高")
    image_size = tuple(int(value) for value in image_size_float)
    if image_size != EXPECTED_IMAGE_SIZE:
        raise CalibrationError(
            f"标定尺寸必须为 {EXPECTED_IMAGE_SIZE[0]}×{EXPECTED_IMAGE_SIZE[1]}，"
            f"实际为 {image_size[0]}×{image_size[1]}"
        )

    return CalibrationData(
        camera_matrix=camera_matrix.copy(),
        dist_coeffs=dist_coeffs.copy(),
        image_size=image_size,
    )

def order_quad_points(points: np.ndarray) -> np.ndarray:
    """
    将任意顺序的四个二维点排序为：

    0: TL 左上
    1: TR 右上
    2: BR 右下
    3: BL 左下
    """

    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.all(np.isfinite(pts)):
        raise ValueError("四边形角点必须是 finite 数值")
    if len(np.unique(pts, axis=0)) != 4:
        raise ValueError("四边形角点必须互不相同")

    # 按四点绕质心的角度排序，避免 x+y / y-x 在旋转约 45° 时重复选点。
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    # 图像坐标 y 向下；先取顶部的一小段，再在这段中选择最左点。
    # 这样可以避免透视或轮廓像素误差让右上角比左上角低 1~几像素，
    # 从而错误地把右上角作为 TL。
    y_values = ordered[:, 1]
    top_tolerance = max(2.0, 0.10 * float(y_values.max() - y_values.min()))
    top_indices = np.flatnonzero(y_values <= y_values.min() + top_tolerance)
    start = int(top_indices[np.argmin(ordered[top_indices, 0])])
    return np.roll(ordered, -start, axis=0).astype(np.float32)

def build_object_points(side_mm: float = TARGET_SIDE_MM) -> np.ndarray:
    """Return TL/TR/BR/BL points around the square centre in millimetres."""
    side_mm = float(side_mm)
    if not np.isfinite(side_mm) or side_mm <= 0:
        raise ValueError("正方形边长必须是正的有限数值")
    half = side_mm / 2.0
    return np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float64,
    )


def preview_to_image(
    point_xy: tuple[int, int],
    preview_size: tuple[int, int] = PREVIEW_SIZE,
    image_size: tuple[int, int] = EXPECTED_IMAGE_SIZE,
) -> tuple[float, float]:
    """Map a point in the preview to the corresponding raw-image coordinate."""
    preview_width, preview_height = preview_size
    image_width, image_height = image_size
    if preview_width <= 0 or preview_height <= 0:
        raise ValueError("预览尺寸必须为正")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图像尺寸必须为正")
    x, y = point_xy
    return (
        float(x) * float(image_width) / float(preview_width),
        float(y) * float(image_height) / float(preview_height),
    )

def _align_quad_to_reference(
    corners: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Align a cyclicly ordered quadrilateral to the previous frame."""

    current = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    previous = np.asarray(reference, dtype=np.float32).reshape(4, 2)
    best: tuple[float, np.ndarray] | None = None
    for shift in range(4):
        candidate = np.roll(current, shift, axis=0)
        distance = float(np.mean(np.linalg.norm(candidate - previous, axis=1)))
        if best is None or distance < best[0]:
            best = (distance, candidate)
    assert best is not None
    return best[1].copy(), best[0]


def _reference_jump_limit(reference: np.ndarray) -> float:
    """Return a motion gate scaled to the previous quadrilateral."""

    points = np.asarray(reference, dtype=np.float32).reshape(4, 2)
    side_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    reference_side = float(np.median(side_lengths))
    return max(MIN_REFERENCE_SIDE_PX, reference_side * MAX_CORNER_JUMP_RATIO)


def _quad_candidate(
    contour: np.ndarray,
    mask: np.ndarray,
    hsv: np.ndarray,
    red_excess: np.ndarray,
    frame_area: int,
) -> tuple[float, np.ndarray] | None:
    """Validate one contour and return its quality score and ordered corners."""

    area = float(cv2.contourArea(contour))
    if area < max(MIN_FRAME_AREA_PX, frame_area * MIN_QUAD_AREA_FRACTION):
        return None

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None
    try:
        corners = order_quad_points(approx.reshape(4, 2))
    except ValueError:
        return None

    x, y, width, height = cv2.boundingRect(approx)
    if width < 20 or height < 20:
        return None
    aspect_ratio = max(width / float(height), height / float(width))
    if aspect_ratio > MAX_QUAD_ASPECT_RATIO:
        return None
    rectangularity = area / float(width * height)
    if rectangularity < MIN_QUAD_RECTANGULARITY:
        return None

    max_angle_cosine = 0.0
    for index in range(4):
        previous = corners[index - 1] - corners[index]
        following = corners[(index + 1) % 4] - corners[index]
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
        if denominator <= 1e-6:
            return None
        cosine = abs(float(np.dot(previous, following)) / denominator)
        max_angle_cosine = max(max_angle_cosine, cosine)
    if max_angle_cosine > MAX_QUAD_ANGLE_COSINE:
        return None

    polygon = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(polygon, np.rint(corners).astype(np.int32), 255)
    colored = cv2.bitwise_and(mask, polygon)
    colored_count = cv2.countNonZero(colored)
    polygon_count = cv2.countNonZero(polygon)
    if colored_count == 0 or polygon_count == 0:
        return None

    fill_ratio = colored_count / float(polygon_count)
    edge_width = max(2, min(15, int(round(min(width, height) * 0.04))))
    erosion_kernel = np.ones((edge_width * 2 + 1, edge_width * 2 + 1), dtype=np.uint8)
    edge_band = cv2.subtract(polygon, cv2.erode(polygon, erosion_kernel))
    edge_count = cv2.countNonZero(cv2.bitwise_and(mask, edge_band))
    edge_fraction = edge_count / float(colored_count)

    selected = cv2.bitwise_and(mask, polygon) > 0
    mean_excess = float(np.mean(red_excess[selected]))
    mean_saturation = float(np.mean(hsv[:, :, 1][selected]))
    border_like = (
        fill_ratio <= MAX_BORDER_FILL_RATIO
        and edge_fraction >= MIN_BORDER_EDGE_FRACTION
    )
    solid_like = (
        fill_ratio >= MIN_SOLID_RED_FILL_RATIO
        and mean_excess >= MIN_SOLID_RED_EXCESS
        and mean_saturation >= MIN_SOLID_RED_SATURATION
    )
    if not border_like and not solid_like:
        return None

    aspect_score = 1.0 - min(1.0, max(0.0, aspect_ratio - 1.0) / 2.5)
    angle_score = 1.0 - max_angle_cosine / MAX_QUAD_ANGLE_COSINE
    color_score = min(1.0, mean_excess / MIN_SOLID_RED_EXCESS)
    edge_score = edge_fraction if border_like else fill_ratio
    score = (
        rectangularity
        * max(0.0, aspect_score)
        * max(0.0, angle_score)
        * (0.6 + 0.25 * color_score + 0.15 * edge_score)
    )
    return float(score), corners


def detect_red_frame(
    frame: np.ndarray,
    previous_corners: Optional[np.ndarray] = None,
):
    """
    从一帧图像中检测红色方形门框。

    返回：
        corners:
            shape=(4,2)
            顺序为 TL, TR, BR, BL
            如果没有找到则为 None

        mask:
            红色区域的二值图
    """

    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("输入帧必须是 BGR 三通道图像")

    # 1. BGR -> HSV，并增加红色通道优势约束。
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(frame)
    red_excess = red.astype(np.int16) - np.maximum(
        green.astype(np.int16), blue.astype(np.int16)
    )
    red_excess = np.maximum(red_excess, 0).astype(np.uint8)

    # 2. 提取两段红色
    mask1 = cv2.inRange(
        hsv,
        RED_LOWER_1,
        RED_UPPER_1,
    )

    mask2 = cv2.inRange(
        hsv,
        RED_LOWER_2,
        RED_UPPER_2,
    )

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_and(
        mask,
        (red_excess >= MIN_RED_EXCESS).astype(np.uint8) * 255,
    )

    # 3. 只做闭运算连接细边框，不能使用开运算，否则会抹掉 1~3 px 的线。
    kernel = np.ones(
        (RED_CLOSE_KERNEL_SIZE, RED_CLOSE_KERNEL_SIZE),
        dtype=np.uint8,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # 4. 查找外轮廓
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None, mask

    candidates: list[tuple[float, np.ndarray]] = []
    frame_area = int(frame.shape[0] * frame.shape[1])
    for contour in contours:
        candidate = _quad_candidate(
            contour,
            mask,
            hsv,
            red_excess,
            frame_area,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None, mask

    if previous_corners is None:
        _, best_corners = max(candidates, key=lambda item: item[0])
        return best_corners, mask

    reference = np.asarray(previous_corners, dtype=np.float32).reshape(4, 2)
    jump_limit = _reference_jump_limit(reference)
    eligible: list[tuple[float, float, np.ndarray]] = []
    for score, candidate_corners in candidates:
        aligned, distance = _align_quad_to_reference(candidate_corners, reference)
        if distance <= jump_limit:
            temporal_score = score + 0.5 * (1.0 - distance / jump_limit)
            eligible.append((temporal_score, distance, aligned))
    if not eligible:
        return None, mask
    _, _, best_corners = max(eligible, key=lambda item: item[0])
    return best_corners, mask

def solve_square_pose(
    image_points: np.ndarray,
    calibration: CalibrationData,
) -> PoseEstimate:
    """Estimate target-to-camera pose from four ordered square corners."""
    points = np.asarray(image_points, dtype=np.float64)
    if points.shape != (4, 2):
        return _invalid_pose(f"需要 4 个二维点，实际形状为 {points.shape}")
    if not np.all(np.isfinite(points)):
        return _invalid_pose("图像点必须全部是 finite 数值")
    if calibration.image_size != EXPECTED_IMAGE_SIZE:
        return _invalid_pose(
            f"标定尺寸必须为 {EXPECTED_IMAGE_SIZE[0]}×{EXPECTED_IMAGE_SIZE[1]}"
        )

    object_points = build_object_points()
    try:
        result = cv2.solvePnPGeneric(
            object_points,
            points.reshape(-1, 1, 2),
            calibration.camera_matrix,
            calibration.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error as exc:
        return _invalid_pose(f"IPPE 求解失败：{exc}")

    if len(result) < 3:
        return _invalid_pose("OpenCV 没有返回完整的 PnP 结果")
    retval, rvecs, tvecs = result[:3]
    candidate_count = len(rvecs) if rvecs is not None else 0
    if not retval or candidate_count == 0:
        return _invalid_pose("没有可用的 PnP 候选解", candidate_count)

    valid_candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for rvec_raw, tvec_raw in zip(rvecs, tvecs):
        rvec = np.asarray(rvec_raw, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec_raw, dtype=np.float64).reshape(3, 1)
        if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue
        try:
            rotation, _ = cv2.Rodrigues(rvec)
            camera_points = (rotation @ object_points.T + tvec).T
            if not np.all(np.isfinite(camera_points)) or np.any(camera_points[:, 2] <= 0):
                continue
            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                calibration.camera_matrix,
                calibration.dist_coeffs,
            )
        except cv2.error:
            continue
        projected = projected.reshape(-1, 2)
        delta = projected - points
        rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
        if np.isfinite(rms):
            valid_candidates.append((rms, rvec, tvec))

    if not valid_candidates:
        return _invalid_pose("没有所有点均为正深度的 PnP 解", candidate_count)

    rms, rvec, tvec = min(valid_candidates, key=lambda candidate: candidate[0])
    if rms > MAX_REPROJECTION_RMS_PX:
        return PoseEstimate(
            valid=False,
            rvec=None,
            tvec=None,
            reprojection_rms_px=rms,
            candidate_count=candidate_count,
            message=(
                f"重投影 RMS {rms:.2f} px 超过阈值 "
                f"{MAX_REPROJECTION_RMS_PX:.2f} px"
            ),
        )
    return PoseEstimate(
        valid=True,
        rvec=rvec.copy(),
        tvec=tvec.copy(),
        reprojection_rms_px=rms,
        candidate_count=candidate_count,
        message="OK",
    )


def camera_position_in_target(pose: PoseEstimate) -> Optional[np.ndarray]:
    """Return the camera origin in the target frame, in millimetres.

    ``solvePnP`` returns the target-to-camera transform:
    ``X_camera = R @ X_target + t``.  Therefore the camera origin expressed in
    target coordinates is ``-R.T @ t``; ``tvec`` itself is the target origin in
    camera coordinates.
    """

    if not pose.valid or pose.rvec is None or pose.tvec is None:
        return None
    try:
        rotation, _ = cv2.Rodrigues(pose.rvec)
    except cv2.error:
        return None
    position = -rotation.T @ np.asarray(pose.tvec, dtype=np.float64).reshape(3, 1)
    position = position.reshape(3)
    if not np.all(np.isfinite(position)):
        return None
    return position


def process_auto_frame(
    frame: np.ndarray,
    calibration: CalibrationData,
    previous_corners: Optional[np.ndarray] = None,
) -> AutoFrameResult:
    """Run red-frame detection and PnP once for one camera frame."""

    corners, mask = detect_red_frame(frame, previous_corners=previous_corners)
    if corners is None:
        return AutoFrameResult(
            corners=None,
            mask=mask,
            pose=_invalid_pose("未检测到红色方框"),
            camera_position_mm=None,
        )

    if previous_corners is not None:
        reference = np.asarray(previous_corners, dtype=np.float32).reshape(4, 2)
        corners, distance = _align_quad_to_reference(corners, reference)
        if distance > _reference_jump_limit(reference):
            return AutoFrameResult(
                corners=None,
                mask=mask,
                pose=_invalid_pose("检测结果发生跳变，等待重新锁定"),
                camera_position_mm=None,
            )
        corners = (
            (1.0 - CORNER_SMOOTH_ALPHA) * reference
            + CORNER_SMOOTH_ALPHA * corners
        ).astype(np.float32)

    pose = solve_square_pose(corners, calibration)
    return AutoFrameResult(
        corners=corners.copy(),
        mask=mask,
        pose=pose,
        camera_position_mm=camera_position_in_target(pose),
    )


def draw_auto_geometry(
    frame: np.ndarray,
    result: AutoFrameResult,
    calibration: CalibrationData,
) -> np.ndarray:
    """Draw the current automatic detection and valid PnP axes."""

    canvas = frame.copy()
    if result.corners is None:
        return canvas

    integer_corners = np.rint(result.corners).astype(np.int32)
    cv2.polylines(
        canvas,
        [integer_corners.reshape(-1, 1, 2)],
        True,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )
    for index, (x, y) in enumerate(integer_corners):
        cv2.circle(canvas, (int(x), int(y)), 10, (0, 255, 255), -1)
        cv2.putText(
            canvas,
            POINT_LABELS[index],
            (int(x) + 12, int(y) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if result.pose.valid and result.pose.rvec is not None and result.pose.tvec is not None:
        cv2.drawFrameAxes(
            canvas,
            calibration.camera_matrix,
            calibration.dist_coeffs,
            result.pose.rvec,
            result.pose.tvec,
            AXIS_LENGTH_MM,
            3,
        )
    return canvas


def auto_status_lines(result: AutoFrameResult) -> list[str]:
    """Build status text for one automatic frame result."""

    lines = ["Automatic red-frame detection | Esc: exit", result.pose.message]
    if result.pose.reprojection_rms_px is not None:
        lines.append(f"RMS={result.pose.reprojection_rms_px:.2f}px")
    lines.append(f"Candidates={result.pose.candidate_count}")
    if result.pose.valid and result.pose.tvec is not None:
        x, y, z = result.pose.tvec.reshape(3)
        lines.append(
            f"target_center_in_camera_mm: X={x:.1f} Y={y:.1f} Z={z:.1f}"
        )
    if result.camera_position_mm is not None:
        x, y, z = result.camera_position_mm
        lines.append(f"camera_in_target_mm: X={x:.1f} Y={y:.1f} Z={z:.1f}")
    return lines


@dataclass
class DemoSession:
    calibration: CalibrationData
    state: CaptureState = CaptureState.LIVE
    image_points: list[tuple[float, float]] | None = None
    pose: PoseEstimate | None = None
    frozen_frame: np.ndarray | None = None
    latest_frame: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.image_points is None:
            self.image_points = []

    def reset(self) -> None:
        self.state = CaptureState.LIVE
        self.image_points = []
        self.pose = None
        self.frozen_frame = None

    def click_preview_point(self, preview_point: tuple[int, int]) -> None:
        if self.latest_frame is None or self.state in {
            CaptureState.POSE_VALID,
            CaptureState.POSE_INVALID,
        }:
            return
        if len(self.image_points) == 0:
            self.frozen_frame = self.latest_frame.copy()
        if len(self.image_points) >= 4:
            return
        self.image_points.append(preview_to_image(preview_point))
        if len(self.image_points) < 4:
            self.state = CaptureState.SELECTING
            return
        self.pose = solve_square_pose(np.asarray(self.image_points), self.calibration)
        if self.pose.valid:
            self.state = CaptureState.POSE_VALID
            self.frozen_frame = None
        else:
            self.state = CaptureState.POSE_INVALID

    def status_lines(self) -> list[str]:
        lines = [
            "L-click: TL -> TR -> BR -> BL | R: reset | Esc: exit",
            f"State: {self.state.value}",
        ]
        if self.state == CaptureState.LIVE:
            lines.append("Click TL to freeze the current frame")
        elif self.state == CaptureState.SELECTING:
            lines.append(f"Selected {len(self.image_points)}/4 corners")
        elif self.state == CaptureState.POSE_INVALID:
            lines.append("POSE INVALID")

        if self.pose is not None:
            rms_text = (
                f"RMS={self.pose.reprojection_rms_px:.2f}px"
                if self.pose.reprojection_rms_px is not None
                else "RMS=n/a"
            )
            lines.append(self.pose.message)
            lines.append(rms_text)
            lines.append(f"Candidates={self.pose.candidate_count}")
            if self.pose.valid and self.pose.tvec is not None:
                x, y, z = self.pose.tvec.reshape(3)
                lines.append(
                    f"target_center_in_camera_mm: X={x:.1f} Y={y:.1f} Z={z:.1f}"
                )
        return lines

    def render(self, live_frame: np.ndarray) -> np.ndarray:
        if self.state in {CaptureState.SELECTING, CaptureState.POSE_INVALID}:
            source = self.frozen_frame if self.frozen_frame is not None else live_frame
        else:
            source = live_frame
        canvas = source.copy()

        if self.image_points:
            raw_points = np.asarray(self.image_points, dtype=np.float64)
            integer_points = np.rint(raw_points).astype(np.int32)
            for index, (x, y) in enumerate(integer_points):
                cv2.circle(canvas, (int(x), int(y)), 8, (0, 255, 255), -1)
                cv2.putText(
                    canvas,
                    POINT_LABELS[index],
                    (int(x) + 12, int(y) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if len(integer_points) >= 2:
                cv2.polylines(
                    canvas,
                    [integer_points.reshape(-1, 1, 2)],
                    isClosed=len(integer_points) == 4,
                    color=(0, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

        if self.state == CaptureState.POSE_VALID and self.pose is not None:
            cv2.drawFrameAxes(
                canvas,
                self.calibration.camera_matrix,
                self.calibration.dist_coeffs,
                self.pose.rvec,
                self.pose.tvec,
                AXIS_LENGTH_MM,
                3,
            )

        preview = cv2.resize(canvas, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
        status_lines = self.status_lines()

        for index, line in enumerate(status_lines):
            cv2.putText(
                preview,
                line,
                (20, 32 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if self.state != CaptureState.POSE_INVALID else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return preview


def _open_camera(camera_index: int) -> cv2.VideoCapture:
    backend = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
    capture = cv2.VideoCapture(camera_index, backend)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开摄像头设备 {camera_index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, EXPECTED_IMAGE_SIZE[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, EXPECTED_IMAGE_SIZE[1])
    return capture


def _mouse_callback(event: int, x: int, y: int, _flags: int, session: DemoSession) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        session.click_preview_point((x, y))


def run_demo(calibration_path: Path, camera_index: int = 0) -> None:
    calibration = load_calibration(calibration_path)
    capture = _open_camera(camera_index)
    previous_corners: Optional[np.ndarray] = None
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Red Mask", cv2.WINDOW_AUTOSIZE)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("摄像头读取失败")
            if (int(frame.shape[1]), int(frame.shape[0])) != EXPECTED_IMAGE_SIZE:
                raise RuntimeError("摄像头分辨率在运行中发生变化")

            result = process_auto_frame(
                frame,
                calibration,
                previous_corners=previous_corners,
            )
            previous_corners = (
                result.corners.copy() if result.corners is not None else None
            )
            debug_frame = draw_auto_geometry(frame, result, calibration)
            preview = cv2.resize(debug_frame, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
            status_color = (0, 255, 0) if result.pose.valid else (0, 0, 255)
            for index, line in enumerate(auto_status_lines(result)):
                cv2.putText(
                    preview,
                    line,
                    (20, 32 + index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, preview)
            cv2.imshow(
                "Red Mask",
                cv2.resize(
                    result.mask,
                    (640, 360),
                    interpolation=cv2.INTER_NEAREST,
                ),
            )
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automatic red-frame PnP demo")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(__file__).resolve().with_name("camera_calibration.npz"),
    )
    parser.add_argument("--camera-index", type=int, default=0)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        run_demo(args.calibration, args.camera_index)
    except (CalibrationError, RuntimeError, cv2.error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
