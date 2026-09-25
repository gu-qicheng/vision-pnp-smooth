import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import pnp_square_demo as demo
from pnp_square_demo import (
    CalibrationData,
    CalibrationError,
    build_object_points,
    load_calibration,
    preview_to_image,
    solve_square_pose,
    CaptureState,
    DemoSession,
    PoseEstimate,
)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.camera_matrix = np.array(
            [[1211.60788, 0.0, 967.494893],
             [0.0, 1207.25088, 532.453665],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.dist_coeffs = np.array(
            [[0.09217848, -0.576192, -0.00258982, -0.00561771, 0.88987501]],
            dtype=np.float64,
        )
        self.calibration = CalibrationData(
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            image_size=(1920, 1080),
        )

    def test_object_points_are_centered_and_105mm(self):
        points = build_object_points()
        expected = np.array(
            [[-52.5, -52.5, 0.0], [52.5, -52.5, 0.0],
             [52.5, 52.5, 0.0], [-52.5, 52.5, 0.0]],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(points, expected)
        self.assertAlmostEqual(np.linalg.norm(points[1] - points[0]), 105.0)

    def test_preview_mapping_1280_to_1920(self):
        self.assertEqual(preview_to_image((0, 0)), (0.0, 0.0))
        self.assertEqual(preview_to_image((640, 360)), (960.0, 540.0))
        self.assertEqual(preview_to_image((1279, 719)), (1918.5, 1078.5))

    def test_synthetic_pose_round_trip(self):
        object_points = build_object_points()
        rvec = np.array([[0.08], [-0.12], [0.03]], dtype=np.float64)
        tvec = np.array([[25.0], [-15.0], [800.0]], dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            object_points, rvec, tvec,
            self.camera_matrix, self.dist_coeffs,
        )
        estimate = solve_square_pose(image_points.reshape(-1, 2), self.calibration)
        self.assertTrue(estimate.valid, estimate.message)
        self.assertIsNotNone(estimate.tvec)
        np.testing.assert_allclose(estimate.tvec.reshape(3), tvec.reshape(3), atol=1e-2)
        self.assertLess(estimate.reprojection_rms_px, 1e-5)

    def test_wrong_calibration_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'bad.npz'
            np.savez(
                path,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                image_size=np.array([1280, 720]),
            )
            with self.assertRaisesRegex(CalibrationError, '1920.*1080'):
                load_calibration(path)

    def test_valid_calibration_file_loads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'valid.npz'
            np.savez(
                path,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                image_size=np.array([1920, 1080]),
            )
            loaded = load_calibration(path)
            self.assertEqual(loaded.image_size, (1920, 1080))
            np.testing.assert_array_equal(loaded.camera_matrix, self.camera_matrix)

    def test_invalid_or_nonfinite_points_return_invalid_pose(self):
        points = np.array(
            [[100.0, 100.0], [200.0, 100.0], [np.nan, 200.0], [100.0, 200.0]],
            dtype=np.float64,
        )
        estimate = solve_square_pose(points, self.calibration)
        self.assertFalse(estimate.valid)
        self.assertIsNone(estimate.tvec)
        self.assertIn('finite', estimate.message.lower())

    def test_first_click_freezes_frame_and_reset_clears_state(self):
        session = DemoSession(self.calibration)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        session.latest_frame = frame
        session.click_preview_point((100, 100))
        self.assertEqual(session.state, CaptureState.SELECTING)
        self.assertIsNotNone(session.frozen_frame)
        np.testing.assert_array_equal(session.frozen_frame, frame)
        session.reset()
        self.assertEqual(session.state, CaptureState.LIVE)
        self.assertEqual(session.image_points, [])
        self.assertIsNone(session.frozen_frame)
        self.assertIsNone(session.pose)

    def test_invalid_pose_status_is_explicit(self):
        session = DemoSession(self.calibration)
        session.state = CaptureState.POSE_INVALID
        session.pose = PoseEstimate(
            valid=False,
            rvec=None,
            tvec=None,
            reprojection_rms_px=8.0,
            candidate_count=2,
            message='重投影误差过大',
        )
        self.assertIn('POSE INVALID', session.status_lines())

    def test_order_quad_points_keeps_four_distinct_points_when_rotated(self):
        # A 45-degree diamond used to make the sum/difference heuristic select
        # the same point for two labels.
        points = np.array(
            [[960.0, 220.0], [1320.0, 540.0], [960.0, 860.0], [600.0, 540.0]],
            dtype=np.float32,
        )
        ordered = demo.order_quad_points(points)
        self.assertEqual(ordered.shape, (4, 2))
        self.assertEqual(len(np.unique(ordered, axis=0)), 4)

    def test_order_quad_points_uses_leftmost_point_in_top_band_as_tl(self):
        points = np.array(
            [[1120.0, 278.0], [1122.0, 850.0], [470.0, 852.0], [468.0, 280.0]],
            dtype=np.float32,
        )
        ordered = demo.order_quad_points(points)
        np.testing.assert_array_equal(
            ordered,
            np.array(
                [[468.0, 280.0], [1120.0, 278.0], [1122.0, 850.0], [470.0, 852.0]],
                dtype=np.float32,
            ),
        )

    def test_process_auto_frame_handles_missing_red_frame(self):
        processor = getattr(demo, 'process_auto_frame', None)
        self.assertIsNotNone(processor)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = processor(frame, self.calibration)
        self.assertIsNone(result.corners)
        self.assertFalse(result.pose.valid)
        self.assertIsNone(result.camera_position_mm)

    def test_process_auto_frame_detects_red_quad_and_does_not_reuse_pose(self):
        processor = getattr(demo, 'process_auto_frame', None)
        self.assertIsNotNone(processor)
        red_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.rectangle(red_frame, (650, 300), (1250, 900), (0, 0, 255), 20)
        valid_result = processor(red_frame, self.calibration)
        self.assertIsNotNone(valid_result.corners)
        self.assertTrue(valid_result.pose.valid, valid_result.pose.message)

        blank_result = processor(
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            self.calibration,
        )
        self.assertIsNone(blank_result.corners)
        self.assertFalse(blank_result.pose.valid)
        self.assertIsNone(blank_result.camera_position_mm)

    def test_camera_position_is_inverse_target_pose(self):
        converter = getattr(demo, 'camera_position_in_target', None)
        self.assertIsNotNone(converter)
        pose = PoseEstimate(
            valid=True,
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.array([[25.0], [-15.0], [800.0]], dtype=np.float64),
            reprojection_rms_px=0.0,
            candidate_count=1,
            message='OK',
        )
        np.testing.assert_allclose(
            converter(pose),
            np.array([-25.0, 15.0, -800.0]),
            atol=1e-9,
        )

    def test_thin_pale_red_frame_beats_skin_colored_distractor(self):
        frame = np.full((1080, 1920, 3), 205, dtype=np.uint8)
        # Low-saturation red ink, similar to the supplied paper frame.
        cv2.rectangle(frame, (470, 280), (1120, 850), (145, 150, 180), 3)
        # Skin-colored filled region that satisfies the old HSV mask.
        cv2.rectangle(frame, (1320, 120), (1760, 850), (80, 85, 120), -1)

        corners, _ = demo.detect_red_frame(frame)
        self.assertIsNotNone(corners)
        center = np.asarray(corners).mean(axis=0)
        np.testing.assert_allclose(center, np.array([795.0, 565.0]), atol=35.0)

    def test_previous_corners_stabilize_detection_and_reject_large_jump(self):
        make_frame = lambda offset: self._make_thin_red_frame(offset)
        first, _ = demo.detect_red_frame(make_frame((0, 0)))
        self.assertIsNotNone(first)

        shifted, _ = demo.detect_red_frame(
            make_frame((4, 3)),
            previous_corners=first,
        )
        self.assertIsNotNone(shifted)
        self.assertLess(float(np.max(np.linalg.norm(shifted - first, axis=1))), 10.0)

        distractor = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.rectangle(distractor, (1350, 100), (1800, 900), (80, 85, 120), -1)
        rejected, _ = demo.detect_red_frame(
            distractor,
            previous_corners=first,
        )
        self.assertIsNone(rejected)

    @staticmethod
    def _make_thin_red_frame(offset=(0, 0)):
        dx, dy = offset
        frame = np.full((1080, 1920, 3), 205, dtype=np.uint8)
        cv2.rectangle(
            frame,
            (470 + dx, 280 + dy),
            (1120 + dx, 850 + dy),
            (145, 150, 180),
            3,
        )
        return frame

if __name__ == '__main__':
    unittest.main()





