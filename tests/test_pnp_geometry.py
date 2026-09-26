import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import pnp_square_demo as demo
from pnp_square_demo import (
    AutoTracker,
    CalibrationData,
    CalibrationError,
    build_object_points,
    load_calibration,
    solve_square_pose,
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

    def test_object_points_are_centered_and_110mm(self):
        points = build_object_points()
        expected = np.array(
            [[-55.0, -55.0, 0.0], [55.0, -55.0, 0.0],
             [55.0, 55.0, 0.0], [-55.0, 55.0, 0.0]],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(points, expected)
        self.assertAlmostEqual(np.linalg.norm(points[1] - points[0]), 110.0)

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

    def test_auto_tracker_handles_missing_red_frame(self):
        tracker = AutoTracker(self.calibration)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = tracker.update(frame)
        self.assertIsNone(result.corners)
        self.assertFalse(result.pose.valid)
        self.assertIsNone(result.camera_position_mm)

    def test_auto_tracker_detects_red_quad(self):
        tracker = AutoTracker(self.calibration)
        red_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.rectangle(red_frame, (650, 300), (1250, 900), (0, 0, 255), 20)
        valid_result = tracker.update(red_frame)
        self.assertIsNotNone(valid_result.corners)
        self.assertIsNotNone(tracker.raw_corners)
        self.assertTrue(valid_result.pose.valid, valid_result.pose.message)
        self.assertFalse(valid_result.tracked)

    def test_auto_tracker_tracks_three_color_misses_then_resets(self):
        tracker = AutoTracker(self.calibration)
        red_frame = self._make_projected_frame(0.0, (0, 0, 255))
        first = tracker.update(red_frame)
        self.assertTrue(first.pose.valid, first.pose.message)

        for miss in range(1, 4):
            tracked = tracker.update(
                self._make_projected_frame(miss * 8.0, (170, 170, 170))
            )
            self.assertIsNotNone(tracked.corners)
            self.assertTrue(tracked.tracked)
            self.assertEqual(tracked.tracking_misses, miss)

        lost = tracker.update(np.zeros((1080, 1920, 3), dtype=np.uint8))
        self.assertIsNone(lost.corners)
        self.assertFalse(lost.pose.valid)
        self.assertEqual(lost.tracking_misses, 0)
        self.assertIsNone(lost.camera_position_mm)

    def test_auto_tracker_clears_invalid_tracked_pose(self):
        tracker = AutoTracker(self.calibration)
        first = tracker.update(self._make_projected_frame(0.0))
        self.assertTrue(first.pose.valid, first.pose.message)
        invalid_pose = PoseEstimate(
            valid=False,
            rvec=None,
            tvec=None,
            reprojection_rms_px=6.0,
            candidate_count=1,
            message="重投影 RMS 超过阈值",
        )
        with patch.object(demo, "solve_square_pose", return_value=invalid_pose):
            result = tracker.update(
                self._make_projected_frame(8.0, (170, 170, 170))
            )
        self.assertIsNone(result.corners)
        self.assertFalse(result.pose.valid)
        self.assertIsNone(tracker.raw_corners)
        self.assertIsNone(tracker.corners)

    def test_auto_tracker_follows_moving_projected_110mm_frame(self):
        tracker = AutoTracker(self.calibration)
        centers = []
        for x in (0.0, 8.0, 16.0):
            result = tracker.update(self._make_projected_frame(x))
            self.assertTrue(result.pose.valid, result.pose.message)
            self.assertFalse(result.tracked)
            centers.append(result.corners.mean(axis=0))
        self.assertGreater(centers[1][0], centers[0][0])
        self.assertGreater(centers[2][0], centers[1][0])

    def test_auto_tracker_uses_velocity_prediction_for_fast_color_gap(self):
        tracker = AutoTracker(self.calibration)
        tracker.update(self._make_projected_frame(0.0))
        second = tracker.update(self._make_projected_frame(20.0))
        self.assertTrue(second.pose.valid, second.pose.message)

        tracked = tracker.update(
            self._make_projected_frame(80.0, (170, 170, 170))
        )
        self.assertTrue(tracked.tracked)
        self.assertTrue(tracked.pose.valid, tracked.pose.message)
        self.assertGreater(
            float(tracked.corners.mean(axis=0)[0]),
            float(second.corners.mean(axis=0)[0]) + 50.0,
        )

    def test_auto_tracker_uses_scaled_roi_for_far_frame(self):
        tracker = AutoTracker(self.calibration)
        near = tracker.update(self._make_projected_frame(0.0, z_mm=800.0))
        self.assertTrue(near.pose.valid, near.pose.message)

        far = tracker.update(self._make_projected_frame(0.0, z_mm=1800.0))
        self.assertIsNotNone(far.corners)
        self.assertTrue(far.pose.valid, far.pose.message)
        self.assertFalse(far.tracked)

    def test_auto_tracker_raises_smoothing_weight_for_fast_detection(self):
        tracker = AutoTracker(self.calibration)
        tracker.update(self._make_projected_frame(0.0))
        second = tracker.update(self._make_projected_frame(4.0))
        fast_frame = self._make_projected_frame(40.0)
        raw_corners, _ = demo.detect_red_frame(fast_frame)
        self.assertIsNotNone(raw_corners)

        fast = tracker.update(fast_frame)
        self.assertTrue(fast.pose.valid, fast.pose.message)
        raw_center = raw_corners.mean(axis=0)
        fast_center = fast.corners.mean(axis=0)
        previous_center = second.corners.mean(axis=0)
        self.assertLess(
            float(raw_center[0] - fast_center[0]),
            float((raw_center[0] - previous_center[0]) * 0.35),
        )

    def test_auto_tracker_does_not_follow_far_red_distractor(self):
        tracker = AutoTracker(self.calibration)
        first = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.rectangle(first, (650, 300), (1250, 900), (0, 0, 255), 20)
        self.assertTrue(tracker.update(first).pose.valid)

        distractor = np.zeros_like(first)
        cv2.rectangle(distractor, (1350, 100), (1800, 900), (0, 0, 255), 20)
        result = tracker.update(distractor)
        self.assertIsNone(result.corners)
        self.assertFalse(result.pose.valid)

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

    def test_fast_translation_and_reversal_keep_current_target(self):
        tracker = AutoTracker(self.calibration)
        for x in (0.0, 100.0, 200.0, 100.0, 0.0):
            result = tracker.update(self._make_projected_frame(x))
            self.assertTrue(result.pose.valid, (x, result.pose.message))
            # The raw tracker must stay on the current image, not lag with EMA.
            expected, _ = cv2.projectPoints(
                build_object_points(), np.array([0.08, -0.12, 0.03]),
                np.array([x, -15.0, 800.0]), self.camera_matrix, self.dist_coeffs,
            )
            np.testing.assert_allclose(
                tracker.raw_corners.mean(axis=0), expected.reshape(4, 2).mean(axis=0),
                atol=3.0,
            )

    def test_prediction_stays_one_frame_ahead_during_color_gaps(self):
        tracker = AutoTracker(self.calibration)
        for index, x in enumerate((0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0)):
            color = (170, 170, 170) if 3 <= index <= 5 else (0, 0, 255)
            result = tracker.update(self._make_projected_frame(x, color))
            self.assertTrue(result.pose.valid, (index, result.pose.message))
            if 3 <= index <= 5:
                self.assertTrue(result.tracked)
                expected, _ = cv2.projectPoints(
                    build_object_points(), np.array([0.08, -0.12, 0.03]),
                    np.array([x + 20.0, -15.0, 800.0]),
                    self.camera_matrix, self.dist_coeffs,
                )
                np.testing.assert_allclose(
                    tracker._predicted_corners().mean(axis=0),
                    expected.reshape(4, 2).mean(axis=0), atol=5.0,
                )

    def test_solid_skin_and_red_quads_never_initialize_pose(self):
        points = np.array([[886, 423], [1053, 429], [1046, 594], [882, 591]])
        for color in ((70, 95, 170), (30, 50, 120), (0, 0, 255)):
            with self.subTest(color=color):
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                cv2.fillConvexPoly(frame, points, color)
                result = AutoTracker(self.calibration).update(frame)
                self.assertIsNone(result.corners)
                self.assertFalse(result.pose.valid)

    def test_flow_rejects_four_consistent_but_unsupported_corners(self):
        previous = np.array([[500, 300], [1100, 300], [1100, 900], [500, 900]],
                            dtype=np.float32).reshape(4, 1, 2)
        wrong = np.array([[540, 330], [1140, 330], [1160, 430], [540, 930]],
                         dtype=np.float32).reshape(4, 1, 2)
        gray = np.zeros((1080, 1920), dtype=np.uint8)
        cv2.polylines(gray, [previous.astype(np.int32)], True, 170, 8)
        status = np.ones((4, 1), dtype=np.uint8)
        # Simulate a reversible LK mismatch. Real geometry/image checks must reject it.
        with patch.object(cv2, 'calcOpticalFlowPyrLK', side_effect=[
            (wrong.copy(), status, None), (previous.copy(), status, None),
        ]):
            self.assertIsNone(demo._track_corners(gray, gray, previous, previous.copy()))

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

    def _make_projected_frame(self, x_mm, color=(0, 0, 255), z_mm=800.0):
        rvec = np.array([[0.08], [-0.12], [0.03]], dtype=np.float64)
        tvec = np.array([[x_mm], [-15.0], [z_mm]], dtype=np.float64)
        points, _ = cv2.projectPoints(
            build_object_points(),
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.polylines(
            frame,
            [np.rint(points.reshape(-1, 2)).astype(np.int32)],
            True,
            color,
            8,
        )
        return frame

if __name__ == '__main__':
    unittest.main()





