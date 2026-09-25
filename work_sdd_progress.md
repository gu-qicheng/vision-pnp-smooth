# SDD ledger — plan: approved inline 110 mm square manual PnP implementation plan

Pre-flight: target directory is not a Git repository; user explicitly required direct files under C:\Users\ASUS\Desktop\视觉pnp自测, so no additional worktree or repository initialization was created. Cost if wrong: no branch-level rollback history for these local deliverables.

Shared interfaces:
- Task 1 -> Task 2: CalibrationData/load_calibration must expose finite 3x3 camera_matrix, dist_coeffs, and image_size=(1920,1080).
- Task 2 -> Task 3: build_object_points, preview_to_image, solve_square_pose and PoseEstimate are consumed by the window state machine.
- Task 3 -> Task 4: run_demo and keyboard/visual behavior are documented and manually verified.

Ruling: user changed target side length from 100 mm to 110 mm — update object points, tests, documentation, and UI constants to preserve the requested physical scale — cost if wrong: reported Z scale would be systematically wrong by 10 percent.


Task 1: complete — requirements, setup script, batch launcher, calibration loader, and actual camera_calibration.npz load verified (image_size=(1920,1080)).
Task 2: complete — 8 unittest cases passed; synthetic PnP round-trip, 110 mm object points, preview mapping, calibration validation, and invalid finite-point handling verified.
Task 3: complete with hardware-click QA unverified — DemoSession freeze/reset/invalid-state tests passed; synthetic four-click render produced pose_valid and 1280x720 output; camera probe opened device 0 and read 1920x1080. CUA could not bind the native OpenCV window for physical clicks.
Task 4: complete — README/spec/launchers written; py_compile and CLI help passed.
Final review: self-review after independent read-only preliminary review. No Critical or Important defect found. Minor/deferred: no dedicated automated test for RMS-above-threshold candidate filtering or runtime resolution-change branch; behavior is implemented but not separately regression-pinned.
