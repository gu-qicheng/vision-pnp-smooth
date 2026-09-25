from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 1. 棋盘格参数
# ============================================================

CHECKERBOARD = (9, 5)       # 内角点数量：横9，竖5
SQUARE_SIZE = 22.0          # 每个小格边长，单位 mm

# 如果你的棋盘实际上是 9×6 个内角点：
# CHECKERBOARD = (9, 6)


# ============================================================
# 2. 文件路径
# ============================================================

IMAGE_DIR = Path(
    r"C:\Users\ASUS\Desktop\视觉pnp自测\标定图片"
)

OUTPUT_FILE = Path(
    r"C:\Users\ASUS\Desktop\视觉pnp自测\camera_calibration.npz"
)


# ============================================================
# 3. 亚像素优化终止条件
# ============================================================

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


# ============================================================
# 4. 构造棋盘格真实 3D 坐标
# ============================================================

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

# 例如：
# (0, 0, 0)
# (22, 0, 0)
# (44, 0, 0)
# ...
# 因为棋盘是平面，所以所有点 Z = 0


# ============================================================
# 5. 保存标定数据
# ============================================================

objpoints = []      # 世界坐标中的 3D 点
imgpoints = []      # 图像中的 2D 像素点


# ============================================================
# 6. 解决 Windows 中文路径读取问题
# ============================================================

def imread_unicode(path: Path):
    """
    比 cv2.imread() 对 Windows 中文路径更加稳定。
    """

    data = np.fromfile(
        str(path),
        dtype=np.uint8
    )

    if data.size == 0:
        return None

    return cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )


# ============================================================
# 7. 检查图片文件夹
# ============================================================

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"标定图片文件夹不存在：\n{IMAGE_DIR}"
    )


# 支持多种图片格式
allowed_suffixes = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in allowed_suffixes
)

print(f"找到 {len(images)} 张图片")


# 没有图片时直接终止
if not images:
    raise FileNotFoundError(
        "没有找到任何标定图片。\n"
        f"请检查文件夹：\n{IMAGE_DIR}\n"
        "支持格式：jpg / jpeg / png / bmp"
    )


# ============================================================
# 8. 检测棋盘格
# ============================================================

valid_count = 0

# 不再使用 gray.shape 保存尺寸
# 而是专门记录真正参与标定图片的尺寸
image_size = None

flags = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE
)


for filename in images:

    # --------------------------------------------------------
    # 读取图片
    # --------------------------------------------------------

    img = imread_unicode(filename)

    if img is None:
        print(
            f"[跳过] 无法读取：{filename}"
        )
        continue


    # --------------------------------------------------------
    # 转灰度图
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    current_size = gray.shape[::-1]


    # --------------------------------------------------------
    # 寻找棋盘格角点
    # --------------------------------------------------------

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        flags
    )


    if not ret:

        print(
            f"[失败] 未检测到 "
            f"{CHECKERBOARD[0]}×{CHECKERBOARD[1]} "
            f"内角点：{filename.name}"
        )

        continue


    # --------------------------------------------------------
    # 检查图片分辨率是否一致
    # --------------------------------------------------------

    if image_size is None:

        # 第一张有效标定图
        image_size = current_size

    elif current_size != image_size:

        print(
            f"[跳过] 图片尺寸不一致：{filename.name}\n"
            f"当前尺寸：{current_size}\n"
            f"要求尺寸：{image_size}"
        )

        continue


    # --------------------------------------------------------
    # 亚像素角点优化
    # --------------------------------------------------------

    corners2 = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria
    )


    # --------------------------------------------------------
    # 保存标定数据
    # --------------------------------------------------------

    objpoints.append(
        objp.copy()
    )

    imgpoints.append(
        corners2
    )

    valid_count += 1


    # --------------------------------------------------------
    # 显示识别结果
    # --------------------------------------------------------

    cv2.drawChessboardCorners(
        img,
        CHECKERBOARD,
        corners2,
        True
    )

    print(
        f"[成功] {filename.name}"
    )

    cv2.imshow(
        "Corners",
        img
    )

    cv2.waitKey(300)


cv2.destroyAllWindows()


print(
    f"\n有效图片：{valid_count} 张"
)


# ============================================================
# 9. 标定前安全检查
# ============================================================

if valid_count == 0:

    raise RuntimeError(
        "\n没有任何图片成功检测到棋盘格角点。\n\n"
        "请依次检查：\n"
        "1. CHECKERBOARD 是否等于真实内角点数量\n"
        "2. 棋盘格是否完整出现在照片中\n"
        "3. 图片是否模糊\n"
        "4. 是否存在严重反光\n"
        "5. 棋盘是否过小\n"
        "6. 棋盘是否被遮挡"
    )


if valid_count < 10:

    print(
        "\n警告：有效图片少于 10 张。"
    )

    print(
        "建议使用 10~20 张不同位置、"
        "不同角度的清晰棋盘照片进行标定。"
    )


# ============================================================
# 10. 相机标定
# ============================================================

rms, camera_matrix, dist_coeffs, rvecs, tvecs = \
    cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )


# ============================================================
# 11. 检查标定结果
# ============================================================

if not np.isfinite(rms):

    raise RuntimeError(
        "标定失败：RMS 重投影误差不是有限数值。"
    )


# ============================================================
# 12. 输出结果
# ============================================================

print("\n=========================")
print("标定完成")
print("=========================")

print("\n有效标定图片数量：")
print(valid_count)

print("\n标定图像尺寸：")
print(image_size)

print("\nRMS 重投影误差：")
print(rms)

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(dist_coeffs)


# ============================================================
# 13. 保存结果
# ============================================================

np.savez(
    OUTPUT_FILE,

    camera_matrix=camera_matrix,

    dist_coeffs=dist_coeffs,

    image_size=np.array(
        image_size
    ),

    checkerboard=np.array(
        CHECKERBOARD
    ),

    square_size=np.array(
        SQUARE_SIZE
    )
)


print(
    f"\n参数已经保存到：\n{OUTPUT_FILE}"
)