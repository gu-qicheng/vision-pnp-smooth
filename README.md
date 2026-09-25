# 11 cm 正方形自动红框 PnP

## 功能

程序使用电脑摄像头和已有的 `camera_calibration.npz`，逐帧检测红色正方形外轮廓的四个角点，调用 PnP 计算目标和相机的相对位置，并显示 XYZ 坐标轴。

当前版本使用 HSV 阈值和轮廓几何，不依赖 YOLO。每帧先独立检测，再用上一帧状态做平滑和短时跟踪；跟踪超时后才清除姿态。

红色掩码同时检查 HSV 红色范围和红色通道相对蓝/绿通道的优势，避免仅凭肤色相近的 HSV 值判定为门框。掩码只做闭运算连接细线，不做会抹掉窄边框的开运算。候选四边形还要通过凸性、矩形度、角度、长宽比和“颜色集中在边缘”的检查；填充红色物体则必须具有更强的红色通道优势。

逐帧处理会优先在速度预测的 ROI 中检测，并对放大后的 ROI 做角点精修。红框颜色短暂丢失时，程序在边缘图上使用带预测初值的四角 LK 光流，并用前后向误差、单应性、几何约束和 PnP RMS 校验；最多连续跟踪 3 帧，失败或超过上限后清除旧姿态。慢速移动使用较强平滑，快速移动自动提高新角点权重，减少滞后。

## 物理与坐标约定

- 正方形外轮廓边长：**110 mm（11 cm）**。
- 自动角点顺序：按图像轮廓顺序排列，从最靠上的点开始。
- 目标坐标原点：正方形中心。
- 目标 X 轴：向右；目标 Y 轴：向下；目标 Z 轴：远离镜头。
- `target_center_in_camera_mm`：目标中心在摄像头坐标系中的 `X/Y/Z`。
- `camera_in_target_mm`：相机原点在目标坐标系中的 `X/Y/Z`。
- 摄像头画面必须为 `1920×1080`，与标定文件一致。

纯正方形没有物理方向标记时，图像几何无法确认真实的物理 `TL/TR/BR/BL`；若需要稳定的物理角点身份，应增加非对称标记或 AprilTag/ArUco。

## 安装与启动

双击 `run_pnp_demo.bat`。首次启动会在当前目录创建 `.venv`，并安装 NumPy 与带 GUI 的 OpenCV。

启动 PnP 前请关闭 Windows 自带“相机”预览，避免两个程序同时占用同一摄像头。

也可以在 PowerShell 中运行：

```powershell
.\setup_pnp_env.ps1
.\.venv\Scripts\python.exe .\pnp_square_demo.py
```

默认读取当前目录的 `camera_calibration.npz`，使用摄像头设备 0。可手动指定：

```powershell
.\.venv\Scripts\python.exe .\pnp_square_demo.py --calibration .\camera_calibration.npz --camera-index 0
```

## 操作

1. 确认画面为 `1920×1080`。
2. 启动后程序自动打开实时画面和红色掩码窗口。
3. 检测到红框且 PnP 有效时显示角点、RMS、候选数量、坐标和 XYZ 轴。
4. 按 `Esc` 退出。

## 结果解释

- `target_center_in_camera_mm` 是目标中心在摄像头坐标系中的位置。
- `camera_in_target_mm` 由 `-R.T @ tvec` 计算，是相机相对门框的位置。
- `Z` 通常表示目标沿摄像头光轴方向的距离，目标离摄像头越远，Z 通常越大。
- RMS 重投影误差越小越好；超过 5 px 时程序显示 `POSE INVALID`，不显示伪姿态。
- 预览缩放只影响显示；自动检测和 PnP 使用原始 `1920×1080` 图像坐标。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m py_compile pnp_square_demo.py
```

自动测试使用合成投影、合成红框、移动帧和短时颜色丢失帧验证 110 mm 对象点、PnP 恢复、角点排序、相机坐标逆变换、RMS、速度预测光流、单应性校验、放大 ROI、快速移动自适应平滑和跟踪超时清除；真实摄像头操作仍需要现场验收。

## Git 版本

本目录是独立 Git 仓库。查看版本：

```powershell
git log --oneline --decorate
```

本次 OpenCV 预测、ROI 和自适应平滑迭代位于独立分支
`feature/opencv-predictive-roi`；旧版 `main` 和 `v0.5-verified` 保持不变。查看所有版本：

```powershell
git log --all --oneline --decorate
git switch main
git switch feature/opencv-predictive-roi
```

回退到基线：

```powershell
git switch --detach v0.1-baseline
```
