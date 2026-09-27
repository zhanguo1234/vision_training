# Vision Training：OpenCV / Eigen / Ceres 综合练习

本工程完成三个图像处理与优化任务：

1. 任务1：对郁金香图片做颜色空间转换、滤波、红色提取、形态学、轮廓筛选、绘制与几何变换。
2. 任务2：对合成旋转视频识别青色目标，拟合角速度模型 omega(t) = b + A*sin(Omega*t + phi)。
3. 任务3：对真实能量机关视频做目标识别与稳定跟踪。

---

## 环境依赖

- Ubuntu 22.04
- CMake >= 3.16
- g++ 11.4.0
- OpenCV 4.5.4
- Eigen 3.4.0
- Ceres Solver 2.0.0

安装：

    sudo apt update
    sudo apt install build-essential cmake libopencv-dev libeigen3-dev libceres-dev -y

---

## 编译与运行

    # 配置
    cmake -S . -B build

    # 编译
    cmake --build build -j4

    # 运行任务1
    cd ~/workspace/vision_training
    ./build/task1_image

    # 运行任务2
    ./build/task2_fit

    # 运行任务3（待实现）
    ./build/task3_windmill

注意：运行时必须 cd 到工程根目录，因为代码中使用相对路径 resources/ 和 result/。

---

## 目录结构

    vision_training/
    ├── CMakeLists.txt
    ├── README.md
    ├── include/
    ├── src/
    │   ├── common/
    │   ├── task1_image/main.cpp
    │   ├── task2_fit/main.cpp
    │   └── task3_windmill/main.cpp
    ├── config/
    ├── resources/
    │   ├── test_image.jpg
    │   ├── task_2.mp4
    │   ├── task_3.mp4
    │   └── task_4.mp4
    └── result/
        ├── task1_images/
        ├── task2_fit/
        ├── task3_windmill/
        ├── task2_fit_result.md
        └── task3_tracking_result.md

---

## 任务1：OpenCV 图片处理（30分）

### 1.1 输入

- 图片：resources/test_image.jpg
- 尺寸：1280 x 853（郁金香图）

### 1.2 处理步骤与参数

#### 步骤 1：灰度与 HSV 拆分

- 灰度转换：cvtColor(img, gray, COLOR_BGR2GRAY)
- HSV 转换：cvtColor(img, hsv, COLOR_BGR2HSV)
- 拆分：split(hsv, hsvChannels)
- OpenCV HSV 范围：H 在 [0, 179]，S、V 在 [0, 255]

输出：gray.png、hsv_h.png、hsv_s.png、hsv_v.png

#### 步骤 2：三种滤波

| 滤波 | 核尺寸 | 标准差 | 输出 |
|---|---|---|---|
| 均值 | 5x5 | - | mean_filter.png |
| 高斯 | 5x5 | sigma = 1.5 | gaussian_filter.png |
| 中值 | 5 | - | median_filter.png |

观察：均值滤波模糊最明显；高斯滤波边缘保留稍好；中值滤波对椒盐噪声效果最好。

#### 步骤 3：红色双区间掩膜

- 低端区间：H 在 [0, 10]，S >= 100，V >= 100
- 高端区间：H 在 [170, 179]，S >= 100，V >= 100
- 合并：bitwise_or(maskLow, maskHigh, redMask)

效果说明：
- 红色区域（H 在 0-10 或 170-179）被正确提取。
- 黄色区域（H 约 20-35）被过滤。
- 阴影区域（V < 100）被过滤。
- 过曝区域（S < 100）被过滤。

输出：red_mask.png

#### 步骤 4：形态学操作

- 结构元素：5x5 矩形（MORPH_RECT）
- 腐蚀（erode）：前景缩小
- 膨胀（dilate）：前景扩大
- 开运算（MORPH_OPEN）：去小白噪点
- 闭运算（MORPH_CLOSE）：填小黑洞

输出：erode.png、dilate.png、open.png、close.png

#### 步骤 5：轮廓提取与筛选

- 输入：闭运算结果 closed
- 检索模式：RETR_EXTERNAL（只取最外层）
- 近似方法：CHAIN_APPROX_SIMPLE（压缩冗余点）
- 面积阈值：>= 500 像素平方
- 长宽比：0.2 到 5.0

筛选后轮廓（本次运行）：
- contour 7：area = 5078.0，box = (184, 564, 98, 170)
- contour 10：area = 766.5，box = (552, 541, 32, 42)
- contour 30：area = 44952.5，box = (0, 93, 224, 391)
- contour 31：area = 224598.0，box = (254, 35, 1026, 521)

输出：contours_boxes.png

#### 步骤 6：绘制与几何变换

- 画圆：圆心 (100, 100)，半径 50，蓝色，线宽 2
- 画矩形：Rect(200, 200, 150, 100)，绿色，线宽 3
- 画文字："Tulip"，位置 (300, 400)，红色，字号 1.5，线宽 3
- 旋转：绕图像中心 (640, 426.5)，逆时针 35 度，不缩放
- 裁剪：左上角 1/4（宽 640，高 426）

输出：drawing.png、rotated_35deg.png、crop_top_left.png

### 1.3 任务1输出文件（共16张）

    result/task1_images/
    ├── gray.png
    ├── hsv_h.png
    ├── hsv_s.png
    ├── hsv_v.png
    ├── mean_filter.png
    ├── gaussian_filter.png
    ├── median_filter.png
    ├── red_mask.png
    ├── erode.png
    ├── dilate.png
    ├── open.png
    ├── close.png
    ├── contours_boxes.png
    ├── drawing.png
    ├── rotated_35deg.png
    └── crop_top_left.png

---

## 任务2：合成旋转视频参数拟合（30分）

### 2.1 输入

- 视频：resources/task_2.mp4
- 参数：960x720，60 FPS，1440 帧，24 秒
- 旋转中心：(480, 360)（已知）
- 跟踪目标：青色圆点

### 2.2 检测方法

- 颜色空间：BGR -> HSV
- 青色掩膜：H 在 [80, 100]，S >= 100，V >= 100
- 形态学：5x5 矩形，先开后闭
- 轮廓：RETR_EXTERNAL + CHAIN_APPROX_SIMPLE
- 中心：moments 求质心（m.m10/m.m00, m.m01/m.m00）

检测结果：1440 / 1440 帧全部识别成功。

### 2.3 角度计算

角度定义（讲义第 21 页）：

    theta = atan2(cy - y, x - cx)

- (cx, cy) = (480, 360)
- 从右方起算，逆时针为正
- 角度展开：消除正负 pi 跳变

### 2.4 拟合方法

模型：

    omega(t) = b + A * sin(Omega * t + phi)

对 t 积分，得到角度模型：

    theta(t) = theta0 + b*t - (A/Omega)*(cos(Omega*t + phi) - cos(phi))

参数：A、b、Omega、phi、theta0，共 5 个。

优化方法：

1. 从差分角速度估计初值：A_init = 0.5*(omega_max - omega_min)，b_init = omega_mean，Omega_init = 3.14，phi_init = 0，theta0_init = theta[0]。
2. 扫描 Omega 初值：从 0.5 到 6.0，步长 0.2，每个 Omega 跑 30 次迭代，找 cost 最小的 Omega。
3. 用最优 Omega 作为初值，Ceres 精细拟合（DENSE_QR，Huber 损失 0.1，最大 500 次迭代）。

为什么扫描 Omega：Omega 是最敏感的非线性参数，初值差得远会陷入局部极小值（让 A 趋于 0）。扫描能找到正确的吸引域。

### 2.5 拟合结果

| 参数 | 值 | 单位 |
|---|---|---|
| A | 0.549941 | rad/s |
| b | 1.35003 | rad/s |
| Omega | 1.64987 | rad/s |
| phi | 0.702918 | rad |
| theta0 | 0.350315 | rad |
| 速度变化周期 T | 3.80829 | s |

### 2.6 误差

- RMSE (theta) = 0.00233645 rad （约 0.13 度）
- RMSE (omega) = 0.0622903 rad/s
- 有效样本数：1440
- 帧范围：0 - 1439
- 视频参数：960x720，60 FPS，1440 帧

详细结果：见 result/task2_fit_result.md

### 2.7 任务2输出文件

    result/task2_fit/
    ├── tracking_overlay.mp4    # 带跟踪标记的视频
    ├── fit_comparison.png      # 观测 vs 拟合曲线
    ├── angular_velocity.png    # 角速度曲线
    └── residuals.png           # 残差图

    result/task2_fit_result.md  # 参数与分析

---

## 任务3：真实能量机关视频识别与稳定跟踪（40分）

状态：待实现

输入：
- resources/task_3.mp4：小能量机关，最多同时 1 个目标亮起
- resources/task_4.mp4：大能量机关，最多同时 2 个目标亮起

输出：
- result/task3_windmill/task_3/recognition_overlay.mp4
- result/task3_windmill/task_4/recognition_overlay.mp4
- result/task3_tracking_result.md

方法（待补充）：
- 检测 R 标中心
- 检测亮起扇叶
- 稳定锁定与重选
- 可视化

---

## 参考资料

- OpenCV 4.x 文档：https://docs.opencv.org/4.x/
- Eigen 官方文档：https://libeigen.gitlab.io/
- Ceres Solver 官方文档：http://ceres-solver.org/
EOF