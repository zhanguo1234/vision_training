# 任务2：合成旋转视频参数拟合结果

## 模型

omega(t) = b + A * sin(Omega * t + phi)

theta(t) = theta0 + b*t - (A/Omega)*(cos(Omega*t+phi) - cos(phi))

## 估计参数

- A = 0.549941 rad/s
- b = 1.35003 rad/s
- Omega = 1.64987 rad/s
- phi = 0.702918 rad
- theta0 = 0.350315 rad
- 速度变化周期 T = 3.80829 s

## 误差

- RMSE (theta) = 0.00233645 rad
- RMSE (omega) = 0.0622903 rad/s
- 有效样本数: 1440
- 帧范围: 0 - 1439
- 视频参数: 960x720, 60 FPS, 1440 帧
