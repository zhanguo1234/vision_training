#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;

struct ThetaResidual {
    ThetaResidual(double t, double theta_obs)
        : t_(t), theta_obs_(theta_obs) {}

    template <typename T>
    bool operator()(const T* const p, T* residual) const {
        T pred = p[4] + p[1] * T(t_)
               - (p[0] / p[2]) * (cos(p[2] * T(t_) + p[3]) - cos(p[3]));
        residual[0] = T(theta_obs_) - pred;
        return true;
    }

    double t_, theta_obs_;
};

int main() {
    string videoPath = "resources/task_2.mp4";
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Cannot open video: " << videoPath << "\n";
        return 1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int totalFrames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    cout << "video: " << width << "x" << height
         << ", fps=" << fps
         << ", totalFrames=" << totalFrames << endl;

    Point2f center(480.0f, 360.0f);

    VideoWriter writer("result/task2_fit/tracking_overlay.mp4",
                       VideoWriter::fourcc('m', 'p', '4', 'v'),
                       fps, Size(width, height));

    vector<double> times;
    vector<double> angles;

    Mat frame;
    int frameIdx = 0;
    while (cap.read(frame)) {
        double t = frameIdx / fps;

        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        Mat mask;
        inRange(hsv, Scalar(80, 100, 100), Scalar(100, 255, 255), mask);

        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Point2f target(-1, -1);
        bool found = false;

        if (!contours.empty()) {
            int maxIdx = 0;
            double maxArea = 0;
            for (size_t i = 0; i < contours.size(); ++i) {
                double a = contourArea(contours[i]);
                if (a > maxArea) { maxArea = a; maxIdx = static_cast<int>(i); }
            }
            if (maxArea > 50.0) {
                Moments m = moments(contours[maxIdx]);
                if (m.m00 > 0) {
                    target.x = static_cast<float>(m.m10 / m.m00);
                    target.y = static_cast<float>(m.m01 / m.m00);
                    found = true;
                }
            }
        }

        if (found) {
            double theta = atan2(center.y - target.y, target.x - center.x);

            if (!angles.empty()) {
                double last = angles.back();
                while (theta - last > M_PI) theta -= 2 * M_PI;
                while (theta - last < -M_PI) theta += 2 * M_PI;
            }

            times.push_back(t);
            angles.push_back(theta);

            circle(frame, center, 5, Scalar(255, 255, 255), -1);
            circle(frame, target, 10, Scalar(255, 255, 0), 2);
            line(frame, center, target, Scalar(0, 255, 0), 2);
        }

        writer.write(frame);
        frameIdx++;
    }

    cap.release();
    writer.release();

    cout << "valid frames: " << times.size() << " / " << totalFrames << endl;

    if (times.size() < 10) {
        cerr << "Not enough valid frames.\n";
        return 1;
    }

    // 从差分估计初值
    vector<double> omegaRaw;
    double omegaMean = 0.0;
    for (size_t i = 0; i + 1 < times.size(); ++i) {
        double dt = times[i + 1] - times[i];
        if (dt <= 0) continue;
        double w = (angles[i + 1] - angles[i]) / dt;
        omegaRaw.push_back(w);
        omegaMean += w;
    }
    omegaMean /= omegaRaw.size();

    double wMin = omegaRaw[0], wMax = omegaRaw[0];
    for (double w : omegaRaw) {
        if (w < wMin) wMin = w;
        if (w > wMax) wMax = w;
    }

    double A_init = 0.5 * (wMax - wMin);
    double b_init = omegaMean;
    double Omega_init = 3.14;
    double phi_init = 0.0;
    double theta0_init = angles[0];

    cout << "\n===== omega stats =====" << endl;
    cout << "min = " << wMin << "  max = " << wMax << endl;
    cout << "mean = " << omegaMean << endl;

    // ============ 扫描 Omega 初值 ============
    cout << "\n===== scanning Omega init =====" << endl;
    double bestOmegaInit = Omega_init;
    double bestScanCost = 1e18;

    for (double omegaTest = 0.5; omegaTest <= 6.0; omegaTest += 0.2) {
        double pTest[5] = {A_init, b_init, omegaTest, phi_init, theta0_init};

        ceres::Problem probTest;
        for (size_t i = 0; i < times.size(); ++i) {
            ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<ThetaResidual, 1, 5>(
                    new ThetaResidual(times[i], angles[i]));
            probTest.AddResidualBlock(cost, nullptr, pTest);
        }
        probTest.SetParameterLowerBound(pTest, 0, 1e-6);
        probTest.SetParameterLowerBound(pTest, 2, 1e-6);

        ceres::Solver::Options optTest;
        optTest.linear_solver_type = ceres::DENSE_QR;
        optTest.max_num_iterations = 30;
        optTest.logging_type = ceres::SILENT;
        optTest.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary sumTest;
        ceres::Solve(optTest, &probTest, &sumTest);

        cout << "Omega_test=" << omegaTest
             << "  cost=" << sumTest.final_cost
             << "  A=" << pTest[0]
             << "  b=" << pTest[1] << endl;

        if (sumTest.final_cost < bestScanCost) {
            bestScanCost = sumTest.final_cost;
            bestOmegaInit = omegaTest;
        }
    }

    cout << "\nbest Omega init = " << bestOmegaInit
         << "  cost = " << bestScanCost << endl;

    // ============ 最终精细拟合 ============
    double p[5] = {A_init, b_init, bestOmegaInit, phi_init, theta0_init};

    ceres::Problem problem;
    for (size_t i = 0; i < times.size(); ++i) {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<ThetaResidual, 1, 5>(
                new ThetaResidual(times[i], angles[i]));
        problem.AddResidualBlock(cost, new ceres::HuberLoss(0.1), p);
    }

    problem.SetParameterLowerBound(p, 0, 1e-6);
    problem.SetParameterLowerBound(p, 2, 1e-6);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;

    double A = p[0], b = p[1], Omega = p[2], phi = p[3], theta0 = p[4];

    while (phi >= M_PI) phi -= 2 * M_PI;
    while (phi < -M_PI) phi += 2 * M_PI;

    cout << "fitted: A=" << A << " b=" << b
         << " Omega=" << Omega << " phi=" << phi
         << " theta0=" << theta0 << endl;

    // 计算 RMSE
    double sseTheta = 0.0;
    vector<double> thetaPred(times.size());
    for (size_t i = 0; i < times.size(); ++i) {
        thetaPred[i] = theta0 + b * times[i]
                     - (A / Omega) * (cos(Omega * times[i] + phi) - cos(phi));
        double r = angles[i] - thetaPred[i];
        sseTheta += r * r;
    }
    double rmseTheta = sqrt(sseTheta / times.size());

    vector<double> omegaTimes, omegaObs, omegaPred;
    double sseOmega = 0.0;
    for (size_t i = 0; i + 1 < times.size(); ++i) {
        double dt = times[i + 1] - times[i];
        if (dt <= 0) continue;
        double wObs = (angles[i + 1] - angles[i]) / dt;
        double wPred = (thetaPred[i + 1] - thetaPred[i]) / dt;
        sseOmega += (wObs - wPred) * (wObs - wPred);
        omegaTimes.push_back(0.5 * (times[i] + times[i + 1]));
        omegaObs.push_back(wObs);
        omegaPred.push_back(wPred);
    }
    double rmseOmega = sqrt(sseOmega / omegaTimes.size());

    cout << "RMSE (theta) = " << rmseTheta << " rad" << endl;
    cout << "RMSE (omega) = " << rmseOmega << " rad/s" << endl;

    // 画拟合图
    int plotW = 1200, plotH = 600;
    Mat fitImg(plotH, plotW, CV_8UC3, Scalar(255, 255, 255));

    double wMinP = omegaObs[0], wMaxP = omegaObs[0];
    for (double w : omegaObs) {
        if (w < wMinP) wMinP = w;
        if (w > wMaxP) wMaxP = w;
    }
    for (double w : omegaPred) {
        if (w < wMinP) wMinP = w;
        if (w > wMaxP) wMaxP = w;
    }
    double wRangeP = wMaxP - wMinP;
    if (wRangeP < 1e-6) wRangeP = 1.0;
    wMinP -= 0.1 * wRangeP;
    wMaxP += 0.1 * wRangeP;

    double tMin = omegaTimes.front();
    double tMax = omegaTimes.back();

    auto toPix = [&](double t, double w) -> Point {
        int x = static_cast<int>((t - tMin) / (tMax - tMin) * (plotW - 1));
        int y = static_cast<int>((wMaxP - w) / (wMaxP - wMinP) * (plotH - 1));
        return Point(x, y);
    };

    for (size_t i = 0; i < omegaTimes.size(); ++i) {
        circle(fitImg, toPix(omegaTimes[i], omegaObs[i]), 2,
               Scalar(255, 0, 0), -1);
    }
    for (size_t i = 0; i + 1 < omegaTimes.size(); ++i) {
        line(fitImg,
             toPix(omegaTimes[i], omegaPred[i]),
             toPix(omegaTimes[i + 1], omegaPred[i + 1]),
             Scalar(0, 0, 255), 2);
    }
    imwrite("result/task2_fit/fit_comparison.png", fitImg);
    imwrite("result/task2_fit/angular_velocity.png", fitImg);

    // 残差图
    Mat resImg(plotH, plotW, CV_8UC3, Scalar(255, 255, 255));
    double rMax = 1e-6;
    for (size_t i = 0; i < omegaTimes.size(); ++i) {
        double r = fabs(omegaObs[i] - omegaPred[i]);
        if (r > rMax) rMax = r;
    }
    rMax *= 1.2;

    auto toPixRes = [&](double t, double r) -> Point {
        int x = static_cast<int>((t - tMin) / (tMax - tMin) * (plotW - 1));
        int y = static_cast<int>((rMax - r) / (2 * rMax) * (plotH - 1));
        return Point(x, y);
    };

    line(resImg,
         toPixRes(omegaTimes.front(), 0),
         toPixRes(omegaTimes.back(), 0),
         Scalar(0, 0, 0), 1);

    for (size_t i = 0; i < omegaTimes.size(); ++i) {
        double r = omegaObs[i] - omegaPred[i];
        circle(resImg, toPixRes(omegaTimes[i], r), 2, Scalar(0, 0, 255), -1);
    }
    imwrite("result/task2_fit/residuals.png", resImg);

    // 写结果文件
    ofstream ofs("result/task2_fit_result.md");
    ofs << "# 任务2：合成旋转视频参数拟合结果\n\n";
    ofs << "## 模型\n\n";
    ofs << "omega(t) = b + A * sin(Omega * t + phi)\n\n";
    ofs << "theta(t) = theta0 + b*t - (A/Omega)*(cos(Omega*t+phi) - cos(phi))\n\n";
    ofs << "## 估计参数\n\n";
    ofs << "- A = " << A << " rad/s\n";
    ofs << "- b = " << b << " rad/s\n";
    ofs << "- Omega = " << Omega << " rad/s\n";
    ofs << "- phi = " << phi << " rad\n";
    ofs << "- theta0 = " << theta0 << " rad\n";
    ofs << "- 速度变化周期 T = " << (2 * M_PI / Omega) << " s\n\n";
    ofs << "## 误差\n\n";
    ofs << "- RMSE (theta) = " << rmseTheta << " rad\n";
    ofs << "- RMSE (omega) = " << rmseOmega << " rad/s\n";
    ofs << "- 有效样本数: " << times.size() << "\n";
    ofs << "- 帧范围: 0 - " << (totalFrames - 1) << "\n";
    ofs << "- 视频参数: " << width << "x" << height
        << ", " << fps << " FPS, " << totalFrames << " 帧\n";
    ofs.close();

    cout << "Task2 done.\n";
    return 0;
}