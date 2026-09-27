#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // ============ 读图 ============
    Mat img = imread("resources/test_image.jpg");
    if (img.empty()) {
        cerr << "Cannot read image! 请确认 resources/test_image.jpg 存在。\n";
        return 1;
    }
    imwrite("result/task1_images/original.png", img);

    // ============ 1. 灰度图 ============
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    imwrite("result/task1_images/gray.png", gray);

    // ============ 1. HSV 拆分 ============
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    vector<Mat> hsvChannels;
    split(hsv, hsvChannels);

    imwrite("result/task1_images/hsv_h.png", hsvChannels[0]);
    imwrite("result/task1_images/hsv_s.png", hsvChannels[1]);
    imwrite("result/task1_images/hsv_v.png", hsvChannels[2]);

    // ============ 2. 三种滤波 ============
    Mat meanImg, gaussianImg, medianImg;
    blur(img, meanImg, Size(5, 5));
    GaussianBlur(img, gaussianImg, Size(5, 5), 1.5);
    medianBlur(img, medianImg, 5);

    imwrite("result/task1_images/mean_filter.png", meanImg);
    imwrite("result/task1_images/gaussian_filter.png", gaussianImg);
    imwrite("result/task1_images/median_filter.png", medianImg);

    // ============ 3. 红色双区间掩膜 ============
    Mat maskLow, maskHigh, redMask;
    inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), maskLow);
    inRange(hsv, Scalar(170, 100, 100), Scalar(179, 255, 255), maskHigh);
    bitwise_or(maskLow, maskHigh, redMask);

    imwrite("result/task1_images/red_mask.png", redMask);

    // ============ 4. 形态学操作 ============
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

    Mat eroded, dilated, opened, closed;
    erode(redMask, eroded, kernel);
    dilate(redMask, dilated, kernel);
    morphologyEx(redMask, opened, MORPH_OPEN, kernel);
    morphologyEx(redMask, closed, MORPH_CLOSE, kernel);

    imwrite("result/task1_images/erode.png", eroded);
    imwrite("result/task1_images/dilate.png", dilated);
    imwrite("result/task1_images/open.png", opened);
    imwrite("result/task1_images/close.png", closed);

    // ============ 5. 轮廓提取与筛选 ============
    vector<vector<Point>> contours;
    findContours(closed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat result = img.clone();
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area < 500.0) continue;

        Rect box = boundingRect(contours[i]);
        double ratio = static_cast<double>(box.width) / box.height;
        if (ratio < 0.2 || ratio > 5.0) continue;

        rectangle(result, box, Scalar(0, 0, 255), 2);
        drawContours(result, contours, static_cast<int>(i), Scalar(0, 255, 0), 2);

        cout << "contour " << i << " area = " << area
             << " box = (" << box.x << ", " << box.y << ", "
             << box.width << ", " << box.height << ")" << endl;
    }

    imwrite("result/task1_images/contours_boxes.png", result);

    // ============ 6. 绘制 ============
    Mat drawing = img.clone();
    circle(drawing, Point(100, 100), 50, Scalar(255, 0, 0), 2);
    rectangle(drawing, Rect(200, 200, 150, 100), Scalar(0, 255, 0), 3);
    putText(drawing, "Tulip", Point(300, 400),
            FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
    imwrite("result/task1_images/drawing.png", drawing);

    // ============ 7. 旋转 35 度 ============
    Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    Mat rot = getRotationMatrix2D(center, 35, 1.0);
    Mat rotated;
    warpAffine(img, rotated, rot, img.size());
    imwrite("result/task1_images/rotated_35deg.png", rotated);

    // ============ 8. 裁剪左上 1/4 ============
    Rect cropRect(0, 0, img.cols / 2, img.rows / 2);
    Mat crop = img(cropRect);
    imwrite("result/task1_images/crop_top_left.png", crop);

    cout << "image size: " << img.cols << " x " << img.rows << endl;
    cout << "Task1 complete!\n";
    return 0;
}