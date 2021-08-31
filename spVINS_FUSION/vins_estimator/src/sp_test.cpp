#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "superPoint/SuperPoint.h"
#include "estimator/estimator.h"
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "usage: feature_extraction img1 img2" << std::endl;
    return 1;
  }
  //-- 读取图像
  cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr);

  cv::resize(img_1, img_1, cv::Size(640, 480), cv::INTER_AREA);
  cv::resize(img_2, img_2, cv::Size(640, 480), cv::INTER_AREA);

  cv::cvtColor(img_1, img_1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img_2, img_2, cv::COLOR_BGR2GRAY);
  
  Estimator estimator;

  return 0;
}
