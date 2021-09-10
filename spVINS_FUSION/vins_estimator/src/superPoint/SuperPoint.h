#ifndef SUPERPOINT_H
#define SUPERPOINT_H
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <torch/script.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
//#include "../utility/tic_toc.h"
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif

struct SuperPointNet : torch::nn::Module {
  SuperPointNet();

  std::vector<torch::Tensor> forward(torch::Tensor x, int nms_dist);


  torch::nn::Conv2d conv1a;
  torch::nn::Conv2d conv1b;

  torch::nn::Conv2d conv2a;
  torch::nn::Conv2d conv2b;

  torch::nn::Conv2d conv3a;
  torch::nn::Conv2d conv3b;

  torch::nn::Conv2d conv4a;
  torch::nn::Conv2d conv4b;

  torch::nn::Conv2d convPa;
  torch::nn::Conv2d convPb;

  // descriptor
  torch::nn::Conv2d convDa;
  torch::nn::Conv2d convDb;

};


class SPDetector {
public:
    SPDetector(std::string _weights_path, int _nms_dist=4, float _conf_thresh=0.1,  bool _cuda=false);
    bool detect(cv::Mat img, cv::Mat mask, std::vector<cv::Point2f>& pts, cv::Mat& descriptors, const int num_pts);
    void detect(cv::Mat img, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors);
    bool detect(cv::Mat img, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors, cv::Mat& compressed_descriptors);
private:
    std::shared_ptr<SuperPointNet> net;
    std::string weights_path;
    int nms_dist;
    float conf_thresh;
    float nn_thresh;
    bool cuda;
    torch::DeviceType device_type;
};

class SPGlue{
public:
    SPGlue(std::string _weights_path, bool _cuda=false);
    std::vector<cv::DMatch> match(std::vector<cv::KeyPoint> pts_1, std::vector<cv::KeyPoint> pts_2, cv::Mat descriptors_1, cv::Mat descriptors_2);
private:
    std::string weights_path;
    bool cuda;
    torch::jit::script::Module module;
    //torch::DeviceType device_type;
    //torch::Device device;
};
class SPMatcher{
public:
    SPMatcher(float _nn_thresh=0.4);
    void match(cv::Mat _desc_1, cv::Mat _desc_2, std::vector<cv::DMatch>& matches);
    std::vector<cv::DMatch> getGoodMathces(std::vector<cv::DMatch> matches);
private:
    float computeDistance(const cv::Mat &a, const cv::Mat &b);
    static bool cmp(const cv::DMatch a, const cv::DMatch b){return a.distance < b.distance;};
    float nn_thresh;
    float distance;
    cv::Mat desc_1, desc_2;
};

#endif
