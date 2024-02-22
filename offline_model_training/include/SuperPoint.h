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
    SPDetector(std::shared_ptr<SuperPointNet> model, std::string _weights_path, int _nms_dist=4, float _conf_thresh=0.1,  bool _cuda=false);
    void detect(cv::Mat img, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors);
    bool detect(cv::Mat img, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors, cv::Mat& compressed_descriptors);
    bool detect(cv::Mat img, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors, cv::Mat& compressed_descriptors, const int c_num_pts);
    bool detectWindow(cv::Mat img, std::vector<cv::Point2f>& pts, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
private:
    std::shared_ptr<SuperPointNet> net;
    std::string weights_path;
    int nms_dist;
    float conf_thresh;
    float nn_thresh;
    bool cuda;
    torch::DeviceType device_type;
};

struct KeypointEncoder : torch::nn::Module {
  KeypointEncoder();
  std::shared_ptr<torch::nn::SequentialImpl> encoder{nullptr};
  torch::Tensor forward(torch::Tensor kpts, torch::Tensor scores);
};
struct MultiHeadedAttention : torch::nn::Module {
  MultiHeadedAttention();
  torch::nn::Conv1d merge{nullptr};
  torch::nn::ModuleList proj;
  int num_heads;
  int d_model;
  int dim;
  torch::Tensor forward(torch::Tensor query, torch::Tensor key, torch::Tensor value);
};

struct AttentionalPropagation : torch::nn::Module {
  AttentionalPropagation();
  std::shared_ptr<MultiHeadedAttention> attn{nullptr};
  std::shared_ptr<torch::nn::SequentialImpl> mlp{nullptr};
  torch::Tensor forward(torch::Tensor x, torch::Tensor source);
};
struct AttentionalGNN : torch::nn::Module {
  AttentionalGNN();
  torch::nn::ModuleList layers;
  void forward(torch::Tensor &desc0, torch::Tensor &desc1);
  std::vector<std::string> layer_names;
};


struct SuperGlueNet : torch::nn::Module {
  SuperGlueNet(float _thres);
  std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);
  std::shared_ptr<KeypointEncoder> kenc;
  std::shared_ptr<AttentionalGNN> gnn;
  torch::nn::Conv1d final_proj;
  float match_threshold;
};
class SPGlue {
public:
    SPGlue(std::string _weights_path, float _match_threshold, bool _cuda);
    std::vector<cv::DMatch> match(std::vector<cv::KeyPoint> pts_1, std::vector<cv::KeyPoint> pts_2, cv::Mat descriptors_1, cv::Mat descriptors_2);
 
private:
    std::shared_ptr<SuperGlueNet> net;
    std::string weights_path;
    float match_threshold;
    bool cuda;
    torch::DeviceType device_type;
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
