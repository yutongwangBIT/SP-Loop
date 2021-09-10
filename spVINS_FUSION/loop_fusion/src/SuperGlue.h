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
  SuperGlueNet();
  std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);
  std::shared_ptr<KeypointEncoder> kenc;
  std::shared_ptr<AttentionalGNN> gnn;
  torch::nn::Conv1d final_proj;
};
class SPGlue {
public:
    SPGlue(std::string _weights_path, bool _cuda);
    std::vector<cv::DMatch> match(std::vector<cv::KeyPoint> pts_1, std::vector<cv::KeyPoint> pts_2, cv::Mat descriptors_1, cv::Mat descriptors_2);
private:
    std::shared_ptr<SuperGlueNet> net;
    std::string weights_path;
    bool cuda;
    torch::DeviceType device_type;
};

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
    bool detect(cv::Mat img, std::vector<cv::KeyPoint>& kpts, cv::Mat& descriptors, const int num_pts);
private:
    std::shared_ptr<SuperPointNet> net;
    std::string weights_path;
    int nms_dist;
    float conf_thresh;
    float nn_thresh;
    bool cuda;
    torch::DeviceType device_type;
};


#endif
