#include "SuperPoint.h"


const int c1 = 64;
const int c2 = 64;
const int c3 = 128;
const int c4 = 128;
const int c5 = 256;
const int d1 = 256;

torch::Tensor max_pool(torch::Tensor x, int nms_dist){
    return torch::nn::functional::max_pool2d(x, torch::nn::functional::MaxPool2dFuncOptions(2*nms_dist+1).stride(1).padding(nms_dist));
}
void simple_nms(torch::Tensor& input_tensor, int nms_dist){
    auto mask = torch::eq(input_tensor, max_pool(input_tensor,nms_dist));
    //std::cout<<"msak::"<<mask<<std::endl;
    auto zeros = torch::zeros_like(input_tensor);
    for(auto i=0;i<2;i++){
        auto supp_mask = torch::ge(max_pool(mask.to(torch::kFloat),nms_dist),0);
        auto supp_tensor = torch::where(supp_mask,zeros,input_tensor);
        auto new_max_mask = torch::eq(supp_tensor, max_pool(supp_tensor,nms_dist));
        mask = torch::__or__(mask, torch::__and__(new_max_mask, torch::logical_not(supp_mask)));
    }
    input_tensor = torch::where(mask, input_tensor, zeros);
}

SuperPointNet::SuperPointNet()
      : conv1a(torch::nn::Conv2dOptions( 1, c1, 3).stride(1).padding(1)),
        conv1b(torch::nn::Conv2dOptions(c1, c1, 3).stride(1).padding(1)),

        conv2a(torch::nn::Conv2dOptions(c1, c2, 3).stride(1).padding(1)),
        conv2b(torch::nn::Conv2dOptions(c2, c2, 3).stride(1).padding(1)),

        conv3a(torch::nn::Conv2dOptions(c2, c3, 3).stride(1).padding(1)),
        conv3b(torch::nn::Conv2dOptions(c3, c3, 3).stride(1).padding(1)),

        conv4a(torch::nn::Conv2dOptions(c3, c4, 3).stride(1).padding(1)),
        conv4b(torch::nn::Conv2dOptions(c4, c4, 3).stride(1).padding(1)),

        convPa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
        convPb(torch::nn::Conv2dOptions(c5, 65, 1).stride(1).padding(0)),

        convDa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
        convDb(torch::nn::Conv2dOptions(c5, d1, 1).stride(1).padding(0))
        
  {
    register_module("conv1a", conv1a);
    register_module("conv1b", conv1b);

    register_module("conv2a", conv2a);
    register_module("conv2b", conv2b);

    register_module("conv3a", conv3a);
    register_module("conv3b", conv3b);

    register_module("conv4a", conv4a);
    register_module("conv4b", conv4b);

    register_module("convPa", convPa);
    register_module("convPb", convPb);

    register_module("convDa", convDa);
    register_module("convDb", convDb);
  }


std::vector<torch::Tensor> SuperPointNet::forward(torch::Tensor x, int nms_dist) {

    x = torch::relu(conv1a->forward(x));
    x = torch::relu(conv1b->forward(x));
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv2a->forward(x));
    x = torch::relu(conv2b->forward(x));
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv3a->forward(x));
    x = torch::relu(conv3b->forward(x));
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv4a->forward(x));
    x = torch::relu(conv4b->forward(x));

    auto cPa = torch::relu(convPa->forward(x));
    auto semi = convPb->forward(cPa);  // [B, 65, H/8, W/8]

    auto cDa = torch::relu(convDa->forward(x));
    auto desc = convDb->forward(cDa);  // [B, d1, H/8, W/8]

    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    semi = torch::softmax(semi, 1);
    semi = semi.slice(1, 0, 64);
    semi = semi.permute({0, 2, 3, 1});  // [B, H/8, W/8, 64]


    int Hc = semi.size(1);
    int Wc = semi.size(2);
    semi = semi.contiguous().view({-1, Hc, Wc, 8, 8});
    semi = semi.permute({0, 1, 3, 2, 4});
    semi = semi.contiguous().view({-1, Hc * 8, Wc * 8});  // [B, H, W]

    simple_nms(semi, nms_dist);

    std::vector<torch::Tensor> ret;
    ret.push_back(semi);
    ret.push_back(desc);

    return ret;
}

SPDetector::SPDetector(std::string _weights_path, int _nms_dist, float _conf_thresh, bool _cuda)
    :weights_path(_weights_path), nms_dist(_nms_dist), conf_thresh(_conf_thresh), cuda(_cuda)
{
    net = std::make_shared<SuperPointNet>();
    torch::load(net, weights_path); 
    bool use_cuda = cuda && torch::cuda::is_available();
    if (use_cuda){
        device_type = torch::kCUDA;
        std::cout<<"USE CUDA!"<<std::endl;
    }
    else
        device_type = torch::kCPU;
    torch::Device device0(device_type);
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    net->to(device0);
    std::chrono::steady_clock::time_point t01 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used0 = std::chrono::duration_cast<std::chrono::duration<double>>(t01 - t0);
    std::cout << "time to put net on cuda= " << time_used0.count() << " seconds. " << std::endl;
}

bool SPDetector::detect(cv::Mat img, cv::Mat pts_mask, std::vector<cv::Point2f>& pts, cv::Mat& descriptors, const int c_num_pts){
    cv::Mat img_re, pts_mask_re;
    cv::resize(img, img_re, cv::Size(640, 480), cv::INTER_AREA);
    cv::resize(pts_mask, pts_mask_re, cv::Size(640, 480), cv::INTER_NEAREST);
    auto input = torch::from_blob(img.clone().data, {1, 1, img.rows, img.cols}, torch::kByte);
    
    input = input.to(torch::kFloat) / 255;
    torch::Device device(device_type);
    input = input.set_requires_grad(false);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    
    auto out = net->forward(input.to(device), nms_dist);  
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    //std::cout << "extract SP cost = " << 1000*time_used.count() << " ms. " << std::endl;
    
    torch::Tensor mProb = out[0].squeeze(0);  // [H, W] # squeeze(0) leaves the tensor unchanged
    torch::Tensor mDesc = out[1];
    
    TicToc t_2;
    auto mask_input = torch::from_blob(pts_mask.clone().data, { pts_mask.rows, pts_mask.cols}, torch::kU8);
    mask_input = mask_input.to(device);
    auto zeros = torch::zeros_like(mProb);
    mProb = torch::where(mask_input==255, mProb, zeros);
    //USE TORCH TOPK AVOIDED SEND MPROB DATA INTO VECTOR
    auto kpts = (mProb > conf_thresh);
    kpts = torch::nonzero(kpts); 
    //std::cout<<"kpts size :"<<kpts.size(0)<<std::endl;
    auto mask_score = torch::ge(mProb,conf_thresh);
    auto scores = torch::masked_select(mProb, mask_score);
    std::cout<<"score size:"<<scores.size(0)<<std::endl;
    auto num_pts = std::min(c_num_pts, int(scores.size(0)));
    auto top = scores.topk(num_pts);
    auto top_values = std::get<0>(top);
    auto top_index = std::get<1>(top);
    auto top_min = torch::min(top_values);

    auto grid = torch::zeros({1, 1, kpts.size(0), 2});  // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * kpts.slice(1, 1, 2) / mProb.size(1) - 1;  // x
    grid[0][0].slice(1, 1, 2) = 2.0 * kpts.slice(1, 0, 1) / mProb.size(0) - 1;  // y
    //printf("2 time %f\n", t_2.toc());
    TicToc t_3;
    auto desc = torch::grid_sampler(mDesc, grid.to(device), 0, 0, true);  // [1, 256, 1, n_keypoints]
    //printf("3 time %f\n", t_3.toc());
    TicToc t_4;
    desc = desc.squeeze(0).squeeze(1);  // [256, n_keypoints] Returns a tensor with all the dimensions of input of size 1 removed
    // normalize to 1
    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));
    desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]
    desc = desc.to(torch::kCPU);
    //printf("4 time %f\n", t_4.toc());
    TicToc t_5;
    cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data_ptr());
    cv::Mat descriptors_whole = desc_mat.clone();
    //printf("5 time %f\n", t_5.toc());
    TicToc t_6;
    //std::cout<<"top_index size :"<<top_index.size(0)<<std::endl;
    for(auto i=0; i<top_index.size(0); ++i){
        int index = top_index[i].item<int>();
        //std::cout<<"index :"<<index<<std::endl;
        float x = std::round(kpts[index][1].item<float>()*(img.rows/480));
        float y = std::round(kpts[index][0].item<float>()*(img.cols/640));
        pts.push_back(cv::Point2f(x, y));
        descriptors.push_back(descriptors_whole.row(index)); //WRONG?
    }
    //printf("6 time %f\n", t_6.toc());
    //if(pts.size()<10)
      //  return false;
    //else    
        return true;
}

bool SPDetector::detect(cv::Mat img, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors, cv::Mat& compressed_descriptors){
    auto input = torch::from_blob(img.clone().data, {1, 1, img.rows, img.cols}, torch::kByte);
    input = input.to(torch::kFloat) / 255;
    bool use_cuda = cuda && torch::cuda::is_available();
    torch::DeviceType device_type;
    if (use_cuda)
        device_type = torch::kCUDA;
    else
        device_type = torch::kCPU;
    torch::Device device(device_type);

    net->to(device);
    input = input.set_requires_grad(false);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto out = net->forward(input.to(device), nms_dist);  
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract SP using CPU cost = " << time_used.count() << " seconds. " << std::endl;
    
    torch::Tensor mProb = out[0].squeeze(0);  // [H, W] # squeeze(0) leaves the tensor unchanged
    torch::Tensor mDesc = out[1]; 
    auto kpts = (mProb > conf_thresh);
    kpts = torch::nonzero(kpts);  // [n_keypoints, 2]  (y, x)
    for (int i = 0; i < kpts.size(0); i++) {
        float response = mProb[kpts[i][0]][kpts[i][1]].item<float>();
        pts.push_back(cv::KeyPoint(kpts[i][1].item<float>(), kpts[i][0].item<float>(), 8, -1, response));
    }


    auto grid = torch::zeros({1, 1, kpts.size(0), 2});  // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * kpts.slice(1, 1, 2) / mProb.size(1) - 1;  // x
    grid[0][0].slice(1, 1, 2) = 2.0 * kpts.slice(1, 0, 1) / mProb.size(0) - 1;  // y

    auto desc = torch::grid_sampler(mDesc, grid, 0, 0, true);  // [1, 256, 1, n_keypoints]
    desc = desc.squeeze(0).squeeze(1);  // [256, n_keypoints] Returns a tensor with all the dimensions of input of size 1 removed

    // normalize to 1
    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]
    desc = desc.to(torch::kCPU);

    cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data_ptr());

    descriptors = desc_mat.clone();
    std::cout<<descriptors.size()<<std::endl;
    if(descriptors.rows == 0 || descriptors.cols == 0)
        return false;

    cv::Mat comp_desc_mat(descriptors.size(), CV_8UC1);
    for(auto i=0;i<descriptors.rows;++i){
        cv::Mat d_i_float = descriptors.row(i);
        cv::Mat d_i_int(d_i_float.size(), CV_8UC1);
        double minv = 0.0, maxv = 0.0;  
        double* minp = &minv;  
        double* maxp = &maxv;  
        minMaxIdx(d_i_float,minp,maxp);  
        cv::Mat mat_min(d_i_float.size(),CV_32FC1);
        mat_min = cv::Scalar::all(minv); 
        //std::cout<<mat_min<<std::endl;
        d_i_int = 255*(d_i_float-mat_min)/(maxv-minv);
        comp_desc_mat.row(i) = d_i_int.clone();
    }
    compressed_descriptors = comp_desc_mat.clone();
    /*cv::PCA pca(descriptors, cv::Mat(), CV_PCA_DATA_AS_ROW, 4);
    std::cout << "pca mean:" << pca.mean.size() << std::endl;
    compressed_descriptors = pca.project(descriptors);
    std::cout<<"comp desc size:"<<compressed_descriptors.size()<<std::endl;
    if(compressed_descriptors.cols != 4)
        return false;*/
    return true;
}

void SPDetector::detect(cv::Mat img, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors){
    auto input = torch::from_blob(img.clone().data, {1, 1, img.rows, img.cols}, torch::kByte);
    input = input.to(torch::kFloat) / 255;
    bool use_cuda = cuda && torch::cuda::is_available();
    torch::DeviceType device_type;
    if (use_cuda)
        device_type = torch::kCUDA;
    else
        device_type = torch::kCPU;
    torch::Device device(device_type);

    net->to(device);
    input = input.set_requires_grad(false);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto out = net->forward(input.to(device), nms_dist);  
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract SP using CPU cost = " << time_used.count() << " seconds. " << std::endl;
    
   // std::cout<<"out0 size:"<<out[0].size(0)<<","<<out[0].size(1)<<","<<out[0].size(2)<<std::endl; //1 480 640
  //  std::cout<<"out1 size:"<<out[1].size(0)<<","<<out[1].size(1)<<","<<out[1].size(2)<<std::endl;
    torch::Tensor mProb = out[0].squeeze(0);  // [H, W] # squeeze(0) leaves the tensor unchanged
    torch::Tensor mDesc = out[1];
  //  std::cout<<"mProb size:"<<mProb.size(0)<<std::endl;
    auto kpts = (mProb > conf_thresh);
    kpts = torch::nonzero(kpts);  // [n_keypoints, 2]  (y, x)
  //  std::cout<<"kpts size:"<<kpts.size(0)<<","<<kpts.size(1)<<<<std::endl;
    for (int i = 0; i < kpts.size(0); i++) {
        float response = mProb[kpts[i][0]][kpts[i][1]].item<float>();
        //std::cout<<response<<",";
        pts.push_back(cv::KeyPoint(kpts[i][1].item<float>(), kpts[i][0].item<float>(), 8, -1, response));
    }

    auto grid = torch::zeros({1, 1, kpts.size(0), 2});  // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * kpts.slice(1, 1, 2) / mProb.size(1) - 1;  // x
    grid[0][0].slice(1, 1, 2) = 2.0 * kpts.slice(1, 0, 1) / mProb.size(0) - 1;  // y

    auto desc = torch::grid_sampler(mDesc, grid, 0, 0, true);  // [1, 256, 1, n_keypoints]
    desc = desc.squeeze(0).squeeze(1);  // [256, n_keypoints] Returns a tensor with all the dimensions of input of size 1 removed

    // normalize to 1
    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]
    desc = desc.to(torch::kCPU);

    cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data_ptr());

    descriptors = desc_mat.clone();
}
SPGlue::SPGlue(std::string _weights_path, bool _cuda):cuda(_cuda){
    try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(_weights_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }
}
std::vector<cv::DMatch> SPGlue::match(std::vector<cv::KeyPoint> pts_1, std::vector<cv::KeyPoint> pts_2, cv::Mat descriptors_1, cv::Mat descriptors_2){
    //CPU OR GPU
    bool use_cuda = cuda && torch::cuda::is_available();
    torch::DeviceType device_type;
    if (use_cuda)
        device_type = torch::kCUDA;
    else
        device_type = torch::kCPU;
    torch::Device device(device_type);
    module.to(device);
    //CONVERT DATA TO TENSOR
    cv::Mat pointmatrix1(pts_1.size(), 2, CV_32F); // [2 n]
    cv::Mat scoresmatrix1(pts_1.size(), 1, CV_32F);
    int column1 = 0;
    for (auto& kp: pts_1) {
        pointmatrix1.at<float>(column1,0) = kp.pt.x;
        pointmatrix1.at<float>(column1,1) = kp.pt.y;
        scoresmatrix1.at<float>(column1, 0) = kp.response;
        column1++;
    }
    cv::Mat pointmatrix2(pts_2.size(), 2, CV_32F); // [2 n]
    cv::Mat scoresmatrix2(pts_2.size(), 1, CV_32F); //[1 n]
    int column2 = 0;
    for (auto& kp: pts_2) {
        pointmatrix2.at<float>(column2,0) = kp.pt.x;
        pointmatrix2.at<float>(column2,1) = kp.pt.y;
        scoresmatrix2.at<float>(column2,0) = kp.response;
        column2++;
    }
    descriptors_1 = descriptors_1.t();
    descriptors_2 = descriptors_2.t();

    auto input_kpts1 = torch::from_blob(pointmatrix1.data, {1, pointmatrix1.rows, pointmatrix1.cols}); //[1,n,2]
    auto input_scores1 = torch::from_blob(scoresmatrix1.data, {1, scoresmatrix1.rows});//[1,n]
    auto input_desc1 = torch::from_blob(descriptors_1.data, {1, descriptors_1.rows, descriptors_1.cols});//[1,256,n]
    auto input_kpts2 = torch::from_blob(pointmatrix2.data, {1, pointmatrix2.rows, pointmatrix2.cols});
    auto input_scores2 = torch::from_blob(scoresmatrix2.data, {1, scoresmatrix2.rows});
    auto input_desc2 = torch::from_blob(descriptors_2.data, {1, descriptors_2.rows, descriptors_2.cols});

    std::vector<torch::jit::IValue> inputs; //multiple inputs defined in script
    inputs.push_back(input_kpts1.to(device));
    inputs.push_back(input_scores1.to(device));
    inputs.push_back(input_desc1.to(device));
    inputs.push_back(input_kpts2.to(device));
    inputs.push_back(input_scores2.to(device));
    inputs.push_back(input_desc2.to(device));

    //CALCULATE AND GET OUTPUTS
    auto output = module.forward(inputs).toTuple();

    auto matches1  = output->elements()[0].toTensor().to(torch::kCPU).squeeze();
    auto matches2 = output->elements()[1].toTensor().to(torch::kCPU).squeeze();
    auto matching_scores1  = output->elements()[2].toTensor().to(torch::kCPU).squeeze();
    auto matching_scores2 = output->elements()[3].toTensor().to(torch::kCPU).squeeze();

    auto valid1 = (matches1 > -1);  
    auto valid2 = (matches2 > -1); 

    auto mkpts1 = torch::nonzero(valid1);
    auto mkpts2 = torch::nonzero(valid2);

    //GET MATCHES
    std::vector<cv::DMatch> mvec_dmatches;
    for (int i = 0; i < mkpts1.size(0); i++) {
        cv::DMatch dmatch = cv::DMatch();
        dmatch.queryIdx = mkpts1[i][0].item<int>();
        dmatch.trainIdx = mkpts2[i][0].item<int>();
        dmatch.distance = matching_scores1[mkpts1[i][0]].item<float>();
        mvec_dmatches.push_back(dmatch);
    }
    return mvec_dmatches;
}


SPMatcher::SPMatcher(float _nn_thresh):nn_thresh(_nn_thresh)
{

}

void SPMatcher::match(cv::Mat _desc_1, cv::Mat _desc_2, std::vector<cv::DMatch>& matches)
{
    desc_1 = _desc_1;
    desc_2 = _desc_2;
    float min_dist = 10000;
    float max_dist = -1;
    int rows_1 = desc_1.rows;
    int rows_2 = desc_2.rows;
    std::cout<<"rows_1:"<<rows_1<<std::endl;
    for(int i = 0; i < rows_1; i++)
    {
        float bestDist = 256;
        int bestIndex = -1;
        cv::DMatch best_match = cv::DMatch();
        for(int j = 0; j < rows_2; j++)
        { 
            float dis = computeDistance(desc_1.row(i), desc_2.row(j));
            //printf("dist %f", dis);
            if(dis < bestDist)
            {
                bestDist = dis;
                bestIndex = j;
            }
        }
        //printf("best dist %f", bestDist);
        if (bestIndex != -1 && bestDist < nn_thresh)
        {
            best_match.queryIdx = i;
            best_match.trainIdx = bestIndex;
            best_match.distance = bestDist;
            matches.push_back(best_match);
            if(bestDist>max_dist)
                max_dist = bestDist;
            if(bestDist<min_dist)
                min_dist = bestDist;
        }
        else{
            continue;
        }
    }
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    std::cout<<"matches number:"<<matches.size()<<std::endl;
}

float SPMatcher::computeDistance(const cv::Mat &a, const cv::Mat &b){
    //std::cout<<a.size()<<std::endl;
    float dist = (float)cv::norm(a, b, cv::NORM_L2); //TODO dot
    return dist;
}

std::vector<cv::DMatch> SPMatcher::getGoodMathces(std::vector<cv::DMatch> matches){
    std::sort(matches.begin(),matches.end(),cmp);
    std::vector<cv::DMatch> good_matches(matches.begin(),matches.begin()+60);
    return good_matches;
}
             