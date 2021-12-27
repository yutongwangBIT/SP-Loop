#include "SuperGlue.h"


const int c1 = 64;
const int c2 = 64;
const int c3 = 128;
const int c4 = 128;
const int c5 = 256;
const int d1 = 256;


bool cmp(cv::KeyPoint a,cv::KeyPoint b){
    return a.response < b.response;
}


torch::Tensor max_pool(torch::Tensor x, int nms_dist){
    return torch::nn::functional::max_pool2d(x, torch::nn::functional::MaxPool2dFuncOptions(2*nms_dist+1).stride(1).padding(nms_dist));
}
void simple_nms(torch::Tensor& input_tensor, int nms_dist){
    auto mask = torch::eq(input_tensor, max_pool(input_tensor,nms_dist));
    //std::cout<<"msak::"<<mask<<std::endl;
    auto zeros = torch::zeros_like(input_tensor);
    for(auto i=0;i<0;i++){
        auto supp_mask = torch::ge(max_pool(mask.to(torch::kFloat),nms_dist),0);
        auto supp_tensor = torch::where(supp_mask,zeros,input_tensor);
        auto new_max_mask = torch::eq(supp_tensor, max_pool(supp_tensor,nms_dist));
        mask = torch::__or__(mask, torch::__and__(new_max_mask, torch::logical_not(supp_mask)));
    }
    input_tensor = torch::where(mask, input_tensor, zeros);
}

void normalize_keypoints(torch::Tensor &kpts){
    int height = 480;
    int width = 640;
    auto one = torch::tensor(1);
    auto size = torch::stack({one*width, one*height});
    auto center = size / 2;
    auto scaling  = torch::max(size)*0.7;
    //std::cout<<scaling<<std::endl;
    kpts = (kpts - center.to(torch::kCUDA)) / scaling.to(torch::kCUDA);
}

torch::Tensor attention(torch::Tensor query, torch::Tensor key, torch::Tensor value){
    auto dim = query.size(1);
    auto scores = torch::einsum("bdhn,bdhm->bhnm", {query, key}) / 8.0;
    auto prob = torch::nn::functional::softmax(scores, -1);
    return torch::einsum("bhnm,bdhm->bdhn", {prob, value});
}   
torch::nn::Sequential MLP(int channels[], int size_c){
    torch::nn::Sequential seq;
    for(size_t i=1; i<size_c;i++){
        seq->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions( *(channels+i-1), *(channels+i), 1).stride(1).bias(true)));
        if( i < size_c-1){
            seq->push_back(torch::nn::BatchNorm1d(*(channels+i)));
            seq->push_back(torch::nn::ReLU());
        }
    }
   // std::cout<<"seq:"<<seq<<std::endl;
    return seq;
}
torch::Tensor log_sinkhorn_iterations(torch::Tensor Z, torch::Tensor log_mu, torch::Tensor log_nu, int iters){
    //Perform Sinkhorn Normalization in Log-space for stability
    auto u = torch::zeros_like(log_mu);
    auto v = torch::zeros_like(log_nu);
    for(size_t i=0;i<iters;i++){
        u = log_mu - torch::logsumexp(Z+v.unsqueeze(1), 2);
        v = log_nu - torch::logsumexp(Z+u.unsqueeze(2), 1);
    }
    return Z + u.unsqueeze(2) + v.unsqueeze(1);
}
torch::Tensor log_optimal_transport(torch::Tensor scores, torch::Tensor alpha, int iters){
    auto b = scores.size(0);
    auto m = scores.size(1);
    auto n = scores.size(2);
    auto one = torch::tensor(1);
    auto ms = (m*one).to(scores); 
    auto ns = (n*one).to(scores); 
    auto bins0 = alpha.expand({b, m, 1});
    auto bins1 = alpha.expand({b, 1, n});
    alpha = alpha.expand({b, 1, 1});
    auto couplings = torch::cat({torch::cat({scores, bins0},-1), torch::cat({bins1, alpha},-1)},1);
    auto norm = -(ms + ns).log();
    norm.to(torch::kCUDA);
    auto log_mu = torch::cat({norm.expand(m), ns.log().reshape(1)+norm});
    auto log_nu = torch::cat({norm.expand(n), ms.log().reshape(1)+norm});
    log_mu = log_mu.expand({b, -1});
    log_nu = log_nu.expand({b, -1});
    auto Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters);
    Z = Z - norm;
    return Z;
}
torch::Tensor arange_like(torch::Tensor x, int dim){
    return torch::ones(x.size(dim)).cumsum(0) - 1;
}

KeypointEncoder::KeypointEncoder(){
  int channels[] = {3, 32, 64, 128, 256, 256}; //[3] + layers + [feature_dim]
  encoder = register_module("encoder", MLP(channels, std::end(channels)-std::begin(channels)));
  //torch::nn::init::constant_(encoder[-1].bias, 0.0);
}

torch::Tensor KeypointEncoder::forward(torch::Tensor kpts, torch::Tensor scores){
    torch::Tensor a = kpts.transpose(1, 2);
    torch::Tensor b = scores.unsqueeze(1);
    return encoder->forward(torch::cat({a,b}, 1));
}
MultiHeadedAttention::MultiHeadedAttention(){
    merge = register_module("merge", torch::nn::Conv1d(torch::nn::Conv1dOptions( 256, 256, 1).stride(1)));
    for(size_t i=0; i<3; i++){ //length of layer_names
        proj->push_back(merge->clone());
    }
    register_module("proj", proj);
    num_heads = 4;
    d_model = 256;
    dim = 64; //d_model/num_heads

}

torch::Tensor MultiHeadedAttention::forward(torch::Tensor query, torch::Tensor key, torch::Tensor value){
    auto batch_dim = query.size(0);
    auto l0 = proj[0]->as<torch::nn::Conv1d>();
    auto l1 = proj[1]->as<torch::nn::Conv1d>();
    auto l2 = proj[2]->as<torch::nn::Conv1d>();
    auto q = l0->forward(query).view({batch_dim, dim, num_heads, -1}); //1, 64, 4, n
    auto k = l1->forward(key).view({batch_dim, dim, num_heads, -1});
    auto v = l2->forward(value).view({batch_dim, dim, num_heads, -1});
    auto x = attention(q, k, v);
    return merge->forward(x.contiguous().view({batch_dim,dim*num_heads,-1}));
}


AttentionalPropagation::AttentionalPropagation(){
    int channels[] = {256*2, 256*2, 256};
    attn = register_module("attn", std::make_shared<MultiHeadedAttention>());
    mlp = register_module("mlp", MLP(channels, std::end(channels)-std::begin(channels)));
}
torch::Tensor AttentionalPropagation::forward(torch::Tensor x, torch::Tensor source){
    torch::Tensor message = attn->forward(x, source, source);
    return mlp->forward(torch::cat({x, message},1));
}

AttentionalGNN::AttentionalGNN(){
    for(size_t i=0; i<18;i++){
        if(i%2==0){
            layer_names.push_back("self");
        }
        else
            layer_names.push_back("cross");
    }
    std::cout<<layer_names<<std::endl;
    for(size_t i=0; i<18; i++){ //length of layer_names
        layers->push_back(AttentionalPropagation());
    }
    //std::cout<<layers<<std::endl;
    register_module("layers", layers);
}
void AttentionalGNN::forward(torch::Tensor &desc0, torch::Tensor &desc1){
    for(size_t i=0;i<18;i++){
        torch::Tensor src0, src1;
        if(layer_names[i]=="cross"){
            src0 = desc1;
            src1 = desc0;
        }
        else{
            src0 = desc0;
            src1 = desc1;
        }
        auto layer = layers[i]->as<AttentionalPropagation>();
        auto delta0 = layer->forward(desc0, src0);
        auto delta1 = layer->forward(desc1, src1);
        desc0 = desc0 + delta0;
        desc1 = desc1 + delta1;
    }
}

SuperGlueNet::SuperGlueNet(float _thres)
    :final_proj(torch::nn::Conv1dOptions( 256, 256, 1).stride(1).bias(true))
{
    kenc = register_module<KeypointEncoder>("kenc", std::make_shared<KeypointEncoder>());
    gnn = register_module<AttentionalGNN>("gnn", std::make_shared<AttentionalGNN>());
    register_module("final_proj", final_proj);
    match_threshold = _thres;
}


std::vector<torch::Tensor> SuperGlueNet::forward(std::vector<torch::Tensor> x) {
    auto kpts0 = x[0];
    auto score0 = x[1];
    auto desc0 = x[2];
    auto kpts1 = x[3];
    auto score1 = x[4];
    auto desc1 = x[5];

    //normalize
    normalize_keypoints(kpts0);
    normalize_keypoints(kpts1);
    //keypointencoder
    desc0 = desc0 + kenc->forward(kpts0, score0);
    desc1 = desc1 + kenc->forward(kpts1, score1);
    //std::cout<<desc0<<std::endl;
    gnn->forward(desc0,desc1);
    auto mdesc0 = final_proj->forward(desc0);
    auto mdesc1 = final_proj->forward(desc1);
    //std::cout<<mdesc0<<std::endl;
    //std::cout<<mdesc1.sizes()<<std::endl;
    auto scores = torch::einsum("bdn,bdm->bnm",{mdesc0, mdesc1});
    scores = (scores / 16.0).to(torch::kCUDA);
    auto bin_score = torch::tensor(1.).to(torch::kCUDA);;
    bin_score = bin_score.set_requires_grad(true);
    scores = log_optimal_transport(scores, bin_score, 100); //'sinkhorn_iterations'
    //Get the matches with score above "match_threshold".
    //std::cout<<"match_threshold:"<<match_threshold<<std::endl;
    auto scores_split = scores.split_with_sizes({scores.size(1)-1, 1}, 1)[0].split_with_sizes({scores.size(2)-1, 1}, 2)[0];
    //std::cout<<"scores_split sizes:"<<scores_split.sizes()<<std::endl;
    auto max0 = scores_split.max(2); //scores[:, :-1, :-1].max(2) libtorch cannot do the same thing, without split, the index are always be the last one
    auto max1 = scores_split.max(1);
    auto indices0 = std::get<1>(max0);
    auto indices1 = std::get<1>(max1);
    auto a0 = arange_like(indices0, 1);
    auto a1 = arange_like(indices1, 1);
    a0 = a0.reshape({1, a0.size(0)});
    a1 = a1.reshape({1, a1.size(0)});
    auto g0 = indices1.gather(1, indices0);
    auto g1 = indices0.gather(1, indices1);
    auto mutual0 = a0.to(torch::kCUDA) == g0.to(torch::kCUDA);
    auto mutual1 = a1.to(torch::kCUDA) == g1.to(torch::kCUDA);
    auto zeros = torch::tensor(0.);
    auto mscores0 = torch::where(mutual0, std::get<0>(max0).exp().to(torch::kCUDA), zeros.to(torch::kCUDA));
    auto mscores1 = torch::where(mutual1, mscores0.gather(1, indices1), zeros.to(torch::kCUDA));
    auto valid0 = torch::__and__(mutual0, (mscores0 > match_threshold));
    auto valid1 = torch::__and__(mutual1, valid0.gather(1, indices1));
    indices0 = torch::where(valid0, indices0, torch::tensor(-1).to(torch::kCUDA));
    indices1 = torch::where(valid1, indices1, torch::tensor(-1).to(torch::kCUDA));
    std::vector<torch::Tensor> ret;
    ret.push_back(indices0);
    ret.push_back(indices1);
    ret.push_back(mscores0);
    ret.push_back(mscores1);
    return ret;
}

SPGlue::SPGlue(std::string _weights_path, float _match_threshold, float _sp_glue_score_thres, bool _cuda)
:weights_path(_weights_path),match_threshold(_match_threshold), sp_glue_score_thres(_sp_glue_score_thres),cuda(_cuda)
{
    net = std::make_shared<SuperGlueNet>(match_threshold);
    torch::load(net, weights_path); 
    std::cout<<"weight Loaded"<<std::endl;
    net->eval();
    bool use_cuda = cuda && torch::cuda::is_available();
    if (use_cuda){
        std::cout<<"use_cuda"<<std::endl;
        device_type = torch::kCUDA;
    }
    else
        device_type = torch::kCPU;
    torch::Device device(device_type);
    net->to(device);
}
void SPGlue::match(std::vector<cv::KeyPoint> pts_1, std::vector<cv::KeyPoint> pts_2,
                          cv::Mat descriptors_1,cv::Mat descriptors_2,  
                          std::vector<cv::KeyPoint> pts_2_norm, 
                          std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<float> &matched_scores, 
                          std::vector<uchar> &status){
    std::cout<<"1:"<<pts_1.size()<<"2:"<<pts_2.size()<<std::endl;
    torch::Device device(device_type);
    cv::Mat pointmatrix1(pts_1.size(), 2, CV_32F); // [2 n]
    cv::Mat scoresmatrix1(pts_1.size(), 1, CV_32F);
    int column1 = 0;
    for (auto& kp: pts_1) {
        status.push_back(0);
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

    std::vector<torch::Tensor> inputs; //multiple inputs defined in script
    inputs.push_back(input_kpts1.to(device));
    inputs.push_back(input_scores1.to(device));
    inputs.push_back(input_desc1.to(device));
    inputs.push_back(input_kpts2.to(device));
    inputs.push_back(input_scores2.to(device));
    inputs.push_back(input_desc2.to(device));

    //std::cout<<"put input on cuda"<<std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    std::vector<torch::Tensor> output = net->forward(inputs);
    //std::cout<<output<<std::endl;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "SP GLUE cost = " << 1000*time_used.count() << " ms. " << std::endl;

    auto matches1 = output[0].squeeze();
    auto matches2 = output[1].squeeze();
    auto matching_scores1  = output[2].squeeze();
    auto matching_scores2 = output[3].squeeze();

    auto valid1 = (matches1 > -1);  
    //auto valid2 = (matches2 > -1); 
    auto mkpts1 = torch::nonzero(valid1);
    //auto x = torch::nonzero(valid2);
    auto mkpts2 = torch::masked_select(matches1, valid1);
    matching_scores1 = torch::masked_select(matching_scores1, valid1);
    //GET MATCHES
    //std::cout<<"status size:"<<status.size()<<std::endl;
    std::cout<<"match size:"<<mkpts1.size(0)<<std::endl;
    for (int i = 0; i < mkpts1.size(0); i++) {
        int ind1 = mkpts1[i][0].item<int>();
        int ind2 = mkpts2[i].item<int>();
        status[ind1] = 1;
        matched_2d_old.push_back(pts_2[ind2].pt);
        matched_2d_old_norm.push_back(pts_2_norm[ind2].pt);
        matched_scores.push_back(matching_scores1[i].item<float>());
    }     
}

void SPGlue::match(std::vector<cv::KeyPoint> pts_1, std::vector<cv::KeyPoint> pts_2,
                          cv::Mat descriptors_1,cv::Mat descriptors_2,  std::vector<cv::KeyPoint> pts_1_norm, 
                          std::vector<cv::KeyPoint> pts_2_norm, std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,std::vector<cv::Point2f> &matched_2d_cur,
                          std::vector<cv::Point2f> &matched_2d_cur_norm,
                          std::vector<float> &matched_scores, std::vector<uchar> &status){
    torch::Device device(device_type);
    cv::Mat pointmatrix1(pts_1.size(), 2, CV_32F); // [2 n]
    cv::Mat scoresmatrix1(pts_1.size(), 1, CV_32F);
    int column1 = 0;
    for (auto& kp: pts_1) {
        status.push_back(0);
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

    std::vector<torch::Tensor> inputs; //multiple inputs defined in script
    inputs.push_back(input_kpts1.to(device));
    inputs.push_back(input_scores1.to(device));
    inputs.push_back(input_desc1.to(device));
    inputs.push_back(input_kpts2.to(device));
    inputs.push_back(input_scores2.to(device));
    inputs.push_back(input_desc2.to(device));

    //std::cout<<"put input on cuda"<<std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    std::vector<torch::Tensor> output = net->forward(inputs);
    //std::cout<<output<<std::endl;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    //std::cout << "SP GLUE cost = " << 1000*time_used.count() << " ms. " << std::endl;

    auto matches1 = output[0].squeeze();
    auto matches2 = output[1].squeeze();
    auto matching_scores1  = output[2].squeeze();
    auto matching_scores2 = output[3].squeeze();

    auto valid1 = (matches1 > -1);  
    //auto valid2 = (matches2 > -1); 
    auto mkpts1 = torch::nonzero(valid1);
    //auto x = torch::nonzero(valid2);
    auto mkpts2 = torch::masked_select(matches1, valid1);
    matching_scores1 = torch::masked_select(matching_scores1, valid1);
    //GET MATCHES
    //std::cout<<"status size:"<<status.size()<<std::endl;
    //std::cout<<"match size:"<<mkpts1.size(0)<<std::endl;
    for (int i = 0; i < mkpts1.size(0); i++) {
        int ind1 = mkpts1[i][0].item<int>();
        int ind2 = mkpts2[i].item<int>();
        status[ind1] = 1;
        matched_2d_cur.push_back(pts_1[ind1].pt);
        matched_2d_cur_norm.push_back(pts_1_norm[ind1].pt);
        matched_2d_old.push_back(pts_2[ind2].pt);
        matched_2d_old_norm.push_back(pts_2_norm[ind2].pt);
        matched_scores.push_back(matching_scores1[i].item<float>());
    }           
    //GET MATCHES
    //std::cout<<"status size:"<<status.size()<<std::endl;
    //std::cout<<"match size:"<<matches1.sizes()<<std::endl;
   /* for (int i = 0; i < matches1.size(0); i++) {
        int ind1 = matches1[i].item<int>();
        int ind2 = matches2[i].item<int>();
        status[ind1] = 1;
        matched_2d_old.push_back(pts_2[ind2].pt);
        matched_2d_old_norm.push_back(pts_2_norm[ind2].pt);
        matched_scores.push_back(matching_scores1[i].item<float>());
    }   */    
}
std::vector<cv::DMatch> SPGlue::match(std::vector<cv::KeyPoint> pts_1, std::vector<cv::KeyPoint> pts_2, cv::Mat descriptors_1, cv::Mat descriptors_2){
    torch::Device device(device_type);
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

    std::vector<torch::Tensor> inputs; //multiple inputs defined in script
    inputs.push_back(input_kpts1.to(device));
    inputs.push_back(input_scores1.to(device));
    inputs.push_back(input_desc1.to(device));
    inputs.push_back(input_kpts2.to(device));
    inputs.push_back(input_scores2.to(device));
    inputs.push_back(input_desc2.to(device));

    std::cout<<"put input on cuda"<<std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    std::vector<torch::Tensor> output = net->forward(inputs);
    //std::cout<<output<<std::endl;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    //std::cout << "SP GLUE cost = " << 1000*time_used.count() << " ms. " << std::endl;

    auto matches1 = output[0].squeeze();
    auto matches2 = output[1].squeeze();
    auto matching_scores1  = output[2].squeeze();
    auto matching_scores2 = output[3].squeeze();

    auto valid1 = (matches1 > -1);  
    auto valid2 = (matches2 > -1); 
    auto mkpts1 = torch::nonzero(valid1);
    auto x = torch::nonzero(valid2);
    std::cout<<x<<std::endl;
    auto mkpts2 = torch::masked_select(matches1, valid1);
    std::cout<<mkpts2<<std::endl;
    //GET MATCHES
    std::vector<cv::DMatch> mvec_dmatches;
    for (int i = 0; i < mkpts1.size(0); i++) {
        cv::DMatch dmatch = cv::DMatch();
        dmatch.queryIdx = mkpts1[i][0].item<int>();
        dmatch.trainIdx = mkpts2[i].item<int>();
       // dmatch.trainIdx = x[i][0].item<int>();
        dmatch.distance = matching_scores1[mkpts1[i][0]].item<float>();
        mvec_dmatches.push_back(dmatch);
    }
    return mvec_dmatches;
    
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


std::vector<torch::Tensor> SuperPointNet::forward(torch::Tensor x,int nms_dist, bool bNms) {

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

    //if(use_nms)
        simple_nms(semi, nms_dist);

    std::vector<torch::Tensor> ret;
    ret.push_back(semi);
    ret.push_back(desc);

    return ret;
}

SPDetector::SPDetector(torch::DeviceType device_type_, int _nms_dist, float _conf_thresh, bool _cuda)
    :device_type(device_type_),nms_dist(_nms_dist), conf_thresh(_conf_thresh), cuda(_cuda)
{
    //net = std::make_shared<SuperPointNet>();
    //torch::load(net, weights_path); 
   /* bool use_cuda = cuda && torch::cuda::is_available();
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
    std::cout << "time to put net on cuda= " << time_used0.count() << " seconds. " << std::endl;*/
}



bool SPDetector::detect(std::shared_ptr<SuperPointNet> net,cv::Mat img, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors, const int c_num_pts){

    cv::Mat img_re;
    //cv::resize(img, img_re, cv::Size(640, 480), cv::INTER_AREA);
    cv::resize(img, img_re, cv::Size(640, 480));
    auto input = torch::from_blob(img_re.clone().data, {1, 1, img_re.rows, img_re.cols}, torch::kByte);
    
    input = input.to(torch::kFloat) / 255;
    torch::Device device(device_type);
    input = input.set_requires_grad(false);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    
    auto out = net->forward(input.to(device), nms_dist);  
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract SP cost = " << 1000*time_used.count() << " ms. " << std::endl;
    
    torch::Tensor mProb = out[0].squeeze(0);  // [H, W] # squeeze(0) leaves the tensor unchanged
    torch::Tensor mDesc = out[1];

    //TicToc t_2;
    //USE TORCH TOPK AVOIDED SEND MPROB DATA INTO VECTOR
    auto kpts = (mProb > conf_thresh);
    kpts = torch::nonzero(kpts); 
    
    auto mask_score = torch::ge(mProb,conf_thresh);
    auto scores = torch::masked_select(mProb, mask_score);
    //std::cout<<"score size:"<<scores.size(0)<<std::endl;
    if(scores.size(0)==0)
        return false;
    auto num_pts = std::min(c_num_pts, int(scores.size(0)));
    auto top = scores.topk(num_pts);
    auto top_values = std::get<0>(top);
    auto top_index = std::get<1>(top);

    auto grid = torch::zeros({1, 1, kpts.size(0), 2});  // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * kpts.slice(1, 1, 2) / mProb.size(1) - 1;  // x
    grid[0][0].slice(1, 1, 2) = 2.0 * kpts.slice(1, 0, 1) / mProb.size(0) - 1;  // y
    auto desc = torch::grid_sampler(mDesc, grid.to(device), 0, 0, true);  // [1, 256, 1, n_keypoints]
    desc = desc.squeeze(0).squeeze(1);  // [256, n_keypoints] Returns a tensor with all the dimensions of input of size 1 removed
    // normalize to 1
    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));
    desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]
    desc = desc.to(torch::kCPU);
    cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data_ptr());
    cv::Mat descriptors_whole = desc_mat.clone();
    for(auto i=0; i<top_index.size(0); ++i){
        int index = top_index[i].item<int>();
        float response = top_values[i].item<float>();
        float x = std::floor(kpts[index][1].item<float>()*(img.cols/(float(img_re.cols))));
        float y = std::floor(kpts[index][0].item<float>()*(img.rows/(float(img_re.rows))));
        pts.push_back(cv::KeyPoint(x, y, 8, -1, response)); //8 is size, can be changed
        descriptors.push_back(descriptors_whole.row(index)); 
    }  
    
    return true;
}

bool SPDetector::detect(std::shared_ptr<SuperPointNet> net,cv::Mat img, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors){
    auto input = torch::from_blob(img.clone().data, {1, 1, img.rows, img.cols}, torch::kByte);
    input = input.to(torch::kFloat) / 255;
    input = input.set_requires_grad(false);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    torch::Device device(device_type);
    auto out = net->forward(input.to(device), nms_dist);  
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract SP cost = " << 1000*time_used.count() << " ms. " << std::endl;
    
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

    auto desc = torch::grid_sampler(mDesc, grid.to(device), 0, 0, true);  // [1, 256, 1, n_keypoints]
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

    return true;
}

bool SPDetector::detectWindow(std::shared_ptr<SuperPointNet> net,cv::Mat img, std::vector<cv::Point2f>& pts, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){

    cv::Mat img_re;
    cv::resize(img, img_re, cv::Size(640, 480));
    auto input = torch::from_blob(img_re.clone().data, {1, 1, img_re.rows, img_re.cols}, torch::kByte);
    input = input.to(torch::kFloat) / 255;
    torch::Device device(device_type);
    input = input.set_requires_grad(false);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto out = net->forward(input.to(device), nms_dist, false);   //no nms
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    //std::cout << "extract SP cost = " << 1000*time_used.count() << " ms. " << std::endl;
    
    torch::Tensor mProb = out[0].squeeze(0);  // [H, W] # squeeze(0) leaves the tensor unchanged
    torch::Tensor mDesc = out[1];
    // std::cout<<mProb<<std::endl;
    //set mask according to pts
    float row = (float)img.rows;
    float col = (float)img.cols;
    torch::Tensor kpts = torch::zeros({pts.size(),2});
    int i = 0;
    for(auto &pt : pts){
        int u = (int)pt.x*640.0/col; //col
        int v = (int)pt.y; //row
        kpts[i][0] = v;
        kpts[i][1] = u;
        float response = mProb[v][u].item<float>();
        keypoints.push_back(cv::KeyPoint(pt.x , pt.y, 8, -1, response)); //use original
        i++;
    }
    auto grid = torch::zeros({1, 1, kpts.size(0), 2});  // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * kpts.slice(1, 1, 2) / mProb.size(1) - 1;  // x
    grid[0][0].slice(1, 1, 2) = 2.0 * kpts.slice(1, 0, 1) / mProb.size(0) - 1;  // y
    auto desc = torch::grid_sampler(mDesc, grid.to(device), 0, 0, true);  // [1, 256, 1, n_keypoints]
    desc = desc.squeeze(0).squeeze(1);  // [256, n_keypoints] Returns a tensor with all the dimensions of input of size 1 removed
    // normalize to 1
    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));
    desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]
    desc = desc.to(torch::kCPU);
    cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data_ptr());
    descriptors = desc_mat.clone();
    return true;
   /* cv::Mat mat_mask_pts = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
    for(auto &pt : pts){
        int u = (int)pt.x*640.0/col; //col
        //std::cout<<"u:"<<u<<",";
        int v = (int)pt.y; //row
        mat_mask_pts.at<uchar>(v,u) = 255;
    }
    auto mask_input = torch::from_blob(mat_mask_pts.clone().data, { mat_mask_pts.rows, mat_mask_pts.cols}, torch::kU8);
    mask_input = mask_input.to(device);
    auto zeros = torch::zeros_like(mProb);
    mProb = torch::where(mask_input==255, mProb, zeros);
    auto kpts = (mProb > 0);
    kpts = torch::nonzero(kpts); 

    if(kpts.size(0)!=pts.size()){
        std::cout<<"kpts size:"<<kpts.size(0)<<",  pts size:"<<pts.size()<<std::endl;
        return false;
    }

    for (int i = 0; i < kpts.size(0); i++) {
        float response = mProb[kpts[i][0]][kpts[i][1]].item<float>();
        float u = kpts[i][1].item<float>() * col / 640.0;
        keypoints.push_back(cv::KeyPoint(u , kpts[i][0].item<float>(), 8, -1, response));
    }

    auto grid = torch::zeros({1, 1, kpts.size(0), 2});  // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * kpts.slice(1, 1, 2) / mProb.size(1) - 1;  // x
    grid[0][0].slice(1, 1, 2) = 2.0 * kpts.slice(1, 0, 1) / mProb.size(0) - 1;  // y
    auto desc = torch::grid_sampler(mDesc, grid.to(device), 0, 0, true);  // [1, 256, 1, n_keypoints]
    desc = desc.squeeze(0).squeeze(1);  // [256, n_keypoints] Returns a tensor with all the dimensions of input of size 1 removed
    // normalize to 1
    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));
    desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]
    desc = desc.to(torch::kCPU);
    cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data_ptr());
    descriptors = desc_mat.clone();
    return true;*/
}

bool SPDetector::detect(std::shared_ptr<SuperPointNet> net, cv::Mat img, std::vector<cv::Point2f>& window_pts, std::vector<cv::KeyPoint>& window_keypoints,
             cv::Mat& window_descriptors, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const int c_num_pts){
    cv::Mat img_re;
    cv::resize(img, img_re, cv::Size(640, 480));
    auto input = torch::from_blob(img_re.clone().data, {1, 1, img_re.rows, img_re.cols}, torch::kByte);
    input = input.to(torch::kFloat) / 255;
    torch::Device device(device_type);
    input = input.set_requires_grad(false);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto out = net->forward(input.to(device), nms_dist, false);   //no nms
    //at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    //AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    //std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    //std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    //std::cout << "extract SP cost = " << 1000*time_used.count() << " ms. " << std::endl;
    
    torch::Tensor mProb = out[0].squeeze(0);  // [H, W] # squeeze(0) leaves the tensor unchanged
    torch::Tensor mDesc = out[1];      


    /////////////////////////////////////////////
    float row = (float)img.rows;
    float col = (float)img.cols;
    torch::Tensor kpts = torch::zeros({window_pts.size(),2});
    int i = 0;
    for(auto &pt : window_pts){
        int u = (int)pt.x*640.0/col; //col
        int v = (int)pt.y; //row
        kpts[i][0] = v;
        kpts[i][1] = u;
        float response = mProb[v][u].item<float>();
        window_keypoints.push_back(cv::KeyPoint(pt.x , pt.y, 8, -1, response)); //use original
        i++;
    }
    auto grid = torch::zeros({1, 1, kpts.size(0), 2});  // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * kpts.slice(1, 1, 2) / mProb.size(1) - 1;  // x
    grid[0][0].slice(1, 1, 2) = 2.0 * kpts.slice(1, 0, 1) / mProb.size(0) - 1;  // y
    auto desc = torch::grid_sampler(mDesc, grid.to(device), 0, 0, true);  // [1, 256, 1, n_keypoints]
    desc = desc.squeeze(0).squeeze(1);  // [256, n_keypoints] Returns a tensor with all the dimensions of input of size 1 removed
    // normalize to 1
    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));
    desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]
    desc = desc.to(torch::kCPU);
    cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data_ptr());
    window_descriptors = desc_mat.clone();         

////////////////////////////////////////////////////////////
    auto kpts2 = (mProb > conf_thresh);
    kpts2 = torch::nonzero(kpts2);
    auto mask_score = torch::ge(mProb,conf_thresh);
    auto scores = torch::masked_select(mProb, mask_score);
    //std::cout<<"score size:"<<scores.size(0)<<std::endl;
    if(scores.size(0)==0)
        return false;
    auto num_pts = std::min(c_num_pts, int(scores.size(0)));
    auto top = scores.topk(num_pts);
    auto top_values = std::get<0>(top);
    auto top_index = std::get<1>(top);

    auto grid2 = torch::zeros({1, 1, kpts2.size(0), 2});  // [1, 1, n_keypoints, 2]
    grid2[0][0].slice(1, 0, 1) = 2.0 * kpts2.slice(1, 1, 2) / mProb.size(1) - 1;  // x
    grid2[0][0].slice(1, 1, 2) = 2.0 * kpts2.slice(1, 0, 1) / mProb.size(0) - 1;  // y
    auto desc2 = torch::grid_sampler(mDesc, grid2.to(device), 0, 0, true);  // [1, 256, 1, n_keypoints]
    desc2 = desc2.squeeze(0).squeeze(1);  // [256, n_keypoints] Returns a tensor with all the dimensions of input of size 1 removed
    // normalize to 1
    auto dn2 = torch::norm(desc2, 2, 1);
    desc2 = desc2.div(torch::unsqueeze(dn, 1));
    desc2 = desc2.transpose(0, 1).contiguous();  // [n_keypoints, 256]
    desc2 = desc2.to(torch::kCPU);
    cv::Mat desc_mat2(cv::Size(desc2.size(1), desc2.size(0)), CV_32FC1, desc2.data_ptr());
    cv::Mat descriptors_whole = desc_mat2.clone();
    for(auto i=0; i<top_index.size(0); ++i){
        int index = top_index[i].item<int>();
        float response = top_values[i].item<float>();
        float x = std::floor(kpts2[index][1].item<float>()*(img.cols/(float(img_re.cols))));
        float y = std::floor(kpts2[index][0].item<float>()*(img.rows/(float(img_re.rows))));
        keypoints.push_back(cv::KeyPoint(x, y, 8, -1, response)); //8 is size, can be changed
        descriptors.push_back(descriptors_whole.row(index)); 
    }  
    return true;
}