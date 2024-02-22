//#include "../include/SuperGlue.h"
#include "../include/SuperPoint.h"
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "../Thirdparty/DVision/DVision.h"
#include "../Thirdparty/DVision/BRIEF_Extractor.h"

using namespace std;
using namespace cv;

void drawMatches(cv::Mat img_1, cv::Mat img_2, std::vector<cv::KeyPoint> pts_1, 
      std::vector<cv::KeyPoint> pts_2, std::vector<DMatch> matches, string filename){

  int a, b, c;
	cv::Mat compareKpts;
	cv::hconcat(img_1, img_2, compareKpts);
	cv::cvtColor(compareKpts, compareKpts, cv::COLOR_GRAY2RGB);

  std::vector<cv::KeyPoint> pts_1_ = pts_1;
  std::vector<cv::KeyPoint> pts_2_ = pts_2;

  for(size_t i = 0; i < pts_1_.size(); i++){
    a = (rand() % static_cast<int>(256));
    b = (rand() % static_cast<int>(256));
    c = (rand() % static_cast<int>(256));
    cv::circle(compareKpts, pts_1_[i].pt, 3, cv::Scalar(a, b, c), 1, LINE_AA);
  }

  for(size_t i = 0; i < pts_2_.size(); i++){
    a = (rand() % static_cast<int>(256));
    b = (rand() % static_cast<int>(256));
    c = (rand() % static_cast<int>(256));
    pts_2_[i].pt.x += 640;
    cv::circle(compareKpts, pts_2_[i].pt, 3, cv::Scalar(a, b, c), 1, LINE_AA);
  }
  
	for(size_t i = 0; i < matches.size(); i++){
    a = (rand() % static_cast<int>(256));
    b = (rand() % static_cast<int>(256));
    c = (rand() % static_cast<int>(256));
		cv::Point p1 = pts_1_[matches[i].queryIdx].pt;
		cv::Point p2 = pts_2_[matches[i].trainIdx].pt;
		//cv::circle(compareKpts, p1, 2, cv::Scalar(a, b, 255), 2);
		//cv::circle(compareKpts, p2, 2, cv::Scalar(0, 0, 255), 2);
   /* line( outImg,
          Point(cvRound(pt1.x*draw_multiplier), cvRound(pt1.y*draw_multiplier)),
          Point(cvRound(dpt2.x*draw_multiplier), cvRound(dpt2.y*draw_multiplier)),
          color, matchesThickness, LINE_AA, draw_shift_bits );*/
		cv::line(compareKpts, p1, p2, cv::Scalar(a, b, c), 1, LINE_AA);
	}
  std::string image_path0 = "/home/yutong/sp_c/plots/" + filename + ".png";
  cv::imwrite(image_path0.c_str(), compareKpts);
}

bool cmp(DMatch a,DMatch b){
    return a.distance < b.distance;
}

bool cmp2(cv::KeyPoint a,cv::KeyPoint b){
    return a.response < b.response;
}

float computeDistance(const cv::Mat &a, const cv::Mat &b){
    //std::cout<<a.size()<<std::endl;
    float dist = (float)cv::norm(a, b, cv::NORM_L2); //TODO dot
    return dist;
}

float computeDistance(const DVision::BRIEF::bitset &a, const DVision::BRIEF::bitset &b){
    return (a^b).count();
}

void match(cv::Mat _desc_1, cv::Mat _desc_2, std::vector<cv::DMatch>& matches)
{
    cv::Mat desc_1 = _desc_1;
    cv::Mat desc_2 = _desc_2;
    float min_dist = 10000;
    float max_dist = -1;
    int rows_1 = desc_1.rows;
    int rows_2 = desc_2.rows;
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
        if (bestIndex != -1 && bestDist < 256)
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

void match(vector<DVision::BRIEF::bitset> desc_1, vector<DVision::BRIEF::bitset> desc_2, std::vector<cv::DMatch>& matches)
{
    float min_dist = 10000;
    float max_dist = -1;
    int rows_1 = desc_1.size();
    int rows_2 = desc_2.size();
    for(int i = 0; i < rows_1; i++)
    {
        float bestDist = 256;
        int bestIndex = -1;
        cv::DMatch best_match = cv::DMatch();
        for(int j = 0; j < rows_2; j++)
        { 
            float dis = computeDistance(desc_1[i], desc_2[j]);
            //printf("dist %f", dis);
            if(dis < bestDist)
            {
                bestDist = dis;
                bestIndex = j;
            }
        }
        //printf("best dist %f", bestDist);
        if (bestIndex != -1 && bestDist < 80)
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
    std::cout<<"brief matches number:"<<matches.size()<<std::endl;
}

std::vector<cv::DMatch> getGoodMathces(std::vector<cv::DMatch> matches){
    std::sort(matches.begin(),matches.end(),cmp);
    std::vector<cv::DMatch> good_matches(matches.begin(),matches.begin()+60);
    return good_matches;
}

std::vector<size_t> sort_indexes(const std::vector<cv::KeyPoint> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  std::stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1].response > v[i2].response;});

  return idx;
}


int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "usage: feature_extraction img1 img2" << std::endl;
    return 1;
  }
  //-- 读取图像
  cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr);
  std::cout << "img size:" << img_1.size() << std::endl;

  cv::resize(img_1, img_1, cv::Size(640, 480), cv::INTER_AREA);
  std::cout << "img size:" << img_1.size() << std::endl;
  cv::resize(img_2, img_2, cv::Size(640, 480), cv::INTER_AREA);

  cv::cvtColor(img_1, img_1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img_2, img_2, cv::COLOR_BGR2GRAY);

  std::cout<<"put sp net on cuda"<<std::endl;
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    
  std::string weights_path = "/home/yutong/sp_c/weights/superpoint.pt";
  std::shared_ptr<SuperPointNet> net(new SuperPointNet(), std::default_delete<SuperPointNet>());
  torch::load(net, weights_path); 
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "sp weight Loaded cost = " << 1000*time_used.count() << " ms. " << std::endl;
  // = std::make_shared<SuperPointNet>();
  SPDetector detector(net,weights_path, 4, 0.01,true);
  //SPDetector* detector = new SPDetector(weights_path, 4, 0.1,true);
  //SPDetector detector(weights_path, 4, 0.01,true);
  std::vector<cv::KeyPoint> pts_0;
  cv::Mat descriptors_0;
  std::vector<cv::KeyPoint> pts_1;
  cv::Mat descriptors_1;
  std::vector<cv::KeyPoint> pts_2;
  cv::Mat descriptors_2;
  //pts_1.clear();
  //detector->detect(img_1, pts_1, descriptors_1);
  cv::Mat compressed_descriptors1;
  cv::Mat compressed_descriptors2;
  //bool dec0 = detector.detect(img_1, pts_0, descriptors_0, compressed_descriptors1, 600);
  bool dec = detector.detect(img_1, pts_1, descriptors_1, compressed_descriptors1, 1000);
  bool dec2 = detector.detect(img_2, pts_2, descriptors_2, compressed_descriptors2, 1000);

  Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L1, true));
  vector<DMatch> matches;                   // 存储匹配点的向量
  vector<DMatch> matches_bi; 
  std::vector<DMatch> good_matches;
  //std::vector<DMatch> good_matches_bi;
  matcher->match(descriptors_1, descriptors_2, matches);
  good_matches = getGoodMathces(matches);
 // matcher->match(descriptors_1_bi, descriptors_2_bi, matches_bi);
 //good_matches_bi = getGoodMathces(matches_bi);

  drawMatches(img_1, img_2, pts_1, pts_2, matches, "tum_sp_float");
  drawMatches(img_1, img_2, pts_1, pts_2, good_matches, "tum_sp_float_good");
  //drawMatches(img_1, img_2, pts_1, pts_2, matches_bi, "tum_sp_binary");
  //drawMatches(img_1, img_2, pts_1, pts_2, good_matches_bi, "tum_sp_binary_good");

 /*std::string weights_path_glue = "/home/yutong/sp_c/weights/superglue_indoor_cuda.pt";
  float thre = 0.3;
  SPGlue* glue = new SPGlue(weights_path_glue, thre, true);
  int N = std::min((int)pts_2.size(),300);
        
  cv::Mat descriptors_2_;  
  std::vector<cv::KeyPoint> p2;
  cv::Mat pointmatrix2(N, 2, CV_32F); // [2 n]
  cv::Mat scoresmatrix2(N, 1, CV_32F); //[1 n]
  for (size_t i=0;i<N;i++) {
      p2.push_back(pts_2[i]);
      std::cout<<pts_2[i].response<<",";
      descriptors_2_.push_back(descriptors_2.row(i));
      //cv::Mat v = descriptors_2.row(i);
      //for(size_t j=0;j<256;j++)
         // descriptors_2_.at<float>(i,j) = v.at<float>(j);
  }*/

  /*std::vector<cv::DMatch> mvec_dmatches = glue->match(pts_1, pts_2, descriptors_1, descriptors_2);
  Mat outimg1;
  drawMatches(img_1, pts_1, img_2, pts_2, mvec_dmatches, outimg1);
  std::cout<<mvec_dmatches.size()<<std::endl;
  cv::imshow("x", outimg1);
  cv::imwrite("match.png", outimg1);
  cv::cvtColor(img_2, img_2, cv::COLOR_GRAY2RGB);
  int a, b, c;
  for(size_t i = 0; i < pts_2.size(); i++){
    a = (rand() % static_cast<int>(256));
    b = (rand() % static_cast<int>(256));
    c = (rand() % static_cast<int>(256));
    cv::circle(img_2, pts_2[i].pt, 3, cv::Scalar(a, b, c), 1, LINE_AA);
  }
  cv::imwrite("key_im2.png", img_2);
  cv::waitKey(0);*/

  //drawKeypoints(img_1, pts_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  //cv::imwrite("sp_nms_10_3.png", outimg1);

  /*cv::Mat descriptors_1_bi;
  cv::threshold(descriptors_1, descriptors_1_bi, 0, 1, cv::THRESH_BINARY);
  cv::Mat descriptors_2_bi;
  cv::threshold(descriptors_2, descriptors_2_bi, 0, 1, cv::THRESH_BINARY);

  Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L1, true));
  vector<DMatch> matches;                   // 存储匹配点的向量
  vector<DMatch> matches_bi; 
  std::vector<DMatch> good_matches;
  std::vector<DMatch> good_matches_bi;
  matcher->match(descriptors_1, descriptors_2, matches);
  matcher->match(descriptors_1_bi, descriptors_2_bi, matches_bi);
  //match(descriptors_1, descriptors_2, matches);
  good_matches = getGoodMathces(matches);
  //match(descriptors_1_bi, descriptors_2_bi, matches_bi);
  good_matches_bi = getGoodMathces(matches_bi);


  

  //-- 第五步:绘制匹配结果

  drawMatches(img_1, img_2, pts_1, pts_2, matches, "sp_float");
  drawMatches(img_1, img_2, pts_1, pts_2, good_matches, "sp_float_good");
  drawMatches(img_1, img_2, pts_1, pts_2, matches_bi, "sp_binary");
  drawMatches(img_1, img_2, pts_1, pts_2, good_matches_bi, "sp_binary_good");

  Ptr<DescriptorExtractor> descriptor_orb = ORB::create();
  cv::Mat descriptors_1_orb, descriptors_2_orb;
  descriptor_orb->compute(img_1, pts_1, descriptors_1_orb);
  descriptor_orb->compute(img_2, pts_2, descriptors_2_orb);
  vector<DMatch> matches_orb;
  matcher->match(descriptors_1_orb, descriptors_2_orb, matches_orb);
  std::cout<<"match orb:"<<matches_orb.size()<<std::endl;
  std::cout<<"match:"<<matches.size()<<std::endl;
  std::vector<DMatch> good_matches_orb = getGoodMathces(matches_orb);

  std::string BRIEF_PATTERN_FILE= "brief_pattern.yml";
  BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
  vector<DVision::BRIEF::bitset> brief_descriptors_1, brief_descriptors_2;
  extractor(img_1, pts_1, brief_descriptors_1);
  extractor(img_2, pts_2, brief_descriptors_2);
  vector<DMatch> matches_brief; 
  match(brief_descriptors_1, brief_descriptors_2, matches_brief);
  std::vector<DMatch> good_matches_brief = getGoodMathces(matches_brief);
  drawMatches(img_1, img_2, pts_1, pts_2, good_matches_orb, "orb_good");
  drawMatches(img_1, img_2, pts_1, pts_2, good_matches_brief, "brief_good");*/

  /*Mat img_match;
  Mat img_goodmatch;
  Mat img_match_bi;
  Mat img_goodmatch_bi;
  Mat img_match_orb;
  Mat img_goodmatch_orb;
  Mat img_match_brief;
  Mat img_goodmatch_brief;

  drawMatches(img_1, pts_1, img_2, pts_2, matches, img_match);
  std::cout<<"1"<<std::endl;
  drawMatches(img_1, pts_1, img_2, pts_2, good_matches, img_goodmatch);
  std::cout<<"2"<<std::endl;
  drawMatches(img_1, pts_1, img_2, pts_2, matches_bi, img_match_bi);
  std::cout<<"3"<<std::endl;
  drawMatches(img_1, pts_1, img_2, pts_2, good_matches_bi, img_goodmatch_bi);
  std::cout<<"4"<<std::endl;
  

  drawMatches(img_1, pts_1, img_2, pts_2, matches_orb, img_match_orb);
  std::cout<<"5"<<std::endl;
  drawMatches(img_1, pts_1, img_2, pts_2, good_matches_orb, img_goodmatch_orb);
  std::cout<<"6"<<std::endl;
  drawMatches(img_1, pts_1, img_2, pts_2, matches_brief, img_match_brief);
  std::cout<<"7"<<std::endl;
  drawMatches(img_1, pts_1, img_2, pts_2, good_matches_brief, img_goodmatch_brief);
  cv::imwrite("SuperPoint_float_brute_force_mathces.png", img_match);
  cv::imwrite("SuperPoint_bianry_brute_force_mathces.png", img_match_bi);
  cv::imwrite("SuperPoint_float_brute_force_mathces_good.png", img_goodmatch);
  cv::imwrite("SuperPoint_bianry_brute_force_mathces_good.png", img_goodmatch_bi);
  cv::imwrite("orb_mathces.png", img_match_orb);
  cv::imwrite("brief_mathces.png", img_match_brief);
  cv::imwrite("orb_mathces_good.png", img_goodmatch_orb);
  cv::imwrite("brief_mathces_good.png", img_goodmatch_brief);*/
  //imshow("ORB features", outimg2);
  //waitKey(0);
  /*imshow("SP all matches", img_match);
  imshow("SP good matches", img_goodmatch);
  imshow("SP binary all matches", img_match_bi);
  imshow("SP binary good matches", img_goodmatch_bi);
  waitKey(0);*/

  /*cv::Mat mask1 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
  cv::Mat mask2 = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
  for(size_t i=0; i<pts_1.size(); i++){
    mask1.at<uchar>(int(pts_1[i].pt.y), int(pts_1[i].pt.x)) = 1;
    mask1.at<uchar>(int(pts_1[i].pt.y)+1, int(pts_1[i].pt.x)) = 1;
    mask1.at<uchar>(int(pts_1[i].pt.y)-1, int(pts_1[i].pt.x)) = 1;
    mask1.at<uchar>(int(pts_1[i].pt.y), int(pts_1[i].pt.x)+1) = 1;
    mask1.at<uchar>(int(pts_1[i].pt.y), int(pts_1[i].pt.x)-1) = 1;
  }
  for(size_t i=0; i<pts_2.size(); i++){
    mask2.at<uchar>(int(pts_2[i].pt.y), int(pts_2[i].pt.x)) = 1;
  }
  cv::Mat mask12=mask1.mul(mask2);
  std::vector<cv::KeyPoint> pts_12;
  cv::Mat nonZeroCoordinates;
  cv::findNonZero(mask12, nonZeroCoordinates);
  
  for (int i = 0; i < nonZeroCoordinates.total(); i++ ) {
    pts_12.push_back(cv::KeyPoint(nonZeroCoordinates.at<cv::Point>(i).x, nonZeroCoordinates.at<cv::Point>(i).y, 1, -1, 0));
  }*/
/*  cv::Mat mask1 = cv::Mat::zeros(cv::Size(640, 480), CV_32FC1);
  cv::Mat mask2 = cv::Mat::zeros(cv::Size(640, 480), CV_32FC1);
	for(size_t i=0; i<pts_1.size(); i++){
		mask1.at<float>(int(pts_1[i].pt.y), int(pts_1[i].pt.x)) = 1;
	}
	for(size_t i=0; i<pts_2.size(); i++){
		mask2.at<float>(int(pts_2[i].pt.y), int(pts_2[i].pt.x)) = pts_2[i].response;
	}
	cv::Mat mask12=mask1.mul(mask2);
	std::vector<cv::KeyPoint> pts_12;
	for(int i = 0; i < mask12.rows; i++){
		for(int j = 0; j < mask12.cols; j++){
			if(mask12.at<float>(i, j)>0)
				pts_12.push_back(cv::KeyPoint(j,i,8,-1,mask12.at<float>(i, j)));
		}
	}

  cv::Mat outImage;
  cv::drawKeypoints(img_1, pts_1, outImage);
  cv::imshow("keypoints",outImage);
  cv::waitKey(0);
  cv::Mat outImage1;
  cv::drawKeypoints(img_1, pts_2, outImage1);
  cv::imshow("keypoints",outImage1);
  cv::waitKey(0);
  cv::Mat outImage2;
  cv::drawKeypoints(img_1, pts_12, outImage2);
  cv::imshow("keypoints",outImage2);
  cv::waitKey(0);*/

 /* std::vector<size_t> indexes_2 = sort_indexes(pts_2);
  std::vector<cv::KeyPoint> window_pts_2;
  cv::Mat window_descriptors_2;
  int num_pts2 = std::min(500, int(indexes_2.size()));
  for(size_t i=0; i<num_pts2; ++i){
    window_pts_2.push_back(pts_2[indexes_2[i]]);
    //std::cout<<window_pts_2[i].response<<",";
    window_descriptors_2.push_back(descriptors_2.row(indexes_2[i]));
  }*/

 // std::string weights_path_glue = "/home/yutong/sp_c/weights/superglue_indoor_trace.pt";
 // SPGlue* glue = new SPGlue(weights_path_glue, true); //cuda is false as default
 // std::vector<cv::DMatch> mvec_dmatches = glue->match(pts_1, pts_2, descriptors_1, descriptors_2);

  /*std::ofstream myfile;
  myfile.open ("/home/yutong/slambook2/test_sp/csv/cos_sim.csv");
  //myfile << "distance" << std::endl;
  myfile << "cos_sim" << "," << "score" << std::endl;

  for(size_t k=0; k<mvec_dmatches.size(); k++){
    cv::DMatch dmatch = mvec_dmatches[k];
    cv::Mat d1 = descriptors_1.row(dmatch.queryIdx);
    cv::Mat d2 = descriptors_2.row(dmatch.trainIdx);
    //float dist = (float)cv::norm(d1, d2, cv::NORM_L1);
    float cosSim = d1.dot(d2) / (cv::norm(d1) * cv::norm(d2));
    float score = dmatch.distance;
    myfile << cosSim << "," << score << std::endl;
  }
  for(size_t i=0; i<350; i++){
    int ind_1 = std::rand() % 200 +1;
    int ind_2 = std::rand() % 200 +1;
    cv::Mat d1 = descriptors_1.row(ind_1);
    cv::Mat d2 = descriptors_2.row(ind_2);
    float dist = (float)cv::norm(d1, d2, cv::NORM_L2);
    myfile << dist << std::endl;
  }


  myfile.close();*/

 /* cv::Mat img_match;
  cv::drawMatches(img_1, pts_1, img_2, pts_2, mvec_dmatches, img_match);
  cv::imshow("spglue c++: all matches", img_match);
  cv::waitKey(0);*/

  return 0;
}
