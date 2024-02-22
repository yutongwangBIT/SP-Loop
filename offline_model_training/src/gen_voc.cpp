#include "../include/SuperPoint.h"
#include "../Thirdparty/DBow3_src/src/DBoW3.h"
#include <dirent.h>
using namespace DBoW3;

int main(int argc, char **argv) {
  std::string weights_path = "/home/yutong/sp_c/weights/superpoint.pt";
  SPDetector* detector = new SPDetector(weights_path, 4, 0.01, true);
  std::string dirName = "/home/yutong/data/gen_voc3/";
  DIR *dir;
  dir = opendir(dirName.c_str());
  struct dirent *ent;
  std::vector<cv::Mat> vdescriptors;
  std::vector<cv::Mat> vcomp_descriptors;
  size_t num=0;
  if (dir != NULL) {
      while ((ent = readdir (dir)) != NULL) {
          if(strlen(ent->d_name)<5)
            continue;
          std::string imgPath(dirName + ent->d_name);
          std::cout<<imgPath<<std::endl;
          cv::Mat img = cv::imread(imgPath, CV_LOAD_IMAGE_COLOR);
          if(img.rows==0 || img.cols==0)
            continue;
          std::cout<<img.size()<<std::endl;
          cv::resize(img, img, cv::Size(640, 480), cv::INTER_AREA);
          cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
          std::vector<cv::KeyPoint> pts;
          cv::Mat descriptors;
          cv::Mat compressed_descriptors;
          bool dec = detector->detect(img, pts, descriptors, compressed_descriptors, 1000);
          if(!dec)
            continue;
          cv::Mat dst;
          cv::threshold(descriptors, dst, 0, 1, cv::THRESH_BINARY);
          vcomp_descriptors.push_back(dst);
          size_t num_pts = pts.size();
          num += num_pts;
      }
      closedir (dir);
  } else {
      std::cout<<"not present"<<std::endl;
  }
  std::cout<<"extracted:"<<num<<" features"<<std::endl;
  std::cout<<"done"<<std::endl;
  //myfile.close();
  const int k = 10;
  const int L = 5;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  DBoW3::Vocabulary voc;

  std::cout << "Creating a large " << k << "^" << L << " vocabulary..." << std::endl;
  voc.create(vcomp_descriptors);
  std::cout << "... done!" << std::endl;

  std::cout << "Vocabulary information: " << std::endl
        << voc << std::endl << std::endl;

  // save the vocabulary to disk
  std::cout << std::endl << "Saving vocabulary..." << std::endl;
  voc.save("/home/yutong/sp_c/voc/voc.yml.gz");
  std::cout << "Done" << std::endl;
  return 0;
}
