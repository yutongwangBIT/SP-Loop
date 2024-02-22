#include "DVision.h"

using namespace DVision;

class BriefExtractor
{
public:
  virtual void operator()(const cv::Mat &im, std::vector<cv::KeyPoint> &keys, std::vector<BRIEF::bitset> &descriptors) const;
  BriefExtractor(const std::string &pattern_file);

  DVision::BRIEF m_brief;
};



