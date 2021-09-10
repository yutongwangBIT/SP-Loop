#include "keyframe_sp.h"

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// create KeyFrameSP online
KeyFrameSP::KeyFrameSP(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
		           vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
		           vector<double> &_point_id ,SPDetector* _sp, SPGlue* _sp_glue, int _sequence)
{
	sp_detector = _sp;
	sp_glue = _sp_glue;
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;		
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	img_size = image.size();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	//computeWindowBRIEFPoint();
	extractSuperPoints();
	computeWindowPoints();
	draw();
	if(!DEBUG_IMAGE)
		image.release();
}

// load previous KeyFrameSP
KeyFrameSP::KeyFrameSP(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
					cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
					vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, cv::Mat &_descriptors)
{
	time_stamp = _time_stamp;
	index = _index;
	//vio_T_w_i = _vio_T_w_i;
	//vio_R_w_i = _vio_R_w_i;
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
	if (DEBUG_IMAGE)
	{
		image = _image.clone();
		cv::resize(image, thumbnail, cv::Size(80, 60));
	}
	if (_loop_index != -1)
		has_loop = true;
	else
		has_loop = false;
	loop_index = _loop_index;
	loop_info = _loop_info;
	has_fast_point = false;
	sequence = 0;
	keypoints = _keypoints;
	keypoints_norm = _keypoints_norm;
}
void KeyFrameSP::extractSuperPoints(){
	bool dec = sp_detector->detect(image, keypoints, descriptors, 10000);
	binarize_descriptors();
	std::cout<<"int desc:"<<descriptors_converted.size()<<std::endl;
}
void KeyFrameSP::draw(){
	cv::hconcat(image, image, compareKpts);
	cv::cvtColor(compareKpts, compareKpts, cv::COLOR_GRAY2RGB);
	for (size_t i = 0; i < point_2d_uv.size(); i++)
	{
		cv::Point2f Pt = point_2d_uv[i];
		cv::circle(compareKpts, Pt, 2, cv::Scalar(0, 255, 0), 2);
	}
	for (size_t i = 0; i < keypoints.size(); i++)
	{
		cv::Point2f Pt = keypoints[i].pt;
		Pt.x += 752;
		cv::circle(compareKpts, Pt, 2, cv::Scalar(0, 255, 0), 2);
	}
	for (size_t i = 0; i < window_keypoints.size(); i++)
	{
		cv::Point2f Pt = window_keypoints[i].pt;
		cv::circle(compareKpts, Pt, 6, cv::Scalar(0, 0, 255), 2); 
		//void cv::circle (InputOutputArray img, Point center, int radius, const Scalar &color, int thickness=1, int lineType=LINE_8, int shift=0)
	}
	
}

void KeyFrameSP::computeWindowPoints(){
	std::cout<<"point_2d_uv size:"<<point_2d_uv.size()<<std::endl;
	std::cout<<"keypoints size:"<<keypoints.size()<<std::endl;
	cv::Mat mask1 = cv::Mat::zeros(img_size, CV_32FC1);
  	cv::Mat mask2 = cv::Mat::zeros(img_size, CV_32FC1);
	//float max_p2d_x = 0;
	//float max_kp_x = 0;
	int range = 5;
	for(size_t i=0; i<point_2d_uv.size(); i++){
		for(int k=-range;k<range;k++){
			for(int m=-range;m<range;m++){
				if((int(point_2d_uv[i].y)+k)<(mask1.rows-2) && (int(point_2d_uv[i].x)+m)<(mask1.cols-2)
				   && (int(point_2d_uv[i].y)+k)>2 && (int(point_2d_uv[i].x)+m)>2 )
					mask1.at<float>(int(point_2d_uv[i].y)+k, int(point_2d_uv[i].x)+m) = 1.0;
			}
		}
		//if(point_2d_uv[i].x > max_p2d_x)
		//	max_p2d_x = point_2d_uv[i].x;
	}
	//std::cout<<"max 2d:"<<max_p2d_x<<std::endl;
	for(size_t i=0; i<keypoints.size(); i++){
		mask2.at<float>(int(keypoints[i].pt.y), int(keypoints[i].pt.x)) = keypoints[i].response;
		//if(keypoints[i].pt.x > max_kp_x)
			//max_kp_x = keypoints[i].pt.x;
	}
	//std::cout<<"max kp:"<<max_kp_x<<std::endl;
	cv::Mat mask12=mask1.mul(mask2);
	for(int i = 0; i < mask12.rows; i++){
		for(int j = 0; j < mask12.cols; j++){
			if(mask12.at<float>(i, j)>0)
				window_keypoints.push_back(cv::KeyPoint(j,i,8,-1,mask12.at<float>(i, j)));
		}
	}
	std::cout<<"window_keypoints size:"<<window_keypoints.size()<<std::endl;
}
void KeyFrameSP::binarize_descriptors(){
	descriptors_converted = cv::Mat(descriptors.size(), 5);
	for(size_t i=0; i<descriptors.rows; ++i){
		cv::threshold(descriptors.row(i), descriptors_converted.row(i), 0, 1, cv::THRESH_BINARY);
	}
}

void KeyFrameSP::float2int_descriptors(){
	descriptors_converted = cv::Mat(descriptors.size(), CV_8UC1);
	for(size_t i=0; i<descriptors.rows; ++i){
		cv::Mat d_i_float = descriptors.row(i);
		cv::Mat d_i_int(d_i_float.size(), CV_8UC1);
		double minv = 0.0, maxv = 0.0;  
        double* minp = &minv;  
        double* maxp = &maxv;  
        cv::minMaxIdx(d_i_float,minp,maxp);  
        cv::Mat mat_min(d_i_float.size(),CV_32FC1);
        mat_min = cv::Scalar::all(minv); 
        //std::cout<<mat_min<<std::endl;
        d_i_int = 255*(d_i_float-mat_min)/(maxv-minv);
		descriptors_converted.row(i) = d_i_int.clone();
	}
}

void KeyFrameSP::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrameSP::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrameSP::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrameSP::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrameSP::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrameSP::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrameSP::getLoopRelativeYaw()
{
    return loop_info(7);
}

