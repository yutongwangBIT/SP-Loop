#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ThirdParty/DBoW3/DBoW3.h"
#include "SuperGlue.h"

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;

typedef struct {
    double r,g,b;
} COLOUR;


class KeyFrameSP
{
public:
	KeyFrameSP(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal, 
			 vector<double> &_point_id, SPDetector* _sp, SPGlue* _sp_glue, int _sequence);
	KeyFrameSP(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, cv::Mat &_descriptors);
	
	
	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);

	void extractSuperPoints();
	void extractWindowPoints();
	void computeWindowPoints();
	bool findConnection(KeyFrameSP* old_kf);

	Eigen::Vector3d getLoopRelativeT();
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();

	void float2int_descriptors();
	void binarize_descriptors();

	double time_stamp; 
	int index;
	int local_index;
	Eigen::Vector3d vio_T_w_i; 
	Eigen::Matrix3d vio_R_w_i; 
	Eigen::Vector3d T_w_i;
	Eigen::Matrix3d R_w_i;
	Eigen::Vector3d origin_vio_T;		
	Eigen::Matrix3d origin_vio_R;
	cv::Mat image;
	cv::Mat thumbnail;
	vector<cv::Point3f> point_3d; 
	vector<cv::Point2f> point_2d_uv;
	vector<cv::Point2f> point_2d_norm;
	vector<cv::Point3f> point_3d_window;  //correspond with window_keypoints
	vector<cv::Point2f> point_2d_uv_window;
	vector<cv::Point2f> point_2d_norm_window;
	vector<double> point_id;
	vector<double> point_id_window;
	vector<cv::KeyPoint> keypoints;
	vector<cv::KeyPoint> keypoints_norm;
	vector<cv::KeyPoint> window_keypoints;
	cv::Mat descriptors; //float n*256
	cv::Mat descriptors_converted; //int
	cv::Mat window_descriptors;
	bool has_fast_point;
	int sequence;

	bool has_loop;
	int loop_index;
	Eigen::Matrix<double, 8, 1 > loop_info;

	//visualization
	cv::Mat compareKpts;
private:
	SPGlue* sp_glue;
	SPDetector* sp_detector;
	cv::Size img_size;
	void draw();
	void drawLoopMatch(KeyFrameSP* old_kf, vector<cv::Point2f> matched_2d_old, 
                      vector<cv::Point2f> matched_2d_cur, vector<float> matched_scores, float rela_t=0.0, float rela_y=0.0);
	void computeWindowPointsAfter(vector<cv::Point2f> matched_2d_cur, vector<uchar> &status);
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
	               const std::vector<cv::Point3f> &matched_3d,
	               std::vector<uchar> &status,
	               Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
};

