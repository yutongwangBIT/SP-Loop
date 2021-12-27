#include "keyframe_sp.h"

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}
template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

cv::Mat shortenVector(cv::Mat binarized_desc){
    cv::Mat shorted_desc = cv::Mat::zeros(cv::Size(32,1),CV_8UC1);
    for(size_t i=0;i<32;i++){
        for(size_t j=0;j<8;j++){
            shorted_desc.at<uchar>(i) += pow(2*binarized_desc.at<float>(8*i+j), j);
        }
    }
    return shorted_desc;
}

cv::Mat compressDesc(cv::Mat mat){
    cv::Mat compreesed = cv::Mat(cv::Size(32,mat.rows),CV_8UC1);
    for(size_t i=0;i<mat.rows;i++){
        cv::Mat m = mat.row(i);
        cv::Mat s = shortenVector(m);
        for(size_t j=0;j<32;j++){
            compreesed.at<uchar>(i,j) = s.at<uchar>(j);
        }
    }
    return compreesed;
}

COLOUR GetColour(double v,double vmin,double vmax)
{
   COLOUR c = {255,255,255}; // white
   double dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      c.r = 0;
      c.g = 255*(4 * (v - vmin) / dv);
   } else if (v < (vmin + 0.5 * dv)) {
      c.r = 0;
      c.b = 255 + 255*(4 * (vmin + 0.25 * dv - v) / dv);
   } else if (v < (vmin + 0.75 * dv)) {
      c.r = 255*(4 * (v - vmin - 0.5 * dv) / dv);
      c.b = 0;
   } else {
      c.g = 255 + 255*(4 * (vmin + 0.75 * dv - v) / dv);
      c.b = 0;
   }

   return(c);
}

// create KeyFrameSP online
KeyFrameSP::KeyFrameSP(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
		           vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
		           vector<double> &_point_id ,SPDetector* _sp, SPGlue* _sp_glue, int _sequence, std::shared_ptr<SuperPointNet> net_)
{
	net = net_;
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
	bool dec = sp_detector->detect(net,image, point_2d_uv, window_keypoints, window_descriptors, keypoints, descriptors, 1000);
	binarize_descriptors();
	//std::cout<<"size:"<<window_keypoints.size()<<std::endl;
	std::cout<<"size:"<<keypoints.size()<<std::endl;

	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
	//extractWindowPoints();
	//extractSuperPoints();
	//computeWindowPoints();
	//draw();
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
	bool dec = sp_detector->detect(net, image, keypoints, descriptors, 1000);
	//bool dec = sp_detector->detect(image, keypoints, descriptors);
	binarize_descriptors();
	std::cout<<"size:"<<keypoints.size()<<std::endl;

	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
}
void KeyFrameSP::extractWindowPoints(){
	bool dec = sp_detector->detectWindow(net, image, point_2d_uv, window_keypoints, window_descriptors);
	if(!dec)
		std::cout<<"window kps size changed"<<std::endl;
}

int KeyFrameSP::findConnection2(KeyFrameSP* old_kf){
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<float> matched_scores;
	vector<uchar> status;

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

	sp_glue->match(window_keypoints,  old_kf->keypoints, window_descriptors, 
	               old_kf->descriptors,  old_kf->keypoints_norm, matched_2d_old, 
				   matched_2d_old_norm, matched_scores, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_id, status);

	//drawLoopMatch(old_kf,  matched_2d_old, matched_2d_cur, matched_scores, 0.0, 0.0);
	return matched_2d_cur.size();
}
bool KeyFrameSP::findConnection(KeyFrameSP* old_kf){
	TicToc tmp_t;
	//printf("find Connection\n");
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<float> matched_scores;
	vector<uchar> status;

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

	TicToc t_match;

	//std::cout<<"before match size:"<<window_keypoints.size()<<std::endl;

	sp_glue->match(window_keypoints,  old_kf->keypoints, window_descriptors, 
	               old_kf->descriptors,  old_kf->keypoints_norm, matched_2d_old, 
				   matched_2d_old_norm, matched_scores, status);

	reduceVector(matched_3d, status);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_id, status);
	//drawLoopMatch(old_kf,  matched_2d_old, matched_2d_cur, matched_scores, 0.0, 0.0, "glue");
	//std::cout<<"matched_2d_cur size:"<<matched_2d_cur.size()<<std::endl;

	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;

	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
		status.clear();
		PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
		reduceVector(matched_2d_cur, status);
	    reduceVector(matched_2d_old, status);
	    reduceVector(matched_2d_cur_norm, status);
	    reduceVector(matched_2d_old_norm, status);
	    reduceVector(matched_3d, status);
	    reduceVector(matched_id, status);
	}
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
	    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
	    relative_q = PnP_R_old.transpose() * origin_vio_R;
	    relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
		float rela_pitch = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).y() - Utility::R2ypr(PnP_R_old).y());
		float rela_roll = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).z() - Utility::R2ypr(PnP_R_old).z());
	    //printf("PNP relative\n");
	    cout << "pnp relative_t " << relative_t.transpose() << endl;
		cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
	    //cout << "pnp relative_yaw " << relative_yaw << endl;
		//if (abs(relative_yaw) < 60.0 && relative_t.norm() < 20.0)
	    if ((abs(relative_yaw) > 30.0 || relative_t.norm() > 2.5) && abs(relative_yaw) < 60.0 && relative_t.norm() < 20)
		{
	    	has_loop = true;
	    	loop_index = old_kf->index;
	    	loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
	    	             relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
	    	             relative_yaw;
	    	//cout << "pnp relative_t " << relative_t.transpose() << endl;
			//cout << "pnp relative_q " << relative_q.transpose() << endl;
			//cout << "abs(relative_yaw) " << abs(relative_yaw) << endl;
			drawLoopMatch(old_kf,  matched_2d_old, matched_2d_cur, matched_scores, relative_t.norm(), abs(relative_yaw),abs(rela_pitch),abs(rela_roll),"pnp");
	    	//cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
	        return true;
	    }
	}
	return false;
}

void KeyFrameSP::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

void KeyFrameSP::binarize_descriptors(){
	//descriptors_converted = cv::Mat(descriptors.size(), 5);
	descriptors_converted = cv::Mat(descriptors.size(), CV_8UC1); //n * 256 
	for(size_t i=0; i<descriptors.rows; ++i){
		cv::Mat tmp;
		cv::threshold(descriptors.row(i), tmp, 0, 1, cv::THRESH_BINARY); // tmp is float 32
		//std::cout<<"tmp:"<<tmp<<std::endl;
		for(size_t j=0;j<descriptors.cols;j++){
            descriptors_converted.at<uchar>(i,j) = tmp.at<float>(j);
        }
		//std::cout<<"converted:"<<descriptors_converted.row(i)<<std::endl;
	}
    //compressed_descriptors = compressDesc(descriptors_converted);
	//std::cout<<compressed_descriptors<<std::endl;
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
	putText(compareKpts, "size:" + to_string(keypoints.size()), cv::Point2f(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255));
        
	std::string image_path0 = "/home/yutong/spVINS_ws/results/match_img/sp/" + to_string(index) + ".png";
    cv::imwrite(image_path0.c_str(), compareKpts);
}

void KeyFrameSP::drawLoopMatch(KeyFrameSP* old_kf, vector<cv::Point2f> matched_2d_old, 
                              vector<cv::Point2f> matched_2d_cur, vector<float> matched_scores,
							  float rela_t, float rela_y, float rela_p,float rela_r, string name){
	cv::Mat im;
	cv::Mat image_old = old_kf->image;
	std::vector<cv::KeyPoint> kpts_old = old_kf->keypoints;
	cv::hconcat(image, image_old, im);
	cv::cvtColor(im, im, cv::COLOR_GRAY2RGB);
	float max_score = 0.0;
	float min_score = 10.0;
	for (size_t i = 0; i < matched_scores.size(); i++)
	{
		if(matched_scores[i]>max_score)
			max_score = matched_scores[i];
		if(matched_scores[i]<min_score)
			min_score = matched_scores[i];
	}

/*	for (size_t i = 0; i < window_keypoints.size(); i++)
	{
		cv::Point2f Pt = window_keypoints[i].pt;
		cv::circle(im, Pt, 2, cv::Scalar(0, 0, 255), 2); 
	}*/
/*	for (size_t i = 0; i < kpts_old.size(); i++)
	{
		cv::Point2f Pt = kpts_old[i].pt;
		Pt.x += image_old.cols;
		cv::circle(im, Pt, 2, cv::Scalar(0, 0, 255), 1); 
	}*/

	//std::cout<<"max sc:"<<max_score<<std::endl;
	for(size_t i = 0; i < matched_2d_old.size(); i++){
		cv::Point2f p1 = matched_2d_cur[i];
		cv::Point2f p2 = matched_2d_old[i];
		p2.x += image_old.cols;
		cv::circle(im, p1, 2, cv::Scalar(0, 255, 0), 1);
		cv::circle(im, p2, 2, cv::Scalar(0, 255, 0), 1);
		COLOUR c =  GetColour(matched_scores[i], 0, 1);
		cv::line(im, p1, p2, cv::Scalar(c.b, c.g, c.r), 1, cv::LINE_AA);
	}
	//putText(im, "relative t norm:" + to_string(rela_t) + "yaw:" + to_string(rela_y), cv::Point2f(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255));
    putText(im, "relative translation:" + to_string_with_precision(rela_t,1) + ", y:" + to_string_with_precision(rela_y,1)+ ", p:" + to_string_with_precision(rela_p,1)+ ", r:" + to_string_with_precision(rela_r,1), cv::Point2f(30, image.rows-30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255),2);    
	std::string image_path0 = "/home/yutong/spVINS_ws/results/match_img/sp_pnp_thres/" + name + to_string(index) + ".png";
    cv::imwrite(image_path0.c_str(), im);
	//std::string image_path1 = "/home/yutong/spVINS_ws/results/match_img/sp/" + to_string(index) + "_1.png";
    //cv::imwrite(image_path1.c_str(), image_old);
	//std::string image_path = "/home/yutong/spVINS_ws/results/match_img/sp_window_after/" + to_string(index) + "_match.png";
    //cv::imwrite(image_path.c_str(), compareKpts);
	/*if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
		std::string image_path = "/home/yutong/spVINS_ws/results/match_img/tomasi_sp_loop/" + to_string(index) + "_match.png";
        cv::imwrite(image_path.c_str(), compareKpts);
	}*/
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

void KeyFrameSP::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
	/*for (int i = 0; i < matched_3d.size(); i++){
		printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
		printf("2d old x: %f, y: %f\n",matched_2d_old_norm[i].x, matched_2d_old_norm[i].y );
	}*/
		
		
	//printf("match size %d \n", matched_3d.size());
	//std::cout<<qic<<std::endl;
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0); // cause we use norm 2d
    Matrix3d R_inital;
    Vector3d P_inital;
    Matrix3d R_w_c = origin_vio_R * qic; //qic is sent by estimator for extrinct, check if without IMU it is identity?
    Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    

    if (CV_MAJOR_VERSION < 3)
        solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
    else
    {
        if (CV_MINOR_VERSION < 2)
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
        else{
			TicToc t_pnp_ransac;
			//solvePnP(matched_3d, matched_2d_old_norm, K, D, rvec, t, true);
			solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers); //this one
			//printf("solvePnP costs: %f \n", t_pnp_ransac.toc());
		}
           

    }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);

    for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
	//cout << "T_pnp " << T_pnp.transpose() << endl;
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * qic.transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * tic;
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

/*if(USE_TRIANGULATE_FACTOR && (int)matched_2d_cur.size() > MIN_LOOP_NUM){	
		int num_points = (int)matched_2d_cur.size();
		double para_s_1[num_points][1]; //cur depth
		double para_s_2[num_points][1]; //old 
		double para_q21[4]; //old to cur
		double para_t21[3];
		//INITIALIZE PARAMETERS
		//Eigen::Vector3d T_w_old;
		//Eigen::Matrix3d R_w_old;
		//old_kf->getPose(T_w_old, R_w_old);
		//relative_t = R_w_old.transpose() * (origin_vio_T - T_w_old);//t_old_cur
		//relative_q = R_w_old.transpose() * origin_vio_R;//q_old_cur
		relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);//t_old_cur
		relative_q = PnP_R_old.transpose() * origin_vio_R;//q_old_cur
		cout << "initial relative_t " << relative_t.transpose() << endl;
		cout << "initial relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
		relative_q.normalized();
		cout << "initial relative_q normalized " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
		para_q21[0] = relative_q.x();
		para_q21[1] = relative_q.y();
		para_q21[2] = relative_q.z();
		para_q21[3] = relative_q.w();
		para_t21[0] = relative_t.x();
		para_t21[1] = relative_t.y();
		para_t21[2] = relative_t.z();
		//SET CERES PROBLEM
		ceres::Problem problem;
		ceres::LossFunction *loss_function;
		loss_function = new ceres::HuberLoss(1.0);

		for(int i = 0; i < num_points; i++){
			Eigen::Vector3d point_i_w(matched_3d[i].x,matched_3d[i].y,matched_3d[i].z);
			Eigen::Vector3d point_i_1 = origin_vio_R.transpose() * (point_i_w - origin_vio_T);//r_1_w(p_w - t_w_1)
			//Eigen::Vector3d point_i_2 = R_w_old.transpose() * (point_i_w - T_w_old);
			Eigen::Vector3d point_i_2 = PnP_R_old.transpose() * (point_i_w - PnP_T_old);
			para_s_1[i][0] = point_i_1.z();
			para_s_2[i][0] = point_i_2.z();
			problem.AddParameterBlock(para_s_1[i], 1);
			problem.AddParameterBlock(para_s_2[i], 1);
			problem.AddParameterBlock(para_q21, 4);
			problem.AddParameterBlock(para_t21, 3);
		}
		for(int i = 0; i < num_points; i++){
			ceres::CostFunction* cost_function_point;
			Eigen::Vector3d match_pts_cur_norm_i(matched_2d_cur_norm[i].x, matched_2d_cur_norm[i].y, 1);
			Eigen::Vector3d match_pts_old_norm_i(matched_2d_old_norm[i].x, matched_2d_old_norm[i].y, 1);
			if(i==0){
				std::cout<<"diff1:"<<( para_s_2[i][0]*match_pts_old_norm_i - (para_s_1[i][0] * relative_q.toRotationMatrix() * 
				               match_pts_cur_norm_i + relative_t) ).transpose()<<std::endl;
			}
			cost_function_point = new triangulateFactor( match_pts_cur_norm_i, match_pts_old_norm_i);
			problem.AddResidualBlock(cost_function_point, loss_function, para_s_1[i], para_s_2[i],para_q21, para_t21);
			//std::cout<<"para_s_1[i][0]:"<<para_s_1[i][0]<<std::endl;						
			//cost_function_point = new triangulateFactor2(para_s_1[i][0], match_pts_cur_norm_i, match_pts_old_norm_i);
			//problem.AddResidualBlock(cost_function_point, loss_function, para_s_2[i], para_q21, para_t21);
		}
		// Run the solver!
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_QR;
	//  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		std::cout<<"\n"<<summary.FullReport()<<"\n";
		relative_q.x() = para_q21[0];
		relative_q.y() = para_q21[1];
		relative_q.z() = para_q21[2];
		relative_q.w() = para_q21[3];
		relative_t.x() = para_t21[0];
		relative_t.y() = para_t21[1];
		relative_t.z() = para_t21[2];
		cout << "optimize relative_t " << relative_t.transpose() << endl;
		cout << "optimize yaw relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
	}*/
	/*vector<cv::Point3f> matched_3d_corrected;
		TicToc t_pnp_trian;
		for(size_t i=0; i<matched_3d.size(); i++){
			Eigen::Matrix<double, 3, 4> curPose;
			Eigen::Matrix<double, 3, 4> oldPose;
			Eigen::Vector3d t_cur = origin_vio_T + origin_vio_R * tic; //from imu to camera
			Eigen::Matrix3d R_cur = origin_vio_R * qic;
			Eigen::Vector3d t_old = PnP_T_old + PnP_R_old * tic;
			Eigen::Matrix3d R_old = PnP_R_old * qic;
			curPose.leftCols<3>() = R_cur.transpose();
			curPose.rightCols<1>() = -R_cur.transpose() * t_cur;
			oldPose.leftCols<3>() = R_old.transpose();
			oldPose.rightCols<1>() = -R_old.transpose() * t_old;
			Eigen::Vector2d point_cur(matched_2d_cur_norm[i].x, matched_2d_cur_norm[i].y);
			Eigen::Vector2d point_old(matched_2d_old_norm[i].x, matched_2d_old_norm[i].y);
			Eigen::Vector3d point3d(matched_3d[i].x, matched_3d[i].y, matched_3d[i].z);
			//std::cout<<"before:"<<point3d.transpose()<<std::endl;
			triangulatePoint(curPose, oldPose, point_cur, point_old, point3d);
			//std::cout<<"after:"<<point3d.transpose()<<std::endl; 
			matched_3d_corrected.push_back(cv::Point3f(point3d.x(), point3d.y(), point3d.z()));
		}
		printf("triangulate costs: %f \n", t_pnp_trian.toc());
		PnPRANSAC(matched_2d_old_norm, matched_3d_corrected, status, PnP_T_old, PnP_R_old);
		*/

		/*if(USE_TRIANGULATE_FACTOR && (int)matched_2d_cur.size() > MIN_LOOP_NUM){	
		int num_points = (int)matched_2d_cur.size();
		double para_s_2[num_points][1]; //old 
		double para_q21[4]; //old to cur
		double para_t21[3];
		//INITIALIZE PARAMETERS
		Eigen::Vector3d T_w_old;
		Eigen::Matrix3d R_w_old;
		old_kf->getPose(T_w_old, R_w_old);
		relative_t = R_w_old.transpose() * (origin_vio_T - T_w_old);//t_old_cur
		relative_q = R_w_old.transpose() * origin_vio_R;//q_old_cur
		cout << "initial relative_t " << relative_t.transpose() << endl;
		cout << "initial relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
		relative_q.normalized();
		cout << "initial relative_q normalized " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
		para_q21[0] = relative_q.x();
		para_q21[1] = relative_q.y();
		para_q21[2] = relative_q.z();
		para_q21[3] = relative_q.w();
		para_t21[0] = relative_t.x();
		para_t21[1] = relative_t.y();
		para_t21[2] = relative_t.z();
		//SET CERES PROBLEM
		ceres::Problem problem;
		ceres::LossFunction *loss_function;
		loss_function = new ceres::HuberLoss(1.0);

		for(int i = 0; i < num_points; i++){
			Eigen::Vector3d point_i_w(matched_3d[i].x,matched_3d[i].y,matched_3d[i].z);
			Matrix3d R_w_c = R_w_old * qic;
    		Vector3d T_w_c = T_w_old + R_w_old * tic;
			Eigen::Vector3d point_i_2 = R_w_c.transpose() * (point_i_w - T_w_c);
			para_s_2[i][0] = point_i_2.z();
			problem.AddParameterBlock(para_s_2[i], 1);
			problem.AddParameterBlock(para_q21, 4);
			problem.AddParameterBlock(para_t21, 3);
		}
		for(int i = 0; i < num_points; i++){
			ceres::CostFunction* cost_function_point;
			Eigen::Vector3d point_i_w(matched_3d[i].x,matched_3d[i].y,matched_3d[i].z);
			Matrix3d R_w_c = origin_vio_R * qic;
    		Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;
			Eigen::Vector3d point_i_1 = R_w_c.transpose() * (point_i_w - T_w_c);//point coord in c1 frame
			Eigen::Vector3d match_pts_old_norm_i(matched_2d_old_norm[i].x, matched_2d_old_norm[i].y, 1);
			cost_function_point = new pnpFactor( point_i_1, match_pts_old_norm_i);
			problem.AddResidualBlock(cost_function_point, loss_function, para_s_2[i],para_q21, para_t21);
		}
		// Run the solver!
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_QR;
	//  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
	//	std::cout<<"\n"<<summary.FullReport()<<"\n";
		relative_q.x() = para_q21[0];
		relative_q.y() = para_q21[1];
		relative_q.z() = para_q21[2];
		relative_q.w() = para_q21[3];
		relative_t.x() = para_t21[0];
		relative_t.y() = para_t21[1];
		relative_t.z() = para_t21[2];
		cout << "optimize relative_t " << relative_t.transpose() << endl;
		cout << "optimize relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
		Eigen::Vector3d eulerAngle=Utility::R2ypr(relative_q.matrix());
		relative_yaw = Utility::normalizeAngle(eulerAngle.x());
		if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
	    {

	    	has_loop = true;
	    	loop_index = old_kf->index;
	    	loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
	    	             relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
	    	             relative_yaw;
	    	//cout << "pnp relative_t " << relative_t.transpose() << endl;
			//cout << "pnp relative_q " << relative_q.transpose() << endl;
			cout << "abs(relative_yaw) " << abs(relative_yaw) << endl;
			drawLoopMatch(old_kf,  matched_2d_old, matched_2d_cur, matched_scores, relative_t.norm(), abs(relative_yaw));
	    	//cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
	        return true;
	    }
	}*/

	/*if(USE_TRIANGULATE_FACTOR && (int)matched_2d_cur.size() > MIN_LOOP_NUM){	
		int num_points = (int)matched_2d_cur.size();
		double para_p1[num_points][3]; //old 
		double para_T21[7]; //old to cur
		//INITIALIZE PARAMETERS
		Eigen::Vector3d T_w_old;
		Eigen::Matrix3d R_w_old;
		old_kf->getPose(T_w_old, R_w_old);
		//relative_t = R_w_old.transpose() * (origin_vio_T - T_w_old);//t_old_cur
		//relative_q = R_w_old.transpose() * origin_vio_R;//q_old_cur
		relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
		relative_q = PnP_R_old.transpose() * origin_vio_R;
		cout << "initial relative_t " << relative_t.transpose() << endl;
		cout << "initial relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
		para_T21[0] = relative_t.x();
		para_T21[1] = relative_t.y();
		para_T21[2] = relative_t.z();
		para_T21[3] = relative_q.x();
		para_T21[4] = relative_q.y();
		para_T21[5] = relative_q.z();
		para_T21[6] = relative_q.w();
		
		//SET CERES PROBLEM
		ceres::Problem problem;
		ceres::LossFunction *loss_function;
		//loss_function = new ceres::HuberLoss(0.1);
		loss_function = new ceres::CauchyLoss(0.5);
		problem.AddParameterBlock(para_T21, 7);
		for(int i = 0; i < num_points; i++){
			Eigen::Vector3d point_i_w(matched_3d[i].x,matched_3d[i].y,matched_3d[i].z);
			Matrix3d R_w_c = origin_vio_R * qic;
    		Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;
			Eigen::Vector3d point_i_1 = R_w_c.transpose() * (point_i_w - T_w_c);//point coord in c1 frame
			para_p1[i][0] = point_i_1.x();
			para_p1[i][1] = point_i_1.y();
			para_p1[i][2] = point_i_1.z();
			problem.AddParameterBlock(para_p1[i], 3);
			//problem.SetParameterBlockConstant(para_p1[i]);
		}
		for(int i = 0; i < num_points; i++){
			ceres::CostFunction* cost_function_point;
			Eigen::Vector2d match_pts_old_norm_i(matched_2d_old_norm[i].x, matched_2d_old_norm[i].y);
			double weight_i = matched_scores[i];
			cost_function_point = new pnpFactor(match_pts_old_norm_i, weight_i);
			//std::cout<<i<<":"<<para_p1[i][0]<<std::endl;
			problem.AddResidualBlock(cost_function_point, loss_function, para_T21, para_p1[i]);
		}
		// Run the solver!
		ceres::Solver::Options options;
		//options.max_num_iterations = 100;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		std::cout<<"\n"<<summary.FullReport()<<"\n";
		relative_t.x() = para_T21[0];
		relative_t.y() = para_T21[1];
		relative_t.z() = para_T21[2];
		relative_q.x() = para_T21[3];
		relative_q.y() = para_T21[4];
		relative_q.z() = para_T21[5];
		relative_q.w() = para_T21[6];
		
		cout << "optimize relative_t " << relative_t.transpose() << endl;
		cout << "optimize relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
		Eigen::Vector3d eulerAngle=Utility::R2ypr(relative_q.matrix());
		relative_yaw = Utility::normalizeAngle(eulerAngle.x());
		if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
	    {

	    	has_loop = true;
	    	loop_index = old_kf->index;
	    	loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
	    	             relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
	    	             relative_yaw;
	    	//cout << "pnp relative_t " << relative_t.transpose() << endl;
			//cout << "pnp relative_q " << relative_q.transpose() << endl;
			cout << "abs(relative_yaw) " << abs(relative_yaw) << endl;
			drawLoopMatch(old_kf,  matched_2d_old, matched_2d_cur, matched_scores, relative_t.norm(), abs(relative_yaw));
	    	//cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
	        return true;
	    }
	}*/