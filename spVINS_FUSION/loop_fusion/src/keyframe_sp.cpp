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
	extractWindowPoints();
	extractSuperPoints();
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
	bool dec = sp_detector->detect(image, keypoints, descriptors, 1000);
	binarize_descriptors();
	//std::cout<<"int desc:"<<descriptors_converted.size()<<std::endl;

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
	bool dec = sp_detector->detectWindow(image, point_2d_uv, window_keypoints, window_descriptors);
	if(!dec)
		std::cout<<"window kps size changed"<<std::endl;
}


void KeyFrameSP::computeWindowPoints(){
	//std::cout<<"point_2d_uv size:"<<point_2d_uv.size()<<std::endl;
	//std::cout<<"keypoints size:"<<keypoints.size()<<std::endl;
	cv::Mat mask1 = cv::Mat::zeros(img_size, CV_32FC1);
  	cv::Mat mask2 = cv::Mat::zeros(img_size, CV_32FC1);
	cv::Mat index_matrix = cv::Mat::zeros(img_size, CV_32FC1);

	int range = 7;
	for(size_t i=0; i<point_2d_uv.size(); i++){
		for(int k=-range;k<range;k++){
			for(int m=-range;m<range;m++){
				if((int(point_2d_uv[i].y)+k)<(mask1.rows-2) && (int(point_2d_uv[i].x)+m)<(mask1.cols-2)
				   && (int(point_2d_uv[i].y)+k)>2 && (int(point_2d_uv[i].x)+m)>2 ){
					   mask1.at<float>(int(point_2d_uv[i].y)+k, int(point_2d_uv[i].x)+m) = 1.0;
					   index_matrix.at<float>(int(point_2d_uv[i].y)+k, int(point_2d_uv[i].x)+m) = (float)i;
				   }
			}
		}
	}
	//std::cout<<"max 2d:"<<max_p2d_x<<std::endl;
	for(size_t i=0; i<keypoints.size(); i++){
		mask2.at<float>(int(keypoints[i].pt.y), int(keypoints[i].pt.x)) = keypoints[i].response;
	}
	//std::cout<<"max kp:"<<max_kp_x<<std::endl;
	cv::Mat mask12=mask1.mul(mask2);
	for(int i = 0; i < mask12.rows; i++){
		for(int j = 0; j < mask12.cols; j++){
			if(mask12.at<float>(i, j)>0){
				int index = (int)index_matrix.at<float>(i, j);
				//std::cout<<"index:"<<index<<",";
				point_2d_norm_window.push_back(point_2d_norm[index]);
				point_2d_uv_window.push_back(point_2d_uv[index]);
				point_3d_window.push_back(point_3d[index]); //they should be correspond with window_keypoints;
				window_descriptors.push_back(descriptors.row(index));
				point_id_window.push_back(point_id[index]);
				window_keypoints.push_back(cv::KeyPoint(j,i,8,-1,mask12.at<float>(i, j)));
			}
		}
	}
	//std::cout<<"window_keypoints size:"<<window_keypoints.size()<<std::endl;
	//std::cout<<"window_desc size:"<<window_descriptors.rows<<std::endl;
}

void KeyFrameSP::computeWindowPointsAfter(vector<cv::Point2f> matched_2d_cur, vector<uchar> &status){
	//std::cout<<"point_2d_uv size:"<<point_2d_uv.size()<<std::endl;
	//std::cout<<"matched_2d_cur size:"<<matched_2d_cur.size()<<std::endl;
	cv::Mat mask1 = cv::Mat::zeros(img_size, CV_32FC1);
  	cv::Mat mask2 = cv::Mat::zeros(img_size, CV_32FC1);
	cv::Mat index_matrix = cv::Mat::zeros(img_size, CV_32FC1);
	cv::Mat index_matrix2 = cv::Mat::zeros(img_size, CV_32FC1);

	int range = 6;
	for(size_t i=0; i<point_2d_uv.size(); i++){
		for(int k=-2;k<2;k++){
			for(int m=-range;m<range;m++){
				if((int(point_2d_uv[i].y)+k)<(mask1.rows-2) && (int(point_2d_uv[i].x)+m)<(mask1.cols-2)
				   && (int(point_2d_uv[i].y)+k)>2 && (int(point_2d_uv[i].x)+m)>2 ){
					   mask1.at<float>(int(point_2d_uv[i].y)+k, int(point_2d_uv[i].x)+m) = 1.0;
					   index_matrix.at<float>(int(point_2d_uv[i].y)+k, int(point_2d_uv[i].x)+m) = (float)i;
				   }
			}
		}
	}
	//std::cout<<"max 2d:"<<max_p2d_x<<std::endl;
	for(size_t i=0; i<matched_2d_cur.size(); i++){
		mask2.at<float>(int(matched_2d_cur[i].y), int(matched_2d_cur[i].x)) = 1.0; //index+1.0 for > 0
		index_matrix2.at<float>(int(matched_2d_cur[i].y), int(matched_2d_cur[i].x)) = (float)i;
		status.push_back(0);
	}
	//std::cout<<"max kp:"<<max_kp_x<<std::endl;
	cv::Mat mask12=mask1.mul(mask2);
	for(int i = 0; i < mask12.rows; i++){
		for(int j = 0; j < mask12.cols; j++){
			if(mask12.at<float>(i, j)>0){
				int index_1 = (int)index_matrix.at<float>(i, j); //index for point_2d_uv and point_3d....
				point_3d_window.push_back(point_3d[index_1]); //they should be correspond with window_keypoints;
				point_id_window.push_back(point_id[index_1]);
				point_2d_norm_window.push_back(point_2d_norm[index_1]);
				int index_2 = (int)(index_matrix2.at<float>(i, j)); //index for matched_2d_cur and 3d....
				status[index_2] = 1;
			}
		}
	}
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
		Eigen::Vector3d relative_t_old = PnP_R_old.transpose() * (old_kf->origin_vio_T - PnP_T_old);
		//std::cout<<"old vs old:"<<relative_t_old.transpose()<<std::endl;
	    reduceVector(matched_2d_cur, status);
	    reduceVector(matched_2d_old, status);
	    reduceVector(matched_2d_cur_norm, status);
	    reduceVector(matched_2d_old_norm, status);
	    reduceVector(matched_3d, status);
	    reduceVector(matched_id, status);
		//std::cout<<"matched 3d size after pnp:"<<matched_3d.size()<<std::endl;
	}
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
	    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
	    relative_q = PnP_R_old.transpose() * origin_vio_R;
	    relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
	    //printf("PNP relative\n");
	    //cout << "pnp relative_t " << relative_t.transpose() << endl;
	    //cout << "pnp relative_yaw " << relative_yaw << endl;
	    if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
	    {

	    	has_loop = true;
	    	loop_index = old_kf->index;
	    	loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
	    	             relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
	    	             relative_yaw;
	    	//cout << "pnp relative_t " << relative_t.transpose() << endl;
			//cout << "abs(relative_yaw) " << abs(relative_yaw) << endl;
			drawLoopMatch(old_kf,  matched_2d_old, matched_2d_cur, matched_scores, relative_t.norm(), abs(relative_yaw));
	    	//cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
	        return true;
	    }
	}
	return false;
}

void KeyFrameSP::binarize_descriptors(){
	descriptors_converted = cv::Mat(descriptors.size(), 5);
	for(size_t i=0; i<descriptors.rows; ++i){
		cv::threshold(descriptors.row(i), descriptors_converted.row(i), 0, 1, cv::THRESH_BINARY);
	}
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

void KeyFrameSP::drawLoopMatch(KeyFrameSP* old_kf, vector<cv::Point2f> matched_2d_old, 
                              vector<cv::Point2f> matched_2d_cur, vector<float> matched_scores,
							  float rela_t, float rela_y){
	cv::Mat image_old = old_kf->image;
	std::vector<cv::KeyPoint> kpts_old = old_kf->keypoints;
	cv::hconcat(image, image_old, compareKpts);
	cv::cvtColor(compareKpts, compareKpts, cv::COLOR_GRAY2RGB);
	float max_score = 0.0;
	float min_score = 10.0;
	for (size_t i = 0; i < matched_scores.size(); i++)
	{
		if(matched_scores[i]>max_score)
			max_score = matched_scores[i];
		if(matched_scores[i]<min_score)
			min_score = matched_scores[i];
	}

	for (size_t i = 0; i < window_keypoints.size(); i++)
	{
		cv::Point2f Pt = window_keypoints[i].pt;
		cv::circle(compareKpts, Pt, 2, cv::Scalar(0, 0, 255), 2); 
	}

	//std::cout<<"max sc:"<<max_score<<std::endl;
	for(size_t i = 0; i < matched_2d_old.size(); i++){
		cv::Point2f p1 = matched_2d_cur[i];
		cv::Point2f p2 = matched_2d_old[i];
		p2.x += 752;
		cv::circle(compareKpts, p1, 2, cv::Scalar(0, 255, 0), 2);
		cv::circle(compareKpts, p2, 2, cv::Scalar(0, 255, 0), 2);
		COLOUR c =  GetColour(matched_scores[i], 0, 1);
		cv::line(compareKpts, p1, p2, cv::Scalar(c.b, c.g, c.r), 1);
	}
	putText(compareKpts, "relative t norm:" + to_string(rela_t) + "yaw:" + to_string(rela_y), cv::Point2f(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255));
        
	std::string image_path0 = "/home/yutong/spVINS_ws/results/match_img/sp_pnp_ransac/" + to_string(index) + "_match.png";
    cv::imwrite(image_path0.c_str(), compareKpts);
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
			solvePnP(matched_3d, matched_2d_old_norm, K, D, rvec, t, true);
			//solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 50, 10.0 / 460.0, 0.9, inliers); //this one
			printf("solvePnP costs: %f \n", t_pnp_ransac.toc());
		}
           

    }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(1);

    /*for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }*/

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

