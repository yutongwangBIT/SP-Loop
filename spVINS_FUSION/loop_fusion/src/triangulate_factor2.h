#pragma once
#include <ros/assert.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "utility/utility.h"

#include <ceres/ceres.h>

class triangulateFactor2 : public ceres::SizedCostFunction<3, 4, 3>//res, num of para in block i (s_2, Q12, t12)
{
  public:
    triangulateFactor2() = delete;
    triangulateFactor2(const double &_s_1, const Eigen::Vector3d &_match_pts_cur_norm, const Eigen::Vector3d &_match_pts_old_norm)
        : s_1(_s_1), match_pts_cur_norm(_match_pts_cur_norm), match_pts_old_norm(_match_pts_old_norm){}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
       // double s_1 = parameters[0][0];
        Eigen::Quaterniond Q21(parameters[0][4], parameters[0][1], parameters[0][2], parameters[0][3]);
        Eigen::Vector3d t21(parameters[0][5], parameters[0][6], parameters[0][7]);

        Eigen::Matrix3d R21 = Q21.toRotationMatrix();
       
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual = s_1 * Utility::skewSymmetric(match_pts_old_norm) * R21 * match_pts_cur_norm + Utility::skewSymmetric(match_pts_old_norm) * t21;
        //std::cout<<"res:"<<residual<<std::endl;
       
        if(jacobians){
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> block_2_jacobian(jacobians[0]);
                block_2_jacobian.setZero();
                block_2_jacobian.block<3, 3>(0, 0) = - s_1 *  Utility::skewSymmetric(Utility::skewSymmetric(match_pts_old_norm) * R21 * match_pts_cur_norm);
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> block_3_jacobian(jacobians[1]);
                block_3_jacobian = Utility::skewSymmetric(match_pts_old_norm);
            }
        }

        return true;
    }
private:
    const Eigen::Vector3d match_pts_cur_norm;
    const Eigen::Vector3d match_pts_old_norm;
    double s_1;
};



