#pragma once
#include <ros/assert.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "utility/utility.h"

#include <ceres/ceres.h>

class triangulateFactor : public ceres::SizedCostFunction<3, 1, 1, 4, 3>//res, num of para in block i (s_1, s_2, Q12, t12)
{
  public:
    triangulateFactor() = delete;
    triangulateFactor(const Eigen::Vector3d &_match_pts_cur_norm, const Eigen::Vector3d &_match_pts_old_norm)
        : match_pts_cur_norm(_match_pts_cur_norm), match_pts_old_norm(_match_pts_old_norm){}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        double s_1 = parameters[0][0];
        double s_2 = parameters[0][1];
        Eigen::Quaterniond Q21(parameters[0][5], parameters[0][2], parameters[0][3], parameters[0][4]);
        Eigen::Vector3d t21(parameters[0][6], parameters[0][7], parameters[0][8]);

        Eigen::Matrix3d R21 = Q21.toRotationMatrix();
       
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual = s_2 * match_pts_old_norm - s_1 * R21 * match_pts_cur_norm - t21;
       
        if(jacobians){
            if(jacobians[0]) //s_1
            {
                Eigen::Map<Eigen::Vector3d> block_0_jacobian(jacobians[0]);
                block_0_jacobian = - R21 * match_pts_cur_norm;
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Vector3d> block_1_jacobian(jacobians[1]);
                block_1_jacobian = match_pts_old_norm;
            }
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> block_2_jacobian(jacobians[2]);
                block_2_jacobian.setZero();
                block_2_jacobian.block<3, 3>(0, 0) = s_1 * Utility::skewSymmetric(R21 * match_pts_cur_norm);
            }
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> block_3_jacobian(jacobians[3]);
                block_3_jacobian = -1 * Eigen::Matrix3d::Identity();
            }
        }

        return true;
    }
private:
    const Eigen::Vector3d match_pts_cur_norm;
    const Eigen::Vector3d match_pts_old_norm;
};



