#pragma once
#include <ros/assert.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "utility/utility.h"

#include <ceres/ceres.h>

class pnpFactor : public ceres::SizedCostFunction<2, 7, 3>//res, num of para in block i ( T21, Pi)
{
  public:
    pnpFactor() = delete;
    pnpFactor(const Eigen::Vector2d &_match_pts_old_norm, const double &_weight_i)
        : match_pts_old_norm(_match_pts_old_norm), weight_i(_weight_i){}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d t21(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Q21(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        

        Eigen::Vector3d P1(parameters[1][0], parameters[1][1], parameters[1][2]);

        Eigen::Matrix3d R21 = Q21.toRotationMatrix();
        //std::cout<<"P1:"<<P1.x()<<std::endl;
        Eigen::Vector3d P2 = R21 * P1 + t21;
        double X2 = P2.x();
        double Y2 = P2.y();
        double Z2 = P2.z();
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual.x() = weight_i * (match_pts_old_norm.x() - X2/Z2);
        residual.y() = weight_i * (match_pts_old_norm.y() - Y2/Z2);
        //std::cout<<"residual.x() :"<< residual.x() <<std::endl;
        //std::cout<<"residual.y() :"<< residual.y() <<std::endl;
        if(jacobians){
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> block_0_jacobian(jacobians[0]);
                block_0_jacobian << 1/Z2, 0, -X2/Z2/Z2, -X2*Y2/Z2/Z2, 1 + X2*X2/Z2/Z2, -Y2/Z2, 0,
                                    0, 1/Z2, -Y2/Z2/Z2, -1-Y2*Y2/Z2/Z2, X2*Y2/Z2/Z2, X2/Z2, 0;
                block_0_jacobian = -weight_i * block_0_jacobian;
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> block_1_jacobian(jacobians[1]);
                Eigen::Matrix<double, 2, 3> tmp;
                tmp << 1/Z2, 0, -X2/Z2/Z2,
                       0, 1/Z2, -Y2/Z2/Z2;
                block_1_jacobian = - weight_i * tmp * R21;
            }
        }

        return true;
    }
private:
    const Eigen::Vector2d match_pts_old_norm;
    const double weight_i;
};



