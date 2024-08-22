/**
* This file is part of splio.
* 
* Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot sg>,
* School of EEE
* Nanyang Technological Univertsity, Singapore
* 
* For more information please see <https://britsknguyen.github.io>.
* or <https://github.com/brytsknguyen/splio>.
* If you use this code, please cite the respective publications as
* listed on the above websites.
* 
* splio is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* splio is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with splio.  If not, see <http://www.gnu.org/licenses/>.
*/

//
// Created by Thien-Minh Nguyen on 01/08/22.
//

#include <ceres/ceres.h>
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/utils/sophus_utils.hpp"
#include "../utility.h"

using namespace Eigen;

class GPIMUFactor: public ceres::CostFunction
{
public:

    // Destructor
    ~GPIMUFactor() {};

    // Constructor
    GPIMUFactor(const Vector3d &acc_, const Vector3d &gyro_, const Vector3d &acc_bias_, const Vector3d &gyro_bias_, double w_,
                         GPMixerPtr gpm_, double s_)
    :   acc         (acc_             ),
        gyro        (gyro_            ),
        acc_bias    (acc_bias_        ),
        gyro_bias   (gyro_bias_       ),        
        w           (w_               ),
        Dt          (gpm_->getDt()    ),
        s           (s_               ),
        gpm         (gpm_             )

    {
        // 6-element residual: 
        set_num_residuals(12);

        // Rotation of the first knot
        mutable_parameter_block_sizes()->push_back(4);
        // Angular velocity of the first knot
        mutable_parameter_block_sizes()->push_back(3);
        // Angular acceleration of the first knot
        mutable_parameter_block_sizes()->push_back(3);
        // Position of the first knot
        mutable_parameter_block_sizes()->push_back(3);
        // Velocity of the first knot
        mutable_parameter_block_sizes()->push_back(3);
        // Acceleration of the first knot
        mutable_parameter_block_sizes()->push_back(3);

        // Rotation of the second knot
        mutable_parameter_block_sizes()->push_back(4);
        // Angular velocity of the second knot
        mutable_parameter_block_sizes()->push_back(3);
        // Angular acceleration of the second knot
        mutable_parameter_block_sizes()->push_back(3);
        // Position of the second knot
        mutable_parameter_block_sizes()->push_back(3);
        // Velocity of the second knot
        mutable_parameter_block_sizes()->push_back(3);
        // Acceleration of the second knot
        mutable_parameter_block_sizes()->push_back(3);

        mutable_parameter_block_sizes()->push_back(3);
        mutable_parameter_block_sizes()->push_back(3);
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        GPState Xa(0);  gpm->MapParamToState(parameters, RaIdx, Xa);
        GPState Xb(Dt); gpm->MapParamToState(parameters, RbIdx, Xb);
        Eigen::Vector3d biasW = Eigen::Map<Eigen::Vector3d const>(parameters[12]);        
        Eigen::Vector3d biasA = Eigen::Map<Eigen::Vector3d const>(parameters[13]);    
        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the pose at sampling time --------------------------------------------------------------*/

        GPState Xt(s*Dt); vector<vector<Matrix3d>> DXt_DXa; vector<vector<Matrix3d>> DXt_DXb;

        Eigen::Matrix<double, 9, 1> gammaa;
        Eigen::Matrix<double, 9, 1> gammab;
        Eigen::Matrix<double, 9, 1> gammat;

        gpm->ComputeXtAndJacobians(Xa, Xb, Xt, DXt_DXa, DXt_DXb, gammaa, gammab, gammat);

        // Residual
        Eigen::Map<Matrix<double, 12, 1>> residual(residuals);      
        Eigen::Vector3d g(0, 0, 9.81);

        residual.block<3, 1>(0, 0) = w*(Xt.R.matrix().transpose() * (Xt.A + g) - acc + biasA);
        // residual.block<3, 1>(0, 0) = w*((Xt.A + g) - acc + biasA);
        residual.block<3, 1>(3, 0) = w*(Xt.O - gyro + biasW);
        residual.block<3, 1>(6, 0) = w*(biasW - gyro_bias);
        residual.block<3, 1>(9, 0) = w*(biasA - acc_bias);

        if (!jacobians)
            return true;

        // Matrix3d Dr_DRt  = Xt.R.matrix().transpose() * SO3d::hat(Xt.A + g);
        Matrix3d Dr_DRt  = SO3d::hat(Xt.R.matrix().transpose() * (Xt.A + g));
        // Matrix3d Dr_DRt  = Matrix3d::Zero();
        // Matrix3d Dr_DPt  = Xt.R.matrix().transpose();
        Matrix3d Dr_DAt  = Xt.R.matrix().transpose();

        // Matrix3d Dr_DBg  = Matrix3d::Identity();
        Matrix3d Dr_DOt  = Matrix3d::Identity();    

        // Matrix3d Dr_DBg  = Matrix3d::Identity();
        // Matrix3d Dr_DBa  = Matrix3d::Identity();

        size_t idx;

        // Jacobian on Ra
        idx = RaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 4, Eigen::RowMajor>> Dr_DRa(jacobians[idx]);
            Dr_DRa.setZero();
            Dr_DRa.block<3, 3>(0, 0) = 2*w*Dr_DRt*DXt_DXa[Ridx][Ridx];
            Dr_DRa.block<3, 3>(3, 0) = 2*w*Dr_DOt*DXt_DXa[Oidx][Ridx];
        }

        // Jacobian on Oa
        idx = OaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 3, Eigen::RowMajor>> Dr_DOa(jacobians[idx]);
            Dr_DOa.setZero();
            Dr_DOa.block<3, 3>(0, 0) = w*Dr_DRt*DXt_DXa[Ridx][Oidx];
            Dr_DOa.block<3, 3>(3, 0) = w*Dr_DOt*DXt_DXa[Oidx][Oidx];
        }

        // Jacobian on Sa
        idx = SaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 3, Eigen::RowMajor>> Dr_DSa(jacobians[idx]);
            Dr_DSa.setZero();
            Dr_DSa.block<3, 3>(0, 0) = w*Dr_DRt*DXt_DXa[Ridx][Sidx];
            Dr_DSa.block<3, 3>(3, 0) = w*Dr_DOt*DXt_DXa[Oidx][Sidx];
        }

        // Jacobian on Pa
        idx = PaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 3, Eigen::RowMajor>> Dr_DPa(jacobians[idx]);
            Dr_DPa.setZero();
            Dr_DPa.block<3, 3>(0, 0) = w*Dr_DAt*DXt_DXa[Aidx][Pidx];
        }

        // Jacobian on Va
        idx = VaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 3, Eigen::RowMajor>> Dr_DVa(jacobians[idx]);
            Dr_DVa.setZero();
            Dr_DVa.block<3, 3>(0, 0) = w*Dr_DAt*DXt_DXa[Aidx][Vidx];
        }

        // Jacobian on Aa
        idx = AaIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 3, Eigen::RowMajor>> Dr_DAa(jacobians[idx]);
            Dr_DAa.setZero();
            Dr_DAa.block<3, 3>(0, 0) = w*Dr_DAt*DXt_DXa[Aidx][Aidx];
        }

        // Jacobian on Rb
        idx = RbIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 4, Eigen::RowMajor>> Dr_DRb(jacobians[idx]);
            Dr_DRb.setZero();
            Dr_DRb.block<3, 3>(0, 0) = 2*w*Dr_DRt*DXt_DXb[Ridx][Ridx];
            Dr_DRb.block<3, 3>(3, 0) = 2*w*Dr_DOt*DXt_DXb[Oidx][Ridx];
        }

        // Jacobian on Ob
        idx = ObIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 3, Eigen::RowMajor>> Dr_DOb(jacobians[idx]);
            Dr_DOb.setZero();
            Dr_DOb.block<3, 3>(0, 0) =  w*Dr_DRt*DXt_DXb[Ridx][Oidx];
            Dr_DOb.block<3, 3>(3, 0) = w*Dr_DOt*DXt_DXb[Oidx][Oidx];
        }

        // Jacobian on Sb
        idx = SbIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 3, Eigen::RowMajor>> Dr_DSb(jacobians[idx]);
            Dr_DSb.setZero();
            Dr_DSb.block<3, 3>(0, 0) =  w*Dr_DRt*DXt_DXb[Ridx][Sidx];
            Dr_DSb.block<3, 3>(3, 0) = w*Dr_DOt*DXt_DXb[Oidx][Sidx];
        }

        // Jacobian on Pb
        idx = PbIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 3, Eigen::RowMajor>> Dr_DPb(jacobians[idx]);
            Dr_DPb.setZero();
            Dr_DPb.block<3, 3>(0, 0) = w*Dr_DAt*DXt_DXb[Aidx][Pidx];
        }

        // Jacobian on Vb
        idx = VbIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 3, Eigen::RowMajor>> Dr_DVb(jacobians[idx]);
            Dr_DVb.setZero();
            Dr_DVb.block<3, 3>(0, 0) = w*Dr_DAt*DXt_DXb[Aidx][Vidx];
        }

        // Jacobian on Ab
        idx = AbIdx;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 3, Eigen::RowMajor>> Dr_DAb(jacobians[idx]);
            Dr_DAb.setZero();
            Dr_DAb.block<3, 3>(0, 0) = w*Dr_DAt*DXt_DXb[Aidx][Aidx];
        }

        idx = 12;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 3, Eigen::RowMajor>> Dr_DBg(jacobians[idx]);
            Dr_DBg.setZero();
            Dr_DBg.block<3, 3>(3, 0) = w*Matrix3d::Identity();
            Dr_DBg.block<3, 3>(6, 0) = w*Matrix3d::Identity();
        }        

        idx = 13;
        if (jacobians[idx])
        {
            Eigen::Map<Eigen::Matrix<double, 12, 3, Eigen::RowMajor>> Dr_DBa(jacobians[idx]);
            Dr_DBa.setZero();
            Dr_DBa.block<3, 3>(0, 0) = w*Matrix3d::Identity();
            Dr_DBa.block<3, 3>(9, 0) = w*Matrix3d::Identity();
        }        
        return true;
    }

private:

    // IMU measurements
    Vector3d acc;
    Vector3d gyro;
    Vector3d acc_bias;
    Vector3d gyro_bias;    

    // Weight
    double w = 10;

    // Gaussian process params
    
    const int Ridx = 0;
    const int Oidx = 1;
    const int Sidx = 2;
    const int Pidx = 3;
    const int Vidx = 4;
    const int Aidx = 5;

    const int RaIdx = 0;
    const int OaIdx = 1;
    const int SaIdx = 2;
    const int PaIdx = 3;
    const int VaIdx = 4;
    const int AaIdx = 5;

    const int RbIdx = 6;
    const int ObIdx = 7;
    const int SbIdx = 8;
    const int PbIdx = 9;
    const int VbIdx = 10;
    const int AbIdx = 11;

    // Spline param
    double Dt;
    double s;

    GPMixerPtr gpm;
};