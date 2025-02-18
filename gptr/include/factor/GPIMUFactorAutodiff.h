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

#include <ceres/ceres.h>
#include "GaussianProcess.hpp"
#include "../utility.h"

using namespace Eigen;

class GPIMUFactorAutodiff
{
public:

    // Destructor
    ~GPIMUFactorAutodiff() {};

    // Constructor
    GPIMUFactorAutodiff(const Vector3d &acc_, const Vector3d &gyro_, const Vector3d &acc_bias_, const Vector3d &gyro_bias_, 
                        double wGyro_, double wAcce_, double wBiasGyro_, double wBiasAcce_, GPMixerPtr gpm_, double s_, int n_knots_)
    :   acc         (acc_             ),
        gyro        (gyro_            ),
        acc_bias    (acc_bias_        ),
        gyro_bias   (gyro_bias_       ),
        wGyro       (wGyro_           ),
        wAcce       (wAcce_           ),
        wBiasGyro   (wBiasGyro_       ),
        wBiasAcce   (wBiasAcce_       ),
        Dt          (gpm_->getDt()    ),
        s           (s_               ),
        gpm         (gpm_             ),
        n_knots     (n_knots_         )

    { }

    template <class T>
    bool operator()(T const *const *parameters, T *residuals) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        GPState<T> Xa(0);  gpm->MapParamToState(parameters, RaIdx, Xa);
        GPState<T> Xb(Dt); gpm->MapParamToState(parameters, RbIdx, Xb);
        Eigen::Matrix<T, 3, 1> biasW = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[12]);        
        Eigen::Matrix<T, 3, 1> biasA = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[13]);    
        Eigen::Matrix<T, 3, 1> g = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[14]);   
        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the pose at sampling time --------------------------------------------------------------*/

        GPState<T> Xt(s*Dt); vector<vector<Matrix<T, 3, 3>>> DXt_DXa; vector<vector<Matrix<T, 3, 3>>> DXt_DXb;
        gpm->ComputeXtAndJacobians(Xa, Xb, Xt, DXt_DXa, DXt_DXb);

        // Residual
        Eigen::Map<Matrix<T, 12, 1>> residual(residuals);      
        // Eigen::Vector3d g(0, 0, 9.81);

        residual.template block<3, 1>(0, 0) = wAcce*(Xt.R.matrix().transpose() * (Xt.A + g) - acc + biasA);
        residual.template block<3, 1>(3, 0) = wGyro*(Xt.O - gyro + biasW);
        residual.template block<3, 1>(6, 0) = wBiasGyro*(biasW - gyro_bias);
        residual.template block<3, 1>(9, 0) = wBiasAcce*(biasA - acc_bias);

        return true;
    }

private:

    // IMU measurements
    Vector3d acc;
    Vector3d gyro;
    Vector3d acc_bias;
    Vector3d gyro_bias;

    // Weight
    double wGyro;
    double wAcce;
    double wBiasGyro;
    double wBiasAcce;

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
    int n_knots;

    GPMixerPtr gpm;
};