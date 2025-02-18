#pragma once

#include <ceres/ceres.h>
#include "GaussianProcess.hpp"
#include "utility.h"

class GPMotionPriorTwoKnotsFactorAutodiff
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GPMotionPriorTwoKnotsFactorAutodiff(GPMixerPtr gpm_)
    :   Dt          (gpm_->getDt()   ),
        gpm         (gpm_            )
    {
        // // 6-element residual: (3x1 rotation, 3x1 position)
        // set_num_residuals(STATE_DIM); // Angular diff, angular vel, angular acce, pos diff, vel diff, acc diff

        // for(int j = 0; j < 2; j++)
        // {
        //     mutable_parameter_block_sizes()->push_back(4);
        //     mutable_parameter_block_sizes()->push_back(3);
        //     mutable_parameter_block_sizes()->push_back(3);
        //     mutable_parameter_block_sizes()->push_back(3);
        //     mutable_parameter_block_sizes()->push_back(3);
        //     mutable_parameter_block_sizes()->push_back(3);
        // }

        // Calculate the information matrix
        Matrix<double, STATE_DIM, STATE_DIM> Info;
        Info.setZero();

        double Dtpow[7];
        for(int j = 0; j < 7; j++)
            Dtpow[j] = pow(Dt, j);

        Matrix3d Qtilde;
        Qtilde << 1.0/20.0*Dtpow[5], 1.0/8.0*Dtpow[4], 1.0/6.0*Dtpow[3],
                  1.0/08.0*Dtpow[4], 1.0/3.0*Dtpow[3], 1.0/2.0*Dtpow[2],
                  1.0/06.0*Dtpow[3], 1.0/2.0*Dtpow[2], 1.0/1.0*Dtpow[1];
        Info.block<9, 9>(0, 0) = gpm->kron(Qtilde, gpm->getCovROSJerk());
        Info.block<9, 9>(9, 9) = gpm->kron(Qtilde, gpm->getCovPVAJerk());
        
        // Find the square root info
        // sqrtW = Matrix<double, STATE_DIM, STATE_DIM>::Identity(STATE_DIM, STATE_DIM);
        sqrtW = Eigen::LLT<Matrix<double, STATE_DIM, STATE_DIM>>(Info.inverse()).matrixL().transpose();
    }

    template <class T>
    bool operator()(T const *const *parameters, T *residuals) const
    {
        using Vec18T   = Eigen::Matrix<T, STATE_DIM, 1>;
        using Mat18x3T = Eigen::Matrix<T, STATE_DIM, 3>;

        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        GPState<T> Xa(0);  gpm->MapParamToState<T>(parameters, RaIdx, Xa);
        GPState<T> Xb(Dt); gpm->MapParamToState<T>(parameters, RbIdx, Xb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        constexpr int RES_SIZE = 18;

        // Compute residual and jacobian
        Vec18T residual_; vector<Mat18x3T> Dr_DXa_(6, Mat18x3T::Zero()); vector<Mat18x3T> Dr_DXb_(6, Mat18x3T::Zero());
        gpm->ComputeMotionPriorFactor<T>(Xa, Xb, residual_, Dr_DXa_, Dr_DXb_, false);
        
        // Loa the residual
        Eigen::Map<Vec18T> residual(residuals); residual << residual_;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        return true;
    }

private:

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

    // Square root information
    Matrix<double, STATE_DIM, STATE_DIM> sqrtW;
    
    // Knot length
    double Dt;
    
    // Mixer for gaussian process
    GPMixerPtr gpm;
};