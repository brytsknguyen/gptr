#pragma once

#include <ceres/ceres.h>
#include "GaussianProcess.hpp"
#include "utility.h"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

#define RES_DIM 18

class GPExtrinsicFactorAutodiff
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GPExtrinsicFactorAutodiff(double wR_, double wP_, GPMixerPtr gpms_, GPMixerPtr gpmf_, double ss_, double sf_)
    :   wR          (wR_             ),
        wP          (wP_             ),
        Dts         (gpms_->getDt()  ),
        Dtf         (gpmf_->getDt()  ),
        gpms        (gpms_           ),
        gpmf        (gpmf_           ),
        ss          (ss_             ),
        sf          (sf_             )
    {
        // set_num_residuals(18);

        // // Add the knots
        // for(int j = 0; j < 4; j++)
        // {
        //     mutable_parameter_block_sizes()->push_back(4);
        //     mutable_parameter_block_sizes()->push_back(3);
        //     mutable_parameter_block_sizes()->push_back(3);
        //     mutable_parameter_block_sizes()->push_back(3);
        //     mutable_parameter_block_sizes()->push_back(3);
        //     mutable_parameter_block_sizes()->push_back(3);
        // }

        // // Add the extrinsics
        // mutable_parameter_block_sizes()->push_back(4);
        // mutable_parameter_block_sizes()->push_back(3);
        
        // // Find the square root info
        sqrtW = Matrix<double, RES_DIM, RES_DIM>::Identity(RES_DIM, RES_DIM);
        sqrtW.block<3, 3>(0, 0) = Vector3d(wR, wR, wR).asDiagonal();
        sqrtW.block<3, 3>(9, 9) = Vector3d(wP, wP, wP).asDiagonal();
    }

    template <class T>
    bool operator()(T const *const *parameters, T *residuals) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        GPState<T> Xsa(0);   gpms->MapParamToState<T>(parameters, RsaIdx, Xsa);
        GPState<T> Xsb(Dts); gpms->MapParamToState<T>(parameters, RsbIdx, Xsb);

        // Map parameters to the control point states
        GPState<T> Xfa(0);   gpmf->MapParamToState<T>(parameters, RfaIdx, Xfa);
        GPState<T> Xfb(Dtf); gpmf->MapParamToState<T>(parameters, RfbIdx, Xfb);

        SO3T  Rsf = Eigen::Map<SO3T  const>(parameters[RsfIdx]);
        Vec3T Psf = Eigen::Map<Vec3T const>(parameters[PsfIdx]);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Compute the interpolated states ------------------------------------------------------------------*/

        GPState<T> Xst(ss*Dts); vector<vector<Mat3T>> DXst_DXsa; vector<vector<Mat3T>> DXst_DXsb;
        GPState<T> Xft(sf*Dtf); vector<vector<Mat3T>> DXft_DXfa; vector<vector<Mat3T>> DXft_DXfb;

        Eigen::Matrix<T, 9, 1> gammasa, gammasb, gammast;
        Eigen::Matrix<T, 9, 1> gammafa, gammafb, gammaft;

        gpms->ComputeXtAndJacobians<T>(Xsa, Xsb, Xst, DXst_DXsa, DXst_DXsb, gammasa, gammasb, gammast);
        gpmf->ComputeXtAndJacobians<T>(Xfa, Xfb, Xft, DXft_DXfa, DXft_DXfb, gammafa, gammafb, gammaft);

        /* #endregion Compute the interpolated states ---------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/
        
        Mat3T Ostx = SO3T::hat(Xst.O);
        Mat3T Oftx = SO3T::hat(Xft.O);
        Mat3T Sstx = SO3T::hat(Xst.S);
        Mat3T Sftx = SO3T::hat(Xft.S);
        Vec3T OstxPsf = Ostx*Psf;
        Vec3T SstxPsf = Sstx*Psf;

        Vec3T rR = (Rsf.inverse()*Xst.R.inverse()*Xft.R).log();
        Vec3T rO = Rsf*Xft.O - Xst.O;
        Vec3T rS = Rsf*Xft.S - Xst.S;
        Vec3T rP = Xft.P - Xst.P - Xst.R*Psf;
        Vec3T rV = Xft.V - Xst.V - Xst.R*OstxPsf;
        Vec3T rA = Xft.A - Xst.A - Xst.R*SstxPsf - Xst.R*(Ostx*OstxPsf);

        // Residual
        Eigen::Map<Matrix<T, 18, 1>> residual(residuals);
        residual << rR, rO, rS, rP, rV, rA;
        residual = sqrtW*residual;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        return true;
    }

private:

    const int RIdx = 0;
    const int OIdx = 1;
    const int SIdx = 2;
    const int PIdx = 3;
    const int VIdx = 4;
    const int AIdx = 5;
    
    const int RsaIdx = 0;
    const int OsaIdx = 1;
    const int SsaIdx = 2;
    const int PsaIdx = 3;
    const int VsaIdx = 4;
    const int AsaIdx = 5;

    const int RsbIdx = 6;
    const int OsbIdx = 7;
    const int SsbIdx = 8;
    const int PsbIdx = 9;
    const int VsbIdx = 10;
    const int AsbIdx = 11;

    const int RfaIdx = 12;
    const int OfaIdx = 13;
    const int SfaIdx = 14;
    const int PfaIdx = 15;
    const int VfaIdx = 16;
    const int AfaIdx = 17;

    const int RfbIdx = 18;
    const int OfbIdx = 19;
    const int SfbIdx = 20;
    const int PfbIdx = 21;
    const int VfbIdx = 22;
    const int AfbIdx = 23;

    const int RsfIdx = 24;
    const int PsfIdx = 25;

    double wR;
    double wP;

    // Square root information
    Matrix<double, 18, 18> sqrtW;
    
    // Knot length
    double Dts;
    double Dtf;

    // Normalized time on each traj
    double ss;
    double sf;

    // Mixer for gaussian process
    GPMixerPtr gpms;
    GPMixerPtr gpmf;
};