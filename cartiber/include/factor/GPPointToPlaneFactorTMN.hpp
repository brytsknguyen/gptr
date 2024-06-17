#pragma once

#include <ceres/ceres.h>
// #include "basalt/spline/ceres_spline_helper.h"
// #include "basalt/utils/sophus_utils.hpp"
#include "../utility.h"
#include "GaussianProcess.hpp"

using namespace Eigen;

// Matrix defined for this factor's jacobian

class GPPointToPlaneFactorTMN
{
typedef Eigen::Matrix<double, 1, 2*STATE_DIM> MatJ;
public:

    // Destructor
    ~GPPointToPlaneFactorTMN() {};

    // Constructor
    GPPointToPlaneFactorTMN(const Vector3d &finW_, const Vector3d &f_, const Vector4d &coef, double w_,
                            double Dt_, double s_)
    :   finW       (finW_            ),
        f          (f_               ),
        n          (coef.head<3>()   ),
        m          (coef.tail<1>()(0)),
        w          (w_               ),
        Dt         (Dt_              ),
        s          (s_               ),
        gpm        (Dt_              )

    {
        // // 1-element residual: n^T*(Rt*f + pt) + m
        // set_num_residuals(1);

        // // Rotation of the first knot
        // mutable_parameter_block_sizes()->push_back(4);
        // // Angular velocity of the first knot
        // mutable_parameter_block_sizes()->push_back(3);
        // // Position of the first knot
        // mutable_parameter_block_sizes()->push_back(3);
        // // Velocity of the first knot
        // mutable_parameter_block_sizes()->push_back(3);
        // // Acceleration of the first knot
        // mutable_parameter_block_sizes()->push_back(3);

        // // Rotation of the second knot
        // mutable_parameter_block_sizes()->push_back(4);
        // // Angular velocity of the second knot
        // mutable_parameter_block_sizes()->push_back(3);
        // // Position of the second knot
        // mutable_parameter_block_sizes()->push_back(3);
        // // Velocity of the second knot
        // mutable_parameter_block_sizes()->push_back(3);
        // // Acceleration of the second knot
        // mutable_parameter_block_sizes()->push_back(3);

        residual = 0;
        jacobian.setZero();
    }

    bool Evaluate(const StateStamped<double> &Xa, const StateStamped<double> &Xb, bool computeJacobian=true)
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        // StateStamped Xa(0);  gpm.MapParamToState(parameters, RaIdx, Xa);
        // StateStamped Xb(Dt); gpm.MapParamToState(parameters, RbIdx, Xb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the pose at sampling time --------------------------------------------------------------*/

        StateStamped Xt(s*Dt); vector<vector<Matrix3d>> DXt_DXa; vector<vector<Matrix3d>> DXt_DXb;

        Eigen::Matrix<double, 6, 1> gammaa;
        Eigen::Matrix<double, 6, 1> gammab;
        Eigen::Matrix<double, 6, 1> gammat;

        gpm.ComputeXtAndDerivs(Xa, Xb, Xt, DXt_DXa, DXt_DXb, gammaa, gammab, gammat);

        // Residual
        // Eigen::Map<Matrix<double, 1, 1>> residual(residuals);
        residual = w*(n.dot(Xt.R*f + Xt.P) + m);

        /* #endregion Calculate the pose at sampling time -----------------------------------------------------------*/
    
        if (!computeJacobian)
            return true;

        Matrix<double, 1, 3> Dr_DRt  = -n.transpose()*Xt.R.matrix()*SO3d::hat(f);
        Matrix<double, 1, 3> Dr_DPt  =  n.transpose();

        size_t idx;

        // Jacobian on Ra
        idx = RaIdx;
        Eigen::Block<MatJ, 1, 3> Dr_DRa(jacobian.block<1, 3>(0, idx));
        Dr_DRa.setZero();
        Dr_DRa.block<1, 3>(0, 0) = w*Dr_DRt*DXt_DXa[Ridx][Ridx];


        // Jacobian on Oa
        idx = OaIdx;
        Eigen::Block<MatJ, 1, 3> Dr_DOa(jacobian.block<1, 3>(0, idx));
        Dr_DOa.setZero();
        Dr_DOa.block<1, 3>(0, 0) = w*Dr_DRt*DXt_DXa[Ridx][Oidx];


        // Jacobian on Pa
        idx = PaIdx;
        Eigen::Block<MatJ, 1, 3> Dr_DPa(jacobian.block<1, 3>(0, idx));
        Dr_DPa.setZero();
        Dr_DPa.block<1, 3>(0, 0) = w*Dr_DPt*DXt_DXa[Pidx][Pidx];


        // Jacobian on Va
        idx = VaIdx;
        Eigen::Block<MatJ, 1, 3> Dr_DVa(jacobian.block<1, 3>(0, idx));
        Dr_DVa.setZero();
        Dr_DVa.block<1, 3>(0, 0) = w*Dr_DPt*DXt_DXa[Pidx][Vidx];


        // Jacobian on Aa
        idx = AaIdx;
        Eigen::Block<MatJ, 1, 3> Dr_DAa(jacobian.block<1, 3>(0, idx));
        Dr_DAa.setZero();
        Dr_DAa.block<1, 3>(0, 0) = w*Dr_DPt*DXt_DXa[Pidx][Aidx];


        // Jacobian on Rb
        idx = RbIdx;
        Eigen::Block<MatJ, 1, 3> Dr_DRb(jacobian.block<1, 3>(0, idx));
        Dr_DRb.setZero();
        Dr_DRb.block<1, 3>(0, 0) = w*Dr_DRt*DXt_DXb[Ridx][Ridx];


        // Jacobian on Ob
        idx = ObIdx;
        Eigen::Block<MatJ, 1, 3> Dr_DOb(jacobian.block<1, 3>(0, idx));
        Dr_DOb.setZero();
        Dr_DOb.block<1, 3>(0, 0) =  w*Dr_DRt*DXt_DXb[Ridx][Oidx];


        // Jacobian on Pb
        idx = PbIdx;
        Eigen::Block<MatJ, 1, 3> Dr_DPb(jacobian.block<1, 3>(0, idx));
        Dr_DPb.setZero();
        Dr_DPb.block<1, 3>(0, 0) = w*Dr_DPt*DXt_DXb[Pidx][Pidx];


        // Jacobian on Vb
        idx = VbIdx;
        Eigen::Block<MatJ, 1, 3> Dr_DVb(jacobian.block<1, 3>(0, idx));
        Dr_DVb.setZero();
        Dr_DVb.block<1, 3>(0, 0) = w*Dr_DPt*DXt_DXb[Pidx][Vidx];


        // Jacobian on Ab
        idx = AbIdx;
        Eigen::Block<MatJ, 1, 3> Dr_DAb(jacobian.block<1, 3>(0, idx));
        Dr_DAb.setZero();
        Dr_DAb.block<1, 3>(0, 0) = w*Dr_DPt*DXt_DXb[Pidx][Aidx];


        return true;
    }

    // Residual and Jacobian
    double residual;
    MatJ jacobian;

private:

    // Feature coordinates in world frame
    Vector3d finW;

    // Feature coordinates in body frame
    Vector3d f;

    // Plane normal
    Vector3d n;

    // Plane offset
    double m;

    // Weight
    double w = 0.1;

    // Gaussian process params
    
    const int Ridx = 0;
    const int Oidx = 1;
    const int Pidx = 2;
    const int Vidx = 3;
    const int Aidx = 4;

    const int RaIdx = 0;
    const int OaIdx = 3;
    const int PaIdx = 6;
    const int VaIdx = 9;
    const int AaIdx = 12;

    const int RbIdx = 15;
    const int ObIdx = 18;
    const int PbIdx = 21;
    const int VbIdx = 24;
    const int AbIdx = 27;

    // Interpolation param
    double Dt;
    double s;

    GPMixer gpm;
};