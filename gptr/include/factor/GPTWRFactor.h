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

class GPTWRFactor: public ceres::CostFunction
{
public:

    // Destructor
    ~GPTWRFactor() {};

    // Constructor
    GPTWRFactor(double twr_, const Vector3d &p_W_anc_, const Vector3d &p_B_tag_, double w_,
                GPMixerPtr gpm_, double s_)
    :   twr         (twr_            ),
        p_W_anc     (p_W_anc_         ),
        p_B_tag     (p_B_tag_         ),
        w           (w_               ),
        Dt          (gpm_->getDt()    ),
        s           (s_               ),
        gpm         (gpm_             )

    {
        // 1-element residual: || p_itp - pos_an_i || - || p_itp - pos_an_j || - twr
        set_num_residuals(1);

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
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        GPState Xa(0);  gpm->MapParamToState(parameters, RaIdx, Xa);
        GPState Xb(Dt); gpm->MapParamToState(parameters, RbIdx, Xb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the pose at sampling time --------------------------------------------------------------*/

        GPState Xt(s*Dt); vector<vector<Matrix3d>> DXt_DXa; vector<vector<Matrix3d>> DXt_DXb;
        gpm->ComputeXtAndJacobians(Xa, Xb, Xt, DXt_DXa, DXt_DXb);

        // Residual
        Eigen::Map<Matrix<double, 1, 1>> residual(residuals);
        Eigen::Vector3d p_W_tag = Xt.R.matrix() * p_B_tag + Xt.P;
        Eigen::Vector3d pat = p_W_tag - p_W_anc;
        residual[0] = w*(pat.norm() - twr);

        /* #endregion Calculate the pose at sampling time -----------------------------------------------------------*/

        if (!jacobians)
            return true;

        constexpr int RES_SIZE = 1;

        Matrix<double, 1, 3> Dr_Dpat =  pat.normalized().transpose();   
        Matrix<double, 1, 3> Dr_DRt  = -w*Dr_Dpat * Xt.R.matrix() * SO3d::hat(p_B_tag);
        Matrix<double, 1, 3> Dr_DPt  =  w*Dr_Dpat;        

        size_t idx;

        for(size_t idx = Ridx; idx <= Aidx; idx++)
        {
            size_t idxa = idx, idxb = idx+RbIdx;

            if (idx == Ridx)
            {
                Eigen::Map<Eigen::Matrix<double, RES_SIZE, 4, Eigen::RowMajor>> Dr_DXa(jacobians[idxa]);
                Eigen::Map<Eigen::Matrix<double, RES_SIZE, 4, Eigen::RowMajor>> Dr_DXb(jacobians[idxb]);

                if(jacobians[idxa]) { Dr_DXa.setZero(); Dr_DXa.block<RES_SIZE, 3>(0, 0) = Dr_DRt*DXt_DXa[Ridx][idx] + Dr_DPt*DXt_DXa[Pidx][idx]; }
                if(jacobians[idxb]) { Dr_DXb.setZero(); Dr_DXb.block<RES_SIZE, 3>(0, 0) = Dr_DRt*DXt_DXb[Ridx][idx] + Dr_DPt*DXt_DXb[Pidx][idx]; }
            }
            else
            {
                Eigen::Map<Eigen::Matrix<double, RES_SIZE, 3, Eigen::RowMajor>> Dr_DXa(jacobians[idxa]);
                Eigen::Map<Eigen::Matrix<double, RES_SIZE, 3, Eigen::RowMajor>> Dr_DXb(jacobians[idxb]);

                if(jacobians[idxa]) { Dr_DXa.setZero(); Dr_DXa.block<RES_SIZE, 3>(0, 0) = Dr_DRt*DXt_DXa[Ridx][idx] + Dr_DPt*DXt_DXa[Pidx][idx]; }
                if(jacobians[idxb]) { Dr_DXb.setZero(); Dr_DXb.block<RES_SIZE, 3>(0, 0) = Dr_DRt*DXt_DXb[Ridx][idx] + Dr_DPt*DXt_DXb[Pidx][idx]; }
            }
        }

        // // Jacobian on Ra
        // idx = RaIdx;
        // if (jacobians[idx])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> Dr_DRa(jacobians[idx]);
        //     Dr_DRa.setZero();
        //     Dr_DRa.block<1, 3>(0, 0) = w*Dr_DRt*DXt_DXa[Ridx][Ridx];
        // }

        // // Jacobian on Oa
        // idx = OaIdx;
        // if (jacobians[idx])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> Dr_DOa(jacobians[idx]);
        //     Dr_DOa.setZero();
        //     Dr_DOa.block<1, 3>(0, 0) = w*Dr_DRt*DXt_DXa[Ridx][Oidx];
        // }

        // // Jacobian on Sa
        // idx = SaIdx;
        // if (jacobians[idx])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> Dr_DSa(jacobians[idx]);
        //     Dr_DSa.setZero();
        //     Dr_DSa.block<1, 3>(0, 0) = w*Dr_DRt*DXt_DXa[Ridx][Sidx];
        // }

        // // Jacobian on Pa
        // idx = PaIdx;
        // if (jacobians[idx])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> Dr_DPa(jacobians[idx]);
        //     Dr_DPa.setZero();
        //     Dr_DPa.block<1, 3>(0, 0) = w*Dr_DPt*DXt_DXa[Pidx][Pidx];
        // }

        // // Jacobian on Va
        // idx = VaIdx;
        // if (jacobians[idx])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> Dr_DVa(jacobians[idx]);
        //     Dr_DVa.setZero();
        //     Dr_DVa.block<1, 3>(0, 0) = w*Dr_DPt*DXt_DXa[Pidx][Vidx];
        // }

        // // Jacobian on Aa
        // idx = AaIdx;
        // if (jacobians[idx])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> Dr_DAa(jacobians[idx]);
        //     Dr_DAa.setZero();
        //     Dr_DAa.block<1, 3>(0, 0) = w*Dr_DPt*DXt_DXa[Pidx][Aidx];
        // }

        // // Jacobian on Rb
        // idx = RbIdx;
        // if (jacobians[idx])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> Dr_DRb(jacobians[idx]);
        //     Dr_DRb.setZero();
        //     Dr_DRb.block<1, 3>(0, 0) = w*Dr_DRt*DXt_DXb[Ridx][Ridx];
        // }

        // // Jacobian on Ob
        // idx = ObIdx;
        // if (jacobians[idx])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> Dr_DOb(jacobians[idx]);
        //     Dr_DOb.setZero();
        //     Dr_DOb.block<1, 3>(0, 0) =  w*Dr_DRt*DXt_DXb[Ridx][Oidx];
        // }

        // // Jacobian on Sb
        // idx = SbIdx;
        // if (jacobians[idx])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> Dr_DSb(jacobians[idx]);
        //     Dr_DSb.setZero();
        //     Dr_DSb.block<1, 3>(0, 0) =  w*Dr_DRt*DXt_DXb[Ridx][Sidx];
        // }

        // // Jacobian on Pb
        // idx = PbIdx;
        // if (jacobians[idx])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> Dr_DPb(jacobians[idx]);
        //     Dr_DPb.setZero();
        //     Dr_DPb.block<1, 3>(0, 0) = w*Dr_DPt*DXt_DXb[Pidx][Pidx];
        // }

        // // Jacobian on Vb
        // idx = VbIdx;
        // if (jacobians[idx])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> Dr_DVb(jacobians[idx]);
        //     Dr_DVb.setZero();
        //     Dr_DVb.block<1, 3>(0, 0) = w*Dr_DPt*DXt_DXb[Pidx][Vidx];
        // }

        // // Jacobian on Ab
        // idx = AbIdx;
        // if (jacobians[idx])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> Dr_DAb(jacobians[idx]);
        //     Dr_DAb.setZero();
        //     Dr_DAb.block<1, 3>(0, 0) = w*Dr_DPt*DXt_DXb[Pidx][Aidx];
        // }

        return true;
    }

private:

    // twr measurement
    double twr;

    // Anchor positions
    Vector3d p_W_anc;
    const Vector3d p_B_tag;

    // Weight
    double w = 10;

    // Gaussian process param
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