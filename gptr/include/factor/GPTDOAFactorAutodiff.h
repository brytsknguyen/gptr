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

class GPTDOAFactorAutodiff
{
public:

    // Destructor
    ~GPTDOAFactorAutodiff() {};

    // Constructor
    GPTDOAFactorAutodiff(double tdoa_, const Vector3d &pos_anchor_i_, const Vector3d &pos_anchor_j_, const Vector3d &offset_, double w_,
                         GPMixerPtr gpm_, double s_)
    :   tdoa        (tdoa_            ),
        pos_anchor_i(pos_anchor_i_    ),
        pos_anchor_j(pos_anchor_j_    ),
        offset      (offset_          ),
        w           (w_               ),
        Dt          (gpm_->getDt()    ),
        s           (s_               ),
        gpm         (gpm_             )

    { }

    template <class T>
    bool operator()(T const *const *parameters, T *residuals) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        GPState<T> Xa(0);  gpm->MapParamToState<T>(parameters, RaIdx, Xa);
        GPState<T> Xb(Dt); gpm->MapParamToState<T>(parameters, RbIdx, Xb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the pose at sampling time --------------------------------------------------------------*/

        GPState<T> Xt(s*Dt); 
        vector<vector<Matrix<T, 3, 3>>> DXt_DXa; vector<vector<Matrix<T, 3, 3>>> DXt_DXb;

        Eigen::Matrix<T, Eigen::Dynamic, 1> gammaa;
        Eigen::Matrix<T, Eigen::Dynamic, 1> gammab;
        Eigen::Matrix<T, Eigen::Dynamic, 1> gammat;

        gpm->ComputeXtAndJacobians<T>(Xa, Xb, Xt, DXt_DXa, DXt_DXb, gammaa, gammab, gammat);

        // Residual
        Eigen::Map<Matrix<T, 1, 1>> residual(residuals);
        Eigen::Matrix<T, 3, 1> p_tag_W = Xt.R.matrix() * offset + Xt.P;
        Eigen::Matrix<T, 3, 1> diff_i = p_tag_W - pos_anchor_i;
        Eigen::Matrix<T, 3, 1> diff_j = p_tag_W - pos_anchor_j;        
        residual[0] = w*(diff_j.norm() - diff_i.norm() - tdoa);

        return true;
    }

private:

    // TDOA measurement
    double tdoa;

    // Anchor positions
    Vector3d pos_anchor_i;
    Vector3d pos_anchor_j;
    const Vector3d offset;

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