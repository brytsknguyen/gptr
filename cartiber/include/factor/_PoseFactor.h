/**
 * This file is part of splio.
 *
 * Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot
 * sg>, School of EEE Nanyang Technological Univertsity, Singapore
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


#include <iostream>
#include <ros/assert.h>
#include <ceres/ceres.h>

#include "../utility.h"
#include <basalt/spline/ceres_spline_helper.h>

using namespace basalt;

template <int _N>
class PoseFactor : public basalt::CeresSplineHelper<_N>
{
public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PoseFactor(const Sophus::SE3d &pose_meas_,
               double dt_, double u_,
               double pose_weight_,
               int factor_index_)
        : pose_meas(pose_meas_),
          dt(dt_),
          inv_dt(1.0/dt_),
          u(u_),
          pose_weight(pose_weight_),
          factor_index(factor_index_)
    {}

    template <class T>
    bool operator()(T const *const *sKnots, T *sResiduals) const
    {
        using Vec3T   = Eigen::Matrix<T, 3, 1>;
        using Vec6T   = Eigen::Matrix<T, 6, 1>;
        using SO3T    = Sophus::SO3<T>;
        using Tangent = typename Sophus::SO3<T>::Tangent;

        // Parameter blocks are organized by:
        // [N 4-dimensional quaternions ] + [N 3-dimensional positions]
        size_t R_offset = 0;              // for quaternion
        size_t P_offset = R_offset + _N;  // for position

        SO3T R_w_i;
        CeresSplineHelper<_N>::template evaluate_lie<T, Sophus::SO3>(sKnots + R_offset, u, inv_dt, &R_w_i);

        Vec3T P_w_i;
        CeresSplineHelper<_N>::template evaluate<T, 3, 0>(sKnots + P_offset, u, inv_dt, &P_w_i);

        Vec3T rot_residuals = (pose_meas.so3().inverse()*R_w_i).log();
        Vec3T pos_residuals = P_w_i - pose_meas.translation();

        Eigen::Map<Vec6T> residuals(sResiduals);
        residuals.template block<3, 1>(0, 0) = T(pose_weight) * rot_residuals;
        residuals.template block<3, 1>(3, 0) = T(pose_weight) * pos_residuals;

        return true;
    }

private:
    Sophus::SE3d pose_meas;
    double dt;
    double inv_dt;
    double u;
    double pose_weight;
    int factor_index;   // Used for debugging
};