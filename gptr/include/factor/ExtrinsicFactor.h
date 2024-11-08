#pragma once

#ifndef ExtrinsicFactor_H_
#define ExtrinsicFactor_H_

#include <ceres/ceres.h>
#include "GaussianProcess.hpp"
#include "utility.h"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

class ExtrinsicFactor : public ceres::CostFunction
{
  private:

    SO3d Rij;
    Vector3d Pij;

    // Weight for the pose
    double wR = 1.0;
    double wP = 1.0;

  public:

    ExtrinsicFactor() = delete;
    ExtrinsicFactor( const SO3d     &Rij_,
                     const Vector3d &Pij_,
                     const double wR_,
                     const double wP_)
        : Rij(Rij_), Pij(Pij_), wR(wR_), wP(wP_)
    {
      set_num_residuals(6);
      mutable_parameter_block_sizes()->push_back(4);
      mutable_parameter_block_sizes()->push_back(3);
    }


    Matrix3d rightJacobian(const Vector3d &phi) const
    {
        Matrix3d Jr;
        Sophus::rightJacobianSO3(phi, Jr);
        return Jr;
    }

    Matrix3d rightJacobianInv(const Vector3d &phi) const
    {
        Matrix3d Jr_inv;
        Sophus::rightJacobianInvSO3(phi, Jr_inv);
        return Jr_inv;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        size_t R_offset = 0;            // for quaternion
        size_t P_offset = R_offset + 1; // for position

        SO3d Rh = Eigen::Map<SO3d const>(parameters[R_offset]);
        Vector3d Ph = Eigen::Map<Vector3d const>(parameters[P_offset]);

        Vector3d dR = (Rij.inverse()*Rh).log();
        Vector3d dP = Ph - Pij;

        Eigen::Map<Matrix<double, 6, 1>> residual(residuals);
        residual.block<3, 1>(0, 0) = wR*dR;
        residual.block<3, 1>(3, 0) = wP*dP;

        // cout << "Res:" << endl;
        // cout << residual << endl;

        if (!jacobians)
            return true;

        // Set the jacobian for rotation
        Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> dRdR(jacobians[0]);
        dRdR.setZero();
        dRdR.block<3, 3>(0, 0) = wR*rightJacobianInv(dR);

        // Set the jacobian for position
        Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> dPdP(jacobians[1]);
        dPdP.setZero();
        dPdP.block<3, 3>(3, 0) = wP*Matrix3d::Identity();

        return true;
    }
};

#endif