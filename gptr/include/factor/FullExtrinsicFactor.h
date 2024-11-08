#pragma once

#ifndef FullExtrinsicFactor_H_
#define FullExtrinsicFactor_H_

#include <ceres/ceres.h>
#include "GaussianProcess.hpp"
#include "utility.h"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;
using Vec3 = Vector3d;
using Mat3 = Matrix3d;

class FullExtrinsicFactor : public ceres::CostFunction
{
  private:

    SO3d Ri;
    Vec3 Oi;
    Vec3 Si;
    Vec3 Pi;
    Vec3 Vi;
    Vec3 Ai;

    SO3d Rj;
    Vec3 Oj;
    Vec3 Sj;
    Vec3 Pj;
    Vec3 Vj;
    Vec3 Aj;

    SO3d Rij;
    Vec3 Pij;
    Vec3 Vij;
    Vec3 Aij;
    Mat3 Oix;
    Mat3 Six;
    Mat3 SOx;
    Mat3 Ojx;
    Mat3 Sjx;

    // Weight for the pose
    double wR = 1.0;
    double wP = 1.0;

  public:

    FullExtrinsicFactor() = delete;
    FullExtrinsicFactor( const SO3d &Ri_, const Vec3 &Oi_, const Vec3 &Si_, const Vec3 &Pi_, const Vec3 &Vi_, const Vec3 &Ai_,
                         const SO3d &Rj_, const Vec3 &Oj_, const Vec3 &Sj_, const Vec3 &Pj_, const Vec3 &Vj_, const Vec3 &Aj_,
                         const double wR_,
                         const double wP_)
        : Ri(Ri_), Oi(Oi_), Si(Si_), Pi(Pi_), Vi(Vi_), Ai(Ai_),
          Rj(Rj_), Oj(Oj_), Sj(Sj_), Pj(Pj_), Vj(Vj_), Aj(Aj_),
          wR(wR_), wP(wP_)
    {
      Rij = Ri.inverse()*Rj;

      Pij = Ri.inverse()*(Pj - Pi);
      Vij = Ri.inverse()*(Vj - Vi);
      Aij = Ri.inverse()*(Aj - Ai);

      Oix = SO3d::hat(Oi);
      Six = SO3d::hat(Si);

      SOx = Six + Oix*Oix;

      Ojx = SO3d::hat(Oj);
      Sjx = SO3d::hat(Sj);

      set_num_residuals(18);
      mutable_parameter_block_sizes()->push_back(4);
      mutable_parameter_block_sizes()->push_back(3);
    }


    Matrix3d rightJacobian(const Vec3 &phi) const
    {
        Matrix3d Jr;
        Sophus::rightJacobianSO3(phi, Jr);
        return Jr;
    }

    Matrix3d rightJacobianInv(const Vec3 &phi) const
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
        Vec3 Ph = Eigen::Map<Vec3 const>(parameters[P_offset]);

        Vec3 dR = (Rij.inverse()*Rh).log();
        Vec3 dO = Rij*Oj - Oi;
        Vec3 dS = Rij*Sj - Si;
        Vec3 dP = Ph - Pij;
        Vec3 dV = 0*Oix*Pij - Vij;
        Vec3 dA = 0*SOx*Pij - Aij;

        Eigen::Map<Matrix<double, 18, 1>> residual(residuals);
        residual << wR*dR, wR*dO, wR*dS,
                    wP*dP, wP*dV, wP*dA;

        // cout << "Res:" << endl;
        // cout << residual << endl;

        if (!jacobians)
            return true;

        // Set the jacobian for rotation
        Eigen::Map<Eigen::Matrix<double, 18, 4, Eigen::RowMajor>> dRdR(jacobians[0]);
        dRdR.setZero();
        dRdR.block<3, 3>(0, 0) =  wR*rightJacobianInv(dR);
        dRdR.block<3, 3>(3, 0) = -wR*Rij.matrix()*Ojx;
        dRdR.block<3, 3>(6, 0) = -wR*Rij.matrix()*Sjx;

        // Set the jacobian for position
        Eigen::Map<Eigen::Matrix<double, 18, 3, Eigen::RowMajor>> dPdP(jacobians[1]);
        dPdP.setZero();
        dPdP.block<3, 3>(9,  0) = wP*Matrix3d::Identity();
        // dPdP.block<3, 3>(12, 0) = wP*Oix;
        // dPdP.block<3, 3>(15, 0) = wP*SOx;
        
        return true;
    }
};

#endif