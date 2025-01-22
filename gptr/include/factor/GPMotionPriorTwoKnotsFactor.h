#pragma once

#include <ceres/ceres.h>
#include "GaussianProcess.hpp"
#include "utility.h"

class GPMotionPriorTwoKnotsFactor : public ceres::CostFunction
{
public:

    GPMotionPriorTwoKnotsFactor(GPMixerPtr gpm_)
    :   Dt          (gpm_->getDt()   ),
        gpm         (gpm_            )
    {
        // 6-element residual: (3x1 rotation, 3x1 position)
        set_num_residuals(STATE_DIM); // Angular diff, angular vel, angular acce, pos diff, vel diff, acc diff

        for(int j = 0; j < 2; j++)
        {
            mutable_parameter_block_sizes()->push_back(4);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
        }

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
        Info.block<9, 9>(0, 0) = gpm->kron(Qtilde, gpm->getSigGa());
        Info.block<9, 9>(9, 9) = gpm->kron(Qtilde, gpm->getSigNu());
        
        // Find the square root info
        // sqrtW = Matrix<double, STATE_DIM, STATE_DIM>::Identity(STATE_DIM, STATE_DIM);
        sqrtW = Eigen::LLT<Matrix<double, STATE_DIM, STATE_DIM>>(Info.inverse()).matrixL().transpose();
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        GPState Xa(0);  gpm->MapParamToState(parameters, RaIdx, Xa);
        GPState Xb(Dt); gpm->MapParamToState(parameters, RbIdx, Xb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        SO3d Rab = Xa.R.inverse()*Xb.R;
        Vec3 Theb = Rab.log();

        Mat3 JrInvTheb = gpm->JrInv(Theb);
        Mat3 DJrInvThebOb_DTheb = gpm->DJrInvUV_DU(Theb, Xb.O);

        Vec3 Thedotb = JrInvTheb*Xb.O;
        Vec3 Theddotb = DJrInvThebOb_DTheb*Thedotb + JrInvTheb*Xb.S;

        double Dtsq = Dt*Dt;

        // Rotational residual
        Vec3 rRot = Theb - Dt*Xa.O - 0.5*Dtsq*Xa.S;

        // Rotational rate residual
        Vec3 rRdot = Thedotb - Xa.O - Dt*Xa.S;

        // Rotational acc residual
        Vec3 rRddot = Theddotb - Xa.S;

        // Positional residual
        Vec3 rPos = Xb.P - (Xa.P + Dt*Xa.V + 0.5*Dtsq*Xa.A);

        // Velocity residual
        Vec3 rVel = Xb.V - (Xa.V + Dt*Xa.A);

        // Acceleration residual
        Vec3 rAcc = Xb.A - Xa.A;

        // Residual
        Eigen::Map<Matrix<double, STATE_DIM, 1>> residual(residuals);
        residual << rRot, rRdot, rRddot, rPos, rVel, rAcc;
        residual = sqrtW*residual;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!jacobians)
            return true;

        Mat3 Eye = Mat3::Identity();
        Mat3 DtI = Vec3(Dt, Dt, Dt).asDiagonal();
        double DtsqDiv2 = 0.5*Dtsq;
        Mat3 DtsqDiv2I = Vec3(DtsqDiv2, DtsqDiv2, DtsqDiv2).asDiagonal();

        // Reusable Jacobians
        Mat3 DTheb_DRa = -JrInvTheb*Rab.inverse().matrix();
        Mat3 DTheb_DRb =  JrInvTheb;

        Mat3 DThedotb_DTheb = gpm->DJrInvUV_DU(Theb, Xb.O);
        Mat3 DThedotb_DRa = DThedotb_DTheb*DTheb_DRa;
        Mat3 DThedotb_DRb = DThedotb_DTheb*DTheb_DRb;

        Mat3 DJrInvThebSb_DTheb = gpm->DJrInvUV_DU(Theb, Xb.S);
        // Mat3 DJrInvThebOb_DTheb = gpm->DJrInvUV_DU(Theb, Xb.O);
        Mat3 DDJrInvThebObThedotb_DThebDTheb = gpm->DDJrInvUVW_DUDU(Theb, Xb.O, Thedotb);
        
        Mat3 DTheddotb_DTheb = DJrInvThebSb_DTheb + DDJrInvThebObThedotb_DThebDTheb + DJrInvThebOb_DTheb*DJrInvThebOb_DTheb;
        Mat3 DTheddotb_DRa = DTheddotb_DTheb*DTheb_DRa;
        Mat3 DTheddotb_DRb = DTheddotb_DTheb*DTheb_DRb;

        Mat3 DDJrInvThebObThedotb_DThebDOb = gpm->DDJrInvUVW_DUDV(Theb, Xb.O, Thedotb);
        Mat3 DTheddotb_DOb = DDJrInvThebObThedotb_DThebDOb + DJrInvThebOb_DTheb*JrInvTheb;        

        size_t idx;

        // Work out the Jacobians for SO3 states first
        {
            // dr_dRa
            idx = RaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, STATE_DIM, 4, Eigen::RowMajor>> Dr_DRa(jacobians[idx]);
                Dr_DRa.setZero();
                Dr_DRa.block<3, 3>(0, 0) = DTheb_DRa;
                Dr_DRa.block<3, 3>(3, 0) = DThedotb_DRa;
                Dr_DRa.block<3, 3>(6, 0) = DTheddotb_DRa;
                Dr_DRa = sqrtW*Dr_DRa;
            }

            // dr_dOa
            idx = OaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, STATE_DIM, 3, Eigen::RowMajor>> Dr_DOa(jacobians[idx]);
                Dr_DOa.setZero();
                Dr_DOa.block<3, 3>(0, 0) = -DtI;
                Dr_DOa.block<3, 3>(3, 0) = -Eye;
                Dr_DOa = sqrtW*Dr_DOa;
            }

            // dr_dSa
            idx = SaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, STATE_DIM, 3, Eigen::RowMajor>> Dr_DSa(jacobians[idx]);
                Dr_DSa.setZero();
                Dr_DSa.block<3, 3>(0, 0) = -DtsqDiv2I;
                Dr_DSa.block<3, 3>(3, 0) = -DtI;
                Dr_DSa.block<3, 3>(6, 0) = -Eye;
                Dr_DSa = sqrtW*Dr_DSa;
            }

            // dr_dRb
            idx = RbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, STATE_DIM, 4, Eigen::RowMajor>> Dr_DRb(jacobians[idx]);
                Dr_DRb.setZero();
                Dr_DRb.block<3, 3>(0, 0) = DTheb_DRb;
                Dr_DRb.block<3, 3>(3, 0) = DThedotb_DRb;
                Dr_DRb.block<3, 3>(6, 0) = DTheddotb_DRb;
                Dr_DRb = sqrtW*Dr_DRb;
            }

            // dr_dOb
            idx = ObIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, STATE_DIM, 3, Eigen::RowMajor>> Dr_DOb(jacobians[idx]);
                Dr_DOb.setZero();
                Dr_DOb.block<3, 3>(3, 0) = JrInvTheb;
                Dr_DOb.block<3, 3>(6, 0) = DTheddotb_DOb;
                Dr_DOb = sqrtW*Dr_DOb;
            }

            // dr_dSb
            idx = SbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, STATE_DIM, 3, Eigen::RowMajor>> Dr_DSb(jacobians[idx]);
                Dr_DSb.setZero();
                Dr_DSb.block<3, 3>(6, 0) = JrInvTheb;
                Dr_DSb = sqrtW*Dr_DSb;
            }
        }

        // Jacobians on PVAa states
        {
            idx = PaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, STATE_DIM, 3, Eigen::RowMajor>> Dr_DPa(jacobians[idx]);
                Dr_DPa.setZero();
                Dr_DPa.block<3, 3>(9,  0) = -Eye;
                Dr_DPa = sqrtW*Dr_DPa;
            }

            idx = VaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, STATE_DIM, 3, Eigen::RowMajor>> Dr_DVa(jacobians[idx]);
                Dr_DVa.setZero();
                Dr_DVa.block<3, 3>(9,  0) = -DtI;
                Dr_DVa.block<3, 3>(12, 0) = -Eye;
                Dr_DVa = sqrtW*Dr_DVa;
            }

            idx = AaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, STATE_DIM, 3, Eigen::RowMajor>> Dr_DAa(jacobians[idx]);
                Dr_DAa.setZero();
                Dr_DAa.block<3, 3>(9,  0) = -DtsqDiv2I;
                Dr_DAa.block<3, 3>(12, 0) = -DtI;
                Dr_DAa.block<3, 3>(15, 0) = -Eye;
                Dr_DAa = sqrtW*Dr_DAa;
            }
        }

        // Jacobians on PVAb states
        {
            idx = PbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, STATE_DIM, 3, Eigen::RowMajor>> Dr_DPb(jacobians[idx]);
                Dr_DPb.setZero();
                Dr_DPb.block<3, 3>(9, 0) = Eye;
                Dr_DPb = sqrtW*Dr_DPb;
            }

            idx = VbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, STATE_DIM, 3, Eigen::RowMajor>> Dr_DVb(jacobians[idx]);
                Dr_DVb.setZero();
                Dr_DVb.block<3, 3>(12, 0) = Eye;
                Dr_DVb = sqrtW*Dr_DVb;
            }

            idx = AbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, STATE_DIM, 3, Eigen::RowMajor>> Dr_DAb(jacobians[idx]);
                Dr_DAb.setZero();
                Dr_DAb.block<3, 3>(15, 0) = Eye;
                Dr_DAb = sqrtW*Dr_DAb;
            }
        }

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