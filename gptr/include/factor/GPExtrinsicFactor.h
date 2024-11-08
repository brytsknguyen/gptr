#pragma once

#include <ceres/ceres.h>
#include "GaussianProcess.hpp"
#include "utility.h"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

#define RES_DIM 18

class GPExtrinsicFactor : public ceres::CostFunction
{
public:

    GPExtrinsicFactor(double wR_, double wP_, GPMixerPtr gpms_, GPMixerPtr gpmf_, double ss_, double sf_)
    :   wR          (wR_             ),
        wP          (wP_             ),
        Dts         (gpms_->getDt()  ),
        Dtf         (gpmf_->getDt()  ),
        gpms        (gpms_           ),
        gpmf        (gpmf_           ),
        ss          (ss_             ),
        sf          (sf_             )
    {
        set_num_residuals(RES_DIM);

        // Add the knots
        for(int j = 0; j < 4; j++)
        {
            mutable_parameter_block_sizes()->push_back(4);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
        }

        // Add the extrinsics
        mutable_parameter_block_sizes()->push_back(4);
        mutable_parameter_block_sizes()->push_back(3);
        
        // // Find the square root info
        sqrtW = Matrix<double, RES_DIM, RES_DIM>::Identity(RES_DIM, RES_DIM);
        sqrtW.block<3, 3>(0, 0) = Vector3d(wR, wR, wR).asDiagonal();
        sqrtW.block<3, 3>(9, 9) = Vector3d(wP, wP, wP).asDiagonal();
    }

    GPExtrinsicFactor(GPMixerPtr gpmx, GPMixerPtr gpms_, GPMixerPtr gpmf_, double sx_, double ss_, double sf_)
    :   Dts         (gpms_->getDt()  ),
        Dtf         (gpmf_->getDt()  ),
        gpms        (gpms_           ),
        gpmf        (gpmf_           ),
        ss          (ss_             ),
        sf          (sf_             )
    {
        set_num_residuals(RES_DIM);

        // Add the knots
        for(int j = 0; j < 4; j++)
        {
            mutable_parameter_block_sizes()->push_back(4);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
        }

        // Add the extrinsics
        mutable_parameter_block_sizes()->push_back(4);
        mutable_parameter_block_sizes()->push_back(3);
        
        // Find the square root info
        MatrixXd Cov(RES_DIM, RES_DIM); Cov.setZero();
        Cov.block<9, 9>(0, 0) += gpmx->Qga(sx_, 3);
        Cov.block<9, 9>(9, 9) += gpmx->Qnu(sx_, 3);
        sqrtW = Eigen::LLT<Matrix<double, RES_DIM, RES_DIM>>(Cov.inverse()/1e6).matrixL().transpose();
        // cout << "InvQ\n" << Cov.inverse() << endl;
    }

    GPExtrinsicFactor(GPMixerPtr gpms_, GPMixerPtr gpmf_, double ss_, double sf_)
    :   Dts         (gpms_->getDt()  ),
        Dtf         (gpmf_->getDt()  ),
        gpms        (gpms_           ),
        gpmf        (gpmf_           ),
        ss          (ss_             ),
        sf          (sf_             )
    {
        set_num_residuals(RES_DIM);

        // Add the knots
        for(int j = 0; j < 4; j++)
        {
            mutable_parameter_block_sizes()->push_back(4);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
            mutable_parameter_block_sizes()->push_back(3);
        }

        // Add the extrinsics
        mutable_parameter_block_sizes()->push_back(4);
        mutable_parameter_block_sizes()->push_back(3);
        
        MatrixXd Cov(RES_DIM, RES_DIM); Cov.setZero();
        Cov.block<9, 9>(0, 0) += gpms_->Qga(ss, 3) + gpms_->Qga(sf, 3);
        Cov.block<9, 9>(9, 9) += gpms_->Qnu(ss, 3) + gpms_->Qnu(sf, 3);
        sqrtW = Eigen::LLT<Matrix<double, RES_DIM, RES_DIM>>(Cov.inverse()/1e6).matrixL().transpose();
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        GPState Xsa(0);   gpms->MapParamToState(parameters, RsaIdx, Xsa);
        GPState Xsb(Dts); gpms->MapParamToState(parameters, RsbIdx, Xsb);

        // Map parameters to the control point states
        GPState Xfa(0);   gpmf->MapParamToState(parameters, RfaIdx, Xfa);
        GPState Xfb(Dtf); gpmf->MapParamToState(parameters, RfbIdx, Xfb);

        SO3d Rsf = Eigen::Map<SO3d const>(parameters[RsfIdx]);
        Vec3 Psf = Eigen::Map<Vec3 const>(parameters[PsfIdx]);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Compute the interpolated states ------------------------------------------------------------------*/

        GPState Xst(ss*Dts); vector<vector<Matrix3d>> DXst_DXsa; vector<vector<Matrix3d>> DXst_DXsb;
        GPState Xft(sf*Dtf); vector<vector<Matrix3d>> DXft_DXfa; vector<vector<Matrix3d>> DXft_DXfb;

        Eigen::Matrix<double, 9, 1> gammasa, gammasb, gammast;
        Eigen::Matrix<double, 9, 1> gammafa, gammafb, gammaft;

        gpms->ComputeXtAndJacobians(Xsa, Xsb, Xst, DXst_DXsa, DXst_DXsb, gammasa, gammasb, gammast);
        gpmf->ComputeXtAndJacobians(Xfa, Xfb, Xft, DXft_DXfa, DXft_DXfb, gammafa, gammafb, gammaft);

        /* #endregion Compute the interpolated states ---------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        Mat3 Ostx = SO3d::hat(Xst.O);
        Mat3 Oftx = SO3d::hat(Xft.O);
        Mat3 Sstx = SO3d::hat(Xst.S);
        Mat3 Sftx = SO3d::hat(Xft.S);
        Vec3 OstxPsf = Ostx*Psf;
        Vec3 SstxPsf = Sstx*Psf;

        Vec3 rR = ((Xst.R*Rsf).inverse()*Xft.R).log();
        Vec3 rO = Rsf*Xft.O - Xst.O;
        Vec3 rS = Rsf*Xft.S - Xst.S;
        Vec3 rP = Xft.P - Xst.P - Xst.R*Psf;
        Vec3 rV = Xft.V - Xst.V - Xst.R*OstxPsf;
        Vec3 rA = Xft.A - Xst.A - Xst.R*SstxPsf - Xst.R*(Ostx*OstxPsf);

        // Residual
        Eigen::Map<Matrix<double, RES_DIM, 1>> residual(residuals);
        residual << rR, rO, rS, rP, rV, rA;
        residual = sqrtW*residual;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!jacobians)
            return true;

        Mat3 Eye = Mat3::Identity();

        Mat3 Rsfmat = Rsf.matrix();
        Mat3 Rstmat = Xst.R.matrix();

        Mat3 Psfskw = SO3d::hat(Psf);

        Mat3 RfInvRs = (Xft.R.inverse()*Xst.R).matrix();
        Mat3 JrInvrR =  gpms->JrInv(rR);

        Mat3 DrR_DRft =  JrInvrR;
        Mat3 DrR_DRst = -JrInvrR*RfInvRs;
        Mat3 DrR_DRsf = -JrInvrR*RfInvRs*Rsf.matrix();

        Mat3 DrO_DOft =  Rsfmat;
        Mat3 DrO_DOst = -Eye;
        Mat3 DrO_DRsf = -Rsfmat*Oftx;

        Mat3 DrS_DSft =  Rsfmat;
        Mat3 DrS_DSst = -Eye;
        Mat3 DrS_DRsf = -Rsfmat*Sftx;

        Mat3 DrP_DPft =  Eye;
        Mat3 DrP_DPst = -Eye;
        Mat3 DrP_DRst =  Rstmat*SO3d::hat(Psf);
        Mat3 DrP_DPsf = -Rstmat;

        Mat3 DrV_DVft =  Eye;
        Mat3 DrV_DVst = -Eye;
        Mat3 DrV_DRst =  Rstmat*SO3d::hat(OstxPsf);
        Mat3 DrV_DOst =  Rstmat*Psfskw;
        Mat3 DrV_DPsf = -Rstmat*Ostx;

        Mat3 DrA_DAft =  Eye;
        Mat3 DrA_DAst = -Eye;
        Mat3 DrA_DRst =  Rstmat*SO3d::hat(SstxPsf + Ostx*OstxPsf);
        Mat3 DrA_DOst = -Rstmat*gpms->Fu(Xst.O, Psf);
        Mat3 DrA_DSst =  Rstmat*SO3d::hat(Psf);
        Mat3 DrA_DPsf = -Rstmat*Sstx - Rstmat*Ostx*Ostx;

        size_t idx;

        // Jacobians on SO3s states
        {
            // dr_dRsa
            idx = RsaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 4, Eigen::RowMajor>> Dr_DRsa(jacobians[idx]);
                Dr_DRsa.setZero();
                Dr_DRsa.block<3, 3>(0,  0) = DrR_DRst*DXst_DXsa[RIdx][RIdx];
                Dr_DRsa.block<3, 3>(3,  0) = DrO_DOst*DXst_DXsa[OIdx][RIdx];
                Dr_DRsa.block<3, 3>(6,  0) = DrS_DSst*DXst_DXsa[SIdx][RIdx];
                Dr_DRsa.block<3, 3>(9,  0) = DrP_DRst*DXst_DXsa[RIdx][RIdx];
                Dr_DRsa.block<3, 3>(12, 0) = DrV_DRst*DXst_DXsa[RIdx][RIdx] + DrV_DOst*DXst_DXsa[OIdx][RIdx];
                Dr_DRsa.block<3, 3>(15, 0) = DrA_DRst*DXst_DXsa[RIdx][RIdx] + DrA_DOst*DXst_DXsa[OIdx][RIdx] + DrA_DSst*DXst_DXsa[SIdx][RIdx];
                Dr_DRsa = sqrtW*Dr_DRsa;
            }

            // dr_dOsa
            idx = OsaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DOsa(jacobians[idx]);
                Dr_DOsa.setZero();
                Dr_DOsa.block<3, 3>(0,  0) = DrR_DRst*DXst_DXsa[RIdx][OIdx];
                Dr_DOsa.block<3, 3>(3,  0) = DrO_DOst*DXst_DXsa[OIdx][OIdx];
                Dr_DOsa.block<3, 3>(6,  0) = DrS_DSst*DXst_DXsa[SIdx][OIdx];
                Dr_DOsa.block<3, 3>(9,  0) = DrP_DRst*DXst_DXsa[RIdx][OIdx];
                Dr_DOsa.block<3, 3>(12, 0) = DrV_DRst*DXst_DXsa[RIdx][OIdx] + DrV_DOst*DXst_DXsa[OIdx][OIdx];
                Dr_DOsa.block<3, 3>(15, 0) = DrA_DRst*DXst_DXsa[RIdx][OIdx] + DrA_DOst*DXst_DXsa[OIdx][OIdx] + DrA_DSst*DXst_DXsa[SIdx][OIdx];
                Dr_DOsa = sqrtW*Dr_DOsa;
            }

            // dr_dSsa
            idx = SsaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DSsa(jacobians[idx]);
                Dr_DSsa.setZero();
                Dr_DSsa.block<3, 3>(0,  0) = DrR_DRst*DXst_DXsa[RIdx][SIdx];
                Dr_DSsa.block<3, 3>(3,  0) = DrO_DOst*DXst_DXsa[OIdx][SIdx];
                Dr_DSsa.block<3, 3>(6,  0) = DrS_DSst*DXst_DXsa[SIdx][SIdx];
                Dr_DSsa.block<3, 3>(9,  0) = DrP_DRst*DXst_DXsa[RIdx][SIdx];
                Dr_DSsa.block<3, 3>(12, 0) = DrV_DRst*DXst_DXsa[RIdx][SIdx] + DrV_DOst*DXst_DXsa[OIdx][SIdx];
                Dr_DSsa.block<3, 3>(15, 0) = DrA_DRst*DXst_DXsa[RIdx][SIdx] + DrA_DOst*DXst_DXsa[OIdx][SIdx] + DrA_DSst*DXst_DXsa[SIdx][SIdx];
                Dr_DSsa = sqrtW*Dr_DSsa;
            }

            // dr_dRsb
            idx = RsbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 4, Eigen::RowMajor>> Dr_DRsb(jacobians[idx]);
                Dr_DRsb.setZero();
                Dr_DRsb.block<3, 3>(0,  0) = DrR_DRst*DXst_DXsb[RIdx][RIdx];
                Dr_DRsb.block<3, 3>(3,  0) = DrO_DOst*DXst_DXsb[OIdx][RIdx];
                Dr_DRsb.block<3, 3>(6,  0) = DrS_DSst*DXst_DXsb[SIdx][RIdx];
                Dr_DRsb.block<3, 3>(9,  0) = DrP_DRst*DXst_DXsb[RIdx][RIdx];
                Dr_DRsb.block<3, 3>(12, 0) = DrV_DRst*DXst_DXsb[RIdx][RIdx] + DrV_DOst*DXst_DXsb[OIdx][RIdx];
                Dr_DRsb.block<3, 3>(15, 0) = DrA_DRst*DXst_DXsb[RIdx][RIdx] + DrA_DOst*DXst_DXsb[OIdx][RIdx] + DrA_DSst*DXst_DXsb[SIdx][RIdx];
                Dr_DRsb = sqrtW*Dr_DRsb;
            }

            // dr_dOsb
            idx = OsbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DOsb(jacobians[idx]);
                Dr_DOsb.setZero();
                Dr_DOsb.block<3, 3>(0,  0) = DrR_DRst*DXst_DXsb[RIdx][OIdx];
                Dr_DOsb.block<3, 3>(3,  0) = DrO_DOst*DXst_DXsb[OIdx][OIdx];
                Dr_DOsb.block<3, 3>(6,  0) = DrS_DSst*DXst_DXsb[SIdx][OIdx];
                Dr_DOsb.block<3, 3>(9,  0) = DrP_DRst*DXst_DXsb[RIdx][OIdx];
                Dr_DOsb.block<3, 3>(12, 0) = DrV_DRst*DXst_DXsb[RIdx][OIdx] + DrV_DOst*DXst_DXsb[OIdx][OIdx];
                Dr_DOsb.block<3, 3>(15, 0) = DrA_DRst*DXst_DXsb[RIdx][OIdx] + DrA_DOst*DXst_DXsb[OIdx][OIdx] + DrA_DSst*DXst_DXsb[SIdx][OIdx];
                Dr_DOsb = sqrtW*Dr_DOsb;
            }

            // dr_dSsb
            idx = SsbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DsSb(jacobians[idx]);
                Dr_DsSb.setZero();
                Dr_DsSb.block<3, 3>(0,  0) = DrR_DRst*DXst_DXsb[RIdx][SIdx];
                Dr_DsSb.block<3, 3>(3,  0) = DrO_DOst*DXst_DXsb[OIdx][SIdx];
                Dr_DsSb.block<3, 3>(6,  0) = DrS_DSst*DXst_DXsb[SIdx][SIdx];
                Dr_DsSb.block<3, 3>(9,  0) = DrP_DRst*DXst_DXsb[RIdx][SIdx];
                Dr_DsSb.block<3, 3>(12, 0) = DrV_DRst*DXst_DXsb[RIdx][SIdx] + DrV_DOst*DXst_DXsb[OIdx][SIdx];
                Dr_DsSb.block<3, 3>(15, 0) = DrA_DRst*DXst_DXsb[RIdx][SIdx] + DrA_DOst*DXst_DXsb[OIdx][SIdx] + DrA_DSst*DXst_DXsb[SIdx][SIdx];
                Dr_DsSb = sqrtW*Dr_DsSb;
            }
        }

        // Jacobians on SO3f states
        {
            // dr_dRfa
            idx = RfaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 4, Eigen::RowMajor>> Dr_DRfa(jacobians[idx]);
                Dr_DRfa.setZero();
                Dr_DRfa.block<3, 3>(0, 0) = DrR_DRft*DXft_DXfa[RIdx][RIdx];
                Dr_DRfa.block<3, 3>(3, 0) = DrO_DOft*DXft_DXfa[OIdx][RIdx];
                Dr_DRfa.block<3, 3>(6, 0) = DrS_DSft*DXft_DXfa[SIdx][RIdx];
                Dr_DRfa = sqrtW*Dr_DRfa;
            }

            // dr_dOfa
            idx = OfaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DOfa(jacobians[idx]);
                Dr_DOfa.setZero();
                Dr_DOfa.block<3, 3>(0, 0) = DrR_DRft*DXft_DXfa[RIdx][OIdx];
                Dr_DOfa.block<3, 3>(3, 0) = DrO_DOft*DXft_DXfa[OIdx][OIdx];
                Dr_DOfa.block<3, 3>(6, 0) = DrS_DSft*DXft_DXfa[SIdx][OIdx];
                Dr_DOfa = sqrtW*Dr_DOfa;
            }

            // dr_dSfa
            idx = SfaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DSfa(jacobians[idx]);
                Dr_DSfa.setZero();
                Dr_DSfa.block<3, 3>(0, 0) = DrR_DRft*DXft_DXfa[RIdx][SIdx];
                Dr_DSfa.block<3, 3>(3, 0) = DrO_DOft*DXft_DXfa[OIdx][SIdx];
                Dr_DSfa.block<3, 3>(6, 0) = DrS_DSft*DXft_DXfa[SIdx][SIdx];
                Dr_DSfa = sqrtW*Dr_DSfa;
            }

            // dr_dRfb
            idx = RfbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 4, Eigen::RowMajor>> Dr_DRfb(jacobians[idx]);
                Dr_DRfb.setZero();
                Dr_DRfb.block<3, 3>(0, 0) = DrR_DRft*DXft_DXfb[RIdx][RIdx];
                Dr_DRfb.block<3, 3>(3, 0) = DrO_DOft*DXft_DXfb[OIdx][RIdx];
                Dr_DRfb.block<3, 3>(6, 0) = DrS_DSft*DXft_DXfb[SIdx][RIdx];
                Dr_DRfb = sqrtW*Dr_DRfb;
            }

            // dr_dOfb
            idx = OfbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DOfb(jacobians[idx]);
                Dr_DOfb.setZero();
                Dr_DOfb.block<3, 3>(0, 0) = DrR_DRft*DXft_DXfb[RIdx][OIdx];
                Dr_DOfb.block<3, 3>(3, 0) = DrO_DOft*DXft_DXfb[OIdx][OIdx];
                Dr_DOfb.block<3, 3>(6, 0) = DrS_DSft*DXft_DXfb[SIdx][OIdx];
                Dr_DOfb = sqrtW*Dr_DOfb;
            }

            // dr_dSfb
            idx = SfbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DSfb(jacobians[idx]);
                Dr_DSfb.setZero();
                Dr_DSfb.block<3, 3>(0, 0) = DrR_DRft*DXft_DXfb[RIdx][SIdx];
                Dr_DSfb.block<3, 3>(3, 0) = DrO_DOft*DXft_DXfb[OIdx][SIdx];
                Dr_DSfb.block<3, 3>(6, 0) = DrS_DSft*DXft_DXfb[SIdx][SIdx];
                Dr_DSfb = sqrtW*Dr_DSfb;
            }
        }

        // Jacobians on PVAs states
        {
            idx = PsaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DPsa(jacobians[idx]);
                Dr_DPsa.setZero();
                Dr_DPsa.block<3, 3>(9,  0) = DrP_DPst*DXst_DXsa[PIdx][PIdx];
                Dr_DPsa.block<3, 3>(12, 0) = DrV_DVst*DXst_DXsa[VIdx][PIdx];
                Dr_DPsa.block<3, 3>(15, 0) = DrA_DAst*DXst_DXsa[AIdx][PIdx];
                Dr_DPsa = sqrtW*Dr_DPsa;
            }

            idx = VsaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DVsa(jacobians[idx]);
                Dr_DVsa.setZero();
                Dr_DVsa.block<3, 3>(9,  0) = DrP_DPst*DXst_DXsa[PIdx][VIdx];
                Dr_DVsa.block<3, 3>(12, 0) = DrV_DVst*DXst_DXsa[VIdx][VIdx];
                Dr_DVsa.block<3, 3>(15, 0) = DrA_DAst*DXst_DXsa[AIdx][VIdx];
                Dr_DVsa = sqrtW*Dr_DVsa;
            }

            idx = AsaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DAsa(jacobians[idx]);
                Dr_DAsa.setZero();
                Dr_DAsa.block<3, 3>(9,  0) = DrP_DPst*DXst_DXsa[PIdx][AIdx];
                Dr_DAsa.block<3, 3>(12, 0) = DrV_DVst*DXst_DXsa[VIdx][AIdx];
                Dr_DAsa.block<3, 3>(15, 0) = DrA_DAst*DXst_DXsa[AIdx][AIdx];
                Dr_DAsa = sqrtW*Dr_DAsa;
            }

            idx = PsbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DRsb(jacobians[idx]);
                Dr_DRsb.setZero();
                Dr_DRsb.block<3, 3>(9,  0) = DrP_DPst*DXst_DXsb[PIdx][PIdx];
                Dr_DRsb.block<3, 3>(12, 0) = DrV_DVst*DXst_DXsb[VIdx][PIdx];
                Dr_DRsb.block<3, 3>(15, 0) = DrA_DAst*DXst_DXsb[AIdx][PIdx];
                Dr_DRsb = sqrtW*Dr_DRsb;
            }

            idx = VsbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DVsb(jacobians[idx]);
                Dr_DVsb.setZero();
                Dr_DVsb.block<3, 3>(9,  0) = DrP_DPst*DXst_DXsb[PIdx][VIdx];
                Dr_DVsb.block<3, 3>(12, 0) = DrV_DVst*DXst_DXsb[VIdx][VIdx];
                Dr_DVsb.block<3, 3>(15, 0) = DrA_DAst*DXst_DXsb[AIdx][VIdx];
                Dr_DVsb = sqrtW*Dr_DVsb;
            }

            idx = AsbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DAsb(jacobians[idx]);
                Dr_DAsb.setZero();
                Dr_DAsb.block<3, 3>(9,  0) = DrP_DPst*DXst_DXsb[PIdx][AIdx];
                Dr_DAsb.block<3, 3>(12, 0) = DrV_DVst*DXst_DXsb[VIdx][AIdx];
                Dr_DAsb.block<3, 3>(15, 0) = DrA_DAst*DXst_DXsb[AIdx][AIdx];
                Dr_DAsb = sqrtW*Dr_DAsb;
            }
        }

        // Jacobians on PVAf states
        {
            idx = PfaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DPfa(jacobians[idx]);
                Dr_DPfa.setZero();
                Dr_DPfa.block<3, 3>(9,  0) = DrP_DPft*DXft_DXfa[PIdx][PIdx];
                Dr_DPfa.block<3, 3>(12, 0) = DrV_DVft*DXft_DXfa[VIdx][PIdx];
                Dr_DPfa.block<3, 3>(15, 0) = DrA_DAft*DXft_DXfa[AIdx][PIdx];
                Dr_DPfa = sqrtW*Dr_DPfa;
            }

            idx = VfaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DVfa(jacobians[idx]);
                Dr_DVfa.setZero();
                Dr_DVfa.block<3, 3>(9,  0) = DrP_DPft*DXft_DXfa[PIdx][VIdx];
                Dr_DVfa.block<3, 3>(12, 0) = DrV_DVft*DXft_DXfa[VIdx][VIdx];
                Dr_DVfa.block<3, 3>(15, 0) = DrA_DAft*DXft_DXfa[AIdx][VIdx];
                Dr_DVfa = sqrtW*Dr_DVfa;
            }

            idx = AfaIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DAfa(jacobians[idx]);
                Dr_DAfa.setZero();
                Dr_DAfa.block<3, 3>(9,  0) = DrP_DPft*DXft_DXfa[PIdx][AIdx];
                Dr_DAfa.block<3, 3>(12, 0) = DrV_DVft*DXft_DXfa[VIdx][AIdx];
                Dr_DAfa.block<3, 3>(15, 0) = DrA_DAft*DXft_DXfa[AIdx][AIdx];
                Dr_DAfa = sqrtW*Dr_DAfa;
            }

            idx = PfbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DRfb(jacobians[idx]);
                Dr_DRfb.setZero();
                Dr_DRfb.block<3, 3>(9,  0) = DrP_DPft*DXft_DXfb[PIdx][PIdx];
                Dr_DRfb.block<3, 3>(12, 0) = DrV_DVft*DXft_DXfb[VIdx][PIdx];
                Dr_DRfb.block<3, 3>(15, 0) = DrA_DAft*DXft_DXfb[AIdx][PIdx];
                Dr_DRfb = sqrtW*Dr_DRfb;
            }

            idx = VfbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DVfb(jacobians[idx]);
                Dr_DVfb.setZero();
                Dr_DVfb.block<3, 3>(9,  0) = DrP_DPft*DXft_DXfb[PIdx][VIdx];
                Dr_DVfb.block<3, 3>(12, 0) = DrV_DVft*DXft_DXfb[VIdx][VIdx];
                Dr_DVfb.block<3, 3>(15, 0) = DrA_DAft*DXft_DXfb[AIdx][VIdx];
                Dr_DVfb = sqrtW*Dr_DVfb;
            }

            idx = AfbIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DAfb(jacobians[idx]);
                Dr_DAfb.setZero();
                Dr_DAfb.block<3, 3>(9,  0) = DrP_DPft*DXft_DXfb[PIdx][AIdx];
                Dr_DAfb.block<3, 3>(12, 0) = DrV_DVft*DXft_DXfb[VIdx][AIdx];
                Dr_DAfb.block<3, 3>(15, 0) = DrA_DAft*DXft_DXfb[AIdx][AIdx];
                Dr_DAfb = sqrtW*Dr_DAfb;
            }
        }

        // Jacobian of extrinsics
        {
            idx = RsfIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 4, Eigen::RowMajor>> Dr_DRsf(jacobians[idx]);
                Dr_DRsf.setZero();
                Dr_DRsf.block<3, 3>(0, 0) = DrR_DRsf;
                Dr_DRsf.block<3, 3>(3, 0) = DrO_DRsf;
                Dr_DRsf.block<3, 3>(6, 0) = DrS_DRsf;
                Dr_DRsf = sqrtW*Dr_DRsf;
            }

            idx = PsfIdx;
            if(jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, RES_DIM, 3, Eigen::RowMajor>> Dr_DPsf(jacobians[idx]);
                Dr_DPsf.setZero();
                Dr_DPsf.block<3, 3>(9,  0) = DrP_DPsf;
                Dr_DPsf.block<3, 3>(12, 0) = DrV_DPsf;
                Dr_DPsf.block<3, 3>(15, 0) = DrA_DPsf;
                Dr_DPsf = sqrtW*Dr_DPsf;
            }
        }

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
    Matrix<double, RES_DIM, RES_DIM> sqrtW;
    
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