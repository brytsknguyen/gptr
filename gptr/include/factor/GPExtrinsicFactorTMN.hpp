#pragma once

#include <ceres/ceres.h>
#include "GaussianProcess.hpp"
#include "utility.h"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

#define GPEXT_RES_DIM 18

class GPExtrinsicFactorTMN
{
typedef Eigen::Matrix<double, GPEXT_RES_DIM, 4*STATE_DIM+6> MatJ;
public:

    GPExtrinsicFactorTMN(double wR_, double wP_, GPMixerPtr gpms_, GPMixerPtr gpmf_, double ss_, double sf_)
    :   wR          (wR_             ),
        wP          (wP_             ),
        Dts         (gpms_->getDt()  ),
        Dtf         (gpmf_->getDt()  ),
        gpms        (gpms_           ),
        gpmf        (gpmf_           ),
        ss          (ss_             ),
        sf          (sf_             )
    {
        // set_num_residuals(GPEXT_RES_DIM);

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
        sqrtW = Matrix<double, GPEXT_RES_DIM, GPEXT_RES_DIM>::Identity(GPEXT_RES_DIM, GPEXT_RES_DIM);
        sqrtW.block<3, 3>(0, 0) = Vector3d(wR, wR, wR).asDiagonal();
        sqrtW.block<3, 3>(9, 9) = Vector3d(wP, wP, wP).asDiagonal();

        residual.setZero();
        jacobian.setZero();
    }

    GPExtrinsicFactorTMN(GPMixerPtr gpmx, GPMixerPtr gpms_, GPMixerPtr gpmf_, double sx_, double ss_, double sf_)
    :   Dts         (gpms_->getDt()  ),
        Dtf         (gpmf_->getDt()  ),
        gpms        (gpms_           ),
        gpmf        (gpmf_           ),
        ss          (ss_             ),
        sf          (sf_             )
    {
        // set_num_residuals(GPEXT_RES_DIM);

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
        
        // Find the square root info
        MatrixXd Cov(GPEXT_RES_DIM, GPEXT_RES_DIM); Cov.setZero();
        Cov.block<9, 9>(0, 0) += gpmx->Qga(sx_, 3);
        Cov.block<9, 9>(9, 9) += gpmx->Qnu(sx_, 3);
        sqrtW = Eigen::LLT<Matrix<double, GPEXT_RES_DIM, GPEXT_RES_DIM>>(Cov.inverse()/1e6).matrixL().transpose();
        // cout << "InvQ\n" << Cov.inverse() << endl;

        residual.setZero();
        jacobian.setZero();
    }

    GPExtrinsicFactorTMN(GPMixerPtr gpms_, GPMixerPtr gpmf_, double ss_, double sf_)
    :   Dts         (gpms_->getDt()  ),
        Dtf         (gpmf_->getDt()  ),
        gpms        (gpms_           ),
        gpmf        (gpmf_           ),
        ss          (ss_             ),
        sf          (sf_             )
    {
        // set_num_residuals(GPEXT_RES_DIM);

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
        
        MatrixXd Cov(GPEXT_RES_DIM, GPEXT_RES_DIM); Cov.setZero();
        Cov.block<9, 9>(0, 0) += gpms_->Qga(ss, 3) + gpms_->Qga(sf, 3);
        Cov.block<9, 9>(9, 9) += gpms_->Qnu(ss, 3) + gpms_->Qnu(sf, 3);
        sqrtW = Eigen::LLT<Matrix<double, GPEXT_RES_DIM, GPEXT_RES_DIM>>(Cov.inverse()/1e6).matrixL().transpose();

        residual.setZero();
        jacobian.setZero();
    }

    virtual bool Evaluate(const GPState<double> &Xsa, const GPState<double> &Xsb,
                          const GPState<double> &Xfa, const GPState<double> &Xfb,
                          const SO3d &Rsf, const Vec3 &Psf, bool computeJacobian=true)
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // // Map parameters to the control point states
        // GPState Xsa(0);   gpms->MapParamToState(parameters, RsaIdx, Xsa);
        // GPState Xsb(Dts); gpms->MapParamToState(parameters, RsbIdx, Xsb);

        // // Map parameters to the control point states
        // GPState Xfa(0);   gpmf->MapParamToState(parameters, RfaIdx, Xfa);
        // GPState Xfb(Dtf); gpmf->MapParamToState(parameters, RfbIdx, Xfb);

        // SO3d Rsf = Eigen::Map<SO3d const>(parameters[RsfIdx]);
        // Vec3 Psf = Eigen::Map<Vec3 const>(parameters[PsfIdx]);

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
        // Eigen::Map<Matrix<double, GPEXT_RES_DIM, 1>> residual(residuals);
        residual << rR, rO, rS, rP, rV, rA;
        residual = sqrtW*residual;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!computeJacobian)
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
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DRsa(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
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
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DOsa(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
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
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DSsa(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
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
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DRsb(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
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
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DOsb(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
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
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DsSb(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
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
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DRfa(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DRfa.block<3, 3>(0, 0) = DrR_DRft*DXft_DXfa[RIdx][RIdx];
                Dr_DRfa.block<3, 3>(3, 0) = DrO_DOft*DXft_DXfa[OIdx][RIdx];
                Dr_DRfa.block<3, 3>(6, 0) = DrS_DSft*DXft_DXfa[SIdx][RIdx];
                Dr_DRfa = sqrtW*Dr_DRfa;
            }

            // dr_dOfa
            idx = OfaIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DOfa(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DOfa.block<3, 3>(0, 0) = DrR_DRft*DXft_DXfa[RIdx][OIdx];
                Dr_DOfa.block<3, 3>(3, 0) = DrO_DOft*DXft_DXfa[OIdx][OIdx];
                Dr_DOfa.block<3, 3>(6, 0) = DrS_DSft*DXft_DXfa[SIdx][OIdx];
                Dr_DOfa = sqrtW*Dr_DOfa;
            }

            // dr_dSfa
            idx = SfaIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DSfa(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DSfa.block<3, 3>(0, 0) = DrR_DRft*DXft_DXfa[RIdx][SIdx];
                Dr_DSfa.block<3, 3>(3, 0) = DrO_DOft*DXft_DXfa[OIdx][SIdx];
                Dr_DSfa.block<3, 3>(6, 0) = DrS_DSft*DXft_DXfa[SIdx][SIdx];
                Dr_DSfa = sqrtW*Dr_DSfa;
            }

            // dr_dRfb
            idx = RfbIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DRfb(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DRfb.block<3, 3>(0, 0) = DrR_DRft*DXft_DXfb[RIdx][RIdx];
                Dr_DRfb.block<3, 3>(3, 0) = DrO_DOft*DXft_DXfb[OIdx][RIdx];
                Dr_DRfb.block<3, 3>(6, 0) = DrS_DSft*DXft_DXfb[SIdx][RIdx];
                Dr_DRfb = sqrtW*Dr_DRfb;
            }

            // dr_dOfb
            idx = OfbIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DOfb(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DOfb.block<3, 3>(0, 0) = DrR_DRft*DXft_DXfb[RIdx][OIdx];
                Dr_DOfb.block<3, 3>(3, 0) = DrO_DOft*DXft_DXfb[OIdx][OIdx];
                Dr_DOfb.block<3, 3>(6, 0) = DrS_DSft*DXft_DXfb[SIdx][OIdx];
                Dr_DOfb = sqrtW*Dr_DOfb;
            }

            // dr_dSfb
            idx = SfbIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DSfb(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DSfb.block<3, 3>(0, 0) = DrR_DRft*DXft_DXfb[RIdx][SIdx];
                Dr_DSfb.block<3, 3>(3, 0) = DrO_DOft*DXft_DXfb[OIdx][SIdx];
                Dr_DSfb.block<3, 3>(6, 0) = DrS_DSft*DXft_DXfb[SIdx][SIdx];
                Dr_DSfb = sqrtW*Dr_DSfb;
            }
        }

        // Jacobians on PVAs states
        {
            idx = PsaIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DPsa(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DPsa.block<3, 3>(9,  0) = DrP_DPst*DXst_DXsa[PIdx][PIdx];
                Dr_DPsa.block<3, 3>(12, 0) = DrV_DVst*DXst_DXsa[VIdx][PIdx];
                Dr_DPsa.block<3, 3>(15, 0) = DrA_DAst*DXst_DXsa[AIdx][PIdx];
                Dr_DPsa = sqrtW*Dr_DPsa;
            }

            idx = VsaIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DVsa(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DVsa.block<3, 3>(9,  0) = DrP_DPst*DXst_DXsa[PIdx][VIdx];
                Dr_DVsa.block<3, 3>(12, 0) = DrV_DVst*DXst_DXsa[VIdx][VIdx];
                Dr_DVsa.block<3, 3>(15, 0) = DrA_DAst*DXst_DXsa[AIdx][VIdx];
                Dr_DVsa = sqrtW*Dr_DVsa;
            }

            idx = AsaIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DAsa(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DAsa.block<3, 3>(9,  0) = DrP_DPst*DXst_DXsa[PIdx][AIdx];
                Dr_DAsa.block<3, 3>(12, 0) = DrV_DVst*DXst_DXsa[VIdx][AIdx];
                Dr_DAsa.block<3, 3>(15, 0) = DrA_DAst*DXst_DXsa[AIdx][AIdx];
                Dr_DAsa = sqrtW*Dr_DAsa;
            }

            idx = PsbIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DRsb(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DRsb.block<3, 3>(9,  0) = DrP_DPst*DXst_DXsb[PIdx][PIdx];
                Dr_DRsb.block<3, 3>(12, 0) = DrV_DVst*DXst_DXsb[VIdx][PIdx];
                Dr_DRsb.block<3, 3>(15, 0) = DrA_DAst*DXst_DXsb[AIdx][PIdx];
                Dr_DRsb = sqrtW*Dr_DRsb;
            }

            idx = VsbIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DVsb(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DVsb.block<3, 3>(9,  0) = DrP_DPst*DXst_DXsb[PIdx][VIdx];
                Dr_DVsb.block<3, 3>(12, 0) = DrV_DVst*DXst_DXsb[VIdx][VIdx];
                Dr_DVsb.block<3, 3>(15, 0) = DrA_DAst*DXst_DXsb[AIdx][VIdx];
                Dr_DVsb = sqrtW*Dr_DVsb;
            }

            idx = AsbIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DAsb(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DAsb.block<3, 3>(9,  0) = DrP_DPst*DXst_DXsb[PIdx][AIdx];
                Dr_DAsb.block<3, 3>(12, 0) = DrV_DVst*DXst_DXsb[VIdx][AIdx];
                Dr_DAsb.block<3, 3>(15, 0) = DrA_DAst*DXst_DXsb[AIdx][AIdx];
                Dr_DAsb = sqrtW*Dr_DAsb;
            }
        }

        // Jacobians on PVAf states
        {
            idx = PfaIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DPfa(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DPfa.block<3, 3>(9,  0) = DrP_DPft*DXft_DXfa[PIdx][PIdx];
                Dr_DPfa.block<3, 3>(12, 0) = DrV_DVft*DXft_DXfa[VIdx][PIdx];
                Dr_DPfa.block<3, 3>(15, 0) = DrA_DAft*DXft_DXfa[AIdx][PIdx];
                Dr_DPfa = sqrtW*Dr_DPfa;
            }

            idx = VfaIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DVfa(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DVfa.block<3, 3>(9,  0) = DrP_DPft*DXft_DXfa[PIdx][VIdx];
                Dr_DVfa.block<3, 3>(12, 0) = DrV_DVft*DXft_DXfa[VIdx][VIdx];
                Dr_DVfa.block<3, 3>(15, 0) = DrA_DAft*DXft_DXfa[AIdx][VIdx];
                Dr_DVfa = sqrtW*Dr_DVfa;
            }

            idx = AfaIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DAfa(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DAfa.block<3, 3>(9,  0) = DrP_DPft*DXft_DXfa[PIdx][AIdx];
                Dr_DAfa.block<3, 3>(12, 0) = DrV_DVft*DXft_DXfa[VIdx][AIdx];
                Dr_DAfa.block<3, 3>(15, 0) = DrA_DAft*DXft_DXfa[AIdx][AIdx];
                Dr_DAfa = sqrtW*Dr_DAfa;
            }

            idx = PfbIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DRfb(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DRfb.block<3, 3>(9,  0) = DrP_DPft*DXft_DXfb[PIdx][PIdx];
                Dr_DRfb.block<3, 3>(12, 0) = DrV_DVft*DXft_DXfb[VIdx][PIdx];
                Dr_DRfb.block<3, 3>(15, 0) = DrA_DAft*DXft_DXfb[AIdx][PIdx];
                Dr_DRfb = sqrtW*Dr_DRfb;
            }

            idx = VfbIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DVfb(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DVfb.block<3, 3>(9,  0) = DrP_DPft*DXft_DXfb[PIdx][VIdx];
                Dr_DVfb.block<3, 3>(12, 0) = DrV_DVft*DXft_DXfb[VIdx][VIdx];
                Dr_DVfb.block<3, 3>(15, 0) = DrA_DAft*DXft_DXfb[AIdx][VIdx];
                Dr_DVfb = sqrtW*Dr_DVfb;
            }

            idx = AfbIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DAfb(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DAfb.block<3, 3>(9,  0) = DrP_DPft*DXft_DXfb[PIdx][AIdx];
                Dr_DAfb.block<3, 3>(12, 0) = DrV_DVft*DXft_DXfb[VIdx][AIdx];
                Dr_DAfb.block<3, 3>(15, 0) = DrA_DAft*DXft_DXfb[AIdx][AIdx];
                Dr_DAfb = sqrtW*Dr_DAfb;
            }
        }

        // Jacobian of extrinsics
        {
            idx = RsfIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DRsf(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DRsf.block<3, 3>(0, 0) = DrR_DRsf;
                Dr_DRsf.block<3, 3>(3, 0) = DrO_DRsf;
                Dr_DRsf.block<3, 3>(6, 0) = DrS_DRsf;
                Dr_DRsf = sqrtW*Dr_DRsf;
            }

            idx = PsfIdx;
            {
                Eigen::Block<MatJ, GPEXT_RES_DIM, 3> Dr_DPsf(jacobian.block<GPEXT_RES_DIM, 3>(0, idx));
                Dr_DPsf.block<3, 3>(9,  0) = DrP_DPsf;
                Dr_DPsf.block<3, 3>(12, 0) = DrV_DPsf;
                Dr_DPsf.block<3, 3>(15, 0) = DrA_DPsf;
                Dr_DPsf = sqrtW*Dr_DPsf;
            }
        }

        return true;
    }

    Matrix<double, 4*STATE_DIM+6, 4*STATE_DIM+6> H() { return jacobian.transpose() * jacobian; }
    Matrix<double, 4*STATE_DIM+6, 1> b() { return -(jacobian.transpose() * residual); }

    Matrix<double, GPEXT_RES_DIM, 1> residual;
    Matrix<double, GPEXT_RES_DIM, 4*STATE_DIM+6> jacobian;

private:

    const int RIdx = 0;
    const int OIdx = 1;
    const int SIdx = 2;
    const int PIdx = 3;
    const int VIdx = 4;
    const int AIdx = 5;
    
    const int RsaIdx = 0;
    const int OsaIdx = 3;
    const int SsaIdx = 6;
    const int PsaIdx = 9;
    const int VsaIdx = 12;
    const int AsaIdx = 15;

    const int RsbIdx = 18;
    const int OsbIdx = 21;
    const int SsbIdx = 24;
    const int PsbIdx = 27;
    const int VsbIdx = 30;
    const int AsbIdx = 33;

    const int RfaIdx = 36;
    const int OfaIdx = 39;
    const int SfaIdx = 42;
    const int PfaIdx = 45;
    const int VfaIdx = 48;
    const int AfaIdx = 51;

    const int RfbIdx = 54;
    const int OfbIdx = 57;
    const int SfbIdx = 60;
    const int PfbIdx = 63;
    const int VfbIdx = 66;
    const int AfbIdx = 69;

    const int RsfIdx = 72;
    const int PsfIdx = 75;

    double wR;
    double wP;

    // Square root information
    Matrix<double, GPEXT_RES_DIM, GPEXT_RES_DIM> sqrtW;
    
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