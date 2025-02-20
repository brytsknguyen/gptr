#include "utility.h"
#include "GaussianProcess.hpp"

#include "factor/GPPointToPlaneFactor.h"
#include "factor/GPPointToPlaneFactorAutodiff.h"

#include "factor/GPMotionPriorTwoKnotsFactor.h"
#include "factor/GPMotionPriorTwoKnotsFactorAutodiff.h"

#include "factor/GPExtrinsicFactor.h"
#include "factor/GPExtrinsicFactorAutodiff.h"

#include "AutoDiffSO3Parameterization.hpp"

double mpCovROSJerk = 1.0;
double mpCovPVAJerk = 1.0;
int lidar_ds_rate = 1;

MatrixXd thresholdMatrix(const MatrixXd &matrix, double threshold)
{
    return (matrix.array().abs() >= threshold).cast<double>();
}


struct FactorMeta
{
    vector<double *> so3_parameter_blocks;
    vector<double *> rv3_parameter_blocks;
    vector<ceres::ResidualBlockId> residual_blocks;

    int parameter_blocks()
    {
        return (so3_parameter_blocks.size() + rv3_parameter_blocks.size());
    }
};

Eigen::MatrixXd GetJacobian(ceres::CRSMatrix &J)
{
    Eigen::MatrixXd dense_jacobian(J.num_rows, J.num_cols);
    dense_jacobian.setZero();
    for (int r = 0; r < J.num_rows; ++r)
    {
        for (int idx = J.rows[r]; idx < J.rows[r + 1]; ++idx)
        {
            const int c = J.cols[idx];
            dense_jacobian(r, c) = J.values[idx];
        }
    }

    return dense_jacobian;
}

void EvaluateFactorRJ(ceres::Problem &problem, FactorMeta &factorMeta,
                      int local_pamaterization_type,
                      double &cost, vector<double> &residual,
                      MatrixXd &Jacobian, double &eval_time)
{
    ceres::LocalParameterization *localparameterization;
    for(auto parameter : factorMeta.so3_parameter_blocks)
    {
        if (local_pamaterization_type == 0)
        {
            localparameterization = new AutoDiffSO3Parameterization<SO3d>();
            problem.SetParameterization(parameter, localparameterization);
        }
        else
        {   
            localparameterization = new GPSO3dLocalParameterization();
            problem.SetParameterization(parameter, localparameterization);
        }
    }

    ceres::Problem::EvaluateOptions e_option;
    ceres::CRSMatrix Jacobian_;
    e_option.residual_blocks = factorMeta.residual_blocks;

    TicToc tt_eval;
    problem.Evaluate(e_option, &cost, &residual, NULL, &Jacobian_);
    tt_eval.Toc();
    eval_time = tt_eval.GetLastStop();

    Jacobian = GetJacobian(Jacobian_);
}

void RemoveResidualBlock(ceres::Problem &problem, FactorMeta &factorMeta)
{
    for(auto res_block : factorMeta.residual_blocks)
        problem.RemoveResidualBlock(res_block);
}


void CreateCeresProblem(ceres::Problem &problem, ceres::Solver::Options &options, ceres::Solver::Summary &summary,
                        GaussianProcessPtr &swTraj, double fixed_start, double fixed_end)
{
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = MAX_THREADS;
    options.max_num_iterations = 50;
    int KNOTS = swTraj->getNumKnots();
    // Add the parameter blocks for rotation
    for (int kidx = 0; kidx < KNOTS; kidx++)
    {
        problem.AddParameterBlock(swTraj->getKnotSO3(kidx).data(), 4, new GPSO3dLocalParameterization());
        problem.AddParameterBlock(swTraj->getKnotOmg(kidx).data(), 3);
        problem.AddParameterBlock(swTraj->getKnotAlp(kidx).data(), 3);
        problem.AddParameterBlock(swTraj->getKnotPos(kidx).data(), 3);
        problem.AddParameterBlock(swTraj->getKnotVel(kidx).data(), 3);
        problem.AddParameterBlock(swTraj->getKnotAcc(kidx).data(), 3);
    }
    // Fix the knots
    if (fixed_start >= 0)
        for (int kidx = 0; kidx < KNOTS; kidx++)
        {
            if (swTraj->getKnotTime(kidx) <= swTraj->getMinTime() + fixed_start)
            {
                problem.SetParameterBlockConstant(swTraj->getKnotSO3(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotOmg(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotAlp(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotPos(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotVel(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotAcc(kidx).data());
                // printf("Fixed knot %d\n", kidx);
            }
        }
    if (fixed_end >= 0)
        for (int kidx = 0; kidx < KNOTS; kidx++)
        {
            if (swTraj->getKnotTime(kidx) >= swTraj->getMaxTime() - fixed_end)
            {
                problem.SetParameterBlockConstant(swTraj->getKnotSO3(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotOmg(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotAlp(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotPos(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotVel(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotAcc(kidx).data());
                // printf("Fixed knot %d\n", kidx);
            }
        }
}

double maxDiff(const MatrixXd &A, const MatrixXd &B)
{
    return (A - B).cwiseAbs().maxCoeff();
}

void AddAutodiffGPMP2KFactor(GaussianProcessPtr &traj, ceres::Problem &problem, FactorMeta &gpmpFactorMeta)
{
    vector<double *> so3_param;
    vector<double *> rv3_param;
    vector<ceres::ResidualBlockId> res_ids_gp;
    // Add the GP factors based on knot difference
    for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
    {
        // Create the factor
        double gp_loss_thres = -1;
        ceres::LossFunction *gp_loss_func = gp_loss_thres == -1 ? NULL : new ceres::HuberLoss(gp_loss_thres);
        GPMotionPriorTwoKnotsFactorAutodiff *GPMPFactor = new GPMotionPriorTwoKnotsFactorAutodiff(traj->getGPMixerPtr());
        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<GPMotionPriorTwoKnotsFactorAutodiff>(GPMPFactor);
        cost_function->SetNumResiduals(18);
        vector<double *> factor_param_blocks;
        // Add the parameter blocks
        for (int knot_idx = kidx; knot_idx < kidx + 2; knot_idx++)
        {
            so3_param.push_back(traj->getKnotSO3(knot_idx).data());
            rv3_param.push_back(traj->getKnotOmg(knot_idx).data());
            rv3_param.push_back(traj->getKnotAlp(knot_idx).data());
            rv3_param.push_back(traj->getKnotPos(knot_idx).data());
            rv3_param.push_back(traj->getKnotVel(knot_idx).data());
            rv3_param.push_back(traj->getKnotAcc(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAlp(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            cost_function->AddParameterBlock(4);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
        }
        auto res_block = problem.AddResidualBlock(cost_function, gp_loss_func, factor_param_blocks);
        res_ids_gp.push_back(res_block);
    }
    gpmpFactorMeta.so3_parameter_blocks = so3_param;
    gpmpFactorMeta.rv3_parameter_blocks = rv3_param;
    gpmpFactorMeta.residual_blocks = res_ids_gp;
    // printf("Autodiff params: %d, %d, %d, %d, %d, %d\n",
    //         so3_param.size(), gpmpFactorMeta.so3_parameter_blocks.size(),
    //         rv3_param.size(), gpmpFactorMeta.rv3_parameter_blocks.size(),
    //         res_ids_gp.size(), gpmpFactorMeta.residual_blocks.size());
}

void AddAnalyticGPMP2KFactor(GaussianProcessPtr &traj, ceres::Problem &problem, FactorMeta &gpmpFactorMeta)
{
    vector<double *> so3_param;
    vector<double *> rv3_param;
    vector<ceres::ResidualBlockId> res_ids_gp;
    // Add GP factors between consecutive knots
    for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
    {
        vector<double *> factor_param_blocks;
        // Add the parameter blocks
        for (int knot_idx = kidx; knot_idx < kidx + 2; knot_idx++)
        {
            so3_param.push_back(traj->getKnotSO3(knot_idx).data());
            rv3_param.push_back(traj->getKnotOmg(knot_idx).data());
            rv3_param.push_back(traj->getKnotAlp(knot_idx).data());
            rv3_param.push_back(traj->getKnotPos(knot_idx).data());
            rv3_param.push_back(traj->getKnotVel(knot_idx).data());
            rv3_param.push_back(traj->getKnotAcc(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAlp(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
        }
        // Create the factors
        double mp_loss_thres = -1;
        // nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
        ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
        ceres::CostFunction *cost_function = new GPMotionPriorTwoKnotsFactor(traj->getGPMixerPtr());
        auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
        res_ids_gp.push_back(res_block);
    }
    
    gpmpFactorMeta.so3_parameter_blocks = so3_param;
    gpmpFactorMeta.rv3_parameter_blocks = rv3_param;
    gpmpFactorMeta.residual_blocks = res_ids_gp;
    // printf("Analytic params: %d, %d, %d, %d, %d, %d\n",
    //         so3_param.size(), gpmpFactorMeta.so3_parameter_blocks.size(),
    //         rv3_param.size(), gpmpFactorMeta.rv3_parameter_blocks.size(),
    //         res_ids_gp.size(), gpmpFactorMeta.residual_blocks.size());
}

void AddAutodiffGPLidarFactor(GaussianProcessPtr &traj, ceres::Problem &problem,
                              FactorMeta &gplidarFactorMeta,
                              vector<LidarCoef> &Coef)
{
    vector<double *> so3_param;
    vector<double *> rv3_param;
    vector<ceres::ResidualBlockId> res_ids_lidar;
    int skip = -1;
    for (auto &coef : Coef)
    {
        // Skip if lidar coef is not assigned
        if (coef.t < 0)
            continue;
        if (!traj->TimeInInterval(coef.t, 1e-6))
            continue;
        skip++;
        if (skip % lidar_ds_rate != 0)
            continue;
        
        auto   us = traj->computeTimeIndex(coef.t);
        int    u  = us.first;
        double s  = us.second;
        double lidar_loss_thres = -1.0;
        double gp_loss_thres = -1;
        ceres::LossFunction *gp_loss_func = gp_loss_thres == -1 ? NULL : new ceres::HuberLoss(gp_loss_thres);
        GPPointToPlaneFactorAutodiff *GPLidarFactor = new GPPointToPlaneFactorAutodiff(coef.f, coef.n, coef.plnrty, traj->getGPMixerPtr(), s);
        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<GPPointToPlaneFactorAutodiff>(GPLidarFactor);
        cost_function->SetNumResiduals(1);
        vector<double *> factor_param_blocks;
        // Add the parameter blocks for rotation
        for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
        {
            so3_param.push_back(traj->getKnotSO3(knot_idx).data());
            rv3_param.push_back(traj->getKnotOmg(knot_idx).data());
            rv3_param.push_back(traj->getKnotAlp(knot_idx).data());
            rv3_param.push_back(traj->getKnotPos(knot_idx).data());
            rv3_param.push_back(traj->getKnotVel(knot_idx).data());
            rv3_param.push_back(traj->getKnotAcc(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAlp(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            cost_function->AddParameterBlock(4);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
        }
        
        auto res = problem.AddResidualBlock(cost_function, gp_loss_func, factor_param_blocks);
        res_ids_lidar.push_back(res);
    }
    gplidarFactorMeta.so3_parameter_blocks = so3_param;
    gplidarFactorMeta.rv3_parameter_blocks = rv3_param;
    gplidarFactorMeta.residual_blocks = res_ids_lidar;
}

void AddAnalyticGPLidarFactor(GaussianProcessPtr &traj, ceres::Problem &problem,
                              FactorMeta &gplidarFactorMeta,
                              vector<LidarCoef> &Coef)
{
    vector<double *> so3_param;
    vector<double *> rv3_param;
    vector<ceres::ResidualBlockId> res_ids_lidar;
    int skip = -1;
    for (auto &coef : Coef)
    {
        // Skip if lidar coef is not assigned
        if (coef.t < 0)
            continue;
        if (!traj->TimeInInterval(coef.t, 1e-6))
            continue;
        skip++;
        if (skip % lidar_ds_rate != 0)
            continue;
        
        auto   us = traj->computeTimeIndex(coef.t);
        int    u  = us.first;
        double s  = us.second;
        vector<double *> factor_param_blocks;
        // Add the parameter blocks for rotation
        for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
        {
            so3_param.push_back(traj->getKnotSO3(knot_idx).data());
            rv3_param.push_back(traj->getKnotOmg(knot_idx).data());
            rv3_param.push_back(traj->getKnotAlp(knot_idx).data());
            rv3_param.push_back(traj->getKnotPos(knot_idx).data());
            rv3_param.push_back(traj->getKnotVel(knot_idx).data());
            rv3_param.push_back(traj->getKnotAcc(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAlp(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
        }
        
        double lidar_loss_thres = -1.0;
        ceres::LossFunction *lidar_loss_function = lidar_loss_thres == -1 ? NULL : new ceres::HuberLoss(lidar_loss_thres);
        ceres::CostFunction *cost_function = new GPPointToPlaneFactor(coef.f, coef.n, coef.plnrty, traj->getGPMixerPtr(), s);
        auto res = problem.AddResidualBlock(cost_function, lidar_loss_function, factor_param_blocks);
        res_ids_lidar.push_back(res);
    }
    gplidarFactorMeta.so3_parameter_blocks = so3_param;
    gplidarFactorMeta.rv3_parameter_blocks = rv3_param;
    gplidarFactorMeta.residual_blocks = res_ids_lidar;
}

void AddAutodiffGPXtsFactor(GaussianProcessPtr &traj, ceres::Problem &problem,
                            FactorMeta &factorMeta, SO3d &R_Lx_Ly, Vector3d &P_Lx_Ly)
{
    vector<double *> so3_param;
    vector<double *> rv3_param;
    vector<ceres::ResidualBlockId> res_ids;

    // SO3d R_Lx_Ly(Quaternd(1, 0, 0, 0));
    // problem.AddParameterBlock(R_Lx_Ly.data(), 4, new GPSO3dLocalParameterization());

    // Vector3d P_Lx_Ly(0, 0, 0);
    // problem.AddParameterBlock(P_Lx_Ly.data(), 3);

    // Get the knot time
    double ts = traj->getKnotTime(0) + traj->getDt()/2.0;
    double tf = traj->getKnotTime(2) + traj->getDt()/2.0;
    
    // Skip if time is outside the range of the other trajectory
    pair<int, double> uss, usf;
    uss = traj->computeTimeIndex(ts);
    usf = traj->computeTimeIndex(tf);

    int umins = uss.first;
    int uminf = usf.first;
    double ss = uss.second;
    double sf = usf.second;

    // Create the factor
    double xtsCovROSJerk = 5.7;
    double xtsCovPVAJerk = 5.7;
    double mp_loss_thres = -1;
    // nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
    ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
    GPExtrinsicFactorAutodiff *factor = new GPExtrinsicFactorAutodiff(xtsCovROSJerk, xtsCovPVAJerk, traj->getGPMixerPtr(), traj->getGPMixerPtr(), ss, sf);
    auto *cost_function = new ceres::DynamicAutoDiffCostFunction<GPExtrinsicFactorAutodiff>(factor);
    cost_function->SetNumResiduals(18);

    // Add the parameter blocks
    vector<double *> factor_param_blocks;
    for (int kidx = umins; kidx < umins + 2; kidx++)
    {
        so3_param.push_back(traj->getKnotSO3(kidx).data());
        rv3_param.push_back(traj->getKnotOmg(kidx).data());
        rv3_param.push_back(traj->getKnotAlp(kidx).data());
        rv3_param.push_back(traj->getKnotPos(kidx).data());
        rv3_param.push_back(traj->getKnotVel(kidx).data());
        rv3_param.push_back(traj->getKnotAcc(kidx).data());

        factor_param_blocks.push_back(traj->getKnotSO3(kidx).data());
        factor_param_blocks.push_back(traj->getKnotOmg(kidx).data());
        factor_param_blocks.push_back(traj->getKnotAlp(kidx).data());
        factor_param_blocks.push_back(traj->getKnotPos(kidx).data());
        factor_param_blocks.push_back(traj->getKnotVel(kidx).data());
        factor_param_blocks.push_back(traj->getKnotAcc(kidx).data());

        cost_function->AddParameterBlock(4);
        cost_function->AddParameterBlock(3);
        cost_function->AddParameterBlock(3);
        cost_function->AddParameterBlock(3);
        cost_function->AddParameterBlock(3);
        cost_function->AddParameterBlock(3);
    }

    for (int kidx = uminf; kidx < uminf + 2; kidx++)
    {
        so3_param.push_back(traj->getKnotSO3(kidx).data());
        rv3_param.push_back(traj->getKnotOmg(kidx).data());
        rv3_param.push_back(traj->getKnotAlp(kidx).data());
        rv3_param.push_back(traj->getKnotPos(kidx).data());
        rv3_param.push_back(traj->getKnotVel(kidx).data());
        rv3_param.push_back(traj->getKnotAcc(kidx).data());

        factor_param_blocks.push_back(traj->getKnotSO3(kidx).data());
        factor_param_blocks.push_back(traj->getKnotOmg(kidx).data());
        factor_param_blocks.push_back(traj->getKnotAlp(kidx).data());
        factor_param_blocks.push_back(traj->getKnotPos(kidx).data());
        factor_param_blocks.push_back(traj->getKnotVel(kidx).data());
        factor_param_blocks.push_back(traj->getKnotAcc(kidx).data());

        cost_function->AddParameterBlock(4);
        cost_function->AddParameterBlock(3);
        cost_function->AddParameterBlock(3);
        cost_function->AddParameterBlock(3);
        cost_function->AddParameterBlock(3);
        cost_function->AddParameterBlock(3);
    }

    // Add the extrinsics
    {
        so3_param.push_back(R_Lx_Ly.data());
        rv3_param.push_back(P_Lx_Ly.data());

        factor_param_blocks.push_back(R_Lx_Ly.data());
        factor_param_blocks.push_back(P_Lx_Ly.data());

        cost_function->AddParameterBlock(4);
        cost_function->AddParameterBlock(3);
    }

    // Add res block
    auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
    res_ids.push_back(res_block);

    // Copy the factor meta
    factorMeta.so3_parameter_blocks = so3_param;
    factorMeta.rv3_parameter_blocks = rv3_param;
    factorMeta.residual_blocks = res_ids;
}

void AddAnalyticGPXtsFactor(GaussianProcessPtr &traj, ceres::Problem &problem,
                            FactorMeta &factorMeta, SO3d &R_Lx_Ly, Vector3d &P_Lx_Ly)
{
    vector<double *> so3_param;
    vector<double *> rv3_param;
    vector<ceres::ResidualBlockId> res_ids;

    // SO3d R_Lx_Ly(Quaternd(1, 0, 0, 0));
    // problem.AddParameterBlock(R_Lx_Ly.data(), 4, new GPSO3dLocalParameterization());

    // Vector3d P_Lx_Ly(0, 0, 0);
    // problem.AddParameterBlock(P_Lx_Ly.data(), 3);

    // Get the knot time
    double ts = traj->getKnotTime(0) + traj->getDt()/2.0;
    double tf = traj->getKnotTime(2) + traj->getDt()/2.0;
    
    // Skip if time is outside the range of the other trajectory
    pair<int, double> uss, usf;
    uss = traj->computeTimeIndex(ts);
    usf = traj->computeTimeIndex(tf);

    int umins = uss.first;
    int uminf = usf.first;
    double ss = uss.second;
    double sf = usf.second;

    // Add the parameter blocks
    vector<double *> factor_param_blocks;
    for (int kidx = umins; kidx < umins + 2; kidx++)
    {
        so3_param.push_back(traj->getKnotSO3(kidx).data());
        rv3_param.push_back(traj->getKnotOmg(kidx).data());
        rv3_param.push_back(traj->getKnotAlp(kidx).data());
        rv3_param.push_back(traj->getKnotPos(kidx).data());
        rv3_param.push_back(traj->getKnotVel(kidx).data());
        rv3_param.push_back(traj->getKnotAcc(kidx).data());

        factor_param_blocks.push_back(traj->getKnotSO3(kidx).data());
        factor_param_blocks.push_back(traj->getKnotOmg(kidx).data());
        factor_param_blocks.push_back(traj->getKnotAlp(kidx).data());
        factor_param_blocks.push_back(traj->getKnotPos(kidx).data());
        factor_param_blocks.push_back(traj->getKnotVel(kidx).data());
        factor_param_blocks.push_back(traj->getKnotAcc(kidx).data());
    }

    for (int kidx = uminf; kidx < uminf + 2; kidx++)
    {
        so3_param.push_back(traj->getKnotSO3(kidx).data());
        rv3_param.push_back(traj->getKnotOmg(kidx).data());
        rv3_param.push_back(traj->getKnotAlp(kidx).data());
        rv3_param.push_back(traj->getKnotPos(kidx).data());
        rv3_param.push_back(traj->getKnotVel(kidx).data());
        rv3_param.push_back(traj->getKnotAcc(kidx).data());

        factor_param_blocks.push_back(traj->getKnotSO3(kidx).data());
        factor_param_blocks.push_back(traj->getKnotOmg(kidx).data());
        factor_param_blocks.push_back(traj->getKnotAlp(kidx).data());
        factor_param_blocks.push_back(traj->getKnotPos(kidx).data());
        factor_param_blocks.push_back(traj->getKnotVel(kidx).data());
        factor_param_blocks.push_back(traj->getKnotAcc(kidx).data());
    }

    // Add the extrinsics
    {
        so3_param.push_back(R_Lx_Ly.data());
        rv3_param.push_back(P_Lx_Ly.data());

        factor_param_blocks.push_back(R_Lx_Ly.data());
        factor_param_blocks.push_back(P_Lx_Ly.data());
    }

    // Create the factor
    double xtsCovROSJerk = 5.7;
    double xtsCovPVAJerk = 5.7;
    // GPMixerPtr gpmxts(new GPMixer(traj->getDt(), xtsCovROSJerk, xtsCovPVAJerk, traj->getPoseRepresentation()));

    double mp_loss_thres = -1;
    // nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
    ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
    ceres::CostFunction *cost_function = new GPExtrinsicFactor(xtsCovROSJerk, xtsCovPVAJerk, traj->getGPMixerPtr(), traj->getGPMixerPtr(), ss, sf);
    
    // Add res block
    auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
    res_ids.push_back(res_block);

    // Copy the factor meta
    factorMeta.so3_parameter_blocks = so3_param;
    factorMeta.rv3_parameter_blocks = rv3_param;
    factorMeta.residual_blocks = res_ids;
}

void TestAnalyticJacobian(ceres::Problem &problem, GaussianProcessPtr &swTraj, vector<LidarCoef> &Coef, const int &cidx)
{

    auto EvaluateFactorClass = [](ceres::Problem &problem, FactorMeta &factors, int local_pamaterization_type, VectorXd &res, MatrixXd &Jcb, double &cost, double &total_time, double &eval_time) -> void
    {
        TicToc tt_autodiff;
        
        eval_time = 0;

        res.setZero(); Jcb.setZero();
        
        vector <double> residual;
        MatrixXd J_autodiff;
        int count = 1;
        while(count-- > 0)
        {
            double eval_time_ = 0;
            EvaluateFactorRJ(problem, factors, local_pamaterization_type, cost, residual, J_autodiff, eval_time_);
            eval_time += eval_time_;
        }

        // Remove sidual block
        RemoveResidualBlock(problem, factors);

        // Extract residual and Jacobian
        res = Eigen::Map<Eigen::VectorXd>(residual.data(), residual.size());
        Jcb = J_autodiff;

        total_time = tt_autodiff.Toc();
    };

    // Motion priors
    {
        double cost_autodiff;
        double time_autodiff;
        double time_autodiff_eval;
        VectorXd residual_autodiff;
        MatrixXd Jacobian_autodiff;
        {
            // Test the autodiff Jacobian
            FactorMeta gpmp2kFactorMetaAutodiff;
            AddAutodiffGPMP2KFactor(swTraj, problem, gpmp2kFactorMetaAutodiff);
            if (gpmp2kFactorMetaAutodiff.parameter_blocks() == 0)
                return;

            EvaluateFactorClass(problem, gpmp2kFactorMetaAutodiff, 0, residual_autodiff, Jacobian_autodiff, cost_autodiff, time_autodiff, time_autodiff_eval);
            printf(KCYN "Motion Prior 2Knot Autodiff Jacobian: Size %2d x %2d. Params: %d. Cost: %f. Time: %f, %f\n",
                   Jacobian_autodiff.rows(), Jacobian_autodiff.cols(),
                   gpmp2kFactorMetaAutodiff.parameter_blocks(),
                   cost_autodiff, time_autodiff, time_autodiff_eval);
        }

        double cost_analytic;
        double time_analytic;
        double time_analytic_eval;
        VectorXd residual_analytic;
        MatrixXd Jacobian_analytic;
        {
            // Test the analytic Jacobian
            FactorMeta gpmp2kFactorMetaAnalytic;
            AddAnalyticGPMP2KFactor(swTraj, problem, gpmp2kFactorMetaAnalytic);
            if (gpmp2kFactorMetaAnalytic.parameter_blocks() == 0)
                return;

            EvaluateFactorClass(problem, gpmp2kFactorMetaAnalytic, 1, residual_analytic, Jacobian_analytic, cost_analytic, time_analytic, time_analytic_eval);
            printf(KCYN "Motion Prior 2Knot Autodiff Jacobian: Size %2d x %2d. Params: %d. Cost: %f. Time: %f, %f\n",
                   Jacobian_analytic.rows(), Jacobian_analytic.cols(),
                   gpmp2kFactorMetaAnalytic.parameter_blocks(),
                   cost_analytic, time_analytic, time_analytic_eval);
        }

        printf(KGRN "CIDX: %d. MotionPrior2K Factor. Residual max error: %.4f. Jacobian max error: %.4f. Time: %.3f, %.3f. Ratio: %.0f\%\n\n" RESET,
               cidx, maxDiff(residual_autodiff, residual_analytic), maxDiff(Jacobian_autodiff, Jacobian_analytic),
               time_autodiff, time_analytic,
               time_autodiff_eval/time_analytic_eval*100);
    }

    // Lidar factors
    {
        double cost_autodiff;
        double time_autodiff;
        double time_autodiff_eval;
        VectorXd residual_autodiff;
        MatrixXd Jacobian_autodiff;
        {
            // Test the autodiff Jacobian
            FactorMeta gplidarFactorMetaAutodiff;
            AddAutodiffGPLidarFactor(swTraj, problem, gplidarFactorMetaAutodiff, Coef);
            if (gplidarFactorMetaAutodiff.parameter_blocks() == 0)
                return;

            EvaluateFactorClass(problem, gplidarFactorMetaAutodiff, 0, residual_autodiff, Jacobian_autodiff, cost_autodiff, time_autodiff, time_autodiff_eval);
            printf(KCYN "Lidar Autodiff Jacobian: Size %2d x %2d. Params: %d. Cost: %f. Time: %f, %f\n",
                   Jacobian_autodiff.rows(), Jacobian_autodiff.cols(),
                   gplidarFactorMetaAutodiff.parameter_blocks(),
                   cost_autodiff, time_autodiff, time_autodiff_eval);
        }

        double cost_analytic;
        double time_analytic;
        double time_analytic_eval;
        VectorXd residual_analytic;
        MatrixXd Jacobian_analytic;
        {
            // Test the analytic Jacobian
            FactorMeta gplidarFactorMetaAnalytic;
            AddAnalyticGPLidarFactor(swTraj, problem, gplidarFactorMetaAnalytic, Coef);
            if (gplidarFactorMetaAnalytic.parameter_blocks() == 0)
                return;

            EvaluateFactorClass(problem, gplidarFactorMetaAnalytic, 1, residual_analytic, Jacobian_analytic, cost_analytic, time_analytic, time_analytic_eval);
            printf(KCYN "Lidar Analytic Jacobian: Size %2d x %2d. Params: %d. Cost: %f. Time: %f, %f\n",
                   Jacobian_analytic.rows(), Jacobian_analytic.cols(),
                   gplidarFactorMetaAnalytic.parameter_blocks(),
                   cost_analytic, time_analytic, time_analytic_eval);
        }

        printf(KGRN "CIDX: %d. Lidar Factor. Residual max error: %.4f. Jacobian max error: %.4f. Time: %.3f, %.3f. Ratio: %.0f\%\n\n" RESET,
               cidx, maxDiff(residual_autodiff, residual_analytic), maxDiff(Jacobian_autodiff, Jacobian_analytic),
               time_autodiff, time_analytic,
               time_autodiff_eval/time_analytic_eval*100);
    }

    // Add the extrinsic params
    SO3d R_Lx_Ly(Util::YPR2Quat(43, 57, 91));
    Vector3d P_Lx_Ly(11, 02, 19);
    problem.AddParameterBlock(R_Lx_Ly.data(), 4, new GPSO3dLocalParameterization());
    problem.AddParameterBlock(P_Lx_Ly.data(), 3);
    
    {
        double cost_autodiff;
        double time_autodiff;
        double time_autodiff_eval;
        VectorXd residual_autodiff;
        MatrixXd Jacobian_autodiff;
        {
            // Test the autodiff Jacobian
            FactorMeta factorMetaAutodiff;
            AddAutodiffGPXtsFactor(swTraj, problem, factorMetaAutodiff, R_Lx_Ly, P_Lx_Ly);
            if (factorMetaAutodiff.parameter_blocks() == 0)
                return;
            
            EvaluateFactorClass(problem, factorMetaAutodiff, 0, residual_autodiff, Jacobian_autodiff, cost_autodiff, time_autodiff, time_autodiff_eval);
            // cout << "Xts Jacobian_autodiff\n" << Jacobian_autodiff << endl;
            printf(KCYN "Xts Autodiff Jacobian: Size %2d x %2d. Params: %d. Cost: %f. Time: %f, %f\n",
                   Jacobian_autodiff.rows(), Jacobian_autodiff.cols(),
                   factorMetaAutodiff.parameter_blocks(),
                   cost_autodiff, time_autodiff, time_autodiff_eval);
        }
        
        double cost_analytic;
        double time_analytic;
        double time_analytic_eval;
        VectorXd residual_analytic;
        MatrixXd Jacobian_analytic;
        {
            // Test the analytic Jacobian
            FactorMeta factorMetaAnalytic;
            AddAnalyticGPXtsFactor(swTraj, problem, factorMetaAnalytic, R_Lx_Ly, P_Lx_Ly);
            if (factorMetaAnalytic.parameter_blocks() == 0)
                return;
            
            EvaluateFactorClass(problem, factorMetaAnalytic, 1, residual_analytic, Jacobian_analytic, cost_analytic, time_analytic, time_analytic_eval);
            // cout << "Xts Jacobian_analytic\n" << Jacobian_analytic << endl;
            printf(KCYN "Xts Analytic Jacobian: Size %2d x %2d. Params: %d. Cost: %f. Time: %f, %f\n",
                   Jacobian_analytic.rows(), Jacobian_analytic.cols(),
                   factorMetaAnalytic.parameter_blocks(),
                   cost_analytic, time_analytic, time_analytic_eval);
        }
        
        // cout << KYEL "Xts Jacobian diff:\n" RESET << thresholdMatrix(Jacobian_autodiff - Jacobian_analytic, 1e-9) << endl;
        printf(KGRN "CIDX: %d. Extrinsic Prior Factor. Residual max error: %.4f. Jacobian max error: %.4f. Time: %.3f, %.3f. Ratio: %.0f\%\n\n" RESET,
               cidx, maxDiff(residual_autodiff, residual_analytic), maxDiff(Jacobian_autodiff, Jacobian_analytic),
               time_autodiff, time_analytic,
               time_autodiff_eval/time_analytic_eval*100);
    }
}

int main(int argc, char **argv)
{
    GPMixer gmp(0.01102, Vector3d(10, 10, 10).asDiagonal(), Vector3d(10, 10, 10).asDiagonal());
    Vector3d X(4.3, 5.7, 11);
    Vector3d V(2, 20, 19);
    Vector3d A(15, 07, 24);

    // Test the jacobian internals
    {
        Matrix3d DJrUV_DU_analytic = gmp.DJrUV_DU(X, V);
        Matrix3d DJrUV_DU_autodiff;
        DJrUV_DU_autodiff
          <<  1.4795693330913913078012550976876, -0.09272692763418464904582492184798, -0.60125453973064792810535942193768,
             -0.2078396293352800836208836614803,  2.106797535770925924175386355789,   -0.13284986373953439932235327939372,
             -1.0285119629145335820195952420107, -0.2735388145177689190297496754912,   0.11676300088524277297144403964954;

        printf("DJrUV_DU_analytic error: %f\n",
               (DJrUV_DU_analytic - DJrUV_DU_autodiff).cwiseAbs().maxCoeff());
        cout << DJrUV_DU_analytic << endl;



        Matrix3d DDJrUVW_DUDU_analytic = gmp.DDJrUVW_DUDU(X, V, A);
        Matrix3d DDJrUVW_DUDU_autodiff;
        DDJrUVW_DUDU_autodiff
          << -0.16628561001097342169099087651306,  8.7415558053382500878863515533626,   21.68743802346549542495135587745,
              2.1985844499890389278582167128789,  -8.7635373891101820685190252965589,   0.31275656668973516839002466049998,
             -1.3623292065783150943538504915428,  -0.26089688300928635412337970935968, -6.6683728805496298190367494634219;

        printf("DDJrUVW_DUDU_analytic error: %f\n",
               (DDJrUVW_DUDU_analytic - DDJrUVW_DUDU_autodiff).cwiseAbs().maxCoeff());
        cout << DDJrUVW_DUDU_analytic << endl;



        Matrix3d DDJrUVW_DUDV_analytic = gmp.DDJrUVW_DUDV(X, V, A);
        Matrix3d DDJrUVW_DUDV_autodiff;
        DDJrUVW_DUDV_autodiff
          << 1.8341786490550746913724895583765,  0.72667635330926210250901007030542, -0.58355483236312445440819761247303,
            -1.105371216325137374863550131557,   1.0826099595468673186725431630753,  -0.5789402725088745785550416913433,
             0.3565297928789546325090220978397, -1.3046627728062866767984473282304,   0.57052930243532971888228331378519;

        printf("DDJrUVW_DUDV_analytic error: %f\n",
               (DDJrUVW_DUDV_analytic - DDJrUVW_DUDV_autodiff).cwiseAbs().maxCoeff());
        cout << DDJrUVW_DUDV_analytic << endl;



        Matrix3d DJrInvUV_DU_analytic = gmp.DJrInvUV_DU(X, V);
        Matrix3d DJrInvUV_DU_autodiff;
        DJrInvUV_DU_autodiff
          <<  53.850477114916400753611771760056,  128.29236486057202749493046758121,  230.1945527540237207146427817569,
             -125.85089351206730209166669504274, -210.15361822396008007225049065689, -306.92185420033828389935326258456,
              62.016904967876696903384730702615,  43.162622478265109060739271800087,  70.152538431552944140723451235805;
        printf("DJrInvUV_DU_analytic error: %f\n",
               (DJrInvUV_DU_analytic - DJrInvUV_DU_autodiff).cwiseAbs().maxCoeff());
        cout << DJrInvUV_DU_analytic << endl;



        Matrix3d DDJrInvUV_DUDX_analytic = gmp.DDJrInvUVW_DUDU(X, V, A);
        Matrix3d DDJrInvUV_DUDX_autodiff;
        DDJrInvUV_DUDX_autodiff
          << -6604.1192221875057982056387635241, -11742.978198354910825157363234684, -23023.800403371279667546327746653,
              11716.420889581207373357214899501,   19044.19517655420675726016405003,  30943.951304134615906701614255486,
             -4275.5969103108402017377943730546, -4438.3051006541123211015442832802, -7571.1236170508494051731665778526;

        printf("DDJrInvUV_DUDX_analytic error: %f\n",
               (DDJrInvUV_DUDX_analytic - DDJrInvUV_DUDX_autodiff).cwiseAbs().maxCoeff());
        cout << DDJrInvUV_DUDX_analytic << endl;



        Matrix3d DDJrInvUVW_DUDV_analytic = gmp.DDJrInvUVW_DUDV(X, V, A);
        Matrix3d DDJrInvUVW_DUDV_autodiff;
        DDJrInvUVW_DUDV_autodiff
          << -1085.5492170614632570397368917895,  158.93672159037582796317033606738,  327.51773574524891872295703546664,
              182.93672159037582796317033606738, -972.94238057172044072848333021843,  440.42162827351622385147861947185,
              320.51773574524891872295703546664,  455.42162827351622385147861947185, -359.65343067039712832221988250374;
    
        printf("DDJrInvUVW_DUDV_analytic error: %f\n",
               (DDJrInvUVW_DUDV_analytic - DDJrInvUVW_DUDV_autodiff).cwiseAbs().maxCoeff());
        cout << DDJrInvUVW_DUDV_analytic << endl;
    }


    // Check the SO3xR3 jacobian
    {
        double Dt = 0.04357;
        Matrix3d mpCovROSJerk = Vector3d(10, 10, 10).asDiagonal();
        Matrix3d mpCovPVAJerk = Vector3d(10, 10, 10).asDiagonal();
        GaussianProcessPtr traj(new GaussianProcess(Dt, mpCovROSJerk, mpCovPVAJerk, false, POSE_GROUP::SO3xR3, 1e-3, true));
        traj->setStartTime(0.5743);
        traj->genRandomTrajectory(6);

        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;

        // Create the ceres problem
        CreateCeresProblem(problem, options, summary, traj, -0.1, -0.1);

        // Create a random number engine
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        // Define a distribution (e.g., uniform distribution in the range [1, 100])
        std::uniform_int_distribution<> dis(1, 100);

        // Create the fake lidar factors
        int Npt = 4;
        int Nknot = traj->getNumKnots();
        vector<LidarCoef> Coef((Nknot - 1)*Npt);
        for(int kidx = 0; kidx < Nknot - 1; kidx++)
        {
            for(int pidx = 0; pidx < Npt; pidx++)
            {
                Vector3d n = Vector3d::Random();
                n /= n.norm();

                Coef[kidx*4 + pidx].t =  traj->getKnotTime(kidx) + Dt/(Npt + 1)*(pidx + 1);
                Coef[kidx*4 + pidx].f =  Vector3d::Random();
                Coef[kidx*4 + pidx].n << n, double(dis(gen));
                Coef[kidx*4 + pidx].plnrty = 0.69;
            }
        }

        // Test the jacobian
        TestAnalyticJacobian(problem, traj, Coef, 0);
    }
}