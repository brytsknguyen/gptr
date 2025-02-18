#include "utility.h"
#include "GaussianProcess.hpp"

#include "factor/GPMotionPriorTwoKnotsFactor.h"
#include "factor/GPMotionPriorTwoKnotsFactorAutodiff.h"

#include "factor/GPIMUFactorAutodiff.h"
#include "factor/GPIMUFactor.h"

#include "factor/GPExtrinsicFactor.h"
#include "factor/GPExtrinsicFactorAutodiff.h"

#include "AutoDiffSO3Parameterization.hpp"

struct FactorMeta
{
    vector<double *> so3_parameter_blocks;
    vector<double *> r3_parameter_blocks;
    vector<ceres::ResidualBlockId> residual_blocks;

    int parameter_blocks()
    {
        return (so3_parameter_blocks.size() + r3_parameter_blocks.size());
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

void GetFactorJacobian(ceres::Problem &problem, FactorMeta &factorMeta,
                       int local_pamaterization_type,
                       double &cost, vector<double> &residual,
                       MatrixXd &Jacobian)
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
    problem.Evaluate(e_option, &cost, &residual, NULL, &Jacobian_);
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

void AddAutodiffGPMP2KFactor(GaussianProcessPtr &traj, ceres::Problem &problem, FactorMeta &gpmpFactorMeta)
{
    vector<double *> so3_param;
    vector<double *> r3_param;
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
            r3_param.push_back(traj->getKnotOmg(knot_idx).data());
            r3_param.push_back(traj->getKnotAlp(knot_idx).data());
            r3_param.push_back(traj->getKnotPos(knot_idx).data());
            r3_param.push_back(traj->getKnotVel(knot_idx).data());
            r3_param.push_back(traj->getKnotAcc(knot_idx).data());
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
    gpmpFactorMeta.r3_parameter_blocks = r3_param;
    gpmpFactorMeta.residual_blocks = res_ids_gp;
    // printf("Autodiff params: %d, %d, %d, %d, %d, %d\n",
    //         so3_param.size(), gpmpFactorMeta.so3_parameter_blocks.size(),
    //         r3_param.size(), gpmpFactorMeta.r3_parameter_blocks.size(),
    //         res_ids_gp.size(), gpmpFactorMeta.residual_blocks.size());
}

void AddAnalyticGPMP2KFactor(GaussianProcessPtr &traj, ceres::Problem &problem, FactorMeta &gpmpFactorMeta)
{
    vector<double *> so3_param;
    vector<double *> r3_param;
    vector<ceres::ResidualBlockId> res_ids_gp;
    // Add GP factors between consecutive knots
    for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
    {
        vector<double *> factor_param_blocks;
        // Add the parameter blocks
        for (int knot_idx = kidx; knot_idx < kidx + 2; knot_idx++)
        {
            so3_param.push_back(traj->getKnotSO3(knot_idx).data());
            r3_param.push_back(traj->getKnotOmg(knot_idx).data());
            r3_param.push_back(traj->getKnotAlp(knot_idx).data());
            r3_param.push_back(traj->getKnotPos(knot_idx).data());
            r3_param.push_back(traj->getKnotVel(knot_idx).data());
            r3_param.push_back(traj->getKnotAcc(knot_idx).data());
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
    gpmpFactorMeta.r3_parameter_blocks = r3_param;
    gpmpFactorMeta.residual_blocks = res_ids_gp;
    // printf("Analytic params: %d, %d, %d, %d, %d, %d\n",
    //         so3_param.size(), gpmpFactorMeta.so3_parameter_blocks.size(),
    //         r3_param.size(), gpmpFactorMeta.r3_parameter_blocks.size(),
    //         res_ids_gp.size(), gpmpFactorMeta.residual_blocks.size());
}

void AddAutodiffIMUFactor(GaussianProcessPtr &traj, ceres::Problem &problem, FactorMeta &gpmpFactorMeta, vector<IMUData> &imu_data, Eigen::Vector3d &bg, Eigen::Vector3d &ba, Eigen::Vector3d &g)
{
    vector<double *> so3_param;
    vector<double *> r3_param;
    vector<ceres::ResidualBlockId> res_ids_gp;
    // Add the GP factors based on knot difference
    for (int kidx = 0; kidx < imu_data.size(); kidx++)
    {
        // Create the factor
        IMUData imu = imu_data[kidx];
        vector<double *> factor_param_blocks;
        auto   us = traj->computeTimeIndex(imu.t);
        int    u  = us.first;
        double s  = us.second;        
        double gp_loss_thres = -1;
        ceres::LossFunction *gp_loss_func = gp_loss_thres == -1 ? NULL : new ceres::HuberLoss(gp_loss_thres);
        GPIMUFactorAutodiff *GPMPFactor = new GPIMUFactorAutodiff(imu.acc, imu.gyro, ba, bg, 1, 1, 1, 1, traj->getGPMixerPtr(), s, traj->getNumKnots());
        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<GPIMUFactorAutodiff>(GPMPFactor);
        cost_function->SetNumResiduals(12);
        // Add the parameter blocks
        for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
        {
            so3_param.push_back(traj->getKnotSO3(knot_idx).data());
            r3_param.push_back(traj->getKnotOmg(knot_idx).data());
            r3_param.push_back(traj->getKnotAlp(knot_idx).data());
            r3_param.push_back(traj->getKnotPos(knot_idx).data());
            r3_param.push_back(traj->getKnotVel(knot_idx).data());
            r3_param.push_back(traj->getKnotAcc(knot_idx).data());
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
        factor_param_blocks.push_back(bg.data());
        factor_param_blocks.push_back(ba.data());   
        factor_param_blocks.push_back(g.data());   
        r3_param.push_back(bg.data());
        r3_param.push_back(ba.data());       
        r3_param.push_back(g.data());   
        cost_function->AddParameterBlock(3);
        cost_function->AddParameterBlock(3);           
        cost_function->AddParameterBlock(3);       
        auto res_block = problem.AddResidualBlock(cost_function, gp_loss_func, factor_param_blocks);
        res_ids_gp.push_back(res_block);
    }
    gpmpFactorMeta.so3_parameter_blocks = so3_param;
    gpmpFactorMeta.r3_parameter_blocks = r3_param;
    gpmpFactorMeta.residual_blocks = res_ids_gp;
    // printf("Autodiff params: %d, %d, %d, %d, %d, %d\n",
    //         so3_param.size(), gpmpFactorMeta.so3_parameter_blocks.size(),
    //         r3_param.size(), gpmpFactorMeta.r3_parameter_blocks.size(),
    //         res_ids_gp.size(), gpmpFactorMeta.residual_blocks.size());
}

void AddAnalyticIMUFactor(GaussianProcessPtr &traj, ceres::Problem &problem, FactorMeta &gpmpFactorMeta, vector<IMUData> &imu_data, Eigen::Vector3d &bg, Eigen::Vector3d &ba, Eigen::Vector3d &g)
{
    vector<double *> so3_param;
    vector<double *> r3_param;
    vector<ceres::ResidualBlockId> res_ids_gp;
    for (int kidx = 0; kidx < imu_data.size(); kidx++)
    {
        IMUData imu = imu_data[kidx];
        vector<double *> factor_param_blocks;
        auto   us = traj->computeTimeIndex(imu.t);
        int    u  = us.first;
        double s  = us.second;        
        // Add the parameter blocks
        for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
        {
            so3_param.push_back(traj->getKnotSO3(knot_idx).data());
            r3_param.push_back(traj->getKnotOmg(knot_idx).data());
            r3_param.push_back(traj->getKnotAlp(knot_idx).data());
            r3_param.push_back(traj->getKnotPos(knot_idx).data());
            r3_param.push_back(traj->getKnotVel(knot_idx).data());
            r3_param.push_back(traj->getKnotAcc(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAlp(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
        }
        factor_param_blocks.push_back(bg.data());
        factor_param_blocks.push_back(ba.data());    
        factor_param_blocks.push_back(g.data());  
        r3_param.push_back(bg.data());
        r3_param.push_back(ba.data());       
        r3_param.push_back(g.data());            
        // Create the factors
        double mp_loss_thres = -1;
        // nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
        ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
        ceres::CostFunction *cost_function = new GPIMUFactor(imu.acc, imu.gyro, ba, bg, 1, 1, 1, 1, traj->getGPMixerPtr(), s);
        auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
        res_ids_gp.push_back(res_block);
    }
    
    gpmpFactorMeta.so3_parameter_blocks = so3_param;
    gpmpFactorMeta.r3_parameter_blocks = r3_param;
    gpmpFactorMeta.residual_blocks = res_ids_gp;
    // printf("Analytic params: %d, %d, %d, %d, %d, %d\n",
    //         so3_param.size(), gpmpFactorMeta.so3_parameter_blocks.size(),
    //         r3_param.size(), gpmpFactorMeta.r3_parameter_blocks.size(),
    //         res_ids_gp.size(), gpmpFactorMeta.residual_blocks.size());
}

void TestAnalyticJacobian(ceres::Problem &problem, GaussianProcessPtr &swTraj, vector<IMUData> &imu_data, const int &cidx)
{
    // Motion priors
    {
        double time_autodiff;
        VectorXd residual_autodiff_;
        MatrixXd Jacobian_autodiff_;
        {
            // Test the autodiff Jacobian
            FactorMeta gpmp2kFactorMetaAutodiff;
            AddAutodiffGPMP2KFactor(swTraj, problem, gpmp2kFactorMetaAutodiff);
            if (gpmp2kFactorMetaAutodiff.parameter_blocks() == 0)
                return;
            TicToc tt_autodiff;
            double cost_autodiff;
            vector <double> residual_autodiff;
            MatrixXd J_autodiff;
            int count = 100;
            while(count-- > 0)
                GetFactorJacobian(problem, gpmp2kFactorMetaAutodiff, 0,
                                  cost_autodiff, residual_autodiff, J_autodiff);
            RemoveResidualBlock(problem, gpmp2kFactorMetaAutodiff);
            printf(KCYN "Motion Prior 2Knot Autodiff Jacobian: Size %2d %2d. Params: %d. Cost: %f. Time: %f.\n",
                   J_autodiff.rows(), J_autodiff.cols(),
                   gpmp2kFactorMetaAutodiff.parameter_blocks(),
                   cost_autodiff, time_autodiff = tt_autodiff.Toc());
            residual_autodiff_ = Eigen::Map<Eigen::VectorXd>(residual_autodiff.data(), residual_autodiff.size());
            Jacobian_autodiff_ = J_autodiff;//MatrixXd(15, 9).setZero();
            // Jacobian_autodiff_ = J_autodiff.block(0, 0, 9, 36);
            // cout << "residual:\n" << residual_autodiff_.transpose() << endl;
            // cout << "jacobian:\n" << Jacobian_autodiff_ << RESET << endl;
        }
        double time_analytic;
        VectorXd residual_analytic_;
        MatrixXd Jacobian_analytic_;
        {
            // Test the analytic Jacobian
            FactorMeta gpmp2kFactorMetaAnalytic;
            AddAnalyticGPMP2KFactor(swTraj, problem, gpmp2kFactorMetaAnalytic);
            if (gpmp2kFactorMetaAnalytic.parameter_blocks() == 0)
                return;
            TicToc tt_analytic;
            double cost_analytic;
            vector <double> residual_analytic;
            MatrixXd J_analytic;
            int count = 100;
            while(count-- > 0)
                GetFactorJacobian(problem, gpmp2kFactorMetaAnalytic, 1,
                                  cost_analytic, residual_analytic, J_analytic);
            RemoveResidualBlock(problem, gpmp2kFactorMetaAnalytic);
            printf(KMAG "Motion Prior 2Knot Analytic Jacobian: Size %2d %2d. Params: %d. Cost: %f. Time: %f.\n",
                   J_analytic.rows(), J_analytic.cols(),
                   gpmp2kFactorMetaAnalytic.parameter_blocks(),
                   cost_analytic, time_analytic = tt_analytic.Toc());
            residual_analytic_ = Eigen::Map<Eigen::VectorXd>(residual_analytic.data(), residual_analytic.size());
            Jacobian_analytic_ = J_analytic;//MatrixXd(15, 9).setZero();
            // Jacobian_analytic_ = J_analytic.block(0, 0, 9, 36);
            // cout << "residual:\n" << residual_analytic_.transpose() << endl;
            // cout << "jacobian:\n" << Jacobian_analytic_ << RESET << endl;
        }
        // Compare the two jacobians
        VectorXd resdiff = residual_autodiff_ - residual_analytic_;
        MatrixXd jcbdiff = Jacobian_autodiff_ - Jacobian_analytic_;
        // cout << KRED "residual diff:\n" RESET << resdiff.transpose() << endl;
        // cout << KRED "jacobian diff:\n" RESET << jcbdiff << endl;
        // if (maxCoef < jcbdiff.cwiseAbs().maxCoeff() && cidx != 0)
        //     maxCoef = jcbdiff.cwiseAbs().maxCoeff();
        printf(KGRN "CIDX: %d. MotionPrior 2K Jacobian max error: %.4f. Time: %.3f, %.3f. Ratio: %.0f\%\n\n" RESET,
               cidx, jcbdiff.cwiseAbs().maxCoeff(), time_autodiff, time_analytic, time_autodiff/time_analytic*100);
    }
    
    // IMU
    {
        Eigen::Vector3d bg = Eigen::Vector3d::Zero();        
        Eigen::Vector3d ba = Eigen::Vector3d::Zero();     
        Eigen::Vector3d g = Eigen::Vector3d(0, 0, 9.81);     
        double time_autodiff;
        VectorXd residual_autodiff_;
        MatrixXd Jacobian_autodiff_;
        {
            // Test the autodiff Jacobian
            FactorMeta gpimuFactorMetaAutodiff;
            AddAutodiffIMUFactor(swTraj, problem, gpimuFactorMetaAutodiff, imu_data, bg, ba, g);
            if (gpimuFactorMetaAutodiff.parameter_blocks() == 0)
                return;
            TicToc tt_autodiff;
            double cost_autodiff;
            vector <double> residual_autodiff;
            MatrixXd J_autodiff;
            int count = 100;
            while(count-- > 0)
                GetFactorJacobian(problem, gpimuFactorMetaAutodiff, 0,
                                  cost_autodiff, residual_autodiff, J_autodiff);
            RemoveResidualBlock(problem, gpimuFactorMetaAutodiff);
            printf(KCYN "IMU Autodiff Jacobian: Size %2d %2d. Params: %d. Cost: %f. Time: %f.\n",
                   J_autodiff.rows(), J_autodiff.cols(),
                   gpimuFactorMetaAutodiff.parameter_blocks(),
                   cost_autodiff, time_autodiff = tt_autodiff.Toc());
            residual_autodiff_ = Eigen::Map<Eigen::VectorXd>(residual_autodiff.data(), residual_autodiff.size());
            Jacobian_autodiff_ = J_autodiff;//MatrixXd(15, 9).setZero();
            // Jacobian_autodiff_ = J_autodiff.block(0, 0, 9, 36);
            // cout << "residual:\n" << residual_autodiff_.transpose() << endl;
            // cout << "jacobian:\n" << Jacobian_autodiff_ << RESET << endl;
        }
        double time_analytic;
        VectorXd residual_analytic_;
        MatrixXd Jacobian_analytic_;
        {
            // Test the analytic Jacobian
            FactorMeta gpimuFactorMetaAnalytic;
            AddAnalyticIMUFactor(swTraj, problem, gpimuFactorMetaAnalytic, imu_data, bg, ba, g);
            if (gpimuFactorMetaAnalytic.parameter_blocks() == 0)
                return;
            TicToc tt_analytic;
            double cost_analytic;
            vector <double> residual_analytic;
            MatrixXd J_analytic;
            int count = 100;
            while(count-- > 0)
                GetFactorJacobian(problem, gpimuFactorMetaAnalytic, 1,
                                  cost_analytic, residual_analytic, J_analytic);
            RemoveResidualBlock(problem, gpimuFactorMetaAnalytic);
            printf(KMAG "IMU Analytic Jacobian: Size %2d %2d. Params: %d. Cost: %f. Time: %f.\n",
                   J_analytic.rows(), J_analytic.cols(),
                   gpimuFactorMetaAnalytic.parameter_blocks(),
                   cost_analytic, time_analytic = tt_analytic.Toc());
            residual_analytic_ = Eigen::Map<Eigen::VectorXd>(residual_analytic.data(), residual_analytic.size());
            Jacobian_analytic_ = J_analytic;//MatrixXd(15, 9).setZero();
            // Jacobian_analytic_ = J_analytic.block(0, 0, 9, 36);
            // cout << "residual:\n" << residual_analytic_.transpose() << endl;
            // cout << "jacobian:\n" << Jacobian_analytic_ << RESET << endl;
        }
        // Compare the two jacobians
        VectorXd resdiff = residual_autodiff_ - residual_analytic_;
        MatrixXd jcbdiff = Jacobian_autodiff_ - Jacobian_analytic_;
        // cout << KRED "residual diff:\n" RESET << resdiff.transpose() << endl;
        // cout << KRED "jacobian diff:\n" RESET << jcbdiff << endl;
        // if (maxCoef < jcbdiff.cwiseAbs().maxCoeff() && cidx != 0)
        //     maxCoef = jcbdiff.cwiseAbs().maxCoeff();
        printf(KGRN "CIDX: %d. IMU Jacobian max error: %.4f. Time: %.3f, %.3f. Ratio: %.0f\%\n\n" RESET,
               cidx, jcbdiff.cwiseAbs().maxCoeff(), time_autodiff, time_analytic, time_autodiff/time_analytic*100);
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


    // Check the factor jacobian
    {
        double Dt = 0.04357;
        GaussianProcessPtr traj(new GaussianProcess(Dt, Vector3d(10, 10, 10).asDiagonal(), Vector3d(10, 10, 10).asDiagonal(), false, POSE_GROUP::SO3xR3));
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

        // Create the fake imu factors
        int Nknot = traj->getNumKnots();
        vector<IMUData> imu_data;
        for(int idx = 0; idx < Nknot - 1; idx++)
        {
            double t = traj->getKnotTime(idx) + 0.5 * traj->getDt();
            Eigen::Vector3d acc = Eigen::Vector3d::Random();
            Eigen::Vector3d gyro = Eigen::Vector3d::Random();
            IMUData imu(t, acc, gyro);
            imu_data.push_back(imu);
        }
        // Test the jacobian
        TestAnalyticJacobian(problem, traj, imu_data, 0);
    }
}
