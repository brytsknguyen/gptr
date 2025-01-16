#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <ceres/ceres.h>

#include "utility.h"

// All about gaussian process
#include "GaussianProcess.hpp"

// Utilities to manage params and marginalization
#include "GaussNewtonUtilities.hpp"

// Factors
#include "factor/GPExtrinsicFactor.h"
#include "factor/GPPointToPlaneFactor.h"
#include "factor/GPMotionPriorTwoKnotsFactor.h"

#include "factor/GPExtrinsicFactorTMN.hpp"
#include "factor/GPPointToPlaneFactorTMN.hpp"
#include "factor/GPMotionPriorTwoKnotsFactorTMN.hpp"

struct lidarFeaIdx
{
    lidarFeaIdx(int &cidx_, int &fidx_, int absidx_)
        : cidx(cidx_), fidx(fidx_), absidx(absidx_) {};

    int cidx; int fidx; int absidx;
};

class MLCME
{
    typedef SparseMatrix<double> SMd;

private:

    // Node handle to get information needed
    NodeHandlePtr nh;

    int Nlidar;

    vector<SO3d> R_Lx_Ly;
    vector<Vec3> P_Lx_Ly;

protected:

    double fix_time_begin = -1;
    double fix_time_end = -1;
    
    int max_ceres_iter = 50;

    double lidar_weight = 1.0;
    double ld_loss_thres = -1.0;
    double xt_loss_thres = -1.0;
    double mp_loss_thres = -1.0;
    
    double max_omg = 20.0;
    double max_alp = 10.0;
    
    double max_vel = 10.0;
    double max_acc = 2.0;

    double xtSigGa = 1.0;
    double xtSigNu = 1.0;

    int max_lidarcoefs = 4000;

    deque<int> kidx_marg;
    deque<int> kidx_keep;

    // Map of traj-kidx and parameter id
    map<pair<int, int>, int> tk2p;
    ParamInfoMap paramInfoMap;
    MarginalizationInfoPtr margInfo;
    // MarginalizationFactor* margFactor = NULL;

    bool use_ceres = true;
    bool compute_cost = false;
    bool fuse_marg = false;

public:

//     // Destructor
//    ~MLCME() {};

    // void InsertZeroRow(MatrixXd &M, int row, int size)
    // {
    //     int Mrow = M.rows();
    //     int Mcol = M.cols();

    //     MatrixXd M_(Mrow + size, Mcol);
    //     M_ << M.topRows(row), MatrixXd::Zero(size, Mcol), M.bottomRows(Mrow - row);
    //     M.resize(Mrow + size, Mcol);
    //     M = M_;
    // }

    Eigen::MatrixXd CRSToEigenDense(ceres::CRSMatrix &J)
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

    void FindJrByCeres(ceres::Problem &problem, FactorMeta &factorMeta,
                    double &cost, VectorXd &residual, MatrixXd &Jacobian)
    {
        ceres::Problem::EvaluateOptions e_option;
        ceres::CRSMatrix Jacobian_;
        e_option.residual_blocks = factorMeta.res;
        vector<double> residual_;
        problem.Evaluate(e_option, &cost, &residual_, NULL, &Jacobian_);
        residual = Eigen::Map<VectorXd>(residual_.data(), residual_.size());
        Jacobian = CRSToEigenDense(Jacobian_);
    }

    // Constructor
    MLCME(NodeHandlePtr &nh_, int Nlidar_)
        : nh(nh_), Nlidar(Nlidar_), R_Lx_Ly(vector<SO3d>(Nlidar_, SO3d())), P_Lx_Ly(vector<Vec3>(Nlidar_, Vec3(0, 0, 0)))
    {
        use_ceres = Util::GetBoolParam(nh, "use_ceres", true);
        Util::GetParam(nh, "fix_time_begin", fix_time_begin);
        Util::GetParam(nh, "fix_time_end"  , fix_time_end  );
        Util::GetParam(nh, "max_ceres_iter", max_ceres_iter);
        Util::GetParam(nh, "max_lidarcoefs", max_lidarcoefs);
        Util::GetParam(nh, "lidar_weight"  , lidar_weight  );
        Util::GetParam(nh, "ld_loss_thres" , ld_loss_thres );
        Util::GetParam(nh, "xt_loss_thres" , xt_loss_thres );
        Util::GetParam(nh, "mp_loss_thres" , mp_loss_thres );
        Util::GetParam(nh, "max_omg", max_omg);
        Util::GetParam(nh, "max_alp", max_alp);
        Util::GetParam(nh, "max_vel", max_omg);
        Util::GetParam(nh, "max_acc", max_alp);
        Util::GetParam(nh, "xtSigGa", xtSigGa);
        Util::GetParam(nh, "xtSigNu", xtSigNu);
        fuse_marg = Util::GetBoolParam(nh, "fuse_marg", false);
        compute_cost = Util::GetBoolParam(nh, "compute_cost", false);
    };

    void AddTrajParams(
        ceres::Problem &problem, vector<GaussianProcessPtr> &trajs, int &tidx,
        ParamInfoMap &paramInfoMap, double tmin, double tmax, double tmid)
    {
        GaussianProcessPtr &traj = trajs[tidx];

        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = min(traj->getNumKnots() - 1, usmax.first + 1);

        for (int kidx = kidxmin; kidx <= kidxmax; kidx++)
        {
            problem.AddParameterBlock(traj->getKnotSO3(kidx).data(), 4, new GPSO3dLocalParameterization());
            problem.AddParameterBlock(traj->getKnotOmg(kidx).data(), 3);
            problem.AddParameterBlock(traj->getKnotAlp(kidx).data(), 3);
            problem.AddParameterBlock(traj->getKnotPos(kidx).data(), 3);
            problem.AddParameterBlock(traj->getKnotVel(kidx).data(), 3);
            problem.AddParameterBlock(traj->getKnotAcc(kidx).data(), 3);

            if (max_omg > 0 && kidx == kidxmax)
            {
                problem.SetParameterLowerBound(traj->getKnotOmg(kidx).data(), 0, -max_omg);
                problem.SetParameterLowerBound(traj->getKnotOmg(kidx).data(), 1, -max_omg);
                problem.SetParameterLowerBound(traj->getKnotOmg(kidx).data(), 2, -max_omg);
                problem.SetParameterUpperBound(traj->getKnotOmg(kidx).data(), 0,  max_omg);
                problem.SetParameterUpperBound(traj->getKnotOmg(kidx).data(), 1,  max_omg);
                problem.SetParameterUpperBound(traj->getKnotOmg(kidx).data(), 2,  max_omg);
            }

            if (max_alp > 0 && kidx == kidxmax)
            {
                problem.SetParameterLowerBound(traj->getKnotAlp(kidx).data(), 0, -max_alp);
                problem.SetParameterLowerBound(traj->getKnotAlp(kidx).data(), 1, -max_alp);
                problem.SetParameterLowerBound(traj->getKnotAlp(kidx).data(), 2, -max_alp);
                problem.SetParameterUpperBound(traj->getKnotAlp(kidx).data(), 0,  max_alp);
                problem.SetParameterUpperBound(traj->getKnotAlp(kidx).data(), 1,  max_alp);
                problem.SetParameterUpperBound(traj->getKnotAlp(kidx).data(), 2,  max_alp);
            }

            if (max_vel > 0 && kidx == kidxmax)
            {
                problem.SetParameterLowerBound(traj->getKnotVel(kidx).data(), 0, -max_vel);
                problem.SetParameterLowerBound(traj->getKnotVel(kidx).data(), 1, -max_vel);
                problem.SetParameterLowerBound(traj->getKnotVel(kidx).data(), 2, -max_vel);
                problem.SetParameterUpperBound(traj->getKnotVel(kidx).data(), 0,  max_vel);
                problem.SetParameterUpperBound(traj->getKnotVel(kidx).data(), 1,  max_vel);
                problem.SetParameterUpperBound(traj->getKnotVel(kidx).data(), 2,  max_vel);
            }

            if (max_acc > 0 && kidx == kidxmax)
            {
                problem.SetParameterLowerBound(traj->getKnotAcc(kidx).data(), 0, -max_acc);
                problem.SetParameterLowerBound(traj->getKnotAcc(kidx).data(), 1, -max_acc);
                problem.SetParameterLowerBound(traj->getKnotAcc(kidx).data(), 2, -max_acc);
                problem.SetParameterUpperBound(traj->getKnotAcc(kidx).data(), 0,  max_acc);
                problem.SetParameterUpperBound(traj->getKnotAcc(kidx).data(), 1,  max_acc);
                problem.SetParameterUpperBound(traj->getKnotAcc(kidx).data(), 2,  max_acc);
            }
            
            // Log down the information of the params
            paramInfoMap.insert(traj->getKnotSO3(kidx).data(), ParamInfo(traj->getKnotSO3(kidx).data(), getEigenPtr(traj->getKnotSO3(kidx)), ParamType::SO3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 0));
            paramInfoMap.insert(traj->getKnotOmg(kidx).data(), ParamInfo(traj->getKnotOmg(kidx).data(), getEigenPtr(traj->getKnotOmg(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 1));
            paramInfoMap.insert(traj->getKnotAlp(kidx).data(), ParamInfo(traj->getKnotAlp(kidx).data(), getEigenPtr(traj->getKnotAlp(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 2));
            paramInfoMap.insert(traj->getKnotPos(kidx).data(), ParamInfo(traj->getKnotPos(kidx).data(), getEigenPtr(traj->getKnotPos(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 3));
            paramInfoMap.insert(traj->getKnotVel(kidx).data(), ParamInfo(traj->getKnotVel(kidx).data(), getEigenPtr(traj->getKnotVel(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 4));
            paramInfoMap.insert(traj->getKnotAcc(kidx).data(), ParamInfo(traj->getKnotAcc(kidx).data(), getEigenPtr(traj->getKnotAcc(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 5));

            if (traj->getKnotTime(kidx) < tmin + fix_time_begin && fix_time_begin > 0)
            {
                problem.SetParameterBlockConstant(traj->getKnotSO3(kidx).data());
                problem.SetParameterBlockConstant(traj->getKnotOmg(kidx).data());
                problem.SetParameterBlockConstant(traj->getKnotAlp(kidx).data());
                problem.SetParameterBlockConstant(traj->getKnotPos(kidx).data());
                problem.SetParameterBlockConstant(traj->getKnotVel(kidx).data());
                problem.SetParameterBlockConstant(traj->getKnotAcc(kidx).data());

                paramInfoMap[traj->getKnotSO3(kidx).data()].fixed = true;
                paramInfoMap[traj->getKnotOmg(kidx).data()].fixed = true;
                paramInfoMap[traj->getKnotAlp(kidx).data()].fixed = true;
                paramInfoMap[traj->getKnotPos(kidx).data()].fixed = true;
                paramInfoMap[traj->getKnotVel(kidx).data()].fixed = true;
                paramInfoMap[traj->getKnotAcc(kidx).data()].fixed = true;
            }

            if (traj->getKnotTime(kidx) > tmax - fix_time_end && fix_time_end > 0)
            {
                problem.SetParameterBlockConstant(traj->getKnotSO3(kidx).data());
                problem.SetParameterBlockConstant(traj->getKnotOmg(kidx).data());
                problem.SetParameterBlockConstant(traj->getKnotAlp(kidx).data());
                problem.SetParameterBlockConstant(traj->getKnotPos(kidx).data());
                problem.SetParameterBlockConstant(traj->getKnotVel(kidx).data());
                problem.SetParameterBlockConstant(traj->getKnotAcc(kidx).data());

                paramInfoMap[traj->getKnotSO3(kidx).data()].fixed = true;
                paramInfoMap[traj->getKnotOmg(kidx).data()].fixed = true;
                paramInfoMap[traj->getKnotAlp(kidx).data()].fixed = true;
                paramInfoMap[traj->getKnotPos(kidx).data()].fixed = true;
                paramInfoMap[traj->getKnotVel(kidx).data()].fixed = true;
                paramInfoMap[traj->getKnotAcc(kidx).data()].fixed = true;
            }
        }
    }

    void AddMP2KFactors(
        ceres::Problem &problem, GaussianProcessPtr &traj,
        ParamInfoMap &paramInfoMap, FactorMeta &factorMeta,
        double tmin, double tmax)
    {
        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = min(traj->getNumKnots() - 1, usmax.first + 1);

        for (int kidx = kidxmin; kidx < kidxmax; kidx++)
        {
            vector<double *> factor_param_blocks;
            factorMeta.coupled_params.push_back(vector<ParamInfo>());
            
            // Add the parameter blocks
            for (int kidx_ = kidx; kidx_ < kidx + 2; kidx_++)
            {
                factor_param_blocks.push_back(traj->getKnotSO3(kidx_).data());
                factor_param_blocks.push_back(traj->getKnotOmg(kidx_).data());
                factor_param_blocks.push_back(traj->getKnotAlp(kidx_).data());
                factor_param_blocks.push_back(traj->getKnotPos(kidx_).data());
                factor_param_blocks.push_back(traj->getKnotVel(kidx_).data());
                factor_param_blocks.push_back(traj->getKnotAcc(kidx_).data());

                // Record the param info
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotSO3(kidx_).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotOmg(kidx_).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotAlp(kidx_).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotPos(kidx_).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotVel(kidx_).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotAcc(kidx_).data()]);
            }

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(traj->getKnotTime(kidx+1));

            if(use_ceres)
            {
                // Create the factors
                ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
                ceres::CostFunction *cost_function = new GPMotionPriorTwoKnotsFactor(traj->getGPMixerPtr());
                auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
                
                // Record the residual block
                factorMeta.res.push_back(res_block);
            }
        }
    }

    void AddLidarFactors(
        ceres::Problem &problem, GaussianProcessPtr &traj,
        ParamInfoMap &paramInfoMap, FactorMeta &factorMeta,
        const vector<vector<LidarCoef>> &cloudCoef, const vector<lidarFeaIdx> &featuresSelected,
        double tmin, double tmax)
    {
        for(auto lf : featuresSelected)
        {
            const vector<LidarCoef> &Coef = cloudCoef[lf.cidx];
            const LidarCoef &coef = Coef[lf.fidx];

            // Skip if lidar coef is not assigned                 
            auto   us = traj->computeTimeIndex(coef.t);
            int    u  = us.first;
            double s  = us.second;
            
            assert(tmin < traj->getKnotTime(u) || traj->getKnotTime(u+1) < tmax);
            assert(coef.t >= 0);
            assert(traj->TimeInInterval(coef.t, 1e-6));  

            vector<double *> factor_param_blocks;
            factorMeta.coupled_params.push_back(vector<ParamInfo>());
            // factorMeta.coupled_coef.push_back(getEigenPtr(&coef));

            // Add the parameter blocks for rotation
            for (int kidx = u; kidx < u + 2; kidx++)
            {
                factor_param_blocks.push_back(traj->getKnotSO3(kidx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(kidx).data());
                factor_param_blocks.push_back(traj->getKnotAlp(kidx).data());
                factor_param_blocks.push_back(traj->getKnotPos(kidx).data());
                factor_param_blocks.push_back(traj->getKnotVel(kidx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(kidx).data());

                // Record the param info
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotSO3(kidx).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotOmg(kidx).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotAlp(kidx).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotPos(kidx).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotVel(kidx).data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[traj->getKnotAcc(kidx).data()]);
            }

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(coef.t);

            if(use_ceres)
            {
                // double lidar_loss_thres = -1.0;
                ceres::LossFunction *lidar_loss_function = ld_loss_thres == -1 ? NULL : new ceres::HuberLoss(ld_loss_thres);
                ceres::CostFunction *cost_function = new GPPointToPlaneFactor(coef.f, coef.n, lidar_weight*coef.plnrty, traj->getGPMixerPtr(), s);
                auto res = problem.AddResidualBlock(cost_function, lidar_loss_function, factor_param_blocks);

                // Record the residual block
                factorMeta.res.push_back(res);
            }
        }
    }

    void AddGPExtrinsicFactors(
        ceres::Problem &problem, GaussianProcessPtr &trajx, GaussianProcessPtr &trajy, SO3d &R_Lx_Ly, Vec3 &P_Lx_Ly,
        ParamInfoMap &paramInfoMap, FactorMeta &factorMeta,
        double tmin, double tmax)
    {
        GPMixerPtr gpmx = trajx->getGPMixerPtr();
        GPMixerPtr gpmy = trajy->getGPMixerPtr();
        GPMixerPtr gpmextr(new GPMixer(gpmx->getDt(), (xtSigGa*Vec3(1.0, 1.0, 1.0)).asDiagonal(), (xtSigNu*Vec3(1.0, 1.0, 1.0)).asDiagonal()));

        int XTRZ_DENSITY = 1;
        nh->get_parameter("XTRZ_DENSITY", XTRZ_DENSITY);

        for (int kidx = 0; kidx < trajx->getNumKnots() - 2; kidx++)
        {
            if (trajx->getKnotTime(kidx+1) <= tmin || trajx->getKnotTime(kidx) >= tmax)
            {
                // printf("Skipping %f. %f, %f, %f\n", trajx->getKnotTime(kidx+1), tmin, trajx->getKnotTime(kidx), tmax);
                continue;
            }

            if (trajy->getKnotTime(kidx+1) <= tmin || trajy->getKnotTime(kidx) >= tmax)
            {
                // printf("Skipping %f. %f, %f, %f\n", trajy->getKnotTime(kidx+1), tmin, trajy->getKnotTime(kidx), tmax);
                continue;
            }

            for(int i = 0; i < XTRZ_DENSITY; i++)
            {
                // Get the knot time
                double t = trajx->getKnotTime(kidx) + trajx->getDt()/(XTRZ_DENSITY+1)*(i+1);

                // Skip if time is outside the range of the other trajectory
                if (!trajy->TimeInInterval(t))
                    continue;

                pair<int, double> uss, usf;
                uss = trajx->computeTimeIndex(t);
                usf = trajy->computeTimeIndex(t);

                int umins = uss.first;
                int uminf = usf.first;
                double ss = uss.second;
                double sf = usf.second;

                if(   !paramInfoMap.hasParam(trajx->getKnotSO3(umins).data())  
                   || !paramInfoMap.hasParam(trajx->getKnotSO3(umins+1).data())
                   || !paramInfoMap.hasParam(trajx->getKnotSO3(uminf).data())  
                   || !paramInfoMap.hasParam(trajx->getKnotSO3(uminf+1).data())
                   || !paramInfoMap.hasParam(R_Lx_Ly.data())
                   || !paramInfoMap.hasParam(P_Lx_Ly.data())
                )
                continue;

                // Add the parameter blocks
                vector<double *> factor_param_blocks;
                factorMeta.coupled_params.push_back(vector<ParamInfo>());

                for (int idx = umins; idx < umins + 2; idx++)
                {
                    assert(idx < trajx->getNumKnots());

                    // assert(Util::SO3IsValid(trajx->getKnotSO3(idx)));
                    factor_param_blocks.push_back(trajx->getKnotSO3(idx).data());
                    factor_param_blocks.push_back(trajx->getKnotOmg(idx).data());
                    factor_param_blocks.push_back(trajx->getKnotAlp(idx).data());
                    factor_param_blocks.push_back(trajx->getKnotPos(idx).data());
                    factor_param_blocks.push_back(trajx->getKnotVel(idx).data());
                    factor_param_blocks.push_back(trajx->getKnotAcc(idx).data());

                    assert(paramInfoMap.hasParam(trajx->getKnotSO3(idx).data()));
                    assert(paramInfoMap.hasParam(trajx->getKnotOmg(idx).data()));
                    assert(paramInfoMap.hasParam(trajx->getKnotAlp(idx).data()));
                    assert(paramInfoMap.hasParam(trajx->getKnotPos(idx).data()));
                    assert(paramInfoMap.hasParam(trajx->getKnotVel(idx).data()));
                    assert(paramInfoMap.hasParam(trajx->getKnotAcc(idx).data()));

                    factorMeta.coupled_params.back().push_back(paramInfoMap[trajx->getKnotSO3(idx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[trajx->getKnotOmg(idx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[trajx->getKnotAlp(idx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[trajx->getKnotPos(idx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[trajx->getKnotVel(idx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[trajx->getKnotAcc(idx).data()]);
                }

                for (int idx = uminf; idx < uminf + 2; idx++)
                {
                    assert(idx < trajy->getNumKnots());
                    // assert(Util::SO3IsValid(trajy->getKnotSO3(idx)));
                    factor_param_blocks.push_back(trajy->getKnotSO3(idx).data());
                    factor_param_blocks.push_back(trajy->getKnotOmg(idx).data());
                    factor_param_blocks.push_back(trajy->getKnotAlp(idx).data());
                    factor_param_blocks.push_back(trajy->getKnotPos(idx).data());
                    factor_param_blocks.push_back(trajy->getKnotVel(idx).data());
                    factor_param_blocks.push_back(trajy->getKnotAcc(idx).data());

                    assert(paramInfoMap.hasParam(trajy->getKnotSO3(idx).data()));
                    assert(paramInfoMap.hasParam(trajy->getKnotOmg(idx).data()));
                    assert(paramInfoMap.hasParam(trajy->getKnotAlp(idx).data()));
                    assert(paramInfoMap.hasParam(trajy->getKnotPos(idx).data()));
                    assert(paramInfoMap.hasParam(trajy->getKnotVel(idx).data()));
                    assert(paramInfoMap.hasParam(trajy->getKnotAcc(idx).data()));

                    factorMeta.coupled_params.back().push_back(paramInfoMap[trajy->getKnotSO3(idx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[trajy->getKnotOmg(idx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[trajy->getKnotAlp(idx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[trajy->getKnotPos(idx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[trajy->getKnotVel(idx).data()]);
                    factorMeta.coupled_params.back().push_back(paramInfoMap[trajy->getKnotAcc(idx).data()]);
                }

                factor_param_blocks.push_back(R_Lx_Ly.data());
                factor_param_blocks.push_back(P_Lx_Ly.data());
                factorMeta.coupled_params.back().push_back(paramInfoMap[R_Lx_Ly.data()]);
                factorMeta.coupled_params.back().push_back(paramInfoMap[P_Lx_Ly.data()]);

                assert(paramInfoMap.hasParam(R_Lx_Ly.data()));
                assert(paramInfoMap.hasParam(P_Lx_Ly.data()));
                
                // Record the time stamp of the factor
                factorMeta.stamp.push_back(t);

                if(use_ceres)
                {
                    // Create the factors
                    ceres::LossFunction *xtz_loss_function = xt_loss_thres <= 0 ? NULL : new ceres::HuberLoss(xt_loss_thres);
                    ceres::CostFunction *cost_function = new GPExtrinsicFactor(gpmextr, gpmx, gpmy, 1.0, ss, sf);
                    auto res_block = problem.AddResidualBlock(cost_function, NULL, factor_param_blocks);

                    // Save the residual block
                    factorMeta.res.push_back(res_block);
                }
            }
        }
    }

    void AddPriorFactor(
        ceres::Problem &problem, vector<GaussianProcessPtr> &trajs,
        ParamInfoMap &paramInfo, MarginalizationInfoPtr &margInfo, FactorMeta &factorMeta,
        double tmin, double tmax)
    {
        // Check if kept states are still in the param list
        bool all_kept_states_found = true;
        vector<int> missing_param_idx;
        int removed_dims = 0;
        for(int idx = 0; idx < margInfo->keptParamInfo.size(); idx++)
        {
            ParamInfo &param = margInfo->keptParamInfo[idx];
            bool state_found = paramInfoMap.hasParam(param.address);
            // printf("param 0x%8x of tidx %2d kidx %4d of sidx %4d is %sfound in paramInfoMap\n",
            //         param.address, param.tidx, param.kidx, param.sidx, state_found ? "" : "NOT ");
            
            if (!state_found)
            {
                all_kept_states_found = false;
                missing_param_idx.push_back(idx);
                removed_dims += param.delta_size;
            }
        }
        
        if (missing_param_idx.size() != 0) // If some marginalization states are missing, delete the missing states
        {
            // printf("Remove %d params missing from %d keptParamInfos\n", missing_param_idx.size(), margInfo->keptParamInfo.size());
            auto removeElementsByIndices = [](vector<ParamInfo>& vec, const std::vector<int>& indices) -> void
            {
                // Copy indices to a new vector and sort it in descending order
                std::vector<int> sortedIndices(indices);
                std::sort(sortedIndices.rbegin(), sortedIndices.rend());

                // Remove elements based on sorted indices
                for (int index : sortedIndices)
                {
                    if (index >= 0 && index < vec.size())
                        vec.erase(vec.begin() + index);
                    else
                        std::cerr << "Index out of bounds: " << index << std::endl;
                }
            };

            auto removeColOrRow = [](const MatrixXd& matrix, const vector<int>& idxToRemove, int cor) -> MatrixXd // set cor = 0 to remove cols, 1 to remove rows
            {
                MatrixXd matrix_tp = matrix;

                if (cor == 1)
                    matrix_tp.transposeInPlace();

                vector<int> idxToRemove_;
                for(int idx : idxToRemove)
                    if(idx < matrix_tp.cols())
                        idxToRemove_.push_back(idx);

                // for(auto idx : idxToRemove_)
                //     printf("To remove: %d\n", idx);

                // Determine the number of columns to keep
                int idxToKeep = matrix_tp.cols() - idxToRemove_.size();
                if (idxToKeep <= 0)
                    throw std::invalid_argument("All columns (all rows) are removed or invalid number of columns (or rows) to keep");

                // Create a new matrix with the appropriate size
                MatrixXd result(matrix_tp.rows(), idxToKeep);

                // Copy columns that are not in idxToRemove
                int currentCol = 0;
                for (int col = 0; col < matrix_tp.cols(); ++col)
                    if (std::find(idxToRemove_.begin(), idxToRemove_.end(), col) == idxToRemove_.end())
                    {
                        result.col(currentCol) = matrix_tp.col(col);
                        currentCol++;
                    }

                if (cor == 1)
                    result.transposeInPlace();

                return result;
            };

            int cidx = 0;
            vector<int> removed_cidx;
            for(int idx = 0; idx < margInfo->keptParamInfo.size(); idx++)
            {
                int &param_cols = margInfo->keptParamInfo[idx].delta_size;
                if(std::find(missing_param_idx.begin(), missing_param_idx.end(), idx) != missing_param_idx.end())
                    for(int c = 0; c < param_cols; c++)
                    {
                        removed_cidx.push_back(cidx + c);
                        // yolos("%d %d %d\n", cidx, c, removed_cidx.size());
                        // printf("idx: %d. param_cols: %d. cidx: %d. c: %d\n", idx, param_cols, cidx, c);
                    }
                cidx += (margInfo->keptParamInfo[idx].delta_size);
            }

            // Remove the rows and collumns of the marginalization matrices
            margInfo->Hkeep = removeColOrRow(margInfo->Hkeep, removed_cidx, 0);
            margInfo->Hkeep = removeColOrRow(margInfo->Hkeep, removed_cidx, 1);
            margInfo->bkeep = removeColOrRow(margInfo->bkeep, removed_cidx, 1);

            margInfo->Jkeep = removeColOrRow(margInfo->Jkeep, removed_cidx, 0);
            margInfo->Jkeep = removeColOrRow(margInfo->Jkeep, removed_cidx, 1);
            margInfo->rkeep = removeColOrRow(margInfo->rkeep, removed_cidx, 1);

            // printf("Jkeep: %d %d. rkeep: %d %d. Hkeep: %d %d. bkeep: %d %d. ParamPrior: %d. ParamInfo: %d. missing_param_idx: %d\n",
            //         margInfo->Jkeep.rows(), margInfo->Jkeep.cols(),
            //         margInfo->rkeep.rows(), margInfo->rkeep.cols(),
            //         margInfo->Hkeep.rows(), margInfo->Hkeep.cols(),
            //         margInfo->bkeep.rows(), margInfo->bkeep.cols(),
            //         margInfo->keptParamPrior.size(),
            //         margInfo->keptParamInfo.size(),
            //         missing_param_idx.size());

            // Remove the stored priors
            for(auto &param_idx : missing_param_idx)
            {
                // printf("Deleting %d\n", param_idx);
                margInfo->keptParamPrior.erase(margInfo->keptParamInfo[param_idx].address);
            }

            // Remove the unfound states
            removeElementsByIndices(margInfo->keptParamInfo, missing_param_idx);

            // printf("Jkeep: %d %d. rkeep: %d %d. Hkeep: %d %d. bkeep: %d %d. ParamPrior: %d. ParamInfo: %d. missing_param_idx: %d\n",
            //         margInfo->Jkeep.rows(), margInfo->Jkeep.cols(),
            //         margInfo->rkeep.rows(), margInfo->rkeep.cols(),
            //         margInfo->Hkeep.rows(), margInfo->Hkeep.cols(),
            //         margInfo->bkeep.rows(), margInfo->bkeep.cols(),
            //         margInfo->keptParamPrior.size(),
            //         margInfo->keptParamInfo.size(),
            //         missing_param_idx.size());
        }

        if(margInfo->keptParamInfo.size() != 0)
        // Add the factor
        {
            // Record the time stamp of the factor
            factorMeta.stamp.push_back(tmin);

            if(use_ceres)
            {
                MarginalizationFactor* margFactor = new MarginalizationFactor(margInfo);

                // Add the involved parameters blocks
                auto res_block = problem.AddResidualBlock(margFactor, NULL, margInfo->getAllParamBlocks());

                // Save the residual block
                factorMeta.res.push_back(res_block);

                // Add the coupled param
                factorMeta.coupled_params.push_back(margInfo->keptParamInfo);
            }
        }
        else
            printf(KYEL "All kept params in marginalization missing. Please check\n" RESET);
    }

    void EvaluateMP2KFactors(
        vector<GaussianProcessPtr> &trajs,
        ParamInfoMap &paramInfoMap, FactorMeta &factorMeta, MatrixXd &J, VectorXd &r)
    {
        int MP2K_COUNT = factorMeta.size();
        int RES_LSIZE = STATE_DIM;
        int RES_GSIZE = RES_LSIZE*MP2K_COUNT;
        J = MatrixXd(RES_GSIZE, paramInfoMap.XSIZE);
        r = VectorXd(RES_GSIZE, 1);
        J.setZero();
        r.setZero();

        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int ridx = 0; ridx < MP2K_COUNT; ridx++)
        {
            vector<ParamInfo> &coupled_params = factorMeta.coupled_params[ridx];
            GaussianProcessPtr &traj = trajs[coupled_params[0].tidx];
            int &kidx = coupled_params[0].kidx;
            int &xidx = coupled_params[0].xidx;
            
            assert(kidx == coupled_params[1].kidx);

            // Calculate the factor with eigen method
            GPMotionPriorTwoKnotsFactorTMN mpFactor(traj->getGPMixerPtr());
            mpFactor.Evaluate(traj->getKnot(kidx), traj->getKnot(kidx+1));

            int row = ridx*RES_LSIZE;
            int col = xidx;
            J.block<STATE_DIM, 2*STATE_DIM>(row, col) = mpFactor.jacobian;
            r.block<STATE_DIM, 1          >(row, 0)   = mpFactor.residual;
        }
    }

    void EvaluateLidarFactors(
        vector<GaussianProcessPtr> &trajs,
        ParamInfoMap &paramInfoMap, FactorMeta &factorMeta,
        const vector<vector<vector<LidarCoef>>> &cloudCoef, const vector<vector<lidarFeaIdx>> &featuresSelected,
        double tmin, double tmax, MatrixXd &J, VectorXd &r)
    {
        int LDR_COUNT = factorMeta.size();
        int RES_LSIZE = 1;
        int RES_GSIZE = RES_LSIZE*LDR_COUNT;
        J = MatrixXd(RES_GSIZE, paramInfoMap.XSIZE);
        r = VectorXd(RES_GSIZE, 1);
        J.setZero();
        r.setZero();

        for(int lidx = 0; lidx < featuresSelected.size(); lidx++)
        {
            GaussianProcessPtr &traj = trajs[lidx];
            int RES_BASE = lidx == 0 ? 0 : featuresSelected[lidx - 1].size();

            #pragma omp parallel for num_threads(MAX_THREADS)
            for(int lfidx = 0; lfidx < featuresSelected[lidx].size(); lfidx++)
            {
                auto &lf = featuresSelected[lidx][lfidx];
                auto &coef = cloudCoef[lidx][lf.cidx][lf.fidx];

                // Skip if lidar coef is not assigned                 
                auto   us = traj->computeTimeIndex(coef.t);
                int    u  = us.first;
                double s  = us.second;
                
                assert(tmin < traj->getKnotTime(u) || traj->getKnotTime(u+1) < tmax);
                assert(coef.t >= 0);
                assert(traj->TimeInInterval(coef.t, 1e-6));  
                
                // Calculate the factor with eigen method
                GPPointToPlaneFactorTMN ldFactor(coef.f, coef.n, lidar_weight*coef.plnrty, traj->getGPMixerPtr(), s);
                ldFactor.Evaluate(traj->getKnot(u), traj->getKnot(u+1));

                int row = RES_BASE + lfidx*RES_LSIZE;
                int col = paramInfoMap[traj->getKnotSO3(u).data()].xidx;
                J.block<1, 2*STATE_DIM>(row, col) = ldFactor.jacobian;
                r.block<1, 1          >(row, 0)  << ldFactor.residual;
            }
        }
    }

    void EvaluateGPExtrinsicFactors(
        vector<GaussianProcessPtr> &trajs, vector<SO3d> &R_Lx_Ly, vector<Vec3> &P_Lx_Ly,
        ParamInfoMap &paramInfo, FactorMeta &factorMeta,
        MatrixXd &J, VectorXd &r)
    {
        int GPX_COUNT = factorMeta.size();
        int RES_LSIZE = STATE_DIM;
        int RES_GSIZE = RES_LSIZE*GPX_COUNT;
        J = MatrixXd(RES_GSIZE, paramInfoMap.XSIZE);
        r = VectorXd(RES_GSIZE, 1);
        J.setZero();
        r.setZero();
        
        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int fidx = 0; fidx < GPX_COUNT; fidx++)
        {
            double t = factorMeta.stamp[fidx];
            GaussianProcessPtr &trajx = trajs[factorMeta.coupled_params[fidx][0].tidx];
            GaussianProcessPtr &trajy = trajs[factorMeta.coupled_params[fidx][12].tidx];
            int xtzidx = factorMeta.coupled_params[fidx][12].tidx;

            pair<int, double> uss, usf;
            uss = trajx->computeTimeIndex(t);
            usf = trajy->computeTimeIndex(t);

            int umins = uss.first;
            int uminf = usf.first;
            double ss = uss.second;
            double sf = usf.second;

            GPMixerPtr gpmx = trajx->getGPMixerPtr();
            GPMixerPtr gpmy = trajy->getGPMixerPtr();
            GPMixerPtr gpmextr(new GPMixer(gpmx->getDt(), (xtSigGa*Vec3(1.0, 1.0, 1.0)).asDiagonal(), (xtSigNu*Vec3(1.0, 1.0, 1.0)).asDiagonal()));

            // Calculate the factor with eigen method
            GPExtrinsicFactorTMN xtFactor(gpmextr, gpmx, gpmy, 1.0, ss, sf);
            xtFactor.Evaluate(trajx->getKnot(umins), trajx->getKnot(umins+1),
                              trajy->getKnot(uminf), trajy->getKnot(uminf+1), R_Lx_Ly[xtzidx], P_Lx_Ly[xtzidx]);

            // Add the factor to the block
            const int GPX2SIZE = 2*STATE_DIM; const int XTRZSIZE = 6;
            int row  = fidx*RES_LSIZE;
            int cols = paramInfoMap[trajx->getKnotSO3(umins).data()].xidx;
            int colf = paramInfoMap[trajy->getKnotSO3(uminf).data()].xidx;
            int colx = paramInfoMap[R_Lx_Ly[xtzidx].data()].xidx;

            J.block<GPEXT_RES_DIM, 2*STATE_DIM>(row, cols) = xtFactor.jacobian.block<GPEXT_RES_DIM, 2*STATE_DIM>(0, 0);
            J.block<GPEXT_RES_DIM, 2*STATE_DIM>(row, colf) = xtFactor.jacobian.block<GPEXT_RES_DIM, 2*STATE_DIM>(0, GPX2SIZE);
            J.block<GPEXT_RES_DIM, XTRZSIZE   >(row, colx) = xtFactor.jacobian.block<GPEXT_RES_DIM, XTRZSIZE   >(0, GPX2SIZE + GPX2SIZE);
            r.block<GPEXT_RES_DIM, 1          >(row, 0)   << xtFactor.residual;
        }
    }

    void EvaluatePriorFactor(
        ParamInfoMap &paramInfo,
        MarginalizationInfoPtr &margInfo, FactorMeta &factorMeta,
        MatrixXd &J, VectorXd &r)
    {
        int RES_GSIZE = margInfo->Jkeep.rows();
        vector<int> RES_BASE(1, 0);
        for (int aidx = 1; aidx < margInfo->keptParamInfo.size(); aidx++)
            RES_BASE.push_back(RES_BASE.back() + margInfo->keptParamInfo[aidx-1].delta_size);

        assert(RES_BASE.back() + margInfo->keptParamInfo.back().delta_size == RES_GSIZE);

        // Find the size of the problem
        J = MatrixXd(RES_GSIZE, paramInfoMap.XSIZE);
        r = VectorXd(RES_GSIZE, 1);
        J.setZero();
        r.setZero();

        if(margInfo->keptParamInfo.size() != 0)
        {      
            // Calculate the factor with eigen method
            MarginalizationFactorTMN priFactor(margInfo);
            priFactor.Evaluate();

            // Add the marginalization blocks into the global J and r
            {
                int row = 0;
                int XA_LBASE = 0; int XB_LBASE = 0;
                for (int aidx = 0; aidx < margInfo->keptParamInfo.size(); aidx++)
                {
                    ParamInfo &xa = margInfo->keptParamInfo[aidx];
                    int XA_GBASE = paramInfoMap[xa.address].xidx;
                    int row = RES_BASE[aidx];

                    XB_LBASE = 0;
                    for (int bidx = 0; bidx < margInfo->keptParamInfo.size(); bidx++)
                    {
                        ParamInfo &xb = margInfo->keptParamInfo[bidx];
                        int XB_GBASE = paramInfoMap[xb.address].xidx;

                        J.block(row, XB_GBASE, xa.delta_size, xb.delta_size)
                            = priFactor.jacobian.block(XA_LBASE, XB_LBASE, xa.delta_size, xb.delta_size);

                        XB_LBASE += xb.delta_size;
                    }

                    r.block(row, 0, xa.delta_size, 1) = priFactor.residual.block(XA_LBASE, 0, xa.delta_size, 1);
                    XA_LBASE += xa.delta_size;
                }
            }
        }
        else
            printf(KYEL "All kept params in marginalization missing. Please check\n" RESET);
    }
    
    void CheckParamMarginalization(
        double tmin, double tmax, double tmid, ParamInfoMap &paramInfo,
        const vector<FactorMetaPtr> &factorMetas,
        FactorMeta &factorMetaRmvd, FactorMeta &factorMetaRtnd,
        vector<ParamInfo> &removed_params, vector<ParamInfo> &priored_params
    )
    {
        // Insanity check to keep track of all the params
        for(auto &factorMeta : factorMetas)
            for(auto &cpset : factorMeta->coupled_params)
                for(auto &cp : cpset)
                    assert(paramInfoMap.hasParam(cp.address));

        // Determine removed factors
        auto FindRemovedFactors = [&tmid](FactorMeta &factorMeta, FactorMeta &factorMetaRemoved, FactorMeta &factorMetaRetained) -> void
        {
            for(int ridx = 0; ridx < factorMeta.stamp.size(); ridx++)
            {
                // ceres::ResidualBlockId &res = factorMeta.res[ridx];
                // int KC = factorMeta.kidx[ridx].size();
                bool removed = factorMeta.stamp[ridx] <= tmid;

                if (removed)
                {
                    // factorMetaRemoved.knots_coupled = factorMeta.res[ridx].knots_coupled;
                    if (factorMeta.res.size() != 0) factorMetaRemoved.res.push_back(factorMeta.res[ridx]);
                    factorMetaRemoved.coupled_params.push_back(factorMeta.coupled_params[ridx]);
                    factorMetaRemoved.stamp.push_back(factorMeta.stamp[ridx]);

                    // for(int coupling_idx = 0; coupling_idx < KC; coupling_idx++)
                    // {
                    //     int kidx = factorMeta.kidx[ridx][coupling_idx];
                    //     int tidx = factorMeta.tidx[ridx][coupling_idx];
                    //     tk_removed_res.push_back(make_pair(tidx, kidx));
                    //     // printf("Removing knot %d of traj %d, param %d.\n", kidx, tidx, tk2p[tk_removed_res.back()]);
                    // }
                }
                else
                {
                    if (factorMeta.res.size() != 0) factorMetaRetained.res.push_back(factorMeta.res[ridx]);
                    factorMetaRetained.coupled_params.push_back(factorMeta.coupled_params[ridx]);
                    factorMetaRetained.stamp.push_back(factorMeta.stamp[ridx]);
                }
            }
        };

        for(auto &factorMeta : factorMetas)
        {
            FactorMeta factorMetaRmvd_, factorMetaRtnd_;
            FindRemovedFactors(*factorMeta, factorMetaRmvd_, factorMetaRtnd_);
            factorMetaRmvd += factorMetaRmvd_;
            factorMetaRtnd += factorMetaRtnd_;
        }

        // Find the set of params belonging to removed factors
        map<double*, ParamInfo> removed_factors_params;
        for(auto &cpset : factorMetaRmvd.coupled_params)
            for(auto &cp : cpset)
            {
                assert(paramInfoMap.hasParam(cp.address));
                removed_factors_params[cp.address] = (paramInfoMap[cp.address]);
            }

        // Find the set of params belonging to the retained factors
        map<double*, ParamInfo> retained_factors_params;
        for(auto &cpset : factorMetaRtnd.coupled_params)
            for(auto &cp : cpset)
            {
                assert(paramInfoMap.hasParam(cp.address));
                retained_factors_params[cp.address] = (paramInfoMap[cp.address]);
            }

        // Find the intersection of the two sets, which will be the kept parameters
        for(auto &param : removed_factors_params)
        {
            // ParamInfo &param = removed_params[param.first];
            if(retained_factors_params.find(param.first) != retained_factors_params.end())            
                priored_params.push_back(param.second);
            else
                removed_params.push_back(param.second);
        }

        // Compare the params following a hierarchy: traj0 knot < traj1 knots < extrinsics
        auto compareParam = [](const ParamInfo &a, const ParamInfo &b) -> bool
        {
            bool abpidx = a.pidx < b.pidx;

            if (a.tidx == -1 && b.tidx != -1)
            {
                assert(abpidx == false);
                return false;
            }

            if (a.tidx != -1 && b.tidx == -1)
            {
                assert(abpidx == true);
                return true;
            }
            
            if ((a.tidx != -1 && b.tidx != -1) && (a.tidx < b.tidx))
            {
                assert(abpidx == true);
                return true;
            }
            
            if ((a.tidx != -1 && b.tidx != -1) && (a.tidx > b.tidx))
            {
                assert(abpidx == false);
                return false;
            }

            // Including the situation that two knots are 01
            if (a.tidx == b.tidx)
            {
                if (a.kidx == -1 && b.kidx != -1)
                {
                    assert(abpidx == false);
                    return false;
                }

                if (a.kidx != -1 && b.kidx == -1)
                {
                    assert(abpidx == true);
                    return true;
                }

                if ((a.kidx != -1 && b.kidx != -1) && (a.kidx < b.kidx))
                {
                    assert(abpidx == true);
                    return true;
                }

                if ((a.kidx != -1 && b.kidx != -1) && (a.kidx > b.kidx))
                {
                    assert(abpidx == false);
                    return false;
                }

                if (a.kidx == b.kidx)
                {
                    if (a.sidx == -1 && b.sidx != -1)
                    {
                        assert(abpidx == false);
                        return false;
                    }

                    if (a.sidx != -1 && b.sidx == -1)
                    {
                        assert(abpidx == true);
                        return true;
                    }

                    if ((a.sidx != -1 && b.sidx != -1) && (a.sidx < b.sidx))
                    {
                        assert(abpidx == true);
                        return true;
                    }
                    
                    if ((a.sidx != -1 && b.sidx != -1) && (a.sidx > b.sidx))
                    {
                        assert(abpidx == false);
                        return false;
                    }

                    assert(abpidx == false);
                    return false;    
                }
            }
        };

        std::sort(removed_params.begin(), removed_params.end(), compareParam);
        std::sort(priored_params.begin(), priored_params.end(), compareParam);

        int removed_count = removed_params.size();
        int priored_count = priored_params.size();

        assert(priored_count != 0);

        // Just make sure that all of the column index will increase
        {
            int prev_idx = -1;
            for(auto &param : priored_params)
            {
                assert(param.pidx > prev_idx);
                prev_idx = param.pidx;
            }
        }
    }

    void Marginalize(
        ceres::Problem &problem, vector<GaussianProcessPtr> &trajs,
        double tmin, double tmax, double tmid, ParamInfoMap &paramInfoMap,
        const vector<FactorMetaPtr> &factorMetas,
        const VectorXd RESIDUAL, const MatrixXd JACOBIAN)
    {
        FactorMeta factorMetaRmvd, factorMetaRtnd;
        vector<ParamInfo> removed_params, priored_params;
        CheckParamMarginalization(tmin, tmax, tmid, paramInfoMap, factorMetas, factorMetaRmvd, factorMetaRtnd, removed_params, priored_params);

        // Calculate the global H and b
        SMd r = RESIDUAL.sparseView(); r.makeCompressed();
        SMd J = JACOBIAN.sparseView(); J.makeCompressed();
        MatrixXd H = ( J.transpose()*J).toDense();
        VectorXd b = (-J.transpose()*r).toDense();

        // Find the size of the marginalization
        int REMOVED_SIZE = 0; for(auto &param : removed_params) REMOVED_SIZE += param.delta_size;
        int PRIORED_SIZE = 0; for(auto &param : priored_params) PRIORED_SIZE += param.delta_size;

        MatrixXd Hrr(REMOVED_SIZE, REMOVED_SIZE);
        MatrixXd Hrp(REMOVED_SIZE, PRIORED_SIZE);
        MatrixXd Hpr(PRIORED_SIZE, REMOVED_SIZE);
        MatrixXd Hpp(PRIORED_SIZE, PRIORED_SIZE);

        auto CopyParamBlock = [](const vector<ParamInfo> &paramsi, const vector<ParamInfo> &paramsj, MatrixXd &Mtarg, const MatrixXd &Msrc) ->void
        {
            int RBASE = 0;
            for(int piidx = 0; piidx < paramsi.size(); piidx++)
            {
                const ParamInfo &pi = paramsi[piidx];

                int CBASE = 0;
                for(int pjidx = 0; pjidx < paramsj.size(); pjidx++)
                {
                    const ParamInfo &pj = paramsj[pjidx];
                    Mtarg.block(RBASE, CBASE, pi.delta_size, pj.delta_size) = Msrc.block(pi.xidx, pj.xidx, pi.delta_size, pj.delta_size);
                    CBASE += pj.delta_size;
                }

                RBASE += pi.delta_size;
            }
        };

        CopyParamBlock(removed_params, removed_params, Hrr, H);
        CopyParamBlock(removed_params, priored_params, Hrp, H);
        CopyParamBlock(priored_params, removed_params, Hpr, H);
        CopyParamBlock(priored_params, priored_params, Hpp, H);

        VectorXd br(REMOVED_SIZE, 1);
        VectorXd bp(PRIORED_SIZE, 1);

        auto CopyRowBlock = [](const vector<ParamInfo> &params, VectorXd &btarg, const VectorXd &bsrc) -> void
        {
            int RBASE = 0;
            for(int pidx = 0; pidx < params.size(); pidx++)
            {
                ParamInfo param = params[pidx];
                btarg.block(RBASE, 0, param.delta_size, 1) = bsrc.block(param.xidx, 0, param.delta_size, 1);
                RBASE += param.delta_size;
            }
        };

        CopyRowBlock(removed_params, br, b);
        CopyRowBlock(priored_params, bp, b);

        // Create the Schur Complement
        MatrixXd Hrrinv    = Hrr.inverse();
        MatrixXd HprHrrinv = Hpr*Hrrinv;

        MatrixXd Hpriored = Hpp - HprHrrinv*Hrp;
        VectorXd bpriored = bp  - HprHrrinv*br;

        MatrixXd Jpriored; VectorXd rpriored;
        margInfo->HbToJr(Hpriored, bpriored, Jpriored, rpriored);

        // Save the marginalization factors and states
        if (margInfo == nullptr)
            margInfo = MarginalizationInfoPtr(new MarginalizationInfo());

        // Copy the marginalization matrices
        margInfo->Hkeep = Hpriored;
        margInfo->bkeep = bpriored;
        margInfo->Jkeep = Jpriored;
        margInfo->rkeep = rpriored;
        margInfo->keptParamInfo = priored_params;

        // Add the prior of kept params
        margInfo->keptParamPrior.clear();
        for(auto &param : priored_params)
        {
            margInfo->keptParamPrior[param.address] = vector<double>();
            for(int idx = 0; idx < param.param_size; idx++)
                margInfo->keptParamPrior[param.address].push_back(param.address[idx]);
        }

    }

    void SelectFeature(
        vector<GaussianProcessPtr> &trajs, double tmin, double tmax,
        const vector<vector<vector<LidarCoef>>> &SwCloudCoef, vector<vector<lidarFeaIdx>> &featuresSelected)
    {
        // Iterate over each lidar
        int Nlidar = SwCloudCoef.size();
        featuresSelected.resize(Nlidar);

        // Count the total number of extracted factors
        int max_coef_per_lidar = int(double(max_lidarcoefs)/Nlidar);

        // Find feature for each lidar
        for(int lidx = 0; lidx < SwCloudCoef.size(); lidx++)
        {
            GaussianProcessPtr &traj = trajs[lidx];

            // Iterate over each point segment
            const vector<vector<LidarCoef>> &cloudCoef = SwCloudCoef[lidx];
            int WINDOW_SIZE = SwCloudCoef[lidx].size();

            // Counting the number of factors
            int total_factors = 0;
            for(auto &Coef : cloudCoef)
                total_factors += Coef.size();

            // Determine the downsampling rate
            int lidar_ds_rate = (max_lidarcoefs == -1 ? 1 : max(1, (int)std::floor(double(total_factors)/max_coef_per_lidar)));

            // A temporary container of selected features by steps in the sliding window for further downsampling
            int total_selected = 0;
            vector<vector<lidarFeaIdx>> featureBySwStep(WINDOW_SIZE);
            for (int cidx = 0; cidx < WINDOW_SIZE; cidx++)
            {
                const vector<LidarCoef> &Coef = cloudCoef[cidx];
                for (int fidx = 0; fidx < Coef.size(); fidx++)
                {
                    const LidarCoef &coef = Coef[fidx];

                    // Skip if lidar coef is not assigned
                    if (coef.t < 0)
                        continue;

                    if (!traj->TimeInInterval(coef.t, 1e-6))
                        continue;

                    // A lot of factors are calculated but only a subset are used for optimization (time constraint).
                    // By adding a counter we can shuffle the factors so all factors have the chance to be used.
                    if ((cidx + fidx) % lidar_ds_rate != 0)
                        continue;

                    auto   us = traj->computeTimeIndex(coef.t);
                    int    u  = us.first;
                    double s  = us.second;

                    if (traj->getKnotTime(u) <= tmin || traj->getKnotTime(u+1) >= tmax)
                        continue;

                    total_selected++;
                    featureBySwStep[cidx].push_back(lidarFeaIdx(cidx, fidx, total_selected));    
                }
            }

            // If number of lidar feature remains large, randomly select a subset
            if (total_selected > max_coef_per_lidar)
            {
                // Define Fisher-Yates shuffle lambda function
                auto fisherYatesShuffle = [](std::vector<int>& array)
                {
                    std::random_device rd;
                    std::mt19937 gen(rd());

                    for (int i = array.size() - 1; i > 0; --i)
                    {
                        std::uniform_int_distribution<int> distribution(0, i);
                        int j = distribution(gen);
                        std::swap(array[i], array[j]);
                    }
                };

                // How many features each swstep do we need?
                int maxFeaPerSwStep = ceil(double(max_coef_per_lidar) / WINDOW_SIZE);

                // Container for shuffled features
                // vector<vector<lidarFeaIdx>> featureBySwStepShuffled(WINDOW_SIZE);
                vector<vector<int>> shuffledIdx(WINDOW_SIZE);

                #pragma omp parallel for num_threads(MAX_THREADS)
                for(int cidx = 0; cidx < WINDOW_SIZE; cidx++)
                {
                    shuffledIdx[cidx] = vector<int>(featureBySwStep[cidx].size());
                    std::iota(shuffledIdx[cidx].begin(), shuffledIdx[cidx].end(), 0);
                    
                    // Shuffle the feature set a few times
                    fisherYatesShuffle(shuffledIdx[cidx]);
                    fisherYatesShuffle(shuffledIdx[cidx]);
                    fisherYatesShuffle(shuffledIdx[cidx]);
                }

                for(int cidx = 0; cidx < WINDOW_SIZE; cidx++)
                    for(int idx = 0; idx < min(maxFeaPerSwStep, (int)featureBySwStep[cidx].size()); idx++)
                        featuresSelected[lidx].push_back(featureBySwStep[cidx][shuffledIdx[cidx][idx]]);
            }
            else
            {
                for(int cidx = 0; cidx < WINDOW_SIZE; cidx++)
                    for(int idx = 0; idx < featureBySwStep[cidx].size(); idx++)
                        featuresSelected[lidx].push_back(featureBySwStep[cidx][idx]);
            }
        }

    }

    void Evaluate(
        int inner_iter, int outer_iter, vector<GaussianProcessPtr> &trajs,
        double tmin, double tmax, double tmid,
        const vector<vector<vector<LidarCoef>>> &cloudCoef, const vector<vector<lidarFeaIdx>> &featuresSelected,
        bool prepare_marginalization,
        OptReport &report)
    {
        TicToc tt_build;

        int Nlidar = trajs.size();

        static bool traj_sufficient_length = false;
        auto pose_tmin = trajs[0]->pose(tmin);
        auto pose_tmax = trajs[0]->pose(tmax);
        auto trans = (pose_tmin.inverse()*pose_tmax).translation();
        if (trans.norm() > 0.2)
            traj_sufficient_length = true;

        // Build the problem

        // Ceres problem
        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;

        // Set up the ceres problem
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = MAX_THREADS;
        options.max_num_iterations = max_ceres_iter;

        // Documenting the parameter blocks
        paramInfoMap.clear();
        // Add the parameter blocks
        {
            // Add the parameter blocks for rotation
            for(int tidx = 0; tidx < trajs.size(); tidx++)
                AddTrajParams(problem, trajs, tidx, paramInfoMap, tmin, tmax, tmid);

            // Only add extrinsic if there are multiple trajectories
            if (trajs.size() > 1)
                for(int lidx = 1; lidx < Nlidar; lidx++)
                    {
                        // printf("Adding %x, %x extrinsic params\n", R_Lx_Ly[lidx].data(), P_Lx_Ly[lidx].data());
                        // Add the extrinsic params
                        problem.AddParameterBlock(R_Lx_Ly[lidx].data(), 4, new GPSO3dLocalParameterization());
                        problem.AddParameterBlock(P_Lx_Ly[lidx].data(), 3);
                        paramInfoMap.insert(R_Lx_Ly[lidx].data(), ParamInfo(R_Lx_Ly[lidx].data(), getEigenPtr(R_Lx_Ly[lidx]), ParamType::SO3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 0));
                        paramInfoMap.insert(P_Lx_Ly[lidx].data(), ParamInfo(P_Lx_Ly[lidx].data(), getEigenPtr(P_Lx_Ly[lidx]), ParamType::RV3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1));
                    }

            // Sanity check
            for(auto &param_ : paramInfoMap.params_info)
            {
                ParamInfo param = param_.second;

                int tidx = param.tidx;
                int kidx = param.kidx;
                int sidx = param.sidx;

                if(param.tidx != -1 && param.kidx != -1)
                {
                    switch(sidx)
                    {
                        case 0:
                            assert(param.address == trajs[tidx]->getKnotSO3(kidx).data());
                            break;
                        case 1:
                            assert(param.address == trajs[tidx]->getKnotOmg(kidx).data());
                            break;
                        case 2:
                            assert(param.address == trajs[tidx]->getKnotAlp(kidx).data());
                            break;
                        case 3:
                            assert(param.address == trajs[tidx]->getKnotPos(kidx).data());
                            break;
                        case 4:
                            assert(param.address == trajs[tidx]->getKnotVel(kidx).data());
                            break;
                        case 5:
                            assert(param.address == trajs[tidx]->getKnotAcc(kidx).data());
                            break;
                        default:
                            printf("Unrecognized param block! %d, %d, %d\n", tidx, kidx, sidx);
                            break;
                    }
                }
                else
                {
                    if(sidx == 0)
                    {
                        bool found = false;
                        for(int lidx = 0; lidx < Nlidar; lidx++)
                            found = found || (param.address == R_Lx_Ly[lidx].data());
                        assert(found);    
                    }

                    if(sidx == 1)
                    {
                        bool found = false;
                        for(int lidx = 0; lidx < Nlidar; lidx++)
                            found = found || (param.address == P_Lx_Ly[lidx].data());
                        assert(found);    
                    }
                }
            }
        }

        // Sample the trajectories before optimization
        vector <GPState<double>> gpX0;
        for(auto &traj : trajs)
            gpX0.push_back(traj->getStateAt(tmax));

        // Calculating the downsampling rate for lidar factors
        vector<int> lidar_factor_ds(trajs.size(), 0);
        vector<int> total_lidar_factors(trajs.size(), 0);
        for(int tidx = 0; tidx < trajs.size(); tidx++)
            for(int cidx = 0; cidx < cloudCoef[tidx].size(); cidx++)
                total_lidar_factors[tidx] += cloudCoef[tidx][cidx].size();
        for(int tidx = 0; tidx < trajs.size(); tidx++)
        {
            lidar_factor_ds[tidx] = max(int(std::ceil(total_lidar_factors[tidx] / max_lidarcoefs)), 1);
            // printf("lidar_factor_ds[%d]: %d\n", tidx, lidar_factor_ds[tidx]);
        }

        TicToc tt_addmp2k;
        // Add the motion prior factor
        FactorMeta factorMetaMp2k; MatrixXd Jmp2k; VectorXd rmp2k;
        double cost_mp2k_init = -1, cost_mp2k_final = -1;
        for(int tidx = 0; tidx < trajs.size(); tidx++)
            AddMP2KFactors(problem, trajs[tidx], paramInfoMap, factorMetaMp2k, tmin, tmax);
        // RINFO("tt_addmp2k: %f\n", tt_addmp2k.Toc());

        {
            TicToc tt_mp2k;
            EvaluateMP2KFactors(trajs, paramInfoMap, factorMetaMp2k, Jmp2k, rmp2k);
            tt_mp2k.Toc();

            // TicToc tt_mp2k_ceres;
            // VectorXd rmp2k_ceres; MatrixXd Jmp2k_ceres;
            // FindJrByCeres(problem, factorMetaMp2k, cost_mp2k_init, rmp2k_ceres, Jmp2k_ceres);
            // tt_mp2k_ceres.Toc();

            // RINFO("rmp2k_ceres: %d x %d. rmp2k: %d x %d. Jmp2k_ceres: %d x %d. Jmp2k: %d x %d. tt_mp2k: %f. tt_mp2k_ceres: %f.\n",
            //        rmp2k_ceres.rows(), rmp2k_ceres.cols(), rmp2k.rows(), rmp2k.cols(),
            //        Jmp2k_ceres.rows(), Jmp2k_ceres.cols(), Jmp2k.rows(), Jmp2k.cols(),
            //        tt_mp2k.GetLastStop(), tt_mp2k_ceres.GetLastStop());

            // RINFO("rmp2k error: %f. Max value: %f.\n", (rmp2k - rmp2k_ceres).norm(), rmp2k.cwiseAbs().maxCoeff());
            // RINFO("Jmp2k error: %f\n", (Jmp2k - Jmp2k_ceres).cwiseAbs().maxCoeff());
        }

        TicToc tt_addldr;
        // Add the lidar factors
        FactorMeta factorMetaLidar; MatrixXd Jldr; VectorXd rldr;
        double cost_lidar_init = -1; double cost_lidar_final = -1;
        for(int tidx = 0; tidx < trajs.size(); tidx++)
            AddLidarFactors(problem, trajs[tidx], paramInfoMap, factorMetaLidar, cloudCoef[tidx], featuresSelected[tidx], tmin, tmax);
        // RINFO("tt_addldr: %f\n", tt_addldr.Toc());

        {
            TicToc tt_ldr;
            EvaluateLidarFactors(trajs, paramInfoMap, factorMetaLidar, cloudCoef, featuresSelected, tmin, tmax, Jldr, rldr);
            tt_ldr.Toc();

            // TicToc tt_ldr_ceres;
            // VectorXd rldr_ceres; MatrixXd Jldr_ceres;
            // FindJrByCeres(problem, factorMetaLidar, cost_lidar_init, rldr_ceres, Jldr_ceres);
            // tt_ldr_ceres.Toc();

            // RINFO("rldr_ceres: %d x %d. rldr: %d x %d. Jldr_ceres: %d x %d. Jldr: %d x %d. tt_ldr: %f. tt_ldr_ceres: %f.\n",
            //        rldr_ceres.rows(), rldr_ceres.cols(), rldr.rows(), rldr.cols(),
            //        Jldr_ceres.rows(), Jldr_ceres.cols(), Jldr.rows(), Jldr.cols(),
            //        tt_ldr.GetLastStop(), tt_ldr_ceres.GetLastStop());
            // RINFO("rldr error: %f. Max value: %f.\n", (rldr - rldr_ceres).norm(), rldr.cwiseAbs().maxCoeff());
            // RINFO("Jldr error: %f\n", (Jldr - Jldr_ceres).cwiseAbs().maxCoeff());
        }

        TicToc tt_addgpx;
        // Add the extrinsics factors
        FactorMeta factorMetaGpx; MatrixXd Jgpx; VectorXd rgpx;
        double cost_gpx_init = -1; double cost_gpx_final = -1;
        // Check if each trajectory is sufficiently long
        for(int tidxx = 0; tidxx < trajs.size(); tidxx++)
            for(int tidxy = tidxx+1; tidxy < trajs.size(); tidxy++)
                AddGPExtrinsicFactors(problem, trajs[tidxx], trajs[tidxy], R_Lx_Ly[tidxy], P_Lx_Ly[tidxy], paramInfoMap, factorMetaGpx, tmin, tmax);
        // RINFO("tt_addgpx: %f\n", tt_addgpx.Toc());

        {
            TicToc tt_gpx;
            EvaluateGPExtrinsicFactors(trajs, R_Lx_Ly, P_Lx_Ly, paramInfoMap, factorMetaGpx, Jgpx, rgpx);
            tt_gpx.Toc();

            // TicToc tt_gpx_ceres;
            // VectorXd rgpx_ceres; MatrixXd Jgpx_ceres;
            // FindJrByCeres(problem, factorMetaGpx, cost_gpx_init, rgpx_ceres, Jgpx_ceres);
            // tt_gpx_ceres.Toc();

            // RINFO("rgpx_ceres: %d x %d. rgpx: %d x %d. Jgpx_ceres: %d x %d. Jgpx: %d x %d. tt_gpx: %f. tt_gpx_ceres: %f.\n",
            //        rgpx_ceres.rows(), rgpx_ceres.cols(), rgpx.rows(), rgpx.cols(),
            //        Jgpx_ceres.rows(), Jgpx_ceres.cols(), Jgpx.rows(), Jgpx.cols(),
            //        tt_gpx.GetLastStop(), tt_gpx_ceres.GetLastStop());
            // RINFO("rgpx error: %f. Max value: %f.\n", (rgpx - rgpx_ceres).norm(), rgpx.cwiseAbs().maxCoeff());
            // RINFO("Jgpx error: %f\n", (Jgpx - Jgpx_ceres).cwiseAbs().maxCoeff());
        }

        // Add the prior factor
        FactorMeta factorMetaPrior; MatrixXd Jpri; VectorXd rpri;
        double cost_prior_init = -1; double cost_prior_final = -1;
        if (margInfo != NULL && fuse_marg)
        {
            TicToc tt_pri;
            AddPriorFactor(problem, trajs, paramInfoMap, margInfo, factorMetaPrior, tmin, tmax);
            // RINFO("tt_pri: %f\n", tt_pri.Toc());

            // Cross check the jacobian
            {
                TicToc tt_pri;
                EvaluatePriorFactor(paramInfoMap, margInfo, factorMetaPrior, Jpri, rpri);
                tt_pri.Toc();

                // TicToc tt_pri_ceres;
                // VectorXd rpri_ceres; MatrixXd Jpri_ceres;
                // FindJrByCeres(problem, factorMetaPrior, cost_prior_init, rpri_ceres, Jpri_ceres);
                // tt_pri_ceres.Toc();

                // RINFO("rpri_ceres: %d x %d. rpri: %d x %d. Jpri_ceres: %d x %d. Jpri: %d x %d. tt_pri: %f, tt_pri_ceres: %f.\n",
                //        rpri_ceres.rows(), rpri_ceres.cols(), rpri.rows(), rpri.cols(),
                //        Jpri_ceres.rows(), Jpri_ceres.cols(), Jpri.rows(), Jpri.cols(),
                //        tt_pri.GetLastStop(), tt_pri_ceres.GetLastStop());

                // RINFO("rpri error: %f. Max value: %f.\n", (rpri - rpri_ceres).norm(), rpri.cwiseAbs().maxCoeff());
                // RINFO("Jpri error: %f\n", (Jpri - Jpri_ceres).cwiseAbs().maxCoeff());
            }
        }

        tt_build.Toc();

        TicToc tt_slv;

        // Find the initial cost
        // if(compute_cost)
        // {
        //     Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_init,  problem);
        //     Util::ComputeCeresCost(factorMetaLidar.res, cost_lidar_init, problem);
        //     Util::ComputeCeresCost(factorMetaGpx.res,   cost_gpx_init,   problem);
        //     Util::ComputeCeresCost(factorMetaPrior.res, cost_prior_init, problem);
        // }

        if(use_ceres)
            ceres::Solve(options, &problem, &summary);
        else
        {
            int &XSIZE = paramInfoMap.XSIZE;

            // Solve using solver and LM method
            SMd H(XSIZE, XSIZE); H.setZero();
            SMd b(XSIZE, 1); b.setZero();

            auto AddrJToHb = [](const VectorXd &r, const MatrixXd &J, SMd &H, SMd &b)
            {
                if (r.rows() == 0 || J.rows() == 0)
                    return;

                SMd rsparse = r.sparseView(); rsparse.makeCompressed();
                SMd Jsparse = J.sparseView(); Jsparse.makeCompressed();    
                SMd Jtp = Jsparse.transpose();

                if (b.size() == 0)
                {
                    H =  Jtp*Jsparse;
                    b = -Jtp*rsparse;
                }
                else
                {
                    H +=  Jtp*Jsparse;
                    b += -Jtp*rsparse;
                }
            };

            AddrJToHb(rmp2k, Jmp2k, H, b);
            AddrJToHb(rldr,  Jldr,  H, b);
            AddrJToHb(rgpx,  Jgpx,  H, b);
            AddrJToHb(rpri,  Jpri,  H, b);

            double lambda = 0.01; bool solver_failed = true;
            SMd I(H.cols(), H.cols()); I.setIdentity();

            Eigen::SparseLU<SMd> solver;
            SMd S = H + lambda*I;
            solver.analyzePattern(S);
            solver.factorize(S);
            solver_failed = solver.info() != Eigen::Success;
            MatrixXd dX = solver.solve(b);

            // Cap the change
            if (dX.norm() > 0.1)
                dX = dX / dX.norm();

            // Update the trajectory
            for(auto &param : paramInfoMap.params_info)
            {
                int &xidx = param.second.xidx;
                int &tidx = param.second.tidx;
                int &kidx = param.second.kidx;
                int &sidx = param.second.sidx;

                ParamType &type = param.second.type;
                VectorXd dx = dX.block(xidx, 0, param.second.delta_size, 1);
                if (type == ParamType::SO3)
                {
                    SO3d &xr = *static_pointer_cast<SO3d>(param.second.ptr);
                    xr = xr*SO3d::exp(dx);
                }
                else if (type == ParamType::RV3)
                {
                    Vec3 &xv = *static_pointer_cast<Vec3>(param.second.ptr);
                    xv += dx;
                }
                else
                {
                    RINFO(KRED"Unexpected state type %d!"RESET, type);
                    exit(-1);
                }

                if (tidx != -1 && kidx != -1)
                {
                    // if (type == ParamType::SO3)
                    // {
                    //     SO3d so3a = *static_pointer_cast<SO3d>(param.second.ptr);
                    //     SO3d so3b = trajs[tidx]->getKnotSO3(kidx);
                    //     SO3d so3c = *(trajs[tidx]->getKnotSO3Ptr(kidx));
                    //     SO3d so3d = *static_pointer_cast<SO3d>(trajs[tidx]->getKnotSO3Ptr(kidx));
                    //     // Vector3d ang = (so3a.inverse()*so3b).log().norm();
                    //     RINFO("Traj %d. Knot %d, SO3. A-B: %f. A-C: %f. A-D: %f. %s. Hex: %p. %p, %p.",
                    //            tidx, kidx,
                    //            (so3a.inverse()*so3b).log().norm(),
                    //            (so3a.inverse()*so3c).log().norm(),
                    //            (so3a.inverse()*so3d).log().norm(),
                    //            param.second.ptr == trajs[tidx]->getKnotSO3Ptr(kidx) ? "true" : "false",
                    //            param.second.ptr, trajs[tidx]->getKnotSO3Ptr(kidx), trajs[tidx]->getKnotSO3Ptr(kidx));
                    // }
                    // else
                    // if (type == ParamType::RV3 && sidx == 3)
                    // {
                    //     Vec3 veca = *static_pointer_cast<Vec3>(param.second.ptr);
                    //     Vec3 vecb = trajs[tidx]->getKnotPos(kidx);
                    //     Vec3 vecc = *(trajs[tidx]->getKnotPosPtr(kidx));
                    //     Vec3 vecd = *static_pointer_cast<Vec3>(trajs[tidx]->getKnotPosPtr(kidx));
                    //     // Vector3d ang = (so3a.inverse()*so3b).log().norm();
                    //     RINFO("Traj %d. Knot %d, RV3 SIDX %d. A-B: %f. A-C: %f. A-D: %f. C-D: %f. %s. Hex: %p. %p, %p. Ptr: %p, %p, %p",
                    //            tidx, kidx, sidx,
                    //            (veca - vecb).norm(),
                    //            (veca - vecc).norm(),
                    //            (veca - vecd).norm(),
                    //            (vecc - vecd).norm(),
                    //            param.second.ptr == getEigenPtr(trajs[tidx]->getKnotPos(kidx)) ? "true" : "false",
                    //            param.second.ptr, trajs[tidx]->getKnotPosPtr(kidx), trajs[tidx]->getKnotPosPtr(kidx),
                    //            param.second.ptr.get(), trajs[tidx]->getKnotPosPtr(kidx).get(), trajs[tidx]->getKnotPosPtr(kidx).get());
                    // }
                    // else
                    // {
                    //     Vec3 YOLO(0.57, 0.43, 0.91);
                    //     std::shared_ptr<Vec3> veca = std::shared_ptr<Vec3>(&YOLO, [](Vec3 *ptr){});
                    //     std::shared_ptr<Vec3> vecb = std::shared_ptr<Vec3>(&YOLO, [](Vec3 *ptr){});
                    //     std::shared_ptr<Vec3> vecc = std::shared_ptr<Vec3>(&YOLO, [](Vec3 *ptr){});
                    //     RINFO("YOLO. A-B: %f. A-C: %f. B-C: %f. %s. Hex: %p. %p, %p. Ptr: %p, %p, %p.",
                    //         (*veca - *vecb).norm(),
                    //         (*veca - *vecc).norm(),
                    //         (*vecb - *vecc).norm(),
                    //         veca == vecb ? "true" : "false",
                    //         veca,       vecb,       vecc,
                    //         veca.get(), vecb.get(), vecc.get());
                    // }
                }
            }
        }

        // if(compute_cost)
        // {
        //     Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_final,  problem);
        //     Util::ComputeCeresCost(factorMetaLidar.res, cost_lidar_final, problem);
        //     Util::ComputeCeresCost(factorMetaGpx.res,   cost_gpx_final,   problem);
        //     Util::ComputeCeresCost(factorMetaPrior.res, cost_prior_final, problem);
        // }

        // Determine the factors to remove
        if (prepare_marginalization && fuse_marg)
        {
            vector<FactorMetaPtr> factorMetas;
            factorMetas.push_back(getEigenPtr(factorMetaMp2k));
            factorMetas.push_back(getEigenPtr(factorMetaLidar));
            factorMetas.push_back(getEigenPtr(factorMetaGpx));
            factorMetaPrior.res.size() != 0 ? factorMetas.push_back(getEigenPtr(factorMetaPrior)) : (void)0;

            VectorXd RESIDUAL; MatrixXd JACOBIAN;
            if(use_ceres)
            {
                // Make all parameter block variables
                std::vector<double*> parameter_blocks;
                problem.GetParameterBlocks(&parameter_blocks);
                for (auto &paramblock : parameter_blocks)
                    problem.SetParameterBlockVariable(paramblock);

                FactorMeta factorMeta;
                ceres::Problem::EvaluateOptions e_option;
                for(auto &fm : factorMetas)
                    factorMeta += *fm;
                
                double cost;
                FindJrByCeres(problem, factorMeta, cost, RESIDUAL, JACOBIAN);
            }
            else
            {
                EvaluateMP2KFactors(trajs, paramInfoMap, factorMetaMp2k, Jmp2k, rmp2k);
                EvaluateLidarFactors(trajs, paramInfoMap, factorMetaLidar, cloudCoef, featuresSelected, tmin, tmax, Jldr, rldr);
                EvaluateGPExtrinsicFactors(trajs, R_Lx_Ly, P_Lx_Ly, paramInfoMap, factorMetaGpx, Jgpx, rgpx);
                if(fuse_marg && margInfo != NULL)
                    EvaluatePriorFactor(paramInfoMap, margInfo, factorMetaPrior, Jpri, rpri);

                int RSIZE = rmp2k.rows() + rldr.rows() + rgpx.rows() + rpri.rows();
                RESIDUAL = VectorXd(RSIZE, 1);
                JACOBIAN = MatrixXd(RSIZE, paramInfoMap.XSIZE);

                if(fuse_marg && rpri.rows()!=0)
                {
                    JACOBIAN << Jmp2k, Jldr, Jgpx, Jpri;
                    RESIDUAL << rmp2k, rldr, rgpx, rpri;
                }
                else if(fuse_marg && rpri.rows()==0)
                {

                    JACOBIAN << Jmp2k, Jldr, Jgpx;
                    RESIDUAL << rmp2k, rldr, rgpx;
                }
                else
                    RINFO(KRED "ERROR: Marginalization not possible!\n" RESET);
            }

            Marginalize(problem, trajs, tmin, tmax, tmid, paramInfoMap, factorMetas, RESIDUAL, JACOBIAN);
            report.marginalization_done = true;
        }
        else
        {
            report.marginalization_done = false;
        }

        tt_slv.Toc();

        // Sample the trajectories before optimization
        vector <GPState<double>> gpXt;
        for(auto &traj : trajs)
            gpXt.push_back(traj->getStateAt(tmax));

        // Put information to the report
        report.ceres_iterations  = summary.iterations.size();
        report.tictocs["t_ceres_build"] = tt_build.GetLastStop();
        report.tictocs["t_ceres_solve"] = tt_slv.GetLastStop();
        report.factors["MP2K"]   = factorMetaMp2k.size();
        report.factors["GPXTRZ"] = factorMetaGpx.size();
        report.factors["LIDAR"]  = factorMetaLidar.size();
        report.factors["PRIOR"]  = factorMetaPrior.size();
        report.costs["J0"]       = summary.initial_cost;
        report.costs["JK"]       = summary.final_cost;
        report.costs["MP2K0"]    = cost_mp2k_init;
        report.costs["MP2KK"]    = cost_mp2k_final;
        report.costs["GPXTRZ0"]  = cost_gpx_init;
        report.costs["GPXTRZK"]  = cost_gpx_final;
        report.costs["LIDAR0"]   = cost_lidar_init;
        report.costs["LIDARK"]   = cost_lidar_final;
        report.costs["PRIOR0"]   = cost_prior_init;
        report.costs["PRIORK"]   = cost_prior_final;
        report.X0 = gpX0;
        report.Xt = gpXt;
    } 

    SE3d GetExtrinsics(int lidx)
    {
        return SE3d(R_Lx_Ly[lidx], P_Lx_Ly[lidx]);
    }

    void Reset()
    {
        margInfo.reset();
    }
};

typedef std::shared_ptr<MLCME> MLCMEPtr;
