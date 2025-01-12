#pragma once

#include <Eigen/Sparse>

#include "GaussNewtonUtilities.hpp"

#include "factor/GPTDOAFactor.h"
#include "factor/GPIMUFactor.h"
#include "factor/GPTWRFactor.h"
#include "factor/GPPoseFactor.h"
// #include "factor/GPIMUFactorAutodiff.h"
#include "factor/GPMotionPriorTwoKnotsFactorUI.h"
// #include "factor/GPMotionPriorTwoKnotsFactorAutodiff.h"

#include "geometry_msgs/TransformStamped.h"
#include "nlink_parser/LinktrackNodeframe3.h"

typedef sensor_msgs::Imu ImuMsg;
typedef ImuMsg::ConstPtr ImuMsgPtr;
typedef nlink_parser::LinktrackNodeframe3 TwrMsg;
typedef TwrMsg::ConstPtr TwrMsgPtr;
typedef geometry_msgs::TransformStamped GtrMsg;
typedef geometry_msgs::TransformStamped::ConstPtr GtrMsgPtr;

class GPNL
{

private:
    // Node handle to get information needed
    NodeHandlePtr nh;

    // Map of traj-kidx and parameter id
    map<pair<int, int>, int> tk2p;
    map<double *, ParamInfo> paramInfoMap;
    MarginalizationInfoPtr margInfo;

public:
    // Destructor
    ~GPNL() {};

    // Constructor
    GPNL(NodeHandlePtr &nh_) : nh(nh_) {};

    void AddTrajParams(ceres::Problem &problem,
                       GaussianProcessPtr &traj, int tidx,
                       map<double *, ParamInfo> &paramInfoMap,
                       double tmin, double tmax, double tmid)
    {
        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = usmax.first + 1;

        // Create local parameterization for so3
        ceres::LocalParameterization *so3parameterization = new GPSO3dLocalParameterization();

        int pidx = -1;

        for (int kidx = 0; kidx < traj->getNumKnots(); kidx++)
        {
            if (kidx < kidxmin || kidx > kidxmax)
                continue;

            problem.AddParameterBlock(traj->getKnotSO3(kidx).data(), 4, so3parameterization);
            problem.AddParameterBlock(traj->getKnotOmg(kidx).data(), 3);
            problem.AddParameterBlock(traj->getKnotAlp(kidx).data(), 3);
            problem.AddParameterBlock(traj->getKnotPos(kidx).data(), 3);
            problem.AddParameterBlock(traj->getKnotVel(kidx).data(), 3);
            problem.AddParameterBlock(traj->getKnotAcc(kidx).data(), 3);

            // Log down the information of the params
            paramInfoMap.insert(make_pair(traj->getKnotSO3(kidx).data(), ParamInfo(traj->getKnotSO3(kidx).data(), ParamType::SO3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 0)));
            paramInfoMap.insert(make_pair(traj->getKnotOmg(kidx).data(), ParamInfo(traj->getKnotOmg(kidx).data(), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 1)));
            paramInfoMap.insert(make_pair(traj->getKnotAlp(kidx).data(), ParamInfo(traj->getKnotAlp(kidx).data(), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 2)));
            paramInfoMap.insert(make_pair(traj->getKnotPos(kidx).data(), ParamInfo(traj->getKnotPos(kidx).data(), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 3)));
            paramInfoMap.insert(make_pair(traj->getKnotVel(kidx).data(), ParamInfo(traj->getKnotVel(kidx).data(), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 4)));
            paramInfoMap.insert(make_pair(traj->getKnotAcc(kidx).data(), ParamInfo(traj->getKnotAcc(kidx).data(), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 5)));
        }
    }

    void AddMP2KFactorsUI(
        ceres::Problem &problem, GaussianProcessPtr &traj,
        map<double *, ParamInfo> &paramInfoMap, FactorMeta &factorMeta,
        double tmin, double tmax, double mp_loss_thres)
    {
        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = usmax.first + 1;
        // Add the GP factors based on knot difference
        for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
        {
            if (kidx < kidxmin || kidx + 1 > kidxmax)
            {
                continue;
            }
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

            // Create the factors
            ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
            ceres::CostFunction *cost_function = new GPMotionPriorTwoKnotsFactorUI(traj->getGPMixerPtr());
            auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);

            // Record the residual block
            factorMeta.res.push_back(res_block);

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(traj->getKnotTime(kidx + 1));
        }
    }

    void AddPriorFactor(ceres::Problem &problem, GaussianProcessPtr &traj, FactorMeta &factorMeta, double tmin, double tmax)
    {
        // Check if kept states are still in the param list
        bool kept_state_present = true;
        vector<int> missing_param_idx;
        int removed_dims = 0;
        for (int idx = 0; idx < margInfo->keptParamInfo.size(); idx++)
        {
            ParamInfo &param = margInfo->keptParamInfo[idx];
            bool state_found = (paramInfoMap.find(param.address) != paramInfoMap.end());
            // printf("param 0x%8x of tidx %2d kidx %4d of sidx %4d is %sfound in paramInfoMap\n",
            // param.address, param.tidx, param.kidx, param.sidx, state_found ? "" : "NOT ");

            if (!state_found)
            {
                kept_state_present = false;
                missing_param_idx.push_back(idx);
                removed_dims += param.delta_size;
            }
        }

        // If all marginalized states are found, add the marginalized factor
        if (kept_state_present)
        {
            MarginalizationFactor *margFactor = new MarginalizationFactor(margInfo);

            // Add the involved parameters blocks
            auto res_block = problem.AddResidualBlock(margFactor, NULL, margInfo->getAllParamBlocks());
            // Save the residual block
            factorMeta.res.push_back(res_block);

            // Add the coupled param
            factorMeta.coupled_params.push_back(margInfo->keptParamInfo);

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(tmin);
        }
        else if (missing_param_idx.size() <= margInfo->keptParamInfo.size()) // If some marginalization states are missing, delete the missing states
        {
            // printf("Remove %d params missing from %d keptParamInfos\n", missing_param_idx.size(), margInfo->keptParamInfo.size());
            auto removeElementsByIndices = [](vector<ParamInfo> &vec, const std::vector<int> &indices) -> void
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

            auto removeColOrRow = [](const MatrixXd &matrix, const vector<int> &idxToRemove, int cor) -> MatrixXd // set cor = 0 to remove cols, 1 to remove rows
            {
                MatrixXd matrix_tp = matrix;

                if (cor == 1)
                    matrix_tp.transposeInPlace();

                vector<int> idxToRemove_;
                for (int idx : idxToRemove)
                    if (idx < matrix_tp.cols())
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
            for (int idx = 0; idx < margInfo->keptParamInfo.size(); idx++)
            {

                int &param_cols = margInfo->keptParamInfo[idx].delta_size;
                if (std::find(missing_param_idx.begin(), missing_param_idx.end(), idx) != missing_param_idx.end())
                    for (int c = 0; c < param_cols; c++)
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
            for (auto &param_idx : missing_param_idx)
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

            // Add the factor
            {
                MarginalizationFactor *margFactor = new MarginalizationFactor(margInfo);

                // Add the involved parameters blocks
                auto res_block = problem.AddResidualBlock(margFactor, NULL, margInfo->getAllParamBlocks());

                // Save the residual block
                factorMeta.res.push_back(res_block);

                // Add the coupled param
                factorMeta.coupled_params.push_back(margInfo->keptParamInfo);

                // Record the time stamp of the factor
                factorMeta.stamp.push_back(tmin);
            }
        }
        else
            printf(KYEL "All kept params in marginalization missing. Please check\n" RESET);
    }

    void AddTwrFactors(
        ceres::Problem &problem, GaussianProcessPtr &traj,
        map<double *, ParamInfo> &paramInfo, FactorMeta &factorMeta,
        const deque<TwrMsgPtr> &twrData, const vector<Vector3d> &anc_pos, const vector<Vector3d> &tag_pos,
        double tmin, double tmax, double wTwr, double twr_loss_thres)
    {
        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = usmax.first + 1;

        for (auto &twr : twrData)
        {
            double ts = twr->header.stamp.toSec();

            if (!traj->TimeInInterval(ts, 1e-6))
                continue;

            auto us = traj->computeTimeIndex(ts);
            int u = us.first;
            double s = us.second;

            if (u < kidxmin || u + 1 > kidxmax)
                continue;

            for (int nodeidx = 0; nodeidx < twr->nodes.size(); nodeidx++)
            {
                int tidx = twr->id;
                int aidx = twr->nodes[nodeidx].id;

                Vector3d p_W_a = anc_pos[aidx];
                Vector3d p_B_t = tag_pos[tidx];
                double range = twr->nodes[nodeidx].dis;

                vector<double *> factor_param_blocks;
                factorMeta.coupled_params.push_back(vector<ParamInfo>());
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
                ceres::LossFunction *twr_loss_function = twr_loss_thres == -1 ? NULL : new ceres::HuberLoss(twr_loss_thres);
                ceres::CostFunction *cost_function = new GPTWRFactor(range, p_W_a, p_B_t, wTwr, traj->getGPMixerPtr(), s);
                auto res = problem.AddResidualBlock(cost_function, twr_loss_function, factor_param_blocks);
                // Record the residual block
                factorMeta.res.push_back(res);

                // Record the time stamp of the factor
                factorMeta.stamp.push_back(ts);
            }
        }
    }

    void AddPoseFactors(
        ceres::Problem &problem, GaussianProcessPtr &traj,
        map<double *, ParamInfo> &paramInfo, FactorMeta &factorMeta,
        const deque<GtrMsgPtr> &poseData,
        double tmin, double tmax, double wpose, double pose_loss_thres)
    {

        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = usmax.first + 1;

        for (auto &pose : poseData)
        {
            double ts = pose->header.stamp.toSec();

            if (!traj->TimeInInterval(ts, 1e-6))
                continue;

            auto us = traj->computeTimeIndex(ts);
            int u = us.first;
            double s = us.second;

            if (u < kidxmin || u + 1 > kidxmax)
                continue;

            Quaternd q_W_B(pose->transform.rotation.w, pose->transform.rotation.x, pose->transform.rotation.y, pose->transform.rotation.z);
            Vector3d p_W_B(pose->transform.translation.x, pose->transform.translation.y, pose->transform.translation.z);

            vector<double *> factor_param_blocks;
            factorMeta.coupled_params.push_back(vector<ParamInfo>());

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

            ceres::LossFunction *pose_loss_function = pose_loss_thres == -1 ? NULL : new ceres::HuberLoss(pose_loss_thres);

            ceres::CostFunction *cost_function = new GPPoseFactor(q_W_B, p_W_B, wpose, traj->getGPMixerPtr(), s);

            auto res = problem.AddResidualBlock(cost_function, pose_loss_function, factor_param_blocks);

            // Record the residual block
            factorMeta.res.push_back(res);

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(ts);
        }
    }

    void AddIMUFactors(ceres::Problem &problem, GaussianProcessPtr &traj, Vector3d &XBIG, Vector3d &XBIA, Vector3d &g,
                       map<double *, ParamInfo> &paramInfo, FactorMeta &factorMeta,
                       const deque<ImuMsgPtr> &imuData, double tmin, double tmax,
                       double wGyro, double wAcce, double wBiasGyro, double wBiasAcce)
    {
        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = usmax.first + 1;
        for (auto &imu : imuData)
        {
            double ts = imu->header.stamp.toSec();

            if (!traj->TimeInInterval(ts, 1e-6))
                continue;

            auto us = traj->computeTimeIndex(ts);
            int u = us.first;
            double s = us.second;

            if (u < kidxmin || u + 1 > kidxmax)
                continue;

            Vector3d gyr(imu->angular_velocity.x, -imu->angular_velocity.y, -imu->angular_velocity.z);
            Vector3d acc(imu->linear_acceleration.x, -imu->linear_acceleration.y, -imu->linear_acceleration.z);

            // cout << gyr.transpose() << endl;
            // cout << acc.transpose() << endl;

            vector<double *> factor_param_blocks;
            factorMeta.coupled_params.push_back(vector<ParamInfo>());
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
            factor_param_blocks.push_back(XBIG.data());
            factor_param_blocks.push_back(XBIA.data());
            factor_param_blocks.push_back(g.data());
            factorMeta.coupled_params.back().push_back(paramInfoMap[XBIG.data()]);
            factorMeta.coupled_params.back().push_back(paramInfoMap[XBIA.data()]);
            // factorMeta.coupled_params.back().push_back(paramInfoMap[g.data()]);

            double imu_loss_thres = -1.0;
            ceres::LossFunction *imu_loss_function = imu_loss_thres == -1 ? NULL : new ceres::HuberLoss(imu_loss_thres);
            ceres::CostFunction *cost_function = new GPIMUFactor(acc, gyr, XBIA, XBIG, wGyro, wAcce, wBiasGyro, wBiasAcce, traj->getGPMixerPtr(), s);
            auto res = problem.AddResidualBlock(cost_function, imu_loss_function, factor_param_blocks);

            // Record the residual block
            factorMeta.res.push_back(res);

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(ts);
        }
    }

    void Marginalize(ceres::Problem &problem, GaussianProcessPtr &traj,
                     double tmin, double tmax, double tmid,
                     map<double *, ParamInfo> &paramInfoMap,
                     FactorMeta &factorMetaMp2k, FactorMeta &factorMetaTwr,
                     FactorMeta &factorMetaIMU, FactorMeta &factorMetaPose, FactorMeta &factorMetaPrior)
    {
        // Deskew, Transform and Associate
        auto FindRemovedFactors = [&tmid](FactorMeta &factorMeta, FactorMeta &factorMetaRemoved, FactorMeta &factorMetaRetained) -> void
        {
            for (int ridx = 0; ridx < factorMeta.res.size(); ridx++)
            {
                ceres::ResidualBlockId &res = factorMeta.res[ridx];
                // int KC = factorMeta.kidx[ridx].size();
                bool removed = factorMeta.stamp[ridx] < tmid;
                if (removed)
                {
                    // factorMetaRemoved.knots_coupled = factorMeta.res[ridx].knots_coupled;
                    factorMetaRemoved.res.push_back(res);
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
                    factorMetaRetained.res.push_back(res);
                    factorMetaRetained.coupled_params.push_back(factorMeta.coupled_params[ridx]);
                    factorMetaRetained.stamp.push_back(factorMeta.stamp[ridx]);
                }
                for (int i = 0; i < factorMeta.coupled_params[ridx].size(); i++)
                {
                    assert(factorMeta.coupled_params[ridx].at(i).pidx >= 0);
                }
            }
        };
        
        // Find the MP2k factors that will be removed
        FactorMeta factorMetaMp2kRemoved, factorMetaMp2kRetained;
        FindRemovedFactors(factorMetaMp2k, factorMetaMp2kRemoved, factorMetaMp2kRetained);
        // printf("factorMetaMp2k: %d. Removed: %d\n", factorMetaMp2k.size(), factorMetaMp2kRemoved.size());

        // Find the TDOA factors that will be removed
        FactorMeta factorMetaTwrRemoved, factorMetaTwrRetained;
        FindRemovedFactors(factorMetaTwr, factorMetaTwrRemoved, factorMetaTwrRetained);
        // printf("factorMetaTwr: %d. Removed: %d\n", factorMetaTwr.size(), factorMetaTwrRemoved.size());

        // Find the extrinsic factors that will be removed
        FactorMeta factorMetaIMURemoved, factorMetaIMURetained;
        FindRemovedFactors(factorMetaIMU, factorMetaIMURemoved, factorMetaIMURetained);
        // printf("factorMetaIMU: %d. Removed: %d\n", factorMetaIMU.size(), factorMetaIMURemoved.size());

        FactorMeta factorMetaPoseRemoved, factorMetaPoseRetained;
        FindRemovedFactors(factorMetaPose, factorMetaPoseRemoved, factorMetaPoseRetained);
        // printf("factorMetaIMU: %d. Removed: %d\n", factorMetaIMU.size(), factorMetaIMURemoved.size());

        FactorMeta factorMetaPriorRemoved, factorMetaPriorRetained;
        FindRemovedFactors(factorMetaPrior, factorMetaPriorRemoved, factorMetaPriorRetained);

        FactorMeta factorMetaRemoved;
        FactorMeta factorMetaRetained;

        factorMetaRemoved = factorMetaMp2kRemoved + factorMetaTwrRemoved + factorMetaIMURemoved + factorMetaPoseRemoved + factorMetaPriorRemoved;
        if (factorMetaPriorRetained.res.empty())
            factorMetaRetained = factorMetaMp2kRetained + factorMetaTwrRetained + factorMetaIMURetained + factorMetaPoseRetained;
        else
            factorMetaRetained = factorMetaMp2kRetained + factorMetaTwrRetained + factorMetaIMURetained + factorMetaPoseRetained + factorMetaPriorRetained;

        // printf("Factor retained: %d. Factor removed %d.\n", factorMetaRetained.size(), factorMetaRemoved.size());

        // Find the set of params belonging to removed factors
        map<double *, ParamInfo> removed_params;
        for (auto &cpset : factorMetaRemoved.coupled_params)
            for (auto &cp : cpset)
                removed_params[cp.address] = paramInfoMap[cp.address];

        // Find the set of params belonging to the retained factors
        map<double *, ParamInfo> retained_params;
        for (auto &cpset : factorMetaRetained.coupled_params)
            for (auto &cp : cpset)
                retained_params[cp.address] = paramInfoMap[cp.address];

        // printf("removed_params: %d. retained_params %d.\n", removed_params.size(), retained_params.size());

        // Find the intersection of the two sets, which will be the kept parameters
        vector<ParamInfo> marg_params;
        vector<ParamInfo> kept_params;
        for (auto &param : removed_params)
        {
            // ParamInfo &param = removed_params[param.first];
            if (retained_params.find(param.first) != retained_params.end())
                kept_params.push_back(param.second);
            else
                marg_params.push_back(param.second);

            assert(param.second.pidx >= 0);
        }
        // printf("kept_params: %d. marg_params %d.\n", kept_params.size(), marg_params.size());

        auto compareParam = [](const ParamInfo &a, const ParamInfo &b) -> bool
        {
            // No need to do insanity check since we only work with one trajectory here
            return a.pidx < b.pidx;
        };

        std::sort(marg_params.begin(), marg_params.end(), compareParam);
        std::sort(kept_params.begin(), kept_params.end(), compareParam);

        int marg_count = 0;
        for (auto &param : marg_params)
        {
            marg_count++;
            // printf(KMAG
            //        "Marg param %3d. Addr: %9x. Type: %2d. Role: %2d. "
            //        "Pidx: %4d. Tidx: %2d. Kidx: %4d. Sidx: %2d.\n"
            //        RESET,
            //        marg_count, param.address, param.type, param.role,
            //        param.pidx, param.tidx, param.kidx, param.sidx);
        }

        int kept_count = 0;
        for (auto &param : kept_params)
        {
            kept_count++;
            // printf(KCYN
            //        "Kept param %3d. Addr: %9x. Type: %2d. Role: %2d. "
            //        "Pidx: %4d. Tidx: %2d. Kidx: %4d. Sidx: %2d.\n"
            //        RESET,
            //        marg_count, param.address, param.type, param.role,
            //        param.pidx, param.tidx, param.kidx, param.sidx);
        }

        assert(kept_count != 0);

        // Just make sure that all of the column index will increase
        {
            int prev_idx = -1;
            for (auto &param : kept_params)
            {
                ROS_ASSERT_MSG(param.pidx > prev_idx, "%d, %d", param.pidx, prev_idx);
                prev_idx = param.pidx;
            }
        }
        // Make all parameter block variables
        std::vector<double *> parameter_blocks;
        problem.GetParameterBlocks(&parameter_blocks);
        for (auto &paramblock : parameter_blocks)
            problem.SetParameterBlockVariable(paramblock);

        auto GetJacobian = [](ceres::CRSMatrix &J) -> MatrixXd
        {
            MatrixXd eJ(J.num_rows, J.num_cols);
            eJ.setZero();
            for (int r = 0; r < J.num_rows; ++r)
            {
                for (int idx = J.rows[r]; idx < J.rows[r + 1]; ++idx)
                {
                    const int c = J.cols[idx];
                    eJ(r, c) = J.values[idx];
                }
            }
            return eJ;
        };

        // Find the jacobians of factors that will be removed
        ceres::Problem::EvaluateOptions e_option;
        e_option.residual_blocks = factorMetaRemoved.res;
        double marg_cost;
        vector<double> residual_;
        ceres::CRSMatrix Jacobian_;
        problem.Evaluate(e_option, &marg_cost, &residual_, NULL, &Jacobian_);
        VectorXd residual = Eigen::Map<VectorXd>(residual_.data(), residual_.size());
        MatrixXd Jacobian = GetJacobian(Jacobian_);
        // Extract all collumns corresponding to the marginalized states
        int MARG_SIZE = 0;
        for (auto &param : marg_params)
            MARG_SIZE += param.delta_size;
        int KEPT_SIZE = 0;
        for (auto &param : kept_params)
            KEPT_SIZE += param.delta_size;

        MatrixXd Jmk = MatrixXd::Zero(Jacobian.rows(), MARG_SIZE + KEPT_SIZE);

        auto CopyCol = [](string msg, MatrixXd &Jtarg, MatrixXd &Jsrc, ParamInfo param, int BASE_TARGET) -> void
        {
            int XBASE = param.pidx * param.delta_size;

            for (int c = 0; c < param.delta_size; c++)
            {
                // printf("%d. %d. %d. %d. %d\n", Jtarg.cols(), Jsrc.cols(), BASE_TARGET, XBASE, c);

                // Copy the column from source to target
                Jtarg.col(BASE_TARGET + c) << Jsrc.col(XBASE + c);

                // Zero out this column
                Jsrc.col(XBASE + c).setZero();
            }
        };
        int MARG_BASE = 0;
        int KEPT_BASE = MARG_SIZE;

        int TARGET_BASE = 0;

        // Copy the Jacobians of marginalized states
        for (int idx = 0; idx < marg_params.size(); idx++)
        {
            CopyCol(string("marg"), Jmk, Jacobian, marg_params[idx], TARGET_BASE);
            TARGET_BASE += marg_params[idx].delta_size;
        }

        assert(TARGET_BASE == KEPT_BASE);

        // Copy the Jacobians of kept states
        for (int idx = 0; idx < kept_params.size(); idx++)
        {
            CopyCol(string("kept"), Jmk, Jacobian, kept_params[idx], TARGET_BASE);
            TARGET_BASE += kept_params[idx].delta_size;
        }
        // // Copy the Jacobians of the extrinsic states
        // Jmkx.rightCols(XTRS_SIZE) = Jacobian.rightCols(XTRS_SIZE);
        // Jacobian.rightCols(XTRS_SIZE).setZero();

        // Calculate the Hessian
        typedef SparseMatrix<double> SMd;
        SMd r = residual.sparseView();
        r.makeCompressed();
        SMd J = Jmk.sparseView();
        J.makeCompressed();
        SMd H = J.transpose() * J;
        SMd b = J.transpose() * r;

        // // Divide the Hessian into corner blocks
        // int MARG_SIZE = RMVD_SIZE;
        // int KEEP_SIZE = KEPT_SIZE + XTRS_SIZE;

        SMd Hmm = H.block(0, 0, MARG_SIZE, MARG_SIZE);
        SMd Hmk = H.block(0, MARG_SIZE, MARG_SIZE, KEPT_SIZE);
        SMd Hkm = H.block(MARG_SIZE, 0, KEPT_SIZE, MARG_SIZE);
        SMd Hkk = H.block(MARG_SIZE, MARG_SIZE, KEPT_SIZE, KEPT_SIZE);

        SMd bm = b.block(0, 0, MARG_SIZE, 1);
        SMd bk = b.block(MARG_SIZE, 0, KEPT_SIZE, 1);

        // Create the Schur Complement
        MatrixXd Hmminv = Hmm.toDense().inverse();
        MatrixXd HkmHmminv = Hkm * Hmminv;
        MatrixXd Hkeep = Hkk - HkmHmminv * Hmk;
        MatrixXd bkeep = bk - HkmHmminv * bm;

        MatrixXd Jkeep;
        VectorXd rkeep;
        margInfo->HbToJr(Hkeep, bkeep, Jkeep, rkeep);

        // printf("Jacobian %d x %d. Jmkx: %d x %d. Params: %d.\n"
        //        "Jkeep: %d x %d. rkeep: %d x %d. Keep size: %d.\n"
        //        "Hkeepmax: %f. bkeepmap: %f. rkeep^2: %f. mcost: %f. Ratio: %f\n",
        //         Jacobian.rows(), Jacobian.cols(), Jmk.rows(), Jmk.cols(), tk2p.size(),
        //         Jkeep.rows(), Jkeep.cols(), rkeep.rows(), rkeep.cols(), KEPT_SIZE,
        //         Hkeep.cwiseAbs().maxCoeff(), bkeep.cwiseAbs().maxCoeff(),
        //         0.5*pow(rkeep.norm(), 2), marg_cost, marg_cost/(0.5*pow(rkeep.norm(), 2)));

        // // Show the marginalization matrices
        // cout << "Jkeep\n" << Hkeep << endl;
        // cout << "rkeep\n" << bkeep << endl;

        // Making sanity checks
        map<ceres::ResidualBlockId, int> wierdRes;
        for (auto &param_ : paramInfoMap.params_info)
        {
            ParamInfo &param = param_.second;

            int tidx = param.tidx;
            // assert(tidx == -1);
            int kidx = param.kidx;
            int sidx = param.sidx;

            // if(param.tidx != -1 && param.kidx != -1)
            if (param.kidx != -1)
            {
                MatrixXd Jparam = Jacobian.block(0, param.pidx * param.delta_size, Jacobian.rows(), param.delta_size);
                if (Jparam.cwiseAbs().maxCoeff() != 0)
                {
                    vector<ceres::ResidualBlockId> resBlocks;
                    problem.GetResidualBlocksForParameterBlock(traj->getKnotSO3(kidx).data(), &resBlocks);
                    // printf("Found %2d res blocks for param %2d. Knot %2d. Traj %2d.\n",
                    //         resBlocks.size(), param.pidx,  param.kidx,  param.tidx);
                    for (auto &res : resBlocks)
                        wierdRes[res] = 1;
                }
            }

            // printf("KnotParam: %2d. Traj %2d, Knot %2d. Max: %f\n",
            //         pidx, tkp.first.first, tkp.first.second, Jparam.cwiseAbs().maxCoeff());
        }
        // cout << endl;

        // Check to see if removed res are among the wierd res
        for (auto &res : factorMetaMp2kRemoved.res)
            assert(wierdRes.find(res) == wierdRes.end());
        // printf("Wierd res: %d. No overlap with mp2k\n");

        for (auto &res : factorMetaTwrRemoved.res)
            assert(wierdRes.find(res) == wierdRes.end());
        // printf("Wierd res: %d. No overlap with lidar\n");

        // for(auto &res : factorMetaGpxRemoved.res)
        //     assert(wierdRes.find(res) == wierdRes.end());
        // printf("Wierd res: %d. No overlap with Gpx\n", wierdRes.size());

        for (auto &res : factorMetaPriorRemoved.res)
            assert(wierdRes.find(res) == wierdRes.end());
        // printf("Wierd res: %d. No overlap with Gpx\n", wierdRes.size());

        // Save the marginalization factors and states
        if (margInfo == nullptr)
            margInfo = MarginalizationInfoPtr(new MarginalizationInfo());

        // Copy the marginalization matrices
        margInfo->Hkeep = Hkeep;
        margInfo->bkeep = bkeep;
        margInfo->Jkeep = Jkeep;
        margInfo->rkeep = rkeep;
        margInfo->keptParamInfo = kept_params;
        // Add the prior of kept params
        margInfo->keptParamPrior.clear();
        for (auto &param : kept_params)
        {
            margInfo->keptParamPrior[param.address] = vector<double>();
            for (int idx = 0; idx < param.param_size; idx++)
                margInfo->keptParamPrior[param.address].push_back(param.address[idx]);
        }
    }

    void Evaluate(
        GaussianProcessPtr &traj, Vector3d &XBIG, Vector3d &XBIA, Vector3d &g,
        double tmin, double tmax, double tmid,
        const deque<TwrMsgPtr> &twrData, const deque<ImuMsgPtr> &imuData, const deque<GtrMsgPtr> &poseData,
        const vector<Vector3d> &anc_pos, const vector<Vector3d> &tag_pos,
        bool do_marginalization, double wTwr, double wGyro, double wAcce, double wBiasGyro, double wBiasAcce, double twr_loss_thres, double mp_loss_thres)
    {
        static int cnt = 0;
        TicToc tt_build;
        cnt++;

        // Ceres problem
        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;

        // Set up the ceres problem
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = MAX_THREADS;
        options.max_num_iterations = 100;
        options.check_gradients = false;

        options.gradient_check_relative_precision = 0.02;

        // Documenting the parameter blocks
        paramInfoMap.clear();
        // Add the parameter blocks
        {
            // Add the parameter blocks for rotation
            AddTrajParams(problem, traj, 0, paramInfoMap, tmin, tmax, tmid);
            problem.AddParameterBlock(XBIG.data(), 3);
            problem.AddParameterBlock(XBIA.data(), 3);
            problem.AddParameterBlock(g.data(), 3);

            paramInfoMap.insert(make_pair(XBIG.data(), ParamInfo(XBIG.data(), ParamType::RV3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1)));
            paramInfoMap.insert(make_pair(XBIA.data(), ParamInfo(XBIA.data(), ParamType::RV3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1)));
            paramInfoMap.insert(make_pair(g.data(), ParamInfo(g.data(), ParamType::RV3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1)));

            problem.SetParameterBlockConstant(g.data());

            // Sanity check
            for (auto &param_ : paramInfoMap.params_info)
            {
                ParamInfo param = param_.second;

                int tidx = 0;
                int kidx = param.kidx;
                int sidx = param.sidx;

                if (param.tidx != -1 && param.kidx != -1)
                {
                    switch (sidx)
                    {
                    case 0:
                        assert(param.address == traj->getKnotSO3(kidx).data());
                        break;
                    case 1:
                        assert(param.address == traj->getKnotOmg(kidx).data());
                        break;
                    case 2:
                        assert(param.address == traj->getKnotAlp(kidx).data());
                        break;
                    case 3:
                        assert(param.address == traj->getKnotPos(kidx).data());
                        break;
                    case 4:
                        assert(param.address == traj->getKnotVel(kidx).data());
                        break;
                    case 5:
                        assert(param.address == traj->getKnotAcc(kidx).data());
                        break;
                    default:
                        printf("Unrecognized param block! %d, %d, %d\n", tidx, kidx, sidx);
                        break;
                    }
                }
                else
                {
                    // if(sidx == 0)
                    //     assert(param.address == R_Lx_Ly.data());
                    // if(sidx == 1)
                    //     assert(param.address == P_Lx_Ly.data());
                }
            }
        }

        // Add the motion prior factor
        FactorMeta factorMetaMp2k;
        double cost_mp2k_init = -1, cost_mp2k_final = -1;
        AddMP2KFactorsUI(problem, traj, paramInfoMap, factorMetaMp2k, tmin, tmax, mp_loss_thres);

        // Add the TDOA factors
        FactorMeta factorMetaTwr;
        double cost_twr_init = -1;
        double cost_twr_final = -1;
        AddTwrFactors(problem, traj, paramInfoMap, factorMetaTwr, twrData, anc_pos, tag_pos, tmin, tmax, wTwr, twr_loss_thres);

        FactorMeta factorMetaIMU;
        double cost_imu_init = -1;
        double cost_imu_final = -1;
        AddIMUFactors(problem, traj, XBIG, XBIA, g, paramInfoMap, factorMetaIMU, imuData, tmin, tmax, wGyro, wAcce, wBiasGyro, wBiasAcce);

        FactorMeta factorMetaPose;
        double cost_pose_init = -1;
        double cost_pose_final = -1;
        // AddPoseFactors(problem, traj, paramInfoMap, factorMetaPose, poseData, tmin, tmax, 1.0, -1);

        // Add the prior factor
        FactorMeta factorMetaPrior;
        double cost_prior_init = -1;
        double cost_prior_final = -1;
        if (margInfo != NULL)
            AddPriorFactor(problem, traj, factorMetaPrior, tmin, tmax);

        tt_build.Toc();

        TicToc tt_slv;

        // Find the initial cost
        Util::ComputeCeresCost(factorMetaMp2k.res, cost_mp2k_init, problem);
        Util::ComputeCeresCost(factorMetaTwr.res, cost_twr_init, problem);
        Util::ComputeCeresCost(factorMetaIMU.res, cost_imu_init, problem);
        Util::ComputeCeresCost(factorMetaPose.res, cost_pose_init, problem);
        Util::ComputeCeresCost(factorMetaPrior.res, cost_prior_init, problem);

        ceres::Solve(options, &problem, &summary);

        // std::cout << summary.FullReport() << std::endl;

        Util::ComputeCeresCost(factorMetaMp2k.res, cost_mp2k_final, problem);
        Util::ComputeCeresCost(factorMetaTwr.res, cost_twr_final, problem);
        Util::ComputeCeresCost(factorMetaIMU.res, cost_imu_final, problem);
        Util::ComputeCeresCost(factorMetaPose.res, cost_pose_final, problem);
        Util::ComputeCeresCost(factorMetaPrior.res, cost_prior_final, problem);

        // Determine the factors to remove
        if (do_marginalization)
            Marginalize(problem, traj, tmin, tmax, tmid, paramInfoMap, factorMetaMp2k, factorMetaTwr, factorMetaIMU, factorMetaPose, factorMetaPrior);

        tt_slv.Toc();
    }
};

typedef std::shared_ptr<GPNL> GPNLPtr;
