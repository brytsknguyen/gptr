#pragma once

#include <Eigen/Sparse>

#include "GaussNewtonUtilities.hpp"

#include "factor/GPMotionPriorTwoKnotsFactor.h"
#include "factor/GPTDOAFactor.h"
#include "factor/GPIMUFactor.h"

class GPUI
{
typedef SparseMatrix<double> SMd;

private:

    // Node handle to get information needed
    NodeHandlePtr nh;

    // Map of traj-kidx and parameter id
    map<pair<int, int>, int> tk2p;
    ParamInfoMap paramInfoMap;
    MarginalizationInfoPtr margInfo;

    double mp_loss_thres;

public:

//     // Destructor
//    ~GPUI() {};

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
    GPUI(NodeHandlePtr &nh_) : nh(nh_)
    {
        Util::GetParam(nh, "mp_loss_thres" , mp_loss_thres );
    }

    void AddTrajParams(
        ceres::Problem &problem, double tmin, double tmax, double tmid,
        GaussianProcessPtr &traj, int tidx,
        ParamInfoMap &paramInfoMap)
    {
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

            // Log down the information of the params
            paramInfoMap.insert(traj->getKnotSO3(kidx).data(), ParamInfo(traj->getKnotSO3(kidx).data(), getEigenPtr(traj->getKnotSO3(kidx)), ParamType::SO3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 0));
            paramInfoMap.insert(traj->getKnotOmg(kidx).data(), ParamInfo(traj->getKnotOmg(kidx).data(), getEigenPtr(traj->getKnotOmg(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 1));
            paramInfoMap.insert(traj->getKnotAlp(kidx).data(), ParamInfo(traj->getKnotAlp(kidx).data(), getEigenPtr(traj->getKnotAlp(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 2));
            paramInfoMap.insert(traj->getKnotPos(kidx).data(), ParamInfo(traj->getKnotPos(kidx).data(), getEigenPtr(traj->getKnotPos(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 3));
            paramInfoMap.insert(traj->getKnotVel(kidx).data(), ParamInfo(traj->getKnotVel(kidx).data(), getEigenPtr(traj->getKnotVel(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 4));
            paramInfoMap.insert(traj->getKnotAcc(kidx).data(), ParamInfo(traj->getKnotAcc(kidx).data(), getEigenPtr(traj->getKnotAcc(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), tidx, kidx, 5));
        }
    }

    void AddMP2KFactors(
        ceres::Problem &problem, double tmin, double tmax,
        GaussianProcessPtr &traj,
        double mp_loss_thres,
        ParamInfoMap &paramInfoMap, FactorMeta &factorMeta)
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

            // Create the factors
            ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
            ceres::CostFunction *cost_function = new GPMotionPriorTwoKnotsFactor(traj->getGPMixerPtr());
            auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);

            // Record the residual block
            factorMeta.res.push_back(res_block);

        }
    }

    void AddIMUFactors(
        ceres::Problem &problem, double tmin, double tmax,
        GaussianProcessPtr &traj, Vector3d &XBIG, Vector3d &XBIA, Vector3d &g,
        const vector<IMUData> &imuData, double wGyro, double wAcce, double wBiasGyro, double wBiasAcce,
        ParamInfoMap &paramInfo, FactorMeta &factorMeta)
    {
        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = min(traj->getNumKnots() - 1, usmax.first + 1);

        for (auto &imu : imuData)
        {
            if (!traj->TimeInInterval(imu.t, 1e-6))
                continue;
            
            auto   us = traj->computeTimeIndex(imu.t);
            int    u  = us.first;
            double s  = us.second;

            if (u < kidxmin || u + 1 >= kidxmax)
                continue;
      
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
            factorMeta.coupled_params.back().push_back(paramInfoMap[g.data()]);                       

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(imu.t);

            double imu_loss_thres = -1.0;
            ceres::LossFunction *imu_loss_function = imu_loss_thres == -1 ? NULL : new ceres::HuberLoss(imu_loss_thres);
            ceres::CostFunction *cost_function = new GPIMUFactor(imu.acc, imu.gyro, XBIA, XBIG, wGyro, wAcce, wBiasGyro, wBiasAcce, traj->getGPMixerPtr(), s);
            auto res = problem.AddResidualBlock(cost_function, imu_loss_function, factor_param_blocks);

            // Record the residual block
            factorMeta.res.push_back(res);
        }
    }

    void AddTDOAFactors(
        ceres::Problem &problem, double tmin, double tmax,
        GaussianProcessPtr &traj,
        const vector<TDOAData> &tdoaData, std::map<uint16_t, Eigen::Vector3d>& pos_anchors, const Vector3d &P_I_tag, double w_tdoa, double tdoa_loss_thres,
        ParamInfoMap &paramInfo, FactorMeta &factorMeta)
    {
        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = min(traj->getNumKnots() - 1, usmax.first + 1);

        for (auto &tdoa : tdoaData)
        {
            if (!traj->TimeInInterval(tdoa.t, 1e-6))
                continue;
            
            auto   us = traj->computeTimeIndex(tdoa.t);
            int    u  = us.first;
            double s  = us.second;

            if (u < kidxmin || u + 1 >= kidxmax)
                continue;

            Eigen::Vector3d pos_an_A = pos_anchors[tdoa.idA];
            Eigen::Vector3d pos_an_B = pos_anchors[tdoa.idB];          

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

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(tdoa.t);

            ceres::LossFunction *tdoa_loss_function = tdoa_loss_thres == -1 ? NULL : new ceres::HuberLoss(tdoa_loss_thres);
            ceres::CostFunction *cost_function = new GPTDOAFactor(tdoa.data, pos_an_A, pos_an_B, P_I_tag, w_tdoa, traj->getGPMixerPtr(), s);
            auto res = problem.AddResidualBlock(cost_function, tdoa_loss_function, factor_param_blocks);
            
            // Record the residual block
            factorMeta.res.push_back(res);
        }
    }

    void AddPriorFactor(
        ceres::Problem &problem, double tmin, double tmax,
        GaussianProcessPtr &traj,
        MarginalizationInfoPtr &margInfo, ParamInfoMap &paramInfo, FactorMeta &factorMeta)
    {
        // Check if kept states are still in the param list
        bool all_kept_states_found = true;
        vector<int> missing_param_idx;
        int removed_dims = 0;
        for(int idx = 0; idx < margInfo->keptParamInfo.size(); idx++)
        {
            ParamInfo &param = margInfo->keptParamInfo[idx];
            bool state_found = paramInfoMap.hasParam(param.address);
            // RINFO("param 0x%8x of tidx %2d kidx %4d of sidx %4d is %sfound in paramInfoMap",
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
            // RINFO("Remove %d params missing from %d keptParamInfos", missing_param_idx.size(), margInfo->keptParamInfo.size());
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
                //     RINFO("To remove: %d", idx);

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
                        // yolos("%d %d %d", cidx, c, removed_cidx.size());
                        // RINFO("idx: %d. param_cols: %d. cidx: %d. c: %d", idx, param_cols, cidx, c);
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

            // RINFO("Jkeep: %d %d. rkeep: %d %d. Hkeep: %d %d. bkeep: %d %d. ParamPrior: %d. ParamInfo: %d. missing_param_idx: %d",
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
                // RINFO("Deleting %d", param_idx);
                margInfo->keptParamPrior.erase(margInfo->keptParamInfo[param_idx].address);
            }

            // Remove the unfound states
            removeElementsByIndices(margInfo->keptParamInfo, missing_param_idx);

            // RINFO("Jkeep: %d %d. rkeep: %d %d. Hkeep: %d %d. bkeep: %d %d. ParamPrior: %d. ParamInfo: %d. missing_param_idx: %d",
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

            MarginalizationFactor* margFactor = new MarginalizationFactor(margInfo);

            // Add the involved parameters blocks
            auto res_block = problem.AddResidualBlock(margFactor, NULL, margInfo->getAllParamBlocks());

            // Save the residual block
            factorMeta.res.push_back(res_block);

            // Add the coupled param
            factorMeta.coupled_params.push_back(margInfo->keptParamInfo);
        }
        else
            RINFO(KYEL "All kept params in marginalization missing. Please check" RESET);
    }

    void CheckMarginalizedResAndParams(
        double tmid, ParamInfoMap &paramInfo,
        const vector<FactorMetaPtr> &factorMetas,
        vector<FactorMeta> &factorMetasRmvd, vector<FactorMeta> &factorMetasRtnd,
        FactorMeta &factorsRmvd, FactorMeta &factorsRtnd,
        vector<ParamInfo> &removed_params, vector<ParamInfo> &priored_params
    )
    {
        // Insanity check to keep track of all the params
        for(auto &factorMeta : factorMetas)
            for(auto &cpset : factorMeta->coupled_params)
                for(auto &cp : cpset)
                    assert(paramInfoMap.hasParam(cp.address));

        // Determine removed factors
        auto FindRemovedFactors = [&tmid](const FactorMeta &factorMeta, FactorMeta &factorMetaRemoved, FactorMeta &factorMetaRetained) -> void
        {
            for(int ridx = 0; ridx < factorMeta.stamp.size(); ridx++)
            {
                // ceres::ResidualBlockId &res = factorMeta.res[ridx];
                // int KC = factorMeta.kidx[ridx].size();
                bool removed = factorMeta.stamp[ridx] <= tmid;

                if (removed)
                {
                    // factorMetaRemoved.knots_coupled = factorMeta.res[ridx].knots_coupled;
                    factorMetaRemoved.stamp.push_back(factorMeta.stamp[ridx]);
                    factorMetaRemoved.coupled_params.push_back(factorMeta.coupled_params[ridx]);
                    if (factorMeta.res.size() != 0) factorMetaRemoved.res.push_back(factorMeta.res[ridx]);
                    if (factorMeta.coupled_coef.size() != 0) factorMetaRemoved.coupled_coef.push_back(factorMeta.coupled_coef[ridx]);

                    // for(int coupling_idx = 0; coupling_idx < KC; coupling_idx++)
                    // {
                    //     int kidx = factorMeta.kidx[ridx][coupling_idx];
                    //     int tidx = factorMeta.tidx[ridx][coupling_idx];
                    //     tk_removed_res.push_back(make_pair(tidx, kidx));
                    //     // RINFO("Removing knot %d of traj %d, param %d.", kidx, tidx, tk2p[tk_removed_res.back()]);
                    // }
                }
                else
                {
                    factorMetaRetained.stamp.push_back(factorMeta.stamp[ridx]);
                    factorMetaRetained.coupled_params.push_back(factorMeta.coupled_params[ridx]);
                    if (factorMeta.res.size() != 0) factorMetaRetained.res.push_back(factorMeta.res[ridx]);
                    if (factorMeta.coupled_coef.size() != 0) factorMetaRetained.coupled_coef.push_back(factorMeta.coupled_coef[ridx]);
                }
            }
        };

        for(int gidx = 0; gidx < factorMetas.size(); gidx++)
        {
            const FactorMetaPtr &factorMeta = factorMetas[gidx];
            FindRemovedFactors(*factorMeta, factorMetasRmvd[gidx], factorMetasRtnd[gidx]);
            factorsRmvd += factorMetasRmvd[gidx];
            factorsRtnd += factorMetasRtnd[gidx];
        }

        // Find the set of params belonging to removed factors
        map<double*, ParamInfo> removed_factors_params;
        for(auto &cpset : factorsRmvd.coupled_params)
            for(auto &cp : cpset)
            {
                assert(paramInfoMap.hasParam(cp.address));
                removed_factors_params[cp.address] = (paramInfoMap[cp.address]);
            }

        // Find the set of params belonging to the retained factors
        map<double*, ParamInfo> retained_factors_params;
        for(auto &cpset : factorsRtnd.coupled_params)
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
            return a.xidx < b.xidx;
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
        vector<ParamInfo> &removed_params, vector<ParamInfo> &priored_params,
        const VectorXd RESIDUAL, const MatrixXd JACOBIAN)
    {
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

            if(param.type == ParamType::SO3)
            {
                Vec3 error = ((*static_pointer_cast<SO3d>(param.ptr)).inverse()
                            *(margInfo->DoubleToSO3<double>(margInfo->keptParamPrior[param.address]))).log();
                ROS_ASSERT_MSG(error.norm() == 0, "error.norm(): %f\n", error.norm());
            }
            
            if(param.type == ParamType::RV3)
            {
                Vec3 error = (*static_pointer_cast<Vec3>(param.ptr) - margInfo->DoubleToRV3<double>(margInfo->keptParamPrior[param.address]));
                ROS_ASSERT_MSG(error.norm() == 0, "error.norm(): %f\n", error.norm());
            }
        }

    }

    void Evaluate(
        double tmin, double tmax, double tmid,
        GaussianProcessPtr &traj, Vector3d &XBIG, Vector3d &XBIA, Vector3d &g,
        const vector<TDOAData> &tdoaData, const vector<IMUData> &imuData,
        std::map<uint16_t, Eigen::Vector3d>& pos_anchors, const Vector3d &P_I_tag, 
        double w_tdoa, double wGyro, double wAcce, double wBiasGyro, double wBiasAcce, double tdoa_loss_thres, double mp_loss_thres, bool do_marginalization)
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
            AddTrajParams(problem, tmin, tmax, tmid, traj, 0, paramInfoMap);
            problem.AddParameterBlock(XBIG.data(), 3);
            problem.AddParameterBlock(XBIA.data(), 3);
            problem.AddParameterBlock(g.data(), 3);

            paramInfoMap.insert(XBIG.data(), ParamInfo(XBIG.data(), getEigenPtr(XBIG), ParamType::RV3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1));
            paramInfoMap.insert(XBIA.data(), ParamInfo(XBIA.data(), getEigenPtr(XBIA), ParamType::RV3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1));
            paramInfoMap.insert(g.data(),    ParamInfo(g.data(),    getEigenPtr(g),    ParamType::RV3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1));
            problem.SetParameterBlockConstant(g.data());
            
            // Sanity check
            for(auto &param_ : paramInfoMap.params_info)
            {
                ParamInfo param = param_.second;

                int tidx = 0;
                int kidx = param.kidx;
                int sidx = param.sidx;

                if(param.tidx != -1 && param.kidx != -1)
                {
                    switch(sidx)
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
                            RINFO("Unrecognized param block! %d, %d, %d", tidx, kidx, sidx);
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
        AddMP2KFactors(problem, tmin, tmax, traj, mp_loss_thres, paramInfoMap, factorMetaMp2k);

        FactorMeta factorMetaIMU;
        double cost_imu_init = -1; double cost_imu_final = -1;
        AddIMUFactors(problem, tmin, tmax, traj, XBIG, XBIA, g, imuData, wGyro, wAcce, wBiasGyro, wBiasAcce, paramInfoMap, factorMetaIMU);

        // Add the TDOA factors
        FactorMeta factorMetaTDOA;
        double cost_tdoa_init = -1; double cost_tdoa_final = -1;
        AddTDOAFactors(problem, tmin, tmax, traj, tdoaData, pos_anchors, P_I_tag, w_tdoa, tdoa_loss_thres, paramInfoMap, factorMetaTDOA);

        // Add the prior factor
        FactorMeta factorMetaPrior;
        double cost_prior_init = -1; double cost_prior_final = -1;
        if (margInfo != NULL)
            AddPriorFactor(problem, tmin, tmax, traj, margInfo, paramInfoMap, factorMetaPrior);
            
        tt_build.Toc();

        TicToc tt_slv;

        // Find the initial cost
        Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_init,  problem);
        Util::ComputeCeresCost(factorMetaTDOA.res,  cost_tdoa_init,  problem);
        Util::ComputeCeresCost(factorMetaIMU.res,   cost_imu_init,   problem);
        Util::ComputeCeresCost(factorMetaPrior.res, cost_prior_init, problem);

        ceres::Solve(options, &problem, &summary);

        // std::cout << summary.FullReport() << std::endl;

        Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_final,  problem);
        Util::ComputeCeresCost(factorMetaTDOA.res,  cost_tdoa_final,  problem);
        Util::ComputeCeresCost(factorMetaIMU.res,   cost_imu_final,   problem);
        Util::ComputeCeresCost(factorMetaPrior.res, cost_prior_final, problem);

        // Determine the factors to remove
        if (do_marginalization)
        {   
            vector<FactorMetaPtr> factorMetas;
            factorMetas.push_back(getEigenPtr(factorMetaMp2k));
            factorMetas.push_back(getEigenPtr(factorMetaTDOA));
            factorMetas.push_back(getEigenPtr(factorMetaIMU));
            factorMetaPrior.res.size() != 0 ? factorMetas.push_back(getEigenPtr(factorMetaPrior)) : (void)0;

            vector<FactorMeta> factorMetasRmvd(factorMetas.size(), FactorMeta());
            vector<FactorMeta> factorMetasRtnd(factorMetas.size(), FactorMeta());
            FactorMeta factorsRmvd;
            FactorMeta factorsRtnd;
            vector<ParamInfo> removed_params, priored_params;
            CheckMarginalizedResAndParams(tmid, paramInfoMap, factorMetas, factorMetasRmvd, factorMetasRtnd,
                                          factorsRmvd, factorsRtnd, removed_params, priored_params);

            // Evaluate the remove factors
            VectorXd RESIDUAL; MatrixXd JACOBIAN;
            {
                // Make all parameter block variables
                std::vector<double*> parameter_blocks;
                problem.GetParameterBlocks(&parameter_blocks);
                for (auto &paramblock : parameter_blocks)
                    problem.SetParameterBlockVariable(paramblock);

                // FactorMeta factorMeta;
                // ceres::Problem::EvaluateOptions e_option;
                // for(auto &fm : factorMetas)
                //     factorMeta += *fm;
                
                double cost;
                FindJrByCeres(problem, factorsRmvd, cost, RESIDUAL, JACOBIAN);
            }
            
            // Do the Schur complement to find prior factor
            Marginalize(removed_params, priored_params, RESIDUAL, JACOBIAN);
            // report.marginalization_done = true;
        }

        tt_slv.Toc();
    }

};

typedef std::shared_ptr<GPUI> GPUIPtr;
