#include "unistd.h"
#include <algorithm> // for std::sort
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_cpp/storage_options.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

// ROS utilities
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/odometry.hpp"

// Topics
#include "cf_msgs/msg/tdoa.hpp"
#include "cf_msgs/msg/tof.hpp"

// Custom built utilities
#include "GaussianProcess.hpp"
// #include "GPUI.hpp"
#include "utility.h"

#include <Eigen/Sparse>

#include "GaussNewtonUtilities.hpp"

#include "factor/GPMotionPriorTwoKnotsFactor.h"
#include "factor/GPTDOAFactor.h"
#include "factor/GPIMUFactor.h"

namespace fs = std::filesystem;

using namespace std;

NodeHandlePtr nh_ptr;

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
        double w_tdoa, double wGyro, double wAcce, double wBiasGyro, double wBiasAcce, double tdoa_loss_thres, double mp_loss_thres, bool do_marginalization,
        CloudPosePtr &gtrPoseCloud, string &report_, map<string, double> &report)
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
        options.max_num_iterations = 50;
        options.function_tolerance = 0.0;
        options.gradient_tolerance = 0.0;
        options.parameter_tolerance = 0.0;

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
            
        tt_build.Toc();


        // Find the initial cost
        Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_init,  problem);
        Util::ComputeCeresCost(factorMetaTDOA.res,  cost_tdoa_init,  problem);
        Util::ComputeCeresCost(factorMetaIMU.res,   cost_imu_init,   problem);

        TicToc tt_solve;
        ceres::Solve(options, &problem, &summary);
        tt_solve.Toc();

        // std::cout << summary.FullReport() << std::endl;

        Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_final,  problem);
        Util::ComputeCeresCost(factorMetaTDOA.res,  cost_tdoa_final,  problem);
        Util::ComputeCeresCost(factorMetaIMU.res,   cost_imu_final,   problem);
        RINFO(KGRN"Done. %fms"RESET, tt_solve.GetLastStop());

        // tt_slv.Toc();


        RINFO("Calculating error...");
        TicToc tt_rmse;

        auto umeyama_alignment = [](vector<Vector3d>& src, vector<Vector3d>& tgt) -> myTf<double>
        {
            if (src.size() != tgt.size() || src.empty()) {
                throw std::runtime_error("Source and target must be same size and non-empty.");
            }
        
            MatrixXd src_mat(3, src.size());
            MatrixXd tgt_mat(3, tgt.size());
            for (size_t i = 0; i < src.size(); ++i)
            {
                src_mat.col(i) = src[i];
                tgt_mat.col(i) = tgt[i];
            }
        
            Eigen::Matrix4d transformation = (Eigen::umeyama(src_mat, tgt_mat, false));
            myTf<double> T_tgt_src(transformation);

            // RINFO("UMEYAMA ALIGN: %f, %f, %f. %f, %f, %f.\n",
            //        T_tgt_src.pos(0), T_tgt_src.pos(1), T_tgt_src.pos(2),
            //        T_tgt_src.yaw(), T_tgt_src.pitch(), T_tgt_src.roll());

            for (size_t i = 0; i < src.size(); ++i)
                src[i] = T_tgt_src*src[i];

            return T_tgt_src;
        };

        // Sample the trajectories
        vector<Vector3d> pos_est;
        vector<Vector3d> pos_gtr;
        for(auto &pose : gtrPoseCloud->points)
        {
            double ts = pose.t;

            if (ts < traj->getMinTime() || ts > traj->getMaxTime())
                continue;

            myTf poseEst = myTf(traj->pose(ts));
            myTf poseGtr = myTf(pose);

            pos_est.push_back(poseEst.pos);
            pos_gtr.push_back(poseGtr.pos);
        }

        // Align the estimate to the gtr
        umeyama_alignment(pos_est, pos_gtr);

        // Calculate the pose error
        vector<Vector3d> pos_err;
        for(int i = 0; i < pos_est.size(); i++)
            pos_err.push_back(pos_est[i] - pos_gtr[i]);
        vector<Vector3d> se3_err = pos_err;

        double pos_rmse = 0;
        for (auto &err : pos_err)
            pos_rmse += err.dot(err);
        pos_rmse /= pos_err.size();
        pos_rmse = sqrt(pos_rmse);

        double se3_rmse = 0;
        for (auto &err : se3_err)
            se3_rmse += err.dot(err);
        se3_rmse /= se3_err.size();
        se3_rmse = sqrt(se3_rmse);
        
        RINFO(KGRN"Done. %fms"RESET, tt_rmse.Toc());


        RINFO("Drafting report ...");
        TicToc tt_report;

        report_ = myprintf(
            "Pose group: %s. Method: %s. Dt: %.3f. "
            "Tslv: %.0f. Iterations: %d.\n"
            "Factors: MP2K: %05d, IMU: %05d, TDOA: %05d\n"
            "J0: %16.3f, MP2K: %16.3f, IMU: %7.3f. TDOA: %7.3f..\n"
            "JK: %16.3f, MP2K: %16.3f, IMU: %7.3f. TDOA: %7.3f.\n"
            "RMSE: POS: %.12f. POSE: %.12f.\n"
            ,
            traj->getGPMixerPtr()->getPoseRepresentation() == POSE_GROUP::SO3xR3 ? "SO3xR3" : "SE3",
            traj->getGPMixerPtr()->getJacobianForm() ? "AP" : "CF",
            traj->getDt(),
            tt_solve.GetLastStop(), summary.iterations.size(),
            factorMetaMp2k.size(), factorMetaIMU.size(), factorMetaTDOA.size(),
            summary.initial_cost, cost_mp2k_init,  cost_imu_init,  cost_tdoa_init,  
            summary.final_cost,   cost_mp2k_final, cost_imu_final, cost_tdoa_final,
            pos_rmse, se3_rmse
        );

        // cout << "yolo " << report_ << endl;

        report["iter"]    = summary.iterations.size();
        report["tslv"]    = summary.total_time_in_seconds;
        report["rmse"]    = pos_rmse;
        report["J0"]      = summary.initial_cost;
        report["JK"]      = summary.final_cost;
        report["MP2KJ0"]  = cost_mp2k_init;
        report["IMUJ0"]   = cost_imu_init;
        report["TDOAJ0"]  = cost_tdoa_init;
        report["MP2KJK"]  = cost_mp2k_final;
        report["IMUJK"]   = cost_imu_final;
        report["TDOAJK"]  = cost_tdoa_final;

        RINFO(KGRN"Done. %fms\n"RESET, tt_report.Toc());        
    }

};

typedef std::shared_ptr<GPUI> GPUIPtr;

std::map<uint16_t, Eigen::Vector3d> getAnchorListFromUTIL(const std::string &anchor_path)
{
    std::map<uint16_t, Eigen::Vector3d> anchor_list;
    std::string line;
    std::ifstream infile;
    infile.open(anchor_path);
    if (!infile)
    {
        std::cerr << "Unable to open file: " << anchor_path << std::endl;
        exit(1);
    }
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        char comma, tmp, tmp2;
        int anchor_id;
        double x, y, z;
        iss >> tmp >> tmp >> anchor_id >> tmp >> tmp2 >> comma >> x >> comma >> y >> comma >> z;
        if (tmp2 == 'p')
        {
            anchor_list[anchor_id] = Eigen::Vector3d(x, y, z);
        }
    }
    infile.close();
    return anchor_list;
}

typedef sensor_msgs::msg::Imu ImuMsg;
typedef ImuMsg::SharedPtr ImuMsgPtr;
typedef cf_msgs::msg::Tdoa TdoaMsg;
typedef TdoaMsg::SharedPtr TdoaMsgPtr;
typedef cf_msgs::msg::Tof TofMsg;
typedef TofMsg::SharedPtr TofMsgPtr;
typedef geometry_msgs::msg::PoseWithCovarianceStamped PoseCovStamped;

const double POSINF = std::numeric_limits<double>::infinity();
const double NEGINF = -std::numeric_limits<double>::infinity();

vector<SE3d> anc_pose;

Matrix3d gpQr;
Matrix3d gpQc;

bool auto_exit;
double w_tdoa = 0.1;
double GYR_N = 10;
double ACC_N = 0.5;
double tdoa_loss_thres = -1;
double mp_loss_thres = -1;

GaussianProcessPtr traj;

bool fuse_tdoa = true;
bool fuse_tof = false;
bool fuse_imu = true;

bool acc_ratio = true;
bool gyro_unit = true;

struct UwbImuBuf
{
    vector<TDOAData> tdoa_data;
    vector<IMUData> imu_data;

    double minTime()
    {
        double tmin = std::numeric_limits<double>::infinity();
        if (tdoa_data.size() != 0)
            tmin = min(tmin, tdoa_data.front().t);
        if (imu_data.size() != 0)
            tmin = min(tmin, imu_data.front().t);
        return tmin;
    }

    double maxTime()
    {
        double tmax = -std::numeric_limits<double>::infinity();
        if (tdoa_data.size() != 0)
            tmax = max(tmax, tdoa_data.back().t);
        if (imu_data.size() != 0)
            tmax = max(tmax, imu_data.back().t);
        return tmax;        
    }
};

UwbImuBuf UIBuf;
CloudPosePtr gtrPoseCloud(new CloudPose());

Eigen::Vector3d bg = Eigen::Vector3d::Zero();
Eigen::Vector3d ba = Eigen::Vector3d::Zero();
Eigen::Vector3d g = Eigen::Vector3d(0, 0, 9.81);
const Eigen::Vector3d P_I_tag = Eigen::Vector3d(-0.012, 0.001, 0.091);

bool if_save_traj;
std::string traj_save_path;

void saveTraj(GaussianProcessPtr traj, const std::string& save_path)
{
    if (!std::filesystem::is_directory(save_path) || !std::filesystem::exists(save_path))
    {
        std::filesystem::create_directories(save_path);
    }
    std::string traj_file_name = save_path + "traj.txt";
    std::ofstream f_traj(traj_file_name);

    std::string gt_file_name = save_path + "gt.txt";
    std::ofstream f_gt(gt_file_name);    
    for (auto &pose : gtrPoseCloud->points)
    {
        double t_gt = pose.t;
        if (t_gt < traj->getMinTime() || t_gt > traj->getMaxTime())
            continue;        
        auto est_pose = traj->pose(t_gt);
        Eigen::Vector3d est_pos = est_pose.translation();
        Eigen::Quaterniond est_ort = est_pose.unit_quaternion();
        f_traj << std::fixed << t_gt << std::setprecision(7)
               << " " << est_pos.x() << " " << est_pos.y() << " " << est_pos.z()
               << " " << est_ort.x() << " " << est_ort.y() << " " << est_ort.z() << " " << est_ort.w() << std::endl;

        f_gt << std::fixed << t_gt << std::setprecision(7)
             << " " << pose.x << " " << pose.y << " " << pose.z
             << " " << pose.qx << " " << pose.qy << " " << pose.qz << " " << pose.qw << std::endl;               
    }
    f_traj.close();
    f_gt.close();
}

void readBag(const std::string& bag_file)
{
    gtrPoseCloud->clear();
    UIBuf.tdoa_data.clear();
    UIBuf.imu_data.clear();
    int cnt_imu = 0;

    auto reader = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = bag_file;
    storage_options.storage_id = "sqlite3";  // or whatever your storage backend is

    rosbag2_cpp::ConverterOptions converter_options;
    converter_options.input_serialization_format = "cdr";
    converter_options.output_serialization_format = "cdr";
    reader->open(storage_options, converter_options);

    while (reader->has_next()) {
        auto message = reader->read_next();
        if (!(message->topic_name).compare("/imu_data")) {
            rclcpp::Serialization<sensor_msgs::msg::Imu> serialization;
            auto imu_msg = std::make_shared<sensor_msgs::msg::Imu>();
            rclcpp::SerializedMessage serialized_data(*message->serialized_data);
            serialization.deserialize_message(&serialized_data, imu_msg.get());
            if (cnt_imu % 10 == 0)
            {
                Eigen::Vector3d acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
                acc *= 9.81;
                Eigen::Vector3d gyro(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
                gyro *= M_PI / 180.0;
                IMUData imu(rclcpp::Time(imu_msg->header.stamp).seconds(), acc, gyro);
                UIBuf.imu_data.push_back(imu);
            }
            cnt_imu++;            
        } else if (!(message->topic_name).compare("/tdoa_data")) {
            rclcpp::Serialization<cf_msgs::msg::Tdoa> serialization;
            auto tdoa_msg = std::make_shared<cf_msgs::msg::Tdoa>();
            rclcpp::SerializedMessage serialized_data(*message->serialized_data);
            serialization.deserialize_message(&serialized_data, tdoa_msg.get());   

            TDOAData tdoa(rclcpp::Time(tdoa_msg->header.stamp).seconds(), tdoa_msg->ida, tdoa_msg->idb, tdoa_msg->data);
            UIBuf.tdoa_data.push_back(tdoa);                     
        } else if (!(message->topic_name).compare("/pose_data")) {
            rclcpp::Serialization<geometry_msgs::msg::PoseWithCovarianceStamped> serialization;
            auto gt_msg = std::make_shared<geometry_msgs::msg::PoseWithCovarianceStamped>();
            rclcpp::SerializedMessage serialized_data(*message->serialized_data);
            serialization.deserialize_message(&serialized_data, gt_msg.get());   

            PointPose pose; pose.t = rclcpp::Time(gt_msg->header.stamp).seconds();
            pose.x = gt_msg->pose.pose.position.x; pose.y = gt_msg->pose.pose.position.y; pose.z = gt_msg->pose.pose.position.z;
            pose.qx = gt_msg->pose.pose.orientation.x; pose.qy = gt_msg->pose.pose.orientation.y; pose.qz = gt_msg->pose.pose.orientation.z; pose.qw = gt_msg->pose.pose.orientation.w;

            gtrPoseCloud->push_back(pose);
        }
    }
    reader->close();
    std::cout << "Loaded IMU messages: " << UIBuf.imu_data.size() << ". "
              << "TDOA messages: " << UIBuf.tdoa_data.size() << std::endl;
}

int main(int argc, char **argv)
{
    // Initalize ros nodes
    rclcpp::init(argc, argv);

    nh_ptr = rclcpp::Node::make_shared("gptrui");

    // Determine if we exit if no data is received after a while
    bool auto_exit = Util::GetBoolParam(nh_ptr, "auto_exit", false);

    // Parameters for the GP trajectory
    double gpQr_ = 1.0, gpQc_ = 1.0;
    Util::GetParam(nh_ptr, "gpQr", gpQr_);
    Util::GetParam(nh_ptr, "gpQc", gpQc_);
    gpQr = gpQr_ * Matrix3d::Identity(3, 3);
    gpQc = gpQc_ * Matrix3d::Identity(3, 3);

    double lie_epsilon = 1e-3;
    Util::GetParam(nh_ptr, "lie_epsilon", lie_epsilon);

    // Find the path to anchor position
    string anchor_pose_path;
    Util::GetParam(nh_ptr, "anchor_pose_path", anchor_pose_path);

    // Load the anchor pose
    std::map<uint16_t, Eigen::Vector3d> anc_pose_ = getAnchorListFromUTIL(anchor_pose_path);

    string bag_file;
    Util::GetParam(nh_ptr, "bag_file", bag_file);
    readBag(bag_file);

    Util::GetParam(nh_ptr, "w_tdoa", w_tdoa);
    Util::GetParam(nh_ptr, "GYR_N", GYR_N);
    Util::GetParam(nh_ptr, "ACC_N", ACC_N);
    if_save_traj = Util::GetBoolParam(nh_ptr, "if_save_traj", if_save_traj);
    Util::GetParam(nh_ptr, "traj_save_path", traj_save_path);

    double tskew0 = 1.0;
    double tskewmax = 1.0;
    double tskewstep = 0.1;
    Util::GetParam(nh_ptr, "tskew0", tskew0);
    Util::GetParam(nh_ptr, "tskewmax", tskewmax);
    Util::GetParam(nh_ptr, "tskewstep", tskewstep);  

    vector<double> Dtstep = {0.01};
    Util::GetParam(nh_ptr, "Dtstep", Dtstep);     
    
    // Create the trajectory
    GPUIPtr gpmui(new GPUI(nh_ptr));

    fs::create_directories(traj_save_path);
    std::ofstream logfile(traj_save_path + "/gptrui.csv", std::ios::out);
    logfile << std::fixed << std::setprecision(6);
    logfile << "tskew,dt,"   
               "so3xr3ap_tslv,so3xr3cf_tslv,se3ap_tslv,se3cf_tslv,"
               "so3xr3ap_JK,so3xr3cf_JK,se3ap_JK,se3cf_JK,"
               "so3xr3ap_rmse,so3xr3cf_rmse,se3ap_rmse,se3cf_rmse\n";
    
    auto AssessTraj = [&anc_pose_, &gpmui](UwbImuBuf &data, GaussianProcessPtr &traj, CloudPosePtr &gtrPoseCloud, map<string, double> &report) -> string
    {
        Eigen::Vector3d gravity_sum(0, 0, 0);
        size_t n_imu = 20;
        for (size_t i = 0; i < n_imu; i++) {
            gravity_sum += data.imu_data.at(i).acc;
        }
        gravity_sum /= n_imu;  
        g = gravity_sum;  
        std::cout << "g: " << g.transpose() << std::endl;

        bg = Eigen::Vector3d::Zero();
        ba = Eigen::Vector3d::Zero();        

        double t0 = data.minTime();
        traj->setStartTime(t0);
        // Set initial pose
        SE3d initial_pose;
        initial_pose.translation() = Eigen::Vector3d(0, 0, 0);
        traj->setKnot(0, GPState(t0, initial_pose));

        double newMaxTime = min(data.imu_data.back().t, data.tdoa_data.back().t);

        // Extend the trajectory
        traj->extendKnotsTo(newMaxTime, GPState(t0, initial_pose), false);

        // // Assign the trajectory by groundtruth
        // for(int kidx = 0; kidx < traj->getNumKnots(); kidx++)
        // {
        //     double tknot = traj->getKnotTime(kidx);
        //     for(auto &pose : gtrPoseCloud->points)
        //     {
        //         if (abs(pose.t - tknot) < traj->getDt())
        //         {
        //             traj->getKnotSO3(kidx) = SO3d(Quaternd(pose.qw, pose.qx, pose.qy, pose.qz));
        //             traj->getKnotPos(kidx) = Vector3d(pose.x, pose.y, pose.z);
        //             break;
        //         }
        //     }
        // }

        TicToc tt_solve;
        double tmin = traj->getMinTime();
        double tmax = traj->getKnotTime(traj->getNumKnots() - 1);
        double tmid = tmin;

        string report_;
        gpmui->Evaluate(tmin, tmax, tmid, traj, bg, ba, g,
                        data.tdoa_data, data.imu_data,
                        anc_pose_, P_I_tag, w_tdoa, GYR_N, ACC_N, 0.0, 0.0,
                        tdoa_loss_thres, mp_loss_thres, false,
                        gtrPoseCloud, report_, report);
        tt_solve.Toc();

        // RINFO("Traj: %.6f. Sw: %.3f -> %.3f. Data: %4d, %4d. Num knots: %d",
        //         traj->getMaxTime(), data.minTime(), data.maxTime(),
        //         data.tdoa_data.size(), data.imu_data.size(), traj->getNumKnots());

        return report_;
    };

    double tspan = 5.0;
    for(double &m : Dtstep)
    {
        for(double tskew = tskew0; tskew <= tskewmax; tskew += tskewstep)
        {
            // double tdatamin = gtrPoseCloud->points.front().t/tskew;
            // double tdatamax = max(tdatamin, gtrPoseCloud->points.back().t/tskew - tspan);

            // // Fixed seed for reproducibility
            // std::mt19937 rng(1102); 
            // std::uniform_real_distribution<double> t0udist(0, 1);

            // double tmin = tdatamin;
            // double tmax = tdatamax;

            // RINFO("tmin: %f, tmax: %f\n", tmin, tmax);

            // tmin /= tskew;
            // tmax /= tskew;

            CloudPosePtr gtrPoseCloud_scaled(new CloudPose());        
            // #pragma omp parallel for num_threads(MAX_THREADS)
            for(auto &pose : gtrPoseCloud->points)
            {
                double ts = pose.t / tskew;
                // if(ts > tmin && ts < tmax)
                {
                    auto pose_ = pose; pose_.t = ts;
                    gtrPoseCloud_scaled->push_back(pose_);
                }
            }

            UwbImuBuf UIBuf_scale;
            {
                std::mt19937 rng(1102); 
                std::uniform_real_distribution<double> sampler(0.0, 1.0);
                // #pragma omp parallel for num_threads(MAX_THREADS)
                for(auto &tdoadata : UIBuf.tdoa_data)
                {
                    // bool admitted = (sampler(rng) < 1.0/tskew);
                    // if (!admitted)
                    //     continue;
                    
                    double ts = tdoadata.t / tskew;
                    // if(ts > tmin && ts < tmax)
                    {
                        auto tdoadata_ = tdoadata; tdoadata_.t = ts;
                        UIBuf_scale.tdoa_data.push_back(tdoadata_);
                    }
                }
            }
            {
                std::mt19937 rng(4357); 
                std::uniform_real_distribution<double> sampler(0.0, 1.0);
                // #pragma omp parallel for num_threads(MAX_THREADS)
                for(auto &imudata : UIBuf.imu_data)
                {
                    // bool admitted = (sampler(rng) < 1.0/tskew);
                    // if (!admitted)
                    //     continue;

                    double ts = imudata.t / tskew;
                    // if(ts > tmin && ts < tmax)
                    {
                        auto imudata_ = imudata; imudata_.t = ts;
                        imudata_.acc *= (tskew*tskew);
                        imudata_.gyro *= tskew;
                        UIBuf_scale.imu_data.push_back(imudata_);
                    }
                }
            }

            double deltaTm = m;

            map<string, double> so3xr3ap_report;
            map<string, double> so3xr3cf_report;
            map<string, double> se3ap_report;
            map<string, double> se3cf_report;            

            GaussianProcessPtr trajSO3xR3AP(new GaussianProcess(deltaTm, gpQr, gpQc, false, POSE_GROUP::SO3xR3, lie_epsilon, true));
            GaussianProcessPtr trajSO3xR3CF(new GaussianProcess(deltaTm, gpQr, gpQc, false, POSE_GROUP::SO3xR3, lie_epsilon, false));
            GaussianProcessPtr trajSE3AP(new GaussianProcess(deltaTm, gpQr, gpQc, false, POSE_GROUP::SE3, lie_epsilon, true));
            GaussianProcessPtr trajSE3CF(new GaussianProcess(deltaTm, gpQr, gpQc, false, POSE_GROUP::SE3, lie_epsilon, false));

            string report_SO3xR3_by_SO3xR3AP = AssessTraj(UIBuf_scale, trajSO3xR3AP, gtrPoseCloud_scaled, so3xr3ap_report);
            // for(int kidx = 0; kidx < trajSO3xR3AP->getNumKnots(); kidx++)
            //     trajSO3xR3CF->setKnot(kidx, trajSO3xR3AP->getKnot(kidx));
            string report_SO3xR3_by_SO3xR3CF = AssessTraj(UIBuf_scale, trajSO3xR3CF, gtrPoseCloud_scaled, so3xr3cf_report);
            string report_SO3xR3_by_SE3AP___ = AssessTraj(UIBuf_scale, trajSE3AP, gtrPoseCloud_scaled, se3ap_report);
            string report_SO3xR3_by_SE3CF___ = AssessTraj(UIBuf_scale, trajSE3CF, gtrPoseCloud_scaled, se3cf_report);            

            RINFO("UIBTraj Dt=%2f, tskew: %.3f. %s", m, tskew, report_SO3xR3_by_SO3xR3AP.c_str());
            RINFO("UIBTraj Dt=%2f, tskew: %.3f. %s", m, tskew, report_SO3xR3_by_SO3xR3CF.c_str());
            RINFO("UIBTraj Dt=%2f, tskew: %.3f. %s", m, tskew, report_SO3xR3_by_SE3AP___.c_str());
            RINFO("UIBTraj Dt=%2f, tskew: %.3f. %s", m, tskew, report_SO3xR3_by_SE3CF___.c_str());
            RINFO("");            

            // Save the rmse result to the log
            logfile << tskew << ","
                    << deltaTm << ","
                    << so3xr3ap_report["tslv"] << ","
                    << so3xr3cf_report["tslv"] << ","
                    << se3ap_report["tslv"] << ","
                    << se3cf_report["tslv"] << ","
                    << so3xr3ap_report["JK"] << ","
                    << so3xr3cf_report["JK"] << ","
                    << se3ap_report["JK"] << ","
                    << se3cf_report["JK"] << ","
                    << so3xr3ap_report["rmse"] << ","
                    << so3xr3cf_report["rmse"] << ","
                    << se3ap_report["rmse"] << ","
                    << se3cf_report["rmse"]
                    << endl;
        }
    }    
    logfile.close();

    rclcpp::shutdown();

    RINFO(KGRN "Program Finsihed" RESET);
    return 0;
} 
