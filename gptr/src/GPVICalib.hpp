#pragma once

#include "DoubleSphereCameraModel.hpp"
#include "GaussNewtonUtilities.hpp"

#include "factor/GPMotionPriorTwoKnotsFactorUI.h"
#include "factor/GPIMUFactor.h"
#include "factor/GPProjectionFactor.h"

#include "utility.h"

class GPMVICalib
{

private:

    // Node handle to get information needed
    NodeHandlePtr nh;

    // Map of traj-kidx and parameter id
    map<pair<int, int>, int> tk2p;
    map<double*, ParamInfo> paramInfoMap;
    MarginalizationInfoPtr margInfo;

public:

    // Destructor
   ~GPMVICalib() {};
   
    // Constructor
    GPMVICalib(NodeHandlePtr &nh_) : nh(nh_) {};

    void AddTrajParams(ceres::Problem &problem,
        GaussianProcessPtr &traj, int tidx,
        map<double*, ParamInfo> &paramInfoMap,
        double tmin, double tmax, double tmid)
    {
        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = usmax.first+1;

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
        map<double*, ParamInfo> &paramInfoMap, FactorMeta &factorMeta,
        double tmin, double tmax, double mp_loss_thres)
    {
        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = usmax.first+1;      
        // Add the GP factors based on knot difference
        for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
        {
            if (kidx < kidxmin || kidx+1 > kidxmax) {        
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
            factorMeta.stamp.push_back(traj->getKnotTime(kidx+1));
        }
    }

    void AddIMUFactors(ceres::Problem &problem, GaussianProcessPtr &traj, Vector3d &XBIG, Vector3d &XBIA, Vector3d &g,
        map<double*, ParamInfo> &paramInfo, FactorMeta &factorMeta,
        const vector<IMUData> &imuData, double tmin, double tmax, 
        double wGyro, double wAcce, double wBiasGyro, double wBiasAcce)
    {
        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = usmax.first+1;  
        for (auto &imu : imuData)
        {
            if (!traj->TimeInInterval(imu.t, 1e-6)) {
                continue;
            }
            
            auto   us = traj->computeTimeIndex(imu.t);
            int    u  = us.first;
            double s  = us.second;

            if (u < kidxmin || u+1 > kidxmax) {
                continue;
            }
      
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
            ceres::CostFunction *cost_function = new GPIMUFactor(imu.acc, imu.gyro, XBIA, XBIG, wGyro, wAcce, wBiasGyro, wBiasAcce, traj->getGPMixerPtr(), s);
            auto res = problem.AddResidualBlock(cost_function, imu_loss_function, factor_param_blocks);

            // Record the residual block
            factorMeta.res.push_back(res);                

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(imu.t);
        }
    }

    void AddProjFactors(
        ceres::Problem &problem, GaussianProcessPtr &traj, 
        map<double*, ParamInfo> &paramInfo, FactorMeta &factorMeta,
        const vector<CornerData> &corner_data_cam, std::map<int, Eigen::Vector3d> &corner_pos_3d, 
        CameraCalibration *cam_calib, int cam_id, 
        double tmin, double tmax, double w_corner, double proj_loss_thres)
    {

        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = usmax.first+1;        
        for (auto &corners : corner_data_cam)
        {
            if (!traj->TimeInInterval(corners.t, 1e-6)) {
                continue;
            }
            
            auto   us = traj->computeTimeIndex(corners.t);
            int    u  = us.first;
            double s  = us.second;

            if (u < kidxmin || u+1 > kidxmax) {
                continue;
            }

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

            factor_param_blocks.push_back(cam_calib->T_i_c[cam_id].so3().data());
            factor_param_blocks.push_back(cam_calib->T_i_c[cam_id].translation().data());

            factorMeta.coupled_params.back().push_back(paramInfoMap[cam_calib->T_i_c[cam_id].so3().data()]);
            factorMeta.coupled_params.back().push_back(paramInfoMap[cam_calib->T_i_c[cam_id].translation().data()]);
            ceres::LossFunction *proj_loss_function = proj_loss_thres == -1 ? NULL : new ceres::HuberLoss(proj_loss_thres);
            ceres::CostFunction *cost_function = new GPProjFactor(corners.proj, corners.id, cam_calib->intrinsics[cam_id], corner_pos_3d, w_corner, traj->getGPMixerPtr(), s);
            auto res = problem.AddResidualBlock(cost_function, proj_loss_function, factor_param_blocks);
            // Record the residual block
            factorMeta.res.push_back(res);

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(corners.t);
        }
    }

    void Evaluate(GaussianProcessPtr &traj, Vector3d &XBIG, Vector3d &XBIA, Vector3d &g, CameraCalibration *cam_calib,
        double tmin, double tmax, double tmid,
        const vector<CornerData> &corner_data_cam0, const vector<CornerData> &corner_data_cam1, const vector<IMUData> &imuData,
        std::map<int, Eigen::Vector3d> &corner_pos_3d, 
        bool do_marginalization, double w_corner, double wGyro, double wAcce, double wBiasGyro, double wBiasAcce, double corner_loss_thres, double mp_loss_thres)
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

            ceres::LocalParameterization *so3parameterization = new GPSO3dLocalParameterization();

            for (int i = 0; i < cam_calib->T_i_c.size(); i++) {
                problem.AddParameterBlock(cam_calib->T_i_c[i].so3().data(), 4, so3parameterization);
                problem.AddParameterBlock(cam_calib->T_i_c[i].translation().data(), 3);

                paramInfoMap.insert(make_pair(cam_calib->T_i_c[i].so3().data(), ParamInfo(cam_calib->T_i_c[i].so3().data(), ParamType::SO3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1)));
                paramInfoMap.insert(make_pair(cam_calib->T_i_c[i].translation().data(), ParamInfo(cam_calib->T_i_c[i].translation().data(), ParamType::RV3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1)));
            }
            
            // Sanity check
            for(auto &param_ : paramInfoMap)
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

        // Add the projection factors
        FactorMeta factorMetaProjCam0;
        double cost_proj_init0 = -1; double cost_proj_final0 = -1;
        AddProjFactors(problem, traj, paramInfoMap, factorMetaProjCam0, corner_data_cam0, corner_pos_3d, cam_calib, 0, tmin, tmax, w_corner, corner_loss_thres);

        FactorMeta factorMetaProjCam1;
        double cost_proj_init1 = -1; double cost_proj_final1 = -1;
        AddProjFactors(problem, traj, paramInfoMap, factorMetaProjCam1, corner_data_cam1, corner_pos_3d, cam_calib, 1, tmin, tmax, w_corner, corner_loss_thres);

        FactorMeta factorMetaIMU;
        double cost_imu_init = -1; double cost_imu_final = -1;
        AddIMUFactors(problem, traj, XBIG, XBIA, g, paramInfoMap, factorMetaIMU, imuData, tmin, tmax, wGyro, wAcce, wBiasGyro, wBiasAcce);

        tt_build.Toc();

        TicToc tt_slv;

        // Find the initial cost
        Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_init,  problem);
        Util::ComputeCeresCost(factorMetaProjCam0.res, cost_proj_init0, problem);
        Util::ComputeCeresCost(factorMetaProjCam1.res, cost_proj_init1, problem);
        Util::ComputeCeresCost(factorMetaIMU.res,   cost_imu_init,   problem);

        ceres::Solve(options, &problem, &summary);

        std::cout << summary.FullReport() << std::endl;

        Util::ComputeCeresCost(factorMetaMp2k.res,  cost_mp2k_final,  problem);
        Util::ComputeCeresCost(factorMetaProjCam0.res, cost_proj_final0, problem);
        Util::ComputeCeresCost(factorMetaProjCam1.res, cost_proj_final1, problem);
        Util::ComputeCeresCost(factorMetaIMU.res,   cost_imu_final,   problem);
        

        tt_slv.Toc();
    }

};

typedef std::shared_ptr<GPMVICalib> GPMVICalibPtr;
