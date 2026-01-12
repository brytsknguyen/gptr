#pragma once

#include "DoubleSphereCameraModel.hpp"
#include "GaussNewtonUtilities.hpp"

#include "factor/GPMotionPriorTwoKnotsFactor.h"
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
    ParamInfoMap paramInfoMap;
    MarginalizationInfoPtr margInfo;

public:
    // Destructor
    ~GPMVICalib() {};

    // Constructor
    GPMVICalib(NodeHandlePtr &nh_) : nh(nh_) {};

    void AddTrajParams(
        ceres::Problem &problem, double tmin, double tmax, double tmid,
        GaussianProcessPtr &traj, int tidx,
        ParamInfoMap &paramInfoMap)
    {
        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = min(traj->getNumKnots() - 1, usmax.first + 1);

        for (int kidx = 0; kidx < traj->getNumKnots(); kidx++)
        {
            if (kidx < kidxmin || kidx > kidxmax)
                continue;

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

    void AddProjFactors(
        ceres::Problem &problem, double tmin, double tmax,
        GaussianProcessPtr &traj, CameraCalibration *cam_calib,
        const vector<CornerData> &corner_data_cam, std::map<int, Eigen::Vector3d> &corner_pos_3d, int cam_id,
        double w_corner, double proj_loss_thres,
        ParamInfoMap &paramInfo, FactorMeta &factorMeta)
    {
        auto usmin = traj->computeTimeIndex(tmin);
        auto usmax = traj->computeTimeIndex(tmax);

        int kidxmin = usmin.first;
        int kidxmax = min(traj->getNumKnots() - 1, usmax.first + 1);

        for (auto &corners : corner_data_cam)
        {
            if (!traj->TimeInInterval(corners.t, 1e-6))
                continue;

            auto us = traj->computeTimeIndex(corners.t);
            int u = us.first;
            double s = us.second;

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

            factor_param_blocks.push_back(cam_calib->T_i_c[cam_id].so3().data());
            factor_param_blocks.push_back(cam_calib->T_i_c[cam_id].translation().data());

            // Record the time stamp of the factor
            factorMeta.stamp.push_back(corners.t);

            factorMeta.coupled_params.back().push_back(paramInfoMap[cam_calib->T_i_c[cam_id].so3().data()]);
            factorMeta.coupled_params.back().push_back(paramInfoMap[cam_calib->T_i_c[cam_id].translation().data()]);

            ceres::LossFunction *proj_loss_function = proj_loss_thres == -1 ? NULL : new ceres::HuberLoss(proj_loss_thres);
            ceres::CostFunction *cost_function = new GPProjFactor(corners.proj, corners.id, cam_calib->intrinsics[cam_id], corner_pos_3d, w_corner, traj->getGPMixerPtr(), s);
            auto res = problem.AddResidualBlock(cost_function, proj_loss_function, factor_param_blocks);

            // Record the residual block
            factorMeta.res.push_back(res);
        }
    }

    void Evaluate(
        double tmin, double tmax, double tmid,
        GaussianProcessPtr &traj, Vector3d &XBIG, Vector3d &XBIA, Vector3d &g, CameraCalibration *cam_calib,
        const vector<IMUData> &imuData, const vector<CornerData> &corner_data_cam0, const vector<CornerData> &corner_data_cam1, std::map<int, Eigen::Vector3d> &corner_pos_3d,
        double w_corner, double wGyro, double wAcce, double wBiasGyro, double wBiasAcce, double corner_loss_thres, double mp_loss_thres, bool do_marginalization,
        const CloudPosePtr &gtrPoseCloud, map<string, double> &report_map, string &report_str)
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

            ceres::Manifold *so3parameterization = new GPSO3dLocalParameterization();

            for (int i = 0; i < cam_calib->T_i_c.size(); i++)
            {
                SO3d &R_i_c = cam_calib->T_i_c[i].so3();
                Vec3 &t_i_c = cam_calib->T_i_c[i].translation();

                problem.AddParameterBlock(R_i_c.data(), 4, so3parameterization);
                problem.AddParameterBlock(t_i_c.data(), 3);

                paramInfoMap.insert(R_i_c.data(), ParamInfo(R_i_c.data(), getEigenPtr(R_i_c), ParamType::SO3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1));
                paramInfoMap.insert(t_i_c.data(), ParamInfo(t_i_c.data(), getEigenPtr(t_i_c), ParamType::RV3, ParamRole::EXTRINSIC, paramInfoMap.size(), -1, -1, 1));
            }

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
        TicToc tt_addmp2k;
        RINFO("Add mp2k...");
        FactorMeta factorMetaMp2k;
        double cost_mp2k_init = -1, cost_mp2k_final = -1;
        AddMP2KFactors(problem, tmin, tmax, traj, mp_loss_thres, paramInfoMap, factorMetaMp2k);
        RINFO("Done, %.0fms", tt_addmp2k.Toc());

        TicToc tt_imu;
        RINFO("Add imu...");
        FactorMeta factorMetaIMU;
        double cost_imu_init = -1;
        double cost_imu_final = -1;
        AddIMUFactors(problem, tmin, tmax, traj, XBIG, XBIA, g, imuData, wGyro, wAcce, wBiasGyro, wBiasAcce, paramInfoMap, factorMetaIMU);
        RINFO("Done, %.0fms", tt_imu.Toc());

        // Add the projection factors
        TicToc tt_projcam0;
        RINFO("Add projcam0...");
        FactorMeta factorMetaProjCam0;
        double cost_proj_init0 = -1;
        double cost_proj_final0 = -1;
        AddProjFactors(problem, tmin, tmax, traj, cam_calib, corner_data_cam0, corner_pos_3d, 0, w_corner, corner_loss_thres, paramInfoMap, factorMetaProjCam0);
        RINFO("Done, %.0fms", tt_projcam0.Toc());

        TicToc tt_projcam1;
        RINFO("Add projcam1...");
        FactorMeta factorMetaProjCam1;
        double cost_proj_init1 = -1;
        double cost_proj_final1 = -1;
        AddProjFactors(problem, tmin, tmax, traj, cam_calib, corner_data_cam1, corner_pos_3d, 1, w_corner, corner_loss_thres, paramInfoMap, factorMetaProjCam1);
        RINFO("Done, %.0fms", tt_projcam1.Toc());

        tt_build.Toc();


        // Find the initial
        RINFO(KYEL"Solving..."RESET);
        Util::ComputeCeresCost(factorMetaMp2k.res, cost_mp2k_init, problem);
        Util::ComputeCeresCost(factorMetaIMU.res, cost_imu_init, problem);
        Util::ComputeCeresCost(factorMetaProjCam0.res, cost_proj_init0, problem);
        Util::ComputeCeresCost(factorMetaProjCam1.res, cost_proj_init1, problem);

        TicToc tt_solve;
        ceres::Solve(options, &problem, &summary);
        tt_solve.Toc();

        // std::cout << summary.FullReport() << std::endl;

        Util::ComputeCeresCost(factorMetaMp2k.res, cost_mp2k_final, problem);
        Util::ComputeCeresCost(factorMetaIMU.res, cost_imu_final, problem);
        Util::ComputeCeresCost(factorMetaProjCam0.res, cost_proj_final0, problem);
        Util::ComputeCeresCost(factorMetaProjCam1.res, cost_proj_final1, problem);
        RINFO(KGRN"Done. %fms"RESET, tt_solve.GetLastStop());

        // RINFO("Factors: MP2K: %d, Proj0: %d, Proj1: %d, IMU: %d.",
        //       factorMetaMp2k.size(), factorMetaProjCam0.size(), factorMetaProjCam1.size(), factorMetaIMU.size());

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

        report_str = myprintf(
            "Pose group: %s. Method: %s. Dt: %.3f. "
            "Tslv: %.0f. Iterations: %d.\n"
            "Factors: MP2K: %05d, IMU: %05d, Cam0Proj: %05d, Cam1Proj: %05d\n"
            "J0: %16.3f, MP2K: %16.3f, IMU: %7.3f. Cam0Proj: %7.3f. Cam1Proj: %7.3f.\n"
            "JK: %16.3f, MP2K: %16.3f, IMU: %7.3f. Cam0Proj: %7.3f. Cam1Proj: %7.3f.\n"
            "RMSE: POS: %.12f. POSE: %.12f.\n"
            ,
            traj->getGPMixerPtr()->getPoseRepresentation() == POSE_GROUP::SO3xR3 ? "SO3xR3" : "SE3",
            traj->getGPMixerPtr()->getJacobianForm() ? "AP" : "CF",
            traj->getDt(),
            tt_solve.GetLastStop(), summary.iterations.size(),
            factorMetaMp2k.size(), factorMetaIMU.size(), factorMetaProjCam0.size(), factorMetaProjCam1.size(),
            summary.initial_cost, cost_mp2k_init,  cost_imu_init,  cost_proj_init0,  cost_proj_init1,
            summary.final_cost,   cost_mp2k_final, cost_imu_final, cost_proj_final0, cost_proj_final1,
            pos_rmse, se3_rmse
        );

        // cout << "yolo " << report_ << endl;

        report_map["iter"]    = summary.iterations.size();
        report_map["tslv"]    = summary.total_time_in_seconds;
        report_map["rmse"]    = pos_rmse;
        report_map["J0"]      = summary.initial_cost;
        report_map["JK"]      = summary.final_cost;
        report_map["MP2KJ0"]  = cost_mp2k_init;
        report_map["IMUJ0"]   = cost_imu_init;
        report_map["CAM0J0"]  = cost_proj_init0;
        report_map["CAM1J0"]  = cost_proj_init1;
        report_map["MP2KJK"]  = cost_mp2k_final;
        report_map["IMUJK"]   = cost_imu_final;
        report_map["CAM0JK"]  = cost_proj_final0;
        report_map["CAM1JK"]  = cost_proj_final1;

        RINFO(KGRN"Done. %fms\n"RESET, tt_report.Toc());

    }
};

typedef std::shared_ptr<GPMVICalib> GPMVICalibPtr;
