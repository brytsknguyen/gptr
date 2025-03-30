#include "unistd.h"
#include <algorithm> // for std::sort

// ROS utilities
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/odometry.hpp"

// Custom built utilities
#include "GaussianProcess.hpp"
#include "GaussNewtonUtilities.hpp"

#include "factor/GPMotionPriorTwoKnotsFactor.h"
#include "factor/GPTWRFactor.h"
#include "utility.h"

using namespace std;

NodeHandlePtr nh_ptr;

// Ground truth trajectory

double wqx1 = 3*5.0;
double wqy1 = 3*5.0;
double wqz1 = 1*5.0;
double wpx1 = 3*0.15;
double wpy1 = 3*0.15;
double wpz1 = 1*0.15;

double rqx1 = M_PI*0.5;
double rqy1 = M_PI*0.5;
double rqz1 = M_PI*sqrt(3)/2;
double rpx1 = 5.0;
double rpy1 = 5.0;
double rpz1 = 5.0;

// double wqx2 = 3*5.0;
// double wqy2 = 3*5.0;
// double wqz2 = 1*5.0;
double wpx2 = 3*0.15;
double wpy2 = 3*0.15;
double wpz2 = 1*0.15;

// double rqx2 = M_PI*0.5;
// double rqy2 = M_PI*0.5;
// double rqz2 = M_PI*sqrt(3)/2;
double rpx2 = 5.0;
double rpy2 = 5.0;
double rpz2 = 5.0;

class GtTrajSO3xR3
{
public:

   ~GtTrajSO3xR3() {};
    GtTrajSO3xR3() {};

    myTf<double> pose(double t) const
    {
        Quaternd Q = SO3d::exp(Vector3d(rqx1*cos(wqx1*t + 57),
                                        rqy1*sin(wqy1*t + 57),
                                        rqz1*sin(wqz1*t + 43))).unit_quaternion();
        Vector3d P( rpx1*sin(wpx1*t + 43),
                    rpy1*cos(wpy1*t + 43),
                    rpz1*cos(wpz1*t + 57));
        return myTf(Q, P);
    }

private:
    // double r1 = M_PI*0.5;
    // double r2 = M_PI*sqrt(3)/2;
    // double wr = 1.0;
    // double wp = 0.015;
};

class GtTrajSE3
{
public:

   ~GtTrajSE3() {};
    GtTrajSE3()
        : gpm(GPMixer())
    {};

    GPMixer gpm;
            
    myTf<double> pose(double t) const
    {
        Vector3d P( rpx2*sin(wpx2*t + 43),
                    rpy2*cos(wpy2*t + 43),
                    rpz2*cos(wpz2*t + 57));
        Vector3d V( rpx2*wpx2*cos(wpx2*t + 43),
                   -rpy2*wpy2*sin(wpy2*t + 43),
                   -rpz2*wpz2*sin(wpz2*t + 57));
        Vector3d ex = V/V.norm();
        Vector3d rn = P/P.norm();
        Vector3d ez = SO3d::hat(rn)*ex; ez = ez/ez.norm();
        Vector3d ey = SO3d::hat(ez)*ex; ey = ey/ey.norm();
        Mat3 R;
        R.col(0) = ex;
        R.col(1) = ey;
        R.col(2) = ez;
        return myTf(R, P);
    }

private:
    // double r1 = M_PI*0.5;
    // double r2 = M_PI*sqrt(3)/2;
    // double wr = 1.0;
    // double wp = 0.015;
};

// Create the ground truth trajectory
GtTrajSO3xR3 gtTrajSO3xR3;
GtTrajSE3 gtTrajSE3;




// Trajectory estimate params

double maxTime = 69.0;
double deltaT = 0.1;
double mpCovROSJerk = 5.0;
double mpCovPVAJerk = 5.0;
double lie_epsilon  = 1e-3;
POSE_GROUP pose_type = POSE_GROUP::SO3xR3;

int max_ceres_iter = 100;



// UWB params

// Anchor and tag configs
vector<Vector3d> tags = {
    // Vector3d( 0.2,  0.2, 0.0),
    // Vector3d(-0.2,  0.2, 0.0),
    // Vector3d(-0.2, -0.2, 0.0),
    // Vector3d( 0.2, -0.2, 0.0)};
    Vector3d( 0.2, 0.0, 0.0),
    Vector3d(-0.2, 0.0, 0.0)
};

vector<Vector3d> anchors = {
    Vector3d( 10.0,  10.0, 0.5),
    Vector3d(-10.0,  10.0, 1.5),
    Vector3d(-10.0, -10.0, 0.5),
    Vector3d( 10.0, -10.0, 1.5)};

double uwb_rate = 200.0;
double uwb_noise = 0.05;


// Creating the param info
ParamInfoMap paramInfoMap;



void AddTrajParams(ceres::Problem &problem, GaussianProcessPtr &traj)
{
    for (int kidx = 0; kidx < traj->getNumKnots(); kidx++)
    {
        problem.AddParameterBlock(traj->getKnotSO3(kidx).data(), 4, new GPSO3dLocalParameterization());
        problem.AddParameterBlock(traj->getKnotOmg(kidx).data(), 3);
        problem.AddParameterBlock(traj->getKnotAlp(kidx).data(), 3);
        problem.AddParameterBlock(traj->getKnotPos(kidx).data(), 3);
        problem.AddParameterBlock(traj->getKnotVel(kidx).data(), 3);
        problem.AddParameterBlock(traj->getKnotAcc(kidx).data(), 3);

        paramInfoMap.insert(traj->getKnotSO3(kidx).data(), ParamInfo(traj->getKnotSO3(kidx).data(), getEigenPtr(traj->getKnotSO3(kidx)), ParamType::SO3, ParamRole::GPSTATE, paramInfoMap.size(), 0, kidx, 0));
        paramInfoMap.insert(traj->getKnotOmg(kidx).data(), ParamInfo(traj->getKnotOmg(kidx).data(), getEigenPtr(traj->getKnotOmg(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), 0, kidx, 1));
        paramInfoMap.insert(traj->getKnotAlp(kidx).data(), ParamInfo(traj->getKnotAlp(kidx).data(), getEigenPtr(traj->getKnotAlp(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), 0, kidx, 2));
        paramInfoMap.insert(traj->getKnotPos(kidx).data(), ParamInfo(traj->getKnotPos(kidx).data(), getEigenPtr(traj->getKnotPos(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), 0, kidx, 3));
        paramInfoMap.insert(traj->getKnotVel(kidx).data(), ParamInfo(traj->getKnotVel(kidx).data(), getEigenPtr(traj->getKnotVel(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), 0, kidx, 4));
        paramInfoMap.insert(traj->getKnotAcc(kidx).data(), ParamInfo(traj->getKnotAcc(kidx).data(), getEigenPtr(traj->getKnotAcc(kidx)), ParamType::RV3, ParamRole::GPSTATE, paramInfoMap.size(), 0, kidx, 5));
    }
}

void AddMP2KFactors(ceres::Problem &problem, GaussianProcessPtr &traj, FactorMeta &factorMeta)
{
    for (int kidx = 0; kidx < traj->getNumKnots()-1; kidx++)
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
        ceres::CostFunction *cost_function = new GPMotionPriorTwoKnotsFactor(traj->getGPMixerPtr());
        auto res_block = problem.AddResidualBlock(cost_function, NULL, factor_param_blocks);
        
        // Record the residual block
        factorMeta.res.push_back(res_block);
    }
}

template <typename T>
void AddUWBFactors(ceres::Problem &problem, GaussianProcessPtr &traj, const T &trajGtr, FactorMeta &factorMeta)
{
    // Random number generator with fixed seed for reproducibility
    std::mt19937 rng(57);  
    std::normal_distribution<double> noise_dist(0.0, uwb_noise);

    for (double ts = traj->getMinTime() + traj->getDt()/10.0; ts < traj->getMaxTime(); ts+=(1.0/uwb_rate))
    {
        myTf tf_W_B = trajGtr.pose(ts);

        auto   us = traj->computeTimeIndex(ts);
        int    u  = us.first;
        double s  = us.second;

        for(int tidx = 0; tidx < tags.size(); tidx++)
        {
            for(int aidx = 0; aidx < anchors.size(); aidx++)
            {
                Vector3d pos_tag = tags[tidx];
                Vector3d pos_anc = anchors[aidx];

                double twr = (tf_W_B.rot*pos_tag + tf_W_B.pos - pos_anc).norm() + noise_dist(rng);

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
                factorMeta.stamp.push_back(ts);

                ceres::CostFunction *cost_function = new GPTWRFactor(twr, pos_anc, pos_tag, 1.0, traj->getGPMixerPtr(), s);
                auto res = problem.AddResidualBlock(cost_function, NULL, factor_param_blocks);
                
                // Record the residual block
                factorMeta.res.push_back(res);
            }
        }
    }
}

template <typename T>
void InitializeTrajEst(GaussianProcessPtr &traj, const T &gtTraj)
{
    traj->setStartTime(0);
    traj->setKnot(0, GPState(0));

    // Extend the trajectory to maxTime
    RINFO("Extending trajectory %s-%s.",
          traj->getGPMixerPtr()->getPoseRepresentation() == POSE_GROUP::SO3xR3 ? "SO3xR3" : "SE3",
          traj->getGPMixerPtr()->getJacobianForm() ? "AP" : "CF");

    while(traj->getMaxTime() < maxTime)
        traj->extendOneKnot(traj->getKnot(traj->getNumKnots()-1));

    // Fixed seed for reproducibility
    std::mt19937 rng(43); 
    // Define a uniform distribution (e.g., integers between 1 and 100)
    std::normal_distribution<double> rdist(0.0, 0.1);
    std::normal_distribution<double> pdist(0.0, 0.2);

    // Initialize the pose with erred ground truth
    for(int kidx = 0; kidx < traj->getNumKnots(); kidx++)
    {
        double ts = traj->getKnotTime(kidx);
        myTf pose = gtTraj.pose(ts);

        Vector3d theErr(rdist(rng), rdist(rng), rdist(rng));
        Vector3d posErr(pdist(rng), pdist(rng), pdist(rng));
        myTf pose_err(SO3d::exp(theErr).unit_quaternion(), posErr);

        traj->setKnot(kidx, GPState(traj->getMaxTime(), (pose*pose_err).getSE3()));
    }
}


template <typename T>
string AssessTraj(GaussianProcessPtr &traj, const T &trajGtr)
{
    paramInfoMap.clear();

    // Ceres problem
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    // Set up the ceres problem
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = MAX_THREADS;
    options.max_num_iterations = max_ceres_iter;

    // Add trajectory params
    TicToc tt_addparam;
    RINFO("Add params...");
    AddTrajParams(problem, traj);
    RINFO("Done, %.0f\n", tt_addparam.Toc());

    // Add the motion prior factors
    TicToc tt_addmp2k;
    RINFO("Add mp2k...");
    FactorMeta mp2kFactorMeta;
    double cost_mp2k_init = -1, cost_mp2k_final = -1;
    AddMP2KFactors(problem, traj, mp2kFactorMeta);
    RINFO("Done, %.0f\n", tt_addmp2k.Toc());

    // Add the UWB factors
    TicToc tt_uwb;
    RINFO("Add uwb...");
    FactorMeta uwbFactorMeta;
    double cost_uwb_init = -1, cost_uwb_final = -1;
    AddUWBFactors(problem, traj, trajGtr, uwbFactorMeta);
    RINFO("Done, %.0f\n", tt_uwb.Toc());

    TicToc tt_solve;
    RINFO(KYEL"Solving..."RESET);
    Util::ComputeCeresCost(mp2kFactorMeta.res, cost_mp2k_init, problem);
    Util::ComputeCeresCost(uwbFactorMeta.res,  cost_uwb_init,  problem);
    ceres::Solve(options, &problem, &summary);
    Util::ComputeCeresCost(mp2kFactorMeta.res, cost_mp2k_final, problem);
    Util::ComputeCeresCost(uwbFactorMeta.res,  cost_uwb_final,  problem);
    RINFO(KGRN"Done. %f"RESET, tt_solve.Toc());


    // Calculate the pose errors
    vector<Vector3d> pos_err;
    vector<Vector3d> se3_err;
    for(double ts = traj->getMinTime() + traj->getDt()/0.5; ts < traj->getMaxTime(); ts += traj->getDt())
    {
        myTf poseEst = myTf(traj->pose(ts));
        myTf poseGtr = trajGtr.pose(ts);
        pos_err.push_back(poseGtr.pos - poseEst.pos);
        se3_err.push_back((poseEst.inverse()*poseGtr).pos);
    }

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
    
    string report = myprintf(
        "Pose group: %s. Method: %s. Dt: %.3f."
        "Tslv: %.0f. Iterations: %d.\n"
        "Factors: MP2K: %05d, UWB: %05d.\n"
        "J0: %16.3f, MP2K: %16.3f, UWB: %7.3f.\n"
        "JK: %16.3f, MP2K: %16.3f, UWB: %7.3f.\n"
        "RMSE: POS: %9.3f. POSE: %9.3f.\n"
        ,
        traj->getGPMixerPtr()->getPoseRepresentation() == POSE_GROUP::SO3xR3 ? "SO3xR3" : "SE3",
        traj->getGPMixerPtr()->getJacobianForm() ? "AP" : "CF",
        traj->getDt(),
        tt_solve.GetLastStop(), summary.iterations.size(),
        mp2kFactorMeta.size(), uwbFactorMeta.size(),
        summary.initial_cost, cost_mp2k_init,  cost_uwb_init,
        summary.final_cost,   cost_mp2k_final, cost_uwb_final,
        pos_rmse, se3_rmse
    );

    return report;
}



int main(int argc, char **argv)
{
    // Initalize ros nodes
    rclcpp::init(argc, argv);
    nh_ptr = rclcpp::Node::make_shared("GPTRPP");



    // Get the params for ground truth

    Util::GetParam(nh_ptr, "wqx1", wqx1);
    Util::GetParam(nh_ptr, "wqy1", wqy1);
    Util::GetParam(nh_ptr, "wqz1", wqz1);
    Util::GetParam(nh_ptr, "wpx1", wpx1);
    Util::GetParam(nh_ptr, "wpy1", wpy1);
    Util::GetParam(nh_ptr, "wpz1", wpz1);
    Util::GetParam(nh_ptr, "rqx1", rqx1);
    Util::GetParam(nh_ptr, "rqy1", rqy1);
    Util::GetParam(nh_ptr, "rqz1", rqz1);
    Util::GetParam(nh_ptr, "rpx1", rpx1);
    Util::GetParam(nh_ptr, "rpy1", rpy1);
    Util::GetParam(nh_ptr, "rpz1", rpz1);

    Util::GetParam(nh_ptr, "wpx2", wpx2);
    Util::GetParam(nh_ptr, "wpy2", wpy2);
    Util::GetParam(nh_ptr, "wpz2", wpz2);
    Util::GetParam(nh_ptr, "rpx2", rpx2);
    Util::GetParam(nh_ptr, "rpy2", rpy2);
    Util::GetParam(nh_ptr, "rpz2", rpz2);

    RINFO("Found param wqx1: %f", wqx1);
    RINFO("Found param wqy1: %f", wqy1);
    RINFO("Found param wqz1: %f", wqz1);
    RINFO("Found param wpx1: %f", wpx1);
    RINFO("Found param wpy1: %f", wpy1);
    RINFO("Found param wpz1: %f", wpz1);
    RINFO("Found param rqx1: %f", rqx1);
    RINFO("Found param rqy1: %f", rqy1);
    RINFO("Found param rqz1: %f", rqz1);
    RINFO("Found param rpx1: %f", rpx1);
    RINFO("Found param rpy1: %f", rpy1);
    RINFO("Found param rpz1: %f", rpz1);

    RINFO("Found param wpx2: %f", wpx2);
    RINFO("Found param wpy2: %f", wpy2);
    RINFO("Found param wpz2: %f", wpz2);
    RINFO("Found param rpx2: %f", rpx2);
    RINFO("Found param rpy2: %f", rpy2);
    RINFO("Found param rpz2: %f", rpz2);



    // Get params and config the trajectory

    // Get the params
    Util::GetParam(nh_ptr, "maxTime", maxTime);
    Util::GetParam(nh_ptr, "deltaT", deltaT);
    Util::GetParam(nh_ptr, "mpCovROSJerk", mpCovROSJerk);
    Util::GetParam(nh_ptr, "mpCovPVAJerk", mpCovPVAJerk);
    Util::GetParam(nh_ptr, "lie_epsilon", lie_epsilon);
    string pose_type_; Util::GetParam(nh_ptr, "pose_type", pose_type_); pose_type = pose_type_ == "SE3" ? POSE_GROUP::SE3 : POSE_GROUP::SO3xR3;
    bool use_approx_drv =  Util::GetBoolParam(nh_ptr, "use_approx_drv", true);
    Util::GetParam(nh_ptr, "max_ceres_iter", max_ceres_iter);

    // Report the param values
    RINFO("Trajectory max time:   %f.", maxTime);
    RINFO("DeltaT:                %f.", deltaT);
    RINFO("mpCovROSJerk:          %f.", mpCovROSJerk);
    RINFO("mpCovPVAJerk:          %f.", mpCovPVAJerk);
    RINFO("Pose representation:   %s. Num: %d.", pose_type_.c_str(), pose_type);
    RINFO("use_approx_drv set to  %d.", use_approx_drv);
    RINFO("Maximu ceres iteraton: %d.", max_ceres_iter);

    // Creating the trajectory
    Matrix3d CovROSJerk = Vector3d(mpCovROSJerk, mpCovROSJerk, mpCovROSJerk).asDiagonal();
    Matrix3d CovPVAJerk = Vector3d(mpCovPVAJerk, mpCovPVAJerk, mpCovPVAJerk).asDiagonal();




    // Get param of UWB
    Util::GetParam(nh_ptr, "uwb_rate", uwb_rate);
    Util::GetParam(nh_ptr, "uwb_noise", uwb_noise);
    
    RINFO("UWB rate: %f", uwb_rate);
    RINFO("UWB noise: %f", uwb_noise);





    // Creating the trajectory and assess

    GaussianProcessPtr trajSO3xR3AP(new GaussianProcess(deltaT, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SO3xR3, lie_epsilon, true));
    GaussianProcessPtr trajSO3xR3CF(new GaussianProcess(deltaT, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SO3xR3, lie_epsilon, false));
    GaussianProcessPtr trajSE3AP(new GaussianProcess(deltaT, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SE3, lie_epsilon, true));
    GaussianProcessPtr trajSE3CF(new GaussianProcess(deltaT, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SE3, lie_epsilon, false));

    InitializeTrajEst(trajSO3xR3AP, gtTrajSO3xR3);
    InitializeTrajEst(trajSO3xR3CF, gtTrajSO3xR3);
    InitializeTrajEst(trajSE3AP, gtTrajSO3xR3);
    InitializeTrajEst(trajSE3CF, gtTrajSO3xR3);

    // Confirm that the trajectories have the same control points
    for(int kidx = 0; kidx < trajSO3xR3AP->getNumKnots(); kidx++)
    {
        SO3d dR = (trajSO3xR3AP->getKnotSO3(kidx).inverse() * trajSO3xR3CF->getKnotSO3(kidx));
        ROS_ASSERT_MSG(dR.log().norm() < 1.0e-9, "%f", dR.log().norm());

        assert((trajSO3xR3AP->getKnotOmg(kidx) - trajSO3xR3CF->getKnotOmg(kidx)).norm() == 0.0);
        assert((trajSO3xR3AP->getKnotAlp(kidx) - trajSO3xR3CF->getKnotAlp(kidx)).norm() == 0.0);
        assert((trajSO3xR3AP->getKnotPos(kidx) - trajSO3xR3CF->getKnotPos(kidx)).norm() == 0.0);
        assert((trajSO3xR3AP->getKnotVel(kidx) - trajSO3xR3CF->getKnotVel(kidx)).norm() == 0.0);
        assert((trajSO3xR3AP->getKnotAcc(kidx) - trajSO3xR3CF->getKnotAcc(kidx)).norm() == 0.0);
    }
    
    // Assess with the SO3xR3 trajectory
    string report_SO3xR3_by_SO3xR3AP = AssessTraj(trajSO3xR3AP, gtTrajSO3xR3);
    string report_SO3xR3_by_SO3xR3CF = AssessTraj(trajSO3xR3CF, gtTrajSO3xR3);
    string report_SO3xR3_by_SE3AP___ = AssessTraj(trajSE3AP,    gtTrajSO3xR3);
    string report_SO3xR3_by_SE3CF___ = AssessTraj(trajSE3CF,    gtTrajSO3xR3);

    RINFO("SO3xR3Traj %s", report_SO3xR3_by_SO3xR3AP.c_str());
    RINFO("SO3xR3Traj %s", report_SO3xR3_by_SO3xR3CF.c_str());
    RINFO("SO3xR3Traj %s", report_SO3xR3_by_SE3AP___.c_str());
    RINFO("SO3xR3Traj %s", report_SO3xR3_by_SE3CF___.c_str());

    InitializeTrajEst(trajSO3xR3AP, gtTrajSE3);
    InitializeTrajEst(trajSO3xR3CF, gtTrajSE3);
    InitializeTrajEst(trajSE3AP, gtTrajSE3);
    InitializeTrajEst(trajSE3CF, gtTrajSE3);

    // Assess with the SE3 trajectory
    string report_SE3_by_SO3xR3AP = AssessTraj(trajSO3xR3AP, gtTrajSE3);
    string report_SE3_by_SO3xR3CF = AssessTraj(trajSO3xR3CF, gtTrajSE3);
    string report_SE3_by_SE3AP___ = AssessTraj(trajSE3AP,    gtTrajSE3);
    string report_SE3_by_SE3CF___ = AssessTraj(trajSE3CF,    gtTrajSE3);

    RINFO("SE3Traj %s", report_SE3_by_SO3xR3AP.c_str());
    RINFO("SE3Traj %s", report_SE3_by_SO3xR3CF.c_str());
    RINFO("SE3Traj %s", report_SE3_by_SE3AP___.c_str());
    RINFO("SE3Traj %s", report_SE3_by_SE3CF___.c_str());



    // Visualizing the result

    rclcpp::Publisher<RosPc2Msg>::SharedPtr cloudTrajEstSO3xR3APPub = nh_ptr->create_publisher<RosPc2Msg>("/gp_traj_est_so3xr3_cf", 1);
    rclcpp::Publisher<RosPc2Msg>::SharedPtr cloudTrajEstSO3xR3CFPub = nh_ptr->create_publisher<RosPc2Msg>("/gp_traj_est_so3xr3_ap", 1);
    rclcpp::Publisher<RosPc2Msg>::SharedPtr cloudTrajEstSE3APPub = nh_ptr->create_publisher<RosPc2Msg>("/gp_traj_est_se3_cf", 1);
    rclcpp::Publisher<RosPc2Msg>::SharedPtr cloudTrajEstSE3CFPub = nh_ptr->create_publisher<RosPc2Msg>("/gp_traj_est_se3_ap", 1);

    rclcpp::Publisher<RosPc2Msg>::SharedPtr cloudTrajGtrSO3xR3Pub = nh_ptr->create_publisher<RosPc2Msg>("/gp_trajso3xr3_gtr", 1);
    rclcpp::Publisher<RosPc2Msg>::SharedPtr cloudTrajGtrSE3Pub = nh_ptr->create_publisher<RosPc2Msg>("/gp_trajse3_gtr", 1);
    
    rclcpp::Publisher<RosOdomMsg>::SharedPtr odomTrajGtrSO3xR3Pub = nh_ptr->create_publisher<RosOdomMsg>("/gp_trajso3xr3_gtr_odom", 1);
    rclcpp::Publisher<RosOdomMsg>::SharedPtr odomTrajGtrSE3Pub = nh_ptr->create_publisher<RosOdomMsg>("/gp_trajse3_gtr_odom", 1);

    CloudPosePtr cloudTrajEstSO3xR3AP = CloudPosePtr(new CloudPose());
    CloudPosePtr cloudTrajEstSO3xR3CF = CloudPosePtr(new CloudPose());
    CloudPosePtr cloudTrajEstSE3AP = CloudPosePtr(new CloudPose());
    CloudPosePtr cloudTrajEstSE3CF = CloudPosePtr(new CloudPose());

    CloudPosePtr cloudTrajGtrSO3xR3 = CloudPosePtr(new CloudPose());
    CloudPosePtr cloudTrajGtrSE3 = CloudPosePtr(new CloudPose());

    for(int kidx = 0; kidx < trajSO3xR3AP->getNumKnots(); kidx++)
    {
        double knot_time = trajSO3xR3AP->getKnotTime(kidx);
        ROS_ASSERT_MSG(trajSO3xR3AP->getKnotTime(kidx) == trajSO3xR3CF->getKnotTime(kidx),
                       "Knot time not match: %f, %f",
                       trajSO3xR3AP->getKnotTime(kidx), trajSO3xR3CF->getKnotTime(kidx));

        cloudTrajEstSO3xR3AP->points.push_back(myTf(trajSO3xR3AP->getKnotPose(kidx)).Pose6D(knot_time));
        cloudTrajEstSO3xR3CF->points.push_back(myTf(trajSO3xR3CF->getKnotPose(kidx)).Pose6D(knot_time));
        cloudTrajEstSE3AP->points.push_back(myTf(trajSE3AP->getKnotPose(kidx)).Pose6D(knot_time));
        cloudTrajEstSE3CF->points.push_back(myTf(trajSE3CF->getKnotPose(kidx)).Pose6D(knot_time));

        cloudTrajGtrSO3xR3->points.push_back(gtTrajSO3xR3.pose(knot_time).Pose6D(knot_time));
        cloudTrajGtrSE3->points.push_back(gtTrajSE3.pose(knot_time).Pose6D(knot_time));
    }

    int kidx = 0;
    while(rclcpp::ok())
    {
        // Publish global trajectory
        Util::publishCloud(cloudTrajEstSO3xR3APPub, *cloudTrajEstSO3xR3AP, rclcpp::Clock().now(), "world");
        Util::publishCloud(cloudTrajEstSO3xR3CFPub, *cloudTrajEstSO3xR3CF, rclcpp::Clock().now(), "world");
        Util::publishCloud(cloudTrajEstSE3APPub, *cloudTrajEstSE3AP, rclcpp::Clock().now(), "world");
        Util::publishCloud(cloudTrajEstSE3CFPub, *cloudTrajEstSE3CF, rclcpp::Clock().now(), "world");
        
        Util::publishCloud(cloudTrajGtrSO3xR3Pub, *cloudTrajGtrSO3xR3, rclcpp::Clock().now(), "world");
        Util::publishCloud(cloudTrajGtrSE3Pub, *cloudTrajGtrSE3, rclcpp::Clock().now(), "world");

        double tk = trajSO3xR3AP->getKnotTime(kidx);

        RosOdomMsg odom_so3xr3 = gtTrajSO3xR3.pose(tk).rosOdom();
        odom_so3xr3.header.frame_id = "world";
        odom_so3xr3.header.stamp = rclcpp::Clock().now();
        odomTrajGtrSO3xR3Pub->publish(odom_so3xr3);

        RosOdomMsg odom_se3 = gtTrajSE3.pose(tk).rosOdom();
        odom_se3.header.frame_id = "world";
        odom_se3.header.stamp = rclcpp::Clock().now();
        odomTrajGtrSE3Pub->publish(odom_se3);

        kidx = kidx >= trajSO3xR3AP->getNumKnots() - 1 ? 0 : kidx + 1;

        this_thread::sleep_for(chrono::milliseconds(int(deltaT*1000)));
    }

    return 0;
}