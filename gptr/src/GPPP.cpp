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
namespace fs = std::filesystem;

NodeHandlePtr nh_ptr;
string log_dir;

// Ground truth trajectory

double wqx1 = 3*0.1;
double wqy1 = 3*0.1;
double wqz1 = 1*0.1;
double wpx1 = 3*0.15;
double wpy1 = 3*0.15;
double wpz1 = 1*0.15;

double rqx1 = M_PI*0.5;
double rqy1 = M_PI*0.5;
double rqz1 = M_PI*sqrt(3)/2;
double rpx1 = 5.0;
double rpy1 = 5.0;
double rpz1 = 5.0;

double wpx2 = 3*0.15;
double wpy2 = 3*0.15;
double wpz2 = 1*0.15;

double rpx2 = 5.0;
double rpy2 = 5.0;
double rpz2 = 5.0;

vector<double> Dtstep = {1, 10};
vector<long int> Wstep = {1, 50};

class GtTrajSO3xR3
{
public:

   ~GtTrajSO3xR3() {};
    GtTrajSO3xR3() {};

    void setn(int n_) { n = n_; }

    double getwqx() { return wqx1*n; }
    double getwqy() { return wqy1*n; }
    double getwqz() { return wqz1*n; }

    double getwpx() { return wpx1*n; }
    double getwpy() { return wpy1*n; }
    double getwpz() { return wpz1*n; }

    myTf<double> pose(double t) const
    {
        Quaternd Q = SO3d::exp(Vector3d(rqx1*cos(wqx1*n*t + 57),
                                        rqy1*sin(wqy1*n*t + 57),
                                        rqz1*sin(wqz1*n*t + 43))).unit_quaternion();
        Vector3d P( rpx1*sin(wpx1*t + 43),
                    rpy1*cos(wpy1*t + 43),
                    rpz1*cos(wpz1*t + 57));
        return myTf(Q, P);
    }

private:
    
    int n = 1;
};

class GtTrajSE3
{
public:

   ~GtTrajSE3() {};
    GtTrajSE3() {};

    void setn(int n_) { n = n_; }

    double getwqx() { return wqx1*n; }
    double getwqy() { return wqy1*n; }
    double getwqz() { return wqz1*n; }

    double getwpx() { return wpx2*n; }
    double getwpy() { return wpy2*n; }
    double getwpz() { return wpz2*n; }
            
    myTf<double> pose(double t) const
    {
        Vector3d P( rpx2*sin(wpx2*n*t + 43),
                    rpy2*cos(wpy2*n*t + 43),
                    rpz2*cos(wpz2*n*t + 57));
        Vector3d V( rpx2*wpx2*n*cos(wpx2*n*t + 43),
                   -rpy2*wpy2*n*sin(wpy2*n*t + 43),
                   -rpz2*wpz2*n*sin(wpz2*n*t + 57));
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

    int n = 1;
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
    Vector3d( 12.0,  12.0, 0.5),
    Vector3d(-12.0, -12.0, 2.5),
    Vector3d(-12.0,  12.0, 2.5),
    Vector3d( 12.0, -12.0, 0.5),
    Vector3d(-12.0, -12.0, 0.5),
    Vector3d( 12.0,  12.0, 2.5),
    Vector3d( 12.0, -12.0, 2.5),
    Vector3d(-12.0,  12.0, 0.5)};

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
    static std::mt19937 rng(571991);  
    static std::normal_distribution<double> noise_dist(0.0, uwb_noise);

    for (double ts = traj->getMinTime(); ts < traj->getMaxTime(); ts+=(1.0/uwb_rate))
    {
        myTf tf_W_B = trajGtr.pose(ts);

        auto   us = traj->computeTimeIndex(ts);
        int    u  = us.first;
        double s  = us.second;
        
        // static vector<vector<pair<int, int>>> pairs(tags.size());
        // if (pairs[0].size() == 0)
        // {
        //     for(int tidx = 0; tidx < tags.size(); tidx++)
        //     {
        //         for(int aidx = 0; aidx < anchors.size(); aidx++)
        //         {
        //             int aidx_ = (aidx + tidx) % anchors.size();
        //             pairs[tidx].push_back(make_pair(tidx, aidx_));
        //             RINFO("Added ranging pair: %d -> %d\n", tidx, aidx_);
        //         }
        //     }
        // }
                    
        // int curr_pair_idx = -1;
        // curr_pair_idx = (curr_pair_idx + 1) % anchors.size();
        for(int tidx = 0; tidx < tags.size(); tidx++)
        {
            // int tidx = pairs[tidx][curr_pair_idx].first;
            // int aidx = pairs[tidx][curr_pair_idx].second;
            // ROS_ASSERT_MSG(tidx == pairs[tidx][curr_pair_idx].first, "Mismatch %d, %d\n", tidx, aidx);

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
void InitializeTrajEst(double t0, GaussianProcessPtr &traj, const T &gtTraj)
{
    traj->setStartTime(t0);
    traj->setKnot(0, GPState(t0));

    // Extend the trajectory to maxTime
    RINFO("Initializing trajectory %s-%s.",
          traj->getGPMixerPtr()->getPoseRepresentation() == POSE_GROUP::SO3xR3 ? "SO3xR3" : "SE3",
          traj->getGPMixerPtr()->getJacobianForm() ? "AP" : "CF");

    while(traj->getMaxTime() < t0 + maxTime)
        traj->extendOneKnot(traj->getKnot(traj->getNumKnots()-1));

    // Fixed seed for reproducibility
    std::mt19937 rng(43); 
    // Define a uniform distribution (e.g., integers between 1 and 100)
    std::normal_distribution<double> rdist(0.0, 0.2);
    std::normal_distribution<double> pdist(0.0, 0.5);

    // Initialize the pose with erred ground truth
    for(int kidx = 0; kidx < traj->getNumKnots(); kidx++)
    {
        double ts = traj->getKnotTime(kidx);
        myTf pose = gtTraj.pose(ts);

        Vector3d theErr(rdist(rng), rdist(rng), rdist(rng));
        Vector3d posErr(pdist(rng), pdist(rng), pdist(rng));
        myTf pose_err(SO3d::exp(theErr).unit_quaternion(), posErr);

        traj->setKnot(kidx, GPState(traj->getKnotTime(kidx), (pose*pose_err).getSE3()));
    }
}

template <typename T>
string AssessTraj(GaussianProcessPtr &traj, const T &trajGtr, map<string, double> &report)
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
    options.function_tolerance  = 0.0;
    options.gradient_tolerance  = 0.0;
    options.parameter_tolerance = 0.0;

    // Add trajectory params
    TicToc tt_addparam;
    RINFO("Add params...");
    AddTrajParams(problem, traj);
    RINFO("Done, %.0fms", tt_addparam.Toc());

    // Add the motion prior factors
    TicToc tt_addmp2k;
    RINFO("Add mp2k...");
    FactorMeta mp2kFactorMeta;
    double cost_mp2k_init = -1, cost_mp2k_final = -1;
    AddMP2KFactors(problem, traj, mp2kFactorMeta);
    RINFO("Done, %.0fms", tt_addmp2k.Toc());

    // Add the UWB factors
    TicToc tt_uwb;
    RINFO("Add uwb...");
    FactorMeta uwbFactorMeta;
    double cost_uwb_init = -1, cost_uwb_final = -1;
    AddUWBFactors(problem, traj, trajGtr, uwbFactorMeta);
    RINFO("Done, %.0fms", tt_uwb.Toc());

    // RINFO(KYEL"Solving..."RESET);
    Util::ComputeCeresCost(mp2kFactorMeta.res, cost_mp2k_init, problem);
    Util::ComputeCeresCost(uwbFactorMeta.res,  cost_uwb_init,  problem);

    TicToc tt_solve;
    ceres::Solve(options, &problem, &summary);
    tt_solve.Toc();

    Util::ComputeCeresCost(mp2kFactorMeta.res, cost_mp2k_final, problem);
    Util::ComputeCeresCost(uwbFactorMeta.res,  cost_uwb_final,  problem);
    RINFO(KGRN"Done. %fms"RESET, tt_solve.GetLastStop());


    // Calculate the pose errors
    RINFO("Calculating error...");
    TicToc tt_rmse;
    vector<Vector3d> pos_err; vector<Vector3d> se3_err;
    for(double ts = traj->getMinTime(); ts < traj->getMaxTime(); ts += 0.01)
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
    
    RINFO(KGRN"Done. %fms"RESET, tt_rmse.Toc());


    RINFO("Drafting report ...");
    TicToc tt_report;
    
    string report_ = myprintf(
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

    report["iter"]    = summary.iterations.size();
    report["tslv"]    = summary.total_time_in_seconds;
    report["rmse"]    = pos_rmse;
    report["J0"]      = summary.initial_cost;
    report["JK"]      = summary.final_cost;
    report["MP2KJ0"]  = cost_mp2k_init;
    report["UWBJ0"]   = cost_uwb_init;
    report["MP2KJK"]  = cost_mp2k_final;
    report["UWBJK"]   = cost_uwb_final;
    
    RINFO(KGRN"Done. %fms\n"RESET, tt_report.Toc());

    
    return report_;
}



int main(int argc, char **argv)
{
    // Initalize ros nodes
    rclcpp::init(argc, argv);
    nh_ptr = rclcpp::Node::make_shared("GPPP");

    // Log direction
    Util::GetParam(nh_ptr, "log_dir", log_dir);   RINFO("Log dir: %s", log_dir.c_str());
    fs::create_directories(log_dir);

    // Get the params for ground truth

    Util::GetParam(nh_ptr, "wqx1", wqx1);   RINFO("Found param wqx1: %f", wqx1);
    Util::GetParam(nh_ptr, "wqy1", wqy1);   RINFO("Found param wqy1: %f", wqy1);
    Util::GetParam(nh_ptr, "wqz1", wqz1);   RINFO("Found param wqz1: %f", wqz1);
    Util::GetParam(nh_ptr, "rqx1", rqx1);   RINFO("Found param rqx1: %f", rqx1);
    Util::GetParam(nh_ptr, "rqy1", rqy1);   RINFO("Found param rqy1: %f", rqy1);
    Util::GetParam(nh_ptr, "rqz1", rqz1);   RINFO("Found param rqz1: %f", rqz1);

    Util::GetParam(nh_ptr, "wpx1", wpx1);   RINFO("Found param wpx1: %f", wpx1);
    Util::GetParam(nh_ptr, "wpy1", wpy1);   RINFO("Found param wpy1: %f", wpy1);
    Util::GetParam(nh_ptr, "wpz1", wpz1);   RINFO("Found param wpz1: %f", wpz1);
    Util::GetParam(nh_ptr, "rpx1", rpx1);   RINFO("Found param rpx1: %f", rpx1);
    Util::GetParam(nh_ptr, "rpy1", rpy1);   RINFO("Found param rpy1: %f", rpy1);
    Util::GetParam(nh_ptr, "rpz1", rpz1);   RINFO("Found param rpz1: %f", rpz1);

    Util::GetParam(nh_ptr, "wpx2", wpx2);   RINFO("Found param wpx2: %f", wpx2);
    Util::GetParam(nh_ptr, "wpy2", wpy2);   RINFO("Found param wpy2: %f", wpy2);
    Util::GetParam(nh_ptr, "wpz2", wpz2);   RINFO("Found param wpz2: %f", wpz2);
    Util::GetParam(nh_ptr, "rpx2", rpx2);   RINFO("Found param rpx2: %f", rpx2);
    Util::GetParam(nh_ptr, "rpy2", rpy2);   RINFO("Found param rpy2: %f", rpy2);
    Util::GetParam(nh_ptr, "rpz2", rpz2);   RINFO("Found param rpz2: %f", rpz2);



    // Get params and config the trajectory

    // Get the params
    Util::GetParam(nh_ptr, "maxTime", maxTime);
    Util::GetParam(nh_ptr, "deltaT", deltaT);
    Util::GetParam(nh_ptr, "mpCovROSJerk", mpCovROSJerk);
    Util::GetParam(nh_ptr, "mpCovPVAJerk", mpCovPVAJerk);
    Util::GetParam(nh_ptr, "lie_epsilon", lie_epsilon);
    string pose_type_; Util::GetParam(nh_ptr, "pose_type", pose_type_); pose_type = pose_type_ == "SE3" ? POSE_GROUP::SE3 : POSE_GROUP::SO3xR3;
    bool use_approx_drv =  Util::GetBoolParam(nh_ptr, "use_approx_drv", true);
    bool random_start =  Util::GetBoolParam(nh_ptr, "random_start", true);
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


    // Get param for experiments
    Util::GetParam(nh_ptr, "Dtstep", Dtstep);
    Util::GetParam(nh_ptr, "Wstep",  Wstep);
    
    RINFO("Dtstep: %f -> %f", Dtstep.front(), Dtstep.back());
    RINFO("Wstep: %d -> %d", Wstep.front(), Wstep.back());


    // Fixed seed for reproducibility
    std::mt19937 rng(1102); 
    // Define a uniform distribution (e.g., integers between 1 and 100)
    std::uniform_real_distribution<double> t0udist(-10, 10);
    

    auto Experiment = [&](string logname, auto gtrTraj, string gtrTrajType, vector<double> &M, vector<long int> &N)
    {
        std::ofstream logfile(logname, std::ios::out);
        logfile << std::fixed << std::setprecision(6);
        logfile << "wqx1,wqy1,wqz1,"
                   "wpx1,wpy1,wpz1,"
                   "dt,"
                   "so3xr3ap_tslv,so3xr3cf_tslv,so3xr3rf_tslv,"
                   "se3ap_tslv,se3cf_tslv,se3rf_tslv,"
                   "so3xr3ap_JK,so3xr3cf_JK,so3xr3rf_JK,"
                   "se3ap_JK,se3cf_JK,se3rf_JK,"
                   "so3xr3ap_rmse,so3xr3cf_rmse,so3xr3rf_rmse,"
                   "se3ap_rmse,se3cf_rmse,se3rf_rmse\n";

        for (double &m : M)
        {
            double deltaTm = m;

            map<string, double> so3xr3ap_report;
            map<string, double> so3xr3cf_report;
            map<string, double> so3xr3rf_report;
            map<string, double> se3ap_report;
            map<string, double> se3cf_report;
            map<string, double> se3rf_report;

            for (int n = N.front(); n <= N.back(); n++)
            {
                GaussianProcessPtr trajSO3xR3AP(new GaussianProcess(deltaTm, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SO3xR3, lie_epsilon, true));
                GaussianProcessPtr trajSO3xR3CF(new GaussianProcess(deltaTm, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SO3xR3, lie_epsilon, false));
                GaussianProcessPtr trajSO3xR3RF(new GaussianProcess(deltaTm, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SO3xR3, lie_epsilon, false));
                GaussianProcessPtr trajSE3AP(new GaussianProcess(deltaTm, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SE3, lie_epsilon, true));
                GaussianProcessPtr trajSE3CF(new GaussianProcess(deltaTm, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SE3, lie_epsilon, false));
                GaussianProcessPtr trajSE3RF(new GaussianProcess(deltaTm, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SE3, lie_epsilon, false));

                gtrTraj.setn(n);

                double t0 = 0;
                if (random_start)
                    t0 = t0udist(rng);

                InitializeTrajEst(t0, trajSO3xR3AP, gtrTraj);
                InitializeTrajEst(t0, trajSO3xR3CF, gtrTraj);
                InitializeTrajEst(t0, trajSO3xR3RF, gtrTraj);
                InitializeTrajEst(t0, trajSE3AP, gtrTraj);
                InitializeTrajEst(t0, trajSE3CF, gtrTraj);
                InitializeTrajEst(t0, trajSE3RF, gtrTraj);

                // double rmse = -1;
                // Assess with the SO3xR3 trajectory
                string report_SO3xR3AP = AssessTraj(trajSO3xR3AP, gtrTraj, so3xr3ap_report);
                string report_SO3xR3CF = AssessTraj(trajSO3xR3CF, gtrTraj, so3xr3cf_report);
                string report_SE3AP___ = AssessTraj(trajSE3AP,    gtrTraj, se3ap_report);
                string report_SE3CF___ = AssessTraj(trajSE3CF,    gtrTraj, se3cf_report);

                for(int kidx = 0; kidx < trajSO3xR3AP->getNumKnots(); kidx++)
                {
                    trajSO3xR3RF->setKnot(kidx, trajSO3xR3AP->getKnot(kidx));
                    trajSE3RF->setKnot(kidx, trajSE3AP->getKnot(kidx));
                }
                string report_SO3xR3RF = AssessTraj(trajSO3xR3RF, gtrTraj, so3xr3rf_report);
                string report_SE3RF___ = AssessTraj(trajSE3RF,    gtrTraj, se3rf_report);

                RINFO("%s n: %2d, Omega: %.3f, %s", gtrTrajType.c_str(), n, n*wqx1, report_SO3xR3AP.c_str());
                RINFO("%s n: %2d, Omega: %.3f, %s", gtrTrajType.c_str(), n, n*wqx1, report_SO3xR3CF.c_str());
                RINFO("%s n: %2d, Omega: %.3f, %s", gtrTrajType.c_str(), n, n*wqx1, report_SO3xR3RF.c_str());
                RINFO("%s n: %2d, Omega: %.3f, %s", gtrTrajType.c_str(), n, n*wqx1, report_SE3AP___.c_str());
                RINFO("%s n: %2d, Omega: %.3f, %s", gtrTrajType.c_str(), n, n*wqx1, report_SE3CF___.c_str());
                RINFO("%s n: %2d, Omega: %.3f, %s", gtrTrajType.c_str(), n, n*wqx1, report_SE3RF___.c_str());

                // Save the rmse result to the log
                logfile << gtrTraj.getwqx() << ","
                        << gtrTraj.getwqy() << ","
                        << gtrTraj.getwqz() << ","
                        << gtrTraj.getwpx() << ","
                        << gtrTraj.getwpy() << ","
                        << gtrTraj.getwpz() << ","
                        << deltaTm << ","
                        << so3xr3ap_report["tslv"] << ","
                        << so3xr3cf_report["tslv"] << ","
                        << so3xr3rf_report["tslv"] << ","
                        << se3ap_report["tslv"] << ","
                        << se3cf_report["tslv"] << ","
                        << se3rf_report["tslv"] << ","
                        << so3xr3ap_report["JK"] << ","
                        << so3xr3cf_report["JK"] << ","
                        << so3xr3rf_report["JK"] << ","
                        << se3ap_report["JK"] << ","
                        << se3cf_report["JK"] << ","
                        << se3rf_report["JK"] << ","
                        << so3xr3ap_report["rmse"] << ","
                        << so3xr3cf_report["rmse"] << ","
                        << so3xr3rf_report["rmse"] << ","
                        << se3ap_report["rmse"] << ","
                        << se3cf_report["rmse"] << ","
                        << se3rf_report["rmse"]
                        << endl;
            }
        }
        logfile.close();
    };


    RINFO(KGRN "Experimenting with SO3xR3 gtr traj" RESET);
    Experiment(log_dir + "/gtrTrajso3xr3.csv", gtTrajSO3xR3, string("gtTrajSO3xR3"), Dtstep, Wstep);
    RINFO(KGRN "Experimenting with SE3 gtr traj" RESET);
    Experiment(log_dir + "/gtrTrajse3.csv",    gtTrajSE3,    string("gtTrajSE3"),    Dtstep, Wstep);

    RINFO(KGRN"Program finished!"RESET);

    return 0;
}