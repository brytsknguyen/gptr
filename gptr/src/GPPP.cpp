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

double wq  = 1.0;
double wp  = 0.15;
double rq1 = M_PI*0.5;
double rq2 = M_PI*sqrt(3)/2;
double rp  = 5;

class GtTrajSO3xR3
{
public:

   ~GtTrajSO3xR3() {};
    GtTrajSO3xR3() {};

    myTf<double> pose(double t) const
    {
        Quaternd Q = SO3d::exp(Vector3d(rq1*cos(3*wq*t + 57),
                                        rq1*sin(3*wq*t + 57),
                                        rq2*sin(  wq*t + 43))).unit_quaternion();
        Vector3d P(-rp*sin(3*wp*t + 43),
                    rp*cos(3*wp*t + 43),
                    rp*cos(  wp*t + 57));
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
    GtTrajSE3() {};

    myTf<double> pose(double t) const
    {
        Quaternd Q = SO3d::exp(Vector3d(rq1*cos(3*wq*t + 57),
                                        rq1*sin(3*wq*t + 57),
                                        rq2*sin(  wq*t + 43))).unit_quaternion();
        Vector3d P(-rp*sin(3*wp*t + 43),
                    rp*cos(3*wp*t + 43),
                    rp*cos(  wp*t + 57));
        return myTf(Q, P);
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

void AddUWBFactors(ceres::Problem &problem, GaussianProcessPtr &traj, FactorMeta &factorMeta)
{
    for (double ts = traj->getMinTime() + traj->getDt()/10.0; ts < traj->getMaxTime(); ts+=(1.0/uwb_rate))
    {
        myTf tf_W_B = gtTrajSO3xR3.pose(ts);

        auto   us = traj->computeTimeIndex(ts);
        int    u  = us.first;
        double s  = us.second;

        for(int tidx = 0; tidx < tags.size(); tidx++)
        {
            for(int aidx = 0; aidx < anchors.size(); aidx++)
            {
                Vector3d pos_tag = tags[tidx];
                Vector3d pos_anc = anchors[aidx];

                double twr = (tf_W_B.rot*pos_tag + tf_W_B.pos - pos_anc).norm();

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

void InitializeTrajEst(GaussianProcessPtr &traj)
{
    traj->setStartTime(0);
    traj->setKnot(0, GPState(0));

    // Extend the trajectory to maxTime
    RINFO("Extending trajectory %s - %s.",
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
        myTf pose = gtTrajSO3xR3.pose(ts);

        Vector3d theErr(rdist(rng), rdist(rng), rdist(rng));
        Vector3d posErr(pdist(rng), pdist(rng), pdist(rng));
        myTf pose_err(SO3d::exp(theErr).unit_quaternion(), posErr);

        traj->setKnot(kidx, GPState(traj->getMaxTime(), (pose*pose_err).getSE3()));
    }
}


template <typename T>
string AssessTraj(GaussianProcessPtr &traj, const T &trajGtr)
{   
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
    AddUWBFactors(problem, traj, uwbFactorMeta);
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

    Util::GetParam(nh_ptr, "wq" ,  wq);
    Util::GetParam(nh_ptr, "wp" ,  wp);
    Util::GetParam(nh_ptr, "rq1", rq1);
    Util::GetParam(nh_ptr, "rq2", rq2);
    Util::GetParam(nh_ptr, "rp" ,  rp);

    RINFO("Found param wq  %f", wq );
    RINFO("Found param wp  %f", wp );
    RINFO("Found param rq1 %f", rq1);
    RINFO("Found param rq2 %f", rq2);
    RINFO("Found param rp  %f", rp );



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
    RINFO("UWB rate: %f", uwb_rate);





    // Creating the trajectory and assess

    GaussianProcessPtr trajSO3xR3AP(new GaussianProcess(deltaT, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SO3xR3, lie_epsilon, true));
    GaussianProcessPtr trajSO3xR3CF(new GaussianProcess(deltaT, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SO3xR3, lie_epsilon, false));
    GaussianProcessPtr trajSE3AP(new GaussianProcess(deltaT, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SE3, lie_epsilon, true));
    GaussianProcessPtr trajSE3CF(new GaussianProcess(deltaT, CovROSJerk, CovPVAJerk, true, POSE_GROUP::SE3, lie_epsilon, false));

    InitializeTrajEst(trajSO3xR3AP);
    InitializeTrajEst(trajSO3xR3CF);
    InitializeTrajEst(trajSE3AP);
    InitializeTrajEst(trajSE3CF);

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
    
    string reportSO3xR3AP = AssessTraj(trajSO3xR3AP, gtTrajSO3xR3);
    string reportSO3xR3CF = AssessTraj(trajSO3xR3CF, gtTrajSO3xR3);
    string reportSE3AP = AssessTraj(trajSE3AP, gtTrajSO3xR3);
    string reportSE3CF = AssessTraj(trajSE3CF, gtTrajSO3xR3);

    RINFO("%s", reportSO3xR3AP.c_str());
    RINFO("%s", reportSO3xR3CF.c_str());
    RINFO("%s", reportSE3AP.c_str());
    RINFO("%s", reportSE3CF.c_str());




    rclcpp::Publisher<RosPc2Msg>::SharedPtr trajEstSO3xR3APPub = nh_ptr->create_publisher<RosPc2Msg>("/gp_traj_est_so3xr3_cf", 1);
    rclcpp::Publisher<RosPc2Msg>::SharedPtr trajEstSO3xR3CFPub = nh_ptr->create_publisher<RosPc2Msg>("/gp_traj_est_so3xr3_ap", 1);
    rclcpp::Publisher<RosPc2Msg>::SharedPtr trajEstSE3APPub = nh_ptr->create_publisher<RosPc2Msg>("/gp_traj_est_se3_cf", 1);
    rclcpp::Publisher<RosPc2Msg>::SharedPtr trajEstSE3CFPub = nh_ptr->create_publisher<RosPc2Msg>("/gp_traj_est_se3_ap", 1);
    rclcpp::Publisher<RosPc2Msg>::SharedPtr trajGtrPub = nh_ptr->create_publisher<RosPc2Msg>("/gp_traj_gtr", 1);

    CloudPosePtr trajEstSO3xR3AP = CloudPosePtr(new CloudPose());
    CloudPosePtr trajEstSO3xR3CF = CloudPosePtr(new CloudPose());
    CloudPosePtr trajEstSE3AP = CloudPosePtr(new CloudPose());
    CloudPosePtr trajEstSE3CF = CloudPosePtr(new CloudPose());
    CloudPosePtr trajGtr = CloudPosePtr(new CloudPose());

    for(int kidx = 0; kidx < trajSO3xR3AP->getNumKnots(); kidx++)
    {
        double knot_time = trajSO3xR3AP->getKnotTime(kidx);
        ROS_ASSERT_MSG(trajSO3xR3AP->getKnotTime(kidx) == trajSO3xR3CF->getKnotTime(kidx),
                       "Knot time not match: %f, %f", trajSO3xR3AP->getKnotTime(kidx), trajSO3xR3CF->getKnotTime(kidx));

        trajEstSO3xR3AP->points.push_back(myTf(trajSO3xR3AP->getKnotPose(kidx)).Pose6D(knot_time));
        trajEstSO3xR3CF->points.push_back(myTf(trajSO3xR3CF->getKnotPose(kidx)).Pose6D(knot_time));
        trajEstSE3AP->points.push_back(myTf(trajSE3AP->getKnotPose(kidx)).Pose6D(knot_time));
        trajEstSE3CF->points.push_back(myTf(trajSE3CF->getKnotPose(kidx)).Pose6D(knot_time));
        
        trajGtr->points.push_back(gtTrajSO3xR3.pose(knot_time).Pose6D(knot_time));
    }

    while(rclcpp::ok())
    {

        // Publish global trajectory
        Util::publishCloud(trajEstSO3xR3APPub, *trajEstSO3xR3AP, rclcpp::Clock().now(), "world");
        Util::publishCloud(trajEstSO3xR3CFPub, *trajEstSO3xR3CF, rclcpp::Clock().now(), "world");
        Util::publishCloud(trajEstSE3APPub, *trajEstSE3AP, rclcpp::Clock().now(), "world");
        Util::publishCloud(trajEstSE3CFPub, *trajEstSE3CF, rclcpp::Clock().now(), "world");
        Util::publishCloud(trajGtrPub, *trajGtr, rclcpp::Clock().now(), "world");

        this_thread::sleep_for(chrono::milliseconds(1000));
    }

    return 0;
}