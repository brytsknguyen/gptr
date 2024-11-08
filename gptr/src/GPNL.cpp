#include "unistd.h"
#include <algorithm>  // for std::sort

// ROS utilities
#include "ros/ros.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include "sensor_msgs/PointCloud2.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/TransformStamped.h"

// Custom built utilities
#include "utility.h"
#include "GaussianProcess.hpp"
#include "GPNL.hpp"

// Topics
// #include "cf_msgs/Tdoa.h"
// #include "cf_msgs/Tof.h"
#include "nlink_parser/LinktrackNodeframe3.h"

using namespace std;

boost::shared_ptr<ros::NodeHandle> nh_ptr;
const double POSINF =  std::numeric_limits<double>::infinity();
const double NEGINF = -std::numeric_limits<double>::infinity();

vector<Vector3d> anc_pos;
vector<Vector3d> tag_pos;

double gpDt = 0.02;
Matrix3d gpQr;
Matrix3d gpQc;

bool auto_exit;
int  WINDOW_SIZE = 4;
int  SLIDE_SIZE  = 2;
double w_twr = 0.1;
double GYR_N = 10;
double GYR_W = 10;
double ACC_N = 0.5;
double ACC_W = 10;
double twr_loss_thres = -1;
double mp_loss_thres = -1;

GaussianProcessPtr traj;

bool fuse_twr = true;
bool fuse_imu = true;

bool acc_ratio = true;
bool gyro_unit = true;

struct UwbImuBuf
{
    deque<TwrMsgPtr> twrBuf;
    deque<ImuMsgPtr> imuBuf;
    deque<GtrMsgPtr> gtrBuf;

    mutex twrBuf_mtx;
    mutex imuBuf_mtx;
    mutex gtrBuf_mtx;

    double minTime()
    {
        double tmin = std::numeric_limits<double>::infinity();
        if (twrBuf.size() != 0 && fuse_twr)
            tmin = min(tmin, twrBuf.front()->header.stamp.toSec());
        if (imuBuf.size() != 0 && fuse_imu)
            tmin = min(tmin, imuBuf.front()->header.stamp.toSec());
        if (gtrBuf.size() != 0)
            tmin = min(tmin, gtrBuf.front()->header.stamp.toSec());    
        return tmin;
    }

    double maxTime()
    {
        double tmax = -std::numeric_limits<double>::infinity();
        if (twrBuf.size() != 0 && fuse_twr)
            tmax = max(tmax, twrBuf.back()->header.stamp.toSec());
        if (imuBuf.size() != 0 && fuse_imu)
            tmax = max(tmax, imuBuf.back()->header.stamp.toSec());
        if (gtrBuf.size() != 0)
            tmax = max(tmax, gtrBuf.back()->header.stamp.toSec());    
        return tmax;
    }

    template<typename T>
    void transferDataOneBuf(deque<T> &selfbuf, deque<T> &otherbuf, mutex &otherbufmtx, double tmax)
    {
        while(otherbuf.size() != 0)
        {
            if (otherbuf.front()->header.stamp.toSec() <= tmax)
            {
                lock_guard<mutex> lg(otherbufmtx);
                selfbuf.push_back(otherbuf.front());
                otherbuf.pop_front();
            }
            else
                break;
        }
    }

    void transferData(UwbImuBuf &other, double tmax)
    {
        if (fuse_twr) transferDataOneBuf(twrBuf, other.twrBuf, other.twrBuf_mtx, tmax);
        if (fuse_imu) transferDataOneBuf(imuBuf, other.imuBuf, other.imuBuf_mtx, tmax);
        if (fuse_imu) transferDataOneBuf(gtrBuf, other.gtrBuf, other.gtrBuf_mtx, tmax);
    }

    template<typename T>
    void slideForwardOneBuf(deque<T> &buf, double tremove)
    {
        while(buf.size() != 0)
            if(buf.front()->header.stamp.toSec() < tremove)
                buf.pop_front();
            else
                break;
    }

    void slideForward(double tremove)
    {
        if (fuse_twr) slideForwardOneBuf(twrBuf, tremove);
        if (fuse_imu) slideForwardOneBuf(imuBuf, tremove);
        slideForwardOneBuf(gtrBuf, tremove);
    }
};

UwbImuBuf UIBuf;
vector<Eigen::Vector3d> gtBuf;
nav_msgs::Path est_path;
nav_msgs::Path gt_path;

ros::Publisher gt_pub;
ros::Publisher gt_path_pub;
ros::Publisher est_pub;
ros::Publisher odom_pub;
ros::Publisher knot_pub;

vector<ros::Subscriber> twrSub;
ros::Subscriber imuSub;
ros::Subscriber gtSub;

Eigen::Vector3d bg = Eigen::Vector3d::Zero();
Eigen::Vector3d ba = Eigen::Vector3d::Zero();
Eigen::Vector3d g = Eigen::Vector3d(0, 0, 9.81);
const Eigen::Vector3d P_I_tag = Eigen::Vector3d(-0.012, 0.001, 0.091);

bool if_save_traj;
std::string traj_save_path;

void twrCb(const TwrMsgPtr &msg)
{
    lock_guard<mutex> lg(UIBuf.twrBuf_mtx);
    UIBuf.twrBuf.push_back(msg);
    // printf(KCYN "Receive twr\n" RESET);
}

void imuCb(const ImuMsgPtr &msg)
{
    lock_guard<mutex> lg(UIBuf.imuBuf_mtx);
    UIBuf.imuBuf.push_back(msg);
    // printf(KMAG "Receive imu\n" RESET);
}

void gtCb(const geometry_msgs::TransformStampedConstPtr& gt_msg)
{
    {
        lock_guard<mutex> lg(UIBuf.gtrBuf_mtx);
        UIBuf.gtrBuf.push_back(gt_msg);
    }

    Eigen::Quaterniond q(gt_msg->transform.rotation.w, gt_msg->transform.rotation.x,
                         gt_msg->transform.rotation.y, gt_msg->transform.rotation.z);
    Eigen::Vector3d pos(gt_msg->transform.translation.x, gt_msg->transform.translation.y, gt_msg->transform.translation.z);
    gtBuf.push_back(pos);    

    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = gt_msg->header.stamp;
    odom_msg.header.frame_id = "map";
    odom_msg.pose.pose.position.x = pos[0];
    odom_msg.pose.pose.position.y = pos[1];
    odom_msg.pose.pose.position.z = pos[2];
    odom_msg.pose.pose.orientation.w = q.w();
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    gt_pub.publish(odom_msg);  

    geometry_msgs::PoseStamped traj_msg;
    traj_msg.header.stamp = gt_msg->header.stamp;
    traj_msg.pose.position.x = pos.x();
    traj_msg.pose.position.y = pos.y();
    traj_msg.pose.position.z = pos.z();
    traj_msg.pose.orientation.w = q.w();
    traj_msg.pose.orientation.x = q.x();
    traj_msg.pose.orientation.y = q.y();
    traj_msg.pose.orientation.z = q.z();
    gt_path.poses.push_back(traj_msg);
    gt_path_pub.publish(gt_path);          
}

void processData(GaussianProcessPtr traj, GPNLPtr gpmui)
{
    UwbImuBuf swUIBuf;

    // Loop and optimize
    while(ros::ok())
    {
        // Step 0: Check if there is data that can be admitted to the sw buffer
        double newMaxTime = traj->getMaxTime() + SLIDE_SIZE*gpDt;

        ros::Time timeout = ros::Time::now();
        if(UIBuf.maxTime() < newMaxTime)
        {
            if(auto_exit && (ros::Time::now() - timeout).toSec() > 20.0)
            {
                printf("Polling time out exiting.\n");
                exit(-1);
            }
            static int msWait = int(SLIDE_SIZE*gpDt*1000);
            this_thread::sleep_for(chrono::milliseconds(msWait));
            continue;
        }
        timeout = ros::Time::now();

        // Step 1: Extract the data to the local buffer
        swUIBuf.transferData(UIBuf, newMaxTime);

        // Step 2: Extend the trajectory
        if (traj->getMaxTime() < newMaxTime && (newMaxTime - traj->getMaxTime()) > gpDt*0.01)
            traj->extendOneKnot();

        // Step 3: Optimization
        TicToc tt_solve;          
        double tmin = traj->getKnotTime(traj->getNumKnots() - WINDOW_SIZE) + 1e-3;     // Start time of the sliding window
        double tmax = traj->getKnotTime(traj->getNumKnots() - 1) + 1e-3;               // End time of the sliding window              
        double tmid = tmin + SLIDE_SIZE*traj->getDt() + 1e-3;                          // Next start time of the sliding window,
                                                                                       // also determines the marginalization time limit
        gpmui->Evaluate(traj, bg, ba, g, tmin, tmax, tmid, swUIBuf.twrBuf, swUIBuf.imuBuf, swUIBuf.gtrBuf,
                        anc_pos, tag_pos, traj->getNumKnots() >= WINDOW_SIZE, 
                        w_twr, GYR_N, ACC_N, GYR_W, ACC_W, twr_loss_thres, mp_loss_thres);
        tt_solve.Toc();

        // Step 4: Report, visualize
        printf("Traj: %f. Sw: %.3f -> %.3f. Buf: %d, %d, %d. Num knots: %d\n",
                traj->getMaxTime(), swUIBuf.minTime(), swUIBuf.maxTime(),
                UIBuf.twrBuf.size(), UIBuf.imuBuf.size(), UIBuf.gtrBuf.size(), traj->getNumKnots());

        // Visualize knots
        pcl::PointCloud<pcl::PointXYZ> est_knots;
        for (int i = 0; i < traj->getNumKnots(); i++)
        {
            Eigen::Vector3d knot_pos = traj->getKnotPose(i).translation();
            est_knots.points.push_back(pcl::PointXYZ(knot_pos.x(), knot_pos.y(), knot_pos.z()));
        }
        sensor_msgs::PointCloud2 knot_msg;
        pcl::toROSMsg(est_knots, knot_msg);
        knot_msg.header.stamp = ros::Time::now();
        knot_msg.header.frame_id = "map";        
        knot_pub.publish(knot_msg);

        // Visualize estimated trajectory
        auto est_pose = traj->pose(swUIBuf.twrBuf.front()->header.stamp.toSec());
        Eigen::Vector3d est_pos = est_pose.translation();
        Eigen::Quaterniond est_ort = est_pose.unit_quaternion();
        geometry_msgs::PoseStamped traj_msg;
        traj_msg.header.stamp = ros::Time::now();
        traj_msg.pose.position.x = est_pos.x();
        traj_msg.pose.position.y = est_pos.y();
        traj_msg.pose.position.z = est_pos.z();
        traj_msg.pose.orientation.w = 1;
        traj_msg.pose.orientation.x = 0;
        traj_msg.pose.orientation.y = 0;
        traj_msg.pose.orientation.z = 0;
        est_path.poses.push_back(traj_msg);
        est_pub.publish(est_path);

        // Visualize odometry
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = ros::Time::now();
        odom_msg.header.frame_id = "map";
        est_pose = traj->pose(traj->getKnotTime(traj->getNumKnots() - 1));
        est_pos = est_pose.translation();
        est_ort = est_pose.unit_quaternion();
        odom_msg.pose.pose.position.x = est_pos[0];
        odom_msg.pose.pose.position.y = est_pos[1];
        odom_msg.pose.pose.position.z = est_pos[2];
        odom_msg.pose.pose.orientation.w = est_ort.w();
        odom_msg.pose.pose.orientation.x = est_ort.x();
        odom_msg.pose.pose.orientation.y = est_ort.y();
        odom_msg.pose.pose.orientation.z = est_ort.z();
        odom_pub.publish(odom_msg);             

        // Step 5: Slide the window forward
        if (traj->getNumKnots() >= WINDOW_SIZE)
        {
            double removeTime = traj->getKnotTime(traj->getNumKnots() - WINDOW_SIZE + SLIDE_SIZE);
            swUIBuf.slideForward(removeTime);
        }
    }
}

void saveTraj(GaussianProcessPtr traj)
{
    if (!std::filesystem::is_directory(traj_save_path) || !std::filesystem::exists(traj_save_path)) {
        std::filesystem::create_directories(traj_save_path);
    }
    std::string traj_file_name = traj_save_path + "traj.txt";
    std::ofstream f_traj(traj_file_name);    
    for (int i = 0; i < gt_path.poses.size(); i++) {
        double t_gt = gt_path.poses[i].header.stamp.toSec();
        auto   us = traj->computeTimeIndex(t_gt);
        int    u  = us.first;
        double s  = us.second;

        if (u < 0 || u+1 >= traj->getNumKnots()) {
            continue;
        }        
        auto est_pose = traj->pose(t_gt);     
        Eigen::Vector3d est_pos = est_pose.translation();
        Eigen::Quaterniond est_ort = est_pose.unit_quaternion();    
        f_traj << std::fixed << t_gt << std::setprecision(7) 
               << " " << est_pos.x() << " " << est_pos.y() << " " << est_pos.z() 
               << " " << est_ort.x() << " " << est_ort.y() << " " << est_ort.z()  << " " << est_ort.w() << std::endl;
    }
    f_traj.close();

    std::string gt_file_name = traj_save_path + "gt.txt";
    std::ofstream f_gt(gt_file_name);    
    for (int i = 0; i < gt_path.poses.size(); i++) {
        double t_gt = gt_path.poses[i].header.stamp.toSec();
    
        f_gt << std::fixed << t_gt << std::setprecision(7) 
             << " " << gt_path.poses[i].pose.position.x << " " << gt_path.poses[i].pose.position.y << " " << gt_path.poses[i].pose.position.z
             << " " << gt_path.poses[i].pose.orientation.x << " " << gt_path.poses[i].pose.orientation.y << " " << gt_path.poses[i].pose.orientation.z  << " " << gt_path.poses[i].pose.orientation.w << std::endl;
    }
    f_gt.close();    
}

int main(int argc, char **argv)
{
    // Initialize the node
    ros::init(argc, argv, "gpui");
    ros::NodeHandle nh("~");
    nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    // Determine if we exit if no data is received after a while
    bool auto_exit = Util::GetBoolParam(nh_ptr, "auto_exit", false);

    // Parameters for the GP trajectory
    double gpQr_ = 1.0, gpQc_ = 1.0;
    nh_ptr->getParam("gpDt", gpDt );
    nh_ptr->getParam("gpQr", gpQr_);
    nh_ptr->getParam("gpQc", gpQc_);
    gpQr = gpQr_*Matrix3d::Identity(3, 3);
    gpQc = gpQc_*Matrix3d::Identity(3, 3);

    // Find the path to anchor position
    // string anchor_pose_path;
    // nh_ptr->getParam("anchor_pose_path", anchor_pose_path);
    
    // Load the anchor pose 
    vector<double> anc_pos_;
    nh_ptr->getParam("anc_pos", anc_pos_);
    ROS_ASSERT(anc_pos_.size() % 3 == 0);
    printf("Found %d anchors.\n", anc_pos_.size()/3);
    for(int aidx = 0; aidx < anc_pos_.size(); aidx+=3)
    {
        anc_pos.push_back(Vector3d(anc_pos_[aidx], anc_pos_[aidx+1], anc_pos_[aidx+2]));
        cout << anc_pos.back().transpose() << endl;
    }

    // Load the anchor pose 
    vector<double> tag_pos_;
    nh_ptr->getParam("tag_pos", tag_pos_);
    ROS_ASSERT(tag_pos_.size() % 3 == 0);
    printf("Found %d tags.\n", tag_pos_.size()/3);
    for(int tidx = 0; tidx < tag_pos_.size(); tidx+=3)
    {
        tag_pos.push_back(Vector3d(tag_pos_[tidx], tag_pos_[tidx+1], tag_pos_[tidx+2]));
        cout << tag_pos.back().transpose() << endl;
    }

    // Topics to subscribe to
    vector<string> twr_topic;
    nh_ptr->getParam("twr_topic", twr_topic);
    string imu_topic; nh_ptr->getParam("imu_topic", imu_topic);
    string gt_topic; nh_ptr->getParam("gt_topic", gt_topic);
    fuse_twr = Util::GetBoolParam(nh_ptr, "fuse_twr", fuse_twr);
    fuse_imu = Util::GetBoolParam(nh_ptr, "fuse_imu", fuse_imu);

    vector<double> init_pose; nh_ptr->getParam("init_pose", init_pose);

    // Subscribe to the topics
    for(int idx = 0; idx < twr_topic.size(); idx++)
        twrSub.push_back(nh_ptr->subscribe(twr_topic[idx], 10, twrCb));

    imuSub = nh_ptr->subscribe(imu_topic, 10, imuCb);
    gtSub = nh_ptr->subscribe(gt_topic, 10, gtCb);

    // Publish estimates
    knot_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/estimated_knot", 10);
    gt_pub = nh_ptr->advertise<nav_msgs::Odometry>("/ground_truth", 10);
    gt_path_pub = nh_ptr->advertise<nav_msgs::Path>("/ground_truth_path", 10);
    est_pub = nh_ptr->advertise<nav_msgs::Path>("/estimated_trajectory", 10);
    odom_pub = nh_ptr->advertise<nav_msgs::Odometry>("/estimated_pose", 10);

    est_path.header.frame_id = "map";
    gt_path.header.frame_id = "map";

    // Time to check the buffers and perform optimization
    nh_ptr->getParam("WINDOW_SIZE", WINDOW_SIZE);
    nh_ptr->getParam("SLIDE_SIZE", SLIDE_SIZE);
    nh_ptr->getParam("w_twr", w_twr);
    nh_ptr->getParam("GYR_N", GYR_N);
    nh_ptr->getParam("GYR_W", GYR_W);
    nh_ptr->getParam("ACC_N", ACC_N);
    nh_ptr->getParam("ACC_W", ACC_W);    
    nh_ptr->getParam("twr_loss_thres", twr_loss_thres);
    nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
    if_save_traj = Util::GetBoolParam(nh_ptr, "if_save_traj", if_save_traj);
    nh_ptr->getParam("traj_save_path", traj_save_path);

    // Create the trajectory
    traj = GaussianProcessPtr(new GaussianProcess(gpDt, gpQr, gpQc, true));
    GPNLPtr gpmui(new GPNL(nh_ptr));

    // Wait to get the initial time
    while(ros::ok())
    {
        if(UIBuf.minTime() == POSINF)
        {
            this_thread::sleep_for(chrono::milliseconds(100));
            ros::spinOnce();
            continue;
        }

        double t0 = UIBuf.minTime();
        traj->setStartTime(t0);

        // Set initial pose
        SE3d initial_pose;
        initial_pose.so3() = SO3d(Quaternd(init_pose[6], init_pose[3], init_pose[4], init_pose[5]));
        initial_pose.translation() = Eigen::Vector3d(init_pose[0], init_pose[1], init_pose[2]);
        traj->setKnot(0, GPState(t0, initial_pose));

        break;
    }
    printf(KGRN "Start time: %f\n" RESET, traj->getMinTime());

    // Start polling and processing the data
    thread pdthread(processData, traj, gpmui);

    // Spin
    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();
    pdthread.join();
    if (if_save_traj) {
        saveTraj(traj);
    }
    return 0;
}