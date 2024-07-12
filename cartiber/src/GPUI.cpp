#include "unistd.h"
#include <algorithm>  // for std::sort

// PCL utilities
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/impl/uniform_sampling.hpp>

// ROS utilities
#include "ros/ros.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include "sensor_msgs/PointCloud2.h"
#include "livox_ros_driver/CustomMsg.h"

// Add ikdtree
#include <ikdTree/ikd_Tree.h>

// Basalt
#include "basalt/spline/se3_spline.h"
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/spline/ceres_local_param.hpp"
#include "basalt/spline/posesplinex.h"

// Custom built utilities
#include "utility.h"
#include "GaussianProcess.hpp"

// Topics
#include "cf_msgs/Tdoa.h"
#include "cf_msgs/Tof.h"

using namespace std;

boost::shared_ptr<ros::NodeHandle> nh_ptr;

template <typename Scalar = double, int RowSize = Dynamic, int ColSize = Dynamic>
Matrix<Scalar, RowSize, ColSize> load_dlm(const std::string &path, string dlm, int r_start = 0, int col_start = 0)
{
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    int row_idx = -1;
    int rows = 0;
    while (std::getline(indata, line))
    {
        row_idx++;
        if (row_idx < r_start)
            continue;

        std::stringstream lineStream(line);
        std::string cell;
        int col_idx = -1;
        while (std::getline(lineStream, cell, dlm[0]))
        {
            if (cell == dlm || cell.size() == 0)
                continue;

            col_idx++;
            if (col_idx < col_start)
                continue;

            values.push_back(std::stod(cell));
        }

        rows++;
    }

    return Map<const Matrix<Scalar, RowSize, ColSize, RowMajor>>(values.data(), rows, values.size() / rows);
}

typedef sensor_msgs::Imu  ImuMsg    ;
typedef ImuMsg::ConstPtr  ImuMsgPtr ;
typedef cf_msgs::Tdoa     TdoaMsg   ;
typedef TdoaMsg::ConstPtr TdoaMsgPtr;
typedef cf_msgs::Tof      TofMsg    ;
typedef TofMsg::ConstPtr  TofMsgPtr ;

const double POSINF =  std::numeric_limits<double>::infinity();
const double NEGINF = -std::numeric_limits<double>::infinity();

vector<SE3d> anc_pose;

double gpDt = 0.02;
Matrix3d gpQr;
Matrix3d gpQc;

bool auto_exit;
int  WINDOW_SIZE = 4;
int  SLIDE_SIZE  = 2;

GaussianProcessPtr traj;

bool fuse_tdoa = true;
bool fuse_tof  = false;
bool fuse_imu  = true;

struct UwbImuBuf
{
    deque<TdoaMsgPtr> tdoaBuf;
    deque<TofMsgPtr>  tofBuf;
    deque<ImuMsgPtr>  imuBuf;

    mutex tdoaBuf_mtx;
    mutex tofBuf_mtx;
    mutex imuBuf_mtx;

    double minTime()
    {
        double tmin = std::numeric_limits<double>::infinity();
        if (tdoaBuf.size() != 0 && fuse_tdoa)
            tmin = min(tmin, tdoaBuf.front()->header.stamp.toSec());
        if (tofBuf.size() != 0 && fuse_tof)
            tmin = min(tmin, tofBuf.front()->header.stamp.toSec());
        if (imuBuf.size() != 0 && fuse_imu)
            tmin = min(tmin, imuBuf.front()->header.stamp.toSec());
        return tmin;
    }

    double maxTime()
    {
        double tmax = -std::numeric_limits<double>::infinity();
        if (tdoaBuf.size() != 0 && fuse_tdoa)
            tmax = max(tmax, tdoaBuf.back()->header.stamp.toSec());
        if (tofBuf.size() != 0 && fuse_tof)
            tmax = max(tmax, tofBuf.back()->header.stamp.toSec());
        if (imuBuf.size() != 0 && fuse_imu)
            tmax = max(tmax, imuBuf.back()->header.stamp.toSec());
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
        if (fuse_tdoa) transferDataOneBuf(tdoaBuf, other.tdoaBuf, other.tdoaBuf_mtx, tmax);
        if (fuse_tof ) transferDataOneBuf(tofBuf,  other.tofBuf,  other.tofBuf_mtx,  tmax);
        if (fuse_imu ) transferDataOneBuf(imuBuf,  other.imuBuf,  other.imuBuf_mtx,  tmax);
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
        if (fuse_tdoa) slideForwardOneBuf(tdoaBuf, tremove);
        if (fuse_tof ) slideForwardOneBuf(tofBuf,  tremove);
        if (fuse_imu ) slideForwardOneBuf(imuBuf,  tremove);
    }
};
UwbImuBuf UIBuf;

ros::Subscriber tdoaSub;
ros::Subscriber tofSub ;
ros::Subscriber imuSub ;

void tdoaCb(const TdoaMsgPtr &msg)
{
    lock_guard<mutex> lg(UIBuf.tdoaBuf_mtx);
    UIBuf.tdoaBuf.push_back(msg);
    // printf(KCYN "Receive tdoa\n" RESET);
}

void tofCb(const TofMsgPtr &msg)
{
    lock_guard<mutex> lg(UIBuf.tofBuf_mtx);
    UIBuf.tofBuf.push_back(msg);
    // printf(KBLU "Receive tof\n" RESET);
}

void imuCb(const ImuMsgPtr &msg)
{
    lock_guard<mutex> lg(UIBuf.imuBuf_mtx);
    UIBuf.imuBuf.push_back(msg);
    // printf(KMAG "Receive imu\n" RESET);
}

void processData()
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
        while(traj->getMaxTime() < newMaxTime)
            traj->extendOneKnot();

        // Step 3: Optimization
        // ...

        // Step 4: Report, visualize
        printf("Traj: %f. Sw: %.3f -> %.3f. Buf: %d, %d, %d\n",
                traj->getMaxTime(), swUIBuf.minTime(), swUIBuf.maxTime(),
                UIBuf.tdoaBuf.size(), UIBuf.tofBuf.size(), UIBuf.imuBuf.size());


        // Step 5: Slide the window forward
        if (traj->getNumKnots() >= WINDOW_SIZE)
        {
            double removeTime = traj->getKnotTime(traj->getNumKnots() - SLIDE_SIZE);
            swUIBuf.slideForward(removeTime);
        }
    }
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
    string anchor_pose_path;
    nh_ptr->getParam("anchor_pose_path", anchor_pose_path);
    // Load the anchor pose 
    MatrixXd anc_pose_ = load_dlm(anchor_pose_path, ",", 1, 0);
    for(int ridx = 0; ridx < anc_pose_.rows(); ridx++)
    {
        Vector4d Q_; Q_ << anc_pose_.block<1, 4>(ridx, 3).transpose();
        Quaternd Q(Q_.w(), Q_.x(), Q_.y(), Q_.z());
        Vector3d P = anc_pose_.block<1, 3>(ridx, 0).transpose();
        anc_pose.push_back(SE3d(Q, P));

        myTf tf(anc_pose.back());
        printf("Anchor: %2d. Pos: %6.3f, %6.3f, %6.3f. Rot: %4.0f, %4.0f, %4.0f.\n",
                ridx, P.x(), P.y(), P.z(), tf.yaw(), tf.pitch(), tf.roll());
    }

    // Topics to subscribe to
    string tdoa_topic; nh_ptr->getParam("tdoa_topic", tdoa_topic);
    string tof_topic;  nh_ptr->getParam("tof_topic", tof_topic);
    string imu_topic;  nh_ptr->getParam("imu_topic", imu_topic);
    fuse_tdoa = Util::GetBoolParam(nh_ptr, "fuse_tdoa", fuse_tdoa);
    fuse_tof  = Util::GetBoolParam(nh_ptr, "fuse_tof" , fuse_tof );
    fuse_imu  = Util::GetBoolParam(nh_ptr, "fuse_imu" , fuse_imu );

    // Subscribe to the topics
    tdoaSub = nh_ptr->subscribe(tdoa_topic, 10, tdoaCb);
    tofSub  = nh_ptr->subscribe(tof_topic,  10, tofCb);
    imuSub  = nh_ptr->subscribe(imu_topic,  10, imuCb);

    // while(ros::ok())
    //     this_thread::sleep_for(chrono::milliseconds(100));

    // Time to check the buffers and perform optimization
    nh_ptr->getParam("WINDOW_SIZE", WINDOW_SIZE);
    nh_ptr->getParam("SLIDE_SIZE", SLIDE_SIZE);

    // Create the trajectory
    traj = GaussianProcessPtr(new GaussianProcess(gpDt, gpQr, gpQc, true));

    // Wait to get the initial time
    while(ros::ok())
    {
        if(UIBuf.minTime() == POSINF)
        {
            this_thread::sleep_for(chrono::milliseconds(5));
            ros::spinOnce();
            continue;
        }

        double t0 = UIBuf.minTime();
        traj->setStartTime(t0);
        traj->setKnot(0, GPState(t0));
        break;
    }
    printf(KGRN "Start time: %f\n" RESET, traj->getMinTime());

    // Start polling and processing the data
    thread pdthread(processData);

    // Spin
    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    return 0;
}