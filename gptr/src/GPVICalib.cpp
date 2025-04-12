#include "unistd.h"
#include <algorithm> // for std::sort

// // PCL utilities
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/filters/uniform_sampling.h>
// #include <pcl/filters/impl/uniform_sampling.hpp>

// For json parser
#include <opencv2/opencv.hpp>

// ROS utilities
// #include "ros/ros.h"
// #include "rosbag/bag.h"
// #include "rosbag/view.h"
// #include "sensor_msgs/msg/point_cloud2.hpp"
// #include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.h"

// Custom built utilities
#include "GaussianProcess.hpp"
#include "GPVICalib.hpp"
#include "utility.h"

namespace fs = std::filesystem;

using namespace std;

NodeHandlePtr nh_ptr;

std::map<int, Eigen::Vector3d> getCornerPosition3D(const std::string &data_path)
{
    std::map<int, Eigen::Vector3d> corner_list;
    std::ifstream infile(data_path);
    std::string line;
    while (std::getline(infile, line))
    {
        int idx;
        double x, y, z;
        char comma;
        std::istringstream iss(line);
        iss >> idx >> comma >> x >> comma >> y >> comma >> z;
        Eigen::Vector3d pos(x, y, z);
        corner_list[idx] = pos;
    }
    infile.close();
    std::cout << "loaded " << corner_list.size() << " 3D positions of corners" << std::endl;
    return corner_list;
}

void getCornerPosition2D(const std::string &data_path, vector<CornerData> &corner_meas)
{
    corner_meas.clear();
    std::string line;
    std::ifstream infile;
    infile.open(data_path);
    if (!infile)
    {
        std::cerr << "Unable to open file: " << data_path << std::endl;
        exit(1);
    }
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);

        double t_s, px, py;
        char comma;
        iss >> t_s >> comma;

        vector<Eigen::Vector2d> corners;
        vector<int> ids;
        int idx = 0;
        for (; iss >> px >> comma >> py; iss >> comma)
        {
            if (px < 0 || py < 0)
            {
                idx++;
                continue;
            }
            corners.push_back(Eigen::Vector2d(px, py));
            ids.push_back(idx);
            idx++;
        }
        corner_meas.emplace_back(t_s, ids, corners);
    }
    infile.close();
    std::cout << "loaded " << corner_meas.size() << " images wih corner positions" << std::endl;
}

void getCameraModel(const std::string &data_path, CameraCalibration &cam_calib)
{
    cv::FileStorage fsSettings(data_path, cv::FileStorage::READ);

    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings " << data_path << std::endl;
        return;
    }

    cv::FileNode root = fsSettings["value0"];
    cv::FileNode T_imu_cam = root["T_imu_cam"];
    cv::FileNode intrinsics = root["intrinsics"];
    cv::FileNode resolution = root["resolution"];

    for (int i = 0; i < 2; i++)
    {
        double x, y, z;
        x = T_imu_cam[i]["px"];
        y = T_imu_cam[i]["py"];
        z = T_imu_cam[i]["pz"];

        double qx, qy, qz, qw;
        qx = T_imu_cam[i]["qx"];
        qy = T_imu_cam[i]["qy"];
        qz = T_imu_cam[i]["qz"];
        qw = T_imu_cam[i]["qw"];

        cv::FileNode intrinsic = intrinsics[i];
        cv::FileNode subintrinsic = intrinsic["intrinsics"];

        double fx, fy, cx, cy, xi, alpha;
        fx = subintrinsic["fx"];
        fy = subintrinsic["fy"];
        cx = subintrinsic["cx"];
        cy = subintrinsic["cy"];
        xi = subintrinsic["xi"];
        alpha = subintrinsic["alpha"];

        Eigen::Quaterniond qic(qw, qx, qy, qz);
        Sophus::SE3d Tic;
        Tic.translation() = Eigen::Vector3d(x, y, z);
        Tic.so3() = Sophus::SO3d::fitToSO3(qic.toRotationMatrix());
        cam_calib.T_i_c.push_back(Tic);

        DoubleSphereCamera<double> intr;
        intr.setFromInit(fx, fy, cx, cy, xi, alpha);
        cam_calib.intrinsics.push_back(intr);
        std::cout << "intr.setFromI: " << intr.getParam().transpose() << std::endl;
        std::cout << "qic: " << qic.coeffs().transpose() << " Ric: " << Tic.so3().matrix() << std::endl;
    }
}

void getIMUMeasurements(const std::string &data_path, vector<IMUData> &imu_meas)
{
    imu_meas.clear();
    std::ifstream infile(data_path + "imu_data.csv");
    std::string line;
    while (std::getline(infile, line))
    {
        if (line[0] == '#')
            continue;

        std::stringstream ss(line);

        char tmp;
        uint64_t timestamp;
        Eigen::Vector3d gyro, accel;

        ss >> timestamp >> tmp >> gyro[0] >> tmp >> gyro[1] >> tmp >> gyro[2]
           >> tmp >> accel[0] >> tmp >> accel[1] >> tmp >> accel[2];

        double t_s = timestamp * 1.0e-9;

        IMUData imu(t_s, accel, gyro);
        imu_meas.push_back(imu);
    }
    infile.close();
    std::cout << "loaded " << imu_meas.size() << " IMU measurements" << std::endl;
}

void getGT(const std::string &data_path, CloudPosePtr &gtrPoseCloud)
{
    gtrPoseCloud->clear();

    std::ifstream infile(data_path + "cam_pose.csv");
    std::string line;
    while (std::getline(infile, line))
    {
        if (line[0] == '#')
            continue;

        std::stringstream ss(line);

        char tmp;
        uint64_t timestamp;
        Eigen::Quaterniond q;
        Eigen::Vector3d pos;

        ss >> timestamp >> tmp >> pos[0] >> tmp >> pos[1] >> tmp >> pos[2]
           >> tmp >> q.w() >> tmp >> q.x() >> tmp >> q.y() >> tmp >> q.z();

        // geometry_msgs::msg::PoseStamped traj_msg;
        // traj_msg.header.stamp = rclcpp::Time(timestamp);
        // traj_msg.pose.position.x = pos.x();
        // traj_msg.pose.position.y = pos.y();
        // traj_msg.pose.position.z = pos.z();
        // traj_msg.pose.orientation.w = q.w();
        // traj_msg.pose.orientation.x = q.x();
        // traj_msg.pose.orientation.y = q.y();
        // traj_msg.pose.orientation.z = q.z();

        PointPose pose; pose.t = timestamp / 1.0e9;
        pose.x = pos.x(); pose.y = pos.y(); pose.z = pos.z();
        pose.qx = q.x(); pose.qy = q.y(); pose.qz = q.z(); pose.qw = q.w();

        gtrPoseCloud->push_back(pose);
    }

    infile.close();
    std::cout << "loaded " << gtrPoseCloud->size() << " gt data" << std::endl;
}

const double POSINF = std::numeric_limits<double>::infinity();
const double NEGINF = -std::numeric_limits<double>::infinity();

double gpDt = 0.02;
Matrix3d gpQr;
Matrix3d gpQc;

bool auto_exit;
int WINDOW_SIZE = 4;
int SLIDE_SIZE = 2;
double w_corner = 0.1;
double GYR_N = 10;
double GYR_W = 10;
double ACC_N = 0.5;
double ACC_W = 10;
double corner_loss_thres = -1;
double mp_loss_thres = -1;

GaussianProcessPtr traj;

bool acc_ratio = false;
bool gyro_unit = false;

struct CameraImuBuf
{
    vector<CornerData> corner_data_cam0;
    vector<CornerData> corner_data_cam1;
    vector<IMUData> imu_data;

    double minTime()
    {
        double tmin = std::numeric_limits<double>::infinity();
        if (corner_data_cam0.size() != 0)
            tmin = min(tmin, corner_data_cam0.front().t);
        if (corner_data_cam1.size() != 0)
            tmin = min(tmin, corner_data_cam1.front().t);
        if (imu_data.size() != 0)
            tmin = min(tmin, imu_data.front().t);
        return tmin;
    }

    double maxTime()
    {
        double tmax = -std::numeric_limits<double>::infinity();
        if (corner_data_cam0.size() != 0)
            tmax = max(tmax, corner_data_cam0.back().t);
        if (corner_data_cam1.size() != 0)
            tmax = max(tmax, corner_data_cam1.back().t);
        if (imu_data.size() != 0)
            tmax = max(tmax, imu_data.back().t);
        return tmax;
    }
};

CameraImuBuf CIBuf;
// RosPathMsg est_path;
CloudPosePtr gtrPoseCloud(new CloudPose());

// rclcpp::Publisher<> gt_pub;
rclcpp::Publisher<RosPc2Msg>::SharedPtr gtrPoseCloud_pub;
rclcpp::Publisher<RosPc2Msg>::SharedPtr est_pub;
rclcpp::Publisher<RosOdomMsg>::SharedPtr odom_pub;
rclcpp::Publisher<RosPc2Msg>::SharedPtr knot_pub;
rclcpp::Publisher<RosPc2Msg>::SharedPtr corner_pub;

Eigen::Vector3d bg = Eigen::Vector3d::Zero();
Eigen::Vector3d ba = Eigen::Vector3d::Zero();
Eigen::Vector3d g = Eigen::Vector3d(0, 0, 9.81);

bool if_save_traj;
std::string traj_save_path;

void publishCornerPos(std::map<int, Eigen::Vector3d> &corner_pos_3d)
{
    RosPc2Msg corners_msg;
    pcl::PointCloud<pcl::PointXYZ> pc_corners;
    for (const auto &iter : corner_pos_3d)
    {
        Eigen::Vector3d pos_i = iter.second;
        pc_corners.points.push_back(pcl::PointXYZ(pos_i[0], pos_i[1], pos_i[2]));
    }
    pcl::toROSMsg(pc_corners, corners_msg);
    corners_msg.header.stamp = rclcpp::Clock().now();
    corners_msg.header.frame_id = "map";
    corner_pub->publish(corners_msg);
}

// void processData(GaussianProcessPtr traj, GPMVICalibPtr gpmui, std::map<int, Eigen::Vector3d> corner_pos_3d,
//                  CameraCalibration *cam_calib)
// {
//     // Step: Optimization
//     TicToc tt_solve;
//     double tmin = traj->getKnotTime(0) + 1e-3;                       // Start time of the sliding window
//     double tmax = traj->getKnotTime(traj->getNumKnots() - 1) + 1e-3; // End time of the sliding window
//     double tmid = tmin + SLIDE_SIZE * traj->getDt() + 1e-3;          // Next start time of the sliding window,
//                                                                         // also determines the marginalization time limit
//     gpmui->Evaluate(tmin, tmax, tmid, traj, bg, ba, g, cam_calib, CIBuf.imu_data,
//                     CIBuf.corner_data_cam0, CIBuf.corner_data_cam1, corner_pos_3d, w_corner,
//                     GYR_N, ACC_N, GYR_W, ACC_W, corner_loss_thres, mp_loss_thres, false);
//     tt_solve.Toc();

//     // for (int i = 0; i < 2; i++)
//     // {
//     //     std::cout << "Ric"   << i << ": \n" << cam_calib->T_i_c[i].so3().matrix() << std::endl;
//     //     std::cout << "tic"   << i << ": \n" << cam_calib->T_i_c[i].translation().transpose() << std::endl;
//     //     std::cout << "param" << i << ": \n" << cam_calib->intrinsics[i].getParam().transpose() << std::endl;
//     // }
//     // std::cout << "ba: " << ba.transpose() << endl;
//     // std::cout << "bg: " << bg.transpose() << endl;
//     // std::cout << "g : " << g.transpose()  << endl;

//     // Visualize knots
//     // pcl::PointCloud<pcl::PointXYZ> est_knots;
//     // for (int i = 0; i < traj->getNumKnots(); i++)
//     // {
//     //     Eigen::Vector3d knot_pos = traj->getKnotPose(i).translation();
//     //     est_knots.points.push_back(pcl::PointXYZ(knot_pos.x(), knot_pos.y(), knot_pos.z()));
//     // }

//     // RosPc2Msg knot_msg;
//     // pcl::toROSMsg(est_knots, knot_msg);

//     // // while(rclcpp::ok())
//     // {           
//     //     publishCornerPos(corner_pos_3d);

//     //     gtrPoseCloud.header.stamp = rclcpp::Clock().now();
//     //     // gtrPoseCloud_pub->publish(gtrPoseCloud);
//     //     Util::publishCloud(gtrPoseCloud_pub, gtrPoseCloud, rclcpp::Clock().now(), "world")

//     //     knot_msg.header.stamp = rclcpp::Clock().now();
//     //     knot_msg.header.frame_id = "map";
//     //     knot_pub->publish(knot_msg);

//     //     this_thread::sleep_for(chrono::milliseconds(100));
//     // }
// }

// void saveTraj(GaussianProcessPtr traj)
// {
//     if (!std::filesystem::is_directory(traj_save_path) || !std::filesystem::exists(traj_save_path))
//     {
//         std::filesystem::create_directories(traj_save_path);
//     }
//     std::string traj_file_name = traj_save_path + "traj.txt";
//     std::ofstream f_traj(traj_file_name);
//     for (int i = 0; i < gtrPoseCloud->size(); i++)
//     {
//         double t_gt = rclcpp::Time(gtrPoseCloud.poses[i].header.stamp).seconds();
//         auto us = traj->computeTimeIndex(t_gt);
//         int u = us.first;
//         double s = us.second;

//         if (u < 0 || u + 1 >= traj->getNumKnots())
//         {
//             continue;
//         }
//         auto est_pose = traj->pose(t_gt);
//         Eigen::Vector3d est_pos = est_pose.translation();
//         Eigen::Quaterniond est_ort = est_pose.unit_quaternion();
//         f_traj << std::fixed << t_gt << std::setprecision(7)
//                << " " << est_pos.x() << " " << est_pos.y() << " " << est_pos.z()
//                << " " << est_ort.x() << " " << est_ort.y() << " " << est_ort.z() << " " << est_ort.w() << std::endl;
//     }
//     f_traj.close();

//     std::string gt_file_name = traj_save_path + "gt.txt";
//     std::ofstream f_gt(gt_file_name);
//     for (int i = 0; i < gtrPoseCloud.poses.size(); i++)
//     {
//         double t_gt = rclcpp::Time(gtrPoseCloud.poses[i].header.stamp).seconds();

//         f_gt << std::fixed << t_gt << std::setprecision(7)
//              << " " << gtrPoseCloud.poses[i].pose.position.x << " " << gtrPoseCloud.poses[i].pose.position.y << " " << gtrPoseCloud.poses[i].pose.position.z
//              << " " << gtrPoseCloud.poses[i].pose.orientation.x << " " << gtrPoseCloud.poses[i].pose.orientation.y << " " << gtrPoseCloud.poses[i].pose.orientation.z << " " << gtrPoseCloud.poses[i].pose.orientation.w << std::endl;
//     }
//     f_gt.close();
// }


void Visualization(map<int, Vector3d> &corner_pos_3d, GaussianProcessPtr &traj)
{
    publishCornerPos(corner_pos_3d);
    Util::publishCloud(gtrPoseCloud_pub, *gtrPoseCloud, rclcpp::Clock().now(), "world");

    CloudPosePtr estPoseCloud(new CloudPose());
    for(int kidx = 0; kidx < traj->getNumKnots(); kidx++)
        estPoseCloud->push_back(myTf(traj->getKnotPose(kidx)).Pose6D());
    
    Util::publishCloud(est_pub, *estPoseCloud, rclcpp::Clock().now(), "world");
}

int main(int argc, char **argv)
{
    // Initialize the node
    rclcpp::init(argc, argv);

    nh_ptr = rclcpp::Node::make_shared("GPVICalib");

    // Determine if we exit if no data is received after a while
    bool auto_exit = Util::GetBoolParam(nh_ptr, "auto_exit", false);

    // Parameters for the GP trajectory
    double gpQr_ = 1.0, gpQc_ = 1.0;
    Util::GetParam(nh_ptr, "gpDt", gpDt);
    Util::GetParam(nh_ptr, "gpQr", gpQr_);
    Util::GetParam(nh_ptr, "gpQc", gpQc_);
    gpQr = gpQr_ * Matrix3d::Identity(3, 3);
    gpQc = gpQc_ * Matrix3d::Identity(3, 3);

    POSE_GROUP pose_type; string pose_type_;
    Util::GetParam(nh_ptr, "pose_type", pose_type_);
    pose_type = pose_type_ == "SE3" ? POSE_GROUP::SE3 : POSE_GROUP::SO3xR3;
    RINFO("Pose representation: %s. Num: %d\n", pose_type_.c_str(), pose_type);

    double lie_epsilon = 1e-3;
    Util::GetParam(nh_ptr, "lie_epsilon", lie_epsilon);

    bool use_approx_drv =  Util::GetBoolParam(nh_ptr, "use_approx_drv", true);

    // Find the path to data
    string data_path;
    Util::GetParam(nh_ptr, "data_path", data_path);

    // Load the corner positions in 3D and measurements
    string corner3d_path = data_path + "corners3D.csv";
    std::cout << "data_path: " << data_path << " corner3d_path: " << corner3d_path << std::endl;
    std::map<int, Eigen::Vector3d> corner_pos_3d = getCornerPosition3D(corner3d_path);

    string corner2d_path0 = data_path + "corners2D_cam0.csv";
    getCornerPosition2D(corner2d_path0, CIBuf.corner_data_cam0);
    string corner2d_path1 = data_path + "corners2D_cam1.csv";
    getCornerPosition2D(corner2d_path1, CIBuf.corner_data_cam1);

    CameraCalibration cam_calib;

    string cam_path = data_path + "initial_calibration.json";
    getCameraModel(cam_path, cam_calib);

    getIMUMeasurements(data_path, CIBuf.imu_data);
    getGT(data_path, gtrPoseCloud);

    // Publish estimates
    knot_pub = nh_ptr->create_publisher<RosPc2Msg>("/estimated_knot", 100);
    // gt_pub = nh_ptr->create_publisher<RosOdomMsg>("/ground_truth", 10);
    gtrPoseCloud_pub = nh_ptr->create_publisher<RosPc2Msg>("/ground_truth_path", 100);
    est_pub = nh_ptr->create_publisher<RosPc2Msg>("/estimated_trajectory", 100);
    odom_pub = nh_ptr->create_publisher<RosOdomMsg>("/estimated_pose", 100);
    corner_pub = nh_ptr->create_publisher<RosPc2Msg>("/corners", 100);

    // est_path.header.frame_id = "map";
    // gtrPoseCloud.header.frame_id = "map";

    // Time to check the buffers and perform optimization
    Util::GetParam(nh_ptr, "WINDOW_SIZE", WINDOW_SIZE);
    Util::GetParam(nh_ptr, "SLIDE_SIZE", SLIDE_SIZE);
    Util::GetParam(nh_ptr, "w_corner", w_corner);
    Util::GetParam(nh_ptr, "GYR_N", GYR_N);
    Util::GetParam(nh_ptr, "GYR_W", GYR_W);
    Util::GetParam(nh_ptr, "ACC_N", ACC_N);
    Util::GetParam(nh_ptr, "ACC_W", ACC_W);
    Util::GetParam(nh_ptr, "corner_loss_thres", corner_loss_thres);
    Util::GetParam(nh_ptr, "mp_loss_thres", mp_loss_thres);
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
    

    CameraImuBuf CIBuf_ = CIBuf;
    CloudPosePtr gtrPoseCloud_(new CloudPose()); *gtrPoseCloud_  = *gtrPoseCloud;
    GPMVICalibPtr gpmui(new GPMVICalib(nh_ptr));

    fs::create_directories(traj_save_path);
    std::ofstream logfile(traj_save_path + "/vicalib.csv", std::ios::out);
    logfile << std::fixed << std::setprecision(6);
    logfile << "tskew,dt,"
               "so3xr3ap_tslv,so3xr3cf_tslv,se3ap_tslv,se3cf_tslv,"
               "so3xr3ap_JK,so3xr3cf_JK,se3ap_JK,se3cf_JK,"
               "so3xr3ap_rmse,so3xr3cf_rmse,se3ap_rmse,se3cf_rmse\n";

    auto AssessTraj = [&corner_pos_3d, &cam_calib, &gpmui](CameraImuBuf &data, GaussianProcessPtr &traj, CloudPosePtr &gtrPoseCloud, map<string, double> &report) -> string
    {
        // double t0 = CIBuf.minTime();
        double t0 = max(CIBuf.imu_data.front().t, CIBuf.corner_data_cam0.front().t);
        traj->setStartTime(t0);
        SE3d initial_pose;
        Eigen::Matrix3d rwi;
        rwi << -0.9978650,  0.0135724, 0.0638772,
                0.0628005, -0.0687564, 0.9956550,
                0.0179054,  0.9975410, 0.0677573;
        initial_pose.so3() = Sophus::SO3d::fitToSO3(rwi);
        initial_pose.translation() = Eigen::Vector3d(0.290213, 0.393962, 0.642399);
        traj->setKnot(0, GPState(t0, initial_pose));

        Eigen::Matrix3d rai;
        rai << -0.9979450,  0.00867498, 0.0634844,
                0.0627571, -0.06752980, 0.9957420,
                0.0129251,  0.99768000, 0.0668466;

        for (size_t i = 0; i < CIBuf.imu_data.size(); i++)
        {
            const Eigen::Vector3d ad = CIBuf.imu_data[i].acc;
            if (std::abs(CIBuf.imu_data[i].t - CIBuf.corner_data_cam0[1].t) < 3000000 * 1e-9)
            {
                g = rai * ad;
                // std::cout << "g_a initialized with " << g.transpose() << std::endl;
                break;
            }
        }

        double newMaxTime = min(CIBuf.imu_data.back().t, CIBuf.corner_data_cam0.back().t);

        // Step 2: Extend the trajectory
        if (traj->getMaxTime() < newMaxTime && (newMaxTime - traj->getMaxTime()) > gpDt * 0.01)
            traj->extendKnotsTo(newMaxTime, GPState(t0, initial_pose));

        // // Start polling and processing the data
        // processData(traj, gpmui, corner_pos_3d, &cam_calib);

        double tmin = traj->getKnotTime(0) + 1e-3;                       // Start time of the sliding window
        double tmax = traj->getKnotTime(traj->getNumKnots() - 1) + 1e-3; // End time of the sliding window
        double tmid = tmin + SLIDE_SIZE * traj->getDt() + 1e-3;          // Next start time of the sliding window,

        string report_;
        gpmui->Evaluate(tmin, tmax, tmid, traj, bg, ba, g, &cam_calib,
                        CIBuf.imu_data, CIBuf.corner_data_cam0, CIBuf.corner_data_cam1,
                        corner_pos_3d, w_corner,
                        GYR_N, ACC_N, GYR_W, ACC_W, corner_loss_thres, mp_loss_thres, false,
                        gtrPoseCloud, report_, report);

        return report_;
    };

    for(double tskew = tskew0; tskew <= tskewmax; tskew += tskewstep)
    {
        for(double &m : Dtstep)
        {
            CIBuf = CIBuf_;
            *gtrPoseCloud = *gtrPoseCloud_;

            #pragma omp parallel for num_threads(MAX_THREADS)
            for(auto &cornerdata : CIBuf.corner_data_cam0)
                cornerdata.t /= tskew;

            #pragma omp parallel for num_threads(MAX_THREADS)
            for(auto &cornerdata : CIBuf.corner_data_cam1)
                cornerdata.t /= tskew;

            #pragma omp parallel for num_threads(MAX_THREADS)
            for(auto &imudata : CIBuf.imu_data)
            {
                imudata.t /= tskew;
                imudata.acc *= (tskew*tskew);
                imudata.gyro *= tskew;
            }

            #pragma omp parallel for num_threads(MAX_THREADS)
            for(auto &pose : gtrPoseCloud->points)
                pose.t /= tskew;

            double deltaTm = m;

            map<string, double> so3xr3ap_report;
            map<string, double> so3xr3cf_report;
            map<string, double> se3ap_report;
            map<string, double> se3cf_report;

            GaussianProcessPtr trajSO3xR3AP(new GaussianProcess(deltaTm, gpQr, gpQc, false, POSE_GROUP::SO3xR3, lie_epsilon, true));
            GaussianProcessPtr trajSO3xR3CF(new GaussianProcess(deltaTm, gpQr, gpQc, false, POSE_GROUP::SO3xR3, lie_epsilon, false));
            GaussianProcessPtr trajSE3AP(new GaussianProcess(deltaTm, gpQr, gpQc, false, POSE_GROUP::SE3, lie_epsilon, true));
            GaussianProcessPtr trajSE3CF(new GaussianProcess(deltaTm, gpQr, gpQc, false, POSE_GROUP::SE3, lie_epsilon, false));

            string report_SO3xR3_by_SO3xR3AP = AssessTraj(CIBuf, trajSO3xR3AP, gtrPoseCloud, so3xr3ap_report);
            string report_SO3xR3_by_SO3xR3CF = AssessTraj(CIBuf, trajSO3xR3CF, gtrPoseCloud, so3xr3cf_report);
            string report_SO3xR3_by_SE3AP___ = AssessTraj(CIBuf, trajSE3AP,    gtrPoseCloud, se3ap_report);
            string report_SO3xR3_by_SE3CF___ = AssessTraj(CIBuf, trajSE3CF,    gtrPoseCloud, se3cf_report);

            RINFO("VICalibTraj Dt=%2f, tskew: %.3f. %s", m, tskew, report_SO3xR3_by_SO3xR3AP.c_str());
            RINFO("VICalibTraj Dt=%2f, tskew: %.3f. %s", m, tskew, report_SO3xR3_by_SO3xR3CF.c_str());
            RINFO("VICalibTraj Dt=%2f, tskew: %.3f. %s", m, tskew, report_SO3xR3_by_SE3AP___.c_str());
            RINFO("VICalibTraj Dt=%2f, tskew: %.3f. %s", m, tskew, report_SO3xR3_by_SE3CF___.c_str());
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

            // Publish the estimated trajectory for visualization
            // Visualization(corner_pos_3d, trajSO3xR3AP);
        }
    }

    logfile.close();

    RINFO(KGRN"Program finished!"RESET);

    return 0;
}