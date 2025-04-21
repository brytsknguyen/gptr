#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_cpp/storage_options.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include "cf_msgs/msg/tdoa.hpp"
#include "cf_msgs/msg/tof.hpp"

#include "ceres_calib_spline_se3.h"
#include "ceres_calib_spline_split.h"
#include "basalt/so3_spline.h"
#include <sophus/average.hpp>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

using namespace std;

Eigen::Vector3d bg = Eigen::Vector3d::Zero();
Eigen::Vector3d ba = Eigen::Vector3d::Zero();
Eigen::Vector3d g = Eigen::Vector3d(0, 0, 9.81);
const Eigen::Vector3d P_I_tag = Eigen::Vector3d(-0.012, 0.001, 0.091);
double w_tdoa = 0.1;
double GYR_N = 10;
double GYR_W = 10;
double ACC_N = 0.5;
double ACC_W = 10;
double tdoa_loss_thres = -1.0;
std::string traj_save_path;

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
    std::cout << "load IMU messages: " << UIBuf.imu_data.size() << " TDOA messages: " << UIBuf.tdoa_data.size() << std::endl;
}

template <class SplineT>
string AssessTraj(UwbImuBuf &data, SplineT &traj, CloudPosePtr &gtPoseCloud, std::string pose_type,
                std::map<uint16_t, Eigen::Vector3d> &anc_pose_, map<string, double> &report)
{
    Eigen::Vector3d gravity_sum(0, 0, 0);
    size_t n_imu = 20;
    for (size_t i = 0; i < n_imu; i++) {
        gravity_sum += data.imu_data.at(i).acc;
    }
    gravity_sum /= n_imu;  
    std::cout << "g: " << gravity_sum.transpose() << std::endl;
    traj.setG(gravity_sum);    

    int num_imu = 0;
    int num_tdoa = 0;
    double tmin = traj.minTimes();
    double tmax = traj.maxTimes();  

    for (auto &imu : data.imu_data) {
        if (imu.t >= tmin && imu.t <= tmax) {
            traj.addGyroMeasurement(imu.gyro, imu.t*1e9, GYR_N);
            traj.addAccelMeasurement(imu.acc, imu.t*1e9, ACC_N);
            num_imu++;
        }
    }

    for (auto &tdoa : data.tdoa_data) {
        if (tdoa.t >= tmin && tdoa.t <= tmax) {
            traj.addTDOAMeasurement(tdoa.data, tdoa.t*1e9, anc_pose_[tdoa.idA], anc_pose_[tdoa.idB], P_I_tag, w_tdoa);
            num_tdoa++;
        }
    }
    TicToc tt_solve;
    ceres::Solver::Summary summary = traj.optimize();
    tt_solve.Toc();
    std::cout << summary.FullReport() << std::endl;


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
    for(auto &pose : gtPoseCloud->points)
    {
        double ts = pose.t;

        if (ts < tmin || ts > tmax)
            continue;

        myTf poseEst = myTf(traj.getPose(ts*1e9));
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
    

    // RINFO("Drafting report ...");
    TicToc tt_report;
    string report_;
    report_ = myprintf(
        "Pose group: %s. Dt: %.3f. "
        "Tslv: %.0f. Iterations: %d.\n"
        "J0: %16.3f.\n"
        "JK: %16.3f.\n"
        "RMSE: POS: %.12f. POSE: %.12f.\n"
        ,
        pose_type,
        traj.getDt(),
        tt_solve.GetLastStop(), summary.iterations.size(),
        summary.initial_cost, 
        summary.final_cost,
        pos_rmse, se3_rmse
    );

    report["iter"]    = summary.iterations.size();
    report["tslv"]    = summary.total_time_in_seconds;
    report["rmse"]    = pos_rmse;
    report["J0"]      = summary.initial_cost;
    report["JK"]      = summary.final_cost;  

    return report_;
};       


int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::Node::SharedPtr nh_ptr = rclcpp::Node::make_shared("splineui");

    string anchor_pose_path;
    Util::GetParam(nh_ptr, "anchor_pose_path", anchor_pose_path);
    std::map<uint16_t, Eigen::Vector3d> anc_pose_ = getAnchorListFromUTIL(anchor_pose_path);

    std::string bag_file;
    Util::GetParam(nh_ptr, "bag_file", bag_file);
    readBag(bag_file);  

    Util::GetParam(nh_ptr, "w_tdoa", w_tdoa);
    Util::GetParam(nh_ptr, "GYR_N", GYR_N);
    Util::GetParam(nh_ptr, "GYR_W", GYR_W);
    Util::GetParam(nh_ptr, "ACC_N", ACC_N);
    Util::GetParam(nh_ptr, "ACC_W", ACC_W);
    Util::GetParam(nh_ptr, "tdoa_loss_thres", tdoa_loss_thres);      

    double tskew0 = 1.0;
    double tskewmax = 1.0;
    double tskewstep = 0.1;
    Util::GetParam(nh_ptr, "tskew0", tskew0);
    Util::GetParam(nh_ptr, "tskewmax", tskewmax);
    Util::GetParam(nh_ptr, "tskewstep", tskewstep);  
    vector<double> Dtstep = {0.01};
    Util::GetParam(nh_ptr, "Dtstep", Dtstep);  
    Util::GetParam(nh_ptr, "traj_save_path", traj_save_path);

    constexpr int N = 4;
    fs::create_directories(traj_save_path);
    std::ofstream logfile(traj_save_path + "/splineui.csv", std::ios::out);
    logfile << std::fixed << std::setprecision(6);
    logfile << "tskew,dt,"   
               "so3xr3_tslv,se3_tslv,"
               "so3xr3_JK,se3_JK,"
               "so3xr3_rmse,se3_rmse\n";       

 

    for(double tskew = tskew0; tskew <= tskewmax; tskew += tskewstep)
    {
        UwbImuBuf UIBuf_scale = UIBuf; 
        #pragma omp parallel for num_threads(MAX_THREADS)
        for(auto &tdoadata : UIBuf_scale.tdoa_data)
            tdoadata.t /= tskew;

        #pragma omp parallel for num_threads(MAX_THREADS)
        for(auto &imudata : UIBuf_scale.imu_data)
        {
            imudata.t /= tskew;
            imudata.acc *= (tskew*tskew);
            imudata.gyro *= tskew;
        }
        
	CloudPosePtr gtPoseCloud_scale(new CloudPose());
	*gtPoseCloud_scale = *gtrPoseCloud;

        #pragma omp parallel for num_threads(MAX_THREADS)
        for(auto &pose : gtPoseCloud_scale->points) {
            pose.t /= tskew;
        }        

        for(double &m : Dtstep)
        {
            double deltaTm = m;
            int64_t dt_ns = deltaTm * 1e9;
            int64_t t0 = UIBuf_scale.minTime() * 1e9;
            int64_t tmax = UIBuf_scale.maxTime() * 1e9;

            SE3d initial_pose;
            initial_pose.translation() = Eigen::Vector3d(1.25, 0.0, 0.07);
            CeresCalibrationSplineSplit<N> trajSO3xR3(dt_ns, t0);
            trajSO3xR3.init(initial_pose, (tmax - t0) / dt_ns + N);
            CeresCalibrationSplineSe3<N> trajSE3(dt_ns, t0);
            trajSE3.init(initial_pose, (tmax - t0) / dt_ns + N);

            map<string, double> so3xr3_report;
            map<string, double> se3_report;

            string report_SO3xR3_by_SO3xR3 = AssessTraj<CeresCalibrationSplineSplit<N>>(UIBuf_scale, trajSO3xR3, gtPoseCloud_scale, "SO3xR3", anc_pose_, so3xr3_report);
            string report_SO3xR3_by_SE3___ = AssessTraj<CeresCalibrationSplineSe3<N>>(UIBuf_scale, trajSE3, gtPoseCloud_scale, "SE3", anc_pose_, se3_report);
       

            // Save the rmse result to the log
            logfile << tskew << ","
                    << deltaTm << ","
                    << so3xr3_report["tslv"] << ","
                    << se3_report["tslv"] << ","
                    << so3xr3_report["JK"] << ","
                    << se3_report["JK"] << ","
                    << so3xr3_report["rmse"] << ","
                    << se3_report["rmse"] 
                    << endl;
        }
    }    
    logfile.close();    

    return 0;
}
