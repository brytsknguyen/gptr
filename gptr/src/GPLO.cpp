#include "unistd.h"
#include <algorithm>  // for std::sort
#include <filesystem>
#include <boost/algorithm/string.hpp>

// PCL utilities
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/impl/uniform_sampling.hpp>

/* All needed for kdtree of custom point type----------*/
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
/* All needed for kdtree of custom point type----------*/

// ROS utilities
#include "sensor_msgs/msg/point_cloud2.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

// Add ikdtree
#include <ikdTree/ikd_Tree.h>

// Custom built utilities
#include "utility.h"
#include "CloudMatcher.hpp"
#include "GaussianProcess.hpp"
#include "i2EKFLO.hpp"
#include "LOAM.hpp"
#include "GPLO.hpp"

using namespace std;
namespace fs = std::filesystem;

// Node handle
NodeHandlePtr nh_ptr;

// Get the dense prior map
string priormap_file = "";

// Dense prior map
CloudXYZIPtr priormap(new CloudXYZI());

// Get the lidar bag file
string lidar_bag_file = "";

// Number of clouds to work with
int MAX_CLOUDS = -1;

// Time to skip the estimation
double SKIPPED_TIME = 0.0;
bool RECURRENT_SKIP = false;

// Get the imu topics
vector<string> imu_topic = {""};

// Get the lidar topics
vector<string> lidar_topic = {"/lidar_0/points"};

// Get the lidar type
vector<string> lidar_type = {"ouster"};

// Get the lidar stamp time (start /  end)
vector<string> stamp_time = {"start"};

// Check for log and load the GP trajectory with the control points in log
vector<long int> resume_from_log = {0};

// Get the prior map leaf size
double pmap_leaf_size = 0.15;
vector<double> cloud_ds;

// Kdtree for priormap
ikdtreePtr ikdTreeMap;
CloudPosePtr kfPose;

double kf_min_dis = 1.0;
double kf_min_angle = 1.0;

// Spline knot length
double deltaT = 0.01;

// ikdtree of the priormap
ikdtreePtr ikdtPM;

// Number of poses per knot in the extrinsic optimization
int SW_CLOUDNUM = 20;
int SW_CLOUDSTEP = 2;
bool VIZ_ONLY = false;

double t_shift = 0.0;
int max_outer_iter = 3;
int min_inner_iter = 3;
int max_inner_iter = 5;
int conv_thres = 3;
double dJ_conv_thres = 5.0;
vector<double> conv_dX_thres = {0.05, 0.2, 1.0, 0.05, 0.2, 1.0};
vector<double> change_thres = {0.5, 3.0, 8.0, 5.0};

// Range of ypr
vector<vector<vector<double>>> pr_range = {{{}, {}}, {{}, {}}};

vector<myTf<double>> T_B_Li_gndtr;

string log_dir = "/home/tmn/logs";
double log_period = 10;

bool runkf = false;

// Mutex for the node handle
mutex nh_mtx;

// Define the posespline
typedef std::shared_ptr<LOAM> LOAMPtr;
typedef std::shared_ptr<GaussianProcess> GaussianProcessPtr;

// vector<i2EKFLOPtr> i2EKFLO;
vector<LOAMPtr> gpmaplo;

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

template <typename PointType>
typename pcl::PointCloud<PointType>::Ptr uniformDownsample(const typename pcl::PointCloud<PointType>::Ptr &cloudin, double sampling_radius)
{
    if (sampling_radius > 0)
    {
        // Downsample the pointcloud
        pcl::UniformSampling<PointType> downsampler;
        downsampler.setRadiusSearch(sampling_radius);
        downsampler.setInputCloud(cloudin);

        typename pcl::PointCloud<PointType>::Ptr cloudout(new pcl::PointCloud<PointType>());
        downsampler.filter(*cloudout);
        return cloudout;
    }
    else
    {
        typename pcl::PointCloud<PointType>::Ptr cloudout(new pcl::PointCloud<PointType>());
        *cloudout = *cloudin;
        return cloudout;
    }
}

void getInitPose(int lidx,
                 const vector<vector<CloudXYZITPtr>> &clouds,
                 const vector<vector<rclcpp::Time>> &cloudstamp,
                 CloudXYZIPtr &priormap,
                 vector<double> &timestart,
                 const vector<double> &xyzypr_W_L0,
                 vector<CloudXYZIPtr> &pc0,
                 vector<myTf<double>> &tf_W_Li0,
                 vector<myTf<double>> &tf_W_Li0_refined)
{
    // Number of lidars
    int Nlidar = cloudstamp.size();

    // Time period to merge initial clouds
    double startup_merge_time = 3.0;

    assert(pc0.size() == Nlidar);
    assert(tf_W_Li0.size() == Nlidar);

    // Merge the pointclouds in the first few seconds
    pc0[lidx] = CloudXYZIPtr(new CloudXYZI());
    int Ncloud = cloudstamp[lidx].size();
    for(int cidx = 0; cidx < Ncloud; cidx++)
    {
        // Check if pointcloud is later
        if ((cloudstamp[lidx][cidx] - cloudstamp[lidx][0]).seconds() > startup_merge_time)
        {
            timestart[lidx] = cloudstamp[lidx][cidx].seconds();
            break;
        }
        // Merge lidar
        CloudXYZI temp; pcl::copyPointCloud(*clouds[lidx][cidx], temp);
        *pc0[lidx] += temp;
        // RINFO("P0 lidar %d, Cloud %d. Points: %d. Copied: %d", lidx, cidx, clouds[lidx][cidx]->size(), pc0[lidx]->size());
    }
    int Norg = pc0[lidx]->size();
    // Downsample the pointcloud
    pc0[lidx] = uniformDownsample<PointXYZI>(pc0[lidx], pmap_leaf_size);
    RINFO("Intial cloud of lidar %d, Points: %d -> %d\n", lidx, Norg, pc0[lidx]->size());
    // Find ICP alignment and refine
    CloudMatcher cm(0.1, 0.1);
    // Set the original position of the anchors
    Vector3d p_W_L0(xyzypr_W_L0[lidx*6 + 0], xyzypr_W_L0[lidx*6 + 1], xyzypr_W_L0[lidx*6 + 2]);
    Quaternd q_W_L0 = Util::YPR2Quat(xyzypr_W_L0[lidx*6 + 3], xyzypr_W_L0[lidx*6 + 4], xyzypr_W_L0[lidx*6 + 5]);
    myTf tf_W_L0(q_W_L0, p_W_L0);

    // // Find ICP pose
    // Matrix4f tfm_W_Li0;
    // double   icpFitness   = 0;
    // double   icpTime      = 0;
    // bool     icpconverged = cm.CheckICP(priormap, pc0[lidx], tf_W_L0.cast<float>().tfMat(), tfm_W_Li0, 0.2, 10, 1.0, icpFitness, icpTime);
    
    // tf_W_L0 = myTf(tfm_W_Li0);
    // RINFO("Lidar %d initial pose. %s. Time: %f. Fn: %f. XYZ: %f, %f, %f. YPR: %f, %f, %f.\n",
    //       lidx, icpconverged ? "Conv" : "Not Conv", icpTime, icpFitness,
    //       tf_W_L0.pos.x(), tf_W_L0.pos.y(), tf_W_L0.pos.z(),
    //       tf_W_L0.yaw(), tf_W_L0.pitch(), tf_W_L0.roll());
    
    // Find the refined pose
    IOAOptions ioaOpt;
    ioaOpt.init_tf = tf_W_L0;
    ioaOpt.max_iterations = 20;
    ioaOpt.show_report = true;
    ioaOpt.text = myprintf( "T_W_L(%d,0)_refined_%d", lidx, 10);
    IOASummary ioaSum;
    ioaSum.final_tf = ioaOpt.init_tf;
    cm.IterateAssociateOptimize(ioaOpt, ioaSum, priormap, pc0[lidx]);
    RINFO("Refined: \n");
    cout << ioaSum.final_tf.tfMat() << endl;
    
    // Save the result to external buffer
    tf_W_Li0[lidx] = ioaOpt.init_tf;
    tf_W_Li0_refined[lidx] = ioaSum.final_tf;

    return;
}

void syncLidar(const vector<CloudXYZITPtr> &cloudbufi, const vector<CloudXYZITPtr> &cloudbufj, vector<CloudXYZITPtr> &cloudji)
{
    int Ncloudi = cloudbufi.size();
    int Ncloudj = cloudbufj.size();

    int last_cloud_j = 0;
    for(int cidxi = 0; cidxi < Ncloudi; cidxi++)
    {
        // Create a cloud
        cloudji.push_back(CloudXYZITPtr(new CloudXYZIT()));

        const CloudXYZITPtr &cloudi = cloudbufi[cidxi];
        CloudXYZITPtr &cloudx = cloudji.back();

        // *cloudx = *cloudi;
        
        for(int cidxj = last_cloud_j; cidxj < Ncloudj; cidxj++)
        {
            const CloudXYZITPtr &cloudj = cloudbufj[cidxj];

            double tif = cloudi->points.front().t;
            double tib = cloudi->points.back().t;
            double tjf = cloudj->points.front().t;
            double tjb = cloudj->points.back().t;
            
            if (tjf > tib)
                break;
            
            if (tjb < tif)
                continue;

            // Now there is overlap, extract the points in the cloudi interval
            // ROS_ASSERT_MSG((tif <= tjf && tjf <= tib) || (tif <= tjb && tjb <= tib),
            //                "cidxi: %d. tif: %f. tib: %f. tjf: %f. tjb: %f",
            //                 cidxi, tif, tib, tjf, tjb);

            // Insert points to the cloudx
            for(auto &point : cloudj->points)
                if(point.t >= tif && point.t <= tib)
                    cloudx->push_back(point);
            
            last_cloud_j = cidxj;

            // RINFO("cloudj %d is split. Cloudx of cloudi %d now has %d points\n", last_cloudj, cidx1, cloudx->size());
        }
    }
}

void VisualizeGndtr(vector<CloudPosePtr> &gndtrCloud)
{
    // Number of lidars
    int Nlidar = gndtrCloud.size();
    
    // Create the publisher
    rclcpp::Publisher<RosPc2Msg>::SharedPtr pmpub = nh_ptr->create_publisher<RosPc2Msg>("/priormap_viz", 1);
    rclcpp::Publisher<RosPc2Msg>::SharedPtr gndtrPub[Nlidar];
    for(int idx = 0; idx < Nlidar; idx++)
        gndtrPub[idx] = nh_ptr->create_publisher<RosPc2Msg>(myprintf( "/lidar_%d/gndtr", idx), 10);

    // Publish gndtr every x seconds
    rclcpp::Rate rate(10);
    while(rclcpp::ok())
    {
        rclcpp::Time currTime = rclcpp::Clock().now();

        // Publish the prior map for visualization
        static int count = 0;
        count++;
        if (count < 50)
            Util::publishCloud(pmpub, *priormap, currTime, "world");

        // Publish the grountruth
        for(int lidx = 0; lidx < Nlidar; lidx++)
        {
            if(gndtrCloud[lidx]->size() == 0)
            {
                // RINFO(KYEL "GND pose is empty\n" RESET);
                continue;
            }

            // RINFO("Publish GND pose cloud of %d points\n", gndtrCloud[lidx]->size());
            Util::publishCloud(gndtrPub[lidx], *gndtrCloud[lidx], rclcpp::Clock().now(), "world");
        }

        // Sleep
        rate.sleep();
    }    
}

// Check new KF
bool IsKfCandidate(CloudPosePtr &kfPose, PointPose &kfPoseCand)
{
    // tt_margcloud.Tic();

    static double last_kf_time = kfPose->points.front().t;

    double kfPoseCandTime = kfPoseCand.t;

    static KdTreeFLANN<PointPose> kdTreeKeyFrames;
    kdTreeKeyFrames.setInputCloud(kfPose);

    int knnKfNbrSize = min(10, (int)kfPose->size());
    vector<int> knnNbrIdx(knnKfNbrSize); vector<float> knnNbrSqDis(knnKfNbrSize);
    kdTreeKeyFrames.nearestKSearch(kfPoseCand, knnKfNbrSize, knnNbrIdx, knnNbrSqDis);
    
    bool far_distance = knnNbrSqDis.front() > kf_min_dis*kf_min_dis;
    bool far_angle = true;
    for(int idx = 0; idx < knnNbrIdx.size(); idx++)
    {
        int kfidx = knnNbrIdx[idx];

        // Collect the angle difference
        Quaternionf Qa(kfPose->points[kfidx].qw,
                       kfPose->points[kfidx].qx,
                       kfPose->points[kfidx].qy,
                       kfPose->points[kfidx].qz);

        Quaternionf Qb(kfPoseCand.qw, kfPoseCand.qx, kfPoseCand.qy, kfPoseCand.qz);

        // If the angle is more than 10 degrees, add this to the key pose
        if (fabs(Util::angleDiff(Qa, Qb)) < kf_min_angle)
        {
            far_angle = false;
            break;
        }
    }
    bool kf_timeout = fabs(kfPoseCandTime - last_kf_time) > 2.0 && (knnNbrSqDis.front() > 0.1*0.1);

    if (far_distance || far_angle || kf_timeout)
        return true;
    else
        return false;
}

int main(int argc, char **argv)
{
    // Initalize ros nodes
    rclcpp::init(argc, argv);
    
    nh_ptr = rclcpp::Node::make_shared("gptrlo");
    
    // Supress the pcl warning
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR); // Suppress warnings by pcl load

    RINFO(KGRN "Multi-Lidar Coupled Motion Estimation Started.\n" RESET);

    rclcpp::Time programstart = rclcpp::Clock().now();
 
    /* #region Read parameters --------------------------------------------------------------------------------------*/

    runkf = Util::GetBoolParam(nh_ptr, "runkf", false);

    // Knot length
    Util::GetParam(nh_ptr, "deltaT", deltaT);
    RINFO("Gaussian process with knot length: %f\n", deltaT);

    Util::GetParam(nh_ptr, "kf_min_dis", kf_min_dis);

    // Get the user define parameters
    Util::GetParam(nh_ptr, "priormap_file", priormap_file);
    Util::GetParam(nh_ptr, "lidar_bag_file", lidar_bag_file);
    
    Util::GetParam(nh_ptr, "MAX_CLOUDS", MAX_CLOUDS);
    Util::GetParam(nh_ptr, "SKIPPED_TIME", SKIPPED_TIME);
    Util::GetBoolParam(nh_ptr, "RECURRENT_SKIP", RECURRENT_SKIP);
    
    Util::GetParam(nh_ptr, "imu_topic", imu_topic);
    Util::GetParam(nh_ptr, "lidar_topic", lidar_topic);
    Util::GetParam(nh_ptr, "lidar_type", lidar_type);
    Util::GetParam(nh_ptr, "stamp_time", stamp_time);
    Util::GetParam(nh_ptr, "resume_from_log", resume_from_log);

    // Determine the number of lidar
    int Nlidar = lidar_topic.size();
    int Nimu = imu_topic.size();

    // Get the leaf size for prior map
    Util::GetParam(nh_ptr, "pmap_leaf_size", pmap_leaf_size);

    // Get the leaf size for lidar pointclouds
    cloud_ds = vector<double>(Nlidar, 0.1);
    Util::GetParam(nh_ptr, "cloud_ds", cloud_ds);

    // Find the settings for cross trajectory optimmization
    VIZ_ONLY = Util::GetBoolParam(nh_ptr, "VIZ_ONLY", false);
    Util::GetParam(nh_ptr, "SW_CLOUDNUM"   , SW_CLOUDNUM   );
    Util::GetParam(nh_ptr, "SW_CLOUDSTEP"  , SW_CLOUDSTEP  );
    Util::GetParam(nh_ptr, "t_shift"       , t_shift       );
    Util::GetParam(nh_ptr, "max_outer_iter", max_outer_iter);
    Util::GetParam(nh_ptr, "max_inner_iter", max_inner_iter);
    Util::GetParam(nh_ptr, "min_inner_iter", min_inner_iter);
    Util::GetParam(nh_ptr, "conv_thres"    , conv_thres    );
    Util::GetParam(nh_ptr, "dJ_conv_thres" , dJ_conv_thres );
    Util::GetParam(nh_ptr, "conv_dX_thres" , conv_dX_thres );
    Util::GetParam(nh_ptr, "change_thres"  , change_thres  );

    // Location to save the logs
    Util::GetParam(nh_ptr, "log_dir", log_dir);
    Util::GetParam(nh_ptr, "log_period", log_period);

    // Some notifications
    RINFO("Get bag at %s and prior map at %s.\n", lidar_bag_file.c_str(), priormap_file.c_str());
    RINFO("Lidar info: \n");
    for(int lidx = 0; lidx < Nlidar; lidx++)
        RINFO("Type: %s.\tDs: %f. Topic %s.\n", lidar_type[lidx].c_str(), cloud_ds[lidx], lidar_topic[lidx].c_str());
    RINFO("Maximum number of clouds: %d\n", MAX_CLOUDS);

    RINFO("IMU info: \n");
    int imuCount = 0;
    for(int iidx = 0; iidx < Nimu; iidx++)
        RINFO("Topic %s.\n", imu_topic[iidx].c_str());

    // Get the initial position of the lidars
    vector<double> xyzypr_W_L0(Nlidar*6, 0.0);
    if( Util::GetParam(nh_ptr, "xyzypr_W_L0", xyzypr_W_L0) )
    {
        if (xyzypr_W_L0.size() < Nlidar*6)
        {
            RINFO(KYEL "T_W_L0 missing values. Setting all to zeros \n" RESET);
            xyzypr_W_L0 = vector<double>(Nlidar*6, 0.0);
        }
        else
        {
            RINFO("T_W_L0 found: \n");
            for(int i = 0; i < Nlidar; i++)
                for(int j = 0; j < 6; j++)
                    RINFO("%f, ", xyzypr_W_L0[i*6 + j]);
                cout << endl;
        }
    }
    else
    {
        RINFO("Failed to get xyzypr_W_L0. Setting all to zeros\n");
        xyzypr_W_L0 = vector<double>(Nlidar*6, 0.0);
    }

    T_B_Li_gndtr.resize(Nlidar);
    vector<double> xtrz_gndtr(Nlidar*6, 0.0);
    if( Util::GetParam(nh_ptr, "xtrz_gndtr", xtrz_gndtr) )
    {
        if (xtrz_gndtr.size() < Nlidar*6)
        {
            RINFO(KYEL "xtrz_gndtr missing values. Setting all to zeros \n" RESET);
            xtrz_gndtr = vector<double>(Nlidar*6, 0.0);
        }
        else
        {
            RINFO("xtrz_gndtr found: \n");
            for(int i = 0; i < Nlidar; i++)
            {
                T_B_Li_gndtr[i] = myTf(Util::YPR2Quat(xtrz_gndtr[i*6 + 3], xtrz_gndtr[i*6 + 4], xtrz_gndtr[i*6 + 5]),
                                             Vector3d(xtrz_gndtr[i*6 + 0], xtrz_gndtr[i*6 + 1], xtrz_gndtr[i*6 + 2]));

                for(int j = 0; j < 6; j++)
                    RINFO("%f, ", xtrz_gndtr[i*6 + j]);
                cout << endl;
            }
        }
    }
    else
    {
        RINFO("Failed to get xyzypr_W_L0. Setting all to zeros\n");
        xyzypr_W_L0 = vector<double>(Nlidar*6, 0.0);
    }

    vector<myTf<double>> T_E_G(Nlidar, myTf());
    vector<double> T_E_G_(Nlidar*6, 0.0);
    if( Util::GetParam(nh_ptr, "T_E_G", T_E_G_) )
    {
        if (T_E_G_.size() < Nlidar*6)
        {
            RINFO(KYEL "T_E_G_ missing values. Setting all to zeros \n" RESET);
            T_E_G_ = vector<double>(Nlidar*6, 0.0);
        }
        else
        {
            RINFO("T_E_G found: \n");
            for(int i = 0; i < Nlidar; i++)
            {
                T_E_G[i] = myTf(Util::YPR2Quat(T_E_G_[i*6 + 3], T_E_G_[i*6 + 4], T_E_G_[i*6 + 5]),
                                Vector3d(T_E_G_[i*6 + 0], T_E_G_[i*6 + 1], T_E_G_[i*6 + 2]));

                for(int j = 0; j < 6; j++)
                    RINFO("%f, ", T_E_G_[i*6 + j]);
                cout << endl;
            }
        }
    }
    else
        RINFO("Failed to get T_E_G. Setting all to identity.\n");

    /* #endregion Read parameters -----------------------------------------------------------------------------------*/
 
    /* #region Load the priormap ------------------------------------------------------------------------------------*/

    ikdTreeMap = ikdtreePtr(new ikdtree(0.5, 0.6, pmap_leaf_size));
    kfPose = CloudPosePtr(new CloudPose());
    if (priormap_file != "none")
    {
        pcl::io::loadPCDFile<PointXYZI>(priormap_file, *priormap);
        priormap = uniformDownsample<PointXYZI>(priormap, pmap_leaf_size);
        // Create the kd tree
        RINFO(KYEL "Building the prior map. Size: %d\n" RESET, priormap->size());
        
        // Insert points to the prior map
        insertCloudToikdTree(ikdTreeMap, *priormap);
        kfPose->push_back(myTf().Pose6D(0));    // Just add one random pose
    }

    /* #endregion Load the priormap ---------------------------------------------------------------------------------*/
 
    /* #region Load the data ----------------------------------------------------------------------------------------*/

    map<string, int> imutopicidx;
    for(int iidx = 0; iidx < Nimu; iidx++)
        imutopicidx[imu_topic[iidx]] = iidx;

    // Converting the topics to index
    map<string, int> pctopicidx;
    for(int lidx = 0; lidx < Nlidar; lidx++)
        pctopicidx[lidar_topic[lidx]] = lidx;

    // Storage of the pointclouds
    vector<vector<RosImuMsgPtr>> imus(Nimu);
    vector<vector<CloudXYZITPtr>> clouds(Nlidar);
    vector<vector<rclcpp::Time>> cloudstamp(Nlidar);
    vector<RosTf2Msg> gndtr;

    vector<string> queried_topics = lidar_topic;
    for(auto &topic : imu_topic)
        queried_topics.push_back(topic);
    queried_topics.push_back("/tf");

    // Check the number of point clouds in each topic
    for(int lidx = 0; lidx < Nlidar; lidx++)
    {
        // Specify the directory path
        string topic_dir = lidar_topic[lidx]; boost::replace_all(topic_dir, "/", "__");
        string path = lidar_bag_file + "/clouds/" + topic_dir + "/";

        RINFO("Looking into %s\n", path.c_str());
        
        vector<fs::directory_entry> pcd_files;
        // Iterate through the directory and add .pcd files to the vector
        for (const auto& entry : fs::directory_iterator(path))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".pcd") 
            {
                pcd_files.push_back(entry);
                // RINFO("Found %s", entry.path().string().c_str());
            }
        }

        auto inferStamp = [](const fs::directory_entry &tstr) -> rclcpp::Time
        {
            string tstr_ = tstr.path().filename().string();
            boost::replace_all(tstr_, ".pcd", "");
            vector<string> tstr_parts; boost::split(tstr_parts, tstr_, boost::is_any_of("."));
            return rclcpp::Time(stod(tstr_parts[0]), stoul(tstr_parts[1]));
        };
        // Sort the files alphabetically by their path
        std::sort(pcd_files.begin(), pcd_files.end(), 
                  [&inferStamp](const fs::directory_entry& a, const fs::directory_entry& b)
                  { return inferStamp(a) < inferStamp(b); }
                 );

        // Resize the buffer
        int numClouds = MAX_CLOUDS < 0 ? pcd_files.size() : min(int(pcd_files.size()), MAX_CLOUDS);
        clouds[lidx].resize(numClouds);
        cloudstamp[lidx].resize(numClouds);

        RINFO("Found %d pointclouds for %s topic", numClouds, lidar_topic[lidx].c_str());

        // Generate random number to diversify downsampler
        auto max_ds_it = std::max_element(cloud_ds.begin(), cloud_ds.end());
        double max_ds = *max_ds_it;
        std::random_device rd; 
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> offsetgen(-max_ds, max_ds);

        // Load the pcd files
        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int cidx = 0; cidx < clouds[lidx].size(); cidx++)
        {
            string filename = pcd_files[cidx].path().string();

            // Infer the time stamp
            rclcpp::Time timestamp = inferStamp(pcd_files[cidx]);

            TicToc tt_read;

            // Create cloud and reserve the memory
            CloudOusterPtr cloudRaw(new CloudOuster());

            if (pcl::io::loadPCDFile<PointOuster>(filename, *cloudRaw) == -1)
            {
                PCL_ERROR("Couldn't read file %s", filename.c_str());
                exit(-1);
            }

            std::sort(cloudRaw->points.begin(), cloudRaw->points.end(),
                      [](const PointOuster& pa, const PointOuster& pb)
                      { return pa.t < pb.t; });

            double sweeptime = (cloudRaw->points.back().t - cloudRaw->points.front().t)/1.0e9;
            // if(fabs(sweeptime - 0.1) > 1e-3)
            //     RINFO(KYEL "Irregular sweep time: %f, %f\n" RESET, sweeptime, fabs(sweeptime - 0.1));
            double timebase = stamp_time[lidx] == "start" ? timestamp.seconds() : timestamp.seconds() - sweeptime;

            // Preserve the start point and end point
            PointOuster pb = cloudRaw->points.front();
            PointOuster pf = cloudRaw->points.back();

            // Downsample the pointcloud
            Vector3d offset(offsetgen(gen), offsetgen(gen), offsetgen(gen));
            
            // Transform all the points back to the original, downsample, then transform back
            pcl::transformPointCloud(*cloudRaw, *cloudRaw, myTf<double>(Quaternf(1, 0, 0, 0),  offset).cast<float>().tfMat());
            cloudRaw = uniformDownsample<PointOuster>(cloudRaw, cloud_ds[lidx]);
            pcl::transformPointCloud(*cloudRaw, *cloudRaw, myTf<double>(Quaternf(1, 0, 0, 0), -offset).cast<float>().tfMat());

            // Check if we should reinsert the first and last points
            if ( !(pb.x == cloudRaw->points.front().x
                && pb.y == cloudRaw->points.front().y
                && pb.z == cloudRaw->points.front().z
                && pb.t == cloudRaw->points.front().t))
            {
                cloudRaw->points.insert(cloudRaw->points.begin(), pb);
                // RINFO("Reinserting the front point.\n");
            }

            if ( !(pf.x == cloudRaw->points.back().x
                && pf.y == cloudRaw->points.back().y
                && pf.z == cloudRaw->points.back().z
                && pf.t == cloudRaw->points.back().t))
            {
                cloudRaw->push_back(pf);
                // RINFO("Reinserting the final point.\n");
            }

            // Save the cloud time stamp
            cloudstamp[lidx][cidx] = timestamp;

            // Copy and restamp the points
            clouds[lidx][cidx] = CloudXYZITPtr(new CloudXYZIT());
            clouds[lidx][cidx]->resize(cloudRaw->size());
            #pragma omp parallel for num_threads(MAX_THREADS)
            for(int pidx = 0; pidx < cloudRaw->size(); pidx++)
            {
                auto &po = clouds[lidx][cidx]->points[pidx];
                auto &pi = cloudRaw->points[pidx];
                po.x = pi.x;
                po.y = pi.y;
                po.z = pi.z;
                po.t = pi.t/1.0e9 + timebase;
                po.intensity = pi.intensity;

                // RINFO("t: %f, %f, %f, %f", po.t, po.x, po.y, po.z);
            }

            if (cidx % 100 == 0)
                RINFO("Loading file %s at time %f. CIDX: %05d. Read Time: %f. Time: %f -> %f.",
                      filename.c_str(), timestamp.seconds(), cidx, tt_read.Toc(), 
                      clouds[lidx][cidx]->points.front().t, clouds[lidx][cidx]->points.back().t);
        }
    }

    /* #endregion Load the data -------------------------------------------------------------------------------------*/

    /* #region Initialize the map if no prior specified -------------------------------------------------------------*/

    if (priormap_file == "none")
    {
        printf("Initializing map with scan of %d points\n", clouds.front().front()->size());

        kfPose->push_back(myTf().Pose6D(clouds.front().front()->points.front().t));
        insertCloudToikdTree(ikdTreeMap, *clouds.front().front());

        // Copy to the prior map for initialization
        pcl::copyPointCloud(*clouds.front().front(), *priormap);
    }

    /* #endregion Initialize the map if no prior specified ----------------------------------------------------------*/
 
    /* #region Extract the ground truth and publish -----------------------------------------------------------------*/

    vector<vector<double>> gndtr_ts(Nlidar);
    vector<CloudPosePtr> gndtrCloud(Nlidar);
    vector<fs::directory_entry> gtr_files;
    if(fs::exists(lidar_bag_file + "/gtr") && fs::is_directory(lidar_bag_file + "/gtr"))
    {
        // Iterate through the directory and add .pcd files to the vector
        for (const auto& entry : fs::directory_iterator(lidar_bag_file + "/gtr"))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".pcd") 
                gtr_files.push_back(entry);
        }
    }

    for(int lidx = 0; lidx < Nlidar; lidx++)
    {
        gndtrCloud[lidx] = CloudPosePtr(new CloudPose());
        if(lidx < gtr_files.size())
        {
            CloudPosePtr temp(new CloudPose());
            pcl::io::loadPCDFile<PointPose>(gtr_files[lidx].path().string(), *temp);

            for(auto pose : temp->points)
            {
                if (pose.t < cloudstamp.front().back().seconds())
                {
                    Vector3d p(pose.x, pose.y, pose.z);
                    Quaternd q(pose.qw, pose.qx, pose.qy, pose.qz);

                    p = T_E_G[lidx].rot*p + T_E_G[lidx].pos;
                    q = T_E_G[lidx].rot*q;

                    pose.x = p.x();
                    pose.y = p.y();
                    pose.z = p.z();

                    pose.qx = q.x();
                    pose.qy = q.y();
                    pose.qz = q.z();
                    pose.qw = q.w();

                    gndtrCloud[lidx]->push_back(pose);
                }
            }

            RINFO("GNDTR cloud size: %d point(s)\n");
        }
    }

    // Create thread for visualizing groundtruth
    thread vizGtr = thread(VisualizeGndtr, std::ref(gndtrCloud));


    /* #endregion Extract the ground truth and publish --------------------------------------------------------------*/
 
    /* #region Initialize the pose of each lidar --------------------------------------------------------------------*/
    
    // Initial coordinates of the lidar
    vector<myTf<double>> tf_W_Li0(Nlidar, myTf());
    vector<myTf<double>> tf_W_Li0_refined(Nlidar, myTf());
    vector<CloudXYZIPtr> pc0(Nlidar);

    if(priormap_file != "none")
    {
        vector<double> timestart(Nlidar);
        vector<thread> poseInitThread(Nlidar);
        for(int lidx = 0; lidx < Nlidar; lidx++)
            poseInitThread[lidx] = thread(getInitPose, lidx, std::ref(clouds), std::ref(cloudstamp), std::ref(priormap),
                                        std::ref(timestart), std::ref(xyzypr_W_L0), std::ref(pc0), std::ref(tf_W_Li0), std::ref(tf_W_Li0_refined));

        for(int lidx = 0; lidx < Nlidar; lidx++)
            poseInitThread[lidx].join();
    }

    /* #endregion Initialize the pose of each lidar -----------------------------------------------------------------*/
 
    /* #region Split the pointcloud by time -------------------------------------------------------------------------*/
    
    // Split the secondary point clouds by the primary pointcloud
    double TSTART = clouds.front().front()->points.front().t;
    double TFINAL = clouds.front().back()->points.back().t;
    RINFO("TSTART, TFINAL: %f, %f", TSTART, TFINAL);

    auto tcloudStart = [&TSTART, &deltaT](int cidx) -> double
    {
        return TSTART + cidx*deltaT;
    };

    auto tcloudFinal = [&TSTART, &deltaT](int cidx) -> double
    {
        return TSTART + cidx*deltaT + deltaT;
    };

    auto splitCloud = [&tcloudStart, &tcloudFinal](double tstart, double tfinal, double dt, vector<CloudXYZITPtr> &cloudsIn, vector<CloudXYZITPtr> &cloudsOut) -> void
    {
        // Create clouds in cloudsOut
        cloudsOut.clear();
        while(true)
        {
            cloudsOut.push_back(CloudXYZITPtr(new CloudXYZIT()));
            if (tcloudFinal(cloudsOut.size()-1) >= tfinal)
                break;
        }

        // Extract points from cloudsIn to cloudsOut
        for(auto &cloud : cloudsIn)
        {
            for(auto &point : cloud->points)
            {
                if (point.t < tstart || point.t > tfinal)
                    continue;

                // if (point.t < (tstart + 1e-3))
                //     point.t = tstart;

                // if (point.t > (tfinal - 1e-3))
                //     point.t = tfinal;

                // Find the cloudOut index
                int cidx = int(std::floor((point.t - tstart)/dt));
                // ROS_ASSERT_MSG(tcloudStart(cidx) <= point.t && point.t <= tcloudFinal(cidx),
                //                "point.t: %f. cidx: %d. dt: %f. tcoutstart: %f. tcoutfinal: %f",
                //                 point.t, cidx, dt, tcloudStart(cidx), tcloudFinal(cidx));
                assert(cidx < cloudsOut.size() && myprintf( "%d, %d, %f, %f, %f", cidx, cloudsOut.size(), point.t, tstart, tfinal).c_str());
                // ROS_ASSERT_MSG(coutidx < cloudsOut.size(), "%d %d. %f, %f, %f", coutidx, cloudsOut.size(), point.t, tfinal, tstart + cloudsOut.size()*dt);
                cloudsOut[cidx]->push_back(point);
            }
        }
    };

    vector<vector<CloudXYZITPtr>> cloudsx(Nlidar);
    for(int lidx = 0; lidx < Nlidar; lidx++)
    {
        splitCloud(TSTART, TFINAL, deltaT, clouds[lidx], cloudsx[lidx]);
        RINFO("Split cloud: %d -> %d", clouds[lidx].size(), cloudsx[lidx].size());
    }

    // Check for empty cloud and fill in place holders
    for(int lidx = 0; lidx < Nlidar; lidx++)
    {
        for(int cidx = 0; cidx < cloudsx[lidx].size(); cidx++)
        {
            // ROS_ASSERT_MSG(cloudsx[lidx][cidx]->size() != 0, "%d", cloudsx[lidx][cidx]->size());
            if(cloudsx[lidx][cidx]->size() == 0)
            {
                // RINFO(KYEL "cloud[%2d][%6d] is empty with %d points.\n" RESET, cidx, lidx, cloudsx[lidx][cidx]->size());

                PointXYZIT p;
                p.x = 0; p.y = 0; p.z = 0; p.intensity = 0;
                cloudsx[lidx][cidx]->push_back(p);
                cloudsx[lidx][cidx]->push_back(p);
                cloudsx[lidx][cidx]->points.front().t = tcloudStart(cidx);
                cloudsx[lidx][cidx]->points.back().t = tcloudFinal(cidx);
            }
        }
    }

    /* #endregion Split the pointcloud by time ----------------------------------------------------------------------*/

    /* #region Split the imu by cloud time --------------------------------------------------------------------------*/

    vector<vector<ImuSequence>> imusx(Nimu);
    for(int iidx = 0; iidx < Nimu; iidx++)
    {
        RINFO("Split imu %d", iidx);
        imusx[iidx].resize(cloudsx[0].size());
        
        // #pragma omp parallel num_threads(MAX_THREADS)
        for(int isidx = 0; isidx < imus[iidx].size(); isidx++)
        {
            ImuSample imu(imus[iidx][isidx]);
            for(int cidx = 0; cidx < cloudsx[0].size(); cidx++)
            {
                if(imu.t > tcloudFinal(cidx))
                    continue;
                else if(tcloudStart(cidx) <= imu.t && imu.t < tcloudFinal(cidx))
                    imusx[iidx][cidx].push_back(imu);
            }
        }
    }

    // // Report the distribution
    // for(int iidx = 0; iidx < Nimu; iidx++)
    // {
    //     for(int cidx = 0; cidx < cloudsx[0].size(); cidx++)
    //     {
    //         RINFO("IMU %2d Sequence %4d, sample %3d. ImuItv: [%.3f %.3f]. CloudItv. [%.3f %.3f].",
    //                 iidx, cidx, imusx[iidx][cidx].size(),
    //                 imusx[iidx][cidx].startTime(), imusx[iidx][cidx].finalTime(),
    //                 cloudsx[0][cidx]->points.front().t, cloudsx[0][cidx]->points.back().t);
    //     }
    // }
    
    /* #endregion Split the imu by cloud time -----------------------------------------------------------------------*/
  
    /* #region Run the KF method ------------------------------------------------------------------------------------*/
    
    if(runkf)
    {
        typedef std::shared_ptr<i2EKFLO> i2EKFLOPtr;
        // Find a preliminary trajectory for each lidar sequence
        vector<i2EKFLOPtr> i2kflo;
        vector<thread> trajEst;
        vector<CloudPosePtr> posePrior(Nlidar);
        double UW_NOISE = 100.0, UV_NOISE = 100.0;
        for(int lidx = 0; lidx < Nlidar; lidx++)
        {
            // Creating the trajectory estimator
            StateWithCov Xhat0(cloudstamp[lidx].front().seconds(), tf_W_Li0[lidx].rot, tf_W_Li0[lidx].pos, Vector3d(0, 0, 0), Vector3d(0, 0, 0), 1.0);

            i2kflo.push_back(i2EKFLOPtr(new i2EKFLO(lidx, Xhat0, UW_NOISE, UV_NOISE, 0.5*0.5, 0.4, nh_ptr, nh_mtx)));

            // Estimate the trajectory
            posePrior[lidx] = CloudPosePtr(new CloudPose());
            trajEst.push_back(thread(std::bind(&i2EKFLO::FindTraj, i2kflo[lidx],
                                                std::ref(ikdTreeMap), std::ref(priormap),
                                                std::ref(clouds[lidx]), std::ref(cloudstamp[lidx]),
                                                std::ref(posePrior[lidx]))));
        }
        // Wait for the trajectory estimate to finish
        for(int lidx = 0; lidx < Nlidar; lidx++)
        {
            trajEst[lidx].join();
            
            // Save the trajectory
            string cloud_dir = log_dir + "/gptr_kf/";
            fs::create_directories(cloud_dir);
            string cloud_name = cloud_dir + "/lidar_" + to_string(lidx) + "_pose.pcd";
            // pcl::io::savePCDFileBinary(cloud_name, *posePrior[lidx]);
            PCDWriter writer;
            writer.writeASCII<PointPose>(cloud_name, *posePrior[lidx], 18);
        }

        trajEst.clear();
        i2kflo.clear();

        // Clear the clouds data to free memory
        for(auto &cs : clouds)
        {
            for(auto &c : cs)
                c->clear();
            cs.clear();
        }
    }

    /* #endregion ---------------------------------------------------------------------------------------------------*/
    
    /* #region Create the LOAM modules ------------------------------------------------------------------------------*/

    gpmaplo = vector<LOAMPtr>(Nlidar);
    for(int lidx = 0; lidx < Nlidar; lidx++)
        // Create the gpmaplo objects
        gpmaplo[lidx] = LOAMPtr(new LOAM(nh_ptr, nh_mtx, tf_W_Li0[lidx].getSE3(), TSTART, lidx));

    // If there is a log, load them up
    for(int lidx = 0; lidx < Nlidar; lidx++)
    {
        string log_file = log_dir + myprintf( "/gptraj_%d.csv", lidx);
        if(resume_from_log[lidx] == 1 && file_exist(log_file))
        {
            RINFO("Loading traj file: %s", log_file.c_str());
            gpmaplo[lidx]->GetTraj()->loadTrajectory(log_file);

            // GaussianProcessPtr &traj = gpmaplo[lidx]->GetTraj();
            // for(int kidx = 0; kidx < traj->getNumKnots(); kidx++)
            // {
            //     GPState<double> x = traj->getKnot(kidx);
            //     Quaternd q = x.R.unit_quaternion();
            //     RINFO("Lidar %d. Knot: %d. XYZ: %9.3f, %9.3f, %9.3f. Q: %9.3f, %9.3f, %9.3f, %9.3f.",
            //             lidx, kidx, x.P.x(), x.P.y(), x.P.z(), q.x(), q.y(), q.z(), q.w());
            // }
        }
    }

    // Create the estimation module
    MLCMEPtr gpmlc(new MLCME(nh_ptr, Nlidar));
    vector<GaussianProcessPtr> trajs;
    for(auto &lo : gpmaplo)
        trajs.push_back(lo->GetTraj());

    vector<deque<RosPoseStampedMsg>> extrinsic_poses(Nlidar);

    /* #endregion Create the LOAM modules ---------------------------------------------------------------------------*/
 
    /* #region Do optimization with inter-trajectory factors --------------------------------------------------------*/

    for(int outer_iter = 0; outer_iter < max_outer_iter; outer_iter++)
    {
        // Last time that was logged
        double last_logged_time = -1;

        // Check if loam has diverged
        bool loam_diverges = false;

        // Parameters controlling the slide
        int cidx = outer_iter;
        int SW_CLOUDSTEP_NOW = SW_CLOUDSTEP;
        int SW_CLOUDSTEP_NXT = SW_CLOUDSTEP;
        while(cidx < cloudsx[0].size() - SW_CLOUDSTEP)
        {
            // Check the skipping condition
            {
                double tcloudNow = cloudsx.front()[cidx]->points.front().t;
                double tcloudStart = cloudsx.front().front()->points.front().t;
                double tcloudSinceStart = tcloudNow - tcloudStart;

                if (RECURRENT_SKIP || outer_iter == 0)
                {
                    if (tcloudSinceStart < SKIPPED_TIME)
                    {
                        // RINFO("tcloudSinceStart %f. SKIPPED_TIME: %f. SKIPPING.", tcloudSinceStart, SKIPPED_TIME);
                        
                        SW_CLOUDSTEP_NXT = SW_CLOUDSTEP;
                        cidx += SW_CLOUDSTEP_NOW;
                        SW_CLOUDSTEP_NOW = SW_CLOUDSTEP_NXT;
                        
                        continue;
                    }
                }
            }

            int SW_BEG = cidx;
            int SW_END = min(cidx + SW_CLOUDNUM, int(cloudsx[0].size())-1);
            int SW_MID = min(cidx + SW_CLOUDSTEP_NOW, int(cloudsx[0].size())-1);
            static int SW_END_PREV = SW_END;

            // The effective length of the sliding window by the number of point clouds
            int SW_CLOUDNUM_EFF = SW_END - SW_BEG;

            double tmin = tcloudStart(SW_BEG);     // Start time of the sliding window
            double tmax = tcloudFinal(SW_END);     // End time of the sliding window
            double tmid = tcloudStart(SW_MID);     // Next start time of the sliding window, also determines the marginalization time limit

            vector<int> traj_curr_knots(Nlidar);
            // Extend the trajectories
            for(int lidx = 0; lidx < Nlidar; lidx++)
            {
                traj_curr_knots[lidx] = trajs[lidx]->getNumKnots();
                while(trajs[lidx]->getMaxTime() < tmax)
                    // trajs[lidx]->extendOneKnot();
                    trajs[lidx]->extendOneKnot(trajs[lidx]->getKnot(trajs[lidx]->getNumKnots()-1));
            }

            // Estimation change
            vector<Matrix<double, STATE_DIM, 1>> dX(Nlidar, Matrix<double, STATE_DIM, 1>::Zero());
            int convergence_count = 0;
            bool converged = false;
            bool marginalization_done = false;

            // Create buffers for lidar coefficients
            vector<vector<CloudXYZITPtr>> swCloud(Nlidar, vector<CloudXYZITPtr>(SW_CLOUDNUM_EFF));
            vector<vector<CloudXYZIPtr >> swCloudUndi(Nlidar, vector<CloudXYZIPtr>(SW_CLOUDNUM_EFF));
            vector<vector<CloudXYZIPtr >> swCloudUndiInW(Nlidar, vector<CloudXYZIPtr>(SW_CLOUDNUM_EFF));
            vector<vector<vector<LidarCoef>>> swCloudCoef(Nlidar, vector<vector<LidarCoef>>(SW_CLOUDNUM_EFF));

            // Deskew, Associate, Estimate, repeat max_inner_iter times
            for(int inner_iter = 0; inner_iter < max_inner_iter; inner_iter++)
            {
                TicToc tt_inner_loop;

                // Deskew, Transform and Associate
                auto ProcessCloud = [&ikdTreeMap, &priormap](LOAMPtr &gpmaplo, CloudXYZITPtr &cloudRaw, CloudXYZIPtr &cloudUndi,
                                                             CloudXYZIPtr &cloudUndiInW, vector<LidarCoef> &cloudCoeff) -> void
                {
                    // Get the trajectory
                    GaussianProcessPtr &traj = gpmaplo->GetTraj();

                    // Deskew
                    cloudUndi = CloudXYZIPtr(new CloudXYZI());
                    gpmaplo->Deskew(traj, cloudRaw, cloudUndi);

                    // Transform
                    cloudUndiInW = CloudXYZIPtr(new CloudXYZI());
                    SE3d pose = traj->pose(cloudRaw->points.back().t);
                    pcl::transformPointCloud(*cloudUndi, *cloudUndiInW, pose.translation(), pose.so3().unit_quaternion());

                    // Associate
                    gpmaplo->Associate(traj, ikdTreeMap, priormap, cloudRaw, cloudUndi, cloudUndiInW, cloudCoeff);
                };
                for(int lidx = 0; lidx < Nlidar; lidx++)
                {
                    for(int idx = SW_BEG; idx < SW_END; idx++)
                    {
                        int swIdx = idx - SW_BEG;
                        swCloud[lidx][swIdx] = uniformDownsample<PointXYZIT>(cloudsx[lidx][idx], cloud_ds[lidx]);
                        ProcessCloud(gpmaplo[lidx], swCloud[lidx][swIdx], swCloudUndi[lidx][swIdx], swCloudUndiInW[lidx][swIdx], swCloudCoef[lidx][swIdx]);

                        // RINFO("%f, %d, %d, %d, %d.", lidarWeightUpScale, idx, SW_END, SW_CLOUDSTEP, SW_CLOUDNUM);
                        // for(auto &coef : swCloudCoef[lidx][swIdx])
                        //     coef.plnrty *= (swIdx + 1);
                    }
                }

                // Prepare a report
                OptReport report;

                // Count the number of extracted factors
                TicToc tt_selectfeature;
                vector<vector<lidarFeaIdx>> featuresSelected;
                gpmlc->SelectFeature(trajs, tmin, tmax, swCloudCoef, featuresSelected);
                report.tictocs["t_select_feature"] = tt_selectfeature.Toc();

                // Optimize
                gpmlc->Evaluate(inner_iter, outer_iter, trajs, tmin, tmax, tmid, swCloudCoef,
                                featuresSelected, inner_iter >= max_inner_iter - 1 || converged,
                                report);


                // Exit if divergent
                if (report.factors["LIDAR"] == 0)
                {
                    loam_diverges = true;
                    RINFO(KRED"LOAM DIVERGES!" RESET);
                }


                // Buffer the extrinsic estimate
                for(int lidx = 0; lidx < Nlidar; lidx++)
                {
                    SE3d se3 = gpmlc->GetExtrinsics(lidx);
                    RosPoseStampedMsg pose;
                    pose.header.stamp = rclcpp::Time(tmax);
                    pose.header.frame_id = "lidar_0";
                    pose.pose.position.x = se3.translation().x();
                    pose.pose.position.y = se3.translation().y();
                    pose.pose.position.z = se3.translation().z();
                    pose.pose.orientation.x = se3.so3().unit_quaternion().x();
                    pose.pose.orientation.y = se3.so3().unit_quaternion().y();
                    pose.pose.orientation.z = se3.so3().unit_quaternion().z();
                    pose.pose.orientation.w = se3.so3().unit_quaternion().w();

                    if(extrinsic_poses[lidx].size() == 0)
                        extrinsic_poses[lidx].push_back(pose);
                    else if(rclcpp::Time(extrinsic_poses[lidx].back().header.stamp).seconds() != tmax)
                        extrinsic_poses[lidx].push_back(pose);
                }


                // Visualize the result on each trajectory
                {
                    static vector<rclcpp::Publisher<RosPc2Msg>::SharedPtr> gdntrPub(Nlidar, nullptr);
                    static vector<rclcpp::Publisher<RosOdomMsg>::SharedPtr> odomPub(Nlidar, nullptr);
                    static vector<rclcpp::Publisher<RosMarkerMsg>::SharedPtr> marker_pub(Nlidar, nullptr);
                    static vector<RosOdomMsg> odomMsg(Nlidar, RosOdomMsg());

                    for(int lidx = 0; lidx < Nlidar; lidx++)
                    {
                        gpmaplo[lidx]->Visualize(tmin, tmax, swCloudCoef[lidx], swCloudUndiInW[lidx].back(), true);

                        // Publish an odom topic for each lidar
                        if (gdntrPub[lidx] == nullptr)
                            gdntrPub[lidx] = nh_ptr->create_publisher<RosPc2Msg>(myprintf( "/lidar_%d/gtr_sw", lidx), 1);

                        CloudPose gtrPose;
                        for(auto &pose : gndtrCloud[lidx]->points)
                            if(tmin < pose.t && pose.t < tmax)
                                gtrPose.push_back(pose);

                        if (gtrPose.size() > 0)
                            Util::publishCloud(gdntrPub[lidx], gtrPose, rclcpp::Time(gtrPose.points.back().t), "world");

                        // Publish an odom topic for each lidar
                        if (odomPub[lidx] == nullptr)
                            odomPub[lidx] = nh_ptr->create_publisher<RosOdomMsg>(myprintf( "/lidar_%d/odom", lidx), 1);

                        double ts = tmax - trajs[lidx]->getDt()/2;
                        SE3d pose = trajs[lidx]->pose(tmax);    
                        odomMsg[lidx].header.stamp = rclcpp::Time(tmax);
                        odomMsg[lidx].header.frame_id = "world";
                        odomMsg[lidx].child_frame_id = myprintf( "lidar_%d_body", lidx);
                        odomMsg[lidx].pose.pose.position.x = pose.translation().x();
                        odomMsg[lidx].pose.pose.position.y = pose.translation().y();
                        odomMsg[lidx].pose.pose.position.z = pose.translation().z();
                        odomMsg[lidx].pose.pose.orientation.x = pose.unit_quaternion().x();
                        odomMsg[lidx].pose.pose.orientation.y = pose.unit_quaternion().y();
                        odomMsg[lidx].pose.pose.orientation.z = pose.unit_quaternion().z();
                        odomMsg[lidx].pose.pose.orientation.w = pose.unit_quaternion().w();
                        odomPub[lidx]->publish(odomMsg[lidx]);

                        // Publish a tf at the last inner iterations
                        // static tf2_ros::TransformBroadcaster br;   // Define a TransformStamped message
                        // geometry_msgs::TransformStamped transformStamped;
                        
                        // // Populate the transform message
                        // static double last_tmax = -1;
                        // transformStamped.header.stamp    = rclcpp::Time(tmax);
                        // transformStamped.header.frame_id = "world";
                        // transformStamped.child_frame_id  = myprintf( "lidar_%d", lidx);

                        // // Set the translation values
                        // transformStamped.transform.translation.x = odomMsg[lidx].pose.pose.position.x;
                        // transformStamped.transform.translation.y = odomMsg[lidx].pose.pose.position.y;
                        // transformStamped.transform.translation.z = odomMsg[lidx].pose.pose.position.z;

                        // // Set the rotation as a quaternion (identity quaternion in this example)
                        // transformStamped.transform.rotation.x = odomMsg[lidx].pose.pose.orientation.x;
                        // transformStamped.transform.rotation.y = odomMsg[lidx].pose.pose.orientation.y;
                        // transformStamped.transform.rotation.z = odomMsg[lidx].pose.pose.orientation.z;
                        // transformStamped.transform.rotation.w = odomMsg[lidx].pose.pose.orientation.w;

                        // // Broadcast the transform
                        // if (tmax != last_tmax)
                        // {
                        //     br.sendTransform(transformStamped);
                        //     last_tmax = tmax;
                        // }

                        if (lidx == 0)
                            continue;

                        // RINFO("Lidar %d. Pos: %f, %f, %f", lidx, pose.translation().x(), pose.translation().y(), pose.translation().z());
                        if(marker_pub[lidx] == nullptr)
                            marker_pub[lidx] = nh_ptr->create_publisher<RosMarkerMsg>(myprintf( "/lidar_%d/extr_marker", lidx), 1);

                        // Publish a line between the lidars
                        RosMarkerMsg line_strip;
                        line_strip.header.frame_id = "world";
                        line_strip.header.stamp = rclcpp::Clock().now();
                        line_strip.ns = "lines";
                        line_strip.action = RosMarkerMsg::ADD;
                        line_strip.pose.orientation.w = 1.0;
                        line_strip.id = 0;
                        line_strip.type = RosMarkerMsg::LINE_STRIP;
                        line_strip.scale.x = 0.05;
                        // Line strip is red
                        line_strip.color.r = 0.0;
                        line_strip.color.g = 0.0;
                        line_strip.color.b = 1.0;
                        line_strip.color.a = 1.0;
                        // Create the vertices for the points and lines
                        geometry_msgs::msg::Point p;
                        p.x = odomMsg[0].pose.pose.position.x;
                        p.y = odomMsg[0].pose.pose.position.y;
                        p.z = odomMsg[0].pose.pose.position.z;
                        line_strip.points.push_back(p);
                        p.x = odomMsg[lidx].pose.pose.position.x;;
                        p.y = odomMsg[lidx].pose.pose.position.y;;
                        p.z = odomMsg[lidx].pose.pose.position.z;;
                        line_strip.points.push_back(p);
                        marker_pub[lidx]->publish(line_strip);
                    }
                }
                

                // Check if the motion is fast to slow down the window sliding
                bool fastMotion = false;
                bool fastR = false; bool fastP = false;
                bool fastO = false; bool fastS = false; bool fastV = false; bool fastA = false;
                // Check if we should reduce or increase the slide
                for(int lidx = 0; lidx < Nlidar; lidx++)
                {
                    GPState<double> Xc = trajs[lidx]->getStateAt(tmax);
                    GPState<double> Xp = trajs[lidx]->predictState(SW_CLOUDSTEP);
                    // double DT = deltaT*SW_CLOUDSTEP;
                    // double dRpred = (X.O.norm()*DT + X.S.norm()*DT*DT/2)/M_PI*180.0;
                    // double dPpred = (X.V.norm()*DT + X.A.norm()*DT*DT/2);
                    double dRpred = (Xc.R.inverse()*Xp.R).log().norm()/M_PI*180.0;
                    double dPpred = (Xp.P - Xc.P).norm();
                    fastR = fastR || (change_thres[0] < 0 ? false : (dRpred      > change_thres[0]));
                    fastO = fastO || (change_thres[1] < 0 ? false : (Xc.O.norm() > change_thres[1]));
                    fastS = fastS || (change_thres[2] < 0 ? false : (Xc.S.norm() > change_thres[2]));
                    fastP = fastP || (change_thres[3] < 0 ? false : (dPpred      > change_thres[3]));
                    fastV = fastV || (change_thres[4] < 0 ? false : (Xc.V.norm() > change_thres[4]));
                    fastA = fastA || (change_thres[5] < 0 ? false : (Xc.A.norm() > change_thres[5]));
                    // RINFO("Predicted Change: %.3f, %.3f,\n", dRpred, dPpred);
                }
                fastMotion = fastR || fastO || fastS || fastP || fastV || fastA;
                if(fastMotion)
                    SW_CLOUDSTEP_NXT = 1;
                else
                    SW_CLOUDSTEP_NXT = SW_CLOUDSTEP;

                // if(fastR)
                // {
                //     // Constrain the pitch and roll to avoid losing track
                //     for(int lidx = 0; lidx < Nlidar; lidx++)
                //         for(int kidx = traj_curr_knots[lidx]; kidx < trajs[lidx]->getNumKnots(); kidx++)
                //         {
                //             Vec3 ypr = Util::Quat2YPR(trajs[lidx]->getKnotSO3(kidx).unit_quaternion());

                //             // If new pitch and roll angles change suddenly, reset them
                //             if(pr_range[lidx][1].size() != 0)
                //             {
                //                 if (ypr(1) < pr_range[lidx][1][0] || ypr(1) > pr_range[lidx][1][1])
                //                 {
                //                     RINFO("Resetting pitch of %d traj from %f to ", lidx, ypr(1));
                //                     ypr(1) = 0;//ypr(1)/fabs(ypr(1))*10;
                //                     RINFO("%f", ypr(1));
                //                     trajs[lidx]->getKnotSO3(kidx) = SO3d(Util::YPR2Quat(ypr));
                //                     trajs[lidx]->getKnotOmg(kidx) *= 0;
                //                     trajs[lidx]->getKnotAlp(kidx) *= 0;
                //                 }
                //             }

                //             if(pr_range[lidx][2].size() != 0)
                //             {
                //                 if (ypr(2) < pr_range[lidx][2][0] || ypr(2) > pr_range[lidx][2][1])
                //                 {
                //                     RINFO("Resetting roll %d traj from %f to ", lidx, ypr(2));
                //                     ypr(2) = 0;//ypr(2)/fabs(ypr(2))*10;
                //                     RINFO("%f", ypr(2));
                //                     trajs[lidx]->getKnotSO3(kidx) = SO3d(Util::YPR2Quat(ypr));
                //                     trajs[lidx]->getKnotOmg(kidx) *= 0;
                //                     trajs[lidx]->getKnotAlp(kidx) *= 0;
                //                 }
                //             }
                //         }
                // }

                
                // Check the convergence criteria
                bool dRconv = true, dOconv = true, dSconv = true, dPconv = true, dVconv = true, dAconv = true;
                {
                    // Extract the change in state estimate
                    for(int lidx = 0; lidx < Nlidar; lidx++)
                        dX[lidx] = report.Xt[lidx].boxminus(report.X0[lidx]);

                    // Calculate the percentage change in lidar cost
                    // double dJLidarPerc = fabs(report.costs["LIDAR0"] - report.costs["LIDARK"])/report.costs["LIDAR0"]*100;
                    double dJPerc = fabs(report.costs["J0"] - report.costs["JK"])/report.costs["J0"]*100;

                    // Check the dR, dP and other derivatives
                    for(int lidx = 0; lidx < Nlidar; lidx++)
                    {
                        // Checking the convergence
                        dRconv = dRconv && (conv_dX_thres[0] <= 0 ? true : dX[lidx].block<3, 1>( 0, 0).norm() < conv_dX_thres[0]);
                        dOconv = dOconv && (conv_dX_thres[1] <= 0 ? true : dX[lidx].block<3, 1>( 3, 0).norm() < conv_dX_thres[1]);
                        dSconv = dSconv && (conv_dX_thres[2] <= 0 ? true : dX[lidx].block<3, 1>( 6, 0).norm() < conv_dX_thres[2]);
                        dPconv = dPconv && (conv_dX_thres[3] <= 0 ? true : dX[lidx].block<3, 1>( 9, 0).norm() < conv_dX_thres[3]);
                        dVconv = dVconv && (conv_dX_thres[4] <= 0 ? true : dX[lidx].block<3, 1>(12, 0).norm() < conv_dX_thres[4]);
                        dAconv = dAconv && (conv_dX_thres[5] <= 0 ? true : dX[lidx].block<3, 1>(12, 0).norm() < conv_dX_thres[5]);
                    }
                    bool dXconv = dRconv && dOconv && dSconv && dPconv && dVconv && dAconv;

                    if (dXconv && dJPerc < dJ_conv_thres)  // Increment the counter if all criterias are met
                        convergence_count += 1;
                    else
                        convergence_count = 0;

                    // Set the flag when convergence has been acheived, one more iteration will be run with marginalization
                    if(convergence_count >= conv_thres && (inner_iter >= min_inner_iter))
                    {
                        // RINFO("Convergent. Slide window.\n");
                        converged = true;
                    }
                    // RINFO("CC: %d. Norm: %f", convergence_count, dX.block<3, 1>(9, 0).norm());
                }
    
                // Make the report
                if(report.ceres_iterations != -1)
                {
                    bool do_marginalization = inner_iter >= max_inner_iter - 1 || converged;
                    static int optnum = -1;
                    optnum++;
                    
                    string report_opt =
                        myprintf( "%s"
                                 "GPXOpt# %4d.%2d.%2d: CeresIter: %d. Tfs: %3.0f. Tbd: %3.0f. Tslv: %.0f. Tinner: %.3f. Conv: %d, %d, %d, %d, %d, %d. Count %d. dJ%: %f,\n"
                                 "TSTART: %.3f. TFIN: + %.3f. Tmin-Tmid-Tmax: +[%.3f, %.3f, %.3f]. Trun: %.3f. FASTCHG: %d, %d, %d, %d, %d, %d. Slide: %d, %d.\n"
                                 "Factor: MP2K: %3d, Cross: %4d. Ldr: %4d. MPri: %2d.\n"
                                 "J0: %12.3f. MP2k: %9.3f. Xtrs: %9.3f. LDR: %9.3f. MPri: %9.3f\n"
                                 "Jk: %12.3f. MP2k: %9.3f. Xtrs: %9.3f. LDR: %9.3f. MPri: %9.3f\n"
                                 RESET,
                                 do_marginalization ? "" : KGRN,
                                 optnum, inner_iter, outer_iter,
                                 report.ceres_iterations, report.tictocs["t_select_feature"], report.tictocs["t_ceres_build"], report.tictocs["t_ceres_solve"], tt_inner_loop.Toc(),
                                 dRconv, dOconv, dSconv, dPconv, dVconv, dAconv, convergence_count, fabs(report.costs["J0"] - report.costs["JK"])/report.costs["J0"]*100,
                                 TSTART, TFINAL - TSTART, tmin - TSTART, tmid - TSTART, tmax - TSTART, (rclcpp::Clock().now() - programstart).seconds(),
                                 fastR, fastO, fastS, fastP, fastV, fastA, SW_CLOUDSTEP_NOW, SW_CLOUDSTEP_NXT,
                                 report.factors["MP2K"], report.factors["GPXTRZ"], report.factors["LIDAR"], report.factors["PRIOR"],
                                 report.costs["J0"], report.costs["MP2K0"], report.costs["GPXTRZ0"], report.costs["LIDAR0"], report.costs["PRIOR0"],
                                 report.costs["JK"], report.costs["MP2KK"], report.costs["GPXTRZK"], report.costs["LIDARK"], report.costs["PRIORK"]);
                    
                    string report_state = "";
                    for(int lidx = 0; lidx < Nlidar; lidx++)
                    {
                        dX[lidx] = report.Xt[lidx].boxminus(report.X0[lidx]);
                        report_state +=
                        myprintf( "%s"
                                 "Traj%2d. YPR: %4.0f, %4.0f, %4.0f. XYZ: %7.3f, %7.3f, %7.3f. |O|: %6.3f, %6.3f. |S|: %6.3f, %6.3f. |V|: %6.3f, %6.3f. |A|: %6.3f, %6.3f.\n"
                                  " AftOp: YPR: %4.0f, %4.0f, %4.0f. XYZ: %7.3f, %7.3f, %7.3f. |O|: %6.3f, %6.3f. |S|: %6.3f, %6.3f. |V|: %6.3f, %6.3f. |A|: %6.3f, %6.3f.\n"
                                  " DX:   |dR|: %5.2f. |dO|: %5.2f, |dS|: %5.2f, |dP| %5.2f. |dV|: %5.2f, |dA|: %5.2f.\n"
                                 RESET,
                                 do_marginalization ? "" : KGRN,
                                 lidx, report.X0[lidx].yaw(), report.X0[lidx].pitch(), report.X0[lidx].roll(),
                                       report.X0[lidx].P.x(), report.X0[lidx].P.y(),   report.X0[lidx].P.z(),
                                       report.X0[lidx].O.norm(), report.X0[lidx].O.cwiseAbs().maxCoeff(),
                                       report.X0[lidx].S.norm(), report.X0[lidx].S.cwiseAbs().maxCoeff(),
                                       report.X0[lidx].V.norm(), report.X0[lidx].V.cwiseAbs().maxCoeff(),
                                       report.X0[lidx].A.norm(), report.X0[lidx].A.cwiseAbs().maxCoeff(),
                                       report.Xt[lidx].yaw(), report.Xt[lidx].pitch(), report.Xt[lidx].roll(),
                                       report.Xt[lidx].P.x(), report.Xt[lidx].P.y(),   report.Xt[lidx].P.z(),
                                       report.Xt[lidx].O.norm(), report.Xt[lidx].O.cwiseAbs().maxCoeff(),
                                       report.Xt[lidx].S.norm(), report.Xt[lidx].S.cwiseAbs().maxCoeff(),
                                       report.Xt[lidx].V.norm(), report.Xt[lidx].V.cwiseAbs().maxCoeff(),
                                       report.Xt[lidx].A.norm(), report.Xt[lidx].A.cwiseAbs().maxCoeff(),
                                       dX[lidx].block<3, 1>(0, 0).norm(), dX[lidx].block<3, 1>(03, 0).norm(), dX[lidx].block<3, 1>(06, 0).norm(), 
                                       dX[lidx].block<3, 1>(9, 0).norm(), dX[lidx].block<3, 1>(12, 0).norm(), dX[lidx].block<3, 1>(15, 0).norm());
                    }
                    
                    string report_xtrs = "";
                    for(int lidx = 0; lidx < Nlidar; lidx++)
                    {
                        SE3d T_L0_Li = gpmlc->GetExtrinsics(lidx);
                        myTf tf_L0_Li(T_L0_Li);
                        double T_err;
                        SE3d T_err_1 = T_B_Li_gndtr[lidx].getSE3().inverse()*T_L0_Li;
                        SE3d T_err_2 = T_L0_Li.inverse()*T_B_Li_gndtr[lidx].getSE3();
                        T_err = sqrt(T_err_1.translation().norm()*T_err_1.translation().norm()
                                     + T_err_2.translation().norm()*T_err_2.translation().norm());
                        report_xtrs += 
                            myprintf( "%s"
                                     "T_L0_L%d. YPR: %4.0f, %4.0f, %4.0f. XYZ: %6.2f, %6.2f, %6.2f. Error: %.3f.\n"
                                     RESET,
                                     do_marginalization ? "" : KGRN,
                                     lidx, tf_L0_Li.yaw(),   tf_L0_Li.pitch(), tf_L0_Li.roll(),
                                           tf_L0_Li.pos.x(), tf_L0_Li.pos.y(), tf_L0_Li.pos.z(), T_err);
                    }
                    RINFO((report_opt + report_state + report_xtrs + "\n").c_str());
                }

                // // Save the pointclouds
                // if (tcloudStart(cidx) - TSTART < 100.0)
                // {
                //     static vector<int> cloud_idx(Nlidar, -1);
                //     for(int lidx = 0; lidx < Nlidar; lidx++)
                //     {
                //         cloud_idx[lidx] += 1;
                //         string cloud_dir = log_dir + "/gptr_deskewed_cloud/";
                //         fs::create_directories(cloud_dir + "/lidar_" + to_string(lidx));
                //         string cloud_name = cloud_dir + "/lidar_" + to_string(lidx) + "/cloud_" + to_string(cloud_idx[lidx]) + ".pcd";
                //         pcl::io::savePCDFileBinary(cloud_name, *swCloudUndiInW[lidx].back());
                //     }
                // }

                // if the system has converged and marginalization done, slide window
                if (converged && report.marginalization_done)
                    break;
            }

            // Log the result every 10 seconds
            if((int(floor(tcloudStart(cidx) - TSTART)) % int(log_period) == 0
                && tcloudStart(cidx) - last_logged_time >= 0.9*log_period)
                || last_logged_time == -1
                || (SW_END >= cloudsx[0].size() - SW_CLOUDSTEP))
            {
                last_logged_time = tcloudStart(cidx);

                // Create directories if they do not exist
                string output_dir = log_dir + myprintf( "/run_%02d/time_%04.0f/", outer_iter, tcloudStart(cidx)-TSTART);
                std::filesystem::create_directories(output_dir);

                // Save the trajectory and estimation result
                for(int lidx = 0; lidx < Nlidar; lidx++)
                {
                    // string log_file = log_dir + myprintf( "/gptraj_%d.csv", lidx);
                    RINFO("Exporting trajectory logs to %s.\n", output_dir.c_str());
                    gpmaplo[lidx]->GetTraj()->saveTrajectory(output_dir, lidx, gndtr_ts[lidx]);
                }

                if (Nlidar > 1)
                {
                    // Log the extrinsics
                    string xts_log = output_dir + "/extrinsics_" + std::to_string(1) + ".csv";
                    std::ofstream xts_logfile;
                    xts_logfile.open(xts_log); // Open the file for writing
                    xts_logfile.precision(std::numeric_limits<double>::digits10 + 1);
                    xts_logfile << "t, x, y, z, qx, qy, qz, qw" << endl;
                    for(auto &pose : extrinsic_poses[1])
                    {
                        xts_logfile << rclcpp::Time(pose.header.stamp).seconds() << ","
                                    << pose.pose.position.x << ","
                                    << pose.pose.position.y << ","
                                    << pose.pose.position.z << ","
                                    << pose.pose.orientation.x << ","
                                    << pose.pose.orientation.y << ","
                                    << pose.pose.orientation.z << ","
                                    << pose.pose.orientation.w << endl;
                    }
                    xts_logfile.close();
                }
            }

            // Exit if loam diverges
            if (loam_diverges)
                break;

            // Set the index for next step
            cidx+= int(SW_CLOUDSTEP_NOW);
            SW_CLOUDSTEP_NOW = SW_CLOUDSTEP_NXT;
            SW_END_PREV = SW_END;

            if (SW_END == int(cloudsx[0].size())-1)
                break;

            // Update map
            if(priormap_file == "none")
            {
                // Check if a new keyframe should be added
                PointPose kfPoseCand = myTf(trajs.front()->pose(tmin)).Pose6D(tmin);
                bool newKf = IsKfCandidate(kfPose, kfPoseCand);

                if (newKf)
                {
                    // Add new keyframe pose
                    kfPose->push_back(kfPoseCand);
                    
                    // Extend the map
                    CloudXYZIPtr kfCloud(new CloudXYZI());
                    for(int lidx = 0; lidx < Nlidar; lidx++)
                    {
                        for(int cidx = 0; cidx < swCloudUndiInW[lidx].size(); cidx++)
                        {
                            if (swCloud[lidx][cidx]->points.back().t - tmin > 0.1) 
                                break;
                            else
                                *kfCloud += *swCloudUndiInW[lidx][cidx];
                        }
                    }
                    insertCloudToikdTree(ikdTreeMap, *kfCloud);

                    // Notice
                    printf(KYEL "Extending map with %d points\n" RESET, kfCloud->size());
                }
            }
        }

        if (loam_diverges)
            break;

        // Reset the marginalization factor
        gpmlc->Reset();
    }

    /* #endregion Do optimization with inter-trajectory factors -----------------------------------------------------*/
 
    /* #region Create some logs for visualization -------------------------------------------------------------------*/

    for(int lidx = 0; lidx < Nlidar; lidx++)
    {
        CloudXYZIPtr cloudMergedUndi(new CloudXYZI());
        CloudXYZITPtr cloudMergedRaw(new CloudXYZIT());
        for(int cidx = 0; cidx < cloudsx[lidx].size(); cidx++)
        {
            if (tcloudStart(cidx) - TSTART < 5.0)
                continue;

            GaussianProcessPtr &traj = gpmaplo[lidx]->GetTraj();
            CloudXYZITPtr &cloudRaw = cloudsx[lidx][cidx];

            CloudXYZIPtr cloudUndi(new CloudXYZI());
            gpmaplo[lidx]->Deskew(traj, cloudsx[lidx][cidx], cloudUndi);

            SE3d pose = traj->pose(cloudRaw->points.back().t);
            pcl::transformPointCloud(*cloudUndi, *cloudUndi, pose.translation(), pose.so3().unit_quaternion());
            pcl::transformPointCloud(*cloudRaw, *cloudRaw, pose.translation(), pose.so3().unit_quaternion());
            
            *cloudMergedUndi += *cloudUndi;
            *cloudMergedRaw += *cloudRaw;

            if (tcloudStart(cidx) - TSTART > 100.0)
                break;
        }

        string cloud_dir = log_dir + "/gptr_deskewed_cloud/";
        fs::create_directories(cloud_dir);

        {
            string cloud_name = cloud_dir + "/lidar_" + to_string(lidx) + ".pcd";
            RINFO("Saving cloud %s", cloud_name.c_str());
            pcl::io::savePCDFileBinary(cloud_name, *cloudMergedUndi);
        }

        {
            string cloud_name = cloud_dir + "/lidar_" + to_string(lidx) + "_raw.pcd";
            RINFO("Saving cloud %s", cloud_name.c_str());
            pcl::io::savePCDFileBinary(cloud_name, *cloudMergedRaw);
        }
    }

    /* #endregion Create some logs for visualization ----------------------------------------------------------------*/

    // A happy exit

    exit(0);
}