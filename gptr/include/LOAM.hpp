#include <Eigen/Dense>
#include <Eigen/Sparse>

// Ros stuff
#include "rclcpp/rclcpp.hpp"

/* All needed for filter of custom point type----------*/
#include <pcl/pcl_base.h>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/filters/filter.h>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/impl/uniform_sampling.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/impl/crop_box.hpp>
/* All needed for filter of custom point type----------*/

// All about gaussian process
#include "GaussianProcess.hpp"
#include "utility.h"

class LOAM
{
private:

    NodeHandlePtr nh_ptr;
    
    // Index for distinguishing between clouds
    int LIDX;

    // Feature to map association parameters
    double min_planarity = 0.2;
    double max_plane_dis = 0.3;

    // Initial pose of the lidars
    SE3d T_W_Li0;

    // Gaussian Process for the trajectory of each lidar
    GaussianProcessPtr traj;

    // Knot length
    double deltaT = 0.1;
    double mpCovROSJerk = 10;
    double mpCovPVAJerk = 10;
    
    // Associate params
    int knnSize = 6;
    double minKnnSqDis = 0.5*0.5;

    // Buffer for the pointcloud segments
    mutex cloud_seg_buf_mtx;
    deque<CloudXYZITPtr> cloud_seg_buf;

    // Publisher
    rclcpp::Publisher<RosPc2Msg>::SharedPtr trajPub;
    rclcpp::Publisher<RosPc2Msg>::SharedPtr swTrajPub;
    rclcpp::Publisher<RosPc2Msg>::SharedPtr assocCloudPub;
    rclcpp::Publisher<RosPc2Msg>::SharedPtr deskewedCloudPub;

public:

    // Destructor
   ~LOAM() {};

    LOAM(NodeHandlePtr &nh_ptr_, mutex & nh_mtx, const SE3d &T_W_Li0_, double t0, int &LIDX_)
        : nh_ptr(nh_ptr_), T_W_Li0(T_W_Li0_), LIDX(LIDX_)
    {
        lock_guard<mutex> lg(nh_mtx);

        // Trajectory estimate
        Util::GetParam(nh_ptr, "deltaT", deltaT);

        // Weight for the motion prior
        Util::GetParam(nh_ptr, "mpCovROSJerk", mpCovROSJerk);
        Util::GetParam(nh_ptr, "mpCovPVAJerk", mpCovPVAJerk);

        // Association params
        Util::GetParam(nh_ptr, "min_planarity", min_planarity);
        Util::GetParam(nh_ptr, "max_plane_dis", max_plane_dis);
        Util::GetParam(nh_ptr, "knnSize", knnSize);

        trajPub = nh_ptr->create_publisher<RosPc2Msg>(myprintf("/lidar_%d/gp_traj", LIDX), 1);
        swTrajPub = nh_ptr->create_publisher<RosPc2Msg>(myprintf("/lidar_%d/sw_opt", LIDX), 1);
        assocCloudPub = nh_ptr->create_publisher<RosPc2Msg>(myprintf("/lidar_%d/assoc_cloud", LIDX), 1);
        deskewedCloudPub = nh_ptr->create_publisher<RosPc2Msg>(myprintf("/lidar_%d/cloud_inW", LIDX), 1);

        Matrix3d CovROSJerk = Vector3d(mpCovROSJerk, mpCovROSJerk, mpCovROSJerk).asDiagonal();
        Matrix3d CovPVAJerk = Vector3d(mpCovPVAJerk, mpCovPVAJerk, mpCovPVAJerk).asDiagonal();

        traj = GaussianProcessPtr(new GaussianProcess(deltaT, CovROSJerk, CovPVAJerk, true));
        traj->setStartTime(t0);
        traj->setKnot(0, GPState(t0, T_W_Li0));
    }

    void Associate(GaussianProcessPtr &traj, const ikdtreePtr &ikdTreeMap, const CloudXYZIPtr &priormap,
                   const CloudXYZITPtr &cloudRaw, const CloudXYZIPtr &cloudInB, const CloudXYZIPtr &cloudInW,
                   vector<LidarCoef> &Coef)
    {
        assert(cloudRaw->size() == cloudInB->size() && myprintf("cloudRaw: %d. cloudInB: %d", cloudRaw->size(), cloudInB->size()).c_str());

        if (priormap->size() > knnSize)
        {
            int pointsCount = cloudInW->points.size();
            vector<LidarCoef> Coef_;
            Coef_.resize(pointsCount);

            #pragma omp parallel for num_threads(MAX_THREADS)
            for (int pidx = 0; pidx < pointsCount; pidx++)
            {
                double tpoint = cloudRaw->points[pidx].t;
                PointXYZIT pointRaw = cloudRaw->points[pidx];
                PointXYZI  pointInB = cloudInB->points[pidx];
                PointXYZI  pointInW = cloudInW->points[pidx];

                Coef_[pidx].n = Vector4d(0, 0, 0, 0);
                Coef_[pidx].t = -1;

                if(!Util::PointIsValid(pointInB))
                {
                    // printf(KRED "Invalid surf point!: %f, %f, %f\n" RESET, pointInB.x, pointInB.y, pointInB.z);
                    pointInB.x = 0; pointInB.y = 0; pointInB.z = 0; pointInB.intensity = 0;
                    continue;
                }

                if(!Util::PointIsValid(pointInW))
                    continue;

                if (!traj->TimeInInterval(tpoint, 1e-6))
                    continue;

                ikdtPointVec nbrPoints; vector<float> knnSqDis(knnSize, 0);
                ikdTreeMap->Nearest_Search(pointInW, knnSize, nbrPoints, knnSqDis);

                if (nbrPoints.size() < knnSize)
                    continue;
                    
                // Fit the plane
                if(Util::fitPlane(nbrPoints, 0.5, 0.2, Coef_[pidx].n, Coef_[pidx].plnrty))
                {
                    // assert(tpoint >= 0);
                    Coef_[pidx].t = tpoint;
                    Coef_[pidx].f = Vector3d(pointRaw.x, pointRaw.y, pointRaw.z);
                    Coef_[pidx].finW = Vector3d(pointInW.x, pointInW.y, pointInW.z);
                    Coef_[pidx].fdsk = Vector3d(pointInB.x, pointInB.y, pointInB.z);
                }
            }

            // Copy the coefficients to the buffer
            Coef.clear();
            int totalFeature = 0;
            for(int pidx = 0; pidx < pointsCount; pidx++)
            {
                LidarCoef &coef = Coef_[pidx];
                if (coef.t >= 0)
                {
                    Coef.push_back(coef);
                    Coef.back().ptIdx = totalFeature;
                    totalFeature++;
                }
            }
        }
    }

    void Deskew(GaussianProcessPtr &traj, CloudXYZITPtr &cloudRaw, CloudXYZIPtr &cloudDeskewedInB)
    {
        int Npoints = cloudRaw->size();

        if (Npoints == 0)
            return;

        cloudDeskewedInB->resize(Npoints);

        SE3d T_Be_W = traj->pose(cloudRaw->points.back().t).inverse();
        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int pidx = 0; pidx < Npoints; pidx++)
        {
            PointXYZIT &pi = cloudRaw->points[pidx];
            PointXYZI  &po = cloudDeskewedInB->points[pidx];

            double ts = pi.t;
            SE3d T_Be_Bs = T_Be_W*traj->pose(ts);

            Vector3d pinBs(pi.x, pi.y, pi.z);
            Vector3d pinBe = T_Be_Bs*pinBs;

            po.x = pinBe.x();
            po.y = pinBe.y();
            po.z = pinBe.z();
            // po.t = pi.t;
            po.intensity = pi.intensity;
        }
    }

    void Visualize(double tmin, double tmax, vector<vector<LidarCoef>> &swCloudCoef, CloudXYZIPtr &cloudUndiInW, bool publish_full_traj=false)
    {
        if (publish_full_traj)
        {
            CloudPosePtr trajCP = CloudPosePtr(new CloudPose());
            for(int kidx = 0; kidx < traj->getNumKnots(); kidx++)
            {
                trajCP->points.push_back(myTf(traj->getKnotPose(kidx)).Pose6D(traj->getKnotTime(kidx)));
                trajCP->points.back().intensity = (tmax - trajCP->points.back().t) < 0.1 ? 1.0 : 0.0;
            }

            // Publish global trajectory
            Util::publishCloud(trajPub, *trajCP, rclcpp::Clock().now(), "world");
        }

        // Sample and publish the slinding window trajectory
        CloudPosePtr poseSampled = CloudPosePtr(new CloudPose());
        for(double ts = tmin; ts < tmax; ts += traj->getDt()/5)
            if(traj->TimeInInterval(ts))
                poseSampled->points.push_back(myTf(traj->pose(ts)).Pose6D(ts));

        // static ros::Publisher swTrajPub = nh_ptr->advertise<RosPc2Msg>(myprintf("/lidar_%d/sw_opt", LIDX), 1);
        Util::publishCloud(swTrajPub, *poseSampled, rclcpp::Clock().now(), "world");

        CloudXYZIPtr assoc_cloud(new CloudXYZI());
        for (int widx = 0; widx < swCloudCoef.size(); widx++)
        {
            for(auto &coef : swCloudCoef[widx])
                {
                    PointXYZI p;
                    p.x = coef.finW.x();
                    p.y = coef.finW.y();
                    p.z = coef.finW.z();
                    p.intensity = widx;
                    assoc_cloud->push_back(p);
                }
        }
        
        // static ros::Publisher assocCloudPub = nh_ptr->advertise<RosPc2Msg>(myprintf("/lidar_%d/assoc_cloud", LIDX), 1);
        if (assoc_cloud->size() != 0)
            Util::publishCloud(assocCloudPub, *assoc_cloud, rclcpp::Clock().now(), "world");

        // Publish the deskewed pointCloud
        Util::publishCloud(deskewedCloudPub, *cloudUndiInW, rclcpp::Clock().now(), "world");
    }

    GaussianProcessPtr &GetTraj()
    {
        return traj;
    }
};
