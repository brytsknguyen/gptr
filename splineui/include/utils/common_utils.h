#pragma once

#include <rclcpp/rclcpp.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <deque>
#include <map>
#include <unordered_map>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#define MAX_THREADS std::thread::hardware_concurrency()/2


namespace Eigen {

template <typename T>
using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

template <typename T>
using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

template <typename K, typename V>
using aligned_map = std::map<K, V, std::less<K>,
                    Eigen::aligned_allocator<std::pair<K const, V>>>;

template <typename K, typename V>
using aligned_unordered_map = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                              Eigen::aligned_allocator<std::pair<K const, V>>>;

}

using namespace std;
using namespace Eigen;

struct PointTQXYZI
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY              // preferred way of adding a XYZ+padding
    double t;
    float  qx;
    float  qy;
    float  qz;
    float  qw;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(PointTQXYZI,
                                 (float,  x, x) (float,  y, y) (float,  z, z)
                                 (float,  intensity, intensity)
                                 (double, t,  t)
                                 (float,  qx, qx)
                                 (float,  qy, qy)
                                 (float,  qz, qz)
                                 (float,  qw, qw))


typedef PointTQXYZI PointPose;
typedef pcl::PointCloud<PointPose> CloudPose;
typedef pcl::PointCloud<PointPose>::Ptr CloudPosePtr;
typedef rclcpp::Node::SharedPtr RosNodeHandlePtr;

typedef Sophus::SO3<double> SO3d;
typedef Sophus::SE3<double> SE3d;
typedef pcl::PointXYZ PointXYZ;
typedef pcl::PointXYZI PointXYZI;
typedef geometry_msgs::msg::PoseStamped RosPoseStampedMsg;
typedef geometry_msgs::msg::PoseStamped::SharedPtr RosPoseStampedMsgPtr;

typedef nav_msgs::msg::Odometry RosOdomMsg;
typedef nav_msgs::msg::Odometry::SharedPtr RosOdomMsgPtr;

typedef Eigen::Quaterniond Quaternd;
typedef Eigen::Quaterniond Quaternf;

struct TDOAData {
  double t;         // Time stamp
  const int idA;          // Index of anchor A
  const int idB;          // Index of anchor B
  const double data;      // TDOA measurement
  TDOAData(double s, int idxA, int idxB, double r) : t(s), idA(idxA), idB(idxB), data(r) {};
};

struct IMUData {
  double t;            // Time stamp
  Eigen::Vector3d acc; // acceleration measurement
  Eigen::Vector3d gyro;// gyroscope measurement
  IMUData(double s, const Eigen::Vector3d& acc_, const Eigen::Vector3d& gyro_) : t(s), acc(acc_), gyro(gyro_) {};
};

class TicToc
{
public:
    TicToc()
    {
        Tic();
    }

    void Tic()
    {
        start_ = std::chrono::system_clock::now();
    }

    double Toc()
    {
        end_ = std::chrono::system_clock::now();
        elapsed_seconds_ = end_ - start_;
        return elapsed_seconds_.count() * 1000;
    }

    double TocTic()
    {
        end_ = std::chrono::system_clock::now();
        elapsed_seconds_ = end_ - start_;
        Tic();
        return elapsed_seconds_.count() * 1000;
    }

    double GetLastStop()
    {
        return elapsed_seconds_.count() * 1000;
    }

#define LASTSTOP(x) printf(#x".LastStop : %f\n", x.GetLastStop());
#define TOCPRINT(x) x.Toc(); printf(#x".Toc : %f\n", x.GetLastStop());

private:
    std::chrono::time_point<std::chrono::system_clock> start_, end_;
    std::chrono::duration<double> elapsed_seconds_;
};


template <typename T=double>
struct myTf
{
    Eigen::Quaternion<T>   rot;
    Eigen::Matrix<T, 3, 1> pos;

    myTf Identity()
    {
        return myTf();
    }

    myTf(const myTf<T> &other)
    {
        rot = other.rot;
        pos = other.pos;
    }

    myTf()
    {
        rot = Quaternd(1, 0, 0, 0);
        pos = Vector3d(0, 0, 0);
    }

    myTf(Eigen::Quaternion<T> rot_in, Eigen::Matrix<T, 3, 1> pos_in)
    {
        this->rot = rot_in;
        this->pos = pos_in;
    }

    myTf(Eigen::Matrix<T, 3, 3> rot_in, Eigen::Matrix<T, 3, 1> pos_in)
    {
        this->rot = Quaternion<T>(rot_in);
        this->pos = pos_in;
    }

    myTf(Eigen::Matrix<T, 3, 1> axisangle_in, Eigen::Matrix<T, 3, 1> pos_in)
    {
        this->rot = Quaternd(Eigen::AngleAxis<T>(axisangle_in.norm(),
                                                 axisangle_in/axisangle_in.norm()));
        this->pos = pos_in;
    }

    template <typename Tin>
    myTf(Eigen::Matrix<Tin, 4, 4> tfMat)
    {
        Eigen::Matrix<T, 3, 3> M = tfMat.block(0, 0, 3, 3).template cast<T>();
        this->rot = Quaternion<T>(M);
        this->pos = tfMat.block(0, 3, 3, 1).template cast<T>();
    }

    myTf(PointPose point)
    {
        this->rot = Quaternion<T>(point.qw, point.qx, point.qy, point.qz);
        this->pos << point.x, point.y, point.z;
    }

    myTf(const nav_msgs::msg::Odometry &odom)
    {
        this->rot = Quaternion<T>(odom.pose.pose.orientation.w,
                                  odom.pose.pose.orientation.x,
                                  odom.pose.pose.orientation.y,
                                  odom.pose.pose.orientation.z);

        this->pos << odom.pose.pose.position.x,
                     odom.pose.pose.position.y,
                     odom.pose.pose.position.z;
    }

    myTf(const geometry_msgs::msg::PoseStamped &pose)
    {
        this->rot = Quaternion<T>(pose.pose.orientation.w,
                                  pose.pose.orientation.x,
                                  pose.pose.orientation.y,
                                  pose.pose.orientation.z);

        this->pos << pose.pose.position.x,
                     pose.pose.position.y,
                     pose.pose.position.z;
    }

    myTf(const Sophus::SE3<T> se3)
    {
        this->rot = se3.unit_quaternion();
        this->pos = se3.translation();
    }

    myTf(Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> transform)
    {
        this->rot = Eigen::Quaternion<T>{transform.linear()}.normalized();
        this->pos = transform.translation();
    }

    Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> transform() const
    {
        Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> transform;
        transform.linear() = rot.normalized().toRotationMatrix();
        transform.translation() = pos;
        return transform;
    }

    Eigen::Matrix<T, 4, 4> tfMat() const
    {
        Eigen::Matrix<T, 4, 4> M = Matrix<T, 4, 4>::Identity();
        M.block(0, 0, 3, 3) = rot.normalized().toRotationMatrix();
        M.block(0, 3, 3, 1) = pos;
        return M;
    }

    PointXYZI Point3D() const
    {
        PointXYZI p;

        p.x = (float)pos.x();
        p.y = (float)pos.y();
        p.z = (float)pos.z();

        p.intensity = -1;

        return p;
    }

    PointPose Pose6D() const
    {
        PointPose p;

        p.t = -1;

        p.x = (float)pos.x();
        p.y = (float)pos.y();
        p.z = (float)pos.z();

        p.qx = (float)rot.x();
        p.qy = (float)rot.y();
        p.qz = (float)rot.z();
        p.qw = (float)rot.w();

        p.intensity = -1;

        return p;
    }

    PointPose Pose6D(double time) const
    {
        PointPose p;

        p.t = time;

        p.x = (float)pos.x();
        p.y = (float)pos.y();
        p.z = (float)pos.z();

        p.qx = (float)rot.x();
        p.qy = (float)rot.y();
        p.qz = (float)rot.z();
        p.qw = (float)rot.w();

        p.intensity = -1;

        return p;
    }

    RosOdomMsg rosOdom()
    {
        RosOdomMsg msg;
        msg.pose.pose.position.x = pos.x();
        msg.pose.pose.position.y = pos.y();
        msg.pose.pose.position.z = pos.z();
        msg.pose.pose.orientation.x = rot.x();
        msg.pose.pose.orientation.y = rot.y();
        msg.pose.pose.orientation.z = rot.z();
        msg.pose.pose.orientation.w = rot.w();
        return msg;
    }

    myTf<T> slerp(double s, myTf<T> tfend)
    {
        Quaternion<T> qs = this->rot.slerp(s, tfend.rot);
        Eigen::Matrix<T, 3, 1> ps = (1-s)*(this->pos) + s*(tfend.pos);
        return myTf<T>(qs, ps);
    }

    template <typename Tout = double>
    Sophus::SE3<Tout> getSE3() const
    {
        return Sophus::SE3<Tout>(this->rot.template cast<Tout>(),
                                 this->pos.template cast<Tout>());
    }

    double roll() const
    {
        return atan2(rot.x()*rot.w() + rot.y()*rot.z(), 0.5 - (rot.x()*rot.x() + rot.y()*rot.y()))/M_PI*180.0;
    }

    double pitch() const
    {
        return asin(-2*(rot.x()*rot.z() - rot.w()*rot.y()))/M_PI*180.0;
    }

    double yaw() const
    {
        return atan2(rot.x()*rot.y() + rot.w()*rot.z(), 0.5 - (rot.y()*rot.y() + rot.z()*rot.z()))/M_PI*180.0;
    }

    Matrix<T, 3, 1> SO3Log()
    {
        Eigen::AngleAxis<T> phi(rot);
        return phi.angle()*phi.axis();
    }

    myTf inverse() const
    {
        Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> transform_inv = this->transform().inverse();
        myTf tf_inv;
        tf_inv.rot = transform_inv.linear();
        tf_inv.pos = transform_inv.translation();
        return tf_inv;
    }

    myTf operator*(const myTf &other) const
    {
        Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> transform_out = this->transform() * other.transform();
        return myTf(transform_out);
    }

    Vector3d operator*(const Vector3d &v) const
    {
        return (rot*v + pos);
    }

    Quaternd operator*(const Quaternd &q) const
    {
        return (rot*q);
    }

    template <typename NewType>
    myTf<NewType> cast() const
    {
        myTf<NewType> tf_new{this->rot.template cast<NewType>(), this->pos.template cast<NewType>()};
        return tf_new;
    }

    friend std::ostream &operator<<(std::ostream &os, const myTf &tf)
    {
        os << tf.pos.x() << " " << tf.pos.y() << " " << tf.pos.z() << " " << tf.rot.w() << " "
           << tf.rot.x() << " " << tf.rot.y() << " " << tf.rot.z();
        return os;
    }
}; // class myTf

typedef myTf<> mytf;

inline std::string myprintf(const std::string& format, ...)
{
    va_list args;
    va_start(args, format);
    size_t len = std::vsnprintf(NULL, 0, format.c_str(), args);
    va_end(args);
    std::vector<char> vec(len + 1);
    va_start(args, format);
    std::vsnprintf(&vec[0], len + 1, format.c_str(), args);
    va_end(args);

    return string(vec.begin(), vec.end() - 1);
}

class Util
{
public:
    template <typename T>
    static inline bool GetParam(const RosNodeHandlePtr &nh, const string &param_name, T &param)
    {
        if(!nh->has_parameter(param_name))
            nh->declare_parameter(param_name, param);
        return nh->get_parameter(param_name, param);
    }

    template <typename T>
    static T readParam(rclcpp::Node::SharedPtr &n, std::string name)
    {
        T ans;
        if (!n->has_parameter(name)) {
            n->declare_parameter<T>(name);
        }
        if (!n->get_parameter(name, ans)) {
            RCLCPP_FATAL_STREAM(n->get_logger(), "Failed to load " << name);
            exit(1);
        }
        return ans;
    }

    template <typename T>
    static T readParam(rclcpp::Node::SharedPtr &n, std::string name, const T& alternative)
    {
        T ans;
        if (!n->has_parameter(name)) {
            n->declare_parameter<T>(name, alternative);
        }
        n->get_parameter_or(name, ans, alternative);
        return ans;
    }
};
