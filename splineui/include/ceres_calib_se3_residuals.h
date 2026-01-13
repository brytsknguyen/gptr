#pragma once


#include "basalt/ceres_spline_helper.h"

namespace basalt {

template <int _N, bool OLD_TIME_DERIV>
struct CalibGyroCostFunctorSE3 {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CalibGyroCostFunctorSE3(const Eigen::Vector3d& measurement, double u,
                          double inv_dt, double inv_std = 1)
      : measurement(measurement), u(u), inv_dt(inv_dt), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Vec6 = Eigen::Matrix<T, 6, 1>;

    Eigen::Map<Vec3> residuals(sResiduals);

    Vec6 rot_vel;

    CeresSplineHelper<_N>::template evaluate_lie<T, Sophus::SE3>(
        sKnots, u, inv_dt, nullptr, &rot_vel, nullptr);

    Eigen::Map<Vec3 const> const bias(sKnots[_N]);

    residuals =
        inv_std * (rot_vel.template tail<3>() - measurement.cast<T>() + bias);

    return true;
  }

  Eigen::Vector3d measurement;
  double u, inv_dt, inv_std;
};

template <int _N, bool OLD_TIME_DERIV>
struct CalibAccelerationCostFunctorSE3 {
  static constexpr int N = _N;  // Order of the spline.

  using VecN = Eigen::Matrix<double, _N, 1>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CalibAccelerationCostFunctorSE3(const Eigen::Vector3d& measurement, double u,
                                  double inv_dt, double inv_std)
      : measurement(measurement), u(u), inv_dt(inv_dt), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vector3 = Eigen::Matrix<T, 3, 1>;
    using Vector6 = Eigen::Matrix<T, 6, 1>;

    using Matrix4 = Eigen::Matrix<T, 4, 4>;

    Eigen::Map<Vector3> residuals(sResiduals);

    Sophus::SE3<T> T_w_i;
    Vector6 vel, accel;

    CeresSplineHelper<N>::template evaluate_lie<T, Sophus::SE3>(
        sKnots, u, inv_dt, &T_w_i, &vel, &accel);

    Matrix4 vel_hat = Sophus::SE3<T>::hat(vel);
    Matrix4 accel_hat = Sophus::SE3<T>::hat(accel);

    Matrix4 ddpose = T_w_i.matrix() * (vel_hat * vel_hat + accel_hat);

    Vector3 accel_w = ddpose.col(3).template head<3>();

    // Gravity
    Eigen::Map<Vector3 const> const g(sKnots[N]);
    Eigen::Map<Vector3 const> const bias(sKnots[N + 1]);

    residuals = inv_std * (T_w_i.so3().inverse() * (accel_w + g) -
                           measurement.cast<T>() + bias);

    return true;
  }

  Eigen::Vector3d measurement;
  double u, inv_dt, inv_std;
};

template <int _N, bool OLD_TIME_DERIV>
struct TDOACostFunctorSE3 {
  static constexpr int N = _N;        // Order of the spline.

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TDOACostFunctorSE3(double tdoa_, const Vector3d &pos_anchor_i_, const Vector3d &pos_anchor_j_, 
                       const Vector3d &offset_, double w_, double u, double inv_dt)
    :   tdoa        (tdoa_            ),
        pos_anchor_i(pos_anchor_i_    ),
        pos_anchor_j(pos_anchor_j_    ),
        offset      (offset_          ),
        w           (w_               ),
        u(u),
        inv_dt(inv_dt) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vector2 = Eigen::Matrix<T, 2, 1>;
    using Vector3 = Eigen::Matrix<T, 3, 1>;
    using Vector4 = Eigen::Matrix<T, 4, 1>;

    using Matrix4 = Eigen::Matrix<T, 4, 4>;

    Eigen::Map<Matrix<T, 1, 1>> residual(sResiduals);

    Sophus::SE3<T> T_w_i;

    CeresSplineHelper<N>::template evaluate_lie<T, Sophus::SE3>(
        sKnots, u, inv_dt, &T_w_i);

    Vector3 p_tag_W = T_w_i.so3() * offset + T_w_i.translation();
    Vector3 diff_i = p_tag_W - pos_anchor_i;
    Vector3 diff_j = p_tag_W - pos_anchor_j;  
    residual[0] = w*(diff_j.norm() - diff_i.norm() - tdoa);

    return true;
  }
    // TDOA measurement
    double tdoa;

    // Anchor positions
    Vector3d pos_anchor_i;
    Vector3d pos_anchor_j;
    const Vector3d offset;

    // Weight
    double w = 10;

    double u, inv_dt;
};


}  // namespace basalt