#pragma once

#include "basalt/ceres_spline_helper.h"

namespace basalt {

template <int _N>
struct CalibAccelerationCostFunctorSplit : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CalibAccelerationCostFunctorSplit(const Eigen::Vector3d& measurement,
                                    double u, double inv_dt, double inv_std)
      : measurement(measurement), u(u), inv_dt(inv_dt), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<Vector3> residuals(sResiduals);

    Sophus::SO3<T> R_w_i;
    CeresSplineHelper<N>::template evaluate_lie<T, Sophus::SO3>(sKnots, u,
                                                                inv_dt, &R_w_i);

    Vector3 accel_w;
    CeresSplineHelper<N>::template evaluate<T, 3, 2>(sKnots + N, u, inv_dt,
                                                     &accel_w);

    // Gravity
    Eigen::Map<Vector3 const> const g(sKnots[2 * N]);
    Eigen::Map<Vector3 const> const bias(sKnots[2 * N + 1]);

    residuals =
        inv_std * (R_w_i.inverse() * (accel_w + g) - measurement + bias);

    return true;
  }

  Eigen::Vector3d measurement;
  double u, inv_dt, inv_std;
};

template <int _N, template <class> class GroupT, bool OLD_TIME_DERIV>
struct CalibGyroCostFunctorSplit : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  using Tangentd = typename GroupT<double>::Tangent;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CalibGyroCostFunctorSplit(const Tangentd& measurement, double u,
                            double inv_dt, double inv_std = 1)
      : measurement(measurement), u(u), inv_dt(inv_dt), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Tangent = typename GroupT<T>::Tangent;

    Eigen::Map<Tangent> residuals(sResiduals);

    Tangent rot_vel;

    CeresSplineHelper<N>::template evaluate_lie<T, GroupT>(sKnots, u, inv_dt,
                                                            nullptr, &rot_vel);

    Eigen::Map<Tangent const> const bias(sKnots[N]);

    residuals = inv_std * (rot_vel - measurement + bias);

    return true;
  }

  Tangentd measurement;
  double u, inv_dt, inv_std;
};

template <int _N>
struct TDOACostFunctorSplit : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TDOACostFunctorSplit(double tdoa_, const Vector3d &pos_anchor_i_, const Vector3d &pos_anchor_j_, 
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

    Sophus::SO3<T> R_w_i;
    CeresSplineHelper<N>::template evaluate_lie<T, Sophus::SO3>(sKnots, u,
                                                                inv_dt, &R_w_i);

    Vector3 t_w_i;
    CeresSplineHelper<N>::template evaluate<T, 3, 0>(sKnots + N, u, inv_dt,
                                                     &t_w_i);

    Vector3 p_tag_W = R_w_i * offset + t_w_i;
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