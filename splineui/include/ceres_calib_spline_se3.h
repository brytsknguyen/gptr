#pragma once

#include "utils/assert.h"
#include "utils/common_utils.h"
#include "basalt/ceres_local_param.hpp"

#include <ceres/ceres.h>
#include "ceres_calib_se3_residuals.h"

template <int _N, bool OLD_TIME_DERIV = false>
class CeresCalibrationSplineSe3 {
 public:
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  static constexpr double ns_to_s = 1e-9;  ///< Nanosecond to second conversion
  static constexpr double s_to_ns = 1e9;   ///< Second to nanosecond conversion

  CeresCalibrationSplineSe3(int64_t time_interval_ns, int64_t start_time_ns = 0)
      : dt_ns(time_interval_ns), start_t_ns(start_time_ns) {
    inv_dt = s_to_ns / dt_ns;

    accel_bias.setZero();
    gyro_bias.setZero();
  };

  Sophus::SE3d getPose(int64_t time_ns) const {
    int64_t st_ns = (time_ns - start_t_ns);

    BASALT_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns);

    int64_t s = st_ns / dt_ns;
    double u = double(st_ns % dt_ns) / double(dt_ns);

    BASALT_ASSERT_STREAM(s >= 0, "s " << s);
    BASALT_ASSERT_STREAM(size_t(s + N) <= knots.size(), "s " << s << " N " << N
                                                             << " knots.size() "
                                                             << knots.size());

    Sophus::SE3d res;

    Sophus::SO3d rot;
    Eigen::Vector3d trans;

    std::vector<const double*> vec;
    for (int i = 0; i < N; i++) {
      vec.emplace_back(knots[s + i].data());
    }

    basalt::CeresSplineHelper<N>::template evaluate_lie<double, Sophus::SE3>(
        &vec[0], u, inv_dt, &res);

    return res;
  }

  void init(const Sophus::SE3d& init, int num_knots) {
    knots = Eigen::aligned_vector<Sophus::SE3d>(num_knots, init);

    for (int i = 0; i < num_knots; i++) {
      ceres::Manifold* local_parameterization =
          new basalt::LieLocalParameterization<Sophus::SE3d>();
      problem.AddParameterBlock(knots[i].data(), Sophus::SE3d::num_parameters, local_parameterization);
    }
    problem.AddParameterBlock(gyro_bias.data(), 3);
    problem.AddParameterBlock(accel_bias.data(), 3);
    problem.AddParameterBlock(g.data(), 3);
    problem.SetParameterBlockConstant(g.data());

  }

  void addGyroMeasurement(const Eigen::Vector3d& meas, int64_t time_ns, double w) {
    int64_t st_ns = (time_ns - start_t_ns);

    BASALT_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns);

    int64_t s = st_ns / dt_ns;
    double u = double(st_ns % dt_ns) / double(dt_ns);

    BASALT_ASSERT_STREAM(s >= 0, "s " << s);
    BASALT_ASSERT_STREAM(size_t(s + N) <= knots.size(), "s " << s << " N " << N
                                                             << " knots.size() "
                                                             << knots.size());

    using FunctorT = basalt::CalibGyroCostFunctorSE3<_N, OLD_TIME_DERIV>;

    FunctorT* functor = new FunctorT(
        meas, u, inv_dt, w);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
        new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    for (int i = 0; i < N; i++) {
      cost_function->AddParameterBlock(7);
    }
    cost_function->AddParameterBlock(3);
    cost_function->SetNumResiduals(3);

    std::vector<double*> vec;
    for (int i = 0; i < N; i++) {
      vec.emplace_back(knots[s + i].data());
    }
    vec.emplace_back(gyro_bias.data());

    problem.AddResidualBlock(cost_function, NULL, vec);
  }

  void addAccelMeasurement(const Eigen::Vector3d& meas, int64_t time_ns, double w) {
    int64_t st_ns = (time_ns - start_t_ns);

    BASALT_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns);

    int64_t s = st_ns / dt_ns;
    double u = double(st_ns % dt_ns) / double(dt_ns);

    BASALT_ASSERT_STREAM(s >= 0, "s " << s);
    BASALT_ASSERT_STREAM(size_t(s + N) <= knots.size(), "s " << s << " N " << N
                                                             << " knots.size() "
                                                             << knots.size());

    using FunctorT = basalt::CalibAccelerationCostFunctorSE3<N, OLD_TIME_DERIV>;

    FunctorT* functor = new FunctorT(
        meas, u, inv_dt, w);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
        new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    for (int i = 0; i < N; i++) {
      cost_function->AddParameterBlock(7);
    }
    cost_function->AddParameterBlock(3);
    cost_function->AddParameterBlock(3);
    cost_function->SetNumResiduals(3);

    std::vector<double*> vec;
    for (int i = 0; i < N; i++) {
      vec.emplace_back(knots[s + i].data());
    }
    vec.emplace_back(g.data());
    vec.emplace_back(accel_bias.data());

    problem.AddResidualBlock(cost_function, NULL, vec);
  }

  void addTDOAMeasurement(double data, int64_t time_ns, const Eigen::Vector3d &pos_anA, const Eigen::Vector3d &pos_anB,
                          const Eigen::Vector3d &offset, double w) {
    int64_t st_ns = (time_ns - start_t_ns);

    BASALT_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns);

    int64_t s = st_ns / dt_ns;
    double u = double(st_ns % dt_ns) / double(dt_ns);

    BASALT_ASSERT_STREAM(s >= 0, "s " << s);
    BASALT_ASSERT_STREAM(
        size_t(s + N) <= knots.size(),
        "s " << s << " N " << N << " knots.size() " << knots.size());

    using FunctorT = basalt::TDOACostFunctorSE3<N, OLD_TIME_DERIV>;

    FunctorT* functor = new FunctorT(data, pos_anA, pos_anB, offset, w, u, inv_dt);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
        new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    for (int i = 0; i < N; i++) {
      cost_function->AddParameterBlock(7);
    }

    cost_function->SetNumResiduals(1);

    std::vector<double*> vec;
    for (int i = 0; i < N; i++) {
      vec.emplace_back(knots[s + i].data());
    }

    problem.AddResidualBlock(cost_function, NULL, vec);
  }

  int64_t maxTimeNs() const {
    return start_t_ns + (knots.size() - N + 1) * dt_ns - 1;
  }

  int64_t minTimeNs() const { return start_t_ns; }

  double maxTimes() const {
    return (start_t_ns + (knots.size() - N + 1) * dt_ns - 1)*1e-9;
  }

  double minTimes() const { return start_t_ns*1e-9; }

  ceres::Solver::Summary optimize() {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = MAX_THREADS;
    options.max_num_iterations = 50;
    options.function_tolerance = 0.0;
    options.gradient_tolerance = 0.0;
    options.parameter_tolerance = 0.0;

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    return summary;
  }

  Sophus::SE3d getKnot(int i) const { return knots[i]; }

  size_t numKnots() { return knots.size(); }


  void setG(Eigen::Vector3d& a) { g = a; }
  const Eigen::Vector3d& getG() { return g; }

  Eigen::Vector3d getGyroBias() { return gyro_bias; }
  Eigen::Vector3d getAccelBias() { return accel_bias; }
  double getDt() { return dt_ns*1e-9; }

 private:
  int64_t dt_ns, start_t_ns;
  double inv_dt;

  Eigen::aligned_vector<Sophus::SE3d> knots;
  Eigen::Vector3d g, accel_bias, gyro_bias;

  ceres::Problem problem;
};
