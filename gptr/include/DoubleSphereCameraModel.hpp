#pragma once

#define UNUSED(x) (void)(x)

template <typename Scalar_ = double>
class DoubleSphereCamera
{
public:
  using Scalar = Scalar_;
  static constexpr int N = 6; ///< Number of intrinsic parameters.

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  DoubleSphereCamera() { param_.setZero(); }

  explicit DoubleSphereCamera(const VecN &p) { param_ = p; }

  template <class Scalar2>
  DoubleSphereCamera<Scalar2> cast() const
  {
    return DoubleSphereCamera<Scalar2>(param_.template cast<Scalar2>());
  }

  static std::string getName() { return "ds"; }

  template <class DerivedPoint3D, class DerivedPoint2D,
            class DerivedJ3D = std::nullptr_t,
            class DerivedJparam = std::nullptr_t>
  inline bool project(const Eigen::MatrixBase<DerivedPoint3D> &p3d,
                      Eigen::MatrixBase<DerivedPoint2D> &proj,
                      DerivedJ3D d_proj_d_p3d = nullptr,
                      DerivedJparam d_proj_d_param = nullptr) const
  {
    // checkProjectionDerivedTypes<DerivedPoint3D, DerivedPoint2D, DerivedJ3D,
    //                             DerivedJparam, N>();

    // const typename EvalOrReference<DerivedPoint3D>::Type p3d_eval(p3d);

    const Scalar &fx = param_[0];
    const Scalar &fy = param_[1];
    const Scalar &cx = param_[2];
    const Scalar &cy = param_[3];

    const Scalar &xi = param_[4];
    const Scalar &alpha = param_[5];

    // const Scalar& x = p3d_eval[0];
    // const Scalar& y = p3d_eval[1];
    // const Scalar& z = p3d_eval[2];

    const Scalar &x = p3d[0];
    const Scalar &y = p3d[1];
    const Scalar &z = p3d[2];

    const Scalar xx = x * x;
    const Scalar yy = y * y;
    const Scalar zz = z * z;

    const Scalar r2 = xx + yy;

    const Scalar d1_2 = r2 + zz;
    const Scalar d1 = sqrt(d1_2);

    const Scalar w1 = alpha > Scalar(0.5) ? (Scalar(1) - alpha) / alpha
                                          : alpha / (Scalar(1) - alpha);
    const Scalar w2 =
        (w1 + xi) / sqrt(Scalar(2) * w1 * xi + xi * xi + Scalar(1));

    const bool is_valid = (z > -w2 * d1);

    const Scalar k = xi * d1 + z;
    const Scalar kk = k * k;

    const Scalar d2_2 = r2 + kk;
    const Scalar d2 = sqrt(d2_2);

    const Scalar norm = alpha * d2 + (Scalar(1) - alpha) * k;

    const Scalar mx = x / norm;
    const Scalar my = y / norm;

    proj[0] = fx * mx + cx;
    proj[1] = fy * my + cy;

    if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>)
    {
      // BASALT_ASSERT(d_proj_d_p3d);

      const Scalar norm2 = norm * norm;
      const Scalar xy = x * y;
      const Scalar tt2 = xi * z / d1 + Scalar(1);

      const Scalar d_norm_d_r2 = (xi * (Scalar(1) - alpha) / d1 +
                                  alpha * (xi * k / d1 + Scalar(1)) / d2) /
                                 norm2;

      const Scalar tmp2 =
          ((Scalar(1) - alpha) * tt2 + alpha * k * tt2 / d2) / norm2;

      d_proj_d_p3d->setZero();
      (*d_proj_d_p3d)(0, 0) = fx * (Scalar(1) / norm - xx * d_norm_d_r2);
      (*d_proj_d_p3d)(1, 0) = -fy * xy * d_norm_d_r2;

      (*d_proj_d_p3d)(0, 1) = -fx * xy * d_norm_d_r2;
      (*d_proj_d_p3d)(1, 1) = fy * (Scalar(1) / norm - yy * d_norm_d_r2);

      (*d_proj_d_p3d)(0, 2) = -fx * x * tmp2;
      (*d_proj_d_p3d)(1, 2) = -fy * y * tmp2;
    }
    else
    {
      UNUSED(d_proj_d_p3d);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>)
    {
      // BASALT_ASSERT(d_proj_d_param);

      const Scalar norm2 = norm * norm;

      (*d_proj_d_param).setZero();
      (*d_proj_d_param)(0, 0) = mx;
      (*d_proj_d_param)(0, 2) = Scalar(1);
      (*d_proj_d_param)(1, 1) = my;
      (*d_proj_d_param)(1, 3) = Scalar(1);

      const Scalar tmp4 = (alpha - Scalar(1) - alpha * k / d2) * d1 / norm2;
      const Scalar tmp5 = (k - d2) / norm2;

      (*d_proj_d_param)(0, 4) = fx * x * tmp4;
      (*d_proj_d_param)(1, 4) = fy * y * tmp4;

      (*d_proj_d_param)(0, 5) = fx * x * tmp5;
      (*d_proj_d_param)(1, 5) = fy * y * tmp5;
    }
    else
    {
      UNUSED(d_proj_d_param);
    }

    return is_valid;
  }

  inline void setFromInit(double fx, double fy, double cx, double cy, double xi, double alpha)
  {
    param_[0] = fx;
    param_[1] = fy;
    param_[2] = cx;
    param_[3] = cy;
    param_[4] = xi;
    param_[5] = alpha;
  }

  //   void operator+=(const VecN& inc) {
  //     param_ += inc;
  //     param_[4] = std::clamp(param_[4], Scalar(-1), Scalar(1));
  //     param_[5] = std::clamp(param_[5], Scalar(0), Scalar(1));
  //   }

  const VecN &getParam() const { return param_; }

  //   static Eigen::aligned_vector<DoubleSphereCamera> getTestProjections() {
  //     Eigen::aligned_vector<DoubleSphereCamera> res;

  //     VecN vec1;
  //     vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0.5 * -0.150694, 0.5 * 1.48785;
  //     res.emplace_back(vec1);

  //     return res;
  //   }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  VecN param_;
};

struct CameraCalibration
{
  template <typename T>
  using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

  aligned_deque<Sophus::SE3d> T_i_c;
  aligned_deque<DoubleSphereCamera<double>> intrinsics;
  // Eigen::aligned_vector<Eigen::Vector2i> resolution;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};