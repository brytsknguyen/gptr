#include <ceres/ceres.h>

/// @brief Local parametrization for ceres that can be used with Sophus Lie
/// group implementations.
template <class Groupd>
class AutoDiffSO3Parameterization : public ceres::Manifold
{
public:
    virtual ~AutoDiffSO3Parameterization() {}

    using Tangentd = typename Groupd::Tangent;

    virtual bool Plus(double const *x_, double const *delta_, double *x_plus_delta_) const
    {
        Eigen::Map<Groupd const> const x(x_);
        Eigen::Map<Tangentd const> const delta(delta_);
        Eigen::Map<Groupd> x_plus_delta(x_plus_delta_);

        x_plus_delta = x * Groupd::exp(delta);

        return true;
    }

    virtual bool PlusJacobian(const double* x_, double* jacobian_) const
    {
        Eigen::Map<Groupd const> const x(x_);
        Eigen::Map<Eigen::Matrix<double, Groupd::num_parameters, Groupd::DoF, Eigen::RowMajor>> jacobian(jacobian_);

        jacobian = x.Dx_this_mul_exp_x_at_0();

        return true;
    }

    bool Minus(const double *y_, const double *x_, double *y_minus_x_) const
    {
        Eigen::Map<Groupd const> const x(x_);
        Eigen::Map<Groupd const> const y(y_);
        Eigen::Map<Tangentd> y_minus_x(y_minus_x_);

        y_minus_x = (x.inverse()*y).log();

        return true;
    }

    virtual bool MinusJacobian(const double* x_, double* jacobian_) const
    {
        Eigen::Map<Groupd const> const x(x_);
        Eigen::Map<Eigen::Matrix<double, Groupd::DoF, Groupd::num_parameters, Eigen::RowMajor>> jacobian(jacobian_);
        jacobian.setZero();

        jacobian = x.Dx_log_this_inv_by_x_at_this();

        return true;
    }

    ///@brief Global size
    virtual int AmbientSize() const { return Groupd::num_parameters; }

    ///@brief Local size
    virtual int TangentSize() const { return Groupd::DoF; }
};

typedef AutoDiffSO3Parameterization<SO3d> AutoDiffSO3dParameterization;