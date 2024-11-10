#include <ceres/ceres.h>

/// @brief Local parametrization for ceres that can be used with Sophus Lie
/// group implementations.
template <class Groupd>
class AutoDiffSO3Parameterization : public ceres::LocalParameterization
{
public:
    virtual ~AutoDiffSO3Parameterization() {}

    using Tangentd = typename Groupd::Tangent;

    /// @brief plus operation for Ceres
    ///
    ///  T * exp(x)
    ///
    virtual bool Plus(double const *T_raw, double const *delta_raw,
                        double *T_plus_delta_raw) const
    {
        Eigen::Map<Groupd const> const T(T_raw);
        Eigen::Map<Tangentd const> const delta(delta_raw);
        Eigen::Map<Groupd> T_plus_delta(T_plus_delta_raw);
        T_plus_delta = T * Groupd::exp(delta);
        return true;
    }

    ///@brief Jacobian of plus operation for Ceres
    ///
    /// Dx T * exp(x)  with  x=0
    ///
    /// For SO3 group, function 'Dx_this_mul_exp_x_at_0()' is supposed to return a
    /// 4*3 matrix, which is
    ///       /        \ 
    ///      |  w -z  y |
    /// J =  |  z  w -x |
    ///      | -y  x  w |
    ///      | -x -y -z |
    ///       \        /
    virtual bool ComputeJacobian(double const *T_raw,
                                    double *jacobian_raw) const
    {
        Eigen::Map<Groupd const> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, Groupd::num_parameters, Groupd::DoF,
                                    Eigen::RowMajor>>
            jacobian(jacobian_raw);
        /// check function `Dx_this_mul_exp_x_at_0` in "sophus/so3.hpp" to decide
        /// which one to use
#if true
        jacobian = T.Dx_this_mul_exp_x_at_0();
#else
        jacobian = T.Dx_this_mul_exp_x_at_0().transpose();
#endif
        return true;
    }

    ///@brief Global size
    virtual int GlobalSize() const { return Groupd::num_parameters; }

    ///@brief Local size
    virtual int LocalSize() const { return Groupd::DoF; }
};