#include <ceres/ceres.h>
#include "GaussianProcess.hpp"
#include "utility.h"

class GPPointToPlaneFactorAutodiff
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GPPointToPlaneFactorAutodiff(const Vector3d &f_, const Vector4d &coef, double w_, GPMixerPtr gpm_, double s_)
    :   f          (f_               ),
        n          (coef.head<3>()   ),
        m          (coef.tail<1>()(0)),
        w          (w_               ),
        Dt         (gpm_->getDt()    ),
        s          (s_               ),
        gpm        (gpm_             )
    {}

    template <class T>
    bool operator()(T const *const *parameters, T *residuals) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        GPState<T> Xa(0);  gpm->MapParamToState<T>(parameters, RaIdx, Xa);
        GPState<T> Xb(Dt); gpm->MapParamToState<T>(parameters, RbIdx, Xb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the pose at sampling time --------------------------------------------------------------*/

        GPState<T> Xt(s*Dt); vector<vector<Mat3T>> DXt_DXa; vector<vector<Mat3T>> DXt_DXb;
        gpm->ComputeXtAndJacobians<T>(Xa, Xb, Xt, DXt_DXa, DXt_DXb);

        // Residual
        Eigen::Map<Matrix<T, 1, 1>> residual(residuals);
        residual[0] = w*(n.dot(Xt.R*f + Xt.P) + m);

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        return true;
    }

private:

    // Feature coordinates in body frame
    Vector3d f;

    // Plane normal
    Vector3d n;

    // Plane offset
    double m;

    // Weight
    double w = 0.1;

    // Gaussian process params
    
    const int Ridx = 0;
    const int Oidx = 1;
    const int Sidx = 2;
    const int Pidx = 3;
    const int Vidx = 4;
    const int Aidx = 5;

    const int RaIdx = 0;
    const int OaIdx = 1;
    const int SaIdx = 2;
    const int PaIdx = 3;
    const int VaIdx = 4;
    const int AaIdx = 5;

    const int RbIdx = 6;
    const int ObIdx = 7;
    const int SbIdx = 8;
    const int PbIdx = 9;
    const int VbIdx = 10;
    const int AbIdx = 11;

    // Interpolation param
    double Dt;
    double s;
    GPMixerPtr gpm;
};