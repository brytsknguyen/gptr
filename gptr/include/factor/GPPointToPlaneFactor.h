#include <ceres/ceres.h>
#include "GaussianProcess.hpp"
#include "utility.h"

using namespace Eigen;

class GPPointToPlaneFactor: public ceres::CostFunction
{
public:

    // Destructor
    ~GPPointToPlaneFactor() {};

    // Constructor
    GPPointToPlaneFactor(const Vector3d &f_, const Vector4d &coef, double w_,
                         GPMixerPtr gpm_, double s_)
    :   f          (f_               ),
        n          (coef.head<3>()   ),
        m          (coef.tail<1>()(0)),
        w          (w_               ),
        Dt         (gpm_->getDt()    ),
        s          (s_               ),
        gpm        (gpm_             )

    {
        // 1-element residual: n^T*(Rt*f + pt) + m
        set_num_residuals(1);

        // Rotation of the first knot
        mutable_parameter_block_sizes()->push_back(4);
        // Angular velocity of the first knot
        mutable_parameter_block_sizes()->push_back(3);
        // Angular acceleration of the first knot
        mutable_parameter_block_sizes()->push_back(3);
        // Position of the first knot
        mutable_parameter_block_sizes()->push_back(3);
        // Velocity of the first knot
        mutable_parameter_block_sizes()->push_back(3);
        // Acceleration of the first knot
        mutable_parameter_block_sizes()->push_back(3);

        // Rotation of the second knot
        mutable_parameter_block_sizes()->push_back(4);
        // Angular velocity of the second knot
        mutable_parameter_block_sizes()->push_back(3);
        // Angular acceleration of the second knot
        mutable_parameter_block_sizes()->push_back(3);
        // Position of the second knot
        mutable_parameter_block_sizes()->push_back(3);
        // Velocity of the second knot
        mutable_parameter_block_sizes()->push_back(3);
        // Acceleration of the second knot
        mutable_parameter_block_sizes()->push_back(3);
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Map parameters to the control point states
        GPState Xa(0);  gpm->MapParamToState(parameters, RaIdx, Xa);
        GPState Xb(Dt); gpm->MapParamToState(parameters, RbIdx, Xb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the pose at sampling time --------------------------------------------------------------*/

        GPState Xt(s*Dt); vector<vector<Matrix3d>> DXt_DXa; vector<vector<Matrix3d>> DXt_DXb;

        Eigen::Matrix<double, Eigen::Dynamic, 1> gammaa;
        Eigen::Matrix<double, Eigen::Dynamic, 1> gammab;
        Eigen::Matrix<double, Eigen::Dynamic, 1> gammat;

        gpm->ComputeXtAndJacobians(Xa, Xb, Xt, DXt_DXa, DXt_DXb, gammaa, gammab, gammat);

        // Residual
        Eigen::Map<Matrix<double, 1, 1>> residual(residuals);
        residual[0] = w*(n.dot(Xt.R*f + Xt.P) + m);

        /* #endregion Calculate the pose at sampling time -----------------------------------------------------------*/
    
        if (!jacobians)
            return true;

        Matrix<double, 1, 3> Dr_DRt = w*(-n.transpose()*Xt.R.matrix()*SO3d::hat(f));
        Matrix<double, 1, 3> Dr_DPt = w*( n.transpose());

        for(size_t idx = Ridx; idx <= Aidx; idx++)
        {
            size_t idxa = idx, idxb = idx+RbIdx;

            if (idx == Ridx)
            {
                Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> Dr_DXa(jacobians[idxa]);
                Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> Dr_DXb(jacobians[idxb]);

                if(jacobians[idxa]) { Dr_DXa.setZero(); Dr_DXa.block<1, 3>(0, 0) = Dr_DRt*DXt_DXa[Ridx][idx] + Dr_DPt*DXt_DXa[Pidx][idx]; }
                if(jacobians[idxb]) { Dr_DXb.setZero(); Dr_DXb.block<1, 3>(0, 0) = Dr_DRt*DXt_DXb[Ridx][idx] + Dr_DPt*DXt_DXb[Pidx][idx]; }
            }
            else
            {
                Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> Dr_DXa(jacobians[idxa]);
                Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> Dr_DXb(jacobians[idxb]);

                if(jacobians[idxa]) { Dr_DXa.setZero(); Dr_DXa.block<1, 3>(0, 0) = Dr_DRt*DXt_DXa[Ridx][idx] + Dr_DPt*DXt_DXa[Pidx][idx]; }
                if(jacobians[idxb]) { Dr_DXb.setZero(); Dr_DXb.block<1, 3>(0, 0) = Dr_DRt*DXt_DXb[Ridx][idx] + Dr_DPt*DXt_DXb[Pidx][idx]; }
            }
        }

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