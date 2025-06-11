#include "utility.h"

#include "GaussianProcess.hpp"
#include "SE3JQ.hpp"

#include "AutoDiffSO3Parameterization.hpp"

using namespace Eigen;
using namespace std;

MatrixXd thresholdMatrix(const MatrixXd &matrix, double threshold)
{
    return (matrix.array().abs() >= threshold).cast<double>();
}

namespace Sophus
{
    namespace test
    {

        class LocalParameterizationSE3 : public ceres::LocalParameterization
        {
        public:
            virtual ~LocalParameterizationSE3() {}

            // SE3 plus operation for Ceres
            //
            //  T * exp(x)
            //
            virtual bool Plus(double const *T_raw, double const *delta_raw,
                              double *T_plus_delta_raw) const
            {
                Eigen::Map<SE3d const> const T(T_raw);
                Eigen::Map<Vector6d const> const delta(delta_raw);
                Vector3d dThe = delta.head<3>();
                Vector3d dRho = delta.tail<3>();
                Vector6d delta_; delta_ << dThe, dRho;

                Eigen::Map<SE3d> T_plus_delta(T_plus_delta_raw);
                T_plus_delta = T * SE3d::exp(delta_);
                return true;
            }

            // Jacobian of SE3 plus operation for Ceres
            //
            // Dx T * exp(x)  with  x=0
            //
            virtual bool ComputeJacobian(double const *T_raw,
                                         double *jacobian_raw) const
            {
                Eigen::Map<SE3d const> T(T_raw);
                Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jacobian(jacobian_raw);
                jacobian.setZero();

                Matrix<double, 7, 6> J = T.Dx_this_mul_exp_x_at_0();
                jacobian.block<4, 3>(0, 0) = J.block<4, 3>(0, 3);
                jacobian.block<3, 3>(4, 3) = J.block<3, 3>(4, 0);
                return true;
            }

            virtual int GlobalSize() const { return SE3d::num_parameters; }
            virtual int LocalSize() const { return SE3d::DoF; }
        };
    } // namespace test
} // namespace Sophus

static constexpr int RESITRZ_SIZE = 18;

class IntrinsicJcbTestAutodiffFactor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IntrinsicJcbTestAutodiffFactor(GPMixerPtr gpm_, double s_)
    :   s          (s_               ),
        gpm        (gpm_             )
    {}

    template <class T>
    bool operator()(T const *const *parameters, T *residuals) const
    {
        using SO3T  = Sophus::SO3<T>;
        using SE3T  = Sophus::SE3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Vec6T = Eigen::Matrix<T, 6, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        /* #region Map the memory to control points -----------------------------------------------------------------*/

        double Dt = gpm->getDt();

        // Map parameters to the control point states
        GPState<T> Xa(0);  gpm->MapParamToState<T>(parameters, RaIdx, Xa);
        GPState<T> Xb(Dt); gpm->MapParamToState<T>(parameters, RbIdx, Xb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the pose at sampling time --------------------------------------------------------------*/

        GPState<T> Xt(s*Dt); vector<vector<Mat3T>> DXt_DXa; vector<vector<Mat3T>> DXt_DXb;
        gpm->ComputeXtAndJacobians<T>(Xa, Xb, Xt, DXt_DXa, DXt_DXb);

        // Residual
        Eigen::Map<Matrix<T, RESITRZ_SIZE, 1>> residual(residuals);
        residual << Xt.R.log(), Xt.O, Xt.S, Xt.P, Xt.V, Xt.A;

        // cout << residual.cast<double>() << endl;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        return true;
    }

private:

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
    double s;
    GPMixerPtr gpm;
};

map<string, MatrixXd> Jdebug;

class IntrinsicJcbTestAnalyticFactor : public ceres::CostFunction
{
public:
    // Destructor
    ~IntrinsicJcbTestAnalyticFactor() {};

    // Constructor
    IntrinsicJcbTestAnalyticFactor(GPMixerPtr gpm_, double s_)
    :   s          (s_               ),
        gpm        (gpm_             )

    {
        // 1-element residual: n^T*(Rt*f + pt) + m
        set_num_residuals(RESITRZ_SIZE);

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
        using SO3d  = Sophus::SO3<double>;
        using SE3d  = Sophus::SE3<double>;
        using Vec3d = Eigen::Matrix<double, 3, 1>;
        using Vec6d = Eigen::Matrix<double, 6, 1>;
        using Mat3d = Eigen::Matrix<double, 3, 3>;

        /* #region Map the memory to control points -----------------------------------------------------------------*/

        double Dt = gpm->getDt();

        // Map parameters to the control point states
        GPState Xa(0);  gpm->MapParamToState(parameters, RaIdx, Xa);
        GPState Xb(Dt); gpm->MapParamToState(parameters, RbIdx, Xb);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the pose at sampling time --------------------------------------------------------------*/

        GPState Xt(s*Dt); vector<vector<Matrix3d>> DXt_DXa; vector<vector<Matrix3d>> DXt_DXb;
        gpm->ComputeXtAndJacobians(Xa, Xb, Xt, DXt_DXa, DXt_DXb);

        SE3d Tft; Vec6d Twt; Vec6d Wrt;
        Xt.GetTUW(Tft, Twt, Wrt);

        // Residual
        Eigen::Map<Matrix<double, RESITRZ_SIZE, 1>> residual(residuals);
        residual << Xt.R.log(), Xt.O, Xt.S, Xt.P, Xt.V, Xt.A;

        // cout << residual << endl;

        /* #endregion Calculate the pose at sampling time -----------------------------------------------------------*/

        if (!jacobians)
            return true;

        Mat3 Eye = Mat3::Identity(3, 3);
        Matrix<double, RESITRZ_SIZE, 3> Dr_DRt; Dr_DRt.setZero(); Dr_DRt.block<3, 3>(Ridx*3, 0) = gpm->JrInv(Xt.R.log());
        Matrix<double, RESITRZ_SIZE, 3> Dr_DOt; Dr_DOt.setZero(); Dr_DOt.block<3, 3>(Oidx*3, 0) = Eye;
        Matrix<double, RESITRZ_SIZE, 3> Dr_DSt; Dr_DSt.setZero(); Dr_DSt.block<3, 3>(Sidx*3, 0) = Eye;
        Matrix<double, RESITRZ_SIZE, 3> Dr_DPt; Dr_DPt.setZero(); Dr_DPt.block<3, 3>(Pidx*3, 0) = Eye;
        Matrix<double, RESITRZ_SIZE, 3> Dr_DVt; Dr_DVt.setZero(); Dr_DVt.block<3, 3>(Vidx*3, 0) = Eye;
        Matrix<double, RESITRZ_SIZE, 3> Dr_DAt; Dr_DAt.setZero(); Dr_DAt.block<3, 3>(Aidx*3, 0) = Eye;

        for(size_t idx = Ridx; idx <= Aidx; idx++)
        {
            if (!jacobians[idx])
                continue;

            if (idx == Ridx)
            {
                Eigen::Map<Eigen::Matrix<double, RESITRZ_SIZE, 4, Eigen::RowMajor>> Dr_DXa(jacobians[idx]);
                Eigen::Map<Eigen::Matrix<double, RESITRZ_SIZE, 4, Eigen::RowMajor>> Dr_DXb(jacobians[idx + RbIdx]);

                Dr_DXa.setZero();
                Dr_DXa.block<RESITRZ_SIZE, 3>(0, 0) = Dr_DRt * DXt_DXa[Ridx][idx]
                                                    + Dr_DOt * DXt_DXa[Oidx][idx]
                                                    + Dr_DSt * DXt_DXa[Sidx][idx]
                                                    + Dr_DPt * DXt_DXa[Pidx][idx]
                                                    + Dr_DVt * DXt_DXa[Vidx][idx]
                                                    + Dr_DAt * DXt_DXa[Aidx][idx];

                Dr_DXb.setZero();
                Dr_DXb.block<RESITRZ_SIZE, 3>(0, 0) = Dr_DRt * DXt_DXb[Ridx][idx]
                                                    + Dr_DOt * DXt_DXb[Oidx][idx]
                                                    + Dr_DSt * DXt_DXb[Sidx][idx]
                                                    + Dr_DPt * DXt_DXb[Pidx][idx]
                                                    + Dr_DVt * DXt_DXb[Vidx][idx]
                                                    + Dr_DAt * DXt_DXb[Aidx][idx];
            }
            else
            {
                Eigen::Map<Eigen::Matrix<double, RESITRZ_SIZE, 3, Eigen::RowMajor>> Dr_DXa(jacobians[idx]);
                Eigen::Map<Eigen::Matrix<double, RESITRZ_SIZE, 3, Eigen::RowMajor>> Dr_DXb(jacobians[idx + RbIdx]);

                Dr_DXa.setZero();
                Dr_DXa.block<RESITRZ_SIZE, 3>(0, 0) = Dr_DRt * DXt_DXa[Ridx][idx]
                                                    + Dr_DOt * DXt_DXa[Oidx][idx]
                                                    + Dr_DSt * DXt_DXa[Sidx][idx]
                                                    + Dr_DPt * DXt_DXa[Pidx][idx]
                                                    + Dr_DVt * DXt_DXa[Vidx][idx]
                                                    + Dr_DAt * DXt_DXa[Aidx][idx];

                Dr_DXb.setZero();
                Dr_DXb.block<RESITRZ_SIZE, 3>(0, 0) = Dr_DRt * DXt_DXb[Ridx][idx]
                                                    + Dr_DOt * DXt_DXb[Oidx][idx]
                                                    + Dr_DSt * DXt_DXb[Sidx][idx]
                                                    + Dr_DPt * DXt_DXb[Pidx][idx]
                                                    + Dr_DVt * DXt_DXb[Vidx][idx]
                                                    + Dr_DAt * DXt_DXb[Aidx][idx];
            }
        }

        return true;
    }

private:
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
    double s;
    GPMixerPtr gpm;
};

struct FactorMeta
{
    vector<double *> so3_parameter_blocks;
    vector<double *> rv3_parameter_blocks;
    vector<double *> se3_parameter_blocks;
    vector<double *> rv6_parameter_blocks;
    vector<ceres::ResidualBlockId> residual_blocks;

    int parameter_blocks()
    {
        return (so3_parameter_blocks.size() + rv3_parameter_blocks.size() + se3_parameter_blocks.size() + rv6_parameter_blocks.size());
    }
};

Eigen::MatrixXd GetJacobian(ceres::CRSMatrix &J)
{
    Eigen::MatrixXd dense_jacobian(J.num_rows, J.num_cols);
    dense_jacobian.setZero();
    for (int r = 0; r < J.num_rows; ++r)
    {
        for (int idx = J.rows[r]; idx < J.rows[r + 1]; ++idx)
        {
            const int c = J.cols[idx];
            dense_jacobian(r, c) = J.values[idx];
        }
    }

    return dense_jacobian;
}

void EvaluateFactorRJ(ceres::Problem &problem, FactorMeta &factorMeta,
                      int local_pamaterization_type,
                      double &cost, VectorXd &residual,
                      MatrixXd &Jacobian, double &eval_time)
{
    ceres::LocalParameterization *localparameterization;
    for (auto parameter : factorMeta.so3_parameter_blocks)
    {
        if (local_pamaterization_type == 0)
        {
            localparameterization = new AutoDiffSO3Parameterization<SO3d>();
            problem.SetParameterization(parameter, localparameterization);
        }
        else if (local_pamaterization_type == 1)
        {
            localparameterization = new GPSO3dLocalParameterization();
            problem.SetParameterization(parameter, localparameterization);
        }
        else if (local_pamaterization_type == 2)
        {
            localparameterization = new Sophus::test::LocalParameterizationSE3();
            problem.SetParameterization(parameter, localparameterization);
        }
        else
            cout << "Unknown local parameterization!" << endl;
    }

    ceres::Problem::EvaluateOptions e_option;
    ceres::CRSMatrix Jacobian_;
    e_option.residual_blocks = factorMeta.residual_blocks;

    vector<double> residual_;
    TicToc tt_eval;
    problem.Evaluate(e_option, &cost, &residual_, NULL, &Jacobian_);
    tt_eval.Toc();
    eval_time = tt_eval.GetLastStop();

    residual = Eigen::Map<Eigen::VectorXd>(residual_.data(), residual_.size());
    Jacobian = GetJacobian(Jacobian_);
}

void RemoveResidualBlock(ceres::Problem &problem, FactorMeta &factorMeta)
{
    for (auto res_block : factorMeta.residual_blocks)
        problem.RemoveResidualBlock(res_block);
}

void CreateCeresProblem(ceres::Problem &problem, ceres::Solver::Options &options, ceres::Solver::Summary &summary,
                        GaussianProcessPtr &swTraj, double fixed_start, double fixed_end)
{
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = MAX_THREADS;
    options.max_num_iterations = 50;
    int KNOTS = swTraj->getNumKnots();
    // Add the parameter blocks for rotation
    for (int kidx = 0; kidx < KNOTS; kidx++)
    {
        problem.AddParameterBlock(swTraj->getKnotSO3(kidx).data(), 4, new GPSO3dLocalParameterization());
        problem.AddParameterBlock(swTraj->getKnotOmg(kidx).data(), 3);
        problem.AddParameterBlock(swTraj->getKnotAlp(kidx).data(), 3);
        problem.AddParameterBlock(swTraj->getKnotPos(kidx).data(), 3);
        problem.AddParameterBlock(swTraj->getKnotVel(kidx).data(), 3);
        problem.AddParameterBlock(swTraj->getKnotAcc(kidx).data(), 3);
    }
    // Fix the knots
    if (fixed_start >= 0)
        for (int kidx = 0; kidx < KNOTS; kidx++)
        {
            if (swTraj->getKnotTime(kidx) <= swTraj->getMinTime() + fixed_start)
            {
                problem.SetParameterBlockConstant(swTraj->getKnotSO3(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotOmg(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotAlp(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotPos(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotVel(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotAcc(kidx).data());
                // printf("Fixed knot %d\n", kidx);
            }
        }
    if (fixed_end >= 0)
        for (int kidx = 0; kidx < KNOTS; kidx++)
        {
            if (swTraj->getKnotTime(kidx) >= swTraj->getMaxTime() - fixed_end)
            {
                problem.SetParameterBlockConstant(swTraj->getKnotSO3(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotOmg(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotAlp(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotPos(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotVel(kidx).data());
                problem.SetParameterBlockConstant(swTraj->getKnotAcc(kidx).data());
                // printf("Fixed knot %d\n", kidx);
            }
        }
}

double maxDiff(const MatrixXd &A, const MatrixXd &B)
{
    return (A - B).cwiseAbs().maxCoeff();
}

void compare(string s, const MatrixXd &A, const MatrixXd &B)
{
    double maxError = (A - B).cwiseAbs().maxCoeff();
    cout << s << maxError << endl;
    assert(maxError < 1e-9);
}

void AddAutodiffIntrzJcbFactor(GaussianProcessPtr &traj, ceres::Problem &problem, FactorMeta &factorMeta)
{
    vector<double *> so3_param;
    vector<double *> rv3_param;
    vector<ceres::ResidualBlockId> res_ids_gp;
    // Add the GP factors based on knot difference
    for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
    {
        // Create the factor
        double gp_loss_thres = -1;
        IntrinsicJcbTestAutodiffFactor *intrzJcb = new IntrinsicJcbTestAutodiffFactor(traj->getGPMixerPtr(), 0.57);
        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<IntrinsicJcbTestAutodiffFactor>(intrzJcb);
        cost_function->SetNumResiduals(RESITRZ_SIZE);
        vector<double *> factor_param_blocks;
        // Add the parameter blocks
        for (int knot_idx = kidx; knot_idx < kidx + 2; knot_idx++)
        {
            so3_param.push_back(traj->getKnotSO3(knot_idx).data());
            rv3_param.push_back(traj->getKnotOmg(knot_idx).data());
            rv3_param.push_back(traj->getKnotAlp(knot_idx).data());
            rv3_param.push_back(traj->getKnotPos(knot_idx).data());
            rv3_param.push_back(traj->getKnotVel(knot_idx).data());
            rv3_param.push_back(traj->getKnotAcc(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAlp(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            cost_function->AddParameterBlock(4);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
            cost_function->AddParameterBlock(3);
        }

        auto res_block = problem.AddResidualBlock(cost_function, NULL, factor_param_blocks);
        res_ids_gp.push_back(res_block);

        break;
    }

    factorMeta.so3_parameter_blocks = so3_param;
    factorMeta.rv3_parameter_blocks = rv3_param;
    factorMeta.residual_blocks = res_ids_gp;
}

void AddAnalyticIntrzJcbFactor(GaussianProcessPtr &traj, ceres::Problem &problem, FactorMeta &factorMeta)
{
    vector<double *> so3_param;
    vector<double *> rv3_param;
    vector<ceres::ResidualBlockId> res_ids_gp;
    // Add GP factors between consecutive knots
    for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
    {
        vector<double *> factor_param_blocks;
        // Add the parameter blocks
        for (int knot_idx = kidx; knot_idx < kidx + 2; knot_idx++)
        {
            so3_param.push_back(traj->getKnotSO3(knot_idx).data());
            rv3_param.push_back(traj->getKnotOmg(knot_idx).data());
            rv3_param.push_back(traj->getKnotAlp(knot_idx).data());
            rv3_param.push_back(traj->getKnotPos(knot_idx).data());
            rv3_param.push_back(traj->getKnotVel(knot_idx).data());
            rv3_param.push_back(traj->getKnotAcc(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAlp(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
        }
        // Create the factors
        double mp_loss_thres = -1;
        // nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
        ceres::CostFunction *cost_function = new IntrinsicJcbTestAnalyticFactor(traj->getGPMixerPtr(), 0.57);
        auto res_block = problem.AddResidualBlock(cost_function, NULL, factor_param_blocks);
        res_ids_gp.push_back(res_block);

        break;
    }

    factorMeta.so3_parameter_blocks = so3_param;
    factorMeta.rv3_parameter_blocks = rv3_param;
    factorMeta.residual_blocks = res_ids_gp;
}

struct TestSE3CostFunctor
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TestSE3CostFunctor() {};

    template <class T>
    bool operator()(const T *const paramTft, // SE3 (7D: qx, qy, qz, qw, tx, ty, tz)
                    const T *const paramTwt, // Vec6 (6D)
                    const T *const paramWrt, T *sResiduals) const
    {
        Eigen::Map<Sophus::SE3<T>  const> const Tft(paramTft);
        Eigen::Map<Matrix<T, 6, 1> const> const Twt(paramTwt);
        Eigen::Map<Matrix<T, 6, 1> const> const Wrt(paramWrt);

        GPState<T> Xt(0, Tft, Twt, Wrt);

        Eigen::Map<Eigen::Matrix<T, 18, 1>> residuals(sResiduals);

        residuals.block(0,  0, 3, 1) = Xt.R.log();
        residuals.block(3,  0, 3, 1) = Xt.O;
        residuals.block(6,  0, 3, 1) = Xt.S;
        residuals.block(9,  0, 3, 1) = Xt.P;
        residuals.block(12, 0, 3, 1) = Xt.V;
        residuals.block(15, 0, 3, 1) = Xt.A;

        return true;
    }
};

void testUnifiedJacobians()
{
    Matrix<double, 3, 1> The_(5.7, 4.3, 9.1); // The = Vec3(0, 1, 0);
    Matrix<double, 3, 1> Rho_(1.9, 0.2, 1.1);
    Matrix<double, 6, 1> Xi_; Xi_ << Rho_, The_;

    // Create an SE3 variable
    SE3d Tft = SE3d::exp(Xi_);
    Matrix<double, 6, 1> Twt;
    Twt << 8.491293058687772, 9.339932477575505, 6.787351548577734, 7.577401305783335, 7.431324681249162, 3.922270195341682;
    Matrix<double, 6, 1> Wrt;
    Wrt << 6.554778901775567, 1.711866878115618, 7.060460880196088, 0.318328463774207, 2.769229849608900, 0.461713906311539;

    // Build the problem.
    ceres::Problem problem;

    // Factormeta
    FactorMeta factorMeta;

    // Add the param block
    problem.AddParameterBlock(Tft.data(), Sophus::SE3d::num_parameters, new Sophus::test::LocalParameterizationSE3);
    problem.AddParameterBlock(Twt.data(), 6, NULL);
    problem.AddParameterBlock(Wrt.data(), 6, NULL);

    factorMeta.se3_parameter_blocks.push_back(Tft.data());
    factorMeta.rv6_parameter_blocks.push_back(Twt.data());
    factorMeta.rv6_parameter_blocks.push_back(Wrt.data());

    // Add the residual block
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<TestSE3CostFunctor, 18, SE3d::num_parameters, 6, 6>(new TestSE3CostFunctor());

    auto res_block = problem.AddResidualBlock(cost_function, NULL, Tft.data(), Twt.data(), Wrt.data());
    factorMeta.residual_blocks.push_back(res_block);

    double cost;
    double time;
    VectorXd residual;
    MatrixXd Jacobian;
    EvaluateFactorRJ(problem, factorMeta, 2, cost, residual, Jacobian, time);

    Vec3 The = residual.block<3, 1>(0,  0);
    SO3d Rt = SO3d::exp(The);
    Vec3 Ot = residual.block<3, 1>(3,  0);
    Vec3 St = residual.block<3, 1>(6,  0);
    Vec3 Pt = residual.block<3, 1>(9,  0);
    Vec3 Vt = residual.block<3, 1>(12, 0);
    Vec3 At = residual.block<3, 1>(15, 0);

    double Un = The.norm();
    Vec3 Ub = The / Un;
    Un = Util::wrapTo180(Un / M_PI * 180) / 180 * M_PI;
    Ub = Ub * Un;

    using MatT = Matrix<double, 6, 3>;
    using MatL = Matrix<double, 3, 6>;

    MatL U; U.setZero(); U.block<3, 3>(0, 0) = Mat3::Identity();
    MatL D; D.setZero(); D.block<3, 3>(0, 3) = Mat3::Identity();

    Mat3 Rtm = Rt.matrix();
    Mat3 RtInvm = Rt.inverse().matrix();

    Mat3 Othat = SO3d::hat(Ot);
    Vec3 Nuy = RtInvm*Vt;
    Vec3 Bta = RtInvm*At - Othat*Nuy;
    Mat3 Nuyhat = SO3d::hat(Nuy);

    Mat3 J_The_Tft = GPMixer::JrInv(The);

    MatL J_Rt_Tft  =  U;
    MatL J_Pt_Tft  =  Rtm*D;

    MatL J_Ot_Twt  =  U;
    Mat3 J_Vt_Rt   = -Rtm*Nuyhat;
    MatL J_Vt_Tft  =  J_Vt_Rt*J_Rt_Tft;
    MatL J_Vt_Twt  =  Rtm*D;

    MatL J_St_Wrt  =  U;
    Mat3 J_At_Rt   = -Rtm*(SO3d::hat(Bta) + SO3d::hat(Othat*Nuy));
    MatL J_At_Tft  =  J_At_Rt*J_Rt_Tft;
    Mat3 J_At_Ot   = -Rtm*Nuyhat;
    MatL J_At_Twt  =  J_At_Ot*J_Ot_Twt + Rtm*Othat*D;
    MatL J_At_Wrt  =  Rtm*D;

    cout << "res:"      << residual.transpose() << endl;
    cout << "J_r_Xt:\n" << Jacobian << endl;

    cout << "J_The_Tft:\n" << J_The_Tft << endl;
    cout << "J_Pt_Tft :\n" << J_Pt_Tft << endl;
    cout << "J_Ot_Twt :\n" << J_Ot_Twt << endl;
    cout << "J_Vt_Tft :\n" << J_Vt_Tft << endl;
    cout << "J_Vt_Twt :\n" << J_Vt_Twt << endl;
    cout << "J_At_Tft :\n" << J_At_Tft << endl;
    cout << "J_At_Twt :\n" << J_At_Twt << endl;
    cout << "J_At_Wrt :\n" << J_At_Wrt << endl;


    // // Set solver options (precision / method)
    // ceres::Solver::Options options;
    // options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
    // options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
    // options.linear_solver_type = ceres::DENSE_QR;

    // // Solve
    // ceres::Solver::Summary summary;
    // ceres::Solve(options, &problem, &summary);
    // std::cout << summary.BriefReport() << std::endl;
}

int main(int argc, char **argv)
{
    double Dt = 0.1102;
    Mat3 CovROSJerk = Vec3(9.4, 4.7, 3.1).asDiagonal();
    Mat3 CovPVAJerk = Vec3(6.3, 6.5, 0.7).asDiagonal();
    GPMixer mySE3GPMixer(Dt, CovROSJerk, CovPVAJerk, POSE_GROUP::SE3, 1e-3, false);

    Vec3 X(4.3, 5.7, 91.0);
    Vec3 Xd(11, 2, 19);

    SO3d R = SO3d::exp(X);

    Mat3 Jr = GPMixer::Jr(X);
    Mat3 JrInv = GPMixer::JrInv(X);

    // Vec3 _X = -X;
    // Mat3 Jr_ = GPMixer::Jl(_X);
    // Mat3 JrInv_ = GPMixer::JlInv(_X);

    // compare("Error Jr      : ", Jr, Jr_);
    // compare("Error JrInv   : ", JrInv, JrInv_);
    compare("Error Jr*JrInv: ", Jr * JrInv, Mat3::Identity(3, 3));

    Vec3 O = GPMixer::Jr(X) * Xd;
    Matrix3d HX_XXd_direct =  mySE3GPMixer.DJrUV_DU(X, Xd);
    Matrix3d HX_XXd_circle = -mySE3GPMixer.Jr(X) * mySE3GPMixer.DJrInvUV_DU(X, O);

    compare("HX_XXd error  : ", HX_XXd_direct, HX_XXd_circle);

    SE3Q<double> myQ;
    SE3Qp<double> myQp;

    Vector3d The(4.3, 5.7, 91.0);
    Vector3d Thed(7.1, 10.1, 10.0);
    Vector3d Thedd(1.5, 0.7, 20.15);

    Vector3d Rho(11.02, 3.04, 26.0);
    Vector3d Rhod(3.3, 60.05, 6.9);
    Vector3d Rhodd(2.0, 0.3, 4.2);

    Vector3d Omg = GPMixer::Jr(The) * Thed;

    TicToc tt_qtime;
    myQ.ComputeQSC(The, Rho, Thed, Rhod);
    tt_qtime.Toc();

    TicToc tt_qptime;
    myQp.ComputeQSC(The, Rho, Thed, Rhod, Omg);
    tt_qptime.Toc();

    printf("tt_qtime  : %f ms\n", tt_qtime.GetLastStop());
    printf("tt_qptime : %f ms\n", tt_qptime.GetLastStop());

    // Confirm that Q = -Exp(-The)*H1(-The, Rho)
    compare("Q  error: ", myQ.Q, Mat3(-SO3d::exp(-The).matrix() * mySE3GPMixer.DJrUV_DU(Vector3d(-The), Rho)));
    // Confirm that Qp = -JrInv*Q*JrInv
    compare("Q' error: ", myQp.Q, -(GPMixer::JrInv(The) * myQ.Q * GPMixer::JrInv(The)));

    Matrix3d Q_, S1_, C11_, C12_, C13_, S2_, C21_, C22_, C23_;

    /* #region  */
    Q_<<
    -0.27606657890206515171271917097329,
    -0.04841190528495784017870562365715,
    0.12325770315319962977529399950072,
    0.06491007847231609895199966331347,
    -0.28379564658289702094862150261179,
    0.03220353738991836578531291479521,
    0.11683202260787924064988629879736,
    0.03289294694641983068938984047236,
    -0.01374133978251581081342358459096;
    S1_<<
    -0.10420562289324271365575924619407,
    -0.12869000662689725200671375660022,
    -2.32312976367504386843165775644593,
    0.09719175238912111658429182625696,
    0.15956808520656370897761178184737,
    2.41398708817864582343304391542915,
    -0.02068253410064482128438712038587,
    -0.01256748627909954653392166079584,
    -0.03797362184689343639343306335832;
    C11_<<
    0.93021696434225409344520585364080,
    1.78412981592265174057843069022056,
    26.84968784906627448094695864710957,
    1.09021381635161129786126821272774,
    1.70894777746216064429063408169895,
    24.64528508888971458645755774341524,
    0.17202529838747851465008409377333,
    -0.47525372243718416731539377906302,
    -2.80130307661426236620627605589107;
    C12_<<
    -0.03147871058691103368643027238249,
    -0.05995514974879118108574971302005,
    -0.95160644539692451626677893727901,
    0.04399108688419393897728326692231,
    0.07779768217335686353308688012476,
    0.98924166357941112348584056235268,
    -0.01612754522900548131980613675296,
    -0.02901172064896630209407391021159,
    -0.01490740933868923458194277742450;
    C13_<<
    0.62421075908884549043875722418306,
    -3.10990564836310401375385481514968,
    0.17072240327124241221490308362263,
    3.13036713850171999240501463646069,
    0.61079303147904506765542009816272,
    -0.19530463169650552623934913754056,
    -0.23403962168482345118647458548367,
    0.11427493105525755867635240292657,
    -0.01460093116982727826091981171430;
    S2_<<
    0.11876213355420794925976224476472,
    -0.00369119382432474534699862189768,
    -0.09669205402277558647483601816930,
    0.00250165625552032202041918829138,
    0.12064793271368141891475289639857,
    -0.09529911795461759593806050361309,
    0.06447284493262382676359578681513,
    0.09787755900559241828418066688755,
    0.00062593062750420540903822930190;
    C21_<<
    -0.03147871058691103368643027238249,
    -0.05995514974879118108574971302005,
    -0.95160644539692451626677893727901,
    0.04399108688419393897728326692231,
    0.07779768217335686353308688012476,
    0.98924166357941112348584056235268,
    -0.01612754522900548131980613675296,
    -0.02901172064896630209407391021159,
    -0.01490740933868923458194277742450;
    C22_<<
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000;
    C23_<<
    -0.11300216006672469948313164422871,
    0.00914570269991075296411864314905,
    0.02130274180256805025002186937400,
    0.05676738965062743530465283470221,
    -0.03430025194173392383278908823740,
    0.65389839893049828134508061339147,
    0.04672620417746253806967615673784,
    0.65473164041046005845458921612590,
    -0.08499188899123681639746763494259;
    /* #endregion */

    compare("Q    numerical error: ", myQ.Q  , Q_  );
    compare("S1   numerical error: ", myQ.S1 , S1_ );
    compare("C11  numerical error: ", myQ.C11, C11_);
    compare("C12  numerical error: ", myQ.C12, C12_);
    compare("C13  numerical error: ", myQ.C13, C13_);
    compare("S2   numerical error: ", myQ.S2 , S2_ );
    compare("C11  numerical error: ", myQ.C11, C11_);
    compare("C12  numerical error: ", myQ.C12, C12_);
    compare("C13  numerical error: ", myQ.C13, C13_);

    Matrix3d Qp_, Sp1_, Cp11_, Cp12_, Cp13_, Sp2_, Cp21_, Cp22_, Cp23_;

    /* #region  */
    Qp_<<
    -612.02081473804355482570827007293701,
    -11.15875168041248954864386178087443,
    30.85758226454096586621744791045785,
    14.84124831958751045135613821912557,
    -611.03080504751687840325757861137390,
    32.75155188362483471564701176248491,
    27.81758226454096316615505202207714,
    43.77155188362483784203504910692573,
    -3.81279170105399023427139582054224;
    Sp1_<<
    73.39070156778583964296558406203985,
    0.54142539261656974503011952037923,
    10.58440348954987619833900680532679,
    0.94226685812990007562461869383696,
    73.92327245304497296274348627775908,
    9.95246662411237714707112900214270,
    -2.33744956271806891834330599522218,
    -5.73379746327464623334435600554571,
    -0.69142956879064099151577238444588;
    Cp11_<<
    45.77333863973847627448776620440185,
    -36.65303911686708460138106602244079,
    -598.84617063737869102624244987964630,
    32.30568463458443062563674175180495,
    111.19922081656719115017040167003870,
    613.10778811629165829799603670835495,
    -20.12645736850905819892432191409171,
    -22.58256014507567144278255000244826,
    -12.72211659922898974173222086392343;
    Cp12_<<
    30.34190198654745884709882375318557,
    0.22762584599059751400140783061943,
    11.43564115494773680836715357145295,
    1.63804810269898770691554545919644,
    32.22731327278113866441344725899398,
    28.33913424110010481626886758022010,
    -1.09270605465602899109001100441674,
    -2.60249986777981190400055311329197,
    -2.36416621699686757551717164460570;
    Cp13_<<
    -740.10960400373289758135797455906868,
    9.85048403738305289323307079030201,
    105.60191234461105125319591024890542,
    9.85048403738305289323307079030201,
    -737.24689677762535211513750255107880,
    112.84216398173451523234689375385642,
    105.60191234461105125319591024890542,
    112.84216398173451523234689375385642,
    -19.72279557830668750284530688077211;
    Sp2_<<
    0.38505995833400119554568163948716,
    5.17131526429971888347836284083314,
    -4.71935719217318450091624981723726,
    -5.27686001514027847036913954070769,
    0.82468335518477076107046741526574,
    4.02565256495616452525609929580241,
    0.24209207430390178306112147765816,
    -0.40127505179751982167246637800417,
    -0.01924134699306674287089258257311;
    Cp21_<<
    -0.03147871058691103368643027238249,
    -0.05995514974879118108574971302005,
    -0.95160644539692451626677893727901,
    0.04399108688419393897728326692231,
    0.07779768217335686353308688012476,
    0.98924166357941112348584056235268,
    -0.01612754522900548131980613675296,
    -0.02901172064896630209407391021159,
    -0.01490740933868923458194277742450;
    Cp22_<<
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000;
    Cp23_<<
    -247.85823676870697340746119152754545,
    -2.55745770312542219926399411633611,
    41.83353235761325095154461450874805,
    4.34254229687457726782895406358875,
    -247.05144848320588835122180171310902,
    17.02473773260695821818444528616965,
    -18.21646764238674620628444245085120,
    20.32473773260695892872718104626983,
    -1.93801260852489187769265299721155;
    /* #endregion */

    compare("Qp   numerical error: ", myQp.Q  , Qp_  );
    compare("Sp1  numerical error: ", myQp.S1 , Sp1_ );
    compare("Cp11 numerical error: ", myQp.C11, Cp11_);
    compare("Cp12 numerical error: ", myQp.C12, Cp12_);
    compare("Cp13 numerical error: ", myQp.C13, Cp13_);
    compare("Sp2  numerical error: ", myQp.S2 , Sp2_ );
    compare("Cp11 numerical error: ", myQp.C11, Cp11_);
    compare("Cp12 numerical error: ", myQp.C12, Cp12_);
    compare("Cp13 numerical error: ", myQp.C13, Cp13_);

    // Create the H and H' matrices of SE3
    Matrix<double, 6, 6> SE3H; SE3H.setZero();
    SE3H.block<3, 3>(0, 0) = mySE3GPMixer.DJrUV_DU(The, Thed);
    SE3H.block<3, 3>(3, 0) = myQ.S1 + mySE3GPMixer.DJrUV_DU(The, Rhod);
    SE3H.block<3, 3>(3, 3) = myQ.S2;

    Matrix<double, 6, 6> SE3H_;

    /* #region  */
    SE3H_<<
    0.11876213355420794925976224476472,
    -0.00369119382432474534699862189768,
    -0.09669205402277558647483601816930,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00250165625552032202041918829138,
    0.12064793271368141891475289639857,
    -0.09529911795461759593806050361309,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.06447284493262382676359578681513,
    0.09787755900559241828418066688755,
    0.00062593062750420540903822930190,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00000000000000000000000000000000,
    0.00729734407524723713983627604307,
    -0.11064588872881084280486163606838,
    -2.48664104388213580421052029123530,
    0.11876213355420794925976224476472,
    -0.00369119382432474534699862189768,
    -0.09669205402277558647483601816930,
    0.07060639254901469874514674529564,
    0.27867591707033195769227518212574,
    1.77265485534088540475750050973147,
    0.00250165625552032202041918829138,
    0.12064793271368141891475289639857,
    -0.09529911795461759593806050361309,
    -0.00775017680108034279912176955918,
    0.63355161584749275505146215436980,
    -0.03184918929561989658916232315278,
    0.06447284493262382676359578681513,
    0.09787755900559241828418066688755,
    0.00062593062750420540903822930190;
    /* #endregion */

    compare("SEH3 numerical error: ", SE3H, SE3H_);

    Matrix<double, 6, 1> Xi; Xi << The, Rho;
    Matrix<double, 6, 1> Xid; Xid << Thed, Rhod;
    Matrix<double, 6, 1> Xidd; Xidd << Thedd, Rhodd;

    Matrix<double, 6, 1> &Xid0 = Xi;
    Matrix<double, 6, 1> &Xid1 = Xid;
    Matrix<double, 6, 1> &Xid2 = Xidd;

    Matrix<double, 6, 1> Tau = GPMixer::Jr(Xi)*Xid;
    Matrix<double, 6, 1> Wrn = GPMixer::Jr(Xi)*Xidd + SE3H*Xid;

    Matrix<double, 6, 1> Tau_, Wrn_;

    /* #region  */
    Tau_<<
    0.70803449619167235784544800480944,
    0.52132104746770180359050073093385,
    10.90202111753637304047970246756449,
    0.57877335681299157954526890534908,
    -1.58678174153978579496993006614503,
    11.72568159674946386417104804422706;
    Wrn_<<
    0.77749061080929748879242424663971,
    1.53406182339223273736195096716983,
    21.59461581651028438955108867958188,
    -24.19341526664975461358153552282602,
    28.41509436231157081920173368416727,
    16.33084332327182153221656335517764;
    /* #endregion */

    compare("Tau numerical error: ", Tau, Tau_);
    compare("Wrn numerical error: ", Wrn, Wrn_);

    Vector3d Nuy = Tau.block<3, 1>(3, 0);

    Matrix<double, 6, 6> SE3Hp;
    SE3Hp.setZero();
    SE3Hp.block<3, 3>(0, 0) = mySE3GPMixer.DJrInvUV_DU(The, Omg);
    SE3Hp.block<3, 3>(3, 0) = myQp.S1 + mySE3GPMixer.DJrInvUV_DU(The, Nuy);
    SE3Hp.block<3, 3>(3, 3) = myQp.S2;

    Matrix<double, 6, 6> JrXi = GPMixer::Jr(Xi);
    Matrix<double, 6, 6> JrInvXi = GPMixer::JrInv(Xi);
    compare("SE3Jr error: ", JrXi * JrInvXi, Matrix<double, 6, 6>::Identity(6, 6));

    Matrix<double, 6, 1> Xid1_reverse = JrInvXi * Tau;
    Matrix<double, 6, 1> Xid2_reverse = JrInvXi * Wrn + SE3Hp * Xid1;

    compare("Xd1_reverse error: ", Xid1_reverse, Xid1);
    compare("Xd2_reverse error: ", Xid2_reverse, Xid2);

    Mat3 S1, S2;
    S1.setZero();
    S2.setZero();

    SE3Q<double> myQ_;
    TicToc tt_s;
    myQ_.ComputeS(The, Rho, Thed);
    tt_s.Toc();

    compare("S1 numerical error: ", myQ.S1, myQ_.S1);
    compare("S2 numerical error: ", myQ.S2, myQ_.S2);

    Mat3 Sp1, Sp2;
    Sp1.setZero();
    Sp2.setZero();

    SE3Qp<double> myQp_;
    TicToc tt_sp;
    myQp_.ComputeS(The, Rho, Omg);
    tt_sp.Toc();

    compare("Sp1 numerical error: ", myQp.S1, myQp.S1);
    compare("Sp2 numerical error: ", myQp.S2, myQp.S2);

    // Compare our SE3Jr SE3JrInv with sophus implemenation
    {
        int count = 1;

        Matrix<double, 6, 1> Xi;
        Xi << The, Rho;
        Matrix<double, 6, 1> Xiuo;
        Xiuo << Rho, The;

        double tt_sophusJr_avr = 0;
        count = 1;
        Matrix<double, 6, 6> JrXiuo;
        while (count <= 20)
        {
            TicToc tt_sophusJr;
            JrXiuo = SE3d::leftJacobian(-Xiuo);
            tt_sophusJr_avr += tt_sophusJr.Toc();

            count++;
        }
        tt_sophusJr_avr /= 20;

        double tt_myJr_avr = 0;
        count = 1;
        Matrix<double, 6, 6> JrXi;
        while (count <= 20)
        {
            TicToc tt_myJr;
            JrXi = GPMixer::Jr(Xi);
            tt_myJr_avr += tt_myJr.Toc();

            count++;
        }
        tt_myJr_avr /= 20;

        cout << "Sophus right Jacobian time: " << tt_sophusJr_avr << endl;
        cout << "XXX right Jacobian time   : " << tt_myJr_avr << endl;

        compare("JrXiuo 11 block error: ", JrXiuo.block(0, 0, 3, 3), JrXi.block(0, 0, 3, 3));
        compare("JrXiuo 12 block error: ", JrXiuo.block(0, 3, 3, 3), JrXi.block(3, 0, 3, 3));
        compare("JrXiuo 21 block error: ", JrXiuo.block(3, 0, 3, 3), JrXi.block(0, 3, 3, 3));
        compare("JrXiuo 22 block error: ", JrXiuo.block(3, 3, 3, 3), JrXi.block(3, 3, 3, 3));
    }

    testUnifiedJacobians();
    // exit(0);

    GPState Xa(0.213,
        SO3d::exp(Vec3(5.7, 4.3, 9.1)),
        Vec3(9.6489, 1.5761, 9.7059),
        Vec3(9.1338, 6.3236, 0.9754),
        Vec3(2.7850, 5.4688, 9.5751),
        Vec3(1.4189, 4.2176, 9.1574),
        Vec3(7.9221, 9.5949, 6.5574));

    GPState Xb(Xa.t + Dt,
        SO3d::exp(Vec3(9.3399, 8.4913, 0.3571)),
        Vec3(3.9223, 2.7692, 3.1710),
        Vec3(6.5548, 0.4617, 9.5022),
        Vec3(1.7119, 0.9713, 0.3445),
        Vec3(7.0605, 8.2346, 4.3874),
        Vec3(0.3183, 6.9483, 3.8156));

    double ts = 0.57 * Dt;
    GPState Xt(Xa.t + ts);
    vector<vector<Matrix3d>> DXt_DXa;
    vector<vector<Matrix3d>> DXt_DXb;

    // Interpolate and find Jacobian
    TicToc tt_split;
    mySE3GPMixer.ComputeXtAndJacobiansSO3xR3(Xa, Xb, Xt, DXt_DXa, DXt_DXb);
    tt_split.Toc();

    TicToc tt_se3;
    mySE3GPMixer.ComputeXtAndJacobiansSE3(Xa, Xb, Xt, DXt_DXa, DXt_DXb);
    tt_se3.Toc();

    GPMixer mySE3GPMixerApr(Dt, CovROSJerk, CovPVAJerk, POSE_GROUP::SE3, 1e-3, true);

    // Interpolate and find Jacobian
    TicToc tt_splitap;
    mySE3GPMixerApr.ComputeXtAndJacobiansSO3xR3(Xa, Xb, Xt, DXt_DXa, DXt_DXb);
    tt_splitap.Toc();

    TicToc tt_se3ap;
    mySE3GPMixerApr.ComputeXtAndJacobiansSE3(Xa, Xb, Xt, DXt_DXa, DXt_DXb);
    tt_se3ap.Toc();

    printf("tt_split   : %f ms\n", tt_split.GetLastStop());
    printf("tt_split ap: %f ms\n", tt_splitap.GetLastStop());
    printf("tt_se3     : %f ms\n", tt_se3.GetLastStop());
    printf("tt_se3 ap  : %f ms\n", tt_se3ap.GetLastStop());

    // POSE_GROUP pr = POSE_GROUP::SO3xR3;
    POSE_GROUP pr = POSE_GROUP::SE3;
    GaussianProcessPtr traj(new GaussianProcess(Dt, CovROSJerk, CovPVAJerk, false, POSE_GROUP::SE3));
    traj->setStartTime(Xa.t);
    traj->genRandomTrajectory(2);
    traj->setKnot(0, Xa);
    traj->setKnot(1, Xb);

    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    // Create the ceres problem
    CreateCeresProblem(problem, options, summary, traj, -0.1, -0.1);

    auto EvaluateFactorClass = [](ceres::Problem &problem, FactorMeta &factors, int local_pamaterization_type, VectorXd &res, MatrixXd &Jcb, double &cost, double &total_time, double &eval_time) -> void
    {
        TicToc tt_autodiff;

        eval_time = 0;

        res.setZero();
        Jcb.setZero();

        int count = 1;
        while (count-- > 0)
        {
            double eval_time_ = 0;
            EvaluateFactorRJ(problem, factors, local_pamaterization_type, cost, res, Jcb, eval_time_);
            eval_time += eval_time_;
        }

        // Remove sidual block
        RemoveResidualBlock(problem, factors);

        total_time = tt_autodiff.Toc();
    };

    // Motion priors
    {
        double cost_autodiff;
        double time_autodiff;
        double time_autodiff_eval;
        VectorXd residual_autodiff;
        MatrixXd Jacobian_autodiff;
        {
            // Test the autodiff Jacobian
            FactorMeta intrzJcbFactorMetaAutodiff;
            AddAutodiffIntrzJcbFactor(traj, problem, intrzJcbFactorMetaAutodiff);
            // if (intrzJcbFactorMetaAutodiff.parameter_blocks() == 0)
            //     return;

            EvaluateFactorClass(problem, intrzJcbFactorMetaAutodiff, 0, residual_autodiff, Jacobian_autodiff, cost_autodiff, time_autodiff, time_autodiff_eval);
            printf(KCYN "Intrinsic Jacobian Autodiff Jacobian: Size %2d x %2d. Params: %d. Cost: %f. Time (ms): %f, %f\n",
                   Jacobian_autodiff.rows(), Jacobian_autodiff.cols(),
                   intrzJcbFactorMetaAutodiff.parameter_blocks(),
                   cost_autodiff, time_autodiff, time_autodiff_eval);
        }

        double cost_analytic;
        double time_analytic;
        double time_analytic_eval;
        VectorXd residual_analytic;
        MatrixXd Jacobian_analytic;
        {
            // Test the analytic Jacobian
            FactorMeta intrzJcbFactorMetaAnalytic;
            AddAnalyticIntrzJcbFactor(traj, problem, intrzJcbFactorMetaAnalytic);
            // if (intrzJcbFactorMetaAnalytic.parameter_blocks() == 0)
            //     return;

            EvaluateFactorClass(problem, intrzJcbFactorMetaAnalytic, 1, residual_analytic, Jacobian_analytic, cost_analytic, time_analytic, time_analytic_eval);
            printf(KCYN "Intrinsic Jacobian Analytic Jacobian: Size %2d x %2d. Params: %d. Cost: %f. Time (ms): %f, %f\n",
                   Jacobian_analytic.rows(), Jacobian_analytic.cols(),
                   intrzJcbFactorMetaAnalytic.parameter_blocks(),
                   cost_analytic, time_analytic, time_analytic_eval);
        }

        printf(KGRN "CIDX: %d. Intrinsic Jacobian Factor. Residual max error: %.4f. Jacobian max error: %.4f. Time (ms): %.3f, %.3f. Ratio: %.0f\%\n\n" RESET,
               0, maxDiff(residual_autodiff, residual_analytic), maxDiff(Jacobian_autodiff, Jacobian_analytic),
               time_autodiff, time_analytic,
               time_autodiff_eval / time_analytic_eval * 100);

        cout << "residual_autodiff: " << residual_autodiff.transpose() << endl;
        cout << "residual_analytic: " << residual_analytic.transpose() << endl;
        cout << "residual_diff    : " << (residual_autodiff - residual_analytic).transpose() << endl;

        // cout << KBLU "Jacobian Tft Xa analytic:\n" RESET << Jdebug["J_Tft_Xa"] << endl;
        // cout << KBLU "Jacobian Tft Xa_autodiff:\n" RESET << Jacobian_autodiff.block(0, 0, 6, 18) << endl;
        // cout << KYEL "Jacobian Tft Xa diff:\n"     RESET << thresholdMatrix(Jacobian_autodiff.block(0, 0, 6, 18) - Jdebug["J_Tft_Xa"], 1e-9) << endl;
        // cout << KBLU "Jacobian Twt Xa analytic:\n" RESET << Jdebug["J_Twt_Xa"] << endl;
        // cout << KBLU "Jacobian Twt Xa_autodiff:\n" RESET << Jacobian_autodiff.block(6, 0, 6, 18) << endl;
        // cout << KYEL "Jacobian Twt Xa diff:\n"     RESET << thresholdMatrix(Jacobian_autodiff.block(6, 0, 6, 18) - Jdebug["J_Twt_Xa"], 1e-9) << endl;
        // cout << KBLU "Jacobian Wrt Xa analytic:\n" RESET << Jdebug["J_Wrt_Xa"] << endl;
        // cout << KBLU "Jacobian Wrt Xa_autodiff:\n" RESET << Jacobian_autodiff.block(12, 0, 6, 18) << endl;
        // cout << KYEL "Jacobian Wrt Xa diff:\n"     RESET << thresholdMatrix(Jacobian_autodiff.block(12, 0, 6, 18) - Jdebug["J_Wrt_Xa"], 1e-9) << endl;

        // cout << endl;

        // cout << KBLU "Jacobian Tft Xb analytic:\n" RESET << Jdebug["J_Tft_Xb"] << endl;
        // cout << KBLU "Jacobian Tft Xb_autodiff:\n" RESET << Jacobian_autodiff.block(0, 18, 6, 18) << endl;
        // cout << KYEL "Jacobian Tft Xb diff:\n"     RESET << thresholdMatrix(Jacobian_autodiff.block(0, 18, 6, 18) - Jdebug["J_Tft_Xb"], 1e-9) << endl;
        // cout << KBLU "Jacobian Twt Xb analytic:\n" RESET << Jdebug["J_Twt_Xb"] << endl;
        // cout << KBLU "Jacobian Twt Xb_autodiff:\n" RESET << Jacobian_autodiff.block(6, 18, 6, 18) << endl;
        // cout << KYEL "Jacobian Twt Xb diff:\n"     RESET << thresholdMatrix(Jacobian_autodiff.block(6, 18, 6, 18) - Jdebug["J_Twt_Xb"], 1e-9) << endl;
        // cout << KBLU "Jacobian Wrt Xb analytic:\n" RESET << Jdebug["J_Wrt_Xb"] << endl;
        // cout << KBLU "Jacobian Wrt Xb_autodiff:\n" RESET << Jacobian_autodiff.block(12, 18, 6, 18) << endl;
        // cout << KYEL "Jacobian Wrt Xb diff:\n"     RESET << thresholdMatrix(Jacobian_autodiff.block(12, 18, 6, 18) - Jdebug["J_Wrt_Xb"], 1e-9) << endl;

        cout << KBLU "Jacobian Xa analytic:\n" RESET << Jacobian_analytic.block(0, 0, 18, 18) << endl;
        cout << KBLU "Jacobian Xa_autodiff:\n" RESET << Jacobian_autodiff.block(0, 0, 18, 18) << endl;
        cout << KYEL "Jacobian Xa diff:\n" RESET << thresholdMatrix(Jacobian_autodiff.block(0, 0, 18, 18) - Jacobian_analytic.block(0, 0, 18, 18), 1e-9) << endl;
        // cout << KYEL "J_At_Ra diff:\n" RESET << Jdebug["J_At_Ra"] << endl;
        // cout << KYEL "J_At_Tft*J_Tft_Ra diff:\n" RESET << Jdebug["J_At_Tft*J_Tft_Ra"] << endl;
        // cout << KYEL "J_At_Twt*J_Twt_Ra diff:\n" RESET << Jdebug["J_At_Twt*J_Twt_Ra"] << endl;
        // cout << KYEL "J_At_Wrt*J_Wrt_Ra diff:\n" RESET << Jdebug["J_At_Wrt*J_Wrt_Ra"] << endl;

        cout << endl;

        cout << KBLU "Jacobian Xb analytic:\n" RESET << Jacobian_analytic.block(0, 18, 18, 18) << endl;
        cout << KBLU "Jacobian Xb_autodiff:\n" RESET << Jacobian_autodiff.block(0, 18, 18, 18) << endl;
        cout << KYEL "Jacobian Xb diff:\n" RESET << thresholdMatrix(Jacobian_autodiff.block(0, 18, 18, 18) - Jacobian_analytic.block(0, 18, 18, 18), 1e-9) << endl;
    }

}