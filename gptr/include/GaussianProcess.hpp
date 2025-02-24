#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <deque>

#include <algorithm>    // Include this header for std::max
#include <Eigen/Dense>

// Sophus
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>

// Include the library for Q matrix
#include <SE3JQ.hpp>

// Ceres parameterization
#include <ceres/ceres.h>

typedef Sophus::SO3<double> SO3d;
typedef Sophus::SE3<double> SE3d;
typedef Eigen::Vector3d Vec3;
typedef Eigen::Matrix3d Mat3;
typedef Eigen::Quaterniond Quaternd;

using namespace std;
using namespace Eigen;

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
#define RESET "\033[0m"

// Threshold to use approximation to avoid numerical issues
#define DOUBLE_EPSILON 1e-4

template <typename Derived>
Eigen::MatrixXd CastJetToDouble(const MatrixBase<Derived> &M_)
{
    MatrixXd M(M_.rows(), M_.cols());
    for(int ridx = 0; ridx < M_.rows(); ridx++)
        for(int cidx = 0; cidx < M_.cols(); cidx++)
        {
            if constexpr (!std::is_same_v<typename Derived::Scalar, double>)
                M(ridx, cidx) = M_(ridx, cidx).a;
            else
                M(ridx, cidx) = M_(ridx, cidx);
        }
    return M;
}

/* #region Define the states for convenience in initialization and copying ------------------------------------------*/

#define STATE_DIM 18
template <class T = double>
class GPState
{
public:
    using SO3T  = Sophus::SO3<T>;
    using SE3T  = Sophus::SE3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Vec6T = Eigen::Matrix<T, 6, 1>;
    using Mat3T = Eigen::Matrix<T, 3, 3>;
    using Mat6T = Eigen::Matrix<T, 6, 6>;

    double t;
    SO3T  R;
    Vec3T O;
    Vec3T S;
    Vec3T P;
    Vec3T V;
    Vec3T A;

    // Destructor
    ~GPState(){};
    
    // Constructor
    GPState()
        : t(0), R(SO3T()), O(Vec3T(0, 0, 0)), S(Vec3T(0, 0, 0)), P(Vec3T(0, 0, 0)), V(Vec3T(0, 0, 0)), A(Vec3T(0, 0, 0)) {}
    
    GPState(double t_)
        : t(t_), R(SO3T()), O(Vec3T()), S(Vec3T()), P(Vec3T()), V(Vec3T()), A(Vec3T()) {}

    GPState(double t_, const SE3d &pose)
        : t(t_), R(pose.so3().cast<T>()), O(Vec3T(0, 0, 0)), S(Vec3T(0, 0, 0)), P(pose.translation().cast<T>()), V(Vec3T(0, 0, 0)), A(Vec3T(0, 0, 0)) {}

    GPState(double t_, const SO3d &R_, const Vec3 &O_, const Vec3 &S_, const Vec3 &P_, const Vec3 &V_, const Vec3 &A_)
        : t(t_), R(R_.cast<T>()), O(O_.cast<T>()), S(S_.cast<T>()), P(P_.cast<T>()), V(V_.cast<T>()), A(A_.cast<T>()) {}

    GPState(const GPState<T> &other)
        : t(other.t), R(other.R), O(other.O), S(other.S), P(other.P), V(other.V), A(other.A) {}

    GPState(double t_, const GPState<T> &other)
        : t(t_), R(other.R), O(other.O), S(other.S), P(other.P), V(other.V), A(other.A) {}
    
    GPState(double t_, const SE3T &Tf, const Vec6T &Tw, const Vec6T &Wr)
    {
        t = t_;

        R = Tf.so3();
        O = Tw.template head<3>();
        S = Wr.template head<3>();

        Vec3T N = Tw.template tail<3>();
        Vec3T B = Wr.template tail<3>();

        P = Tf.translation();
        V = R*N;
        A = R*B + R*(SO3T::hat(O)*N);
    }

    GPState &operator=(const GPState &Xother)
    {
        this->t = Xother.t;
        this->R = Xother.R;
        this->O = Xother.O;
        this->S = Xother.S;
        this->P = Xother.P;
        this->V = Xother.V;
        this->A = Xother.A;
        return *this;
    }

    void GetTUW(SE3T &Tf, Vec6T &Tw, Vec6T &Wr) const
    {
        Tf = SE3T(R, P);

        SO3T Rinv = R.inverse();
        Vec3T N = Rinv*V;
        Vec3T B = Rinv*A - SO3T::hat(O)*N;

        Tw << O, N;
        Wr << S, B;
    }

    Matrix<double, STATE_DIM, 1> boxminus(const GPState &Xother) const
    {
        Matrix<double, STATE_DIM, 1> dX;
        dX << (Xother.R.inverse()*R).log(),
               O - Xother.O,
               S - Xother.S,
               P - Xother.P,
               V - Xother.V,
               A - Xother.A;
        return dX;
    }

    double yaw()
    {
        Eigen::Vector3d n = R.matrix().col(0);
        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        return y / M_PI * 180.0;
    }

    double pitch()
    {
        Eigen::Vector3d n = R.matrix().col(0);
        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        return p / M_PI * 180.0;
    }

    double roll()
    {
        Eigen::Vector3d n = R.matrix().col(0);
        Eigen::Vector3d o = R.matrix().col(1);
        Eigen::Vector3d a = R.matrix().col(2);
        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        return r / M_PI * 180.0;
    }
};

/* #endregion Define the states for convenience in initialization and copying ---------------------------------------*/


/* #region Utility for propagation and interpolation matrices, elementary jacobians dXt/dXk, J_r, H_r, Hprime_r.. ---*/

enum class POSE_GROUP
{
    SO3xR3, SE3
};

class GPMixer
{
private:

    // Knot length
    double Dt = 0.0;

    // Identity matrix
    const Mat3 Eye = Mat3::Identity();

    // 3x3 Zero matrix
    const Mat3 Zr3 = Mat3::Zero();

    // Covariance of so3 GP jerk
    Mat3 CovROSJerk = Eye;

    // Covariance of R3 GP jerk
    Mat3 CovPVAJerk = Eye;

    // Covariance of SE3 Jerk
    Matrix<double, 6, 6> CovTTWJerk = Matrix<double, 6, 6>::Identity(6, 6);

    // Transition matrix
    Eigen::SparseMatrix<double> Fmat;

    // Square root of GP convariance, needed for the motion prior factor
    Eigen::SparseMatrix<double> sqrtW = Eigen::SparseMatrix<double>(STATE_DIM, STATE_DIM);

    // Type of pose representation
    POSE_GROUP pose_representation = POSE_GROUP::SO3xR3;

    // Tolerance to use SE3
    double lie_epsilon = 1e-3;

    // Use / not use closed form derivatives
    bool use_approx_drv = false;

public:

//     // Destructor
//    ~GPMixer() {};

    // Constructor
    GPMixer(double Dt_, const Mat3 CovROSJerk_, const Mat3 CovPVAJerk_,
            const POSE_GROUP pose_representation_ = POSE_GROUP::SO3xR3,
            double lie_epsilon_ = 1e-3, bool use_approx_drv_ = false)
        : Dt(Dt_), CovROSJerk(CovROSJerk_), CovPVAJerk(CovPVAJerk_),
          pose_representation(pose_representation_),
          lie_epsilon(lie_epsilon_), use_approx_drv(use_approx_drv_)
    {
        CovTTWJerk.block<3, 3>(0, 0) = CovROSJerk;
        CovTTWJerk.block<3, 3>(3, 3) = CovPVAJerk;

        // Calculate the transition matrix
        if(pose_representation == POSE_GROUP::SO3xR3)
            Fmat = kron(Fbase(Dt, 3), Eye).sparseView();
        else if(pose_representation == POSE_GROUP::SE3)
            Fmat = kron(Fbase(Dt, 3), Matrix<double, 6, 6>::Identity(6, 6)).sparseView();
        Fmat.makeCompressed();
        
        // Calculate the information matrix for motion prior
        Matrix3d Qtilde = Qbase(Dt, 3);
        Matrix<double, STATE_DIM, STATE_DIM> Info; Info.setZero();
        if(pose_representation == POSE_GROUP::SO3xR3)
        {
            Info.block<9, 9>(0, 0) = kron(Qtilde, getCovROSJerk());
            Info.block<9, 9>(9, 9) = kron(Qtilde, getCovPVAJerk());
        }
        else if(pose_representation == POSE_GROUP::SE3)
            Info = kron(Qtilde, CovTTWJerk);

        sqrtW = MatrixXd(Eigen::LLT<Matrix<double, STATE_DIM, STATE_DIM>>(Info.inverse()).matrixL().transpose()).sparseView();
        sqrtW.makeCompressed();
    };

    Matrix3d   getCovROSJerk()         const { return CovROSJerk;          }
    Matrix3d   getCovPVAJerk()         const { return CovPVAJerk;          }
    Matrix3d   getCovTTWJerk()         const { return CovPVAJerk;          }
    double     getDt()                 const { return Dt;                  }
    POSE_GROUP getPoseRepresentation() const { return pose_representation; }
    double     getEps()                const { return lie_epsilon;         }
    bool       getJacobianForm()       const { return use_approx_drv;     }

    template <typename MatrixType1, typename MatrixType2>
    MatrixXd kron(const MatrixType1& A, const MatrixType2& B) const
    {
        MatrixXd result(A.rows() * B.rows(), A.cols() * B.cols());
        for (int i = 0; i < A.rows(); ++i)
            for (int j = 0; j < A.cols(); ++j)
                result.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;

        return result;
    }

    void setCovROSJerk(const Mat3 &m)
    {
        CovROSJerk = m;
    }

    void setCovPVAJerk(const Mat3 &m)
    {
        CovPVAJerk = m;
    }

    // Transition Matrix, PHI(tau, 0)
    MatrixXd Fbase(const double Dtau, int N) const
    {
        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};

        MatrixXd Phi = MatrixXd::Identity(N, N);
        for(int n = 0; n < N; n++)
            for(int m = n + 1; m < N; m++)
                Phi(n, m) = pow(Dtau, m-n)/factorial(m-n);

        return Phi;
    }

    // Gaussian Process covariance, Q = \int{Phi*F*CovPVAJerk*F'*Phi'}
    MatrixXd Qbase(const double Dtau, int N) const 
    {
        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};
        
        MatrixXd Q(N, N);
        for(int n = 0; n < N; n++)
            for(int m = 0; m < N; m++)
                Q(n, m) = pow(Dtau, 2*N-1-n-m)/double(2*N-1-n-m)/double(factorial(N-1-n))/double(factorial(N-1-m));
        // cout << "MyQ: " << Q << endl;
        return Q;
    }

    MatrixXd Qga(const double s, int N) const 
    {
        double Dtau = s*Dt;

        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};
        
        MatrixXd Q(N, N);
        for(int n = 0; n < N; n++)
            for(int m = 0; m < N; m++)
                Q(n, m) = pow(Dtau, 2*N-1-n-m)/double(2*N-1-n-m)/double(factorial(N-1-n))/double(factorial(N-1-m));

        return kron(Qbase(Dt, 3), CovROSJerk);
    }

    MatrixXd Qnu(const double s, int N) const 
    {
        double Dtau = s*Dt;

        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};
        
        MatrixXd Q(N, N);
        for(int n = 0; n < N; n++)
            for(int m = 0; m < N; m++)
                Q(n, m) = pow(Dtau, 2*N-1-n-m)/double(2*N-1-n-m)/double(factorial(N-1-n))/double(factorial(N-1-m));

        return kron(Qbase(Dt, 3), CovPVAJerk);
    }

    Matrix<double, STATE_DIM, STATE_DIM> PropagateFullCov(Matrix<double, STATE_DIM, STATE_DIM> P0) const
    {
        Matrix<double, STATE_DIM, STATE_DIM> F; F.setZero();
        Matrix<double, STATE_DIM, STATE_DIM> Q; Q.setZero();
        
        F.block<9, 9>(0, 0) = kron(Fbase(Dt, 3), Eye);
        F.block<9, 9>(9, 9) = kron(Fbase(Dt, 3), Eye);

        Q.block<9, 9>(0, 0) = kron(Qbase(Dt, 3), CovROSJerk);
        Q.block<9, 9>(9, 9) = kron(Qbase(Dt, 3), CovPVAJerk);

        return F*P0*F.transpose() + Q;
    }

    MatrixXd PSI(const double Dtau, const MatrixXd &Q) const
    {
        MatrixXd Eye = MatrixXd::Identity(Q.rows(), Q.rows());

        if (Dtau < DOUBLE_EPSILON)
            return kron(MatrixXd::Zero(3, 3), Eye);

        MatrixXd PhiDtaubar = kron(Fbase(Dt - Dtau, 3), Eye);
        MatrixXd QDtau = kron(Qbase(Dtau, 3), Q);
        MatrixXd QDt = kron(Qbase(Dt, 3), Q);

        return QDtau*PhiDtaubar.transpose()*QDt.inverse();
    }

    MatrixXd PSI_ROS(const double Dtau) const
    {
        return PSI(Dtau, CovROSJerk);
    }

    MatrixXd PSI_PVA(const double Dtau) const
    {
        return PSI(Dtau, CovPVAJerk);
    }

    MatrixXd LMD(const double Dtau, const MatrixXd &Q) const
    {
        MatrixXd Eye = MatrixXd::Identity(Q.rows(), Q.rows());

        MatrixXd PSIDtau = PSI(Dtau, Q);
        MatrixXd FDtau = kron(Fbase(Dtau, 3), Eye);
        MatrixXd FDt = kron(Fbase(Dt, 3), Eye);
        
        return FDtau - PSIDtau*FDt;
    }

    MatrixXd LMD_ROS(const double Dtau) const
    {
        return LMD(Dtau, CovROSJerk);
    }

    MatrixXd LMD_PVA(const double Dtau) const
    {
        return LMD(Dtau, CovPVAJerk);
    }

    template <class T = double>
    static Eigen::Matrix<T, 3, 3> hatSquare(const Eigen::Matrix<T, 3, 1> &The)
    {
        return Sophus::SO3<T>::hat(The)*Sophus::SO3<T>::hat(The);
    }

    template <class T = double>
    static Eigen::Matrix<T, 3, 3> Fu(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V)
    {
        // Extract the elements of input
        T ux = U(0); T uy = U(1); T uz = U(2);
        T vx = V(0); T vy = V(1); T vz = V(2);

        Eigen::Matrix<T, 3, 3> Fu_;
        Fu_ << uy*vy +     uz*vz, ux*vy - 2.0*uy*vx, ux*vz - 2.0*uz*vx,
               uy*vx - 2.0*ux*vy, ux*vx +     uz*vz, uy*vz - 2.0*uz*vy,
               uz*vx - 2.0*ux*vz, uz*vy - 2.0*uy*vz, ux*vx +     uy*vy;
        return Fu_; 
    }

    template <class T = double>
    static Eigen::Matrix<T, 3, 3> Fv(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V)
    {
        return Sophus::SO3<T>::hat(U)*Sophus::SO3<T>::hat(U);
    }

    template <class T = double>
    static Eigen::Matrix<T, 3, 3> Fuu(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A)
    {
        // Extract the elements of input
        // T ux = U(0); T uy = U(1); T uz = U(2);
        T vx = V(0); T vy = V(1); T vz = V(2);
        T ax = A(0); T ay = A(1); T az = A(2);

        Eigen::Matrix<T, 3, 3> Fuu_;
        Fuu_ << ay*vy +     az*vz, ax*vy - 2.0*ay*vx, ax*vz - 2.0*az*vx,
                ay*vx - 2.0*ax*vy, ax*vx +     az*vz, ay*vz - 2.0*az*vy,
                az*vx - 2.0*ax*vz, az*vy - 2.0*ay*vz, ax*vx +     ay*vy; 
        return Fuu_; 
    }

    template <class T = double>
    static Eigen::Matrix<T, 3, 3> Fuv(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &A)
    {
        // Extract the elements of input
        T ux = U(0); T uy = U(1); T uz = U(2);
        // T vx = V(0); T vy = V(1); T vz = V(2);
        T ax = A(0); T ay = A(1); T az = A(2);

        Eigen::Matrix<T, 3, 3> Fuv_;
        Fuv_ << -2.0*ay*uy - 2.0*az*uz,      ax*uy +     ay*ux,      ax*uz +     az*ux,
                     ax*uy +     ay*ux, -2.0*ax*ux - 2.0*az*uz,      ay*uz +     az*uy,
                     ax*uz +     az*ux,      ay*uz +     az*uy, -2.0*ax*ux - 2.0*ay*uy;
        return Fuv_; 
    }


    /* #region Lie operations for SO3 --------------------------------------------------------------------------------------------------------------------------------*/

    // right Jacobian for SO3
    template <class T = double>
    static Eigen::Matrix<T, 3, 3> Jr(const Eigen::Matrix<T, 3, 1> &The)
    {
        if (The.norm() < DOUBLE_EPSILON)
            return Eigen::Matrix<T, 3, 3>::Identity() - 0.5*Sophus::SO3<T>::hat(The) + (1.0/6.0)*hatSquare<T>(The);

        return Sophus::SO3<T>::leftJacobian(-The);
    }

    // inverse right Jacobian for SO3
    template <class T = double>
    static Eigen::Matrix<T, 3, 3> JrInv(const Eigen::Matrix<T, 3, 1> &The)
    {
        if (The.norm() < DOUBLE_EPSILON)
            return Eigen::Matrix<T, 3, 3>::Identity() + 0.5*Sophus::SO3<T>::hat(The) + (1.0/12.0)*hatSquare(The);

        return Sophus::SO3<T>::leftJacobianInverse(-The);
    }


    // left Jacobian for SO3
    template <class T = double>
    static Eigen::Matrix<T, 3, 3> Jl(const Eigen::Matrix<T, 3, 1> &The)
    {
        return Sophus::SO3<T>::leftJacobian(The);
    }

    // inverse right Jacobian for SO3
    template <class T = double>
    static Eigen::Matrix<T, 3, 3> JlInv(const Eigen::Matrix<T, 3, 1> &The)
    {
        return Sophus::SO3<T>::leftJacobianInverse(The);;
    }

    // For calculating HThe_
    template <class T = double>
    Eigen::Matrix<T, 3, 3> DJrUV_DU(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        if(use_approx_drv)
        {
            // cout << "approx form" << endl;
            return 0.5*SO3T::hat(V);
        }

        T Un = U.norm();

        if(Un < DOUBLE_EPSILON)
            return 0.5*SO3T::hat(V) + Fu(U, V)/6.0;

        T Unp2 = Un*Un;
        T Unp3 = Unp2*Un;
        T Unp4 = Unp3*Un;

        T sUn = sin(Un);
        // T sUnp2 = sUn*sUn;
        
        T cUn = cos(Un);
        // T cUnp2 = cUn*cUn;
        
        T gUn = (1.0 - cUn)/Unp2;
        T DgUn_DUn = sUn/Unp2 - 2.0*(1.0 - cUn)/Unp3;

        T hUn = (Un - sUn)/Unp3;
        T DhUn_DUn = (1.0 - cUn)/Unp3 - 3.0*(Un - sUn)/Unp4;

        Vec3T Ub = U/Un;
        
        Vec3T UsksqV = SO3T::hat(U)*SO3T::hat(U)*V;
        Mat3T DUsksqV_DU = Fu<T>(U, V);

        return SO3T::hat(V)*gUn + SO3T::hat(V)*U*DgUn_DUn*Ub.transpose() + DUsksqV_DU*hUn + UsksqV*DhUn_DUn*Ub.transpose();
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DDJrUVW_DUDU(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &W) const
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Un = U.norm();

        if(use_approx_drv)
        {
            // cout << "approx form" << endl;
            return Mat3T::Zero(3, 3);
        }

        if(Un < DOUBLE_EPSILON)
            return Fuu(U, V, W)/6.0;

        T Unp2 = Un*Un;
        T Unp3 = Unp2*Un;
        T Unp4 = Unp3*Un;
        T Unp5 = Unp4*Un;

        T sUn = sin(Un);
        // T sUnp2 = sUn*sUn;
        
        T cUn = cos(Un);
        // T cUnp2 = cUn*cUn;
        
        // T gUn = (1.0 - cUn)/Unp2;
        T DgUn_DUn = sUn/Unp2 - 2.0*(1.0 - cUn)/Unp3;
        T DDgUn_DUnDUn = cUn/Unp2 - 4.0*sUn/Unp3 + 6.0*(1.0 - cUn)/Unp4;

        T hUn = (Un - sUn)/Unp3;
        T DhUn_DUn = (1.0 - cUn)/Unp3 - 3.0*(Un - sUn)/Unp4;
        T DDhUn_DUnDUn = 6.0/Unp4 + sUn/Unp3 + 6.0*cUn/Unp4 - 12.0*sUn/Unp5;

        Vec3T Ub = U/Un;
        Mat3T DUb_DU = 1.0/Un*(Mat3T::Identity(3, 3) - Ub*Ub.transpose());

        Vec3T UsksqV = SO3T::hat(U)*SO3T::hat(U)*V;
        Mat3T DUsksqV_DU = Fu(U, V);
        Mat3T DDUsksqVW_DUDU = Fuu(U, V, W);

        Mat3T Vsk = SO3T::hat(V);
        T WtpUb = W.transpose()*Ub;
        Eigen::Matrix<T, 1, 3> WtpDUb = W.transpose()*DUb_DU;

        return  Vsk*W*DgUn_DUn*Ub.transpose()

              + Vsk*WtpUb*DgUn_DUn
              + Vsk*U*WtpDUb*DgUn_DUn
              + Vsk*U*WtpUb*Ub.transpose()*DDgUn_DUnDUn

              + DDUsksqVW_DUDU*hUn
              + DUsksqV_DU*W*Ub.transpose()*DhUn_DUn

              + DUsksqV_DU*WtpUb*DhUn_DUn
              + UsksqV*WtpDUb*DhUn_DUn
              + UsksqV*WtpUb*Ub.transpose()*DDhUn_DUnDUn;
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DDJrUVW_DUDV(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &W) const
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Un = U.norm();

        if(use_approx_drv)
        {
            // cout << "approx form" << endl;
            return -0.5*SO3T::hat(W);
        }

        if(Un < DOUBLE_EPSILON)
            return -0.5*SO3T::hat(W) + Fuv(U, V, W)/6.0;

        T Unp2 = Un*Un;
        T Unp3 = Unp2*Un;
        T Unp4 = Unp3*Un;
        // T Unp5 = Unp4*Un;

        T sUn = sin(Un);
        // T sUnp2 = sUn*sUn;
        
        T cUn = cos(Un);
        // T cUnp2 = cUn*cUn;
        
        T gUn = (1.0 - cUn)/Unp2;
        T DgUn_DUn = sUn/Unp2 - 2.0*(1.0 - cUn)/Unp3;
        // T DDgUn_DUnDUn = cUn/Unp2 - 4.0*sUn/Unp3 + 6.0*(1.0 - cUn)/Unp4;

        T hUn = (Un - sUn)/Unp3;
        T DhUn_DUn = (1.0 - cUn)/Unp3 - 3.0*(Un - sUn)/Unp4;
        // T DDhUn_DUnDUn = 6.0/Unp4 + sUn/Unp3 + 6.0*cUn/Unp4 - 12*sUn/Unp5;

        Vec3T Ub = U/Un;
        // Mat3T DUb_DU = 1.0/Un*(Mat3T::Identity(3, 3) - Ub*Ub.transpose());

        Mat3T DUsksqV_DV = Fv(U, V);
        Mat3T DDUsksqVW_DUDV = Fuv(U, V, W);

        T WtpUb = W.transpose()*Ub;

        return -SO3T::hat(W)*gUn - SO3T::hat(U)*DgUn_DUn*WtpUb + DDUsksqVW_DUDV*hUn + DUsksqV_DV*DhUn_DUn*WtpUb;
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DJrInvUV_DU(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Un = U.norm();

        if(use_approx_drv)
        {
            // cout << "approx form" << endl;
            return -0.5*SO3T::hat(V);
        }

        if(Un < DOUBLE_EPSILON)
            return -0.5*SO3T::hat(V) + Fu(U, V)/12.0;

        T Unp2 = Un*Un;
        T Unp3 = Unp2*Un;
        
        T sUn = sin(Un);
        T sUnp2 = sUn*sUn;
        
        T cUn = cos(Un);
        // T cUnp2 = cUn*cUn;
        
        T gUn = (1.0/Unp2 - (1.0 + cUn)/(2.0*Un*sUn));
        T DgUn_DUn = -2.0/Unp3 + (Un*sUnp2 + (sUn + Un*cUn)*(1.0 + cUn))/(2.0*Unp2*sUnp2);

        Vec3T Ub = U/Un;

        Vec3T UsksqV = SO3T::hat(U)*SO3T::hat(U)*V;
        Mat3T DUsksqV_DU = Fu(U, V);

        return -0.5*SO3T::hat(V) + DUsksqV_DU*gUn + UsksqV*DgUn_DUn*Ub.transpose();
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DDJrInvUVW_DUDU(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &W) const
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Un = U.norm();

        if(use_approx_drv)
        {
            // cout << "approx form" << endl;
            return Mat3T::Zero();
        }

        if(Un < DOUBLE_EPSILON)
            return Fuu(U, V, W)/12.0;

        T Unp2 = Un*Un;
        T Unp3 = Unp2*Un;
        T Unp4 = Unp3*Un;
        // T Unp5 = Unp4*Un;

        T sUn = sin(Un);
        T sUnp2 = sUn*sUn;
        // T s2Un = sin(2.0*Un);
        
        T cUn = cos(Un);
        // T cUnp2 = cUn*cUn;
        T c2Un = cos(2.0*Un);
        
        T gUn = 1.0/Unp2 - (1.0 + cUn)/(2.0*Un*sUn);
        T DgUn_DUn = -2.0/Unp3 + (sUn + Un)*(1.0 + cUn)/(2.0*Unp2*sUnp2);
        // T DDgUn_DUnDUn = 6.0/Unp4 + (1.0 - c2Un + Unp2*cUn + 2.0*Un*sUn + Unp2)/(Unp3*2.0*sUn*(cUn - 1.0));
        T DDgUn_DUnDUn = 6.0/Unp4 + sUn/(Unp3*(cUn - 1.0)) + (Un*cUn + 2.0*sUn + Un)/(2.0*Unp2*sUn*(cUn - 1.0));

        Vec3T Ub = U/Un;
        Mat3T DUb_DU = 1.0/Un*(Mat3T::Identity(3, 3) - Ub*Ub.transpose());

        Vec3T UsksqV = SO3T::hat(U)*SO3T::hat(U)*V;
        Mat3T DUsksqV_DU = Fu(U, V);
        Mat3T DDUsksqVW_DUDU = Fuu(U, V, W);

        // Mat3T Vsk = SO3T::hat(V);
        T WtpUb = W.transpose()*Ub;
        Eigen::Matrix<T, 1, 3> WtpDUb = W.transpose()*DUb_DU;

        return   DDUsksqVW_DUDU*gUn
               + DUsksqV_DU*W*Ub.transpose()*DgUn_DUn

               + DUsksqV_DU*WtpUb*DgUn_DUn
               + UsksqV*WtpDUb*DgUn_DUn
               + UsksqV*WtpUb*Ub.transpose()*DDgUn_DUnDUn;
    }

    template <class T = double>
    Eigen::Matrix<T, 3, 3> DDJrInvUVW_DUDV(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &W) const
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Un = U.norm();

        if(use_approx_drv)
        {
            // cout << "approx form" << endl;
            return 0.5*SO3T::hat(W);
        }

        if(Un < DOUBLE_EPSILON)
            return 0.5*SO3T::hat(W) + Fuv(U, V, W)/12.0;

        T Unp2 = Un*Un;
        T Unp3 = Unp2*Un;
        // T Unp4 = Unp3*Un;
        // T Unp5 = Unp4*Un;

        T sUn = sin(Un);
        T sUnp2 = sUn*sUn;
        // T s2Un = sin(2*Un);
        
        T cUn = cos(Un);
        // T cUnp2 = cUn*cUn;
        // T c2Un = cos(2*Un);
        
        T gUn = (1.0/Unp2 - (1.0 + cUn)/(2.0*Un*sUn));
        T DgUn_DUn = -2.0/Unp3 + (Un*sUnp2 + (sUn + Un*cUn)*(1.0 + cUn))/(2.0*Unp2*sUnp2);
        // T DDgUn_DUnDUn = (Un + 6.0*s2Un - 12.0*sUn - Un*c2Un + Unp3*cUn + 2.0*Unp2*sUn + Unp3)/(Unp4*(s2Un - 2.0*sUn));

        Vec3T Ub = U/Un;
        // Mat3T DUb_DU = 1.0/Un*(Mat3T::Identity(3, 3) - Ub*Ub.transpose());

        Mat3T DUsksqV_DV = Fv(U, V);
        // Mat3T DUsksqV_DU = Fu(U, V);
        Mat3T DDUsksqVW_DUDV = Fuv(U, V, W);

        T WtpUb = W.transpose()*Ub;

        return 0.5*SO3T::hat(W) + DDUsksqVW_DUDV*gUn + DUsksqV_DV*DgUn_DUn*WtpUb;
    }

    /* #endregion Lie operations for SO3 -----------------------------------------------------------------------------------------------------------------------------*/


    /* #region Lie operations for SE3 --------------------------------------------------------------------------------------------------------------------------------*/


    template <class T = double>
    Eigen::Matrix<T, 6, 6> SE3hat(const Eigen::Matrix<T, 6, 1> &Xi) const
    {
        Eigen::Matrix<T, 3, 1> The = Xi.template head<3>();
        Eigen::Matrix<T, 3, 1> Rho = Xi.template tail<3>();

        Eigen::Matrix<T, 3, 3> Thehat = Sophus::SO3<T>::hat(The);
        Eigen::Matrix<T, 3, 3> Rhohat = Sophus::SO3<T>::hat(Rho);
        Eigen::Matrix<T, 3, 3> MZero3 = Eigen::Matrix<T, 3, 3>::Zero();

        Eigen::Matrix<T, 6, 6> M;
        M << Thehat, MZero3,
             Rhohat, Thehat;

        return M;
    }

    template <class T = double>
    Sophus::SE3<T> SE3Exp(const Eigen::Matrix<T, 6, 1> &Xi) const
    {
        Eigen::Matrix<T, 3, 1> The = Xi.template head<3>();
        Eigen::Matrix<T, 3, 1> Rho = Xi.template tail<3>();

        return Sophus::SE3<T>(Sophus::SO3<T>::exp(The), Jr((-The).eval())*Rho);
    }

    template <class T = double>
    Eigen::Matrix<T, 6, 1> SE3Log(const Sophus::SE3<T> &Tf) const
    {
        Eigen::Matrix<T, 6, 1> Xi_ = Tf.log();
        Eigen::Matrix<T, 6, 1> Xi;
        Xi << Xi_.template tail<3>(), Xi_.template head<3>();

        return Xi;
    }

    template <class T = double>
    Eigen::Matrix<T, 6, 6> SE3Adj(const Sophus::SE3<T> &Tf) const
    {
        Eigen::Matrix<T, 3, 1> The = Tf.so3().log();
        Eigen::Matrix<T, 3, 1> Pos = Tf.translation();

        Eigen::Matrix<T, 3, 3> R = Sophus::SO3<T>::exp(The).matrix();

        Matrix<T, 6, 6> Adj;
        Adj << R, Eigen::Matrix<T, 3, 3>::Zero(),
               Sophus::SO3<T>::hat(Pos)*R, R;

        return Adj;
    }

    template <class T = double>
    Eigen::Matrix<T, 6, 6> SE3AdjInv(const Sophus::SE3<T> &Tf) const
    {
        Eigen::Matrix<T, 3, 1> The = Tf.so3().log();
        Eigen::Matrix<T, 3, 1> Pos = Tf.translation();

        Eigen::Matrix<T, 3, 3> Rinv = Sophus::SO3<T>::exp(The).matrix().transpose();

        Matrix<T, 6, 6> AdjInv;
        AdjInv << Rinv, Eigen::Matrix<T, 3, 3>::Zero(),
                 -Rinv*Sophus::SO3<T>::hat(Pos), Rinv;

        return AdjInv;
    }

    template <class T = double>
    static Eigen::Matrix<T, 3, 3> SE3Qr(const Matrix<T, 6, 1> &Xi)
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        Eigen::Matrix<T, 3, 1> U = -Xi.template head<3>();
        Eigen::Matrix<T, 3, 1> V =  Xi.template tail<3>();

        T Un = U.norm();

        Mat3T Exp_U = SO3T::exp(U).matrix();

        if(Un < DOUBLE_EPSILON)
            return -(Exp_U*(0.5*SO3T::hat(V) + Fu(U, V)/6.0));

        T Unp2 = Un*Un;
        T Unp3 = Unp2*Un;
        T Unp4 = Unp3*Un;

        T sUn = sin(Un);
        // T sUnp2 = sUn*sUn;
        
        T cUn = cos(Un);
        // T cUnp2 = cUn*cUn;
        
        T gUn = (1.0 - cUn)/Unp2;
        T DgUn_DUn = sUn/Unp2 - 2.0*(1.0 - cUn)/Unp3;

        T hUn = (Un - sUn)/Unp3;
        T DhUn_DUn = (1.0 - cUn)/Unp3 - 3.0*(Un - sUn)/Unp4;

        Vec3T Ub = U/Un;
        
        Vec3T UsksqV = SO3T::hat(U)*SO3T::hat(U)*V;
        Mat3T DUsksqV_DU = Fu<T>(U, V);

        return -Exp_U*(SO3T::hat(V)*gUn + SO3T::hat(V)*U*DgUn_DUn*Ub.transpose() + DUsksqV_DU*hUn + UsksqV*DhUn_DUn*Ub.transpose());
    }

    // right Jacobian for SE3
    template <class T = double>
    static Eigen::Matrix<T, 6, 6> Jr(const Eigen::Matrix<T, 6, 1> &Xi)
    {
        Eigen::Matrix<T, 3, 1> The = Xi.template head<3>();
        Eigen::Matrix<T, 3, 1> Rho = Xi.template tail<3>();
        
        Eigen::Matrix<T, 6, 6> Jr_Xi;
        Eigen::Matrix<T, 3, 3> Jr_The = Jr(The);
        Eigen::Matrix<T, 3, 3> Qr = SE3Qr(Xi);

        Jr_Xi << Jr_The, Matrix<T, 3, 3>::Zero(),
                 Qr, Jr_The;
        return Jr_Xi;
    }

    // inverse right Jacobian for SE3
    template <class T = double>
    static Eigen::Matrix<T, 6, 6> JrInv(const Eigen::Matrix<T, 6, 1> &Xi)
    {
        Eigen::Matrix<T, 3, 1> The = Xi.template head<3>();
        Eigen::Matrix<T, 3, 1> Rho = Xi.template tail<3>();

        Eigen::Matrix<T, 6, 6> JrInv_Xi;
        Eigen::Matrix<T, 3, 3> JrInv_The = JrInv<T>(The);
        Eigen::Matrix<T, 3, 3> Qr = SE3Qr(Xi);

        JrInv_Xi << JrInv_The, Matrix<T, 3, 3>::Zero(),
                   -JrInv_The*Qr*JrInv_The, JrInv_The;

        return JrInv_Xi;
    }

    // left Jacobian for SE3
    template <class T = double>
    static Eigen::Matrix<T, 6, 6> Jl(const Eigen::Matrix<T, 6, 1> &Xi)
    {
        return Jr<T>(-Xi);
    }

    // inverse left Jacobian for SE3
    template <class T = double>
    static Eigen::Matrix<T, 6, 6> JlInv(const Eigen::Matrix<T, 6, 1> &Xi)
    {
        return JrInv<T>(-Xi);
    }

    template <class T = double>
    void Get_JHL(const Eigen::Matrix<T, 6, 1> &Xi,
                 const Eigen::Matrix<T, 6, 1> &Xid,
                 const Eigen::Matrix<T, 6, 1> &Xidd,
                 Eigen::Matrix<T, 6, 6> &Jr_Xi,
                 Eigen::Matrix<T, 6, 6> &H1_XiXid,
                 Eigen::Matrix<T, 6, 6> &H1_XiXidd,
                 Eigen::Matrix<T, 6, 6> &L11_XiXidXid,
                 Eigen::Matrix<T, 6, 6> &L12_XiXidXid) const
    {
        // using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;
        using Mat6T = Eigen::Matrix<T, 6, 6>;

        Vec3T The = Xi.template head(3);
        Vec3T Rho = Xi.template tail(3);

        Vec3T Thed = Xid.template head(3);
        Vec3T Rhod = Xid.template tail(3);

        Vec3T Thedd = Xidd.template head(3);
        Vec3T Rhodd = Xidd.template tail(3);

        Jr_Xi = Jr(Xi);

        if (The.norm() < lie_epsilon || use_approx_drv)
        {
            // cout << "approx form " << The.transpose() << endl;

            H1_XiXid     =  0.5*SE3hat(Xid);
            H1_XiXidd    =  0.5*SE3hat(Xidd);
            L11_XiXidXid =  Mat6T::Zero();
            L12_XiXidXid = -0.5*SE3hat(Xid);
        }
        else
        {
            // cout << "closed form" << endl;

            SE3Q<T> myQ_XiXid;
            myQ_XiXid.ComputeQSC(The, Rho, Thed, Rhod);
            
            SE3Q<T> myQ_XiXidd;
            myQ_XiXidd.ComputeS(The, Rho, Thedd);
            
            Mat3T Zero = Mat3T::Zero(3, 3);
            Mat3T Jr_The = Jr(The);
            Mat3T H1_TheThed = DJrUV_DU(The, Thed);
            Mat3T H1_TheRhod = DJrUV_DU(The, Rhod);
            Mat3T H1_TheThedd = DJrUV_DU(The, Thedd);
            Mat3T H1_TheRhodd = DJrUV_DU(The, Rhodd);
            Mat3T L11_TheThedThed = DDJrUVW_DUDU(The, Thed, Thed);
            Mat3T L11_TheRhodThed = DDJrUVW_DUDU(The, Rhod, Thed);
            Mat3T L12_TheThedThed = DDJrUVW_DUDV(The, Thed, Thed);
            Mat3T L12_TheRhodThed = DDJrUVW_DUDV(The, Rhod, Thed);
        
            H1_XiXid
                << H1_TheThed, Zero,
                   myQ_XiXid.S1 + H1_TheRhod, myQ_XiXid.S2;

            H1_XiXidd
                << H1_TheThedd, Zero,
                   myQ_XiXidd.S1 + H1_TheRhodd, myQ_XiXidd.S2;

            L11_XiXidXid
                << L11_TheThedThed, Zero,
                   myQ_XiXid.C11 + L11_TheRhodThed + myQ_XiXid.C21, myQ_XiXid.C12 + myQ_XiXid.C22;

            L12_XiXidXid
                << L12_TheThedThed, Zero,
                   myQ_XiXid.C13 + myQ_XiXid.C23, L12_TheRhodThed;

            // assert(!myQ_XiXid.Q.array().isNaN().any());
            // assert(!myQ_XiXid.S1.array().isNaN().any());
            // assert(!myQ_XiXid.S2.array().isNaN().any());
            // assert(!myQ_XiXid.C11.array().isNaN().any());
            // assert(!myQ_XiXid.C12.array().isNaN().any());
            // assert(!myQ_XiXid.C13.array().isNaN().any());
            // assert(!myQ_XiXid.C21.array().isNaN().any());
            // assert(!myQ_XiXid.C22.array().isNaN().any());
            // assert(!myQ_XiXid.C23.array().isNaN().any());

            // assert(!myQ_XiXidd.S1.array().isNaN().any());
            // assert(!myQ_XiXidd.S2.array().isNaN().any());

            // assert(!H1_XiXid.array().isNaN().any());
            // assert(!H1_XiXidd.array().isNaN().any());
            // assert(!L11_XiXidXid.array().isNaN().any());
            // assert(!L12_XiXidXid.array().isNaN().any());
        }
    }

    template <class T = double>
    void Get_JrInvHpLp(const Eigen::Matrix<T, 6, 1> &Xi,
                       const Eigen::Matrix<T, 6, 1> &Tw,
                       const Eigen::Matrix<T, 6, 1> &Wr,
                       Eigen::Matrix<T, 6, 1> &Xid,
                       Eigen::Matrix<T, 6, 1> &Xidd,
                       Eigen::Matrix<T, 6, 6> &JrInv_Xi,
                       Eigen::Matrix<T, 6, 6> &Hp1_XiTw,
                       Eigen::Matrix<T, 6, 6> &Hp1_XiWr,
                       Eigen::Matrix<T, 6, 6> &Lp11_XiTwXid,
                       Eigen::Matrix<T, 6, 6> &Lp12_XiTwXid) const
    {
        // using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Vec6T = Eigen::Matrix<T, 6, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;
        using Mat6T = Eigen::Matrix<T, 6, 6>;

        Vec3T The = Xi.template head(3);
        Vec3T Rho = Xi.template tail(3);

        Vec3T Omg = Tw.template head(3);
        Vec3T Nuy = Tw.template tail(3);

        Vec3T Alp = Wr.template head(3);
        Vec3T Bta = Wr.template tail(3);

        JrInv_Xi = JrInv(Xi);
        Xid = JrInv_Xi*Tw;

        Vec3T Thed = Xid.template head(3);
        Vec3T Rhod = Xid.template tail(3);

        if (The.norm() < lie_epsilon || use_approx_drv)
        {
            // cout << "approxed form " << The.transpose() << endl;

            Hp1_XiTw     = -0.5*SE3hat(Tw);
            Hp1_XiWr     = -0.5*SE3hat(Wr);
            Lp11_XiTwXid =  Mat6T::Zero();
            Lp12_XiTwXid =  0.5*SE3hat(Xid);
        }
        else
        {
            // cout << "closed form" << endl;
            
            SE3Qp<T> myQp_XiTw;
            myQp_XiTw.ComputeQSC(The, Rho, Thed, Rhod, Omg);
            
            SE3Qp<T> myQp_XiWr;
            myQp_XiWr.ComputeS(The, Rho, Alp);
            
            Mat3T Zero = Mat3T::Zero(3, 3);
            Mat3T JrInv_The = JrInv(The);
            
            Mat3T Hp1_TheOmg = DJrInvUV_DU(The, Omg);
            Mat3T Hp1_TheNuy = DJrInvUV_DU(The, Nuy);
            Mat3T Hp1_TheAlp = DJrInvUV_DU(The, Alp);
            Mat3T Hp1_TheBta = DJrInvUV_DU(The, Bta);
            Mat3T Lp11_TheOmgThed = DDJrInvUVW_DUDU(The, Omg, Thed);
            Mat3T Lp11_TheNuyThed = DDJrInvUVW_DUDU(The, Nuy, Thed);
            Mat3T Lp12_TheOmgThed = DDJrInvUVW_DUDV(The, Omg, Thed);
            Mat3T Lp12_TheNuyThed = DDJrInvUVW_DUDV(The, Nuy, Thed);

            Hp1_XiTw
                << Hp1_TheOmg, Zero,
                   myQp_XiTw.S1 + Hp1_TheNuy, myQp_XiTw.S2;

            Hp1_XiWr
                << Hp1_TheAlp, Zero,
                   myQp_XiWr.S1 + Hp1_TheBta, myQp_XiWr.S2;

            Lp11_XiTwXid
                << Lp11_TheOmgThed, Zero,
                   myQp_XiTw.C11 + Lp11_TheNuyThed + myQp_XiTw.C21, myQp_XiTw.C12 + myQp_XiTw.C22;

            Lp12_XiTwXid
                << Lp12_TheOmgThed, Zero,
                   myQp_XiTw.C13 + myQp_XiTw.C23, Lp12_TheNuyThed;

            // assert(!myQp_XiTw.Q.array().isNaN().any());
            // assert(!myQp_XiTw.S1.array().isNaN().any());
            // assert(!myQp_XiTw.S2.array().isNaN().any());
            // assert(!myQp_XiTw.C11.array().isNaN().any());
            // assert(!myQp_XiTw.C12.array().isNaN().any());
            // assert(!myQp_XiTw.C13.array().isNaN().any());
            // assert(!myQp_XiTw.C21.array().isNaN().any());
            // assert(!myQp_XiTw.C22.array().isNaN().any());
            // assert(!myQp_XiTw.C23.array().isNaN().any());
            
            // assert(!myQp_XiWr.S1.array().isNaN().any());
            // assert(!myQp_XiWr.S2.array().isNaN().any());
            
            // assert(!Hp1_XiTw.array().isNaN().any());
            // assert(!Hp1_XiWr.array().isNaN().any());
            // assert(!Lp11_XiTwXid.array().isNaN().any());
            // assert(!Lp12_XiTwXid.array().isNaN().any());
        }

        Xidd = JrInv_Xi*Wr + Hp1_XiTw*Xid;
    }

    /* #endregion Lie operations for SE3 -----------------------------------------------------------------------------------------------------------------------------*/


    template <class T = double>
    void MapParamToState(T const *const *parameters, int base, GPState<T> &X) const
    {
        X.R = Eigen::Map<Sophus::SO3<T> const>(parameters[base + 0]);
        X.O = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 1]);
        X.S = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 2]);
        X.P = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 3]);
        X.V = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 4]);
        X.A = Eigen::Map<Eigen::Matrix<T, 3, 1> const>(parameters[base + 5]);
    }

    template <class T = double>
    void ComputeXtAndJacobians(const GPState<T> &Xa,
                               const GPState<T> &Xb,
                                     GPState<T> &Xt,
                               vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXa,
                               vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXb,
                               bool find_jacobian = true
                              ) const
    {
        if (pose_representation == POSE_GROUP::SO3xR3)
            ComputeXtAndJacobiansSO3xR3(Xa, Xb, Xt, DXt_DXa, DXt_DXb, find_jacobian);
        else if (pose_representation == POSE_GROUP::SE3)
            ComputeXtAndJacobiansSE3(Xa, Xb, Xt, DXt_DXa, DXt_DXb, find_jacobian);
    }

    template <class T = double>
    void ComputeXtAndJacobiansSO3xR3(const GPState<T> &Xa,
                                     const GPState<T> &Xb,
                                           GPState<T> &Xt,
                                     vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXa,
                                     vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXb,
                                     bool find_jacobian = true
                                    ) const
    {
        // Local index for the states in the state vector
        const int RIDX = 0;
        const int OIDX = 1;
        const int SIDX = 2;
        const int PIDX = 3;
        const int VIDX = 4;
        const int AIDX = 5;

        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Vec6T = Eigen::Matrix<T, 6, 1>;
        using Vec9T = Eigen::Matrix<T, 9, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        double Dtau = (Xt.t - Xa.t)/(Xb.t - Xa.t)*Dt;

        // Map the variables of the state
        SO3T   &Rt =  Xt.R;
        Vec3T  &Ot =  Xt.O;
        Vec3T  &St =  Xt.S;
        Vec3T  &Pt =  Xt.P;
        Vec3T  &Vt =  Xt.V;
        Vec3T  &At =  Xt.A;
        
        /* #region Processing the RO3 states ------------------------------------------------------------------------*/

        // Prepare the the mixer matrixes
        Matrix<T, 9, 9> LAM_ROSt = LMD(Dtau, CovROSJerk).cast<T>();
        Matrix<T, 9, 9> PSI_ROSt = PSI(Dtau, CovROSJerk).cast<T>();

        // Find the relative rotation
        SO3T Rab = Xa.R.inverse()*Xb.R;

        // Calculate the SO3 knots in relative form
        Vec3T Thead0 = Vec3T::Zero();
        Vec3T Thead1 = Xa.O;
        Vec3T Thead2 = Xa.S;

        // Find the local variable at tb and the associated Jacobians
        Vec3T Theb = Rab.log();
        Mat3T JrInv_Theb = JrInv<T>(Theb);
        Mat3T Hp1_ThebOb = DJrInvUV_DU(Theb, Xb.O);

        Vec3T Thebd0 = Theb;
        Vec3T Thebd1 = JrInv_Theb*Xb.O;
        Vec3T Thebd2 = JrInv_Theb*Xb.S + Hp1_ThebOb*Thebd1;
        
        Mat3T Hp1_ThebSb = DJrInvUV_DU(Theb, Xb.S);
        Mat3T Lp11_ThebObThebd1 = DDJrInvUVW_DUDU(Theb, Xb.O, Thebd1);
        Mat3T Lp12_ThebObThebd1 = DDJrInvUVW_DUDV(Theb, Xb.O, Thebd1);

        // Put them in vector form
        Vec9T gammaa; gammaa << Thead0, Thead1, Thead2;
        Vec9T gammab; gammab << Thebd0, Thebd1, Thebd2;
        // Mix the knots to get the interpolated states
        Vec9T gammat = LAM_ROSt*gammaa + PSI_ROSt*gammab;

        // Retrive the interpolated SO3 in relative form
        Vec3T Thetd0 = gammat.block(0, 0, 3, 1);
        Vec3T Thetd1 = gammat.block(3, 0, 3, 1);
        Vec3T Thetd2 = gammat.block(6, 0, 3, 1);

        // Do all jacobians needed for L4-L3 interface 
        Mat3T Jr_Thet = Jr<T>(Thetd0);
        Mat3T H1_ThetThetd1 = DJrUV_DU(Thetd0, Thetd1);
        Mat3T H1_ThetThetd2 = DJrUV_DU(Thetd0, Thetd2);
        Mat3T L11_ThetThetd1Thetd1 = DDJrUVW_DUDU(Thetd0, Thetd1, Thetd1);
        Mat3T L12_ThetThetd1Thetd1 = DDJrUVW_DUDV(Thetd0, Thetd1, Thetd1);
        
        SO3T Exp_Thet = SO3T::exp(Thetd0);

        // Calculate the interpolated global states from the interpolated local
        Rt = Xa.R*Exp_Thet;
        Ot = Jr_Thet*Thetd1;
        St = Jr_Thet*Thetd2 + H1_ThetThetd1*Thetd1;

        // If calculating the Jacobian
        if(find_jacobian)
        {
            // Calculate the Jacobian
            DXt_DXa = vector<vector<Mat3T>>(6, vector<Mat3T>(6, Mat3T::Zero()));
            DXt_DXb = vector<vector<Mat3T>>(6, vector<Mat3T>(6, Mat3T::Zero()));


            // Jacobians from L2 to L1
            Mat3T  J_Thead1_Oa = Mat3T::Identity(); Mat3T J_Thead2_Sa = Mat3T::Identity();

            Mat3T  J_Thebd0_Ra = -JrInv_Theb*Rab.inverse().matrix();
            Mat3T &J_Thebd0_Rb =  JrInv_Theb;

            Mat3T  J_Thebd1_Ra = Hp1_ThebOb*J_Thebd0_Ra;
            Mat3T  J_Thebd1_Rb = Hp1_ThebOb*J_Thebd0_Rb;
            Mat3T &J_Thebd1_Ob = JrInv_Theb;

            Mat3T  J_Thebd2_Ra = Hp1_ThebSb*J_Thebd0_Ra + Hp1_ThebOb*J_Thebd1_Ra + Lp11_ThebObThebd1*J_Thebd0_Ra;
            Mat3T  J_Thebd2_Rb = Hp1_ThebSb*J_Thebd0_Rb + Hp1_ThebOb*J_Thebd1_Rb + Lp11_ThebObThebd1*J_Thebd0_Rb;
            Mat3T  J_Thebd2_Ob = Lp12_ThebObThebd1 + Hp1_ThebOb*J_Thebd1_Ob;
            Mat3T &J_Thebd2_Sb = JrInv_Theb;


            // Jacobians from L3 to L2
            Mat3T J_Thetd0_Thead0 = LAM_ROSt.block(0, 0, 3, 3); Mat3T J_Thetd0_Thead1 = LAM_ROSt.block(0, 3, 3, 3); Mat3T J_Thetd0_Thead2 = LAM_ROSt.block(0, 6, 3, 3);
            Mat3T J_Thetd1_Thead0 = LAM_ROSt.block(3, 0, 3, 3); Mat3T J_Thetd1_Thead1 = LAM_ROSt.block(3, 3, 3, 3); Mat3T J_Thetd1_Thead2 = LAM_ROSt.block(3, 6, 3, 3);
            Mat3T J_Thetd2_Thead0 = LAM_ROSt.block(6, 0, 3, 3); Mat3T J_Thetd2_Thead1 = LAM_ROSt.block(6, 3, 3, 3); Mat3T J_Thetd2_Thead2 = LAM_ROSt.block(6, 6, 3, 3);
            Mat3T J_Thetd0_Thebd0 = PSI_ROSt.block(0, 0, 3, 3); Mat3T J_Thetd0_Thebd1 = PSI_ROSt.block(0, 3, 3, 3); Mat3T J_Thetd0_Thebd2 = PSI_ROSt.block(0, 6, 3, 3);
            Mat3T J_Thetd1_Thebd0 = PSI_ROSt.block(3, 0, 3, 3); Mat3T J_Thetd1_Thebd1 = PSI_ROSt.block(3, 3, 3, 3); Mat3T J_Thetd1_Thebd2 = PSI_ROSt.block(3, 6, 3, 3);
            Mat3T J_Thetd2_Thebd0 = PSI_ROSt.block(6, 0, 3, 3); Mat3T J_Thetd2_Thebd1 = PSI_ROSt.block(6, 3, 3, 3); Mat3T J_Thetd2_Thebd2 = PSI_ROSt.block(6, 6, 3, 3);


            // Jacobians from L3 to L1
            Mat3T J_Thetd0_Ra = J_Thetd0_Thebd0*J_Thebd0_Ra + J_Thetd0_Thebd1*J_Thebd1_Ra + J_Thetd0_Thebd2*J_Thebd2_Ra;
            Mat3T J_Thetd0_Rb = J_Thetd0_Thebd0*J_Thebd0_Rb + J_Thetd0_Thebd1*J_Thebd1_Rb + J_Thetd0_Thebd2*J_Thebd2_Rb;
            Mat3T J_Thetd0_Oa = J_Thetd0_Thead1*J_Thead1_Oa;
            Mat3T J_Thetd0_Ob = J_Thetd0_Thebd1*J_Thebd1_Ob + J_Thetd0_Thebd2*J_Thebd2_Ob;
            Mat3T J_Thetd0_Sa = J_Thetd0_Thead2*J_Thead2_Sa;
            Mat3T J_Thetd0_Sb = J_Thetd0_Thebd2*J_Thebd2_Sb;

            Mat3T J_Thetd1_Ra = J_Thetd1_Thebd0*J_Thebd0_Ra + J_Thetd1_Thebd1*J_Thebd1_Ra + J_Thetd1_Thebd2*J_Thebd2_Ra;
            Mat3T J_Thetd1_Rb = J_Thetd1_Thebd0*J_Thebd0_Rb + J_Thetd1_Thebd1*J_Thebd1_Rb + J_Thetd1_Thebd2*J_Thebd2_Rb;
            Mat3T J_Thetd1_Oa = J_Thetd1_Thead1*J_Thead1_Oa;
            Mat3T J_Thetd1_Ob = J_Thetd1_Thebd1*J_Thebd1_Ob + J_Thetd1_Thebd2*J_Thebd2_Ob;
            Mat3T J_Thetd1_Sa = J_Thetd1_Thead2*J_Thead2_Sa;
            Mat3T J_Thetd1_Sb = J_Thetd1_Thebd2*J_Thebd2_Sb;

            Mat3T J_Thetd2_Ra = J_Thetd2_Thebd0*J_Thebd0_Ra + J_Thetd2_Thebd1*J_Thebd1_Ra + J_Thetd2_Thebd2*J_Thebd2_Ra;
            Mat3T J_Thetd2_Rb = J_Thetd2_Thebd0*J_Thebd0_Rb + J_Thetd2_Thebd1*J_Thebd1_Rb + J_Thetd2_Thebd2*J_Thebd2_Rb;
            Mat3T J_Thetd2_Oa = J_Thetd2_Thead1*J_Thead1_Oa;
            Mat3T J_Thetd2_Ob = J_Thetd2_Thebd1*J_Thebd1_Ob + J_Thetd2_Thebd2*J_Thebd2_Ob;
            Mat3T J_Thetd2_Sa = J_Thetd2_Thead2*J_Thead2_Sa;
            Mat3T J_Thetd2_Sb = J_Thetd2_Thebd2*J_Thebd2_Sb;


            // Jacobians from L4 to L3
            Mat3T &J_Rt_Thetd0 = Jr_Thet;
            Mat3T &J_Ot_Thetd0 = H1_ThetThetd1;
            Mat3T &J_Ot_Thetd1 = Jr_Thet;
            Mat3T  J_St_Thetd0 = H1_ThetThetd2 + L11_ThetThetd1Thetd1;
            Mat3T  J_St_Thetd1 = L12_ThetThetd1Thetd1 + H1_ThetThetd1;
            Mat3T &J_St_Thetd2 = Jr_Thet;


            // Jacobians from L4 to L1
            DXt_DXa[RIDX][RIDX] = Exp_Thet.Adj().inverse() + J_Rt_Thetd0*J_Thetd0_Ra;                          // DRt_DRa
            DXt_DXa[RIDX][OIDX] = J_Rt_Thetd0*J_Thetd0_Oa;                                                     // DRt_DOa
            DXt_DXa[RIDX][SIDX] = J_Rt_Thetd0*J_Thetd0_Sa;                                                     // DRt_DSa
            //-----------------------------------------------------------------------------------------------------------
            DXt_DXa[OIDX][RIDX] = J_Ot_Thetd0*J_Thetd0_Ra + J_Ot_Thetd1*J_Thetd1_Ra;                           // DOt_DRa
            DXt_DXa[OIDX][OIDX] = J_Ot_Thetd0*J_Thetd0_Oa + J_Ot_Thetd1*J_Thetd1_Oa;                           // DOt_DOa
            DXt_DXa[OIDX][SIDX] = J_Ot_Thetd0*J_Thetd0_Sa + J_Ot_Thetd1*J_Thetd1_Sa;                           // DOt_DSa
            //-----------------------------------------------------------------------------------------------------------
            DXt_DXa[SIDX][RIDX] = J_St_Thetd0*J_Thetd0_Ra + J_St_Thetd1*J_Thetd1_Ra + J_St_Thetd2*J_Thetd2_Ra; // DSt_DRa
            DXt_DXa[SIDX][OIDX] = J_St_Thetd0*J_Thetd0_Oa + J_St_Thetd1*J_Thetd1_Oa + J_St_Thetd2*J_Thetd2_Oa; // DSt_DOa
            DXt_DXa[SIDX][SIDX] = J_St_Thetd0*J_Thetd0_Sa + J_St_Thetd1*J_Thetd1_Sa + J_St_Thetd2*J_Thetd2_Sa; // DSt_DSa
            //-----------------------------------------------------------------------------------------------------------
            DXt_DXb[RIDX][RIDX] = J_Rt_Thetd0*J_Thetd0_Rb;                                                     // DRt_DRb
            DXt_DXb[RIDX][OIDX] = J_Rt_Thetd0*J_Thetd0_Ob;                                                     // DRt_DOb
            DXt_DXb[RIDX][SIDX] = J_Rt_Thetd0*J_Thetd0_Sb;                                                     // DRt_DSb
            //-----------------------------------------------------------------------------------------------------------
            DXt_DXb[OIDX][RIDX] = J_Ot_Thetd0*J_Thetd0_Rb + J_Ot_Thetd1*J_Thetd1_Rb;                           // DOt_DRb
            DXt_DXb[OIDX][OIDX] = J_Ot_Thetd0*J_Thetd0_Ob + J_Ot_Thetd1*J_Thetd1_Ob;                           // DOt_DOb
            DXt_DXb[OIDX][SIDX] = J_Ot_Thetd0*J_Thetd0_Sb + J_Ot_Thetd1*J_Thetd1_Sb;                           // DOt_DSb
            //-----------------------------------------------------------------------------------------------------------
            DXt_DXb[SIDX][RIDX] = J_St_Thetd0*J_Thetd0_Rb + J_St_Thetd1*J_Thetd1_Rb + J_St_Thetd2*J_Thetd2_Rb; // DSt_DRb
            DXt_DXb[SIDX][OIDX] = J_St_Thetd0*J_Thetd0_Ob + J_St_Thetd1*J_Thetd1_Ob + J_St_Thetd2*J_Thetd2_Ob; // DSt_DOb
            DXt_DXb[SIDX][SIDX] = J_St_Thetd0*J_Thetd0_Sb + J_St_Thetd1*J_Thetd1_Sb + J_St_Thetd2*J_Thetd2_Sb; // DSt_DSb
        }

        /* #endregion Processing the RO3 states ---------------------------------------------------------------------*/


        /* #region Processing the R3 states -------------------------------------------------------------------------*/
        
        // Performing interpolation on R3
        Matrix<T, 9, 9> LAM_PVAt = LMD(Dtau, CovPVAJerk).cast<T>();
        Matrix<T, 9, 9> PSI_PVAt = PSI(Dtau, CovPVAJerk).cast<T>();

        // Calculate the knot euclid states and put them in vector form
        Vec9T pvaa; pvaa << Xa.P, Xa.V, Xa.A;
        Vec9T pvab; pvab << Xb.P, Xb.V, Xb.A;
        Vec9T pvat = LAM_PVAt*pvaa + PSI_PVAt*pvab;

        Pt = pvat.block(0, 0, 3, 1);
        Vt = pvat.block(3, 0, 3, 1);
        At = pvat.block(6, 0, 3, 1);

        if(find_jacobian)
        {
            // Extract the blocks of R3 states
            Mat3T LAM_PVA11 = LAM_PVAt.block(0, 0, 3, 3); Mat3T LAM_PVA12 = LAM_PVAt.block(0, 3, 3, 3); Mat3T LAM_PVA13 = LAM_PVAt.block(0, 6, 3, 3);
            Mat3T LAM_PVA21 = LAM_PVAt.block(3, 0, 3, 3); Mat3T LAM_PVA22 = LAM_PVAt.block(3, 3, 3, 3); Mat3T LAM_PVA23 = LAM_PVAt.block(3, 6, 3, 3);
            Mat3T LAM_PVA31 = LAM_PVAt.block(6, 0, 3, 3); Mat3T LAM_PVA32 = LAM_PVAt.block(6, 3, 3, 3); Mat3T LAM_PVA33 = LAM_PVAt.block(6, 6, 3, 3);

            Mat3T PSI_PVA11 = PSI_PVAt.block(0, 0, 3, 3); Mat3T PSI_PVA12 = PSI_PVAt.block(0, 3, 3, 3); Mat3T PSI_PVA13 = PSI_PVAt.block(0, 6, 3, 3);
            Mat3T PSI_PVA21 = PSI_PVAt.block(3, 0, 3, 3); Mat3T PSI_PVA22 = PSI_PVAt.block(3, 3, 3, 3); Mat3T PSI_PVA23 = PSI_PVAt.block(3, 6, 3, 3);
            Mat3T PSI_PVA31 = PSI_PVAt.block(6, 0, 3, 3); Mat3T PSI_PVA32 = PSI_PVAt.block(6, 3, 3, 3); Mat3T PSI_PVA33 = PSI_PVAt.block(6, 6, 3, 3);

            // DPt_DPa
            DXt_DXa[PIDX][PIDX] = LAM_PVA11;
            // DPt_DVa
            DXt_DXa[PIDX][VIDX] = LAM_PVA12;
            // DPt_DAa
            DXt_DXa[PIDX][AIDX] = LAM_PVA13;
            
            // DVt_DPa
            DXt_DXa[VIDX][PIDX] = LAM_PVA21;
            // DVt_DVa
            DXt_DXa[VIDX][VIDX] = LAM_PVA22;
            // DVt_DAa
            DXt_DXa[VIDX][AIDX] = LAM_PVA23;

            // DAt_DPa
            DXt_DXa[AIDX][PIDX] = LAM_PVA31;
            // DAt_DVa
            DXt_DXa[AIDX][VIDX] = LAM_PVA32;
            // DAt_DAa
            DXt_DXa[AIDX][AIDX] = LAM_PVA33;

            // DPt_DPb
            DXt_DXb[PIDX][PIDX] = PSI_PVA11;
            // DRt_DPb
            DXt_DXb[PIDX][VIDX] = PSI_PVA12;
            // DRt_DAb
            DXt_DXb[PIDX][AIDX] = PSI_PVA13;

            // DVt_DPb
            DXt_DXb[VIDX][PIDX] = PSI_PVA21;
            // DVt_DVb
            DXt_DXb[VIDX][VIDX] = PSI_PVA22;
            // DVt_DAb
            DXt_DXb[VIDX][AIDX] = PSI_PVA23;
            
            // DAt_DPb
            DXt_DXb[AIDX][PIDX] = PSI_PVA31;
            // DAt_DVb
            DXt_DXb[AIDX][VIDX] = PSI_PVA32;
            // DAt_DAb
            DXt_DXb[AIDX][AIDX] = PSI_PVA33;
        }

        /* #endregion Processing the R3 states ----------------------------------------------------------------------*/
    
    }

    template <class T = double>
    void ComputeXtAndJacobiansSE3(const GPState<T> &Xa,
                                  const GPState<T> &Xb,
                                        GPState<T> &Xt,
                                  vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXa,
                                  vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXb,
                                  bool find_jacobian = true
                                 ) const
    {
        // Local index for the states in the state vector
        const int RIDX = 0;
        const int OIDX = 1;
        const int SIDX = 2;
        const int PIDX = 3;
        const int VIDX = 4;
        const int AIDX = 5;

        using SO3T   = Sophus::SO3<T>;
        using SE3T   = Sophus::SE3<T>;
        using Vec3T  = Eigen::Matrix<T, 3, 1>;
        using Vec6T  = Eigen::Matrix<T, 6, 1>;
        using Mat3T  = Eigen::Matrix<T, 3, 3>;
        using Mat6T  = Eigen::Matrix<T, 6, 6>;
        using MatLT  = Eigen::Matrix<T, 3, 6>;   // L is for long T is for tall
        using MatTT  = Eigen::Matrix<T, 6, 3>;   // L is for long T is for tall

        double Dtau = (Xt.t - Xa.t)/(Xb.t - Xa.t)*Dt;

        // Prepare the the mixer matrixes
        Matrix<T, 18, 18> LAM_TTWt = LMD(Dtau, CovTTWJerk).cast<T>();
        Matrix<T, 18, 18> PSI_TTWt = PSI(Dtau, CovTTWJerk).cast<T>();

        // Find the global 6DOF states

        SE3T Tfa; Vec6T Twa; Vec6T Wra;
        Xa.GetTUW(Tfa, Twa, Wra);

        SE3T Tfb; Vec6T Twb; Vec6T Wrb;
        Xb.GetTUW(Tfb, Twb, Wrb);

        SE3T Tfab = Tfa.inverse()*Tfb;

        // Calculate the local variable at the two ends
        Vec6T Xiad0 = Vec6T::Zero();
        Vec6T Xiad1; Xiad1 << Twa;
        Vec6T Xiad2; Xiad2 << Wra;

        Vec6T Xib = SE3Log(Tfab);
        Vec6T Xibd0 = Xib;
        Vec6T Xibd1;
        Vec6T Xibd2;

        Mat6T JrInv_Xib;         JrInv_Xib.setZero();
        Mat6T Hp1_XibTwb;        Hp1_XibTwb.setZero();
        Mat6T Hp1_XibWrb;        Hp1_XibWrb.setZero();
        Mat6T Lp11_XibTwbXibd1;  Lp11_XibTwbXibd1.setZero();
        Mat6T Lp12_XibTwbXibd1;  Lp12_XibTwbXibd1.setZero();

        // Populate the matrices related to Xib
        Get_JrInvHpLp<T>(Xib, Twb, Wrb, Xibd1, Xibd2, JrInv_Xib, Hp1_XibTwb, Hp1_XibWrb, Lp11_XibTwbXibd1, Lp12_XibTwbXibd1);

        // Stack the local variables in vector form
        Matrix<T, 18, 1> gammaa; gammaa << Xiad0, Xiad1, Xiad2;
        Matrix<T, 18, 1> gammab; gammab << Xibd0, Xibd1, Xibd2;
        // Mix the knots to get the interpolated states
        Matrix<T, 18, 1> gammat = LAM_TTWt*gammaa + PSI_TTWt*gammab;

        // Extract the interpolated local states
        Vec6T Xitd0 = gammat.block(0,  0, 6, 1);
        Vec6T Xitd1 = gammat.block(6,  0, 6, 1);
        Vec6T Xitd2 = gammat.block(12, 0, 6, 1);

        // Do all jacobians needed for L4-L3 interface 
        Mat6T Jr_Xit;
        Mat6T H1_XitXitd1;
        Mat6T H1_XitXitd2;
        Mat6T L11_XitXitd1Xitd1;
        Mat6T L12_XitXitd1Xitd1;

        // Populate the matrices related to Xit
        Get_JHL<T>(Xitd0, Xitd1, Xitd2, Jr_Xit, H1_XitXitd1, H1_XitXitd2, L11_XitXitd1Xitd1, L12_XitXitd1Xitd1);

        SE3T Exp_Xit = SE3Exp(Xitd0);

        SE3T  Tft = Tfa*Exp_Xit;
        Vec6T Twt = Jr_Xit*Xitd1;
        Vec6T Wrt = Jr_Xit*Xitd2 + H1_XitXitd1*Xitd1;

        // Get the interpolated states as variable
        Xt = GPState<T>(Xt.t, Tft, Twt, Wrt);

        if (find_jacobian)
        {
            // Calculate the Jacobian
            DXt_DXa = vector<vector<Mat3T>>(6, vector<Mat3T>(6, Mat3T::Zero()));
            DXt_DXb = vector<vector<Mat3T>>(6, vector<Mat3T>(6, Mat3T::Zero()));

            // Jacobians from L2 to L1
            Mat6T  J_Xiad1_Twa = Mat6T::Identity();
            Mat6T  J_Xiad2_Wra = Mat6T::Identity();

            Mat6T  J_Xibd0_Tfa = -JrInv((-Xibd0).eval());    // Technically we can use JrInv_Xib*Tab.Adj().inverse().matrix() but the order of The and Rho in Sophus is different
            Mat6T &J_Xibd0_Tfb =  JrInv_Xib;

            Mat6T  J_Xibd1_Tfa = Hp1_XibTwb*J_Xibd0_Tfa;
            Mat6T  J_Xibd1_Tfb = Hp1_XibTwb*J_Xibd0_Tfb;
            Mat6T &J_Xibd1_Twb = JrInv_Xib;

            Mat6T  J_Xibd2_Tfa = Hp1_XibWrb*J_Xibd0_Tfa + Hp1_XibTwb*J_Xibd1_Tfa + Lp11_XibTwbXibd1*J_Xibd0_Tfa;
            Mat6T  J_Xibd2_Tfb = Hp1_XibWrb*J_Xibd0_Tfb + Hp1_XibTwb*J_Xibd1_Tfb + Lp11_XibTwbXibd1*J_Xibd0_Tfb;
            Mat6T  J_Xibd2_Twb = Lp12_XibTwbXibd1 + Hp1_XibTwb*J_Xibd1_Twb;
            Mat6T &J_Xibd2_Wrb = JrInv_Xib;


            // Jacobians from L3 to L2
            Mat6T J_Xitd0_Xiad0 = LAM_TTWt.block(0,  0, 6, 6); Mat6T J_Xitd0_Xiad1 = LAM_TTWt.block(0,  6, 6, 6); Mat6T J_Xitd0_Xiad2 = LAM_TTWt.block(0,  12, 6, 6);
            Mat6T J_Xitd1_Xiad0 = LAM_TTWt.block(6,  0, 6, 6); Mat6T J_Xitd1_Xiad1 = LAM_TTWt.block(6,  6, 6, 6); Mat6T J_Xitd1_Xiad2 = LAM_TTWt.block(6,  12, 6, 6);
            Mat6T J_Xitd2_Xiad0 = LAM_TTWt.block(12, 0, 6, 6); Mat6T J_Xitd2_Xiad1 = LAM_TTWt.block(12, 6, 6, 6); Mat6T J_Xitd2_Xiad2 = LAM_TTWt.block(12, 12, 6, 6);
            Mat6T J_Xitd0_Xibd0 = PSI_TTWt.block(0,  0, 6, 6); Mat6T J_Xitd0_Xibd1 = PSI_TTWt.block(0,  6, 6, 6); Mat6T J_Xitd0_Xibd2 = PSI_TTWt.block(0,  12, 6, 6);
            Mat6T J_Xitd1_Xibd0 = PSI_TTWt.block(6,  0, 6, 6); Mat6T J_Xitd1_Xibd1 = PSI_TTWt.block(6,  6, 6, 6); Mat6T J_Xitd1_Xibd2 = PSI_TTWt.block(6,  12, 6, 6);
            Mat6T J_Xitd2_Xibd0 = PSI_TTWt.block(12, 0, 6, 6); Mat6T J_Xitd2_Xibd1 = PSI_TTWt.block(12, 6, 6, 6); Mat6T J_Xitd2_Xibd2 = PSI_TTWt.block(12, 12, 6, 6);


            // Jacobians from L3 to L1
            Mat6T J_Xitd0_Tfa = J_Xitd0_Xibd0*J_Xibd0_Tfa + J_Xitd0_Xibd1*J_Xibd1_Tfa + J_Xitd0_Xibd2*J_Xibd2_Tfa;
            Mat6T J_Xitd0_Tfb = J_Xitd0_Xibd0*J_Xibd0_Tfb + J_Xitd0_Xibd1*J_Xibd1_Tfb + J_Xitd0_Xibd2*J_Xibd2_Tfb;
            Mat6T J_Xitd0_Twa = J_Xitd0_Xiad1*J_Xiad1_Twa;
            Mat6T J_Xitd0_Twb = J_Xitd0_Xibd1*J_Xibd1_Twb + J_Xitd0_Xibd2*J_Xibd2_Twb;
            Mat6T J_Xitd0_Wra = J_Xitd0_Xiad2*J_Xiad2_Wra;
            Mat6T J_Xitd0_Wrb = J_Xitd0_Xibd2*J_Xibd2_Wrb;

            Mat6T J_Xitd1_Tfa = J_Xitd1_Xibd0*J_Xibd0_Tfa + J_Xitd1_Xibd1*J_Xibd1_Tfa + J_Xitd1_Xibd2*J_Xibd2_Tfa;
            Mat6T J_Xitd1_Tfb = J_Xitd1_Xibd0*J_Xibd0_Tfb + J_Xitd1_Xibd1*J_Xibd1_Tfb + J_Xitd1_Xibd2*J_Xibd2_Tfb;
            Mat6T J_Xitd1_Twa = J_Xitd1_Xiad1*J_Xiad1_Twa;
            Mat6T J_Xitd1_Twb = J_Xitd1_Xibd1*J_Xibd1_Twb + J_Xitd1_Xibd2*J_Xibd2_Twb;
            Mat6T J_Xitd1_Wra = J_Xitd1_Xiad2*J_Xiad2_Wra;
            Mat6T J_Xitd1_Wrb = J_Xitd1_Xibd2*J_Xibd2_Wrb;

            Mat6T J_Xitd2_Tfa = J_Xitd2_Xibd0*J_Xibd0_Tfa + J_Xitd2_Xibd1*J_Xibd1_Tfa + J_Xitd2_Xibd2*J_Xibd2_Tfa;
            Mat6T J_Xitd2_Tfb = J_Xitd2_Xibd0*J_Xibd0_Tfb + J_Xitd2_Xibd1*J_Xibd1_Tfb + J_Xitd2_Xibd2*J_Xibd2_Tfb;
            Mat6T J_Xitd2_Twa = J_Xitd2_Xiad1*J_Xiad1_Twa;
            Mat6T J_Xitd2_Twb = J_Xitd2_Xibd1*J_Xibd1_Twb + J_Xitd2_Xibd2*J_Xibd2_Twb;
            Mat6T J_Xitd2_Wra = J_Xitd2_Xiad2*J_Xiad2_Wra;
            Mat6T J_Xitd2_Wrb = J_Xitd2_Xibd2*J_Xibd2_Wrb;

            
            // Jacobians from L4 to L3
            Mat6T &J_Tft_Xitd0 = Jr_Xit;
            Mat6T &J_Twt_Xitd0 = H1_XitXitd1;
            Mat6T &J_Twt_Xitd1 = Jr_Xit;
            Mat6T  J_Wrt_Xitd0 = H1_XitXitd2 + L11_XitXitd1Xitd1;
            Mat6T  J_Wrt_Xitd1 = L12_XitXitd1Xitd1 + H1_XitXitd1;
            Mat6T &J_Wrt_Xitd2 = Jr_Xit;


            // Jacobian from L4 to L1
            Mat6T J_Tft_Tfa = SE3AdjInv(Exp_Xit) + J_Tft_Xitd0*J_Xitd0_Tfa;                                // DTft_DTfa
            Mat6T J_Tft_Twa = J_Tft_Xitd0*J_Xitd0_Twa;                                                     // DTft_DTwa
            Mat6T J_Tft_Wra = J_Tft_Xitd0*J_Xitd0_Wra;                                                     // DTft_DWra
            //---------------------------------------------------------------------------------------------------------
            Mat6T J_Twt_Tfa = J_Twt_Xitd0*J_Xitd0_Tfa + J_Twt_Xitd1*J_Xitd1_Tfa;                           // DTwt_DTfa
            Mat6T J_Twt_Twa = J_Twt_Xitd0*J_Xitd0_Twa + J_Twt_Xitd1*J_Xitd1_Twa;                           // DTwt_DTwa
            Mat6T J_Twt_Wra = J_Twt_Xitd0*J_Xitd0_Wra + J_Twt_Xitd1*J_Xitd1_Wra;                           // DTwt_DWra
            //---------------------------------------------------------------------------------------------------------
            Mat6T J_Wrt_Tfa = J_Wrt_Xitd0*J_Xitd0_Tfa + J_Wrt_Xitd1*J_Xitd1_Tfa + J_Wrt_Xitd2*J_Xitd2_Tfa; // DTwt_DTfa
            Mat6T J_Wrt_Twa = J_Wrt_Xitd0*J_Xitd0_Twa + J_Wrt_Xitd1*J_Xitd1_Twa + J_Wrt_Xitd2*J_Xitd2_Twa; // DTwt_DTwa
            Mat6T J_Wrt_Wra = J_Wrt_Xitd0*J_Xitd0_Wra + J_Wrt_Xitd1*J_Xitd1_Wra + J_Wrt_Xitd2*J_Xitd2_Wra; // DTwt_DWra
            //---------------------------------------------------------------------------------------------------------
            Mat6T J_Tft_Tfb = J_Tft_Xitd0*J_Xitd0_Tfb;                                                     // DTft_DTfb
            Mat6T J_Tft_Twb = J_Tft_Xitd0*J_Xitd0_Twb;                                                     // DTft_DTwb
            Mat6T J_Tft_Wrb = J_Tft_Xitd0*J_Xitd0_Wrb;                                                     // DTft_DWrb
            //---------------------------------------------------------------------------------------------------------
            Mat6T J_Twt_Tfb = J_Twt_Xitd0*J_Xitd0_Tfb + J_Twt_Xitd1*J_Xitd1_Tfb;                           // DTwt_DTfb
            Mat6T J_Twt_Twb = J_Twt_Xitd0*J_Xitd0_Twb + J_Twt_Xitd1*J_Xitd1_Twb;                           // DTwt_DTwb
            Mat6T J_Twt_Wrb = J_Twt_Xitd0*J_Xitd0_Wrb + J_Twt_Xitd1*J_Xitd1_Wrb;                           // DTwt_DWrb
            //---------------------------------------------------------------------------------------------------------
            Mat6T J_Wrt_Tfb = J_Wrt_Xitd0*J_Xitd0_Tfb + J_Wrt_Xitd1*J_Xitd1_Tfb + J_Wrt_Xitd2*J_Xitd2_Tfb; // DTwt_DTfb
            Mat6T J_Wrt_Twb = J_Wrt_Xitd0*J_Xitd0_Twb + J_Wrt_Xitd1*J_Xitd1_Twb + J_Wrt_Xitd2*J_Xitd2_Twb; // DTwt_DTwb
            Mat6T J_Wrt_Wrb = J_Wrt_Xitd0*J_Xitd0_Wrb + J_Wrt_Xitd1*J_Xitd1_Wrb + J_Wrt_Xitd2*J_Xitd2_Wrb; // DTwt_DWrb

            
            // Jacobian from L5 to L4
            MatLT U; U.setZero(); U.block(0, 0, 3, 3) = Mat3T::Identity();
            MatLT D; D.setZero(); D.block(0, 3, 3, 3) = Mat3T::Identity();
            MatTT Utp = U.transpose();
            MatTT Dtp = D.transpose();

            const SO3T  &Rt = Xt.R;       const SO3T  &Ra = Xa.R;       const SO3T  &Rb = Xb.R;
            const Vec3T &Ot = Xt.O;       const Vec3T &Oa = Xa.O;       const Vec3T &Ob = Xb.O;
            const Vec3T &St = Xt.S;       const Vec3T &Sa = Xa.S;       const Vec3T &Sb = Xb.S;
            const Vec3T &Pt = Xt.P;       const Vec3T &Pa = Xa.P;       const Vec3T &Pb = Xb.P;
            const Vec3T &Vt = Xt.V;       const Vec3T &Va = Xa.V;       const Vec3T &Vb = Xb.V;
            const Vec3T &At = Xt.A;       const Vec3T &Aa = Xa.A;       const Vec3T &Ab = Xb.A;
            const Vec3T Nt = Twt.tail(3); const Vec3T Na = Twa.tail(3); const Vec3T Nb = Twb.tail(3);
            const Vec3T Bt = Wrt.tail(3); const Vec3T Ba = Wra.tail(3); const Vec3T Bb = Wrb.tail(3);

            Mat3T Rtmat  = Rt.matrix();
            MatLT RtmatD = Rtmat*D;
            Mat3T RtmatNthat = Rtmat*SO3T::hat(Nt);

            MatLT &J_Rt_Tft = U;    const MatLT &J_Pt_Tft = RtmatD;
            MatLT &J_Ot_Twt = U;    const MatLT &J_Vt_Twt = RtmatD;
            MatLT &J_St_Wrt = U;    const MatLT &J_At_Wrt = RtmatD;

            Mat3T  J_Vt_Rt  = -RtmatNthat;
            MatLT  J_Vt_Tft =  J_Vt_Rt*J_Rt_Tft;

            Mat3T  J_At_Rt  = -Rtmat*(SO3T::hat(Bt) + SO3T::hat(SO3T::hat(Ot)*Nt));
            MatLT  J_At_Tft =  J_At_Rt*J_Rt_Tft;
            
            Mat3T &J_At_Ot  =  J_Vt_Rt;
            MatLT  J_At_Twt =  J_At_Ot*J_Ot_Twt + Rtmat*SO3T::hat(Ot)*D;
            

            // Jacobian from L1 to L0
            Mat3T hat_Oa = SO3T::hat(Oa);                                                       Mat3T hat_Ob = SO3T::hat(Ob);                           
            Mat3T Ratp = Ra.inverse().matrix();                                                 Mat3T Rbtp = Rb.inverse().matrix();                     
            Mat3T hat_RatpVa = SO3T::hat(Ratp*Va);                                              Mat3T hat_RbtpVb = SO3T::hat(Rbtp*Vb);                  
            Mat3T hat_RatpAa = SO3T::hat(Ratp*Aa);                                              Mat3T hat_RbtpAb = SO3T::hat(Rbtp*Ab);                  
                                                                                                                                                        
            MatTT J_Tfa_Ra =  Utp;                                                              MatTT J_Tfb_Rb =  Utp;                                  
            MatTT J_Tfa_Pa =  Dtp*Ratp;                                                         MatTT J_Tfb_Pb =  Dtp*Rbtp;                             
                                                                                                                                                        
            MatTT J_Twa_Oa =  Utp;                                                              MatTT J_Twb_Ob =  Utp;                                  
            MatTT J_Twa_Ra =  Dtp*hat_RatpVa;                                                   MatTT J_Twb_Rb =  Dtp*hat_RbtpVb;                       
            MatTT J_Twa_Va =  Dtp*Ratp;                                                         MatTT J_Twb_Vb =  Dtp*Rbtp;                             
                                                                                                                                                        
            MatTT J_Wra_Sa =  Utp;                                                              MatTT J_Wrb_Sb =  Utp;                                  
            MatTT J_Wra_Ra =  Dtp*(hat_RatpAa - hat_Oa*hat_RatpVa);                             MatTT J_Wrb_Rb =  Dtp*(hat_RbtpAb - hat_Ob*hat_RbtpVb); 
            MatTT J_Wra_Aa =  Dtp*Ratp;                                                         MatTT J_Wrb_Ab =  Dtp*Rbtp;                             
            MatTT J_Wra_Oa =  Dtp*hat_RatpVa;                                                   MatTT J_Wrb_Ob =  Dtp*hat_RbtpVb;                       
            MatTT J_Wra_Va = -Dtp*hat_Oa*Ratp;                                                  MatTT J_Wrb_Vb = -Dtp*hat_Ob*Rbtp;                      


            // Jacobians from L4 to L0
            MatTT J_Tft_Ra = J_Tft_Tfa*J_Tfa_Ra + J_Tft_Twa*J_Twa_Ra + J_Tft_Wra*J_Wra_Ra;      MatTT J_Tft_Rb = J_Tft_Tfb*J_Tfb_Rb + J_Tft_Twb*J_Twb_Rb + J_Tft_Wrb*J_Wrb_Rb;
            MatTT J_Tft_Oa =                      J_Tft_Twa*J_Twa_Oa + J_Tft_Wra*J_Wra_Oa;      MatTT J_Tft_Ob =                      J_Tft_Twb*J_Twb_Ob + J_Tft_Wrb*J_Wrb_Ob;
            MatTT J_Tft_Sa =                                           J_Tft_Wra*J_Wra_Sa;      MatTT J_Tft_Sb =                                           J_Tft_Wrb*J_Wrb_Sb;
            MatTT J_Tft_Pa = J_Tft_Tfa*J_Tfa_Pa                                          ;      MatTT J_Tft_Pb = J_Tft_Tfb*J_Tfb_Pb                                          ;
            MatTT J_Tft_Va =                      J_Tft_Twa*J_Twa_Va + J_Tft_Wra*J_Wra_Va;      MatTT J_Tft_Vb =                      J_Tft_Twb*J_Twb_Vb + J_Tft_Wrb*J_Wrb_Vb;
            MatTT J_Tft_Aa =                                           J_Tft_Wra*J_Wra_Aa;      MatTT J_Tft_Ab =                                           J_Tft_Wrb*J_Wrb_Ab;

            MatTT J_Twt_Ra = J_Twt_Tfa*J_Tfa_Ra + J_Twt_Twa*J_Twa_Ra + J_Twt_Wra*J_Wra_Ra;      MatTT J_Twt_Rb = J_Twt_Tfb*J_Tfb_Rb + J_Twt_Twb*J_Twb_Rb + J_Twt_Wrb*J_Wrb_Rb;
            MatTT J_Twt_Oa =                      J_Twt_Twa*J_Twa_Oa + J_Twt_Wra*J_Wra_Oa;      MatTT J_Twt_Ob =                      J_Twt_Twb*J_Twb_Ob + J_Twt_Wrb*J_Wrb_Ob;
            MatTT J_Twt_Sa =                                           J_Twt_Wra*J_Wra_Sa;      MatTT J_Twt_Sb =                                           J_Twt_Wrb*J_Wrb_Sb;
            MatTT J_Twt_Pa = J_Twt_Tfa*J_Tfa_Pa                                          ;      MatTT J_Twt_Pb = J_Twt_Tfb*J_Tfb_Pb                                          ;
            MatTT J_Twt_Va =                      J_Twt_Twa*J_Twa_Va + J_Twt_Wra*J_Wra_Va;      MatTT J_Twt_Vb =                      J_Twt_Twb*J_Twb_Vb + J_Twt_Wrb*J_Wrb_Vb;
            MatTT J_Twt_Aa =                                           J_Twt_Wra*J_Wra_Aa;      MatTT J_Twt_Ab =                                           J_Twt_Wrb*J_Wrb_Ab;

            MatTT J_Wrt_Ra = J_Wrt_Tfa*J_Tfa_Ra + J_Wrt_Twa*J_Twa_Ra + J_Wrt_Wra*J_Wra_Ra;      MatTT J_Wrt_Rb = J_Wrt_Tfb*J_Tfb_Rb + J_Wrt_Twb*J_Twb_Rb + J_Wrt_Wrb*J_Wrb_Rb;
            MatTT J_Wrt_Oa =                      J_Wrt_Twa*J_Twa_Oa + J_Wrt_Wra*J_Wra_Oa;      MatTT J_Wrt_Ob =                      J_Wrt_Twb*J_Twb_Ob + J_Wrt_Wrb*J_Wrb_Ob;
            MatTT J_Wrt_Sa =                                           J_Wrt_Wra*J_Wra_Sa;      MatTT J_Wrt_Sb =                                           J_Wrt_Wrb*J_Wrb_Sb;
            MatTT J_Wrt_Pa = J_Wrt_Tfa*J_Tfa_Pa                                          ;      MatTT J_Wrt_Pb = J_Wrt_Tfb*J_Tfb_Pb                                          ;
            MatTT J_Wrt_Va =                      J_Wrt_Twa*J_Twa_Va + J_Wrt_Wra*J_Wra_Va;      MatTT J_Wrt_Vb =                      J_Wrt_Twb*J_Twb_Vb + J_Wrt_Wrb*J_Wrb_Vb;
            MatTT J_Wrt_Aa =                                           J_Wrt_Wra*J_Wra_Aa;      MatTT J_Wrt_Ab =                                           J_Wrt_Wrb*J_Wrb_Ab;


            // Jacobians from L5 to L0, dXt/dXa

            DXt_DXa[RIDX][RIDX] = J_Rt_Tft*J_Tft_Ra;                                            DXt_DXb[RIDX][RIDX] = J_Rt_Tft*J_Tft_Rb;                                         // DRt_DRa
            DXt_DXa[RIDX][OIDX] = J_Rt_Tft*J_Tft_Oa;                                            DXt_DXb[RIDX][OIDX] = J_Rt_Tft*J_Tft_Ob;                                         // DRt_DOa
            DXt_DXa[RIDX][SIDX] = J_Rt_Tft*J_Tft_Sa;                                            DXt_DXb[RIDX][SIDX] = J_Rt_Tft*J_Tft_Sb;                                         // DRt_DSa
            DXt_DXa[RIDX][PIDX] = J_Rt_Tft*J_Tft_Pa;                                            DXt_DXb[RIDX][PIDX] = J_Rt_Tft*J_Tft_Pb;                                         // DRt_DPa
            DXt_DXa[RIDX][VIDX] = J_Rt_Tft*J_Tft_Va;                                            DXt_DXb[RIDX][VIDX] = J_Rt_Tft*J_Tft_Vb;                                         // DRt_DVa
            DXt_DXa[RIDX][AIDX] = J_Rt_Tft*J_Tft_Aa;                                            DXt_DXb[RIDX][AIDX] = J_Rt_Tft*J_Tft_Ab;                                         // DRt_DAa
            //------------------------------------------------------------------------------    //-----------------------------------------------------------------------------------------
            DXt_DXa[OIDX][RIDX] =                     J_Ot_Twt*J_Twt_Ra;                        DXt_DXb[OIDX][RIDX] =                     J_Ot_Twt*J_Twt_Rb;                     // DOt_DRa
            DXt_DXa[OIDX][OIDX] =                     J_Ot_Twt*J_Twt_Oa;                        DXt_DXb[OIDX][OIDX] =                     J_Ot_Twt*J_Twt_Ob;                     // DOt_DOa
            DXt_DXa[OIDX][SIDX] =                     J_Ot_Twt*J_Twt_Sa;                        DXt_DXb[OIDX][SIDX] =                     J_Ot_Twt*J_Twt_Sb;                     // DOt_DSa
            DXt_DXa[OIDX][PIDX] =                     J_Ot_Twt*J_Twt_Pa;                        DXt_DXb[OIDX][PIDX] =                     J_Ot_Twt*J_Twt_Pb;                     // DOt_DPa
            DXt_DXa[OIDX][VIDX] =                     J_Ot_Twt*J_Twt_Va;                        DXt_DXb[OIDX][VIDX] =                     J_Ot_Twt*J_Twt_Vb;                     // DOt_DVa
            DXt_DXa[OIDX][AIDX] =                     J_Ot_Twt*J_Twt_Aa;                        DXt_DXb[OIDX][AIDX] =                     J_Ot_Twt*J_Twt_Ab;                     // DOt_DAa
            //------------------------------------------------------------------------------    //-----------------------------------------------------------------------------------------
            DXt_DXa[SIDX][RIDX] =                                         J_St_Wrt*J_Wrt_Ra;    DXt_DXb[SIDX][RIDX] =                                         J_St_Wrt*J_Wrt_Rb; // DSt_DRa
            DXt_DXa[SIDX][OIDX] =                                         J_St_Wrt*J_Wrt_Oa;    DXt_DXb[SIDX][OIDX] =                                         J_St_Wrt*J_Wrt_Ob; // DSt_DOa
            DXt_DXa[SIDX][SIDX] =                                         J_St_Wrt*J_Wrt_Sa;    DXt_DXb[SIDX][SIDX] =                                         J_St_Wrt*J_Wrt_Sb; // DSt_DSa
            DXt_DXa[SIDX][PIDX] =                                         J_St_Wrt*J_Wrt_Pa;    DXt_DXb[SIDX][PIDX] =                                         J_St_Wrt*J_Wrt_Pb; // DSt_DPa
            DXt_DXa[SIDX][VIDX] =                                         J_St_Wrt*J_Wrt_Va;    DXt_DXb[SIDX][VIDX] =                                         J_St_Wrt*J_Wrt_Vb; // DSt_DVa
            DXt_DXa[SIDX][AIDX] =                                         J_St_Wrt*J_Wrt_Aa;    DXt_DXb[SIDX][AIDX] =                                         J_St_Wrt*J_Wrt_Ab; // DSt_DAa
            //------------------------------------------------------------------------------    //-----------------------------------------------------------------------------------------------------------------------------------------------------------
            DXt_DXa[PIDX][RIDX] = J_Pt_Tft*J_Tft_Ra;                                            DXt_DXb[PIDX][RIDX] = J_Pt_Tft*J_Tft_Rb;                                         // DRt_DRb
            DXt_DXa[PIDX][OIDX] = J_Pt_Tft*J_Tft_Oa;                                            DXt_DXb[PIDX][OIDX] = J_Pt_Tft*J_Tft_Ob;                                         // DRt_DOb
            DXt_DXa[PIDX][SIDX] = J_Pt_Tft*J_Tft_Sa;                                            DXt_DXb[PIDX][SIDX] = J_Pt_Tft*J_Tft_Sb;                                         // DRt_DSb
            DXt_DXa[PIDX][PIDX] = J_Pt_Tft*J_Tft_Pa;                                            DXt_DXb[PIDX][PIDX] = J_Pt_Tft*J_Tft_Pb;                                         // DRt_DPb
            DXt_DXa[PIDX][VIDX] = J_Pt_Tft*J_Tft_Va;                                            DXt_DXb[PIDX][VIDX] = J_Pt_Tft*J_Tft_Vb;                                         // DRt_DVb
            DXt_DXa[PIDX][AIDX] = J_Pt_Tft*J_Tft_Aa;                                            DXt_DXb[PIDX][AIDX] = J_Pt_Tft*J_Tft_Ab;                                         // DRt_DAb
            //------------------------------------------------------------------------------    //-----------------------------------------------------------------------------------------------------------------------------------------------------------
            DXt_DXa[VIDX][RIDX] = J_Vt_Tft*J_Tft_Ra + J_Vt_Twt*J_Twt_Ra;                        DXt_DXb[VIDX][RIDX] = J_Vt_Tft*J_Tft_Rb + J_Vt_Twt*J_Twt_Rb;                     // DOt_DRb
            DXt_DXa[VIDX][OIDX] = J_Vt_Tft*J_Tft_Oa + J_Vt_Twt*J_Twt_Oa;                        DXt_DXb[VIDX][OIDX] = J_Vt_Tft*J_Tft_Ob + J_Vt_Twt*J_Twt_Ob;                     // DOt_DOb
            DXt_DXa[VIDX][SIDX] = J_Vt_Tft*J_Tft_Sa + J_Vt_Twt*J_Twt_Sa;                        DXt_DXb[VIDX][SIDX] = J_Vt_Tft*J_Tft_Sb + J_Vt_Twt*J_Twt_Sb;                     // DOt_DSb
            DXt_DXa[VIDX][PIDX] = J_Vt_Tft*J_Tft_Pa + J_Vt_Twt*J_Twt_Pa;                        DXt_DXb[VIDX][PIDX] = J_Vt_Tft*J_Tft_Pb + J_Vt_Twt*J_Twt_Pb;                     // DOt_DPb
            DXt_DXa[VIDX][VIDX] = J_Vt_Tft*J_Tft_Va + J_Vt_Twt*J_Twt_Va;                        DXt_DXb[VIDX][VIDX] = J_Vt_Tft*J_Tft_Vb + J_Vt_Twt*J_Twt_Vb;                     // DOt_DVb
            DXt_DXa[VIDX][AIDX] = J_Vt_Tft*J_Tft_Aa + J_Vt_Twt*J_Twt_Aa;                        DXt_DXb[VIDX][AIDX] = J_Vt_Tft*J_Tft_Ab + J_Vt_Twt*J_Twt_Ab;                     // DOt_DAb
            //------------------------------------------------------------------------------    //-----------------------------------------------------------------------------------------------------------------------------------------------------------
            DXt_DXa[AIDX][RIDX] = J_At_Tft*J_Tft_Ra + J_At_Twt*J_Twt_Ra + J_At_Wrt*J_Wrt_Ra;    DXt_DXb[AIDX][RIDX] = J_At_Tft*J_Tft_Rb + J_At_Twt*J_Twt_Rb + J_At_Wrt*J_Wrt_Rb; // DOt_DRb
            DXt_DXa[AIDX][OIDX] = J_At_Tft*J_Tft_Oa + J_At_Twt*J_Twt_Oa + J_At_Wrt*J_Wrt_Oa;    DXt_DXb[AIDX][OIDX] = J_At_Tft*J_Tft_Ob + J_At_Twt*J_Twt_Ob + J_At_Wrt*J_Wrt_Ob; // DOt_DOb
            DXt_DXa[AIDX][SIDX] = J_At_Tft*J_Tft_Sa + J_At_Twt*J_Twt_Sa + J_At_Wrt*J_Wrt_Sa;    DXt_DXb[AIDX][SIDX] = J_At_Tft*J_Tft_Sb + J_At_Twt*J_Twt_Sb + J_At_Wrt*J_Wrt_Sb; // DOt_DSb
            DXt_DXa[AIDX][PIDX] = J_At_Tft*J_Tft_Pa + J_At_Twt*J_Twt_Pa + J_At_Wrt*J_Wrt_Pa;    DXt_DXb[AIDX][PIDX] = J_At_Tft*J_Tft_Pb + J_At_Twt*J_Twt_Pb + J_At_Wrt*J_Wrt_Pb; // DOt_DPb
            DXt_DXa[AIDX][VIDX] = J_At_Tft*J_Tft_Va + J_At_Twt*J_Twt_Va + J_At_Wrt*J_Wrt_Va;    DXt_DXb[AIDX][VIDX] = J_At_Tft*J_Tft_Vb + J_At_Twt*J_Twt_Vb + J_At_Wrt*J_Wrt_Vb; // DOt_DVb
            DXt_DXa[AIDX][AIDX] = J_At_Tft*J_Tft_Aa + J_At_Twt*J_Twt_Aa + J_At_Wrt*J_Wrt_Aa;    DXt_DXb[AIDX][AIDX] = J_At_Tft*J_Tft_Ab + J_At_Twt*J_Twt_Ab + J_At_Wrt*J_Wrt_Ab; // DOt_DAb
        }

    }

    template <class T = double>
    void ComputeMotionPriorFactor(const GPState<T> &Xa,
                                  const GPState<T> &Xb,
                                  Eigen::Matrix<T, STATE_DIM, 1> &residual,
                                  vector<Eigen::Matrix<T, STATE_DIM, 3>> &Dr_DXa,
                                  vector<Eigen::Matrix<T, STATE_DIM, 3>> &Dr_DXb,
                                  bool find_jacobian = true
                                 ) const
    {
        if(pose_representation == POSE_GROUP::SO3xR3)
            ComputeMotionPriorFactorSO3xR3(Xa, Xb, residual, Dr_DXa, Dr_DXb, find_jacobian);
        else if (pose_representation == POSE_GROUP::SE3)
            ComputeMotionPriorFactorSE3(Xa, Xb, residual, Dr_DXa, Dr_DXb, find_jacobian);
    }

    template <class T = double>
    void ComputeMotionPriorFactorSO3xR3(const GPState<T> &Xa,
                                        const GPState<T> &Xb,
                                        Eigen::Matrix<T, STATE_DIM, 1> &residual,
                                        vector<Eigen::Matrix<T, STATE_DIM, 3>> &Dr_DXa,
                                        vector<Eigen::Matrix<T, STATE_DIM, 3>> &Dr_DXb,
                                        bool find_jacobian = true
                                       ) const
    {
        // Local index for the states in the state vector
        const int RIDX = 0;
        const int OIDX = 1;
        const int SIDX = 2;
        const int PIDX = 3;
        const int VIDX = 4;
        const int AIDX = 5;

        using SO3T   = Sophus::SO3<T>;
        using SE3T   = Sophus::SE3<T>;
        using Vec3T  = Eigen::Matrix<T, 3, 1>;
        using Vec6T  = Eigen::Matrix<T, 6, 1>;
        using Vec9T  = Eigen::Matrix<T, 9, 1>;
        using Mat3T  = Eigen::Matrix<T, 3, 3>;
        using Mat6T  = Eigen::Matrix<T, 6, 6>;
        using MatLT  = Eigen::Matrix<T, 3, 6>;   // L is for long T is for tall
        using MatTT  = Eigen::Matrix<T, 6, 3>;   // L is for long T is for tall

        // Find the relative rotation
        SO3T Rab = Xa.R.inverse()*Xb.R;

        // Calculate the SO3 knots in relative form
        Vec3T Thead0 = Vec3T::Zero();
        Vec3T Thead1 = Xa.O;
        Vec3T Thead2 = Xa.S;

        // Find the local variable at tb and the associated Jacobians
        Vec3T Theb = Rab.log();
        Mat3T JrInvTheb = JrInv<T>(Theb);
        Mat3T Hp1ThebOb = DJrInvUV_DU(Theb, Xb.O);

        Vec3T Thebd0 = Theb;
        Vec3T Thebd1 = JrInvTheb*Xb.O;
        Vec3T Thebd2 = JrInvTheb*Xb.S + Hp1ThebOb*Thebd1;

        // Put them in vector form
        Vec9T gammaa; gammaa << Thead0, Thead1, Thead2;
        Vec9T gammab; gammab << Thebd0, Thebd1, Thebd2;

        Vec9T PVAa; PVAa << Xa.P, Xa.V, Xa.A;
        Vec9T PVAb; PVAb << Xb.P, Xb.V, Xb.A;

        // Calculate the residual
        residual << gammab - Fmat*gammaa, PVAb - Fmat*PVAa;
        residual = sqrtW*residual;

        if (find_jacobian)
        {
            T Dtsq = T(Dt*Dt);
            Mat3T Eye = Mat3T::Identity();
            Mat3T DtI = Vec3T(T(Dt), T(Dt), T(Dt)).asDiagonal();
            T     DtsqDiv2 = 0.5*Dtsq;
            Mat3T DtsqDiv2I = Vec3T(DtsqDiv2, DtsqDiv2, DtsqDiv2).asDiagonal();

            // Reusable Jacobians
            Mat3T DTheb_DRa = -JrInvTheb*Rab.inverse().matrix();
            Mat3T DTheb_DRb =  JrInvTheb;

            Mat3T &DThebd1_DTheb = Hp1ThebOb;
            Mat3T &DJrInvThebOb_DTheb = Hp1ThebOb;
            Mat3T DThebd1_DRa = DThebd1_DTheb*DTheb_DRa;
            Mat3T DThebd1_DRb = DThebd1_DTheb*DTheb_DRb;

            Mat3T DJrInvThebSb_DTheb = DJrInvUV_DU(Theb, Xb.S);
            Mat3T DDJrInvThebObThebd1_DThebDTheb = DDJrInvUVW_DUDU(Theb, Xb.O, Thebd1);
            
            Mat3T DThebd2_DTheb = DJrInvThebSb_DTheb + DDJrInvThebObThebd1_DThebDTheb + DJrInvThebOb_DTheb*DJrInvThebOb_DTheb;
            Mat3T DThebd2_DRa = DThebd2_DTheb*DTheb_DRa;
            Mat3T DThebd2_DRb = DThebd2_DTheb*DTheb_DRb;

            Mat3T DDJrInvThebObThebd1_DThebDOb = DDJrInvUVW_DUDV(Theb, Xb.O, Thebd1);
            Mat3T DThebd2_DOb = DDJrInvThebObThebd1_DThebDOb + DJrInvThebOb_DTheb*JrInvTheb;

            // Set the jacobian on Ra                                    // Set the jacobian on Rb
            Dr_DXa[RIDX].template block<3, 3>(0, 0) = DTheb_DRa;         Dr_DXb[RIDX].template block<3, 3>(0, 0) = DTheb_DRb;
            Dr_DXa[RIDX].template block<3, 3>(3, 0) = DThebd1_DRa;       Dr_DXb[RIDX].template block<3, 3>(3, 0) = DThebd1_DRb;
            Dr_DXa[RIDX].template block<3, 3>(6, 0) = DThebd2_DRa;       Dr_DXb[RIDX].template block<3, 3>(6, 0) = DThebd2_DRb;
            Dr_DXa[RIDX] = sqrtW*Dr_DXa[RIDX];                           Dr_DXb[RIDX] = sqrtW*Dr_DXb[RIDX];

            // Set the jacobian on Oa                                    // Set the jacobian on Ob
            Dr_DXa[OIDX].template block<3, 3>(0, 0) = -DtI;              Dr_DXb[OIDX].template block<3, 3>(3, 0) = JrInvTheb;
            Dr_DXa[OIDX].template block<3, 3>(3, 0) = -Eye;              Dr_DXb[OIDX].template block<3, 3>(6, 0) = DThebd2_DOb;
            Dr_DXa[OIDX] = sqrtW*Dr_DXa[OIDX];                           Dr_DXb[OIDX] = sqrtW*Dr_DXb[OIDX];

            // Set the jacobian on Sa                                    // Set the jacobian on Sb
            Dr_DXa[SIDX].template block<3, 3>(0, 0) = -DtsqDiv2I;        Dr_DXb[SIDX].template block<3, 3>(6, 0) = JrInvTheb;
            Dr_DXa[SIDX].template block<3, 3>(3, 0) = -DtI;              Dr_DXb[SIDX] = sqrtW*Dr_DXb[SIDX];
            Dr_DXa[SIDX].template block<3, 3>(6, 0) = -Eye;
            Dr_DXa[SIDX] = sqrtW*Dr_DXa[SIDX];    
        
            // Set the jacobian on Pa                                    // Set the jacobian on Pb
            Dr_DXa[PIDX].template block<3, 3>(9,  0) = -Eye;             Dr_DXb[PIDX].template block<3, 3>(9, 0) = Eye;
            Dr_DXa[PIDX] = sqrtW*Dr_DXa[PIDX];                           Dr_DXb[PIDX] = sqrtW*Dr_DXb[PIDX];

            // Set the jacobian on Va                                    // Set the jacobian on Vb
            Dr_DXa[VIDX].template block<3, 3>(9,  0) = -DtI;             Dr_DXb[VIDX].template block<3, 3>(12, 0) = Eye;
            Dr_DXa[VIDX].template block<3, 3>(12, 0) = -Eye;             Dr_DXb[VIDX] = sqrtW*Dr_DXb[VIDX];
            Dr_DXa[VIDX] = sqrtW*Dr_DXa[VIDX];                             

            // Set the jacobian on Aa                                    // Set the jacobian on Ab
            Dr_DXa[AIDX].template block<3, 3>(9,  0) = -DtsqDiv2I;       Dr_DXb[AIDX].template block<3, 3>(15, 0) = Eye;
            Dr_DXa[AIDX].template block<3, 3>(12, 0) = -DtI;             Dr_DXb[AIDX] = sqrtW*Dr_DXb[AIDX];
            Dr_DXa[AIDX].template block<3, 3>(15, 0) = -Eye;
            Dr_DXa[AIDX] = sqrtW*Dr_DXa[AIDX];
        }
    }

    template <class T = double>
    void ComputeMotionPriorFactorSE3(const GPState<T> &Xa,
                                     const GPState<T> &Xb,
                                     Eigen::Matrix<T, STATE_DIM, 1> &residual,
                                     vector<Eigen::Matrix<T, STATE_DIM, 3>> &Dr_DXa,
                                     vector<Eigen::Matrix<T, STATE_DIM, 3>> &Dr_DXb,
                                     bool find_jacobian = true
                                    ) const
    {
        // Local index for the states in the state vector
        const int Ridx = 0;
        const int Oidx = 1;
        const int Sidx = 2;
        const int Pidx = 3;
        const int Vidx = 4;
        const int Aidx = 5;

        using SO3T   = Sophus::SO3<T>;
        using SE3T   = Sophus::SE3<T>;
        using Vec3T  = Eigen::Matrix<T, 3, 1>;
        using Vec6T  = Eigen::Matrix<T, 6, 1>;
        using Vec9T  = Eigen::Matrix<T, 9, 1>;
        using Mat3T  = Eigen::Matrix<T, 3, 3>;
        using Mat6T  = Eigen::Matrix<T, 6, 6>;
        using MatLT  = Eigen::Matrix<T, 3, 6>;   // L is for long T is for tall
        using MatTT  = Eigen::Matrix<T, 6, 3>;   // L is for long T is for tall
        
        // Find the global 6DOF states
        SE3T Tfa; Vec6T Twa; Vec6T Wra; Xa.GetTUW(Tfa, Twa, Wra);
        SE3T Tfb; Vec6T Twb; Vec6T Wrb; Xb.GetTUW(Tfb, Twb, Wrb);

        // Find the relative pose
        SE3T Tfab = Tfa.inverse()*Tfb;

        // Calculate the local variable at the two ends
        Vec6T Xiad0 = Vec6T::Zero();
        Vec6T Xiad1; Xiad1 << Twa;
        Vec6T Xiad2; Xiad2 << Wra;

        Vec6T Xib = SE3Log(Tfab);
        Vec6T Xibd0 = Xib;
        Vec6T Xibd1;
        Vec6T Xibd2;

        Mat6T JrInv_Xib;         JrInv_Xib.setZero();
        Mat6T Hp1_XibTwb;        Hp1_XibTwb.setZero();
        Mat6T Hp1_XibWrb;        Hp1_XibWrb.setZero();
        Mat6T Lp11_XibTwbXibd1;  Lp11_XibTwbXibd1.setZero();
        Mat6T Lp12_XibTwbXibd1;  Lp12_XibTwbXibd1.setZero();

        // Populate the matrices related to Xib
        Get_JrInvHpLp<T>(Xib, Twb, Wrb, Xibd1, Xibd2, JrInv_Xib, Hp1_XibTwb, Hp1_XibWrb, Lp11_XibTwbXibd1, Lp12_XibTwbXibd1);

        // Stack the local variables in vector form
        Matrix<T, 18, 1> gammaa; gammaa << Xiad0, Xiad1, Xiad2;
        Matrix<T, 18, 1> gammab; gammab << Xibd0, Xibd1, Xibd2;

        // Calculate the residual
        residual << gammab - Fmat*gammaa;
        residual = sqrtW*residual;

        if(find_jacobian)
        {
            // Basic variables
            const SO3T  &Ra = Xa.R;          const SO3T  &Rb = Xb.R;
            const Vec3T &Oa = Xa.O;          const Vec3T &Ob = Xb.O;
            const Vec3T &Sa = Xa.S;          const Vec3T &Sb = Xb.S;
            const Vec3T &Pa = Xa.P;          const Vec3T &Pb = Xb.P;
            const Vec3T &Va = Xa.V;          const Vec3T &Vb = Xb.V;
            const Vec3T &Aa = Xa.A;          const Vec3T &Ab = Xb.A;
            const Vec3T Na = Twa.tail(3);    const Vec3T Nb = Twb.tail(3);
            const Vec3T Ba = Wra.tail(3);    const Vec3T Bb = Wrb.tail(3);


            // Jacobians from L2 to L1
            MatLT U; U.setZero(); U.block(0, 0, 3, 3) = Mat3T::Identity();
            MatLT D; D.setZero(); D.block(0, 3, 3, 3) = Mat3T::Identity();
            MatTT Utp = U.transpose();
            MatTT Dtp = D.transpose();

            Mat6T  J_Xiad1_Twa = Mat6T::Identity();
            Mat6T  J_Xiad2_Wra = Mat6T::Identity();

            Mat6T  J_Xibd0_Tfa = -JrInv((-Xibd0).eval());    // Technically we can use JrInv_Xib*Tab.Adj().inverse().matrix() but the order of The and Rho in Sophus is different
            Mat6T &J_Xibd0_Tfb =  JrInv_Xib;

            Mat6T  J_Xibd1_Tfa = Hp1_XibTwb*J_Xibd0_Tfa;
            Mat6T  J_Xibd1_Tfb = Hp1_XibTwb*J_Xibd0_Tfb;
            Mat6T &J_Xibd1_Twb = JrInv_Xib;

            Mat6T  J_Xibd2_Tfa = Hp1_XibWrb*J_Xibd0_Tfa + Hp1_XibTwb*J_Xibd1_Tfa + Lp11_XibTwbXibd1*J_Xibd0_Tfa;
            Mat6T  J_Xibd2_Tfb = Hp1_XibWrb*J_Xibd0_Tfb + Hp1_XibTwb*J_Xibd1_Tfb + Lp11_XibTwbXibd1*J_Xibd0_Tfb;
            Mat6T  J_Xibd2_Twb = Lp12_XibTwbXibd1 + Hp1_XibTwb*J_Xibd1_Twb;
            Mat6T &J_Xibd2_Wrb = JrInv_Xib;


            // Jacobian from L1 to L0
            Mat3T hat_Oa = SO3T::hat(Oa);                               Mat3T hat_Ob = SO3T::hat(Ob);                           
            Mat3T Ratp = Ra.inverse().matrix();                         Mat3T Rbtp = Rb.inverse().matrix();                     
            Mat3T hat_RatpVa = SO3T::hat(Ratp*Va);                      Mat3T hat_RbtpVb = SO3T::hat(Rbtp*Vb);                  
            Mat3T hat_RatpAa = SO3T::hat(Ratp*Aa);                      Mat3T hat_RbtpAb = SO3T::hat(Rbtp*Ab);                  
                                                                                                                                
            MatTT J_Tfa_Ra =  Utp;                                      MatTT J_Tfb_Rb =  Utp;                                  
            MatTT J_Tfa_Pa =  Dtp*Ratp;                                 MatTT J_Tfb_Pb =  Dtp*Rbtp;                                  
                                                                                                                                
            MatTT J_Twa_Oa =  Utp;                                      MatTT J_Twb_Ob =  Utp;                                  
            MatTT J_Twa_Ra =  Dtp*hat_RatpVa;                           MatTT J_Twb_Rb =  Dtp*hat_RbtpVb;                       
            MatTT J_Twa_Va =  Dtp*Ratp;                                 MatTT J_Twb_Vb =  Dtp*Rbtp;                             
                                                                                                                                
            MatTT J_Wra_Sa =  Utp;                                      MatTT J_Wrb_Sb =  Utp;                                  
            MatTT J_Wra_Ra =  Dtp*(hat_RatpAa - hat_Oa*hat_RatpVa);     MatTT J_Wrb_Rb =  Dtp*(hat_RbtpAb - hat_Ob*hat_RbtpVb); 
            MatTT J_Wra_Aa =  Dtp*Ratp;                                 MatTT J_Wrb_Ab =  Dtp*Rbtp;                             
            MatTT J_Wra_Oa =  Dtp*hat_RatpVa;                           MatTT J_Wrb_Ob =  Dtp*hat_RbtpVb;                       
            MatTT J_Wra_Va = -Dtp*hat_Oa*Ratp;                          MatTT J_Wrb_Vb = -Dtp*hat_Ob*Rbtp;                      


            // Jacobian from L2 to L0

            MatTT J_Xiad1_Ra = J_Xiad1_Twa*J_Twa_Ra;
            MatTT J_Xiad1_Oa = J_Xiad1_Twa*J_Twa_Oa;
            MatTT J_Xiad1_Va = J_Xiad1_Twa*J_Twa_Va;

            MatTT J_Xiad2_Ra = J_Xiad2_Wra*J_Wra_Ra;
            MatTT J_Xiad2_Oa = J_Xiad2_Wra*J_Wra_Oa;
            MatTT J_Xiad2_Sa = J_Xiad2_Wra*J_Wra_Sa;
            MatTT J_Xiad2_Va = J_Xiad2_Wra*J_Wra_Va;
            MatTT J_Xiad2_Aa = J_Xiad2_Wra*J_Wra_Aa;

            MatTT J_Xibd0_Ra = J_Xibd0_Tfa*J_Tfa_Ra;
            MatTT J_Xibd0_Pa = J_Xibd0_Tfa*J_Tfa_Pa;

            MatTT J_Xibd1_Ra = J_Xibd1_Tfa*J_Tfa_Ra;
            MatTT J_Xibd1_Pa = J_Xibd1_Tfa*J_Tfa_Pa;

            MatTT J_Xibd2_Ra = J_Xibd2_Tfa*J_Tfa_Ra;
            MatTT J_Xibd2_Pa = J_Xibd2_Tfa*J_Tfa_Pa;

            MatTT J_Xibd0_Rb = J_Xibd0_Tfb*J_Tfb_Rb;
            MatTT J_Xibd0_Pb = J_Xibd0_Tfb*J_Tfb_Pb;

            MatTT J_Xibd1_Rb = J_Xibd1_Tfb*J_Tfb_Rb + J_Xibd1_Twb*J_Twb_Rb;
            MatTT J_Xibd1_Ob =                        J_Xibd1_Twb*J_Twb_Ob;
            MatTT J_Xibd1_Pb = J_Xibd1_Tfb*J_Tfb_Pb;
            MatTT J_Xibd1_Vb =                        J_Xibd1_Twb*J_Twb_Vb;

            MatTT J_Xibd2_Rb = J_Xibd2_Tfb*J_Tfb_Rb + J_Xibd2_Twb*J_Twb_Rb + J_Xibd2_Wrb*J_Wrb_Rb;
            MatTT J_Xibd2_Ob =                        J_Xibd2_Twb*J_Twb_Ob + J_Xibd2_Wrb*J_Wrb_Ob;
            MatTT J_Xibd2_Sb =                                               J_Xibd2_Wrb*J_Wrb_Sb;
            MatTT J_Xibd2_Pb = J_Xibd2_Tfb*J_Tfb_Pb;
            MatTT J_Xibd2_Vb =                        J_Xibd2_Twb*J_Twb_Vb + J_Xibd2_Wrb*J_Wrb_Vb;
            MatTT J_Xibd2_Ab =                                               J_Xibd2_Wrb*J_Wrb_Ab;

            // Find J_r_Xia and J_r_Xib
            Matrix<T, 18, 18> mF = -Fmat.cast<T>().toDense();
            SparseMatrix<T> J_r_Xiad0 = (mF.template block<STATE_DIM, 6>(0, 0 )).sparseView(); J_r_Xiad0.makeCompressed();
            SparseMatrix<T> J_r_Xiad1 = (mF.template block<STATE_DIM, 6>(0, 6 )).sparseView(); J_r_Xiad1.makeCompressed();
            SparseMatrix<T> J_r_Xiad2 = (mF.template block<STATE_DIM, 6>(0, 12)).sparseView(); J_r_Xiad2.makeCompressed();

            SparseMatrix<T> J_r_Xibd0(18, 6); Util::SetSparseMatBlock<T>(J_r_Xibd0, 0,  0, Mat6T::Identity()); J_r_Xibd0.makeCompressed();
            SparseMatrix<T> J_r_Xibd1(18, 6); Util::SetSparseMatBlock<T>(J_r_Xibd1, 6,  0, Mat6T::Identity()); J_r_Xibd1.makeCompressed();
            SparseMatrix<T> J_r_Xibd2(18, 6); Util::SetSparseMatBlock<T>(J_r_Xibd2, 12, 0, Mat6T::Identity()); J_r_Xibd2.makeCompressed();

            Dr_DXa[Ridx] = sqrtW*(J_r_Xiad1*J_Xiad1_Ra + J_r_Xiad2*J_Xiad2_Ra + J_r_Xibd0*J_Xibd0_Ra + J_r_Xibd1*J_Xibd1_Ra + J_r_Xibd2*J_Xibd2_Ra);
            Dr_DXa[Oidx] = sqrtW*(J_r_Xiad1*J_Xiad1_Oa + J_r_Xiad2*J_Xiad2_Oa                                                                     );
            Dr_DXa[Sidx] = sqrtW*(                       J_r_Xiad2*J_Xiad2_Sa                                                                     );
            Dr_DXa[Pidx] = sqrtW*(                                              J_r_Xibd0*J_Xibd0_Pa + J_r_Xibd1*J_Xibd1_Pa + J_r_Xibd2*J_Xibd2_Pa);
            Dr_DXa[Vidx] = sqrtW*(J_r_Xiad1*J_Xiad1_Va + J_r_Xiad2*J_Xiad2_Va                                                                     );
            Dr_DXa[Aidx] = sqrtW*(                       J_r_Xiad2*J_Xiad2_Aa                                                                     );

            Dr_DXb[Ridx] = sqrtW*(                                              J_r_Xibd0*J_Xibd0_Rb + J_r_Xibd1*J_Xibd1_Rb + J_r_Xibd2*J_Xibd2_Rb);
            Dr_DXb[Oidx] = sqrtW*(                                                                     J_r_Xibd1*J_Xibd1_Ob + J_r_Xibd2*J_Xibd2_Ob);
            Dr_DXb[Sidx] = sqrtW*(                                                                                            J_r_Xibd2*J_Xibd2_Sb);
            Dr_DXb[Pidx] = sqrtW*(                                              J_r_Xibd0*J_Xibd0_Pb + J_r_Xibd1*J_Xibd1_Pb + J_r_Xibd2*J_Xibd2_Pb);
            Dr_DXb[Vidx] = sqrtW*(                                                                     J_r_Xibd1*J_Xibd1_Vb + J_r_Xibd2*J_Xibd2_Vb);
            Dr_DXb[Aidx] = sqrtW*(                                                                                            J_r_Xibd2*J_Xibd2_Ab);
        }
    }

    GPMixer &operator=(const GPMixer &other)
    {
        this->Dt = other.Dt;
        this->CovROSJerk = other.CovROSJerk;
        this->CovPVAJerk = other.CovPVAJerk;
        this->CovTTWJerk = other.CovTTWJerk;
        this->Fmat = other.Fmat;
        this->sqrtW = other.sqrtW;
        this->pose_representation = other.pose_representation;
        this->lie_epsilon = other.lie_epsilon;
        this->use_approx_drv = other.use_approx_drv;

        return *this;
    }
};

// Define the shared pointer
typedef std::shared_ptr<GPMixer> GPMixerPtr;

/* #endregion Utility for propagation and interpolation matrices, elementary jacobians dXt/dXk, J_r, H_r, Hprime_r.. */


/* #region Managing control points: cration, extension, queries, ... ------------------------------------------------*/

class GaussianProcess
{
    using CovM = Eigen::Matrix<double, STATE_DIM, STATE_DIM>;

private:
    
    // The invalid covariance
    const CovM CovMZero = CovM::Zero();

    // Start time
    double t0 = 0;

    // Knot length
    double Dt = 0.0;

    // Mixer
    GPMixerPtr gpm;

    // Set to true to maintain a covariance of each state
    bool keepCov = false;

    template <typename T>
    using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

    // Covariance
    aligned_deque<CovM> C;

    // State vector
    aligned_deque<SO3d> R;
    aligned_deque<Vec3> O;
    aligned_deque<Vec3> S;
    aligned_deque<Vec3> P;
    aligned_deque<Vec3> V;
    aligned_deque<Vec3> A;

public:

    // // Destructor
    // ~GaussianProcess(){};

    // Constructor
    GaussianProcess(double Dt_, Mat3 CovROSJerk_, Mat3 CovPVAJerk_, bool keepCov_ = false,
                    POSE_GROUP pose_representation_ = POSE_GROUP::SO3xR3, double lie_epsilon_ = 1e-3, bool use_approx_drv_ = true)
        : Dt(Dt_), keepCov(keepCov_),
          gpm(GPMixerPtr(new GPMixer(Dt_, CovROSJerk_, CovPVAJerk_, pose_representation_, lie_epsilon_, use_approx_drv_))) {};

          // Constructor
    GaussianProcess(const GPMixerPtr &gpm_)
            : Dt(gpm_->getDt()), keepCov(false), gpm(gpm_) {};

    Mat3 getCovROSJerk() const { return gpm->getCovROSJerk(); }
    Mat3 getCovPVAJerk() const { return gpm->getCovPVAJerk(); }
    bool getKeepCov() const {return keepCov;}

    GPMixerPtr getGPMixerPtr()
    {
        return gpm;
    }

    POSE_GROUP getPoseRepresentation()
    {
        return gpm->getPoseRepresentation();
    }

    double getMinTime() const
    {
        return t0;
    }

    double getMaxTime() const
    {
        return t0 + max(0, int(R.size()) - 1)*Dt;
    }

    int getNumKnots() const
    {
        return int(R.size());
    }

    double getKnotTime(int kidx) const
    {
        return t0 + kidx*Dt;
    }

    double getDt() const
    {
        return Dt;
    }

    bool TimeInInterval(double t, double eps=0.0) const
    {
        return (t >= getMinTime() + eps && t < getMaxTime() - eps);
    }

    pair<int, double> computeTimeIndex(double t) const
    {
        int u = int((t - t0)/Dt);
        double s = double(t - t0)/Dt - u;
        return make_pair(u, s);
    }

    GPState<double> getStateAt(double t) const
    {
        // Find the index of the interval to find interpolation
        auto   us = computeTimeIndex(t);
        int    u  = us.first;
        double s  = us.second;

        int ua = u;  
        int ub = u+1;

        if (ub >= R.size() && fabs(1.0 - s) < DOUBLE_EPSILON)
        {
            // printf(KYEL "Boundary issue: ub: %d, Rsz: %d, s: %f, 1-s: %f\n" RESET, ub, R.size(), s, fabs(1.0 - s));
            return GPState(t0 + ua*Dt, R[ua], O[ua], S[ua], P[ua], V[ua], A[ua]);
        }

        // Extract the states of the two adjacent knots
        GPState Xa = GPState(t0 + ua*Dt, R[ua], O[ua], S[ua], P[ua], V[ua], A[ua]);
        if (fabs(s) < DOUBLE_EPSILON)
        {
            // printf(KYEL "Boundary issue: ub: %d, Rsz: %d, s: %f, 1-s: %f\n" RESET, ub, R.size(), s, fabs(1.0 - s));
            return Xa;
        }

        GPState Xb = GPState(t0 + ub*Dt, R[ub], O[ub], S[ua], P[ub], V[ub], A[ub]);
        if (fabs(1.0 - s) < DOUBLE_EPSILON)
        {
            // printf(KYEL "Boundary issue: ub: %d, Rsz: %d, s: %f, 1-s: %f\n" RESET, ub, R.size(), s, fabs(1.0 - s));
            return Xb;
        }

        GPState Xt(t); vector<vector<Matrix3d>> DXt_DXa, DXt_DXb;
        gpm->ComputeXtAndJacobians(Xa, Xb, Xt, DXt_DXa, DXt_DXb, false);

        return Xt;
    }

    GPState<double> getKnot(int kidx) const
    {
        return GPState(getKnotTime(kidx), R[kidx], O[kidx], S[kidx], P[kidx], V[kidx], A[kidx]);
    }

    SE3d getKnotPose(int kidx) const
    {
        return SE3d(R[kidx], P[kidx]);
    }

    SE3d pose(double t) const
    {
        GPState X = getStateAt(t);
        return SE3d(X.R, X.P);
    }

    GPState<double> predictState(int steps)
    {
        SO3d Rc = R.back();
        Vec3 Oc = O.back();
        Vec3 Sc = S.back();
        Vec3 Pc = P.back();
        Vec3 Vc = V.back();
        Vec3 Ac = A.back();
        
        for(int k = 0; k < steps; k++)
        {
            SO3d Rpred = Rc*SO3d::exp(Dt*Oc + 0.5*Dt*Dt*Sc);
            Vec3 Opred = Oc + Dt*Sc;
            Vec3 Spred = Sc;
            Vec3 Ppred = Pc + Dt*Vc + 0.5*Dt*Dt*Ac;
            Vec3 Vpred = Vc + Dt*Ac;
            Vec3 Apred = Ac;

            Rc = Rpred;
            Oc = Opred;
            Sc = Spred;
            Pc = Ppred;
            Vc = Vpred;
            Ac = Apred;
        }

        return GPState<double>(getMaxTime() + steps*Dt, Rc, Oc, Sc, Pc, Vc, Ac);
    }

    inline SO3d &getKnotSO3(size_t kidx) { return R[kidx]; }
    inline Vec3 &getKnotOmg(size_t kidx) { return O[kidx]; }
    inline Vec3 &getKnotAlp(size_t kidx) { return S[kidx]; }
    inline Vec3 &getKnotPos(size_t kidx) { return P[kidx]; }
    inline Vec3 &getKnotVel(size_t kidx) { return V[kidx]; }
    inline Vec3 &getKnotAcc(size_t kidx) { return A[kidx]; }
    inline CovM &getKnotCov(size_t kidx) { return C[kidx]; }

    inline shared_ptr<SO3d> getKnotSO3Ptr(size_t kidx) { return shared_ptr<SO3d>(&R[kidx], [](SO3d *X){}); }
    inline shared_ptr<Vec3> getKnotOmgPtr(size_t kidx) { return shared_ptr<Vec3>(&O[kidx], [](Vec3 *X){}); }
    inline shared_ptr<Vec3> getKnotAlpPtr(size_t kidx) { return shared_ptr<Vec3>(&S[kidx], [](Vec3 *X){}); }
    inline shared_ptr<Vec3> getKnotPosPtr(size_t kidx) { return shared_ptr<Vec3>(&P[kidx], [](Vec3 *X){}); }
    inline shared_ptr<Vec3> getKnotVelPtr(size_t kidx) { return shared_ptr<Vec3>(&V[kidx], [](Vec3 *X){}); }
    inline shared_ptr<Vec3> getKnotAccPtr(size_t kidx) { return shared_ptr<Vec3>(&A[kidx], [](Vec3 *X){}); }
    inline shared_ptr<CovM> getKnotCovPtr(size_t kidx) { return shared_ptr<CovM>(&C[kidx], [](CovM *X){}); }

    void setStartTime(double t)
    {
        t0 = t;
        if (R.size() == 0)
        {
            R = {SO3d()};
            O = {Vec3(0, 0, 0)};
            S = {Vec3(0, 0, 0)};
            P = {Vec3(0, 0, 0)};
            V = {Vec3(0, 0, 0)};
            A = {Vec3(0, 0, 0)};
            
            if (keepCov)
                C = {CovMZero};
        }
    }
    
    void propagateCovariance()
    {
        CovM Cn = CovMZero;

        // If previous
        if (C.back().cwiseAbs().maxCoeff() != 0.0)
            Cn = gpm->PropagateFullCov(C.back());

        // Add the covariance to buffer
        C.push_back(Cn);
        assert(C.size() == R.size());
    }

    void extendKnotsTo(double t, const GPState<double> &Xn=GPState())
    {
        if(t0 == 0)
        {
            printf("MIN TIME HAS NOT BEEN INITIALIZED. "
                   "PLEASE CHECK, OR ELSE THE KNOT NUMBERS CAN BE VERY LARGE!");
            exit(-1);
        }
        
        // double tmax = getMaxTime();
        // if (tmax > t)
        //     return;

        // // Find the total number of knots at the new max time
        // int newknots = (t - t0 + dt - 1)/dt + 1;

        // Push the new state to the queue
        while(getMaxTime() < t)
        {
            R.push_back(Xn.R);
            O.push_back(Xn.O);
            S.push_back(Xn.S);
            P.push_back(Xn.P);
            V.push_back(Xn.V);
            A.push_back(Xn.A);

            if (keepCov)
                propagateCovariance();
        }
    }

    void extendOneKnot()
    {
        SO3d Rc = R.back();
        Vec3 Oc = O.back();
        Vec3 Sc = S.back();
        Vec3 Pc = P.back();
        Vec3 Vc = V.back();
        Vec3 Ac = A.back();

        SO3d Rn = Rc*SO3d::exp(Dt*Oc);
        Vec3 On = Oc;
        Vec3 Sn = Sc;
        Vec3 Pn = Pc + Dt*Vc + 0.5*Dt*Dt*Ac;
        Vec3 Vn = Vc + Dt*Ac;
        Vec3 An = Ac;

        R.push_back(Rn);
        O.push_back(On);
        S.push_back(Sn);
        P.push_back(Pn);
        V.push_back(Vn);
        A.push_back(An);

        if (keepCov)
            propagateCovariance();
    }

    void extendOneKnot(const GPState<double> &Xn)
    {
        R.push_back(Xn.R);
        O.push_back(Xn.O);
        S.push_back(Xn.S);
        P.push_back(Xn.P);
        V.push_back(Xn.V);
        A.push_back(Xn.A);

        if (keepCov)
            propagateCovariance();
    }

    void setCovPVAJerk(const Matrix3d &m)
    {
        gpm->setCovPVAJerk(m);
    }

    void setCovROSJerk(const Matrix3d &m)
    {
        gpm->setCovROSJerk(m);
    }

    void setKnot(int kidx, const GPState<double> &Xn)
    {
        R[kidx] = Xn.R;
        O[kidx] = Xn.O;
        S[kidx] = Xn.S;
        P[kidx] = Xn.P;
        V[kidx] = Xn.V;
        A[kidx] = Xn.A;
    }

    void setKnotState(int kidx, const SO3d &X)
    {
        R[kidx] = X;
    }

    void setKnotState(int kidx, const Vec3 &X, int sidx)
    {
        switch(sidx)
        {
            case 1:
                O[kidx] = X; break;
            case 2:
                S[kidx] = X; break;
            case 3:
                P[kidx] = X; break;
            case 4:
                V[kidx] = X; break;
            case 5:
                A[kidx] = X; break;
            default:
                printf("Error. Trying to set sidx out of range 1 to 5\n");
        }
    }

    void setKnotCovariance(int kidx, const CovM &Cov)
    {
        C[kidx] = Cov;
        assert(C.size() == R.size());
    }

    void updateKnot(int kidx, Matrix<double, STATE_DIM, 1> dX)
    {
        R[kidx] = R[kidx]*SO3d::exp(dX.block<3, 1>(0, 0));
        O[kidx] = O[kidx] + dX.block<3, 1>(3, 0);
        S[kidx] = S[kidx] + dX.block<3, 1>(6, 0);
        P[kidx] = P[kidx] + dX.block<3, 1>(9, 0);
        V[kidx] = V[kidx] + dX.block<3, 1>(12, 0);
        A[kidx] = A[kidx] + dX.block<3, 1>(15, 0);
    }

    void genRandomTrajectory(int n, double scale = 5.0)
    {
        R.clear(); O.clear(); S.clear(); P.clear(); V.clear(); A.clear();

        for(int kidx = 0; kidx < n; kidx++)
        {
            R.push_back(SO3d::exp(Vec3::Random()* M_PI));
            O.push_back(Vec3::Random() * scale);
            S.push_back(Vec3::Random() * scale);
            P.push_back(Vec3::Random() * scale);
            V.push_back(Vec3::Random() * scale);
            A.push_back(Vec3::Random() * scale);
        }
    }

    // Copy constructor
    GaussianProcess &operator=(GaussianProcess &other)
    {
        this->t0 = other.getMinTime();
        this->Dt = other.getDt();

        *(this->gpm) = (*other.getGPMixerPtr());

        this->keepCov = other.keepCov;
        this->C = other.C;

        this->R = other.R;
        this->O = other.O;
        this->S = other.S;
        this->P = other.P;
        this->V = other.V;
        this->A = other.A;

        return *this;
    }

    bool saveTrajectory(string log_dir, int lidx)
    {
        string log_ = log_dir + "/gptraj_" + std::to_string(lidx) + ".csv";
        std::ofstream logfile;
        logfile.open(log_); // Open the file for writing
        logfile.precision(std::numeric_limits<double>::digits10 + 1);

        logfile << "Dt:" << Dt << ";Order:" << 3 << ";Knots:" << getNumKnots() << ";MinTime:" << t0 << ";MaxTime:" << getMaxTime()
                << ";CovROSJerk:" << getCovROSJerk()(0, 0) << "," << getCovROSJerk()(0, 1) << "," << getCovROSJerk()(0, 2) << ","
                             << getCovROSJerk()(1, 0) << "," << getCovROSJerk()(1, 1) << "," << getCovROSJerk()(1, 2) << ","
                             << getCovROSJerk()(2, 0) << "," << getCovROSJerk()(2, 1) << "," << getCovROSJerk()(2, 2)
                << ";CovPVAJerk:" << getCovPVAJerk()(0, 0) << "," << getCovPVAJerk()(0, 1) << "," << getCovPVAJerk()(0, 2) << ","
                             << getCovPVAJerk()(1, 0) << "," << getCovPVAJerk()(1, 1) << "," << getCovPVAJerk()(1, 2) << ","
                             << getCovPVAJerk()(2, 0) << "," << getCovPVAJerk()(2, 1) << "," << getCovPVAJerk()(2, 2)
                << ";keepCov:" << getKeepCov()
                << ";poseType:" << int(getPoseRepresentation())
                << ";closedform:" << getGPMixerPtr()->getJacobianForm()
                << endl;

        for(int kidx = 0; kidx < getNumKnots(); kidx++)
        {
            logfile << kidx << ", "
                    << getKnotTime(kidx) << ", "
                    << getKnotSO3(kidx).unit_quaternion().x() << ", "
                    << getKnotSO3(kidx).unit_quaternion().y() << ", "
                    << getKnotSO3(kidx).unit_quaternion().z() << ", "
                    << getKnotSO3(kidx).unit_quaternion().w() << ", "
                    << getKnotOmg(kidx).x() << ", "
                    << getKnotOmg(kidx).y() << ", "
                    << getKnotOmg(kidx).z() << ", "
                    << getKnotAlp(kidx).x() << ", "
                    << getKnotAlp(kidx).y() << ", "
                    << getKnotAlp(kidx).z() << ", "
                    << getKnotPos(kidx).x() << ", "
                    << getKnotPos(kidx).y() << ", "
                    << getKnotPos(kidx).z() << ", "
                    << getKnotVel(kidx).x() << ", "
                    << getKnotVel(kidx).y() << ", "
                    << getKnotVel(kidx).z() << ", "
                    << getKnotAcc(kidx).x() << ", "
                    << getKnotAcc(kidx).y() << ", "
                    << getKnotAcc(kidx).z() << endl;
        }

        logfile.close();
        return true;
    }

    bool loadTrajectory(string log_file)
    {
        std::ifstream file(log_file);

        auto splitstr = [](const string s_, const char d) -> vector<string>
        {
            std::istringstream s(s_);
            vector<string> o; string p;
            while(std::getline(s, p, d))
                o.push_back(p);
            return o;    
        };

        double Dt_inLog;
        double t0_inLog;
        GPMixerPtr gpm_inLog;

        // Get the first line for specification
        if (file.is_open())
        {
            // Read the first line from the file
            std::string header;
            std::getline(file, header);

            printf("Get header: %s\n", header.c_str());
            vector<string> fields = splitstr(header, ';');
            map<string, int> fieldidx;
            for(auto &field : fields)
            {
                vector<string> fv = splitstr(field, ':');
                fieldidx[fv[0]] = fieldidx.size();
                printf("Field: %s. Value: %s\n", fv[0].c_str(), splitstr(fields[fieldidx[fv[0]]], ':').back().c_str());
            }

            auto strToMat3 = [&splitstr](const string &s, char d) -> Matrix3d
            {
                vector<string> Mstr = splitstr(s, d);
                for(int idx = 0; idx < Mstr.size(); idx++)
                    printf("Mstr[%d] = %s. S: %s\n", idx, Mstr[idx].c_str(), s.c_str());

                vector<double> Mdbl = {stod(Mstr[0]), stod(Mstr[1]), stod(Mstr[2]),
                                       stod(Mstr[3]), stod(Mstr[4]), stod(Mstr[5]), 
                                       stod(Mstr[6]), stod(Mstr[7]), stod(Mstr[8])};

                Eigen::Map<Matrix3d, Eigen::RowMajor> M(&Mdbl[0]);
                return M;
            };
            Matrix3d logCovPVAJerk = strToMat3(splitstr(fields[fieldidx["CovPVAJerk"]], ':').back(), ',');
            Matrix3d logCovROSJerk = strToMat3(splitstr(fields[fieldidx["CovROSJerk"]], ':').back(), ',');
            double logDt = stod(splitstr(fields[fieldidx["Dt"]], ':').back());
            double logMinTime = stod(splitstr(fields[fieldidx["MinTime"]], ':').back());
            bool logkeepCov = (stoi(splitstr(fields[fieldidx["keepCov"]], ':').back()) == 1);

            printf("Log configs:\n");
            printf("Dt: %f\n", logDt);
            printf("MinTime: %f\n", logMinTime);
            printf("CovPVAJerk: \n");
            cout << logCovPVAJerk << endl;
            printf("CovROSJerk: \n");
            cout << logCovROSJerk << endl;

            Dt_inLog = logDt;
            t0_inLog = logMinTime;
            gpm_inLog = GPMixerPtr(new GPMixer(logDt, logCovROSJerk, logCovPVAJerk));

            if (logkeepCov == keepCov)
                printf(KYEL "Covariance tracking is disabled\n" RESET);

            keepCov = false;
        }

        // Read txt to matrix
        auto read_csv =  [](const std::string &path, string dlm, int r_start = 0, int col_start = 0) -> MatrixXd
        {
            std::ifstream indata;
            indata.open(path);
            std::string line;
            std::vector<double> values;
            int row_idx = -1;
            int rows = 0;
            while (std::getline(indata, line))
            {
                row_idx++;
                if (row_idx < r_start)
                    continue;

                // printf("line: %s\n", line.c_str());

                std::stringstream lineStream(line);
                std::string cell;
                int col_idx = -1;
                while (std::getline(lineStream, cell, dlm[0]))
                {
                    if (cell == dlm || cell.size() == 0)
                        continue;

                    col_idx++;
                    if (col_idx < col_start)
                        continue;

                    values.push_back(std::stod(cell));

                    // printf("cell: %s\n", cell.c_str());
                }

                rows++;
            }

            return Eigen::Map<Matrix<double, -1, -1, Eigen::RowMajor>>(values.data(), rows, values.size() / rows);
        };

        // Load the control point values
        MatrixXd traj = read_csv(log_file, ",", 1, 0);
        printf("Found %d control points.\n", traj.rows());
        // for(int ridx = 0; ridx < traj.rows(); ridx++)
        // {
        //     cout << "Row: " << traj.row(ridx) << endl;
        //     if (ridx == 10)
        //         exit(-1);
        // }
        
        if(Dt == 0 || Dt_inLog == Dt)
        {
            printf("dt has not been set. Use log's dt %f.\n", Dt_inLog);
            
            // Clear the knots
            R.clear(); O.clear(); S.clear(); P.clear(); V.clear(); A.clear(); C.clear();

            // Set the knot values
            for(int ridx = 0; ridx < traj.rows(); ridx++)
            {
                VectorXd X = traj.row(ridx);
                R.push_back(SO3d(Quaternd(X(5), X(2), X(3), X(4))));
                O.push_back(Vec3(X(6),  X(7),  X(8)));
                S.push_back(Vec3(X(9),  X(10), X(11)));
                P.push_back(Vec3(X(12), X(13), X(14)));
                V.push_back(Vec3(X(15), X(16), X(17)));
                A.push_back(Vec3(X(18), X(19), X(20)));

                // C.push_back(CovMZero);
            }
        }
        else
        {
            printf(KYEL "Logged GPCT is has different knot length. Chosen: %f. Log: %f.\n" RESET, Dt, Dt_inLog);
            
            // Create a trajectory
            GaussianProcess trajLog(Dt_inLog, gpm_inLog->getCovROSJerk(), gpm_inLog->getCovPVAJerk());
            trajLog.setStartTime(t0_inLog);

            // Create the trajectory
            for(int ridx = 0; ridx < traj.rows(); ridx++)
            {
                VectorXd X = traj.row(ridx);
                trajLog.extendOneKnot(GPState<double>(ridx*Dt_inLog+t0_inLog, SO3d(Quaternd(X(5), X(2), X(3), X(4))),
                                                                              Vec3(X(6),  X(7),  X(8)),
                                                                              Vec3(X(9),  X(10), X(11)),
                                                                              Vec3(X(12), X(13), X(14)),
                                                                              Vec3(X(15), X(16), X(17)),
                                                                              Vec3(X(18), X(19), X(20))));
            }

            // Sample the log trajectory to initialize current trajectory
            t0 = t0_inLog;
            R.clear(); O.clear(); S.clear(); P.clear(); V.clear(); A.clear(); C.clear();
            for(double ts = t0; ts < trajLog.getMaxTime() - trajLog.getDt(); ts += Dt)
                extendOneKnot(trajLog.getStateAt(ts));
        }

        return true;
    }
};
// Define the shared pointer
typedef std::shared_ptr<GaussianProcess> GaussianProcessPtr;

/* #endregion Managing control points: cration, extension, queries, ... ---------------------------------------------*/


/* #region Local parameterization when using ceres ------------------------------------------------------------------*/

template <class Groupd>
class GPSO3LocalParameterization : public ceres::LocalParameterization
{
public:
    // virtual ~GPSO3LocalParameterization() {}

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

    virtual bool ComputeJacobian(double const *T_raw,
                                 double *jacobian_raw) const
    {
        Eigen::Map<Groupd const> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, Groupd::num_parameters, Groupd::DoF, Eigen::RowMajor>>
        
        jacobian(jacobian_raw);
        jacobian.setZero();

        jacobian(0, 0) = 1;
        jacobian(1, 1) = 1;
        jacobian(2, 2) = 1;
        return true;
    }

    ///@brief Global size
    virtual int GlobalSize() const { return Groupd::num_parameters; }

    ///@brief Local size
    virtual int LocalSize() const { return Groupd::DoF; }
};
typedef GPSO3LocalParameterization<SO3d> GPSO3dLocalParameterization;

/* #endregion Local parameterization when using ceres ----------------------------------------------------------------*/
