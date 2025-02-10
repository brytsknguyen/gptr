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

/* #region Define the states for convenience in initialization and copying ------------------------------------------*/

#define STATE_DIM 18
template <class T = double>
class GPState
{
public:

    using SO3T  = Sophus::SO3<T>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Mat3T = Eigen::Matrix<T, 3, 3>;

    double t;
    SO3T  R;
    Vec3T O;
    Vec3T S;
    Vec3T P;
    Vec3T V;
    Vec3T A;

    // // Destructor
    // ~GPState(){};
    
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
    double dt = 0.0;

    // Identity matrix
    const Mat3 Eye = Mat3::Identity();

    // 3x3 Zero matrix
    const Mat3 Zr3 = Mat3::Zero();

    // Covariance of angular jerk
    Mat3 SigGa = Eye;

    // Covariance of translational jerk
    Mat3 SigNu = Eye;

    // Covariance of wrench
    Matrix<double, 6, 6> SigGN = Matrix<double, 6, 6>::Identity(6, 6);

public:

//     // Destructor
//    ~GPMixer() {};

    // Constructor
    GPMixer(double dt_, const Mat3 SigGa_, const Mat3 SigNu_) : dt(dt_), SigGa(SigGa_), SigNu(SigNu_)
    {
        SigGN.block<3, 3>(0, 0) = SigGa;
        SigGN.block<3, 3>(3, 3) = SigNu;
    };

    double getDt()    const { return dt;    }
    Mat3   getSigGa() const { return SigGa; }
    Mat3   getSigNu() const { return SigNu; }

    template <typename MatrixType1, typename MatrixType2>
    MatrixXd kron(const MatrixType1& A, const MatrixType2& B) const
    {
        MatrixXd result(A.rows() * B.rows(), A.cols() * B.cols());
        for (int i = 0; i < A.rows(); ++i)
            for (int j = 0; j < A.cols(); ++j)
                result.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;

        return result;
    }

    void setSigGa(const Mat3 &m)
    {
        SigGa = m;
    }

    void setSigNu(const Mat3 &m)
    {
        SigNu = m;
    }

    // Transition Matrix, PHI(tau, 0)
    MatrixXd Fbase(const double dtau, int N) const
    {
        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};

        MatrixXd Phi = MatrixXd::Identity(N, N);
        for(int n = 0; n < N; n++)
            for(int m = n + 1; m < N; m++)
                Phi(n, m) = pow(dtau, m-n)/factorial(m-n);

        return Phi;
    }

    // Gaussian Process covariance, Q = \int{Phi*F*SigNu*F'*Phi'}
    MatrixXd Qbase(const double dtau, int N) const 
    {
        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};
        
        MatrixXd Q(N, N);
        for(int n = 0; n < N; n++)
            for(int m = 0; m < N; m++)
                Q(n, m) = pow(dtau, 2*N-1-n-m)/double(2*N-1-n-m)/double(factorial(N-1-n))/double(factorial(N-1-m));
        // cout << "MyQ: " << Q << endl;
        return Q;
    }

    MatrixXd Qga(const double s, int N) const 
    {
        double dtau = s*dt;

        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};
        
        MatrixXd Q(N, N);
        for(int n = 0; n < N; n++)
            for(int m = 0; m < N; m++)
                Q(n, m) = pow(dtau, 2*N-1-n-m)/double(2*N-1-n-m)/double(factorial(N-1-n))/double(factorial(N-1-m));

        return kron(Qbase(dt, 3), SigGa);
    }

    MatrixXd Qnu(const double s, int N) const 
    {
        double dtau = s*dt;

        std::function<int(int)> factorial = [&factorial](int n) -> int {return (n <= 1) ? 1 : n * factorial(n - 1);};
        
        MatrixXd Q(N, N);
        for(int n = 0; n < N; n++)
            for(int m = 0; m < N; m++)
                Q(n, m) = pow(dtau, 2*N-1-n-m)/double(2*N-1-n-m)/double(factorial(N-1-n))/double(factorial(N-1-m));

        return kron(Qbase(dt, 3), SigNu);
    }

    Matrix<double, STATE_DIM, STATE_DIM> PropagateFullCov(Matrix<double, STATE_DIM, STATE_DIM> P0) const
    {
        Matrix<double, STATE_DIM, STATE_DIM> F; F.setZero();
        Matrix<double, STATE_DIM, STATE_DIM> Q; Q.setZero();
        
        F.block<9, 9>(0, 0) = kron(Fbase(dt, 3), Eye);
        F.block<9, 9>(9, 9) = kron(Fbase(dt, 3), Eye);

        Q.block<9, 9>(0, 0) = kron(Qbase(dt, 3), SigGa);
        Q.block<9, 9>(9, 9) = kron(Qbase(dt, 3), SigNu);

        return F*P0*F.transpose() + Q;
    }

    MatrixXd PSI(const double dtau, const MatrixXd &Q) const
    {
        if (dtau < DOUBLE_EPSILON)
            return kron(MatrixXd::Zero(3, 3), Eye);

        MatrixXd Phidtaubar = kron(Fbase(dt - dtau, 3), Eye);
        MatrixXd Qdtau = kron(Qbase(dtau, 3), Q);
        MatrixXd Qdt = kron(Qbase(dt, 3), Q);

        return Qdtau*Phidtaubar.transpose()*Qdt.inverse();
    }

    MatrixXd PSI_ROS(const double dtau) const
    {
        return PSI(dtau, SigGa);
    }

    MatrixXd PSI_PVA(const double dtau) const
    {
        return PSI(dtau, SigNu);
    }

    MatrixXd LAMDA(const double dtau, const MatrixXd &Q) const
    {
        MatrixXd PSIdtau = PSI(dtau, Q);
        MatrixXd Fdtau = kron(Fbase(dtau, 3), Eye);
        MatrixXd Fdt = kron(Fbase(dt, 3), Eye);

        return Fdtau - PSIdtau*Fdt;
    }

    MatrixXd LAMDA_ROS(const double dtau) const
    {
        return LAMDA(dtau, SigGa);
    }

    MatrixXd LAMDA_PVA(const double dtau) const
    {
        return LAMDA(dtau, SigNu);
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

    // left Jacobian for SO3
    template <class T = double>
    static Eigen::Matrix<T, 3, 3> Jl(const Eigen::Matrix<T, 3, 1> &The)
    {
        // if (phi.norm() < DOUBLE_EPSILON)
        //     return Eigen::Matrix<T, 3, 3>::Identity() - 0.5*Sophus::SO3<T>::hat(phi) + 1.0/6*hatSquare(phi);

        return Sophus::SO3<T>::leftJacobian(The);
    }

    // right Jacobian for SO3
    template <class T = double>
    static Eigen::Matrix<T, 3, 3> Jr(const Eigen::Matrix<T, 3, 1> &The)
    {
        // if (phi.norm() < DOUBLE_EPSILON)
        //     return Eigen::Matrix<T, 3, 3>::Identity() - 0.5*Sophus::SO3<T>::hat(phi) + 1.0/6*hatSquare(phi);

        return Sophus::SO3<T>::leftJacobian(-The);
    }

    // inverse right Jacobian for SO3
    template <class T = double>
    static Eigen::Matrix<T, 3, 3> JlInv(const Eigen::Matrix<T, 3, 1> &The)
    {
        // if (The.norm() < DOUBLE_EPSILON)
        //     return Eigen::Matrix<T, 3, 3>::Identity() + 0.5*Sophus::SO3<T>::hat(The) + 1.0/12*hatSquare(The);

        return Sophus::SO3<T>::leftJacobianInverse(The);;
    }

    // inverse right Jacobian for SO3
    template <class T = double>
    static Eigen::Matrix<T, 3, 3> JrInv(const Eigen::Matrix<T, 3, 1> &The)
    {
        // if (The.norm() < DOUBLE_EPSILON)
        //     return Eigen::Matrix<T, 3, 3>::Identity() + 0.5*Sophus::SO3<T>::hat(The) + 1.0/12*hatSquare(The);

        return Sophus::SO3<T>::leftJacobianInverse(-The);
    }

    // For calculating HThe_
    template <class T = double>
    static Eigen::Matrix<T, 3, 3> DJrUV_DU(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V)
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Un = U.norm();

        if(Un < DOUBLE_EPSILON)
            return 0.5*SO3T::hat(V);

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
    static Eigen::Matrix<T, 3, 3> DDJrUVW_DUDU(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &W)
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Un = U.norm();

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
    static Eigen::Matrix<T, 3, 3> DDJrUVW_DUDV(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &W)
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Un = U.norm();

        if(Un < DOUBLE_EPSILON)
            return -0.5*SO3T::hat(W);

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
    static Eigen::Matrix<T, 3, 3> DJrInvUV_DU(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V)
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Un = U.norm();
        if(Un < DOUBLE_EPSILON)
            return -0.5*SO3T::hat(V);

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
    static Eigen::Matrix<T, 3, 3> DDJrInvUVW_DUDU(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &W)
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Un = U.norm();

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
    static Eigen::Matrix<T, 3, 3> DDJrInvUVW_DUDV(const Eigen::Matrix<T, 3, 1> &U, const Eigen::Matrix<T, 3, 1> &V, const Eigen::Matrix<T, 3, 1> &W)
    {
        using SO3T = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        T Un = U.norm();

        if(Un < DOUBLE_EPSILON)
            return 0.5*SO3T::hat(W);

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

    // right Jacobian for SE3
    template <class T = double>
    static Eigen::Matrix<T, 6, 6> Jr(const Eigen::Matrix<T, 6, 1> &Xi)
    {
        Eigen::Matrix<T, 3, 1> The = Xi.template head<3>();
        Eigen::Matrix<T, 3, 1> Rho = Xi.template tail<3>();
        
        Eigen::Matrix<T, 6, 6> Jr_Xi;
        Eigen::Matrix<T, 3, 3> Jr_The = Jr(The);
        Eigen::Matrix<T, 3, 3> Q = -(Sophus::SO3<T>::exp(-The).matrix()*DJrUV_DU(Eigen::Matrix<T, 3, 1>(-The), Rho));

        Jr_Xi << Jr_The, Matrix<T, 3, 3>::Zero(),
                 Q, Jr_The;
        return Jr_Xi;
    }

    // inverse right Jacobian for SE3
    template <class T = double>
    static Eigen::Matrix<T, 6, 6> JrInv(const Eigen::Matrix<T, 6, 1> &Xi)
    {
        Eigen::Matrix<T, 3, 1> The = Xi.template head<3>();
        Eigen::Matrix<T, 3, 1> Rho = Xi.template tail<3>();

        Eigen::Matrix<T, 6, 6> JrInv_Xi;
        Eigen::Matrix<T, 3, 3> JrInv_The = JrInv(The);
        Eigen::Matrix<T, 3, 3> Q = -(Sophus::SO3<T>::exp(-The).matrix()*DJrUV_DU(Eigen::Matrix<T, 3, 1>(-The), Rho));

        JrInv_Xi << JrInv_The, Matrix<T, 3, 3>::Zero(),
                   -JrInv_The*Q*JrInv_The, JrInv_The;

        return JrInv_Xi;
    }

    // left Jacobian for SE3
    template <class T = double>
    static Eigen::Matrix<T, 6, 6> Jl(const Eigen::Matrix<T, 6, 1> &Xi)
    {
        return Jr(Eigen::Matrix<T, 6, 1>(-Xi));
    }

    // inverse left Jacobian for SE3
    template <class T = double>
    static Eigen::Matrix<T, 6, 6> JlInv(const Eigen::Matrix<T, 6, 1> &Xi)
    {
        return JrInv(Eigen::Matrix<T, 6, 1>(-Xi));
    }

    // // For calculating H_Xi(Xi, Xid)
    // template <class T = double>
    // static Eigen::Matrix<T, 6, 6> DJrUV_DU(const Eigen::Matrix<T, 6, 1> &U, const Eigen::Matrix<T, 6, 1> &V)
    // {
    //     using SO3T  = Sophus::SO3<T>;
    //     using Vec3T = Eigen::Matrix<T, 3, 1>;
    //     using Mat3T = Eigen::Matrix<T, 3, 3>;
    //     using Mat6T = Eigen::Matrix<T, 6, 6>;

    //     T Un = U.norm();
    //     if(Un < DOUBLE_EPSILON)
    //         return Eigen::Matrix<T, 6, 6>::Zero();    // To do: find the near-zero form

    //     Vec3T The = U.template head(3);
    //     Vec3T Rho = U.template tail(3);

    //     Vec3T Thed = V.template head(3);
    //     Vec3T Rhod = V.template tail(3);

    //     Mat3T Zero3x3 = Mat3T::Zero();
    //     Mat3T HThe_TheThed = DJrUV_DU(The, Thed);
    //     Mat3T HThe_TheRhod = DJrUV_DU(The, Rhod);
    //     Mat3T LTheThe_TheRhoThed = DDJrUVW_DUDU(The, Rho, Thed);
    //     Mat3T LTheRho_TheRhoThed = DDJrUVW_DUDV(The, Rho, Thed);

    //     Mat6T HXi_XiXid;
    //     HXi_XiXid << HThe_TheThed, Zero3x3,
    //                  LTheThe_TheRhoThed + HThe_TheRhod, LTheRho_TheRhoThed;

    //     return HXi_XiXid;
    // }

    // // For calculating H'_Xi(Xi, tau)
    // template <class T = double>
    // static Eigen::Matrix<T, 6, 6> DJrInvUV_DU(const Eigen::Matrix<T, 6, 1> &U, const Eigen::Matrix<T, 6, 1> &V)
    // {
    //     T Un = U.norm();
    //     if(Un < DOUBLE_EPSILON)
    //         return Eigen::Matrix<T, 6, 6>::Zero();    // To do: find the near-zero form

    //     Eigen::Matrix<T, 6, 1> O = JrInv(U)*V;
    //     return -JrInv(U)*DJrUV_DU(U, O);
    // }

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
                               Eigen::Matrix<T, 9, 1> &gammaa_,
                               Eigen::Matrix<T, 9, 1> &gammab_,
                               Eigen::Matrix<T, 9, 1> &gammat_,
                               POSE_GROUP representation = POSE_GROUP::SO3xR3,
                               bool debug = false
                              ) const
    {
        if (representation == POSE_GROUP::SO3xR3)
            ComputeXtAndJacobiansSO3R3(Xa, Xb, Xt, DXt_DXa, DXt_DXb, gammaa_, gammab_, gammat_);
        else if (representation == POSE_GROUP::SE3)
            ComputeXtAndJacobiansSE3(Xa, Xb, Xt, DXt_DXa, DXt_DXb);
    }

    template <class T = double>
    void ComputeXtAndJacobiansSO3R3(const GPState<T> &Xa,
                                    const GPState<T> &Xb,
                                          GPState<T> &Xt,
                                    vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXa,
                                    vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXb,
                                    Eigen::Matrix<T, 9, 1> &gammaa_,
                                    Eigen::Matrix<T, 9, 1> &gammab_,
                                    Eigen::Matrix<T, 9, 1> &gammat_,
                                    bool debug = false
                                   ) const
    {
        using SO3T  = Sophus::SO3<T>;
        using Vec3T = Eigen::Matrix<T, 3, 1>;
        using Vec6T = Eigen::Matrix<T, 6, 1>;
        using Vec9T = Eigen::Matrix<T, 9, 1>;
        using Mat3T = Eigen::Matrix<T, 3, 3>;

        // Map the variables of the state
        double tau = Xt.t;
        SO3T   &Rt = Xt.R;
        Vec3T  &Ot = Xt.O;
        Vec3T  &St = Xt.S;
        Vec3T  &Pt = Xt.P;
        Vec3T  &Vt = Xt.V;
        Vec3T  &At = Xt.A;
        
        // Calculate the the mixer matrixes
        Matrix<T, Dynamic, Dynamic> LAM_ROSt = LAMDA(tau, SigGa).cast<T>();
        Matrix<T, Dynamic, Dynamic> PSI_ROSt = PSI(tau,   SigGa).cast<T>();
        Matrix<T, Dynamic, Dynamic> LAM_PVAt = LAMDA(tau, SigNu).cast<T>();
        Matrix<T, Dynamic, Dynamic> PSI_PVAt = PSI(tau,   SigNu).cast<T>();

        // Find the relative rotation
        SO3T Rab = Xa.R.inverse()*Xb.R;

        // Calculate the SO3 knots in relative form
        Vec3T Thead0 = Vec3T::Zero();
        Vec3T Thead1 = Xa.O;
        Vec3T Thead2 = Xa.S;

        Vec3T Theb = Rab.log();
        Mat3T JrInvTheb = JrInv(Theb);
        Mat3T HpTheb_ThebOb = DJrInvUV_DU(Theb, Xb.O);
        
        Vec3T Thebd0 = Theb;
        Vec3T Thebd1 = JrInvTheb*Xb.O;
        Vec3T Thebd2 = JrInvTheb*Xb.S + HpTheb_ThebOb*Thebd1;

        // Put them in vector form
        Vec9T gammaa; gammaa << Thead0, Thead1, Thead2;
        Vec9T gammab; gammab << Thebd0, Thebd1, Thebd2;

        // Calculate the knot euclid states and put them in vector form
        Vec9T pvaa; pvaa << Xa.P, Xa.V, Xa.A;
        Vec9T pvab; pvab << Xb.P, Xb.V, Xb.A;

        // Mix the knots to get the interpolated states
        Vec9T gammat = LAM_ROSt*gammaa + PSI_ROSt*gammab;
        Vec9T pvat   = LAM_PVAt*pvaa   + PSI_PVAt*pvab;

        // Retrive the interpolated SO3 in relative form
        Vec3T Thetd0 = gammat.block(0, 0, 3, 1);
        Vec3T Thetd1 = gammat.block(3, 0, 3, 1);
        Vec3T Thetd2 = gammat.block(6, 0, 3, 1);

        Mat3T JrThet  = Jr(Thetd0);
        SO3T  ExpThet = SO3T::exp(Thetd0);
        Mat3T HThet_ThetThetd1 = DJrUV_DU(Thetd0, Thetd1);

        // Assign the interpolated state
        Rt = Xa.R*ExpThet;
        Ot = JrThet*Thetd1;
        St = JrThet*Thetd2 + HThet_ThetThetd1*Thetd1;
        Pt = pvat.block(0, 0, 3, 1);
        Vt = pvat.block(3, 0, 3, 1);
        At = pvat.block(6, 0, 3, 1);

        // Calculate the Jacobian
        DXt_DXa = vector<vector<Mat3T>>(6, vector<Mat3T>(6, Mat3T::Zero()));
        DXt_DXb = vector<vector<Mat3T>>(6, vector<Mat3T>(6, Mat3T::Zero()));


        // Local index for the states in the state vector
        const int RIDX = 0;
        const int OIDX = 1;
        const int SIDX = 2;
        const int PIDX = 3;
        const int VIDX = 4;
        const int AIDX = 5;


        // Some reusable matrices
        SO3T ExpThetInv = ExpThet.inverse();
        Mat3T HpTheb_ThebSb = DJrInvUV_DU(Theb, Xb.S);
        Mat3T LpThebTheb_ThebObThebd1 = DDJrInvUVW_DUDU(Theb, Xb.O, Thebd1);
        Mat3T LpThebOb_ThebObThebd1 = DDJrInvUVW_DUDV(Theb, Xb.O, Thebd1);

        Mat3T HThet_ThetThetd2 = DJrUV_DU(Thetd0, Thetd2);
        Mat3T LThetThet_ThetThetd1Thetd1 = DDJrUVW_DUDU(Thetd0, Thetd1, Thetd1);
        Mat3T LThetThetd1_ThetThetd1Thetd1 = DDJrUVW_DUDV(Thetd0, Thetd1, Thetd1);
        

        // Jacobians from L1 to L0
        Mat3T JThead1Oa = Mat3T::Identity(); Mat3T JThead2Sa = Mat3T::Identity();

        Mat3T  JThebd0Ra = -JrInvTheb*Rab.inverse().matrix();
        Mat3T &JThebd0Rb =  JrInvTheb;

        Mat3T  JThebd1Ra = HpTheb_ThebOb*JThebd0Ra;
        Mat3T  JThebd1Rb = HpTheb_ThebOb*JThebd0Rb;
        Mat3T &JThebd1Ob = JrInvTheb;

        Mat3T  JThebd2Ra = HpTheb_ThebSb*JThebd0Ra + HpTheb_ThebOb*JThebd1Ra + LpThebTheb_ThebObThebd1*JThebd0Ra;
        Mat3T  JThebd2Rb = HpTheb_ThebSb*JThebd0Rb + HpTheb_ThebOb*JThebd1Rb + LpThebTheb_ThebObThebd1*JThebd0Rb;
        Mat3T  JThebd2Ob = LpThebOb_ThebObThebd1   + HpTheb_ThebOb*JThebd1Ob;
        Mat3T &JThebd2Sb = JrInvTheb;


        // Jacobians from L2 to L1
        Mat3T JThetd0Thead0 = LAM_ROSt.block(0, 0, 3, 3); Mat3T JThetd0Thead1 = LAM_ROSt.block(0, 3, 3, 3); Mat3T JThetd0Thead2 = LAM_ROSt.block(0, 6, 3, 3);
        Mat3T JThetd1Thead0 = LAM_ROSt.block(3, 0, 3, 3); Mat3T JThetd1Thead1 = LAM_ROSt.block(3, 3, 3, 3); Mat3T JThetd1Thead2 = LAM_ROSt.block(3, 6, 3, 3);
        Mat3T JThetd2Thead0 = LAM_ROSt.block(6, 0, 3, 3); Mat3T JThetd2Thead1 = LAM_ROSt.block(6, 3, 3, 3); Mat3T JThetd2Thead2 = LAM_ROSt.block(6, 6, 3, 3);

        Mat3T JThetd0Thebd0 = PSI_ROSt.block(0, 0, 3, 3); Mat3T JThetd0Thebd1 = PSI_ROSt.block(0, 3, 3, 3); Mat3T JThetd0Thebd2 = PSI_ROSt.block(0, 6, 3, 3);
        Mat3T JThetd1Thebd0 = PSI_ROSt.block(3, 0, 3, 3); Mat3T JThetd1Thebd1 = PSI_ROSt.block(3, 3, 3, 3); Mat3T JThetd1Thebd2 = PSI_ROSt.block(3, 6, 3, 3);
        Mat3T JThetd2Thebd0 = PSI_ROSt.block(6, 0, 3, 3); Mat3T JThetd2Thebd1 = PSI_ROSt.block(6, 3, 3, 3); Mat3T JThetd2Thebd2 = PSI_ROSt.block(6, 6, 3, 3);


        // Jacobians from L2 to L0
        Mat3T JThetd0Ra = JThetd0Thebd0*JThebd0Ra + JThetd0Thebd1*JThebd1Ra + JThetd0Thebd2*JThebd2Ra;
        Mat3T JThetd0Rb = JThetd0Thebd0*JThebd0Rb + JThetd0Thebd1*JThebd1Rb + JThetd0Thebd2*JThebd2Rb;
        Mat3T JThetd0Oa = JThetd0Thead1*JThead1Oa;
        Mat3T JThetd0Ob = JThetd0Thebd1*JThebd1Ob + JThetd0Thebd2*JThebd2Ob;
        Mat3T JThetd0Sa = JThetd0Thead2*JThead2Sa;
        Mat3T JThetd0Sb = JThetd0Thebd2*JThebd2Sb;

        Mat3T JThetd1Ra = JThetd1Thebd0*JThebd0Ra + JThetd1Thebd1*JThebd1Ra + JThetd1Thebd2*JThebd2Ra;
        Mat3T JThetd1Rb = JThetd1Thebd0*JThebd0Rb + JThetd1Thebd1*JThebd1Rb + JThetd1Thebd2*JThebd2Rb;
        Mat3T JThetd1Oa = JThetd1Thead1*JThead1Oa;
        Mat3T JThetd1Ob = JThetd1Thebd1*JThebd1Ob + JThetd1Thebd2*JThebd2Ob;
        Mat3T JThetd1Sa = JThetd1Thead2*JThead2Sa;
        Mat3T JThetd1Sb = JThetd1Thebd2*JThebd2Sb;

        Mat3T JThetd2Ra = JThetd2Thebd0*JThebd0Ra + JThetd2Thebd1*JThebd1Ra + JThetd2Thebd2*JThebd2Ra;
        Mat3T JThetd2Rb = JThetd2Thebd0*JThebd0Rb + JThetd2Thebd1*JThebd1Rb + JThetd2Thebd2*JThebd2Rb;
        Mat3T JThetd2Oa = JThetd2Thead1*JThead1Oa;
        Mat3T JThetd2Ob = JThetd2Thebd1*JThebd1Ob + JThetd2Thebd2*JThebd2Ob;
        Mat3T JThetd2Sa = JThetd2Thead2*JThead2Sa;
        Mat3T JThetd2Sb = JThetd2Thebd2*JThebd2Sb;


        // Jacobians from L3 to L2
        Mat3T &JRtThetd0 = JrThet;

        Mat3T &JOtThetd0 = HThet_ThetThetd1;
        Mat3T &JOtThetd1 = JrThet;

        Mat3T  JStThetd0 = HThet_ThetThetd2 + LThetThet_ThetThetd1Thetd1;
        Mat3T  JStThetd1 = LThetThetd1_ThetThetd1Thetd1 + HThet_ThetThetd1;
        Mat3T &JStThetd2 = JrThet;


        // DRt_DRa
        DXt_DXa[RIDX][RIDX] = ExpThetInv.matrix() + JRtThetd0*JThetd0Ra;
        // DRt_DOa
        DXt_DXa[RIDX][OIDX] = JRtThetd0*JThetd0Oa;
        // DRt_DSa
        DXt_DXa[RIDX][SIDX] = JRtThetd0*JThetd0Sa;
        // DRt_DPa DRt_DVa DRt_DAa are all zeros
        
        // DOt_Ra
        DXt_DXa[OIDX][RIDX] = JOtThetd0*JThetd0Ra + JOtThetd1*JThetd1Ra;
        // DOt_Oa
        DXt_DXa[OIDX][OIDX] = JOtThetd0*JThetd0Oa + JOtThetd1*JThetd1Oa;
        // DOt_Sa
        DXt_DXa[OIDX][SIDX] = JOtThetd0*JThetd0Sa + JOtThetd1*JThetd1Sa;
        // DOt_DPa DOt_DVa DOt_DAa are all zeros

        // DSt_Ra
        DXt_DXa[SIDX][RIDX] = JStThetd0*JThetd0Ra + JStThetd1*JThetd1Ra + JStThetd2*JThetd2Ra;
        // DSt_Oa
        DXt_DXa[SIDX][OIDX] = JStThetd0*JThetd0Oa + JStThetd1*JThetd1Oa + JStThetd2*JThetd2Oa;
        // DSt_Sa
        DXt_DXa[SIDX][SIDX] = JStThetd0*JThetd0Sa + JStThetd1*JThetd1Sa + JStThetd2*JThetd2Sa;
        // DSt_DPa DSt_DVa DSt_DAa are all zeros


        // DRt_DRb
        DXt_DXb[RIDX][RIDX] = JRtThetd0*JThetd0Rb;
        // DRt_DOb
        DXt_DXb[RIDX][OIDX] = JRtThetd0*JThetd0Ob;
        // DRt_DSb
        DXt_DXb[RIDX][SIDX] = JRtThetd0*JThetd0Sb;
        // DRt_DPb DRt_DVb DRt_DAb are all zeros
        
        // DOt_Rb
        DXt_DXb[OIDX][RIDX] = JOtThetd0*JThetd0Rb + JOtThetd1*JThetd1Rb;
        // DOt_Ob
        DXt_DXb[OIDX][OIDX] = JOtThetd0*JThetd0Ob + JOtThetd1*JThetd1Ob;
        // DOt_Sb
        DXt_DXb[OIDX][SIDX] = JOtThetd0*JThetd0Sb + JOtThetd1*JThetd1Sb;
        // DOt_DPb DOt_DVb DOt_DAb are all zeros

        // DSt_Rb
        DXt_DXb[SIDX][RIDX] = JStThetd0*JThetd0Rb + JStThetd1*JThetd1Rb + JStThetd2*JThetd2Rb;
        // DSt_Ob
        DXt_DXb[SIDX][OIDX] = JStThetd0*JThetd0Ob + JStThetd1*JThetd1Ob + JStThetd2*JThetd2Ob;
        // DSt_Sb
        DXt_DXb[SIDX][SIDX] = JStThetd0*JThetd0Sb + JStThetd1*JThetd1Sb + JStThetd2*JThetd2Sb;
        // DSt_DPb DSt_DVb DSt_DAb are all zeros




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

        gammaa_ = gammaa;
        gammab_ = gammab;
        gammat_ = gammat;
    }

    template <class T = double>
    void ComputeXtAndJacobiansSE3(const GPState<T> &Xa,
                                  const GPState<T> &Xb,
                                        GPState<T> &Xt,
                                  vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXa,
                                  vector<vector<Eigen::Matrix<T, 3, 3>>> &DXt_DXb,
                                  bool debug = false
                                 ) const
    {   
        using SO3T   = Sophus::SO3<T>;
        using SE3T   = Sophus::SE3<T>;

        using Vec3T  = Eigen::Matrix<T, 3,  1>;
        using Vec6T  = Eigen::Matrix<T, 6,  1>;
        using Vec18T = Eigen::Matrix<T, 18, 1>;

        // using Mat3T  = Eigen::Matrix<T, 3,  3>;
        using Mat6T  = Eigen::Matrix<T, 6,  6>;

        // Map the variables of the state
        double tau = Xt.t;
        SO3T   &Rt = Xt.R;
        Vec3T  &Ot = Xt.O;
        Vec3T  &St = Xt.S;
        Vec3T  &Pt = Xt.P;
        Vec3T  &Vt = Xt.V;
        Vec3T  &At = Xt.A;
        
        // // Calculate the the mixer matrixes
        // Matrix<T, 18, 18> LAM_ = LAMDA(tau, SigGN).template cast<T>();
        // Matrix<T, 18, 18> PSI_ = PSI(tau,   SigGN).template cast<T>();

        // // Find the relative transform
        
        // SE3T  Ta = SE3T(Xa.R, Xa.P);
        // Vec6T Ua; Ua << Xa.O, Xa.V;
        // Vec6T Wa; Wa << Xa.S, Xa.A;

        // SE3T  Tb = SE3T(Xb.R, Xb.P);
        // Vec6T Ub; Ub << Xb.O, Xb.V;
        // Vec6T Wb; Wb << Xb.S, Xb.A;

        // SE3T Tab = Ta.inverse()*Tb;

        // // Calculate the SO3 knots in relative form
        // Vec6T Xiad0 = Vec6T::Zero();
        // Vec6T Xiad1; Xiad1 << Xa.O, Xa.V;
        // Vec6T Xiad2; Xiad2 << Xa.S, Xa.A;

        // Vec6T Xib = Tab.log();
        // Mat6T JrInv_Xib = JrInv(Xib);
        // Mat6T JlInv_Xib = JlInv(Xib);
        // Mat6T HpXib_XibUb = DJrInvUV_DU(Xib, Ub);
        
        // Vec6T Xibd0 = Xib;
        // Vec6T Xibd1 = JrInv_Xib*Ub;
        // Vec6T Xibd2 = JrInv_Xib*Wb + HpXib_XibUb*Xibd1;

        // // Stack the  in vector form
        // Vec18T zetaa; zetaa << Xiad0, Xiad1, Xiad2;
        // Vec18T zetab; zetab << Xibd0, Xibd1, Xibd2;

        // // Mix the knots to get the interpolated states
        // Vec18T zetat = LAM_*zetaa + PSI_*zetab;

        // // Retrive the interpolated SO3 in relative form
        // Vec6T Xitd0 = zetat.block(0,  0, 6, 1);
        // Vec6T Xitd1 = zetat.block(6,  0, 6, 1);
        // Vec6T Xitd2 = zetat.block(12, 0, 6, 1);

        // Mat6T Jr_Xit  = Jr(Xitd0);
        // SE3T  Exp_Xit = SE3T::exp(Xitd0);
        // Mat6T HXit_XitXitd1 = DJrUV_DU(Xitd0, Xitd1);

        // // Assign the interpolated state
        // SE3T  Tt = Ta*Exp_Xit;
        // Vec6T Ut = Jr_Xit*Xitd1;
        // Vec6T Wt = Jr_Xit*Xitd2 + HXit_XitXitd1*Xitd1;

        // // Calculate the Jacobian
        // // DXt_DXa = vector<vector<Mat3T>>(6, vector<Mat3T>(6, Mat3T::Zero()));
        // // DXt_DXb = vector<vector<Mat3T>>(6, vector<Mat3T>(6, Mat3T::Zero()));


        // // // Local index for the states in the state vector
        // // const int RIDX = 0;
        // // const int OIDX = 1;
        // // const int SIDX = 2;
        // // const int PIDX = 3;
        // // const int VIDX = 4;
        // // const int AIDX = 5;


        // // // Some reusable matrices
        // // SO3T Exp_XitInv = Exp_Xit.inverse();
        // Mat6T HpXib_XibWb = DJrInvUV_DU(Xib, Wb);
        // // Mat6T LpXibXib_XibUbXibd1 = DDJrInvUVW_DUDU(Xib, Ub, Xibd1);
        // // Mat6T LpXibUb_XibUbXibd1 = DDJrInvUVW_DUDV(Xib, Ub, Xibd1);

        // // Mat3T HXit_XitXitd2 = DJrUV_DU(Xitd0, Xitd2);
        // // Mat3T LXitXit_XitXitd1Xitd1 = DDJrUVW_DUDU(Xitd0, Xitd1, Xitd1);
        // // Mat3T LXitXitd1_XitXitd1Xitd1 = DDJrUVW_DUDV(Xitd0, Xitd1, Xitd1);
        

        // // Jacobians from L1 to L0
        // Mat6T J_Xiad1_Ua = Mat6T::Identity(); Mat6T J_Xiad2_Wa = Mat6T::Identity();

        // Mat6T  J_Xibd0_Ta = -JlInv_Xib;
        // Mat6T &J_Xibd0_Tb =  JrInv_Xib;

        // Mat6T  JXibd1_Ta = HpXib_XibUb*J_Xibd0_Ta;
        // Mat6T  JXibd1_Tb = HpXib_XibUb*J_Xibd0_Tb;
        // Mat6T &JXibd1_Ub = JrInv_Xib;

        // // Mat6T  JXibd2_Ta = HpXib_XibWb*J_Xibd0_Ta + HpXib_XibUb*JXibd1_Ta + LpXibXib_XibUbXibd1*J_Xibd0_Ta;
        // // Mat6T  JXibd2_Tb = HpXib_XibWb*J_Xibd0_Tb + HpXib_XibUb*JXibd1_Tb + LpXibXib_XibUbXibd1*J_Xibd0_Tb;
        // // Mat6T  JXibd2_Ub = LpXibUb_XibUbXibd1     + HpXib_XibUb*JXibd1_Ub;
        // Mat6T &JXibd2_Wb = JrInv_Xib;


        // // // Jacobians from L2 to L1
        // // Mat3T JXitd0Xiad0 = LAM_ROSt.block(0, 0, 3, 3); Mat3T JXitd0Xiad1 = LAM_ROSt.block(0, 3, 3, 3); Mat3T JXitd0Xiad2 = LAM_ROSt.block(0, 6, 3, 3);
        // // Mat3T JXitd1Xiad0 = LAM_ROSt.block(3, 0, 3, 3); Mat3T JXitd1Xiad1 = LAM_ROSt.block(3, 3, 3, 3); Mat3T JXitd1Xiad2 = LAM_ROSt.block(3, 6, 3, 3);
        // // Mat3T JXitd2Xiad0 = LAM_ROSt.block(6, 0, 3, 3); Mat3T JXitd2Xiad1 = LAM_ROSt.block(6, 3, 3, 3); Mat3T JXitd2Xiad2 = LAM_ROSt.block(6, 6, 3, 3);

        // // Mat3T JXitd0Xibd0 = PSI_ROSt.block(0, 0, 3, 3); Mat3T JXitd0Xibd1 = PSI_ROSt.block(0, 3, 3, 3); Mat3T JXitd0Xibd2 = PSI_ROSt.block(0, 6, 3, 3);
        // // Mat3T JXitd1Xibd0 = PSI_ROSt.block(3, 0, 3, 3); Mat3T JXitd1Xibd1 = PSI_ROSt.block(3, 3, 3, 3); Mat3T JXitd1Xibd2 = PSI_ROSt.block(3, 6, 3, 3);
        // // Mat3T JXitd2Xibd0 = PSI_ROSt.block(6, 0, 3, 3); Mat3T JXitd2Xibd1 = PSI_ROSt.block(6, 3, 3, 3); Mat3T JXitd2Xibd2 = PSI_ROSt.block(6, 6, 3, 3);


        // // // Jacobians from L2 to L0
        // // Mat3T JXitd0Ta = JXitd0Xibd0*J_Xibd0_Ta + JXitd0Xibd1*JXibd1Ta + JXitd0Xibd2*JXibd2Ta;
        // // Mat3T JXitd0Tb = JXitd0Xibd0*J_Xibd0_Tb + JXitd0Xibd1*JXibd1Tb + JXitd0Xibd2*JXibd2Tb;
        // // Mat3T JXitd0Ua = JXitd0Xiad1*J_Xiad1_Ua;
        // // Mat3T JXitd0Ub = JXitd0Xibd1*JXibd1Ub + JXitd0Xibd2*JXibd2Ub;
        // // Mat3T JXitd0Wa = JXitd0Xiad2*J_Xiad2_Wa;
        // // Mat3T JXitd0Wb = JXitd0Xibd2*JXibd2Wb;

        // // Mat3T JXitd1Ta = JXitd1Xibd0*J_Xibd0_Ta + JXitd1Xibd1*JXibd1Ta + JXitd1Xibd2*JXibd2Ta;
        // // Mat3T JXitd1Tb = JXitd1Xibd0*J_Xibd0_Tb + JXitd1Xibd1*JXibd1Tb + JXitd1Xibd2*JXibd2Tb;
        // // Mat3T JXitd1Ua = JXitd1Xiad1*J_Xiad1_Ua;
        // // Mat3T JXitd1Ub = JXitd1Xibd1*JXibd1Ub + JXitd1Xibd2*JXibd2Ub;
        // // Mat3T JXitd1Wa = JXitd1Xiad2*J_Xiad2_Wa;
        // // Mat3T JXitd1Wb = JXitd1Xibd2*JXibd2Wb;

        // // Mat3T JXitd2Ta = JXitd2Xibd0*J_Xibd0_Ta + JXitd2Xibd1*JXibd1Ta + JXitd2Xibd2*JXibd2Ta;
        // // Mat3T JXitd2Tb = JXitd2Xibd0*J_Xibd0_Tb + JXitd2Xibd1*JXibd1Tb + JXitd2Xibd2*JXibd2Tb;
        // // Mat3T JXitd2Ua = JXitd2Xiad1*J_Xiad1_Ua;
        // // Mat3T JXitd2Ub = JXitd2Xibd1*JXibd1Ub + JXitd2Xibd2*JXibd2Ub;
        // // Mat3T JXitd2Wa = JXitd2Xiad2*J_Xiad2_Wa;
        // // Mat3T JXitd2Wb = JXitd2Xibd2*JXibd2Wb;


        // // // Jacobians from L3 to L2
        // // Mat3T &JRtXitd0 = Jr_Xit;

        // // Mat3T &JOtXitd0 = HXit_XitXitd1;
        // // Mat3T &JOtXitd1 = Jr_Xit;

        // // Mat3T  JStXitd0 = HXit_XitXitd2 + LXitXit_XitXitd1Xitd1;
        // // Mat3T  JStXitd1 = LXitXitd1_XitXitd1Xitd1 + HXit_XitXitd1;
        // // Mat3T &JStXitd2 = Jr_Xit;


        // // // DRt_DTa
        // // DXt_DXa[RIDX][RIDX] = Exp_XitInv.matrix() + JRtXitd0*JXitd0Ta;
        // // // DRt_DUa
        // // DXt_DXa[RIDX][OIDX] = JRtXitd0*JXitd0Ua;
        // // // DRt_DWa
        // // DXt_DXa[RIDX][SIDX] = JRtXitd0*JXitd0Wa;
        // // // DRt_DPa DRt_DVa DRt_DWa are all zeros
        
        // // // DOt_Ta
        // // DXt_DXa[OIDX][RIDX] = JOtXitd0*JXitd0Ta + JOtXitd1*JXitd1Ta;
        // // // DOt_Ua
        // // DXt_DXa[OIDX][OIDX] = JOtXitd0*JXitd0Ua + JOtXitd1*JXitd1Ua;
        // // // DOt_Wa
        // // DXt_DXa[OIDX][SIDX] = JOtXitd0*JXitd0Wa + JOtXitd1*JXitd1Wa;
        // // // DOt_DPa DOt_DVa DOt_DWa are all zeros

        // // // DSt_Ta
        // // DXt_DXa[SIDX][RIDX] = JStXitd0*JXitd0Ta + JStXitd1*JXitd1Ta + JStXitd2*JXitd2Ta;
        // // // DSt_Ua
        // // DXt_DXa[SIDX][OIDX] = JStXitd0*JXitd0Ua + JStXitd1*JXitd1Ua + JStXitd2*JXitd2Ua;
        // // // DSt_Wa
        // // DXt_DXa[SIDX][SIDX] = JStXitd0*JXitd0Wa + JStXitd1*JXitd1Wa + JStXitd2*JXitd2Wa;
        // // // DSt_DPa DSt_DVa DSt_DWa are all zeros


        // // // DRt_DTb
        // // DXt_DXb[RIDX][RIDX] = JRtXitd0*JXitd0Tb;
        // // // DRt_DUb
        // // DXt_DXb[RIDX][OIDX] = JRtXitd0*JXitd0Ub;
        // // // DRt_DWb
        // // DXt_DXb[RIDX][SIDX] = JRtXitd0*JXitd0Wb;
        // // // DRt_DPb DRt_DVb DRt_DWb are all zeros
        
        // // // DOt_Tb
        // // DXt_DXb[OIDX][RIDX] = JOtXitd0*JXitd0Tb + JOtXitd1*JXitd1Tb;
        // // // DOt_Ub
        // // DXt_DXb[OIDX][OIDX] = JOtXitd0*JXitd0Ub + JOtXitd1*JXitd1Ub;
        // // // DOt_Wb
        // // DXt_DXb[OIDX][SIDX] = JOtXitd0*JXitd0Wb + JOtXitd1*JXitd1Wb;
        // // // DOt_DPb DOt_DVb DOt_DWb are all zeros

        // // // DSt_Tb
        // // DXt_DXb[SIDX][RIDX] = JStXitd0*JXitd0Tb + JStXitd1*JXitd1Tb + JStXitd2*JXitd2Tb;
        // // // DSt_Ub
        // // DXt_DXb[SIDX][OIDX] = JStXitd0*JXitd0Ub + JStXitd1*JXitd1Ub + JStXitd2*JXitd2Ub;
        // // // DSt_Wb
        // // DXt_DXb[SIDX][SIDX] = JStXitd0*JXitd0Wb + JStXitd1*JXitd1Wb + JStXitd2*JXitd2Wb;
        // // // DSt_DPb DSt_DVb DSt_DWb are all zeros




        // // // Extract the blocks of R3 states
        // // Mat3T LAM_PVA11 = LAM_PVAt.block(0, 0, 3, 3); Mat3T LAM_PVA12 = LAM_PVAt.block(0, 3, 3, 3); Mat3T LAM_PVA13 = LAM_PVAt.block(0, 6, 3, 3);
        // // Mat3T LAM_PVA21 = LAM_PVAt.block(3, 0, 3, 3); Mat3T LAM_PVA22 = LAM_PVAt.block(3, 3, 3, 3); Mat3T LAM_PVA23 = LAM_PVAt.block(3, 6, 3, 3);
        // // Mat3T LAM_PVA31 = LAM_PVAt.block(6, 0, 3, 3); Mat3T LAM_PVA32 = LAM_PVAt.block(6, 3, 3, 3); Mat3T LAM_PVA33 = LAM_PVAt.block(6, 6, 3, 3);

        // // Mat3T PSI_PVA11 = PSI_PVAt.block(0, 0, 3, 3); Mat3T PSI_PVA12 = PSI_PVAt.block(0, 3, 3, 3); Mat3T PSI_PVA13 = PSI_PVAt.block(0, 6, 3, 3);
        // // Mat3T PSI_PVA21 = PSI_PVAt.block(3, 0, 3, 3); Mat3T PSI_PVA22 = PSI_PVAt.block(3, 3, 3, 3); Mat3T PSI_PVA23 = PSI_PVAt.block(3, 6, 3, 3);
        // // Mat3T PSI_PVA31 = PSI_PVAt.block(6, 0, 3, 3); Mat3T PSI_PVA32 = PSI_PVAt.block(6, 3, 3, 3); Mat3T PSI_PVA33 = PSI_PVAt.block(6, 6, 3, 3);

        // // // DPt_DPa
        // // DXt_DXa[PIDX][PIDX] = LAM_PVA11;
        // // // DPt_DVa
        // // DXt_DXa[PIDX][VIDX] = LAM_PVA12;
        // // // DPt_DWa
        // // DXt_DXa[PIDX][AIDX] = LAM_PVA13;
        
        // // // DVt_DPa
        // // DXt_DXa[VIDX][PIDX] = LAM_PVA21;
        // // // DVt_DVa
        // // DXt_DXa[VIDX][VIDX] = LAM_PVA22;
        // // // DVt_DWa
        // // DXt_DXa[VIDX][AIDX] = LAM_PVA23;

        // // // DAt_DPa
        // // DXt_DXa[AIDX][PIDX] = LAM_PVA31;
        // // // DAt_DVa
        // // DXt_DXa[AIDX][VIDX] = LAM_PVA32;
        // // // DAt_DWa
        // // DXt_DXa[AIDX][AIDX] = LAM_PVA33;

        // // // DPt_DPb
        // // DXt_DXb[PIDX][PIDX] = PSI_PVA11;
        // // // DRt_DPb
        // // DXt_DXb[PIDX][VIDX] = PSI_PVA12;
        // // // DRt_DWb
        // // DXt_DXb[PIDX][AIDX] = PSI_PVA13;

        // // // DVt_DPb
        // // DXt_DXb[VIDX][PIDX] = PSI_PVA21;
        // // // DVt_DVb
        // // DXt_DXb[VIDX][VIDX] = PSI_PVA22;
        // // // DVt_DWb
        // // DXt_DXb[VIDX][AIDX] = PSI_PVA23;
        
        // // // DAt_DPb
        // // DXt_DXb[AIDX][PIDX] = PSI_PVA31;
        // // // DAt_DVb
        // // DXt_DXb[AIDX][VIDX] = PSI_PVA32;
        // // // DAt_DWb
        // // DXt_DXb[AIDX][AIDX] = PSI_PVA33;

        // // gammaa_ = gammaa;
        // // gammab_ = gammab;
        // // gammat_ = gammat;
    }

    GPMixer &operator=(const GPMixer &other)
    {
        this->dt = other.dt;
        this->SigGa = other.SigGa;
        this->SigNu = other.SigNu;
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
    double dt = 0.0;

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
    GaussianProcess(double dt_, Mat3 SigGa_, Mat3 SigNu_, bool keepCov_=false)
        : dt(dt_), gpm(GPMixerPtr(new GPMixer(dt_, SigGa_, SigNu_))), keepCov(keepCov_) {};

    Mat3 getSigGa() const { return gpm->getSigGa(); }
    Mat3 getSigNu() const { return gpm->getSigNu(); }
    bool getKeepCov() const {return keepCov;}

    GPMixerPtr getGPMixerPtr()
    {
        return gpm;
    }

    double getMinTime() const
    {
        return t0;
    }

    double getMaxTime() const
    {
        return t0 + max(0, int(R.size()) - 1)*dt;
    }

    int getNumKnots() const
    {
        return int(R.size());
    }

    double getKnotTime(int kidx) const
    {
        return t0 + kidx*dt;
    }

    double getDt() const
    {
        return dt;
    }

    bool TimeInInterval(double t, double eps=0.0) const
    {
        return (t >= getMinTime() + eps && t < getMaxTime() - eps);
    }

    pair<int, double> computeTimeIndex(double t) const
    {
        int u = int((t - t0)/dt);
        double s = double(t - t0)/dt - u;
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
            return GPState(t0 + ua*dt, R[ua], O[ua], S[ua], P[ua], V[ua], A[ua]);
        }

        // Extract the states of the two adjacent knots
        GPState Xa = GPState(t0 + ua*dt, R[ua], O[ua], S[ua], P[ua], V[ua], A[ua]);
        if (fabs(s) < DOUBLE_EPSILON)
        {
            // printf(KYEL "Boundary issue: ub: %d, Rsz: %d, s: %f, 1-s: %f\n" RESET, ub, R.size(), s, fabs(1.0 - s));
            return Xa;
        }

        GPState Xb = GPState(t0 + ub*dt, R[ub], O[ub], S[ua], P[ub], V[ub], A[ub]);
        if (fabs(1.0 - s) < DOUBLE_EPSILON)
        {
            // printf(KYEL "Boundary issue: ub: %d, Rsz: %d, s: %f, 1-s: %f\n" RESET, ub, R.size(), s, fabs(1.0 - s));
            return Xb;
        }

        SO3d Tab = Xa.R.inverse()*Xb.R;

        // Relative angle between two knots
        Vec3 Thea     = Vec3::Zero();
        Vec3 Thedota  = Xa.O;
        Vec3 Theddota = Xa.S;

        Vec3 Theb     = Tab.log();
        Vec3 Thedotb  = gpm->JrInv(Theb)*Xb.O;
        Vec3 Theddotb = gpm->JrInv(Theb)*Xb.S + gpm->DJrInvUV_DU(Theb, Xb.O)*Thedotb;

        Eigen::Matrix<double, 9, 1> gammaa; gammaa << Thea, Thedota, Theddota;
        Eigen::Matrix<double, 9, 1> gammab; gammab << Theb, Thedotb, Theddotb;

        Eigen::Matrix<double, 9, 1> pvaa; pvaa << Xa.P, Xa.V, Xa.A;
        Eigen::Matrix<double, 9, 1> pvab; pvab << Xb.P, Xb.V, Xb.A;

        Eigen::Matrix<double, 9, 1> gammat; // Containing on-manifold states (rotation and angular velocity)
        Eigen::Matrix<double, 9, 1> pvat;   // Position, velocity, acceleration

        gammat = gpm->LAMDA_ROS(s*dt) * gammaa + gpm->PSI_ROS(s*dt) * gammab;
        pvat   = gpm->LAMDA_PVA(s*dt) * pvaa   + gpm->PSI_PVA(s*dt) * pvab;

        // Retrive the interpolated SO3 in relative form
        Vec3 Thet     = gammat.block(0, 0, 3, 1);
        Vec3 Thedott  = gammat.block(3, 0, 3, 1);
        Vec3 Theddott = gammat.block(6, 0, 3, 1);

        // Assign the interpolated state
        SO3d Rt = Xa.R*SO3d::exp(Thet);
        Vec3 Ot = gpm->Jr(Thet)*Thedott;
        Vec3 St = gpm->Jr(Thet)*Theddott + gpm->DJrUV_DU(Thet, Thedott)*Thedott;
        Vec3 Pt = pvat.block<3, 1>(0, 0);
        Vec3 Vt = pvat.block<3, 1>(3, 0);
        Vec3 At = pvat.block<3, 1>(6, 0);

        return GPState<double>(t, Rt, Ot, St, Pt, Vt, At);
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
            SO3d Rpred = Rc*SO3d::exp(dt*Oc + 0.5*dt*dt*Sc);
            Vec3 Opred = Oc + dt*Sc;
            Vec3 Spred = Sc;
            Vec3 Ppred = Pc + dt*Vc + 0.5*dt*dt*Ac;
            Vec3 Vpred = Vc + dt*Ac;
            Vec3 Apred = Ac;

            Rc = Rpred;
            Oc = Opred;
            Sc = Spred;
            Pc = Ppred;
            Vc = Vpred;
            Ac = Apred;
        }

        return GPState<double>(getMaxTime() + steps*dt, Rc, Oc, Sc, Pc, Vc, Ac);
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

        SO3d Rn = Rc*SO3d::exp(dt*Oc + 0.5*dt*dt*Sc);
        Vec3 On = Oc + dt*Sc;
        Vec3 Sn = Sc;
        Vec3 Pn = Pc + dt*Vc + 0.5*dt*dt*Ac;
        Vec3 Vn = Vc + dt*Ac;
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

    void setSigNu(const Matrix3d &m)
    {
        gpm->setSigNu(m);
    }

    void setSigGa(const Matrix3d &m)
    {
        gpm->setSigGa(m);
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
        this->dt = other.getDt();
        
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

    bool saveTrajectory(string log_dir, int lidx, vector<double> ts)
    {
        string log_ = log_dir + "/gptraj_" + std::to_string(lidx) + ".csv";
        std::ofstream logfile;
        logfile.open(log_); // Open the file for writing
        logfile.precision(std::numeric_limits<double>::digits10 + 1);

        logfile << "Dt:" << dt << ";Order:" << 3 << ";Knots:" << getNumKnots() << ";MinTime:" << t0 << ";MaxTime:" << getMaxTime()
                << ";SigGa:" << getSigGa()(0, 0) << "," << getSigGa()(0, 1) << "," << getSigGa()(0, 2) << ","
                             << getSigGa()(1, 0) << "," << getSigGa()(1, 1) << "," << getSigGa()(1, 2) << ","
                             << getSigGa()(2, 0) << "," << getSigGa()(2, 1) << "," << getSigGa()(2, 2)
                << ";SigNu:" << getSigNu()(0, 0) << "," << getSigNu()(0, 1) << "," << getSigNu()(0, 2) << ","
                             << getSigNu()(1, 0) << "," << getSigNu()(1, 1) << "," << getSigNu()(1, 2) << ","
                             << getSigNu()(2, 0) << "," << getSigNu()(2, 1) << "," << getSigNu()(2, 2)
                << ";keepCov:" << getKeepCov()
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

        double dt_inLog;
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
            Matrix3d logSigNu = strToMat3(splitstr(fields[fieldidx["SigNu"]], ':').back(), ',');
            Matrix3d logSigGa = strToMat3(splitstr(fields[fieldidx["SigGa"]], ':').back(), ',');
            double logDt = stod(splitstr(fields[fieldidx["Dt"]], ':').back());
            double logMinTime = stod(splitstr(fields[fieldidx["MinTime"]], ':').back());
            bool logkeepCov = (stoi(splitstr(fields[fieldidx["keepCov"]], ':').back()) == 1);

            printf("Log configs:\n");
            printf("Dt: %f\n", logDt);
            printf("MinTime: %f\n", logMinTime);
            printf("SigNu: \n");
            cout << logSigNu << endl;
            printf("SigGa: \n");
            cout << logSigGa << endl;

            dt_inLog = logDt;
            t0_inLog = logMinTime;
            gpm_inLog = GPMixerPtr(new GPMixer(logDt, logSigGa, logSigNu));

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
        
        if(dt == 0 || dt_inLog == dt)
        {
            printf("dt has not been set. Use log's dt %f.\n", dt_inLog);
            
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
            printf(KYEL "Logged GPCT is has different knot length. Chosen: %f. Log: %f.\n" RESET, dt, dt_inLog);
            
            // Create a trajectory
            GaussianProcess trajLog(dt_inLog, gpm_inLog->getSigGa(), gpm_inLog->getSigNu());
            trajLog.setStartTime(t0_inLog);

            // Create the trajectory
            for(int ridx = 0; ridx < traj.rows(); ridx++)
            {
                VectorXd X = traj.row(ridx);
                trajLog.extendOneKnot(GPState<double>(ridx*dt_inLog+t0_inLog, SO3d(Quaternd(X(5), X(2), X(3), X(4))),
                                                                              Vec3(X(6),  X(7),  X(8)),
                                                                              Vec3(X(9),  X(10), X(11)),
                                                                              Vec3(X(12), X(13), X(14)),
                                                                              Vec3(X(15), X(16), X(17)),
                                                                              Vec3(X(18), X(19), X(20))));
            }

            // Sample the log trajectory to initialize current trajectory
            t0 = t0_inLog;
            R.clear(); O.clear(); S.clear(); P.clear(); V.clear(); A.clear(); C.clear();
            for(double ts = t0; ts < trajLog.getMaxTime() - trajLog.getDt(); ts += dt)
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
