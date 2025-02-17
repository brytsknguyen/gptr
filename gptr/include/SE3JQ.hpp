#include "utility.h"
#include <Eigen/Dense>

#ifndef SE3JQ_HPP
#define SE3JQ_HPP

using namespace Eigen;

template <typename T>
class SE3Q
{
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Mat3T = Eigen::Matrix<T, 3, 3>;

private:
    static constexpr int COMPONENTS = 4;

    static constexpr int S1_IDX = 3;
    static constexpr int C11_IDX = 6;
    static constexpr int C12_IDX = 9;
    static constexpr int C13_IDX = 12;

    static constexpr int S2_IDX = 15;
    static constexpr int C21_IDX = 18;
    static constexpr int C22_IDX = 21;
    static constexpr int C23_IDX = 24;

public:
    // Constructor
    SE3Q()
    {
        // ResetQSC();
    }

    void ResetQSC()
    {
        Q.setZero(); S1.setZero(); S2.setZero();
        
        C11.setZero(); C21.setZero();
        C12.setZero(); C22.setZero();
        C13.setZero(); C23.setZero();
    }

    void ComputeS(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed);

    void ComputeQSC(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod);

    Mat3T Q;
    Mat3T S1;
    Mat3T C11;
    Mat3T C12;
    Mat3T C13;
    Mat3T S2;
    Mat3T C21;
    Mat3T C22;
    Mat3T C23;
};

template <typename T>
class SE3Qp
{
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Mat3T = Eigen::Matrix<T, 3, 3>;

private:
    static constexpr int COMPONENTS = 36;

    static constexpr int S1_IDX = 3;
    static constexpr int C11_IDX = 6;
    static constexpr int C12_IDX = 9;
    static constexpr int C13_IDX = 12;

    static constexpr int S2_IDX = 15;
    static constexpr int C21_IDX = 18;
    static constexpr int C22_IDX = 21;
    static constexpr int C23_IDX = 24;

public:

    // Constructor
    SE3Qp()
    {
        // ResetQSC();
    }

    void ResetQSC()
    {
        Q.setZero(); S1.setZero(); S2.setZero();
        
        C11.setZero(); C21.setZero();
        C12.setZero(); C22.setZero();
        C13.setZero(); C23.setZero();
    }

    void ComputeS(const Vec3T &The, const Vec3T &Rho, const Vec3T &Omg);
    
    void ComputeQSC(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);

    Matrix<T, 3, 27> f01(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f02(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f03(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f04(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f05(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f06(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f07(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f08(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f09(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f10(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f11(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f12(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f13(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f14(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f15(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f16(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f17(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f18(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f19(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f20(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f21(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f22(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f23(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f24(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f25(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f26(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f27(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f28(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f29(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f30(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f31(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f32(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f33(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f34(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f35(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);
    Matrix<T, 3, 27> f36(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg);

    Mat3T Q;
    Mat3T S1;
    Mat3T C11;
    Mat3T C12;
    Mat3T C13;
    Mat3T S2;
    Mat3T C21;
    Mat3T C22;
    Mat3T C23;
};

extern template class SE3Q<double>;
extern template class SE3Q<ceres::Jet<double, 4>>;
extern template class SE3Qp<double>;
extern template class SE3Qp<ceres::Jet<double, 4>>;

#endif // SE3Q_HPP
