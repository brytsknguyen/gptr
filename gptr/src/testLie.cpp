#include "utility.h"
// #include "SophusExtras.hpp"
#include "GaussianProcess.hpp"

using namespace Eigen;
using namespace std;

typedef Eigen::Vector3d Vec3;
typedef Eigen::Matrix3d Mat3;


int main(int argc, char **argv)
{
    Vec3 X(4.3, 5.7, 91.0);
    Vec3 V(11, 2, 19);
    SO3d R = SO3d::exp(X);

    Mat3 Jr = GPMixer::Jr(X);
    Mat3 JrInv = GPMixer::JrInv(X);

    Vec3 _X = -X;
    Mat3 Jr_ = GPMixer::Jl(_X);
    Mat3 JrInv_ = GPMixer::JlInv(_X);

    cout << "Jr\n" << Jr << endl;
    cout << "JrInv\n" << JrInv << endl;
    cout << "Jr*JrInv\n" << Jr*JrInv << endl;

    cout << "Jr\n" << Jr_ << endl;
    cout << "JrInv\n" << JrInv_ << endl;
    cout << "Jr*JrInv\n" << Jr_*JrInv_ << endl;

    cout << "Jr err \n" << Jr - Jr_ << endl;
    cout << "JrInv err\n" << JrInv - JrInv_ << endl;

    // GPMixer gmp(0.01102, Vector3d(10, 10, 10).asDiagonal(), Vector3d(10, 10, 10).asDiagonal());

    Vec3 O = GPMixer::Jr(X)*V;
    Matrix3d DJrUV_DU_analytic = GPMixer::DJrUV_DU(X, V);
    Matrix3d DJrInvXVA_DX_analytic = GPMixer::DJrInvUV_DU(X, O);

    cout << "Hu(u,v)\n" << DJrUV_DU_analytic << endl;
    cout << "O\n" << O << endl;
    cout << "Hu(u,w)\n" << DJrInvXVA_DX_analytic << endl;
}