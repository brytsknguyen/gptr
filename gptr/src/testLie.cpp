#include "utility.h"
#include "GaussianProcess.hpp"
#include "SE3Q.hpp"

using namespace Eigen;
using namespace std;

typedef Eigen::Vector3d Vec3;
typedef Eigen::Matrix3d Mat3;


int main(int argc, char **argv)
{
    Vec3 X(4.3, 5.7, 91.0);
    Vec3 Xd(11, 2, 19);
    
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

    Vec3 O = GPMixer::Jr(X)*Xd;
    Matrix3d HX_XXd_direct = GPMixer::DJrUV_DU(X, Xd);
    Matrix3d HX_XXd_circle = -GPMixer::Jr(X)*GPMixer::DJrInvUV_DU(X, O);

    cout << "HX_XXd error:\n" << HX_XXd_direct - HX_XXd_circle << endl;
    cout << "HX_XXd_direct\n" << HX_XXd_direct << endl;
    cout << "HX_XXd_circle\n" << HX_XXd_circle << endl;   


    SE3Q<double> myQ;
    SE3Q<double> myQp;

    Vector3d The(4.3, 5.7, 91.0);
    Vector3d Rho(11.02, 03.04, 26.0);
    Vector3d Thed(7.1, 10.1, 10.0);
    Vector3d Rhod(3.3, 60.05, 26);
    Vector3d Omg = GPMixer::Jr(X)*The;
    
    myQ.ComputeQSC(The, Rho, Thed, Rhod);
}