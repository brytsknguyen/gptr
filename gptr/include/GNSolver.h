#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

#include "utility.h"

#include "factor/GPPointToPlaneFactorXXX.hpp"
#include "factor/GPMotionPriorTwoKnotsFactorXXX.hpp"

struct GNSolverReport
{
   double J0lidar = -1;
   double J0mp2k  = -1;
   double J0prior = -1;
   double JKlidar = -1;
   double JKmp2k  = -1;
   double JKprior = -1;

   int lidarFactors = 0;
   int mp2kFactors = 0;
   int priorFactors = 0;
};

class GNSolver
{
private:

    // Node handle to get information needed
    NodeHandlePtr nh;

    int LIDX;

    // Lidar params
    double lidar_weight = 1.0;

    // Report of the latest optimizatio
    GNSolverReport report;

    // Weight for the motion prior factors
    double mpCovROSJerk = 1.0;
    double mpCovPVAJerk = 1.0;

    // Solver parameters
    int max_gniter = 3;
    double lambda = 1.0;
    double dxmax = 0.5;

    map<int, int> knots_keep_lckidx_gbkidx; // idx of knots to keep in graph
    map<int, int> knots_marg_lckidx_gbkidx; // idx of knots to remove from graph

    // Marginalization info
    MatrixXd Hkeep;
    VectorXd bkeep;
    MatrixXd Jm;
    VectorXd rm;

    // Matrix to solve the covariance
    SparseMatrix<double> InvCov;

    // Value of the keep state for prior
    map<int, GPState<double>>  knots_keep_gbidx_state;

    // Dictionary to covert absolute to local state idx
    map<int, int> absKidxToLocal;

    static mutex solver_mtx;

public:

    // Destructor
   ~GNSolver();

    // Constructor
    GNSolver(NodeHandlePtr &nh_, int &LIDX_);

    void EvaluateMotionPriorFactors
    (
        GaussianProcessPtr &traj,
        VectorXd &r, MatrixXd &J, double* cost
    );

    void EvaluateLidarFactors
    (
        const GaussianProcessPtr &traj,
        const deque<vector<LidarCoef>> &SwLidarCoef,
        VectorXd &r, MatrixXd &J, double* cost
    );

    void EvaluatePriorFactors
    (
        GaussianProcessPtr &traj,
        deque<int> swAbsKidx,
        SparseMatrix<double> &bprior_sparse,
        SparseMatrix<double> &Hprior_sparse,
        // VectorXd* bprior_reduced,
        // MatrixXd* Hprior_reduced,
        double* cost
    );

    void Marginalize
    (
        GaussianProcessPtr &traj,
        VectorXd &RESIDUAL,
        MatrixXd &JACOBIAN,
        SparseMatrix<double> &bprior_sparse,
        SparseMatrix<double> &Hprior_sparse,
        const deque<int> &swAbsKidx,
        const int &swNextBaseKnot,
        deque<vector<LidarCoef>> &SwLidarCoef
    );

    void HbToJr(const MatrixXd &H, const VectorXd &b, MatrixXd &J, VectorXd &r);

    bool Solve
    (
        GaussianProcessPtr &traj,
        deque<vector<LidarCoef>> &SwLidarCoef,
        const int iter,
        const deque<int> swAbsKidx,
        const int swNextBaseKnot
    );

    GNSolverReport &GetReport()
    {
        return report;
    }

    SparseMatrix<double> GetInvCov()
    {
        return InvCov;
    }
};

typedef std::shared_ptr<GNSolver> GNSolverPtr;