#include "GNSolver.h"

// Local size at individual factors
#define RES_LDR_LSIZE 1
#define RES_MP2_LSIZE 18
// Global size for the whole problem, to be changed in each solve
int RES_LDR_GSIZE;
int RES_MP2_GSIZE;
int RES_ALL_GSIZE;
int RES_LDR_GBASE;
int RES_MP2_GBASE;
// Size of each parameter block at individual level
// #define XROT_LSIZE 0
// #define XOMG_LSIZE 3
// #define XALP_LSIZE 6
// #define XPOS_LSIZE 9
// #define XVEL_LSIZE 12
// #define XACC_LSIZE 15
#define XALL_LSIZE STATE_DIM
int XALL_GSIZE;

void UpdateDimensions(int &numldr, int &nummp2, int &numKnots)
{
    RES_LDR_GSIZE = RES_LDR_LSIZE*numldr;
    RES_MP2_GSIZE = RES_MP2_LSIZE*nummp2;
    RES_ALL_GSIZE = RES_LDR_GSIZE + RES_MP2_GSIZE;

    RES_LDR_GBASE = 0;
    RES_MP2_GBASE = RES_LDR_GBASE + RES_LDR_GSIZE;

    XALL_GSIZE = XALL_LSIZE*numKnots;
}

mutex GNSolver::solver_mtx;

GNSolver::~GNSolver(){};

GNSolver::GNSolver(NodeHandlePtr &nh_, int &LIDX_) : nh(nh_), LIDX(LIDX_)
{
    nh->getParam("/lidar_weight", lidar_weight);

    nh->getParam("/max_gniter", max_gniter);
    nh->getParam("/lambda", lambda);
    nh->getParam("/dxmax", dxmax);

    // Weight for the motion prior
    nh->getParam("mpCovROSJerk", mpCovROSJerk);
    nh->getParam("mpCovPVAJerk", mpCovPVAJerk);
}

void GNSolver::EvaluateLidarFactors
(
    const GaussianProcessPtr &traj,
    const deque<vector<LidarCoef>> &SwLidarCoef,
    VectorXd &r, MatrixXd &J, double* cost
)
{
    // Index the points
    vector<int> lidarIdxBase(SwLidarCoef.size(), 0);
    for(int widx = 1; widx < SwLidarCoef.size(); widx++)
        lidarIdxBase[widx] += lidarIdxBase[widx-1] + SwLidarCoef[widx-1].size();

    // Create factors and calculate their residual and jacobian
    for(int widx = 0; widx < SwLidarCoef.size(); widx++)
    {
        auto &Coefs = SwLidarCoef[widx];

        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int cidx = 0; cidx < Coefs.size(); cidx++)
        {
            const LidarCoef &coef = Coefs[cidx];
            assert(cidx == coef.ptIdx);

            // Skip if lidar coef is not assigned
            if (coef.t < 0)
                continue;

            if (!traj->TimeInInterval(coef.t, 1e-6))
                continue;

            auto   us = traj->computeTimeIndex(coef.t);
            int    u  = us.first;
            double s  = us.second;

            typedef GPPointToPlaneFactorXXX ppFactor;
            ppFactor factor = ppFactor(coef.f, coef.n, coef.plnrty*lidar_weight, traj->getGPMixerPtr(), s);

            // Calculate the residual and jacobian
            factor.Evaluate(traj->getKnot(u), traj->getKnot(u+1));

            int row  = RES_LDR_GBASE + (lidarIdxBase[widx] + cidx)*RES_LDR_LSIZE;
            int cola = u*XALL_LSIZE;
            int colb = cola + XALL_LSIZE;

            r.block(row, 0, RES_LDR_LSIZE, 1).setZero();
            J.block(row, 0, RES_LDR_LSIZE, XALL_GSIZE).setZero();

            r.block<RES_LDR_LSIZE, 1>(row, 0) << factor.residual;
            J.block<RES_LDR_LSIZE, XALL_LSIZE>(row, cola) << factor.jacobian.block<RES_LDR_LSIZE, XALL_LSIZE>(0, 0);
            J.block<RES_LDR_LSIZE, XALL_LSIZE>(row, colb) << factor.jacobian.block<RES_LDR_LSIZE, XALL_LSIZE>(0, XALL_LSIZE);
        }
    }

    // Calculate the cost
    if (cost != NULL)
        *cost = pow(r.block(RES_LDR_GBASE, 0, RES_LDR_GSIZE, 1).norm(), 2);
}

void GNSolver::EvaluateMotionPriorFactors
(
    GaussianProcessPtr &traj,
    VectorXd &r, MatrixXd &J, double* cost
)
{
    // Number of lidar factors
    report.mp2kFactors = traj->getNumKnots() - 1;

    // Add GP factors between consecutive knots
    #pragma omp parallel for num_threads(MAX_THREADS)
    for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
    {
        // Create the factors
        typedef GPMotionPriorTwoKnotsFactorXXX mp2Factor;
        mp2Factor factor = mp2Factor(traj->getGPMixerPtr());

        // Calculate the residual and jacobian
        factor.Evaluate(traj->getKnot(kidx), traj->getKnot(kidx + 1));

        int row  = RES_MP2_GBASE + kidx*RES_MP2_LSIZE;
        int cola = kidx*XALL_LSIZE;
        int colb = cola + XALL_LSIZE;

        r.block(row, 0, RES_MP2_LSIZE, 1).setZero();
        J.block(row, 0, RES_MP2_LSIZE, XALL_GSIZE).setZero();

        r.block<RES_MP2_LSIZE, 1>(row, 0) << factor.residual;
        J.block<RES_MP2_LSIZE, XALL_LSIZE>(row, cola) << factor.jacobian.block<RES_MP2_LSIZE, XALL_LSIZE>(0, 0);
        J.block<RES_MP2_LSIZE, XALL_LSIZE>(row, colb) << factor.jacobian.block<RES_MP2_LSIZE, XALL_LSIZE>(0, XALL_LSIZE);
    }

    // Calculate the cost
    if (cost != NULL)
        *cost = pow(r.block(RES_MP2_GBASE, 0, RES_MP2_GSIZE, 1).norm(), 2);
}

void GNSolver::EvaluatePriorFactors
(
    GaussianProcessPtr &traj,
    deque<int> swAbsKidx,
    SparseMatrix<double> &bprior_sparse,
    SparseMatrix<double> &Hprior_sparse,
    // VectorXd* bprior_reduced,
    // MatrixXd* Hprior_reduced,
    double* cost
)
{
    report.priorFactors = 1;

    VectorXd rprior = VectorXd::Zero(XALL_GSIZE);
    MatrixXd Jprior = MatrixXd::Zero(XALL_GSIZE, XALL_GSIZE);
    VectorXd bprior = VectorXd::Zero(XALL_GSIZE);
    MatrixXd Hprior = MatrixXd::Zero(XALL_GSIZE, XALL_GSIZE);

    // Create marginalizing factor if there is states kept from last optimization
    SparseMatrix<double> rprior_sparse;
    SparseMatrix<double> Jprior_sparse;

    // printf("absKidx: \n");
    // for(auto &xkidx : absKidxToLocal)
    //     cout << xkidx.first << ", ";
    // cout << endl;

    // Calculate the prior residual
    for(auto &xkidx : knots_keep_gbidx_state)
    {
        int kidx = absKidxToLocal[xkidx.first];

        // printf("Build prior for knot %d, %d, %d.\n", kidx, xkidx.first);

        const GPState<> &Xbar = xkidx.second;
        const GPState<> &Xcur = traj->getKnot(absKidxToLocal[xkidx.first]);

        Matrix<double, STATE_DIM, 1> dX = Xcur.boxminus(Xbar);
        rprior.block<XALL_LSIZE, 1>(kidx*XALL_LSIZE, 0) << dX;
        Jprior.block<XALL_LSIZE, XALL_LSIZE>(kidx*XALL_LSIZE, kidx*XALL_LSIZE).setIdentity();

        // Add the inverse jacobian on the SO3 state
        Vec3 dR = dX.block<3, 1>(0, 0);
        if ( !(dR.norm() < 1e-3 || dR.hasNaN()) )
            Jprior.block<3, 3>(kidx*XALL_LSIZE, kidx*XALL_LSIZE) = GPMixer::JrInv(dR);
    }

    // Copy the blocks in big step
    int XKEEP_SIZE = knots_keep_lckidx_gbkidx.size()*XALL_LSIZE;
    Hprior.block(0, 0, XKEEP_SIZE, XKEEP_SIZE) = Hkeep.block(0, 0, XKEEP_SIZE, XKEEP_SIZE);

    // Update the b block
    bprior.block(0, 0, XKEEP_SIZE, 1) = bkeep.block(0, 0, XKEEP_SIZE, 1);

    if (Hprior.hasNaN() || bprior.hasNaN())
        return;

    // Update the hessian
    rprior_sparse = rprior.sparseView(); rprior_sparse.makeCompressed();
    Jprior_sparse = Jprior.sparseView(); Jprior_sparse.makeCompressed();

    bprior_sparse = bprior.sparseView(); bprior_sparse.makeCompressed();
    Hprior_sparse = Hprior.sparseView(); Hprior_sparse.makeCompressed();

    bprior_sparse = bprior_sparse - Hprior_sparse*Jprior_sparse*rprior_sparse;
    Hprior_sparse = Jprior_sparse.transpose()*Hprior_sparse*Jprior_sparse;

    // if(bprior_reduced != NULL)
    // {
    //     *bprior_reduced = VectorXd(XKEEP_SIZE, 1);
    //     *bprior_reduced << bprior_sparse.block(0, 0, XKEEP_SIZE, 1).toDense();
    // }

    // if(Hprior_reduced != NULL)
    // {
    //     *Hprior_reduced = MatrixXd(XKEEP_SIZE, XKEEP_SIZE);
    //     *Hprior_reduced << Hprior_sparse.block(0, 0, XKEEP_SIZE, XKEEP_SIZE).toDense();
    // }

    if (cost != NULL)
    {
        // For cost computation
        VectorXd rprior_reduced(XKEEP_SIZE, 1);
        MatrixXd Jprior_reduced(XKEEP_SIZE, XKEEP_SIZE);
        rprior_reduced << rprior.block(0, 0, XKEEP_SIZE, 1);
        Jprior_reduced << Jprior.block(0, 0, XKEEP_SIZE, XKEEP_SIZE);

        // printf("Sizes: rm: %d, %d. Jm: %d, %d. Jr: %d, %d. r: %d, %d\n",
        //         rm.rows(), rm.cols(),
        //         Jm.rows(), Jm.cols(),
        //         Jprior_reduced.rows(), Jprior_reduced.cols(),
        //         rprior_reduced.rows(), rprior_reduced.cols());

        if (cost != NULL)
            *cost = pow((rm + Jm*Jprior_reduced*rprior_reduced).norm(), 2);
    }
}

void GNSolver::HbToJr(const MatrixXd &H, const VectorXd &b, MatrixXd &J, VectorXd &r)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(H);
    Eigen::VectorXd S = Eigen::VectorXd((saes.eigenvalues().array() > 0).select(saes.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes.eigenvalues().array() > 0).select(saes.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    J = S_sqrt.asDiagonal() * saes.eigenvectors().transpose();
    r = S_inv_sqrt.asDiagonal() * saes.eigenvectors().transpose() * (-b);
}

void GNSolver::Marginalize
(
    GaussianProcessPtr &traj,
    VectorXd &RESIDUAL,
    MatrixXd &JACOBIAN,
    SparseMatrix<double> &bprior_sparse,
    SparseMatrix<double> &Hprior_sparse,
    const deque<int> &swAbsKidx,
    const int &swNextBaseKnot,
    deque<vector<LidarCoef>> &SwLidarCoef
)
{
    RESIDUAL.setZero();
    JACOBIAN.setZero();
    // VectorXd bprior_final_reduced;
    // MatrixXd Hprior_final_reduced;

    // Calculate the factors again at new linearized points
    EvaluateLidarFactors(traj, SwLidarCoef, RESIDUAL, JACOBIAN, &report.JKlidar);
    EvaluateMotionPriorFactors(traj, RESIDUAL, JACOBIAN, &report.JKmp2k);
    EvaluatePriorFactors(traj, swAbsKidx, bprior_sparse, Hprior_sparse, &report.JKprior);

    // Determine the marginalized states
    if (swNextBaseKnot > 0)
    {
        ROS_ASSERT_MSG(swNextBaseKnot == swAbsKidx[1], "%d, %d, %d, %f, %f, %f\n",
                       LIDX,
                       swNextBaseKnot,
                       swAbsKidx[1],
                       traj->getKnotTime(0),
                       traj->getKnotTime(absKidxToLocal[swAbsKidx[1]]),
                       traj->getKnotTime(absKidxToLocal[swNextBaseKnot]));

        // Index for each coef
        vector<int> lidarIdxBase(SwLidarCoef.size(), 0);
        for(int widx = 1; widx < SwLidarCoef.size(); widx++)
            lidarIdxBase[widx] += lidarIdxBase[widx-1] + SwLidarCoef[widx-1].size();

        // Store the states that currently couples with marginalization
        map<int, int> knots_keep_lckidx_gbkidx_prev = knots_keep_lckidx_gbkidx;

        // Determine the marginalized and keep knots
        knots_keep_lckidx_gbkidx.clear();
        knots_marg_lckidx_gbkidx.clear();
        for(int kidx = 0; kidx < traj->getNumKnots(); kidx++)
            if (swAbsKidx[kidx] < swNextBaseKnot)
                knots_marg_lckidx_gbkidx[kidx] = swAbsKidx[kidx];

        // Find the marginalized lidar residuals and the adjacent knots in the first bundle
        vector<int> res_ldr_marg;
        {
            auto &Coefs = SwLidarCoef[0];
            for(int cidx = 0; cidx < Coefs.size(); cidx++)
            {
                LidarCoef &coef = Coefs[cidx];
                assert(cidx == coef.ptIdx);

                // Skip if lidar coef is not assigned
                if (coef.t < 0)
                    continue;

                if (!traj->TimeInInterval(coef.t, 1e-6))
                    continue;

                auto   us = traj->computeTimeIndex(coef.t);
                int    u  = us.first;
                double s  = us.second;

                // If the base knot is in the remove set, log down the residual rows
                if (knots_marg_lckidx_gbkidx.find(u) != knots_marg_lckidx_gbkidx.end())
                {
                    // The base row of the residual
                    int RES_ROW_BASE = RES_LDR_GBASE + (lidarIdxBase[0] + cidx)*RES_LDR_LSIZE;
                    // Record the rows to marginalize
                    for(int row_idx = 0; row_idx < RES_LDR_LSIZE; row_idx++)
                        res_ldr_marg.push_back(RES_ROW_BASE + row_idx);
                    // If the coupled knot is not in the marginalized set, it is in the keep set
                    for(int kidx = u; kidx < u + 2; kidx++)
                        if(knots_marg_lckidx_gbkidx.find(kidx) == knots_marg_lckidx_gbkidx.end())
                            knots_keep_lckidx_gbkidx[kidx] = swAbsKidx.front() + kidx;
                }
            }
        }

        // Find the marginalized motion prior residuals and the adjacent knots
        vector<int> res_mp2_marg;
        {
            for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
            {
                if (knots_marg_lckidx_gbkidx.find(kidx) != knots_marg_lckidx_gbkidx.end())
                {
                    // The base row of the residual
                    int RES_ROW_BASE = RES_MP2_GBASE + kidx*RES_MP2_LSIZE;
                    // Record the rows to marginalize
                    for(int row_idx = 0; row_idx < RES_MP2_LSIZE; row_idx++)
                        res_mp2_marg.push_back(RES_ROW_BASE + row_idx);
                    // If the coupled knot is not in the marginalized set, it is in the keep set
                    for(int u = kidx; u < kidx + 2; u++)
                        if(knots_marg_lckidx_gbkidx.find(u) == knots_marg_lckidx_gbkidx.end())
                            knots_keep_lckidx_gbkidx[u] = swAbsKidx.front() + u;
                }
            }
        }

        // In this specific LO problem, all marginalized and kept states are the first two knots
        for (auto &xkidx : knots_marg_lckidx_gbkidx)
            ROS_ASSERT_MSG(xkidx.first == 0, "%d, %d\n", xkidx.first, xkidx.second);
        for (auto &xkidx : knots_keep_lckidx_gbkidx)
            ROS_ASSERT_MSG(xkidx.first == 1, "%d, %d\n", xkidx.first, xkidx.second);

        int RES_MARG_GSIZE = res_ldr_marg.size() + res_mp2_marg.size();
        int XM_XK_GSIZE = (knots_marg_lckidx_gbkidx.size() + knots_keep_lckidx_gbkidx.size())*STATE_DIM;

        // Extract the blocks of J and r related to marginalization
        MatrixXd Jmarg = MatrixXd::Zero(RES_MARG_GSIZE, XM_XK_GSIZE);
        MatrixXd rmarg = MatrixXd::Zero(RES_MARG_GSIZE, 1);

        // Extract the corresponding blocks
        int row_marg = 0;
        for(int row : res_mp2_marg)
        {
            Jmarg.row(row_marg) << JACOBIAN.block(row, 0, 1, XM_XK_GSIZE);
            rmarg.row(row_marg) = RESIDUAL.row(row);
            row_marg++;
        }
        for(int row : res_ldr_marg)
        {
            Jmarg.row(row_marg) << JACOBIAN.block(row, 0, 1, XM_XK_GSIZE);
            rmarg.row(row_marg) = RESIDUAL.row(row);
            row_marg++;
        }

        // Calculate the post-optimization hessian and gradient
        SparseMatrix<double> Jsparse = Jmarg.sparseView(); Jsparse.makeCompressed();
        SparseMatrix<double> Jtp = Jsparse.transpose();
        SparseMatrix<double> H = Jtp*Jsparse;
        MatrixXd b = -Jtp*rmarg;

        // Due to the simple structure of GP, the marginalized block
        // only consists of one state that will be marginalized
        MatrixXd H_xtra_(H.rows(), H.cols()); H_xtra_.setZero();
        VectorXd b_xtra_(b.rows(),        1); b_xtra_.setZero();
        H_xtra_.block<STATE_DIM, STATE_DIM>(0, 0) = Hprior_sparse.toDense().block<STATE_DIM, STATE_DIM>(0, 0);
        b_xtra_.block<STATE_DIM,         1>(0, 0) = bprior_sparse.toDense().block<STATE_DIM,         1>(0, 0);
        SparseMatrix<double> H_xtra = H_xtra_.sparseView(); H_xtra.makeCompressed();
        SparseMatrix<double> b_xtra = b_xtra_.sparseView(); b_xtra.makeCompressed();
        // Adding the old marginalization to the current one.
        H += H_xtra;
        b += b_xtra;

        // Divide the Hessian into corner blocks
        int MARG_GSIZE = knots_marg_lckidx_gbkidx.size()*STATE_DIM;
        int KEEP_GSIZE = H.cols() - MARG_GSIZE;
        SparseMatrix<double> Hmm = H.block(0, 0, MARG_GSIZE, MARG_GSIZE);
        SparseMatrix<double> Hmk = H.block(0, MARG_GSIZE, MARG_GSIZE, KEEP_GSIZE);
        SparseMatrix<double> Hkm = H.block(MARG_GSIZE, 0, KEEP_GSIZE, MARG_GSIZE);
        SparseMatrix<double> Hkk = H.block(MARG_GSIZE, MARG_GSIZE, KEEP_GSIZE, KEEP_GSIZE);

        MatrixXd bm = b.block(0, 0, MARG_GSIZE, 1);
        MatrixXd bk = b.block(MARG_GSIZE, 0, KEEP_GSIZE, 1);

        // Save the schur complement
        MatrixXd Hmminv = Hmm.toDense().inverse();
        MatrixXd HkmHmminv = Hkm*Hmminv;
        Hkeep = Hkk - HkmHmminv*Hmk;
        bkeep = bk  - HkmHmminv*bm;

        // Convert Hb to Jr for easier use
        HbToJr(Hkeep, bkeep, Jm, rm);

        // Store the values of the keep knots for reference in the next loop
        knots_keep_gbidx_state.clear();
        for(auto &xkidx : knots_keep_lckidx_gbkidx)
            knots_keep_gbidx_state[xkidx.second] = traj->getKnot(xkidx.first);
            //           AbsKnotIdx     state value at the knot idx

    }
}

bool GNSolver::Solve
(
    GaussianProcessPtr &traj,
    deque<vector<LidarCoef>> &SwLidarCoef,
    const int iter,
    const deque<int> swAbsKidx,
    const int swNextBaseKnot
)
{
    lock_guard<mutex> lg(solver_mtx);

    // Make a dictionary of the local idx and the abs knot idx
    absKidxToLocal.clear();
    for (int kidx = 0; kidx < swAbsKidx.size(); kidx++)
        absKidxToLocal[swAbsKidx[kidx]] = kidx;

    // Find the dimensions of the problem
    int numX = 0;
    int numLidarFactors = 0;
    int numMp2Factors = 0;

    // Each knot is counted as one state
    numX = traj->getNumKnots();
    // Each coefficient makes one factor
    for(auto &c : SwLidarCoef)
        numLidarFactors += c.size();
    // One mp2 factor between each knot
    numMp2Factors = traj->getNumKnots() - 1;

    // Number of lidar factors
    report.lidarFactors = numLidarFactors;
    report.mp2kFactors  = numMp2Factors;
    report.priorFactors = 1;

    // Determine the dimension of the Gauss-Newton Problem
    UpdateDimensions(numLidarFactors, numMp2Factors, numX);

    // Create the big Matrices
    VectorXd RESIDUAL(RES_ALL_GSIZE, 1);
    MatrixXd JACOBIAN(RES_ALL_GSIZE, XALL_GSIZE);
    SparseMatrix<double> bprior_sparse;
    SparseMatrix<double> Hprior_sparse;

    RESIDUAL.setZero();
    JACOBIAN.setZero();

    report.JKlidar = -1;
    report.JKmp2k  = -1;
    report.JKprior = -1;

    // Evaluate the lidar factors
    EvaluateLidarFactors(traj, SwLidarCoef, RESIDUAL, JACOBIAN, &report.J0lidar);

    // Evaluate the motion prior factors
    EvaluateMotionPriorFactors(traj, RESIDUAL, JACOBIAN, &report.J0mp2k);

    // Evaluate the motion prior factors
    EvaluatePriorFactors(traj, swAbsKidx, bprior_sparse, Hprior_sparse, &report.J0prior);

    // Build the problem and solve
    SparseMatrix<double> rsparse = RESIDUAL.sparseView(); rsparse.makeCompressed();
    SparseMatrix<double> Jsparse = JACOBIAN.sparseView(); Jsparse.makeCompressed();
    SparseMatrix<double> Jtp = Jsparse.transpose();
    SparseMatrix<double> b = -Jtp*rsparse;
    SparseMatrix<double> H =  Jtp*Jsparse;

    // Add the prior factor
    if(Hprior_sparse.rows() > 0 && bprior_sparse.rows() > 0)
    {
        H += Hprior_sparse;
        b += bprior_sparse;
        // printf("Hprior: %f\n", Hprior_sparse.toDense().trace());
    }

    // Fix the first knot
    // if(H.rows() > XALL_LSIZE && H.cols() > XALL_LSIZE)
    // {
    //     MatrixXd H_ = H.toDense();
    //     MatrixXd b_ = b.toDense();
    //     H_.block(0, 0, XALL_LSIZE, H_.cols()).setZero();
    //     H_.block(0, 0, H_.rows(), XALL_LSIZE).setZero();
    //     b_.block(0, 0, XALL_LSIZE, b_.cols()).setZero();
    //     H = H_.sparseView(); H.makeCompressed();
    //     b = b_.sparseView(); b.makeCompressed();
    // }

    MatrixXd dX = MatrixXd::Zero(XALL_GSIZE, 1);
    bool solver_failed = false;

    // Solving using dense QR
    // dX = S.toDense().colPivHouseholderQr().solve(b);

    // Solve using solver and LM method
    SparseMatrix<double> I(H.cols(), H.cols()); I.setIdentity();
    SparseMatrix<double> S = H + lambda/pow(2, iter)*I;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(S);
    solver.factorize(S);
    solver_failed = solver.info() != Eigen::Success;
    dX = solver.solve(b);

    // If solving is not successful, return false
    if (solver_failed || dX.hasNaN())
    {
        printf(KRED"Failed to solve!\n"RESET);
        for(int kidx = 0; kidx < traj->getNumKnots(); kidx++)
        {
            printf("Knot %d dX:\n", kidx);
            cout << dX.block<STATE_DIM, 1>(kidx*STATE_DIM, 0) << endl;
        }
    }

    // Apply saturation
    if (dX.norm() > dxmax)
        dX = dX / dX.norm() * dxmax;

    // Update the states
    for(int kidx = 0; kidx < traj->getNumKnots(); kidx++)
        traj->updateKnot(kidx, dX.block<STATE_DIM, 1>(kidx*STATE_DIM, 0));

    // Perform marginalization
    if(iter == max_gniter - 1 && swNextBaseKnot > 0)
        Marginalize(traj, RESIDUAL, JACOBIAN, bprior_sparse, Hprior_sparse, swAbsKidx, swNextBaseKnot, SwLidarCoef);

    // Save the covariance
    {
        SparseMatrix<double> rsparse = RESIDUAL.sparseView(); rsparse.makeCompressed();
        SparseMatrix<double> Jsparse = JACOBIAN.sparseView(); Jsparse.makeCompressed();
        SparseMatrix<double> Jtp = Jsparse.transpose();
        SparseMatrix<double> b = -Jtp*rsparse;
        SparseMatrix<double> H =  Jtp*Jsparse;

        // Add the prior factor
        if(Hprior_sparse.rows() > 0 && bprior_sparse.rows() > 0)
        {
            H += Hprior_sparse;
            b += bprior_sparse;
            // printf("Hprior: %f\n", Hprior_sparse.toDense().trace());
        }

        InvCov = H;
    }

    return true;
}