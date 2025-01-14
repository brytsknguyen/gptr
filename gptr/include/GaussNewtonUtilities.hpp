
#include "GaussianProcess.hpp"

struct OptReport
{
    int ceres_iterations = -1;
    map<string, double> tictocs;

    map<string, int> factors;
    map<string, double> costs;

    vector<GPState<double>> X0;
    vector<GPState<double>> Xt;

    bool marginalization_done = false;
};

enum class ParamType
{
    SO3, RV3, UNKNOWN
};

enum class ParamRole
{
    GPSTATE, EXTRINSIC, UNKNOWN
};

class ParamInfo
{
public:
    double* address = NULL; // Actual address of the param block
    ParamType type;         // Type of the param block (SO3 or RV3)
    ParamRole role;         // What this param is used for, state or extrinsics
    shared_ptr<void> ptr;   // Pointer to param directly
    bool fixed = false;     // Set to true to fix it

    int xidx = -1;          // Index of the parameter in the Hessian, to be set by the ParamInfoMap
    int pidx;               // Index of the parameter in the problem
    int tidx;               // Index of the trajectory
    int kidx;               // Index of the knot
    int sidx;               // Index of state in the knot

    int param_size;         // Size of the param in doubles
    int delta_size;         // Size of the dx in the "X \boxplus dx" of the param

    ParamInfo()
    {
        address = NULL;
        type = ParamType::UNKNOWN;
        role = ParamRole::UNKNOWN;
        // param_size = -1;
        pidx = -1;
        tidx = -1;
        kidx = -1;
        sidx = -1;
    }

    ParamInfo(double* address_, shared_ptr<void> ptr_, ParamType type_, ParamRole role_, int pidx_, int tidx_, int kidx_, int sidx_)
        : address(address_), ptr(ptr_), type(type_), role(role_), pidx(pidx_), tidx(tidx_), kidx(kidx_), sidx(sidx_)
    {
        if(type == ParamType::SO3)
        {
            param_size = 4;
            delta_size = 3;
        }
        else if(type == ParamType::RV3)
        {
            param_size = 3;
            delta_size = 3;
        }
        else
        {
            printf("Unknown type! %d\n", type);
            exit(-1);
        }
    };

    bool operator<(const ParamInfo &other) const
    {
        if (tidx == -1 && other.tidx != -1)
            return false;

        if (tidx != -1 && other.tidx == -1)
            return true;

        if ((tidx != -1 && other.tidx != -1) && (tidx < other.tidx))
            return true;

        if ((tidx != -1 && other.tidx != -1) && (tidx > other.tidx))
            return false;

        // Including the situation that two knots are 01
        if (tidx == other.tidx)
        {
            if (kidx == -1 && other.kidx != -1)
                return false;

            if (kidx != -1 && other.kidx == -1)
                return true;

            if ((kidx != -1 && other.kidx != -1) && (kidx < other.kidx))
                return true;

            if ((kidx != -1 && other.kidx != -1) && (kidx > other.kidx))
                return false;

            if (kidx == other.kidx)
            {
                if (sidx == -1 && other.sidx != -1)
                    return false;

                if (sidx != -1 && other.sidx == -1)
                    return true;

                if ((sidx != -1 && other.sidx != -1) && (sidx < other.sidx))
                    return true;

                if ((sidx != -1 && other.sidx != -1) && (sidx > other.sidx))
                    return false;

                return false;    
            }    
        }                
    }
};

// This one manages all params within one evaluation
struct ParamInfoMap
{
    int XSIZE = -1;
    map<double*, ParamInfo> params_info;
    
    // Queries
    int  size()                         { return params_info.size(); }
    ParamInfo& operator[](double *addr) { return params_info[addr];  }
    bool hasParam(double *addr)         { return params_info.find(addr) != params_info.end(); }

    // Modify
    void clear()                        { params_info.clear(); XSIZE = -1;}
    void insert(double *addr, ParamInfo info)
    {
        assert(params_info.find(addr) == params_info.end());
        params_info[addr] = info;
        XSIZE = XSIZE < 0 ? info.delta_size : XSIZE + info.delta_size;
        params_info[addr].xidx = XSIZE - info.delta_size;
    }
};

class FactorMeta
{
public:

    vector<double> stamp; // Time of the factor
    vector<ceres::ResidualBlockId> res;
    vector<vector<ParamInfo>> coupled_params;
    vector<shared_ptr<void>> coupled_coef;

    FactorMeta() {};
    FactorMeta(const FactorMeta &other)
        : res(other.res), coupled_params(other.coupled_params), coupled_coef(other.coupled_coef), stamp(other.stamp)
    {};

    // FactorMeta(int knots_coupled_)
    //     : knots_coupled(knots_coupled_)
    // {};

    // void ResizeTidx(int idx)
    // {
    //     tidx.resize(kidx.size(), vector<int>(knots_coupled, idx));
    // }

    FactorMeta operator+(const FactorMeta &other)
    {
        FactorMeta added(*this);
        
        added.stamp.insert(added.stamp.end(), other.stamp.begin(), other.stamp.end());
        added.res.insert(added.res.end(), other.res.begin(), other.res.end());
        added.coupled_params.insert(added.coupled_params.end(), other.coupled_params.begin(), other.coupled_params.end());
        added.coupled_coef.insert(added.coupled_coef.end(), other.coupled_coef.begin(), other.coupled_coef.end());

        return added;
    }

    FactorMeta &operator+=(const FactorMeta &other)
    {
        stamp.insert(stamp.end(), other.stamp.begin(), other.stamp.end());
        res.insert(res.end(), other.res.begin(), other.res.end());
        coupled_params.insert(coupled_params.end(), other.coupled_params.begin(), other.coupled_params.end());
        coupled_coef.insert(coupled_coef.end(), other.coupled_coef.begin(), other.coupled_coef.end());

        return *this;
    }

    int size()
    {
        return coupled_params.size();
    }
};

typedef std::shared_ptr<FactorMeta> FactorMetaPtr;

class MarginalizationInfo
{
public:
    MatrixXd Hkeep;
    VectorXd bkeep;
    MatrixXd Jkeep;
    VectorXd rkeep;

    vector<ParamInfo> keptParamInfo;
    map<double*, vector<double>> keptParamPrior;

    MarginalizationInfo() {};

    template<typename T>
    vector<T> SO3ToDouble(Sophus::SO3<T> &rot)
    {
        Quaternion<T> q = rot.unit_quaternion();
        return {q.x(), q.y(), q.z(), q.w()};
    }

    template<typename T>
    vector<T> RV3ToDouble(Matrix<T, 3, 1> &vec)
    {
        return {vec.x(), vec.y(), vec.z()};
    }

    template<typename T>
    Sophus::SO3<T> DoubleToSO3(vector<T> &rot)
    {
        return Sophus::SO3<T>(Quaternion<T>(rot[3], rot[0], rot[1], rot[2]));
    }

    template<typename T>
    Matrix<T, 3, 1> DoubleToRV3(vector<T> &vec)
    {
        return Matrix<T, 3, 1>(vec[0], vec[1], vec[2]);
    }

    template<typename T>
    Sophus::SO3<T> DoubleToSO3(const double* rot)
    {
        return Sophus::SO3<T>(Quaternion<T>(rot[3], rot[0], rot[1], rot[2]));
    }

    template<typename T>
    Matrix<T, 3, 1> DoubleToRV3(const double* vec)
    {
        return Matrix<T, 3, 1>(vec[0], vec[1], vec[2]);
    }

    vector<double*> getAllParamBlocks()
    {
        vector<double*> params;
        for(auto &kpi : keptParamInfo)
            params.push_back(kpi.address);
        return params;    
    }

    // Convert the H, b matrices to J, r matrices
    void HbToJr(const MatrixXd &H, const VectorXd &b, MatrixXd &J, VectorXd &r)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(H);

        Eigen::VectorXd S = Eigen::VectorXd((saes.eigenvalues().array() > 0).select(saes.eigenvalues().array(), 0));
        Eigen::VectorXd S_inv = Eigen::VectorXd((saes.eigenvalues().array() > 0).select(saes.eigenvalues().array().inverse(), 0));
        
        Eigen::VectorXd S_sqrt = S.cwiseSqrt();
        Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

        J = S_sqrt.asDiagonal() * saes.eigenvectors().transpose();
        r = S_inv_sqrt.asDiagonal() * saes.eigenvectors().transpose() * b;
    }
};

typedef std::shared_ptr<MarginalizationInfo> MarginalizationInfoPtr;

class MarginalizationFactor : public ceres::CostFunction
{
private:
        MarginalizationInfoPtr margInfo;

public:

    MarginalizationFactor(MarginalizationInfoPtr margInfo_) : margInfo(margInfo_)
    {
        int RES_SIZE = 0;
        // Set the parameter blocks sizes
        for(auto &param : margInfo->keptParamInfo)
        {
            // Confirm that the param is in the new map
            // assert(paramInfoMap.find(param.address) != paramInfoMap.end());

            if (param.type == ParamType::SO3)
                mutable_parameter_block_sizes()->push_back(4);
            else
                mutable_parameter_block_sizes()->push_back(3);

            RES_SIZE += param.delta_size;
        }

        // Set the residual sizes
        set_num_residuals(RES_SIZE);
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        vector<Vector3d> rprior_;
        vector<Matrix3d> Jprior_;
        vector<ParamType> type;
        
        vector<int> RES_BASE;
        int RES_SIZE = 0;

        // Iterate over groups of param blocks (9 for GP state)
        for(int idx = 0; idx < margInfo->keptParamInfo.size(); idx++)
        {
            ParamInfo &param = margInfo->keptParamInfo[idx];
            // assert(keptParamMap.find(parameters[idx]) != keptParamMap.end());

            // Find the residual
            if (param.type == ParamType::SO3)
            {
                SO3d xso3_est = margInfo->DoubleToSO3<double>(parameters[idx]);
                SO3d xso3_pri = margInfo->DoubleToSO3<double>(margInfo->keptParamPrior[param.address]);
                Vector3d res = (xso3_pri.inverse()*xso3_est).log();
                rprior_.push_back(res);
                Jprior_.push_back(GPMixer::JrInv(res));
                type.push_back(ParamType::SO3);

                RES_BASE.push_back(RES_SIZE);
                RES_SIZE += param.delta_size;
            }
            else if (param.type == ParamType::RV3)
            {
                Vector3d xr3_est = margInfo->DoubleToRV3<double>(parameters[idx]);
                Vector3d xr3_pri = margInfo->DoubleToRV3<double>(margInfo->keptParamPrior[param.address]);
                Vector3d res = xr3_est - xr3_pri;
                rprior_.push_back(res);
                Jprior_.push_back(Matrix3d::Identity());
                type.push_back(ParamType::RV3);

                RES_BASE.push_back(RES_SIZE);
                RES_SIZE += param.delta_size;
            }
            else
            {
                printf("Unknown param type! %d\n", param.type);
                exit(-1);
            }
        }

        int Nprior = rprior_.size();
        VectorXd rprior = VectorXd::Zero(RES_SIZE, 1);
        MatrixXd Jprior = MatrixXd::Zero(RES_SIZE, RES_SIZE);
        for(int idx = 0; idx < Nprior; idx++)
        {
            rprior.block<3, 1>(RES_BASE[idx], 0) = rprior_[idx];
            Jprior.block<3, 3>(RES_BASE[idx], RES_BASE[idx]) = Jprior_[idx];
        }

        VectorXd &rkeep = margInfo->rkeep;
        MatrixXd &Jkeep = margInfo->Jkeep;

        const MatrixXd &bkeep = margInfo->bkeep;
        const MatrixXd &Hkeep = margInfo->Hkeep;
        MatrixXd bmarg = bkeep + Hkeep*Jprior*rprior;
        MatrixXd Hmarg = Jprior.transpose()*Hkeep*Jprior;
        
        VectorXd rmarg(RES_SIZE, 1);
        MatrixXd Jmarg(RES_SIZE, RES_SIZE);
        // margInfo->HbToJr(Hmarg, bmarg, Jmarg, rmarg);
        rmarg = rkeep + Jkeep*Jprior*rprior;
        Jmarg = Jkeep*Jprior;

        // Export the residual
        Eigen::Map<VectorXd>(residuals, rmarg.rows()) = rmarg;

        // (*iteration)++;
        // printf("Iter: %d. rkeep: %.3f. rmarg: %.3f. Jkeep: %.3f. Jmarg: %.3f. Dif: %.3f. %.3f\n",
        //         (*iteration),
        //         rkeep.cwiseAbs().maxCoeff(), rmarg.cwiseAbs().maxCoeff(),
        //         Jkeep.cwiseAbs().maxCoeff(), Jmarg.cwiseAbs().maxCoeff(),
        //         (rkeep - rmarg).cwiseAbs().maxCoeff(),
        //         (Jkeep - Jmarg).cwiseAbs().maxCoeff());
        //         // cout << rkeep - rmarg << endl;

        // if (*iteration == 1 && (rmarg.hasNaN() || Jmarg.hasNaN()))
        // {
        //     printf("Marg has NaN\n");
        //     cout << "rmarg\n" << endl;
        //     cout << rmarg << endl;
        //     cout << "Jmarg\n" << endl;
        //     cout << Jmarg << endl;
        // }

        // Export the Jacobian
        if(jacobians)
        {
            for(int pidx = 0; pidx < Nprior; pidx++)
            {
                ParamInfo &param = margInfo->keptParamInfo[pidx];
                if(jacobians[pidx])
                {
                    Eigen::Map<Matrix<double, -1, -1, Eigen::RowMajor>> J(jacobians[pidx], rprior.rows(), param.param_size);
                    J.setZero();
                    J.leftCols(3) = Jmarg.middleCols(pidx*param.delta_size, param.delta_size);

                    // MatrixXd Jtmp = MatrixXd::Zero(rprior.rows(), param.param_size);
                    // Jtmp.leftCols(3) = J.leftCols(3);
                    // printf("Jprior %3d: %9.3f, %9.3f\n", pidx, Jtmp.cwiseAbs().maxCoeff(), Jkeep.middleCols(pidx*param.delta_size, param.delta_size).cwiseAbs().maxCoeff());
                    // cout << rprior << endl;
                }
            }
        }

        return true;
    }
};

class MarginalizationFactorTMN
{
private:
        MarginalizationInfoPtr margInfo;

public:

    MarginalizationFactorTMN(MarginalizationInfoPtr margInfo_) : margInfo(margInfo_)
    {
        int RES_SIZE = 0;
        // Set the parameter blocks sizes
        for(auto &param : margInfo->keptParamInfo)
        {
            // Confirm that the param is in the new map
            // assert(paramInfoMap.find(param.address) != paramInfoMap.end());

            // if (param.type == ParamType::SO3)
            //     mutable_parameter_block_sizes()->push_back(4);
            // else
            //     mutable_parameter_block_sizes()->push_back(3);

            RES_SIZE += param.delta_size;
        }

        // residual = VectorXd(RES_SIZE);
        // jacobian = MatrixXd(RES_SIZE, RES_SIZE);
    }

    bool Evaluate(bool computeJacobian=true)
    {
        vector<Vector3d> rprior_;
        vector<Matrix3d> Jprior_;
        vector<ParamType> type;
        
        vector<int> RES_BASE;
        int RES_SIZE = 0;

        // Iterate over groups of param blocks (9 for GP state)
        for(int idx = 0; idx < margInfo->keptParamInfo.size(); idx++)
        {
            ParamInfo &param = margInfo->keptParamInfo[idx];
            // assert(keptParamMap.find(param.address) != keptParamMap.end());

            // Find the residual
            if (param.type == ParamType::SO3)
            {
                SO3d xso3_est = *static_pointer_cast<SO3d>(param.ptr);
                SO3d xso3_pri = margInfo->DoubleToSO3<double>(margInfo->keptParamPrior[param.address]);
                Vec3 res = (xso3_pri.inverse()*xso3_est).log();
                rprior_.push_back(res);
                Jprior_.push_back(GPMixer::JrInv(res));
                type.push_back(ParamType::SO3);

                RES_BASE.push_back(RES_SIZE);
                RES_SIZE += param.delta_size;
            }
            else if (param.type == ParamType::RV3)
            {
                Vec3 xr3_est = *static_pointer_cast<Vec3>(param.ptr);
                Vec3 xr3_pri = margInfo->DoubleToRV3<double>(margInfo->keptParamPrior[param.address]);
                Vec3 res = xr3_est - xr3_pri;
                rprior_.push_back(res);
                Jprior_.push_back(Matrix3d::Identity());
                type.push_back(ParamType::RV3);

                RES_BASE.push_back(RES_SIZE);
                RES_SIZE += param.delta_size;
            }
            else
            {
                printf("Unknown param type! %d\n", param.type);
                exit(-1);
            }
        }

        // Assemble the new residual and jacobian into matrices
        int Nprior = rprior_.size();
        VectorXd rprior = VectorXd::Zero(RES_SIZE, 1);
        MatrixXd Jprior = MatrixXd::Zero(RES_SIZE, RES_SIZE);
        for(int idx = 0; idx < Nprior; idx++)
        {
            rprior.block<3, 1>(RES_BASE[idx], 0) = rprior_[idx];
            Jprior.block<3, 3>(RES_BASE[idx], RES_BASE[idx]) = Jprior_[idx];
        }

        VectorXd &rkeep = margInfo->rkeep;
        MatrixXd &Jkeep = margInfo->Jkeep;

        // const MatrixXd &bkeep = margInfo->bkeep;
        // const MatrixXd &Hkeep = margInfo->Hkeep;
        // MatrixXd bmarg = bkeep + Hkeep*Jprior*rprior;
        // MatrixXd Hmarg = Jprior.transpose()*Hkeep*Jprior;
        
        // VectorXd rmarg(RES_SIZE, 1);
        // MatrixXd Jmarg(RES_SIZE, RES_SIZE);
        residual = rkeep + Jkeep*Jprior*rprior;
        jacobian = Jkeep*Jprior;

        // Export the residual
        // Eigen::Map<VectorXd>(residuals, bkeep.rows()) = rmarg;

        // (*iteration)++;
        // printf("Iter: %d. rkeep: %.3f. rmarg: %.3f. Jkeep: %.3f. Jmarg: %.3f. Dif: %.3f. %.3f\n",
        //         (*iteration),
        //         rkeep.cwiseAbs().maxCoeff(), rmarg.cwiseAbs().maxCoeff(),
        //         Jkeep.cwiseAbs().maxCoeff(), Jmarg.cwiseAbs().maxCoeff(),
        //         (rkeep - rmarg).cwiseAbs().maxCoeff(),
        //         (Jkeep - Jmarg).cwiseAbs().maxCoeff());
        //         // cout << rkeep - rmarg << endl;

        // if (*iteration == 1 && (rmarg.hasNaN() || Jmarg.hasNaN()))
        // {
        //     printf("Marg has NaN\n");
        //     cout << "rmarg\n" << endl;
        //     cout << rmarg << endl;
        //     cout << "Jmarg\n" << endl;
        //     cout << Jmarg << endl;
        // }

        // Export the Jacobian
        // if(computeJacobian)
        // {
        //     for(int pidx = 0; pidx < Nprior; pidx++)
        //     {
        //         ParamInfo &param = margInfo->keptParamInfo[pidx];
        //         // if(jacobians[pidx])
        //         // {
        //             Eigen::Map<Matrix<double, -1, -1, Eigen::RowMajor>> J(jacobians[pidx], rprior.rows(), param.param_size);
        //             J.setZero();
        //             J.leftCols(3) = Jmarg.middleCols(pidx*param.delta_size, param.delta_size);

        //             MatrixXd Jtmp = MatrixXd::Zero(rprior.rows(), param.param_size);
        //             Jtmp.leftCols(3) = J.leftCols(3);

        //             // printf("Jprior %3d: %9.3f, %9.3f\n", pidx, Jtmp.cwiseAbs().maxCoeff(), Jkeep.middleCols(pidx*param.delta_size, param.delta_size).cwiseAbs().maxCoeff());
        //             // cout << rprior << endl;
        //         // }
        //     }
        // }

        return true;
    }

    MatrixXd H() { return  jacobian.transpose()*jacobian; }
    VectorXd b() { return -jacobian.transpose()*residual; }

    VectorXd residual;
    MatrixXd jacobian;
};
