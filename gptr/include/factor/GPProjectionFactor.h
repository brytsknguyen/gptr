/**
* This file is part of splio.
* 
* Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot sg>,
* School of EEE
* Nanyang Technological Univertsity, Singapore
* 
* For more information please see <https://britsknguyen.github.io>.
* or <https://github.com/brytsknguyen/splio>.
* If you use this code, please cite the respective publications as
* listed on the above websites.
* 
* splio is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* splio is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with splio.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <ceres/ceres.h>
#include "GaussianProcess.hpp"
#include "../utility.h"

using namespace Eigen;

class GPProjFactor: public ceres::CostFunction
{
public:

    // Destructor
    ~GPProjFactor() {};

    // Constructor
    GPProjFactor(const vector<Eigen::Vector2d> &proj_, const vector<int> &id_, const DoubleSphereCamera<double> &cam_model_, std::map<int, Eigen::Vector3d> corner_pos_3d_, double w_,
                 GPMixerPtr gpm_, double s_)
    :   proj        (proj_            ),
        id          (id_              ),
        corner_pos_3d(corner_pos_3d_  ),
        cam_model   (cam_model_       ),
        // cam_id      (cam_id_          ),
        w           (w_               ),
        Dt          (gpm_->getDt()    ),
        s           (s_               ),
        gpm         (gpm_             )

    {
        set_num_residuals(2*proj.size());

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

        mutable_parameter_block_sizes()->push_back(4);
        mutable_parameter_block_sizes()->push_back(3);
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/
        // Map parameters to the control point states
        GPState Xa(0);  gpm->MapParamToState(parameters, RaIdx, Xa);
        GPState Xb(Dt); gpm->MapParamToState(parameters, RbIdx, Xb);
        Sophus::SO3d R_i_c_so3 = Eigen::Map<Sophus::SO3d const>(parameters[RicIdx]);      
        Eigen::Matrix3d R_i_c =  R_i_c_so3.matrix();
        Eigen::Vector3d t_i_c = Eigen::Map<Eigen::Vector3d const>(parameters[ticIdx]);            

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the pose at sampling time --------------------------------------------------------------*/

        GPState Xt(s*Dt); vector<vector<Matrix3d>> DXt_DXa; vector<vector<Matrix3d>> DXt_DXb;
        gpm->ComputeXtAndJacobians(Xa, Xb, Xt, DXt_DXa, DXt_DXb);

        Eigen::Matrix3d R_c_w = (Xt.R.matrix() * R_i_c).transpose();
        Eigen::Vector3d t_c_w = - R_i_c.transpose() * (Xt.R.matrix().transpose() * Xt.P + t_i_c);

        if (!jacobians) {
            for (size_t i = 0; i < proj.size(); i++) {
                auto iter = corner_pos_3d.find(id[i]);
                assert(iter != corner_pos_3d.end());
                Vector3d p_w = iter->second;
                Vector3d p_i = Xt.R.matrix().transpose() * (p_w - Xt.P);
                Vector3d p_c = R_i_c.transpose() * (p_i - t_i_c);
                // Vector3d p_c = R_c_w * corner_pos_3d[id[i]].cast<double>() + t_c_w;
                Vector4d p_c4;
                p_c4 << p_c[0], p_c[1], p_c[2], 1;

                Eigen::Vector2d proj2d;
                bool success = cam_model.project(p_c4, proj2d);          
                if (success) {
                    residuals[2 * i + 0] = proj2d[0] - proj[i][0];
                    residuals[2 * i + 1] = proj2d[1] - proj[i][1];        

                    residuals[2 * i + 0] *= w;
                    residuals[2 * i + 1] *= w;  
                } else {
                    residuals[2 * i + 0] = 0;
                    residuals[2 * i + 1] = 0;                      
                }
            }
        } else {

            Matrix3d Rictp = R_i_c.transpose();
            Matrix<double, 3, 3> Dp_DPt  = - Rictp * Xt.R.matrix().transpose();   
            // Matrix<double, 3, 3> Dp_DRt  = - Dr_DPW * Xt.R.matrix() * SO3d::hat(offset);
            // Matrix<double, 3, 3> Dp_DRic  = Dr_DPW;        
            Matrix<double, 3, 3> Dp_Dtic  = - Rictp;   

            size_t num_proj = proj.size();
            SparseMatrix<double> Dr_DRt(num_proj*2, 3);
            SparseMatrix<double> Dr_DPt(num_proj*2, 3);
            SparseMatrix<double> Dr_DRic(num_proj*2, 3);
            SparseMatrix<double> Dr_Dtic(num_proj*2, 3);

            for (size_t i = 0; i < num_proj; i++) {

                auto iter = corner_pos_3d.find(id[i]);
                assert(iter != corner_pos_3d.end());
                Vector3d p_w = iter->second;              
                Vector3d p_i = Xt.R.matrix().transpose() * (p_w - Xt.P);
                Vector3d p_c = Rictp * (p_i - t_i_c);
                Vector4d p_c4;
                p_c4 << p_c[0], p_c[1], p_c[2], 1;

                Eigen::Vector2d proj2d;
                Eigen::Matrix<double, 2, 4> d_r_d_p;
                bool success = cam_model.project(p_c4, proj2d, &d_r_d_p);

                if (success) {
                    residuals[2 * i + 0] = w * (proj2d[0] - proj[i][0]);
                    residuals[2 * i + 1] = w * (proj2d[1] - proj[i][1]);

                } else {
                    residuals[2 * i + 0] = 0;
                    residuals[2 * i + 1] = 0;    
                    d_r_d_p.setZero();                  
                }

                Util::SetSparseMatBlock<double>(Dr_DRt,  i * 2, 0, w * d_r_d_p.leftCols<3>() * Rictp * SO3d::hat(p_i), false);
                Util::SetSparseMatBlock<double>(Dr_DPt,  i * 2, 0, w * d_r_d_p.leftCols<3>() * Dp_DPt                , false);
                Util::SetSparseMatBlock<double>(Dr_DRic, i * 2, 0, w * d_r_d_p.leftCols<3>() * SO3d::hat(p_c)        , false);
                Util::SetSparseMatBlock<double>(Dr_Dtic, i * 2, 0, w * d_r_d_p.leftCols<3>() * Dp_Dtic               , false);
            }

            Dr_DRt.makeCompressed();
            Dr_DPt.makeCompressed();
            Dr_DRic.makeCompressed();
            Dr_Dtic.makeCompressed();

            size_t idx;

            for(size_t idx = Ridx; idx <= Aidx; idx++)
            {
                size_t idxa = idx, idxb = idx+RbIdx;

                if (idx == Ridx)
                {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>> Dr_DXa(jacobians[idxa], num_proj*2, 4);
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>> Dr_DXb(jacobians[idxb], num_proj*2, 4);

                    if(jacobians[idxa]) { Dr_DXa.setZero(); Dr_DXa.block(0, 0, num_proj*2, 3) = Dr_DRt*DXt_DXa[Ridx][idx] + Dr_DPt*DXt_DXa[Pidx][idx]; }
                    if(jacobians[idxb]) { Dr_DXb.setZero(); Dr_DXb.block(0, 0, num_proj*2, 3) = Dr_DRt*DXt_DXb[Ridx][idx] + Dr_DPt*DXt_DXb[Pidx][idx]; }

                }
                else
                {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> Dr_DXa(jacobians[idxa], num_proj*2, 3);
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> Dr_DXb(jacobians[idxb], num_proj*2, 3);

                    if(jacobians[idxa]) { Dr_DXa.setZero(); Dr_DXa.block(0, 0, num_proj*2, 3) = Dr_DRt*DXt_DXa[Ridx][idx] + Dr_DPt*DXt_DXa[Pidx][idx]; }
                    if(jacobians[idxb]) { Dr_DXb.setZero(); Dr_DXb.block(0, 0, num_proj*2, 3) = Dr_DRt*DXt_DXb[Ridx][idx] + Dr_DPt*DXt_DXb[Pidx][idx]; }

                }
            }

            // // Jacobian on Ra
            // idx = RaIdx;
            // if (jacobians[idx])
            // {
            //     Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> Dr_DRa;
            //     Dr_DRa.resize(2*num_proj, 4);
            //     Dr_DRa.setZero();
            //     for (size_t i = 0; i < num_proj; i++) {
            //         Dr_DRa.block<2, 3>(2 * i, 0) = w*vec_Dr_DRt[i]*DXt_DXa[Ridx][Ridx];
            //     }
            //     int idx_jac = 0;
            //     for (size_t i = 0; i < 2*num_proj; i++) {
            //         for (size_t j = 0; j < 4; j++) {
            //             jacobians[idx][idx_jac] = Dr_DRa.coeff(i,j);
            //             idx_jac++;
            //         }
            //     }
                
            // }

            // // Jacobian on Oa
            // idx = OaIdx;
            // if (jacobians[idx])
            // {
            //     Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Dr_DOa;
            //     // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> Dr_DOa(jacobians[idx]);
            //     Dr_DOa.resize(2*num_proj, 3);
            //     Dr_DOa.setZero();
            //     for (size_t i = 0; i < num_proj; i++) {
            //         Dr_DOa.block<2, 3>(2 * i, 0) = w*vec_Dr_DRt[i]*DXt_DXa[Ridx][Oidx];
            //     }                
            //     int idx_jac = 0;
            //     for (size_t i = 0; i < 2*num_proj; i++) {
            //         for (size_t j = 0; j < 3; j++) {
            //             jacobians[idx][idx_jac] = Dr_DOa.coeff(i,j);
            //             idx_jac++;
            //         }
            //     }            
            // }

            // // Jacobian on Sa
            // idx = SaIdx;
            // if (jacobians[idx])
            // {
            //     Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Dr_DSa;
            //     // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> Dr_DSa(jacobians[idx]);
            //     Dr_DSa.resize(2*num_proj, 3);
            //     Dr_DSa.setZero();
            //     for (size_t i = 0; i < num_proj; i++) {
            //         Dr_DSa.block<2, 3>(2 * i, 0) = w*vec_Dr_DRt[i]*DXt_DXa[Ridx][Sidx];
            //     }                      
            //     int idx_jac = 0;
            //     for (size_t i = 0; i < 2*num_proj; i++) {
            //         for (size_t j = 0; j < 3; j++) {
            //             jacobians[idx][idx_jac] = Dr_DSa.coeff(i,j);
            //             idx_jac++;
            //         }
            //     }            
            // }

            // // Jacobian on Pa
            // idx = PaIdx;
            // if (jacobians[idx])
            // {
            //     Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Dr_DPa;
            //     // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> Dr_DPa(jacobians[idx]);
            //     Dr_DPa.resize(2*num_proj, 3);
            //     Dr_DPa.setZero();
            //     for (size_t i = 0; i < num_proj; i++) {
            //         Dr_DPa.block<2, 3>(2 * i, 0) = w*vec_Dr_DPt[i]*DXt_DXa[Pidx][Pidx];
            //     }                      
            //     int idx_jac = 0;
            //     for (size_t i = 0; i < 2*num_proj; i++) {
            //         for (size_t j = 0; j < 3; j++) {
            //             jacobians[idx][idx_jac] = Dr_DPa.coeff(i,j);
            //             idx_jac++;
            //         }
            //     }
            // }

            // // Jacobian on Va
            // idx = VaIdx;
            // if (jacobians[idx])
            // {
            //     Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Dr_DVa;
            //     // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> Dr_DVa(jacobians[idx]);
            //     Dr_DVa.resize(2*num_proj, 3);
            //     Dr_DVa.setZero();
            //     for (size_t i = 0; i < num_proj; i++) {
            //         Dr_DVa.block<2, 3>(2 * i, 0) = w*vec_Dr_DPt[i]*DXt_DXa[Pidx][Vidx];
            //     }                       
            //     int idx_jac = 0;
            //     for (size_t i = 0; i < 2*num_proj; i++) {
            //         for (size_t j = 0; j < 3; j++) {
            //             jacobians[idx][idx_jac] = Dr_DVa.coeff(i,j);
            //             idx_jac++;
            //         }
            //     }
            // }

            // // Jacobian on Aa
            // idx = AaIdx;
            // if (jacobians[idx])
            // {
            //     Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Dr_DAa;
            //     // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> Dr_DAa(jacobians[idx]);
            //     Dr_DAa.resize(2*num_proj, 3);
            //     Dr_DAa.setZero();
            //     for (size_t i = 0; i < num_proj; i++) {
            //         Dr_DAa.block<2, 3>(2 * i, 0) = w*vec_Dr_DPt[i]*DXt_DXa[Pidx][Aidx];
            //     }                   
            //     int idx_jac = 0;
            //     for (size_t i = 0; i < 2*num_proj; i++) {
            //         for (size_t j = 0; j < 3; j++) {
            //             jacobians[idx][idx_jac] = Dr_DAa.coeff(i,j);
            //             idx_jac++;
            //         }
            //     }
            // }

            // // Jacobian on Rb
            // idx = RbIdx;
            // if (jacobians[idx])
            // {
            //     Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> Dr_DRb;
            //     // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>> Dr_DRb(jacobians[idx]);
            //     Dr_DRb.resize(2*num_proj, 4);
            //     Dr_DRb.setZero();
            //     for (size_t i = 0; i < num_proj; i++) {
            //         Dr_DRb.block<2, 3>(2 * i, 0) = w*vec_Dr_DRt[i]*DXt_DXb[Ridx][Ridx];
            //     }                
            //     int idx_jac = 0;
            //     for (size_t i = 0; i < 2*num_proj; i++) {
            //         for (size_t j = 0; j < 4; j++) {
            //             jacobians[idx][idx_jac] = Dr_DRb.coeff(i,j);
            //             idx_jac++;
            //         }
            //     }
            // }

            // // Jacobian on Ob
            // idx = ObIdx;
            // if (jacobians[idx])
            // {
            //     Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Dr_DOb;
            //     // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> Dr_DOb(jacobians[idx]);
            //     Dr_DOb.resize(2*num_proj, 3);
            //     Dr_DOb.setZero();
            //     for (size_t i = 0; i < num_proj; i++) {
            //         Dr_DOb.block<2, 3>(2 * i, 0) = w*vec_Dr_DRt[i]*DXt_DXb[Ridx][Oidx];
            //     }                        
            //     int idx_jac = 0;
            //     for (size_t i = 0; i < 2*num_proj; i++) {
            //         for (size_t j = 0; j < 3; j++) {
            //             jacobians[idx][idx_jac] = Dr_DOb.coeff(i,j);
            //             idx_jac++;
            //         }
            //     }
            // }

            // // Jacobian on Sb
            // idx = SbIdx;
            // if (jacobians[idx])
            // {
            //     Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Dr_DSb;
            //     // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> Dr_DSb(jacobians[idx]);
            //     Dr_DSb.resize(2*num_proj, 3);
            //     Dr_DSb.setZero();
            //     for (size_t i = 0; i < num_proj; i++) {
            //         Dr_DSb.block<2, 3>(2 * i, 0) = w*vec_Dr_DRt[i]*DXt_DXb[Ridx][Sidx];
            //     }                        
            //     int idx_jac = 0;
            //     for (size_t i = 0; i < 2*num_proj; i++) {
            //         for (size_t j = 0; j < 3; j++) {
            //             jacobians[idx][idx_jac] = Dr_DSb.coeff(i,j);
            //             idx_jac++;
            //         }
            //     }
            // }

            // // Jacobian on Pb
            // idx = PbIdx;
            // if (jacobians[idx])
            // {
            //     Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Dr_DPb;
            //     // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> Dr_DPb(jacobians[idx]);
            //     Dr_DPb.resize(2*num_proj, 3);
            //     Dr_DPb.setZero();
            //     for (size_t i = 0; i < num_proj; i++) {
            //         Dr_DPb.block<2, 3>(2 * i, 0) = w*vec_Dr_DPt[i]*DXt_DXb[Pidx][Pidx];
            //     }                     
            //     int idx_jac = 0;
            //     for (size_t i = 0; i < 2*num_proj; i++) {
            //         for (size_t j = 0; j < 3; j++) {
            //             jacobians[idx][idx_jac] = Dr_DPb.coeff(i,j);
            //             idx_jac++;
            //         }
            //     }
            // }

            // // Jacobian on Vb
            // idx = VbIdx;
            // if (jacobians[idx])
            // {
            //     Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Dr_DVb;
            //     // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> Dr_DVb(jacobians[idx]);
            //     Dr_DVb.resize(2*num_proj, 3);
            //     Dr_DVb.setZero();
            //     for (size_t i = 0; i < num_proj; i++) {
            //         Dr_DVb.block<2, 3>(2 * i, 0) = w*vec_Dr_DPt[i]*DXt_DXb[Pidx][Vidx];
            //     }                      
            //     int idx_jac = 0;
            //     for (size_t i = 0; i < 2*num_proj; i++) {
            //         for (size_t j = 0; j < 3; j++) {
            //             jacobians[idx][idx_jac] = Dr_DVb.coeff(i,j);
            //             idx_jac++;
            //         }
            //     }
            // }

            // // Jacobian on Ab
            // idx = AbIdx;
            // if (jacobians[idx])
            // {
            //     Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> Dr_DAb;
            //     // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> Dr_DAb(jacobians[idx]);
            //     Dr_DAb.resize(2*num_proj, 3);
            //     Dr_DAb.setZero();
            //     for (size_t i = 0; i < num_proj; i++) {
            //         Dr_DAb.block<2, 3>(2 * i, 0) = w*vec_Dr_DPt[i]*DXt_DXb[Pidx][Aidx];
            //     }                      
            //     int idx_jac = 0;
            //     for (size_t i = 0; i < 2*num_proj; i++) {
            //         for (size_t j = 0; j < 3; j++) {
            //             jacobians[idx][idx_jac] = Dr_DAb.coeff(i,j);
            //             idx_jac++;
            //         }
            //     }
            // }

            idx = RicIdx;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>> Dr_DRic_(jacobians[idx], num_proj*2, 4);
                if(jacobians[idx]) { Dr_DRic_.setZero(); Dr_DRic_.block(0, 0, num_proj*2, 3) = Dr_DRic; }
            }

            idx = ticIdx;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> Dr_Dtic_(jacobians[idx], num_proj*2, 3);
                if(jacobians[idx]) { Dr_Dtic_.setZero(); Dr_Dtic_.block(0, 0, num_proj*2, 3) = Dr_Dtic; }
            }

        }

        /* #endregion Calculate the pose at sampling time -----------------------------------------------------------*/

        return true;
    }

private:

    // corner measurement
    vector<Eigen::Vector2d> proj;
    vector<int> id;
    std::map<int, Eigen::Vector3d> corner_pos_3d;
    DoubleSphereCamera<double> cam_model;

    // Weight
    double w = 10;

    // Gaussian process param
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

    const int RicIdx = 12;
    const int ticIdx = 13;

    // Spline param
    double Dt;
    double s;

    GPMixerPtr gpm;
};