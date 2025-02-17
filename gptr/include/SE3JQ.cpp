#include "utility.h"
#include <Eigen/Dense>
#include "SE3JQ.hpp"

using namespace Eigen;

template <typename T>
void SE3Q<T>::ComputeS(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed)
{
    T Un = The.norm();
    Matrix<T, 3, 1> Ub = The / Un;
    Matrix<T, 1, 3> Ubtp = Ub.transpose();
    Matrix<T, 3, 3> JUb = (Mat3T::Identity(3, 3) - Ub*Ubtp) / Un;

    // This Q has 4 g and each has 3 derivatives 0th, 1st
    Matrix<T, COMPONENTS, 2> gdrv; gdrv.setZero();

    // This Q has 4 fs and each f has 3 derivatives (f, S1, S2) (4x3 x 3x3)
    Matrix<T, 3*COMPONENTS, 9> fdrv; fdrv.setZero();

    /* #region Calculating the derivatives of g -----------------------------------------------------------------*/

    {
        T t2 = cos(Un), t3 = sin(Un), t4 = Un*Un, t5 = 1.0/(Un*Un*Un), t7 = 1.0/(Un*Un*Un*Un*Un), t6 = 1.0/(t4*t4), t8 = -t3, t9 = Un+t8;

        gdrv(0, 0) = T(1.0/2.0); gdrv(1, 0) = t5*t9; gdrv(1, 1) = t6*t9*-3.0-t5*(t2-1.0); gdrv(2, 0) = (t6*(t2*2.0+t4-2.0))/2.0;
        gdrv(2, 1) = -t7*(t2*4.0+t4+Un*t3-4.0); gdrv(3, 0) = (t7*(Un*2.0-t3*3.0+Un*t2))/2.0; gdrv(3, 1) = 1.0/(t4*t4*t4)*(Un*8.0-t3*1.5E+1+Un*t2*7.0+t3*t4)*(-1.0/2.0);
    }

    /* #endregion Calculating the derivatives of g --------------------------------------------------------------*/

    /* #region Calculating the derivatives of f -----------------------------------------------------------------*/

    T tx = The(0), ty = The(1), tz = The(2);
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);

    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2);

    {
        T t2 = rx*tx, t3 = rx*ty, t4 = ry*tx, t5 = rx*tz, t6 = ry*ty, t7 = rz*tx, t8 = ry*tz, t9 = rz*ty, t10 = rz*tz, t11 = tx*ty, t12 = tx*tz, t13 = ty*tz, t14 = tx*tx, t15 = ty*ty, t16 = tz*tz, t17 = t2*2.0, t18 = t2*3.0, t19 = t3*2.0;
        T t20 = t4*2.0, t21 = t5*2.0, t22 = t6*2.0, t23 = t7*2.0, t24 = t6*3.0, t25 = t8*2.0, t26 = t9*2.0, t27 = t10*2.0, t28 = t10*3.0, t29 = t2*ty, t30 = t2*tz, t31 = t4*ty, t32 = t3*tz, t33 = t4*tz, t34 = t7*ty, t35 = t6*tz, t36 = t7*tz;
        T t37 = t9*tz, t38 = t11*tdx, t39 = t11*tdy, t40 = t12*tdx, t41 = t12*tdz, t42 = t13*tdy, t43 = t13*tdz, t44 = t13+tx, t45 = t12+ty, t46 = t11+tz, t53 = -t11, t54 = -t12, t55 = -t13, t56 = t14*tdy, t57 = t14*tdz, t58 = t15*tdx, t59 = t15*tdz;
        T t60 = t16*tdx, t61 = t16*tdy, t68 = -t14, t69 = -t15, t70 = -t16, t79 = t14+t15, t80 = t14+t16, t81 = t15+t16, t88 = t2+t6, t89 = t2+t10, t90 = t6+t10, t91 = t2*tdx*tx*-2.0, t92 = t6*tdy*ty*-2.0, t93 = t10*tdz*tz*-2.0, t47 = -t17;
        T t48 = -t20, t49 = -t22, t50 = -t23, t51 = -t26, t52 = -t27, t62 = t29*tdx*4.0, t63 = t30*tdx*4.0, t64 = t31*tdy*4.0, t65 = t35*tdy*4.0, t66 = t36*tdz*4.0, t67 = t37*tdz*4.0, t71 = -t33, t72 = -t34, t94 = rx*t79, t95 = rx*t80, t96 = ry*t79;
        T t97 = ry*t81, t98 = rz*t80, t99 = rz*t81, t100 = t18+t24, t101 = t18+t28, t102 = t24+t28, t106 = t27+t88, t107 = t22+t89, t108 = t17+t90, t115 = t17+t22+t27, t116 = t68+t81, t117 = t69+t80, t118 = t70+t79, t103 = t19+t48, t104 = t21+t50;
        T t105 = t25+t51, t109 = t32+t71, t110 = t32+t72, t111 = t33+t72, t121 = t31+t95, t122 = t36+t94, t123 = t29+t97, t124 = t37+t96, t125 = t30+t99, t126 = t35+t98, t127 = t115*tdx, t128 = t115*tdy, t129 = t115*tdz, t112 = t103*tdz;
        T t113 = t104*tdy, t114 = t105*tdx, t119 = -t112, t120 = -t114;

        fdrv(0, 1) = rz; fdrv(0, 2) = -ry; fdrv(0, 7) = -tdz; fdrv(0, 8) = tdy; fdrv(1, 0) = -rz; fdrv(1, 2) = rx; fdrv(1, 6) = tdz; fdrv(1, 8) = -tdx; fdrv(2, 0) = ry; fdrv(2, 1) = -rx; fdrv(2, 6) = -tdy;
        fdrv(2, 7) = tdx; fdrv(3, 0) = t49+t52; fdrv(3, 1) = t3+t4-t30-t90*tz; fdrv(3, 2) = t5+t7+t29+t90*ty; fdrv(3, 3) = tdz*(rz+t3)+tdy*(ry-t5); fdrv(3, 4) = ry*tdx*-2.0+t107*tdz+tdy*(rx-t8); fdrv(3, 5) = tdz*(rx+t9)-rz*tdx*2.0-t106*tdy;
        fdrv(3, 6) = t46*tdz-tdy*(t12-ty); fdrv(3, 7) = t59-tdx*ty*2.0-tdy*(t13-tx); fdrv(3, 8) = -t61+t44*tdz-tdx*tz*2.0; fdrv(4, 0) = t3+t4+t35+t89*tz; fdrv(4, 1) = t47+t52; fdrv(4, 2) = t8+t9-t31-t89*tx; fdrv(4, 3) = tdx*(ry+t5)-rx*tdy*2.0-t108*tdz;
        fdrv(4, 4) = tdx*(rx+t8)+tdz*(rz-t4); fdrv(4, 5) = rz*tdy*-2.0+t106*tdx+tdz*(ry-t7); fdrv(4, 6) = -t57+t45*tdx-tdy*tx*2.0; fdrv(4, 7) = t44*tdx-tdz*(t11-tz); fdrv(4, 8) = t60-tdy*tz*2.0-tdz*(t12-ty); fdrv(5, 0) = t5+t7-t37-t88*ty;
        fdrv(5, 1) = t8+t9+t36+t88*tx; fdrv(5, 2) = t47+t49; fdrv(5, 3) = rx*tdz*-2.0+t108*tdy+tdx*(rz-t3); fdrv(5, 4) = tdy*(rz+t4)-ry*tdz*2.0-t107*tdx; fdrv(5, 5) = tdy*(ry+t7)+tdx*(rx-t9); fdrv(5, 6) = t56-tdz*tx*2.0-tdx*(t11-tz);
        fdrv(5, 7) = -t58+t46*tdy-tdz*ty*2.0; fdrv(5, 8) = t45*tdy-tdx*(t13-tx); fdrv(6, 1) = -t99-t126+t17*tz+t102*tz; fdrv(6, 2) = t29*-2.0+t97+t124-t102*ty; fdrv(6, 3) = t113+t119; fdrv(6, 4) = -t129+t105*tdy; fdrv(6, 5) = t128+t105*tdz;
        fdrv(6, 6) = tx*(tdz*ty-tdy*tz)*-2.0; fdrv(6, 7) = t42*2.0+t117*tdz; fdrv(6, 8) = t43*-2.0-t118*tdy; fdrv(7, 0) = t35*-2.0+t98+t125-t101*tz; fdrv(7, 2) = -t95-t122+t101*tx+t20*ty; fdrv(7, 3) = t129-t104*tdx; fdrv(7, 4) = t119+t120;
        fdrv(7, 5) = -t127-t104*tdz; fdrv(7, 6) = t40*-2.0-t116*tdz; fdrv(7, 7) = ty*(tdz*tx-tdx*tz)*2.0; fdrv(7, 8) = t41*2.0+t118*tdx; fdrv(8, 0) = -t96-t123+t100*ty+t26*tz; fdrv(8, 1) = t36*-2.0+t94+t121-t100*tx; fdrv(8, 3) = -t128+t103*tdx;
        fdrv(8, 4) = t127+t103*tdy; fdrv(8, 5) = t113+t120; fdrv(8, 6) = t38*2.0+t116*tdy; fdrv(8, 7) = t39*-2.0-t117*tdx; fdrv(8, 8) = tz*(tdy*tx-tdx*ty)*-2.0; fdrv(9, 0) = t2*t15+t2*t16+t81*t90+t123*ty+t125*tz;
        fdrv(9, 1) = t3*t16-t3*t80+t53*t90-t123*tx+t111*tz; fdrv(9, 2) = t3*t13-t5*t79+t54*t90-t125*tx-t111*ty; fdrv(9, 3) = t92+t93-t29*tdy*4.0-t37*tdy*2.0-t30*tdz*4.0-t35*tdz*2.0+t19*tdx*ty+t21*tdx*tz;
        fdrv(9, 4) = t62-t64+t37*tdx*4.0-t36*tdy*2.0-t33*tdz*2.0-t2*tdy*tx*2.0+t6*tdx*ty*6.0+t25*tdx*tz; fdrv(9, 5) = t63-t66+t35*tdx*4.0-t34*tdy*2.0-t31*tdz*2.0-t2*tdz*tx*2.0+t26*tdx*ty+t10*tdx*tz*6.0; fdrv(9, 6) = tx*(t39+t41-t58-t60)*-2.0;
        fdrv(9, 7) = ty*(t39+t41-t58-t60)*-2.0; fdrv(9, 8) = tz*(t39+t41-t58-t60)*-2.0; fdrv(10, 0) = t4*t16-t4*t81+t53*t89-t121*ty+t110*tz; fdrv(10, 1) = t4*t11+t6*t16+t80*t89+t121*tx+t126*tz; fdrv(10, 2) = t4*t12-t8*t79+t55*t89-t110*tx-t126*ty;
        fdrv(10, 3) = -t62+t64-t37*tdx*2.0+t36*tdy*4.0-t32*tdz*2.0+t2*tdy*tx*6.0-t6*tdx*ty*2.0+t21*tdy*tz; fdrv(10, 4) = t91+t93-t31*tdx*4.0-t36*tdx*2.0-t30*tdz*2.0-t35*tdz*4.0+t20*tdy*tx+t25*tdy*tz;
        fdrv(10, 5) = t65-t67-t34*tdx*2.0+t30*tdy*4.0-t29*tdz*2.0+t23*tdy*tx-t6*tdz*ty*2.0+t10*tdy*tz*6.0; fdrv(10, 6) = tx*(t38+t43-t56-t61)*-2.0; fdrv(10, 7) = ty*(t38+t43-t56-t61)*-2.0; fdrv(10, 8) = tz*(t38+t43-t56-t61)*-2.0;
        fdrv(11, 0) = t7*t15-t7*t81+t54*t88+t109*ty-t122*tz; fdrv(11, 1) = t7*t11-t9*t80+t55*t88-t109*tx-t124*tz; fdrv(11, 2) = t7*t12+t9*t13+t79*t88+t122*tx+t124*ty;
        fdrv(11, 3) = -t63+t66-t35*tdx*2.0-t32*tdy*2.0+t31*tdz*4.0+t2*tdz*tx*6.0+t19*tdz*ty-t10*tdx*tz*2.0; fdrv(11, 4) = -t65+t67-t33*tdx*2.0-t30*tdy*2.0+t29*tdz*4.0+t20*tdz*tx+t6*tdz*ty*6.0-t10*tdy*tz*2.0;
        fdrv(11, 5) = t91+t92-t31*tdx*2.0-t36*tdx*4.0-t29*tdy*2.0-t37*tdy*4.0+t23*tdz*tx+t26*tdz*ty; fdrv(11, 6) = tx*(t40+t42-t57-t59)*-2.0; fdrv(11, 7) = ty*(t40+t42-t57-t59)*-2.0; fdrv(11, 8) = tz*(t40+t42-t57-t59)*-2.0;
    }
    
    /* #endregion Calculating the derivatives of f --------------------------------------------------------------*/

    S1.setZero(); S2.setZero();

    // Calculating the component jacobians
    for (int idx = 0; idx < COMPONENTS; idx++)
    {
        T &g = gdrv(idx, 0);
        T &dg = gdrv(idx, 1);

        Matrix<T, 1, 3> Jdg_U = dg*Ubtp;

        auto fXi = fdrv.block(idx*3, 0, 3, 3);
        auto dfXi_W_dU = fdrv.block(idx*3, 3, 3, 3);
        auto dfXi_W_dV = fdrv.block(idx*3, 6, 3, 3);

        // Q += fXi*g;

        // Find the S1 S2 jacobians
        const Vec3T &W = Thed;
        Vec3T fXiW = fXi*W;

        S1 += dfXi_W_dU*g + fXiW*Jdg_U;
        S2 += dfXi_W_dV*g;

        // cout << "ComputeS " << idx << ": g\n" << g << endl;
        // cout << "ComputeS " << idx << ": dfXi_W_dU\n" << dfXi_W_dU << endl;
        // cout << "ComputeS " << idx << ": dfXi_W_dV\n" << dfXi_W_dV << endl;
        // cout << "ComputeS " << idx << ": S1\n" << S1_ << endl;
        // cout << "ComputeS " << idx << ": S2\n" << S2_ << endl;
    }
}

template <typename T>
void SE3Q<T>::ComputeQSC(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod)
{
    T Un = The.norm();
    Matrix<T, 3, 1> Ub = The / Un;
    Matrix<T, 1, 3> Ubtp = Ub.transpose();
    Matrix<T, 3, 3> JUb = (Mat3T::Identity(3, 3) - Ub*Ubtp) / Un;

    // This Q has 4 g and each has 3 derivatives 0th, 1st and 2nd
    Matrix<T, COMPONENTS, 3> gdrv; gdrv.setZero();

    // This Q has 4 fs and each f has 9 derivatives (f, S1, C11, C12, C13, S2, C21, C22, C23) (4x3 x 9x3)
    Matrix<T, 3*COMPONENTS, 27> fdrv; fdrv.setZero();

    /* #region Calculating the derivatives of g -----------------------------------------------------------------*/

    {
        T t2 = cos(Un), t3 = sin(Un), t4 = Un*Un, t5 = 1.0/(Un*Un*Un), t7 = 1.0/(Un*Un*Un*Un*Un), t6 = 1.0/(t4*t4), t8 = 1.0/(t4*t4*t4), t9 = -t3, t10 = t3*t4, t11 = Un+t9;

        gdrv(0, 0) = T(1.0/2.0); gdrv(1, 0) = t5*t11; gdrv(1, 1) = t6*t11*-3.0-t5*(t2-1.0);
        gdrv(1, 2) = t7*(Un*6.0-t3*1.2E+1+t10+Un*t2*6.0); gdrv(2, 0) = (t6*(t2*2.0+t4-2.0))/2.0; gdrv(2, 1) = -t7*(t2*4.0+t4+Un*t3-4.0); gdrv(2, 2) = t8*(t2*2.0E+1+t4*3.0+Un*t3*8.0-t2*t4-2.0E+1); gdrv(3, 0) = (t7*(Un*2.0-t3*3.0+Un*t2))/2.0;
        gdrv(3, 1) = t8*(Un*8.0-t3*1.5E+1+t10+Un*t2*7.0)*(-1.0/2.0); gdrv(3, 2) = (1.0/(Un*Un*Un*Un*Un*Un*Un)*(Un*4.0E+1-t3*9.0E+1+t10*1.1E+1+Un*t2*5.0E+1-t2/t5))/2.0;
    }

    /* #endregion Calculating the derivatives of g --------------------------------------------------------------*/

    /* #region Calculating the derivatives of f -----------------------------------------------------------------*/

    T tx = The(0), ty = The(1), tz = The(2);
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);

    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2);
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2);

    {
        /* #region Repeated terms -------------------------------------------------------------------------------*/

        T t2 = rdx*tdx, t3 = rdy*tdy, t4 = rdz*tdz, t5 = rdx*tx, t6 = rx*tdx, t7 = rdy*ty, t8 = ry*tdy, t9 = rdz*tz, t10 = rz*tdz, t11 = rx*tx, t12 = rx*ty, t13 = ry*tx, t14 = rx*tz, t15 = ry*ty, t16 = rz*tx, t17 = ry*tz, t18 = rz*ty, t19 = rz*tz;
        T t20 = tdx*tdy, t21 = tdx*tdz, t22 = tdy*tdz, t23 = tdx*tx, t24 = tdy*tx, t25 = tdz*tx, t26 = tdx*ty, t27 = tdy*ty, t28 = tdz*ty, t29 = tdx*tz, t30 = tdy*tz, t31 = tdz*tz, t32 = tx*ty, t33 = tx*tz, t34 = ty*tz, t35 = tdx*2.0, t36 = tdy*2.0;
        T t37 = tdz*2.0, t38 = tx*tx, t39 = ty*ty, t40 = tz*tz, t106 = ry*tdx*-2.0, t110 = rz*tdx*-2.0, t111 = rz*tdy*-2.0, t41 = t5*2.0, t42 = t6*2.0, t43 = rx*t36, t44 = rx*t37, t45 = ry*t35, t46 = t7*2.0, t47 = t8*2.0, t48 = ry*t37, t49 = rz*t35;
        T t50 = rz*t36, t51 = t9*2.0, t52 = t10*2.0, t53 = t11*2.0, t54 = t11*3.0, t55 = t12*2.0, t56 = t13*2.0, t57 = t14*2.0, t58 = t15*2.0, t59 = t16*2.0, t60 = t15*3.0, t61 = t17*2.0, t62 = t18*2.0, t63 = t19*2.0, t64 = t19*3.0, t65 = t23*2.0;
        T t66 = t24*2.0, t67 = t25*2.0, t68 = t26*2.0, t69 = t27*2.0, t70 = t28*2.0, t71 = t29*2.0, t72 = t30*2.0, t73 = t31*2.0, t74 = t2*tx, t75 = t3*ty, t76 = t4*tz, t77 = t11*ty, t78 = t11*tz, t79 = t13*ty, t80 = t12*tz, t81 = t13*tz;
        T t82 = t16*ty, t83 = t15*tz, t84 = t16*tz, t85 = t18*tz, t86 = t23*ty, t87 = t24*ty, t88 = t23*tz, t89 = t25*tz, t90 = t27*tz, t91 = t28*tz, t92 = rx+t17, t93 = rx+t18, t94 = ry+t14, t95 = ry+t16, t96 = rz+t12, t97 = rz+t13, t98 = t25+tdy;
        T t99 = t26+tdz, t100 = t30+tdx, t101 = t34+tx, t102 = t33+ty, t103 = t32+tz, t107 = -t8, t112 = -t10, t116 = -t12, t117 = -t13, t119 = -t14, t120 = -t16, t123 = -t17, t124 = -t18, t127 = -t24, t129 = -t26, t131 = -t28, t133 = -t29;
        T t135 = -t30, t137 = -t32, t138 = -t33, t139 = -t34, t140 = t5*tx, t141 = t7*ty, t142 = t9*tz, t143 = t24*tx, t144 = t25*tx, t145 = t26*ty, t146 = t28*ty, t147 = t29*tz, t148 = t30*tz, t149 = t6*t36, t152 = t8*t37, t153 = t10*t35;
        T t155 = t6*tx*4.0, t156 = t11*t36, t157 = t11*tdy*4.0, t158 = t11*t37, t159 = t11*tdz*4.0, t161 = t13*t35, t162 = t6*ty*4.0, t164 = t12*t36, t165 = t8*tx*4.0, t166 = t12*t37, t168 = t12*tdz*4.0, t169 = t13*tdz*4.0, t171 = t15*t35;
        T t172 = t16*t35, t173 = t6*tz*4.0, t174 = t15*tdx*4.0, t175 = t14*t36, t177 = t14*tdy*4.0, t178 = t8*ty*4.0, t179 = t16*tdy*4.0, t181 = t14*t37, t182 = t15*t37, t183 = t15*tdz*4.0, t184 = t10*tx*4.0, t185 = t17*t35, t186 = t18*t35;
        T t187 = t17*tdx*4.0, t188 = t18*tdx*4.0, t190 = t18*t36, t191 = t8*tz*4.0, t193 = t17*t37, t194 = t10*ty*4.0, t195 = t19*t35, t196 = t19*tdx*4.0, t197 = t19*t36, t198 = t19*tdy*4.0, t199 = t10*tz*4.0, t217 = t25*ty*4.0, t221 = t24*tz*4.0;
        T t225 = t26*tz*4.0, t230 = t2*t32*4.0, t231 = t3*t32*4.0, t232 = t2*t33*4.0, t233 = t4*t33*4.0, t234 = t3*t34*4.0, t235 = t4*t34*4.0, t236 = t6*t32*4.0, t237 = t6*t33*4.0, t238 = t8*t32*4.0, t242 = t8*t34*4.0, t243 = t10*t33*4.0;
        T t244 = t10*t34*4.0, t245 = -t38, t246 = -t39, t247 = -t40, t249 = t11*tx*6.0, t251 = t15*ty*6.0, t253 = t19*tz*6.0, t263 = t6*tdz*-2.0, t264 = t8*tdx*-2.0, t265 = t10*tdy*-2.0, t290 = t26*tz*-2.0, t297 = t38+t39, t298 = t38+t40;
        T t299 = t39+t40, t306 = t2*t38*2.0, t307 = t3*t39*2.0, t308 = t4*t40*2.0, t318 = t6*t34*-2.0, t319 = t8*t33*-2.0, t320 = t10*t32*-2.0, t324 = t11+t15, t325 = t11+t19, t326 = t15+t19, t330 = t6*t38*-2.0, t331 = t8*t39*-2.0;
        T t332 = t10*t40*-2.0, t357 = t5+t7+t9, t104 = -t41, t105 = -t42, t108 = -t46, t109 = -t47, t113 = -t51, t114 = -t52, t115 = -t53, t118 = -t56, t121 = -t58, t122 = -t59, t125 = -t62, t126 = -t63, t128 = -t66, t130 = -t68, t132 = -t70;
        T t134 = -t71, t136 = -t72, t160 = t41*ty, t163 = t46*tx, t170 = t41*tz, t180 = t51*tx, t189 = t46*tz, t192 = t51*ty, t200 = t53*ty, t201 = t77*4.0, t202 = t53*tz, t203 = t56*ty, t204 = t78*4.0, t205 = t79*4.0, t206 = t58*tz, t207 = t59*tz;
        T t208 = t83*4.0, t209 = t84*4.0, t210 = t62*tz, t211 = t85*4.0, t212 = t65*ty, t213 = t86*4.0, t214 = t66*ty, t215 = t87*4.0, t216 = t67*ty, t218 = t65*tz, t219 = t88*4.0, t222 = t67*tz, t223 = t89*4.0, t226 = t69*tz, t227 = t90*4.0;
        T t228 = t70*tz, t229 = t91*4.0, t248 = t53*tx, t250 = t58*ty, t252 = t63*tz, t254 = t65*tx, t255 = t66*tx, t256 = t67*tx, t257 = t68*ty, t258 = t69*ty, t259 = t70*ty, t260 = t71*tz, t261 = t72*tz, t262 = t73*tz, t266 = -t162, t267 = -t165;
        T t268 = -t173, t270 = -t177, t271 = -t179, t272 = -t184, t276 = -t191, t277 = -t194, t278 = -t81, t279 = -t82, t283 = t87*-2.0, t286 = t88*-2.0, t291 = -t225, t295 = t91*-2.0, t300 = rx+t123, t301 = rx+t124, t302 = ry+t119, t303 = ry+t120;
        T t304 = rz+t116, t305 = rz+t117, t321 = t94*tdx, t322 = t97*tdy, t323 = t93*tdz, t327 = -t306, t328 = -t307, t329 = -t308, t333 = rx*t297, t334 = rx*t298, t335 = ry*t297, t336 = ry*t299, t337 = rz*t298, t338 = rz*t299, t339 = t6+t107;
        T t340 = t43+t45, t341 = t6+t112, t342 = t44+t49, t343 = t8+t112, t344 = t48+t50, t348 = t54+t60, t349 = t54+t64, t350 = t60+t64, t351 = t24+t129, t352 = t65+t69, t353 = t25+t133, t354 = t65+t73, t355 = t28+t135, t356 = t69+t73;
        T t367 = t63+t324, t368 = t58+t325, t369 = t53+t326, t388 = t53+t58+t63, t389 = t245+t299, t390 = t246+t298, t391 = t247+t297, t392 = t37+t68+t127, t393 = t36+t67+t133, t394 = t35+t72+t131, t395 = t32*t357*2.0, t396 = t33*t357*2.0;
        T t397 = t34*t357*2.0, t284 = -t215, t289 = -t223, t296 = -t229, t345 = t304*tdx, t346 = t300*tdy, t347 = t303*tdz, t358 = t42+t109, t359 = t42+t114, t360 = t47+t114, t361 = t55+t118, t362 = t57+t122, t363 = t61+t125, t364 = t66+t130;
        T t365 = t67+t134, t366 = t70+t136, t370 = t80+t278, t371 = t80+t279, t372 = t81+t279, t382 = t26+t37+t128, t383 = t25+t36+t134, t384 = t30+t35+t132, t385 = t369*tdx, t386 = t368*tdy, t387 = t367*tdz, t402 = t79+t334, t403 = t84+t333;
        T t404 = t77+t336, t405 = t85+t335, t406 = t78+t338, t407 = t83+t337, t408 = t388*tdx, t409 = t388*tdy, t410 = t388*tdz, t411 = rdx*t389, t412 = rdy*t390, t413 = rdz*t391, t414 = -t395, t415 = -t396, t416 = -t397, t420 = t203+t209+t248;
        T t421 = t205+t207+t248, t422 = t200+t211+t250, t423 = t201+t210+t250, t424 = t202+t208+t252, t425 = t204+t206+t252, t442 = t166+t186+t270+t271, t444 = t155+t164+t171+t196+t272, t445 = t155+t174+t181+t195+t267, t446 = t156+t161+t178+t198+t277;
        T t447 = t157+t178+t193+t197+t266, t448 = t158+t172+t183+t199+t276, t449 = t159+t182+t190+t199+t268, t373 = t361*tdx, t374 = t361*tdy, t375 = t361*tdz, t376 = t362*tdx, t377 = t362*tdy, t378 = t362*tdz, t379 = t363*tdx, t380 = t363*tdy;
        T t381 = t363*tdz, t417 = -t408, t418 = -t409, t419 = -t410, t426 = t421*tdy, t427 = t420*tdz, t428 = t423*tdx, t429 = t422*tdz, t430 = t425*tdx, t431 = t424*tdy, t398 = -t375, t399 = -t376, t400 = -t378, t401 = -t379, t432 = -t426;
        T t433 = -t427, t434 = -t428, t435 = -t429, t436 = -t430, t437 = -t431, t450 = t320+t432+t434, t451 = t319+t433+t436, t452 = t318+t435+t437;

        /* #endregion Repeated terms ----------------------------------------------------------------------------*/

        /* #region Calculating f and its jacobian ---------------------------------------------------------------*/

        fdrv(0, 1) = rz; fdrv(0, 2) = -ry; fdrv(0, 16) = -tdz; fdrv(0, 17) = tdy; fdrv(0, 25) = rdz; fdrv(0, 26) = -rdy;
        fdrv(1, 0) = -rz; fdrv(1, 2) = rx; fdrv(1, 15) = tdz; fdrv(1, 17) = -tdx; fdrv(1, 24) = -rdz; fdrv(1, 26) = rdx; fdrv(2, 0) = ry; fdrv(2, 1) = -rx; fdrv(2, 15) = -tdy; fdrv(2, 16) = tdx; fdrv(2, 24) = rdy; fdrv(2, 25) = -rdx; fdrv(3, 0) = t121+t126;
        fdrv(3, 1) = t12+t13-t78-t326*tz; fdrv(3, 2) = t14+t16+t77+t326*ty; fdrv(3, 3) = t302*tdy+t96*tdz; fdrv(3, 4) = t106+t346+t368*tdz; fdrv(3, 5) = t110+t323-t367*tdy; fdrv(3, 7) = t152+t6*tdz-t343*tdz; fdrv(3, 8) = t265-t6*tdy-t343*tdy;
        fdrv(3, 9) = t98*tdy+t355*tdx-tdz*(t24-tdz); fdrv(3, 10) = t20-t384*tdy-t22*ty; fdrv(3, 11) = t21-t394*tdz+t22*tz; fdrv(3, 12) = t109+t114; fdrv(3, 13) = t346-t387+t302*tdx; fdrv(3, 14) = t323+t386+t96*tdx; fdrv(3, 15) = t103*tdz-tdy*(t33-ty);
        fdrv(3, 16) = t130+t146-tdy*(t34-tx); fdrv(3, 17) = t134+t101*tdz+t135*tz; fdrv(3, 18) = t3+t4+rdx*t355; fdrv(3, 19) = t76+rdx*t98-rdy*t384; fdrv(3, 20) = -t75-rdz*t394-rdx*(t24-tdz); fdrv(3, 24) = t108+t113; fdrv(3, 25) = -t142-rdy*(t34-tx)-rdx*(t33-ty);
        fdrv(3, 26) = t141+rdx*t103+rdz*t101; fdrv(4, 0) = t12+t13+t83+t325*tz; fdrv(4, 1) = t115+t126; fdrv(4, 2) = t17+t18-t79-t325*tx; fdrv(4, 3) = t321-rx*tdy*2.0-t369*tdz; fdrv(4, 4) = t92*tdx+t305*tdz; fdrv(4, 5) = t111+t347+t367*tdx;
        fdrv(4, 6) = t263+t107*tdz+t341*tdz; fdrv(4, 8) = t153+t8*tdx+t341*tdx; fdrv(4, 9) = t20-t393*tdx+t21*tx; fdrv(4, 10) = -t353*tdy+t99*tdz-tdx*(t28-tdx); fdrv(4, 11) = t22-t383*tdz-t21*tz; fdrv(4, 12) = t321+t387+t92*tdy; fdrv(4, 13) = t105+t114;
        fdrv(4, 14) = t347-t385+t305*tdy; fdrv(4, 15) = t128-t144+t102*tdx; fdrv(4, 16) = t101*tdx-tdz*(t32-tz); fdrv(4, 17) = t136+t147-tdz*(t33-ty); fdrv(4, 18) = -t76-rdx*t393-rdy*(t28-tdx); fdrv(4, 19) = t2+t4-rdy*t353; fdrv(4, 20) = t74+rdy*t99-rdz*t383;
        fdrv(4, 24) = t142+rdx*t102+rdy*t101; fdrv(4, 25) = t104+t113; fdrv(4, 26) = -t140-rdz*(t33-ty)-rdy*(t32-tz); fdrv(5, 0) = t14+t16-t85-t324*ty; fdrv(5, 1) = t17+t18+t84+t324*tx; fdrv(5, 2) = t115+t121; fdrv(5, 3) = t345-rx*tdz*2.0+t369*tdy;
        fdrv(5, 4) = t322-ry*tdz*2.0-t368*tdx; fdrv(5, 5) = t301*tdx+t95*tdy; fdrv(5, 6) = t149+t10*tdy-t339*tdy; fdrv(5, 7) = t264+t112*tdx-t339*tdx; fdrv(5, 9) = t21-t382*tdx-t20*tx; fdrv(5, 10) = t22-t392*tdy+t20*ty;
        fdrv(5, 11) = t100*tdx+t351*tdz-tdy*(t29-tdy); fdrv(5, 12) = t345-t386+t301*tdz; fdrv(5, 13) = t322+t385+t95*tdz; fdrv(5, 14) = t105+t109; fdrv(5, 15) = -t67+t143-tdx*(t32-tz); fdrv(5, 16) = t132+t103*tdy+t129*ty; fdrv(5, 17) = t102*tdy-tdx*(t34-tx);
        fdrv(5, 18) = t75-rdx*t382+rdz*t100; fdrv(5, 19) = -t74-rdy*t392-rdz*(t29-tdy); fdrv(5, 20) = t2+t3+rdz*t351; fdrv(5, 24) = -t141-rdz*(t34-tx)-rdx*(t32-tz); fdrv(5, 25) = t140+rdy*t103+rdz*t102; fdrv(5, 26) = t104+t108;
        fdrv(6, 1) = t202-t338-t407+t350*tz; fdrv(6, 2) = t77*-2.0+t336+t405-t350*ty; fdrv(6, 3) = t377+t398; fdrv(6, 4) = t380+t419; fdrv(6, 5) = t381+t409; fdrv(6, 6) = tdx*(t48+t111); fdrv(6, 7) = t263-t344*tdy+t360*tdz; fdrv(6, 8) = t149+t360*tdy+t344*tdz;
        fdrv(6, 9) = -t366*tdx; fdrv(6, 10) = -t366*tdy+t356*tdz+t21*tx*2.0; fdrv(6, 11) = -t356*tdy-t366*tdz-t20*tx*2.0; fdrv(6, 13) = t376+t380+t410; fdrv(6, 14) = -t373+t381+t418; fdrv(6, 15) = t355*tx*-2.0; fdrv(6, 16) = t226+t390*tdz;
        fdrv(6, 17) = t295-t391*tdy; fdrv(6, 18) = rdy*t67-rdx*t355*2.0-rdz*t24*2.0; fdrv(6, 19) = -rdy*t366-rdz*t356-t5*tdz*2.0; fdrv(6, 20) = rdy*t356-rdz*t366+t5*t36; fdrv(6, 25) = t170+t189-t413; fdrv(6, 26) = t412-t5*ty*2.0-t9*ty*2.0;
        fdrv(7, 0) = t83*-2.0+t337+t406-t349*tz; fdrv(7, 2) = t203-t334-t403+t349*tx; fdrv(7, 3) = t399+t410; fdrv(7, 4) = t398+t401; fdrv(7, 5) = t400+t417; fdrv(7, 6) = t152+t342*tdx-t359*tdz; fdrv(7, 7) = -tdy*(t44+t110); fdrv(7, 8) = t264-t359*tdx-t342*tdz;
        fdrv(7, 9) = t365*tdx-t354*tdz-t22*ty*2.0; fdrv(7, 10) = t365*tdy; fdrv(7, 11) = t354*tdx+t365*tdz+t20*ty*2.0; fdrv(7, 12) = -t380+t399+t419; fdrv(7, 14) = -t374+t400+t408; fdrv(7, 15) = t286-t389*tdz; fdrv(7, 16) = t353*ty*2.0; fdrv(7, 17) = t222+t391*tdx;
        fdrv(7, 18) = rdx*t365+rdz*t354+t7*t37; fdrv(7, 19) = rdx*t28*-2.0+rdz*t68+rdy*t353*2.0; fdrv(7, 20) = -rdx*t354+rdz*t365-t7*tdx*2.0; fdrv(7, 24) = t413-t5*tz*2.0-t7*tz*2.0; fdrv(7, 26) = t163+t180-t411; fdrv(8, 0) = t210-t335-t404+t348*ty;
        fdrv(8, 1) = t84*-2.0+t333+t402-t348*tx; fdrv(8, 3) = t373+t418; fdrv(8, 4) = t374+t408; fdrv(8, 5) = t377+t401; fdrv(8, 6) = t265-t340*tdx+t358*tdy; fdrv(8, 7) = t153+t358*tdx+t340*tdy; fdrv(8, 8) = tdz*(t43+t106);
        fdrv(8, 9) = -t364*tdx+t352*tdy+t22*tz*2.0; fdrv(8, 10) = -t352*tdx-t364*tdy-t21*tz*2.0; fdrv(8, 11) = -t364*tdz; fdrv(8, 12) = t373-t381+t409; fdrv(8, 13) = t374+t378+t417; fdrv(8, 15) = t212+t389*tdy; fdrv(8, 16) = t283-t390*tdx;
        fdrv(8, 17) = t351*tz*-2.0; fdrv(8, 18) = -rdx*t364-rdy*t352-t9*tdy*2.0; fdrv(8, 19) = rdx*t352-rdy*t364+t9*t35; fdrv(8, 20) = rdx*t72-rdy*t29*2.0-rdz*t351*2.0; fdrv(8, 24) = t160+t192-t412; fdrv(8, 25) = t411-t7*tx*2.0-t9*tx*2.0;
        fdrv(9, 0) = t11*t39+t11*t40+t299*t326+t404*ty+t406*tz; fdrv(9, 1) = t12*t40+t116*t298+t137*t326-t404*tx+t372*tz; fdrv(9, 2) = t12*t34+t119*t297+t138*t326-t406*tx-t372*ty;
        fdrv(9, 3) = t331+t332-t11*t27*4.0-t11*t31*4.0-t15*t31*2.0-t18*t30*2.0+t39*t42+t40*t42; fdrv(9, 4) = t236-t238-t11*t24*2.0+t15*t26*6.0-t13*t31*2.0-t16*t30*2.0+t18*t29*4.0+t29*t61;
        fdrv(9, 5) = t237-t243-t11*t25*2.0-t13*t28*2.0-t16*t27*2.0+t15*t29*4.0+t19*t29*6.0+t26*t62; fdrv(9, 6) = t6*(t27+t31)*-4.0-t447*tdy-t449*tdz; fdrv(9, 7) = -t447*tdx+tdz*(t187+t188-t16*tdy*2.0-t13*tdz*2.0)+tdy*(t155+t196+t267+t15*tdx*1.2E+1);
        fdrv(9, 8) = -t449*tdx+tdy*(t187+t188-t16*tdy*2.0-t13*tdz*2.0)+tdz*(t155+t174+t272+t19*tdx*1.2E+1); fdrv(9, 9) = tdz*(t219-t256)+tdx*(t145*2.0+t147*2.0+t284+t289)+tdy*(t213+t128*tx);
        fdrv(9, 10) = -tdx*(t228+t258)-tdz*(t216+t291)+tdy*(t145*6.0+t147*2.0-t222+t284); fdrv(9, 11) = -tdx*(t226+t262)-tdz*(t147*-6.0+t214+t223+t130*ty)+tdy*(t225+t128*tz); fdrv(9, 12) = t42*t299+tdz*(t204+t208+t253+t62*ty)+tdy*(t201+t211+t251+t61*tz);
        fdrv(9, 13) = t450; fdrv(9, 14) = t451; fdrv(9, 15) = tx*(t87+t89+t129*ty+t133*tz)*-2.0; fdrv(9, 16) = ty*(t87+t89+t129*ty+t133*tz)*-2.0; fdrv(9, 17) = tz*(t87+t89+t129*ty+t133*tz)*-2.0;
        fdrv(9, 18) = t328+t329-t5*t27*4.0-t5*t31*4.0-t9*t27*2.0-t7*t31*2.0+t2*t39*2.0+t2*t40*2.0; fdrv(9, 19) = t230-t231-rdy*t89*2.0+rdy*t260-t5*t24*2.0+t7*t26*6.0-t9*t24*2.0+t9*t26*4.0;
        fdrv(9, 20) = t232-t233+rdz*t257+rdz*t283-t5*t25*2.0-t7*t25*2.0+t7*t29*4.0+t9*t29*6.0; fdrv(9, 24) = t299*t357*2.0; fdrv(9, 25) = t414; fdrv(9, 26) = t415; fdrv(10, 0) = t13*t40+t117*t299+t137*t325-t402*ty+t371*tz;
        fdrv(10, 1) = t13*t32+t15*t40+t298*t325+t402*tx+t407*tz; fdrv(10, 2) = t13*t33+t123*t297+t139*t325-t371*tx-t407*ty; fdrv(10, 3) = -t236+t238+t11*t24*6.0-t15*t26*2.0-t12*t31*2.0+t16*t30*4.0-t18*t29*2.0+t30*t57;
        fdrv(10, 4) = t330+t332-t13*t26*4.0-t11*t31*2.0-t16*t29*2.0-t15*t31*4.0+t38*t47+t40*t47; fdrv(10, 5) = t242-t244-t11*t28*2.0+t11*t30*4.0-t16*t26*2.0-t15*t28*2.0+t19*t30*6.0+t24*t59;
        fdrv(10, 6) = -t445*tdy-t442*tdz+tdx*(t178+t198+t266+t11*tdy*1.2E+1); fdrv(10, 7) = t8*(t23+t31)*-4.0-t445*tdx-t448*tdz; fdrv(10, 8) = -t442*tdx-t448*tdy+tdz*(t157+t178+t277+t19*tdy*1.2E+1);
        fdrv(10, 9) = -tdy*(t222+t254)-tdz*(t216-t221)-tdx*(t91*2.0-t143*6.0+t213+t136*tz); fdrv(10, 10) = -tdx*(t145*2.0+t284)-tdy*(t213+t229+t128*tx+t136*tz)+tdz*(t227+t132*ty);
        fdrv(10, 11) = tdx*(t221+t290)-tdy*(t218+t262)-tdz*(t86*2.0-t148*6.0+t229+t128*tx); fdrv(10, 12) = t450; fdrv(10, 13) = t47*t298+tdz*(t204+t208+t253+t59*tx)+tdx*(t205+t209+t249+t57*tz); fdrv(10, 14) = t452;
        fdrv(10, 15) = tx*(t86+t91+t127*tx+t135*tz)*-2.0; fdrv(10, 16) = ty*(t86+t91+t127*tx+t135*tz)*-2.0; fdrv(10, 17) = tz*(t86+t91+t127*tx+t135*tz)*-2.0; fdrv(10, 18) = -t230+t231+rdx*t261+rdx*t295+t5*t24*6.0-t7*t26*2.0+t9*t24*4.0-t9*t26*2.0;
        fdrv(10, 19) = t327+t329-t7*t23*4.0-t9*t23*2.0-t5*t31*2.0-t7*t31*4.0+t3*t38*2.0+t3*t40*2.0; fdrv(10, 20) = t234-t235-rdz*t86*2.0+rdz*t255-t5*t28*2.0+t5*t30*4.0-t7*t28*2.0+t9*t30*6.0; fdrv(10, 24) = t414; fdrv(10, 25) = t298*t357*2.0;
        fdrv(10, 26) = t416; fdrv(11, 0) = t16*t39+t120*t299+t138*t324+t370*ty-t403*tz; fdrv(11, 1) = t16*t32+t124*t298+t139*t324-t370*tx-t405*tz; fdrv(11, 2) = t16*t33+t18*t34+t297*t324+t403*tx+t405*ty;
        fdrv(11, 3) = -t237+t243+t11*t25*6.0+t13*t28*4.0-t12*t30*2.0-t15*t29*2.0-t19*t29*2.0+t28*t55; fdrv(11, 4) = -t242+t244+t11*t28*4.0-t11*t30*2.0-t13*t29*2.0+t15*t28*6.0-t19*t30*2.0+t25*t56;
        fdrv(11, 5) = t330+t331-t11*t27*2.0-t13*t26*2.0-t16*t29*4.0-t18*t30*4.0+t38*t52+t39*t52; fdrv(11, 6) = -t444*tdz+tdy*(t168+t169-t175-t185)+tdx*(t183+t199+t268+t11*tdz*1.2E+1);
        fdrv(11, 7) = -t446*tdz+tdx*(t168+t169-t175-t185)+tdy*(t159+t199+t276+t15*tdz*1.2E+1); fdrv(11, 8) = t10*(t23+t27)*-4.0-t444*tdx-t446*tdy; fdrv(11, 9) = -tdz*(t214+t254)-tdx*(t90*2.0-t144*6.0+t219+t132*ty)+tdy*(t217-t24*tz*2.0);
        fdrv(11, 10) = tdx*(t217+t290)-tdz*(t212+t258)-tdy*(t88*2.0-t146*6.0+t227-t256); fdrv(11, 11) = -tdz*(t219+t227-t256+t132*ty)-tdx*(t147*2.0+t289)-tdy*(t148*2.0+t296); fdrv(11, 12) = t451; fdrv(11, 13) = t452;
        fdrv(11, 14) = t52*t297+tdy*(t201+t211+t251+t56*tx)+tdx*(t205+t209+t249+t55*ty); fdrv(11, 15) = tx*(t88+t90-t144+t131*ty)*-2.0; fdrv(11, 16) = ty*(t88+t90-t144+t131*ty)*-2.0; fdrv(11, 17) = tz*(t88+t90-t144+t131*ty)*-2.0;
        fdrv(11, 18) = -t232+t233-rdx*t90*2.0+rdx*t259+t5*t25*6.0+t7*t25*4.0-t7*t29*2.0-t9*t29*2.0; fdrv(11, 19) = -t234+t235+rdy*t256+rdy*t286+t5*t28*4.0-t5*t30*2.0+t7*t28*6.0-t9*t30*2.0;
        fdrv(11, 20) = t327+t328-t7*t23*2.0-t5*t27*2.0-t9*t23*4.0-t9*t27*4.0+t4*t38*2.0+t4*t39*2.0; fdrv(11, 24) = t415; fdrv(11, 25) = t416; fdrv(11, 26) = t297*t357*2.0;

        /* #endregion Calculating f and its jacobian ------------------------------------------------------------*/
    }

    /* #endregion Calculating the derivatives of f --------------------------------------------------------------*/

    ResetQSC();

    // Calculating the component jacobians
    for (int idx = 0; idx < COMPONENTS; idx++)
    {
        T &g = gdrv(idx, 0);
        T &dg = gdrv(idx, 1);
        T &ddg = gdrv(idx, 2);

        Matrix<T, 1, 3> Jdg_U = dg*Ubtp;
        Matrix<T, 1, 3> Jddg_U = ddg*Ubtp;

        auto fXi = fdrv.block(idx*3, 0, 3, 3);

        auto dfXi_W_dU = fdrv.block(idx*3, S1_IDX, 3, 3);
        auto ddfXiW_X_dUdU = fdrv.block(idx*3, C11_IDX, 3, 3);
        auto ddfXiW_X_dUdV = fdrv.block(idx*3, C12_IDX, 3, 3);
        auto ddfXiW_X_dUdW = fdrv.block(idx*3, C13_IDX, 3, 3);

        auto dfXi_W_dV = fdrv.block(idx*3, S2_IDX, 3, 3);
        auto ddfXiW_X_dVdU = fdrv.block(idx*3, C21_IDX, 3, 3);
        auto ddfXiW_X_dVdV = fdrv.block(idx*3, C22_IDX, 3, 3);
        auto ddfXiW_X_dVdW = fdrv.block(idx*3, C23_IDX, 3, 3);

        Q += fXi*g;

        // Find the S1 C1 jacobians
        {
            const Vec3T &W = Thed;
            const Vec3T &X = Thed; Matrix<T, 1, 3> Xtp = X.transpose();
            Vec3T fXiW = fXi*W;
            T UbtpX = Ubtp.dot(X);

            S1  += fXiW*Jdg_U + dfXi_W_dU*g;
            C11 += dfXi_W_dU*(Jdg_U*X) + fXiW*Jddg_U*UbtpX + fXiW*Xtp*dg*JUb + ddfXiW_X_dUdU*g + dfXi_W_dU*X*Jdg_U;
            C12 += dfXi_W_dV*(Jdg_U*X) + ddfXiW_X_dUdV*g;
            C13 += fXi*(Jdg_U*X) + ddfXiW_X_dUdW*g;
        }

        // Find the S2 C2 jacobians
        {
            const Vec3T &W = Thed;
            const Vec3T &X = Rhod; Matrix<T, 1, 3> Xtp = X.transpose();
            Vec3T fXiW = fXi*W;
            T UbtpX = Ubtp.dot(X);

            S2  += dfXi_W_dV*g;
            C21 += ddfXiW_X_dVdU*g + dfXi_W_dV*X*Jdg_U;
            C22 += ddfXiW_X_dVdV*g;
            C23 += ddfXiW_X_dVdW*g;
        }

        // cout << "ComputeQSC " << idx << ": g\n" << g << endl;
        // cout << "ComputeQSC " << idx << ": dfXi_W_dU\n" << dfXi_W_dU << endl;
        // cout << "ComputeQSC " << idx << ": dfXi_W_dV\n" << dfXi_W_dV << endl;
        // cout << "ComputeQSC " << idx << ": S1\n" << S1 << endl;
        // cout << "ComputeQSC " << idx << ": S2\n" << S2 << endl;
    }
}

template <typename T>
void SE3Qp<T>::ComputeS(const Vec3T &The, const Vec3T &Rho, const Vec3T &Omg)
{
    T Un = The.norm();
    Matrix<T, 3, 1> Ub = The / Un;
    Matrix<T, 1, 3> Ubtp = Ub.transpose();
    Matrix<T, 3, 3> JUb = (Mat3T::Identity(3, 3) - Ub*Ubtp) / Un;

    // This Q has 4 g and each has 3 derivatives 0th, 1st
    Matrix<T, COMPONENTS, 2> gdrv; gdrv.setZero();

    // This Q has 4 fs and each f has 3 derivatives (f, S1, S2) (4x3 x 3x3)
    Matrix<T, 3*COMPONENTS, 9> fdrv; fdrv.setZero();

    /* #region Calculating the derivatives of g -----------------------------------------------------------------*/

    {
        T t2 = cos(Un), t3 = sin(Un), t4 = Un*2.0, t5 = Un*3.0, t6 = Un*Un, t7 = Un*Un*Un, t20 = Un*8.0, t21 = 1.0/Un, t8 = cos(t4), t9 = t2*2.0, t10 = cos(t5), t11 = t2*4.0, t12 = t2*t2, t13 = t2*t2*t2, t14 = sin(t4), t15 = t3*2.0, t16 = sin(t5);
        T t17 = t3*3.0, t18 = Un*t2, t19 = Un*t3, t22 = 1.0/t6, t23 = 1.0/t7, t25 = t21*t21*t21*t21*t21, t27 = t21*t21*t21*t21*t21*t21*t21, t28 = Un*t4, t29 = t2*1.0E+1, t30 = t2*4.2E+1, t31 = t2-1.0, t32 = -t3, t35 = t3*1.5E+1, t36 = 1.0/t3;
        T t42 = t6*1.0E+1, t43 = t2/2.0, t48 = t2*t6, t49 = t3*t6, t50 = t3*t7, t24 = t22*t22, t26 = t22*t22*t22, t33 = -t15, t34 = -t17, t37 = t36*t36, t38 = t18*7.0, t40 = t23*t23*t23, t41 = t22*t22*t22*t22*t22, t44 = -t29, t45 = -t30, t46 = -t35;
        T t47 = t19*1.2E+1, t51 = t3*t18*6.0, t52 = t10*4.2E+1, t53 = t12*1.0E+1, t54 = t13*1.0E+1, t56 = t19*1.32E+2, t57 = Un+t32, t58 = t22/2.0, t59 = t22/4.0, t60 = t6*t13, t61 = t7*t16, t62 = t8*8.4E+1, t64 = 1.0/t31, t66 = t6*t29;
        T t67 = Un*t14*1.2E+1, t68 = Un*t16*3.6E+1, t69 = t50*1.1E+1, t72 = t4*t6*t14, t74 = t12*t19*6.0, t77 = t6+t9-2.0, t78 = t6*t12*-2.0, t81 = t43+1.0/2.0, t83 = t6*t8*-1.0E+1, t84 = t6*t10*-1.0E+1, t87 = (t23*t31)/2.0, t91 = t6+t11+t19-4.0;
        T t39 = t24*t24, t55 = -t47, t63 = -t53, t65 = -t56, t71 = t4+t33, t73 = -t62, t79 = -t60, t80 = -t61, t82 = t14+t33, t85 = Un+t18+t33, t88 = t4+t18+t34, t90 = -t87, t92 = (t23*t57)/2.0, t93 = t24*t57*(3.0/2.0), t95 = (t24*t77)/4.0;
        T t96 = t81/t19, t98 = t20+t38+t46+t49, t100 = (t25*t91)/2.0, t108 = (t23*t64*t91)/4.0, t111 = (t23*t64*t91)/8.0, t124 = (t27*t64*t77*t91)/4.0, t126 = (t27*t64*t77*t91)/8.0, t86 = 1.0/t82, t89 = t85*t85, t94 = -t93, t97 = (t25*t88)/4.0;
        T t99 = -t96, t101 = t96/2.0, t102 = t96/4.0, t103 = -t100, t109 = (t26*t98)/4.0, t110 = -t108, t114 = ((t21*t21*t21*t21*t21*t21)*t77*t85)/t19, t115 = -t111, t118 = (t26*t36*t71*t85)/4.0, t121 = (t26*t36*t71*t85)/8.0, t125 = -t124;
        T t127 = -t126, t133 = t28+t44+t48+t50+t51+t54+t55+t63+t74+t78+t79+1.0E+1, t140 = t42+t45+t52+t65+t66+t67+t68+t69+t72+t73+t80+t83+t84+8.4E+1, t104 = -t101, t105 = -t102, t106 = t22+t99, t112 = t90+t94, t113 = -t109, t119 = t114/2.0;
        T t120 = -t118, t122 = -t121, t134 = t26*t86*t133, t141 = (t39*t86*t140)/8.0, t142 = (t39*t86*t140)/1.6E+1, t107 = t106*t106, t116 = t58+t104, t117 = t59+t105, t123 = t23*t57*t106, t128 = t92*t106, t129 = (t24*t77*t106)/2.0, t130 = t95*t106;
        T t131 = (t25*t88*t106)/2.0, t132 = t97*t106, t135 = -t134, t136 = t134/2.0, t138 = t114+t120+t125, t139 = t119+t122+t127, t143 = -t141, t144 = -t142, t137 = -t136;

        gdrv(0, 0) = T(1.0/2.0); gdrv(1, 0) = t23*t57; gdrv(1, 1) = -t23*t31-t24*t57*3.0;
        gdrv(2, 0) = (t24*t77)/2.0; gdrv(2, 1) = -t25*t91; gdrv(3, 0) = (t25*t88)/2.0; gdrv(3, 1) = t26*t98*(-1.0/2.0); gdrv(4, 0) = T(1.0/4.0); gdrv(5, 0) = t92; gdrv(5, 1) = t112; gdrv(6, 0) = t95; gdrv(6, 1) = t103; gdrv(7, 0) = t97; gdrv(7, 1) = t113; gdrv(8, 0) = t116;
        gdrv(8, 1) = t110; gdrv(9, 0) = t123; gdrv(9, 1) = t135; gdrv(10, 0) = t129; gdrv(10, 1) = t138; gdrv(11, 0) = t131; gdrv(11, 1) = t143; gdrv(12, 0) = T(1.0/4.0); gdrv(13, 0) = t92; gdrv(13, 1) = t112; gdrv(14, 0) = t95; gdrv(14, 1) = t103; gdrv(15, 0) = t97;
        gdrv(15, 1) = t113; gdrv(16, 0) = T(1.0/8.0); gdrv(17, 0) = (t23*t57)/4.0; gdrv(17, 1) = t23*t31*(-1.0/4.0)-t24*t57*(3.0/4.0); gdrv(18, 0) = (t24*t77)/8.0; gdrv(18, 1) = t25*t91*(-1.0/4.0); gdrv(19, 0) = (t25*t88)/8.0; gdrv(19, 1) = t26*t98*(-1.0/8.0);
        gdrv(20, 0) = t117; gdrv(20, 1) = t115; gdrv(21, 0) = t128; gdrv(21, 1) = t137; gdrv(22, 0) = t130; gdrv(22, 1) = t139; gdrv(23, 0) = t132; gdrv(23, 1) = t144; gdrv(24, 0) = t116; gdrv(24, 1) = t110; gdrv(25, 0) = t123; gdrv(25, 1) = t135; gdrv(26, 0) = t129;
        gdrv(26, 1) = t138; gdrv(27, 0) = t131; gdrv(27, 1) = t143; gdrv(28, 0) = t117; gdrv(28, 1) = t115; gdrv(29, 0) = t128; gdrv(29, 1) = t137; gdrv(30, 0) = t130; gdrv(30, 1) = t139; gdrv(31, 0) = t132; gdrv(31, 1) = t144; gdrv(32, 0) = t106*t116;
        gdrv(32, 1) = (t25*(t64*t64)*(t2*8.0+t6-t12*8.0+t19*6.0+t31*8.0+t6*t12-t5*t14-t4*t18+t7*t32))/4.0; gdrv(33, 0) = t23*t57*t107; gdrv(33, 1) = t37*t39*t57*t89*(-3.0/4.0)-(1.0/(t19*t19)*t25*t31*t89)/4.0+(t36*t39*t57*t64*t85*t91)/2.0;
        gdrv(34, 0) = (t24*t77*t107)/2.0; gdrv(34, 1) = (t37*t39*t71*t89)/8.0-(t37*t40*t77*t89)/2.0+(t36*t40*t64*t77*t85*t91)/4.0; gdrv(35, 0) = (t25*t88*t107)/2.0;
        gdrv(35, 1) = t37*t40*t89*(t9+t19-2.0)*(-1.0/8.0)-t37*t41*t88*t89*(5.0/8.0)+(t36*t41*t64*t85*t88*t91)/4.0;
    }

    /* #endregion Calculating the derivatives of g --------------------------------------------------------------*/

    /* #region Calculating the derivatives of f -----------------------------------------------------------------*/

    T tx = The(0), ty = The(1), tz = The(2);
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);

    {
        /* #region Repeated terms -------------------------------------------------------------------------------*/

        T t2 = ox*rx, t3 = ox*ry, t4 = oy*rx, t5 = ox*rz, t6 = oy*ry, t7 = oz*rx, t8 = oy*rz, t9 = oz*ry, t10 = oz*rz, t11 = ox*tx, t12 = ox*ty, t13 = oy*tx, t14 = ox*tz, t15 = oy*ty, t16 = oz*tx, t17 = oy*tz, t18 = oz*ty, t19 = oz*tz;
        T t20 = rx*tx, t21 = rx*ty, t22 = ry*tx, t23 = rx*tz, t24 = ry*ty, t25 = rz*tx, t26 = ry*tz, t27 = rz*ty, t28 = rz*tz, t29 = tx*ty, t30 = tx*tz, t31 = ty*tz, t32 = tx*tx, t33 = tx*tx*tx, t35 = tx*tx*tx*tx*tx, t36 = ty*ty, t37 = ty*ty*ty;
        T t39 = ty*ty*ty*ty*ty, t40 = tz*tz, t41 = tz*tz*tz, t43 = tz*tz*tz*tz*tz, t34 = t32*t32, t38 = t36*t36, t42 = t40*t40, t44 = t12*2.0, t45 = t13*2.0, t46 = t14*2.0, t47 = t16*2.0, t48 = t17*2.0, t49 = t18*2.0, t50 = t20*2.0, t51 = t20*3.0;
        T t52 = t21*2.0, t53 = t22*2.0, t54 = t23*2.0, t55 = t24*2.0, t56 = t25*2.0, t57 = t24*3.0, t58 = t26*2.0, t59 = t27*2.0, t60 = t28*2.0, t61 = t28*3.0, t62 = t29*2.0, t63 = t30*2.0, t64 = t31*2.0, t65 = t2*tx, t66 = t2*ty, t67 = t3*tx;
        T t68 = t2*tz, t69 = t5*tx, t70 = t4*ty, t71 = t6*tx, t72 = t6*ty, t73 = t6*tz, t74 = t8*ty, t75 = t7*tz, t76 = t10*tx, t77 = t9*tz, t78 = t10*ty, t79 = t10*tz, t80 = t11*ty, t81 = t11*tz, t82 = t13*ty, t83 = t15*tz, t84 = t16*tz;
        T t85 = t18*tz, t86 = t20*ty, t87 = t20*tz, t88 = t22*ty, t89 = t21*tz, t90 = t22*tz, t91 = t25*ty, t92 = t24*tz, t93 = t25*tz, t94 = t27*tz, t95 = rx+t26, t96 = rx+t27, t97 = ry+t23, t98 = ry+t25, t99 = rz+t21, t100 = rz+t22, t101 = t31+tx;
        T t102 = t30+ty, t103 = t29+tz, t104 = t3*t29, t105 = t4*t29, t106 = t3*t30, t107 = t5*t29, t108 = t4*t30, t109 = t7*t29, t110 = t3*t31, t111 = t5*t30, t112 = t4*t31, t113 = t8*t29, t114 = t7*t30, t115 = t9*t29, t116 = t5*t31, t117 = t8*t30;
        T t118 = t7*t31, t119 = t9*t30, t120 = t8*t31, t121 = t9*t31, t122 = t11*t31, t123 = t13*t31, t124 = t16*t31, t125 = t32*2.0, t126 = t36*2.0, t127 = t40*2.0, t128 = -t3, t129 = -t4, t130 = -t5, t131 = -t7, t132 = -t8, t133 = -t9, t134 = -t12;
        T t135 = -t13, t137 = -t14, t138 = -t16, t140 = -t17, t141 = -t18, t143 = -t20, t144 = -t21, t145 = -t22, t147 = -t23, t148 = -t24, t149 = -t25, t151 = -t26, t152 = -t27, t154 = -t28, t155 = -t29, t156 = -t30, t157 = -t31, t158 = t11*tx;
        T t159 = t12*ty, t160 = t13*tx, t162 = t13*t32, t163 = t14*tz, t164 = t15*ty, t165 = t16*tx, t166 = t14*t40, t168 = t17*tz, t169 = t18*ty, t171 = t18*t36, t172 = t19*tz, t173 = t21*ty, t174 = t22*tx, t175 = t23*tz, t176 = t25*tx;
        T t177 = t26*tz, t178 = t27*ty, t179 = t29*ty, t180 = t29*tx, t181 = t29*t36, t182 = t29*t32, t183 = t30*tz, t184 = t30*tx, t185 = t30*t40, t186 = t30*t32, t187 = t31*tz, t188 = t31*ty, t189 = t31*t40, t190 = t31*t36, t193 = t4*tx*4.0;
        T t196 = t3*ty*4.0, t197 = t7*tx*4.0, t201 = t5*tz*4.0, t202 = t9*ty*4.0, t204 = t8*tz*4.0, t225 = t2*t32, t226 = t2*t33, t228 = t2*t35, t229 = t2*t36, t230 = t3*t32, t231 = t4*t32, t232 = t2*t37, t233 = t3*t33, t234 = t4*t33, t237 = t2*t40;
        T t238 = t3*t36, t239 = t5*t32, t240 = t4*t36, t241 = t6*t32, t242 = t7*t32, t243 = t2*t41, t244 = t3*t37, t245 = t5*t33, t246 = t4*t37, t247 = t6*t33, t248 = t7*t33, t253 = t3*t40, t254 = t5*t36, t255 = t4*t40, t256 = t6*t36, t257 = t8*t32;
        T t258 = t7*t36, t259 = t9*t32, t260 = t3*t41, t261 = t5*t37, t262 = t4*t41, t263 = t6*t37, t264 = t8*t33, t265 = t7*t37, t266 = t9*t33, t274 = t6*t39, t275 = t5*t40, t276 = t6*t40, t277 = t8*t36, t278 = t7*t40, t279 = t9*t36, t280 = t10*t32;
        T t281 = t5*t41, t282 = t6*t41, t283 = t8*t37, t284 = t7*t41, t285 = t9*t37, t286 = t10*t33, t291 = t8*t40, t292 = t9*t40, t293 = t10*t36, t294 = t8*t41, t295 = t9*t41, t296 = t10*t37, t299 = t10*t40, t300 = t10*t41, t302 = t10*t43;
        T t304 = t11*t30, t305 = t13*t36, t307 = t12*t31, t308 = t13*t40, t311 = t16*t29, t314 = t18*t40, t315 = t20*t36, t316 = t20*t40, t317 = t22*t29, t318 = t21*t40, t319 = t21*t31, t320 = t22*t40, t321 = t22*t30, t322 = t25*t36, t323 = t25*t29;
        T t324 = t24*t40, t325 = t25*t30, t326 = t27*t31, t327 = t29*t40, t328 = t29*t31, t329 = t29*t30, t331 = t2*t29*4.0, t335 = t2*t30*4.0, t345 = t6*t29*4.0, t365 = t6*t31*4.0, t372 = t10*t30*4.0, t374 = t10*t31*4.0, t378 = -t32, t379 = -t36;
        T t380 = -t40, t388 = t20*tx*5.0, t390 = t24*ty*5.0, t392 = t28*tz*5.0, t438 = t29*tz*-2.0, t439 = t32+t36, t440 = t32+t40, t441 = t36+t40, t556 = t2*t29*-2.0, t558 = t2*t30*-2.0, t566 = t2*t31*-2.0, t568 = t6*t29*-2.0, t579 = t6*t30*-2.0;
        T t589 = t6*t31*-2.0, t591 = t10*t29*-2.0, t597 = t10*t30*-2.0, t599 = t10*t31*-2.0, t709 = t20+t24, t710 = t20+t28, t711 = t24+t28, t1408 = t2*(t29*t29), t1409 = t2*(t30*t30), t1413 = t6*(t29*t29), t1420 = t6*(t31*t31), t1424 = t10*(t30*t30);
        T t1425 = t10*(t31*t31), t1435 = -tx*(t31-tx), t1436 = -ty*(t30-ty), t1437 = -tz*(t29-tz), t1593 = t2*(t31*t31)*4.0, t1605 = t6*(t30*t30)*4.0, t1617 = t10*(t29*t29)*4.0, t1848 = t29*(t29-tz), t1849 = t30*(t30-ty), t1850 = t31*(t31-tx);
        T t136 = -t45, t139 = -t47, t142 = -t49, t146 = -t53, t150 = -t56, t153 = -t59, t191 = t65*2.0, t192 = t67*2.0, t194 = t69*2.0, t195 = t70*2.0, t198 = t72*2.0, t199 = t74*2.0, t200 = t75*2.0, t203 = t77*2.0, t205 = t79*2.0, t206 = t80*2.0;
        T t207 = t81*2.0, t208 = t45*ty, t209 = t83*2.0, t210 = t47*tz, t211 = t49*tz, t212 = t50*ty, t213 = t86*4.0, t214 = t50*tz, t215 = t53*ty, t216 = t87*4.0, t217 = t88*4.0, t218 = t55*tz, t219 = t56*tz, t220 = t92*4.0, t221 = t93*4.0;
        T t222 = t59*tz, t223 = t94*4.0, t224 = t62*tz, t227 = t2*t34, t235 = t2*t38, t236 = t4*t34, t249 = t2*t42, t250 = t3*t38, t251 = t6*t34, t252 = t7*t34, t267 = t3*t42, t268 = t5*t38, t269 = t4*t42, t270 = t6*t38, t271 = t8*t34, t272 = t7*t38;
        T t273 = t9*t34, t287 = t5*t42, t288 = t6*t42, t289 = t9*t38, t290 = t10*t34, t297 = t8*t42, t298 = t10*t38, t301 = t10*t42, t330 = t2*t62, t332 = t2*t63, t333 = t3*t62, t334 = t4*t62, t336 = t104*4.0, t337 = t105*4.0, t338 = t2*t64;
        T t339 = t3*t63, t340 = t5*t62, t341 = t4*t63, t342 = t6*t62, t343 = t7*t62, t344 = t108*4.0, t346 = t109*4.0, t347 = t3*t64, t348 = t5*t63, t349 = t4*t64, t350 = t6*t63, t351 = t8*t62, t352 = t7*t63, t353 = t9*t62, t354 = t110*4.0;
        T t355 = t111*4.0, t356 = t114*4.0, t357 = t115*4.0, t358 = t5*t64, t359 = t6*t64, t360 = t8*t63, t361 = t7*t64, t362 = t9*t63, t363 = t10*t62, t364 = t116*4.0, t366 = t117*4.0, t367 = t8*t64, t368 = t9*t64, t369 = t10*t63, t370 = t120*4.0;
        T t371 = t121*4.0, t373 = t10*t64, t375 = t11*t64, t376 = t31*t45, t377 = t31*t47, t381 = t44*ty, t383 = t46*tz, t387 = t51*tx, t389 = t57*ty, t391 = t61*tz, t393 = t62*ty, t394 = t62*tx, t395 = t63*tz, t396 = t63*tx, t397 = t64*tz;
        T t398 = t64*ty, t399 = -t67, t400 = -t68, t401 = -t69, t402 = -t70, t403 = -t71, t404 = -t196, t405 = -t197, t406 = -t74, t407 = -t75, t408 = -t77, t409 = -t78, t410 = -t204, t411 = -t80, t413 = -t81, t414 = -t82, t417 = -t83, t418 = -t84;
        T t420 = t84*-2.0, t421 = -t85, t423 = -t86, t424 = t86*-2.0, t425 = -t87, t426 = -t88, t427 = t87*-2.0, t428 = t88*-2.0, t429 = -t89, t430 = -t90, t431 = -t91, t432 = -t92, t433 = -t93, t434 = t92*-2.0, t435 = t93*-2.0, t436 = -t94;
        T t437 = t94*-2.0, t442 = t2*t125, t443 = t226*2.0, t446 = t4*t125, t447 = t229*3.0, t448 = t232*2.0, t449 = t230*3.0, t450 = t231*3.0, t451 = t234*2.0, t454 = t234*4.0, t455 = t231*6.0, t456 = t234*6.0, t459 = t3*t126, t461 = t7*t125;
        T t462 = t237*3.0, t463 = t243*2.0, t464 = t238*3.0, t465 = t244*2.0, t466 = t239*3.0, t467 = t240*3.0, t468 = t241*3.0, t469 = t247*2.0, t470 = t242*3.0, t471 = t248*2.0, t476 = t244*4.0, t477 = t248*4.0, t478 = t238*6.0, t479 = t242*6.0;
        T t480 = t244*6.0, t482 = t248*6.0, t487 = t6*t126, t491 = t260*2.0, t492 = t261*2.0, t493 = t262*2.0, t494 = t263*2.0, t495 = t264*2.0, t496 = t265*2.0, t497 = t266*2.0, t505 = t5*t127, t507 = t9*t126, t509 = t275*3.0, t510 = t281*2.0;
        T t511 = t276*3.0, t512 = t282*2.0, t513 = t277*3.0, t514 = t278*3.0, t515 = t279*3.0, t516 = t285*2.0, t517 = t280*3.0, t518 = t286*2.0, t523 = t281*4.0, t524 = t285*4.0, t525 = t275*6.0, t526 = t279*6.0, t527 = t281*6.0, t529 = t285*6.0;
        T t531 = t8*t127, t533 = t291*3.0, t534 = t294*2.0, t535 = t292*3.0, t536 = t293*3.0, t537 = t296*2.0, t540 = t294*4.0, t541 = t291*6.0, t542 = t294*6.0, t544 = t10*t127, t545 = t300*2.0, t547 = rx+t151, t548 = rx+t152, t549 = ry+t147;
        T t550 = ry+t149, t551 = rz+t144, t552 = rz+t145, t557 = -t331, t559 = t104*-2.0, t560 = t105*-2.0, t561 = -t335, t562 = -t106, t563 = -t107, t564 = -t108, t565 = -t109, t567 = t108*-2.0, t569 = t109*-2.0, t571 = -t345, t573 = -t110;
        T t574 = -t112, t575 = -t113, t576 = -t115, t577 = t110*-2.0, t578 = t111*-2.0, t580 = t114*-2.0, t581 = t115*-2.0, t584 = -t116, t585 = -t117, t586 = -t118, t587 = -t119, t588 = t116*-2.0, t590 = t117*-2.0, t593 = -t365, t595 = t120*-2.0;
        T t596 = t121*-2.0, t598 = -t372, t600 = -t374, t601 = -t122, t602 = -t123, t603 = -t124, t604 = t29*t66, t605 = t30*t68, t606 = t104*ty, t607 = t29*t67, t608 = t29*t70, t609 = t105*tx, t610 = t36*t104, t611 = t32*t104, t612 = t36*t105;
        T t613 = t32*t105, t614 = t31*t68, t615 = t31*t66, t616 = t30*t67, t617 = t29*t69, t618 = t108*tx, t619 = t29*t71, t620 = t109*tx, t621 = t40*t106, t622 = t32*t106, t623 = t36*t107, t624 = t32*t107, t625 = t40*t108, t626 = t32*t108;
        T t627 = t36*t109, t628 = t32*t109, t629 = t110*ty, t630 = t111*tz, t631 = t30*t69, t632 = t31*t70, t633 = t30*t73, t634 = t30*t71, t635 = t29*t74, t636 = t30*t75, t637 = t114*tx, t638 = t115*ty, t639 = t40*t110, t640 = t36*t110;
        T t641 = t40*t111, t642 = t32*t111, t643 = t40*t112, t644 = t36*t112, t645 = t36*t113, t646 = t32*t113, t647 = t40*t114, t648 = t32*t114, t649 = t36*t115, t650 = t32*t115, t651 = t116*tz, t652 = t31*t73, t653 = t117*tz, t654 = t31*t75;
        T t655 = t30*t77, t656 = t29*t78, t657 = t29*t76, t658 = t40*t116, t659 = t36*t116, t660 = t40*t117, t661 = t32*t117, t662 = t40*t118, t663 = t36*t118, t664 = t40*t119, t665 = t32*t119, t666 = t120*tz, t667 = t31*t74, t668 = t31*t77;
        T t669 = t121*ty, t670 = t30*t76, t671 = t40*t120, t672 = t36*t120, t673 = t40*t121, t674 = t36*t121, t675 = t31*t78, t676 = t62*t68, t677 = t29*t68*4.0, t683 = t62*t73, t684 = t62*t75, t685 = t107*tz*4.0, t686 = t29*t73*4.0;
        T t687 = t29*t75*4.0, t689 = t62*t77, t690 = t113*tz*4.0, t691 = t29*t77*4.0, t692 = t62*t79, t693 = t29*t79*4.0, t694 = t134*ty, t695 = t135*tx, t697 = t137*tz, t698 = t138*tx, t700 = t140*tz, t701 = t141*ty, t703 = t179*-2.0;
        T t704 = t180*-2.0, t705 = t183*-2.0, t706 = t184*-2.0, t707 = t187*-2.0, t708 = t188*-2.0, t712 = t101*tx, t713 = t102*ty, t714 = t103*tz, t715 = -t225, t716 = -t226, t720 = -t229, t721 = t32*t128, t722 = t32*t129, t723 = t229*-2.0;
        T t724 = -t232, t725 = t33*t129, t728 = t34*t129, t733 = t234*8.0, t735 = -t237, t736 = t36*t128, t737 = t32*t130, t738 = t36*t129, t739 = -t241, t740 = t32*t131, t741 = t237*-2.0, t742 = -t243, t743 = t37*t128, t744 = t241*-2.0, t745 = -t247;
        T t746 = t33*t131, t751 = t38*t128, t756 = t34*t131, t765 = t244*8.0, t766 = t248*8.0, t769 = t40*t128, t770 = t36*t130, t771 = t40*t129, t772 = -t256, t773 = t32*t132, t774 = t36*t131, t775 = t32*t133, t776 = t253*-2.0, t777 = t41*t128;
        T t778 = t254*-2.0, t779 = t37*t130, t780 = t255*-2.0, t781 = t41*t129, t782 = -t263, t783 = t257*-2.0, t784 = t33*t132, t785 = t258*-2.0, t786 = t37*t131, t787 = t259*-2.0, t788 = t33*t133, t790 = t42*t128, t791 = t38*t130, t792 = t42*t129;
        T t796 = t34*t132, t798 = t38*t131, t799 = t34*t133, t801 = t40*t130, t802 = -t276, t803 = t36*t132, t804 = t40*t131, t805 = t36*t133, t806 = -t280, t807 = t41*t130, t808 = t276*-2.0, t809 = -t282, t810 = t37*t133, t811 = t280*-2.0;
        T t812 = -t286, t814 = t42*t130, t817 = t38*t133, t827 = t281*8.0, t828 = t285*8.0, t831 = t40*t132, t832 = t40*t133, t833 = -t293, t834 = t41*t132, t835 = t293*-2.0, t836 = -t296, t839 = t42*t132, t846 = t294*8.0, t848 = -t299, t849 = -t300;
        T t853 = t11*t156, t854 = t36*t135, t858 = t40*t141, t859 = t40*t144, t860 = t31*t144, t861 = t40*t145, t862 = t30*t145, t863 = t36*t149, t864 = t29*t149, t865 = t29*t101, t866 = t29*t102, t867 = t30*t101, t868 = t30*t103, t869 = t31*t102;
        T t870 = t31*t103, t873 = t62*t229, t874 = t29*t65*3.0, t875 = t62*t225, t877 = t2*t181*4.0, t878 = t29*t65*5.0, t879 = t2*t182*4.0, t881 = t29*t65*6.0, t882 = t63*t68, t884 = t62*t67, t885 = t62*t70, t888 = t63*t237, t889 = t30*t65*3.0;
        T t890 = t63*t225, t891 = t62*t238, t892 = t62*t230, t893 = t62*t240, t894 = t62*t231, t900 = t2*t185*4.0, t901 = t30*t65*5.0, t902 = t2*t186*4.0, t908 = t30*t65*6.0, t909 = t64*t68, t914 = t62*t69, t917 = t62*t71, t920 = t106*tz*3.0;
        T t924 = t107*ty*3.0, t928 = t108*tz*3.0, t932 = t29*t72*3.0, t933 = t62*t256, t935 = t62*t241, t936 = t109*ty*3.0, t944 = t108*tz*5.0, t947 = t29*t72*5.0, t948 = t6*t181*4.0, t950 = t6*t182*4.0, t951 = t109*ty*5.0, t954 = t108*tz*6.0;
        T t956 = t29*t72*6.0, t958 = t109*ty*6.0, t963 = t63*t69, t965 = t64*t70, t967 = t63*t71, t970 = t63*t75, t974 = t110*tz*3.0, t978 = t63*t275, t979 = t63*t239, t980 = t112*tz*3.0, t986 = t113*tx*3.0, t988 = t63*t278, t989 = t63*t242;
        T t992 = t115*tx*3.0, t1002 = t110*tz*5.0, t1010 = t115*tx*5.0, t1012 = t110*tz*6.0, t1015 = t115*tx*6.0, t1023 = t63*t77, t1025 = t62*t78, t1029 = t116*ty*3.0, t1032 = t64*t276, t1033 = t31*t72*3.0, t1034 = t64*t256, t1037 = t117*tx*3.0;
        T t1041 = t118*ty*3.0, t1045 = t119*tx*3.0, t1052 = t116*ty*5.0, t1055 = t6*t189*4.0, t1056 = t31*t72*5.0, t1057 = t6*t190*4.0, t1059 = t117*tx*5.0, t1062 = t116*ty*6.0, t1064 = t31*t72*6.0, t1066 = t117*tx*6.0, t1068 = t64*t74;
        T t1069 = t64*t77, t1072 = t64*t291, t1073 = t64*t277, t1074 = t64*t292, t1075 = t64*t279, t1076 = t30*t79*3.0, t1077 = t63*t299, t1079 = t63*t280, t1088 = t30*t79*5.0, t1089 = t10*t185*4.0, t1091 = t10*t186*4.0, t1092 = t30*t79*6.0;
        T t1094 = t64*t78, t1095 = t31*t79*3.0, t1096 = t64*t299, t1098 = t64*t293, t1099 = t31*t79*5.0, t1100 = t10*t189*4.0, t1102 = t10*t190*4.0, t1103 = t31*t79*6.0, t1105 = t29*t68*-2.0, t1109 = t104*tz*-4.0, t1110 = t105*tz*-4.0;
        T t1111 = t107*tz*-2.0, t1112 = t29*t73*-2.0, t1113 = t29*t75*-2.0, t1117 = t113*tz*-2.0, t1118 = t29*t77*-2.0, t1121 = t29*t79*-2.0, t1123 = t20+t55, t1124 = t24+t50, t1125 = t20+t60, t1126 = t28+t50, t1127 = t24+t60, t1128 = t28+t55;
        T t1132 = t66*t155, t1135 = t2*t181*-2.0, t1137 = t2*t182*-2.0, t1141 = t2*t181*8.0, t1142 = t2*t182*8.0, t1143 = t68*t156, t1144 = t67*t155, t1145 = t70*t155, t1147 = t104*t379, t1149 = t104*t378, t1151 = t105*t379, t1153 = t105*t378;
        T t1155 = t2*t185*-2.0, t1157 = t2*t186*-2.0, t1174 = t2*t185*8.0, t1175 = t2*t186*8.0, t1178 = t68*t157, t1179 = t66*t157, t1180 = t69*t155, t1181 = t71*t155, t1183 = t106*tz*-2.0, t1184 = t106*t380, t1186 = t106*t378, t1187 = t107*t379;
        T t1188 = t107*t378, t1190 = t108*t380, t1191 = t108*t378, t1193 = t109*t379, t1195 = t109*t378, t1205 = t6*t181*-2.0, t1207 = t6*t182*-2.0, t1211 = t108*tz*-4.0, t1226 = t6*t181*8.0, t1227 = t6*t182*8.0, t1230 = t69*t156, t1231 = t70*t157;
        T t1232 = t73*t156, t1233 = t71*t156, t1234 = t75*t156, t1236 = t110*t380, t1238 = t110*t379, t1240 = t111*t380, t1242 = t111*t378, t1243 = t112*t380, t1244 = t112*t379, t1247 = t113*t379, t1248 = t113*tx*-2.0, t1249 = t113*t378;
        T t1251 = t114*t380, t1253 = t114*t378, t1254 = t115*t379, t1256 = t115*t378, t1276 = t115*tx*-4.0, t1293 = t73*t157, t1294 = t77*t156, t1295 = t78*t155, t1296 = t76*t155, t1297 = t116*t380, t1299 = t116*t379, t1302 = t117*t380;
        T t1304 = t117*t378, t1306 = t118*t380, t1307 = t118*ty*-2.0, t1308 = t118*t379, t1309 = t119*t380, t1310 = t119*t378, t1317 = t6*t189*-2.0, t1319 = t6*t190*-2.0, t1328 = t116*ty*-4.0, t1340 = t6*t189*8.0, t1341 = t6*t190*8.0;
        T t1344 = t74*t157, t1345 = t77*t157, t1346 = t76*t156, t1348 = t120*t380, t1350 = t120*t379, t1352 = t121*t380, t1354 = t121*t379, t1361 = t10*t185*-2.0, t1363 = t10*t186*-2.0, t1376 = t10*t185*8.0, t1377 = t10*t186*8.0, t1378 = t78*t157;
        T t1380 = t10*t189*-2.0, t1382 = t10*t190*-2.0, t1388 = t10*t189*8.0, t1389 = t10*t190*8.0, t1390 = t709*tx, t1391 = t709*ty, t1392 = t710*tx, t1393 = t710*tz, t1394 = t711*ty, t1395 = t711*tz, t1396 = ox*t441, t1397 = oy*t440;
        T t1398 = oz*t439, t1399 = rx*t439, t1400 = rx*t440, t1401 = ry*t439, t1402 = ry*t441, t1403 = rz*t440, t1404 = rz*t441, t1405 = t12+t135, t1406 = t14+t138, t1407 = t17+t141, t1410 = t30*t106, t1411 = t29*t107, t1412 = t30*t108;
        T t1414 = t29*t109, t1415 = t31*t110, t1416 = t31*t112, t1417 = t29*t113, t1418 = t29*t115, t1419 = t31*t116, t1421 = t30*t117, t1422 = t31*t118, t1423 = t30*t119, t1426 = t21+t145, t1427 = t50+t55, t1428 = t51+t57, t1429 = t23+t149;
        T t1430 = t50+t60, t1431 = t51+t61, t1432 = t26+t152, t1433 = t55+t60, t1434 = t57+t61, t1438 = t62*t237, t1441 = t29*t237*4.0, t1442 = t2*t328*4.0, t1443 = t2*t329*4.0, t1444 = t29*t243*4.0, t1445 = t66*t328*4.0, t1446 = t65*t329*4.0;
        T t1447 = t62*t253, t1448 = t62*t110, t1449 = t62*t106, t1450 = t62*t255, t1451 = t62*t112, t1452 = t62*t108, t1453 = t40*t104*3.0, t1454 = t31*t104*3.0, t1455 = t30*t104*3.0, t1456 = t40*t105*3.0, t1457 = t31*t105*3.0, t1458 = t30*t105*3.0;
        T t1465 = t40*t104*6.0, t1466 = t31*t104*6.0, t1467 = t30*t104*6.0, t1468 = t40*t105*6.0, t1469 = t31*t105*6.0, t1470 = t30*t105*6.0, t1471 = t62*t275, t1472 = t62*t116, t1473 = t62*t111, t1474 = t62*t276, t1477 = t62*t278, t1478 = t62*t118;
        T t1479 = t62*t114, t1480 = t40*t107*3.0, t1481 = t31*t107*3.0, t1482 = t30*t107*3.0, t1483 = t40*t109*3.0, t1484 = t31*t109*3.0, t1485 = t30*t109*3.0, t1486 = t40*t107*4.0, t1487 = t31*t107*4.0, t1488 = t30*t107*4.0, t1489 = t29*t276*4.0;
        T t1490 = t6*t328*4.0, t1491 = t6*t329*4.0, t1495 = t29*t282*4.0, t1496 = t72*t328*4.0, t1497 = t71*t329*4.0, t1498 = t40*t107*6.0, t1499 = t31*t107*6.0, t1500 = t30*t107*6.0, t1501 = t40*t109*6.0, t1502 = t31*t109*6.0, t1503 = t30*t109*6.0;
        T t1504 = t62*t291, t1505 = t62*t120, t1506 = t62*t117, t1507 = t62*t292, t1508 = t62*t121, t1509 = t62*t119, t1510 = t40*t113*3.0, t1511 = t31*t113*3.0, t1512 = t30*t113*3.0, t1513 = t40*t115*3.0, t1514 = t31*t115*3.0, t1515 = t30*t115*3.0;
        T t1516 = t40*t113*4.0, t1517 = t31*t113*4.0, t1518 = t30*t113*4.0, t1522 = t40*t113*6.0, t1523 = t31*t113*6.0, t1524 = t30*t113*6.0, t1525 = t40*t115*6.0, t1526 = t31*t115*6.0, t1527 = t30*t115*6.0, t1528 = t62*t299, t1531 = t29*t299*4.0;
        T t1532 = t10*t328*4.0, t1533 = t10*t329*4.0, t1534 = t29*t300*4.0, t1535 = t78*t328*4.0, t1536 = t76*t329*4.0, t1537 = t30*t709, t1538 = t29*t710, t1539 = t31*t709, t1540 = t29*t711, t1541 = t31*t710, t1542 = t30*t711, t1543 = t21*t440;
        T t1544 = t23*t439, t1545 = t22*t441, t1546 = t26*t439, t1547 = t25*t441, t1548 = t27*t440, t1549 = t155*(t31-tx), t1550 = t155*(t30-ty), t1552 = t156*(t31-tx), t1554 = t156*(t29-tz), t1556 = t157*(t30-ty), t1557 = t157*(t29-tz);
        T t1570 = t439*tx*2.0, t1571 = t441*ty*2.0, t1572 = t440*tz*2.0, t1573 = t40+t439, t1575 = t1408*3.0, t1576 = t1408*5.0, t1577 = t1408*6.0, t1579 = t1409*3.0, t1580 = t29*t104*3.0, t1581 = t29*t105*3.0, t1582 = t1409*5.0, t1583 = t1409*6.0;
        T t1584 = t29*t104*6.0, t1585 = t29*t105*6.0, t1587 = t63*t106, t1588 = t62*t107, t1592 = t1413*3.0, t1594 = t1413*5.0, t1596 = t1413*6.0, t1599 = t64*t112, t1601 = t62*t113, t1603 = t30*t111*3.0, t1604 = t30*t114*3.0, t1607 = t30*t111*6.0;
        T t1608 = t30*t114*6.0, t1613 = t64*t118, t1614 = t63*t119, t1616 = t1420*3.0, t1618 = t1420*5.0, t1620 = t1420*6.0, t1623 = t31*t120*3.0, t1624 = t31*t121*3.0, t1625 = t1424*3.0, t1626 = t1424*5.0, t1627 = t31*t120*6.0, t1628 = t31*t121*6.0;
        T t1629 = t1424*6.0, t1631 = t1425*3.0, t1632 = t1425*5.0, t1633 = t1425*6.0, t1638 = t2*t328*-2.0, t1639 = t2*t329*-2.0, t1643 = t29*t237*8.0, t1644 = t2*t328*8.0, t1657 = t40*t104*-4.0, t1658 = t31*t104*-4.0, t1659 = t40*t105*-4.0;
        T t1660 = t30*t105*-4.0, t1667 = t40*t107*-2.0, t1668 = t31*t107*-2.0, t1669 = t30*t107*-2.0, t1671 = t6*t328*-2.0, t1672 = t6*t329*-2.0, t1687 = t31*t109*-4.0, t1688 = t30*t109*-4.0, t1695 = t29*t276*8.0, t1696 = t6*t329*8.0;
        T t1697 = t40*t113*-2.0, t1698 = t31*t113*-2.0, t1699 = t30*t113*-2.0, t1711 = t31*t115*-4.0, t1712 = t30*t115*-4.0, t1719 = t29*t299*-2.0, t1720 = t10*t328*-2.0, t1721 = t10*t329*-2.0, t1725 = t10*t328*8.0, t1726 = t10*t329*8.0;
        T t1727 = t60+t709, t1728 = t55+t710, t1729 = t50+t711, t1730 = t32*t440, t1731 = t36*t439, t1732 = t40*t441, t1733 = t68*t328*6.0, t1734 = t73*t329*6.0, t1735 = (t29*t29)*t79*6.0, t1769 = t2*t29*t155, t1770 = t29*t556, t1773 = t2*t30*t156;
        T t1774 = t30*t558, t1781 = t106*t156, t1782 = t107*t155, t1783 = t108*t156, t1784 = t6*t29*t155, t1785 = t109*t155, t1786 = t31*t566, t1788 = t29*t568, t1796 = t110*t157, t1797 = t112*t157, t1798 = t113*t155, t1799 = t115*t155;
        T t1801 = t30*t579, t1811 = t116*t157, t1812 = t6*t31*t157, t1813 = t117*t156, t1814 = t118*t157, t1815 = t119*t156, t1817 = t31*t589, t1819 = t29*t591, t1826 = t10*t30*t156, t1827 = t30*t597, t1834 = t10*t31*t157, t1835 = t31*t599;
        T t1838 = t143+t148, t1839 = t143+t154, t1840 = t148+t154, t1861 = t378+t441, t1862 = t379+t440, t1863 = t380+t439, t1864 = t103*t439, t1865 = t102*t440, t1866 = t101*t441, t1896 = t439*t709, t1897 = t440*t710, t1898 = t441*t711, t412 = -t206;
        T t419 = -t209, t444 = t227*2.0, t452 = t235*2.0, t453 = t236*2.0, t457 = t236*5.0, t472 = t249*2.0, t473 = t250*2.0, t474 = t251*2.0, t475 = t252*2.0, t481 = t250*5.0, t483 = t252*5.0, t498 = t267*2.0, t499 = t268*2.0, t500 = t269*2.0;
        T t501 = t270*2.0, t502 = t271*2.0, t503 = t272*2.0, t504 = t273*2.0, t519 = t287*2.0, t520 = t288*2.0, t521 = t289*2.0, t522 = t290*2.0, t528 = t287*5.0, t530 = t289*5.0, t538 = t297*2.0, t539 = t298*2.0, t543 = t297*5.0, t546 = t301*2.0;
        T t570 = -t344, t572 = -t346, t582 = -t354, t583 = -t357, t592 = -t364, t594 = -t366, t678 = t3*t224, t679 = t4*t224, t680 = t336*tz, t681 = t337*tz, t682 = t5*t224, t688 = t8*t224, t717 = -t443, t718 = -t227, t726 = -t235, t727 = -t450;
        T t730 = -t454, t731 = -t455, t734 = t236*1.0E+1, t747 = -t463, t748 = -t249, t749 = -t464, t750 = -t465, t752 = -t469, t753 = -t251, t754 = -t470, t755 = -t471, t759 = -t476, t760 = -t477, t761 = -t478, t762 = -t479, t767 = t250*1.0E+1;
        T t768 = t252*1.0E+1, t789 = -t491, t793 = -t494, t794 = -t270, t795 = -t495, t797 = -t496, t813 = -t509, t815 = -t288, t816 = -t515, t818 = -t290, t821 = -t523, t822 = -t524, t823 = -t525, t824 = -t526, t829 = t287*1.0E+1, t830 = t289*1.0E+1;
        T t837 = -t533, t838 = -t534, t840 = -t537, t841 = -t298, t843 = -t540, t844 = -t541, t847 = t297*1.0E+1, t850 = -t545, t851 = -t301, t872 = t604*3.0, t876 = t604*5.0, t880 = t604*6.0, t883 = t333*ty, t886 = t334*tx, t887 = t605*3.0;
        T t895 = t336*ty, t896 = t607*4.0, t897 = t608*4.0, t899 = t605*5.0, t903 = t36*t336, t906 = t32*t337, t907 = t605*6.0, t913 = t340*ty, t915 = t341*tz, t916 = t341*tx, t918 = t343*ty, t919 = t343*tx, t922 = t616*3.0, t926 = t617*3.0;
        T t930 = t618*3.0, t934 = t619*3.0, t938 = t620*3.0, t942 = t346*ty, t943 = t346*tx, t945 = t40*t344, t946 = t32*t344, t949 = t619*5.0, t952 = t36*t346, t953 = t32*t346, t955 = t618*6.0, t957 = t619*6.0, t959 = t620*6.0, t960 = t347*tz;
        T t961 = t347*ty, t962 = t348*tz, t964 = t349*tz, t971 = t352*tx, t972 = t353*ty, t973 = t353*tx, t976 = t629*3.0, t982 = t632*3.0, t984 = t635*3.0, t990 = t638*3.0, t994 = t354*tz, t995 = t354*ty, t997 = t631*4.0, t998 = t636*4.0;
        T t999 = t356*tx, t1003 = t40*t354, t1004 = t36*t354, t1005 = t40*t355, t1008 = t32*t356, t1009 = t36*t357, t1011 = t32*t357, t1013 = t629*6.0, t1014 = t638*6.0, t1016 = t358*tz, t1017 = t358*ty, t1019 = t360*tz, t1020 = t360*tx;
        T t1024 = t362*tx, t1027 = t651*3.0, t1031 = t652*3.0, t1035 = t653*3.0, t1039 = t654*3.0, t1043 = t655*3.0, t1049 = t366*tz, t1050 = t366*tx, t1051 = t40*t364, t1053 = t36*t364, t1054 = t652*5.0, t1058 = t40*t366, t1060 = t32*t366;
        T t1061 = t651*6.0, t1063 = t652*6.0, t1065 = t653*6.0, t1067 = t367*tz, t1070 = t368*ty, t1078 = t670*3.0, t1080 = t370*tz, t1081 = t667*4.0, t1082 = t668*4.0, t1084 = t40*t370, t1087 = t36*t371, t1090 = t670*5.0, t1093 = t670*6.0;
        T t1097 = t675*3.0, t1101 = t675*5.0, t1104 = t675*6.0, t1106 = -t677, t1107 = t559*tz, t1108 = t560*tz, t1114 = -t685, t1115 = -t686, t1116 = -t687, t1119 = -t690, t1120 = -t691, t1122 = -t693, t1129 = -t733, t1130 = -t827, t1131 = -t828;
        T t1133 = t604*-2.0, t1136 = -t874, t1138 = -t877, t1139 = -t878, t1140 = -t879, t1146 = t559*ty, t1148 = t607*-2.0, t1150 = t608*-2.0, t1152 = t560*tx, t1156 = -t889, t1158 = t36*t559, t1159 = t32*t559, t1160 = t36*t560, t1161 = t32*t560;
        T t1162 = t606*-4.0, t1164 = t609*-4.0, t1165 = -t900, t1166 = -t901, t1167 = -t902, t1168 = t610*-4.0, t1169 = t611*-4.0, t1170 = t612*-4.0, t1171 = t613*-4.0, t1173 = -t908, t1176 = t610*8.0, t1177 = t613*8.0, t1182 = t615*-2.0;
        T t1185 = t616*-2.0, t1189 = t567*tz, t1192 = t569*ty, t1194 = t569*tx, t1196 = t621*-2.0, t1197 = t622*-2.0, t1198 = t623*-2.0, t1199 = t624*-2.0, t1200 = -t928, t1201 = t40*t567, t1203 = t32*t567, t1204 = -t932, t1208 = -t936;
        T t1209 = t36*t569, t1210 = t32*t569, t1212 = t618*-4.0, t1213 = t625*-4.0, t1214 = t626*-4.0, t1215 = -t947, t1216 = -t948, t1217 = -t950, t1218 = t627*-4.0, t1219 = t628*-4.0, t1220 = -t956, t1222 = -t958, t1224 = t625*8.0, t1225 = t626*8.0;
        T t1228 = t627*8.0, t1229 = t628*8.0, t1235 = t577*tz, t1237 = t577*ty, t1239 = t578*tz, t1241 = t631*-2.0, t1245 = t633*-2.0, t1246 = t635*-2.0, t1250 = t636*-2.0, t1252 = t580*tx, t1255 = t581*tx, t1257 = -t974, t1258 = t40*t577;
        T t1259 = t36*t577, t1260 = t40*t578, t1261 = t32*t578, t1262 = t643*-2.0, t1263 = t644*-2.0, t1264 = t645*-2.0, t1265 = t646*-2.0, t1266 = t40*t580, t1267 = t32*t580, t1269 = t36*t581, t1270 = -t992, t1271 = t32*t581, t1272 = t630*-4.0;
        T t1274 = t637*-4.0, t1275 = t638*-4.0, t1277 = t639*-4.0, t1278 = t640*-4.0, t1279 = t641*-4.0, t1280 = t642*-4.0, t1281 = t647*-4.0, t1282 = t648*-4.0, t1283 = t649*-4.0, t1284 = t650*-4.0, t1285 = -t1012, t1287 = t639*8.0, t1288 = t640*8.0;
        T t1289 = t641*8.0, t1290 = t648*8.0, t1291 = t649*8.0, t1292 = t650*8.0, t1298 = t588*ty, t1300 = t652*-2.0, t1301 = t590*tz, t1303 = t590*tx, t1305 = t654*-2.0, t1311 = t657*-2.0, t1313 = t40*t588, t1314 = -t1029, t1315 = t36*t588;
        T t1318 = -t1033, t1320 = t40*t590, t1321 = -t1037, t1322 = t32*t590, t1323 = t662*-2.0, t1324 = t663*-2.0, t1325 = t664*-2.0, t1326 = t665*-2.0, t1327 = t651*-4.0, t1329 = t658*-4.0, t1330 = t659*-4.0, t1331 = -t1055, t1332 = -t1056;
        T t1333 = -t1057, t1334 = t660*-4.0, t1335 = t661*-4.0, t1337 = -t1066, t1338 = t658*8.0, t1339 = t659*8.0, t1342 = t660*8.0, t1343 = t661*8.0, t1347 = t595*tz, t1349 = t667*-2.0, t1351 = t668*-2.0, t1353 = t596*ty, t1355 = t670*-2.0;
        T t1356 = t40*t595, t1357 = t36*t595, t1358 = t40*t596, t1359 = t36*t596, t1360 = -t1076, t1364 = t666*-4.0, t1366 = t669*-4.0, t1367 = t671*-4.0, t1368 = t672*-4.0, t1369 = t673*-4.0, t1370 = t674*-4.0, t1371 = -t1088, t1372 = -t1089;
        T t1373 = -t1091, t1374 = t671*8.0, t1375 = t674*8.0, t1379 = -t1095, t1383 = -t1099, t1384 = -t1100, t1385 = -t1102, t1386 = -t1103, t1439 = t31*t330, t1440 = t30*t330, t1459 = t40*t336, t1460 = t31*t336, t1461 = t30*t336, t1462 = t40*t337;
        T t1463 = t31*t337, t1464 = t30*t337, t1475 = t31*t342, t1476 = t30*t342, t1492 = t40*t346, t1493 = t31*t346, t1494 = t30*t346, t1519 = t40*t357, t1520 = t31*t357, t1521 = t30*t357, t1529 = t31*t363, t1530 = t30*t363, t1558 = -t1142;
        T t1559 = -t1175, t1562 = -t1226, t1565 = -t1341, t1568 = -t1376, t1569 = -t1388, t1586 = t31*t338, t1595 = t1412*6.0, t1597 = t1414*6.0, t1600 = t30*t350, t1606 = t1415*6.0, t1609 = t1418*6.0, t1615 = t29*t363, t1619 = t1419*6.0;
        T t1621 = t1421*6.0, t1634 = t52+t146, t1635 = t54+t150, t1636 = t58+t153, t1637 = t29*t741, t1640 = -t1441, t1641 = -t1442, t1642 = -t1443, t1645 = t40*t559, t1646 = t31*t559, t1647 = t30*t559, t1648 = t40*t560, t1649 = t31*t560;
        T t1650 = t30*t560, t1651 = -t1453, t1652 = -t1454, t1653 = -t1455, t1654 = -t1456, t1655 = -t1457, t1656 = -t1458, t1661 = -t1465, t1662 = -t1466, t1663 = -t1467, t1664 = -t1468, t1665 = -t1469, t1666 = -t1470, t1670 = t29*t808;
        T t1673 = t40*t569, t1674 = t31*t569, t1675 = t30*t569, t1676 = -t1480, t1677 = -t1481, t1678 = -t1482, t1679 = -t1483, t1680 = -t1484, t1681 = -t1485, t1682 = -t1486, t1683 = -t1487, t1684 = -t1489, t1685 = -t1490, t1686 = -t1491;
        T t1689 = -t1498, t1690 = -t1499, t1691 = -t1500, t1692 = -t1501, t1693 = -t1502, t1694 = -t1503, t1700 = t40*t581, t1701 = t31*t581, t1702 = t30*t581, t1703 = -t1510, t1704 = -t1511, t1705 = -t1512, t1706 = -t1513, t1707 = -t1514;
        T t1708 = -t1515, t1709 = -t1516, t1710 = -t1518, t1713 = -t1522, t1714 = -t1523, t1715 = -t1524, t1716 = -t1525, t1717 = -t1526, t1718 = -t1527, t1722 = -t1531, t1723 = -t1532, t1724 = -t1533, t1736 = t1405*tz, t1737 = t1406*ty;
        T t1738 = t1407*tx, t1739 = oz*t1426, t1740 = oy*t1429, t1741 = ox*t1432, t1742 = -t1390, t1743 = t1427*tx, t1744 = t1428*tx, t1745 = -t1391, t1746 = t1427*ty, t1747 = t1428*ty, t1748 = -t1392, t1749 = t1430*tx, t1750 = t1431*tx;
        T t1751 = -t1393, t1752 = t1430*tz, t1753 = t1431*tz, t1754 = -t1394, t1755 = t1433*ty, t1756 = t1434*ty, t1757 = -t1395, t1758 = t1433*tz, t1759 = t1434*tz, t1760 = t89+t430, t1761 = t89+t431, t1762 = t90+t431, t1763 = -t1399, t1764 = -t1400;
        T t1765 = -t1401, t1766 = -t1402, t1767 = -t1403, t1768 = -t1404, t1771 = -t1575, t1772 = -t1577, t1775 = -t1579, t1776 = -t1580, t1777 = -t1581, t1778 = -t1583, t1779 = -t1584, t1780 = -t1585, t1787 = t30*t567, t1789 = t29*t569;
        T t1790 = -t1592, t1792 = -t1596, t1794 = t1412*1.2E+1, t1795 = t1414*1.2E+1, t1800 = t31*t577, t1802 = t29*t581, t1803 = -t1603, t1804 = -t1604, t1806 = -t1607, t1807 = -t1608, t1809 = t1415*1.2E+1, t1810 = t1418*1.2E+1, t1816 = t31*t588;
        T t1818 = t30*t590, t1820 = -t1616, t1822 = -t1620, t1824 = t1419*1.2E+1, t1825 = t1421*1.2E+1, t1828 = -t1623, t1829 = -t1624, t1830 = -t1625, t1831 = -t1627, t1832 = -t1628, t1833 = -t1629, t1836 = -t1631, t1837 = -t1633, t1841 = t30*t1427;
        T t1842 = t29*t1430, t1843 = t31*t1427, t1844 = t29*t1433, t1845 = t31*t1430, t1846 = t30*t1433, t1847 = t1573*t1573, t1854 = ox*t1727, t1855 = ox*t1728, t1856 = oy*t1727, t1857 = oy*t1729, t1858 = oz*t1728, t1859 = oz*t1729;
        T t1860 = t60+t1427, t1881 = t88+t1400, t1882 = t93+t1399, t1883 = t86+t1402, t1884 = t94+t1401, t1885 = t87+t1404, t1886 = t92+t1403, t1893 = t1861*tx, t1894 = t1862*ty, t1895 = t1863*tz, t1920 = t439*t1427, t1921 = t440*t1430;
        T t1922 = t441*t1433, t1926 = t21+t22+t92+t1393, t1927 = t23+t25+t86+t1394, t1928 = t26+t27+t93+t1390, t1929 = t1405*t1573*tx, t1930 = t1405*t1573*ty, t1931 = t1406*t1573*tx, t1935 = t1406*t1573*tz, t1936 = t1407*t1573*ty;
        T t1937 = t1407*t1573*tz, t1955 = t160+t168+t411+t421, t2019 = t173+t175+t215+t219+t387, t2020 = t173+t175+t217+t221+t388, t2021 = t174+t177+t212+t222+t389, t2022 = t174+t177+t213+t223+t390, t2023 = t176+t178+t214+t218+t391;
        T t2024 = t176+t178+t216+t220+t392, t2037 = t44+t81+t83+t136+t698+t701, t2039 = t48+t82+t84+t142+t694+t697, t2043 = -t1573*(t82+t163+t418+t694), t2045 = -t1573*(t81+t169+t417+t698), t2049 = -t1573*tx*(t82+t84+t694+t697);
        T t2050 = -t1573*tx*(t80+t85+t695+t700), t2051 = -t1573*ty*(t82+t84+t694+t697), t2052 = -t1573*tx*(t81+t83+t698+t701), t2053 = -t1573*ty*(t80+t85+t695+t700), t2054 = -t1573*tz*(t82+t84+t694+t697), t2055 = -t1573*ty*(t81+t83+t698+t701);
        T t2056 = -t1573*tz*(t80+t85+t695+t700), t2057 = -t1573*tz*(t81+t83+t698+t701), t2058 = t1573*(t80+t168+t421+t695), t2059 = t1573*tx*(t82+t84+t694+t697)*-2.0, t2060 = t1573*tx*(t80+t85+t695+t700)*-2.0;
        T t2061 = t1573*ty*(t82+t84+t694+t697)*-2.0, t2062 = t1573*tx*(t81+t83+t698+t701)*-2.0, t2063 = t1573*ty*(t80+t85+t695+t700)*-2.0, t2064 = t1573*tz*(t82+t84+t694+t697)*-2.0, t2065 = t1573*ty*(t81+t83+t698+t701)*-2.0;
        T t2066 = t1573*tz*(t80+t85+t695+t700)*-2.0, t2067 = t1573*tz*(t81+t83+t698+t701)*-2.0, t719 = -t444, t729 = -t453, t732 = -t457, t757 = -t473, t758 = -t475, t763 = -t481, t764 = -t483, t800 = -t501, t819 = -t519, t820 = -t521, t825 = -t528;
        T t826 = -t530, t842 = -t538, t845 = -t543, t852 = -t546, t1134 = -t872, t1154 = -t887, t1163 = -t897, t1172 = -t907, t1202 = -t930, t1206 = -t934, t1221 = -t957, t1223 = -t959, t1268 = -t990, t1273 = -t997, t1286 = -t1013, t1312 = -t1027;
        T t1316 = -t1031, t1336 = -t1065, t1362 = -t1078, t1365 = -t1082, t1381 = -t1097, t1387 = -t1104, t1560 = -t1176, t1561 = -t1177, t1563 = -t1289, t1564 = -t1290, t1566 = -t1374, t1567 = -t1375, t1791 = -t1595, t1793 = -t1597, t1805 = -t1606;
        T t1808 = -t1609, t1821 = -t1619, t1823 = -t1621, t1851 = oz*t1634, t1852 = oy*t1635, t1853 = ox*t1636, t1867 = t1760*tx, t1868 = t1761*tx, t1869 = t1760*ty, t1870 = t1762*ty, t1871 = t1761*tz, t1872 = t1762*tz, t1873 = -t1737, t1874 = -t1744;
        T t1875 = -t1747, t1876 = -t1750, t1877 = -t1753, t1878 = -t1756, t1879 = -t1759, t1887 = -t1854, t1888 = ox*t1860, t1889 = -t1857, t1890 = oy*t1860, t1891 = -t1858, t1892 = oz*t1860, t1905 = t11*t1860, t1906 = t15*t1860, t1907 = t19*t1860;
        T t1908 = t1881*tx, t1909 = t1882*tx, t1910 = t1883*tx, t1911 = t1881*ty, t1912 = t1883*ty, t1913 = t1885*tx, t1914 = t1882*tz, t1915 = t1884*ty, t1916 = t1886*ty, t1917 = t1884*tz, t1918 = t1885*tz, t1919 = t1886*tz, t1923 = -t1920;
        T t1924 = -t1921, t1925 = -t1922, t1932 = t1573*t1736, t1933 = t1573*t1737, t1934 = t1573*t1738, t1938 = t1926*tx, t1939 = t1928*tx, t1940 = t1927*ty, t1941 = t1928*ty, t1942 = t1926*tz, t1943 = t1927*tz, t1944 = t1929*2.0, t1945 = t1930*2.0;
        T t1946 = t1931*2.0, t1950 = t1935*2.0, t1951 = t1936*2.0, t1952 = t1937*2.0, t1969 = t29*t1926, t1971 = t30*t1927, t1974 = t31*t1928, t1977 = t21+t22+t425+t1757, t1978 = t23+t25+t436+t1745, t1979 = t26+t27+t426+t1748;
        T t1980 = t1405*t1847*tx*2.0, t1981 = t1405*t1847*ty*2.0, t1982 = t1406*t1847*tx*2.0, t1983 = t1736*t1847*2.0, t1984 = t1737*t1847*2.0, t1985 = t1738*t1847*2.0, t1986 = t1406*t1847*tz*2.0, t1987 = t1407*t1847*ty*2.0;
        T t1988 = t1407*t1847*tz*2.0, t1989 = t155*t1926, t1990 = t155*t1927, t1991 = t156*t1927, t1992 = t156*t1928, t1993 = t157*t1926, t1994 = t157*t1928, t1995 = -oy*(t395-t1893), t1996 = -oz*(t393-t1893), t1997 = -ox*(t397-t1894);
        T t1998 = -oz*(t394-t1894), t1999 = -ox*(t398-t1895), t2000 = -oy*(t396-t1895), t2028 = t439*t1928, t2029 = t440*t1926, t2030 = t441*t1927, t2038 = t46+t139+t1955, t2068 = t1407*t2019, t2069 = t1406*t2021, t2070 = t1405*t2023;
        T t2090 = t1573*t2037*tx, t2092 = t1573*t2037*ty, t2093 = t1573*t2039*tx, t2095 = t1573*t2037*tz, t2096 = t1573*t2039*ty, t2098 = t1573*t2039*tz, t2124 = t1407*t1573*t2020*2.0, t2125 = t1406*t1573*t2022*2.0, t2126 = t1405*t1573*t2024*2.0;
        T t2232 = t226+t281+t480+t605+t631+t745+t876+t884+t960+t1052+t1119+t1164+t1215+t1232, t2233 = t263+t294+t456+t652+t667+t724+t885+t915+t949+t1059+t1114+t1139+t1162+t1178;
        T t2234 = t226+t244+t527+t604+t607+t812+t899+t963+t1002+t1017+t1120+t1274+t1295+t1371, t2235 = t285+t300+t482+t668+t675+t742+t918+t970+t1010+t1090+t1109+t1166+t1179+t1272;
        T t2236 = t234+t263+t542+t608+t619+t836+t944+t1020+t1054+t1068+t1116+t1296+t1366+t1383, t2237 = t248+t300+t529+t636+t670+t809+t951+t973+t1069+t1101+t1110+t1233+t1332+t1364;
        T t2238 = t443+t510+t752+t765+t880+t882+t896+t963+t994+t1062+t1119+t1164+t1220+t1245, t2239 = t516+t545+t747+t766+t942+t998+t1015+t1069+t1093+t1094+t1109+t1173+t1182+t1272;
        T t2240 = t451+t494+t840+t846+t885+t917+t954+t1050+t1063+t1081+t1116+t1311+t1366+t1386, t2271 = t267+t481+t728+t875+t877+t1051+t1053+t1207+t1216+t1302+t1304+t1309+t1310+t1410+t1441+t1473+t1580+t1606+t1670+t1675+t1704+t1707+t1719+t1777+t1783;
        T t2272 = t269+t457+t751+t933+t950+t1058+t1060+t1135+t1140+t1297+t1299+t1306+t1308+t1416+t1489+t1505+t1581+t1595+t1637+t1678+t1681+t1701+t1719+t1776+t1796;
        T t2273 = t268+t528+t756+t890+t900+t1003+t1004+t1247+t1249+t1254+t1256+t1363+t1372+t1411+t1442+t1449+t1603+t1619+t1650+t1671+t1703+t1706+t1720+t1785+t1804;
        T t2274 = t272+t483+t814+t1009+t1011+t1077+t1091+t1155+t1167+t1236+t1238+t1243+t1244+t1422+t1507+t1532+t1597+t1604+t1638+t1653+t1656+t1671+t1697+t1803+t1811;
        T t2275 = t271+t543+t817+t945+t946+t1034+t1055+t1187+t1188+t1193+t1195+t1382+t1384+t1417+t1451+t1491+t1621+t1623+t1639+t1646+t1676+t1679+t1721+t1799+t1829;
        T t2276 = t273+t530+t839+t952+t953+t1096+t1102+t1184+t1186+t1190+t1191+t1317+t1333+t1423+t1477+t1533+t1609+t1624+t1639+t1652+t1655+t1667+t1672+t1813+t1828;
        T t2283 = t290+t298+t718+t794+t989+t1075+t1147+t1149+t1151+t1153+t1261+t1279+t1357+t1367+t1478+t1509+t1615+t1625+t1631+t1651+t1654+t1668+t1699+t1769+t1775+t1784+t1820;
        T t2284 = t251+t288+t718+t851+t894+t1072+t1159+t1168+t1240+t1242+t1251+t1253+t1358+t1370+t1450+t1506+t1592+t1600+t1616+t1645+t1677+t1680+t1702+t1771+t1773+t1826+t1836;
        T t2285 = t235+t249+t794+t851+t891+t978+t1160+t1171+t1266+t1282+t1348+t1350+t1352+t1354+t1447+t1472+t1575+t1579+t1586+t1648+t1674+t1705+t1708+t1790+t1812+t1830+t1834, t1880 = -t1852, t1902 = -t1869, t1903 = -t1871, t1904 = -t1872;
        T t1947 = t1932*2.0, t1948 = t1933*2.0, t1949 = t1934*2.0, t1965 = -t1946, t1966 = t1573*t1873, t1968 = -t1950, t2001 = -t1980, t2002 = -t1981, t2003 = -t1983, t2004 = -t1985, t2005 = -t1987, t2006 = -t1988, t2007 = t1978*tx, t2008 = t1979*tx;
        T t2009 = t1977*ty, t2010 = t1978*ty, t2011 = t1977*tz, t2012 = t1979*tz, t2013 = t29*t1977, t2016 = t30*t1978, t2018 = t31*t1979, t2031 = t155*t1977, t2032 = t155*t1979, t2033 = t156*t1977, t2034 = t156*t1978, t2035 = t157*t1978;
        T t2036 = t157*t1979, t2040 = t439*t1978, t2041 = t440*t1979, t2042 = t441*t1977, t2046 = t1743+t1943, t2047 = t1755+t1938, t2048 = t1752+t1941, t2071 = t2068*2.0, t2072 = t2069*2.0, t2073 = t2070*2.0, t2078 = t435+t1399+t1874+t1881;
        T t2079 = t428+t1400+t1876+t1882, t2080 = t437+t1401+t1875+t1883, t2081 = t424+t1402+t1878+t1884, t2082 = t434+t1403+t1877+t1885, t2083 = t427+t1404+t1879+t1886, t2091 = t1573*t2038*tx, t2094 = t1573*t2038*ty, t2097 = t1573*t2038*tz;
        T t2127 = -t2124, t2128 = -t2126, t2138 = t864+t1539+t1548+t1867+t1917, t2139 = t860+t1542+t1544+t1870+t1913, t2140 = t862+t1541+t1546+t1868+t1916, t2150 = t325+t326+t1896+t1909+t1915, t2151 = t317+t324+t1897+t1908+t1919;
        T t2152 = t315+t316+t1898+t1912+t1918, t2241 = t448+t685+t793+t838+t881+t895+t909+t1129+t1163+t1211+t1221+t1300+t1337+t1349, t2242 = t518+t691+t717+t750+t999+t1025+t1092+t1130+t1133+t1148+t1172+t1273+t1285+t1328;
        T t2243 = t512+t681+t755+t850+t967+t1064+t1080+t1131+t1222+t1250+t1276+t1355+t1365+t1387, t2244 = t1573*t2232*2.0, t2245 = t1573*t2233*2.0, t2246 = t1573*t2234*2.0, t2247 = t1573*t2235*2.0, t2248 = t1573*t2236*2.0, t2249 = t1573*t2237*2.0;
        T t2277 = t498+t729+t767+t879+t1141+t1217+t1320+t1322+t1325+t1326+t1338+t1339+t1488+t1562+t1584+t1587+t1643+t1684+t1688+t1714+t1717+t1722+t1780+t1787+t1809;
        T t2278 = t500+t734+t757+t948+t1138+t1227+t1313+t1315+t1323+t1324+t1342+t1343+t1517+t1558+t1585+t1599+t1640+t1691+t1694+t1695+t1711+t1722+t1779+t1794+t1800;
        T t2279 = t499+t758+t829+t902+t1174+t1264+t1265+t1269+t1271+t1287+t1288+t1373+t1461+t1568+t1588+t1607+t1644+t1660+t1685+t1713+t1716+t1723+t1789+t1807+t1824;
        T t2280 = t503+t768+t819+t1089+t1165+t1258+t1259+t1262+t1263+t1291+t1292+t1377+t1519+t1559+t1608+t1613+t1641+t1663+t1666+t1685+t1709+t1725+t1795+t1806+t1816;
        T t2281 = t502+t820+t847+t1057+t1198+t1199+t1209+t1210+t1224+t1225+t1340+t1385+t1463+t1569+t1601+t1627+t1642+t1658+t1689+t1692+t1696+t1724+t1802+t1825+t1832;
        T t2282 = t504+t830+t842+t1100+t1196+t1197+t1201+t1203+t1228+t1229+t1331+t1389+t1492+t1565+t1614+t1628+t1642+t1662+t1665+t1682+t1686+t1726+t1810+t1818+t1831;
        T t2286 = t522+t539+t719+t800+t1008+t1087+t1158+t1159+t1160+t1161+t1280+t1368+t1493+t1521+t1563+t1566+t1617+t1629+t1633+t1661+t1664+t1683+t1710+t1770+t1778+t1788+t1822;
        T t2287 = t474+t520+t719+t852+t906+t1084+t1169+t1260+t1261+t1266+t1267+t1369+t1462+t1518+t1560+t1567+t1596+t1605+t1620+t1657+t1690+t1693+t1712+t1772+t1774+t1827+t1837;
        T t2288 = t452+t472+t800+t852+t903+t1005+t1170+t1281+t1356+t1357+t1358+t1359+t1459+t1487+t1561+t1564+t1577+t1583+t1593+t1659+t1687+t1715+t1718+t1792+t1817+t1833+t1835;
        T t2289 = t227+t270+t492+t610+t611+t612+t613+t677+t795+t818+t841+t914+t979+t995+t1005+t1061+t1073+t1084+t1115+t1212+t1246+t1267+t1336+t1359+t1408+t1413+t1453+t1456+t1472+t1506+t1579+t1616+t1674+t1702+t1819+t1830+t1836;
        T t2290 = t227+t301+t497+t641+t642+t647+t648+t693+t753+t789+t815+t892+t903+t943+t1014+t1023+t1074+t1087+t1106+t1161+t1185+t1286+t1327+t1356+t1409+t1424+t1447+t1481+t1484+t1509+t1575+t1631+t1648+t1699+t1790+t1801+t1820;
        T t2291 = t270+t301+t493+t671+t672+t673+t674+t686+t726+t748+t797+t893+t906+t955+t965+t988+t1008+t1049+t1122+t1158+t1223+t1260+t1275+t1305+t1420+t1425+t1450+t1478+t1512+t1515+t1592+t1625+t1645+t1668+t1771+t1775+t1786;
        T t2292 = t250+t658+t659+t662+t663+t732+t792+t873+t879+t1205+t1217+t1334+t1335+t1415+t1438+t1482+t1485+t1508+t1528+t1580+t1684+t1698+t1777+t1791+t1797+t2239;
        T t2293 = t297+t621+t622+t625+t626+t799+t826+t1032+t1057+t1218+t1219+t1380+t1385+t1421+t1440+t1454+t1457+t1471+t1476+t1623+t1673+t1724+t1808+t1815+t1829+t2238;
        T t2294 = t252+t645+t646+t649+t650+t791+t825+t1079+t1089+t1157+t1165+t1277+t1278+t1414+t1452+t1475+t1510+t1513+t1529+t1604+t1641+t1647+t1782+t1803+t1821+t2240, t1967 = -t1948, t2025 = -t2008, t2026 = -t2010, t2027 = -t2011;
        T t2074 = t1749+t2009, t2075 = t1746+t2012, t2076 = t1758+t2007, t2077 = -t2072, t2084 = t2078*tx, t2085 = t2079*tx, t2086 = t2080*ty, t2087 = t2081*ty, t2088 = t2082*tz, t2089 = t2083*tz, t2099 = t29*t2078, t2100 = t30*t2078;
        T t2101 = t29*t2079, t2102 = t31*t2078, t2103 = t30*t2079, t2104 = t29*t2080, t2105 = t31*t2079, t2106 = t30*t2080, t2107 = t31*t2080, t2108 = t29*t2081, t2111 = t31*t2081, t2112 = t30*t2082, t2113 = t29*t2083, t2114 = t31*t2082;
        T t2115 = t30*t2083, t2116 = t31*t2083, t2117 = -t2091, t2118 = -t2094, t2119 = -t2097, t2120 = t157*t2078, t2121 = t156*t2081, t2122 = t155*t2082, t2123 = t155*t2083, t2129 = t439*t2078, t2130 = t440*t2079, t2131 = t439*t2080;
        T t2132 = t441*t2081, t2133 = t440*t2082, t2134 = t441*t2083, t2141 = t2138*tx, t2142 = t2140*tx, t2143 = t2138*ty, t2144 = t2139*ty, t2145 = t2139*tz, t2146 = t2140*tz, t2147 = t861+t1538+t1545+t1903+t1911;
        T t2148 = t863+t1537+t1547+t1902+t1914, t2149 = t859+t1540+t1543+t1904+t1910, t2153 = t29*t2139, t2154 = t29*t2140, t2155 = t30*t2138, t2156 = t30*t2139, t2157 = t31*t2138, t2158 = t31*t2140, t2167 = t2150*tx, t2168 = t2151*tx;
        T t2169 = t2150*ty, t2170 = t2152*ty, t2171 = t2151*tz, t2172 = t2152*tz, t2179 = t155*t2139, t2180 = t155*t2140, t2181 = t156*t2138, t2182 = t29*t2151, t2183 = t30*t2150, t2184 = t29*t2152, t2185 = t31*t2150, t2186 = t30*t2152;
        T t2187 = t31*t2151, t2196 = t155*t2151, t2197 = t156*t2150, t2198 = t155*t2152, t2199 = t157*t2150, t2200 = t156*t2152, t2201 = t157*t2151, t2202 = t439*t2138, t2203 = t440*t2140, t2204 = t441*t2139, t2205 = t1920+t1971+t2018;
        T t2206 = t1921+t1974+t2013, t2207 = t1922+t1969+t2016, t2211 = t439*t2150, t2212 = t440*t2151, t2213 = t441*t2152, t2214 = t1843+t1990+t2041, t2215 = t1841+t2030+t2032, t2216 = t1845+t2028+t2033, t2217 = t1842+t1992+t2042;
        T t2218 = t1846+t1993+t2040, t2219 = t1844+t2029+t2035, t2250 = -t2244, t2251 = -t2247, t2252 = -t2248;
        T t2295 = t287+t639+t640+t643+t644+t764+t798+t888+t902+t1283+t1284+t1361+t1373+t1419+t1439+t1455+t1458+t1475+t1504+t1603+t1700+t1723+t1793+t1804+t1814+t2241;
        T t2296 = t236+t660+t661+t664+t665+t763+t790+t935+t948+t1137+t1138+t1329+t1330+t1412+t1474+t1479+t1511+t1514+t1528+t1581+t1640+t1669+t1776+t1781+t1805+t2243;
        T t2297 = t289+t623+t624+t627+t628+t796+t845+t1098+t1100+t1213+t1214+t1319+t1331+t1418+t1440+t1448+t1480+t1483+t1530+t1624+t1649+t1686+t1798+t1823+t1828+t2242, t2136 = t1939+t2027, t2159 = t2147*tx, t2160 = t2148*tx, t2161 = t2148*ty;
        T t2162 = t2149*ty, t2163 = t2147*tz, t2164 = t2149*tz, t2165 = -t2141, t2166 = -t2144, t2173 = t29*t2147, t2174 = t29*t2149, t2175 = t30*t2148, t2176 = t30*t2149, t2177 = t31*t2147, t2178 = t31*t2148, t2190 = t156*t2149, t2191 = t157*t2147;
        T t2192 = t157*t2148, t2208 = t439*t2148, t2209 = t440*t2147, t2210 = t441*t2149, t2220 = t2085+t2087, t2221 = t2084+t2089, t2222 = t2086+t2088, t2223 = t2105+t2121, t2224 = t2102+t2123, t2225 = t2106+t2122, t2226 = t2108+t2130;
        T t2227 = t2101+t2132, t2228 = t2115+t2129, t2229 = t2100+t2134, t2230 = t2114+t2131, t2231 = t2107+t2133, t2254 = t2145+t2167, t2255 = t2143+t2171, t2256 = t2146+t2169, t2262 = t2156+t2158+t2211, t2264 = t2179+t2185+t2203;
        T t2265 = t2180+t2183+t2204, t2188 = -t2163, t2189 = -t2164, t2253 = t2142+t2166, t2258 = t2159+t2170, t2259 = t2162+t2168, t2260 = t2160+t2172, t2263 = t2157+t2174+t2212, t2266 = t2173+t2175+t2213, t2267 = t2187+t2190+t2202;
        T t2268 = t2181+t2182+t2210, t2269 = t2186+t2191+t2208, t2270 = t2184+t2192+t2209, t2257 = t2141+t2189, t2261 = t2161+t2188;

        /* #endregion Repeated terms ----------------------------------------------------------------------------*/

        /* #region Calculating f and its jacobian ---------------------------------------------------------------*/
        
        fdrv(0, 1) = -rz; fdrv(0, 2) = ry; fdrv(0, 7) = oz; fdrv(0, 8) = -oy; fdrv(1, 0) = rz; fdrv(1, 2) = -rx; fdrv(1, 6) = -oz;
        fdrv(1, 8) = ox; fdrv(2, 0) = -ry; fdrv(2, 1) = rx; fdrv(2, 6) = oy; fdrv(2, 7) = -ox; fdrv(3, 0) = t1433; fdrv(3, 1) = t87+t144+t145+t1395; fdrv(3, 2) = t147+t149+t423+t1754; fdrv(3, 3) = -oy*t549-oz*t99; fdrv(3, 4) = t3*2.0+t1891-oy*t547;
        fdrv(3, 5) = t5*2.0+t1856-oz*t96; fdrv(3, 6) = -oz*t103+oy*(t30-ty); fdrv(3, 7) = t44+t701+oy*(t31-tx); fdrv(3, 8) = t46+t168-oz*t101; fdrv(4, 0) = t144+t145+t432+t1751; fdrv(4, 1) = t1430; fdrv(4, 2) = t88+t151+t152+t1392; fdrv(4, 3) = t4*2.0+t1859-ox*t97;
        fdrv(4, 4) = -ox*t95-oz*t552; fdrv(4, 5) = t8*2.0+t1887-oz*t550; fdrv(4, 6) = t45+t165-ox*t102; fdrv(4, 7) = -ox*t101+oz*(t29-tz); fdrv(4, 8) = t48+t697+oz*(t30-ty); fdrv(5, 0) = t94+t147+t149+t1391; fdrv(5, 1) = t151+t152+t433+t1742; fdrv(5, 2) = t1427;
        fdrv(5, 3) = t7*2.0+t1889-ox*t551; fdrv(5, 4) = t9*2.0+t1855-oy*t100; fdrv(5, 5) = -ox*t548-oy*t98; fdrv(5, 6) = t47+t695+ox*(t29-tz); fdrv(5, 7) = t49+t159-oy*t103; fdrv(5, 8) = -oy*t102+ox*(t31-tx); fdrv(6, 1) = t2083;
        fdrv(6, 2) = t212+t436+t1756+t1765+t1766; fdrv(6, 3) = t1851+t1880; fdrv(6, 4) = t1892-oy*t1636; fdrv(6, 5) = -t1890-oz*t1636; fdrv(6, 6) = t1738*-2.0; fdrv(6, 7) = t419-oz*t1862; fdrv(6, 8) = t211+oy*t1863; fdrv(7, 0) = t218+t425+t1753+t1767+t1768;
        fdrv(7, 2) = t2079; fdrv(7, 3) = -t1892+ox*t1635; fdrv(7, 4) = t1851+t1853; fdrv(7, 5) = t1888+oz*t1635; fdrv(7, 6) = t207+oz*t1861; fdrv(7, 7) = t1737*2.0; fdrv(7, 8) = t420-ox*t1863; fdrv(8, 0) = t2080; fdrv(8, 1) = t219+t426+t1744+t1763+t1764;
        fdrv(8, 3) = t1890-ox*t1634; fdrv(8, 4) = -t1888-oy*t1634; fdrv(8, 5) = t1853+t1880; fdrv(8, 6) = t412-oy*t1861; fdrv(8, 7) = t208+ox*t1862; fdrv(8, 8) = t1736*-2.0; fdrv(9, 0) = -t1898-t1912-t1918+t36*t143+t40*t143; fdrv(9, 1) = t2149; fdrv(9, 2) = t2139;
        fdrv(9, 3) = t337+t356+t367+t368+t487+t544+t723+t741; fdrv(9, 4) = t345+t360+t362+t446+t557+t592+t761+t776; fdrv(9, 5) = t351+t353+t372+t461+t561+t582+t778+t823; fdrv(9, 6) = tx*(t82+t84+t694+t697)*2.0; fdrv(9, 7) = ty*(t82+t84+t694+t697)*2.0;
        fdrv(9, 8) = tz*(t82+t84+t694+t697)*2.0; fdrv(10, 0) = t2147; fdrv(10, 1) = -t1897-t1908-t1919+t29*t145+t40*t148; fdrv(10, 2) = t2140; fdrv(10, 3) = t331+t358+t361+t459+t571+t594+t731+t780; fdrv(10, 4) = t336+t348+t352+t371+t442+t544+t744+t808;
        fdrv(10, 5) = t340+t343+t374+t507+t570+t593+t783+t844; fdrv(10, 6) = tx*(t80+t85+t695+t700)*2.0; fdrv(10, 7) = ty*(t80+t85+t695+t700)*2.0; fdrv(10, 8) = tz*(t80+t85+t695+t700)*2.0; fdrv(11, 0) = t2148; fdrv(11, 1) = t2138;
        fdrv(11, 2) = -t1896-t1909-t1915+t30*t149+t31*t152; fdrv(11, 3) = t335+t347+t349+t505+t583+t598+t762+t785; fdrv(11, 4) = t339+t341+t365+t531+t572+t600+t787+t824; fdrv(11, 5) = t333+t334+t355+t370+t442+t487+t811+t835;
        fdrv(11, 6) = tx*(t81+t83+t698+t701)*2.0; fdrv(11, 7) = ty*(t81+t83+t698+t701)*2.0; fdrv(11, 8) = tz*(t81+t83+t698+t701)*2.0; fdrv(12, 0) = t1840; fdrv(12, 1) = t22; fdrv(12, 2) = t25; fdrv(12, 3) = t6+t10; fdrv(12, 4) = t128; fdrv(12, 5) = t130;
        fdrv(12, 7) = t13+t134; fdrv(12, 8) = t16+t137; fdrv(13, 0) = t21; fdrv(13, 1) = t1839; fdrv(13, 2) = t27; fdrv(13, 3) = t129; fdrv(13, 4) = t2+t10; fdrv(13, 5) = t132; fdrv(13, 6) = t1405; fdrv(13, 8) = t18+t140; fdrv(14, 0) = t23; fdrv(14, 1) = t26;
        fdrv(14, 2) = t1838; fdrv(14, 3) = t131; fdrv(14, 4) = t133; fdrv(14, 5) = t2+t6; fdrv(14, 6) = t1406; fdrv(14, 7) = t1407; fdrv(15, 0) = t1940+t2027; fdrv(15, 1) = -t1758-t1927*tx; fdrv(15, 2) = t1755+t1977*tx;
        fdrv(15, 3) = ox*(t99*ty-t549*tz)-oy*(t1927+t99*tx)+oz*(t1977+t549*tx); fdrv(15, 4) = t69-t73*2.0+t202+t205+t253+t330+t358+t464+t568+t585+t587+t722+t7*tx; fdrv(15, 5) = t78*2.0-t198+t254+t332+t347+t399+t410+t509+t575+t576+t597+t740+t129*tx;
        fdrv(15, 6) = -tx*(t82+t84+t694+t697+t1407); fdrv(15, 7) = -oy*(t64+t179)+oz*(t126+t1435)+ox*(t37+tz*(t31-tx)); fdrv(15, 8) = -oy*(t127+t712)+ox*(t41+t101*ty)+oz*(t64+t156*tz); fdrv(16, 0) = t1752+t1979*ty; fdrv(16, 1) = t1942+t2025;
        fdrv(16, 2) = -t1749-t1926*ty; fdrv(16, 3) = t68*2.0-t205+t255+t342+t360+t405+t406+t450+t556+t584+t586+t736+t133*ty; fdrv(16, 4) = -oy*(t552*tx-t95*tz)+ox*(t1979+t552*ty)-oz*(t1926+t95*ty);
        fdrv(16, 5) = t70-t76*2.0+t191+t201+t257+t341+t359+t533+t563+t565+t599+t805+t3*ty; fdrv(16, 6) = -oz*(t125+t713)+ox*(t63+t155*tx)+oy*(t33+t102*tz); fdrv(16, 7) = ty*(t1406+t1955); fdrv(16, 8) = ox*(t127+t1436)-oz*(t63+t187)+oy*(t41+tx*(t30-ty));
        fdrv(17, 0) = -t1746-t1928*tz; fdrv(17, 1) = t1743+t1978*tz; fdrv(17, 2) = t1939+t2026; fdrv(17, 3) = t66*-2.0+t77+t193+t198+t258+t353+t369+t470+t558+t573+t574+t801+t8*tz;
        fdrv(17, 4) = t71*2.0-t191+t259+t343+t373+t404+t407+t515+t562+t564+t589+t831+t130*tz; fdrv(17, 5) = oz*(t98*tx-t548*ty)-ox*(t1928+t98*tz)+oy*(t1978+t548*tz); fdrv(17, 6) = -ox*(t62+t184)+oy*(t125+t1437)+oz*(t33+ty*(t29-tz));
        fdrv(17, 7) = -ox*(t126+t714)+oz*(t37+t103*tx)+oy*(t62+t157*ty); fdrv(17, 8) = -tz*(t81+t83+t698+t701+t1405); fdrv(18, 0) = t2087+t2089; fdrv(18, 1) = -t2081*tx; fdrv(18, 2) = -t2083*tx;
        fdrv(18, 3) = t256+t299+t333+t337+t348+t356+t367+t368-t468-t517+t723+t741+t802+t833; fdrv(18, 4) = t230+t342+t360+t362+t446+t557+t588+t591+t749+t769; fdrv(18, 5) = t239+t351+t353+t369+t461+t561+t577+t579+t770+t813;
        fdrv(18, 6) = -ox*(t393+t395)+t29*t45+t30*t47; fdrv(18, 7) = t377+t1997+t135*t1862; fdrv(18, 8) = t376+t1999+t138*t1863; fdrv(19, 0) = -t2079*ty; fdrv(19, 1) = t2085+t2088; fdrv(19, 2) = -t2082*ty;
        fdrv(19, 3) = t240+t330+t358+t361+t459+t571+t590+t591+t727+t771; fdrv(19, 4) = t225+t299+t334+t336+t348+t352+t367+t371-t447-t536+t735+t744+t806+t808; fdrv(19, 5) = t277+t340+t343+t373+t507+t566+t567+t593+t773+t837;
        fdrv(19, 6) = t377+t1995+t134*t1861; fdrv(19, 7) = -oy*(t394+t397)+t31*t49+t11*t126; fdrv(19, 8) = t375+t2000+t141*t1863; fdrv(20, 0) = -t2078*tz; fdrv(20, 1) = -t2080*tz; fdrv(20, 2) = t2084+t2086;
        fdrv(20, 3) = t278+t332+t347+t349+t505+t579+t581+t598+t754+t774; fdrv(20, 4) = t292+t339+t341+t359+t531+t566+t569+t600+t775+t816; fdrv(20, 5) = t225+t256+t333+t334+t352+t355+t368+t370-t462-t511+t720+t739+t811+t835;
        fdrv(20, 6) = t376+t1996+t137*t1861; fdrv(20, 7) = t375+t1998+t140*t1862; fdrv(20, 8) = -oz*(t396+t398)+t11*t127+t15*t127; fdrv(21, 0) = t2164+t2166; fdrv(21, 1) = t2172+t2139*tx; fdrv(21, 2) = -t2170-t2149*tx; fdrv(21, 3) = t2071; fdrv(21, 4) = t2243;
        fdrv(21, 5) = t2240; fdrv(21, 6) = t1949; fdrv(21, 7) = t1951; fdrv(21, 8) = t1952; fdrv(22, 0) = -t2171-t2140*ty; fdrv(22, 1) = t2142+t2188; fdrv(22, 2) = t2168+t2147*ty; fdrv(22, 3) = t2239; fdrv(22, 4) = t2077; fdrv(22, 5) = t2242; fdrv(22, 6) = t1965;
        fdrv(22, 7) = t1967; fdrv(22, 8) = t1968; fdrv(23, 0) = t2169+t2138*tz; fdrv(23, 1) = -t2167-t2148*tz; fdrv(23, 2) = t2161+t2165; fdrv(23, 3) = t2241; fdrv(23, 4) = t2238; fdrv(23, 5) = t2073; fdrv(23, 6) = t1944; fdrv(23, 7) = t1945; fdrv(23, 8) = t1947;
        fdrv(24, 0) = t1762; fdrv(24, 1) = t1886; fdrv(24, 2) = t436+t1765; fdrv(24, 3) = t1741+t8*tx*2.0-t9*tx*2.0; fdrv(24, 4) = t73+t401-oz*t1128; fdrv(24, 5) = t67+t409+oy*t1127; fdrv(24, 7) = t81+t83-t1398; fdrv(24, 8) = t411+t421+t1397;
        fdrv(25, 0) = t425+t1768; fdrv(25, 1) = t91+t429; fdrv(25, 2) = t1882; fdrv(25, 3) = t74+t400+oz*t1126; fdrv(25, 4) = -t1740-t5*ty*2.0+t7*ty*2.0; fdrv(25, 5) = t76+t402-ox*t1125; fdrv(25, 6) = t413+t417+t1398; fdrv(25, 8) = t82+t84-t1396; fdrv(26, 0) = t1883;
        fdrv(26, 1) = t426+t1764; fdrv(26, 2) = t1760; fdrv(26, 3) = t66+t408-oy*t1124; fdrv(26, 4) = t75+t403+ox*t1123; fdrv(26, 5) = t1739+t3*tz*2.0-t4*tz*2.0; fdrv(26, 6) = t80+t85-t1397; fdrv(26, 7) = t414+t418+t1396; fdrv(27, 0) = t1925+t1991+t2031;
        fdrv(27, 1) = t1844+t157*t1927+t440*t1977; fdrv(27, 2) = t1846+t157*t1977+t439*t1927; fdrv(27, 3) = t120+t121+t265+t276+t293+t334+t352+t468+t487+t517+t544+t559+t578+t654+t692+t720+t735+t781+t938+t972+t1112+t1202+t1231+t1301;
        fdrv(27, 4) = t117+t119+t231+t248+t300+t345+t363+t524+t556+t592+t636+t670+t721+t761+t776+t809+t936+t973+t1069+t1097+t1108+t1233+t1318+t1347;
        fdrv(27, 5) = t113+t115+t242+t296+t350+t372+t558+t582+t657+t684+t725+t737+t778+t782+t823+t843+t1070+t1095+t1145+t1181+t1200+t1303+t1316+t1349; fdrv(27, 6) = -ox*(t868+t1550)-oy*(t870+t440*(t30-ty))+oz*(t1864+t31*(t30-ty));
        fdrv(27, 7) = -ox*(t328+t1549+t1571)+oz*(t224+t1731+t1850)-oy*(t190+t703+t440*(t31-tx)); fdrv(27, 8) = t140*(t41+t155+t184+t188)+t137*(t32+t126+t127)+oz*(t33+t179+t189+t190+t329+t395); fdrv(28, 0) = t1842+t156*t1979+t441*t1926;
        fdrv(28, 1) = t1924+t1989+t2036; fdrv(28, 2) = t1845+t156*t1926+t439*t1979; fdrv(28, 3) = t116+t118+t238+t243+t331+t363+t568+t594+t615+t678+t731+t738+t760+t780+t810+t849+t889+t962+t1192+t1250+t1270+t1345+t1362+t1378;
        fdrv(28, 4) = t111+t114+t237+t260+t280+t333+t368+t442+t447+t536+t544+t560+t595+t616+t676+t739+t788+t802+t976+t1016+t1121+t1194+t1268+t1294;
        fdrv(28, 5) = t107+t109+t226+t244+t279+t338+t374+t523+t570+t589+t604+t607+t783+t803+t812+t844+t887+t963+t974+t1017+t1118+t1252+t1295+t1360; fdrv(28, 6) = t138*(t33+t157+t179+t183)+t135*(t36+t125+t127)+ox*(t37+t185+t186+t187+t328+t394);
        fdrv(28, 7) = -oy*(t865+t1557)+ox*(t1866+t30*(t29-tz))-oz*(t867+t439*(t29-tz)); fdrv(28, 8) = ox*(t224+t1732+t1849)-oy*(t327+t1556+t1572)-oz*(t185+t707+t439*(t30-ty)); fdrv(29, 0) = t1841+t155*t1928+t441*t1978;
        fdrv(29, 1) = t1843+t155*t1978+t440*t1928; fdrv(29, 2) = t1923+t1994+t2034; fdrv(29, 3) = t110+t112+t263+t275+t294+t335+t350+t454+t583+t597+t652+t667+t724+t762+t785+t804+t885+t915+t934+t1037+t1111+t1136+t1146+t1178;
        fdrv(29, 4) = t106+t108+t247+t291+t338+t365+t572+t599+t633+t688+t716+t759+t787+t807+t824+t832+t886+t932+t1134+t1143+t1148+t1230+t1235+t1314;
        fdrv(29, 5) = t104+t105+t229+t241+t264+t348+t367+t442+t462+t487+t511+t580+t596+t635+t683+t779+t806+t833+t916+t1035+t1105+t1180+t1237+t1312; fdrv(29, 6) = oy*(t224+t1730+t1848)-oz*(t329+t1554+t1570)-ox*(t182+t706+t441*(t29-tz));
        fdrv(29, 7) = t134*(t37+t156+t180+t187)+t141*(t40+t125+t126)+oy*(t41+t181+t182+t184+t327+t398); fdrv(29, 8) = -oz*(t869+t1552)-ox*(t866+t441*(t31-tx))+oy*(t1865+t29*(t31-tx)); fdrv(30, 0) = t2113+t2121; fdrv(30, 1) = t157*t2081-t440*t2083;
        fdrv(30, 2) = t2116+t439*t2081; fdrv(30, 3) = t261-t264*4.0+t266*4.0+t493+t651+t683+t777+t797-t922+t926+t955+t965+t1023+t1121+t1223+t1246+t1305+t573*ty;
        fdrv(30, 4) = t245+t282+t630+t634+t681+t755+t822+t849+t924+t1033+t1067+t1107+t1222+t1248+t1250+t1346+t1351+t1381; fdrv(30, 5) = t263+t451+t540-t606+t619+t682+t836+t885-t920+t954+t1024+t1031+t1068+t1116+t1296+t1353+t1379+t33*t128;
        fdrv(30, 6) = t1949; fdrv(30, 7) = t2045; fdrv(30, 8) = t2058; fdrv(31, 0) = t2103+t441*t2082; fdrv(31, 1) = t2105+t2122; fdrv(31, 2) = t156*t2082-t439*t2079;
        fdrv(31, 3) = t300+t477+t516-t666+t675+t679+t742+t913+t970-t986+t1015+t1069+t1078+t1109+t1156+t1179+t1239+t37*t132; fdrv(31, 4) = t261*4.0+t262-t265*4.0+t497+t618+t692+t784+t789+t914+t982-t984+t1014+t1023+t1105+t1185+t1286+t1305+t585*tz;
        fdrv(31, 5) = t246+t286+t609+t656+t691+t716+t750+t821+t971+t980+t1076+t1117+t1132+t1148+t1154+t1241+t1285+t1307; fdrv(31, 6) = t2045; fdrv(31, 7) = t1967; fdrv(31, 8) = t2043; fdrv(32, 0) = t155*t2078-t441*t2080; fdrv(32, 1) = t2104+t440*t2078;
        fdrv(32, 2) = t2106+t2120; fdrv(32, 3) = t232+t295+t614+t669+t685+t730+t782+t838+t874+t883+t1045+t1113+t1150+t1183+t1206+t1293+t1337+t1349;
        fdrv(32, 4) = t226+t476+t510+t605-t637+t689+t745+t872+t884+t963+t964-t1041+t1062+t1119+t1152+t1204+t1232+t41*t131; fdrv(32, 5) = t260*-4.0+t262*4.0+t266+t492+t638+t676+t786+t795+t914+t965-t1039+t1043+t1061+t1112+t1185+t1246+t1336+t565*tx;
        fdrv(32, 6) = t2058; fdrv(32, 7) = t2043; fdrv(32, 8) = t1947; fdrv(33, 0) = t2156+t2174+t2213; fdrv(33, 1) = t2198+t31*t2139-t440*t2149; fdrv(33, 2) = t2200+t31*t2149-t439*t2139; fdrv(33, 3) = t2288; fdrv(33, 4) = t2277; fdrv(33, 5) = t2279;
        fdrv(33, 6) = t2059; fdrv(33, 7) = t2061; fdrv(33, 8) = t2064; fdrv(34, 0) = t2196+t30*t2140-t441*t2147; fdrv(34, 1) = t2158+t2173+t2212; fdrv(34, 2) = t2201+t30*t2147-t439*t2140; fdrv(34, 3) = t2278; fdrv(34, 4) = t2287; fdrv(34, 5) = t2281;
        fdrv(34, 6) = t2060; fdrv(34, 7) = t2063; fdrv(34, 8) = t2066; fdrv(35, 0) = t2197+t29*t2138-t441*t2148; fdrv(35, 1) = t2199+t29*t2148-t440*t2138; fdrv(35, 2) = t2157+t2175+t2211; fdrv(35, 3) = t2280; fdrv(35, 4) = t2282; fdrv(35, 5) = t2286;
        fdrv(35, 6) = t2062; fdrv(35, 7) = t2065; fdrv(35, 8) = t2067; fdrv(36, 0) = t1840; fdrv(36, 1) = t21; fdrv(36, 2) = t23; fdrv(36, 4) = t4+t128; fdrv(36, 5) = t7+t130; fdrv(36, 6) = t15+t19; fdrv(36, 7) = t134; fdrv(36, 8) = t137; fdrv(37, 0) = t22;
        fdrv(37, 1) = t1839; fdrv(37, 2) = t26; fdrv(37, 3) = t3+t129; fdrv(37, 5) = t9+t132; fdrv(37, 6) = t135; fdrv(37, 7) = t11+t19; fdrv(37, 8) = t140; fdrv(38, 0) = t25; fdrv(38, 1) = t27; fdrv(38, 2) = t1838; fdrv(38, 3) = t5+t131; fdrv(38, 4) = t8+t133;
        fdrv(38, 6) = t138; fdrv(38, 7) = t141; fdrv(38, 8) = t11+t15; fdrv(39, 0) = t1942+t2026; fdrv(39, 1) = -t2048; fdrv(39, 2) = t2075; fdrv(39, 3) = -ox*(t551*ty-t97*tz)-oy*(t54+t1729*ty)+oz*(t52-t1729*tz);
        fdrv(39, 4) = -t73+t79-t199+t202+t253+t330+t358+t401+t464+t568+t585+t587+t722+t7*tx*2.0; fdrv(39, 5) = t67-t72+t78+t203+t254+t332+t347+t410+t509+t575+t576+t597+t740-t4*tx*2.0; fdrv(39, 6) = -t2039*tx;
        fdrv(39, 7) = oz*(t126+t1437)-t15*t103+ox*(t37+t101*tz); fdrv(39, 8) = -oy*(t127+t713)+ox*(t41+ty*(t31-tx))-t19*(t30-ty); fdrv(40, 0) = t2076; fdrv(40, 1) = t2136; fdrv(40, 2) = -t2046;
        fdrv(40, 3) = t68+t74-t79+t194+t255+t342+t360+t405+t450+t556+t584+t586+t736-t9*ty*2.0; fdrv(40, 4) = oy*(t100*tx-t547*tz)+ox*(t58-t1728*tx)-oz*(t53+t1728*tz); fdrv(40, 5) = t65-t76-t200+t201+t257+t341+t359+t402+t533+t563+t565+t599+t805+t3*ty*2.0;
        fdrv(40, 6) = -oz*(t125+t714)+oy*(t33+tz*(t30-ty))-t11*(t29-tz); fdrv(40, 7) = t2038*ty; fdrv(40, 8) = ox*(t127+t1435)-t19*t101+oy*(t41+t102*tx); fdrv(41, 0) = -t2047; fdrv(41, 1) = t2074; fdrv(41, 2) = t1940+t2025;
        fdrv(41, 3) = -t66+t72-t192+t193+t258+t353+t369+t408+t470+t558+t573+t574+t801+t8*tz*2.0; fdrv(41, 4) = -t65+t71+t75+t195+t259+t343+t373+t404+t515+t562+t564+t589+t831-t5*tz*2.0;
        fdrv(41, 5) = -oz*(t550*tx-t96*ty)-ox*(t59+t1727*tx)+oy*(t56-t1727*ty); fdrv(41, 6) = oy*(t125+t1436)-t11*t102+oz*(t33+t103*ty); fdrv(41, 7) = -ox*(t126+t712)+oz*(t37+tx*(t29-tz))-t15*(t31-tx); fdrv(41, 8) = -t2037*tz; fdrv(42, 0) = t2222;
        fdrv(42, 1) = -t2078*ty; fdrv(42, 2) = -t2079*tz; fdrv(42, 3) = t1906+t1907-ox*(t1634*ty+t1635*tz); fdrv(42, 4) = t118*-2.0+t230+t231+t345+t360+t362-t467+t557+t588+t749+t769+t771;
        fdrv(42, 5) = t112*-2.0+t239+t242+t351+t353+t372-t514+t561+t577+t770+t774+t813; fdrv(42, 6) = t11*t441*-2.0-t15*t1861-t19*t1861; fdrv(42, 7) = t377+t1997+t36*t45; fdrv(42, 8) = t376+t1999+t40*t47; fdrv(43, 0) = -t2080*tx; fdrv(43, 1) = t2221;
        fdrv(43, 2) = -t2081*tz; fdrv(43, 3) = t119*-2.0+t238+t240+t331+t358+t361-t449+t571+t590+t727+t769+t771; fdrv(43, 4) = t1905+t1907+oy*(t1634*tx-t1636*tz); fdrv(43, 5) = t106*-2.0+t277+t279+t340+t343+t374-t535+t567+t593+t773+t775+t837;
        fdrv(43, 6) = t377+t1995+t11*t62; fdrv(43, 7) = t15*t440*-2.0-t11*t1862-t19*t1862; fdrv(43, 8) = t375+t2000+t40*t49; fdrv(44, 0) = -t2082*tx; fdrv(44, 1) = -t2083*ty; fdrv(44, 2) = t2220;
        fdrv(44, 3) = t113*-2.0+t275+t278+t335+t347+t349-t466+t581+t598+t754+t770+t774; fdrv(44, 4) = t107*-2.0+t291+t292+t339+t341+t365-t513+t569+t600+t773+t775+t816; fdrv(44, 5) = t1905+t1906+oz*(t1635*tx+t1636*ty); fdrv(44, 6) = t376+t1996+t11*t63;
        fdrv(44, 7) = t375+t1998+t15*t64; fdrv(44, 8) = t19*t439*-2.0-t11*t1863-t15*t1863; fdrv(45, 0) = t2261; fdrv(45, 1) = t2255; fdrv(45, 2) = -t2256; fdrv(45, 3) = t2071; fdrv(45, 4) = t2243; fdrv(45, 5) = t2240; fdrv(45, 6) = t1949; fdrv(45, 7) = t1951;
        fdrv(45, 8) = t1952; fdrv(46, 0) = -t2260; fdrv(46, 1) = t2164+t2165; fdrv(46, 2) = t2254; fdrv(46, 3) = t2239; fdrv(46, 4) = t2077; fdrv(46, 5) = t2242; fdrv(46, 6) = t1965; fdrv(46, 7) = t1967; fdrv(46, 8) = t1968; fdrv(47, 0) = t2258; fdrv(47, 1) = -t2259;
        fdrv(47, 2) = t2253; fdrv(47, 3) = t2241; fdrv(47, 4) = t2238; fdrv(47, 5) = t2073; fdrv(47, 6) = t1944; fdrv(47, 7) = t1945; fdrv(47, 8) = t1947; fdrv(48, 1) = t87+t1395; fdrv(48, 2) = t423+t1754; fdrv(48, 3) = rx*t1407; fdrv(48, 4) = t73+t1891;
        fdrv(48, 5) = t409+t1856; fdrv(48, 6) = t1738; fdrv(48, 7) = t83+t701; fdrv(48, 8) = t168+t421; fdrv(49, 0) = t432+t1751; fdrv(49, 2) = t88+t1392; fdrv(49, 3) = t400+t1859; fdrv(49, 4) = -ry*t1406; fdrv(49, 5) = t76+t1887; fdrv(49, 6) = t165+t413;
        fdrv(49, 7) = t1873; fdrv(49, 8) = t84+t697; fdrv(50, 0) = t94+t1391; fdrv(50, 1) = t433+t1742; fdrv(50, 3) = t66+t1889; fdrv(50, 4) = t403+t1855; fdrv(50, 5) = rz*t1405; fdrv(50, 6) = t80+t695; fdrv(50, 7) = t159+t414; fdrv(50, 8) = t1736;
        fdrv(51, 0) = -t2075*ty-t2048*tz; fdrv(51, 1) = -tz*(t1942+t2026)+t2075*tx; fdrv(51, 2) = ty*(t1942+t2026)+t2048*tx; fdrv(51, 3) = t265+t337+t356+t367+t368+t487+t544+t654+t692+t723+t741+t781+t938+t972+t1112+t1202+t1231+t1301;
        fdrv(51, 4) = t248+t300+t345+t360+t362+t446+t524+t557+t592+t636+t670+t761+t776+t809+t936+t973+t1069+t1097+t1108+t1233+t1318+t1347;
        fdrv(51, 5) = t296+t351+t353+t372+t461+t561+t582+t657+t684+t725+t778+t782+t823+t843+t1070+t1095+t1145+t1181+t1200+t1303+t1316+t1349; fdrv(51, 6) = tx*(t82*2.0+t84*2.0+t171+t311+t314-t381-t383+t30*t135+t15*t157+t40*t140);
        fdrv(51, 7) = ty*(t82*2.0+t84*2.0+t171+t311+t314-t381-t383+t30*t135+t15*t157+t40*t140); fdrv(51, 8) = tz*(t82*2.0+t84*2.0+t171+t311+t314-t381-t383+t30*t135+t15*t157+t40*t140); fdrv(52, 0) = t2046*ty+t2136*tz; fdrv(52, 1) = -t2046*tx-t2076*tz;
        fdrv(52, 2) = -t2136*tx+t2076*ty; fdrv(52, 3) = t243+t331+t358+t361+t459+t571+t594+t615+t678+t731+t760+t780+t810+t849+t889+t962+t1192+t1250+t1270+t1345+t1362+t1378;
        fdrv(52, 4) = t260+t336+t348+t352+t371+t442+t544+t616+t676+t744+t788+t808+t976+t1016+t1121+t1194+t1268+t1294; fdrv(52, 5) = t226+t244+t340+t343+t374+t507+t523+t570+t593+t604+t607+t783+t812+t844+t887+t963+t974+t1017+t1118+t1252+t1295+t1360;
        fdrv(52, 6) = tx*(t160*-2.0+t166-t168*2.0+t206+t211+t304+t307+t32*t138+t36*t138+t40*t138); fdrv(52, 7) = ty*(t160*-2.0+t166-t168*2.0+t206+t211+t304+t307+t32*t138+t36*t138+t40*t138);
        fdrv(52, 8) = tz*(t160*-2.0+t166-t168*2.0+t206+t211+t304+t307+t32*t138+t36*t138+t40*t138); fdrv(53, 0) = -ty*(t1940+t2025)+t2074*tz; fdrv(53, 1) = tx*(t1940+t2025)+t2047*tz; fdrv(53, 2) = -t2074*tx-t2047*ty;
        fdrv(53, 3) = t263+t294+t335+t347+t349+t454+t505+t583+t598+t652+t667+t724+t762+t785+t885+t915+t934+t1037+t1111+t1136+t1146+t1178;
        fdrv(53, 4) = t247+t339+t341+t365+t531+t572+t600+t633+t688+t716+t759+t787+t807+t824+t886+t932+t1134+t1143+t1148+t1230+t1235+t1314; fdrv(53, 5) = t264+t333+t334+t355+t370+t442+t487+t635+t683+t779+t811+t835+t916+t1035+t1105+t1180+t1237+t1312;
        fdrv(53, 6) = tx*(t162+t207+t209+t305+t308+t11*t155+t36*t134+t40*t134+t139*tx+t142*ty); fdrv(53, 7) = ty*(t162+t207+t209+t305+t308+t11*t155+t36*t134+t40*t134+t139*tx+t142*ty);
        fdrv(53, 8) = tz*(t162+t207+t209+t305+t308+t11*t155+t36*t134+t40*t134+t139*tx+t142*ty); fdrv(54, 0) = t2105+t2120; fdrv(54, 1) = t156*t2079-t2222*tz; fdrv(54, 2) = t2099+t2222*ty; fdrv(54, 3) = t2068;
        fdrv(54, 4) = t282+t634+t679+t746+t822+t849+t1033+t1067+t1208+t1234+t1255+t1346+t1351+t1381; fdrv(54, 5) = t234+t263+t540+t608+t619+t836+t928+t1020+t1031+t1068+t1113+t1296+t1353+t1379; fdrv(54, 6) = t1934; fdrv(54, 7) = t1936; fdrv(54, 8) = t1937;
        fdrv(55, 0) = t2111+t2221*tz; fdrv(55, 1) = t2106+t2121; fdrv(55, 2) = t155*t2080-t2221*tx; fdrv(55, 3) = t285+t300+t477+t668+t675+t742+t918+t970+t992+t1078+t1107+t1156+t1179+t1239; fdrv(55, 4) = -t2069;
        fdrv(55, 5) = t286+t656+t689+t716+t743+t821+t971+t1076+t1132+t1144+t1154+t1241+t1257+t1298; fdrv(55, 6) = -t1931; fdrv(55, 7) = t1966; fdrv(55, 8) = -t1935; fdrv(56, 0) = t157*t2083-t2220*ty; fdrv(56, 1) = t2112+t2220*tx; fdrv(56, 2) = t2113+t2122;
        fdrv(56, 3) = t232+t614+t682+t730+t782+t834+t874+t883+t1150+t1189+t1206+t1293+t1321+t1344; fdrv(56, 4) = t226+t281+t476+t605+t631+t745+t872+t884+t960+t1029+t1117+t1152+t1204+t1232; fdrv(56, 5) = t2070; fdrv(56, 6) = t1929; fdrv(56, 7) = t1930;
        fdrv(56, 8) = t1932; fdrv(57, 0) = t2256*ty+t2255*tz; fdrv(57, 1) = -t2256*tx-t2261*tz; fdrv(57, 2) = -t2255*tx+t2261*ty; fdrv(57, 3) = t2288; fdrv(57, 4) = t2277; fdrv(57, 5) = t2279; fdrv(57, 6) = t2059; fdrv(57, 7) = t2061; fdrv(57, 8) = t2064;
        fdrv(58, 0) = -t2254*ty-t2257*tz; fdrv(58, 1) = t2254*tx+t2260*tz; fdrv(58, 2) = t2257*tx-t2260*ty; fdrv(58, 3) = t2278; fdrv(58, 4) = t2287; fdrv(58, 5) = t2281; fdrv(58, 6) = t2060; fdrv(58, 7) = t2063; fdrv(58, 8) = t2066; fdrv(59, 0) = -t2253*ty-t2259*tz;
        fdrv(59, 1) = t2253*tx-t2258*tz; fdrv(59, 2) = t2259*tx+t2258*ty; fdrv(59, 3) = t2280; fdrv(59, 4) = t2282; fdrv(59, 5) = t2286; fdrv(59, 6) = t2062; fdrv(59, 7) = t2065; fdrv(59, 8) = t2067; fdrv(60, 0) = t315+t316+t1898;
        fdrv(60, 1) = t318+t144*t440+t155*t711; fdrv(60, 2) = t319+t147*t439+t156*t711; fdrv(60, 3) = t2*t441-t15*t1729-t19*t1729; fdrv(60, 4) = t253+t330+t358+t464+t568+t585+t587+t722; fdrv(60, 5) = t254+t332+t347+t509+t575+t576+t597+t740;
        fdrv(60, 6) = -tx*(t82+t84+t694+t697); fdrv(60, 7) = t603+t854+t12*t441; fdrv(60, 8) = t602+t40*t138+t14*t441; fdrv(61, 0) = t320+t145*t441+t155*t710; fdrv(61, 1) = t317+t324+t1897; fdrv(61, 2) = t321+t151*t439+t157*t710;
        fdrv(61, 3) = t255+t342+t360+t450+t556+t584+t586+t736; fdrv(61, 4) = t6*t440-t11*t1728-t19*t1728; fdrv(61, 5) = t257+t341+t359+t533+t563+t565+t599+t805; fdrv(61, 6) = t603+t11*t155+t13*t440; fdrv(61, 7) = -ty*(t80+t85+t695+t700);
        fdrv(61, 8) = t601+t858+t17*t440; fdrv(62, 0) = t322+t149*t441+t156*t709; fdrv(62, 1) = t323+t152*t440+t157*t709; fdrv(62, 2) = t325+t326+t1896; fdrv(62, 3) = t258+t353+t369+t470+t558+t573+t574+t801;
        fdrv(62, 4) = t259+t343+t373+t515+t562+t564+t589+t831; fdrv(62, 5) = t10*t439-t11*t1727-t15*t1727; fdrv(62, 6) = t602+t853+t16*t439; fdrv(62, 7) = t601+t15*t157+t18*t439; fdrv(62, 8) = -tz*(t81+t83+t698+t701);
        fdrv(63, 0) = -t441*(t1942+t2026)+t30*t2075+t155*t2048; fdrv(63, 1) = -t155*(t1942+t2026)+t31*t2075+t440*t2048; fdrv(63, 2) = -t156*(t1942+t2026)+t157*t2048-t439*t2075; fdrv(63, 3) = t2291; fdrv(63, 4) = t2296; fdrv(63, 5) = t2294; fdrv(63, 6) = t2093;
        fdrv(63, 7) = t2096; fdrv(63, 8) = t2098; fdrv(64, 0) = t156*t2046-t155*t2136-t441*t2076; fdrv(64, 1) = t29*t2076+t157*t2046-t440*t2136; fdrv(64, 2) = t30*t2076-t157*t2136+t439*t2046; fdrv(64, 3) = t2292; fdrv(64, 4) = t2290; fdrv(64, 5) = t2297;
        fdrv(64, 6) = t2117; fdrv(64, 7) = t2118; fdrv(64, 8) = t2119; fdrv(65, 0) = -t156*(t1940+t2025)+t29*t2074+t441*t2047; fdrv(65, 1) = -t157*(t1940+t2025)+t155*t2047-t440*t2074; fdrv(65, 2) = -t439*(t1940+t2025)+t31*t2074+t156*t2047; fdrv(65, 3) = t2295;
        fdrv(65, 4) = t2293; fdrv(65, 5) = t2289; fdrv(65, 6) = t2090; fdrv(65, 7) = t2092; fdrv(65, 8) = t2095; fdrv(66, 0) = -t441*t2222+t155*t2078*ty+t156*t2079*tz; fdrv(66, 1) = t29*t2222+t440*t2078*ty+t157*t2079*tz;
        fdrv(66, 2) = t30*t2222+t2120*ty+t439*t2079*tz; fdrv(66, 3) = t2285; fdrv(66, 4) = t2271; fdrv(66, 5) = t2273; fdrv(66, 6) = t2049; fdrv(66, 7) = t2051; fdrv(66, 8) = t2054; fdrv(67, 0) = t29*t2221+t2121*tz+t441*t2080*tx;
        fdrv(67, 1) = -t440*t2221+t155*t2080*tx+t157*t2081*tz; fdrv(67, 2) = t31*t2221+t156*t2080*tx+t439*t2081*tz; fdrv(67, 3) = t2272; fdrv(67, 4) = t2284; fdrv(67, 5) = t2275; fdrv(67, 6) = t2050; fdrv(67, 7) = t2053; fdrv(67, 8) = t2056;
        fdrv(68, 0) = t30*t2220+t2123*ty+t441*t2082*tx; fdrv(68, 1) = t31*t2220+t2122*tx+t440*t2083*ty; fdrv(68, 2) = -t439*t2220+t156*t2082*tx+t157*t2083*ty; fdrv(68, 3) = t2274; fdrv(68, 4) = t2276; fdrv(68, 5) = t2283; fdrv(68, 6) = t2052; fdrv(68, 7) = t2055;
        fdrv(68, 8) = t2057; fdrv(69, 0) = t29*t2255+t156*t2256-t441*t2261; fdrv(69, 1) = t29*t2261+t157*t2256-t440*t2255; fdrv(69, 2) = t31*t2255+t30*t2261+t439*t2256; fdrv(69, 3) = t2127; fdrv(69, 4) = t2249; fdrv(69, 5) = t2252; fdrv(69, 6) = t2004;
        fdrv(69, 7) = t2005; fdrv(69, 8) = t2006; fdrv(70, 0) = t30*t2254+t155*t2257+t441*t2260; fdrv(70, 1) = t31*t2254+t155*t2260+t440*t2257; fdrv(70, 2) = t157*t2257+t156*t2260-t439*t2254; fdrv(70, 3) = t2251; fdrv(70, 4) = t2125; fdrv(70, 5) = t2246;
        fdrv(70, 6) = t1982; fdrv(70, 7) = t1984; fdrv(70, 8) = t1986; fdrv(71, 0) = t30*t2253+t155*t2259-t441*t2258; fdrv(71, 1) = t31*t2253+t29*t2258+t440*t2259; fdrv(71, 2) = t30*t2258+t157*t2259-t439*t2253; fdrv(71, 3) = t2245; fdrv(71, 4) = t2250;
        fdrv(71, 5) = t2128; fdrv(71, 6) = t2001; fdrv(71, 7) = t2002; fdrv(71, 8) = t2003; fdrv(72, 0) = t91+t430; fdrv(72, 1) = t1885; fdrv(72, 2) = t423+t1766; fdrv(72, 3) = -t1741+t131*ty+t4*tz; fdrv(72, 4) = t69+t199-oz*t1123; fdrv(72, 5) = -t203+t399+oy*t1125;
        fdrv(72, 6) = t1738; fdrv(72, 7) = t413-oz*t441; fdrv(72, 8) = t80+oy*t441; fdrv(73, 0) = t432+t1767; fdrv(73, 1) = t1761; fdrv(73, 2) = t1881; fdrv(73, 3) = -t194+t406+oz*t1124; fdrv(73, 4) = t1740+t9*tx+t128*tz; fdrv(73, 5) = t70+t200-ox*t1127;
        fdrv(73, 6) = t83+oz*t440; fdrv(73, 7) = t1873; fdrv(73, 8) = t414-ox*t440; fdrv(74, 0) = t1884; fdrv(74, 1) = t433+t1763; fdrv(74, 2) = t90+t429; fdrv(74, 3) = t77+t192-oy*t1126; fdrv(74, 4) = -t195+t407+ox*t1128; fdrv(74, 5) = -t1739+t132*tx+t5*ty;
        fdrv(74, 6) = t421-oy*t439; fdrv(74, 7) = t84+ox*t439; fdrv(74, 8) = t1736; fdrv(75, 0) = t1925+t1989+t2034; fdrv(75, 1) = t2217; fdrv(75, 2) = t2215;
        fdrv(75, 3) = t120+t121+t256+t265+t299+t337+t356+t559+t578+t654+t692+t720+t735+t781+t938+t972+t1112+t1202+t1231+t1301;
        fdrv(75, 4) = t117+t119+t248+t255+t300+t342+t361+t446+t467+t524+t556+t592+t636+t670+t721+t761+t776+t809+t936+t973+t1069+t1097+t1108+t1233+t1318+t1347;
        fdrv(75, 5) = t113+t115+t258+t296+t349+t369+t461+t514+t558+t582+t657+t684+t725+t737+t778+t782+t823+t843+t1070+t1095+t1145+t1181+t1200+t1303+t1316+t1349; fdrv(75, 6) = -ox*(t866+t1554)+oz*(t182+t396+t103*t441)-oy*(t186+t704+t441*(t30-ty));
        fdrv(75, 7) = -oy*(t868+t441*(t31-tx))-ox*(t865+t1571+t31*t155)+oz*(t224+t1848+t36*t441); fdrv(75, 8) = oz*(t1866+t29*(t30-ty))-oy*(t438+t1732+t30*t102)-ox*(t327+t1552+t441*tz*2.0); fdrv(76, 0) = t2219; fdrv(76, 1) = t1924+t1994+t2031;
        fdrv(76, 2) = t2214; fdrv(76, 3) = t116+t118+t243+t253+t330+t362+t449+t459+t568+t594+t615+t678+t731+t738+t760+t780+t810+t849+t889+t962+t1192+t1250+t1270+t1345+t1362+t1378;
        fdrv(76, 4) = t111+t114+t225+t260+t299+t336+t371+t560+t595+t616+t676+t739+t788+t802+t976+t1016+t1121+t1194+t1268+t1294;
        fdrv(76, 5) = t107+t109+t226+t244+t259+t339+t373+t507+t523+t535+t570+t589+t604+t607+t783+t803+t812+t844+t887+t963+t974+t1017+t1118+t1252+t1295+t1360; fdrv(76, 6) = ox*(t1865+t31*(t29-tz))-oz*(t438+t1730+t29*t103)-oy*(t329+t1550+t440*tx*2.0);
        fdrv(76, 7) = -oy*(t870+t1549)+ox*(t190+t393+t101*t440)-oz*(t181+t708+t440*(t29-tz)); fdrv(76, 8) = -oz*(t865+t440*(t30-ty))+ox*(t224+t1850+t40*t440)-oy*(t869+t1572+t40*t155); fdrv(77, 0) = t2218; fdrv(77, 1) = t2216; fdrv(77, 2) = t1923+t1991+t2036;
        fdrv(77, 3) = t110+t112+t254+t263+t294+t332+t351+t454+t466+t505+t583+t597+t652+t667+t724+t762+t785+t804+t885+t915+t934+t1037+t1111+t1136+t1146+t1178;
        fdrv(77, 4) = t106+t108+t247+t257+t340+t359+t513+t531+t572+t599+t633+t688+t716+t759+t787+t807+t824+t832+t886+t932+t1134+t1143+t1148+t1230+t1235+t1314;
        fdrv(77, 5) = t104+t105+t225+t256+t264+t355+t370+t580+t596+t635+t683+t779+t806+t833+t916+t1035+t1105+t1180+t1237+t1312; fdrv(77, 6) = -ox*(t869+t439*(t29-tz))+oy*(t224+t1849+t32*t439)-oz*(t868+t1570+t30*t155);
        fdrv(77, 7) = oy*(t1864+t30*(t31-tx))-ox*(t438+t1731+t31*t101)-oz*(t328+t1557+t439*ty*2.0); fdrv(77, 8) = -oz*(t867+t1556)+oy*(t185+t397+t102*t439)-ox*(t189+t705+t439*(t31-tx)); fdrv(78, 0) = t2225; fdrv(78, 1) = -t2134+t156*t2078;
        fdrv(78, 2) = t2227; fdrv(78, 3) = t260+t262+t629+t632+t686+t779+t786+t922-t926+t930-t938+t1019+t1023+t1122+t1246+t75*t157+t581*ty+t584*tz;
        fdrv(78, 4) = t283*-4.0+t512-t630+t678+t679+t746+t822+t850-t924+t967+t1064+t1208+t1234+t1248+t1255+t1355+t1387+t33*t130; fdrv(78, 5) = t233+t234+t295*4.0+t494+t540+t606+t608+t840+t917+t920+t928+t1020+t1024+t1063+t1111+t1113+t1311+t1386;
        fdrv(78, 6) = t1934; fdrv(78, 7) = t1573*(t81+t172+t209+t701); fdrv(78, 8) = -t1573*(t80+t164+t211+t700); fdrv(79, 0) = t2231; fdrv(79, 1) = t2113+t2120; fdrv(79, 2) = -t2130+t155*t2081;
        fdrv(79, 3) = t245*4.0+t283+t285+t477+t545+t666+t668+t747+t913+t918+t986+t992+t1093+t1094+t1107+t1108+t1173+t1182; fdrv(79, 4) = t264+t266+t653+t655+t693+t777+t781+t914+t919-t976-t982+t984+t990+t1106+t1305+t67*t156+t564*tx+t588*tz;
        fdrv(79, 5) = t284*-4.0+t518-t609+t688+t689+t717+t743+t821-t980+t1025+t1092+t1133+t1144+t1172+t1257+t1298+t1307+t37*t129; fdrv(79, 6) = -t1573*(t83+t172+t207+t698); fdrv(79, 7) = t1966; fdrv(79, 8) = t1573*(t82+t158+t210+t697);
        fdrv(80, 0) = -t2131+t157*t2082; fdrv(80, 1) = t2228; fdrv(80, 2) = t2223; fdrv(80, 3) = t233*-4.0+t448-t669+t682+t684+t730+t793+t834+t881+t909-t1045+t1183+t1189+t1221+t1300+t1321+t1344+t41*t133;
        fdrv(80, 4) = t246*4.0+t281+t284+t443+t476+t631+t637+t752+t880+t882+t960+t964+t1029+t1041+t1117+t1118+t1220+t1245; fdrv(80, 5) = t261+t265+t617+t620+t677+t784+t788+t961+t965+t1027-t1035+t1039-t1043+t1115+t1185+t74*t155+t567*tx+t576*ty;
        fdrv(80, 6) = t1573*(t85+t164+t206+t695); fdrv(80, 7) = -t1573*(t84+t158+t208+t694); fdrv(80, 8) = t1932; fdrv(81, 0) = t2266; fdrv(81, 1) = t2155+t2196-t2210; fdrv(81, 2) = t2154+t2197-t2204; fdrv(81, 3) = t2288; fdrv(81, 4) = t2277; fdrv(81, 5) = t2279;
        fdrv(81, 6) = t2059; fdrv(81, 7) = t2061; fdrv(81, 8) = t2064; fdrv(82, 0) = t2178+t2198-t2209; fdrv(82, 1) = t2263; fdrv(82, 2) = t2153+t2199-t2203; fdrv(82, 3) = t2278; fdrv(82, 4) = t2287; fdrv(82, 5) = t2281; fdrv(82, 6) = t2060; fdrv(82, 7) = t2063;
        fdrv(82, 8) = t2066; fdrv(83, 0) = t2177+t2200-t2208; fdrv(83, 1) = t2176+t2201-t2202; fdrv(83, 2) = t2262; fdrv(83, 3) = t2280; fdrv(83, 4) = t2282; fdrv(83, 5) = t2286; fdrv(83, 6) = t2062; fdrv(83, 7) = t2065; fdrv(83, 8) = t2067; fdrv(84, 0) = t1912+t1918;
        fdrv(84, 1) = t1872-t1910; fdrv(84, 2) = -t1870-t1913; fdrv(84, 3) = -t120-t121+t229+t237+t560+t580+t772+t848; fdrv(84, 4) = t587+ox*(t222+t1883+t1123*ty)-oy*(t93+t1123*tx); fdrv(84, 5) = t575+ox*(t218+t1885+t1125*tz)-oz*(t88+t1125*tx);
        fdrv(84, 6) = ox*(t179+t183)+t29*t135+t30*t138; fdrv(84, 7) = -ty*(t82+t84+t694+t697); fdrv(84, 8) = -tz*(t82+t84+t694+t697); fdrv(85, 0) = t1871-t1911; fdrv(85, 1) = t1908+t1919; fdrv(85, 2) = -t1868-t1916;
        fdrv(85, 3) = t586+oy*(t219+t1881+t1124*tx)-ox*(t94+t1124*ty); fdrv(85, 4) = -t111-t114+t241+t276+t559+t596+t715+t848; fdrv(85, 5) = t563+oy*(t214+t1886+t1127*tz)-oz*(t86+t1127*ty); fdrv(85, 6) = -tx*(t80+t85+t695+t700);
        fdrv(85, 7) = oy*(t180+t187)+t31*t141+t11*t379; fdrv(85, 8) = -tz*(t80+t85+t695+t700); fdrv(86, 0) = t1869-t1914; fdrv(86, 1) = -t1867-t1917; fdrv(86, 2) = t1909+t1915; fdrv(86, 3) = t574+oz*(t215+t1882+t1126*tx)-ox*(t92+t1126*tz);
        fdrv(86, 4) = t562+oz*(t212+t1884+t1128*ty)-oy*(t87+t1128*tz); fdrv(86, 5) = -t104-t105+t280+t293+t578+t595+t715+t772; fdrv(86, 6) = -tx*(t81+t83+t698+t701); fdrv(86, 7) = -ty*(t81+t83+t698+t701); fdrv(86, 8) = oz*(t184+t188)+t11*t380+t15*t380;
        fdrv(87, 0) = -t2215*ty+t2217*tz; fdrv(87, 1) = t2215*tx+t2207*tz; fdrv(87, 2) = -t2217*tx-t2207*ty; fdrv(87, 3) = t2291; fdrv(87, 4) = t2296; fdrv(87, 5) = t2294; fdrv(87, 6) = t2093; fdrv(87, 7) = t2096; fdrv(87, 8) = t2098; fdrv(88, 0) = -t2214*ty-t2206*tz;
        fdrv(88, 1) = t2214*tx-t2219*tz; fdrv(88, 2) = t2206*tx+t2219*ty; fdrv(88, 3) = t2292; fdrv(88, 4) = t2290; fdrv(88, 5) = t2297; fdrv(88, 6) = t2117; fdrv(88, 7) = t2118; fdrv(88, 8) = t2119; fdrv(89, 0) = t2205*ty+t2216*tz; fdrv(89, 1) = -t2205*tx-t2218*tz;
        fdrv(89, 2) = -t2216*tx+t2218*ty; fdrv(89, 3) = t2295; fdrv(89, 4) = t2293; fdrv(89, 5) = t2289; fdrv(89, 6) = t2090; fdrv(89, 7) = t2092; fdrv(89, 8) = t2095; fdrv(90, 0) = -t2227*ty-t2229*tz; fdrv(90, 1) = t2227*tx-t2225*tz; fdrv(90, 2) = t2229*tx+t2225*ty;
        fdrv(90, 3) = t2285; fdrv(90, 4) = t2271; fdrv(90, 5) = t2273; fdrv(90, 6) = t2049; fdrv(90, 7) = t2051; fdrv(90, 8) = t2054; fdrv(91, 0) = t2226*ty-t2224*tz; fdrv(91, 1) = -t2226*tx-t2231*tz; fdrv(91, 2) = t2224*tx+t2231*ty; fdrv(91, 3) = t2272;
        fdrv(91, 4) = t2284; fdrv(91, 5) = t2275; fdrv(91, 6) = t2050; fdrv(91, 7) = t2053; fdrv(91, 8) = t2056; fdrv(92, 0) = -t2223*ty+t2228*tz; fdrv(92, 1) = t2223*tx+t2230*tz; fdrv(92, 2) = -t2228*tx-t2230*ty; fdrv(92, 3) = t2274; fdrv(92, 4) = t2276;
        fdrv(92, 5) = t2283; fdrv(92, 6) = t2052; fdrv(92, 7) = t2055; fdrv(92, 8) = t2057; fdrv(93, 0) = t2265*ty-t2268*tz; fdrv(93, 1) = -t2265*tx-t2266*tz; fdrv(93, 2) = t2268*tx+t2266*ty; fdrv(93, 3) = t2127; fdrv(93, 4) = t2249; fdrv(93, 5) = t2252;
        fdrv(93, 6) = t2004; fdrv(93, 7) = t2005; fdrv(93, 8) = t2006; fdrv(94, 0) = t2264*ty+t2263*tz; fdrv(94, 1) = -t2264*tx+t2270*tz; fdrv(94, 2) = -t2263*tx-t2270*ty; fdrv(94, 3) = t2251; fdrv(94, 4) = t2125; fdrv(94, 5) = t2246; fdrv(94, 6) = t1982;
        fdrv(94, 7) = t1984; fdrv(94, 8) = t1986; fdrv(95, 0) = -t2262*ty-t2267*tz; fdrv(95, 1) = t2262*tx+t2269*tz; fdrv(95, 2) = t2267*tx-t2269*ty; fdrv(95, 3) = t2245; fdrv(95, 4) = t2250; fdrv(95, 5) = t2128; fdrv(95, 6) = t2001; fdrv(95, 7) = t2002;
        fdrv(95, 8) = t2003; fdrv(96, 0) = t29*t1885+t156*t1883+t441*t1762; fdrv(96, 1) = t155*t1762+t157*t1883-t440*t1885; fdrv(96, 2) = t31*t1885+t156*t1762+t439*t1883; fdrv(96, 3) = -t2068;
        fdrv(96, 4) = t248+t300+t524+t636+t670+t809+t936+t973+t1069+t1097+t1108+t1233+t1318+t1347; fdrv(96, 5) = t296+t657+t684+t725+t782+t843+t1070+t1095+t1145+t1181+t1200+t1303+t1316+t1349; fdrv(96, 6) = -t1934; fdrv(96, 7) = -t1936; fdrv(96, 8) = -t1937;
        fdrv(97, 0) = t29*t1761+t30*t1881+t441*t1886; fdrv(97, 1) = t31*t1881+t155*t1886-t440*t1761; fdrv(97, 2) = t31*t1761+t156*t1886-t439*t1881; fdrv(97, 3) = t243+t615+t678+t760+t810+t849+t889+t962+t1192+t1250+t1270+t1345+t1362+t1378; fdrv(97, 4) = t2069;
        fdrv(97, 5) = t226+t244+t523+t604+t607+t812+t887+t963+t974+t1017+t1118+t1252+t1295+t1360; fdrv(97, 6) = t1931; fdrv(97, 7) = t1933; fdrv(97, 8) = t1935; fdrv(98, 0) = t156*t1760+t155*t1882-t441*t1884; fdrv(98, 1) = t29*t1884+t157*t1760+t440*t1882;
        fdrv(98, 2) = t30*t1884+t157*t1882+t439*t1760; fdrv(98, 3) = t263+t294+t454+t652+t667+t724+t885+t915+t934+t1037+t1111+t1136+t1146+t1178; fdrv(98, 4) = t247+t633+t688+t716+t759+t807+t886+t932+t1134+t1143+t1148+t1230+t1235+t1314; fdrv(98, 5) = -t2070;
        fdrv(98, 6) = -t1929; fdrv(98, 7) = -t1930; fdrv(98, 8) = -t1932; fdrv(99, 0) = t30*t2215+t29*t2217+t441*t2207; fdrv(99, 1) = t31*t2215+t155*t2207-t440*t2217; fdrv(99, 2) = t31*t2217+t156*t2207-t439*t2215;
        fdrv(99, 3) = t1495+t1496+t1497-t1534-t1535-t1536+t2288+t4*t43+t33*t108*5.0-t33*t109*5.0+t37*t112-t37*t115*4.0+t39*t131-t109*t179*6.0+t108*t183*6.0-t115*t180*4.0+t113*t187*4.0-t75*t329*6.0-t77*t328*4.0+t41*t366+t112*t397+t184*t366+t41*t586+t118*t708+t1585*tz;
        fdrv(99, 4) = -t302-t1735+t2277-t9*t39*6.0+t6*t43-t37*t109*5.0-t41*t114+t35*t131+t30*t247-t109*t180*6.0+t31*t263*5.0-t115*t179*8.0+t113*t183*4.0-t121*t188*8.0-t31*t296*5.0+t41*t337+t70*t328*4.0-t75*t328*6.0-t77*t329*4.0+t41*t370+t156*t286+t184*t337+t33*t581+t41*t596+t63*t633+t184*t580+t31*t1063+t31*t1081+(t29*t29)*t73*6.0-(t30*t30)*t79*2.0-(t31*t31)*t79*6.0;
        fdrv(99, 5) = t274+t1734+t2279+t4*t35-t10*t39+t8*t43*6.0+t37*t105+t41*t108*5.0-t41*t109*4.0-t37*t121*4.0+t29*t247+t105*t187*6.0+t108*t184*6.0-t109*t184*4.0-t109*t188*4.0+t117*t183*8.0+t120*t187*8.0+t31*t282*5.0+t63*t264-t31*t300*5.0+t64*t283-t79*t329*6.0+t155*t286+t105*t394-t29*t656*2.0-t31*t668*4.0-t31*t675*6.0+t29*t690+t31*t1064-(t29*t29)*t77*4.0+t29*t62*t72;
        fdrv(99, 6) = -t1573*tx*(t82*2.0+t84*2.0+t171+t311+t314-t381-t383+t30*t135+t15*t157+t40*t140); fdrv(99, 7) = -t1573*ty*(t82*2.0+t84*2.0+t171+t311+t314-t381-t383+t30*t135+t15*t157+t40*t140);
        fdrv(99, 8) = -t1573*tz*(t82*2.0+t84*2.0+t171+t311+t314-t381-t383+t30*t135+t15*t157+t40*t140); fdrv(100, 0) = t30*t2214+t155*t2206-t441*t2219; fdrv(100, 1) = t31*t2214+t29*t2219+t440*t2206; fdrv(100, 2) = t30*t2219+t157*t2206-t439*t2214;
        fdrv(100, 3) = t302+t1735+t2278+t7*t35*6.0-t2*t43+t9*t39-t41*t104*4.0+t33*t115*5.0-t41*t111*4.0+t41*t121-t30*t226*5.0+t109*t180*8.0-t104*t188*4.0-t107*t187*4.0+t115*t179*6.0+t114*t184*8.0+t30*t286*5.0+t31*t296+t62*t265+t63*t284+t157*t232-t67*t329*4.0+t75*t328*4.0+t77*t329*6.0+t121*t398-t30*t605*6.0-t31*t614*2.0-t30*t631*4.0+t30*t1092-(t29*t29)*t68*6.0+t31*t64*t79;
        fdrv(100, 4) = -t1444-t1445-t1446+t1534+t1535+t1536+t2287+t9*t35-t37*t110*5.0+t37*t115*5.0-t41*t116*4.0+t41*t119+t43*t128-t107*t183*4.0+t115*t180*6.0-t110*t187*6.0-t116*t188*4.0+t33*t346+t75*t329*4.0+t77*t328*6.0+t119*t396+t179*t346+t33*t562+t106*t705-t29*t104*tz*6.0;
        fdrv(100, 5) = -t228-t1733+t2281+t10*t35-t5*t43*6.0-t33*t104-t41*t110*5.0+t39*t128-t30*t243*5.0-t104*t183*6.0-t111*t183*8.0-t110*t188*6.0-t116*t187*8.0+t29*t296+t30*t300*5.0+t155*t232+t33*t356+t41*t357+t79*t328*6.0+t184*t357+t188*t357+t33*t578+t37*t588+t29*t687+t62*t657+t179*t559+t30*t998+t30*t1093-t1411*tz*4.0-(t29*t29)*t65*2.0-(t30*t30)*t65*6.0;
        fdrv(100, 6) = -t1573*tx*(t160*-2.0+t166-t168*2.0+t206+t211+t304+t307+t32*t138+t36*t138+t40*t138); fdrv(100, 7) = -t1573*ty*(t160*-2.0+t166-t168*2.0+t206+t211+t304+t307+t32*t138+t36*t138+t40*t138);
        fdrv(100, 8) = -t1573*tz*(t160*-2.0+t166-t168*2.0+t206+t211+t304+t307+t32*t138+t36*t138+t40*t138); fdrv(101, 0) = t29*t2216+t156*t2205-t441*t2218; fdrv(101, 1) = t29*t2218+t157*t2205-t440*t2216; fdrv(101, 2) = t31*t2216+t30*t2218+t439*t2205;
        fdrv(101, 3) = -t274-t1734+t2280-t4*t35*6.0+t2*t39+t41*t107*4.0-t33*t117*5.0-t37*t120+t43*t132+t29*t226*5.0+t31*t243-t29*t247*5.0-t105*t180*8.0-t105*t187*4.0-t108*t184*8.0+t107*t188*4.0-t117*t183*6.0+t37*t336+t68*t329*6.0+t69*t329*4.0+t157*t282+t187*t336+t37*t560+t41*t567+t64*t615+t187*t595+t29*t880+t29*t896-t1417*tz*6.0-(t29*t29)*t72*6.0-(t31*t31)*t72*2.0;
        fdrv(101, 4) = t228+t1733+t2282-t6*t35+t3*t39*6.0+t5*t43-t33*t105*4.0+t33*t111+t37*t116*5.0-t41*t113*4.0+t29*t232*5.0+t30*t243+t104*t179*8.0-t105*t183*4.0-t29*t263*5.0+t62*t233-t113*t184*4.0+t110*t188*8.0+t116*t187*6.0+t64*t260-t73*t328*6.0-t74*t328*4.0+t156*t282+t111*t395+t183*t336-t29*t608*4.0-t29*t619*6.0-t30*t634*2.0+t29*t881+t1411*tz*6.0+t30*t63*t65;
        fdrv(101, 5) = t1444+t1445+t1446-t1495-t1496-t1497+t2286+t5*t39+t33*t107-t33*t108*4.0+t41*t116*5.0-t41*t117*5.0+t35*t132+t107*t183*6.0-t108*t183*4.0-t113*t187*6.0-t117*t184*6.0+t116*t188*6.0+t37*t354+t107*t393+t187*t354+t37*t575+t29*t680+t113*t704+t29*t1110;
        fdrv(101, 6) = -t1573*tx*(t162+t207+t209+t305+t308+t11*t155+t36*t134+t40*t134+t139*tx+t142*ty); fdrv(101, 7) = -t1573*ty*(t162+t207+t209+t305+t308+t11*t155+t36*t134+t40*t134+t139*tx+t142*ty);
        fdrv(101, 8) = -t1573*tz*(t162+t207+t209+t305+t308+t11*t155+t36*t134+t40*t134+t139*tx+t142*ty); fdrv(102, 0) = t30*t2227+t155*t2229-t441*t2225; fdrv(102, 1) = t29*t2225+t31*t2227+t440*t2229; fdrv(102, 2) = t30*t2225+t157*t2229-t439*t2227;
        fdrv(102, 3) = -t1407*t1573*t2020; fdrv(102, 4) = t1573*t2237; fdrv(102, 5) = -t1573*t2236; fdrv(102, 6) = -t1738*t1847; fdrv(102, 7) = -t1407*t1847*ty; fdrv(102, 8) = -t1407*t1847*tz; fdrv(103, 0) = t155*t2224+t156*t2226-t441*t2231;
        fdrv(103, 1) = t29*t2231+t157*t2226+t440*t2224; fdrv(103, 2) = t30*t2231+t157*t2224+t439*t2226; fdrv(103, 3) = -t1573*t2235; fdrv(103, 4) = t1406*t1573*t2022; fdrv(103, 5) = t1573*t2234; fdrv(103, 6) = t1406*t1847*tx; fdrv(103, 7) = t1737*t1847;
        fdrv(103, 8) = t1406*t1847*tz; fdrv(104, 0) = t30*t2223+t29*t2228+t441*t2230; fdrv(104, 1) = t31*t2223+t155*t2230-t440*t2228; fdrv(104, 2) = t31*t2228+t156*t2230-t439*t2223; fdrv(104, 3) = t1573*t2233; fdrv(104, 4) = -t1573*t2232;
        fdrv(104, 5) = -t1405*t1573*t2024; fdrv(104, 6) = -t1405*t1847*tx; fdrv(104, 7) = -t1405*t1847*ty; fdrv(104, 8) = -t1736*t1847; fdrv(105, 0) = t156*t2265+t155*t2268-t441*t2266; fdrv(105, 1) = t29*t2266+t157*t2265+t440*t2268;
        fdrv(105, 2) = t30*t2266+t157*t2268+t439*t2265;
        fdrv(105, 3) = t1573*(t270+t301+t613*6.0+t648*6.0+t671+t672+t673+t674+t726+t748+t893+t988+t1168+t1279+t1420+t1425+t1450+t1478-t1576-t1582+t1594+t1626+t1657+t1683+t1786+t30*t113*5.0+t30*t115*5.0)*2.0;
        fdrv(105, 4) = t1573*(t236-t250*7.0-t658*6.0-t659*6.0+t660+t661+t664+t665+t790+t935+t1137+t1412-t1415*8.0+t1474+t1494+t1531+t1669+t1776+t1781+t29*t105*5.0+t31*t113*5.0+t31*t115*5.0-t2*t181*6.0+t6*t181*6.0-t29*t237*6.0)*2.0;
        fdrv(105, 5) = t1573*(t252-t287*7.0-t639*6.0-t640*6.0+t645+t646+t649+t650+t791+t1079+t1157+t1414-t1419*8.0+t1464+t1490+t1529+t1647+t1782+t1803+t30*t114*5.0+t40*t113*5.0+t40*t115*5.0-t2*t185*6.0+t10*t185*6.0-t2*t328*6.0)*2.0;
        fdrv(105, 6) = t1847*tx*(t82+t84+t694+t697)*2.0; fdrv(105, 7) = t1847*ty*(t82+t84+t694+t697)*2.0; fdrv(105, 8) = t1847*tz*(t82+t84+t694+t697)*2.0; fdrv(106, 0) = t29*t2263+t156*t2264+t441*t2270; fdrv(106, 1) = t157*t2264+t155*t2270-t440*t2263;
        fdrv(106, 2) = t31*t2263+t156*t2270+t439*t2264;
        fdrv(106, 3) = t1573*(t236*-7.0+t250+t658+t659-t660*6.0-t661*6.0+t662+t663+t792+t873+t1205-t1412*8.0+t1415+t1438+t1520+t1531+t1698+t1777+t1797+t29*t104*5.0+t30*t107*5.0+t30*t109*5.0+t2*t182*6.0-t6*t182*6.0-t29*t276*6.0)*2.0;
        fdrv(106, 4) = t1573*(t227+t301+t610*6.0+t641+t642+t647+t648+t674*6.0+t753+t815+t892+t1074+t1171+t1367+t1409+t1424+t1447+t1509+t1576-t1594-t1618+t1632+t1659+t1710+t1801+t31*t107*5.0+t31*t109*5.0)*2.0;
        fdrv(106, 5) = t1573*(t289-t297*7.0+t623+t624-t625*6.0-t626*6.0+t627+t628+t796+t1098+t1319+t1418-t1421*8.0+t1443+t1460+t1530+t1649+t1798+t1828+t40*t107*5.0+t40*t109*5.0+t31*t121*5.0-t6*t189*6.0+t10*t189*6.0-t6*t329*6.0)*2.0;
        fdrv(106, 6) = t1847*tx*(t80+t85+t695+t700)*2.0; fdrv(106, 7) = t1847*ty*(t80+t85+t695+t700)*2.0; fdrv(106, 8) = t1847*tz*(t80+t85+t695+t700)*2.0; fdrv(107, 0) = t30*t2262+t155*t2267+t441*t2269; fdrv(107, 1) = t31*t2262+t155*t2269+t440*t2267;
        fdrv(107, 2) = t157*t2267+t156*t2269-t439*t2262;
        fdrv(107, 3) = t1573*(t252*-7.0+t287+t639+t640+t643+t644-t649*6.0-t650*6.0+t798+t888+t1361-t1414*8.0+t1419+t1439+t1490+t1516+t1700+t1804+t1814+t30*t104*5.0+t30*t105*5.0+t30*t111*5.0+t2*t186*6.0-t10*t186*6.0-t10*t328*6.0)*2.0;
        fdrv(107, 4) = t1573*(t289*-7.0+t297+t621+t622+t625+t626-t627*6.0-t628*6.0+t799+t1032+t1380-t1418*8.0+t1421+t1443+t1476+t1486+t1673+t1815+t1829+t31*t104*5.0+t31*t105*5.0+t31*t120*5.0+t6*t190*6.0-t10*t190*6.0-t10*t329*6.0)*2.0;
        fdrv(107, 5) = t1573*(t227+t270+t610+t611+t612+t613+t641*6.0+t671*6.0+t818+t841+t979+t1073+t1282+t1370+t1408+t1413+t1472+t1506+t1582+t1618-t1626-t1632+t1687+t1712+t1819+t40*t104*5.0+t40*t105*5.0)*2.0;
        fdrv(107, 6) = t1847*tx*(t81+t83+t698+t701)*2.0; fdrv(107, 7) = t1847*ty*(t81+t83+t698+t701)*2.0; fdrv(107, 8) = t1847*tz*(t81+t83+t698+t701)*2.0;
        
        /* #endregion  Calculating f and its jacobian -----------------------------------------------------------*/
    }
    
    /* #endregion Calculating the derivatives of f --------------------------------------------------------------*/

    S1.setZero(); S2.setZero();
    
    // Calculating the component jacobians
    for (int idx = 0; idx < COMPONENTS; idx++)
    {
        T &g = gdrv(idx, 0);
        T &dg = gdrv(idx, 1);

        Matrix<T, 1, 3> Jdg_U = dg*Ubtp;

        auto fXi = fdrv.block(idx*3, 0, 3, 3);
        auto dfXi_W_dU = fdrv.block(idx*3, 3, 3, 3);
        auto dfXi_W_dV = fdrv.block(idx*3, 6, 3, 3);

        // Q += fXi*g;

        // Find the S1 S2 jacobians
        const Vec3T &W = Omg;
        Vec3T fXiW = fXi*W;

        S1 += dfXi_W_dU*g + fXiW*Jdg_U;
        S2 += dfXi_W_dV*g;

        // cout << "ComputeS " << idx << ": g\n" << g << endl;
        // cout << "ComputeS " << idx << ": dfXi_W_dU\n" << dfXi_W_dU << endl;
        // cout << "ComputeS " << idx << ": dfXi_W_dV\n" << dfXi_W_dV << endl;
        // cout << "ComputeS " << idx << ": S1\n" << S1_ << endl;
        // cout << "ComputeS " << idx << ": S2\n" << S2_ << endl;
    }
}

template <typename T>
void SE3Qp<T>::ComputeQSC(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{
    T Un = The.norm();
    Matrix<T, 3, 1> Ub = The / Un;
    Matrix<T, 1, 3> Ubtp = Ub.transpose();
    Matrix<T, 3, 3> JUb = (Mat3T::Identity(3, 3) - Ub*Ubtp) / Un;

    // This Q has 4 g and each has 3 derivatives 0th, 1st and 2nd
    Matrix<T, Eigen::Dynamic, 3> gdrv(COMPONENTS, 3); gdrv.setZero();

    // This Q has 4 fs and each f has 9 derivatives (f, S1, C11, C12, C13, S2, C21, C22, C23) (4x3 x 9x3)
    Matrix<T, Eigen::Dynamic, 27> fdrv(3*COMPONENTS, 27); fdrv.setZero();

    /* #region Calculating the derivatives of g -----------------------------------------------------------------*/

    {
        /* #region Repeated terms -------------------------------------------------------------------------------*/

        T t2 = cos(Un), t3 = sin(Un), t4 = Un*2.0, t5 = Un*3.0, t6 = Un*4.0, t7 = Un*6.0, t8 = Un*Un, t9 = Un*Un*Un, t11 = Un*Un*Un*Un*Un, t28 = Un*8.0, t29 = Un*4.0E+1, t30 = Un*5.0E+1, t31 = Un*5.4E+1, t32 = 1.0/Un, t78 = Un*5.88E+2;
        T t10 = t8*t8, t12 = t8*t8*t8, t13 = cos(t4), t14 = t2*2.0, t15 = cos(t5), t16 = cos(t6), t17 = t2*4.0, t18 = t2*t2, t19 = t2*t2*t2, t20 = sin(t4), t21 = t3*2.0, t22 = sin(t5), t23 = t3*3.0, t24 = sin(t6), t25 = t3*5.0, t26 = Un*t2;
        T t27 = Un*t3, t33 = 1.0/t8, t34 = 1.0/t9, t36 = 1.0/t11, t38 = t32*t32*t32*t32*t32*t32*t32, t39 = Un*t4, t40 = Un*t5, t41 = t9*5.0, t42 = t9*7.0, t43 = t2*1.0E+1, t44 = t2*2.0E+1, t45 = t2*2.4E+1, t46 = t2*4.2E+1, t47 = t2-1.0, t48 = -t3;
        T t51 = t3*1.2E+1, t52 = t3*1.5E+1, t53 = 1.0/t3, t55 = t2*t7, t57 = t3*t4, t58 = -t30, t59 = -t31, t63 = t8*1.0E+1, t64 = t9*6.0E+1, t65 = t2/2.0, t70 = t3*7.5E+1, t71 = t3*9.0E+1, t72 = t3*2.1E+2, t79 = t2*t8, t80 = t2*t9, t81 = t2*t11;
        T t82 = t3*t8, t83 = t3*t9, t86 = t3*1.68E+3, t100 = -t78, t35 = 1.0/t10, t37 = 1.0/t12, t49 = -t21, t50 = -t23, t54 = t53*t53, t56 = t26*7.0, t61 = t34*t34*t34, t62 = t33*t33*t33*t33*t33, t66 = -t43, t67 = -t46, t68 = -t51, t69 = -t52;
        T t73 = t26*2.7E+1, t74 = t26*5.0E+1, t75 = t27*8.0, t76 = t27*1.2E+1, t77 = t20*4.0, t84 = t3*t10, t85 = -t71, t88 = t3*t55, t89 = t26*2.94E+2, t90 = t15*4.2E+1, t91 = t18*1.0E+1, t92 = t19*1.0E+1, t93 = t18*1.2E+1, t95 = t27*1.32E+2;
        T t97 = t22*1.5E+1, t98 = t22*4.2E+1, t99 = t20*6.0E+1, t101 = t4*t26, t102 = Un+t48, t103 = t7*t83, t104 = t33/2.0, t105 = t33/4.0, t106 = t9*t13, t107 = t8*t19, t109 = t9*t22, t110 = t10*t22, t111 = t13*8.4E+1, t114 = 1.0/t47;
        T t118 = t20*1.68E+2, t119 = t22*3.36E+2, t120 = Un*t13*1.0E+1, t121 = Un*t15*1.0E+1, t122 = Un*t15*2.7E+1, t123 = t13*t31, t124 = -t79, t126 = -t80, t127 = t8*t43, t128 = t80*1.5E+1, t129 = Un*t20*1.2E+1, t130 = Un*t22*3.6E+1;
        T t131 = t82*1.1E+1, t132 = t83*1.1E+1, t134 = t82*4.6E+1, t138 = t4*t8*t20, t139 = Un*t6*t20, t144 = t20*1.344E+3, t145 = t3*t7*t18, t148 = Un*t15*2.94E+2, t151 = t82/2.0, t153 = t82*1.2E+2, t158 = t9*t15*1.5E+1, t160 = t8+t14-2.0;
        T t161 = t8*t18*-2.0, t164 = t10*t20*-4.0, t165 = t8*t22*-6.0, t166 = t8*t20*3.2E+1, t167 = t65+1.0/2.0, t171 = t13*t78, t172 = t80*(-1.0/2.0), t173 = t80*(1.7E+1/4.0), t176 = (t9*t15)/2.0, t177 = (t9*t15)/4.0, t178 = t8*t13*-1.0E+1;
        T t179 = t8*t15*-1.0E+1, t181 = t8*t22*(5.0/2.0), t182 = t8*t22*9.6E+1, t183 = t8*t20*2.04E+2, t189 = (t34*t47)/2.0, t193 = t8+t17+t27-4.0, t60 = t35*t35, t94 = -t76, t96 = -t77, t112 = -t91, t115 = t114*t114, t116 = -t95, t117 = -t99;
        T t133 = t84*1.7E+1, t137 = t4+t49, t142 = -t111, t143 = -t118, t146 = -t121, t147 = -t122, t150 = -t128, t152 = -t134, t154 = -t106, t155 = t106*-7.0, t162 = -t107, t163 = -t109, t168 = -t144, t169 = t20+t49, t170 = -t148, t174 = -t151;
        T t175 = -t153, t180 = t106*-6.0E+1, t184 = Un+t26+t49, t185 = -t173, t186 = -t181, t187 = -t182, t190 = t4+t26+t50, t192 = -t189, t194 = (t34*t102)/2.0, t195 = t35*t102*(3.0/2.0), t198 = (t35*t160)/4.0, t200 = t167/t27, t201 = t7+t55+t68+t82;
        T t203 = t28+t56+t69+t82, t205 = (t36*t193)/2.0, t212 = t40+t44+t75+t124-2.0E+1, t215 = (t34*t114*t193)/4.0, t218 = (t34*t114*t193)/8.0, t226 = t29+t74+t85+t126+t131, t233 = (t38*t114*t160*t193)/4.0, t235 = (t38*t114*t160*t193)/8.0;
        T t188 = 1.0/t169, t191 = t184*t184, t196 = t22+t25+t96, t197 = -t195, t202 = (t36*t190)/4.0, t204 = -t200, t206 = t200/2.0, t207 = t200/4.0, t208 = -t205, t214 = (t36*t201)/2.0, t216 = (t37*t203)/4.0, t217 = -t215;
        T t221 = ((t32*t32*t32*t32*t32*t32)*t160*t184)/t27, t222 = -t218, t225 = (t37*t53*t137*t184)/4.0, t229 = (t37*t53*t137*t184)/8.0, t231 = (t37*t212)/2.0, t234 = -t233, t236 = -t235, t237 = (t38*t226)/4.0;
        T t245 = (t35*t115*(-t39-t45+t79*2.0+t93+t3*t26*2.0+t4*t48+t9*t48+1.2E+1))/4.0, t247 = (t35*t115*(-t39-t45+t79*2.0+t93+t3*t26*2.0+t4*t48+t9*t48+1.2E+1))/8.0, t248 = t39+t66+t79+t83+t88+t92+t94+t112+t145+t161+t162+1.0E+1;
        T t255 = t63+t67+t90+t116+t127+t129+t130+t132+t138+t142+t163+t178+t179+8.4E+1, t256 = t41+t58+t70+t74+t84+t97+t117+t120+t139+t146+t154+t174+t177+t185+t186, t257 = t11+t42+t59+t72+t73+t81+t98+t103+t123+t143+t147+t152+t155+t165+t166+t172+t176;
        T t268 = t64+t86+t89+t100+t110+t119+t133+t150+t158+t164+t168+t170+t171+t175+t180+t183+t187, t199 = 1.0/t196, t209 = -t206, t210 = -t207, t211 = t33+t204, t219 = t192+t197, t220 = -t216, t227 = t221/2.0, t228 = -t225, t230 = -t229;
        T t249 = t37*t188*t248, t258 = (t60*t188*t255)/8.0, t259 = (t60*t188*t255)/1.6E+1, t262 = (t38*t115*t256)/2.0, t263 = (t38*t115*t256)/4.0, t266 = (t60*t188*t257)/2.0, t267 = (t60*t188*t257)/4.0, t269 = (t61*t115*t268)/1.6E+1;
        T t270 = (t61*t115*t268)/3.2E+1, t213 = t211*t211, t223 = t104+t209, t224 = t105+t210, t232 = t34*t102*t211, t238 = t194*t211, t239 = (t35*t160*t211)/2.0, t240 = t198*t211, t241 = (t36*t190*t211)/2.0, t242 = t202*t211, t250 = -t249;
        T t251 = t249/2.0, t253 = t221+t228+t234, t254 = t227+t230+t236, t260 = -t258, t261 = -t259, t264 = -t262, t265 = -t263, t271 = -t269, t272 = -t270, t252 = -t251;

        /* #endregion Repeated terms ----------------------------------------------------------------------------*/

        /* #region Calculate g derivatives ----------------------------------------------------------------------*/

        gdrv(0, 0) = T(1.0/2.0); gdrv(1, 0) = t34*t102; gdrv(1, 1) = -t34*t47-t35*t102*3.0;
        gdrv(1, 2) = t36*t201; gdrv(2, 0) = (t35*t160)/2.0; gdrv(2, 1) = -t36*t193; gdrv(2, 2) = t37*t212; gdrv(3, 0) = (t36*t190)/2.0; gdrv(3, 1) = t37*t203*(-1.0/2.0); gdrv(3, 2) = (t38*t226)/2.0; gdrv(4, 0) = T(1.0/4.0); gdrv(5, 0) = t194; gdrv(5, 1) = t219;
        gdrv(5, 2) = t214; gdrv(6, 0) = t198; gdrv(6, 1) = t208; gdrv(6, 2) = t231; gdrv(7, 0) = t202; gdrv(7, 1) = t220; gdrv(7, 2) = t237; gdrv(8, 0) = t223; gdrv(8, 1) = t217; gdrv(8, 2) = t245; gdrv(9, 0) = t232; gdrv(9, 1) = t250; gdrv(9, 2) = t264; gdrv(10, 0) = t239;
        gdrv(10, 1) = t253; gdrv(10, 2) = t266; gdrv(11, 0) = t241; gdrv(11, 1) = t260; gdrv(11, 2) = t271; gdrv(12, 0) = T(1.0/4.0); gdrv(13, 0) = t194; gdrv(13, 1) = t219; gdrv(13, 2) = t214; gdrv(14, 0) = t198; gdrv(14, 1) = t208; gdrv(14, 2) = t231; gdrv(15, 0) = t202;
        gdrv(15, 1) = t220; gdrv(15, 2) = t237; gdrv(16, 0) = T(1.0/8.0); gdrv(17, 0) = (t34*t102)/4.0; gdrv(17, 1) = t34*t47*(-1.0/4.0)-t35*t102*(3.0/4.0); gdrv(17, 2) = (t36*t201)/4.0; gdrv(18, 0) = (t35*t160)/8.0; gdrv(18, 1) = t36*t193*(-1.0/4.0);
        gdrv(18, 2) = (t37*t212)/4.0; gdrv(19, 0) = (t36*t190)/8.0; gdrv(19, 1) = t37*t203*(-1.0/8.0); gdrv(19, 2) = (t38*t226)/8.0; gdrv(20, 0) = t224; gdrv(20, 1) = t222; gdrv(20, 2) = t247; gdrv(21, 0) = t238; gdrv(21, 1) = t252; gdrv(21, 2) = t265;
        gdrv(22, 0) = t240; gdrv(22, 1) = t254; gdrv(22, 2) = t267; gdrv(23, 0) = t242; gdrv(23, 1) = t261; gdrv(23, 2) = t272; gdrv(24, 0) = t223; gdrv(24, 1) = t217; gdrv(24, 2) = t245; gdrv(25, 0) = t232; gdrv(25, 1) = t250; gdrv(25, 2) = t264; gdrv(26, 0) = t239;
        gdrv(26, 1) = t253; gdrv(26, 2) = t266; gdrv(27, 0) = t241; gdrv(27, 1) = t260; gdrv(27, 2) = t271; gdrv(28, 0) = t224; gdrv(28, 1) = t222; gdrv(28, 2) = t247; gdrv(29, 0) = t238; gdrv(29, 1) = t252; gdrv(29, 2) = t265; gdrv(30, 0) = t240; gdrv(30, 1) = t254;
        gdrv(30, 2) = t267; gdrv(31, 0) = t242; gdrv(31, 1) = t261; gdrv(31, 2) = t272; gdrv(32, 0) = t211*t223; gdrv(32, 1) = (t36*t115*(t2*8.0+t8-t18*8.0+t27*6.0+t47*8.0-t101-t5*t20+t8*t18+t9*t48))/4.0;
        gdrv(32, 2) = (t37*t115*(t2*-4.0E+1-t8*9.0+t18*4.0E+1-t27*2.4E+1-t47*4.0E+1+t79*1.2E+1+t129+t2*t10+t4*t9-t8*t18*3.0+t8*t57))/4.0; gdrv(33, 0) = t34*t102*t213;
        gdrv(33, 1) = t54*t60*t102*t191*(-3.0/4.0)-(1.0/(t27*t27)*(t32*t32*t32*t32*t32)*t47*t191)/4.0+(t53*t60*t102*t114*t184*t193)/2.0;
        gdrv(33, 2) = t61*t199*(t8*(-1.55E+2/4.0)+t10*(3.5E+1/8.0)+t13*1.12E+2-t15*1.12E+2+t16*2.8E+1+t27*3.08E+2+t47*1.12E+2-t83*3.8E+1-Un*t20*1.68E+2-Un*t22*2.8E+1+Un*t24*2.8E+1+t8*t13*4.9E+1-t8*t15*4.0-t10*t13*(9.0/2.0)-t8*t16*(4.1E+1/4.0)-(t10*t15)/2.0+(t10*t16)/8.0+t9*t20*(3.9E+1/2.0)+t11*t20+t6*t26-t9*t24*(7.0/4.0)+t10*t65+t6*t84+t4*t8*t22-2.8E+1);
        gdrv(34, 0) = (t35*t160*t213)/2.0; gdrv(34, 1) = (t54*t60*t137*t191)/8.0-(t54*t61*t160*t191)/2.0+(t53*t61*t114*t160*t184*t193)/4.0;
        gdrv(34, 2) = t62*t199*(Un*1.8E+2-t3*5.04E+2-t9*4.0E+1+t20*5.04E+2-t22*2.16E+2+t24*3.6E+1-t26*1.44E+2+t80*2.3E+1+t82*(2.07E+2/2.0)-t84*1.3E+1+t106*3.8E+1-t110*3.0-Un*t13*1.44E+2+Un*t15*1.44E+2-Un*t16*3.6E+1+t5*t10-t9*t15*2.3E+1-t11*t13*3.0-t8*t20*(2.37E+2/2.0)+t8*t22*(1.23E+2/2.0)+t10*t20*(4.3E+1/4.0)-t8*t24*(5.1E+1/4.0)+(t12*t20)/2.0+(t10*t24)/8.0+t11*t57+t4*t8*t16);
        gdrv(35, 0) = (t36*t190*t213)/2.0; gdrv(35, 1) = t54*t61*t191*(t14+t27-2.0)*(-1.0/8.0)-t54*t62*t190*t191*(5.0/8.0)+(t53*t62*t114*t184*t190*t193)/4.0;
        gdrv(35, 2) = (pow(t32,1.1E+1)*t199*(t8*-1.83E+2+t10*(1.21E+2/4.0)+t13*1.08E+3-t15*1.08E+3+t16*2.7E+2+t27*2.232E+3+t47*1.08E+3-t79*1.56E+2-t83*2.64E+2+t109*7.2E+1-Un*t20*8.28E+2-Un*t22*6.48E+2+Un*t24*3.42E+2+t2*t10*1.9E+1+t3*t11*2.3E+1+t8*t13*3.48E+2+t8*t15*1.56E+2-t10*t13*3.5E+1-t8*t16*1.65E+2-t10*t15*1.9E+1+t10*t16*(1.9E+1/4.0)+t9*t20*1.02E+2+t11*t20*(1.5E+1/2.0)-t9*t24*3.9E+1-t11*t22+(t11*t24)/4.0-2.7E+2))/4.0;

        /* #endregion Calculate g derivatives --------------------------------------------------------------------*/
    }

    /* #endregion Calculating the derivatives of g --------------------------------------------------------------*/

    /* #region Calculating the derivatives of f -----------------------------------------------------------------*/
    
    // int fidx = 0;
    // fdrv.block(fidx*36, 0, 36, 27) = f01To12(The, Rho, Thed, Rhod, Omg); fidx +=1;
    // fdrv.block(fidx*36, 0, 36, 27) = f13To24(The, Rho, Thed, Rhod, Omg); fidx +=1;
    // fdrv.block(fidx*36, 0, 36, 27) = f25To36(The, Rho, Thed, Rhod, Omg); fidx +=1;

    int fidx = 0;
    fdrv.block(fidx*3, 0, 3, 27) = f01(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f02(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f03(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f04(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f05(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f06(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f07(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f08(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f09(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f10(The, Rho, Thed, Rhod, Omg); fidx +=1;

    fdrv.block(fidx*3, 0, 3, 27) = f11(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f12(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f13(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f14(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f15(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f16(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f17(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f18(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f19(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f20(The, Rho, Thed, Rhod, Omg); fidx +=1;

    fdrv.block(fidx*3, 0, 3, 27) = f21(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f22(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f23(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f24(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f25(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f26(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f27(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f28(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f29(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f30(The, Rho, Thed, Rhod, Omg); fidx +=1;

    fdrv.block(fidx*3, 0, 3, 27) = f31(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f32(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f33(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f34(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f35(The, Rho, Thed, Rhod, Omg); fidx +=1;
    fdrv.block(fidx*3, 0, 3, 27) = f36(The, Rho, Thed, Rhod, Omg); fidx +=1;

    /* #endregion Calculating the derivatives of f --------------------------------------------------------------*/

    ResetQSC();

    // Calculating the component Jacobians
    for (int idx = 0; idx < COMPONENTS; idx++)
    {
        T &g = gdrv(idx, 0);
        T &dg = gdrv(idx, 1);
        T &ddg = gdrv(idx, 2);

        Matrix<T, 1, 3> Jdg_U = dg*Ubtp;
        Matrix<T, 1, 3> Jddg_U = ddg*Ubtp;

        auto fXi = fdrv.block(idx*3, 0, 3, 3);

        auto dfXi_W_dU = fdrv.block(idx*3, S1_IDX, 3, 3);
        auto ddfXiW_X_dUdU = fdrv.block(idx*3, C11_IDX, 3, 3);
        auto ddfXiW_X_dUdV = fdrv.block(idx*3, C12_IDX, 3, 3);
        auto ddfXiW_X_dUdW = fdrv.block(idx*3, C13_IDX, 3, 3);

        auto dfXi_W_dV = fdrv.block(idx*3, S2_IDX, 3, 3);
        auto ddfXiW_X_dVdU = fdrv.block(idx*3, C21_IDX, 3, 3);
        auto ddfXiW_X_dVdV = fdrv.block(idx*3, C22_IDX, 3, 3);
        auto ddfXiW_X_dVdW = fdrv.block(idx*3, C23_IDX, 3, 3);

        Q += fXi*g;

        {
            const Vec3T &W = Omg;
            const Vec3T &X = Thed; Matrix<T, 1, 3> Xtp = X.transpose();
            Vec3T fXiW = fXi*W;
            T UbtpX = Ubtp.dot(X);
            
            // Find the S1 C1 jacobians
            S1  += fXiW*Jdg_U + dfXi_W_dU*g;
            C11 += dfXi_W_dU*(Jdg_U*X) + fXiW*Jddg_U*UbtpX + fXiW*Xtp*dg*JUb + ddfXiW_X_dUdU*g + dfXi_W_dU*X*Jdg_U;
            C12 += dfXi_W_dV*(Jdg_U*X) + ddfXiW_X_dUdV*g;
            C13 += fXi*(Jdg_U*X) + ddfXiW_X_dUdW*g;
        }

        {
            const Vec3T &W = Omg;
            const Vec3T &X = Rhod; Matrix<T, 1, 3> Xtp = X.transpose();
            Vec3T fXiW = fXi*W;
            T UbtpX = Ubtp.dot(X);

            // Find the S2 C2 jacobians
            S2  += dfXi_W_dV*g;
            C21 += ddfXiW_X_dVdU*g + dfXi_W_dV*X*Jdg_U;
            C22 += ddfXiW_X_dVdV*g;
            C23 += ddfXiW_X_dVdW*g;       
        }
    }
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f01(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

        
    fdrv(0, 1) = -rz; fdrv(0, 2) = ry; fdrv(0, 16) = oz; fdrv(0, 17) = -oy; fdrv(0, 25) = -rdz; fdrv(0, 26) = rdy; fdrv(1, 0) = rz; fdrv(1, 2) = -rx; fdrv(1, 15) = -oz; fdrv(1, 17) = ox; fdrv(1, 24) = rdz; fdrv(1, 26) = -rdx; fdrv(2, 0) = -ry; fdrv(2, 1) = rx;
    fdrv(2, 15) = oy; fdrv(2, 16) = -ox; fdrv(2, 24) = -rdy; fdrv(2, 25) = rdx;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f02(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*rdx, t3 = oy*rdy, t4 = oz*rdz, t5 = ox*rx, t6 = oy*ry, t7 = oz*rz, t8 = ox*tdy, t9 = ox*tdz, t10 = oy*tdx, t11 = oy*tdz, t12 = oz*tdx, t13 = oz*tdy, t14 = ox*ty, t15 = oy*tx, t16 = ox*tz, t17 = oz*tx, t18 = oy*tz, t19 = oz*ty;
    T t20 = rx*tx, t21 = rx*ty, t22 = ry*tx, t23 = rx*tz, t24 = ry*ty, t25 = rz*tx, t26 = ry*tz, t27 = rz*ty, t28 = rz*tz, t29 = tx*ty, t30 = tx*tz, t31 = ty*tz, t32 = ox*2.0, t33 = oy*2.0, t34 = oz*2.0, t35 = tx*tx, t36 = ty*ty, t37 = tz*tz;
    T t44 = rdx*tx*2.0, t45 = rx*tdx*2.0, t46 = rdy*ty*2.0, t47 = ry*tdy*2.0, t48 = rdz*tz*2.0, t49 = rz*tdz*2.0, t38 = t14*2.0, t39 = t15*2.0, t40 = t16*2.0, t41 = t17*2.0, t42 = t18*2.0, t43 = t19*2.0, t50 = t20*2.0, t51 = t24*2.0;
    T t52 = t28*2.0, t53 = t2*tx, t54 = t3*ty, t55 = t4*tz, t56 = ox+t18, t57 = oy+t17, t58 = oz+t14, t59 = rx+t26, t60 = rx+t27, t61 = ry+t23, t62 = ry+t25, t63 = rz+t21, t64 = rz+t22, t65 = t31+tx, t66 = t30+ty, t67 = t29+tz, t68 = -t2;
    T t69 = -t3, t70 = -t4, t71 = -t6, t72 = -t7, t73 = -t10, t74 = -t12, t75 = -t13, t76 = -t15, t78 = -t16, t79 = -t17, t81 = -t19, t83 = -t21, t84 = -t22, t85 = -t23, t86 = -t25, t87 = -t26, t88 = -t27, t92 = rdx*t35, t93 = rdy*t36;
    T t94 = rdz*t37, t107 = t20+t24, t108 = t20+t28, t109 = t24+t28, t77 = -t39, t80 = -t40, t82 = -t43, t95 = ox+t81, t96 = oy+t78, t97 = oz+t76, t98 = rx+t87, t99 = rx+t88, t100 = ry+t85, t101 = ry+t86, t102 = rz+t83, t103 = rz+t84;
    T t110 = t5+t71, t111 = t5+t72, t112 = t6+t72, t113 = t14+t76, t114 = t16+t79, t115 = t18+t81, t116 = t52+t107, t117 = t51+t108, t118 = t50+t109, t125 = t34+t38+t76, t126 = t33+t41+t78, t127 = t32+t42+t81, t119 = t14+t34+t77;
    T t120 = t17+t33+t80, t121 = t18+t32+t82, t122 = t118*tdx, t123 = t117*tdy, t124 = t116*tdz;
    
    fdrv(0, 0) = t51+t52; fdrv(0, 1) = t83+t84+t20*tz+t109*tz; fdrv(0, 2) = t85+t86-t20*ty-t109*ty; fdrv(0, 3) = -oy*t100-oz*t63;
    fdrv(0, 4) = -oy*t98-oz*t117+ry*t32; fdrv(0, 5) = oy*t116-oz*t60+rz*t32; fdrv(0, 6) = rx*(t11+t75); fdrv(0, 7) = rx*t74-ry*t13*2.0+t112*tdz; fdrv(0, 8) = rx*t10+rz*t11*2.0+t112*tdy; fdrv(0, 9) = t115*tdx-t57*tdy-t97*tdz; fdrv(0, 10) = t73+t121*tdy+t11*ty;
    fdrv(0, 11) = t74+t127*tdz+t75*tz; fdrv(0, 12) = t47+t49; fdrv(0, 13) = t124-t100*tdx-t98*tdy; fdrv(0, 14) = -t123-t63*tdx-t60*tdz; fdrv(0, 15) = -oz*t67+oy*(t30-ty); fdrv(0, 16) = t38+t81*ty+oy*(t31-tx); fdrv(0, 17) = t40-oz*t65+t18*tz;
    fdrv(0, 18) = t69+t70+rdx*t115; fdrv(0, 19) = -t55-rdx*t57+rdy*t121; fdrv(0, 20) = t54-rdx*t97+rdz*t127; fdrv(0, 24) = t46+t48; fdrv(0, 25) = t94+rdy*(t31-tx)+rdx*(t30-ty); fdrv(0, 26) = -t93-rdx*t67-rdz*t65; fdrv(1, 0) = t83+t84-t24*tz-t108*tz;
    fdrv(1, 1) = t50+t52; fdrv(1, 2) = t87+t88+t108*tx+t22*ty; fdrv(1, 3) = -ox*t61+oz*t118+rx*t33; fdrv(1, 4) = -ox*t59-oz*t103; fdrv(1, 5) = -ox*t116-oz*t101+rz*t33; fdrv(1, 6) = rx*t12*2.0+ry*t13-t111*tdz; fdrv(1, 7) = -ry*(t9+t74);
    fdrv(1, 8) = -ry*t8-rz*t9*2.0-t111*tdx; fdrv(1, 9) = -t8+t126*tdx-t9*tx; fdrv(1, 10) = -t95*tdx-t114*tdy-t58*tdz; fdrv(1, 11) = t75+t120*tdz+t12*tz; fdrv(1, 12) = -t124-t61*tdx-t59*tdy; fdrv(1, 13) = t45+t49; fdrv(1, 14) = t122-t103*tdy-t101*tdz;
    fdrv(1, 15) = t39-ox*t66+t17*tx; fdrv(1, 16) = -ox*t65+oz*(t29-tz); fdrv(1, 17) = t42+t78*tz+oz*(t30-ty); fdrv(1, 18) = t55+rdx*t126-rdy*t95; fdrv(1, 19) = t68+t70-rdy*t114; fdrv(1, 20) = -t53-rdy*t58+rdz*t120; fdrv(1, 24) = -t94-rdx*t66-rdy*t65;
    fdrv(1, 25) = t44+t48; fdrv(1, 26) = t92+rdz*(t30-ty)+rdy*(t29-tz); fdrv(2, 0) = t85+t86+t107*ty+t27*tz; fdrv(2, 1) = t87+t88-t107*tx+t86*tz; fdrv(2, 2) = t50+t51; fdrv(2, 3) = -ox*t102-oy*t118+rx*t34; fdrv(2, 4) = ox*t117-oy*t64+ry*t34;
    fdrv(2, 5) = -ox*t99-oy*t62; fdrv(2, 6) = rx*t10*-2.0-rz*t11+t110*tdy; fdrv(2, 7) = ry*t8*2.0+rz*t9+t110*tdx; fdrv(2, 8) = rz*(t8+t73); fdrv(2, 9) = -t9+t119*tdx+t8*tx; fdrv(2, 10) = -t11+t125*tdy+t73*ty; fdrv(2, 11) = -t56*tdx-t96*tdy+t113*tdz;
    fdrv(2, 12) = t123-t102*tdx-t99*tdz; fdrv(2, 13) = -t122-t64*tdy-t62*tdz; fdrv(2, 14) = t45+t47; fdrv(2, 15) = t41+t76*tx+ox*(t29-tz); fdrv(2, 16) = t43-oy*t67+t14*ty; fdrv(2, 17) = -oy*t66+ox*(t31-tx); fdrv(2, 18) = -t54+rdx*t119-rdz*t56;
    fdrv(2, 19) = t53+rdy*t125-rdz*t96; fdrv(2, 20) = t68+t69+rdz*t113; fdrv(2, 24) = t93+rdz*(t31-tx)+rdx*(t29-tz); fdrv(2, 25) = -t92-rdy*t67-rdz*t66; fdrv(2, 26) = t44+t46;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f03(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*ty, t3 = oy*tx, t4 = ox*tz, t5 = oz*tx, t6 = oy*tz, t7 = oz*ty, t8 = tx*tx, t9 = ty*ty, t10 = tz*tz, t11 = ox*rx*2.0, t12 = ox*ry*2.0, t13 = oy*rx*2.0, t14 = ox*rz*2.0, t15 = oy*ry*2.0, t16 = oz*rx*2.0, t17 = oy*rz*2.0;
    T t18 = oz*ry*2.0, t19 = oz*rz*2.0, t20 = ox*tx*2.0, t24 = oy*ty*2.0, t28 = oz*tz*2.0, t29 = rx*tx*2.0, t30 = rx*tx*3.0, t31 = rx*ty*2.0, t32 = ry*tx*2.0, t33 = rx*tz*2.0, t34 = ry*ty*2.0, t35 = rz*tx*2.0, t36 = ry*ty*3.0, t37 = ry*tz*2.0;
    T t38 = rz*ty*2.0, t39 = rz*tz*2.0, t40 = rz*tz*3.0, t52 = rdx*tx*ty*2.0, t53 = rdy*tx*ty*2.0, t54 = rdx*tx*tz*2.0, t55 = rdz*tx*tz*2.0, t56 = rdy*ty*tz*2.0, t57 = rdz*ty*tz*2.0, t21 = t2*2.0, t22 = t3*2.0, t23 = t4*2.0, t25 = t5*2.0;
    T t26 = t6*2.0, t27 = t7*2.0, t41 = -t15, t42 = -t19, t43 = -t3, t45 = -t5, t47 = -t7, t49 = -t32, t50 = -t35, t51 = -t38, t58 = -t8, t59 = -t9, t60 = -t10, t61 = t8+t9, t62 = t8+t10, t63 = t9+t10, t70 = t12+t13, t71 = t14+t16, t72 = t17+t18;
    T t74 = t20+t24, t76 = t20+t28, t78 = t24+t28, t79 = t30+t36, t80 = t30+t40, t81 = t36+t40, t100 = t29+t34+t39, t44 = -t22, t46 = -t25, t48 = -t27, t64 = rx*t61, t65 = rx*t62, t66 = ry*t61, t67 = ry*t63, t68 = rz*t62, t69 = rz*t63;
    T t73 = t2+t43, t75 = t4+t45, t77 = t6+t47, t82 = t11+t41, t83 = t11+t42, t84 = t15+t42, t88 = t31+t49, t89 = t33+t50, t90 = t37+t51, t101 = t58+t63, t102 = t59+t62, t103 = t60+t61, t105 = ox*t100, t106 = oy*t100, t107 = oz*t100;
    T t108 = t100*tdx, t109 = t100*tdy, t110 = t100*tdz, t85 = t21+t44, t86 = t23+t46, t87 = t26+t48, t91 = oz*t88, t92 = oy*t89, t93 = ox*t90, t94 = t88*tdx, t95 = t88*tdy, t96 = t89*tdx, t97 = t89*tdz, t98 = t90*tdy, t99 = t90*tdz;
    T t111 = rdx*t101, t112 = rdy*t102, t113 = rdz*t103, t104 = -t92;
    
    fdrv(0, 1) = t68+t69-t81*tz-rx*tx*tz*2.0+ry*ty*tz; fdrv(0, 2) = -t66-t67+t29*ty+t81*ty-rz*ty*tz; fdrv(0, 3) = t91+t104; fdrv(0, 4) = t107-oy*t90; fdrv(0, 5) = -t106-oz*t90;
    fdrv(0, 6) = t16*tdy+tdx*(t17-t18)-oy*rx*tdz*2.0; fdrv(0, 7) = t16*tdx+t72*tdy-t84*tdz; fdrv(0, 8) = -t84*tdy-t72*tdz-oy*rx*tdx*2.0; fdrv(0, 9) = -t87*tdx+t25*tdy-t3*tdz*2.0; fdrv(0, 10) = t5*tdx*-2.0-t87*tdy-t78*tdz;
    fdrv(0, 11) = t22*tdx+t78*tdy-t87*tdz; fdrv(0, 13) = -t96-t98-t110; fdrv(0, 14) = t94-t99+t109; fdrv(0, 15) = t77*tx*-2.0; fdrv(0, 16) = -oz*t102-t6*ty*2.0; fdrv(0, 17) = oy*t103+t27*tz; fdrv(0, 18) = rdx*t77*-2.0-rdy*t5*2.0+rdz*t22;
    fdrv(0, 19) = rdx*t25-rdy*t87+rdz*t78; fdrv(0, 20) = rdx*t3*-2.0-rdy*t78-rdz*t87; fdrv(0, 25) = -t54-t56+t113; fdrv(0, 26) = t52+t57-t112; fdrv(1, 0) = -t68-t69+t34*tz+t80*tz-rx*tx*tz; fdrv(1, 2) = t64+t65-t80*tx-ry*tx*ty*2.0+rz*tx*tz;
    fdrv(1, 3) = -t107+ox*t89; fdrv(1, 4) = t91+t93; fdrv(1, 5) = t105+oz*t89; fdrv(1, 6) = -t71*tdx+t83*tdz-oz*ry*tdy*2.0; fdrv(1, 7) = t12*tdz-tdy*(t14-t16)-oz*ry*tdx*2.0; fdrv(1, 8) = t83*tdx+t12*tdy+t71*tdz; fdrv(1, 9) = t86*tdx+t27*tdy+t76*tdz;
    fdrv(1, 10) = t7*tdx*-2.0+t86*tdy+t21*tdz; fdrv(1, 11) = -t76*tdx-t2*tdy*2.0+t86*tdz; fdrv(1, 12) = t96+t98+t110; fdrv(1, 14) = t95+t97-t108; fdrv(1, 15) = oz*t101+t23*tx; fdrv(1, 16) = t75*ty*2.0; fdrv(1, 17) = -ox*t103-t5*tz*2.0;
    fdrv(1, 18) = rdx*t86-rdy*t7*2.0-rdz*t76; fdrv(1, 19) = rdx*t27+rdy*t75*2.0-rdz*t2*2.0; fdrv(1, 20) = rdx*t76+rdy*t21+rdz*t86; fdrv(1, 24) = t54+t56-t113; fdrv(1, 26) = -t53-t55+t111; fdrv(2, 0) = t66+t67-t79*ty+rx*tx*ty-rz*ty*tz*2.0;
    fdrv(2, 1) = -t64-t65+t79*tx+t35*tz-ry*tx*ty; fdrv(2, 3) = t106-ox*t88; fdrv(2, 4) = -t105-oy*t88; fdrv(2, 5) = t93+t104; fdrv(2, 6) = t70*tdx-t82*tdy+t17*tdz; fdrv(2, 7) = -t82*tdx-t70*tdy-ox*rz*tdz*2.0; fdrv(2, 8) = t17*tdx+tdz*(t12-t13)-ox*rz*tdy*2.0;
    fdrv(2, 9) = -t85*tdx-t74*tdy-t6*tdz*2.0; fdrv(2, 10) = t74*tdx-t85*tdy+t23*tdz; fdrv(2, 11) = t26*tdx-t4*tdy*2.0-t85*tdz; fdrv(2, 12) = -t94+t99-t109; fdrv(2, 13) = -t95-t97+t108; fdrv(2, 15) = -oy*t101-t2*tx*2.0; fdrv(2, 16) = ox*t102+t22*ty;
    fdrv(2, 17) = t73*tz*-2.0; fdrv(2, 18) = -rdx*t85+rdy*t74+rdz*t26; fdrv(2, 19) = -rdx*t74-rdy*t85-rdz*t4*2.0; fdrv(2, 20) = rdx*t6*-2.0+rdy*t23-rdz*t73*2.0; fdrv(2, 24) = -t52-t57+t112; fdrv(2, 25) = t53+t55-t111;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f04(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = oy*ty, t4 = oz*tz, t5 = rdx*tx, t6 = rdy*ty, t7 = rdz*tz, t8 = rx*tx, t9 = ry*ty, t10 = rz*tz, t11 = tx*tx, t12 = ty*ty, t13 = tz*tz, t23 = rx*ty*tz, t24 = ry*tx*tz, t25 = rz*tx*ty, t38 = ox*rx*ty*4.0;
    T t44 = ox*rx*tz*4.0, t46 = oy*ry*tx*4.0, t48 = ox*ry*tz*2.0, t49 = ox*rz*ty*2.0, t50 = oy*rx*tz*2.0, t51 = oy*rz*tx*2.0, t52 = oz*rx*ty*2.0, t53 = oz*ry*tx*2.0, t54 = ox*ry*tz*4.0, t55 = ox*rz*ty*4.0, t56 = oy*rx*tz*4.0, t58 = oy*rz*tx*4.0;
    T t59 = oz*rx*ty*4.0, t60 = oz*ry*tx*4.0, t66 = oy*ry*tz*4.0, t68 = oz*rz*tx*4.0, t72 = oz*rz*ty*4.0, t80 = ox*ty*tz*2.0, t81 = oy*tx*tz*2.0, t82 = oz*tx*ty*2.0, t83 = ox*ty*tz*4.0, t84 = oy*tx*tz*4.0, t85 = oz*tx*ty*4.0, t14 = t2*ty;
    T t15 = t2*tz, t16 = t3*tx, t17 = t3*tz, t18 = t4*tx, t19 = t4*ty, t20 = t8*ty, t21 = t8*tz, t22 = t9*tx, t26 = t9*tz, t27 = t10*tx, t28 = t10*ty, t29 = ox*t12, t30 = oy*t11, t31 = ox*t13, t32 = oz*t11, t33 = oy*t13, t34 = oz*t12;
    T t35 = rx*t2*4.0, t36 = ry*t2*2.0, t37 = oy*t8*2.0, t39 = oy*t8*4.0, t40 = ox*t9*2.0, t41 = rz*t2*2.0, t42 = rx*t3*2.0, t43 = oz*t8*2.0, t45 = ox*t9*4.0, t47 = oz*t8*4.0, t57 = ry*t3*4.0, t61 = ox*t10*2.0, t62 = rz*t3*2.0, t63 = rx*t4*2.0;
    T t64 = oz*t9*2.0, t65 = ox*t10*4.0, t67 = oz*t9*4.0, t69 = oy*t10*2.0, t70 = ry*t4*2.0, t71 = oy*t10*4.0, t73 = rz*t4*4.0, t116 = t23*tdx*2.0, t117 = t24*tdy*2.0, t118 = t25*tdz*2.0, t119 = t2*tx*2.0, t123 = t3*ty*2.0, t127 = t4*tz*2.0;
    T t128 = t8*tx*2.0, t129 = t8*tx*6.0, t130 = t9*ty*2.0, t131 = t9*ty*6.0, t132 = t10*tz*2.0, t133 = t10*tz*6.0, t134 = -t38, t135 = -t44, t136 = -t46, t139 = -t56, t140 = -t58, t141 = -t59, t142 = -t60, t143 = -t66, t144 = -t68, t145 = -t72;
    T t153 = -t85, t160 = -t24, t161 = -t25, t162 = t11+t12, t163 = t11+t13, t164 = t12+t13, t165 = t2*t5*2.0, t166 = t3*t6*2.0, t167 = t4*t7*2.0, t168 = t2*t8*2.0, t169 = t3*t9*2.0, t170 = t4*t10*2.0, t171 = t8+t9, t172 = t8+t10, t173 = t9+t10;
    T t180 = t5+t6+t7, t74 = t14*2.0, t75 = t14*4.0, t76 = t15*2.0, t77 = t16*2.0, t78 = t15*4.0, t79 = t16*4.0, t86 = t17*2.0, t87 = t18*2.0, t88 = t17*4.0, t89 = t18*4.0, t90 = t19*2.0, t91 = t19*4.0, t92 = t20*2.0, t93 = t20*4.0, t94 = t21*2.0;
    T t95 = t22*2.0, t96 = t21*4.0, t97 = t22*4.0, t98 = t26*2.0, t99 = t27*2.0, t100 = t26*4.0, t101 = t27*4.0, t102 = t28*2.0, t103 = t28*4.0, t120 = t29*2.0, t121 = t30*2.0, t122 = t31*2.0, t124 = t32*2.0, t125 = t33*2.0, t126 = t34*2.0;
    T t174 = rx*t162, t175 = rx*t163, t176 = ry*t162, t177 = ry*t164, t178 = rz*t163, t179 = rz*t164, t181 = t23+t160, t182 = t23+t161, t183 = t24+t161, t184 = t180*tx*ty*2.0, t185 = t180*tx*tz*2.0, t186 = t180*ty*tz*2.0, t208 = t48+t50+t141+t142;
    T t209 = t49+t52+t139+t140, t211 = t35+t40+t42+t65+t144, t212 = t35+t45+t61+t63+t136, t213 = t36+t37+t57+t71+t145, t214 = t39+t57+t69+t70+t134, t215 = t41+t43+t67+t73+t143, t216 = t47+t62+t64+t73+t135, t104 = rdx*t75, t105 = rdx*t78;
    T t106 = rdy*t79, t107 = rdy*t88, t108 = rdz*t89, t109 = rdz*t91, t110 = rx*t75, t111 = rx*t78, t112 = ry*t79, t113 = ry*t88, t114 = rz*t89, t115 = rz*t91, t147 = -t75, t150 = -t78, t151 = -t79, t156 = -t88, t157 = -t89, t159 = -t91;
    T t187 = t22+t175, t188 = t27+t174, t189 = t20+t177, t190 = t28+t176, t191 = t21+t179, t192 = t26+t178, t193 = t95+t101+t128, t194 = t97+t99+t128, t195 = t92+t103+t130, t196 = t93+t102+t130, t197 = t94+t100+t132, t198 = t96+t98+t132;
    T t199 = t194*tdy, t200 = t193*tdz, t201 = t196*tdx, t202 = t195*tdz, t203 = t198*tdx, t204 = t197*tdy, t217 = t118+t199+t201, t218 = t117+t200+t203, t219 = t116+t202+t204;
    
    fdrv(0, 0) = -t8*t12-t8*t13-t164*t173-t189*ty-t191*tz;
    fdrv(0, 1) = t189*tx+t175*ty-t183*tz-rx*t13*ty+t173*tx*ty; fdrv(0, 2) = t191*tx+t183*ty+t174*tz-rx*t12*tz+t173*tx*tz; fdrv(0, 3) = t169+t170-rx*t29*2.0-rx*t31*2.0+t3*t8*4.0+t4*t8*4.0+t3*t10*2.0+t4*t9*2.0;
    fdrv(0, 4) = t112-ox*t28*4.0+oy*t99-rx*t14*4.0-ry*t31*2.0+ry*t87+t37*tx-ox*t9*ty*6.0; fdrv(0, 5) = t114-ox*t26*4.0+oz*t95-rx*t15*4.0-rz*t29*2.0+rz*t77+t43*tx-ox*t10*tz*6.0; fdrv(0, 6) = t214*tdy+t216*tdz+rx*tdx*(t3+t4)*4.0;
    fdrv(0, 7) = t214*tdx+tdz*(t51+t53-t54-t55)-tdy*(t35+t65+t136+ox*t9*1.2E+1); fdrv(0, 8) = t216*tdx+tdy*(t51+t53-t54-t55)-tdz*(t35+t45+t144+ox*t10*1.2E+1); fdrv(0, 9) = -tdy*(t75-t121)-tdz*(t78-t124)+tdx*(t79+t89-t120-t122);
    fdrv(0, 10) = tdx*(t90+t123)+tdz*(t82-t83)-tdy*(t29*6.0-t87+t122+t151); fdrv(0, 11) = tdx*(t86+t127)+tdy*(t81-t83)-tdz*(t31*6.0-t77+t120+t157); fdrv(0, 12) = -tdy*(t93+t103+t131+ry*t13*2.0)-tdz*(t96+t100+t133+rz*t12*2.0)-rx*t164*tdx*2.0;
    fdrv(0, 13) = t217; fdrv(0, 14) = t218; fdrv(0, 15) = tx*(t16+t18-t29-t31)*2.0; fdrv(0, 16) = ty*(t16+t18-t29-t31)*2.0; fdrv(0, 17) = tz*(t16+t18-t29-t31)*2.0; fdrv(0, 18) = t166+t167-rdx*t29*2.0-rdx*t31*2.0+t3*t5*4.0+t4*t5*4.0+t3*t7*2.0+t4*t6*2.0;
    fdrv(0, 19) = t106-rdx*t14*4.0-rdy*t31*2.0+rdy*t87+oy*t5*tx*2.0-ox*t6*ty*6.0-ox*t7*ty*4.0+oy*t7*tx*2.0; fdrv(0, 20) = t108-rdx*t15*4.0-rdz*t29*2.0+rdz*t77+oz*t5*tx*2.0-ox*t6*tz*4.0+oz*t6*tx*2.0-ox*t7*tz*6.0; fdrv(0, 24) = t164*t180*-2.0;
    fdrv(0, 25) = t184; fdrv(0, 26) = t185; fdrv(1, 0) = t177*tx+t187*ty-t182*tz-ry*t13*tx+t172*tx*ty; fdrv(1, 1) = -t9*t11-t9*t13-t163*t172-t187*tx-t192*tz; fdrv(1, 2) = t182*tx+t192*ty+t176*tz-ry*t11*tz+t172*ty*tz;
    fdrv(1, 3) = t110+ox*t102-oy*t27*4.0-rx*t33*2.0+rx*t90-ry*t16*4.0+t40*ty-oy*t8*tx*6.0; fdrv(1, 4) = t168+t170-ry*t30*2.0-ry*t33*2.0+t2*t9*4.0+t2*t10*2.0+t4*t8*2.0+t4*t9*4.0;
    fdrv(1, 5) = t115-oy*t21*4.0+oz*t92-ry*t17*4.0-rz*t30*2.0+rz*t74+t64*ty-oy*t10*tz*6.0; fdrv(1, 6) = t212*tdy+t209*tdz-tdx*(t57+t71+t134+oy*t8*1.2E+1); fdrv(1, 7) = t212*tdx+t215*tdz+ry*tdy*(t2+t4)*4.0;
    fdrv(1, 8) = t209*tdx+t215*tdy-tdz*(t39+t57+t145+oy*t10*1.2E+1); fdrv(1, 9) = tdy*(t87+t119)+tdz*(t82-t84)-tdx*(t30*6.0-t90+t125+t147); fdrv(1, 10) = -tdx*(t79-t120)-tdz*(t88-t126)+tdy*(t75+t91-t121-t125);
    fdrv(1, 11) = tdy*(t76+t127)+tdx*(t80-t84)-tdz*(t33*6.0-t74+t121+t159); fdrv(1, 12) = t217; fdrv(1, 13) = -tdx*(t97+t101+t129+rx*t13*2.0)-tdz*(t96+t100+t133+rz*t11*2.0)-ry*t163*tdy*2.0; fdrv(1, 14) = t219; fdrv(1, 15) = tx*(t14+t19-t30-t33)*2.0;
    fdrv(1, 16) = ty*(t14+t19-t30-t33)*2.0; fdrv(1, 17) = tz*(t14+t19-t30-t33)*2.0; fdrv(1, 18) = t104-rdx*t33*2.0+rdx*t90-rdy*t16*4.0-oy*t5*tx*6.0+ox*t6*ty*2.0+ox*t7*ty*2.0-oy*t7*tx*4.0;
    fdrv(1, 19) = t165+t167-rdy*t30*2.0-rdy*t33*2.0+t2*t6*4.0+t2*t7*2.0+t4*t5*2.0+t4*t6*4.0; fdrv(1, 20) = t109-rdy*t17*4.0-rdz*t30*2.0+rdz*t74-oy*t5*tz*4.0+oz*t5*ty*2.0+oz*t6*ty*2.0-oy*t7*tz*6.0; fdrv(1, 24) = t184; fdrv(1, 25) = t163*t180*-2.0;
    fdrv(1, 26) = t186; fdrv(2, 0) = t179*tx-t181*ty+t188*tz-rz*t12*tx+t171*tx*tz; fdrv(2, 1) = t181*tx+t178*ty+t190*tz-rz*t11*ty+t171*ty*tz; fdrv(2, 2) = -t10*t11-t10*t12-t162*t171-t188*tx-t190*ty;
    fdrv(2, 3) = t111+ox*t98-oz*t22*4.0-rx*t34*2.0+rx*t86-rz*t18*4.0+t61*tz-oz*t8*tx*6.0; fdrv(2, 4) = t113+oy*t94-oz*t20*4.0-ry*t32*2.0+ry*t76-rz*t19*4.0+t69*tz-oz*t9*ty*6.0;
    fdrv(2, 5) = t168+t169-rz*t32*2.0-rz*t34*2.0+t2*t9*2.0+t3*t8*2.0+t2*t10*4.0+t3*t10*4.0; fdrv(2, 6) = t208*tdy+t211*tdz-tdx*(t67+t73+t135+oz*t8*1.2E+1); fdrv(2, 7) = t208*tdx+t213*tdz-tdy*(t47+t73+t143+oz*t9*1.2E+1);
    fdrv(2, 8) = t211*tdx+t213*tdy+rz*tdz*(t2+t3)*4.0; fdrv(2, 9) = tdy*(t81+t153)+tdz*(t77+t119)-tdx*(t32*6.0-t86+t126+t150); fdrv(2, 10) = tdx*(t80+t153)+tdz*(t74+t123)-tdy*(t34*6.0-t76+t124+t156);
    fdrv(2, 11) = -tdx*(t89-t122)-tdy*(t91-t125)+tdz*(t78+t88-t124-t126); fdrv(2, 12) = t218; fdrv(2, 13) = t219; fdrv(2, 14) = -tdx*(t97+t101+t129+rx*t12*2.0)-tdy*(t93+t103+t131+ry*t11*2.0)-rz*t162*tdz*2.0; fdrv(2, 15) = tx*(t15+t17-t32-t34)*2.0;
    fdrv(2, 16) = ty*(t15+t17-t32-t34)*2.0; fdrv(2, 17) = tz*(t15+t17-t32-t34)*2.0; fdrv(2, 18) = t105-rdx*t34*2.0+rdx*t86-rdz*t18*4.0-oz*t5*tx*6.0+ox*t6*tz*2.0-oz*t6*tx*4.0+ox*t7*tz*2.0;
    fdrv(2, 19) = t107-rdy*t32*2.0+rdy*t76-rdz*t19*4.0+oy*t5*tz*2.0-oz*t5*ty*4.0-oz*t6*ty*6.0+oy*t7*tz*2.0; fdrv(2, 20) = t165+t166-rdz*t32*2.0-rdz*t34*2.0+t2*t6*2.0+t3*t5*2.0+t2*t7*4.0+t3*t7*4.0; fdrv(2, 24) = t185; fdrv(2, 25) = t186;
    fdrv(2, 26) = t162*t180*-2.0;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f05(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*rdx, t3 = oy*rdy, t4 = oz*rdz, t5 = ox*rx, t6 = oy*ry, t7 = oz*rz, t8 = ox*tdy, t9 = ox*tdz, t10 = oy*tdx, t11 = oy*tdz, t12 = oz*tdx, t13 = oz*tdy, t14 = ox*ty, t15 = oy*tx, t16 = ox*tz, t17 = oz*tx, t18 = oy*tz, t19 = oz*ty;
    T t20 = rdx*tx, t21 = rx*tdx, t22 = rdy*ty, t23 = ry*tdy, t24 = rdz*tz, t25 = rz*tdz, t26 = rx*tx, t27 = ry*ty, t28 = rz*tz, t29 = -t20, t30 = -t21, t31 = -t22, t32 = -t23, t33 = -t24, t34 = -t25, t35 = -t26, t36 = -t27, t37 = -t28;
    
    fdrv(0, 0) = t36+t37; fdrv(0, 1) = ry*tx; fdrv(0, 2) = rz*tx; fdrv(0, 3) = t6+t7; fdrv(0, 4) = -ox*ry; fdrv(0, 5) = -ox*rz; fdrv(0, 10) = -t8+t10; fdrv(0, 11) = -t9+t12; fdrv(0, 12) = t32+t34; fdrv(0, 13) = ry*tdx; fdrv(0, 14) = rz*tdx; fdrv(0, 16) = -t14+t15;
    fdrv(0, 17) = -t16+t17; fdrv(0, 18) = t3+t4; fdrv(0, 19) = -ox*rdy; fdrv(0, 20) = -ox*rdz; fdrv(0, 24) = t31+t33; fdrv(0, 25) = rdy*tx; fdrv(0, 26) = rdz*tx; fdrv(1, 0) = rx*ty; fdrv(1, 1) = t35+t37; fdrv(1, 2) = rz*ty; fdrv(1, 3) = -oy*rx; fdrv(1, 4) = t5+t7;
    fdrv(1, 5) = -oy*rz; fdrv(1, 9) = t8-t10; fdrv(1, 11) = -t11+t13; fdrv(1, 12) = rx*tdy; fdrv(1, 13) = t30+t34; fdrv(1, 14) = rz*tdy; fdrv(1, 15) = t14-t15; fdrv(1, 17) = -t18+t19; fdrv(1, 18) = -oy*rdx; fdrv(1, 19) = t2+t4; fdrv(1, 20) = -oy*rdz;
    fdrv(1, 24) = rdx*ty; fdrv(1, 25) = t29+t33; fdrv(1, 26) = rdz*ty; fdrv(2, 0) = rx*tz; fdrv(2, 1) = ry*tz; fdrv(2, 2) = t35+t36; fdrv(2, 3) = -oz*rx; fdrv(2, 4) = -oz*ry; fdrv(2, 5) = t5+t6; fdrv(2, 9) = t9-t12; fdrv(2, 10) = t11-t13; fdrv(2, 12) = rx*tdz;
    fdrv(2, 13) = ry*tdz; fdrv(2, 14) = t30+t32; fdrv(2, 15) = t16-t17; fdrv(2, 16) = t18-t19; fdrv(2, 18) = -oz*rdx; fdrv(2, 19) = -oz*rdy; fdrv(2, 20) = t2+t3; fdrv(2, 24) = rdx*tz; fdrv(2, 25) = rdy*tz; fdrv(2, 26) = t29+t31;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f06(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*ry, t3 = oy*rx, t4 = ox*rz, t5 = oz*rx, t6 = oy*rz, t7 = oz*ry, t8 = ox*tx, t9 = ox*ty, t10 = oy*tx, t11 = ox*tz, t12 = oy*ty, t13 = oz*tx, t14 = oy*tz, t15 = oz*ty, t16 = oz*tz, t17 = rx*tx, t18 = rx*ty, t19 = ry*tx;
    T t20 = rx*tz, t21 = ry*ty, t22 = rz*tx, t23 = ry*tz, t24 = rz*ty, t25 = rz*tz, t26 = tx*ty, t27 = tx*tz, t28 = ty*tz, t29 = rx*2.0, t30 = ry*2.0, t31 = rz*2.0, t32 = tx*2.0, t33 = tx*4.0, t34 = ty*2.0, t35 = ty*4.0, t36 = tz*2.0;
    T t37 = tz*4.0, t38 = tx*tx, t39 = tx*tx*tx, t40 = ty*ty, t41 = ty*ty*ty, t42 = tz*tz, t43 = tz*tz*tz, t44 = ox*t29, t45 = oy*t30, t46 = oz*t31, t47 = t8*2.0, t48 = t9*2.0, t49 = t10*2.0, t50 = t11*2.0, t51 = t12*2.0, t52 = t13*2.0;
    T t53 = t14*2.0, t54 = t15*2.0, t55 = t16*2.0, t56 = t17*2.0, t57 = t17*4.0, t58 = t18*2.0, t59 = t19*2.0, t60 = t20*2.0, t61 = t21*2.0, t62 = t22*2.0, t63 = t21*4.0, t64 = t23*2.0, t65 = t24*2.0, t66 = t25*2.0, t67 = t25*4.0, t68 = t26*2.0;
    T t69 = t27*2.0, t70 = t28*2.0, t71 = t2*tx, t72 = t3*tx, t73 = t2*ty, t74 = t4*tx, t75 = t3*ty, t76 = t5*tx, t77 = t2*tz, t78 = t4*ty, t79 = t3*tz, t80 = t6*tx, t81 = t5*ty, t82 = t7*tx, t83 = t4*tz, t84 = t6*ty, t85 = t5*tz, t86 = t7*ty;
    T t87 = t6*tz, t88 = t7*tz, t89 = t8*ty, t90 = t8*tz, t91 = t10*ty, t92 = t9*tz, t93 = t10*tz, t94 = t13*ty, t95 = t12*tz, t96 = t13*tz, t97 = t15*tz, t98 = t17*ty, t99 = t17*tz, t100 = t19*ty, t101 = t18*tz, t102 = t19*tz, t103 = t22*ty;
    T t104 = t21*tz, t105 = t22*tz, t106 = t24*tz, t107 = rx+t23, t108 = ry+t22, t109 = rz+t18, t110 = t28+tx, t111 = t27+ty, t112 = t26+tz, t113 = t32*tx, t114 = t38*3.0, t115 = t34*ty, t116 = t40*3.0, t117 = t36*tz, t118 = t42*3.0, t121 = -t9;
    T t122 = -t10, t124 = -t13, t125 = -t14, t126 = -t15, t128 = -t19, t130 = -t20, t132 = -t24, t134 = -t26, t136 = -t27, t138 = -t28, t140 = t8*tx, t141 = t9*ty, t142 = t10*tx, t143 = t11*tz, t144 = t12*ty, t145 = t13*tx, t146 = t14*tz;
    T t147 = t15*ty, t148 = t16*tz, t149 = t17*tx, t150 = t21*ty, t151 = t25*tz, t152 = t26*ty, t155 = t27*tx, t156 = t28*tz, t158 = t8*t29, t160 = t3*t32, t161 = t11*t29, t162 = t2*t34, t163 = t10*t30, t164 = t5*t32, t167 = t3*t36;
    T t168 = t12*t30, t169 = t6*t32, t172 = t4*t36, t174 = t7*t34, t176 = t6*t36, t177 = t15*t31, t178 = t16*t31, t179 = t8*t34, t180 = t8*t36, t182 = t9*t36, t183 = t10*t36, t184 = t13*t34, t185 = t12*t36, t186 = t13*t36, t188 = t17*t34;
    T t189 = t17*t36, t190 = t19*t34, t191 = t21*t36, t192 = t22*t36, t193 = t24*t36, t206 = rx*t9*-2.0, t207 = rx*t11*-2.0, t208 = ry*t10*-2.0, t215 = ry*t14*-2.0, t216 = rz*t13*-2.0, t217 = rz*t15*-2.0, t227 = t38+t40, t228 = t38+t42;
    T t229 = t40+t42, t239 = t28+t32, t240 = t27+t34, t241 = t26+t36, t249 = t17+t21, t254 = t17+t25, t257 = t21+t25, t286 = -tx*(t28-tx), t287 = -tx*(t27-ty), t288 = -ty*(t27-ty), t289 = -ty*(t26-tz), t290 = -tz*(t28-tx), t291 = -tz*(t26-tz);
    T t306 = tx*(t27-ty), t307 = ty*(t26-tz), t308 = tz*(t28-tx), t315 = ox*(t27-t34), t316 = oy*(t26-t36), t317 = oz*(t28-t32), t123 = -t49, t127 = -t54, t129 = -t59, t131 = -t60, t133 = -t65, t135 = -t68, t137 = -t69, t139 = -t70, t194 = t70+tx;
    T t195 = t69+ty, t196 = t68+tz, t197 = t18*t47, t198 = t20*t47, t199 = t21*t49, t200 = t23*t51, t201 = t25*t52, t202 = t25*t54, t203 = t149*3.0, t204 = t150*3.0, t205 = t151*3.0, t209 = -t78, t210 = -t81, t211 = t77*-2.0, t212 = t78*-2.0;
    T t213 = t81*-2.0, t214 = t82*-2.0, t218 = -t89, t219 = -t92, t220 = -t93, t221 = -t94, t222 = -t97, t223 = t97*-2.0, t224 = -t99, t225 = -t100, t226 = -t106, t230 = rx+t132, t231 = ry+t130, t232 = rz+t128, t242 = t121*ty, t243 = -t143;
    T t244 = t124*tx, t245 = t126*ty, t246 = t134*tx, t247 = t136*tz, t248 = t138*ty, t250 = t108*tx, t251 = t109*tx, t252 = t107*ty, t253 = t109*ty, t255 = t107*tz, t256 = t108*tz, t258 = t110*tx, t259 = t112*tx, t260 = t110*ty, t261 = t111*ty;
    T t262 = t111*tz, t263 = t112*tz, t264 = ox*t241, t265 = oy*t239, t266 = oz*t240, t270 = t249*tx, t271 = t249*ty, t272 = t254*tx, t273 = t254*tz, t274 = t257*ty, t275 = t257*tz, t276 = t56+t61, t281 = t56+t66, t285 = t61+t66, t292 = t68+t155;
    T t293 = t70+t152, t294 = t69+t156, t321 = t39+t307, t322 = t43+t306, t323 = t41+t308, t336 = t15+t140+t316, t337 = t10+t148+t315, t338 = t11+t144+t317, t277 = t231*tx, t278 = t232*tx, t279 = t230*ty, t280 = t232*ty, t282 = t230*tz;
    T t283 = t231*tz, t295 = t41+t259, t296 = t39+t262, t297 = t43+t260, t301 = -t271, t302 = -t272, t303 = -t275, t309 = t17+t285, t310 = t21+t281, t311 = t25+t276, t312 = t69+t246, t313 = t68+t248, t314 = t70+t247, t318 = t113+t261;
    T t319 = t117+t258, t320 = t115+t263, t330 = t18+t19+t104+t273, t331 = t20+t22+t98+t274, t332 = t23+t24+t105+t270, t333 = t124+t144+t264, t334 = t125+t140+t266, t335 = t121+t148+t265, t347 = t9+t90+t95+t122+t244+t245;
    T t348 = t11+t124+t142+t146+t218+t222, t349 = t14+t91+t96+t126+t242+t243, t304 = -t279, t305 = -t283, t339 = t18+t19+t224+t303, t340 = t20+t22+t226+t301, t341 = t23+t24+t225+t302, t342 = t251+t331, t343 = t252+t330, t344 = t256+t332;
    T t327 = t250+t304, t329 = t253+t305, t351 = t277+t339, t352 = t280+t341, t353 = t282+t340;
    
    fdrv(0, 0) = t331*ty-t339*tz; fdrv(0, 1) = -t331*tx-t285*tz; fdrv(0, 2) = t339*tx+t285*ty; fdrv(0, 3) = ox*t329-oy*t342+oz*t351;
    fdrv(0, 4) = t74+t76+t178+t197+t215-t10*t21*2.0-t3*t38+t7*t35+t2*t42+t4*t70+t2*t116+t6*t136+t7*t136; fdrv(0, 5) = -t71-t72-t87*4.0+t177+t198-ry*t12*2.0-t13*t25*2.0-t5*t38+t4*t40+t2*t70+t4*t118+t6*t134+t7*t134;
    fdrv(0, 6) = -tdy*(-t4-t5+t87+t88+t160+t168+t206)-tdz*(t2+t3+t84+t86+t164+t178+t207)-tdx*(oy*(t31+t58)-oz*(t30+t131)); fdrv(0, 7) = tdy*(t7*4.0+t73*6.0+t158+t172+t208)-tdz*(t45-t46+t80+t82+t211+t212)+tdx*(ox*(rz+t58)-oy*t311+oz*(rx-t23));
    fdrv(0, 8) = tdz*(t6*-4.0+t83*6.0+t158+t162+t216)-tdy*(t45-t46+t80+t82+t211+t212)-tdx*(oy*(rx+t24)+ox*(ry+t131)+oz*t310); fdrv(0, 9) = -tdx*(-ox*t229+oy*t196+oz*(t69-ty))+tdy*(t13+t179+t122*tx)-tdz*(t10-t90*2.0+t145);
    fdrv(0, 10) = tdy*(t15*4.0-t53-t96+t141*3.0+t143+t34*t122)-t338*tdx-tdz*(t8+t51-t92*2.0+t94); fdrv(0, 11) = tdy*(t8+t55+t182+t220)-t335*tdx-tdz*(t14*4.0+t91+t127-t143*3.0+t186+t242);
    fdrv(0, 12) = t329*tdx+tdz*(t128+t189+t191+t205+t24*ty)+tdy*(t22+t188+t193+t204+t23*tz); fdrv(0, 13) = -tdz*(t17+t61+t67+t103)-tdy*(t64+t105+t149+t190)-t342*tdx; fdrv(0, 14) = -tdz*(t100+t133+t149+t192)+t351*tdx+tdy*(t17+t63+t66-t102);
    fdrv(0, 15) = -t349*tx; fdrv(0, 16) = ox*t323-oy*t293+oz*(t40*2.0+t286); fdrv(0, 17) = ox*t297-oy*t319+oz*t314; fdrv(0, 18) = -rdx*t349-rdy*t338-rdz*t335-rdx*tx*(t12+t16);
    fdrv(0, 19) = rdz*(t55+t220+ox*t194)-rdy*(-ox*(t42+t116)+oy*(t36+t68)+oz*(t27-t35))+rdx*tx*(oz+t48+t122); fdrv(0, 20) = rdz*(ox*(t40+t118)-oy*(t26+t37)+oz*(t34+t137))-rdy*(t51+t94-ox*(t70-tx))-rdx*tx*(oy+t13-t50);
    fdrv(0, 24) = rdy*t323+rdz*t297+rdx*t229*tx; fdrv(0, 25) = -rdx*t259-rdy*t293-rdz*t319; fdrv(0, 26) = rdx*t287+rdz*t314+rdy*(t40*2.0+t286); fdrv(1, 0) = t341*ty+t281*tz; fdrv(1, 1) = -t341*tx+t330*tz; fdrv(1, 2) = -t281*tx-t330*ty;
    fdrv(1, 3) = t76*-4.0-t84-t86+t161+t199-rz*t16*2.0-t8*t18*2.0-t2*t40+t3*t42+t6*t69+t3*t114+t4*t138+t5*t138; fdrv(1, 4) = ox*t352-oz*t343+oy*(t255-t278);
    fdrv(1, 5) = t73+t75+t158+t200+t216-t15*t25*2.0+t4*t37+t6*t38-t7*t40+t3*t69+t6*t118+t4*t134+t5*t134; fdrv(1, 6) = tdx*(t5*-4.0+t72*6.0+t168+t176+t206)+tdz*(t44-t46+t167+t169+t209+t210)-tdy*(oz*(ry+t20)+oy*(rz+t129)+ox*t311);
    fdrv(1, 7) = -tdz*(-t2-t3+t74+t76+t174+t178+t215)-tdx*(t6+t7+t83+t85+t158+t162+t208)+tdy*(ox*(t31+t129)-oz*(t29+t64)); fdrv(1, 8) = tdz*(t4*4.0+t87*6.0+t160+t168+t217)+tdx*(t44-t46+t167+t169+t209+t210)+tdy*(oy*(rx+t64)-oz*t309+ox*(ry-t22));
    fdrv(1, 9) = tdz*(t12+t47+t183+t221)-tdx*(t13*4.0-t50+t89*2.0+t97-t142*3.0+t125*tz)-t334*tdy; fdrv(1, 10) = -tdy*(-oy*t228+oz*t194+ox*(t68-tz))+tdz*(t9+t185+t245)-tdx*(t15-t91*2.0+t141);
    fdrv(1, 11) = tdz*(t11*4.0-t52+t142+t146*3.0+t218+t223)-t337*tdy-tdx*(t12+t55+t92-t93*2.0); fdrv(1, 12) = -tdx*(t106+t131+t150+t188)+t352*tdy+tdz*(t21+t56+t67-t103);
    fdrv(1, 13) = tdz*(t18+t189+t191+t205+t22*tx)+tdx*(t132+t190+t192+t203+t20*tz)+tdy*(t255-t278); fdrv(1, 14) = -tdx*(t21+t57+t66+t101)-tdz*(t62+t98+t150+t193)-t343*tdy; fdrv(1, 15) = ox*t312+oy*t296-oz*t318; fdrv(1, 16) = t348*ty;
    fdrv(1, 17) = oy*t322-oz*t294+ox*(t42*2.0+t288); fdrv(1, 18) = rdx*(ox*(t36+t135)+oy*(t42+t114)-oz*(t28+t33))-rdz*(t55+t92-oy*(t69-ty))-rdy*ty*(oz+t9+t123); fdrv(1, 19) = -rdx*t334+rdy*t348-rdz*t337-rdy*ty*(t8+t16);
    fdrv(1, 20) = rdx*(t47+t221+oy*t195)-rdz*(-oy*(t38+t118)+oz*(t32+t70)+ox*(t26-t37))+rdy*ty*(ox+t53+t126); fdrv(1, 24) = rdx*t312+rdy*t289+rdz*(t42*2.0+t288); fdrv(1, 25) = rdx*t296+rdz*t322+rdy*t228*ty; fdrv(1, 26) = -rdx*t318-rdy*t260-rdz*t294;
    fdrv(2, 0) = -t276*ty-t332*tz; fdrv(2, 1) = t276*tx+t340*tz; fdrv(2, 2) = t332*tx-t340*ty; fdrv(2, 3) = t87+t88+t168+t201+t206-t8*t20*2.0+t3*t33+t5*t40-t4*t42+t7*t68+t5*t114+t2*t138+t3*t138;
    fdrv(2, 4) = t73*-4.0-t83-t85+t163+t202-rx*t8*2.0-t12*t23*2.0+t7*t38-t6*t42+t5*t68+t7*t116+t2*t136+t3*t136; fdrv(2, 5) = -ox*t344+oy*t353+oz*t327;
    fdrv(2, 6) = tdx*(t3*4.0+t76*6.0+t174+t178+t207)-tdy*(t44-t45+t77+t79+t213+t214)+tdz*(oz*(ry+t62)-ox*t310+oy*(rz-t18)); fdrv(2, 7) = tdy*(t2*-4.0+t86*6.0+t164+t178+t215)-tdx*(t44-t45+t77+t79+t213+t214)-tdz*(ox*(rz+t19)+oz*(rx+t133)+oy*t309);
    fdrv(2, 8) = -tdx*(-t6-t7+t73+t75+t158+t172+t216)-tdy*(t4+t5+t71+t72+t168+t176+t217)-tdz*(ox*(t30+t62)-oy*(t29+t133)); fdrv(2, 9) = -t336*tdz-tdy*(t16+t47+t93-t94*2.0)+tdx*(t10*4.0-t48-t95+t145*3.0+t147-t180);
    fdrv(2, 10) = tdx*(t16+t51+t184+t219)-t333*tdz-tdy*(t9*4.0+t90+t123-t147*3.0+t185+t244); fdrv(2, 11) = -tdz*(ox*t195-oz*t227+oy*(t70-tx))+tdx*(t14+t186+t243)-tdy*(t11+t146+t223);
    fdrv(2, 12) = -tdx*(t58+t104+t151+t189)-tdy*(t25+t56+t63+t102)-t344*tdz; fdrv(2, 13) = -tdy*(t99+t129+t151+t191)+t353*tdz+tdx*(t25+t57+t61-t101); fdrv(2, 14) = t327*tdz+tdy*(t130+t188+t193+t204+t19*tx)+tdx*(t23+t190+t192+t203+t18*ty);
    fdrv(2, 15) = -ox*t292+oz*t321+oy*(t38*2.0+t291); fdrv(2, 16) = -ox*t320+oy*t313+oz*t295; fdrv(2, 17) = -t347*tz; fdrv(2, 18) = rdy*(t51+t219+oz*t196)-rdx*(ox*(t34+t69)-oz*(t40+t114)+oy*(t28-t33))+rdz*tz*(oy-t11+t52);
    fdrv(2, 19) = rdy*(-ox*(t27+t35)+oy*(t32+t139)+oz*(t38+t116))-rdx*(t47+t93-oz*(t68-tz))-rdz*tz*(ox+t14+t127); fdrv(2, 20) = -rdx*t336-rdy*t333-rdz*t347-rdz*tz*(t8+t12); fdrv(2, 24) = -rdx*t292-rdy*t320-rdz*t262;
    fdrv(2, 25) = rdy*t313+rdz*t290+rdx*(t38*2.0+t291); fdrv(2, 26) = rdx*t321+rdy*t295+rdz*t227*tz;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f07(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = rx*tx, t3 = ry*ty, t4 = rz*tz, t5 = tx*tx, t6 = ty*ty, t7 = tz*tz, t43 = ox*rx*ty*2.0, t44 = ox*ry*tx*2.0, t46 = ox*rx*ty*4.0, t48 = ox*rx*tz*2.0, t50 = ox*rz*tx*2.0, t51 = oy*rx*ty*2.0, t52 = oy*ry*tx*2.0, t54 = ox*rx*tz*4.0;
    T t56 = oy*ry*tx*4.0, t58 = ox*ry*tz*2.0, t59 = ox*rz*ty*2.0, t60 = oy*rx*tz*2.0, t62 = oy*rz*tx*2.0, t63 = oz*rx*ty*2.0, t64 = oz*ry*tx*2.0, t67 = oy*ry*tz*2.0, t68 = oy*rz*ty*2.0, t69 = oz*rx*tz*2.0, t71 = oz*rz*tx*2.0, t73 = oy*ry*tz*4.0;
    T t75 = oz*rz*tx*4.0, t77 = oz*ry*tz*2.0, t78 = oz*rz*ty*2.0, t80 = oz*rz*ty*4.0, t83 = ox*tx*ty*2.0, t84 = ox*tx*ty*4.0, t85 = ox*tx*tz*2.0, t86 = oy*tx*ty*2.0, t87 = ox*tx*tz*4.0, t88 = oy*tx*ty*4.0, t89 = ox*ty*tz*2.0, t90 = oy*tx*tz*2.0;
    T t91 = oz*tx*ty*2.0, t92 = oy*ty*tz*2.0, t93 = oz*tx*tz*2.0, t94 = oy*ty*tz*4.0, t95 = oz*tx*tz*4.0, t96 = oz*ty*tz*2.0, t97 = oz*ty*tz*4.0, t104 = rx*ty*tz*2.0, t105 = ry*tx*tz*2.0, t106 = rz*tx*ty*2.0, t134 = rdx*tx*ty*tz*2.0;
    T t135 = rdy*tx*ty*tz*2.0, t136 = rdz*tx*ty*tz*2.0, t202 = ox*rx*ty*tz*-2.0, t203 = oy*ry*tx*tz*-2.0, t204 = oz*rz*tx*ty*-2.0, t8 = t2*3.0, t9 = t3*3.0, t10 = t4*3.0, t11 = t2*ty, t12 = t2*tz, t13 = t3*tx, t14 = t3*tz, t15 = t4*tx;
    T t16 = t4*ty, t17 = t5*2.0, t18 = t5*3.0, t19 = t6*2.0, t20 = t6*3.0, t21 = t7*2.0, t22 = t7*3.0, t23 = ox*t5, t24 = ox*t6, t25 = oy*t5, t26 = ox*t7, t27 = oy*t6, t28 = oz*t5, t29 = oy*t7, t30 = oz*t6, t31 = oz*t7, t32 = t2*tx, t33 = rx*t6;
    T t34 = ry*t5, t35 = rx*t7, t36 = t3*ty, t37 = rz*t5, t38 = ry*t7, t39 = rz*t6, t40 = t4*tz, t41 = ox*t2*2.0, t42 = ox*t2*4.0, t45 = oy*t2*2.0, t47 = oy*t2*4.0, t49 = ox*t3*2.0, t53 = oz*t2*2.0, t55 = ox*t3*4.0, t57 = oz*t2*4.0;
    T t61 = oy*t3*2.0, t65 = oy*t3*4.0, t66 = ox*t4*2.0, t70 = oz*t3*2.0, t72 = ox*t4*4.0, t74 = oz*t3*4.0, t76 = oy*t4*2.0, t79 = oy*t4*4.0, t81 = oz*t4*2.0, t82 = oz*t4*4.0, t131 = t83*tz, t132 = t86*tz, t133 = t91*tz, t137 = -t5, t138 = -t6;
    T t139 = -t7, t161 = -t43, t162 = -t46, t163 = -t48, t164 = -t52, t165 = -t54, t166 = -t56, t167 = -t60, t168 = -t62, t169 = -t63, t170 = -t64, t171 = -t67, t172 = -t71, t173 = -t73, t174 = -t75, t175 = -t78, t176 = -t80, t183 = -t89;
    T t184 = -t90, t185 = -t91, t195 = -t106, t199 = t5+t6, t200 = t5+t7, t201 = t6+t7, t238 = t2+t3+t4, t98 = t11*2.0, t99 = t11*4.0, t100 = t12*2.0, t101 = t13*2.0, t102 = t12*4.0, t103 = t13*4.0, t107 = t14*2.0, t108 = t15*2.0, t109 = t14*4.0;
    T t110 = t15*4.0, t111 = t16*2.0, t112 = t16*4.0, t113 = ox*t32, t114 = oy*t36, t115 = oz*t40, t140 = ox*t19, t141 = oy*t17, t144 = ox*t21, t145 = oz*t17, t148 = oy*t21, t149 = oz*t19, t152 = t8*tx, t153 = t9*ty, t154 = t10*tz, t155 = t19*tx;
    T t156 = t17*ty, t157 = t21*tx, t158 = t17*tz, t159 = t21*ty, t160 = t19*tz, t205 = -t24, t206 = -t25, t207 = t24*-3.0, t208 = t25*-3.0, t209 = -t26, t210 = -t28, t211 = t26*-3.0, t212 = t28*-3.0, t213 = -t29, t214 = -t30, t215 = t29*-3.0;
    T t216 = t30*-3.0, t217 = -t33, t218 = -t34, t219 = -t35, t220 = -t37, t221 = -t38, t222 = -t39, t229 = rx*t199, t230 = rx*t200, t231 = ry*t199, t232 = ry*t201, t233 = rz*t200, t234 = rz*t201, t235 = t8+t9, t236 = t8+t10, t237 = t9+t10;
    T t245 = t137+t201, t246 = t138+t200, t247 = t139+t199, t269 = t90+t91+t183, t270 = t89+t91+t184, t271 = t89+t90+t185, t284 = t58+t59+t168+t170, t285 = t58+t60+t169+t170, t286 = t59+t63+t167+t168, t299 = t41+t49+t51+t69+t72+t164+t174;
    T t300 = t41+t51+t55+t66+t69+t166+t172, t301 = t44+t45+t61+t77+t79+t161+t176, t302 = t44+t47+t61+t76+t77+t162+t175, t303 = t50+t53+t68+t74+t81+t163+t173, t304 = t50+t57+t68+t70+t81+t165+t171, t116 = ox*t101, t117 = oy*t98, t119 = oy*t100;
    T t120 = oz*t98, t121 = ox*t107, t122 = ox*t108, t124 = oz*t100, t125 = oz*t101, t126 = ox*t111, t127 = oy*t108, t129 = oy*t111, t130 = oz*t107, t192 = -t98, t193 = -t100, t194 = -t101, t196 = -t107, t197 = -t108, t198 = -t111, t239 = t235*tx;
    T t240 = t235*ty, t241 = t236*tx, t242 = t236*tz, t243 = t237*ty, t244 = t237*tz, t248 = t155+t157, t249 = t156+t159, t250 = t158+t160, t263 = ox*t245, t264 = oy*t246, t265 = oz*t247, t266 = t245*tx, t267 = t246*ty, t268 = t247*tz;
    T t272 = t269*tdx, t273 = t270*tdy, t274 = t271*tdz, t287 = t23+t86+t93+t205+t211, t288 = t23+t86+t93+t207+t209, t289 = t27+t83+t96+t206+t215, t290 = t27+t83+t96+t208+t213, t291 = t31+t85+t92+t210+t216, t292 = t31+t85+t92+t212+t214;
    T t257 = -t239, t258 = -t240, t259 = -t241, t260 = -t242, t261 = -t243, t262 = -t244, t275 = -t272, t276 = -t273, t277 = -t274, t293 = t288*tdy, t294 = t287*tdz, t295 = t290*tdx, t296 = t289*tdz, t297 = t292*tdx, t298 = t291*tdy;
    T t305 = t13+t197+t229+t230+t257, t306 = t15+t194+t229+t230+t259, t307 = t11+t198+t231+t232+t258, t308 = t16+t192+t231+t232+t261, t309 = t12+t196+t233+t234+t260, t310 = t14+t193+t233+t234+t262, t311 = t277+t293+t295, t312 = t276+t294+t297;
    T t313 = t275+t296+t298;
    
    fdrv(0, 0) = t308*ty+t310*tz; fdrv(0, 1) = -t308*tx; fdrv(0, 2) = -t310*tx; fdrv(0, 3) = t114+t115+t116+t122+t129+t130+oy*t99+oz*t102-rx*t24*2.0-rx*t26*2.0+ry*t208+ry*t213+rz*t212+rz*t214;
    fdrv(0, 4) = t127+t204-ox*t11*4.0-ox*t16*2.0-ox*t36*3.0+oy*t32*2.0+oy*t101+ry*t23+ry*t209+t64*tz; fdrv(0, 5) = t125+t203-ox*t12*4.0-ox*t14*2.0-ox*t40*3.0+oz*t32*2.0+oz*t108+rz*t23+rz*t205+t62*ty;
    fdrv(0, 6) = t302*tdy+t304*tdz+tdx*(t49+t66+oy*rx*ty*4.0-oy*ry*tx*6.0+oz*rx*tz*4.0-oz*rz*tx*6.0); fdrv(0, 7) = -tdy*(t42+t66+t71+t164+ox*t3*6.0)+t302*tdx-t284*tdz; fdrv(0, 8) = -tdz*(t42+t49+t52+t172+ox*t4*6.0)+t304*tdx-t284*tdy;
    fdrv(0, 9) = -tdy*(t84-t141)-tdz*(t87-t145)+tdx*(t88+t95-t140-t144); fdrv(0, 10) = t311; fdrv(0, 11) = t312; fdrv(0, 12) = -tdy*(t38+t99+t111+t153+t218)-tdz*(t39+t102+t107+t154+t220)-tdx*(t33*2.0+t35*2.0+t194+t197);
    fdrv(0, 13) = -tdz*(t105+t195)+tdx*(t34*-3.0+t36+t99+t111+t221)+tdy*(t32*2.0+t101+t108); fdrv(0, 14) = tdx*(t37*-3.0+t40+t102+t107+t222)+tdy*tx*(ry*tz-rz*ty)*2.0+t238*tdz*tx*2.0; fdrv(0, 15) = -ox*t248+t141*ty+t145*tz;
    fdrv(0, 16) = t133-t264*tx-ox*(t159-t267); fdrv(0, 17) = t132-t265*tx-ox*(t160-t268); fdrv(0, 18) = rdx*(t88+t95-ox*(t19+t21))+rdy*(t83+t96-t141-t264)+rdz*(t85+t92-t145-t265); fdrv(0, 19) = -rdz*t270-rdx*(t84-t141)+rdy*(t86+t93-ox*(t7+t20+t137));
    fdrv(0, 20) = -rdy*t271-rdx*(t87-t145)+rdz*(t86+t93-ox*(t6+t22+t137)); fdrv(0, 24) = -rdx*t248-rdy*(t159-t267)-rdz*(t160-t268); fdrv(0, 25) = t136+rdx*t156-rdy*t246*tx; fdrv(0, 26) = t135+rdx*t158-rdz*t247*tx; fdrv(1, 0) = -t306*ty;
    fdrv(1, 1) = t306*tx+t309*tz; fdrv(1, 2) = -t309*ty; fdrv(1, 3) = t126+t204+ox*t36*2.0+ox*t98-oy*t13*4.0-oy*t15*2.0-oy*t32*3.0+rx*t27+rx*t213+t63*tz;
    fdrv(1, 4) = t113+t115+t117+t122+t124+t129+ox*t103+oz*t109+rx*t207+rx*t209-ry*t25*2.0-ry*t29*2.0+rz*t210+rz*t216; fdrv(1, 5) = t120+t202-oy*t12*2.0-oy*t14*4.0-oy*t40*3.0+oz*t36*2.0+oz*t111+rz*t27+rz*t206+t50*ty;
    fdrv(1, 6) = -tdx*(t65+t76+t78+t161+oy*t2*6.0)+t300*tdy+t286*tdz; fdrv(1, 7) = t300*tdx+t303*tdz+tdy*(t45+t76-ox*rx*ty*6.0+ox*ry*tx*4.0+oz*ry*tz*4.0-oz*rz*ty*6.0); fdrv(1, 8) = -tdz*(t43+t45+t65+t175+oy*t4*6.0)+t286*tdx+t303*tdy; fdrv(1, 9) = t311;
    fdrv(1, 10) = -tdx*(t88-t140)-tdz*(t94-t149)+tdy*(t84+t97-t141-t148); fdrv(1, 11) = t313; fdrv(1, 12) = -tdz*(t104+t195)+tdy*(t32-t33*3.0+t103+t108+t219)+tdx*(t36*2.0+t98+t111);
    fdrv(1, 13) = -tdx*(t35+t103+t108+t152+t217)-tdz*(t37+t100+t109+t154+t222)-tdy*(t34*2.0+t38*2.0+t192+t198); fdrv(1, 14) = tdy*(t39*-3.0+t40+t100+t109+t220)+tdx*ty*(rx*tz-rz*tx)*2.0+t238*tdz*ty*2.0; fdrv(1, 15) = t133-t263*ty-oy*(t157-t266);
    fdrv(1, 16) = -oy*t249+t140*tx+t149*tz; fdrv(1, 17) = t131-t265*ty-oy*(t158-t268); fdrv(1, 18) = -rdz*t269-rdy*(t88-t140)+rdx*(t83+t96-oy*(t7+t18+t138)); fdrv(1, 19) = rdy*(t84+t97-oy*(t17+t21))+rdx*(t86+t93-t140-t263)+rdz*(t85+t92-t149-t265);
    fdrv(1, 20) = -rdx*t271-rdy*(t94-t149)+rdz*(t83+t96-oy*(t5+t22+t138)); fdrv(1, 24) = t136+rdy*t155-rdx*t245*ty; fdrv(1, 25) = -rdy*t249-rdx*(t157-t266)-rdz*(t158-t268); fdrv(1, 26) = t134+rdy*t160-rdz*t247*ty; fdrv(2, 0) = -t305*tz;
    fdrv(2, 1) = -t307*tz; fdrv(2, 2) = t305*tx+t307*ty; fdrv(2, 3) = t121+t203+ox*t40*2.0+ox*t100-oz*t13*2.0-oz*t15*4.0-oz*t32*3.0+rx*t31+rx*t214+t51*tz; fdrv(2, 4) = t119+t202+oy*t40*2.0+oy*t107-oz*t11*2.0-oz*t16*4.0-oz*t36*3.0+ry*t31+ry*t210+t44*tz;
    fdrv(2, 5) = t113+t114+t116+t117+t124+t130+ox*t110+oy*t112+rx*t205+rx*t211+ry*t206+ry*t215-rz*t28*2.0-rz*t30*2.0; fdrv(2, 6) = -tdx*(t67+t70+t82+t163+oz*t2*6.0)+t285*tdy+t299*tdz; fdrv(2, 7) = -tdy*(t48+t53+t82+t171+oz*t3*6.0)+t285*tdx+t301*tdz;
    fdrv(2, 8) = t299*tdx+t301*tdy+tdz*(t53+t70-ox*rx*tz*6.0+ox*rz*tx*4.0-oy*ry*tz*6.0+oy*rz*ty*4.0); fdrv(2, 9) = t312; fdrv(2, 10) = t313; fdrv(2, 11) = -tdx*(t95-t144)-tdy*(t97-t148)+tdz*(t87+t94-t145-t149);
    fdrv(2, 12) = -tdy*(t104-t105)+tdz*(t32-t35*3.0+t101+t110+t217)+tdx*(t40*2.0+t100+t107); fdrv(2, 13) = tdz*(t36-t38*3.0+t98+t112+t218)+tdx*tz*(rx*ty-ry*tx)*2.0+t238*tdy*tz*2.0;
    fdrv(2, 14) = -tdx*(t33+t101+t110+t152+t219)-tdy*(t34+t98+t112+t153+t221)-tdz*(t37*2.0+t39*2.0+t193+t196); fdrv(2, 15) = t132-t263*tz-oz*(t155-t266); fdrv(2, 16) = t131-t264*tz-oz*(t156-t267); fdrv(2, 17) = -oz*t250+t144*tx+t148*ty;
    fdrv(2, 18) = -rdy*t269-rdz*(t95-t144)+rdx*(t85+t92-oz*(t6+t18+t139)); fdrv(2, 19) = -rdx*t270-rdz*(t97-t148)+rdy*(t85+t92-oz*(t5+t20+t139)); fdrv(2, 20) = rdz*(t87+t94-oz*(t17+t19))+rdx*(t86+t93-t144-t263)+rdy*(t83+t96-t148-t264);
    fdrv(2, 24) = t135+rdz*t157-rdx*t245*tz; fdrv(2, 25) = t134+rdz*t159-rdy*t246*tz; fdrv(2, 26) = -rdz*t250-rdx*(t155-t266)-rdy*(t156-t267);
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f08(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*ty, t3 = oy*tx, t4 = ox*tz, t5 = oz*tx, t6 = oy*tz, t7 = oz*ty, t8 = rdx*tx, t9 = rdy*ty, t10 = rdz*tz, t11 = rx*tx, t12 = ry*ty, t13 = rz*tz, t14 = tx*tx, t16 = ty*ty, t18 = tz*tz, t21 = rx*ty*2.0, t22 = ry*tx*2.0;
    T t23 = rx*tz*2.0, t25 = rz*tx*2.0, t26 = ry*tz*2.0, t27 = rz*ty*2.0, t32 = rx*ty*tz, t33 = ry*tx*tz, t34 = rz*tx*ty, t20 = t11*2.0, t24 = t12*2.0, t28 = t13*2.0, t29 = t11*ty, t30 = t11*tz, t31 = t12*tx, t35 = t12*tz, t36 = t13*tx;
    T t37 = t13*ty, t38 = -t3, t39 = -t5, t40 = -t7, t41 = ox*t14, t42 = t2*ty, t43 = t3*tx, t44 = t4*tz, t45 = oy*t16, t46 = t5*tx, t47 = t6*tz, t48 = t7*ty, t49 = oz*t18, t50 = rx*t16, t51 = ry*t14, t52 = rx*t18, t53 = rz*t14, t54 = ry*t18;
    T t55 = rz*t16, t56 = t2*tx*2.0, t57 = t4*tx*2.0, t58 = t3*ty*2.0, t59 = t6*ty*2.0, t60 = t5*tz*2.0, t61 = t7*tz*2.0, t68 = t11*t16, t69 = t11*t18, t70 = t12*t14, t77 = t12*t18, t78 = t13*t14, t79 = t13*t16, t80 = ry*t2*tx*4.0;
    T t81 = rx*t3*ty*4.0, t82 = rx*t2*tz*4.0, t83 = rz*t4*tx*4.0, t84 = ry*t3*tz*4.0, t85 = rx*t5*tz*4.0, t86 = rz*t5*ty*4.0, t87 = rz*t6*ty*4.0, t88 = ry*t7*tz*4.0, t95 = t11*tx*3.0, t97 = t12*ty*3.0, t99 = t13*tz*3.0, t107 = -t33, t108 = -t34;
    T t109 = t14+t16, t110 = t14+t18, t111 = t16+t18, t113 = ox*t11*tx*6.0, t114 = t2*t21, t116 = t4*t23, t117 = t3*t22, t121 = oy*t12*ty*6.0, t129 = oz*t13*tz*6.0, t148 = t2*t11*1.2E+1, t149 = ry*t2*tx*8.0, t150 = rx*t3*ty*8.0;
    T t151 = t4*t11*1.2E+1, t153 = t3*t12*1.2E+1, t156 = rz*t4*tx*8.0, t157 = rx*t5*tz*8.0, t159 = t6*t12*1.2E+1, t162 = rz*t6*ty*8.0, t163 = ry*t7*tz*8.0, t164 = t5*t13*1.2E+1, t165 = t7*t13*1.2E+1, t166 = t11*t14*8.0, t167 = t12*t16*8.0;
    T t168 = t13*t18*8.0, t169 = t11+t12, t170 = t11+t13, t171 = t12+t13, t198 = t21+t22, t199 = t23+t25, t200 = t26+t27, t201 = t8+t9+t10, t62 = t20*ty, t63 = t20*tz, t64 = t24*tx, t65 = t24*tz, t66 = t28*tx, t67 = t28*ty, t71 = t52*ty;
    T t72 = t50*tz, t73 = t54*tx, t74 = t51*tz, t75 = t55*tx, t76 = t53*ty, t89 = t42*3.0, t90 = t43*3.0, t91 = t44*3.0, t92 = t46*3.0, t93 = t47*3.0, t94 = t48*3.0, t96 = t14*t20, t98 = t16*t24, t100 = t18*t28, t101 = -t56, t102 = -t57;
    T t103 = -t58, t104 = -t59, t105 = -t60, t106 = -t61, t112 = t20*t41, t115 = rx*t42*6.0, t118 = rx*t44*6.0, t119 = ry*t43*6.0, t120 = t24*t45, t124 = ry*t47*6.0, t125 = rz*t46*6.0, t127 = rz*t48*6.0, t128 = t28*t49, t130 = t16*t20;
    T t131 = t68*4.0, t132 = t68*6.0, t133 = t18*t20, t134 = t14*t24, t135 = t69*4.0, t136 = t70*4.0, t137 = t69*6.0, t138 = t70*6.0, t139 = t18*t24, t140 = t14*t28, t141 = t77*4.0, t142 = t78*4.0, t143 = t77*6.0, t144 = t78*6.0, t145 = t16*t28;
    T t146 = t79*4.0, t147 = t79*6.0, t152 = -t82, t154 = -t84, t155 = -t85, t158 = -t86, t160 = -t87, t161 = -t88, t173 = -t121, t174 = ry*t47*-2.0, t175 = rz*t46*-2.0, t177 = rz*t48*-2.0, t179 = -t129, t186 = -t150, t187 = -t157, t188 = -t163;
    T t189 = rx*t109, t190 = rx*t110, t191 = ry*t109, t192 = ry*t111, t193 = rz*t110, t194 = rz*t111, t195 = t2+t38, t196 = t4+t39, t197 = t6+t40, t202 = t169*tx*tz, t203 = t170*tx*ty, t204 = t169*ty*tz, t205 = t171*tx*ty, t206 = t170*ty*tz;
    T t207 = t171*tx*tz, t214 = t18+t109, t215 = t32+t107, t216 = t32+t108, t217 = t33+t108, t230 = t109*t169, t231 = t110*t170, t232 = t111*t171, t172 = -t119, t176 = -t125, t178 = -t127, t180 = -t71, t181 = -t72, t182 = -t73, t183 = -t74;
    T t184 = -t75, t185 = -t76, t208 = t190*ty, t209 = t189*tz, t210 = t192*tx, t211 = t191*tz, t212 = t194*tx, t213 = t193*ty, t218 = t215*tx, t219 = t216*tx, t220 = t215*ty, t221 = t217*ty, t222 = t216*tz, t223 = t217*tz, t224 = t31+t190;
    T t225 = t36+t189, t226 = t29+t192, t227 = t37+t191, t228 = t30+t194, t229 = t35+t193, t233 = ox*t8*t214*2.0, t234 = rdx*t3*t214*2.0, t235 = rdy*t2*t214*2.0, t236 = rdx*t5*t214*2.0, t237 = oy*t9*t214*2.0, t238 = rdz*t4*t214*2.0;
    T t239 = rdy*t7*t214*2.0, t240 = rdz*t6*t214*2.0, t241 = oz*t10*t214*2.0, t257 = t201*t214*tx*2.0, t258 = t201*t214*ty*2.0, t259 = t201*t214*tz*2.0, t260 = t41+t44+t89+t103, t261 = t41+t42+t91+t105, t262 = t45+t47+t90+t101;
    T t263 = t43+t45+t93+t106, t264 = t48+t49+t92+t102, t265 = t46+t49+t94+t104, t266 = t50+t52+t64+t66+t95, t267 = t51+t54+t62+t67+t97, t268 = t53+t55+t63+t65+t99, t284 = t96+t98+t130+t134+t137+t142+t143+t146+t168;
    T t285 = t96+t100+t132+t133+t136+t140+t141+t147+t167, t286 = t98+t100+t131+t135+t138+t139+t144+t145+t166, t242 = -t220, t243 = -t222, t244 = -t223, t245 = t224*tx, t246 = t225*tx, t247 = t226*tx, t248 = t224*ty, t249 = t226*ty, t250 = t228*tx;
    T t251 = t225*tz, t252 = t227*ty, t253 = t229*ty, t254 = t227*tz, t255 = t228*tz, t256 = t229*tz, t269 = t266*tdx*ty*2.0, t270 = t266*tdx*tz*2.0, t271 = t267*tdy*tx*2.0, t272 = t267*tdy*tz*2.0, t273 = t268*tdz*tx*2.0, t274 = t268*tdz*ty*2.0;
    T t287 = t286*tdx, t288 = t285*tdy, t289 = t284*tdz, t290 = t83+t113+t115+t116+t149+t160+t172+t173+t174+t186, t291 = t80+t113+t114+t118+t156+t161+t176+t177+t179+t187, t292 = t81+t117+t121+t124+t155+t162+t175+t178+t179+t188;
    T t275 = t185+t204+t213+t218+t254, t276 = t181+t207+t209+t221+t250, t277 = t183+t206+t211+t219+t253, t278 = t182+t203+t210+t243+t248, t279 = t184+t202+t212+t242+t251, t280 = t180+t205+t208+t244+t247, t281 = t78+t79+t230+t246+t252;
    T t282 = t70+t77+t231+t245+t256, t283 = t68+t69+t232+t249+t255;
    
    fdrv(0, 0) = -t276*ty+t280*tz; fdrv(0, 1) = t276*tx+t283*tz; fdrv(0, 2) = -t280*tx-t283*ty; fdrv(0, 3) = t197*t266*2.0;
    fdrv(0, 4) = t3*t32*4.0-t5*t31*4.0-t5*t36*2.0+t6*t37*4.0-t7*t37*6.0-t5*t50*6.0-t5*t52*2.0-t11*t46*2.0+t6*t54*2.0-t12*t48*8.0-t7*t54*4.0-t13*t49*2.0+t117*tz+t6*t12*ty*6.0;
    fdrv(0, 5) = t120-t5*t32*4.0+t3*t36*4.0-t5*t34*2.0+t6*t35*6.0-t7*t35*4.0+t3*t50*2.0+t3*t52*6.0+t13*t47*8.0+t6*t55*4.0-t7*t55*2.0+t20*t43+t24*t43-t7*t13*tz*6.0;
    fdrv(0, 6) = -tdy*(t154+rx*t48*6.0+rx*t49*2.0+t5*t11*6.0+t5*t12*8.0+t5*t13*4.0-rx*t6*ty*4.0)+tdz*(t158+rx*t45*2.0+rx*t47*6.0+t3*t11*6.0+t3*t12*4.0+t3*t13*8.0-rx*t7*tz*4.0)+t197*tdx*(t11*6.0+t24+t28)*2.0;
    fdrv(0, 7) = -tdy*(-t159+t165+ry*t46*4.0+ry*t49*4.0-t6*t13*4.0+t7*t12*2.4E+1+rx*t5*ty*1.2E+1-rx*t3*tz*4.0)+t292*tdz-oz*t266*tdx*2.0+t197*t198*tdx*2.0;
    fdrv(0, 8) = tdz*(t159-t165+rz*t43*4.0+rz*t45*4.0+t6*t13*2.4E+1-t7*t12*4.0-rx*t5*ty*4.0+rx*t3*tz*1.2E+1)+t292*tdy+oy*t266*tdx*2.0+t197*t199*tdx*2.0; fdrv(0, 9) = t197*tdx*(t14+t109+t110)*2.0-t265*tdy*tx*2.0+t263*tdz*tx*2.0;
    fdrv(0, 10) = tdy*(t6*t16*6.0-t7*t16*8.0+t6*t18*2.0-t7*t18*4.0-t46*ty*4.0+t43*tz*2.0)+t263*tdz*ty*2.0+t197*tdx*tx*ty*4.0; fdrv(0, 11) = tdz*(t6*t16*4.0-t7*t16*2.0+t6*t18*8.0-t7*t18*6.0-t46*ty*2.0+t43*tz*4.0)-t265*tdy*tz*2.0+t197*tdx*tx*tz*4.0;
    fdrv(0, 13) = t270+t272+t289; fdrv(0, 14) = -t269-t274-t288; fdrv(0, 15) = t197*t214*tx*2.0; fdrv(0, 16) = t197*t214*ty*2.0; fdrv(0, 17) = t197*t214*tz*2.0; fdrv(0, 18) = t197*(rdx*t16+rdx*t18+t8*tx*3.0+t9*tx*2.0+t10*tx*2.0)*2.0;
    fdrv(0, 19) = -t236-t239-t241+rdy*t197*t214*2.0+t8*t197*ty*4.0+t9*t197*ty*4.0+t10*t197*ty*4.0; fdrv(0, 20) = t234+t237+t240+rdz*t197*t214*2.0+t8*t197*tz*4.0+t9*t197*tz*4.0+t10*t197*tz*4.0; fdrv(0, 25) = t259; fdrv(0, 26) = -t258;
    fdrv(1, 0) = -t277*ty-t282*tz; fdrv(1, 1) = t277*tx-t278*tz; fdrv(1, 2) = t282*tx+t278*ty;
    fdrv(1, 3) = t128-t2*t32*2.0-t2*t33*4.0+t5*t31*6.0-t4*t36*4.0+t5*t36*6.0+t5*t50*4.0-t4*t52*2.0+t5*t52*4.0+t11*t46*8.0+t7*t54*2.0+t24*t48+t28*t48-t4*t11*tx*6.0; fdrv(1, 4) = t196*t267*-2.0;
    fdrv(1, 5) = t2*t29*-2.0-t4*t30*6.0+t5*t30*4.0-t2*t37*4.0+t5*t35*4.0-t11*t41*2.0-t2*t51*2.0-t12*t42*2.0-t2*t54*6.0-t4*t53*4.0-t13*t44*8.0+t5*t53*2.0+t5*t55*2.0+t5*t13*tz*6.0;
    fdrv(1, 6) = -t291*tdz+tdx*(-t151+t164+rx*t48*4.0+rx*t49*4.0+t5*t11*2.4E+1-t4*t13*4.0+t5*t12*1.2E+1-ry*t2*tz*4.0)+oz*t267*tdy*2.0-t196*t198*tdy*2.0;
    fdrv(1, 7) = tdx*(t152+ry*t46*6.0+ry*t49*2.0+t7*t12*6.0+t7*t13*4.0-ry*t4*tx*4.0+rx*t5*ty*8.0)-tdz*(t158+ry*t41*2.0+ry*t44*6.0+t2*t11*4.0+t2*t12*6.0+t2*t13*8.0-ry*t5*tz*4.0)-t196*tdy*(t12*6.0+t20+t28)*2.0;
    fdrv(1, 8) = -t291*tdx-tdz*(t151-t164+rz*t41*4.0+rz*t42*4.0-t5*t11*4.0+t4*t13*2.4E+1-t5*t12*4.0+ry*t2*tz*1.2E+1)-ox*t267*tdy*2.0-t196*t200*tdy*2.0;
    fdrv(1, 9) = -tdx*(t4*t14*6.0-t5*t14*8.0-t5*t16*4.0+t4*t18*2.0-t5*t18*4.0+t42*tz*2.0)-t261*tdz*tx*2.0-t196*tdy*tx*ty*4.0; fdrv(1, 10) = t196*tdy*(t16+t109+t111)*-2.0+t264*tdx*ty*2.0-t261*tdz*ty*2.0;
    fdrv(1, 11) = -tdz*(t4*t14*4.0-t5*t14*2.0-t5*t16*2.0+t4*t18*8.0-t5*t18*6.0+t42*tz*4.0)+t264*tdx*tz*2.0-t196*tdy*ty*tz*4.0; fdrv(1, 12) = -t270-t272-t289; fdrv(1, 14) = t271+t273+t287; fdrv(1, 15) = t196*t214*tx*-2.0; fdrv(1, 16) = t196*t214*ty*-2.0;
    fdrv(1, 17) = t196*t214*tz*-2.0; fdrv(1, 18) = t236+t239+t241-rdx*t196*t214*2.0-t8*t196*tx*4.0-t9*t196*tx*4.0-t10*t196*tx*4.0; fdrv(1, 19) = t196*(rdy*t14+rdy*t18+t8*ty*2.0+t9*ty*3.0+t10*ty*2.0)*-2.0;
    fdrv(1, 20) = -t233-t235-t238-rdz*t196*t214*2.0-t8*t196*tz*4.0-t9*t196*tz*4.0-t10*t196*tz*4.0; fdrv(1, 24) = -t259; fdrv(1, 26) = t257; fdrv(2, 0) = t281*ty+t275*tz; fdrv(2, 1) = -t281*tx-t279*tz; fdrv(2, 2) = -t275*tx+t279*ty;
    fdrv(2, 3) = t2*t31*4.0-t3*t31*6.0+t2*t36*4.0-t3*t36*6.0-t6*t35*2.0+t2*t50*2.0-t3*t50*4.0+t2*t52*2.0-t11*t43*8.0-t3*t52*4.0-t12*t45*2.0-t13*t47*2.0-t6*t55*2.0+t2*t11*tx*6.0;
    fdrv(2, 4) = t112+t2*t29*6.0-t3*t29*4.0+t2*t37*6.0-t3*t37*4.0+t2*t51*4.0-t3*t51*2.0+t12*t42*8.0+t2*t54*4.0-t3*t54*2.0+t4*t53*2.0+t20*t44+t28*t44-t3*t12*ty*6.0; fdrv(2, 5) = t195*t268*2.0;
    fdrv(2, 6) = -tdx*(-t148+t153+rx*t45*4.0+rx*t47*4.0-t2*t12*4.0+t3*t11*2.4E+1-t2*t13*4.0+t3*t13*1.2E+1)+t290*tdy-oy*t268*tdz*2.0+t195*t199*tdz*2.0;
    fdrv(2, 7) = tdy*(t148-t153+ry*t41*4.0+ry*t44*4.0+t2*t12*2.4E+1-t3*t11*4.0+t2*t13*1.2E+1-t3*t13*4.0)+t290*tdx+ox*t268*tdz*2.0+t195*t200*tdz*2.0;
    fdrv(2, 8) = -tdx*(t152+rz*t43*6.0+rz*t45*2.0+t6*t12*4.0+t6*t13*6.0-rz*t2*tx*4.0+rx*t3*tz*8.0)+tdy*(t154+rz*t41*2.0+rz*t42*6.0+t4*t11*4.0+t4*t13*6.0+ry*t2*tz*8.0-rz*t3*ty*4.0)+t195*tdz*(t13*6.0+t20+t24)*2.0;
    fdrv(2, 9) = tdx*(t2*t14*6.0-t3*t14*8.0+t2*t16*2.0-t3*t16*4.0+t2*t18*2.0-t3*t18*4.0)+t260*tdy*tx*2.0+t195*tdz*tx*tz*4.0; fdrv(2, 10) = tdy*(t2*t14*4.0-t3*t14*2.0+t2*t16*8.0-t3*t16*6.0+t2*t18*4.0-t3*t18*2.0)-t262*tdx*ty*2.0+t195*tdz*ty*tz*4.0;
    fdrv(2, 11) = t262*tdx*tz*-2.0+t260*tdy*tz*2.0+t195*tdz*(t18*2.0+t214)*2.0; fdrv(2, 12) = t269+t274+t288; fdrv(2, 13) = -t271-t273-t287; fdrv(2, 15) = t195*t214*tx*2.0; fdrv(2, 16) = t195*t214*ty*2.0; fdrv(2, 17) = t195*t214*tz*2.0;
    fdrv(2, 18) = -t234-t237-t240+rdx*t195*t214*2.0+t8*t195*tx*4.0+t9*t195*tx*4.0+t10*t195*tx*4.0; fdrv(2, 19) = t233+t235+t238+rdy*t195*t214*2.0+t8*t195*ty*4.0+t9*t195*ty*4.0+t10*t195*ty*4.0;
    fdrv(2, 20) = t195*(rdz*t14+rdz*t16+t8*tz*2.0+t9*tz*2.0+t10*tz*3.0)*2.0; fdrv(2, 24) = t258; fdrv(2, 25) = -t257;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f09(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*rx, t3 = oy*ry, t4 = oz*rz, t5 = ox*tx, t6 = ox*ty, t7 = oy*tx, t8 = ox*tz, t9 = oy*ty, t10 = oz*tx, t11 = oy*tz, t12 = oz*ty, t13 = oz*tz, t14 = rx*tx, t15 = rx*ty, t16 = ry*tx, t17 = rx*tz, t18 = ry*ty, t19 = rz*tx;
    T t20 = ry*tz, t21 = rz*ty, t22 = rz*tz, t23 = tx*tx, t24 = ty*ty, t25 = tz*tz, t26 = t6*2.0, t27 = t7*2.0, t28 = t8*2.0, t29 = t10*2.0, t30 = t11*2.0, t31 = t12*2.0, t32 = t14*2.0, t33 = t18*2.0, t34 = t22*2.0, t35 = t5*ty, t36 = t5*tz;
    T t37 = t7*ty, t38 = t9*tz, t39 = t10*tz, t40 = t12*tz, t41 = t15*tz, t42 = t16*tz, t43 = t19*ty, t44 = -t3, t45 = -t4, t52 = -t16, t53 = -t19, t54 = -t21, t55 = t23+t24, t56 = t23+t25, t57 = t24+t25, t58 = t5+t9, t59 = t5+t13, t60 = t9+t13;
    T t47 = -t27, t49 = -t29, t51 = -t31, t61 = t14+t33, t62 = t18+t32, t63 = t14+t34, t64 = t22+t32, t65 = t18+t34, t66 = t22+t33, t67 = t58*tdz, t68 = t59*tdy, t69 = t60*tdx, t70 = ox*t57, t71 = oy*t56, t72 = oz*t55, t73 = t2+t44, t74 = t2+t45;
    T t75 = t3+t45, t79 = t15+t52, t80 = t17+t53, t81 = t20+t54, t88 = -tdy*(t7-t26), t89 = -tdz*(t10-t28), t90 = -tdz*(t12-t30), t76 = t6+t47, t77 = t8+t49, t78 = t11+t51, t85 = t76*tdx, t86 = t77*tdx, t87 = t78*tdy;
    
    fdrv(0, 0) = t42-t43;
    fdrv(0, 1) = rz*t56+t18*tz; fdrv(0, 2) = -ry*t55+t54*tz; fdrv(0, 3) = ox*t81-ry*t10*2.0+rz*t27; fdrv(0, 4) = -oz*t66-rz*t5+t3*tz; fdrv(0, 5) = oy*t65+ry*t5+t45*ty; fdrv(0, 6) = tdx*(oy*rz*2.0-oz*ry*2.0)+ox*ry*tdz-ox*rz*tdy;
    fdrv(0, 7) = t75*tdz-ox*rz*tdx-oz*ry*tdy*2.0; fdrv(0, 8) = t75*tdy+ox*ry*tdx+oy*rz*tdz*2.0; fdrv(0, 10) = t67+t86+t87; fdrv(0, 11) = -t68-t85+t90; fdrv(0, 12) = t81*tdx+t53*tdy+t16*tdz; fdrv(0, 13) = t19*tdx*2.0+t20*tdy+t65*tdz;
    fdrv(0, 14) = t16*tdx*-2.0-t66*tdy+t54*tdz; fdrv(0, 16) = t36+t38-t72; fdrv(0, 17) = -t35-t40+t71; fdrv(0, 18) = rdy*t77-rdz*t76; fdrv(0, 19) = rdy*t78-rdz*t59; fdrv(0, 20) = rdy*t58-rdz*(t12-t30); fdrv(0, 24) = -tx*(rdz*ty-rdy*tz);
    fdrv(0, 25) = rdz*t56+rdy*ty*tz; fdrv(0, 26) = -rdy*t55-rdz*ty*tz; fdrv(1, 0) = -rz*t57-t14*tz; fdrv(1, 1) = -t41+t43; fdrv(1, 2) = rx*t55+t19*tz; fdrv(1, 3) = oz*t64+rz*t9-t2*tz; fdrv(1, 4) = -oy*t80+rx*t31-rz*t6*2.0; fdrv(1, 5) = -ox*t63-rx*t9+t4*tx;
    fdrv(1, 6) = -t74*tdz+oz*rx*tdx*2.0+oy*rz*tdy; fdrv(1, 7) = -tdy*(ox*rz*2.0-oz*rx*2.0)-oy*rx*tdz+oy*rz*tdx; fdrv(1, 8) = -t74*tdx-oy*rx*tdy-ox*rz*tdz*2.0; fdrv(1, 9) = -t67-t86-t87; fdrv(1, 11) = t69+tdy*(t7-t26)+tdz*(t10-t28);
    fdrv(1, 12) = -t17*tdx-t21*tdy*2.0-t63*tdz; fdrv(1, 13) = t21*tdx-t80*tdy-t15*tdz; fdrv(1, 14) = t64*tdx+t15*tdy*2.0+t19*tdz; fdrv(1, 15) = -t36-t38+t72; fdrv(1, 17) = t37+t39-t70; fdrv(1, 18) = -rdx*t77+rdz*t60; fdrv(1, 19) = -rdx*t78+rdz*(t7-t26);
    fdrv(1, 20) = -rdx*t58+rdz*(t10-t28); fdrv(1, 24) = -rdz*t57-rdx*tx*tz; fdrv(1, 25) = ty*(rdz*tx-rdx*tz); fdrv(1, 26) = rdx*t55+rdz*tx*tz; fdrv(2, 0) = ry*t57+t14*ty; fdrv(2, 1) = -rx*t56+t52*ty; fdrv(2, 2) = t41-t42; fdrv(2, 3) = -oy*t62-ry*t13+t2*ty;
    fdrv(2, 4) = ox*t61+rx*t13+t44*tx; fdrv(2, 5) = oz*t79-rx*t11*2.0+ry*t28; fdrv(2, 6) = t73*tdy-oy*rx*tdx*2.0-oz*ry*tdz; fdrv(2, 7) = t73*tdx+ox*ry*tdy*2.0+oz*rx*tdz; fdrv(2, 8) = tdz*(ox*ry*2.0-oy*rx*2.0)+oz*rx*tdy-oz*ry*tdx;
    fdrv(2, 9) = t68+t85+tdz*(t12-t30); fdrv(2, 10) = -t69+t88+t89; fdrv(2, 12) = t15*tdx+t61*tdy+t20*tdz*2.0; fdrv(2, 13) = -t62*tdx+t52*tdy-t17*tdz*2.0; fdrv(2, 14) = -t20*tdx+t17*tdy+t79*tdz; fdrv(2, 15) = t35+t40-t71; fdrv(2, 16) = -t37-t39+t70;
    fdrv(2, 18) = rdx*t76-rdy*t60; fdrv(2, 19) = rdx*t59-rdy*(t7-t26); fdrv(2, 20) = rdx*(t12-t30)-rdy*(t10-t28); fdrv(2, 24) = rdy*t57+rdx*tx*ty; fdrv(2, 25) = -rdx*t56-rdy*tx*ty; fdrv(2, 26) = -tz*(rdy*tx-rdx*ty);
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f10(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = rx*tx, t3 = rx*ty, t4 = ry*tx, t5 = rx*tz, t6 = ry*ty, t7 = rz*tx, t8 = ry*tz, t9 = rz*ty, t10 = rz*tz, t11 = tx*ty, t12 = tx*tz, t13 = ty*tz, t14 = tx*tx, t15 = tx*tx*tx, t16 = ty*ty, t17 = ty*ty*ty, t18 = tz*tz, t19 = tz*tz*tz;
    T t20 = t2*2.0, t21 = t6*2.0, t22 = t10*2.0, t23 = ox*t4, t24 = oy*t2, t25 = ox*t6, t26 = ox*t7, t27 = oy*t3, t28 = oz*t2, t29 = ox*t8, t30 = ox*t9, t31 = oy*t5, t32 = oy*t7, t33 = oz*t3, t34 = oz*t4, t35 = ox*t10, t36 = oy*t9, t37 = oz*t5;
    T t38 = oz*t6, t39 = oy*t10, t40 = oz*t8, t41 = ox*t11, t42 = ox*t12, t43 = oy*t11, t44 = ox*t13, t45 = oy*t12, t46 = oz*t11, t47 = oy*t13, t48 = oz*t12, t49 = oz*t13, t50 = t2*ty, t51 = t2*tz, t52 = t4*ty, t53 = t3*tz, t54 = t4*tz;
    T t55 = t7*ty, t56 = t6*tz, t57 = t7*tz, t58 = t9*tz, t59 = t13+tx, t60 = t12+ty, t61 = t11+tz, t62 = t14*2.0, t63 = t16*2.0, t64 = t18*2.0, t65 = -t11, t66 = -t12, t67 = -t13, t68 = ox*t14, t69 = ox*t15, t70 = ox*t16, t71 = oy*t14;
    T t72 = ox*t17, t73 = oy*t15, t74 = ox*t18, t75 = oy*t16, t76 = oz*t14, t77 = ox*t19, t78 = oy*t17, t79 = oz*t15, t80 = oy*t18, t81 = oz*t16, t82 = oy*t19, t83 = oz*t17, t84 = oz*t18, t85 = oz*t19, t86 = t2*tx, t87 = t2*t14, t88 = t3*ty;
    T t89 = t4*tx, t90 = t3*t16, t91 = t4*t14, t92 = t5*tz, t93 = t6*ty, t94 = t7*tx, t95 = t5*t18, t96 = t6*t16, t97 = t7*t14, t98 = t8*tz, t99 = t9*ty, t100 = t8*t18, t101 = t9*t16, t102 = t10*tz, t103 = t10*t18, t104 = t11*ty, t105 = t11*tx;
    T t106 = t11*t16, t107 = t11*t14, t108 = t12*tz, t109 = t12*tx, t110 = t12*t18, t111 = t12*t14, t112 = t13*tz, t113 = t13*ty, t114 = t13*t18, t115 = t13*t16, t117 = ox*t2*4.0, t118 = ox*t3*2.0, t121 = ox*t5*2.0, t125 = oy*t4*2.0;
    T t131 = oy*t6*4.0, t136 = oy*t8*2.0, t140 = oz*t7*2.0, t143 = oz*t9*2.0, t145 = oz*t10*4.0, t173 = t11*tz*2.0, t198 = t2*t16, t199 = t2*t18, t200 = t4*t11, t201 = t3*t18, t202 = t3*t13, t203 = t4*t18, t204 = t4*t12, t205 = t7*t16;
    T t206 = t7*t11, t207 = t6*t18, t208 = t7*t12, t209 = t9*t13, t210 = t11*t18, t211 = t11*t13, t212 = t11*t12, t265 = t4*t13*2.0, t266 = t7*t13*2.0, t276 = rdx*t15*2.0, t277 = rdy*t17*2.0, t278 = rdz*t19*2.0, t327 = t14+t16, t328 = t14+t18;
    T t329 = t16+t18, t375 = t2*t11*3.0, t381 = t2*t12*3.0, t382 = t4*t16*3.0, t389 = t6*t13*3.0, t390 = t7*t18*3.0, t394 = t9*t18*3.0, t429 = t2+t6, t430 = t2+t10, t431 = t6+t10, t479 = t11*(t13-tx), t480 = t11*(t11-tz), t481 = t12*(t12-ty);
    T t482 = t12*(t11-tz), t483 = t13*(t13-tx), t484 = t13*(t12-ty), t116 = ox*t20, t119 = t23*2.0, t120 = oy*t20, t122 = ox*t21, t123 = t26*2.0, t124 = t27*2.0, t126 = oz*t20, t127 = oy*t21, t128 = t29*4.0, t129 = t30*4.0, t130 = t31*4.0;
    T t132 = t32*4.0, t133 = t33*4.0, t134 = t34*4.0, t135 = ox*t22, t137 = t36*2.0, t138 = t37*2.0, t139 = oz*t21, t141 = oy*t22, t142 = t40*2.0, t144 = oz*t22, t146 = t41*2.0, t147 = t42*2.0, t148 = t43*2.0, t149 = t44*2.0, t150 = t45*2.0;
    T t151 = t46*2.0, t152 = t44*4.0, t153 = t45*4.0, t154 = t46*4.0, t155 = t47*2.0, t156 = t48*2.0, t157 = t49*2.0, t158 = t20*ty, t159 = t50*4.0, t160 = t20*tz, t161 = t52*2.0, t162 = t51*4.0, t163 = t52*4.0, t164 = t53*2.0, t165 = t54*2.0;
    T t166 = t55*2.0, t167 = t21*tz, t168 = t57*2.0, t169 = t56*4.0, t170 = t57*4.0, t171 = t58*2.0, t172 = t58*4.0, t174 = rdx*t69, t175 = rdy*t78, t176 = rdz*t85, t177 = t2*t68, t178 = ox*t88, t179 = t23*tx, t180 = ox*t92, t181 = t26*tx;
    T t182 = t27*ty, t183 = oy*t89, t184 = t6*t75, t185 = oy*t98, t186 = t36*ty, t187 = t37*tz, t188 = oz*t94, t189 = t40*tz, t190 = oz*t99, t191 = t10*t84, t192 = rdx*t104, t193 = rdy*t105, t194 = rdx*t108, t195 = rdz*t109, t196 = rdy*t112;
    T t197 = rdz*t113, t217 = ox*t50*6.0, t221 = t23*ty*4.0, t222 = t24*ty*4.0, t223 = ox*t51*6.0, t228 = t24*tz*4.0, t229 = t28*ty*4.0, t230 = t24*tz*6.0, t231 = oy*t52*6.0, t232 = t28*ty*6.0, t236 = t32*ty*2.0, t238 = t25*tz*4.0;
    T t239 = t26*tz*4.0, t240 = t28*tz*4.0, t242 = t25*tz*6.0, t243 = t34*ty*6.0, t245 = t33*tz*2.0, t246 = t34*tz*2.0, t250 = t30*tz*6.0, t251 = oy*t56*6.0, t252 = t32*tz*6.0, t256 = t36*tz*4.0, t257 = t38*tz*4.0, t258 = oz*t57*6.0;
    T t260 = oz*t58*6.0, t264 = t13*t20, t267 = ox*t62, t268 = t70*3.0, t269 = t71*3.0, t270 = oy*t63, t271 = t74*3.0, t272 = t76*3.0, t273 = t80*3.0, t274 = t81*3.0, t275 = oz*t64, t279 = t20*tx, t280 = t87*4.0, t281 = t86*6.0, t282 = t21*ty;
    T t283 = t96*4.0, t284 = t93*6.0, t285 = t22*tz, t286 = t103*4.0, t287 = t102*6.0, t288 = t104*2.0, t289 = t105*2.0, t290 = t108*2.0, t291 = t109*2.0, t292 = t112*2.0, t293 = t113*2.0, t296 = -t24, t297 = -t118, t299 = t25*-2.0;
    T t300 = t28*-2.0, t302 = -t31, t304 = -t34, t308 = -t35, t311 = -t38, t312 = -t136, t315 = -t140, t316 = t39*-2.0, t324 = -t51, t325 = -t52, t326 = -t58, t331 = ox*t86*3.0, t334 = t24*tx*3.0, t338 = t25*ty*3.0, t340 = t28*tx*3.0;
    T t341 = t29*tz*2.0, t342 = t30*ty*2.0, t343 = t31*tz*2.0, t345 = t32*tx*2.0, t346 = t33*ty*2.0, t347 = t34*tx*2.0, t348 = t29*tz*3.0, t349 = t30*ty*3.0, t350 = t31*tz*3.0, t351 = oy*t93*3.0, t352 = t32*tx*3.0, t353 = t33*ty*3.0;
    T t354 = t34*tx*3.0, t357 = t35*tz*3.0, t359 = t38*ty*3.0, t362 = t39*tz*3.0, t365 = oz*t102*3.0, t372 = t16*t20, t373 = t11*t20, t374 = t198*3.0, t376 = t18*t20, t377 = t12*t20, t378 = t4*t63, t379 = t200*2.0, t380 = t199*3.0;
    T t383 = t200*3.0, t384 = t18*t21, t385 = t13*t21, t386 = t7*t64, t387 = t208*2.0, t388 = t207*3.0, t391 = t208*3.0, t392 = t9*t64, t393 = t209*2.0, t395 = t209*3.0, t399 = ox*t50*-2.0, t401 = ox*t53*-2.0, t403 = t26*ty*-2.0;
    T t404 = t27*tz*-2.0, t405 = oy*t54*-2.0, t409 = oy*t56*-2.0, t412 = oz*t55*-2.0, t413 = t36*tz*-2.0, t414 = t38*tz*-2.0, t415 = oz*t57*-2.0, t417 = t20*t44, t420 = -t73, t421 = -t77, t422 = -t83, t423 = -t276, t424 = -t277, t425 = -t278;
    T t443 = t44*ty*-2.0, t444 = t45*tz*-2.0, t445 = t46*tx*-2.0, t446 = rdx*t65*ty, t447 = rdy*t65*tx, t448 = rdx*t66*tz, t449 = rdz*t66*tx, t450 = rdy*t67*tz, t451 = rdz*t67*ty, t452 = t11*t60, t453 = t12*t59, t454 = t13*t61, t455 = t429*tx;
    T t456 = t429*ty, t457 = t430*tx, t458 = t430*tz, t459 = t431*ty, t460 = t431*tz, t461 = t20+t21, t462 = t20+t22, t463 = t21+t22, t464 = t65*(t13-tx), t468 = t66*(t11-tz), t471 = t67*(t12-ty), t473 = t14*t328, t474 = t16*t327, t475 = t18*t329;
    T t485 = t61*t327, t486 = t60*t328, t487 = t59*t329, t494 = t15+t67+t104+t108, t495 = t17+t66+t105+t112, t496 = t19+t65+t109+t113, t213 = rdx*t149, t214 = rdy*t150, t215 = rdz*t151, t216 = t116*ty, t218 = t116*tz, t219 = t119*ty;
    T t220 = t120*ty, t224 = ox*t164, t227 = oy*t161, t233 = t123*tz, t235 = oy*t165, t237 = t126*tz, t241 = t134*ty, t244 = t127*tz, t247 = oz*t166, t248 = t129*tz, t253 = t137*tz, t254 = t139*tz, t255 = oz*t168, t259 = oz*t171, t261 = t146*tz;
    T t262 = t148*tz, t263 = t151*tz, t298 = -t119, t305 = -t130, t307 = -t132, t313 = -t137, t314 = -t138, t318 = -t146, t319 = -t147, t320 = -t148, t321 = -t155, t322 = -t156, t323 = -t157, t330 = t116*tx, t332 = t120*tx, t333 = t178*3.0;
    T t337 = t180*3.0, t339 = t183*3.0, t344 = t127*ty, t355 = t135*tz, t356 = t139*ty, t358 = t185*3.0, t360 = t188*3.0, t363 = t190*3.0, t364 = t144*tz, t366 = t149*tz, t367 = t149*ty, t368 = t150*tz, t369 = t150*tx, t370 = t151*ty;
    T t371 = t151*tx, t407 = t300*tz, t408 = -t240, t411 = -t246, t418 = t4*t155, t419 = t7*t157, t432 = t299*ty, t433 = t300*tx, t435 = -t351, t436 = -t185, t437 = -t188, t439 = -t190, t440 = t316*tz, t442 = -t365, t476 = -t456, t477 = -t457;
    T t478 = -t460, t497 = t3+t4+t56+t458, t498 = t5+t7+t50+t459, t499 = t8+t9+t57+t455, t504 = t479+t486, t505 = t484+t485, t508 = t482+t487, t512 = t173+t473+t480, t513 = t173+t474+t483, t514 = t173+t475+t481;
    T t515 = t15+t104+t114+t115+t212+t290, t516 = t19+t106+t107+t109+t210+t293, t517 = t17+t110+t111+t112+t211+t289, t438 = -t360, t500 = t3+t4+t324+t478, t501 = t5+t7+t326+t476, t502 = t8+t9+t325+t477;
    T t518 = t25+t27+t117+t125+t135+t186+t228+t244+t314+t315+t352+t362+t401+t403, t519 = t26+t28+t121+t139+t145+t179+t216+t248+t312+t313+t338+t348+t411+t412, t520 = t39+t40+t120+t131+t143+t187+t241+t255+t297+t298+t340+t353+t404+t405;
    T t524 = t30+t33+t178+t219+t239+t305+t307+t331+t337+t408+t414+t438+t439+t442;
    
    fdrv(0, 0) = t66*t498+t65*t500-t329*t463; fdrv(0, 1) = t11*t463+t67*t498+t328*t500; fdrv(0, 2) = t12*t463+t67*t500+t327*t498;
    fdrv(0, 3) = -t178-t180+t185+t190+t220+t237+t339+t344+t360+t364+t419-t12*t24*3.0+t11*t28*3.0+t16*t33-t18*t32*2.0-t4*t47*2.0+t18*t33+t27*t67+t34*t63+t18*t302-t23*ty*2.0-t26*tz*2.0+t36*tz+t38*tz;
    fdrv(0, 4) = -t179+t191+t247-t341+t399+oy*t163-t13*t24*2.0+t14*t28+t16*t28*3.0+t11*t34*2.0+t18*t28-t4*t45-t6*t47*3.0+t16*t38*4.0-t18*t36*2.0+t7*t48+t9*t49*3.0-t8*t80+t21*t84+t24*tx-t25*ty*6.0-t30*tz*4.0+t32*tz+t34*tz;
    fdrv(0, 5) = -t181-t184+t235-t238-t342-ox*t51*2.0+oz*t170-t18*t24*3.0-t12*t32*2.0-t4*t43-t13*t36*2.0+t7*t46-t18*t39*4.0+t20*t49+t21*t49-t6*t80*3.0+t9*t81+t9*t84*3.0+t14*t296+t16*t296+t28*tx+t32*ty+t34*ty-t35*tz*6.0;
    fdrv(0, 6) = t520*tdy+tdz*(t28*2.0+t36+t38-t121-t123+t136+t145-t182+t245+t247-t334-t350-oy*t52*2.0-t32*tz*4.0)+tdx*(t35*-2.0+t124+t138-t230+t232+t259+t299+t356+t409+t440+oy*t4*6.0+oz*t7*6.0);
    fdrv(0, 7) = t520*tdx+tdy*(t25*-1.2E+1-t35*4.0+t140+t232-t251+t260+t347+t440-ox*t2*2.0+oy*t4*4.0+t38*ty*1.2E+1-t24*tz*2.0+t142*tz)+tdz*(t32+t34-t128-t129-t183+t188-t220-t256+t257-t358+t363+t365+t435+t28*tz*2.0);
    fdrv(0, 8) = tdx*(t28*2.0+t36+t38-t121-t123+t136+t145-t182+t245+t247-t334-t350-oy*t52*2.0-t32*tz*4.0)+tdy*(t32+t34-t128-t129-t183+t188-t220-t256+t257-t358+t363+t365+t435+t28*tz*2.0)-tdz*(t25*4.0+t35*1.2E+1+t116-t125+t230+t251-t260+t345-oz*t7*4.0-t38*ty*2.0+t137*ty+t300*ty+t39*tz*1.2E+1);
    fdrv(0, 9) = -tdx*(t70+t74+t82+t320+t322+t422+t45*tx*3.0-t46*tx*3.0+t47*ty-t49*tz)+tdy*tx*(t76+t84+t274+t321-ox*ty*2.0+oy*tx)-tdz*tx*(t71+t75+t273+t323+ox*tz*2.0-oz*tx);
    fdrv(0, 10) = tdx*(t49+t80+t269+t270+t318+t370-t43*tz*2.0)-tdz*(-t46+t78-t150+t152+t43*tx-t49*ty*2.0+t47*tz*3.0)-tdy*(t43*-4.0-t48+t68+t70*6.0+t82-t83*4.0+t445+ox*t64+t45*tx+t47*ty*3.0-t49*tz*2.0);
    fdrv(0, 11) = tdy*(t45+t85+t151-t152+t48*tx+t49*ty*3.0-t47*tz*2.0)+tdx*(t47+t81+t263+t272+t275+t319+t444)-tdz*(-t43-t48*4.0+t68+t74*6.0+t82*4.0+t369+t422+ox*t63-t46*tx+t155*ty-t49*tz*3.0);
    fdrv(0, 12) = -tdx*(t88+t92+t161+t168)-tdy*(t89+t98*2.0+t158+t172+t284)-tdz*(t94+t99*2.0+t160+t169+t287);
    fdrv(0, 13) = tdx*(t50*2.0+t58+t89*3.0+t93*2.0-t95+t98-t265-t381-t7*t18*2.0+t3*t67)-tdz*(-t55+t87+t96-t165+t198+t200+t286+t380+t387+t388+t393)-tdy*(-t57-t86+t100-t163+t204+t264+t389+t392);
    fdrv(0, 14) = tdx*(t56+t90+t94*3.0+t99+t160+t201+t266+t285+t375+t378)+tdy*(t54+t87+t103+t166+t199+t208+t283+t374+t379+t384+t395)+tdz*(t52+t86+t101+t170+t206+t264+t385+t394);
    fdrv(0, 15) = -oy*(t454+t328*(t12-ty))+oz*t505-ox*(t12*t61+t65*(t12-ty)); fdrv(0, 16) = oz*t513-ox*(t211+t464+t329*ty*2.0)-oy*(t115-t288+t328*(t13-tx)); fdrv(0, 17) = oz*t515-oy*t496*tz-ox*tz*(t14+t63+t64);
    fdrv(0, 18) = -rdz*(t147-oz*(t14*2.0+t64+t173+t327)+oy*tz*(t12*2.0-ty))+rdy*(oz*(t13+t288)-ox*(t11+t113-ty*(t13-tx))+oy*(t63+t328-tx*(t13-tx)*2.0))-rdx*(-oz*(t112+t61*tx*2.0+t327*ty)+oy*(t113+t328*tz+tx*(t12-ty)*2.0)+ox*(t61*tz-ty*(t12-ty)));
    fdrv(0, 19) = t176+t215-rdx*t41*2.0+rdx*t71+rdx*t79+rdy*t43*4.0+rdy*t48-rdy*t68-rdy*t70*6.0-rdy*t74*2.0-rdy*t82+rdy*t83*4.0-rdz*t44*4.0+rdz*t45+rdy*t371-rdy*t45*tx+rdz*t48*tx+rdx*t46*ty*3.0-rdy*t47*ty*3.0+rdz*t49*ty*3.0-rdx*t43*tz*2.0+rdx*t48*tz+rdy*t157*tz-rdz*t47*tz*2.0;
    fdrv(0, 20) = -t175+t214-rdx*t42*2.0+rdx*t76+rdx*t263-rdy*t44*4.0+rdy*t46+rdx*t420+rdz*t43+rdz*t48*4.0-rdz*t68-rdz*t70*2.0-rdz*t74*6.0-rdz*t82*4.0+rdz*t83-rdy*t43*tx-rdz*t45*tx*2.0+rdz*t46*tx-rdx*t43*ty+rdy*t157*ty-rdz*t47*ty*2.0-rdx*t45*tz*3.0-rdy*t47*tz*3.0+rdz*t49*tz*3.0;
    fdrv(0, 24) = t196*-2.0-t197*2.0+t424+t425+t446+t447+t448+t449; fdrv(0, 25) = -rdx*(t454+t328*(t12-ty))-rdy*(t115-t288+t328*(t13-tx))-rdz*t496*tz; fdrv(0, 26) = rdx*t505+rdy*t513+rdz*t515; fdrv(1, 0) = t11*t462+t66*t502+t329*t497;
    fdrv(1, 1) = t65*t497+t67*t502-t328*t462; fdrv(1, 2) = t13*t462+t66*t497+t327*t502;
    fdrv(1, 3) = -t182-t191+t247-t343+ox*t159-oy*t52*2.0-t14*t28*4.0+t2*t42*3.0-t11*t34*3.0+t3*t44-t7*t48*3.0-t9*t49+t5*t74+t26*t64+t13*t119+t16*t300+t18*t300+t16*t311+t18*t311-t24*tx*6.0+t25*ty+t30*tz-t32*tz*4.0+t33*tz;
    fdrv(1, 4) = t180-t183+t188+t219+t254+t330+t333+t363+t364+t413+t417+t436+t12*t23+t13*t25*3.0+t18*t29-t16*t34*3.0-t7*t49*2.0+t30*t64+t11*t300+t14*t304+t18*t304-t24*ty*2.0+t26*tz+t28*tz;
    fdrv(1, 5) = t177-t186+t224-t228-t345+t409+oz*t172+t11*t23+t16*t25+t13*t30*2.0+t18*t25*3.0-t13*t34*2.0+t18*t35*4.0+t2*t70-t7*t76-t7*t81-t7*t84*3.0+t12*t123+t2*t271+t12*t300+t26*ty+t28*ty+t38*ty-t39*tz*6.0;
    fdrv(1, 6) = t524*tdz+tdy*(t35+t37+t117+t122-t124-t125+t140-t189+t224-t229-t354-t359-oz*t58*2.0+t23*tz*2.0)-tdx*(t24*1.2E+1+t39*4.0+t127-t143-t223+t243+t258+t346-ox*t3*4.0+t28*tx*1.2E+1-t35*tz*2.0+t138*tz+t299*tz);
    fdrv(1, 7) = t519*tdz+tdx*(t35+t37+t117+t122-t124-t125+t140-t189+t224-t229-t354-t359-oz*t58*2.0+t23*tz*2.0)+tdy*(t24*-2.0+t119+t142+t218+t242-t243+t316+t355+t415+t433+ox*t3*6.0+oz*t9*6.0);
    fdrv(1, 8) = t524*tdx+t519*tdy+tdz*(t24*-4.0-t39*1.2E+1+t118+t223+t242-t258+t342+t433-oy*t6*2.0+oz*t9*4.0+t123*tx-t34*ty*2.0+t35*tz*1.2E+1);
    fdrv(1, 9) = tdz*(t46+t69+t149-t153-t48*tx*2.0+t41*ty+t42*tz*3.0)+tdy*(t48+t74+t261+t267+t268+t320+t445)-tdx*(t41*-4.0-t49+t71*6.0+t75+t79*4.0+t370+t421+oy*t64-t42*tx*3.0-t44*ty+t156*tz);
    fdrv(1, 10) = -tdy*(t71+t79+t80+t318+t323+t421-t42*tx-t44*ty*3.0+t46*ty*3.0+t48*tz)-tdx*ty*(t81+t84+t272+t319-ox*ty+oy*tx*2.0)+tdz*ty*(t68+t70+t271+t322-oy*tz*2.0+oz*ty);
    fdrv(1, 11) = tdy*(t42+t76+t274+t275+t321+t366-t46*tz*2.0)-tdx*(-t44+t85-t151+t153+t48*tx*3.0+t49*ty-t42*tz*2.0)-tdz*(-t41-t49*4.0+t75-t77*4.0+t79+t80*6.0+t443+oy*t62-t42*tx*2.0+t46*ty+t48*tz*3.0);
    fdrv(1, 12) = tdy*(t57+t88*3.0+t92+t100+t161+t204+t264+t279+t389+t392)+tdz*(t55+t87+t96+t164+t198+t200+t286+t380+t387+t388+t393)+tdx*(t58+t93+t95+t159+t202+t265+t381+t386);
    fdrv(1, 13) = -tdy*(t89+t98+t158+t171)-tdx*(t88+t92*2.0+t161+t170+t281)-tdz*(t94*2.0+t99+t162+t167+t287);
    fdrv(1, 14) = tdy*(t51+t56*2.0-t91+t94+t99*3.0+t102*2.0-t203-t266-t382+t20*t65)-tdx*(-t53+t96+t103-t166+t207+t209+t280+t372+t376+t383+t391)-tdz*(-t50-t93+t97-t172+t205+t265+t377+t390); fdrv(1, 15) = ox*t517-oz*t494*tx-oy*tx*(t16+t62+t64);
    fdrv(1, 16) = -oz*(t453+t327*(t11-tz))+ox*t508-oy*(t11*t59+t67*(t11-tz)); fdrv(1, 17) = ox*t514-oy*(t210+t471+t328*tz*2.0)-oz*(t110-t292+t327*(t12-ty));
    fdrv(1, 18) = -t176+t215+rdx*t41*4.0+rdx*t49-rdx*t71*6.0-rdx*t75+rdx*t77-rdx*t79*4.0-rdx*t80*2.0-rdy*t43*2.0+rdy*t70+rdy*t261+rdz*t44-rdz*t45*4.0+rdy*t422+rdx*t42*tx*3.0-rdy*t46*tx*3.0-rdz*t48*tx*3.0+rdx*t44*ty-rdx*t46*ty*2.0-rdz*t49*ty-rdx*t48*tz*2.0-rdy*t49*tz+rdz*t147*tz;
    fdrv(1, 19) = -rdx*(t148-ox*(t16*2.0+t62+t173+t329)+oz*tx*(t11*2.0-tz))+rdz*(ox*(t12+t292)-oy*(t13+t108-tz*(t12-ty))+oz*(t64+t327-ty*(t12-ty)*2.0))-rdy*(-ox*(t109+t59*ty*2.0+t329*tz)+oz*(t108+t327*tx+ty*(t11-tz)*2.0)+oy*(t59*tx-tz*(t11-tz)));
    fdrv(1, 20) = t174+t213-rdx*t45*4.0+rdx*t46-rdy*t47*2.0+rdy*t72+rdy*t81+rdz*t41+rdz*t49*4.0-rdz*t71*2.0-rdz*t75+rdz*t77*4.0-rdz*t79-rdz*t80*6.0+rdz*t367-rdx*t48*tx*2.0+rdy*t41*tx+rdz*t147*tx+rdx*t41*ty-rdz*t46*ty+rdx*t42*tz*3.0+rdy*t44*tz*3.0-rdy*t46*tz*2.0-rdz*t48*tz*3.0;
    fdrv(1, 24) = rdx*t517+rdy*t508+rdz*t514; fdrv(1, 25) = t194*-2.0-t195*2.0+t423+t425+t446+t447+t450+t451; fdrv(1, 26) = -rdy*(t453+t327*(t11-tz))-rdz*(t110-t292+t327*(t12-ty))-rdx*t494*tx; fdrv(2, 0) = t12*t461+t65*t499+t329*t501;
    fdrv(2, 1) = t13*t461+t65*t501+t328*t499; fdrv(2, 2) = t67*t499+t66*t501-t327*t461;
    fdrv(2, 3) = t184-t187+t235-t346+t415+ox*t162+t14*t24*4.0-t13*t26*2.0-t16*t23*2.0-t2*t41*3.0+t12*t32*3.0+t4*t43*3.0+t13*t36+t18*t39-t3*t70-t3*t74+t6*t80+t20*t75+t20*t80-t28*tx*6.0-t34*ty*4.0+t25*tz+t27*tz+t35*tz;
    fdrv(2, 4) = -t177-t189+t224-t229-t347+oy*t169-oz*t58*2.0-t11*t23*2.0-t16*t25*4.0-t13*t30*3.0+t13*t32*2.0+t20*t43-t2*t70*3.0+t4*t71-t2*t74+t4*t75*3.0+t4*t80+t26*t66+t18*t299+t18*t308-t38*ty*6.0+t23*tz+t24*tz+t39*tz;
    fdrv(2, 5) = t178+t183+t233+t253+t330+t337+t344+t358+t407+t414+t418+t437+t439-t2*t44*2.0+t14*t32-t16*t30+t16*t32-t18*t30*3.0+t18*t32*3.0+t20*t45+t26*t65+t13*t299+t23*ty+t24*ty;
    fdrv(2, 6) = t518*tdz+tdx*(t28*-1.2E+1-t38*4.0+t136-t217+t231+t252+t343+t432+ox*t5*4.0-oz*t10*2.0+t24*tx*1.2E+1+t124*ty-t30*tz*2.0)+tdy*(t29+t31-t133-t134-t180+t185-t221+t222+t253-t331-t333+t339+t351-t26*tz*2.0);
    fdrv(2, 7) = tdz*(t23+t24+t118+t131+t141-t142-t143-t181+t235+t236-t238-t349-t357-ox*t51*2.0)+tdx*(t29+t31-t133-t134-t180+t185-t221+t222+t253-t331-t333+t339+t351-t26*tz*2.0)-tdy*(t28*4.0+t38*1.2E+1-t121+t144+t217-t231+t250+t341-oy*t8*4.0-t24*tx*2.0+t119*tx+t25*ty*1.2E+1-t32*tz*2.0);
    fdrv(2, 8) = t518*tdx+tdy*(t23+t24+t118+t131+t141-t142-t143-t181+t235+t236-t238-t349-t357-ox*t51*2.0)+tdz*(t38*-2.0+t123+t137+t227-t250+t252+t300+t332+t399+t432+ox*t5*6.0+oy*t8*6.0);
    fdrv(2, 9) = tdz*(t43+t70+t267+t271+t322+t369-t41*tz*2.0)-tdy*(-t45+t69-t149+t154-t43*tx*2.0+t41*ty*3.0+t42*tz)-tdx*(t42*-4.0-t47+t72-t73*4.0+t76*6.0+t84+t444+oz*t63+t41*tx*3.0-t43*ty*2.0+t44*tz);
    fdrv(2, 10) = tdx*(t44+t78+t150-t154+t43*tx*3.0-t41*ty*2.0+t47*tz)+tdz*(t41+t71+t262+t270+t273+t323+t443)-tdy*(-t42-t47*4.0+t72*4.0+t81*6.0+t84+t366+t420+oz*t62+t146*tx-t43*ty*3.0-t45*tz);
    fdrv(2, 11) = -tdz*(t72+t76+t81+t319+t321+t420+t41*tx-t43*ty+t44*tz*3.0-t45*tz*3.0)+tdx*tz*(t75+t80+t269+t318+ox*tz-oz*tx*2.0)-tdy*tz*(t68+t74+t268+t320-oy*tz+oz*ty*2.0);
    fdrv(2, 12) = tdz*(t52+t86*2.0+t88+t92*3.0-t101+t168-t394+t7*t65+t20*t67+t21*t67)-tdy*(-t54+t87+t103-t164+t199+t208+t283+t374+t379+t384+t395)-tdx*(-t56+t90-t102-t162+t201+t266+t375+t378);
    fdrv(2, 13) = tdz*(t50+t89+t97+t98*3.0+t171+t205+t265+t282+t377+t390)+tdx*(t53+t96+t103+t165+t207+t209+t280+t372+t376+t383+t391)+tdy*(t51+t91+t102+t169+t203+t266+t373+t382);
    fdrv(2, 14) = -tdz*(t94+t99+t160+t167)-tdx*(t88*2.0+t92+t163+t168+t281)-tdy*(t89*2.0+t98+t159+t171+t284); fdrv(2, 15) = oy*t512-oz*(t212+t468+t327*tx*2.0)-ox*(t107-t291+t329*(t11-tz)); fdrv(2, 16) = oy*t516-ox*t495*ty-oz*ty*(t18+t62+t63);
    fdrv(2, 17) = -ox*(t452+t329*(t13-tx))+oy*t504-oz*(t13*t60+t66*(t13-tx));
    fdrv(2, 18) = t175+t214+rdx*t42*4.0+rdx*t47-rdx*t72+rdx*t73*4.0-rdx*t76*6.0-rdx*t81*2.0-rdx*t84+rdy*t44-rdy*t46*4.0+rdx*t368-rdz*t48*2.0+rdz*t74+rdz*t82-rdx*t41*tx*3.0+rdy*t43*tx*3.0+rdz*t45*tx*3.0+rdx*t148*ty-rdy*t41*ty*2.0+rdz*t47*ty-rdx*t44*tz+rdy*t47*tz-rdz*t41*tz*2.0;
    fdrv(2, 19) = -t174+t213+rdx*t45-rdx*t46*4.0+rdy*t42+rdy*t47*4.0-rdy*t72*4.0+rdy*t73-rdy*t76*2.0-rdy*t81*6.0-rdy*t84-rdz*t49*2.0+rdz*t80+rdz*t262+rdz*t421+rdx*t148*tx-rdy*t41*tx*2.0-rdz*t42*tx-rdx*t41*ty*3.0+rdy*t43*ty*3.0-rdz*t44*ty*3.0-rdx*t42*tz-rdy*t44*tz*2.0+rdy*t45*tz;
    fdrv(2, 20) = -rdy*(t157-oy*(t18*2.0+t63+t173+t328)+ox*ty*(t13*2.0-tx))+rdx*(oy*(t11+t291)-oz*(t12+t105-tx*(t11-tz))+ox*(t62+t329-tz*(t11-tz)*2.0))-rdz*(-oy*(t104+t328*tx+t60*tz*2.0)+ox*(t105+t329*ty+tz*(t13-tx)*2.0)+oz*(t60*ty-tx*(t13-tx)));
    fdrv(2, 24) = -rdz*(t452+t329*(t13-tx))-rdx*(t107-t291+t329*(t11-tz))-rdy*t495*ty; fdrv(2, 25) = rdx*t512+rdy*t516+rdz*t504; fdrv(2, 26) = t192*-2.0-t193*2.0+t423+t424+t448+t449+t450+t451;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f11(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = ox*ty, t4 = oy*tx, t5 = ox*tz, t6 = oy*ty, t7 = oz*tx, t8 = oy*tz, t9 = oz*ty, t10 = oz*tz, t11 = tx*tx, t12 = tx*tx*tx, t13 = ty*ty, t14 = ty*ty*ty, t15 = tz*tz, t16 = tz*tz*tz, t23 = rx*tx*3.0, t24 = ry*ty*3.0;
    T t25 = rz*tz*3.0, t32 = rx*tx*ty, t33 = rx*tx*tz, t34 = ry*tx*ty, t35 = ry*ty*tz, t36 = rz*tx*tz, t37 = rz*ty*tz, t17 = t3*2.0, t18 = t4*2.0, t19 = t5*2.0, t20 = t7*2.0, t21 = t8*2.0, t22 = t9*2.0, t26 = t2*ty, t27 = t2*tz, t28 = t4*ty;
    T t29 = t6*tz, t30 = t7*tz, t31 = t9*tz, t38 = t11*3.0, t39 = t13*3.0, t40 = t15*3.0, t41 = -t4, t43 = -t6, t44 = -t7, t46 = -t9, t48 = -t10, t49 = t2*tx, t50 = t2*t11, t51 = t3*ty, t52 = t4*tx, t53 = t3*t13, t54 = t4*t11, t55 = t5*tz;
    T t56 = t6*ty, t57 = t7*tx, t58 = t5*t15, t59 = t6*t13, t60 = t7*t11, t61 = t8*tz, t62 = t9*ty, t63 = t8*t15, t64 = t9*t13, t65 = t10*tz, t66 = t10*t15, t67 = rdx*t11, t68 = rdy*t13, t69 = rdz*t15, t70 = rx*t11, t71 = rx*t12, t72 = ry*t13;
    T t73 = ry*t14, t74 = rz*t15, t75 = rz*t16, t76 = t32*2.0, t77 = t33*2.0, t78 = t34*2.0, t79 = t35*2.0, t80 = t36*2.0, t81 = t37*2.0, t97 = t2*t13, t98 = t2*t15, t100 = t3*t15, t102 = t4*t15, t104 = t7*t13, t106 = t6*t15, t158 = t32*tz*4.0;
    T t160 = t34*tz*4.0, t162 = t36*ty*4.0, t184 = t11+t13, t185 = t11+t15, t186 = t13+t15, t187 = t2*t23, t203 = t6*t24, t250 = rx*t13*tx*2.0, t251 = rx*t15*tx*2.0, t252 = ry*t11*ty*2.0, t253 = rx*t15*ty*2.0, t254 = rx*t13*tz*2.0;
    T t255 = ry*t15*tx*2.0, t256 = ry*t11*tz*2.0, t257 = rz*t13*tx*2.0, t258 = rz*t11*ty*2.0, t259 = ry*t15*ty*2.0, t260 = rz*t11*tz*2.0, t261 = rz*t13*tz*2.0, t265 = rx*t3*tz*-2.0, t269 = rx*t4*tz*1.2E+1, t270 = rx*t7*ty*1.2E+1;
    T t271 = ry*t4*tz*-2.0, t277 = ry*t3*tz*1.2E+1, t278 = ry*t7*ty*1.2E+1, t279 = rz*t7*ty*-2.0, t283 = rz*t3*tz*1.2E+1, t284 = rz*t4*tz*1.2E+1, t332 = t4*t13*-2.0, t334 = t4*t13*-3.0, t339 = t7*t15*-2.0, t345 = t9*t15*-2.0, t346 = t9*t15*-3.0;
    T t384 = t23+t24, t385 = t23+t25, t386 = t24+t25, t42 = -t18, t45 = -t20, t47 = -t22, t82 = rx*t50, t83 = rx*t51, t84 = rx*t52, t85 = rx*t55, t86 = ry*t51, t87 = ry*t52, t88 = rx*t57, t89 = ry*t59, t90 = rz*t55, t91 = ry*t61, t92 = ry*t62;
    T t93 = rz*t57, t94 = rz*t61, t95 = rz*t62, t96 = rz*t66, t99 = t28*tx, t101 = t51*tz, t105 = t57*ty, t107 = t30*tx, t108 = t31*ty, t109 = rx*t26*2.0, t110 = rx*t26*6.0, t111 = rx*t27*2.0, t112 = ry*t26*2.0, t113 = rx*t18*ty;
    T t114 = ry*t26*4.0, t115 = rx*t28*4.0, t116 = rx*t27*6.0, t118 = rx*t18*tz, t119 = ry*t18*ty, t120 = rx*t20*ty, t121 = ry*t27*4.0, t122 = rz*t26*4.0, t123 = ry*t27*6.0, t124 = rz*t26*6.0, t125 = ry*t28*6.0, t126 = ry*t17*tz;
    T t127 = rz*t27*2.0, t130 = ry*t20*ty, t131 = rz*t27*4.0, t132 = rx*t29*4.0, t133 = rz*t28*4.0, t134 = rx*t30*4.0, t135 = rx*t29*6.0, t136 = rz*t28*6.0, t137 = rz*t17*tz, t138 = ry*t29*2.0, t139 = rz*t18*tz, t141 = rx*t31*4.0;
    T t142 = ry*t30*4.0, t143 = ry*t29*6.0, t144 = rx*t31*6.0, t145 = ry*t30*6.0, t146 = rz*t29*2.0, t148 = rz*t20*tz, t149 = rz*t29*4.0, t150 = ry*t31*4.0, t151 = rz*t30*6.0, t152 = rz*t22*tz, t153 = rz*t31*6.0, t154 = t26*tz*2.0;
    T t155 = t18*ty*tz, t156 = t20*ty*tz, t157 = t76*tz, t159 = t78*tz, t161 = t80*ty, t163 = t53*4.0, t164 = t54*4.0, t165 = t58*4.0, t166 = t60*4.0, t167 = t63*4.0, t168 = t64*4.0, t169 = t71*2.0, t170 = t71*4.0, t171 = t73*2.0, t172 = t73*4.0;
    T t173 = t75*2.0, t174 = t75*4.0, t178 = -t76, t179 = -t77, t180 = -t78, t181 = -t79, t182 = -t80, t183 = -t81, t188 = ry*t49*2.0, t190 = ry*t49*3.0, t192 = rz*t49*2.0, t193 = rx*t56*2.0, t195 = rz*t49*3.0, t196 = rx*t56*3.0;
    T t200 = ry*t55*3.0, t201 = rz*t51*3.0, t202 = rx*t61*3.0, t204 = rz*t52*3.0, t205 = rx*t62*3.0, t206 = ry*t57*3.0, t207 = ry*t55*6.0, t208 = rz*t51*6.0, t209 = rx*t61*6.0, t210 = rz*t52*6.0, t211 = rx*t62*6.0, t212 = ry*t57*6.0;
    T t213 = rz*t56*2.0, t214 = rx*t65*2.0, t216 = rz*t56*3.0, t217 = rx*t65*3.0, t221 = ry*t65*2.0, t222 = ry*t65*3.0, t226 = t97*2.0, t227 = t26*tx*2.0, t228 = t2*t39, t229 = t26*tx*3.0, t230 = t98*2.0, t231 = t27*tx*2.0, t232 = t13*t18;
    T t234 = t2*t40, t235 = t27*tx*3.0, t236 = t4*t39, t238 = t106*2.0, t239 = t29*ty*2.0, t240 = t15*t20, t241 = t20*tx*tz, t242 = t6*t40, t243 = t29*ty*3.0, t244 = t7*t40, t246 = t15*t22, t247 = t22*ty*tz, t248 = t9*t40, t263 = rx*t28*-2.0;
    T t272 = rx*t30*-2.0, t288 = t28*tz*-2.0, t289 = t30*ty*-2.0, t291 = t18*t35, t292 = t20*t37, t293 = t11*t41, t296 = t13*t43, t300 = t46*ty, t301 = t13*t46, t303 = t15*t48, t304 = rx*t41*tx, t307 = rx*t44*tx, t321 = rz*t44*tx;
    T t329 = rz*t65*-3.0, t335 = t15*t41, t341 = t106*-3.0, t348 = -t253, t349 = -t254, t350 = -t255, t351 = -t256, t352 = -t257, t353 = -t258, t366 = ry*t27*tx*-2.0, t367 = rz*t332, t368 = rx*t345, t369 = rx*t184, t370 = rx*t185, t371 = ry*t184;
    T t372 = ry*t186, t373 = rz*t185, t374 = rz*t186, t375 = t2+t43, t376 = t3+t41, t378 = t2+t48, t379 = t5+t44, t381 = t6+t48, t382 = t8+t46, t387 = t15+t184, t391 = t384*tx, t392 = t384*ty, t393 = t385*tx, t394 = t385*tz, t395 = t386*ty;
    T t396 = t386*tz, t397 = t38+t186, t398 = t39+t185, t399 = t40+t184, t189 = t83*3.0, t191 = t84*6.0, t194 = t85*3.0, t197 = t87*3.0, t198 = t86*6.0, t199 = t88*6.0, t215 = t91*3.0, t218 = t93*3.0, t219 = t90*6.0, t220 = t92*6.0;
    T t223 = t95*3.0, t224 = t94*6.0, t237 = t99*3.0, t245 = t107*3.0, t249 = t108*3.0, t262 = -t112, t266 = -t121, t267 = -t122, t268 = -t124, t273 = -t132, t274 = -t133, t275 = -t134, t276 = -t136, t280 = -t141, t281 = -t142, t282 = -t145;
    T t286 = -t150, t287 = -t154, t290 = t109*tz, t294 = -t164, t298 = -t166, t305 = -t190, t306 = -t86, t308 = -t195, t309 = -t196, t311 = -t200, t312 = -t201, t313 = -t202, t315 = -t204, t316 = -t205, t317 = -t206, t318 = -t90, t320 = ry*t300;
    T t322 = -t216, t323 = -t217, t325 = -t94, t327 = -t222, t331 = -t99, t333 = t99*-2.0, t338 = -t107, t340 = t107*-2.0, t354 = t112*tx, t355 = rx*t232, t357 = rz*t227, t358 = t127*tx, t359 = rx*t239, t361 = rx*t240, t363 = ry*t240;
    T t364 = t146*ty, t365 = ry*t246, t377 = t3+t42, t380 = t5+t45, t383 = t8+t47, t400 = -t391, t401 = -t392, t402 = -t393, t403 = -t394, t404 = -t395, t405 = -t396, t409 = -t387*(t28-t30-t51+t55), t411 = -t387*(t27-t29+t62+t44*tx);
    T t412 = t387*(t26-t31+t61+t41*tx), t427 = -tdy*(-t63+t154+t168-t243+t246+t41*tx*tz), t428 = -tdy*(t54+t102-t163-t227+t236+t289), t432 = tdz*(t60+t104-t165-t231+t244+t288), t324 = -t218, t328 = -t223, t347 = -t249;
    T t416 = t34+t182+t369+t370+t400, t417 = t36+t180+t369+t370+t402, t418 = t32+t183+t371+t372+t401, t419 = t37+t178+t371+t372+t404, t420 = t33+t181+t373+t374+t403, t421 = t35+t179+t373+t374+t405, t425 = tdx*(t53+t100-t156+t229-t232+t294);
    T t426 = tdx*(t58+t101-t155+t235-t240+t298), t431 = -tdz*(t64+t105-t167-t239+t248+t287), t433 = t50+t97+t234+t247+t296+t331+t340+t341, t440 = tdx*(t66+t108+t226-t230-t237+t245+t296+t15*t43);
    T t441 = t109+t139+t141+t188+t198+t207+t279+t281+t304+t309+t313, t442 = t119+t137+t142+t191+t193+t209+t279+t280+t305+t306+t311, t443 = t111+t130+t132+t192+t208+t219+t271+t274+t307+t316+t323;
    T t444 = t126+t133+t148+t199+t211+t214+t271+t273+t308+t312+t318, t445 = t120+t121+t138+t210+t213+t224+t265+t267+t317+t320+t327, t446 = t118+t122+t152+t212+t220+t221+t265+t266+t315+t322+t325, t429 = -t426;
    T t434 = t50+t98+t228+t238+t303+t333+t338+t347, t436 = t433*tdz, t449 = t87+t115+t127+t149+t203+t215+t262+t275+t286+t321+t328+t329, t452 = t428+t432+t440, t437 = t434*tdy, t439 = -t436, t450 = t425+t431+t437, t451 = t427+t429+t439;
    
    fdrv(0, 0) = t421*tx*ty-t419*tx*tz; fdrv(0, 1) = -t185*t421-t419*ty*tz; fdrv(0, 2) = t184*t419+t421*ty*tz;
    fdrv(0, 3) = t291+t359+t363+t367+t368-rx*t64*2.0-ry*t58+ry*t166+rz*t53-rz*t54*4.0+rz*t229+rz*t289+t4*t33*6.0-t3*t35-t7*t32*6.0+t3*t74+rx*t15*t21-ry*t27*tx*3.0;
    fdrv(0, 4) = rx*t60*-2.0-rx*t104*6.0+rx*t339+ry*t63-ry*t64*4.0+ry*t345+rz*t50-rz*t108*3.0+rz*t228+rz*t333+rz*t338+t24*t29+t2*t74+t6*t74*2.0+t48*t74+t87*tz+t115*tz-ry*t26*tz*2.0;
    fdrv(0, 5) = t89+t355+t364+rx*t102*6.0-ry*t50-ry*t98*3.0+ry*t99-ry*t108*2.0+ry*t241+ry*t242+rz*t154+rz*t167+rz*t301-t2*t72-t9*t74*3.0+t18*t70+t321*ty-rx*t30*ty*4.0;
    fdrv(0, 6) = -t444*tdy+t442*tdz-tdx*(t123-t138+t152+t213-t221+t268-t269+t270-ry*t57*1.2E+1+rz*t52*1.2E+1); fdrv(0, 7) = -tdy*(t92*1.2E+1-t94*2.0-t143+t153+t221+t268+t270+ry*t27*2.0-rx*t4*tz*4.0+rz*t18*tx)-t444*tdx+t449*tdz;
    fdrv(0, 8) = t442*tdx+t449*tdy+tdz*(t92*-2.0+t94*1.2E+1-t123+t143-t153+t213+t269+rz*t26*2.0-rx*t7*ty*4.0+ry*t20*tx); fdrv(0, 9) = tdy*tx*(t29*-2.0+t57+t62*3.0+t65)*-2.0+tdz*tx*(t31*-2.0+t52+t56+t61*3.0)*2.0+t382*t397*tdx*2.0; fdrv(0, 10) = t451;
    fdrv(0, 11) = t450; fdrv(0, 12) = -tdz*tx*(t72+t183+ry*t11+ry*t40)+tdy*tx*(t74+t181+rz*t11+rz*t39)-t397*tdx*(ry*tz-rz*ty);
    fdrv(0, 13) = tdy*(t158+t353+ry*t16+t74*ty*2.0+ry*t11*tz+ry*t39*tz)+tdz*(t73+t169+t174+t250+t261+t15*t24+rx*t15*tx*6.0+ry*t11*ty)+tdx*(t159+t254+t352+rx*t16*2.0-rz*t12*4.0+t70*tz*6.0);
    fdrv(0, 14) = -tdz*(t158+t351+rz*t14+t72*tz*2.0+rz*t11*ty+rz*t40*ty)-tdy*(t75+t169+t172+t251+t259+t13*t25+rx*t13*tx*6.0+rz*t11*tz)-tdx*(t161+t253+t350+rx*t14*2.0-ry*t12*4.0+t70*ty*6.0); fdrv(0, 15) = t382*t387*tx*2.0; fdrv(0, 16) = t411;
    fdrv(0, 17) = t412; fdrv(0, 18) = t67*t382*4.0-rdy*tx*(t27-t29+t62+t44*tx)*2.0+rdz*tx*(t26-t31+t61+t41*tx)*2.0+rdx*t382*t387*2.0-rdy*t380*t387+rdz*t377*t387;
    fdrv(0, 19) = rdy*ty*(t27-t29+t62+t44*tx)*-2.0+rdz*ty*(t26-t31+t61+t41*tx)*2.0-rdx*t7*t387*2.0+rdy*t383*t387+rdz*t378*t387+rdx*t382*tx*ty*4.0;
    fdrv(0, 20) = rdy*tz*(t27-t29+t62+t44*tx)*-2.0+rdz*tz*(t26-t31+t61+t41*tx)*2.0+rdx*t18*t387-rdy*t375*t387-rdz*t387*(t9-t21)+rdx*t382*tx*tz*4.0; fdrv(0, 24) = t387*tx*(rdz*ty-rdy*tz); fdrv(0, 25) = t387*(t69-rdz*t11+rdx*tx*tz*2.0+rdy*ty*tz);
    fdrv(0, 26) = -t387*(t68-rdy*t11+rdx*tx*ty*2.0+rdz*ty*tz); fdrv(1, 0) = t186*t420+t417*tx*tz; fdrv(1, 1) = -t420*tx*ty+t417*ty*tz; fdrv(1, 2) = -t184*t417-t420*tx*tz;
    fdrv(1, 3) = t96+t361+t365-rx*t58+rx*t166-rz*t99*3.0+rz*t108+rz*t226+rz*t296+t7*t34*6.0-t2*t74*2.0+t25*t57+t22*t72+t43*t74-t83*tz+t113*tz-rx*t27*tx*3.0-ry*t26*tz*4.0;
    fdrv(1, 4) = t292+t357+t363+t366+t368+rx*t63-rx*t64*4.0+rx*t243-ry*t58*2.0+rz*t163+rz*t293+rz*t334+t4*t33-t3*t35*6.0+t7*t72*6.0+t41*t74+ry*t11*t20-rx*t26*tz*2.0;
    fdrv(1, 5) = -t82+rx*t59-rx*t97-rx*t98*3.0+rx*t99-rx*t108*2.0+rx*t242-ry*t53*2.0-ry*t100*6.0-rz*t58*4.0+rz*t60+rz*t104+rz*t244+rz*t288+t20*t33+t142*ty-ry*t26*tx*2.0-rz*t27*tx*2.0;
    fdrv(1, 6) = -tdz*(t83-t95+t114+t131+t146+t187+t194+t263+t275+t286+t324+t329)+tdx*(t88*1.2E+1-t90*2.0-t116+t151+t214+t276+t278+rx*t29*2.0-ry*t3*tz*4.0+rz*t17*ty)+t446*tdy;
    fdrv(1, 7) = t446*tdx-t441*tdz-tdy*(t111-t135+t136-t192+t214+t277-t278+rx*t62*1.2E+1-rz*t30*2.0-rz*t51*1.2E+1);
    fdrv(1, 8) = -tdx*(t83-t95+t114+t131+t146+t187+t194+t263+t275+t286+t324+t329)-tdz*(t88*-2.0+t90*1.2E+1+t116-t135-t151+t192+t277+rx*t22*ty-ry*t7*ty*4.0+rz*t18*ty)-t441*tdy; fdrv(1, 9) = t451;
    fdrv(1, 10) = tdx*ty*(t27*-2.0+t57*3.0+t62+t65)*2.0-tdz*ty*(t30*-2.0+t49+t51+t55*3.0)*2.0-t379*t398*tdy*2.0; fdrv(1, 11) = t452;
    fdrv(1, 12) = -tdx*(t160+t352+rx*t16+t74*tx*2.0+rx*t13*tz+rx*t38*tz)-tdz*(t71+t171+t174+t252+t260+t15*t23+rx*t13*tx+ry*t15*ty*6.0)-tdy*(t157+t256+t353+ry*t16*2.0-rz*t14*4.0+t72*tz*6.0);
    fdrv(1, 13) = tdz*ty*(t70+t182+rx*t13+rx*t40)-tdx*ty*(t74+t179+rz*t13+rz*t38)+t398*tdy*(rx*tz-rz*tx);
    fdrv(1, 14) = tdz*(t160+t349+rz*t12+t70*tz*2.0+rz*t13*tx+rz*t40*tx)+tdx*(t75+t170+t171+t251+t259+t11*t25+ry*t11*ty*6.0+rz*t13*tz)+tdy*(t161+t255+t348-rx*t14*4.0+ry*t12*2.0+t72*tx*6.0); fdrv(1, 15) = t411; fdrv(1, 16) = t379*t387*ty*-2.0;
    fdrv(1, 17) = t409; fdrv(1, 18) = rdz*tx*(t28-t30-t51+t55)*-2.0-rdx*tx*(t27-t29+t62+t44*tx)*2.0+rdy*t22*t387-rdx*t380*t387-rdz*t381*t387-rdy*t379*tx*ty*4.0;
    fdrv(1, 19) = t68*t379*-4.0-rdz*ty*(t28-t30-t51+t55)*2.0-rdx*ty*(t27-t29+t62+t44*tx)*2.0+rdx*t383*t387-rdy*t379*t387*2.0-rdz*t387*(t4-t17);
    fdrv(1, 20) = rdz*tz*(t28-t30-t51+t55)*-2.0-rdx*tz*(t27-t29+t62+t44*tx)*2.0-rdy*t3*t387*2.0-rdx*t375*t387+rdz*t387*(t7-t19)-rdy*t379*ty*tz*4.0; fdrv(1, 24) = -t387*(t69-rdz*t13+rdx*tx*tz+rdy*ty*tz*2.0); fdrv(1, 25) = -t387*ty*(rdz*tx-rdx*tz);
    fdrv(1, 26) = t387*(t67-rdx*t13+rdy*tx*ty*2.0+rdz*tx*tz); fdrv(2, 0) = -t186*t418-t416*tx*ty; fdrv(2, 1) = t185*t416+t418*tx*ty; fdrv(2, 2) = t418*tx*tz-t416*ty*tz;
    fdrv(2, 3) = rx*t53-rx*t54*4.0+rx*t100+rx*t332+ry*t66-ry*t98*2.0-ry*t99*3.0+ry*t108+ry*t245-rz*t63*2.0-t4*t36*6.0+t23*t26+t2*t72*2.0+t43*t72+t272*ty+t122*tz+ry*t15*t43-rz*t29*ty*2.0;
    fdrv(2, 4) = t82+t354+t358+rx*t98-rx*t108*3.0+rx*t228+rx*t238+rx*t303+rx*t333+rx*t338+ry*t163+ry*t293+ry*t335+t3*t37*6.0+t20*t35-t4*t72*3.0+t19*t74-rz*t28*tz*4.0;
    fdrv(2, 5) = t290+t357+t359+t366+t367+rx*t167+rx*t301+rx*t346-ry*t58*4.0+ry*t60+ry*t244+ry*t288-rz*t54*2.0+t32*t44+t3*t74*6.0-t4*t74*6.0+t7*t72+rz*t13*t17;
    fdrv(2, 6) = -t445*tdz-tdy*(-t85+t91-t114+t115-t131+t149-t187-t189+t197+rx*t30*2.0+ry*t56*3.0+ry*t47*tz)-tdx*(t84*1.2E+1-t86*2.0-t110+t125+t193+t282+t284+rx*t22*tz+ry*t19*tz-rz*t3*tz*4.0);
    fdrv(2, 7) = tdy*(t84*-2.0+t86*1.2E+1+t110-t125-t144+t188+t283+rx*t21*tz+ry*t20*tz-rz*t4*tz*4.0)+t443*tdz-tdx*(-t85+t91-t114+t115-t131+t149-t187-t189+t197+rx*t30*2.0+ry*t56*3.0+ry*t47*tz);
    fdrv(2, 8) = -t445*tdx+t443*tdy+tdz*(t109-t144+t145-t188+t193+t283-t284+rx*t61*1.2E+1-ry*t28*2.0-ry*t55*1.2E+1); fdrv(2, 9) = t450; fdrv(2, 10) = t452;
    fdrv(2, 11) = tdx*tz*(t26*-2.0+t52*3.0+t56+t61)*-2.0+tdy*tz*(t28*-2.0+t49+t51*3.0+t55)*2.0+t376*t399*tdz*2.0;
    fdrv(2, 12) = tdx*(t162+t350+rx*t14+t72*tx*2.0+rx*t15*ty+rx*t38*ty)+tdy*(t71+t172+t173+t252+t260+t13*t23+rx*t15*tx+rz*t13*tz*6.0)+tdz*(t157+t258+t351-ry*t16*4.0+rz*t14*2.0+t74*ty*6.0);
    fdrv(2, 13) = -tdy*(t162+t348+ry*t12+t70*ty*2.0+ry*t15*tx+ry*t39*tx)-tdx*(t73+t170+t173+t250+t261+t11*t24+ry*t15*ty+rz*t11*tz*6.0)-tdz*(t159+t257+t349-rx*t16*4.0+rz*t12*2.0+t74*tx*6.0);
    fdrv(2, 14) = -tdy*tz*(t70+t180+rx*t15+rx*t39)+tdx*tz*(t72+t178+ry*t15+ry*t38)-t399*tdz*(rx*ty-ry*tx); fdrv(2, 15) = t412; fdrv(2, 16) = t409; fdrv(2, 17) = t376*t387*tz*2.0;
    fdrv(2, 18) = rdy*tx*(t28-t30-t51+t55)*-2.0+rdx*tx*(t26-t31+t61+t41*tx)*2.0+rdx*t377*t387-rdz*t8*t387*2.0-rdy*t381*t387+rdz*t376*tx*tz*4.0;
    fdrv(2, 19) = rdy*ty*(t28-t30-t51+t55)*-2.0+rdx*ty*(t26-t31+t61+t41*tx)*2.0+rdx*t378*t387+rdz*t19*t387-rdy*t387*(t4-t17)+rdz*t376*ty*tz*4.0;
    fdrv(2, 20) = t69*t376*4.0-rdy*tz*(t28-t30-t51+t55)*2.0+rdx*tz*(t26-t31+t61+t41*tx)*2.0+rdz*t376*t387*2.0-rdx*t387*(t9-t21)+rdy*t387*(t7-t19); fdrv(2, 24) = t387*(t68-rdy*t15+rdx*tx*ty+rdz*ty*tz*2.0);
    fdrv(2, 25) = -t387*(t67-rdx*t15+rdy*tx*ty+rdz*tx*tz*2.0); fdrv(2, 26) = t387*tz*(rdy*tx-rdx*ty);
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f12(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = ox*ty, t4 = oy*tx, t5 = ox*tz, t6 = oy*ty, t7 = oz*tx, t8 = oy*tz, t9 = oz*ty, t10 = oz*tz, t11 = rdx*tx, t12 = rdy*ty, t13 = rdz*tz, t14 = rx*tx, t15 = ry*ty, t16 = rz*tz, t17 = tx*tx, t18 = tx*tx*tx, t20 = ty*ty;
    T t21 = ty*ty*ty, t23 = tz*tz, t24 = tz*tz*tz, t41 = rx*ty*tz, t42 = ry*tx*tz, t43 = rz*tx*ty, t19 = t17*t17, t22 = t20*t20, t25 = t23*t23, t26 = t3*2.0, t27 = t4*2.0, t28 = t5*2.0, t29 = t7*2.0, t30 = t8*2.0, t31 = t9*2.0, t32 = t2*ty;
    T t33 = t2*tz, t34 = t4*ty, t35 = t6*tz, t36 = t7*tz, t37 = t9*tz, t38 = t14*ty, t39 = t14*tz, t40 = t15*tx, t44 = t15*tz, t45 = t16*tx, t46 = t16*ty, t47 = -t4, t49 = -t7, t51 = -t9, t53 = t2*t17, t54 = t3*ty, t55 = t4*tx, t56 = t3*t20;
    T t57 = t4*t17, t58 = t5*tz, t59 = t7*tx, t60 = t5*t23, t61 = t6*t20, t62 = t7*t17, t63 = t8*tz, t64 = t9*ty, t65 = t8*t23, t66 = t9*t20, t67 = t10*t23, t68 = t14*t17, t69 = rx*t20, t70 = ry*t17, t71 = rx*t23, t72 = rz*t17, t73 = t15*t20;
    T t74 = ry*t23, t75 = rz*t20, t76 = t16*t23, t83 = t2*t20, t84 = t2*t23, t86 = t3*t23, t88 = t4*t23, t90 = t7*t20, t92 = t6*t23, t95 = t14*t20, t96 = t14*t23, t97 = t15*t17, t104 = t15*t23, t105 = t16*t17, t106 = t16*t20, t122 = t14*tx*3.0;
    T t124 = t15*ty*3.0, t126 = t16*tz*3.0, t134 = -t42, t135 = -t43, t136 = t17+t20, t137 = t17+t23, t138 = t20+t23, t140 = t2*t14*tx*4.0, t154 = t6*t15*ty*4.0, t162 = t10*t16*tz*4.0, t173 = t4*t20*3.0, t187 = t7*t23*3.0, t191 = t9*t23*3.0;
    T t205 = t3*t21*-2.0, t206 = t4*t18*-2.0, t209 = t5*t24*-2.0, t210 = t7*t18*-2.0, t213 = t8*t24*-2.0, t214 = t9*t21*-2.0, t217 = t14*t18*1.0E+1, t218 = t15*t21*1.0E+1, t219 = t16*t24*1.0E+1, t220 = t2+t6, t221 = t2+t10, t222 = t6+t10;
    T t223 = t14+t15, t224 = t14+t16, t225 = t15+t16, t227 = t2*t14*tx*8.0, t238 = t6*t15*ty*8.0, t249 = t10*t16*tz*8.0, t254 = t4*t20*-2.0, t264 = t7*t23*-2.0, t266 = t9*t23*-2.0, t268 = t14*t21*8.0, t269 = t14*t24*8.0, t270 = t15*t18*8.0;
    T t277 = t15*t24*8.0, t278 = t16*t18*8.0, t279 = t16*t21*8.0, t312 = t7*t15*ty*4.0, t324 = t7*t15*ty*6.0, t326 = t3*t16*tz*4.0, t330 = t4*t16*tz*4.0, t336 = t3*t16*tz*6.0, t338 = t4*t16*tz*6.0, t410 = t3*t41*8.0, t422 = t4*t42*8.0;
    T t429 = t7*t15*ty*2.4E+1, t436 = t7*t43*8.0, t439 = t3*t16*tz*2.4E+1, t441 = t4*t16*tz*2.4E+1, t484 = t11+t12+t13, t48 = -t27, t50 = -t29, t52 = -t31, t77 = t38*2.0, t78 = t39*2.0, t79 = t40*2.0, t80 = t44*2.0, t81 = t45*2.0, t82 = t46*2.0;
    T t85 = t34*tx, t87 = t54*tz, t89 = t55*tz, t91 = t59*ty, t93 = t36*tx, t94 = t37*ty, t98 = t71*ty, t99 = t69*tz, t100 = t74*tx, t101 = t70*tz, t102 = t75*tx, t103 = t72*ty, t107 = t32*tz*2.0, t108 = t27*ty*tz, t109 = t29*ty*tz;
    T t110 = t21*t26, t111 = t18*t27, t112 = t56*4.0, t113 = t57*4.0, t114 = t24*t28, t115 = t18*t29, t116 = t60*4.0, t117 = t62*4.0, t118 = t24*t30, t119 = t21*t31, t120 = t65*4.0, t121 = t66*4.0, t123 = t68*4.0, t125 = t73*4.0, t127 = t76*4.0;
    T t139 = t14*t53*2.0, t141 = ry*t53*2.0, t142 = t14*t27*tx, t143 = t15*t26*ty, t144 = rz*t53*2.0, t145 = rx*t61*2.0, t146 = t14*t29*tx, t147 = t28*t74, t148 = t26*t75, t149 = t30*t71, t150 = t27*t72, t151 = t31*t69, t152 = t29*t70;
    T t153 = t15*t61*2.0, t155 = t16*t28*tz, t156 = rz*t61*2.0, t157 = rx*t67*2.0, t158 = t15*t31*ty, t159 = t16*t30*tz, t160 = ry*t67*2.0, t161 = t16*t67*2.0, t163 = t83*2.0, t164 = t32*tx*2.0, t165 = t83*3.0, t166 = t32*tx*3.0, t167 = t84*2.0;
    T t168 = t33*tx*2.0, t171 = t84*3.0, t172 = t33*tx*3.0, t175 = t86*4.0, t177 = t88*4.0, t179 = t90*4.0, t181 = t92*2.0, t182 = t35*ty*2.0, t185 = t92*3.0, t186 = t35*ty*3.0, t193 = t95*2.0, t194 = t95*3.0, t195 = t96*2.0, t196 = t97*2.0;
    T t197 = t96*3.0, t198 = t97*3.0, t199 = t104*2.0, t200 = t105*2.0, t201 = t104*3.0, t202 = t105*3.0, t203 = t106*2.0, t204 = t106*3.0, t228 = rx*t56*8.0, t229 = t14*t55*8.0, t230 = rx*t60*8.0, t231 = t15*t54*8.0, t232 = ry*t57*8.0;
    T t233 = t14*t59*8.0, t235 = ry*t60*8.0, t236 = rz*t56*8.0, t237 = rx*t65*8.0, t239 = rz*t57*8.0, t240 = rx*t66*8.0, t241 = ry*t62*8.0, t242 = t16*t58*8.0, t243 = ry*t65*8.0, t244 = t15*t64*8.0, t245 = rz*t62*8.0, t246 = t16*t63*8.0;
    T t247 = rz*t66*8.0, t280 = t2*t69*4.0, t281 = t14*t32*4.0, t282 = t2*t71*4.0, t283 = t14*t33*4.0, t284 = t15*t32*6.0, t285 = t2*t40*6.0, t286 = t4*t69*6.0, t287 = t14*t34*6.0, t288 = t2*t74*4.0, t289 = ry*t33*tx*4.0, t290 = t2*t75*4.0;
    T t291 = rz*t32*tx*4.0, t292 = t4*t71*4.0, t293 = t4*t39*4.0, t294 = t15*t34*4.0, t295 = t4*t40*4.0, t296 = t7*t69*4.0, t297 = t7*t38*4.0, t298 = t2*t74*6.0, t299 = ry*t33*tx*6.0, t300 = t2*t75*6.0, t301 = rz*t32*tx*6.0, t302 = t4*t71*6.0;
    T t303 = t4*t39*6.0, t304 = t7*t69*6.0, t305 = t7*t38*6.0, t306 = t3*t74*4.0, t307 = t3*t44*4.0, t308 = t6*t71*4.0, t309 = rx*t35*ty*4.0, t310 = t4*t75*4.0, t313 = t7*t40*4.0, t314 = t3*t74*6.0, t315 = t3*t44*6.0, t316 = t16*t33*6.0;
    T t317 = t2*t45*6.0, t318 = t6*t71*6.0, t319 = rx*t35*ty*6.0, t320 = t4*t75*6.0, t322 = t7*t71*6.0, t323 = t14*t36*6.0, t325 = t7*t40*6.0, t327 = t3*t46*4.0, t328 = t6*t74*4.0, t329 = t15*t35*4.0, t331 = t4*t45*4.0, t332 = t9*t71*4.0;
    T t334 = t7*t74*4.0, t337 = t3*t46*6.0, t339 = t4*t45*6.0, t340 = t9*t71*6.0, t342 = t7*t74*6.0, t344 = t16*t36*4.0, t345 = t7*t45*4.0, t346 = t16*t35*6.0, t347 = t6*t46*6.0, t348 = t9*t74*6.0, t349 = t15*t37*6.0, t350 = t16*t37*4.0;
    T t351 = t9*t46*4.0, t352 = t23*t32*4.0, t353 = t20*t33*4.0, t354 = t23*t34*4.0, t356 = t20*t36*4.0, t358 = rx*t32*tz*8.0, t359 = rx*t32*tz*1.6E+1, t360 = t15*t33*8.0, t361 = rx*t34*tz*8.0, t362 = t15*t33*1.2E+1, t363 = rx*t34*tz*1.2E+1;
    T t364 = t16*t32*8.0, t365 = t4*t44*8.0, t366 = rx*t36*ty*8.0, t367 = t16*t32*1.2E+1, t368 = rx*t36*ty*1.2E+1, t369 = t4*t44*1.6E+1, t370 = t16*t34*8.0, t371 = t15*t36*8.0, t372 = t16*t34*1.2E+1, t373 = t15*t36*1.2E+1, t374 = t7*t46*8.0;
    T t375 = t7*t46*1.6E+1, t388 = t83*tx*6.0, t389 = t84*tx*6.0, t390 = t20*t55*6.0, t391 = t92*ty*6.0, t392 = t23*t59*6.0, t393 = t23*t64*6.0, t394 = t23*t69*2.0, t395 = t23*t70*2.0, t396 = t20*t72*2.0, t397 = t2*t69*1.2E+1;
    T t398 = t14*t32*1.2E+1, t399 = t2*t71*1.2E+1, t400 = t14*t33*1.2E+1, t401 = t15*t32*1.2E+1, t402 = t2*t40*1.2E+1, t403 = t4*t69*1.2E+1, t404 = t14*t34*1.2E+1, t409 = t3*t71*8.0, t411 = t15*t34*1.2E+1, t412 = t4*t40*1.2E+1;
    T t413 = t4*t71*2.4E+1, t414 = t4*t39*2.4E+1, t415 = t7*t69*2.4E+1, t416 = t7*t38*2.4E+1, t421 = t4*t74*8.0, t423 = t16*t33*1.2E+1, t424 = t2*t45*1.2E+1, t425 = t7*t71*1.2E+1, t426 = t14*t36*1.2E+1, t427 = t3*t74*2.4E+1, t428 = t3*t44*2.4E+1;
    T t430 = t7*t40*2.4E+1, t435 = t7*t75*8.0, t437 = t6*t74*1.2E+1, t438 = t15*t35*1.2E+1, t440 = t3*t46*2.4E+1, t442 = t4*t45*2.4E+1, t443 = t16*t35*1.2E+1, t444 = t6*t46*1.2E+1, t445 = t9*t74*1.2E+1, t446 = t15*t37*1.2E+1;
    T t447 = t16*t36*1.2E+1, t448 = t7*t45*1.2E+1, t449 = t16*t37*1.2E+1, t450 = t9*t46*1.2E+1, t466 = rx*t136, t467 = rx*t137, t468 = ry*t136, t469 = ry*t138, t470 = rz*t137, t471 = rz*t138, t475 = t32*t39*4.0, t476 = t15*t84*4.0;
    T t477 = t34*t71*4.0, t478 = t16*t83*4.0, t479 = t34*t44*4.0, t480 = t36*t69*4.0, t481 = t34*t45*4.0, t482 = t36*t40*4.0, t483 = t36*t46*4.0, t485 = t223*tx*tz, t486 = t224*tx*ty, t487 = t223*ty*tz, t488 = t225*tx*ty, t489 = t224*ty*tz;
    T t490 = t225*tx*tz, t500 = t23*t54*1.2E+1, t501 = t23*t55*1.2E+1, t502 = t20*t59*1.2E+1, t511 = -t410, t516 = -t422, t520 = -t429, t522 = -t436, t524 = -t439, t525 = -t441, t530 = t23+t136, t534 = t14*t83*6.0, t535 = t14*t84*6.0;
    T t536 = t32*t40*6.0, t537 = t4*t95*6.0, t538 = t34*t40*6.0, t539 = t33*t45*6.0, t540 = t7*t96*6.0, t541 = t15*t92*6.0, t542 = t35*t46*6.0, t543 = t9*t104*6.0, t544 = t36*t45*6.0, t545 = t37*t46*6.0, t552 = t41+t134, t553 = t41+t135;
    T t554 = t42+t135, t567 = t136*t223, t568 = t137*t224, t569 = t138*t225, t174 = t85*3.0, t176 = t87*4.0, t178 = t89*4.0, t180 = t91*4.0, t188 = t93*3.0, t192 = t94*3.0, t207 = -t112, t208 = -t113, t211 = -t116, t212 = -t117, t215 = -t120;
    T t216 = -t121, t226 = -t139, t234 = -t153, t248 = -t161, t250 = -t163, t251 = -t164, t252 = -t167, t253 = -t168, t255 = t85*-2.0, t256 = -t175, t258 = -t177, t260 = -t179, t262 = -t181, t263 = -t182, t265 = t93*-2.0, t267 = t94*-2.0;
    T t271 = -t98, t272 = -t99, t273 = -t100, t274 = -t101, t275 = -t102, t276 = -t103, t311 = rz*t85*4.0, t321 = rz*t85*6.0, t333 = rx*t94*4.0, t335 = ry*t93*4.0, t341 = rx*t94*6.0, t343 = ry*t93*6.0, t355 = t85*tz*4.0, t357 = t93*ty*4.0;
    T t376 = -t228, t377 = -t230, t378 = -t232, t379 = -t235, t380 = -t236, t381 = -t237, t382 = -t239, t383 = -t240, t384 = -t241, t385 = -t243, t386 = -t245, t387 = -t247, t405 = -t288, t406 = -t289, t407 = -t290, t408 = -t291, t417 = -t308;
    T t418 = -t309, t419 = -t310, t431 = -t332, t433 = -t334, t451 = -t359, t452 = -t360, t453 = -t361, t454 = -t362, t455 = -t363, t456 = -t364, t457 = -t366, t458 = -t367, t459 = -t368, t460 = -t369, t461 = -t370, t462 = -t371, t463 = -t372;
    T t464 = -t373, t465 = -t375, t472 = t3+t48, t473 = t5+t50, t474 = t8+t52, t491 = t467*ty, t492 = t466*tz, t493 = t469*tx, t494 = t468*tz, t495 = t471*tx, t496 = t470*ty, t497 = -t388, t498 = -t389, t499 = -t390, t503 = -t391, t504 = -t392;
    T t505 = -t393, t506 = -t398, t507 = -t400, t508 = -t401, t509 = -t404, t510 = -t409, t512 = -t411, t513 = -t414, t514 = -t416, t515 = -t421, t517 = -t423, t518 = -t426, t519 = -t428, t521 = -t435, t523 = -t438, t526 = -t443, t527 = -t446;
    T t528 = -t447, t529 = -t449, t546 = -t475, t547 = -t479, t548 = -t483, t549 = -t500, t550 = -t501, t551 = -t502, t555 = t552*tx, t556 = t553*tx, t557 = t552*ty, t558 = t554*ty, t559 = t553*tz, t560 = t554*tz, t561 = t40+t467, t562 = t45+t466;
    T t563 = t38+t469, t564 = t46+t468, t565 = t39+t471, t566 = t44+t470, t588 = t484*t530*tx*ty*2.0, t589 = t484*t530*tx*tz*2.0, t590 = t484*t530*ty*tz*2.0, t594 = t69+t71+t79+t81+t122, t595 = t70+t74+t77+t82+t124, t596 = t72+t75+t78+t80+t126;
    T t612 = t68+t73+t95+t97+t127+t197+t200+t201+t203, t613 = t68+t76+t96+t105+t125+t194+t196+t199+t204, t614 = t73+t76+t104+t106+t123+t193+t195+t198+t202, t257 = -t176, t259 = -t178, t261 = -t180, t420 = -t311, t432 = -t333, t434 = -t335;
    T t570 = -t557, t571 = -t559, t572 = -t560, t573 = t561*tx, t574 = t562*tx, t575 = t563*tx, t576 = t561*ty, t577 = t563*ty, t578 = t565*tx, t579 = t562*tz, t580 = t564*ty, t581 = t566*ty, t582 = t564*tz, t583 = t565*tz, t584 = t566*tz;
    T t591 = -t588, t592 = -t589, t593 = -t590, t597 = t594*tdx*ty*tz*2.0, t598 = t595*tdy*tx*tz*2.0, t599 = t596*tdz*tx*ty*2.0, t603 = t56+t86+t109+t166+t208+t254+t258, t604 = t57+t88+t109+t173+t207+t251+t256;
    T t605 = t60+t87+t108+t172+t212+t260+t264, t609 = t53+t61+t83+t85+t171+t185+t265+t267, t610 = t53+t67+t84+t93+t165+t192+t255+t262, t611 = t61+t67+t92+t94+t174+t188+t250+t252, t624 = t613*tdy*tx*2.0, t625 = t612*tdz*tx*2.0;
    T t626 = t614*tdx*ty*2.0, t627 = t612*tdz*ty*2.0, t628 = t614*tdx*tz*2.0, t629 = t613*tdy*tz*2.0, t636 = t150+t152+t293+t297+t320+t324+t338+t342+t365+t374+t379+t380+t406+t408+t451+t519+t524;
    T t637 = t148+t151+t301+t305+t307+t312+t336+t340+t358+t374+t381+t382+t418+t419+t460+t513+t525, t638 = t147+t149+t299+t303+t315+t319+t326+t330+t358+t365+t383+t384+t431+t433+t465+t514+t520;
    T t642 = t155+t157+t227+t231+t282+t306+t317+t323+t337+t341+t344+t371+t378+t397+t402+t417+t461+t509+t512+t515, t644 = t159+t160+t229+t238+t292+t328+t339+t343+t347+t349+t350+t366+t376+t403+t405+t412+t456+t506+t508+t510;
    T t646 = t156+t158+t233+t249+t296+t321+t325+t329+t346+t348+t351+t361+t377+t407+t425+t448+t452+t507+t511+t517, t600 = -t597, t601 = -t598, t602 = -t599, t606 = t62+t90+t108+t187+t211+t253+t257, t607 = t65+t89+t107+t186+t216+t261+t266;
    T t608 = t66+t91+t107+t191+t215+t259+t263, t615 = t276+t487+t496+t555+t582, t616 = t272+t490+t492+t558+t578, t617 = t274+t489+t494+t556+t581, t618 = t273+t486+t493+t571+t576, t619 = t275+t485+t495+t570+t579, t620 = t271+t488+t491+t572+t575;
    T t621 = t105+t106+t567+t574+t580, t622 = t97+t104+t568+t573+t584, t623 = t95+t96+t569+t577+t583, t630 = -t624, t631 = -t625, t632 = -t626, t633 = -t627, t634 = -t628, t635 = -t629;
    T t643 = t143+t145+t227+t242+t280+t285+t287+t294+t314+t318+t327+t370+t386+t399+t424+t432+t462+t518+t521+t528, t645 = t141+t142+t238+t246+t281+t284+t286+t295+t298+t302+t331+t364+t387+t434+t437+t444+t457+t522+t527+t529;
    T t647 = t144+t146+t244+t249+t283+t300+t304+t313+t316+t322+t345+t360+t385+t420+t445+t450+t453+t516+t523+t526, t639 = t600+t633+t635, t640 = t601+t631+t634, t641 = t602+t630+t632;
    
    fdrv(0, 0) = t138*t623+t620*tx*ty+t616*tx*tz;
    fdrv(0, 1) = -t137*t620-t623*tx*ty+t616*ty*tz; fdrv(0, 2) = -t136*t616-t623*tx*tz+t620*ty*tz;
    fdrv(0, 3) = t234+t248+t476-t477+t478-t480+t534+t535-t538-t544+rx*t110+rx*t114-t36*t40*6.0-t16*t61*2.0-t34*t45*6.0-t6*t76*2.0-t37*t46*2.0-t14*t85*8.0-t14*t93*8.0-t15*t92*2.0+t54*t71*4.0+t2*t125+t2*t127+t15*t267-rx*t4*t21*4.0-rx*t7*t24*4.0-ry*t9*t24*2.0;
    fdrv(0, 4) = t536-t537+t548+ry*t114-t14*t57*2.0+t15*t56*1.0E+1-t15*t57*4.0+t16*t56*8.0-t16*t57*2.0-t36*t38*4.0-t4*t73*8.0+t32*t45*4.0+t3*t76*8.0-t4*t76*2.0+t15*t86*1.2E+1+t2*t100*2.0-t14*t88*2.0-t15*t88*4.0+t32*t71*8.0-t36*t70*2.0-t4*t106*6.0+t281*tx+rx*t2*t21*8.0-ry*t7*t24*2.0-t15*t36*ty*6.0;
    fdrv(0, 5) = t539-t540+t547+rz*t110+t33*t40*4.0-t34*t39*4.0-t14*t62*2.0+t16*t60*1.0E+1-t15*t62*2.0-t16*t62*4.0-t7*t73*2.0-t7*t76*8.0+t44*t54*8.0+t33*t69*8.0+t2*t102*2.0-t14*t90*2.0-t16*t90*4.0-t34*t72*2.0-t7*t104*6.0+t283*tx+rx*t2*t24*8.0+ry*t3*t24*8.0-rz*t4*t21*2.0+t3*t46*tz*1.2E+1-t16*t34*tz*6.0;
    fdrv(0, 6) = -t644*tdy-t646*tdz-tdx*(-t306+t308-t327+t333+t372+t373-t397-t399+t411+t447+rx*t61*4.0+rx*t67*4.0+t14*t34*2.4E+1+t14*t36*2.4E+1-t15*t54*4.0-t16*t58*4.0);
    fdrv(0, 7) = -t644*tdx-t636*tdz+tdy*(t140+t242-t344+t402+t427+t440+t463+t464+t509-ry*t57*4.0+t2*t45*4.0-t15*t34*2.4E+1-t14*t36*4.0+t15*t54*4.0E+1+t2*t69*2.4E+1+t2*t71*8.0-t4*t74*4.0);
    fdrv(0, 8) = -t646*tdx-t636*tdy+tdz*(t140+t231-t294+t424+t427+t440+t463+t464+t518-rz*t62*4.0+t2*t40*4.0-t14*t34*4.0-t16*t36*2.4E+1+t2*t69*8.0+t2*t71*2.4E+1+t16*t58*4.0E+1-t7*t75*4.0);
    fdrv(0, 9) = -tdx*(t205+t209+t354+t356+t497+t498+t4*t21*4.0+t7*t24*4.0+t17*t34*8.0+t17*t36*8.0-t23*t54*4.0)-t604*tdy*tx*2.0-t606*tdz*tx*2.0;
    fdrv(0, 10) = -tdy*(t209+t354+t497+t549-t3*t21*1.0E+1+t4*t21*8.0+t17*t34*4.0+t24*t29+t20*t36*6.0-t84*tx*2.0+t17*t29*tz)-t611*tdx*ty*2.0-t606*tdz*ty*2.0;
    fdrv(0, 11) = -tdz*(t205+t356+t498+t549-t5*t24*1.0E+1+t7*t24*8.0+t21*t27+t17*t36*4.0+t23*t34*6.0-t83*tx*2.0+t17*t27*ty)-t611*tdx*tz*2.0-t604*tdy*tz*2.0;
    fdrv(0, 12) = tdy*(t218+t268+t279+t395+ry*t25*2.0+t17*t38*4.0+t23*t38*8.0+t17*t46*4.0+t23*t46*8.0+t97*ty*6.0+t104*ty*1.2E+1)+tdz*(t219+t269+t277+t396+rz*t22*2.0+t17*t39*4.0+t20*t39*8.0+t17*t44*4.0+t20*t44*8.0+t105*tz*6.0+t106*tz*1.2E+1)+t138*t594*tdx*2.0;
    fdrv(0, 13) = t641; fdrv(0, 14) = t640; fdrv(0, 15) = t530*tx*(t34+t36-t54-t58)*-2.0; fdrv(0, 16) = t530*ty*(t34+t36-t54-t58)*-2.0; fdrv(0, 17) = t530*tz*(t34+t36-t54-t58)*-2.0;
    fdrv(0, 18) = rdx*t530*(t34+t36-t54-t58)*-2.0-t11*tx*(t34+t36-t54-t58)*4.0-t12*tx*(t34+t36-t54-t58)*4.0-t13*tx*(t34+t36-t54-t58)*4.0-t11*t222*t530*2.0-t12*t222*t530*2.0-t13*t222*t530*2.0;
    fdrv(0, 19) = rdy*t530*(t34+t36-t54-t58)*-2.0-t11*ty*(t34+t36-t54-t58)*4.0-t12*ty*(t34+t36-t54-t58)*4.0-t13*ty*(t34+t36-t54-t58)*4.0-t11*t530*(t4-t26)*2.0-t12*t530*(t4-t26)*2.0-t13*t530*(t4-t26)*2.0;
    fdrv(0, 20) = rdz*t530*(t34+t36-t54-t58)*-2.0-t11*tz*(t34+t36-t54-t58)*4.0-t12*tz*(t34+t36-t54-t58)*4.0-t13*tz*(t34+t36-t54-t58)*4.0-t11*t530*(t7-t28)*2.0-t12*t530*(t7-t28)*2.0-t13*t530*(t7-t28)*2.0; fdrv(0, 24) = t138*t484*t530*2.0;
    fdrv(0, 25) = t591; fdrv(0, 26) = t592; fdrv(1, 0) = -t138*t618-t622*tx*ty+t617*tx*tz; fdrv(1, 1) = t137*t622+t618*tx*ty+t617*ty*tz; fdrv(1, 2) = -t136*t617+t618*tx*tz-t622*ty*tz;
    fdrv(1, 3) = -t536+t537+t548+rx*t118+t14*t57*1.0E+1-t15*t56*2.0+t15*t57*8.0-t16*t56*2.0+t16*t57*8.0-t36*t38*6.0-t32*t45*6.0-t3*t76*2.0+t4*t76*8.0-t15*t86*2.0+t14*t88*1.2E+1+t15*t88*8.0-t32*t71*4.0+t6*t98*2.0-t37*t69*2.0+t4*t106*4.0+t4*t125-rx*t2*t21*4.0-rx*t9*t24*2.0-t14*t32*tx*8.0-t15*t36*ty*4.0;
    fdrv(1, 4) = t226+t248-t476+t477+t481-t482-t534+t538+t541-t545+ry*t111+ry*t118-t15*t53*4.0-t16*t53*2.0-t2*t73*8.0-t2*t76*2.0-t36*t45*2.0-t14*t84*2.0+t14*t85*4.0-t16*t83*6.0-t36*t69*6.0-t15*t94*8.0+t55*t74*4.0+t6*t127+t14*t265-rx*t7*t24*2.0-ry*t9*t24*4.0;
    fdrv(1, 5) = t542-t543+t546+rz*t111-t32*t44*4.0-t15*t66*2.0+t16*t65*1.0E+1-t16*t66*4.0-t9*t76*8.0+t39*t55*8.0-t38*t59*2.0+t4*t99*4.0-t32*t72*2.0-t7*t98*6.0+t27*t102+t329*ty+rx*t4*t24*8.0-rx*t7*t21*2.0+ry*t6*t24*8.0-rz*t2*t21*2.0-t7*t40*ty*2.0-t7*t45*ty*4.0+t4*t40*tz*8.0-t16*t32*tz*6.0+t4*t45*tz*1.2E+1;
    fdrv(1, 6) = -t642*tdy-t637*tdz+tdx*(t154+t246-t350+t403+t413+t442+t458+t459+t508-rx*t56*4.0+t4*t40*2.4E+1-t14*t32*2.4E+1+t6*t46*4.0-t15*t37*4.0+t14*t55*4.0E+1-t3*t71*4.0+t6*t74*8.0);
    fdrv(1, 7) = -t642*tdx-t647*tdz-tdy*(t288-t292-t331+t335+t367+t368+t398-t412-t437+t449+ry*t53*4.0+ry*t67*4.0+t15*t32*2.4E+1+t15*t37*2.4E+1-t14*t55*4.0-t16*t63*4.0);
    fdrv(1, 8) = -t637*tdx-t647*tdy+tdz*(t154+t229-t281+t413+t442+t444+t458+t459+t527-rz*t66*4.0+t4*t40*8.0-t15*t32*4.0-t7*t43*4.0-t16*t37*2.4E+1+t4*t69*4.0+t16*t63*4.0E+1+t6*t74*2.4E+1);
    fdrv(1, 9) = -tdx*(t213+t352+t499+t550-t4*t18*1.0E+1+t2*t21*4.0+t17*t32*8.0+t24*t31-t92*ty*2.0+t93*ty*6.0+t20*t31*tz)-t610*tdy*tx*2.0-t608*tdz*tx*2.0;
    fdrv(1, 10) = -tdy*(t206+t213+t352+t357+t499+t503+t2*t21*8.0+t9*t24*4.0+t17*t32*4.0+t20*t37*8.0-t23*t55*4.0)-t603*tdx*ty*2.0-t608*tdz*ty*2.0;
    fdrv(1, 11) = -tdz*(t206+t357+t503+t550+t2*t21*2.0-t8*t24*1.0E+1+t9*t24*8.0+t17*t32*2.0+t23*t32*6.0+t20*t37*4.0-t20*t55*2.0)-t603*tdx*tz*2.0-t610*tdy*tz*2.0; fdrv(1, 12) = t641;
    fdrv(1, 13) = tdx*(t217+t270+t278+t394+rx*t25*2.0+t20*t40*4.0+t23*t40*8.0+t20*t45*4.0+t23*t45*8.0+t95*tx*6.0+t96*tx*1.2E+1)+tdz*(t219+t269+t277+t396+rz*t19*2.0+t17*t39*8.0+t20*t39*4.0+t17*t44*8.0+t20*t44*4.0+t105*tz*1.2E+1+t106*tz*6.0)+t137*t595*tdy*2.0;
    fdrv(1, 14) = t639; fdrv(1, 15) = t530*tx*(t32+t37-t63+t47*tx)*-2.0; fdrv(1, 16) = t530*ty*(t32+t37-t63+t47*tx)*-2.0; fdrv(1, 17) = t530*tz*(t32+t37-t63+t47*tx)*-2.0;
    fdrv(1, 18) = rdx*t530*(t32+t37-t63+t47*tx)*-2.0-t11*tx*(t32+t37-t63+t47*tx)*4.0-t12*tx*(t32+t37-t63+t47*tx)*4.0-t13*tx*(t32+t37-t63+t47*tx)*4.0-t11*t472*t530*2.0-t12*t472*t530*2.0-t13*t472*t530*2.0;
    fdrv(1, 19) = rdy*t530*(t32+t37-t63+t47*tx)*-2.0-t11*ty*(t32+t37-t63+t47*tx)*4.0-t12*ty*(t32+t37-t63+t47*tx)*4.0-t13*ty*(t32+t37-t63+t47*tx)*4.0-t11*t221*t530*2.0-t12*t221*t530*2.0-t13*t221*t530*2.0;
    fdrv(1, 20) = rdz*t530*(t32+t37-t63+t47*tx)*-2.0-t11*tz*(t32+t37-t63+t47*tx)*4.0-t12*tz*(t32+t37-t63+t47*tx)*4.0-t13*tz*(t32+t37-t63+t47*tx)*4.0-t11*t530*(t9-t30)*2.0-t12*t530*(t9-t30)*2.0-t13*t530*(t9-t30)*2.0; fdrv(1, 24) = t591;
    fdrv(1, 25) = t137*t484*t530*2.0; fdrv(1, 26) = t593; fdrv(2, 0) = -t138*t619+t615*tx*ty-t621*tx*tz; fdrv(2, 1) = -t137*t615+t619*tx*ty-t621*ty*tz; fdrv(2, 2) = t136*t621+t619*tx*tz+t615*ty*tz;
    fdrv(2, 3) = -t539+t540+t547+rx*t119-t33*t40*6.0-t34*t39*6.0+t14*t62*1.0E+1-t16*t60*2.0+t15*t62*8.0+t16*t62*8.0+t7*t73*8.0-t44*t54*2.0-t33*t69*4.0+t14*t90*1.2E+1-t35*t69*2.0+t16*t90*8.0+t7*t104*4.0+t31*t98+t7*t127-rx*t2*t24*4.0-rx*t6*t24*2.0-ry*t3*t24*2.0-t14*t33*tx*8.0-t3*t46*tz*2.0-t16*t34*tz*4.0;
    fdrv(2, 4) = -t542+t543+t546+ry*t115-t32*t44*6.0+t15*t66*1.0E+1-t16*t65*2.0+t16*t66*8.0-t39*t55*2.0+t38*t59*8.0-t4*t99*6.0-t33*t70*2.0+t7*t98*4.0+t29*t100+t9*t127-rx*t4*t24*2.0+rx*t7*t21*8.0-ry*t2*t24*2.0-ry*t6*t24*4.0+t7*t40*ty*1.2E+1-t15*t35*ty*8.0+t7*t45*ty*8.0-t4*t40*tz*4.0-t16*t32*tz*4.0-t4*t45*tz*2.0;
    fdrv(2, 5) = t226+t234-t478+t480-t481+t482-t535-t541+t544+t545+rz*t115+rz*t119-t15*t53*2.0-t16*t53*4.0-t34*t40*2.0-t2*t73*2.0-t16*t61*4.0-t2*t76*8.0-t6*t76*8.0-t14*t83*2.0-t15*t84*6.0-t34*t71*6.0+t14*t93*4.0+t15*t94*4.0+t59*t75*4.0+t14*t255-rx*t4*t21*2.0;
    fdrv(2, 6) = -t638*tdy-t643*tdz+tdx*(t162+t244-t329+t415+t425+t430+t454+t455+t517-rx*t60*4.0-t3*t41*4.0-t14*t33*2.4E+1-t16*t35*4.0+t7*t45*2.4E+1+t9*t46*8.0+t14*t59*4.0E+1+t9*t74*4.0);
    fdrv(2, 7) = -t638*tdx-t645*tdz+tdy*(t162+t233-t283+t415+t430+t445+t454+t455+t526-ry*t65*4.0-t4*t42*4.0-t16*t33*4.0-t15*t35*2.4E+1+t7*t45*8.0+t9*t46*2.4E+1+t7*t71*4.0+t15*t64*4.0E+1);
    fdrv(2, 8) = -t643*tdx-t645*tdy-tdz*(t290-t296+t311-t313+t362+t363+t400+t438-t448-t450+rz*t53*4.0+rz*t61*4.0+t16*t33*2.4E+1+t16*t35*2.4E+1-t14*t59*4.0-t15*t64*4.0);
    fdrv(2, 9) = -tdx*(t214+t353+t504+t551-t7*t18*1.0E+1+t2*t24*4.0+t6*t24*2.0+t17*t33*8.0+t20*t35*2.0-t23*t64*2.0+t85*tz*6.0)-t607*tdy*tx*2.0-t609*tdz*tx*2.0;
    fdrv(2, 10) = -tdy*(t210+t355+t505+t551+t2*t24*2.0+t6*t24*4.0-t9*t21*1.0E+1+t17*t33*2.0+t20*t33*6.0+t20*t35*8.0-t23*t59*2.0)-t605*tdx*ty*2.0-t609*tdz*ty*2.0;
    fdrv(2, 11) = -tdz*(t210+t214+t353+t355+t504+t505+t2*t24*8.0+t6*t24*8.0+t17*t33*4.0+t20*t35*4.0-t20*t59*4.0)-t605*tdx*tz*2.0-t607*tdy*tz*2.0; fdrv(2, 12) = t640; fdrv(2, 13) = t639;
    fdrv(2, 14) = tdx*(t217+t270+t278+t394+rx*t22*2.0+t20*t40*8.0+t23*t40*4.0+t20*t45*8.0+t23*t45*4.0+t95*tx*1.2E+1+t96*tx*6.0)+tdy*(t218+t268+t279+t395+ry*t19*2.0+t17*t38*8.0+t23*t38*4.0+t17*t46*8.0+t23*t46*4.0+t97*ty*1.2E+1+t104*ty*6.0)+t136*t596*tdz*2.0;
    fdrv(2, 15) = t530*tx*(t33+t35+t49*tx+t51*ty)*-2.0; fdrv(2, 16) = t530*ty*(t33+t35+t49*tx+t51*ty)*-2.0; fdrv(2, 17) = t530*tz*(t33+t35+t49*tx+t51*ty)*-2.0;
    fdrv(2, 18) = rdx*t530*(t33+t35+t49*tx+t51*ty)*-2.0-t11*tx*(t33+t35+t49*tx+t51*ty)*4.0-t12*tx*(t33+t35+t49*tx+t51*ty)*4.0-t13*tx*(t33+t35+t49*tx+t51*ty)*4.0-t11*t473*t530*2.0-t12*t473*t530*2.0-t13*t473*t530*2.0;
    fdrv(2, 19) = rdy*t530*(t33+t35+t49*tx+t51*ty)*-2.0-t11*ty*(t33+t35+t49*tx+t51*ty)*4.0-t12*ty*(t33+t35+t49*tx+t51*ty)*4.0-t13*ty*(t33+t35+t49*tx+t51*ty)*4.0-t11*t474*t530*2.0-t12*t474*t530*2.0-t13*t474*t530*2.0;
    fdrv(2, 20) = rdz*t530*(t33+t35+t49*tx+t51*ty)*-2.0-t11*tz*(t33+t35+t49*tx+t51*ty)*4.0-t12*tz*(t33+t35+t49*tx+t51*ty)*4.0-t13*tz*(t33+t35+t49*tx+t51*ty)*4.0-t11*t220*t530*2.0-t12*t220*t530*2.0-t13*t220*t530*2.0; fdrv(2, 24) = t592;
    fdrv(2, 25) = t593; fdrv(2, 26) = t136*t484*t530*2.0;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f13(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*rdy, t3 = ox*rdz, t4 = oy*rdx, t5 = oy*rdz, t6 = oz*rdx, t7 = oz*rdy, t8 = ox*ry, t9 = oy*rx, t10 = ox*rz, t11 = oz*rx, t12 = oy*rz, t13 = oz*ry, t14 = ox*tdx, t15 = oy*tdy, t16 = oz*tdz, t17 = ox*tx, t18 = oy*ty, t19 = oz*tz;
    T t20 = rdx*tx, t21 = rx*tdx, t22 = rdy*ty, t23 = ry*tdy, t24 = rdz*tz, t25 = rz*tdz, t26 = rx*tx, t27 = ry*ty, t28 = rz*tz, t29 = -t20, t30 = -t21, t31 = -t22, t32 = -t23, t33 = -t24, t34 = -t25, t35 = -t26, t36 = -t27, t37 = -t28;
    
    fdrv(0, 0) = t36+t37; fdrv(0, 1) = rx*ty; fdrv(0, 2) = rx*tz; fdrv(0, 4) = -t8+t9; fdrv(0, 5) = -t10+t11; fdrv(0, 9) = t15+t16; fdrv(0, 10) = -ox*tdy; fdrv(0, 11) = -ox*tdz; fdrv(0, 12) = t32+t34; fdrv(0, 13) = rx*tdy; fdrv(0, 14) = rx*tdz; fdrv(0, 15) = t18+t19;
    fdrv(0, 16) = -ox*ty; fdrv(0, 17) = -ox*tz; fdrv(0, 19) = -t2+t4; fdrv(0, 20) = -t3+t6; fdrv(0, 24) = t31+t33; fdrv(0, 25) = rdx*ty; fdrv(0, 26) = rdx*tz; fdrv(1, 0) = ry*tx; fdrv(1, 1) = t35+t37; fdrv(1, 2) = ry*tz; fdrv(1, 3) = t8-t9; fdrv(1, 5) = -t12+t13;
    fdrv(1, 9) = -oy*tdx; fdrv(1, 10) = t14+t16; fdrv(1, 11) = -oy*tdz; fdrv(1, 12) = ry*tdx; fdrv(1, 13) = t30+t34; fdrv(1, 14) = ry*tdz; fdrv(1, 15) = -oy*tx; fdrv(1, 16) = t17+t19; fdrv(1, 17) = -oy*tz; fdrv(1, 18) = t2-t4; fdrv(1, 20) = -t5+t7;
    fdrv(1, 24) = rdy*tx; fdrv(1, 25) = t29+t33; fdrv(1, 26) = rdy*tz; fdrv(2, 0) = rz*tx; fdrv(2, 1) = rz*ty; fdrv(2, 2) = t35+t36; fdrv(2, 3) = t10-t11; fdrv(2, 4) = t12-t13; fdrv(2, 9) = -oz*tdx; fdrv(2, 10) = -oz*tdy; fdrv(2, 11) = t14+t15; fdrv(2, 12) = rz*tdx;
    fdrv(2, 13) = rz*tdy; fdrv(2, 14) = t30+t32; fdrv(2, 15) = -oz*tx; fdrv(2, 16) = -oz*ty; fdrv(2, 17) = t17+t18; fdrv(2, 18) = t3-t6; fdrv(2, 19) = t5-t7; fdrv(2, 24) = rdz*tx; fdrv(2, 25) = rdz*ty; fdrv(2, 26) = t29+t31;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f14(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*rx, t3 = oy*ry, t4 = oz*rz, t5 = ox*tx, t6 = ox*ty, t7 = oy*tx, t8 = ox*tz, t9 = oy*ty, t10 = oz*tx, t11 = oy*tz, t12 = oz*ty, t13 = oz*tz, t14 = rx*tx, t15 = rx*ty, t16 = ry*tx, t17 = rx*tz, t18 = ry*ty, t19 = rz*tx;
    T t20 = ry*tz, t21 = rz*ty, t22 = rz*tz, t23 = tx*ty, t24 = tx*tz, t25 = ty*tz, t26 = ox*2.0, t27 = oy*2.0, t28 = oz*2.0, t29 = rx*2.0, t30 = ry*2.0, t31 = rz*2.0, t32 = tx*2.0, t33 = tx*4.0, t34 = ty*2.0, t35 = ty*4.0, t36 = tz*2.0;
    T t37 = tz*4.0, t38 = tx*tx, t39 = tx*tx*tx, t40 = ty*ty, t41 = ty*ty*ty, t42 = tz*tz, t43 = tz*tz*tz, t119 = ox*ry*-2.0, t121 = oz*rx*-2.0, t122 = oy*rz*-2.0, t45 = rx*t27, t46 = rz*t26, t49 = ry*t28, t50 = t5*2.0, t51 = t6*2.0, t52 = t7*2.0;
    T t53 = t8*2.0, t54 = t9*2.0, t55 = t10*2.0, t56 = t11*2.0, t57 = t12*2.0, t58 = t13*2.0, t59 = t14*2.0, t60 = t14*4.0, t61 = t15*2.0, t62 = t16*2.0, t63 = t17*2.0, t64 = t18*2.0, t65 = t19*2.0, t66 = t18*4.0, t67 = t20*2.0, t68 = t21*2.0;
    T t69 = t22*2.0, t70 = t22*4.0, t71 = t23*2.0, t72 = t24*2.0, t73 = t25*2.0, t74 = t2*tx, t75 = ry*t5, t76 = rz*t5, t77 = rx*t9, t78 = ry*t8, t79 = rz*t6, t80 = rx*t11, t81 = t3*ty, t82 = rz*t7, t83 = rx*t12, t84 = ry*t10, t85 = rz*t9;
    T t86 = rx*t13, t87 = ry*t13, t88 = t4*tz, t89 = t5*ty, t90 = t5*tz, t91 = t7*ty, t92 = t6*tz, t93 = t7*tz, t94 = t10*ty, t95 = t9*tz, t96 = t10*tz, t97 = t12*tz, t98 = t14*ty, t99 = t14*tz, t100 = t16*ty, t101 = t15*tz, t102 = t16*tz;
    T t103 = t19*ty, t104 = t18*tz, t105 = t19*tz, t106 = t21*tz, t107 = rx+t21, t108 = ry+t17, t109 = rz+t16, t110 = t25+tx, t111 = t24+ty, t112 = t23+tz, t113 = t32*tx, t114 = t38*3.0, t115 = t34*ty, t116 = t40*3.0, t117 = t36*tz;
    T t118 = t42*3.0, t120 = -t3, t123 = -t4, t124 = -t5, t125 = -t7, t127 = -t8, t128 = -t9, t130 = -t12, t132 = -t13, t133 = -t15, t135 = -t19, t137 = -t20, t145 = t5*tx, t146 = t6*ty, t147 = t7*tx, t148 = t8*tz, t149 = t9*ty, t150 = t10*tx;
    T t151 = t11*tz, t152 = t12*ty, t153 = t13*tz, t154 = t14*tx, t155 = t18*ty, t156 = t22*tz, t157 = t2*t32, t159 = t7*t29, t161 = t6*t30, t163 = t10*t29, t166 = t11*t29, t167 = t3*t34, t168 = t7*t31, t171 = t8*t31, t173 = t12*t30;
    T t175 = t11*t31, t177 = t4*t36, t178 = t5*t34, t179 = t5*t36, t180 = t7*t34, t181 = t6*t36, t182 = t7*t36, t183 = t10*t34, t184 = t9*t36, t185 = t10*t36, t187 = t14*t34, t188 = t14*t36, t189 = t16*t34, t190 = t18*t36, t191 = t19*t36;
    T t192 = t21*t36, t202 = t2*ty*-2.0, t203 = t2*tz*-2.0, t204 = t3*tx*-2.0, t211 = t3*tz*-2.0, t212 = t4*tx*-2.0, t213 = t4*ty*-2.0, t220 = t38+t40, t221 = t38+t42, t222 = t40+t42, t229 = t25+t32, t230 = t24+t34, t231 = t23+t36, t236 = t5+t9;
    T t237 = t5+t13, t238 = t9+t13, t239 = t14+t18, t242 = t14+t22, t244 = t18+t22, t268 = -tx*(t25-tx), t269 = -tx*(t23-tz), t270 = -ty*(t25-tx), t271 = -ty*(t24-ty), t272 = -tz*(t24-ty), t273 = -tz*(t23-tz), t284 = tx*(t23-tz);
    T t285 = ty*(t25-tx), t286 = tz*(t24-ty), t126 = -t52, t129 = -t55, t131 = -t57, t134 = -t61, t136 = -t65, t138 = -t67, t140 = -t71, t142 = -t72, t144 = -t73, t193 = t2*t71, t194 = t2*t72, t195 = t3*t71, t196 = t3*t73, t197 = t4*t72;
    T t198 = t4*t73, t199 = t154*3.0, t200 = t155*3.0, t201 = t156*3.0, t205 = -t79, t206 = -t83, t207 = t78*-2.0, t208 = t79*-2.0, t209 = t83*-2.0, t210 = t84*-2.0, t214 = -t89, t215 = -t97, t216 = t97*-2.0, t217 = -t99, t218 = -t100;
    T t219 = -t106, t223 = rx+t137, t224 = ry+t135, t225 = rz+t133, t232 = -t146, t233 = t127*tz, t234 = -t150, t235 = t130*ty, t240 = t109*tx, t241 = t107*ty, t243 = t108*tz, t245 = t110*tx, t246 = t111*tx, t247 = t111*ty, t248 = t112*ty;
    T t249 = t110*tz, t250 = t112*tz, t254 = t239*tx, t255 = t239*ty, t256 = t242*tx, t257 = t242*tz, t258 = t244*ty, t259 = t244*tz, t260 = t59+t64, t264 = t59+t69, t267 = t64+t69, t277 = t69+t239, t278 = t64+t242, t279 = t59+t244;
    T t299 = t10+t89+t149, t300 = t11+t96+t145, t301 = t6+t95+t153, t302 = t41+t284, t303 = t39+t286, t304 = t43+t285, t308 = t91+t130+t145, t309 = t90+t125+t153, t310 = t97+t127+t149, t261 = t224*tx, t262 = t225*ty, t265 = t223*tz;
    T t274 = t39+t248, t275 = t43+t246, t276 = t41+t249, t280 = -t255, t281 = -t256, t282 = -t259, t287 = t14+t267, t288 = t18+t264, t289 = t22+t260, t290 = t277*tx, t291 = t278*tx, t292 = t277*ty, t293 = t279*ty, t294 = t278*tz, t295 = t279*tz;
    T t296 = t115+t245, t297 = t113+t250, t298 = t117+t247, t320 = t15+t16+t104+t257, t321 = t17+t19+t98+t258, t322 = t20+t21+t105+t254, t326 = t2+t78+t80+t120+t209+t210, t327 = t3+t82+t84+t123+t207+t208, t328 = t2+t123+t166+t168+t205+t206;
    T t329 = t51+t90+t95+t126+t234+t235, t330 = t53+t129+t147+t151+t214+t215, t331 = t56+t91+t96+t131+t232+t233, t283 = -t265, t311 = t68+t290, t312 = t63+t293, t313 = t62+t294, t323 = t15+t16+t217+t282, t324 = t17+t19+t219+t280;
    T t325 = t20+t21+t218+t281, t315 = t240+t283;
    
    fdrv(0, 0) = -t324*ty+t320*tz; fdrv(0, 1) = -t322*ty-t264*tz; fdrv(0, 2) = t260*ty+t325*tz; fdrv(0, 3) = -oy*t312+ox*(t243-t262)+oz*(t61-t295);
    fdrv(0, 4) = -t76-t85*2.0+t88+t163+t193+ry*t12*4.0+t6*t18*3.0-t3*t23*2.0+t8*t20+t22*t51+t14*t125+t10*t137+t22*t125+t120*tz;
    fdrv(0, 5) = t75-t81+t194-rx*t7*2.0-rz*t11*4.0-t10*t14+t6*t21-t4*t24*2.0-t10*t18+t8*t22*3.0+t13*t30+t20*t51+t21*t125+t4*ty; fdrv(0, 6) = -tdz*(t45+t85+t163+t177+t203-ox*ry+ry*t12)-tdy*(t87+t121+t159+t167+t202+ox*rz+rz*t11)-rx*t238*tdx*2.0;
    fdrv(0, 7) = -tdx*(ox*(rz+t134)+oy*t289+oz*(t20-t29))-t327*tdz+tdy*(t122+t157+t171+t204+oz*ry*4.0+ry*t6*6.0); fdrv(0, 8) = -t327*tdy+tdz*(t49+t157+t161+t212-oy*rz*4.0+rz*t8*6.0)-tdx*(-ox*(ry+t63)+oy*(t21+t29)+oz*t288);
    fdrv(0, 9) = tdy*(t55+t178+t125*tx)+tdx*(-oy*(t36+t71)+oz*(t34+t142)+ox*t222)-tdz*(t52-t90*2.0+t150); fdrv(0, 10) = -t310*tdx-tdy*(t11-t12*4.0+t96-t146*3.0+t180+t233)+tdz*(t5+t58-t94+t128+t181);
    fdrv(0, 11) = -t301*tdx-tdz*(t11*4.0+t91+t130-t148*3.0+t185+t232)-tdy*(t5+t54-t92*2.0+t93+t132); fdrv(0, 12) = tdz*(t16+t188+t190+t201+t21*ty)+tdy*(t135+t187+t192+t200+t20*tz)+tdx*(t243-t262);
    fdrv(0, 13) = -tdz*(t18+t59+t70+t103)-t312*tdx-tdy*(t20+t68+t105+t154+t189); fdrv(0, 14) = tdy*(t22+t59+t66-t102)+tdx*(t61-t295)-tdz*(-t21+t100+t138+t154+t191); fdrv(0, 15) = -t331*tx; fdrv(0, 16) = ox*t276+t112*t128+oz*(t40*2.0+t273);
    fdrv(0, 17) = ox*t304-oy*t298+t132*(t24-ty); fdrv(0, 18) = -rdx*t331-rdy*t310-rdz*t301-rdx*t238*tx; fdrv(0, 19) = rdz*(t13-oy*t230+ox*(t73-tx))-rdy*(t91-ox*(t42+t116)+oy*t112+oz*(t24-t35))+rdx*tx*(t28+t51+t125);
    fdrv(0, 20) = -rdy*(t9-ox*(t73+tx)+oz*(t23-t36))-rdz*(t96-ox*(t40+t118)+oy*(t23+t37)+oz*(t24-ty))-rdx*tx*(t10+t27-t53); fdrv(0, 24) = rdy*t276+rdz*t304+rdx*t222*tx; fdrv(0, 25) = -rdy*t248-rdz*t298-rdx*t231*tx;
    fdrv(0, 26) = rdz*t272+rdy*(t40*2.0+t273)-rdx*tx*(t24-t34); fdrv(1, 0) = t324*tx+t267*tz; fdrv(1, 1) = t322*tx-t323*tz; fdrv(1, 2) = -t260*tx-t321*tz;
    fdrv(1, 3) = t85-t88+t195-rx*t10*4.0-ry*t12*2.0+t7*t14*3.0-t6*t18-t2*t23*2.0-t6*t22+t11*t17+t5*t31+t22*t52+t17*t130+t2*tz; fdrv(1, 4) = oy*t315-oz*t313+ox*(t67-t291);
    fdrv(1, 5) = t74-t77-t86*2.0+t161+t196+rz*t8*4.0+t7*t19-t4*t25*2.0+t11*t22*3.0+t17*t52+t10*t133+t21*t124+t18*t130+t123*tx; fdrv(1, 6) = t328*tdz+tdx*(t46+t167+t175+t202-oz*rx*4.0+rx*t7*6.0)-tdy*(-oy*(rz+t62)+oz*(t17+t30)+ox*t289);
    fdrv(1, 7) = -tdz*(t76+t119+t173+t177+t211+oy*rx+rx*t10)-tdx*(t49+t86+t157+t161+t204-oy*rz+rz*t8)-ry*t237*tdy*2.0; fdrv(1, 8) = -tdy*(oy*(rx+t138)+oz*t287+ox*(t19-t30))+t328*tdx+tdz*(t121+t159+t167+t213+ox*rz*4.0+rz*t11*6.0);
    fdrv(1, 9) = tdx*(t8-t10*4.0-t89*2.0+t147*3.0+t151+t215)-t300*tdy-tdz*(t9+t58-t93*2.0+t94+t124); fdrv(1, 10) = tdz*(t51+t184+t235)+tdy*(ox*(t36+t140)-oz*(t32+t73)+oy*t221)-tdx*(t57-t91*2.0+t146);
    fdrv(1, 11) = tdz*(t8*4.0-t10+t147+t151*3.0+t214+t216)-t309*tdy+tdx*(t9+t50-t92+t132+t182); fdrv(1, 12) = tdz*(t14+t64+t70-t103)+tdy*(t67-t291)-tdx*(-t17+t106+t136+t155+t187);
    fdrv(1, 13) = t315*tdy+tdz*(t133+t188+t190+t201+t19*tx)+tdx*(t21+t189+t191+t199+t17*tz); fdrv(1, 14) = -tdx*(t22+t60+t64+t101)-t313*tdy-tdz*(t19+t63+t98+t155+t192); fdrv(1, 15) = oy*t303-oz*t297+t124*(t23-tz); fdrv(1, 16) = t330*ty;
    fdrv(1, 17) = oy*t275+t110*t132+ox*(t42*2.0+t268); fdrv(1, 18) = -rdz*(t13-oy*(t72+ty)+ox*(t25-t32))-rdx*(t89-oy*(t42+t114)+oz*(t25+t33)+ox*(t23-tz))-rdy*ty*(t6+t28+t126); fdrv(1, 19) = -rdx*t300+rdy*t330-rdz*t309-rdy*t237*ty;
    fdrv(1, 20) = rdx*(t5-oz*t231+oy*(t72-ty))-rdz*(t97-oy*(t38+t118)+oz*t110+ox*(t23-t37))+rdy*ty*(t26+t56+t130); fdrv(1, 24) = rdx*t269+rdz*(t42*2.0+t268)-rdy*ty*(t23-t36); fdrv(1, 25) = rdx*t303+rdz*t275+rdy*t221*ty;
    fdrv(1, 26) = -rdx*t297-rdz*t249-rdy*t229*ty; fdrv(2, 0) = -t320*tx-t267*ty; fdrv(2, 1) = t264*tx+t323*ty; fdrv(2, 2) = -t325*tx+t321*ty;
    fdrv(2, 3) = t75*-2.0+t81-t87+t175+t197+rx*t7*4.0+t10*t14*3.0-t2*t24*2.0+t12*t15+t18*t55+t6*t137+t17*t128+t22*t127-t2*ty;
    fdrv(2, 4) = -t74+t86+t198-ry*t6*4.0-rz*t8*2.0+t10*t16-t3*t25*2.0+t12*t18*3.0-t11*t22+t9*t29+t15*t55+t17*t125+t20*t124+t3*tx; fdrv(2, 5) = -ox*t311+oy*(t65-t292)+oz*(t241-t261);
    fdrv(2, 6) = -tdz*(oz*(ry+t136)+ox*t288+oy*(t15-t31))-t326*tdy+tdx*(t119+t173+t177+t203+oy*rx*4.0+rx*t10*6.0); fdrv(2, 7) = -t326*tdx+tdy*(t45+t163+t177+t211-ox*ry*4.0+ry*t12*6.0)-tdz*(-oz*(rx+t68)+ox*(t16+t31)+oy*t287);
    fdrv(2, 8) = -tdy*(t46+t75+t167+t175+t213-oz*rx+rx*t7)-tdx*(t77+t122+t157+t171+t212+oz*ry+ry*t6)-rz*t236*tdz*2.0; fdrv(2, 9) = -t308*tdz-tdx*(t6-t7*4.0+t95-t150*3.0+t179+t235)+tdy*(t13+t54-t93+t124+t183);
    fdrv(2, 10) = -t299*tdz-tdy*(t6*4.0+t90+t125-t152*3.0+t184+t234)-tdx*(t13+t50+t92-t94*2.0+t128); fdrv(2, 11) = tdx*(t56+t185+t233)-tdy*(t53+t151+t216)+tdz*(-ox*(t34+t72)+oy*(t32+t144)+oz*t220);
    fdrv(2, 12) = -tdy*(t14+t66+t69+t102)-t311*tdz-tdx*(t15+t62+t104+t156+t188); fdrv(2, 13) = tdx*(t18+t60+t69-t101)+tdz*(t65-t292)-tdy*(-t16+t99+t134+t156+t190);
    fdrv(2, 14) = tdy*(t17+t187+t192+t200+t16*tx)+tdx*(t137+t189+t191+t199+t15*ty)+tdz*(t241-t261); fdrv(2, 15) = oz*t274+t111*t124+oy*(t38*2.0+t271); fdrv(2, 16) = -ox*t296+oz*t302+t128*(t25-tx); fdrv(2, 17) = -t329*tz;
    fdrv(2, 18) = rdy*(t9-ox*t229+oz*(t71-tz))-rdx*(t90-oz*(t40+t114)+ox*t111+oy*(t25-t33))+rdz*tz*(t27+t55+t127); fdrv(2, 19) = -rdx*(t5-oz*(t71+tz)+oy*(t24-t34))-rdy*(t95+ox*(t24+t35)-oz*(t38+t116)+oy*(t25-tx))-rdz*tz*(t11+t26+t131);
    fdrv(2, 20) = -rdx*t308-rdy*t299-rdz*t329-rdz*t236*tz; fdrv(2, 24) = -rdx*t246-rdy*t296-rdz*t230*tz; fdrv(2, 25) = rdy*t270+rdx*(t38*2.0+t271)-rdz*tz*(t25-t32); fdrv(2, 26) = rdx*t274+rdy*t302+rdz*t220*tz;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f15(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = oy*ty, t4 = oz*tz, t5 = rx*ty, t6 = ry*tx, t7 = rx*tz, t8 = rz*tx, t9 = ry*tz, t10 = rz*ty, t11 = tx*tx, t12 = ty*ty, t13 = tz*tz, t14 = rx*tx*2.0, t15 = rx*tx*3.0, t19 = ry*ty*2.0, t21 = ry*ty*3.0, t24 = rz*tz*2.0;
    T t25 = rz*tz*3.0, t27 = oy*rx*tx, t28 = ox*ry*ty, t31 = oz*rx*tx, t32 = ox*rz*tz, t35 = oz*ry*ty, t36 = oy*rz*tz, t110 = ox*ty*tz*2.0, t111 = oy*tx*tz*2.0, t112 = oz*tx*ty*2.0, t158 = rdx*tx*ty*tz*2.0, t159 = rdy*tx*ty*tz*2.0;
    T t160 = rdz*tx*ty*tz*2.0, t16 = t5*2.0, t17 = t6*2.0, t18 = t7*2.0, t20 = t8*2.0, t22 = t9*2.0, t23 = t10*2.0, t26 = ry*t2, t29 = rz*t2, t30 = rx*t3, t33 = rz*t3, t34 = rx*t4, t37 = ry*t4, t38 = t5*tx, t39 = t7*tx, t40 = t6*ty, t41 = t9*ty;
    T t42 = t8*tz, t43 = t10*tz, t44 = t11*2.0, t45 = t11*3.0, t46 = t12*2.0, t47 = t12*3.0, t48 = t13*2.0, t49 = t13*3.0, t50 = -t6, t52 = -t8, t54 = -t10, t56 = t2*tx, t57 = ox*t12, t58 = oy*t11, t59 = ox*t13, t60 = t3*ty, t61 = oz*t11;
    T t62 = oy*t13, t63 = oz*t12, t64 = t4*tz, t65 = rx*t11, t66 = t5*ty, t67 = t6*tx, t68 = t7*tz, t69 = ry*t12, t70 = t8*tx, t71 = t9*tz, t72 = t10*ty, t73 = rz*t13, t74 = rx*t2*2.0, t75 = rx*t2*4.0, t77 = oy*t14, t78 = ox*t5*4.0, t79 = ox*t19;
    T t82 = oz*t14, t83 = ox*t7*4.0, t84 = oy*t6*4.0, t88 = ry*t3*2.0, t92 = ry*t3*4.0, t93 = ox*t24, t96 = oz*t19, t97 = oy*t9*4.0, t98 = oz*t8*4.0, t99 = oy*t24, t101 = oz*t10*4.0, t102 = rz*t4*2.0, t103 = rz*t4*4.0, t104 = t2*ty*2.0;
    T t105 = t2*ty*4.0, t106 = t2*tz*2.0, t107 = t3*tx*2.0, t108 = t2*tz*4.0, t109 = t3*tx*4.0, t113 = t3*tz*2.0, t114 = t4*tx*2.0, t115 = t3*tz*4.0, t116 = t4*tx*4.0, t117 = t4*ty*2.0, t118 = t4*ty*4.0, t137 = t2*t5*4.0, t138 = t2*t7*4.0;
    T t143 = t3*t6*4.0, t152 = t3*t9*4.0, t153 = t4*t8*4.0, t154 = t4*t10*4.0, t161 = -t11, t162 = -t12, t163 = -t13, t182 = ox*t9*-2.0, t183 = ox*t10*-2.0, t184 = oy*t7*-2.0, t185 = oy*t8*-2.0, t186 = oz*t5*-2.0, t187 = oz*t6*-2.0, t191 = -t111;
    T t192 = -t112, t199 = t11+t12, t200 = t11+t13, t201 = t12+t13, t233 = t15+t21, t235 = t15+t25, t237 = t21+t25, t238 = t2+t3+t4, t257 = t14+t19+t24, t51 = -t17, t53 = -t20, t55 = -t23, t76 = t26*2.0, t80 = t29*2.0, t81 = t30*2.0, t85 = ox*t22;
    T t86 = ox*t23, t87 = oy*t18, t89 = oy*t20, t90 = oz*t16, t91 = oz*t17, t94 = t33*2.0, t95 = t34*2.0, t100 = t37*2.0, t119 = t16*tx, t120 = t38*4.0, t121 = t18*tx, t122 = t17*ty, t123 = t39*4.0, t124 = t40*4.0, t125 = t22*ty, t126 = t20*tz;
    T t127 = t41*4.0, t128 = t42*4.0, t129 = t23*tz, t130 = t43*4.0, t131 = ox*t71, t133 = oy*t68, t135 = oz*t66, t139 = t2*t22, t140 = t2*t23, t145 = t3*t18, t146 = t3*t20, t150 = t4*t16, t151 = t4*t17, t155 = t104*tz, t156 = t107*tz;
    T t157 = t114*ty, t164 = t56*2.0, t165 = t56*3.0, t166 = t60*2.0, t167 = t60*3.0, t168 = t64*2.0, t169 = t64*3.0, t170 = rx*t45, t171 = ry*t47, t172 = rz*t49, t173 = t46*tx, t174 = t44*ty, t175 = t48*tx, t176 = t44*tz, t177 = t48*ty;
    T t178 = t46*tz, t179 = -t78, t180 = -t83, t181 = -t84, t188 = -t97, t189 = -t98, t190 = -t101, t193 = t38*-2.0, t194 = t39*-2.0, t195 = t40*-2.0, t196 = t41*-2.0, t197 = t42*-2.0, t198 = t43*-2.0, t202 = -t57, t203 = -t58, t204 = -t59;
    T t205 = -t61, t206 = -t62, t207 = -t63, t208 = -t66, t209 = t50*tx, t210 = -t68, t211 = t52*tx, t212 = -t71, t213 = t54*ty, t226 = rx*t199, t227 = rx*t200, t228 = ry*t199, t229 = ry*t201, t230 = rz*t200, t231 = rz*t201, t232 = t5+t50;
    T t234 = t7+t52, t236 = t9+t54, t242 = t233*tx, t243 = t233*ty, t244 = t235*tx, t245 = t235*tz, t246 = t237*ty, t247 = t237*tz, t254 = t110+t191, t255 = t110+t192, t256 = t111+t192, t258 = t161+t201, t259 = t162+t200, t260 = t163+t199;
    T t267 = rdx*t238*tx*2.0, t268 = rdy*t238*ty*2.0, t269 = rdz*t238*tz*2.0, t277 = t2*t257, t278 = t3*t257, t279 = t4*t257, t141 = t87*tx, t142 = t90*tx, t144 = t85*ty, t147 = t91*ty, t148 = t86*tz, t149 = t89*tz, t220 = -t131, t221 = ox*t213;
    T t222 = -t133, t223 = oy*t211, t224 = -t135, t225 = oz*t209, t239 = t16+t51, t240 = t18+t53, t241 = t22+t55, t261 = -t242, t262 = -t243, t263 = -t244, t264 = -t245, t265 = -t246, t266 = -t247, t270 = rdx*t256, t271 = rdy*t255;
    T t272 = rdz*t254, t274 = t258*tx, t275 = t259*ty, t276 = t260*tz, t289 = t86+t89+t90+t91+t182+t184, t290 = t85+t87+t89+t91+t183+t186, t291 = t85+t86+t87+t90+t185+t187, t248 = t239*tx, t249 = t239*ty, t250 = t240*tx, t251 = t240*tz;
    T t252 = t241*ty, t253 = t241*tz, t292 = t40+t197+t226+t227+t261, t293 = t42+t195+t226+t227+t263, t294 = t38+t198+t228+t229+t262, t295 = t43+t193+t228+t229+t265, t296 = t39+t196+t230+t231+t264, t297 = t41+t194+t230+t231+t266, t273 = -t253;
    T t286 = t249+t251, t287 = t250+t252, t288 = t248+t273;
    
    fdrv(0, 0) = t294*ty+t296*tz; fdrv(0, 1) = -t292*ty; fdrv(0, 2) = -t293*tz; fdrv(0, 3) = t278+t279-ox*t286;
    fdrv(0, 4) = -t137+t143+t149+t151+t220+t222+ox*t198+rx*t58-ry*t57*3.0+t2*t6-t3*t5*3.0-t4*t5*2.0; fdrv(0, 5) = -t138+t146+t147+t153+t221+t224+ox*t196+rx*t61-rz*t59*3.0+t2*t8-t3*t7*2.0-t4*t7*3.0;
    fdrv(0, 6) = tdx*(t81+t95+ox*(t19+t24))+tdy*(t76+t77+t92+t99+t100+t179)+tdz*(t80+t82+t94+t96+t103+t180); fdrv(0, 7) = -t291*tdz-tdy*(t28*6.0+t30*6.0+t75+t93+t95+t181)+tdx*(t26+t27+t36+t37+t88-ox*t5*2.0)*2.0;
    fdrv(0, 8) = -t291*tdy-tdz*(t32*6.0+t34*6.0+t75+t79+t81+t189)+tdx*(t29+t31+t33+t35+t102-ox*t7*2.0)*2.0; fdrv(0, 9) = tdx*(t107+t114-ox*(t46+t48))-tdy*(t62+t105+t117+t167+t203)-tdz*(t63+t108+t113+t169+t205);
    fdrv(0, 10) = -t255*tdz+tdx*(t104+t117+t166)+tdy*(t56-t57*3.0+t109+t114+t204); fdrv(0, 11) = -t254*tdy+tdx*(t106+t113+t168)+tdz*(t56-t59*3.0+t107+t116+t202); fdrv(0, 12) = -t286*tdx-tdy*(t71+t120+t129+t171+t209)-tdz*(t72+t123+t125+t172+t211);
    fdrv(0, 13) = tdy*(t65-t66*3.0+t124+t126+t210)+t257*tdx*ty-t234*tdz*ty*2.0; fdrv(0, 14) = tdz*(t65-t68*3.0+t122+t128+t208)+t257*tdx*tz-t232*tdy*tz*2.0; fdrv(0, 15) = t2*t201*-2.0-t3*t258-t4*t258; fdrv(0, 16) = t157+t166*tx-ox*(t177-t275);
    fdrv(0, 17) = t156+t168*tx-ox*(t178-t276); fdrv(0, 18) = t268+t269+rdx*(t107+t114-ox*t46-ox*t48); fdrv(0, 19) = -t272+rdy*(t109+t114-ox*(t13+t47+t161))-rdx*(t105+t117+t166+oy*t258);
    fdrv(0, 20) = -t271+rdz*(t107+t116-ox*(t12+t49+t161))-rdx*(t108+t113+t168+oz*t258); fdrv(0, 24) = -rdy*(t177-t275)-rdz*(t178-t276)-rdx*t201*tx*2.0; fdrv(0, 25) = t160+rdy*t173-rdx*t258*ty; fdrv(0, 26) = t159+rdz*t175-rdx*t258*tz;
    fdrv(1, 0) = -t294*tx; fdrv(1, 1) = t292*tx+t297*tz; fdrv(1, 2) = -t295*tz; fdrv(1, 3) = t137-t143+t148+t150+t220+t222+oy*t197-rx*t58*3.0+ry*t57-t2*t6*3.0+t3*t5-t4*t6*2.0; fdrv(1, 4) = t277+t279+oy*t288;
    fdrv(1, 5) = t140+t142-t152+t154+t223+t225+oy*t194+ry*t63-rz*t62*3.0-t2*t9*2.0+t3*t10-t4*t9*3.0; fdrv(1, 6) = -t290*tdz-tdx*(t26*6.0+t27*6.0+t92+t99+t100+t179)+tdy*(t28+t30+t32+t34+t74-oy*t6*2.0)*2.0;
    fdrv(1, 7) = tdy*(t76+t100+oy*(t14+t24))+tdx*(t75+t79+t81+t93+t95+t181)+tdz*(t80+t82+t94+t96+t103+t188); fdrv(1, 8) = -t290*tdx-tdz*(t36*6.0+t37*6.0+t76+t77+t92+t190)+tdy*(t29+t31+t33+t35+t102-oy*t9*2.0)*2.0;
    fdrv(1, 9) = -t256*tdz+tdy*(t107+t114+t164)+tdx*(t58*-3.0+t60+t105+t117+t206); fdrv(1, 10) = tdy*(t104+t117-oy*(t44+t48))-tdx*(t59+t109+t114+t165+t202)-tdz*(t61+t106+t115+t169+t207);
    fdrv(1, 11) = tdz*(t60-t62*3.0+t104+t118+t203)+tdx*tz*(ox*ty-oy*tx)*2.0+t238*tdy*tz*2.0; fdrv(1, 12) = tdx*(t67*-3.0+t69+t120+t129+t212)+t257*tdy*tx-t236*tdz*tx*2.0;
    fdrv(1, 13) = t288*tdy-tdx*(t68+t124+t126+t170+t208)-tdz*(t70+t121+t127+t172+t213); fdrv(1, 14) = tdz*(t69-t71*3.0+t119+t130+t209)+t232*tdx*tz*2.0+t257*tdy*tz; fdrv(1, 15) = t157+t164*ty-oy*(t175-t274); fdrv(1, 16) = t3*t200*-2.0-t2*t259-t4*t259;
    fdrv(1, 17) = t155+t168*ty-oy*(t176-t276); fdrv(1, 18) = t272+rdx*(t105+t117-oy*(t13+t45+t162))-rdy*(t109+t114+t164+ox*t259); fdrv(1, 19) = t267+t269+rdy*(t104+t117-oy*t44-oy*t48);
    fdrv(1, 20) = -t270+rdz*(t104+t118-oy*(t11+t49+t162))-rdy*(t106+t115+t168+oz*t259); fdrv(1, 24) = t160+rdx*t174-rdy*t259*tx; fdrv(1, 25) = -rdx*(t175-t274)-rdz*(t176-t276)-rdy*t200*ty*2.0; fdrv(1, 26) = t158+rdz*t177-rdy*t259*tz;
    fdrv(2, 0) = -t296*tx; fdrv(2, 1) = -t297*ty; fdrv(2, 2) = t293*tx+t295*ty; fdrv(2, 3) = t138+t144+t145-t153+t221+t224+oz*t195-rx*t61*3.0+rz*t59-t2*t8*3.0-t3*t8*2.0+t4*t7;
    fdrv(2, 4) = t139+t141+t152-t154+t223+t225+oz*t193-ry*t63*3.0+rz*t62-t2*t10*2.0-t3*t10*3.0+t4*t9; fdrv(2, 5) = t277+t278+oz*t287; fdrv(2, 6) = -t289*tdy-tdx*(t29*6.0+t31*6.0+t94+t96+t103+t180)+tdz*(t28+t30+t32+t34+t74-oz*t8*2.0)*2.0;
    fdrv(2, 7) = -t289*tdx-tdy*(t33*6.0+t35*6.0+t80+t82+t103+t188)+tdz*(t26+t27+t36+t37+t88-oz*t10*2.0)*2.0; fdrv(2, 8) = tdz*(t80+t94+oz*(t14+t19))+tdx*(t75+t79+t81+t93+t95+t189)+tdy*(t76+t77+t92+t99+t100+t190);
    fdrv(2, 9) = tdx*(t61*-3.0+t64+t108+t113+t207)+tdy*tx*(oy*tz-oz*ty)*2.0+t238*tdz*tx*2.0; fdrv(2, 10) = tdy*(t63*-3.0+t64+t106+t115+t205)+tdx*ty*(ox*tz-oz*tx)*2.0+t238*tdz*ty*2.0;
    fdrv(2, 11) = tdz*(t106+t113-oz*(t44+t46))-tdx*(t57+t107+t116+t165+t204)-tdy*(t58+t104+t118+t167+t206); fdrv(2, 12) = tdx*(t70*-3.0+t73+t123+t125+t213)+t236*tdy*tx*2.0+t257*tdz*tx;
    fdrv(2, 13) = tdy*(t72*-3.0+t73+t121+t127+t211)+t234*tdx*ty*2.0+t257*tdz*ty; fdrv(2, 14) = t287*tdz-tdx*(t66+t122+t128+t170+t210)-tdy*(t67+t119+t130+t171+t212); fdrv(2, 15) = t156+t164*tz-oz*(t173-t274); fdrv(2, 16) = t155+t166*tz-oz*(t174-t275);
    fdrv(2, 17) = t4*t199*-2.0-t2*t260-t3*t260; fdrv(2, 18) = t271+rdx*(t108+t113-oz*(t12+t45+t163))-rdz*(t107+t116+t164+ox*t260); fdrv(2, 19) = t270+rdy*(t106+t115-oz*(t11+t47+t163))-rdz*(t104+t118+t166+oy*t260);
    fdrv(2, 20) = t267+t268+rdz*(t106+t113-oz*t44-oz*t46); fdrv(2, 24) = t159+rdx*t176-rdz*t260*tx; fdrv(2, 25) = t158+rdy*t178-rdz*t260*ty; fdrv(2, 26) = -rdx*(t173-t274)-rdy*(t174-t275)-rdz*t199*tz*2.0;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f16(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*ty, t3 = oy*tx, t4 = ox*tz, t5 = oz*tx, t6 = oy*tz, t7 = oz*ty, t8 = rdx*tx, t9 = rdy*ty, t10 = rdz*tz, t11 = rx*tx, t12 = ry*ty, t13 = rz*tz, t14 = tx*tx, t16 = ty*ty, t18 = tz*tz, t21 = rx*ty*2.0, t22 = ry*tx*2.0;
    T t23 = rx*tz*2.0, t25 = rz*tx*2.0, t26 = ry*tz*2.0, t27 = rz*ty*2.0, t32 = rx*ty*tz, t33 = ry*tx*tz, t34 = rz*tx*ty, t20 = t11*2.0, t24 = t12*2.0, t28 = t13*2.0, t29 = t11*ty, t30 = t11*tz, t31 = t12*tx, t35 = t12*tz, t36 = t13*tx;
    T t37 = t13*ty, t38 = -t3, t39 = -t5, t40 = -t7, t41 = ox*t14, t42 = t2*ty, t43 = t3*tx, t44 = t4*tz, t45 = oy*t16, t46 = t5*tx, t47 = t6*tz, t48 = t7*ty, t49 = oz*t18, t50 = rx*t16, t51 = ry*t14, t52 = rx*t18, t53 = rz*t14, t54 = ry*t18;
    T t55 = rz*t16, t56 = t2*tx*2.0, t57 = t4*tx*2.0, t58 = t3*ty*2.0, t59 = t6*ty*2.0, t60 = t5*tz*2.0, t61 = t7*tz*2.0, t68 = t11*t16, t69 = t11*t18, t70 = t12*t14, t77 = t12*t18, t78 = t13*t14, t79 = t13*t16, t80 = ry*t2*tx*4.0;
    T t81 = rx*t3*ty*4.0, t82 = rx*t2*tz*4.0, t83 = rz*t4*tx*4.0, t84 = ry*t3*tz*4.0, t85 = rx*t5*tz*4.0, t86 = rz*t5*ty*4.0, t87 = rz*t6*ty*4.0, t88 = ry*t7*tz*4.0, t95 = t11*tx*3.0, t97 = t12*ty*3.0, t99 = t13*tz*3.0, t107 = -t33, t108 = -t34;
    T t109 = t14+t16, t110 = t14+t18, t111 = t16+t18, t113 = ox*t11*tx*6.0, t114 = t2*t21, t116 = t4*t23, t117 = t3*t22, t121 = oy*t12*ty*6.0, t129 = oz*t13*tz*6.0, t148 = t2*t11*1.2E+1, t149 = ry*t2*tx*8.0, t150 = rx*t3*ty*8.0;
    T t151 = t4*t11*1.2E+1, t153 = t3*t12*1.2E+1, t156 = rz*t4*tx*8.0, t157 = rx*t5*tz*8.0, t159 = t6*t12*1.2E+1, t162 = rz*t6*ty*8.0, t163 = ry*t7*tz*8.0, t164 = t5*t13*1.2E+1, t165 = t7*t13*1.2E+1, t166 = t11*t14*8.0, t167 = t12*t16*8.0;
    T t168 = t13*t18*8.0, t169 = t11+t12, t170 = t11+t13, t171 = t12+t13, t198 = t21+t22, t199 = t23+t25, t200 = t26+t27, t201 = t8+t9+t10, t62 = t20*ty, t63 = t20*tz, t64 = t24*tx, t65 = t24*tz, t66 = t28*tx, t67 = t28*ty, t71 = t52*ty;
    T t72 = t50*tz, t73 = t54*tx, t74 = t51*tz, t75 = t55*tx, t76 = t53*ty, t89 = t42*3.0, t90 = t43*3.0, t91 = t44*3.0, t92 = t46*3.0, t93 = t47*3.0, t94 = t48*3.0, t96 = t14*t20, t98 = t16*t24, t100 = t18*t28, t101 = -t56, t102 = -t57;
    T t103 = -t58, t104 = -t59, t105 = -t60, t106 = -t61, t112 = t20*t41, t115 = rx*t42*6.0, t118 = rx*t44*6.0, t119 = ry*t43*6.0, t120 = t24*t45, t124 = ry*t47*6.0, t125 = rz*t46*6.0, t127 = rz*t48*6.0, t128 = t28*t49, t130 = t16*t20;
    T t131 = t68*4.0, t132 = t68*6.0, t133 = t18*t20, t134 = t14*t24, t135 = t69*4.0, t136 = t70*4.0, t137 = t69*6.0, t138 = t70*6.0, t139 = t18*t24, t140 = t14*t28, t141 = t77*4.0, t142 = t78*4.0, t143 = t77*6.0, t144 = t78*6.0, t145 = t16*t28;
    T t146 = t79*4.0, t147 = t79*6.0, t152 = -t82, t154 = -t84, t155 = -t85, t158 = -t86, t160 = -t87, t161 = -t88, t173 = -t121, t174 = ry*t47*-2.0, t175 = rz*t46*-2.0, t177 = rz*t48*-2.0, t179 = -t129, t186 = -t150, t187 = -t157, t188 = -t163;
    T t189 = rx*t109, t190 = rx*t110, t191 = ry*t109, t192 = ry*t111, t193 = rz*t110, t194 = rz*t111, t195 = t2+t38, t196 = t4+t39, t197 = t6+t40, t202 = t169*tx*tz, t203 = t170*tx*ty, t204 = t169*ty*tz, t205 = t171*tx*ty, t206 = t170*ty*tz;
    T t207 = t171*tx*tz, t214 = t18+t109, t215 = t32+t107, t216 = t32+t108, t217 = t33+t108, t230 = t109*t169, t231 = t110*t170, t232 = t111*t171, t172 = -t119, t176 = -t125, t178 = -t127, t180 = -t71, t181 = -t72, t182 = -t73, t183 = -t74;
    T t184 = -t75, t185 = -t76, t208 = t190*ty, t209 = t189*tz, t210 = t192*tx, t211 = t191*tz, t212 = t194*tx, t213 = t193*ty, t218 = t215*tx, t219 = t216*tx, t220 = t215*ty, t221 = t217*ty, t222 = t216*tz, t223 = t217*tz, t224 = t31+t190;
    T t225 = t36+t189, t226 = t29+t192, t227 = t37+t191, t228 = t30+t194, t229 = t35+t193, t233 = ox*t8*t214*2.0, t234 = rdx*t3*t214*2.0, t235 = rdy*t2*t214*2.0, t236 = rdx*t5*t214*2.0, t237 = oy*t9*t214*2.0, t238 = rdz*t4*t214*2.0;
    T t239 = rdy*t7*t214*2.0, t240 = rdz*t6*t214*2.0, t241 = oz*t10*t214*2.0, t257 = t201*t214*tx*2.0, t258 = t201*t214*ty*2.0, t259 = t201*t214*tz*2.0, t260 = t41+t44+t89+t103, t261 = t41+t42+t91+t105, t262 = t45+t47+t90+t101;
    T t263 = t43+t45+t93+t106, t264 = t48+t49+t92+t102, t265 = t46+t49+t94+t104, t266 = t50+t52+t64+t66+t95, t267 = t51+t54+t62+t67+t97, t268 = t53+t55+t63+t65+t99, t284 = t96+t98+t130+t134+t137+t142+t143+t146+t168;
    T t285 = t96+t100+t132+t133+t136+t140+t141+t147+t167, t286 = t98+t100+t131+t135+t138+t139+t144+t145+t166, t242 = -t220, t243 = -t222, t244 = -t223, t245 = t224*tx, t246 = t225*tx, t247 = t226*tx, t248 = t224*ty, t249 = t226*ty, t250 = t228*tx;
    T t251 = t225*tz, t252 = t227*ty, t253 = t229*ty, t254 = t227*tz, t255 = t228*tz, t256 = t229*tz, t269 = t266*tdx*ty*2.0, t270 = t266*tdx*tz*2.0, t271 = t267*tdy*tx*2.0, t272 = t267*tdy*tz*2.0, t273 = t268*tdz*tx*2.0, t274 = t268*tdz*ty*2.0;
    T t287 = t286*tdx, t288 = t285*tdy, t289 = t284*tdz, t290 = t83+t113+t115+t116+t149+t160+t172+t173+t174+t186, t291 = t80+t113+t114+t118+t156+t161+t176+t177+t179+t187, t292 = t81+t117+t121+t124+t155+t162+t175+t178+t179+t188;
    T t275 = t185+t204+t213+t218+t254, t276 = t181+t207+t209+t221+t250, t277 = t183+t206+t211+t219+t253, t278 = t182+t203+t210+t243+t248, t279 = t184+t202+t212+t242+t251, t280 = t180+t205+t208+t244+t247, t281 = t78+t79+t230+t246+t252;
    T t282 = t70+t77+t231+t245+t256, t283 = t68+t69+t232+t249+t255;
    
    fdrv(0, 0) = t279*ty-t278*tz; fdrv(0, 1) = t275*ty+t282*tz; fdrv(0, 2) = -t281*ty-t277*tz; fdrv(0, 3) = t197*t266*2.0;
    fdrv(0, 4) = t3*t32*4.0-t5*t31*4.0-t5*t36*2.0+t6*t37*4.0-t7*t37*6.0-t5*t50*6.0-t5*t52*2.0-t11*t46*2.0+t6*t54*2.0-t12*t48*8.0-t7*t54*4.0-t13*t49*2.0+t117*tz+t6*t12*ty*6.0;
    fdrv(0, 5) = t120-t5*t32*4.0+t3*t36*4.0-t5*t34*2.0+t6*t35*6.0-t7*t35*4.0+t3*t50*2.0+t3*t52*6.0+t13*t47*8.0+t6*t55*4.0-t7*t55*2.0+t20*t43+t24*t43-t7*t13*tz*6.0;
    fdrv(0, 6) = -tdy*(t154+rx*t48*6.0+rx*t49*2.0+t5*t11*6.0+t5*t12*8.0+t5*t13*4.0-rx*t6*ty*4.0)+tdz*(t158+rx*t45*2.0+rx*t47*6.0+t3*t11*6.0+t3*t12*4.0+t3*t13*8.0-rx*t7*tz*4.0)+t197*tdx*(t11*6.0+t24+t28)*2.0;
    fdrv(0, 7) = -tdy*(-t159+t165+ry*t46*4.0+ry*t49*4.0-t6*t13*4.0+t7*t12*2.4E+1+rx*t5*ty*1.2E+1-rx*t3*tz*4.0)+t292*tdz-oz*t266*tdx*2.0+t197*t198*tdx*2.0;
    fdrv(0, 8) = tdz*(t159-t165+rz*t43*4.0+rz*t45*4.0+t6*t13*2.4E+1-t7*t12*4.0-rx*t5*ty*4.0+rx*t3*tz*1.2E+1)+t292*tdy+oy*t266*tdx*2.0+t197*t199*tdx*2.0; fdrv(0, 9) = t197*tdx*(t14+t109+t110)*2.0-t265*tdy*tx*2.0+t263*tdz*tx*2.0;
    fdrv(0, 10) = tdy*(t6*t16*6.0-t7*t16*8.0+t6*t18*2.0-t7*t18*4.0-t46*ty*4.0+t43*tz*2.0)+t263*tdz*ty*2.0+t197*tdx*tx*ty*4.0; fdrv(0, 11) = tdz*(t6*t16*4.0-t7*t16*2.0+t6*t18*8.0-t7*t18*6.0-t46*ty*2.0+t43*tz*4.0)-t265*tdy*tz*2.0+t197*tdx*tx*tz*4.0;
    fdrv(0, 13) = t270+t272+t289; fdrv(0, 14) = -t269-t274-t288; fdrv(0, 15) = t197*t214*tx*2.0; fdrv(0, 16) = t197*t214*ty*2.0; fdrv(0, 17) = t197*t214*tz*2.0; fdrv(0, 18) = t197*(rdx*t16+rdx*t18+t8*tx*3.0+t9*tx*2.0+t10*tx*2.0)*2.0;
    fdrv(0, 19) = -t236-t239-t241+rdy*t197*t214*2.0+t8*t197*ty*4.0+t9*t197*ty*4.0+t10*t197*ty*4.0; fdrv(0, 20) = t234+t237+t240+rdz*t197*t214*2.0+t8*t197*tz*4.0+t9*t197*tz*4.0+t10*t197*tz*4.0; fdrv(0, 25) = t259; fdrv(0, 26) = -t258;
    fdrv(1, 0) = -t279*tx-t283*tz; fdrv(1, 1) = -t275*tx+t280*tz; fdrv(1, 2) = t281*tx+t276*tz;
    fdrv(1, 3) = t128-t2*t32*2.0-t2*t33*4.0+t5*t31*6.0-t4*t36*4.0+t5*t36*6.0+t5*t50*4.0-t4*t52*2.0+t5*t52*4.0+t11*t46*8.0+t7*t54*2.0+t24*t48+t28*t48-t4*t11*tx*6.0; fdrv(1, 4) = t196*t267*-2.0;
    fdrv(1, 5) = t2*t29*-2.0-t4*t30*6.0+t5*t30*4.0-t2*t37*4.0+t5*t35*4.0-t11*t41*2.0-t2*t51*2.0-t12*t42*2.0-t2*t54*6.0-t4*t53*4.0-t13*t44*8.0+t5*t53*2.0+t5*t55*2.0+t5*t13*tz*6.0;
    fdrv(1, 6) = -t291*tdz+tdx*(-t151+t164+rx*t48*4.0+rx*t49*4.0+t5*t11*2.4E+1-t4*t13*4.0+t5*t12*1.2E+1-ry*t2*tz*4.0)+oz*t267*tdy*2.0-t196*t198*tdy*2.0;
    fdrv(1, 7) = tdx*(t152+ry*t46*6.0+ry*t49*2.0+t7*t12*6.0+t7*t13*4.0-ry*t4*tx*4.0+rx*t5*ty*8.0)-tdz*(t158+ry*t41*2.0+ry*t44*6.0+t2*t11*4.0+t2*t12*6.0+t2*t13*8.0-ry*t5*tz*4.0)-t196*tdy*(t12*6.0+t20+t28)*2.0;
    fdrv(1, 8) = -t291*tdx-tdz*(t151-t164+rz*t41*4.0+rz*t42*4.0-t5*t11*4.0+t4*t13*2.4E+1-t5*t12*4.0+ry*t2*tz*1.2E+1)-ox*t267*tdy*2.0-t196*t200*tdy*2.0;
    fdrv(1, 9) = -tdx*(t4*t14*6.0-t5*t14*8.0-t5*t16*4.0+t4*t18*2.0-t5*t18*4.0+t42*tz*2.0)-t261*tdz*tx*2.0-t196*tdy*tx*ty*4.0; fdrv(1, 10) = t196*tdy*(t16+t109+t111)*-2.0+t264*tdx*ty*2.0-t261*tdz*ty*2.0;
    fdrv(1, 11) = -tdz*(t4*t14*4.0-t5*t14*2.0-t5*t16*2.0+t4*t18*8.0-t5*t18*6.0+t42*tz*4.0)+t264*tdx*tz*2.0-t196*tdy*ty*tz*4.0; fdrv(1, 12) = -t270-t272-t289; fdrv(1, 14) = t271+t273+t287; fdrv(1, 15) = t196*t214*tx*-2.0; fdrv(1, 16) = t196*t214*ty*-2.0;
    fdrv(1, 17) = t196*t214*tz*-2.0; fdrv(1, 18) = t236+t239+t241-rdx*t196*t214*2.0-t8*t196*tx*4.0-t9*t196*tx*4.0-t10*t196*tx*4.0; fdrv(1, 19) = t196*(rdy*t14+rdy*t18+t8*ty*2.0+t9*ty*3.0+t10*ty*2.0)*-2.0;
    fdrv(1, 20) = -t233-t235-t238-rdz*t196*t214*2.0-t8*t196*tz*4.0-t9*t196*tz*4.0-t10*t196*tz*4.0; fdrv(1, 24) = -t259; fdrv(1, 26) = t257; fdrv(2, 0) = t278*tx+t283*ty; fdrv(2, 1) = -t282*tx-t280*ty; fdrv(2, 2) = t277*tx-t276*ty;
    fdrv(2, 3) = t2*t31*4.0-t3*t31*6.0+t2*t36*4.0-t3*t36*6.0-t6*t35*2.0+t2*t50*2.0-t3*t50*4.0+t2*t52*2.0-t11*t43*8.0-t3*t52*4.0-t12*t45*2.0-t13*t47*2.0-t6*t55*2.0+t2*t11*tx*6.0;
    fdrv(2, 4) = t112+t2*t29*6.0-t3*t29*4.0+t2*t37*6.0-t3*t37*4.0+t2*t51*4.0-t3*t51*2.0+t12*t42*8.0+t2*t54*4.0-t3*t54*2.0+t4*t53*2.0+t20*t44+t28*t44-t3*t12*ty*6.0; fdrv(2, 5) = t195*t268*2.0;
    fdrv(2, 6) = -tdx*(-t148+t153+rx*t45*4.0+rx*t47*4.0-t2*t12*4.0+t3*t11*2.4E+1-t2*t13*4.0+t3*t13*1.2E+1)+t290*tdy-oy*t268*tdz*2.0+t195*t199*tdz*2.0;
    fdrv(2, 7) = tdy*(t148-t153+ry*t41*4.0+ry*t44*4.0+t2*t12*2.4E+1-t3*t11*4.0+t2*t13*1.2E+1-t3*t13*4.0)+t290*tdx+ox*t268*tdz*2.0+t195*t200*tdz*2.0;
    fdrv(2, 8) = -tdx*(t152+rz*t43*6.0+rz*t45*2.0+t6*t12*4.0+t6*t13*6.0-rz*t2*tx*4.0+rx*t3*tz*8.0)+tdy*(t154+rz*t41*2.0+rz*t42*6.0+t4*t11*4.0+t4*t13*6.0+ry*t2*tz*8.0-rz*t3*ty*4.0)+t195*tdz*(t13*6.0+t20+t24)*2.0;
    fdrv(2, 9) = tdx*(t2*t14*6.0-t3*t14*8.0+t2*t16*2.0-t3*t16*4.0+t2*t18*2.0-t3*t18*4.0)+t260*tdy*tx*2.0+t195*tdz*tx*tz*4.0; fdrv(2, 10) = tdy*(t2*t14*4.0-t3*t14*2.0+t2*t16*8.0-t3*t16*6.0+t2*t18*4.0-t3*t18*2.0)-t262*tdx*ty*2.0+t195*tdz*ty*tz*4.0;
    fdrv(2, 11) = t262*tdx*tz*-2.0+t260*tdy*tz*2.0+t195*tdz*(t18*2.0+t214)*2.0; fdrv(2, 12) = t269+t274+t288; fdrv(2, 13) = -t271-t273-t287; fdrv(2, 15) = t195*t214*tx*2.0; fdrv(2, 16) = t195*t214*ty*2.0; fdrv(2, 17) = t195*t214*tz*2.0;
    fdrv(2, 18) = -t234-t237-t240+rdx*t195*t214*2.0+t8*t195*tx*4.0+t9*t195*tx*4.0+t10*t195*tx*4.0; fdrv(2, 19) = t233+t235+t238+rdy*t195*t214*2.0+t8*t195*ty*4.0+t9*t195*ty*4.0+t10*t195*ty*4.0;
    fdrv(2, 20) = t195*(rdz*t14+rdz*t16+t8*tz*2.0+t9*tz*2.0+t10*tz*3.0)*2.0; fdrv(2, 24) = t258; fdrv(2, 25) = -t257;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f17(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*rx, t3 = oy*ry, t4 = oz*rz, t5 = ox*ty, t6 = oy*tx, t7 = ox*tz, t8 = oz*tx, t9 = oy*tz, t10 = oz*ty, t11 = rdx*tx, t12 = rdy*ty, t13 = rdz*tz, t14 = rx*tx, t15 = ry*ty, t16 = rz*tz, t32 = rx*tdx*ty, t33 = ry*tdy*tx;
    T t34 = rx*tdx*tz, t35 = rz*tdz*tx, t36 = ry*tdy*tz, t37 = rz*tdz*ty, t20 = t5*2.0, t21 = t6*2.0, t22 = t7*2.0, t23 = t8*2.0, t24 = t9*2.0, t25 = t10*2.0, t26 = t14*2.0, t27 = t15*2.0, t28 = t16*2.0, t29 = ox*t11, t30 = oy*t12, t31 = oz*t13;
    T t38 = -t3, t39 = -t4, t40 = -t6, t42 = -t8, t44 = -t10, t46 = t14+t15, t47 = t14+t16, t48 = t15+t16, t58 = t11+t12+t13, t41 = -t21, t43 = -t23, t45 = -t25, t49 = t2+t38, t50 = t2+t39, t51 = t3+t39, t52 = t5+t40, t54 = t7+t42, t56 = t9+t44;
    T t62 = t28+t46, t63 = t27+t47, t64 = t26+t48, t65 = t58*tx, t66 = t58*ty, t67 = t58*tz, t53 = t5+t41, t55 = t7+t43, t57 = t9+t45, t68 = t64*tdx, t69 = t63*tdy, t70 = t62*tdz;
    
    fdrv(0, 1) = t14*tz+t48*tz; fdrv(0, 2) = -t14*ty-t48*ty;
    fdrv(0, 3) = rx*t56; fdrv(0, 4) = -oz*t63+t3*tz; fdrv(0, 5) = oy*t62+t39*ty; fdrv(0, 6) = rx*(oy*tdz-oz*tdy); fdrv(0, 7) = t51*tdz-oz*rx*tdx-oz*ry*tdy*2.0; fdrv(0, 8) = t51*tdy+oy*rx*tdx+oy*rz*tdz*2.0; fdrv(0, 9) = t56*tdx+t42*tdy+t6*tdz;
    fdrv(0, 10) = t57*tdy+oy*tdz*ty; fdrv(0, 11) = -tdz*(t10-t24)-oz*tdy*tz; fdrv(0, 13) = t34+t36+t70; fdrv(0, 14) = -t32-t37-t69; fdrv(0, 15) = t56*tx; fdrv(0, 16) = t9*ty+t44*ty; fdrv(0, 17) = t9*tz+t44*tz; fdrv(0, 18) = rdx*t56;
    fdrv(0, 19) = -t31+rdx*t42+rdy*t57; fdrv(0, 20) = t30+rdx*t6-rdz*(t10-t24); fdrv(0, 25) = t67; fdrv(0, 26) = -t66; fdrv(1, 0) = -t15*tz-t47*tz; fdrv(1, 2) = t15*tx+t47*tx; fdrv(1, 3) = oz*t64-t2*tz; fdrv(1, 4) = -ry*t54; fdrv(1, 5) = -ox*t62+t4*tx;
    fdrv(1, 6) = -t50*tdz+oz*rx*tdx*2.0+oz*ry*tdy; fdrv(1, 7) = -ry*(ox*tdz-oz*tdx); fdrv(1, 8) = -t50*tdx-ox*ry*tdy-ox*rz*tdz*2.0; fdrv(1, 9) = -t55*tdx-ox*tdz*tx; fdrv(1, 10) = t10*tdx-t54*tdy-t5*tdz; fdrv(1, 11) = tdz*(t8-t22)+oz*tdx*tz;
    fdrv(1, 12) = -t34-t36-t70; fdrv(1, 14) = t33+t35+t68; fdrv(1, 15) = -t7*tx+t8*tx; fdrv(1, 16) = -t54*ty; fdrv(1, 17) = -t7*tz+t8*tz; fdrv(1, 18) = t31-rdx*t55+rdy*t10; fdrv(1, 19) = -rdy*t54; fdrv(1, 20) = -t29-rdy*t5+rdz*(t8-t22); fdrv(1, 24) = -t67;
    fdrv(1, 26) = t65; fdrv(2, 0) = t16*ty+t46*ty; fdrv(2, 1) = -t16*tx-t46*tx; fdrv(2, 3) = -oy*t64+t2*ty; fdrv(2, 4) = ox*t63+t38*tx; fdrv(2, 5) = rz*t52; fdrv(2, 6) = t49*tdy-oy*rx*tdx*2.0-oy*rz*tdz; fdrv(2, 7) = t49*tdx+ox*ry*tdy*2.0+ox*rz*tdz;
    fdrv(2, 8) = rz*(ox*tdy-oy*tdx); fdrv(2, 9) = t53*tdx+ox*tdy*tx; fdrv(2, 10) = -tdy*(t6-t20)-oy*tdx*ty; fdrv(2, 11) = -t9*tdx+t7*tdy+t52*tdz; fdrv(2, 12) = t32+t37+t69; fdrv(2, 13) = -t33-t35-t68; fdrv(2, 15) = t5*tx+t40*tx; fdrv(2, 16) = t5*ty+t40*ty;
    fdrv(2, 17) = t52*tz; fdrv(2, 18) = -t30+rdx*t53-rdz*t9; fdrv(2, 19) = t29+rdz*t7-rdy*(t6-t20); fdrv(2, 20) = rdz*t52; fdrv(2, 24) = t66; fdrv(2, 25) = -t65;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f18(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = oy*ty, t4 = oz*tz, t5 = rdx*tx, t6 = rdy*ty, t7 = rdz*tz, t8 = rx*tx, t9 = rx*ty, t10 = ry*tx, t11 = rx*tz, t12 = ry*ty, t13 = rz*tx, t14 = ry*tz, t15 = rz*ty, t16 = rz*tz, t17 = tx*tx, t18 = tx*tx*tx, t19 = ty*ty;
    T t20 = ty*ty*ty, t21 = tz*tz, t22 = tz*tz*tz, t23 = ox*ty*2.0, t24 = oy*tx*2.0, t25 = ox*ty*4.0, t26 = oy*tx*4.0, t27 = ox*tz*2.0, t28 = oz*tx*2.0, t29 = ox*tz*4.0, t30 = oz*tx*4.0, t31 = oy*tz*2.0, t32 = oz*ty*2.0, t33 = oy*tz*4.0;
    T t34 = oz*ty*4.0, t38 = tx*ty*2.0, t39 = tx*tz*2.0, t40 = ty*tz*2.0, t41 = ox*ty*tz, t42 = oy*tx*tz, t43 = oz*tx*ty, t35 = t8*2.0, t36 = t12*2.0, t37 = t16*2.0, t44 = t8*ty, t45 = t8*tz, t46 = t10*ty, t47 = t12*tz, t48 = t13*tz, t49 = t15*tz;
    T t50 = -t23, t51 = -t25, t52 = -t28, t53 = -t30, t54 = -t31, t55 = -t33, t56 = t2*tx, t57 = ox*t19, t58 = oy*t17, t59 = ox*t20, t60 = oy*t18, t61 = ox*t21, t62 = t3*ty, t63 = oz*t17, t64 = ox*t22, t65 = oz*t18, t66 = oy*t21, t67 = oz*t19;
    T t68 = oy*t22, t69 = oz*t20, t70 = t4*tz, t71 = t8*t17, t72 = t9*t19, t73 = t10*t17, t74 = t11*t21, t75 = t12*t19, t76 = t13*t17, t77 = t14*t21, t78 = t15*t19, t79 = t16*t21, t80 = t19*tx, t81 = t17*ty, t82 = t21*tx, t83 = t17*tz;
    T t84 = t21*ty, t85 = t19*tz, t86 = rx*t2*4.0, t87 = ry*t2*2.0, t89 = ox*t9*4.0, t90 = oy*t8*4.0, t92 = rz*t2*2.0, t93 = rx*t3*2.0, t95 = ox*t11*4.0, t96 = ox*t12*4.0, t97 = oy*t10*4.0, t98 = oz*t8*4.0, t99 = ox*t14*2.0, t100 = ox*t15*2.0;
    T t101 = oy*t11*2.0, t102 = oy*t13*2.0, t103 = oz*t9*2.0, t104 = oz*t10*2.0, t105 = ox*t14*4.0, t106 = ox*t15*4.0, t107 = oy*t11*4.0, t108 = ry*t3*4.0, t109 = oy*t13*4.0, t110 = oz*t9*4.0, t111 = oz*t10*4.0, t113 = rz*t3*2.0, t114 = rx*t4*2.0;
    T t116 = ox*t16*4.0, t117 = oy*t14*4.0, t118 = oz*t12*4.0, t119 = oz*t13*4.0, t121 = ry*t4*2.0, t122 = oy*t16*4.0, t123 = oz*t15*4.0, t124 = rz*t4*4.0, t125 = t2*ty*2.0, t126 = t2*ty*4.0, t127 = t2*tz*2.0, t128 = t3*tx*2.0, t129 = t2*tz*4.0;
    T t130 = t3*tx*4.0, t131 = t3*tz*2.0, t132 = t4*tx*2.0, t133 = t3*tz*4.0, t134 = t4*tx*4.0, t135 = t4*ty*2.0, t136 = t4*ty*4.0, t143 = t9*tz*2.0, t144 = t10*tz*2.0, t145 = t13*ty*2.0, t156 = ox*t9*ty, t157 = t2*t10, t158 = ox*t11*tz;
    T t159 = t2*t13, t160 = t3*t9, t161 = oy*t10*tx, t163 = oy*t14*tz, t164 = t3*t15, t165 = t4*t11, t166 = oz*t13*tx, t167 = t4*t14, t168 = oz*t15*ty, t182 = t8*t19, t183 = t8*t21, t185 = t9*t21, t186 = t9*ty*tz, t187 = t10*t21, t188 = t10*tx*tz;
    T t189 = t13*t19, t190 = t13*tx*ty, t191 = t12*t21, t200 = t2*t9*2.0, t201 = t2*t9*4.0, t202 = t2*t9*6.0, t203 = t2*t11*2.0, t206 = t2*t11*4.0, t207 = t2*t12*4.0, t208 = t3*t8*4.0, t209 = t2*t11*6.0, t210 = t9*t27, t211 = t2*t14*2.0;
    T t212 = t2*t15*2.0, t213 = t8*t31, t214 = t3*t10*2.0, t215 = t8*t32, t216 = t8*t33, t217 = t3*t10*4.0, t220 = t3*t10*6.0, t222 = t12*t27, t224 = t3*t11*2.0, t226 = t3*t13*2.0, t228 = t10*t32, t230 = t2*t16*4.0, t231 = t4*t8*4.0;
    T t232 = t10*t34, t235 = t15*t27, t236 = t3*t14*2.0, t237 = t13*t31, t238 = t4*t9*2.0, t239 = t4*t10*2.0, t241 = t15*t29, t242 = t3*t14*4.0, t243 = t13*t33, t245 = t3*t14*6.0, t249 = t4*t13*2.0, t250 = t3*t16*4.0, t251 = t4*t12*4.0;
    T t252 = t4*t13*4.0, t253 = t4*t13*6.0, t254 = t4*t15*2.0, t255 = t4*t15*4.0, t256 = t4*t15*6.0, t258 = t10*t40, t259 = t13*t40, t274 = t8*tx*6.0, t277 = t12*ty*6.0, t280 = t16*tz*6.0, t283 = oy*t8*-2.0, t294 = ox*t16*-2.0, t297 = oz*t12*-2.0;
    T t308 = -t41, t309 = -t42, t310 = -t43, t320 = t17+t19, t321 = t17+t21, t322 = t19+t21, t323 = t2*t5*2.0, t324 = t3*t6*2.0, t325 = t4*t7*2.0, t327 = t2*t8*3.0, t328 = t8*t24, t330 = oy*t8*tx*3.0, t331 = t12*t23, t332 = t8*t28;
    T t334 = ox*t12*ty*3.0, t336 = oz*t8*tx*3.0, t337 = t14*t27, t339 = t11*t31, t341 = t13*t24, t342 = t9*t32, t344 = ox*t14*tz*3.0, t345 = ox*t15*ty*3.0, t346 = oy*t11*tz*3.0, t347 = t3*t12*3.0, t348 = oy*t13*tx*3.0, t349 = oz*t9*ty*3.0;
    T t350 = oz*t10*tx*3.0, t351 = t16*t27, t352 = t12*t32, t353 = ox*t16*tz*3.0, t355 = oz*t12*ty*3.0, t357 = t16*t31, t358 = oy*t16*tz*3.0, t361 = t4*t16*3.0, t368 = t10*t19*2.0, t369 = t10*t38, t372 = t10*t19*3.0, t376 = t13*t21*2.0;
    T t377 = t13*t39, t380 = t13*t21*3.0, t382 = t15*t21*2.0, t383 = t15*t40, t384 = t15*t21*3.0, t387 = ox*t9*tz*-2.0, t391 = oy*t10*tz*-2.0, t393 = t4*t8*-2.0, t397 = oz*t13*ty*-2.0, t398 = t3*t16*-2.0, t399 = t4*t12*-2.0, t401 = rdx*t2*t40;
    T t402 = rdy*t3*t39, t403 = rdz*t4*t38, t410 = t8*tx*-2.0, t411 = t12*ty*-2.0, t412 = t16*tz*-2.0, t413 = t8+t12, t414 = t8+t16, t415 = t12+t16, t442 = t5+t6+t7, t88 = oy*t35, t91 = ox*t36, t94 = oz*t35, t112 = ox*t37, t120 = oy*t37;
    T t137 = t35*ty, t138 = t44*4.0, t139 = t35*tz, t140 = t46*2.0, t141 = t45*4.0, t142 = t46*4.0, t146 = t36*tz, t147 = t48*2.0, t148 = t47*4.0, t149 = t48*4.0, t150 = t49*2.0, t151 = t49*4.0, t152 = t5*t56, t153 = t6*t62, t154 = t7*t70;
    T t155 = t8*t56, t162 = t12*t62, t169 = t16*t70, t170 = t56*ty, t171 = t56*tz, t172 = t62*tx, t173 = t61*ty, t174 = t57*tz, t175 = t66*tx, t176 = t58*tz, t177 = t67*tx, t178 = t63*ty, t179 = t62*tz, t180 = t70*tx, t181 = t70*ty, t184 = t46*tx;
    T t192 = t48*tx, t193 = t49*ty, t194 = rdx*t126, t195 = rdx*t129, t196 = rdy*t130, t197 = rdy*t133, t198 = rdz*t134, t199 = rdz*t136, t204 = t2*t36, t205 = t3*t35, t219 = oy*t45*6.0, t221 = oz*t44*6.0, t223 = t2*t37, t227 = t4*t35;
    T t233 = ox*t47*6.0, t234 = oz*t46*6.0, t244 = ox*t49*6.0, t246 = oy*t48*6.0, t247 = t3*t37, t248 = t4*t36, t260 = t57*2.0, t261 = t58*2.0, t262 = t57*3.0, t263 = t58*3.0, t264 = t61*2.0, t265 = t63*2.0, t266 = t61*3.0, t267 = t63*3.0;
    T t268 = t66*2.0, t269 = t67*2.0, t270 = t66*3.0, t271 = t67*3.0, t272 = t35*tx, t273 = t71*4.0, t275 = t36*ty, t276 = t75*4.0, t278 = t37*tz, t279 = t79*4.0, t282 = -t87, t284 = -t89, t286 = -t98, t291 = -t107, t292 = -t108, t293 = -t109;
    T t295 = -t113, t298 = -t117, t299 = -t119, t300 = -t122, t301 = -t124, t302 = -t125, t303 = -t126, t304 = -t127, t305 = -t128, t306 = -t129, t307 = -t130, t311 = -t131, t312 = -t132, t313 = -t133, t314 = -t134, t315 = -t135, t317 = -t45;
    T t318 = -t46, t319 = -t49, t326 = t2*t35, t329 = t156*3.0, t333 = t158*3.0, t335 = t161*3.0, t340 = t3*t36, t354 = t163*3.0, t356 = t166*3.0, t359 = t168*3.0, t360 = t4*t37, t362 = t19*t35, t364 = t182*3.0, t365 = t44*tx*3.0, t366 = t21*t35;
    T t370 = t183*3.0, t371 = t45*tx*3.0, t374 = t21*t36, t378 = t191*3.0, t379 = t47*ty*3.0, t389 = -t212, t390 = -t224, t392 = -t226, t394 = -t231, t395 = -t238, t396 = -t239, t404 = t9*t127, t405 = t10*t131, t406 = t13*t135, t407 = -t60;
    T t408 = -t64, t409 = -t69, t416 = ox*t411, t417 = oz*t410, t419 = -t347, t423 = -t168, t424 = oy*t412, t426 = -t361, t433 = t413*tx, t434 = t413*ty, t435 = t414*tx, t436 = t414*tz, t437 = t415*ty, t438 = t415*tz, t439 = t35+t36;
    T t440 = t35+t37, t441 = t36+t37, t452 = t2+t3+t42+t308, t453 = t2+t4+t41+t310, t454 = t3+t4+t43+t309, t257 = t137*tz, t363 = t137*tx, t367 = t139*tx, t373 = t184*3.0, t375 = t146*ty, t381 = t192*3.0, t385 = t193*3.0, t422 = -t356;
    T t428 = -t172, t429 = -t174, t430 = -t175, t431 = -t178, t432 = -t181, t443 = t439*tx, t444 = -t434, t445 = t439*ty, t446 = -t435, t447 = t440*tx, t448 = t440*tz, t449 = t441*ty, t450 = -t438, t451 = t441*tz, t455 = t9+t10+t47+t436;
    T t456 = t11+t13+t44+t437, t457 = t14+t15+t48+t433, t476 = t27+t53+t62+t66+t263+t302, t477 = t34+t54+t56+t61+t262+t305, t478 = t26+t50+t67+t70+t267+t304, t479 = t32+t55+t56+t57+t266+t312, t480 = t24+t51+t63+t70+t271+t311;
    T t481 = t29+t52+t58+t62+t270+t315, t494 = t86+t91+t93+t116+t164+t216+t236+t299+t348+t358+t387+t389, t495 = t92+t94+t118+t124+t157+t200+t241+t298+t334+t344+t396+t397, t496 = t90+t108+t120+t121+t165+t232+t249+t284+t336+t349+t390+t391;
    T t499 = t95+t160+t214+t243+t286+t295+t297+t301+t330+t346+t395+t397, t458 = t455*tx, t459 = t457*tx, t460 = t456*ty, t461 = t457*ty, t462 = t455*tz, t463 = t456*tz, t466 = t9+t10+t317+t450, t467 = t11+t13+t319+t444, t468 = t14+t15+t318+t446;
    T t491 = t59+t170+t173+t265+t269+t304+t311+t407+t428+t430, t493 = t68+t176+t179+t260+t264+t305+t312+t409+t431+t432, t501 = t100+t103+t156+t204+t230+t291+t293+t327+t333+t394+t399+t422+t423+t426, t469 = t467*tx, t470 = t468*tx, t471 = t466*ty;
    T t472 = t467*ty, t473 = t466*tz, t474 = t468*tz, t482 = t443+t463, t483 = t449+t458, t484 = t448+t461, t475 = -t473, t485 = t447+t471, t486 = t445+t474, t487 = t451+t469, t489 = t459+t475;
    
    fdrv(0, 0) = -t486*ty-t484*tz;
    fdrv(0, 1) = t486*tx-tz*(t462-t472); fdrv(0, 2) = t484*tx+ty*(t462-t472); fdrv(0, 3) = t156*-2.0-t158*2.0+t208+t231+t247+t248+t340+t360+t406-t8*t42*3.0+t8*t43*3.0+t9*t67-t11*t66+t9*t70-t13*t66*2.0+t10*t269-t160*tz-t3*t10*tz*2.0;
    fdrv(0, 4) = t169-t201+t217+t237+t239+t328-ox*t49*4.0-t3*t45*2.0+t4*t45-t3*t47*3.0+t8*t63+t28*t46+t12*t67*4.0-t14*t66+t36*t70+t8*t271+t10*t309+t398*tz-ox*t12*ty*6.0-ox*t14*tz*2.0+t4*t13*tx+t4*t15*ty*3.0;
    fdrv(0, 5) = -t162-t206+t226+t228+t252+t332-ox*t47*4.0-t3*t44-t3*t49*2.0+t4*t49*3.0-t13*t42*2.0+t13*t43-t8*t58-t8*t66*3.0+t15*t67-t16*t66*4.0+t4*t137+t248*ty-ox*t15*ty*2.0-ox*t16*tz*6.0-t3*t10*tx-t3*t14*tz*3.0;
    fdrv(0, 6) = tdx*(-t219+t221-t236+t254+t352+t424+rx*t3*4.0+rx*t4*4.0)+t496*tdy-t499*tdz;
    fdrv(0, 7) = t496*tdx-tdy*(t86-t97+t116-t167*2.0+t213-t221+t245-t256+t357+ox*t12*1.2E+1-oz*t10*tx*2.0-oz*t12*ty*1.2E+1)+tdz*(t102+t104-t105-t106-t161+t166-t205-t250+t251-t354+t359+t361+t419+t4*t8*2.0);
    fdrv(0, 8) = -tdz*(t86+t96+t164*2.0+t219+t245-t256+t299+t341+ox*t16*1.2E+1-oz*t44*2.0+t297*ty+oy*t16*tz*1.2E+1)-t499*tdx+tdy*(t102+t104-t105-t106-t161+t166-t205-t250+t251-t354+t359+t361+t419+t4*t8*2.0);
    fdrv(0, 9) = -tdx*(t68+t176*3.0-t178*3.0+t179+t260+t264+t307+t314+t409+t432)+t480*tdy*tx-t481*tdz*tx; fdrv(0, 10) = -tdy*(t57*6.0+t68-t69*4.0+t176-t178*2.0+t179*3.0-t181*2.0+t264+t307+t312)+t454*tdx*ty*2.0-t481*tdz*ty;
    fdrv(0, 11) = -tdz*(t61*6.0+t68*4.0-t181*3.0+t260+t305+t314+t409+t431+t3*t40+t17*t31)+t454*tdx*tz*2.0+t480*tdy*tz; fdrv(0, 12) = -tdz*(t141+t148+t280+t15*ty*2.0)-tdy*(t138+t151+t277+t14*tz*2.0)-rx*t322*tdx*2.0;
    fdrv(0, 13) = -tdz*(t71+t75-t145+t182+t184+t279+t370+t377+t378+t383)-tdx*(t74-t138-t150+t186+t258+t371+t376+t411)-tdy*(t77-t142-t147+t188+t257+t379+t382+t410);
    fdrv(0, 14) = tdy*(t71+t79+t144+t183+t192+t276+t364+t369+t374+t385)+tdx*(t72+t141+t146+t185+t259+t278+t365+t368)+tdz*(t78+t140+t149+t190+t257+t272+t375+t384); fdrv(0, 15) = -t493*tx; fdrv(0, 16) = -t493*ty; fdrv(0, 17) = -t493*tz;
    fdrv(0, 18) = t324+t325+t403-rdx*t57*2.0-rdx*t61*2.0-rdx*t68+rdx*t69-rdx*t179+rdx*t181+t3*t5*4.0+t4*t5*4.0+t3*t7*2.0+t4*t6*2.0-t5*t42*3.0+t5*t43*3.0-t7*t42*2.0+t6*t28*ty-rdy*t3*tx*tz*2.0;
    fdrv(0, 19) = t154+t196-rdy*t61*2.0-rdy*t68+rdy*t132-rdy*t176+t5*t24+t7*t24+t5*t63+t6*t67*4.0+t5*t70+t6*t70*2.0+t6*t265+t5*t271-ox*t6*ty*6.0-ox*t7*ty*4.0+rdz*t4*t17+rdz*t4*t19*3.0-rdx*t2*ty*4.0-t3*t5*tz*2.0-t3*t6*tz*3.0-t3*t7*tz*2.0;
    fdrv(0, 20) = -t153+t198-rdz*t57*2.0+rdz*t69+rdz*t128+rdz*t178+t5*t28+t6*t28-t5*t58-t7*t58*2.0-t5*t62-t7*t62*2.0-t5*t66*3.0-t7*t66*4.0+t5*t135+t6*t135-ox*t6*tz*4.0-ox*t7*tz*6.0-rdy*t3*t17-rdy*t3*t21*3.0-rdx*t2*tz*4.0+t4*t7*ty*3.0;
    fdrv(0, 24) = t322*t442*-2.0; fdrv(0, 25) = -t442*(t22-t38+t83+t85); fdrv(0, 26) = t442*(t20+t39+t81+t84); fdrv(1, 0) = t482*ty+t489*tz; fdrv(1, 1) = -t482*tx-t487*tz; fdrv(1, 2) = -t489*tx+t487*ty;
    fdrv(1, 3) = -t169+t201-t217+t235+t238+t331-oy*t48*4.0+t2*t45*3.0-t4*t45*2.0+t9*t41-t4*t47-t10*t43*3.0-t8*t63*4.0+t11*t61-t8*t67*2.0-t12*t67+t2*t146+t223*tz-oy*t8*tx*6.0-oy*t11*tz*2.0-t4*t13*tx*3.0-t4*t15*ty;
    fdrv(1, 4) = t161*-2.0-t163*2.0+t207+t223+t227+t251+t326+t360+t404-t8*t43*2.0+t12*t41*3.0-t10*t63+t14*t61-t10*t67*3.0-t10*t70+t15*t264+t157*tz-t4*t13*ty*2.0;
    fdrv(1, 5) = t155+t212+t215-t242+t255+t352-oy*t45*4.0+t2*t46-t4*t46*2.0-t4*t48*3.0+t12*t57+t23*t49-t13*t63+t16*t61*4.0-t13*t67+t2*t147+t12*t266+t393*tx-oy*t13*tx*2.0-oy*t16*tz*6.0+t2*t9*ty+t2*t11*tz*3.0;
    fdrv(1, 6) = -tdx*(t108+t122+t165*2.0-t209+t234+t253+t284+t342-ox*t47*2.0+oy*t8*1.2E+1+t294*tz+oz*t8*tx*1.2E+1)+t501*tdz+tdy*(t86+t96-t97+t112+t114-t167+t210+t211-t254-t350-t355-oz*t44*4.0);
    fdrv(1, 7) = tdy*(t203+t233-t234-t249+t351+t417+ry*t2*4.0+ry*t4*4.0)+t495*tdz+tdx*(t86+t96-t97+t112+t114-t167+t210+t211-t254-t350-t355-oz*t44*4.0);
    fdrv(1, 8) = t501*tdx+t495*tdy+tdz*(-t90+t123+t159*2.0+t209-t228+t233-t253+t292-oy*t16*1.2E+1+t8*t52+t100*ty+ox*t16*tz*1.2E+1); fdrv(1, 9) = -tdx*(t58*6.0+t65*4.0-t171*3.0+t177*2.0+t180*2.0+t268+t303+t315+t408+t429)+t453*tdy*tx*2.0+t479*tdz*tx;
    fdrv(1, 10) = tdy*(t64-t65+t126+t136+t171+t174*3.0-t177*3.0-t180-t261-t268)-t478*tdx*ty+t479*tdz*ty; fdrv(1, 11) = tdz*(t64*4.0-t65-t66*6.0+t125+t136-t177-t180*3.0-t261+t2*t39+t19*t27)-t478*tdx*tz+t453*tdy*tz*2.0;
    fdrv(1, 12) = tdz*(t71+t75+t145+t182+t184+t279+t370+t377+t378+t383)+tdx*(t74+t138+t150+t186+t258+t275+t371+t376)+tdy*(t77+t142+t147+t188+t257+t272+t379+t382);
    fdrv(1, 13) = -tdz*(t141+t148+t280+t13*tx*2.0)-tdx*(t142+t149+t274+t11*tz*2.0)-ry*t321*tdy*2.0;
    fdrv(1, 14) = -tdx*(t75+t79-t143+t191+t193+t273+t362+t366+t373+t381)-tdy*(t45*-2.0+t73-t148+t187+t259+t363+t372+t412)-tdz*(t44*-2.0+t76-t151+t189+t258+t367+t380+t411); fdrv(1, 15) = tx*(t64-t65+t125+t135+t171+t174-t177-t180-t261-t268);
    fdrv(1, 16) = ty*(t64-t65+t125+t135+t171+t174-t177-t180-t261-t268); fdrv(1, 17) = tz*(t64-t65+t125+t135+t171+t174-t177-t180-t261-t268);
    fdrv(1, 18) = -t154+t194+rdx*t64-rdx*t66*2.0+rdx*t135+rdx*t174+t6*t23+t7*t23-t5*t63*4.0-t6*t63*3.0-t5*t67*2.0-t6*t67-t5*t70*2.0-t6*t70+t6*t127+t7*t127-oy*t5*tx*6.0-oy*t7*tx*4.0-rdz*t4*t17*3.0-rdz*t4*t19-rdy*t3*tx*4.0+t2*t5*tz*3.0;
    fdrv(1, 19) = t323+t325+t401-rdy*t58*2.0+rdy*t64-rdy*t65-rdy*t66*2.0+rdy*t171-rdy*t180+t2*t6*4.0+t2*t7*2.0+t4*t5*2.0+t4*t6*4.0+t6*t41*3.0-t5*t43*2.0-t6*t43*3.0+t7*t23*tz-rdz*t4*tx*ty*2.0;
    fdrv(1, 20) = t152+t199-rdz*t58*2.0-rdz*t65+rdz*t125-rdz*t177+t5*t32+t6*t32+t6*t56+t6*t57+t7*t56*2.0+t7*t61*4.0+t7*t260+t6*t266-oy*t5*tz*4.0-oy*t7*tz*6.0+rdx*t2*t19+rdx*t2*t21*3.0-rdy*t3*tz*4.0-t4*t5*tx*2.0-t4*t6*tx*2.0-t4*t7*tx*3.0;
    fdrv(1, 24) = t442*(t22+t38+t83+t85); fdrv(1, 25) = t321*t442*-2.0; fdrv(1, 26) = -t442*(t18-t40+t80+t82); fdrv(2, 0) = t485*tz-ty*(t460-t470); fdrv(2, 1) = t483*tz+tx*(t460-t470); fdrv(2, 2) = -t485*tx-t483*ty;
    fdrv(2, 3) = t162+t206+t222+t224-t252+t351-oz*t46*4.0-t2*t44*3.0-t2*t49*2.0+t3*t49+t13*t42*3.0+t8*t58*4.0-t9*t57-t9*t61+t16*t66+t35*t62+t35*t66+t2*t411-oz*t8*tx*6.0-oz*t9*ty*2.0+t3*t10*tx*3.0+t3*t14*tz;
    fdrv(2, 4) = -t155+t211+t213+t242-t255+t357-oz*t44*4.0-t2*t46*2.0+t3*t46*3.0-t2*t48-t15*t41*3.0+t10*t58-t12*t57*4.0-t12*t61*2.0+t10*t66-t16*t61+t3*t147+t205*tx-oz*t10*tx*2.0-oz*t12*ty*6.0-t2*t9*ty*3.0-t2*t11*tz;
    fdrv(2, 5) = t166*-2.0-t168*2.0+t204+t205+t230+t250+t326+t340+t405-t12*t41*2.0+t24*t45+t13*t58-t15*t57+t13*t62-t15*t61*3.0+t13*t270-t159*ty-t2*t9*tz*2.0;
    fdrv(2, 6) = t494*tdz+tdy*(t99+t101-t110-t111-t158+t163-t207+t208+t247-t327-t329+t335+t347-t2*t16*2.0)+tdx*(t95-t118+t160*2.0-t202+t220+t246+t301+t339+t416-ox*t49*2.0-oz*t8*1.2E+1+oy*t8*tx*1.2E+1);
    fdrv(2, 7) = -tdy*(t98+t124+t157*2.0+t202-t220+t244+t298+t337-oy*t48*2.0+oz*t12*1.2E+1+t283*tx+ox*t12*ty*1.2E+1)+tdx*(t99+t101-t110-t111-t158+t163-t207+t208+t247-t327-t329+t335+t347-t2*t16*2.0)-tdz*(-t88+t123+t159+t203+t282+t292+t300+t345+t353+t392+ox*t148+t10*t54);
    fdrv(2, 8) = tdz*(-t200+t214-t244+t246+t328+t416+rz*t2*4.0+rz*t3*4.0)+t494*tdx-tdy*(-t88+t123+t159+t203+t282+t292+t300+t345+t353+t392+ox*t148+t10*t54);
    fdrv(2, 9) = -tdx*(t59-t60*4.0+t63*6.0+t170*3.0-t172*2.0+t173-t175*2.0+t269+t306+t311)-t477*tdy*tx+t452*tdz*tx*2.0; fdrv(2, 10) = -tdy*(t59*4.0+t67*6.0-t172*3.0+t265+t304+t313+t407+t430+t2*t38+t21*t23)+t476*tdx*ty+t452*tdz*ty*2.0;
    fdrv(2, 11) = -tdz*(t59+t170+t173*3.0-t175*3.0+t265+t269+t306+t313+t407+t428)+t476*tdx*tz-t477*tdy*tz;
    fdrv(2, 12) = -tdy*(t71+t79-t144+t183+t192+t276+t364+t369+t374+t385)-tdx*(t47*-2.0+t72-t141+t185+t259+t365+t368+t412)-tdz*(t78-t140-t149+t190+t257+t375+t384+t410);
    fdrv(2, 13) = tdx*(t75+t79+t143+t191+t193+t273+t362+t366+t373+t381)+tdy*(t73+t139+t148+t187+t259+t278+t363+t372)+tdz*(t76+t137+t151+t189+t258+t275+t367+t380);
    fdrv(2, 14) = -tdy*(t138+t151+t277+t10*tx*2.0)-tdx*(t142+t149+t274+t9*ty*2.0)-rz*t320*tdz*2.0; fdrv(2, 15) = -t491*tx; fdrv(2, 16) = -t491*ty; fdrv(2, 17) = -t491*tz;
    fdrv(2, 18) = t153+t195-rdx*t59-rdx*t67*2.0+rdx*t131-rdx*t173+t6*t27+t7*t27+t5*t58*4.0+t5*t62*2.0+t7*t62+t7*t66+t7*t263+t5*t268-oz*t5*tx*6.0-oz*t6*tx*4.0+rdy*t3*t17*3.0+rdy*t3*t21-rdz*t4*tx*4.0-t2*t5*ty*3.0-t2*t6*ty*2.0-t2*t7*ty*2.0;
    fdrv(2, 19) = -t152+t197+rdy*t60-rdy*t63*2.0+rdy*t127+rdy*t175+t5*t31+t7*t31-t6*t56*2.0-t6*t57*4.0-t7*t56-t7*t57*3.0-t6*t61*2.0-t7*t61+t5*t128+t7*t128-oz*t5*ty*4.0-oz*t6*ty*6.0-rdx*t2*t19*3.0-rdx*t2*t21-rdz*t4*ty*4.0+t3*t6*tx*3.0;
    fdrv(2, 20) = t323+t324+t402-rdz*t59+rdz*t60-rdz*t63*2.0-rdz*t67*2.0-rdz*t170+rdz*t172+t2*t6*2.0+t3*t5*2.0+t2*t7*4.0+t3*t7*4.0-t6*t41*2.0-t7*t41*3.0+t7*t42*3.0+t5*t24*tz-rdx*t2*ty*tz*2.0; fdrv(2, 24) = -t442*(t20-t39+t81+t84);
    fdrv(2, 25) = t442*(t18+t40+t80+t82); fdrv(2, 26) = t320*t442*-2.0;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f19(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*ty, t3 = oy*tx, t4 = ox*tz, t5 = oz*tx, t6 = oy*tz, t7 = oz*ty, t8 = rdx*tx, t9 = rdy*ty, t10 = rdz*tz, t11 = tx*tx, t12 = tx*tx*tx, t13 = ty*ty, t14 = ty*ty*ty, t15 = tz*tz, t16 = tz*tz*tz, t17 = rx*tx*2.0, t18 = rx*tx*3.0;
    T t19 = rx*ty*2.0, t20 = ry*tx*2.0, t21 = rx*tz*2.0, t22 = ry*ty*2.0, t23 = rz*tx*2.0, t24 = ry*ty*3.0, t25 = ry*tz*2.0, t26 = rz*ty*2.0, t27 = rz*tz*2.0, t28 = rz*tz*3.0, t29 = rx*tx*ty, t30 = rx*tx*tz, t31 = ry*tx*ty, t32 = ry*ty*tz;
    T t33 = rz*tx*tz, t34 = rz*ty*tz, t35 = -t3, t36 = -t5, t37 = -t7, t38 = ox*t11, t39 = t2*ty, t40 = t3*tx, t41 = t4*tz, t42 = oy*t13, t43 = t5*tx, t44 = t6*tz, t45 = t7*ty, t46 = oz*t15, t47 = rx*t12, t48 = rx*t13, t49 = ry*t11, t50 = rx*t15;
    T t51 = rz*t11, t52 = ry*t14, t53 = ry*t15, t54 = rz*t13, t55 = rz*t16, t56 = t2*tx*2.0, t57 = t4*tx*2.0, t58 = t3*ty*2.0, t59 = t6*ty*2.0, t60 = t5*tz*2.0, t61 = t7*tz*2.0, t62 = t17*ty, t63 = t17*tz, t64 = t20*ty, t65 = t22*tz, t66 = t23*tz;
    T t67 = t26*tz, t83 = rx*t2*tx*6.0, t84 = t2*t20, t85 = t3*t19, t86 = ry*t2*tx*4.0, t87 = rx*t3*ty*4.0, t88 = rx*t4*tx*6.0, t90 = ry*t3*ty*6.0, t91 = t4*t23, t94 = rz*t4*tx*4.0, t95 = rx*t5*tz*4.0, t97 = ry*t6*ty*6.0, t100 = rz*t6*ty*4.0;
    T t101 = ry*t7*tz*4.0, t102 = rz*t5*tz*6.0, t103 = rz*t7*tz*6.0, t110 = rx*t11*3.0, t112 = ry*t13*3.0, t114 = rz*t15*3.0, t122 = t29*-2.0, t123 = t30*-2.0, t124 = t31*-2.0, t125 = t32*-2.0, t126 = t33*-2.0, t127 = t34*-2.0, t128 = t11+t13;
    T t129 = t11+t15, t130 = t13+t15, t140 = t13*t17, t141 = t13*t18, t142 = t15*t17, t143 = t11*t22, t144 = t15*t18, t145 = t11*t24, t146 = t15*t22, t147 = t11*t27, t148 = t15*t24, t149 = t11*t28, t150 = t13*t27, t151 = t13*t28;
    T t153 = rx*t2*tz*-2.0, t154 = ry*t3*tz*-2.0, t155 = rx*t5*tz*-2.0, t157 = rz*t5*ty*-2.0, t158 = rz*t6*ty*-2.0, t159 = ry*t7*tz*-2.0, t178 = t19+t20, t179 = t18+t24, t180 = t21+t23, t181 = t18+t28, t182 = t25+t26, t183 = t24+t28;
    T t184 = t8+t9+t10, t68 = ox*t47, t69 = rx*t39, t70 = rx*t41, t71 = ry*t40, t72 = oy*t52, t73 = ry*t44, t74 = rz*t43, t75 = rz*t45, t76 = oz*t55, t77 = t48*tx, t78 = t50*tx, t79 = t49*ty, t80 = t53*ty, t81 = t51*tz, t82 = t54*tz;
    T t104 = t39*3.0, t105 = t40*3.0, t106 = t41*3.0, t107 = t43*3.0, t108 = t44*3.0, t109 = t45*3.0, t111 = t47*4.0, t113 = t52*4.0, t115 = t55*4.0, t116 = -t56, t117 = -t57, t118 = -t58, t119 = -t59, t120 = -t60, t121 = -t61, t131 = rx*t38*3.0;
    T t135 = ry*t42*3.0, t139 = rz*t46*3.0, t152 = -t87, t156 = -t95, t160 = -t101, t164 = rz*t36*tx, t166 = rz*t37*ty, t169 = rx*t128, t170 = rx*t129, t171 = ry*t128, t172 = ry*t130, t173 = rz*t129, t174 = rz*t130, t175 = t2+t35, t176 = t4+t36;
    T t177 = t6+t37, t185 = t15+t128, t186 = t179*tx, t187 = t179*ty, t188 = t181*tx, t189 = t181*tz, t190 = t183*ty, t191 = t183*tz, t216 = t48+t50+t64+t66+t110, t217 = t49+t53+t62+t67+t112, t218 = t51+t54+t63+t65+t114, t132 = t69*3.0;
    T t133 = t70*3.0, t134 = t71*3.0, t136 = t73*3.0, t137 = t74*3.0, t138 = t75*3.0, t162 = -t135, t163 = -t73, t168 = -t139, t192 = -t186, t193 = -t187, t194 = -t188, t195 = -t189, t196 = -t190, t197 = -t191, t198 = ox*t8*t185;
    T t199 = rdx*t3*t185, t200 = rdy*t2*t185, t201 = rdx*t5*t185, t202 = oy*t9*t185, t203 = rdz*t4*t185, t204 = rdy*t7*t185, t205 = rdz*t6*t185, t206 = oz*t10*t185, t207 = t184*t185*tx, t208 = t184*t185*ty, t209 = t184*t185*tz;
    T t210 = t38+t41+t104+t118, t211 = t38+t39+t106+t120, t212 = t42+t44+t105+t116, t213 = t40+t42+t108+t121, t214 = t45+t46+t107+t117, t215 = t43+t46+t109+t119, t219 = t216*tdx*ty, t220 = t216*tdx*tz, t221 = t217*tdy*tx, t222 = t217*tdy*tz;
    T t223 = t218*tdz*tx, t224 = t218*tdz*ty, t237 = t47+t52+t77+t79+t115+t144+t147+t148+t150, t238 = t47+t55+t78+t81+t113+t141+t143+t146+t151, t239 = t52+t55+t80+t82+t111+t140+t142+t145+t149, t161 = -t134, t165 = -t137, t167 = -t138;
    T t225 = t31+t126+t169+t170+t192, t226 = t33+t124+t169+t170+t194, t227 = t29+t127+t171+t172+t193, t228 = t34+t122+t171+t172+t196, t229 = t30+t125+t173+t174+t195, t230 = t32+t123+t173+t174+t197, t240 = t239*tdx, t241 = t238*tdy;
    T t242 = t237*tdz, t231 = t225*tx, t232 = t226*tx, t233 = t227*ty, t234 = t228*ty, t235 = t229*tz, t236 = t230*tz, t243 = t70+t86+t91+t131+t132+t152+t158+t161+t162+t163, t244 = t69+t84+t94+t131+t133+t156+t159+t165+t166+t168;
    T t245 = t71+t85+t100+t135+t136+t155+t160+t164+t167+t168, t246 = t232+t234, t247 = t231+t236, t248 = t233+t235;
    
    fdrv(0, 0) = -t225*ty*tz+t226*ty*tz; fdrv(0, 1) = -t232*tz-t248*tz; fdrv(0, 2) = t231*ty+t248*ty; fdrv(0, 3) = t177*t216;
    fdrv(0, 4) = -t76-t7*t34*3.0-t5*t48*3.0+t6*t53-t7*t53*2.0+t33*t36+t26*t44+t36*t50+t6*t112+t5*t124+t71*tz+t85*tz+rx*t11*t36-ry*t7*t13*4.0;
    fdrv(0, 5) = t72+t3*t31+t3*t48+t3*t50*3.0+t6*t54*2.0+t24*t44+t3*t66+t37*t54+t7*t125+t155*ty+t164*ty+rx*t3*t11+rz*t6*t15*4.0-rz*t7*t15*3.0;
    fdrv(0, 6) = -tdy*(t154+rx*t46+rx*t109+t5*t18+t5*t27-rx*t6*ty*2.0+ry*t5*ty*4.0)+tdz*(t157+rx*t42+rx*t108+t3*t18+t3*t22-rx*t7*tz*2.0+rz*t3*tz*4.0)+t177*tdx*(t22+t27+rx*tx*6.0);
    fdrv(0, 7) = -tdy*(-t97+t103+ry*t45*1.2E+1+ry*t46*2.0-rz*t44*2.0+t5*t20+rx*t5*ty*6.0-rx*t3*tz*2.0)+t245*tdz-oz*t216*tdx+t177*t178*tdx;
    fdrv(0, 8) = tdz*(t97-t103-ry*t45*2.0+rz*t42*2.0+rz*t44*1.2E+1+t3*t23-rx*t5*ty*2.0+rx*t3*tz*6.0)+t245*tdy+oy*t216*tdx+t177*t180*tdx; fdrv(0, 9) = t177*tdx*(t11+t128+t129)-t215*tdy*tx+t213*tdz*tx;
    fdrv(0, 10) = tdy*(t6*t13*3.0-t7*t13*4.0+t6*t15-t7*t15*2.0-t43*ty*2.0+t40*tz)+t213*tdz*ty+t177*tdx*tx*ty*2.0; fdrv(0, 11) = tdz*(t6*t13*2.0+t6*t15*4.0-t7*t15*3.0+t13*t37+t40*tz*2.0+t36*tx*ty)-t215*tdy*tz+t177*tdx*tx*tz*2.0;
    fdrv(0, 13) = t220+t222+t242; fdrv(0, 14) = -t219-t224-t241; fdrv(0, 15) = t177*t185*tx; fdrv(0, 16) = t177*t185*ty; fdrv(0, 17) = t177*t185*tz; fdrv(0, 18) = t177*(rdx*t13+rdx*t15+t8*tx*3.0+t9*tx*2.0+t10*tx*2.0);
    fdrv(0, 19) = -t206+rdx*t36*t185+rdy*t37*t185+rdy*t177*t185+t8*t177*ty*2.0+t9*t177*ty*2.0+t10*t177*ty*2.0; fdrv(0, 20) = t199+t202+t205+rdz*t177*t185+t8*t177*tz*2.0+t9*t177*tz*2.0+t10*t177*tz*2.0; fdrv(0, 25) = t209; fdrv(0, 26) = -t208;
    fdrv(1, 0) = t234*tz+t247*tz; fdrv(1, 1) = t227*tx*tz-t228*tx*tz; fdrv(1, 2) = -t233*tx-t247*tx; fdrv(1, 3) = t76+t7*t34+t5*t48*2.0-t4*t50+t5*t50*2.0+t7*t53+t24*t43+t28*t43+t4*t126-t69*tz-rx*t4*t11*3.0+rx*t5*t11*4.0+ry*t7*t13-ry*t2*tx*tz*2.0;
    fdrv(1, 4) = -t176*t217; fdrv(1, 5) = -t68-t2*t29-t4*t30*3.0-t2*t49-t2*t53*3.0-t4*t51*2.0+t5*t51+t5*t54+t5*t63+t5*t65+t5*t114+t2*t127-ry*t2*t13-rz*t4*t15*4.0;
    fdrv(1, 6) = tdx*(-t88+t102+rx*t43*1.2E+1+rx*t46*2.0-rz*t41*2.0+t7*t19+ry*t5*ty*6.0-ry*t2*tz*2.0)-t244*tdz+oz*t217*tdy-t176*t178*tdy;
    fdrv(1, 7) = tdx*(t153+ry*t46+ry*t107+t7*t24+t7*t27-ry*t4*tx*2.0+rx*t5*ty*4.0)-tdz*(t157+ry*t38+ry*t106+t2*t17+t2*t24-ry*t5*tz*2.0+rz*t2*tz*4.0)-t176*tdy*(t17+t27+ry*ty*6.0);
    fdrv(1, 8) = -tdz*(t88-t102-rx*t43*2.0+rz*t38*2.0+rz*t41*1.2E+1+t2*t26-ry*t5*ty*2.0+ry*t2*tz*6.0)-t244*tdx-ox*t217*tdy-t176*t182*tdy; fdrv(1, 9) = -tdx*(t4*t11*3.0-t5*t11*4.0-t5*t13*2.0+t4*t15-t5*t15*2.0+t39*tz)-t211*tdz*tx-t176*tdy*tx*ty*2.0;
    fdrv(1, 10) = -t176*tdy*(t13+t128+t130)+t214*tdx*ty-t211*tdz*ty; fdrv(1, 11) = -tdz*(t4*t11*2.0+t4*t15*4.0-t5*t15*3.0+t11*t36+t13*t36+t39*tz*2.0)+t214*tdx*tz-t176*tdy*ty*tz*2.0; fdrv(1, 12) = -t220-t222-t242; fdrv(1, 14) = t221+t223+t240;
    fdrv(1, 15) = -t176*t185*tx; fdrv(1, 16) = -t176*t185*ty; fdrv(1, 17) = -t176*t185*tz; fdrv(1, 18) = t201+t204+t206-rdx*t176*t185-t8*t176*tx*2.0-t9*t176*tx*2.0-t10*t176*tx*2.0; fdrv(1, 19) = -t176*(rdy*t11+rdy*t15+t8*ty*2.0+t9*ty*3.0+t10*ty*2.0);
    fdrv(1, 20) = -t198-t200-t203-rdz*t176*t185-t8*t176*tz*2.0-t9*t176*tz*2.0-t10*t176*tz*2.0; fdrv(1, 24) = -t209; fdrv(1, 26) = t207; fdrv(2, 0) = -t236*ty-t246*ty; fdrv(2, 1) = t235*tx+t246*tx; fdrv(2, 2) = -t229*tx*ty+t230*tx*ty;
    fdrv(2, 3) = -t72-t3*t31*3.0-t3*t33*3.0-t6*t32+t2*t48-t3*t48*2.0+t2*t50-t3*t50*2.0+t20*t39-t6*t54+t2*t66+t2*t110-rx*t3*t11*4.0-rz*t6*t15;
    fdrv(2, 4) = t68+t4*t30+t2*t49*2.0+t2*t53*2.0+t4*t51+t18*t39+t28*t39+t35*t49+t35*t53+t3*t122+t3*t127+ry*t2*t13*4.0-ry*t3*t13*3.0+rz*t4*t15; fdrv(2, 5) = t175*t218;
    fdrv(2, 6) = -tdx*(-t83+t90+rx*t40*1.2E+1+rx*t42*2.0-ry*t39*2.0+t6*t21-rz*t2*tz*2.0+rz*t3*tz*6.0)+t243*tdy-oy*t218*tdz+t175*t180*tdz;
    fdrv(2, 7) = tdy*(t83-t90-rx*t40*2.0+ry*t38*2.0+ry*t39*1.2E+1+t4*t25+rz*t2*tz*6.0-rz*t3*tz*2.0)+t243*tdx+ox*t218*tdz+t175*t182*tdz;
    fdrv(2, 8) = -tdx*(t153+rz*t42+rz*t105+t6*t22+t6*t28-rz*t2*tx*2.0+rx*t3*tz*4.0)+tdy*(t154+rz*t38+rz*t104+t4*t17+t4*t28+ry*t2*tz*4.0-rz*t3*ty*2.0)+t175*tdz*(t17+t22+rz*tz*6.0);
    fdrv(2, 9) = tdx*(t2*t11*3.0-t3*t11*4.0+t2*t13-t3*t13*2.0+t2*t15-t3*t15*2.0)+t210*tdy*tx+t175*tdz*tx*tz*2.0; fdrv(2, 10) = tdy*(t2*t11*2.0+t2*t13*4.0-t3*t13*3.0+t2*t15*2.0+t11*t35+t15*t35)-t212*tdx*ty+t175*tdz*ty*tz*2.0;
    fdrv(2, 11) = -t212*tdx*tz+t210*tdy*tz+t175*tdz*(t15*2.0+t185); fdrv(2, 12) = t219+t224+t241; fdrv(2, 13) = -t221-t223-t240; fdrv(2, 15) = t175*t185*tx; fdrv(2, 16) = t175*t185*ty; fdrv(2, 17) = t175*t185*tz;
    fdrv(2, 18) = -t202-t205+rdx*t35*t185+rdx*t175*t185+t8*t175*tx*2.0+t9*t175*tx*2.0+t10*t175*tx*2.0; fdrv(2, 19) = t198+t200+t203+rdy*t175*t185+t8*t175*ty*2.0+t9*t175*ty*2.0+t10*t175*ty*2.0;
    fdrv(2, 20) = t175*(rdz*t11+rdz*t13+t8*tz*2.0+t9*tz*2.0+t10*tz*3.0); fdrv(2, 24) = t208; fdrv(2, 25) = -t207;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f20(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = ox*ty, t4 = oy*tx, t5 = ox*tz, t6 = oy*ty, t7 = oz*tx, t8 = oy*tz, t9 = oz*ty, t10 = oz*tz, t11 = rdx*tx, t12 = rdy*ty, t13 = rdz*tz, t14 = rx*tx, t15 = ry*ty, t16 = rz*tz, t17 = tx*tx, t18 = tx*tx*tx, t20 = ty*ty;
    T t21 = ty*ty*ty, t23 = tz*tz, t24 = tz*tz*tz, t41 = rx*ty*tz, t42 = ry*tx*tz, t43 = rz*tx*ty, t19 = t17*t17, t22 = t20*t20, t25 = t23*t23, t26 = t3*2.0, t27 = t4*2.0, t28 = t5*2.0, t29 = t7*2.0, t30 = t8*2.0, t31 = t9*2.0, t32 = t2*ty;
    T t33 = t2*tz, t34 = t4*ty, t35 = t6*tz, t36 = t7*tz, t37 = t9*tz, t38 = t14*ty, t39 = t14*tz, t40 = t15*tx, t44 = t15*tz, t45 = t16*tx, t46 = t16*ty, t47 = -t4, t49 = -t7, t51 = -t9, t53 = t2*t17, t54 = t3*ty, t55 = t4*tx, t56 = t3*t20;
    T t57 = t4*t17, t58 = t5*tz, t59 = t7*tx, t60 = t5*t23, t61 = t6*t20, t62 = t7*t17, t63 = t8*tz, t64 = t9*ty, t65 = t8*t23, t66 = t9*t20, t67 = t10*t23, t68 = t14*t17, t69 = rx*t20, t70 = ry*t17, t71 = rx*t23, t72 = rz*t17, t73 = t15*t20;
    T t74 = ry*t23, t75 = rz*t20, t76 = t16*t23, t83 = t2*t20, t84 = t2*t23, t86 = t3*t23, t88 = t4*t23, t90 = t7*t20, t92 = t6*t23, t95 = t14*t20, t96 = t14*t23, t97 = t15*t17, t104 = t15*t23, t105 = t16*t17, t106 = t16*t20, t122 = t14*tx*3.0;
    T t124 = t15*ty*3.0, t126 = t16*tz*3.0, t134 = -t42, t135 = -t43, t136 = t17+t20, t137 = t17+t23, t138 = t20+t23, t140 = t2*t14*tx*4.0, t154 = t6*t15*ty*4.0, t162 = t10*t16*tz*4.0, t173 = t4*t20*3.0, t187 = t7*t23*3.0, t191 = t9*t23*3.0;
    T t205 = t3*t21*-2.0, t206 = t4*t18*-2.0, t209 = t5*t24*-2.0, t210 = t7*t18*-2.0, t213 = t8*t24*-2.0, t214 = t9*t21*-2.0, t217 = t14*t18*1.0E+1, t218 = t15*t21*1.0E+1, t219 = t16*t24*1.0E+1, t220 = t2+t6, t221 = t2+t10, t222 = t6+t10;
    T t223 = t14+t15, t224 = t14+t16, t225 = t15+t16, t227 = t2*t14*tx*8.0, t238 = t6*t15*ty*8.0, t249 = t10*t16*tz*8.0, t254 = t4*t20*-2.0, t264 = t7*t23*-2.0, t266 = t9*t23*-2.0, t268 = t14*t21*8.0, t269 = t14*t24*8.0, t270 = t15*t18*8.0;
    T t277 = t15*t24*8.0, t278 = t16*t18*8.0, t279 = t16*t21*8.0, t312 = t7*t15*ty*4.0, t324 = t7*t15*ty*6.0, t326 = t3*t16*tz*4.0, t330 = t4*t16*tz*4.0, t336 = t3*t16*tz*6.0, t338 = t4*t16*tz*6.0, t410 = t3*t41*8.0, t422 = t4*t42*8.0;
    T t429 = t7*t15*ty*2.4E+1, t436 = t7*t43*8.0, t439 = t3*t16*tz*2.4E+1, t441 = t4*t16*tz*2.4E+1, t484 = t11+t12+t13, t48 = -t27, t50 = -t29, t52 = -t31, t77 = t38*2.0, t78 = t39*2.0, t79 = t40*2.0, t80 = t44*2.0, t81 = t45*2.0, t82 = t46*2.0;
    T t85 = t34*tx, t87 = t54*tz, t89 = t55*tz, t91 = t59*ty, t93 = t36*tx, t94 = t37*ty, t98 = t71*ty, t99 = t69*tz, t100 = t74*tx, t101 = t70*tz, t102 = t75*tx, t103 = t72*ty, t107 = t32*tz*2.0, t108 = t27*ty*tz, t109 = t29*ty*tz;
    T t110 = t21*t26, t111 = t18*t27, t112 = t56*4.0, t113 = t57*4.0, t114 = t24*t28, t115 = t18*t29, t116 = t60*4.0, t117 = t62*4.0, t118 = t24*t30, t119 = t21*t31, t120 = t65*4.0, t121 = t66*4.0, t123 = t68*4.0, t125 = t73*4.0, t127 = t76*4.0;
    T t139 = t14*t53*2.0, t141 = ry*t53*2.0, t142 = t14*t27*tx, t143 = t15*t26*ty, t144 = rz*t53*2.0, t145 = rx*t61*2.0, t146 = t14*t29*tx, t147 = t28*t74, t148 = t26*t75, t149 = t30*t71, t150 = t27*t72, t151 = t31*t69, t152 = t29*t70;
    T t153 = t15*t61*2.0, t155 = t16*t28*tz, t156 = rz*t61*2.0, t157 = rx*t67*2.0, t158 = t15*t31*ty, t159 = t16*t30*tz, t160 = ry*t67*2.0, t161 = t16*t67*2.0, t163 = t83*2.0, t164 = t32*tx*2.0, t165 = t83*3.0, t166 = t32*tx*3.0, t167 = t84*2.0;
    T t168 = t33*tx*2.0, t171 = t84*3.0, t172 = t33*tx*3.0, t175 = t86*4.0, t177 = t88*4.0, t179 = t90*4.0, t181 = t92*2.0, t182 = t35*ty*2.0, t185 = t92*3.0, t186 = t35*ty*3.0, t193 = t95*2.0, t194 = t95*3.0, t195 = t96*2.0, t196 = t97*2.0;
    T t197 = t96*3.0, t198 = t97*3.0, t199 = t104*2.0, t200 = t105*2.0, t201 = t104*3.0, t202 = t105*3.0, t203 = t106*2.0, t204 = t106*3.0, t228 = rx*t56*8.0, t229 = t14*t55*8.0, t230 = rx*t60*8.0, t231 = t15*t54*8.0, t232 = ry*t57*8.0;
    T t233 = t14*t59*8.0, t235 = ry*t60*8.0, t236 = rz*t56*8.0, t237 = rx*t65*8.0, t239 = rz*t57*8.0, t240 = rx*t66*8.0, t241 = ry*t62*8.0, t242 = t16*t58*8.0, t243 = ry*t65*8.0, t244 = t15*t64*8.0, t245 = rz*t62*8.0, t246 = t16*t63*8.0;
    T t247 = rz*t66*8.0, t280 = t2*t69*4.0, t281 = t14*t32*4.0, t282 = t2*t71*4.0, t283 = t14*t33*4.0, t284 = t15*t32*6.0, t285 = t2*t40*6.0, t286 = t4*t69*6.0, t287 = t14*t34*6.0, t288 = t2*t74*4.0, t289 = ry*t33*tx*4.0, t290 = t2*t75*4.0;
    T t291 = rz*t32*tx*4.0, t292 = t4*t71*4.0, t293 = t4*t39*4.0, t294 = t15*t34*4.0, t295 = t4*t40*4.0, t296 = t7*t69*4.0, t297 = t7*t38*4.0, t298 = t2*t74*6.0, t299 = ry*t33*tx*6.0, t300 = t2*t75*6.0, t301 = rz*t32*tx*6.0, t302 = t4*t71*6.0;
    T t303 = t4*t39*6.0, t304 = t7*t69*6.0, t305 = t7*t38*6.0, t306 = t3*t74*4.0, t307 = t3*t44*4.0, t308 = t6*t71*4.0, t309 = rx*t35*ty*4.0, t310 = t4*t75*4.0, t313 = t7*t40*4.0, t314 = t3*t74*6.0, t315 = t3*t44*6.0, t316 = t16*t33*6.0;
    T t317 = t2*t45*6.0, t318 = t6*t71*6.0, t319 = rx*t35*ty*6.0, t320 = t4*t75*6.0, t322 = t7*t71*6.0, t323 = t14*t36*6.0, t325 = t7*t40*6.0, t327 = t3*t46*4.0, t328 = t6*t74*4.0, t329 = t15*t35*4.0, t331 = t4*t45*4.0, t332 = t9*t71*4.0;
    T t334 = t7*t74*4.0, t337 = t3*t46*6.0, t339 = t4*t45*6.0, t340 = t9*t71*6.0, t342 = t7*t74*6.0, t344 = t16*t36*4.0, t345 = t7*t45*4.0, t346 = t16*t35*6.0, t347 = t6*t46*6.0, t348 = t9*t74*6.0, t349 = t15*t37*6.0, t350 = t16*t37*4.0;
    T t351 = t9*t46*4.0, t352 = t23*t32*4.0, t353 = t20*t33*4.0, t354 = t23*t34*4.0, t356 = t20*t36*4.0, t358 = rx*t32*tz*8.0, t359 = rx*t32*tz*1.6E+1, t360 = t15*t33*8.0, t361 = rx*t34*tz*8.0, t362 = t15*t33*1.2E+1, t363 = rx*t34*tz*1.2E+1;
    T t364 = t16*t32*8.0, t365 = t4*t44*8.0, t366 = rx*t36*ty*8.0, t367 = t16*t32*1.2E+1, t368 = rx*t36*ty*1.2E+1, t369 = t4*t44*1.6E+1, t370 = t16*t34*8.0, t371 = t15*t36*8.0, t372 = t16*t34*1.2E+1, t373 = t15*t36*1.2E+1, t374 = t7*t46*8.0;
    T t375 = t7*t46*1.6E+1, t388 = t83*tx*6.0, t389 = t84*tx*6.0, t390 = t20*t55*6.0, t391 = t92*ty*6.0, t392 = t23*t59*6.0, t393 = t23*t64*6.0, t394 = t23*t69*2.0, t395 = t23*t70*2.0, t396 = t20*t72*2.0, t397 = t2*t69*1.2E+1;
    T t398 = t14*t32*1.2E+1, t399 = t2*t71*1.2E+1, t400 = t14*t33*1.2E+1, t401 = t15*t32*1.2E+1, t402 = t2*t40*1.2E+1, t403 = t4*t69*1.2E+1, t404 = t14*t34*1.2E+1, t409 = t3*t71*8.0, t411 = t15*t34*1.2E+1, t412 = t4*t40*1.2E+1;
    T t413 = t4*t71*2.4E+1, t414 = t4*t39*2.4E+1, t415 = t7*t69*2.4E+1, t416 = t7*t38*2.4E+1, t421 = t4*t74*8.0, t423 = t16*t33*1.2E+1, t424 = t2*t45*1.2E+1, t425 = t7*t71*1.2E+1, t426 = t14*t36*1.2E+1, t427 = t3*t74*2.4E+1, t428 = t3*t44*2.4E+1;
    T t430 = t7*t40*2.4E+1, t435 = t7*t75*8.0, t437 = t6*t74*1.2E+1, t438 = t15*t35*1.2E+1, t440 = t3*t46*2.4E+1, t442 = t4*t45*2.4E+1, t443 = t16*t35*1.2E+1, t444 = t6*t46*1.2E+1, t445 = t9*t74*1.2E+1, t446 = t15*t37*1.2E+1;
    T t447 = t16*t36*1.2E+1, t448 = t7*t45*1.2E+1, t449 = t16*t37*1.2E+1, t450 = t9*t46*1.2E+1, t466 = rx*t136, t467 = rx*t137, t468 = ry*t136, t469 = ry*t138, t470 = rz*t137, t471 = rz*t138, t475 = t32*t39*4.0, t476 = t15*t84*4.0;
    T t477 = t34*t71*4.0, t478 = t16*t83*4.0, t479 = t34*t44*4.0, t480 = t36*t69*4.0, t481 = t34*t45*4.0, t482 = t36*t40*4.0, t483 = t36*t46*4.0, t485 = t223*tx*tz, t486 = t224*tx*ty, t487 = t223*ty*tz, t488 = t225*tx*ty, t489 = t224*ty*tz;
    T t490 = t225*tx*tz, t500 = t23*t54*1.2E+1, t501 = t23*t55*1.2E+1, t502 = t20*t59*1.2E+1, t511 = -t410, t516 = -t422, t520 = -t429, t522 = -t436, t524 = -t439, t525 = -t441, t530 = t23+t136, t534 = t14*t83*6.0, t535 = t14*t84*6.0;
    T t536 = t32*t40*6.0, t537 = t4*t95*6.0, t538 = t34*t40*6.0, t539 = t33*t45*6.0, t540 = t7*t96*6.0, t541 = t15*t92*6.0, t542 = t35*t46*6.0, t543 = t9*t104*6.0, t544 = t36*t45*6.0, t545 = t37*t46*6.0, t552 = t41+t134, t553 = t41+t135;
    T t554 = t42+t135, t567 = t136*t223, t568 = t137*t224, t569 = t138*t225, t174 = t85*3.0, t176 = t87*4.0, t178 = t89*4.0, t180 = t91*4.0, t188 = t93*3.0, t192 = t94*3.0, t207 = -t112, t208 = -t113, t211 = -t116, t212 = -t117, t215 = -t120;
    T t216 = -t121, t226 = -t139, t234 = -t153, t248 = -t161, t250 = -t163, t251 = -t164, t252 = -t167, t253 = -t168, t255 = t85*-2.0, t256 = -t175, t258 = -t177, t260 = -t179, t262 = -t181, t263 = -t182, t265 = t93*-2.0, t267 = t94*-2.0;
    T t271 = -t98, t272 = -t99, t273 = -t100, t274 = -t101, t275 = -t102, t276 = -t103, t311 = rz*t85*4.0, t321 = rz*t85*6.0, t333 = rx*t94*4.0, t335 = ry*t93*4.0, t341 = rx*t94*6.0, t343 = ry*t93*6.0, t355 = t85*tz*4.0, t357 = t93*ty*4.0;
    T t376 = -t228, t377 = -t230, t378 = -t232, t379 = -t235, t380 = -t236, t381 = -t237, t382 = -t239, t383 = -t240, t384 = -t241, t385 = -t243, t386 = -t245, t387 = -t247, t405 = -t288, t406 = -t289, t407 = -t290, t408 = -t291, t417 = -t308;
    T t418 = -t309, t419 = -t310, t431 = -t332, t433 = -t334, t451 = -t359, t452 = -t360, t453 = -t361, t454 = -t362, t455 = -t363, t456 = -t364, t457 = -t366, t458 = -t367, t459 = -t368, t460 = -t369, t461 = -t370, t462 = -t371, t463 = -t372;
    T t464 = -t373, t465 = -t375, t472 = t3+t48, t473 = t5+t50, t474 = t8+t52, t491 = t467*ty, t492 = t466*tz, t493 = t469*tx, t494 = t468*tz, t495 = t471*tx, t496 = t470*ty, t497 = -t388, t498 = -t389, t499 = -t390, t503 = -t391, t504 = -t392;
    T t505 = -t393, t506 = -t398, t507 = -t400, t508 = -t401, t509 = -t404, t510 = -t409, t512 = -t411, t513 = -t414, t514 = -t416, t515 = -t421, t517 = -t423, t518 = -t426, t519 = -t428, t521 = -t435, t523 = -t438, t526 = -t443, t527 = -t446;
    T t528 = -t447, t529 = -t449, t546 = -t475, t547 = -t479, t548 = -t483, t549 = -t500, t550 = -t501, t551 = -t502, t555 = t552*tx, t556 = t553*tx, t557 = t552*ty, t558 = t554*ty, t559 = t553*tz, t560 = t554*tz, t561 = t40+t467, t562 = t45+t466;
    T t563 = t38+t469, t564 = t46+t468, t565 = t39+t471, t566 = t44+t470, t588 = t484*t530*tx*ty*2.0, t589 = t484*t530*tx*tz*2.0, t590 = t484*t530*ty*tz*2.0, t594 = t69+t71+t79+t81+t122, t595 = t70+t74+t77+t82+t124, t596 = t72+t75+t78+t80+t126;
    T t612 = t68+t73+t95+t97+t127+t197+t200+t201+t203, t613 = t68+t76+t96+t105+t125+t194+t196+t199+t204, t614 = t73+t76+t104+t106+t123+t193+t195+t198+t202, t257 = -t176, t259 = -t178, t261 = -t180, t420 = -t311, t432 = -t333, t434 = -t335;
    T t570 = -t557, t571 = -t559, t572 = -t560, t573 = t561*tx, t574 = t562*tx, t575 = t563*tx, t576 = t561*ty, t577 = t563*ty, t578 = t565*tx, t579 = t562*tz, t580 = t564*ty, t581 = t566*ty, t582 = t564*tz, t583 = t565*tz, t584 = t566*tz;
    T t591 = -t588, t592 = -t589, t593 = -t590, t597 = t594*tdx*ty*tz*2.0, t598 = t595*tdy*tx*tz*2.0, t599 = t596*tdz*tx*ty*2.0, t603 = t56+t86+t109+t166+t208+t254+t258, t604 = t57+t88+t109+t173+t207+t251+t256;
    T t605 = t60+t87+t108+t172+t212+t260+t264, t609 = t53+t61+t83+t85+t171+t185+t265+t267, t610 = t53+t67+t84+t93+t165+t192+t255+t262, t611 = t61+t67+t92+t94+t174+t188+t250+t252, t630 = t613*tdy*tx*2.0, t631 = t612*tdz*tx*2.0;
    T t632 = t614*tdx*ty*2.0, t633 = t612*tdz*ty*2.0, t634 = t614*tdx*tz*2.0, t635 = t613*tdy*tz*2.0, t666 = t150+t152+t293+t297+t320+t324+t338+t342+t365+t374+t379+t380+t406+t408+t451+t519+t524;
    T t667 = t148+t151+t301+t305+t307+t312+t336+t340+t358+t374+t381+t382+t418+t419+t460+t513+t525, t668 = t147+t149+t299+t303+t315+t319+t326+t330+t358+t365+t383+t384+t431+t433+t465+t514+t520;
    T t672 = t155+t157+t227+t231+t282+t306+t317+t323+t337+t341+t344+t371+t378+t397+t402+t417+t461+t509+t512+t515, t674 = t159+t160+t229+t238+t292+t328+t339+t343+t347+t349+t350+t366+t376+t403+t405+t412+t456+t506+t508+t510;
    T t676 = t156+t158+t233+t249+t296+t321+t325+t329+t346+t348+t351+t361+t377+t407+t425+t448+t452+t507+t511+t517, t600 = -t597, t601 = -t598, t602 = -t599, t606 = t62+t90+t108+t187+t211+t253+t257, t607 = t65+t89+t107+t186+t216+t261+t266;
    T t608 = t66+t91+t107+t191+t215+t259+t263, t615 = t276+t487+t496+t555+t582, t616 = t272+t490+t492+t558+t578, t617 = t274+t489+t494+t556+t581, t624 = t273+t486+t493+t571+t576, t625 = t275+t485+t495+t570+t579, t626 = t271+t488+t491+t572+t575;
    T t627 = t105+t106+t567+t574+t580, t628 = t97+t104+t568+t573+t584, t629 = t95+t96+t569+t577+t583, t636 = -t630, t637 = -t631, t638 = -t632, t639 = -t633, t640 = -t634, t641 = -t635;
    T t673 = t143+t145+t227+t242+t280+t285+t287+t294+t314+t318+t327+t370+t386+t399+t424+t432+t462+t518+t521+t528, t675 = t141+t142+t238+t246+t281+t284+t286+t295+t298+t302+t331+t364+t387+t434+t437+t444+t457+t522+t527+t529;
    T t677 = t144+t146+t244+t249+t283+t300+t304+t313+t316+t322+t345+t360+t385+t420+t445+t450+t453+t516+t523+t526, t618 = t615*tx, t619 = t617*tx, t620 = t615*ty, t621 = t616*ty, t622 = t616*tz, t623 = t617*tz, t642 = t624*tx, t643 = t625*tx;
    T t644 = t625*ty, t645 = t626*ty, t646 = t624*tz, t647 = t626*tz, t649 = t627*tx, t650 = t628*tx, t651 = t627*ty, t652 = t629*ty, t653 = t628*tz, t654 = t629*tz, t669 = t600+t639+t641, t670 = t601+t637+t640, t671 = t602+t636+t638;
    T t648 = -t621, t655 = -t646, t656 = -t647, t658 = t622+t649, t659 = t620+t653, t660 = t623+t651, t662 = t642+t652, t663 = t645+t650, t664 = t643+t654, t657 = t619+t648, t661 = t618+t656, t665 = t644+t655;
    
    fdrv(0, 0) = t660*ty+t659*tz;
    fdrv(0, 1) = -t660*tx-t665*tz; fdrv(0, 2) = -t659*tx+t665*ty;
    fdrv(0, 3) = t234+t248+t476-t477+t478-t480+t534+t535-t538-t544+rx*t110+rx*t114-t36*t40*6.0-t16*t61*2.0-t34*t45*6.0-t6*t76*2.0-t37*t46*2.0-t14*t85*8.0-t14*t93*8.0-t15*t92*2.0+t54*t71*4.0+t2*t125+t2*t127+t15*t267-rx*t4*t21*4.0-rx*t7*t24*4.0-ry*t9*t24*2.0;
    fdrv(0, 4) = t536-t537+t548+ry*t114-t14*t57*2.0+t15*t56*1.0E+1-t15*t57*4.0+t16*t56*8.0-t16*t57*2.0-t36*t38*4.0-t4*t73*8.0+t32*t45*4.0+t3*t76*8.0-t4*t76*2.0+t15*t86*1.2E+1+t2*t100*2.0-t14*t88*2.0-t15*t88*4.0+t32*t71*8.0-t36*t70*2.0-t4*t106*6.0+t281*tx+rx*t2*t21*8.0-ry*t7*t24*2.0-t15*t36*ty*6.0;
    fdrv(0, 5) = t539-t540+t547+rz*t110+t33*t40*4.0-t34*t39*4.0-t14*t62*2.0+t16*t60*1.0E+1-t15*t62*2.0-t16*t62*4.0-t7*t73*2.0-t7*t76*8.0+t44*t54*8.0+t33*t69*8.0+t2*t102*2.0-t14*t90*2.0-t16*t90*4.0-t34*t72*2.0-t7*t104*6.0+t283*tx+rx*t2*t24*8.0+ry*t3*t24*8.0-rz*t4*t21*2.0+t3*t46*tz*1.2E+1-t16*t34*tz*6.0;
    fdrv(0, 6) = -t674*tdy-t676*tdz-tdx*(-t306+t308-t327+t333+t372+t373-t397-t399+t411+t447+rx*t61*4.0+rx*t67*4.0+t14*t34*2.4E+1+t14*t36*2.4E+1-t15*t54*4.0-t16*t58*4.0);
    fdrv(0, 7) = -t674*tdx-t666*tdz+tdy*(t140+t242-t344+t402+t427+t440+t463+t464+t509-ry*t57*4.0+t2*t45*4.0-t15*t34*2.4E+1-t14*t36*4.0+t15*t54*4.0E+1+t2*t69*2.4E+1+t2*t71*8.0-t4*t74*4.0);
    fdrv(0, 8) = -t676*tdx-t666*tdy+tdz*(t140+t231-t294+t424+t427+t440+t463+t464+t518-rz*t62*4.0+t2*t40*4.0-t14*t34*4.0-t16*t36*2.4E+1+t2*t69*8.0+t2*t71*2.4E+1+t16*t58*4.0E+1-t7*t75*4.0);
    fdrv(0, 9) = -tdx*(t205+t209+t354+t356+t497+t498+t4*t21*4.0+t7*t24*4.0+t17*t34*8.0+t17*t36*8.0-t23*t54*4.0)-t604*tdy*tx*2.0-t606*tdz*tx*2.0;
    fdrv(0, 10) = -tdy*(t209+t354+t497+t549-t3*t21*1.0E+1+t4*t21*8.0+t17*t34*4.0+t24*t29+t20*t36*6.0-t84*tx*2.0+t17*t29*tz)-t611*tdx*ty*2.0-t606*tdz*ty*2.0;
    fdrv(0, 11) = -tdz*(t205+t356+t498+t549-t5*t24*1.0E+1+t7*t24*8.0+t21*t27+t17*t36*4.0+t23*t34*6.0-t83*tx*2.0+t17*t27*ty)-t611*tdx*tz*2.0-t604*tdy*tz*2.0;
    fdrv(0, 12) = tdy*(t218+t268+t279+t395+ry*t25*2.0+t17*t38*4.0+t23*t38*8.0+t17*t46*4.0+t23*t46*8.0+t97*ty*6.0+t104*ty*1.2E+1)+tdz*(t219+t269+t277+t396+rz*t22*2.0+t17*t39*4.0+t20*t39*8.0+t17*t44*4.0+t20*t44*8.0+t105*tz*6.0+t106*tz*1.2E+1)+t138*t594*tdx*2.0;
    fdrv(0, 13) = t671; fdrv(0, 14) = t670; fdrv(0, 15) = t530*tx*(t34+t36-t54-t58)*-2.0; fdrv(0, 16) = t530*ty*(t34+t36-t54-t58)*-2.0; fdrv(0, 17) = t530*tz*(t34+t36-t54-t58)*-2.0;
    fdrv(0, 18) = rdx*t530*(t34+t36-t54-t58)*-2.0-t11*tx*(t34+t36-t54-t58)*4.0-t12*tx*(t34+t36-t54-t58)*4.0-t13*tx*(t34+t36-t54-t58)*4.0-t11*t222*t530*2.0-t12*t222*t530*2.0-t13*t222*t530*2.0;
    fdrv(0, 19) = rdy*t530*(t34+t36-t54-t58)*-2.0-t11*ty*(t34+t36-t54-t58)*4.0-t12*ty*(t34+t36-t54-t58)*4.0-t13*ty*(t34+t36-t54-t58)*4.0-t11*t530*(t4-t26)*2.0-t12*t530*(t4-t26)*2.0-t13*t530*(t4-t26)*2.0;
    fdrv(0, 20) = rdz*t530*(t34+t36-t54-t58)*-2.0-t11*tz*(t34+t36-t54-t58)*4.0-t12*tz*(t34+t36-t54-t58)*4.0-t13*tz*(t34+t36-t54-t58)*4.0-t11*t530*(t7-t28)*2.0-t12*t530*(t7-t28)*2.0-t13*t530*(t7-t28)*2.0; fdrv(0, 24) = t138*t484*t530*2.0;
    fdrv(0, 25) = t591; fdrv(0, 26) = t592; fdrv(1, 0) = -t658*ty-t661*tz; fdrv(1, 1) = t658*tx+t664*tz; fdrv(1, 2) = t661*tx-t664*ty;
    fdrv(1, 3) = -t536+t537+t548+rx*t118+t14*t57*1.0E+1-t15*t56*2.0+t15*t57*8.0-t16*t56*2.0+t16*t57*8.0-t36*t38*6.0-t32*t45*6.0-t3*t76*2.0+t4*t76*8.0-t15*t86*2.0+t14*t88*1.2E+1+t15*t88*8.0-t32*t71*4.0+t6*t98*2.0-t37*t69*2.0+t4*t106*4.0+t4*t125-rx*t2*t21*4.0-rx*t9*t24*2.0-t14*t32*tx*8.0-t15*t36*ty*4.0;
    fdrv(1, 4) = t226+t248-t476+t477+t481-t482-t534+t538+t541-t545+ry*t111+ry*t118-t15*t53*4.0-t16*t53*2.0-t2*t73*8.0-t2*t76*2.0-t36*t45*2.0-t14*t84*2.0+t14*t85*4.0-t16*t83*6.0-t36*t69*6.0-t15*t94*8.0+t55*t74*4.0+t6*t127+t14*t265-rx*t7*t24*2.0-ry*t9*t24*4.0;
    fdrv(1, 5) = t542-t543+t546+rz*t111-t32*t44*4.0-t15*t66*2.0+t16*t65*1.0E+1-t16*t66*4.0-t9*t76*8.0+t39*t55*8.0-t38*t59*2.0+t4*t99*4.0-t32*t72*2.0-t7*t98*6.0+t27*t102+t329*ty+rx*t4*t24*8.0-rx*t7*t21*2.0+ry*t6*t24*8.0-rz*t2*t21*2.0-t7*t40*ty*2.0-t7*t45*ty*4.0+t4*t40*tz*8.0-t16*t32*tz*6.0+t4*t45*tz*1.2E+1;
    fdrv(1, 6) = -t672*tdy-t667*tdz+tdx*(t154+t246-t350+t403+t413+t442+t458+t459+t508-rx*t56*4.0+t4*t40*2.4E+1-t14*t32*2.4E+1+t6*t46*4.0-t15*t37*4.0+t14*t55*4.0E+1-t3*t71*4.0+t6*t74*8.0);
    fdrv(1, 7) = -t672*tdx-t677*tdz-tdy*(t288-t292-t331+t335+t367+t368+t398-t412-t437+t449+ry*t53*4.0+ry*t67*4.0+t15*t32*2.4E+1+t15*t37*2.4E+1-t14*t55*4.0-t16*t63*4.0);
    fdrv(1, 8) = -t667*tdx-t677*tdy+tdz*(t154+t229-t281+t413+t442+t444+t458+t459+t527-rz*t66*4.0+t4*t40*8.0-t15*t32*4.0-t7*t43*4.0-t16*t37*2.4E+1+t4*t69*4.0+t16*t63*4.0E+1+t6*t74*2.4E+1);
    fdrv(1, 9) = -tdx*(t213+t352+t499+t550-t4*t18*1.0E+1+t2*t21*4.0+t17*t32*8.0+t24*t31-t92*ty*2.0+t93*ty*6.0+t20*t31*tz)-t610*tdy*tx*2.0-t608*tdz*tx*2.0;
    fdrv(1, 10) = -tdy*(t206+t213+t352+t357+t499+t503+t2*t21*8.0+t9*t24*4.0+t17*t32*4.0+t20*t37*8.0-t23*t55*4.0)-t603*tdx*ty*2.0-t608*tdz*ty*2.0;
    fdrv(1, 11) = -tdz*(t206+t357+t503+t550+t2*t21*2.0-t8*t24*1.0E+1+t9*t24*8.0+t17*t32*2.0+t23*t32*6.0+t20*t37*4.0-t20*t55*2.0)-t603*tdx*tz*2.0-t610*tdy*tz*2.0; fdrv(1, 12) = t671;
    fdrv(1, 13) = tdx*(t217+t270+t278+t394+rx*t25*2.0+t20*t40*4.0+t23*t40*8.0+t20*t45*4.0+t23*t45*8.0+t95*tx*6.0+t96*tx*1.2E+1)+tdz*(t219+t269+t277+t396+rz*t19*2.0+t17*t39*8.0+t20*t39*4.0+t17*t44*8.0+t20*t44*4.0+t105*tz*1.2E+1+t106*tz*6.0)+t137*t595*tdy*2.0;
    fdrv(1, 14) = t669; fdrv(1, 15) = t530*tx*(t32+t37-t63+t47*tx)*-2.0; fdrv(1, 16) = t530*ty*(t32+t37-t63+t47*tx)*-2.0; fdrv(1, 17) = t530*tz*(t32+t37-t63+t47*tx)*-2.0;
    fdrv(1, 18) = rdx*t530*(t32+t37-t63+t47*tx)*-2.0-t11*tx*(t32+t37-t63+t47*tx)*4.0-t12*tx*(t32+t37-t63+t47*tx)*4.0-t13*tx*(t32+t37-t63+t47*tx)*4.0-t11*t472*t530*2.0-t12*t472*t530*2.0-t13*t472*t530*2.0;
    fdrv(1, 19) = rdy*t530*(t32+t37-t63+t47*tx)*-2.0-t11*ty*(t32+t37-t63+t47*tx)*4.0-t12*ty*(t32+t37-t63+t47*tx)*4.0-t13*ty*(t32+t37-t63+t47*tx)*4.0-t11*t221*t530*2.0-t12*t221*t530*2.0-t13*t221*t530*2.0;
    fdrv(1, 20) = rdz*t530*(t32+t37-t63+t47*tx)*-2.0-t11*tz*(t32+t37-t63+t47*tx)*4.0-t12*tz*(t32+t37-t63+t47*tx)*4.0-t13*tz*(t32+t37-t63+t47*tx)*4.0-t11*t530*(t9-t30)*2.0-t12*t530*(t9-t30)*2.0-t13*t530*(t9-t30)*2.0; fdrv(1, 24) = t591;
    fdrv(1, 25) = t137*t484*t530*2.0; fdrv(1, 26) = t593; fdrv(2, 0) = -t657*ty-t663*tz; fdrv(2, 1) = t657*tx-t662*tz; fdrv(2, 2) = t663*tx+t662*ty;
    fdrv(2, 3) = -t539+t540+t547+rx*t119-t33*t40*6.0-t34*t39*6.0+t14*t62*1.0E+1-t16*t60*2.0+t15*t62*8.0+t16*t62*8.0+t7*t73*8.0-t44*t54*2.0-t33*t69*4.0+t14*t90*1.2E+1-t35*t69*2.0+t16*t90*8.0+t7*t104*4.0+t31*t98+t7*t127-rx*t2*t24*4.0-rx*t6*t24*2.0-ry*t3*t24*2.0-t14*t33*tx*8.0-t3*t46*tz*2.0-t16*t34*tz*4.0;
    fdrv(2, 4) = -t542+t543+t546+ry*t115-t32*t44*6.0+t15*t66*1.0E+1-t16*t65*2.0+t16*t66*8.0-t39*t55*2.0+t38*t59*8.0-t4*t99*6.0-t33*t70*2.0+t7*t98*4.0+t29*t100+t9*t127-rx*t4*t24*2.0+rx*t7*t21*8.0-ry*t2*t24*2.0-ry*t6*t24*4.0+t7*t40*ty*1.2E+1-t15*t35*ty*8.0+t7*t45*ty*8.0-t4*t40*tz*4.0-t16*t32*tz*4.0-t4*t45*tz*2.0;
    fdrv(2, 5) = t226+t234-t478+t480-t481+t482-t535-t541+t544+t545+rz*t115+rz*t119-t15*t53*2.0-t16*t53*4.0-t34*t40*2.0-t2*t73*2.0-t16*t61*4.0-t2*t76*8.0-t6*t76*8.0-t14*t83*2.0-t15*t84*6.0-t34*t71*6.0+t14*t93*4.0+t15*t94*4.0+t59*t75*4.0+t14*t255-rx*t4*t21*2.0;
    fdrv(2, 6) = -t668*tdy-t673*tdz+tdx*(t162+t244-t329+t415+t425+t430+t454+t455+t517-rx*t60*4.0-t3*t41*4.0-t14*t33*2.4E+1-t16*t35*4.0+t7*t45*2.4E+1+t9*t46*8.0+t14*t59*4.0E+1+t9*t74*4.0);
    fdrv(2, 7) = -t668*tdx-t675*tdz+tdy*(t162+t233-t283+t415+t430+t445+t454+t455+t526-ry*t65*4.0-t4*t42*4.0-t16*t33*4.0-t15*t35*2.4E+1+t7*t45*8.0+t9*t46*2.4E+1+t7*t71*4.0+t15*t64*4.0E+1);
    fdrv(2, 8) = -t673*tdx-t675*tdy-tdz*(t290-t296+t311-t313+t362+t363+t400+t438-t448-t450+rz*t53*4.0+rz*t61*4.0+t16*t33*2.4E+1+t16*t35*2.4E+1-t14*t59*4.0-t15*t64*4.0);
    fdrv(2, 9) = -tdx*(t214+t353+t504+t551-t7*t18*1.0E+1+t2*t24*4.0+t6*t24*2.0+t17*t33*8.0+t20*t35*2.0-t23*t64*2.0+t85*tz*6.0)-t607*tdy*tx*2.0-t609*tdz*tx*2.0;
    fdrv(2, 10) = -tdy*(t210+t355+t505+t551+t2*t24*2.0+t6*t24*4.0-t9*t21*1.0E+1+t17*t33*2.0+t20*t33*6.0+t20*t35*8.0-t23*t59*2.0)-t605*tdx*ty*2.0-t609*tdz*ty*2.0;
    fdrv(2, 11) = -tdz*(t210+t214+t353+t355+t504+t505+t2*t24*8.0+t6*t24*8.0+t17*t33*4.0+t20*t35*4.0-t20*t59*4.0)-t605*tdx*tz*2.0-t607*tdy*tz*2.0; fdrv(2, 12) = t670; fdrv(2, 13) = t669;
    fdrv(2, 14) = tdx*(t217+t270+t278+t394+rx*t22*2.0+t20*t40*8.0+t23*t40*4.0+t20*t45*8.0+t23*t45*4.0+t95*tx*1.2E+1+t96*tx*6.0)+tdy*(t218+t268+t279+t395+ry*t19*2.0+t17*t38*8.0+t23*t38*4.0+t17*t46*8.0+t23*t46*4.0+t97*ty*1.2E+1+t104*ty*6.0)+t136*t596*tdz*2.0;
    fdrv(2, 15) = t530*tx*(t33+t35+t49*tx+t51*ty)*-2.0; fdrv(2, 16) = t530*ty*(t33+t35+t49*tx+t51*ty)*-2.0; fdrv(2, 17) = t530*tz*(t33+t35+t49*tx+t51*ty)*-2.0;
    fdrv(2, 18) = rdx*t530*(t33+t35+t49*tx+t51*ty)*-2.0-t11*tx*(t33+t35+t49*tx+t51*ty)*4.0-t12*tx*(t33+t35+t49*tx+t51*ty)*4.0-t13*tx*(t33+t35+t49*tx+t51*ty)*4.0-t11*t473*t530*2.0-t12*t473*t530*2.0-t13*t473*t530*2.0;
    fdrv(2, 19) = rdy*t530*(t33+t35+t49*tx+t51*ty)*-2.0-t11*ty*(t33+t35+t49*tx+t51*ty)*4.0-t12*ty*(t33+t35+t49*tx+t51*ty)*4.0-t13*ty*(t33+t35+t49*tx+t51*ty)*4.0-t11*t474*t530*2.0-t12*t474*t530*2.0-t13*t474*t530*2.0;
    fdrv(2, 20) = rdz*t530*(t33+t35+t49*tx+t51*ty)*-2.0-t11*tz*(t33+t35+t49*tx+t51*ty)*4.0-t12*tz*(t33+t35+t49*tx+t51*ty)*4.0-t13*tz*(t33+t35+t49*tx+t51*ty)*4.0-t11*t220*t530*2.0-t12*t220*t530*2.0-t13*t220*t530*2.0; fdrv(2, 24) = t592;
    fdrv(2, 25) = t593; fdrv(2, 26) = t136*t484*t530*2.0;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f21(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = oy*ty, t4 = oz*tz, t5 = rdx*tx, t6 = rdy*ty, t7 = rdz*tz, t8 = rx*tx, t9 = ry*ty, t10 = rz*tz, t11 = tx*tx, t12 = ty*ty, t13 = tz*tz, t23 = ox*ry*tz, t24 = ox*rz*ty, t25 = oy*rx*tz, t26 = oy*rz*tx, t27 = oz*rx*ty;
    T t28 = oz*ry*tx, t38 = ox*ty*tz, t39 = oy*tx*tz, t40 = oz*tx*ty, t47 = rx*tdx*ty*tz, t48 = ry*tdy*tx*tz, t49 = rz*tdz*tx*ty, t63 = ox*rx*ty*2.0, t65 = ox*rx*tz*2.0, t67 = oy*ry*tx*2.0, t77 = oy*ry*tz*2.0, t79 = oz*rz*tx*2.0;
    T t81 = oz*rz*ty*2.0, t14 = t8*2.0, t15 = t9*2.0, t16 = t10*2.0, t17 = ry*t2, t18 = oy*t8, t19 = ox*t9, t20 = rz*t2, t21 = rx*t3, t22 = oz*t8, t29 = ox*t10, t30 = rz*t3, t31 = rx*t4, t32 = oz*t9, t33 = oy*t10, t34 = ry*t4, t35 = t2*ty;
    T t36 = t2*tz, t37 = t3*tx, t41 = t3*tz, t42 = t4*tx, t43 = t4*ty, t50 = t2*tx, t51 = ox*t12, t52 = oy*t11, t53 = ox*t13, t54 = t3*ty, t55 = oz*t11, t56 = oy*t13, t57 = oz*t12, t58 = t4*tz, t59 = t8*tx, t60 = t9*ty, t61 = t10*tz;
    T t62 = rx*t2*2.0, t69 = t23*2.0, t70 = t24*2.0, t71 = t25*2.0, t72 = ry*t3*2.0, t73 = t26*2.0, t74 = t27*2.0, t75 = t28*2.0, t82 = rz*t4*2.0, t86 = t38*2.0, t87 = t39*2.0, t88 = t40*2.0, t98 = t2*t5, t99 = t3*t6, t100 = t4*t7, t110 = -t63;
    T t111 = -t65, t112 = -t67, t119 = -t77, t120 = -t79, t121 = -t81, t138 = t11+t12, t139 = t11+t13, t140 = t12+t13, t144 = -t47, t145 = -t48, t146 = -t49, t147 = t8+t9, t148 = t8+t10, t149 = t9+t10, t156 = t5+t6+t7, t44 = t35*tz, t45 = t37*tz;
    T t46 = t42*ty, t64 = oy*t14, t66 = ox*t15, t68 = oz*t14, t76 = ox*t16, t78 = oz*t15, t80 = oy*t16, t83 = t35*2.0, t84 = t36*2.0, t85 = t37*2.0, t89 = t41*2.0, t90 = t42*2.0, t91 = t43*2.0, t92 = t14*ty, t93 = t14*tz, t94 = t15*tx;
    T t95 = t15*tz, t96 = t16*tx, t97 = t16*ty, t107 = t59*3.0, t108 = t60*3.0, t109 = t61*3.0, t115 = -t71, t116 = -t73, t117 = -t74, t118 = -t75, t122 = -t35, t124 = -t36, t125 = -t37, t130 = -t87, t131 = -t88, t132 = -t41, t133 = -t42;
    T t136 = -t43, t150 = -t98, t151 = -t99, t152 = -t100, t153 = ox*t140, t154 = oy*t139, t155 = oz*t138, t157 = t16+t147, t158 = t15+t148, t159 = t14+t149, t166 = t156*tx*ty, t167 = t156*tx*tz, t168 = t156*ty*tz, t101 = rx*t83, t102 = rx*t84;
    T t103 = ry*t85, t104 = ry*t89, t105 = rz*t90, t106 = rz*t91, t123 = -t83, t126 = -t84, t127 = -t85, t134 = -t89, t135 = -t90, t137 = -t91, t141 = -t44, t142 = -t45, t143 = -t46, t160 = t38+t130, t161 = t38+t131, t162 = t39+t131, t169 = -t166;
    T t170 = -t167, t171 = -t168, t172 = t23+t25+t117+t118, t173 = t24+t27+t115+t116, t175 = t19+t21+t62+t76+t120, t176 = t29+t31+t62+t66+t112, t177 = t17+t18+t72+t80+t121, t178 = t33+t34+t64+t72+t110, t179 = t20+t22+t78+t82+t119;
    T t180 = t30+t32+t68+t82+t111;
    
    fdrv(0, 0) = t8*t12+t8*t13+t140*t149; fdrv(0, 1) = rx*t13*ty-rx*t139*ty-t149*tx*ty; fdrv(0, 2) = rx*t12*tz-rx*t138*tz-t149*tx*tz; fdrv(0, 3) = rx*t153-t3*t159-t4*t159;
    fdrv(0, 4) = t101-ry*t37*2.0+ry*t53-t18*tx-t33*tx-t34*tx+t19*ty*3.0+t76*ty; fdrv(0, 5) = t102-rz*t42*2.0+rz*t51-t22*tx-t30*tx-t32*tx+t29*tz*3.0+t66*tz; fdrv(0, 6) = -t178*tdy-t180*tdz-rx*tdx*(t3+t4)*2.0;
    fdrv(0, 7) = -t178*tdx+tdy*(t19*6.0+t62+t76+t112)-tdz*(t26+t28-t69-t70); fdrv(0, 8) = -t180*tdx+tdz*(t29*6.0+t62+t66+t120)-tdy*(t26+t28-t69-t70); fdrv(0, 9) = -tdy*(t52+t123)-tdz*(t55+t126)-tdx*(t85+t90-t153);
    fdrv(0, 10) = -tdx*(t43+t54)-tdz*(t40-t86)-tdy*(t42-t51*3.0-t53+t85); fdrv(0, 11) = -tdx*(t41+t58)-tdy*(t39-t86)-tdz*(t37-t51-t53*3.0+t90); fdrv(0, 12) = tdy*(t92+t97+t108+ry*t13)+tdz*(t93+t95+t109+rz*t12)+rx*t140*tdx;
    fdrv(0, 13) = t146-tdy*(t59+t94+t10*tx)-t159*tdx*ty; fdrv(0, 14) = t145-tdz*(t59+t96+t9*tx)-t159*tdx*tz; fdrv(0, 15) = -tx*(t37+t42-t51-t53); fdrv(0, 16) = t143+t125*ty+t153*ty; fdrv(0, 17) = t142+t133*tz+t153*tz;
    fdrv(0, 18) = t151+t152+rdx*t51+rdx*t53-t3*t5*2.0-t4*t5*2.0-t3*t7-t4*t6; fdrv(0, 19) = -rdz*(t39-t86)+t5*(ox*ty*2.0-oy*tx)-rdy*(t42-t51*2.0+t85-t153); fdrv(0, 20) = -rdy*(t40-t86)+t5*(ox*tz*2.0-oz*tx)-rdz*(t37-t53*2.0+t90-t153);
    fdrv(0, 24) = t140*t156; fdrv(0, 25) = t169; fdrv(0, 26) = t170; fdrv(1, 0) = ry*t13*tx-ry*t140*tx-t148*tx*ty; fdrv(1, 1) = t9*t11+t9*t13+t139*t148; fdrv(1, 2) = ry*t11*tz-ry*t138*tz-t148*ty*tz;
    fdrv(1, 3) = t103-rx*t35*2.0+rx*t56+t18*tx*3.0+t80*tx-t19*ty-t29*ty-t31*ty; fdrv(1, 4) = ry*t154-t2*t158-t4*t158; fdrv(1, 5) = t104-rz*t43*2.0+rz*t52-t20*ty-t22*ty-t32*ty+t33*tz*3.0+t64*tz; fdrv(1, 6) = -t176*tdy-t173*tdz+tdx*(t18*6.0+t72+t80+t110);
    fdrv(1, 7) = -t176*tdx-t179*tdz-ry*tdy*(t2+t4)*2.0; fdrv(1, 8) = -t173*tdx-t179*tdy+tdz*(t33*6.0+t64+t72+t121); fdrv(1, 9) = -tdy*(t42+t50)-tdz*(t40+t130)-tdx*(t43-t52*3.0-t56+t83); fdrv(1, 10) = -tdx*(t51+t127)-tdz*(t57+t134)-tdy*(t83+t91-t154);
    fdrv(1, 11) = -tdy*(t36+t58)-t160*tdx-tdz*(t35-t52-t56*3.0+t91); fdrv(1, 12) = t146-tdx*(t60+t92+t10*ty)-t158*tdy*tx; fdrv(1, 13) = tdx*(t94+t96+t107+rx*t13)+tdz*(t93+t95+t109+rz*t11)+ry*t139*tdy; fdrv(1, 14) = t144-tdz*(t60+t97+t8*ty)-t158*tdy*tz;
    fdrv(1, 15) = t143+t122*tx+t154*tx; fdrv(1, 16) = -ty*(t35+t43-t52-t56); fdrv(1, 17) = t141+t136*tz+t154*tz; fdrv(1, 18) = -rdz*t160-t6*(ox*ty-oy*tx*2.0)-rdx*(t43-t52*2.0+t83-t154);
    fdrv(1, 19) = t150+t152+rdy*t52+rdy*t56-t2*t6*2.0-t2*t7-t4*t5-t4*t6*2.0; fdrv(1, 20) = -rdx*(t40+t130)+t6*(oy*tz*2.0-oz*ty)-rdz*(t35-t56*2.0+t91-t154); fdrv(1, 24) = t169; fdrv(1, 25) = t139*t156; fdrv(1, 26) = t171;
    fdrv(2, 0) = rz*t12*tx-rz*t140*tx-t147*tx*tz; fdrv(2, 1) = rz*t11*ty-rz*t139*ty-t147*ty*tz; fdrv(2, 2) = t10*t11+t10*t12+t138*t147; fdrv(2, 3) = t105-rx*t36*2.0+rx*t57+t22*tx*3.0+t78*tx-t19*tz-t21*tz-t29*tz;
    fdrv(2, 4) = t106-ry*t41*2.0+ry*t55+t32*ty*3.0+t68*ty-t17*tz-t18*tz-t33*tz; fdrv(2, 5) = rz*t155-t2*t157-t3*t157; fdrv(2, 6) = -t172*tdy-t175*tdz+tdx*(t22*6.0+t78+t82+t111); fdrv(2, 7) = -t172*tdx-t177*tdz+tdy*(t32*6.0+t68+t82+t119);
    fdrv(2, 8) = -t175*tdx-t177*tdy-rz*tdz*(t2+t3)*2.0; fdrv(2, 9) = -tdz*(t37+t50)-t162*tdy-tdx*(t41-t55*3.0-t57+t84); fdrv(2, 10) = -tdz*(t35+t54)-t161*tdx-tdy*(t36-t55-t57*3.0+t89); fdrv(2, 11) = -tdx*(t53+t135)-tdy*(t56+t137)-tdz*(t84+t89-t155);
    fdrv(2, 12) = t145-tdx*(t61+t93+t9*tz)-t157*tdz*tx; fdrv(2, 13) = t144-tdy*(t61+t95+t8*tz)-t157*tdz*ty; fdrv(2, 14) = tdx*(t94+t96+t107+rx*t12)+tdy*(t92+t97+t108+ry*t11)+rz*t138*tdz; fdrv(2, 15) = t142+t124*tx+t155*tx;
    fdrv(2, 16) = t141+t132*ty+t155*ty; fdrv(2, 17) = -tz*(t36+t41-t55-t57); fdrv(2, 18) = -rdy*t161-t7*(ox*tz-oz*tx*2.0)-rdx*(t41-t55*2.0+t84-t155); fdrv(2, 19) = -rdx*t162-t7*(oy*tz-oz*ty*2.0)-rdy*(t36-t57*2.0+t89-t155);
    fdrv(2, 20) = t150+t151+rdz*t55+rdz*t57-t2*t6-t3*t5-t2*t7*2.0-t3*t7*2.0; fdrv(2, 24) = t170; fdrv(2, 25) = t171; fdrv(2, 26) = t138*t156;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f22(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = ox*ty, t4 = oy*tx, t5 = ox*tz, t6 = oy*ty, t7 = oz*tx, t8 = oy*tz, t9 = oz*ty, t10 = oz*tz, t11 = rdx*tx, t12 = rdy*ty, t13 = rdz*tz, t14 = rx*tx, t15 = rx*ty, t16 = ry*tx, t17 = rx*tz, t18 = ry*ty, t19 = rz*tx;
    T t20 = ry*tz, t21 = rz*ty, t22 = rz*tz, t23 = ox*2.0, t24 = oy*2.0, t25 = oz*2.0, t26 = tx*tx, t27 = tx*tx*tx, t29 = ty*ty, t30 = ty*ty*ty, t32 = tz*tz, t33 = tz*tz*tz, t35 = t3*2.0, t36 = t4*2.0, t37 = t5*2.0, t38 = t7*2.0, t39 = t8*2.0;
    T t40 = t9*2.0, t41 = t14*2.0, t42 = t18*2.0, t43 = t22*2.0, t44 = t2*ty, t45 = t2*tz, t46 = t4*ty, t47 = t6*tz, t48 = t7*tz, t49 = t9*tz, t50 = t14*ty, t51 = t14*tz, t52 = t16*ty, t53 = t18*tz, t54 = t19*tz, t55 = t21*tz, t56 = t27*2.0;
    T t57 = t30*2.0, t58 = t33*2.0, t59 = -t4, t61 = -t5, t64 = -t9, t66 = t2*t26, t67 = t3*ty, t68 = t4*tx, t69 = t3*t29, t70 = t4*t26, t71 = t3*t30, t72 = t4*t27, t73 = t5*tz, t74 = t7*tx, t75 = t5*t32, t76 = t6*t29, t77 = t7*t26, t78 = t5*t33;
    T t79 = t7*t27, t80 = t8*tz, t81 = t9*ty, t82 = t8*t32, t83 = t9*t29, t84 = t8*t33, t85 = t9*t30, t86 = t10*t32, t87 = t14*t27, t88 = t18*t30, t89 = t22*t33, t90 = t30*tx, t91 = t27*ty, t92 = t33*tx, t93 = t27*tz, t94 = t33*ty, t95 = t30*tz;
    T t99 = t3*tz*4.0, t100 = t4*tz*4.0, t101 = t7*ty*4.0, t106 = t2*t16*tx, t109 = t2*t19*tx, t110 = t6*t15*ty, t120 = t6*t21*ty, t121 = t10*t17*tz, t124 = t10*t20*tz, t126 = t2*t29, t127 = t2*t32, t129 = t3*t32, t131 = t4*t32, t133 = t7*t29;
    T t135 = t6*t32, t138 = t14*t30, t140 = t14*t33, t142 = t16*t30, t144 = t15*t33, t145 = t15*t29*tz, t146 = t16*t33, t147 = t16*t26*tz, t148 = t19*t30, t149 = t19*t26*ty, t150 = t18*t33, t152 = t19*t33, t154 = t21*t33, t156 = t32*tx*ty;
    T t157 = t29*tx*tz, t158 = t26*ty*tz, t159 = t2*t15*4.0, t160 = t2*t17*4.0, t161 = t2*t18*4.0, t162 = t4*t15*4.0, t163 = t3*t17*4.0, t164 = t2*t20*4.0, t165 = t2*t21*4.0, t166 = t4*t18*4.0, t167 = t2*t22*4.0, t168 = t6*t17*4.0;
    T t169 = t4*t20*4.0, t170 = t4*t21*4.0, t171 = t7*t17*4.0, t172 = t6*t20*4.0, t173 = t9*t17*4.0, t174 = t7*t20*4.0, t175 = t7*t21*4.0, t176 = t6*t22*4.0, t177 = t9*t20*4.0, t178 = t7*t22*4.0, t179 = t9*t22*4.0, t186 = t2*tx*2.0;
    T t196 = t6*ty*2.0, t212 = t10*tz*2.0, t215 = t15*t29*2.0, t216 = t16*t26*2.0, t217 = t17*t32*2.0, t219 = t19*t26*2.0, t221 = t20*t32*2.0, t222 = t21*t29*2.0, t225 = t29*tx*2.0, t226 = t26*ty*2.0, t227 = t32*tx*2.0, t228 = t26*tz*2.0;
    T t229 = t32*ty*2.0, t230 = t29*tz*2.0, t245 = t26+t29, t246 = t26+t32, t247 = t29+t32, t249 = t2*t14*tx*4.0, t250 = t2*t14*6.0, t252 = t2*t16*2.0, t255 = t4*t14*4.0, t258 = t3*t15*6.0, t259 = t4*t14*6.0, t261 = t2*t19*2.0, t262 = t6*t15*2.0;
    T t268 = t3*t18*4.0, t269 = t7*t14*4.0, t274 = t5*t17*6.0, t275 = t3*t18*6.0, t276 = t4*t16*6.0, t277 = t7*t14*6.0, t282 = t6*t18*ty*4.0, t286 = t5*t20*6.0, t287 = t3*t21*6.0, t288 = t8*t17*6.0, t289 = t6*t18*6.0, t290 = t4*t19*6.0;
    T t291 = t9*t15*6.0, t292 = t7*t16*6.0, t294 = t6*t21*2.0, t295 = t10*t17*2.0, t301 = t5*t22*4.0, t302 = t9*t18*4.0, t307 = t5*t22*6.0, t308 = t8*t20*6.0, t309 = t9*t18*6.0, t310 = t7*t19*6.0, t311 = t10*t20*2.0, t315 = t8*t22*4.0;
    T t318 = t8*t22*6.0, t319 = t9*t21*6.0, t321 = t10*t22*tz*4.0, t322 = t10*t22*6.0, t333 = t4*t29*3.0, t347 = t7*t32*3.0, t351 = t9*t32*3.0, t355 = t14*t29*4.0, t359 = t14*t29*6.0, t365 = t14*t32*4.0, t367 = t16*t29*4.0, t373 = t14*t32*6.0;
    T t375 = t16*t29*6.0, t377 = t15*t32*2.0, t378 = t15*ty*tz*2.0, t379 = t16*t32*2.0, t380 = t16*tx*tz*2.0, t381 = t19*t29*2.0, t382 = t19*tx*ty*2.0, t387 = t18*t32*4.0, t389 = t19*t32*4.0, t395 = t18*t32*6.0, t397 = t19*t32*6.0;
    T t401 = t21*t32*4.0, t405 = t21*t32*6.0, t408 = t2*t15*1.2E+1, t410 = t2*t18*8.0, t411 = t4*t15*8.0, t412 = t2*t17*1.2E+1, t415 = t4*t17*8.0, t416 = t7*t15*8.0, t417 = t4*t17*1.2E+1, t418 = t4*t18*1.2E+1, t419 = t7*t15*1.2E+1;
    T t423 = t3*t20*8.0, t424 = t2*t22*8.0, t425 = t7*t17*8.0, t426 = t7*t18*8.0, t427 = t3*t20*1.2E+1, t428 = t7*t18*1.2E+1, t432 = t3*t22*8.0, t433 = t4*t22*8.0, t434 = t3*t22*1.2E+1, t435 = t6*t20*1.2E+1, t436 = t4*t22*1.2E+1;
    T t439 = t6*t22*8.0, t440 = t9*t20*8.0, t441 = t7*t22*1.2E+1, t442 = t9*t22*1.2E+1, t453 = t4*t15*tz*6.0, t483 = t14*t26*8.0, t484 = t18*t29*8.0, t485 = t22*t32*8.0, t486 = t2+t6, t487 = t2+t10, t488 = t6+t10, t489 = t14+t18, t490 = t14+t22;
    T t491 = t18+t22, t492 = t2*t14*tx*-2.0, t494 = t3*t15*-2.0, t508 = t6*t18*ty*-2.0, t519 = t8*t20*-2.0, t522 = t7*t19*-2.0, t532 = t10*t22*tz*-2.0, t537 = t4*t29*-2.0, t549 = t9*t32*-2.0, t558 = t14*t29*tx, t559 = t14*t32*tx;
    T t560 = t16*t29*tx, t561 = t15*t32*ty, t562 = t16*t32*tx, t563 = t19*t29*tx, t564 = t18*t32*ty, t565 = t19*t32*tx, t566 = t21*t32*ty, t601 = t4*t17*tz*3.0, t603 = t7*t15*ty*3.0, t621 = t3*t20*tz*3.0, t631 = t7*t18*ty*3.0;
    T t651 = t3*t22*tz*3.0, t653 = t4*t22*tz*3.0, t697 = t16*t29*tz*3.0, t702 = t19*t32*ty*3.0, t707 = t4*t15*tz*-4.0, t740 = t3*t17*tz*-4.0, t741 = t3*t15*tz*-4.0, t750 = t4*t20*tz*-2.0, t754 = t4*t20*tz*-4.0, t755 = t4*t16*tz*-4.0;
    T t767 = t7*t21*ty*-4.0, t768 = t7*t19*ty*-4.0, t781 = t16*t29*tz*-2.0, t782 = t19*t32*ty*-2.0, t804 = t11+t12+t13, t817 = t7*t18*ty*-1.2E+1, t818 = t3*t22*tz*-1.2E+1, t819 = t4*t22*tz*-1.2E+1, t60 = -t36, t62 = -t37, t63 = -t38, t65 = -t40;
    T t96 = t44*4.0, t97 = t45*4.0, t98 = t46*4.0, t102 = t47*4.0, t103 = t48*4.0, t104 = t49*4.0, t105 = t14*t66, t107 = t14*t68, t108 = t18*t67, t111 = t14*t74, t112 = t20*t73, t113 = t21*t67, t114 = t17*t80, t115 = t19*t68, t116 = t15*t81;
    T t117 = t16*t74, t118 = t18*t76, t119 = t22*t73, t122 = t18*t81, t123 = t22*t80, t125 = t22*t86, t128 = t46*tx, t130 = t67*tz, t132 = t68*tz, t134 = t74*ty, t136 = t48*tx, t137 = t49*ty, t139 = t26*t50, t141 = t26*t51, t143 = t26*t52;
    T t151 = t29*t53, t153 = t26*t54, t155 = t29*t55, t180 = t44*tz*2.0, t181 = t36*ty*tz, t182 = t38*ty*tz, t183 = t50*tz*4.0, t184 = t52*tz*4.0, t185 = t54*ty*4.0, t187 = t35*ty, t188 = t36*tx, t189 = t29*t35, t190 = t26*t36, t191 = t69*4.0;
    T t192 = t70*4.0, t193 = t67*6.0, t194 = t68*6.0, t195 = t37*tz, t197 = t38*tx, t199 = t26*t38, t200 = t75*4.0, t201 = t77*4.0, t202 = t73*6.0, t203 = t74*6.0, t204 = t39*tz, t205 = t40*ty, t206 = t32*t39, t207 = t29*t40, t208 = t82*4.0;
    T t209 = t83*4.0, t210 = t80*6.0, t211 = t81*6.0, t213 = t26*t41, t214 = t87*5.0, t218 = t29*t42, t220 = t88*5.0, t223 = t32*t43, t224 = t89*5.0, t231 = -t44, t235 = -t99, t236 = -t100, t237 = -t101, t240 = -t49, t242 = -t51, t243 = -t52;
    T t244 = -t55, t248 = t2*t41*tx, t256 = t15*t67*4.0, t260 = t17*t37, t263 = t16*t36, t270 = t17*t73*4.0, t272 = t16*t68*4.0, t278 = t6*t42*ty, t304 = t20*t80*4.0, t306 = t19*t74*4.0, t312 = t21*t40, t317 = t21*t81*4.0, t320 = t10*t43*tz;
    T t323 = t126*2.0, t324 = t44*tx*2.0, t325 = t126*3.0, t326 = t44*tx*3.0, t327 = t127*2.0, t328 = t45*tx*2.0, t329 = t29*t36, t331 = t127*3.0, t332 = t45*tx*3.0, t335 = t129*4.0, t337 = t131*4.0, t339 = t133*4.0, t341 = t135*2.0;
    T t342 = t47*ty*2.0, t343 = t32*t38, t345 = t135*3.0, t346 = t47*ty*3.0, t349 = t32*t40, t353 = t29*t41, t354 = t30*t41, t356 = t50*tx*4.0, t357 = t138*4.0, t360 = t50*tx*6.0, t361 = t32*t41, t362 = t52*tx*2.0, t363 = t33*t41;
    T t366 = t51*tx*4.0, t368 = t52*tx*4.0, t369 = t140*4.0, t371 = t142*4.0, t374 = t51*tx*6.0, t376 = t52*tx*6.0, t383 = t32*t42, t384 = t54*tx*2.0, t385 = t33*t42, t388 = t53*ty*4.0, t390 = t54*tx*4.0, t391 = t150*4.0, t393 = t152*4.0;
    T t396 = t53*ty*6.0, t398 = t54*tx*6.0, t399 = t55*ty*2.0, t402 = t55*ty*4.0, t403 = t154*4.0, t406 = t55*ty*6.0, t407 = -t159, t409 = -t161, t413 = -t163, t414 = -t164, t420 = -t169, t421 = -t170, t422 = -t171, t429 = -t172, t430 = -t173;
    T t431 = -t175, t437 = -t176, t438 = -t178, t444 = t46*tz*-2.0, t445 = t48*ty*-2.0, t451 = t15*t100, t452 = t18*t45*6.0, t455 = t4*t53*4.0, t457 = t21*t45*6.0, t458 = t15*t48*6.0, t459 = t4*t55*4.0, t461 = t4*t55*6.0, t462 = t18*t48*6.0;
    T t464 = -t67, t465 = t26*t59, t466 = -t71, t467 = t70*-2.0, t468 = t27*t59, t470 = t61*tz, t471 = -t74, t472 = t32*t61, t473 = t75*-2.0, t474 = t33*t61, t475 = -t79, t477 = t64*ty, t478 = t29*t64, t479 = -t84, t480 = t83*-2.0, t481 = t30*t64;
    T t493 = -t250, t495 = -t252, t496 = -t255, t498 = t17*t73*-2.0, t500 = t16*t68*-2.0, t504 = -t274, t505 = -t275, t506 = -t276, t507 = -t277, t515 = -t286, t516 = -t289, t517 = -t290, t518 = -t291, t520 = -t294, t521 = -t295, t523 = -t301;
    T t524 = -t302, t528 = t21*t81*-2.0, t530 = -t318, t531 = -t319, t533 = -t322, t540 = -t333, t542 = t32*t59, t551 = -t351, t552 = -t411, t553 = -t415, t554 = -t424, t555 = -t426, t556 = -t432, t557 = -t440, t567 = t15*t44*2.0, t568 = t41*t44;
    T t569 = t15*t44*6.0, t570 = t14*t44*6.0, t571 = t17*t45*2.0, t572 = t41*t45, t573 = t16*t44*2.0, t574 = t15*t36*ty, t575 = t18*t44*3.0, t576 = t16*t44*3.0, t577 = t15*t46*3.0, t578 = t14*t46*3.0, t579 = t17*t45*6.0, t580 = t14*t45*6.0;
    T t581 = t18*t44*6.0, t582 = t16*t44*6.0, t583 = t15*t46*6.0, t584 = t14*t46*6.0, t585 = t17*t35*tz, t587 = t20*t45*2.0, t588 = t16*t45*2.0, t589 = t21*t44*2.0, t590 = t19*t44*2.0, t591 = t17*t36*tz, t592 = t36*t51, t593 = t18*t36*ty;
    T t594 = t36*t52, t595 = t15*t38*ty, t596 = t38*t50, t597 = t20*t45*3.0, t598 = t16*t45*3.0, t599 = t21*t44*3.0, t600 = t19*t44*3.0, t602 = t4*t51*3.0, t604 = t7*t50*3.0, t607 = t18*t46*6.0, t608 = t16*t46*6.0, t609 = t20*t35*tz;
    T t610 = t35*t53, t611 = t19*t45*2.0, t612 = t17*t47*2.0, t613 = t15*t47*2.0, t617 = t19*t36*ty, t619 = t18*t38*ty, t620 = t38*t52, t622 = t3*t53*3.0, t623 = t22*t45*3.0, t624 = t19*t45*3.0, t625 = t17*t47*3.0, t626 = t15*t47*3.0;
    T t627 = t21*t46*3.0, t628 = t19*t46*3.0, t629 = t17*t48*3.0, t630 = t14*t48*3.0, t632 = t7*t52*3.0, t635 = t22*t45*6.0, t636 = t19*t45*6.0, t637 = t17*t48*6.0, t638 = t14*t48*6.0, t639 = t22*t35*tz, t640 = t35*t55, t641 = t20*t47*2.0;
    T t642 = t42*t47, t643 = t22*t36*tz, t644 = t36*t54, t646 = t15*t40*tz, t647 = t20*t38*tz, t648 = t16*t38*tz, t649 = t21*t38*ty, t652 = t3*t55*3.0, t654 = t4*t54*3.0, t655 = t17*t49*3.0, t656 = t15*t49*3.0, t657 = t20*t48*3.0;
    T t658 = t16*t48*3.0, t661 = t20*t47*6.0, t662 = t18*t47*6.0, t663 = t21*t47*2.0, t664 = t20*t40*tz, t665 = t22*t38*tz, t666 = t38*t54, t667 = t22*t47*3.0, t668 = t21*t47*3.0, t669 = t20*t49*3.0, t670 = t18*t49*3.0, t671 = t22*t47*6.0;
    T t672 = t21*t47*6.0, t673 = t20*t49*6.0, t674 = t18*t49*6.0, t675 = t22*t48*6.0, t676 = t19*t48*6.0, t677 = t22*t40*tz, t678 = t40*t55, t679 = t22*t49*6.0, t680 = t21*t49*6.0, t681 = t32*t44*2.0, t682 = t29*t45*2.0, t683 = t32*t36*ty;
    T t685 = t29*t38*tz, t689 = t41*tx*ty*tz, t690 = t32*t50*3.0, t691 = t29*t51*3.0, t692 = t50*tx*tz*3.0, t693 = t32*t52*2.0, t694 = t16*t230, t696 = t32*t52*3.0, t698 = t52*tx*tz*3.0, t699 = t19*t229, t700 = t29*t54*2.0, t703 = t29*t54*3.0;
    T t704 = t54*tx*ty*3.0, t705 = t15*t45*8.0, t706 = t18*t45*-4.0, t708 = t21*t45*-4.0, t709 = t15*t48*-4.0, t710 = t4*t53*8.0, t712 = t18*t48*-4.0, t713 = t21*t48*8.0, t716 = t29*t68*3.0, t721 = t32*t74*3.0, t722 = t32*t81*3.0, t723 = t558*3.0;
    T t724 = t559*3.0, t725 = t560*3.0, t726 = t564*3.0, t727 = t565*3.0, t728 = t566*3.0, t735 = t494*tz, t744 = t417*tz, t745 = t4*t51*1.2E+1, t746 = t419*ty, t747 = t7*t50*1.2E+1, t751 = t21*t46*-2.0, t752 = t19*t46*-2.0, t753 = t17*t48*-2.0;
    T t758 = t427*tz, t759 = t3*t53*1.2E+1, t761 = t7*t52*1.2E+1, t762 = t17*t49*-2.0, t763 = t15*t49*-2.0, t764 = t20*t48*-2.0, t765 = t16*t48*-2.0, t766 = t522*ty, t771 = t3*t55*1.2E+1, t773 = t4*t54*1.2E+1, t780 = t50*tx*tz*-2.0;
    T t786 = t489*tx, t787 = t489*ty, t788 = t490*tx, t789 = t490*tz, t790 = t491*ty, t791 = t491*tz, t792 = t41+t42, t793 = t41+t43, t794 = t42+t43, t796 = t42*t127, t797 = t15*t32*t36, t798 = t44*t55*2.0, t799 = t36*t53*ty, t805 = t126*tx*-3.0;
    T t806 = t127*tx*-3.0, t808 = t32*t67*-6.0, t809 = t32*t68*-6.0, t810 = t29*t74*-6.0, t811 = t135*ty*-3.0, t820 = t32+t245, t824 = t14*t333, t825 = t16*t333, t827 = t14*t347, t830 = t18*t351, t831 = t19*t347, t832 = t21*t351;
    T t845 = t25+t35+t59, t846 = t24+t38+t61, t847 = t23+t39+t64, t232 = -t96, t233 = -t97, t234 = -t98, t238 = -t102, t239 = -t103, t241 = -t104, t253 = t15*t187, t254 = t14*t188, t257 = t107*4.0, t271 = t108*4.0, t273 = t111*4.0;
    T t279 = t112*4.0, t280 = t113*4.0, t281 = t114*4.0, t283 = t115*4.0, t284 = t116*4.0, t285 = t117*4.0, t297 = t22*t195, t298 = t20*t204, t299 = t18*t205, t300 = t19*t197, t303 = t119*4.0, t305 = t122*4.0, t316 = t123*4.0, t334 = t128*3.0;
    T t336 = t130*4.0, t338 = t132*4.0, t340 = t134*4.0, t348 = t136*3.0, t352 = t137*3.0, t358 = t139*4.0, t364 = t143*2.0, t370 = t141*4.0, t372 = t143*4.0, t386 = t153*2.0, t392 = t151*4.0, t394 = t153*4.0, t400 = t155*2.0, t404 = t155*4.0;
    T t443 = -t180, t446 = -t183, t447 = -t184, t448 = -t185, t449 = t15*t97, t450 = t18*t97, t454 = t21*t97, t456 = t15*t103, t460 = t18*t103, t463 = t21*t103, t469 = -t192, t476 = -t200, t482 = -t209, t497 = -t256, t499 = t108*-2.0;
    T t501 = t111*-2.0, t502 = -t270, t503 = -t272, t525 = -t304, t526 = -t306, t527 = t123*-2.0, t529 = -t317, t534 = -t323, t535 = -t327, t536 = -t328, t538 = t128*-2.0, t539 = -t332, t541 = -t130, t543 = -t134, t545 = -t337, t547 = -t341;
    T t548 = t136*-2.0, t550 = t137*-2.0, t615 = t263*tz, t684 = t181*tx, t686 = t182*tx, t687 = t361*ty, t688 = t353*tz, t695 = t362*tz, t701 = t384*ty, t711 = -t459, t729 = -t570, t730 = -t573, t731 = -t579, t732 = -t580, t733 = -t581;
    T t734 = -t584, t736 = -t587, t737 = -t588, t738 = -t589, t739 = -t590, t742 = -t607, t743 = -t608, t748 = -t612, t749 = -t613, t756 = -t635, t757 = -t638, t769 = -t662, t774 = -t663, t775 = -t671, t776 = -t674, t777 = -t675, t778 = -t679;
    T t779 = -t680, t783 = -t705, t784 = -t710, t785 = -t713, t795 = t568*tz, t800 = t15*t182, t801 = t644*ty, t802 = t620*tz, t803 = t21*t343, t807 = -t716, t812 = -t721, t813 = -t722, t814 = -t745, t815 = -t747, t816 = -t759, t821 = t14*t325;
    T t822 = t14*t331, t823 = t16*t325, t826 = t19*t331, t828 = t18*t345, t829 = t21*t345, t833 = t792*tx, t834 = -t787, t835 = t792*ty, t836 = -t788, t837 = t793*tx, t838 = t793*tz, t839 = t794*ty, t840 = -t791, t841 = t794*tz, t842 = t3+t25+t60;
    T t843 = t7+t24+t62, t844 = t8+t23+t65, t848 = t15+t16+t53+t789, t849 = t17+t19+t50+t790, t850 = t20+t21+t54+t786, t869 = t35+t45+t47+t60+t471+t477, t870 = t37+t63+t68+t80+t231+t240, t871 = t39+t46+t48+t65+t464+t470, t509 = -t279;
    T t510 = -t280, t511 = -t281, t512 = -t283, t513 = -t284, t514 = -t285, t544 = -t336, t546 = -t340, t851 = t848*tx, t852 = t850*tx, t853 = t849*ty, t854 = t850*ty, t855 = t848*tz, t856 = t849*tz, t859 = t15+t16+t242+t840;
    T t860 = t17+t19+t244+t834, t861 = t20+t21+t243+t836, t881 = t66+t76+t99+t126+t128+t236+t331+t345+t548+t550, t882 = t66+t86+t101+t127+t136+t235+t325+t352+t538+t547, t883 = t76+t86+t100+t135+t137+t237+t334+t348+t534+t535;
    T t884 = t69+t129+t182+t203+t205+t212+t233+t326+t469+t537+t545, t887 = t194+t196+t201+t204+t232+t339+t343+t444+t472+t539+t541, t888 = t191+t197+t211+t212+t238+t324+t335+t445+t465+t540+t542;
    T t889 = t186+t187+t202+t208+t239+t338+t342+t443+t478+t543+t551, t890 = t119+t121+t179+t249+t271+t292+t309+t311+t413+t414+t416+t460+t503+t569+t571+t582+t609+t624+t630+t652+t656+t665+t711+t734+t742+t748+t754;
    T t891 = t106+t107+t160+t261+t282+t287+t307+t316+t420+t421+t423+t454+t529+t568+t575+t577+t594+t597+t601+t644+t661+t672+t709+t765+t768+t776+t778;
    T t892 = t120+t122+t166+t259+t262+t273+t288+t321+t430+t431+t433+t451+t502+t595+t628+t632+t637+t642+t667+t669+t676+t678+t706+t732+t738+t741+t756;
    T t893 = t108+t110+t163+t165+t249+t303+t429+t459+t517+t520+t526+t530+t553+t567+t576+t578+t579+t593+t621+t625+t636+t640+t712+t757+t763+t767+t777;
    T t894 = t123+t124+t168+t169+t257+t282+t438+t456+t497+t507+t518+t521+t555+t583+t591+t608+t641+t654+t658+t668+t670+t677+t708+t729+t733+t736+t740;
    T t895 = t109+t111+t174+t175+t305+t321+t407+t450+t495+t505+t515+t525+t556+t572+t599+t603+t620+t623+t629+t666+t673+t680+t707+t752+t755+t769+t775, t862 = t860*tx, t863 = t861*tx, t864 = t859*ty, t865 = t860*ty, t866 = t859*tz, t867 = t861*tz;
    T t872 = t833+t856, t873 = t839+t851, t874 = t838+t854, t885 = t82+t132+t180+t186+t193+t195+t234+t346+t482+t546+t549, t886 = t77+t133+t181+t188+t196+t210+t241+t347+t476+t536+t544;
    T t896 = t112+t114+t167+t250+t258+t260+t410+t437+t449+t455+t506+t513+t514+t516+t519+t552+t598+t602+t622+t626+t639+t643+t762+t764+t785+t815+t817;
    T t897 = t113+t116+t177+t310+t312+t322+t409+t425+t449+t463+t493+t494+t504+t511+t512+t554+t600+t604+t610+t619+t651+t655+t749+t751+t784+t814+t819;
    T t898 = t115+t117+t162+t263+t289+t308+t422+t439+t455+t463+t509+t510+t522+t531+t533+t557+t592+t596+t627+t631+t653+t657+t737+t739+t783+t816+t818, t868 = -t866, t875 = t837+t864, t876 = t835+t867, t877 = t841+t862, t879 = t852+t868;
    
    fdrv(0, 0) = -t247*(t855-t865)-t874*tx*ty+t876*tx*tz; fdrv(0, 1) = t246*t874+t876*ty*tz+tx*ty*(t855-t865); fdrv(0, 2) = -t245*t876-t874*ty*tz+tx*tz*(t855-t865);
    fdrv(0, 3) = t116*-2.0+t118+t125+t455+t613+t762+t797-t798+t800+t825+t831+t4*t51*6.0-t7*t50*6.0-t21*t48*4.0-t15*t69+t46*t54*3.0+t48*t52*3.0+t22*t100-t14*t126*3.0-t14*t127*3.0-t15*t129*2.0-t18*t126*2.0-t18*t127*2.0-t22*t127*2.0+t18*t135+t18*t137+t22*t135+t17*t204+t15*t329+t17*t343+t17*t472+t9*t20*t32+t9*t21*t32+t14*t98*tx+t14*t103*tx-t7*t18*ty*4.0+t21*t47*ty;
    fdrv(0, 4) = t122*-8.0+t298+t451+t501+t532+t615+t662+t753+t779+t803+t824-t7*t52*4.0-t19*t48*2.0-t20*t49*4.0+t14*t70-t18*t69*5.0-t44*t54*2.0+t46*t55*3.0+t54*t68-t55*t67*4.0+t22*t102-t15*t126*4.0-t15*t127*4.0-t16*t126*3.0-t16*t127+t14*t131-t18*t129*6.0-t22*t129*4.0+t16*t136+t22*t131+t29*t166+t52*t188+t20*t472+t596*tz+t7*t20*t32+t18*t32*t36-t14*t44*tx*2.0-t7*t15*ty*6.0+t18*t48*ty*3.0;
    fdrv(0, 5) = t123*8.0+t254+t278+t528+t574+t594+t661+t709+t766+t778+t799+t827+t4*t54*4.0-t18*t49*4.0-t21*t69+t14*t77-t22*t75*5.0-t53*t67*4.0+t21*t102+t52*t74-t17*t127*4.0-t19*t126-t19*t127*3.0+t14*t133+t19*t128-t20*t129*4.0-t21*t129*6.0+t18*t133+t21*t131*3.0+t32*t178+t54*t197+t18*t347+t4*t21*t29-t14*t45*tx*2.0+t38*t55*ty+t4*t17*tz*6.0-t15*t44*tz*4.0-t16*t44*tz*2.0+t36*t50*tz;
    fdrv(0, 6) = tdx*(t110*2.0-t119*2.0+t121*2.0+t172-t179+t315+t417-t419+t461+t462+t499+t524-t569+t607+t612+t646+t675+t731-t3*t55*2.0+t14*t46*1.2E+1+t14*t48*1.2E+1-t3*t20*tz*2.0)+t894*tdy+t892*tdz;
    fdrv(0, 7) = t894*tdx+t898*tdz-tdy*(t108*2.0E+1+t248+t303-t315+t419-t435+t442-t461-t462+t500+t582+t611+t734+t750+t758+t771-t4*t17*4.0+t7*t16*4.0+t9*t18*2.4E+1+t10*t20*4.0+t15*t44*1.2E+1-t14*t48*2.0-t18*t46*1.2E+1-t22*t48*2.0+t17*t97);
    fdrv(0, 8) = tdz*(t119*-2.0E+1-t271+t300+t417+t435-t442+t461+t462+t492+t524+t593-t636+t638+t649+t730-t771-t7*t15*4.0+t4*t19*4.0+t6*t21*4.0+t8*t22*2.4E+1-t15*t44*4.0-t17*t45*1.2E+1+t22*t48*1.2E+1+t36*t50-t3*t20*tz*1.2E+1)+t892*tdx+t898*tdy;
    fdrv(0, 9) = tdx*(t132*6.0-t134*6.0+t206+t342+t466+t474+t480+t549+t683+t685+t805+t806+t30*t36+t33*t38-t32*t67*2.0+t26*t98+t26*t103)-t888*tdy*tx+t886*tdz*tx;
    fdrv(0, 10) = tdy*(t71*-5.0-t83*8.0+t206+t474+t546+t683+t805+t808+t4*t30*4.0+t7*t33-t9*t32*4.0+t26*t48+t29*t48*3.0-t127*tx+t47*ty*6.0+t190*ty+t188*tz)+t883*tdx*ty+t886*tdz*ty;
    fdrv(0, 11) = tdz*(t78*-5.0+t82*8.0-t134*2.0+t338+t466+t480+t685+t806+t808+t4*t30+t7*t33*4.0-t9*t32*6.0+t26*t46+t32*t46*3.0-t126*tx+t102*ty+t199*tz)+t883*tdx*tz-t888*tdy*tz;
    fdrv(0, 12) = -tdy*(t220+t357+t403+t404+t562+t701+t725+t20*t33+t32*t50*4.0+t213*ty+t395*ty)-tdz*(t224+t369+t391+t392+t563+t695+t727+t21*t30+t29*t51*4.0+t405*ty+t213*tz)-t247*tdx*(t52*2.0+t54*2.0+t14*tx*3.0+t15*ty+t17*tz);
    fdrv(0, 13) = tdx*(t88+t154+t155+t184+t217+t354+t358+t374+t378+t389+t564+t687+t704+t725)+tdy*(t87+t152+t153+t183+t221+t364+t371+t380+t396+t401+t559+t693+t703+t723)+tdz*(t148+t149+t213+t218+t353+t362+t373+t390+t395+t402+t485+t689+t694+t702);
    fdrv(0, 14) = tdx*(t89+t150+t151-t215-t360+t363-t367+t370-t377+t448+t566+t688+t698+t727)+tdz*(t87+t142+t143-t222-t382+t386-t388+t393-t405+t446+t558+t696+t700+t724)-tdy*(-t146-t147+t213+t223+t359+t361+t368+t384+t387+t406+t484-t697+t780+t782);
    fdrv(0, 15) = t820*t871*tx; fdrv(0, 16) = t820*t871*ty; fdrv(0, 17) = t820*t871*tz; fdrv(0, 18) = rdx*t820*t871+t11*t488*t820+t12*t488*t820+t13*t488*t820+t11*t871*tx*2.0+t12*t871*tx*2.0+t13*t871*tx*2.0;
    fdrv(0, 19) = rdy*t820*t871-t11*t820*t845-t12*t820*t845-t13*t820*t845+t11*t871*ty*2.0+t12*t871*ty*2.0+t13*t871*ty*2.0; fdrv(0, 20) = rdz*t820*t871+t11*t820*t843+t12*t820*t843+t13*t820*t843+t11*t871*tz*2.0+t12*t871*tz*2.0+t13*t871*tz*2.0;
    fdrv(0, 24) = -t247*t804*t820; fdrv(0, 25) = t804*(t58+t90+t91+t156+t228+t230); fdrv(0, 26) = -t804*(t57-t92-t93-t157+t226+t229); fdrv(1, 0) = -t247*t877+t879*tx*ty-t872*tx*tz; fdrv(1, 1) = -t246*t879+t877*tx*ty-t872*ty*tz;
    fdrv(1, 2) = t245*t872+t877*tx*tz+t879*ty*tz;
    fdrv(1, 3) = t111*8.0+t299+t320+t498+t664+t676+t678+t706+t732+t735+t803+t823+t7*t52*6.0-t22*t45*4.0-t14*t70*5.0+t18*t69+t44*t54*3.0+t48*t50*3.0-t17*t82-t46*t55*2.0+t15*t101+t17*t103-t54*t68*4.0+t55*t67-t16*t128*4.0-t14*t131*6.0+t18*t129-t18*t131*4.0-t15*t135+t22*t129+t15*t137-t22*t131*4.0+t15*t323+t15*t327+t18*t537-t4*t14*t29*3.0+t9*t17*t32+t14*t96*tx+t38*t53*ty;
    fdrv(1, 4) = t105-t112*2.0+t125+t463+t647+t737+t796+t802+t821+t832-t3*t53*6.0+t7*t50*4.0-t15*t45*4.0+t44*t55*3.0-t46*t54*2.0-t20*t82+t14*t127+t18*t126*4.0-t15*t131*2.0-t16*t131*2.0+t22*t127+t14*t136-t18*t135*3.0-t22*t135*2.0+t16*t197+t16*t324+t20*t349+t16*t465+t14*t538-t4*t16*t29*3.0+t7*t17*t32+t7*t19*t32+t19*t45*tx+t7*t18*ty*6.0+t15*t48*ty*3.0+t18*t104*ty-t3*t22*tz*4.0;
    fdrv(1, 5) = t119*-8.0+t300+t460+t492+t499-t567+t649+t675+t730+t731+t795+t830-t3*t55*4.0-t19*t45*4.0+t18*t83-t22*t82*5.0+t14*t103-t51*t68*4.0+t50*t74+t21*t126+t15*t133-t17*t131*4.0+t16*t133-t19*t131*6.0-t20*t135*4.0-t21*t135*3.0+t32*t179+t55*t205+t21*t331+t15*t347+t15*t444+t19*t465+t666*ty+t19*t29*t59+t19*t44*tx-t18*t47*ty*2.0-t3*t20*tz*6.0-t16*t46*tz*4.0+t42*t44*tz;
    fdrv(1, 6) = tdx*(t107*-2.0E+1+t253-t316-t412+t428+t441+t457+t458+t508+t523+t581-t583+t585+t677-t773+t774+t7*t14*2.4E+1-t3*t20*4.0+t9*t15*4.0+t10*t17*4.0+t14*t44*1.2E+1-t16*t46*1.2E+1-t20*t47*4.0+t40*t53-t4*t17*tz*1.2E+1)+t890*tdy+t897*tdz;
    fdrv(1, 7) = tdy*(t106*2.0-t107*2.0+t124*2.0-t160+t178+t269-t427+t428+t457+t458+t523+t527+t570+t587+t648-t661+t679+t743-t4*t54*2.0+t18*t44*1.2E+1+t18*t49*1.2E+1-t4*t17*tz*2.0)+t890*tdx+t895*tdz;
    fdrv(1, 8) = t897*tdx+t895*tdy-tdz*(t123*2.0E+1+t257-t269+t278+t412+t427-t441-t457-t458+t528+t574+t672+t744+t766+t773+t776+t2*t19*4.0+t3*t21*4.0-t7*t18*4.0+t5*t22*2.4E+1-t14*t44*2.0-t18*t44*2.0+t20*t47*1.2E+1-t22*t49*1.2E+1+t16*t98);
    fdrv(1, 9) = tdx*(t72*-5.0+t77*8.0-t130*2.0+t339+t473+t479+t681+t807+t809+t7*t32*4.0+t9*t33+t2*t57+t29*t49+t26*t96-t45*tx*6.0-t135*ty+t348*ty)+t882*tdy*tx-t889*tdz*tx;
    fdrv(1, 10) = tdy*(t130*-6.0+t133*6.0+t199+t343+t468+t473+t479+t536+t681+t686+t807+t811+t2*t30*4.0+t26*t44*2.0+t33*t40-t32*t68*2.0+t29*t104)+t884*tdx*ty-t889*tdz*ty;
    fdrv(1, 11) = tdz*(t75*-8.0-t84*5.0+t199+t468+t544+t686+t809+t811+t2*t30+t7*t32*6.0+t9*t33*4.0+t29*t38+t26*t44+t32*t44*3.0-t45*tx*4.0+t207*tz+t29*t59*tx)+t884*tdx*tz+t882*tdy*tz;
    fdrv(1, 12) = tdx*(t88+t154+t155-t217+t354+t358-t374-t378-t389+t447+t564+t687+t704+t725)+tdy*(t87+t152+t153-t221+t364+t371-t380-t396-t401+t446+t559+t693+t703+t723)-tdz*(-t148-t149+t213+t218+t353+t362+t373+t390+t395+t402+t485-t702+t780+t781);
    fdrv(1, 13) = -tdx*(t214+t372+t393+t394+t561+t700+t723+t17*t33+t16*t57+t32*t52*4.0+t373*tx)-tdz*(t224+t369+t370+t391+t563+t688+t728+t19*t27+t184*tx+t397*tx+t218*tz)-t246*tdy*(t55*2.0+t16*tx+t18*ty*3.0+t41*ty+t20*tz);
    fdrv(1, 14) = tdx*(t144+t145+t218+t223+t355+t365+t376+t383+t398+t399+t483+t692+t694+t699)+tdy*(t89+t140+t141+t185+t216+t356+t375+t379+t385+t392+t565+t691+t695+t728)+tdz*(t88+t138+t139+t184+t219+t366+t381+t397+t400+t403+t560+t690+t701+t726);
    fdrv(1, 15) = -t820*t870*tx; fdrv(1, 16) = -t820*t870*ty; fdrv(1, 17) = -t820*t870*tz; fdrv(1, 18) = -rdx*t820*t870+t11*t820*t842+t12*t820*t842+t13*t820*t842-t11*t870*tx*2.0-t12*t870*tx*2.0-t13*t870*tx*2.0;
    fdrv(1, 19) = -rdy*t820*t870+t11*t487*t820+t12*t487*t820+t13*t487*t820-t11*t870*ty*2.0-t12*t870*ty*2.0-t13*t870*ty*2.0; fdrv(1, 20) = -rdz*t820*t870-t11*t820*t847-t12*t820*t847-t13*t820*t847-t11*t870*tz*2.0-t12*t870*tz*2.0-t13*t870*tz*2.0;
    fdrv(1, 24) = -t804*(t58-t90-t91-t156+t228+t230); fdrv(1, 25) = -t246*t804*t820; fdrv(1, 26) = t804*(t56+t94+t95+t158+t225+t227); fdrv(2, 0) = t247*t873+t875*tx*ty+tx*tz*(t853-t863); fdrv(2, 1) = -t246*t875-t873*tx*ty+ty*tz*(t853-t863);
    fdrv(2, 2) = -t245*(t853-t863)-t873*tx*tz+t875*ty*tz;
    fdrv(2, 3) = t107*-8.0+t253+t454+t508+t527+t570+t585-t641+t743+t774+t799+t826-t4*t54*6.0-t15*t46*4.0-t14*t77*5.0+t22*t75+t46*t51*3.0+t18*t96+t53*t67-t52*t74*4.0-t14*t133*6.0+t20*t129+t21*t129-t18*t133*4.0+t17*t135-t19*t136*4.0+t15*t180+t17*t327+t15*t478+t576*tz-t7*t14*t32*3.0-t7*t18*t32*2.0-t7*t22*t32*2.0+t21*t32*t36+t15*t32*t64+t14*t97*tx+t15*t47*ty-t21*t48*ty*4.0-t4*t17*tz*4.0;
    fdrv(2, 4) = t108*8.0+t248+t297+t500+t569+t571+t611+t711+t742+t750+t795+t829+t3*t55*6.0-t14*t46*4.0-t16*t77+t44*t53*3.0-t18*t83*5.0+t22*t82+t16*t96+t20*t99+t51*t68-t50*t74*4.0+t20*t127-t15*t133*4.0+t17*t131-t16*t133*6.0+t19*t131-t21*t137*4.0+t21*t327+t20*t341+t22*t549+t577*tz+t594*tz-t7*t15*t32*2.0-t7*t16*t32-t9*t18*t32*3.0+t16*t45*tx-t19*t48*ty*4.0+t18*t102*ty;
    fdrv(2, 5) = t105-t115*2.0+t118+t449-t455+t590+t751+t798+t801+t822+t828-t4*t51*4.0+t3*t53*4.0-t19*t77-t48*t52*2.0+t14*t126+t14*t128+t18*t126+t15*t131*3.0+t22*t127*4.0-t19*t133*2.0+t22*t135*4.0+t21*t187+t19*t328+t18*t331+t21*t342+t15*t445+t21*t478+t14*t548+t18*t550+t4*t15*t29+t4*t16*t29-t7*t19*t32*3.0-t9*t21*t32*3.0+t16*t44*tx+t3*t22*tz*6.0-t4*t22*tz*6.0;
    fdrv(2, 6) = t896*tdy+t893*tdz-tdx*(t111*2.0E+1-t268+t305+t320-t408+t418+t436-t452-t453+t498+t637+t664+t735+t746+t756+t761+t4*t14*2.4E+1+t6*t15*4.0-t3*t22*4.0+t8*t17*4.0-t14*t45*1.2E+1-t18*t47*2.0+t19*t48*1.2E+1-t22*t47*2.0+t21*t104);
    fdrv(2, 7) = tdy*(t122*-2.0E+1-t273+t298+t408-t418+t434+t452+t453+t496+t532+t572+t615+t671-t673+t753-t761+t2*t16*4.0+t3*t18*2.4E+1+t5*t20*4.0-t4*t22*4.0+t18*t47*1.2E+1-t19*t48*4.0-t21*t49*1.2E+1+t43*t45-t7*t15*ty*1.2E+1)+t896*tdx+t891*tdz;
    fdrv(2, 8) = tdz*(t109*2.0+t120*2.0-t122*2.0+t159-t166+t268+t434-t436+t452+t453+t496+t501+t580+t589+t617+t662-t676+t779-t7*t52*2.0+t22*t45*1.2E+1+t22*t47*1.2E+1-t7*t15*ty*2.0)+t893*tdx+t891*tdy;
    fdrv(2, 9) = tdx*(t70*-8.0-t79*5.0+t189+t481+t545+t682+t810+t812-t4*t29*4.0+t6*t33+t2*t58+t32*t35+t29*t47+t26*t97+t32*t477+t44*tx*6.0+t334*tz)+t885*tdy*tx+t881*tdz*tx;
    fdrv(2, 10) = tdy*(t69*8.0-t85*5.0-t131*2.0+t335+t467+t475+t684+t810+t813-t4*t29*6.0+t2*t33+t6*t58+t26*t45+t29*t45*3.0+t29*t102+t32*t471+t96*tx)-t887*tdx*ty+t881*tdz*ty;
    fdrv(2, 11) = tdz*(t129*6.0-t131*6.0+t189+t324+t467+t475+t481+t537+t682+t684+t812+t813+t2*t33*4.0+t6*t33*4.0+t26*t45*2.0+t29*t47*2.0-t29*t74*2.0)-t887*tdx*tz+t885*tdy*tz;
    fdrv(2, 12) = tdx*(t89+t150+t151+t185+t215+t360+t363+t367+t370+t377+t566+t688+t698+t727)+tdy*(t146+t147+t213+t223+t359+t361+t368+t384+t387+t406+t484+t689+t697+t699)+tdz*(t87+t142+t143+t183+t222+t382+t386+t388+t393+t405+t558+t696+t700+t724);
    fdrv(2, 13) = tdy*(t89+t140+t141-t216-t356-t375-t379+t385+t392+t448+t565+t691+t695+t728)+tdz*(t88+t138+t139-t219-t366-t381-t397+t400+t403+t447+t560+t690+t701+t726)-tdx*(-t144-t145+t218+t223+t355+t365+t376+t383+t398+t399+t483-t692+t781+t782);
    fdrv(2, 14) = -tdx*(t214+t371+t372+t394+t561+t693+t724+t15*t30+t19*t58+t29*t54*4.0+t359*tx)-tdy*(t220+t357+t358+t404+t562+t687+t726+t16*t27+t21*t58+t185*tx+t375*tx)-t245*tdz*(t19*tx+t21*ty+t22*tz*3.0+t41*tz+t42*tz); fdrv(2, 15) = t820*t869*tx;
    fdrv(2, 16) = t820*t869*ty; fdrv(2, 17) = t820*t869*tz; fdrv(2, 18) = rdx*t820*t869-t11*t820*t846-t12*t820*t846-t13*t820*t846+t11*t869*tx*2.0+t12*t869*tx*2.0+t13*t869*tx*2.0;
    fdrv(2, 19) = rdy*t820*t869+t11*t820*t844+t12*t820*t844+t13*t820*t844+t11*t869*ty*2.0+t12*t869*ty*2.0+t13*t869*ty*2.0; fdrv(2, 20) = rdz*t820*t869+t11*t486*t820+t12*t486*t820+t13*t486*t820+t11*t869*tz*2.0+t12*t869*tz*2.0+t13*t869*tz*2.0;
    fdrv(2, 24) = t804*(t57+t92+t93+t157+t226+t229); fdrv(2, 25) = -t804*(t56-t94-t95-t158+t225+t227); fdrv(2, 26) = -t245*t804*t820;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f23(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = ox*ty, t4 = oy*tx, t5 = ox*tz, t6 = oy*ty, t7 = oz*tx, t8 = oy*tz, t9 = oz*ty, t10 = oz*tz, t11 = rdx*tx, t12 = rdy*ty, t13 = rdz*tz, t14 = tx*tx, t15 = tx*tx*tx, t17 = ty*ty, t18 = ty*ty*ty, t20 = tz*tz, t21 = tz*tz*tz;
    T t29 = rx*tx*3.0, t30 = ry*ty*3.0, t31 = rz*tz*3.0, t38 = rx*tx*ty, t39 = rx*tx*tz, t40 = ry*tx*ty, t41 = ry*ty*tz, t42 = rz*tx*tz, t43 = rz*ty*tz, t16 = t14*t14, t19 = t17*t17, t22 = t20*t20, t23 = t3*2.0, t24 = t4*2.0, t25 = t5*2.0;
    T t26 = t7*2.0, t27 = t8*2.0, t28 = t9*2.0, t32 = t2*ty, t33 = t2*tz, t34 = t4*ty, t35 = t6*tz, t36 = t7*tz, t37 = t9*tz, t44 = -t4, t46 = -t7, t48 = -t9, t50 = t2*t14, t51 = t3*ty, t52 = t4*tx, t53 = t3*t17, t54 = t4*t14, t55 = t3*t18;
    T t56 = t4*t15, t57 = t5*tz, t58 = t7*tx, t59 = t5*t20, t60 = t6*t17, t61 = t7*t14, t62 = t5*t21, t63 = t7*t15, t64 = t8*tz, t65 = t9*ty, t66 = t8*t20, t67 = t9*t17, t68 = t8*t21, t69 = t9*t18, t70 = t10*t20, t71 = rx*t15, t72 = rx*t17;
    T t73 = ry*t14, t74 = rx*t20, t75 = rz*t14, t76 = ry*t18, t77 = ry*t20, t78 = rz*t17, t79 = rz*t21, t80 = t38*2.0, t81 = t39*2.0, t82 = t40*2.0, t83 = t41*2.0, t84 = t42*2.0, t85 = t43*2.0, t107 = t2*t17, t108 = t2*t20, t110 = t3*t20;
    T t112 = t4*t20, t114 = t7*t17, t116 = t6*t20, t134 = rx*t14*3.0, t137 = ry*t17*3.0, t140 = rz*t20*3.0, t155 = t14+t17, t156 = t14+t20, t157 = t17+t20, t192 = t4*t17*3.0, t206 = t7*t20*3.0, t210 = t9*t20*3.0, t213 = t17*t29;
    T t214 = rx*t18*tx*4.0, t217 = t20*t29, t218 = t14*t30, t219 = rx*t21*tx*4.0, t220 = ry*t15*ty*4.0, t223 = t20*t30, t224 = t14*t31, t225 = ry*t21*ty*4.0, t226 = rz*t15*tz*4.0, t228 = t17*t31, t229 = rz*t18*tz*4.0, t257 = t2+t6, t258 = t2+t10;
    T t259 = t6+t10, t279 = t4*t17*-2.0, t289 = t7*t20*-2.0, t291 = t9*t20*-2.0, t303 = t2*t30*tx, t327 = t4*t29*tz, t329 = t7*t29*ty, t332 = ry*t4*t17*6.0, t343 = t3*t30*tz, t345 = t2*t31*tx, t385 = t6*t31*ty, t392 = rz*t7*t20*6.0;
    T t396 = rz*t9*t20*6.0, t440 = t4*t39*1.2E+1, t442 = t7*t38*1.2E+1, t452 = t3*t41*1.2E+1, t454 = t7*t40*1.2E+1, t463 = t3*t43*1.2E+1, t465 = t4*t42*1.2E+1, t482 = t29+t30, t483 = t29+t31, t484 = t30+t31, t494 = t11+t12+t13, t45 = -t24;
    T t47 = -t26, t49 = -t28, t86 = t2*t71, t87 = ry*t50, t88 = rx*t54, t89 = ry*t53, t90 = rz*t50, t91 = rx*t60, t92 = rx*t61, t93 = ry*t59, t94 = rz*t53, t95 = rx*t66, t96 = rz*t54, t97 = rx*t67, t98 = ry*t61, t99 = t6*t76, t100 = rz*t59;
    T t101 = rz*t60, t102 = rx*t70, t103 = ry*t67, t104 = rz*t66, t105 = ry*t70, t106 = t10*t79, t109 = t34*tx, t111 = t51*tz, t113 = t52*tz, t115 = t58*ty, t117 = t36*tx, t118 = t37*ty, t119 = t72*tx, t120 = t74*tx, t121 = t73*ty, t122 = t77*ty;
    T t123 = t75*tz, t124 = t78*tz, t125 = t32*tz*2.0, t126 = t24*ty*tz, t127 = t26*ty*tz, t128 = t53*4.0, t129 = t54*4.0, t130 = t59*4.0, t131 = t61*4.0, t132 = t66*4.0, t133 = t67*4.0, t135 = t71*4.0, t136 = rx*t16*5.0, t138 = t76*4.0;
    T t139 = ry*t19*5.0, t141 = t79*4.0, t142 = rz*t22*5.0, t143 = -t32, t144 = -t33, t145 = -t34, t146 = -t35, t147 = -t36, t148 = -t37, t149 = -t80, t150 = -t81, t151 = -t82, t152 = -t83, t153 = -t84, t154 = -t85, t158 = rx*t50*2.0;
    T t159 = rx*t50*4.0, t166 = ry*t60*2.0, t170 = ry*t60*4.0, t180 = rz*t70*2.0, t181 = rz*t70*4.0, t182 = t107*2.0, t183 = t32*tx*2.0, t184 = t107*3.0, t185 = t32*tx*3.0, t186 = t108*2.0, t187 = t33*tx*2.0, t188 = t17*t24, t189 = t24*tx*ty;
    T t190 = t108*3.0, t191 = t33*tx*3.0, t194 = t110*4.0, t196 = t112*4.0, t198 = t114*4.0, t200 = t116*2.0, t201 = t35*ty*2.0, t202 = t20*t26, t203 = t26*tx*tz, t204 = t116*3.0, t205 = t35*ty*3.0, t208 = t20*t28, t209 = t28*ty*tz;
    T t230 = rx*t32*tz*4.0, t231 = ry*t32*tz*4.0, t232 = rx*t34*tz*4.0, t233 = ry*t32*tz*6.0, t234 = rx*t34*tz*6.0, t235 = rz*t32*tz*4.0, t236 = ry*t34*tz*4.0, t237 = rx*t36*ty*4.0, t238 = rz*t32*tz*6.0, t239 = rx*t36*ty*6.0, t240 = rz*t34*tz*4.0;
    T t241 = ry*t36*ty*4.0, t242 = rz*t34*tz*6.0, t243 = ry*t36*ty*6.0, t244 = rz*t36*ty*4.0, t245 = -t55, t246 = t15*t44, t249 = -t62, t250 = t15*t46, t253 = -t68, t254 = t18*t48, t261 = rx*t53*-4.0, t262 = rx*t59*-4.0, t263 = ry*t54*-4.0;
    T t271 = ry*t66*-4.0, t272 = rz*t61*-4.0, t273 = rz*t67*-4.0, t293 = t20*t72, t294 = t20*t73, t295 = t17*t75, t296 = t2*t72*2.0, t298 = t2*t72*6.0, t299 = rx*t32*tx*6.0, t300 = t2*t74*2.0, t304 = t4*t72*3.0, t305 = t29*t34, t306 = t2*t74*6.0;
    T t307 = rx*t33*tx*6.0, t308 = ry*t107*6.0, t309 = ry*t32*tx*6.0, t310 = t4*t72*6.0, t312 = t2*t77*2.0, t314 = t2*t78*2.0, t316 = t24*t74, t317 = t24*t39, t319 = t24*t40, t320 = t26*t72, t321 = t26*t38, t322 = t2*t77*3.0, t324 = t2*t78*3.0;
    T t326 = t4*t74*3.0, t328 = t7*t72*3.0, t330 = t3*t74*4.0, t334 = t23*t77, t335 = t23*t41, t336 = t6*t74*2.0, t340 = ry*t17*t26, t341 = t26*t40, t342 = t3*t77*3.0, t346 = t6*t74*3.0, t348 = t4*t78*3.0, t350 = t7*t74*3.0, t351 = t29*t36;
    T t352 = ry*t114*3.0, t353 = t30*t58, t354 = t4*t77*4.0, t356 = rz*t108*6.0, t357 = rz*t33*tx*6.0, t358 = t7*t74*6.0, t360 = rz*t20*t23, t361 = t23*t43, t362 = t6*t77*2.0, t364 = rz*t20*t24, t365 = t24*t42, t370 = rz*t110*3.0, t371 = t31*t51;
    T t372 = rz*t112*3.0, t373 = t31*t52, t374 = t9*t74*3.0, t376 = t7*t77*3.0, t378 = t7*t78*4.0, t380 = t6*t77*6.0, t381 = ry*t35*ty*6.0, t383 = t26*t42, t386 = t9*t77*3.0, t387 = t30*t37, t388 = rz*t116*6.0, t389 = rz*t35*ty*6.0;
    T t390 = t9*t77*6.0, t395 = t28*t43, t398 = t20*t32*2.0, t399 = t17*t33*2.0, t400 = t20*t24*ty, t402 = t17*t26*tz, t404 = rx*t32*tz*8.0, t413 = ry*t34*tz*8.0, t418 = rz*t36*ty*8.0, t421 = t17*t52*3.0, t422 = t20*t51*6.0, t423 = t20*t52*6.0;
    T t424 = t17*t58*6.0, t426 = t20*t58*3.0, t427 = t20*t65*3.0, t433 = ry*t33*tx*-2.0, t435 = rz*t32*tx*-2.0, t438 = -t332, t439 = t4*t74*1.2E+1, t441 = t7*t72*1.2E+1, t444 = rx*t35*ty*-2.0, t445 = t4*t78*-2.0, t451 = t3*t77*1.2E+1;
    T t453 = ry*t114*1.2E+1, t455 = t9*t74*-2.0, t457 = t7*t77*-2.0, t462 = rz*t110*1.2E+1, t464 = rz*t112*1.2E+1, t468 = -t392, t469 = -t396, t473 = rx*t155, t474 = rx*t156, t475 = ry*t155, t476 = ry*t157, t477 = rz*t156, t478 = rz*t157;
    T t486 = t32*t77*2.0, t488 = t33*t78*2.0, t495 = t107*tx*-3.0, t496 = t108*tx*-3.0, t501 = t116*ty*-3.0, t504 = -t440, t505 = -t442, t506 = -t452, t510 = t20+t155, t514 = t29*t107, t515 = t29*t108, t517 = t4*t213, t518 = t52*t137;
    T t520 = t7*t217, t521 = t30*t116, t523 = t9*t223, t524 = t58*t140, t525 = t65*t140, t526 = t32*t39*-2.0, t527 = ry*t279*tz, t528 = rz*t289*ty, t529 = t482*tx, t530 = t482*ty, t531 = t483*tx, t532 = t483*tz, t533 = t484*ty, t534 = t484*tz;
    T t550 = t72+t74+t82+t84+t134, t551 = t73+t77+t80+t85+t137, t552 = t75+t78+t81+t83+t140, t161 = t88*4.0, t163 = t89*4.0, t165 = t92*4.0, t167 = t93*4.0, t168 = t94*4.0, t169 = t95*4.0, t171 = t96*4.0, t172 = t97*4.0, t173 = t98*4.0;
    T t174 = t100*4.0, t176 = t103*4.0, t178 = t104*4.0, t193 = t109*3.0, t195 = t111*4.0, t197 = t113*4.0, t199 = t115*4.0, t207 = t117*3.0, t211 = t118*3.0, t212 = t119*2.0, t215 = t120*2.0, t216 = t121*2.0, t221 = t122*2.0, t222 = t123*2.0;
    T t227 = t124*2.0, t247 = -t128, t248 = -t129, t251 = -t130, t252 = -t131, t255 = -t132, t256 = -t133, t260 = -t86, t264 = -t99, t274 = -t106, t275 = -t182, t276 = -t183, t277 = -t186, t278 = -t187, t280 = t109*-2.0, t281 = -t194;
    T t283 = -t196, t285 = -t198, t287 = -t200, t288 = -t201, t290 = t117*-2.0, t292 = t118*-2.0, t297 = rx*t183, t301 = rx*t187, t302 = ry*t184, t311 = rx*t109*6.0, t318 = ry*t188, t323 = ry*t191, t325 = rz*t185, t333 = ry*t109*6.0;
    T t339 = rz*t189, t344 = rz*t190, t347 = rx*t205, t359 = rx*t117*6.0, t363 = ry*t201, t367 = rx*t209, t369 = ry*t203, t382 = rz*t202, t384 = rz*t204, t391 = ry*t118*6.0, t393 = rz*t117*6.0, t394 = rz*t208, t397 = rz*t118*6.0, t401 = t126*tx;
    T t403 = t127*tx, t405 = -t231, t406 = -t232, t407 = -t233, t408 = -t234, t409 = -t235, t410 = -t237, t411 = -t238, t412 = -t239, t414 = -t240, t415 = -t241, t416 = -t242, t417 = -t243, t428 = -t299, t429 = -t307, t430 = -t308, t432 = -t312;
    T t434 = -t314, t436 = -t330, t437 = rx*t111*-4.0, t443 = -t336, t447 = -t354, t448 = ry*t113*-4.0, t449 = -t356, t459 = -t378, t460 = rz*t115*-4.0, t461 = -t381, t466 = -t388, t470 = -t404, t471 = -t413, t472 = -t418, t479 = t3+t45;
    T t480 = t5+t47, t481 = t8+t49, t487 = t316*ty, t490 = t320*tz, t491 = t365*ty, t492 = t341*tz, t497 = -t421, t498 = -t422, t499 = -t423, t500 = -t424, t502 = -t426, t503 = -t427, t507 = -t453, t508 = -t462, t509 = -t464, t535 = -t529;
    T t536 = -t530, t537 = -t531, t538 = -t532, t539 = -t533, t540 = -t534, t544 = t494*t510*tx*ty, t545 = t494*t510*tx*tz, t546 = t494*t510*ty*tz, t553 = t550*tdx*ty*tz, t554 = t551*tdy*tx*tz, t555 = t552*tdz*tx*ty, t265 = -t167, t266 = -t168;
    T t267 = -t169, t268 = -t171, t269 = -t172, t270 = -t173, t282 = -t195, t284 = -t197, t286 = -t199, t349 = rz*t193, t375 = rx*t211, t377 = ry*t207, t431 = -t311, t446 = rz*t280, t450 = -t359, t456 = rx*t292, t458 = ry*t290, t467 = -t391;
    T t516 = t302*tx, t519 = t344*tx, t522 = t384*ty, t547 = -t544, t548 = -t545, t549 = -t546, t556 = -t553, t557 = -t554, t558 = -t555, t559 = t40+t153+t473+t474+t535, t560 = t42+t151+t473+t474+t537, t561 = t38+t154+t475+t476+t536;
    T t562 = t43+t149+t475+t476+t539, t563 = t39+t152+t477+t478+t538, t564 = t41+t150+t477+t478+t540, t571 = t53+t110+t127+t185+t248+t279+t283, t572 = t54+t112+t127+t192+t247+t276+t281, t573 = t59+t111+t126+t191+t252+t285+t289;
    T t577 = t50+t60+t107+t109+t190+t204+t290+t292, t578 = t50+t70+t108+t117+t184+t211+t280+t287, t579 = t60+t70+t116+t118+t193+t207+t275+t277, t580 = t71+t76+t119+t121+t141+t217+t222+t223+t227, t581 = t71+t79+t120+t123+t138+t213+t216+t221+t228;
    T t582 = t76+t79+t122+t124+t135+t212+t215+t218+t224, t565 = t559*tx, t566 = t560*tx, t567 = t561*ty, t568 = t562*ty, t569 = t563*tz, t570 = t564*tz, t574 = t61+t114+t126+t206+t251+t278+t282, t575 = t66+t113+t125+t205+t256+t286+t291;
    T t576 = t67+t115+t125+t210+t255+t284+t288, t583 = t581*tdy*tx, t584 = t580*tdz*tx, t585 = t582*tdx*ty, t586 = t580*tdz*ty, t587 = t582*tdx*tz, t588 = t581*tdy*tz;
    T t598 = t96+t98+t236+t244+t265+t266+t317+t321+t348+t352+t372+t376+t433+t435+t470+t506+t508, t599 = t94+t97+t230+t244+t267+t268+t325+t329+t335+t340+t370+t374+t444+t445+t471+t504+t509;
    T t600 = t93+t95+t230+t236+t269+t270+t323+t327+t343+t347+t360+t364+t455+t457+t472+t505+t507, t601 = t100+t102+t159+t163+t241+t263+t298+t300+t309+t334+t345+t351+t371+t375+t382+t414+t431+t438+t443+t447;
    T t602 = t89+t91+t159+t174+t240+t272+t296+t303+t305+t306+t318+t342+t346+t357+t361+t415+t450+t456+t459+t468, t603 = t104+t105+t161+t170+t237+t261+t310+t316+t333+t362+t373+t377+t385+t387+t394+t409+t428+t430+t432+t436;
    T t604 = t87+t88+t170+t178+t235+t273+t297+t302+t304+t319+t322+t326+t365+t380+t389+t410+t458+t460+t467+t469, t605 = t101+t103+t165+t181+t232+t262+t320+t349+t353+t358+t363+t384+t386+t393+t395+t405+t429+t434+t437+t449;
    T t606 = t90+t92+t176+t181+t231+t271+t301+t324+t328+t341+t344+t350+t383+t390+t397+t406+t446+t448+t461+t466, t589 = -t583, t590 = -t584, t591 = -t585, t592 = -t586, t593 = -t587, t594 = -t588, t595 = t566+t568, t596 = t565+t570;
    T t597 = t567+t569, t607 = t556+t592+t594, t608 = t557+t590+t593, t609 = t558+t589+t591;
    
    fdrv(0, 0) = -t17*t565-t20*t566-t157*t597; fdrv(0, 1) = -t20*t560*ty+t156*t559*ty+t597*tx*ty; fdrv(0, 2) = -t17*t559*tz+t155*t560*tz+t597*tx*tz;
    fdrv(0, 3) = t264+t274+t486+t488+t514+t515+rx*t55+rx*t62-t34*t42*3.0-t36*t40*3.0+t2*t76*2.0+t2*t79*2.0-t6*t79-t34*t74*2.0-t36*t72*2.0-t6*t122+t78*t146-rx*t4*t18*2.0-rx*t7*t21*2.0-rx*t14*t34*4.0-rx*t14*t36*4.0-ry*t17*t52*3.0+ry*t21*t48+ry*t17*t148-rz*t20*t58*3.0+t23*t74*ty+rz*t20*t48*ty;
    fdrv(0, 4) = t516+t528+ry*t55*5.0+ry*t62-t36*t38*2.0-t4*t76*4.0+t32*t74*4.0-t34*t73*2.0-t34*t77*2.0+t44*t71+t32*t84+t44*t79-t52*t72*3.0-t4*t124*3.0+t51*t77*6.0+t3*t141+t44*t120+t44*t123+t73*t147+t168*tz+rx*t2*t18*4.0+rx*t14*t32*2.0-ry*t17*t36*3.0+ry*t21*t46+t2*t77*tx;
    fdrv(0, 5) = t519+t527+rz*t55+rz*t62*5.0+rz*t422-t34*t39*2.0-t7*t79*4.0+t33*t72*4.0-t36*t75*2.0-t36*t78*2.0+t46*t71+t46*t76-t7*t122*3.0-t58*t74*3.0+t46*t119+t46*t121+t75*t145+t163*tz+rx*t2*t21*4.0+rx*t14*t33*2.0+ry*t3*t21*4.0-rz*t20*t34*3.0+rz*t18*t44+ry*t125*tx+t2*t78*tx;
    fdrv(0, 6) = -t603*tdy-t605*tdz-tdx*(t89*-2.0+t91*2.0-t100*2.0+t102*2.0+t242+t243-t298-t306+t332+t336+t367+t392+rx*t109*1.2E+1+rx*t117*1.2E+1-t3*t43*2.0-t3*t77*2.0);
    fdrv(0, 7) = -t603*tdx-t598*tdz+tdy*(t89*2.0E+1+t158+t174+t309+t416+t417+t431+t451+t463+rx*t290-ry*t54*2.0+rz*t187+rz*t289+t2*t72*1.2E+1+t2*t74*4.0-t4*t77*2.0-ry*t4*t17*1.2E+1);
    fdrv(0, 8) = -t605*tdx-t598*tdy+tdz*(t100*2.0E+1+t158+t163+t357+t416+t417+t450+t451+t463+rx*t280+ry*t183+ry*t279-rz*t61*2.0+t2*t72*4.0+t2*t74*1.2E+1-t7*t78*2.0-rz*t7*t20*1.2E+1);
    fdrv(0, 9) = -tdx*(t245+t249+t400+t402+t495+t496+t18*t24+t21*t26+t14*t34*4.0+t14*t36*4.0-t20*t51*2.0)-t572*tdy*tx-t574*tdz*tx;
    fdrv(0, 10) = -tdy*(t55*-5.0+t249+t400+t495+t498+t4*t18*4.0+t7*t21+t14*t36+t17*t36*3.0-t108*tx+t14*t24*ty)-t579*tdx*ty-t574*tdz*ty;
    fdrv(0, 11) = -tdz*(t62*-5.0+t245+t402+t496+t498+t4*t18+t7*t21*4.0+t14*t34+t20*t34*3.0-t107*tx+t14*t26*tz)-t579*tdx*tz-t572*tdy*tz;
    fdrv(0, 12) = tdy*(t139+t214+t229+t294+ry*t22+t20*t38*4.0+t17*t73*3.0+t17*t77*6.0+t14*t85+t71*ty*2.0+t141*ty)+tdz*(t142+t219+t225+t295+rz*t19+t17*t39*4.0+t20*t75*3.0+t14*t83+t20*t78*6.0+t71*tz*2.0+t138*tz)+t157*t550*tdx; fdrv(0, 13) = t609;
    fdrv(0, 14) = t608; fdrv(0, 15) = -t510*tx*(t34+t36-t51-t57); fdrv(0, 16) = -t510*ty*(t34+t36-t51-t57); fdrv(0, 17) = -t510*tz*(t34+t36-t51-t57);
    fdrv(0, 18) = -rdx*t510*(t34+t36-t51-t57)-t11*tx*(t34+t36-t51-t57)*2.0-t12*tx*(t34+t36-t51-t57)*2.0-t13*tx*(t34+t36-t51-t57)*2.0-t11*t259*t510-t12*t259*t510-t13*t259*t510;
    fdrv(0, 19) = -rdy*t510*(t34+t36-t51-t57)-t11*ty*(t34+t36-t51-t57)*2.0-t12*ty*(t34+t36-t51-t57)*2.0-t13*ty*(t34+t36-t51-t57)*2.0-t11*t510*(t4-t23)-t12*t510*(t4-t23)-t13*t510*(t4-t23);
    fdrv(0, 20) = -rdz*t510*(t34+t36-t51-t57)-t11*tz*(t34+t36-t51-t57)*2.0-t12*tz*(t34+t36-t51-t57)*2.0-t13*tz*(t34+t36-t51-t57)*2.0-t11*t510*(t7-t25)-t12*t510*(t7-t25)-t13*t510*(t7-t25); fdrv(0, 24) = t157*t494*t510; fdrv(0, 25) = t547;
    fdrv(0, 26) = t548; fdrv(1, 0) = -t20*t562*tx+t157*t561*tx+t596*tx*ty; fdrv(1, 1) = -t14*t567-t20*t568-t156*t596; fdrv(1, 2) = -t14*t561*tz+t155*t562*tz+t596*ty*tz;
    fdrv(1, 3) = t517+t528+rx*t56*5.0+rx*t68+ry*t245+ry*t495-t32*t42*3.0-t36*t38*3.0-t3*t79+t24*t76-t32*t74*2.0+t34*t73*4.0+t34*t77*4.0+t52*t74*6.0-t51*t77+t4*t141+t24*t124+t72*t148-t94*tz+t171*tz-rx*t2*t18*2.0-rx*t14*t32*4.0+rx*t21*t48-ry*t17*t36*2.0+t6*t74*ty;
    fdrv(1, 4) = t260+t274-t486+t487+t491+t518+t521+ry*t56+ry*t68-t36*t40*2.0-t2*t76*4.0-t2*t79+t6*t79*2.0-t32*t73*2.0-t36*t72*3.0-t33*t78*3.0-t2*t119*3.0-t2*t120+t75*t144+rx*t21*t46+rx*t14*t147-ry*t9*t21*2.0-ry*t17*t37*4.0-rz*t20*t65*3.0+t24*t77*tx+rx*t14*t24*ty+rz*t20*t46*tx;
    fdrv(1, 5) = t522+t526+rz*t56+rz*t68*5.0+rz*t423-t9*t79*4.0-t37*t78*2.0+t48*t76+t52*t78-t65*t77*3.0+t75*t143+t161*tz+rx*t4*t21*4.0+rx*t18*t46+ry*t6*t21*4.0-ry*t17*t33*2.0+ry*t17*t35*2.0-rz*t2*t18-rz*t20*t32*3.0+ry*t109*tz*4.0+rz*t290*ty-t7*t74*ty*3.0+t24*t72*tz+rx*t14*t46*ty+ry*t17*t46*tx;
    fdrv(1, 6) = -t601*tdy-t599*tdz+tdx*(t88*2.0E+1+t166+t178+t310+t411+t412+t430+t439+t465-rx*t53*2.0+ry*t109*1.2E+1+ry*t292+rz*t201+rz*t291-t3*t74*2.0+t6*t77*4.0-rx*t32*tx*1.2E+1);
    fdrv(1, 7) = -t601*tdx-t606*tdz-tdy*(t87*2.0-t88*2.0-t104*2.0+t105*2.0+t238+t239+t299+t312-t333+t369-t380+t396+ry*t107*1.2E+1+ry*t118*1.2E+1-t4*t42*2.0-t4*t74*2.0);
    fdrv(1, 8) = -t599*tdx-t606*tdy+tdz*(t104*2.0E+1+t161+t166+t389+t411+t412+t439+t465+t467-ry*t107*2.0+ry*t109*4.0-rz*t67*2.0-rz*t115*2.0+t6*t77*1.2E+1+t24*t72-rz*t9*t20*1.2E+1-rx*t32*tx*2.0);
    fdrv(1, 9) = -tdx*(t56*-5.0+t253+t398+t497+t499+t2*t18*2.0+t9*t21+t14*t32*4.0+t17*t37-t116*ty+t207*ty)-t578*tdy*tx-t576*tdz*tx;
    fdrv(1, 10) = -tdy*(t246+t253+t398+t403+t497+t501+t2*t18*4.0+t14*t32*2.0+t21*t28+t17*t37*4.0-t20*t52*2.0)-t571*tdx*ty-t576*tdz*ty;
    fdrv(1, 11) = -tdz*(t68*-5.0+t246+t403+t499+t501+t2*t18+t9*t21*4.0+t14*t32+t20*t32*3.0+t17*t44*tx+t17*t28*tz)-t571*tdx*tz-t578*tdy*tz; fdrv(1, 12) = t609;
    fdrv(1, 13) = tdx*(t136+t220+t226+t293+rx*t22+t20*t40*4.0+t14*t72*3.0+t14*t74*6.0+t17*t84+t76*tx*2.0+t141*tx)+tdz*(t142+t219+t225+t295+rz*t16+t14*t41*4.0+t20*t75*6.0+t17*t81+t20*t78*3.0+t76*tz*2.0+t135*tz)+t156*t551*tdy; fdrv(1, 14) = t607;
    fdrv(1, 15) = -t510*tx*(t32+t37-t64+t44*tx); fdrv(1, 16) = -t510*ty*(t32+t37-t64+t44*tx); fdrv(1, 17) = -t510*tz*(t32+t37-t64+t44*tx);
    fdrv(1, 18) = -rdx*t510*(t32+t37-t64+t44*tx)-t11*tx*(t32+t37-t64+t44*tx)*2.0-t12*tx*(t32+t37-t64+t44*tx)*2.0-t13*tx*(t32+t37-t64+t44*tx)*2.0-t11*t479*t510-t12*t479*t510-t13*t479*t510;
    fdrv(1, 19) = -rdy*t510*(t32+t37-t64+t44*tx)-t11*ty*(t32+t37-t64+t44*tx)*2.0-t12*ty*(t32+t37-t64+t44*tx)*2.0-t13*ty*(t32+t37-t64+t44*tx)*2.0-t11*t258*t510-t12*t258*t510-t13*t258*t510;
    fdrv(1, 20) = -rdz*t510*(t32+t37-t64+t44*tx)-t11*tz*(t32+t37-t64+t44*tx)*2.0-t12*tz*(t32+t37-t64+t44*tx)*2.0-t13*tz*(t32+t37-t64+t44*tx)*2.0-t11*t510*(t9-t27)-t12*t510*(t9-t27)-t13*t510*(t9-t27); fdrv(1, 24) = t547; fdrv(1, 25) = t156*t494*t510;
    fdrv(1, 26) = t549; fdrv(2, 0) = -t17*t564*tx+t157*t563*tx+t595*tx*tz; fdrv(2, 1) = -t14*t563*ty+t156*t564*ty+t595*ty*tz; fdrv(2, 2) = -t14*t569-t17*t570-t155*t595;
    fdrv(2, 3) = t520+t527+rx*t63*5.0+rx*t69+rz*t249+rz*t496-t34*t39*3.0+t26*t79-t33*t72*2.0+t36*t75*4.0+t36*t78*4.0+t58*t72*6.0+t65*t74+t7*t138+t26*t122+t72*t146+t173*ty-t89*tz-rx*t2*t21*2.0-rx*t6*t21-rx*t14*t33*4.0-ry*t3*t21-rz*t20*t34*2.0-rz*t20*t51-ry*t32*tx*tz*3.0;
    fdrv(2, 4) = t523+t526+ry*t63+ry*t69*5.0+ry*t424+rz*t253+rz*t501+t28*t79+t37*t78*4.0+t58*t77+t73*t144+t165*ty+rx*t7*t18*4.0+rx*t21*t44-ry*t2*t21-ry*t6*t21*2.0-ry*t17*t33*3.0-ry*t17*t35*4.0-rz*t20*t32*2.0+rz*t117*ty*4.0+ry*t280*tz+t26*t74*ty-t4*t72*tz*3.0+rx*t14*t44*tz+rz*t20*t44*tx;
    fdrv(2, 5) = t260+t264-t488+t490+t492+t524+t525+rz*t63+rz*t69-t34*t42*2.0-t2*t76-t2*t79*4.0-t6*t79*4.0-t33*t75*2.0-t34*t74*3.0-t32*t77*3.0-t35*t78*2.0-t2*t119-t2*t120*3.0-t6*t122*3.0+t73*t143+rx*t18*t44+rx*t14*t145+t26*t78*tx+ry*t17*t44*tx+rx*t14*t26*tz+ry*t17*t28*tz;
    fdrv(2, 6) = -t600*tdy-t602*tdz+tdx*(t92*2.0E+1+t176+t180+t358+t407+t408+t441+t449+t454-rx*t59*2.0-rx*t111*2.0-rz*t116*2.0+rz*t117*1.2E+1+rz*t118*4.0+t28*t77-rx*t33*tx*1.2E+1-ry*t35*ty*2.0);
    fdrv(2, 7) = -t600*tdx-t604*tdz+tdy*(t103*2.0E+1+t165+t180+t390+t407+t408+t441+t454+t466-ry*t66*2.0-ry*t113*2.0-rz*t108*2.0+rz*t117*4.0+rz*t118*1.2E+1+t26*t74-rx*t33*tx*2.0-ry*t35*ty*1.2E+1);
    fdrv(2, 8) = -t602*tdx-t604*tdy-tdz*(t90*2.0-t92*2.0+t101*2.0-t103*2.0+t233+t234+t307+t314+t339+t381-t393-t397+rz*t108*1.2E+1+rz*t116*1.2E+1-t7*t40*2.0-t7*t72*2.0);
    fdrv(2, 9) = -tdx*(t63*-5.0+t254+t399+t500+t502+t2*t21*2.0+t6*t21+t14*t33*4.0+t17*t35+t193*tz+t20*t48*ty)-t575*tdy*tx-t577*tdz*tx;
    fdrv(2, 10) = -tdy*(t69*-5.0+t250+t401+t500+t503+t2*t21+t6*t21*2.0+t14*t33+t17*t33*3.0+t17*t35*4.0+t20*t46*tx)-t573*tdx*ty-t577*tdz*ty;
    fdrv(2, 11) = -tdz*(t250+t254+t399+t401+t502+t503+t2*t21*4.0+t6*t21*4.0+t14*t33*2.0+t17*t35*2.0-t17*t58*2.0)-t573*tdx*tz-t575*tdy*tz; fdrv(2, 12) = t608; fdrv(2, 13) = t607;
    fdrv(2, 14) = tdx*(t136+t220+t226+t293+rx*t19+t17*t42*4.0+t14*t72*6.0+t14*t74*3.0+t20*t82+t79*tx*2.0+t138*tx)+tdy*(t139+t214+t229+t294+ry*t16+t14*t43*4.0+t17*t73*6.0+t17*t77*3.0+t20*t80+t79*ty*2.0+t135*ty)+t155*t552*tdz;
    fdrv(2, 15) = -t510*tx*(t33+t35+t46*tx+t48*ty); fdrv(2, 16) = -t510*ty*(t33+t35+t46*tx+t48*ty); fdrv(2, 17) = -t510*tz*(t33+t35+t46*tx+t48*ty);
    fdrv(2, 18) = -rdx*t510*(t33+t35+t46*tx+t48*ty)-t11*tx*(t33+t35+t46*tx+t48*ty)*2.0-t12*tx*(t33+t35+t46*tx+t48*ty)*2.0-t13*tx*(t33+t35+t46*tx+t48*ty)*2.0-t11*t480*t510-t12*t480*t510-t13*t480*t510;
    fdrv(2, 19) = -rdy*t510*(t33+t35+t46*tx+t48*ty)-t11*ty*(t33+t35+t46*tx+t48*ty)*2.0-t12*ty*(t33+t35+t46*tx+t48*ty)*2.0-t13*ty*(t33+t35+t46*tx+t48*ty)*2.0-t11*t481*t510-t12*t481*t510-t13*t481*t510;
    fdrv(2, 20) = -rdz*t510*(t33+t35+t46*tx+t48*ty)-t11*tz*(t33+t35+t46*tx+t48*ty)*2.0-t12*tz*(t33+t35+t46*tx+t48*ty)*2.0-t13*tz*(t33+t35+t46*tx+t48*ty)*2.0-t11*t257*t510-t12*t257*t510-t13*t257*t510; fdrv(2, 24) = t548; fdrv(2, 25) = t549;
    fdrv(2, 26) = t155*t494*t510;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f24(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*ty, t3 = oy*tx, t4 = ox*tz, t5 = oz*tx, t6 = oy*tz, t7 = oz*ty, t8 = rdx*tx, t9 = rdy*ty, t10 = rdz*tz, t11 = rx*tx, t12 = ry*ty, t13 = rz*tz, t14 = tx*tx, t16 = ty*ty, t18 = tz*tz, t26 = rx*ty*tz, t27 = ry*tx*tz, t28 = rz*tx*ty;
    T t20 = t11*4.0, t21 = t12*4.0, t22 = t13*4.0, t23 = t11*ty, t24 = t11*tz, t25 = t12*tx, t29 = t12*tz, t30 = t13*tx, t31 = t13*ty, t32 = -t3, t33 = -t5, t34 = -t7, t35 = ox*t14, t36 = t2*ty, t37 = t3*tx, t38 = t4*tz, t39 = oy*t16, t40 = t5*tx;
    T t41 = t6*tz, t42 = t7*ty, t43 = oz*t18, t44 = t11*t14, t45 = rx*t16, t46 = ry*t14, t47 = rx*t18, t48 = rz*t14, t49 = t12*t16, t50 = ry*t18, t51 = rz*t16, t52 = t13*t18, t53 = t2*tx*4.0, t54 = t4*tx*4.0, t55 = t3*ty*4.0, t56 = t6*ty*4.0;
    T t57 = t5*tz*4.0, t58 = t7*tz*4.0, t86 = t11*t16, t87 = t11*t18, t88 = t12*t14, t95 = t12*t18, t96 = t13*t14, t97 = t13*t16, t100 = rx*t2*tz*2.0, t101 = rx*t3*tz*4.0, t102 = rx*t5*ty*4.0, t103 = ry*t3*tz*2.0, t104 = ry*t2*tz*4.0;
    T t108 = rz*t5*ty*2.0, t119 = t11*tx*5.0, t121 = t12*ty*5.0, t123 = t13*tz*5.0, t131 = -t27, t132 = -t28, t133 = t14+t16, t134 = t14+t18, t135 = t16+t18, t136 = ox*t11*tx*3.0, t141 = oy*t12*ty*3.0, t146 = oz*t13*tz*3.0, t160 = t2*t11*1.0E+1;
    T t161 = t4*t11*1.0E+1, t163 = t3*t12*1.0E+1, t165 = rx*t5*tz*-4.0, t167 = t6*t12*1.0E+1, t168 = ry*t7*tz*-4.0, t169 = t5*t13*1.0E+1, t170 = t7*t13*1.0E+1, t176 = t2*t26, t189 = t2*t27*4.0, t190 = t3*t26*4.0, t192 = t5*t26*4.0, t195 = t11+t12;
    T t196 = t11+t13, t197 = t12+t13, t208 = oz*t13*tz*-5.0, t257 = t2*t11*tx*-5.0, t260 = t4*t11*tx*-5.0, t263 = t3*t12*ty*-5.0, t270 = t6*t12*ty*-5.0, t273 = t5*t13*tz*-5.0, t274 = t7*t13*tz*-5.0, t284 = t8+t9+t10, t59 = t20*ty, t60 = t20*tz;
    T t61 = t21*tx, t62 = t21*tz, t63 = t22*tx, t64 = t22*ty, t65 = t11*t35, t66 = rx*t36, t67 = t2*t45, t68 = t11*t37, t69 = rx*t38, t70 = ry*t37, t71 = t4*t47, t72 = t12*t36, t74 = t11*t40, t75 = t12*t39, t76 = ry*t41, t77 = rz*t40;
    T t78 = t13*t38, t79 = t6*t50, t80 = t12*t42, t82 = rz*t42, t83 = t13*t41, t85 = t13*t43, t89 = t47*ty, t90 = t45*tz, t91 = t50*tx, t92 = t46*tz, t93 = t51*tx, t94 = t48*ty, t98 = ry*t53, t99 = rx*t55, t105 = rz*t54, t107 = t5*t21;
    T t109 = t2*t22, t110 = t3*t22, t111 = rz*t56, t113 = t36*5.0, t114 = t37*5.0, t115 = t38*5.0, t116 = t40*5.0, t117 = t41*5.0, t118 = t42*5.0, t120 = t44*6.0, t122 = t49*6.0, t124 = t52*6.0, t125 = -t53, t126 = -t54, t127 = -t55, t128 = -t56;
    T t129 = -t57, t130 = -t58, t137 = ox*t119, t142 = oy*t121, t148 = t86*2.0, t149 = t86*5.0, t150 = t87*2.0, t151 = t88*2.0, t152 = t87*5.0, t153 = t88*5.0, t154 = t95*2.0, t155 = t96*2.0, t156 = t95*5.0, t157 = t96*5.0, t158 = t97*2.0;
    T t159 = t97*5.0, t162 = -t100, t164 = -t103, t166 = -t108, t171 = t2*t23, t172 = t4*t24, t173 = t2*t46, t174 = t3*t45, t175 = t2*t47, t177 = t3*t25, t178 = t4*t48, t181 = t5*t47, t182 = t6*t29, t185 = t6*t51, t186 = t7*t50, t187 = t5*t30;
    T t188 = t7*t31, t200 = t32*t46, t202 = rz*t33*tx, t204 = t33*t48, t205 = rz*t34*ty, t206 = t34*t51, t207 = -t146, t223 = t3*t47*2.0, t224 = t5*t45*2.0, t225 = t3*t47*5.0, t228 = t5*t45*5.0, t229 = t2*t50*2.0, t232 = t5*t25*2.0;
    T t235 = t2*t50*5.0, t236 = t5*t25*5.0, t237 = t2*t31*2.0, t238 = t3*t30*2.0, t239 = t2*t31*5.0, t242 = t3*t30*5.0, t251 = -t189, t252 = -t190, t253 = t2*t30*-4.0, t254 = -t192, t255 = t3*t31*-4.0, t256 = t5*t29*-4.0, t258 = t2*t25*-4.0;
    T t259 = t3*t23*-4.0, t262 = -t176, t264 = t32*t50, t265 = t27*t32, t266 = t4*t30*-4.0, t267 = t5*t24*-4.0, t268 = t33*t51, t269 = t28*t33, t271 = t6*t31*-4.0, t272 = t7*t29*-4.0, t275 = rx*t133, t276 = rx*t134, t277 = ry*t133, t278 = ry*t135;
    T t279 = rz*t134, t280 = rz*t135, t281 = t2+t32, t282 = t4+t33, t283 = t6+t34, t285 = t195*tx*tz, t286 = t196*tx*ty, t287 = t195*ty*tz, t288 = t197*tx*ty, t289 = t196*ty*tz, t290 = t197*tx*tz, t297 = t18+t133, t298 = t26+t131, t299 = t26+t132;
    T t300 = t27+t132, t314 = t133*t195, t315 = t134*t196, t316 = t135*t197, t138 = t68*6.0, t139 = t72*6.0, t140 = t74*6.0, t143 = t78*6.0, t144 = t80*6.0, t145 = t83*6.0, t198 = -t67, t199 = -t71, t201 = -t76, t203 = -t79, t209 = -t89;
    T t210 = -t90, t211 = -t91, t212 = -t92, t213 = -t93, t214 = -t94, t215 = t171*5.0, t217 = t173*2.0, t218 = t174*2.0, t221 = t172*5.0, t227 = t177*5.0, t230 = t178*2.0, t231 = t181*2.0, t240 = t182*5.0, t243 = t185*2.0, t244 = t186*2.0;
    T t248 = t187*5.0, t250 = t188*5.0, t261 = -t175, t291 = t276*ty, t292 = t275*tz, t293 = t278*tx, t294 = t277*tz, t295 = t280*tx, t296 = t279*ty, t301 = t297*t297, t302 = t298*tx, t303 = t299*tx, t304 = t298*ty, t305 = t300*ty, t306 = t299*tz;
    T t307 = t300*tz, t308 = t25+t276, t309 = t30+t275, t310 = t23+t278, t311 = t31+t277, t312 = t24+t280, t313 = t29+t279, t341 = t35+t38+t113+t127, t342 = t35+t36+t115+t129, t343 = t39+t41+t114+t125, t344 = t37+t39+t117+t130;
    T t345 = t42+t43+t116+t126, t346 = t40+t43+t118+t128, t350 = t45+t47+t61+t63+t119, t351 = t46+t50+t59+t64+t121, t352 = t48+t51+t60+t62+t123, t359 = t44+t49+t86+t88+t124+t152+t155+t156+t158, t360 = t44+t52+t87+t96+t122+t149+t151+t154+t159;
    T t361 = t49+t52+t95+t97+t120+t148+t150+t153+t157, t317 = -t304, t318 = -t306, t319 = -t307, t320 = t308*tx, t321 = t309*tx, t322 = t310*tx, t323 = t308*ty, t324 = t310*ty, t325 = t312*tx, t326 = t309*tz, t327 = t311*ty, t328 = t313*ty;
    T t329 = t311*tz, t330 = t312*tz, t331 = t313*tz, t332 = ox*t8*t301*2.0, t333 = rdx*t3*t301*2.0, t334 = rdy*t2*t301*2.0, t335 = rdx*t5*t301*2.0, t336 = oy*t9*t301*2.0, t337 = rdz*t4*t301*2.0, t338 = rdy*t7*t301*2.0, t339 = rdz*t6*t301*2.0;
    T t340 = oz*t10*t301*2.0, t347 = t284*t301*tx*2.0, t348 = t284*t301*ty*2.0, t349 = t284*t301*tz*2.0, t353 = t297*t350*tdx*ty*2.0, t354 = t297*t350*tdx*tz*2.0, t355 = t297*t351*tdy*tx*2.0, t356 = t297*t351*tdy*tz*2.0;
    T t357 = t297*t352*tdz*tx*2.0, t358 = t297*t352*tdz*ty*2.0, t392 = t297*t361*tdx*2.0, t393 = t297*t360*tdy*2.0, t394 = t297*t359*tdz*2.0, t395 = t65+t78+t139+t172+t178+t200+t215+t217+t229+t239+t255+t259+t263+t264;
    T t396 = t75+t83+t138+t182+t185+t198+t218+t223+t227+t242+t253+t257+t258+t261, t397 = t65+t72+t143+t171+t173+t204+t221+t230+t235+t237+t256+t267+t268+t273, t398 = t80+t85+t140+t186+t188+t199+t224+t231+t236+t248+t251+t260+t262+t266;
    T t399 = t68+t75+t145+t174+t177+t206+t225+t238+t240+t243+t254+t269+t272+t274, t400 = t74+t85+t144+t181+t187+t203+t228+t232+t244+t250+t252+t265+t270+t271, t362 = t214+t287+t296+t302+t329, t363 = t210+t290+t292+t305+t325;
    T t364 = t212+t289+t294+t303+t328, t371 = t211+t286+t293+t318+t323, t372 = t213+t285+t295+t317+t326, t373 = t209+t288+t291+t319+t322, t374 = t96+t97+t314+t321+t327, t375 = t88+t95+t315+t320+t331, t376 = t86+t87+t316+t324+t330, t365 = t362*tx;
    T t366 = t364*tx, t367 = t362*ty, t368 = t363*ty, t369 = t363*tz, t370 = t364*tz, t377 = t371*tx, t378 = t372*tx, t379 = t372*ty, t380 = t373*ty, t381 = t371*tz, t382 = t373*tz, t384 = t374*tx, t385 = t375*tx, t386 = t374*ty, t387 = t376*ty;
    T t388 = t375*tz, t389 = t376*tz, t383 = -t368, t390 = -t381, t391 = -t382, t402 = t369+t384, t403 = t367+t388, t404 = t370+t386, t406 = t377+t387, t407 = t380+t385, t408 = t378+t389, t401 = t366+t383, t405 = t365+t391, t409 = t379+t390;
    
    fdrv(0, 0) = -t135*t409+t403*tx*ty-t404*tx*tz; fdrv(0, 1) = -t134*t403+t409*tx*ty-t404*ty*tz; fdrv(0, 2) = t133*t404+t409*tx*tz+t403*ty*tz; fdrv(0, 3) = t283*t297*t350*-2.0; fdrv(0, 4) = t297*t400*2.0; fdrv(0, 5) = t297*t399*-2.0;
    fdrv(0, 6) = t297*tdy*(t107+t164+rx*t43+rx*t118+t5*t11*3.0+t5*t13*2.0-rx*t6*ty*4.0)*2.0-t297*tdz*(t110+t166+rx*t39+rx*t117+t3*t11*3.0+t3*t12*2.0-rx*t7*tz*4.0)*2.0+t400*tdy*tx*4.0-t399*tdz*tx*4.0-t283*t350*tdx*tx*4.0-t283*t297*tdx*(t11*1.0E+1+t21+t22)*2.0;
    fdrv(0, 7) = t297*tdz*(t70+t76*5.0-t82*3.0+t111+t141+t165+t202+t208+rx*t3*ty*2.0-ry*t7*tz*8.0)*-2.0+t297*tdy*(-t101-t167+t170+ry*t40*2.0+ry*t43*2.0-t6*t13*4.0+t7*t12*1.8E+1+rx*t5*ty*1.0E+1)*2.0+t400*tdy*ty*4.0-t399*tdz*ty*4.0+oz*t297*t350*tdx*2.0-t283*t350*tdx*ty*4.0-t283*t297*tdx*(rx*ty*2.0+ry*tx*4.0)*2.0;
    fdrv(0, 8) = t297*tdy*(t70+t76*3.0-t82*5.0+t99+t142+t168+t202+t207-rx*t5*tz*2.0+rz*t6*ty*8.0)*-2.0-t297*tdz*(-t102+t167-t170+rz*t37*2.0+rz*t39*2.0+t6*t13*1.8E+1-t7*t12*4.0+rx*t3*tz*1.0E+1)*2.0+t400*tdy*tz*4.0-t399*tdz*tz*4.0-oy*t297*t350*tdx*2.0-t283*t350*tdx*tz*4.0-t283*t297*tdx*(rx*tz*2.0+rz*tx*4.0)*2.0;
    fdrv(0, 9) = t297*t346*tdy*tx*2.0-t297*t344*tdz*tx*2.0-t283*t297*tdx*(t14*3.0+t133+t134)*2.0; fdrv(0, 10) = t297*tdy*(t6*t16*5.0-t7*t16*6.0+t6*t18-t7*t18*2.0-t40*ty*2.0+t37*tz)*-2.0-t297*t344*tdz*ty*2.0-t283*t297*tdx*tx*ty*8.0;
    fdrv(0, 11) = t297*tdz*(t6*t16*2.0+t6*t18*6.0-t7*t18*5.0+t16*t34+t37*tz*2.0+t33*tx*ty)*-2.0+t297*t346*tdy*tz*2.0-t283*t297*tdx*tx*tz*8.0; fdrv(0, 13) = -t354-t356-t394; fdrv(0, 14) = t353+t358+t393; fdrv(0, 15) = t283*t301*tx*-2.0;
    fdrv(0, 16) = t283*t301*ty*-2.0; fdrv(0, 17) = t283*t301*tz*-2.0; fdrv(0, 18) = t283*t297*(rdx*t16+rdx*t18+t8*tx*5.0+t9*tx*4.0+t10*tx*4.0)*-2.0;
    fdrv(0, 19) = t335+t338+t340-rdy*t283*t301*2.0-t8*t283*t297*ty*8.0-t9*t283*t297*ty*8.0-t10*t283*t297*ty*8.0; fdrv(0, 20) = -t333-t336-t339-rdz*t283*t301*2.0-t8*t283*t297*tz*8.0-t9*t283*t297*tz*8.0-t10*t283*t297*tz*8.0; fdrv(0, 25) = -t349;
    fdrv(0, 26) = t348; fdrv(1, 0) = t135*t408-t405*tx*ty+t402*tx*tz; fdrv(1, 1) = t134*t405-t408*tx*ty+t402*ty*tz; fdrv(1, 2) = -t133*t402-t408*tx*tz-t405*ty*tz; fdrv(1, 3) = t297*t398*-2.0; fdrv(1, 4) = t282*t297*t351*2.0; fdrv(1, 5) = t297*t397*2.0;
    fdrv(1, 6) = t297*tdx*(-t104-t161+t169+rx*t42*2.0+rx*t43*2.0+t5*t11*1.8E+1-t4*t13*4.0+t5*t12*1.0E+1)*-2.0+t297*tdz*(t66+t69*5.0-t77*3.0+t105+t136+t168+t205+t208+ry*t2*tx*2.0-rx*t5*tz*8.0)*2.0-t398*tdx*tx*4.0+t397*tdz*tx*4.0-oz*t297*t351*tdy*2.0+t282*t351*tdy*tx*4.0+t282*t297*tdy*(rx*ty*4.0+ry*tx*2.0)*2.0;
    fdrv(1, 7) = t297*tdx*(t102+t162+ry*t43+ry*t116+t7*t12*3.0+t7*t13*2.0-ry*t4*tx*4.0)*-2.0+t297*tdz*(t109+t166+ry*t35+ry*t115+t2*t11*2.0+t2*t12*3.0-ry*t5*tz*4.0)*2.0-t398*tdx*ty*4.0+t397*tdz*ty*4.0+t282*t351*tdy*ty*4.0+t282*t297*tdy*(t12*1.0E+1+t20+t22)*2.0;
    fdrv(1, 8) = t297*tdz*(t161-t169+rz*t35*2.0+rz*t36*2.0-t5*t11*4.0+t4*t13*1.8E+1-t5*t12*4.0+ry*t2*tz*1.0E+1)*2.0+t297*tdx*(t66+t69*3.0-t77*5.0+t98+t137+t165+t205+t207+rz*t4*tx*8.0-ry*t7*tz*2.0)*2.0-t398*tdx*tz*4.0+t397*tdz*tz*4.0+ox*t297*t351*tdy*2.0+t282*t351*tdy*tz*4.0+t282*t297*tdy*(ry*tz*2.0+rz*ty*4.0)*2.0;
    fdrv(1, 9) = t297*tdx*(t4*t14*5.0-t5*t14*6.0-t5*t16*2.0+t4*t18-t5*t18*2.0+t36*tz)*2.0+t297*t342*tdz*tx*2.0+t282*t297*tdy*tx*ty*8.0; fdrv(1, 10) = t297*t345*tdx*ty*-2.0+t297*t342*tdz*ty*2.0+t282*t297*tdy*(t16*3.0+t133+t135)*2.0;
    fdrv(1, 11) = t297*tdz*(t4*t14*2.0+t4*t18*6.0-t5*t18*5.0+t14*t33+t16*t33+t36*tz*2.0)*2.0-t297*t345*tdx*tz*2.0+t282*t297*tdy*ty*tz*8.0; fdrv(1, 12) = t354+t356+t394; fdrv(1, 14) = -t355-t357-t392; fdrv(1, 15) = t282*t301*tx*2.0;
    fdrv(1, 16) = t282*t301*ty*2.0; fdrv(1, 17) = t282*t301*tz*2.0; fdrv(1, 18) = -t335-t338-t340+rdx*t282*t301*2.0+t8*t282*t297*tx*8.0+t9*t282*t297*tx*8.0+t10*t282*t297*tx*8.0;
    fdrv(1, 19) = t282*t297*(rdy*t14+rdy*t18+t8*ty*4.0+t9*ty*5.0+t10*ty*4.0)*2.0; fdrv(1, 20) = t332+t334+t337+rdz*t282*t301*2.0+t8*t282*t297*tz*8.0+t9*t282*t297*tz*8.0+t10*t282*t297*tz*8.0; fdrv(1, 24) = t349; fdrv(1, 26) = -t347;
    fdrv(2, 0) = -t135*t406-t407*tx*ty+t401*tx*tz; fdrv(2, 1) = t134*t407+t406*tx*ty+t401*ty*tz; fdrv(2, 2) = -t133*t401+t406*tx*tz-t407*ty*tz; fdrv(2, 3) = t297*t396*2.0; fdrv(2, 4) = t297*t395*-2.0; fdrv(2, 5) = t281*t297*t352*-2.0;
    fdrv(2, 6) = t297*tdy*(t66*5.0+t69-t70*3.0+t98+t136+t201-oy*t12*ty*5.0-rx*t3*ty*8.0+rz*t4*tx*2.0-rz*t6*ty*4.0)*-2.0+t297*tdx*(-t160+t163+rx*t39*2.0+rx*t41*2.0-t2*t12*4.0+t3*t11*1.8E+1-t2*t13*4.0+t3*t13*1.0E+1)*2.0+t396*tdx*tx*4.0-t395*tdy*tx*4.0+oy*t297*t352*tdz*2.0-t281*t352*tdz*tx*4.0-t281*t297*tdz*(rx*tz*4.0+rz*tx*2.0)*2.0;
    fdrv(2, 7) = t297*tdx*(t66*3.0+t69-t70*5.0+t105+t137-t141+t201+ry*t2*tx*8.0-rx*t3*ty*4.0-rz*t6*ty*2.0)*-2.0-t297*tdy*(t160-t163+ry*t35*2.0+ry*t38*2.0+t2*t12*1.8E+1-t3*t11*4.0+t2*t13*1.0E+1-t3*t13*4.0)*2.0+t396*tdx*ty*4.0-t395*tdy*ty*4.0-ox*t297*t352*tdz*2.0-t281*t352*tdz*ty*4.0-t281*t297*tdz*(ry*tz*4.0+rz*ty*2.0)*2.0;
    fdrv(2, 8) = t297*tdx*(t101+t162+rz*t39+rz*t114+t6*t12*2.0+t6*t13*3.0-rz*t2*tx*4.0)*2.0-t297*tdy*(t104+t164+rz*t35+rz*t113+t4*t11*2.0+t4*t13*3.0-rz*t3*ty*4.0)*2.0+t396*tdx*tz*4.0-t395*tdy*tz*4.0-t281*t352*tdz*tz*4.0-t281*t297*tdz*(t13*1.0E+1+t20+t21)*2.0;
    fdrv(2, 9) = t297*tdx*(t2*t14*5.0-t3*t14*6.0+t2*t16-t3*t16*2.0+t2*t18-t3*t18*2.0)*-2.0-t297*t341*tdy*tx*2.0-t281*t297*tdz*tx*tz*8.0;
    fdrv(2, 10) = t297*tdy*(t2*t14*2.0+t2*t16*6.0-t3*t16*5.0+t2*t18*2.0+t14*t32+t18*t32)*-2.0+t297*t343*tdx*ty*2.0-t281*t297*tdz*ty*tz*8.0; fdrv(2, 11) = t297*t343*tdx*tz*2.0-t297*t341*tdy*tz*2.0-t281*t297*tdz*(t18*4.0+t297)*2.0;
    fdrv(2, 12) = -t353-t358-t393; fdrv(2, 13) = t355+t357+t392; fdrv(2, 15) = t281*t301*tx*-2.0; fdrv(2, 16) = t281*t301*ty*-2.0; fdrv(2, 17) = t281*t301*tz*-2.0;
    fdrv(2, 18) = t333+t336+t339-rdx*t281*t301*2.0-t8*t281*t297*tx*8.0-t9*t281*t297*tx*8.0-t10*t281*t297*tx*8.0; fdrv(2, 19) = -t332-t334-t337-rdy*t281*t301*2.0-t8*t281*t297*ty*8.0-t9*t281*t297*ty*8.0-t10*t281*t297*ty*8.0;
    fdrv(2, 20) = t281*t297*(rdz*t14+rdz*t16+t8*tz*4.0+t9*tz*4.0+t10*tz*5.0)*-2.0; fdrv(2, 24) = -t348; fdrv(2, 25) = t347;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f25(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*ry, t3 = oy*rx, t4 = ox*rz, t5 = oz*rx, t6 = oy*rz, t7 = oz*ry, t8 = ox*tx, t9 = ox*ty, t10 = oy*tx, t11 = ox*tz, t12 = oy*ty, t13 = oz*tx, t14 = oy*tz, t15 = oz*ty, t16 = oz*tz, t17 = rx*tx, t18 = rx*ty, t19 = ry*tx;
    T t20 = rx*tz, t21 = ry*ty, t22 = rz*tx, t23 = ry*tz, t24 = rz*ty, t25 = rz*tz, t26 = tx*tx, t27 = ty*ty, t28 = tz*tz, t29 = t2*2.0, t30 = t3*2.0, t31 = t4*2.0, t32 = t5*2.0, t33 = t6*2.0, t34 = t7*2.0, t35 = t8*2.0, t36 = t12*2.0;
    T t37 = t16*2.0, t38 = t17*2.0, t39 = t21*2.0, t40 = t25*2.0, t41 = t18*tz, t42 = t19*tz, t43 = t22*ty, t44 = -t3, t46 = -t5, t48 = -t7, t50 = -t10, t51 = -t13, t52 = -t15, t53 = -t19, t54 = -t22, t55 = -t24, t56 = t26+t27, t57 = t26+t28;
    T t58 = t27+t28, t45 = -t30, t47 = -t32, t49 = -t34, t59 = t8+t36, t60 = t12+t35, t61 = t8+t37, t62 = t16+t35, t63 = t12+t37, t64 = t16+t36, t65 = t17+t39, t66 = t21+t38, t67 = t17+t40, t68 = t25+t38, t69 = t21+t40, t70 = t25+t39;
    T t71 = t2+t44, t72 = t4+t46, t73 = t6+t48, t74 = t9+t50, t75 = t11+t51, t76 = t14+t52, t77 = t18+t53, t78 = t20+t54, t79 = t23+t55, t80 = t29+t45, t81 = t31+t47, t82 = t33+t49, t83 = t71*tdz, t84 = t72*tdy, t85 = t73*tdx, t86 = -t83;
    T t87 = -t85;
    
    fdrv(0, 0) = -t42+t43; fdrv(0, 1) = rz*t58+t17*tz; fdrv(0, 2) = -ry*t58-t17*ty; fdrv(0, 3) = -ox*t79+t46*ty+t3*tz; fdrv(0, 4) = -oz*t65+t4*tx+t33*ty; fdrv(0, 5) = oy*t67-t2*tx-t7*tz*2.0; fdrv(0, 6) = t84+t86; fdrv(0, 7) = t72*tdx+t82*tdy;
    fdrv(0, 8) = -t71*tdx+t82*tdz; fdrv(0, 9) = t76*tdx+t51*tdy+t10*tdz; fdrv(0, 10) = -t11*tdx-t15*tdy*2.0-t61*tdz; fdrv(0, 11) = t9*tdx+t59*tdy+t14*tdz*2.0; fdrv(0, 12) = -t79*tdx+t22*tdy+t53*tdz; fdrv(0, 13) = t20*tdx+t24*tdy*2.0+t67*tdz;
    fdrv(0, 14) = -t18*tdx-t65*tdy-t23*tdz*2.0; fdrv(0, 15) = t76*tx; fdrv(0, 16) = -oz*t58-t8*tz; fdrv(0, 17) = oy*t58+t8*ty; fdrv(0, 18) = rdx*t76-rdy*t11+rdz*t9; fdrv(0, 19) = rdx*t51-rdy*t15*2.0+rdz*t59; fdrv(0, 20) = rdx*t10-rdy*t61+rdz*t14*2.0;
    fdrv(0, 24) = tx*(rdz*ty-rdy*tz); fdrv(0, 25) = rdz*t58+rdx*tx*tz; fdrv(0, 26) = -rdy*t58-rdx*tx*ty; fdrv(1, 0) = -rz*t57-t21*tz; fdrv(1, 1) = t41-t43; fdrv(1, 2) = rx*t57+t19*ty; fdrv(1, 3) = oz*t66-t4*tx*2.0-t6*ty; fdrv(1, 4) = oy*t78+t7*tx-t2*tz;
    fdrv(1, 5) = -ox*t69+t3*ty+t32*tz; fdrv(1, 6) = -t81*tdx-t73*tdy; fdrv(1, 7) = t86+t87; fdrv(1, 8) = -t71*tdy-t81*tdz; fdrv(1, 9) = t13*tdx*2.0+t14*tdy+t63*tdz; fdrv(1, 10) = t15*tdx-t75*tdy-t9*tdz; fdrv(1, 11) = -t60*tdx+t50*tdy-t11*tdz*2.0;
    fdrv(1, 12) = t22*tdx*-2.0-t23*tdy-t69*tdz; fdrv(1, 13) = t55*tdx+t78*tdy+t18*tdz; fdrv(1, 14) = t66*tdx+t19*tdy+t20*tdz*2.0; fdrv(1, 15) = oz*t57+t12*tz; fdrv(1, 16) = -t75*ty; fdrv(1, 17) = -ox*t57+t50*ty; fdrv(1, 18) = rdx*t13*2.0+rdy*t15-rdz*t60;
    fdrv(1, 19) = rdx*t14-rdy*t75+rdz*t50; fdrv(1, 20) = rdx*t63-rdy*t9-rdz*t11*2.0; fdrv(1, 24) = -rdz*t57-rdy*ty*tz; fdrv(1, 25) = -ty*(rdz*tx-rdx*tz); fdrv(1, 26) = rdx*t57+rdy*tx*ty; fdrv(2, 0) = ry*t56+t24*tz; fdrv(2, 1) = -rx*t56+t54*tz;
    fdrv(2, 2) = -t41+t42; fdrv(2, 3) = -oy*t68+t29*tx+t7*tz; fdrv(2, 4) = ox*t70-t3*ty*2.0+t46*tz; fdrv(2, 5) = -oz*t77-t6*tx+t4*ty; fdrv(2, 6) = t80*tdx-t73*tdz; fdrv(2, 7) = t80*tdy+t72*tdz; fdrv(2, 8) = t84+t87; fdrv(2, 9) = t10*tdx*-2.0-t64*tdy+t52*tdz;
    fdrv(2, 10) = t62*tdx+t9*tdy*2.0+t13*tdz; fdrv(2, 11) = -t14*tdx+t11*tdy+t74*tdz; fdrv(2, 12) = t19*tdx*2.0+t70*tdy+t24*tdz; fdrv(2, 13) = -t68*tdx-t18*tdy*2.0+t54*tdz; fdrv(2, 14) = t23*tdx-t20*tdy-t77*tdz; fdrv(2, 15) = -oy*t56+t52*tz;
    fdrv(2, 16) = ox*t56+t13*tz; fdrv(2, 17) = t74*tz; fdrv(2, 18) = rdx*t10*-2.0+rdy*t62-rdz*t14; fdrv(2, 19) = -rdx*t64+rdy*t9*2.0+rdz*t11; fdrv(2, 20) = rdx*t52+rdy*t13+rdz*t74; fdrv(2, 24) = rdy*t56+rdz*ty*tz; fdrv(2, 25) = -rdx*t56-rdz*tx*tz;
    fdrv(2, 26) = tz*(rdy*tx-rdx*ty);
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f26(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = oy*ty, t4 = oz*tz, t5 = rx*tx, t6 = rx*ty, t7 = ry*tx, t8 = rx*tz, t9 = ry*ty, t10 = rz*tx, t11 = ry*tz, t12 = rz*ty, t13 = rz*tz, t14 = tx*ty, t15 = tx*tz, t16 = ty*tz, t17 = tx*tx, t18 = tx*tx*tx, t19 = ty*ty;
    T t20 = ty*ty*ty, t21 = tz*tz, t22 = tz*tz*tz, t23 = t2*2.0, t24 = t3*2.0, t25 = t4*2.0, t26 = t5*2.0, t27 = t9*2.0, t28 = t13*2.0, t29 = t14*2.0, t30 = t15*2.0, t31 = t16*2.0, t32 = ry*t2, t33 = oy*t5, t34 = ox*t9, t35 = rz*t2, t36 = rx*t3;
    T t37 = oz*t5, t38 = ox*t11, t39 = ox*t12, t40 = oy*t8, t41 = oy*t10, t42 = oz*t6, t43 = oz*t7, t44 = ox*t13, t45 = rz*t3, t46 = rx*t4, t47 = oz*t9, t48 = oy*t13, t49 = ry*t4, t50 = t5*ty, t51 = t5*tz, t52 = t7*ty, t53 = t6*tz, t54 = t7*tz;
    T t55 = t10*ty, t56 = t9*tz, t57 = t10*tz, t58 = t12*tz, t59 = t14*tz, t60 = t16+tx, t61 = t15+ty, t62 = t14+tz, t63 = t17*2.0, t64 = t18*2.0, t65 = t17*6.0, t66 = t19*2.0, t67 = t20*2.0, t68 = t19*6.0, t69 = t21*2.0, t70 = t22*2.0;
    T t71 = t21*6.0, t75 = -t14, t76 = -t15, t77 = -t16, t78 = t2*tx, t79 = ox*t19, t80 = oy*t17, t81 = ox*t20, t82 = oy*t18, t83 = ox*t21, t84 = t3*ty, t85 = oz*t17, t86 = ox*t22, t87 = oz*t18, t88 = oy*t21, t89 = oz*t19, t90 = oy*t22;
    T t91 = oz*t20, t92 = t4*tz, t93 = t5*tx, t94 = t5*t17, t95 = t6*ty, t96 = t7*tx, t97 = t6*t19, t98 = t7*t17, t99 = t8*tz, t100 = t9*ty, t101 = t10*tx, t102 = t8*t21, t103 = t9*t19, t104 = t10*t17, t105 = t11*tz, t106 = t12*ty, t107 = t11*t21;
    T t108 = t12*t19, t109 = t13*tz, t110 = t13*t21, t111 = t14*ty, t112 = t14*tx, t113 = t14*t19, t114 = t14*t17, t115 = t15*tz, t116 = t15*tx, t117 = t15*t21, t118 = t15*t17, t119 = t16*tz, t120 = t16*ty, t121 = t16*t21, t122 = t16*t19;
    T t124 = ox*t6*2.0, t127 = ox*t8*2.0, t130 = oy*t7*2.0, t146 = oy*t11*2.0, t149 = oz*t10*2.0, t153 = oz*t12*2.0, t162 = ox*t16*4.0, t163 = oy*t15*4.0, t164 = oz*t14*4.0, t187 = t2*t5, t190 = t2*t7, t192 = t2*t10, t193 = t3*t6, t195 = t3*t9;
    T t198 = t3*t12, t199 = t4*t8, t201 = t4*t11, t203 = t4*t13, t205 = t2*t14, t206 = t2*t15, t207 = t3*t14, t208 = t3*t16, t209 = t4*t15, t210 = t4*t16, t217 = t5*t19, t218 = t5*t21, t219 = t7*t14, t220 = t6*t21, t221 = t6*t16, t222 = t7*t21;
    T t223 = t7*t15, t224 = t10*t19, t225 = t10*t14, t226 = t9*t21, t227 = t10*t15, t228 = t12*t16, t229 = t14*t21, t230 = t14*t16, t231 = t14*t15, t233 = t2*t6*6.0, t237 = t2*t9*4.0, t238 = t3*t5*4.0, t239 = t2*t8*6.0, t247 = t3*t7*6.0;
    T t255 = t2*t13*4.0, t256 = t4*t5*4.0, t267 = t3*t11*6.0, t272 = t3*t13*4.0, t273 = t4*t9*4.0, t274 = t4*t10*6.0, t276 = t4*t12*6.0, t310 = rx*t2*-2.0, t317 = ry*t3*-2.0, t333 = rz*t4*-2.0, t334 = t2*ty*-2.0, t335 = t2*tz*-2.0;
    T t336 = t3*tx*-2.0, t337 = ox*t16*-2.0, t338 = oy*t15*-2.0, t339 = oz*t14*-2.0, t340 = t3*tz*-2.0, t341 = t4*tx*-2.0, t342 = t4*ty*-2.0, t348 = t17+t19, t349 = t17+t21, t350 = t19+t21, t393 = t5*t14*3.0, t399 = t5*t15*3.0, t400 = t7*t19*3.0;
    T t407 = t9*t16*3.0, t408 = t10*t21*3.0, t412 = t12*t21*3.0, t417 = t2*t6*-2.0, t418 = t2*t8*-2.0, t419 = t3*t5*-2.0, t422 = t2*t11*-2.0, t423 = t2*t12*-2.0, t424 = t2*t13*-2.0, t425 = t3*t8*-2.0, t427 = t3*t10*-2.0, t429 = t3*t11*-2.0;
    T t430 = t4*t6*-2.0, t431 = t4*t7*-2.0, t433 = t4*t9*-2.0, t444 = rdx*t18*-2.0, t445 = rdy*t20*-2.0, t446 = rdz*t22*-2.0, t450 = t5+t9, t451 = t5+t13, t452 = t9+t13, t507 = -tx*(t14-tz), t508 = -ty*(t16-tx), t509 = -tz*(t15-ty);
    T t537 = tx*(t14-tz)*2.0, t538 = ty*(t16-tx)*2.0, t539 = tz*(t15-ty)*2.0, t540 = t14*(t15-ty), t541 = t14*(t14-tz), t542 = t15*(t16-tx), t543 = t15*(t15-ty), t544 = t16*(t16-tx), t545 = t16*(t14-tz), t72 = -t23, t73 = -t24, t74 = -t25;
    T t123 = rx*t23, t125 = ry*t23, t126 = t33*4.0, t128 = rz*t23, t129 = rx*t24, t131 = t34*4.0, t132 = t37*4.0, t133 = t38*2.0, t134 = t39*2.0, t135 = t40*2.0, t136 = ry*t24, t137 = t41*2.0, t138 = t42*2.0, t139 = t43*2.0, t140 = t38*4.0;
    T t141 = t39*4.0, t142 = t40*4.0, t143 = t41*4.0, t144 = t42*4.0, t145 = t43*4.0, t147 = rz*t24, t148 = rx*t25, t150 = t44*4.0, t151 = t47*4.0, t152 = ry*t25, t154 = t48*4.0, t155 = rz*t25, t156 = t23*ty, t157 = t23*tz, t158 = t24*tx;
    T t159 = ox*t31, t160 = oy*t30, t161 = oz*t29, t165 = t24*tz, t166 = t25*tx, t167 = t25*ty, t168 = t26*ty, t169 = t50*4.0, t170 = t26*tz, t171 = t52*2.0, t172 = t51*4.0, t173 = t52*4.0, t174 = t53*2.0, t175 = t54*2.0, t176 = t55*2.0;
    T t177 = t27*tz, t178 = t57*2.0, t179 = t56*4.0, t180 = t57*4.0, t181 = t58*2.0, t182 = t58*4.0, t183 = t29*tz, t184 = rdx*t78, t185 = rdy*t84, t186 = rdz*t92, t188 = t5*t78, t189 = ox*t95, t191 = ox*t99, t194 = oy*t96, t196 = t9*t84;
    T t197 = oy*t105, t200 = oz*t101, t202 = oz*t106, t204 = t13*t92, t211 = rdx*t111, t212 = rdy*t112, t213 = rdx*t115, t214 = rdz*t116, t215 = rdy*t119, t216 = rdz*t120, t232 = t6*t23, t234 = t8*t23, t235 = t9*t23, t241 = t11*t23;
    T t242 = t12*t23, t243 = t7*t24, t246 = t33*tz*6.0, t248 = t37*ty*6.0, t250 = t8*t24, t252 = t10*t24, t253 = t5*t25, t258 = t34*tz*6.0, t259 = t43*ty*6.0, t260 = t11*t24, t261 = t6*t25, t262 = t7*t25, t266 = t39*tz*6.0, t268 = t41*tz*6.0;
    T t269 = t13*t24, t271 = t10*t25, t275 = t12*t25, t277 = t16*t23, t278 = t15*t24, t279 = t14*t25, t280 = t16*t26, t281 = t7*t31, t282 = t10*t31, t283 = t78*3.0, t284 = ox*t66, t285 = oy*t63, t286 = ox*t69, t287 = oz*t63, t288 = t84*3.0;
    T t289 = oy*t69, t290 = oz*t66, t291 = t92*3.0, t295 = t26*tx, t296 = t94*4.0, t297 = t93*6.0, t298 = t27*ty, t299 = t103*4.0, t300 = t100*6.0, t301 = t28*tz, t302 = t110*4.0, t303 = t109*6.0, t304 = t29*ty, t305 = t29*tx, t306 = t30*tz;
    T t307 = t30*tx, t308 = t31*tz, t309 = t31*ty, t311 = -t32, t312 = -t33, t313 = -t124, t314 = t32*-2.0, t324 = -t44, t325 = -t45, t326 = -t46, t327 = -t47, t328 = -t146, t329 = t45*-2.0, t330 = t46*-2.0, t331 = -t149, t343 = -t51, t344 = -t52;
    T t345 = -t58, t346 = -t59, t347 = t59*-2.0, t351 = t187*3.0, t354 = t33*tx*3.0, t358 = t34*ty*3.0, t360 = t37*tx*3.0, t367 = t38*tz*3.0, t368 = t39*ty*3.0, t369 = t40*tz*3.0, t370 = t195*3.0, t371 = t41*tx*3.0, t372 = t42*ty*3.0;
    T t373 = t43*tx*3.0, t376 = t44*tz*3.0, t378 = t47*ty*3.0, t381 = t48*tz*3.0, t383 = t203*3.0, t384 = ox*t119*3.0, t385 = ox*t120*3.0, t386 = oy*t115*3.0, t387 = oy*t116*3.0, t388 = oz*t111*3.0, t389 = oz*t112*3.0, t390 = t19*t26;
    T t391 = t14*t26, t392 = t217*3.0, t394 = t21*t26, t395 = t15*t26, t396 = t7*t66, t397 = t7*t29, t398 = t218*3.0, t401 = t219*3.0, t402 = t21*t27, t403 = t16*t27, t404 = t10*t69, t405 = t10*t30, t406 = t226*3.0, t409 = t227*3.0;
    T t410 = t12*t69, t411 = t12*t31, t413 = t228*3.0, t420 = -t237, t421 = ox*t53*-2.0, t426 = oy*t54*-2.0, t428 = -t256, t432 = oz*t55*-2.0, t434 = -t272, t438 = t23*t53, t439 = t24*t54, t440 = t25*t55, t441 = -t82, t442 = -t86, t443 = -t91;
    T t453 = t61*tx, t454 = t62*ty, t455 = t60*tz, t460 = t34*ty*-2.0, t462 = t44*tz*-2.0, t466 = t48*tz*-2.0, t468 = t2*t76, t469 = t3*t75, t473 = t4*t77, t474 = rdx*t75*ty, t475 = rdy*t75*tx, t476 = rdx*t76*tz, t477 = rdz*t76*tx;
    T t478 = rdy*t77*tz, t479 = rdz*t77*ty, t480 = t14*t60, t481 = t14*t62, t482 = t15*t61, t483 = t15*t62, t484 = t16*t60, t485 = t16*t61, t489 = t450*tx, t490 = t450*ty, t491 = t451*tx, t492 = t451*tz, t493 = t452*ty, t494 = t452*tz;
    T t495 = t348*tx, t496 = t349*tx, t497 = t348*ty, t498 = t350*tx, t499 = t349*ty, t500 = t348*tz, t501 = t350*ty, t502 = t349*tz, t503 = t350*tz, t504 = t26+t27, t505 = t26+t28, t506 = t27+t28, t510 = t75*(t15-ty), t513 = t76*(t16-tx);
    T t517 = t77*(t14-tz), t528 = t17*t348, t529 = t17*t349, t530 = t19*t348, t531 = t19*t350, t532 = t21*t349, t533 = t21*t350, t546 = t61*t348, t547 = t62*t348, t548 = t60*t349, t549 = t61*t349, t550 = t60*t350, t551 = t62*t350, t244 = t126*tz;
    T t245 = t132*ty, t254 = t131*tz, t257 = t145*ty, t264 = t141*tz, t265 = t143*tz, t315 = -t131, t316 = -t132, t318 = -t140, t319 = -t141, t320 = -t142, t321 = -t143, t322 = -t144, t323 = -t145, t332 = -t154, t352 = oy*t295, t353 = t189*3.0;
    T t355 = ox*t298, t356 = oz*t295, t357 = t191*3.0, t359 = t194*3.0, t361 = t133*tz, t362 = t134*ty, t363 = t135*tz, t364 = t137*tx, t365 = t138*ty, t366 = t139*tx, t374 = ox*t301, t375 = oz*t298, t377 = t197*3.0, t379 = t200*3.0;
    T t380 = oy*t301, t382 = t202*3.0, t435 = rdx*t277, t436 = rdy*t278, t437 = rdz*t279, t456 = -t351, t458 = -t191, t459 = -t194, t461 = -t370, t465 = -t202, t467 = -t383, t470 = -t385, t471 = -t386, t472 = -t389, t486 = t453*2.0;
    T t487 = t454*2.0, t488 = t455*2.0, t519 = t495*2.0, t520 = t496*2.0, t521 = t497*2.0, t522 = t501*2.0, t523 = t502*2.0, t524 = t503*2.0, t534 = -t490, t535 = -t491, t536 = -t494, t558 = t6+t7+t56+t492, t559 = t8+t10+t50+t493;
    T t560 = t11+t12+t57+t489, t561 = t2+t3+t74+t160+t337, t562 = t2+t4+t73+t159+t339, t563 = t3+t4+t72+t161+t338, t567 = t117+t308+t546, t568 = t122+t304+t548, t569 = t114+t307+t551, t570 = t542+t547, t573 = t540+t550, t574 = t545+t549;
    T t576 = t347+t481+t529, t577 = t347+t484+t530, t578 = t347+t482+t533, t582 = t183+t528+t543, t583 = t183+t531+t541, t584 = t183+t532+t544, t457 = -t353, t463 = -t377, t464 = -t379, t564 = t6+t7+t343+t536, t565 = t8+t10+t345+t534;
    T t566 = t11+t12+t344+t535, t585 = t34+t36+t123+t150+t198+t244+t260+t330+t331+t371+t381+t421+t423, t586 = t48+t49+t126+t136+t199+t257+t271+t313+t314+t360+t372+t425+t426, t587 = t35+t37+t151+t155+t190+t232+t264+t328+t329+t358+t367+t431+t432;
    T t588 = t129+t130+t201+t245+t275+t310+t315+t324+t326+t373+t378+t421+t422, t589 = t152+t153+t192+t234+t254+t311+t312+t317+t332+t368+t376+t426+t427, t590 = t127+t128+t193+t243+t265+t316+t325+t327+t333+t354+t369+t430+t432;
    T t591 = t38+t40+t134+t137+t197+t238+t269+t322+t323+t359+t370+t420+t424+t456+t457+t458, t592 = t39+t42+t133+t139+t189+t235+t255+t320+t321+t351+t357+t428+t433+t464+t465+t467;
    T t593 = t41+t43+t135+t138+t200+t253+t273+t318+t319+t382+t383+t419+t434+t459+t461+t463;
    
    fdrv(0, 0) = t75*t558+t76*t565-t350*t506; fdrv(0, 1) = t14*t505+t76*t560+t350*t564; fdrv(0, 2) = t15*t504+t75*t566+t350*t559;
    fdrv(0, 3) = -t189+t195+t203+t238+t256+t424+t440+t458-t2*t9*2.0+t4*t9+t3*t13-t15*t33*3.0+t14*t37*3.0-t3*t53-t3*t54*2.0+t4*t53+t19*t42-t21*t40-t21*t41*2.0+t43*t66;
    fdrv(0, 4) = -t190+t193*3.0+t204+t243+t261+t352+t417+t4*t7-t3*t51*2.0+t17*t37+t4*t51+t19*t37*3.0-t3*t56*3.0+t29*t43+t25*t56-t11*t88+t4*t101+t4*t106*3.0-t3*t109*2.0+t19*t151-t34*ty*6.0-t38*tz*2.0-t39*tz*4.0+t40*tz+t41*tz+oy*t7*t76;
    fdrv(0, 5) = -t192-t196+t199*3.0+t250+t271+t356+t418+oz*t225+t3*t10-t3*t50-t21*t33*3.0-t15*t41*2.0-t3*t58*2.0+t4*t58*3.0-t21*t48*4.0+t25*t50-t3*t96+t12*t89-t3*t105*3.0+t25*t100+t17*t312-t39*ty*2.0+t42*ty+t43*ty-t34*tz*4.0-t44*tz*6.0;
    fdrv(0, 6) = t586*tdy-t590*tdz+tdx*(t36*4.0+t46*4.0-t246+t248-t380-ox*t27-ox*t28+t4*t12*2.0+t11*t73+t47*ty*2.0);
    fdrv(0, 7) = t586*tdx+t593*tdz+tdy*(t34*-1.2E+1+t36*6.0+t130+t148-t150+t248-t267+t276+t310+t366+t466+t11*t25+t47*ty*1.2E+1-t33*tz*2.0);
    fdrv(0, 8) = -t590*tdx+t593*tdy-tdz*(t36*-2.0+t44*1.2E+1-t46*6.0+t123+t131+t246+t267-t276+t331+t364+t12*t24-t37*ty*2.0-t47*ty*2.0+t48*tz*1.2E+1);
    fdrv(0, 9) = tdy*(t87+t88+t167+t209+t285+t288+t334+t388-t3*t15*2.0)+tdz*(t89+t165+t279+t287+t291+t335+t441+t469+t471)-tdx*(t79+t83+t90+t208+t387+t443+t472+t473-t3*tx*4.0-t4*tx*4.0);
    fdrv(0, 10) = -tdy*(t78+t90-t91*4.0+t208*3.0-t210*2.0+t286+t336+ox*t68+oy*t116-oz*t112*2.0-t4*tx)-tdz*ty*(t80+t84+t88*3.0+t342+ox*tz*4.0-oz*tx)+t563*tdx*ty;
    fdrv(0, 11) = -tdz*(t78+t90*4.0-t210*3.0+t284+t341+t443+ox*t71+t16*t24-t3*tx+t160*tx+oz*t75*tx)+tdy*tz*(t85+t89*3.0+t92+t340-ox*ty*4.0+oy*tx)+t563*tdx*tz;
    fdrv(0, 12) = -tdx*(t95+t99+t171+t178)-tdy*(t96+t105*2.0+t168+t182+t300)-tdz*(t101+t106*2.0+t170+t179+t303);
    fdrv(0, 13) = -tdx*(-t100+t102-t169+t221+t281+t345+t399+t404)+tdy*(t57+t93*2.0+t95*3.0+t99-t107+t171-t407-t12*t21*2.0+t7*t76+t26*t77)-tdz*(-t55+t94+t103-t174+t217+t219+t302+t398+t405+t406+t411);
    fdrv(0, 14) = tdz*(t52+t95+t99*3.0+t108+t178+t225+t280+t295+t403+t412)+tdy*(t54+t94+t110+t174+t218+t227+t299+t392+t397+t402+t413)+tdx*(t56+t97+t109+t172+t220+t282+t393+t396);
    fdrv(0, 15) = oz*t569-ox*(t14*t61+t76*(t14-tz))-oy*(t112*-2.0+t118+t350*(t15-ty)); fdrv(0, 16) = -oy*(t483+t350*(t16-tx))+oz*t583-ox*(t480+t522+t16*t75); fdrv(0, 17) = -oy*t578+oz*t573-ox*(t229+t513+t524);
    fdrv(0, 18) = t185+t186+t437-rdx*t79-rdx*t83-rdx*t90+rdx*t91+rdx*t210+rdx*t389+rdy*t334+rdz*t335-oy*rdx*t116*3.0-oy*rdz*t115*2.0+rdx*t3*t77-rdy*t3*t15*2.0+rdx*t3*tx*4.0+rdx*t4*tx*4.0+rdy*t4*ty+rdy*t161*ty+rdz*t3*tz;
    fdrv(0, 19) = -rdy*(-oz*(t30+t67+t112+t522+tx*(t14-tz))+oy*(t116+t503+t538)+ox*(t68+t69+t346+t60*tx))+rdz*(-t162+oy*(t15-t119*2.0)+oz*(t75+t503+t60*ty*2.0+tx*(t15-ty)))+rdx*(-ox*(t14+t453+t76*tx)+oz*(t18+t487+t498)+oy*(t63+t350-ty*(t15-ty)*2.0));
    fdrv(0, 20) = -rdz*(ox*(t59+t66+t71-tx*(t16-tx))-oz*(t112+t488+t501)+oy*(-t29+t70+t116+t453+t524))-rdy*(t162-oz*(t14+t309)+oy*(t15+t501+t62*tx+tz*(t16-tx)*2.0))-rdx*(-oz*(t63+t350+t62*tz*2.0)+ox*(t15+t112+t507)+oy*(t18+t498+t539));
    fdrv(0, 24) = t215*-2.0-t216*2.0+t445+t446+t474+t475+t476+t477; fdrv(0, 25) = -rdy*(t483+t350*(t16-tx))-rdz*t578-rdx*(t112*-2.0+t118+t350*(t15-ty)); fdrv(0, 26) = rdx*t569+rdy*t583+rdz*t573; fdrv(1, 0) = t14*t506+t77*t565+t349*t558;
    fdrv(1, 1) = t77*t560+t75*t564-t349*t505; fdrv(1, 2) = t16*t504+t75*t559+t349*t566;
    fdrv(1, 3) = t190*3.0-t193-t204+t232+t262+t355+ox*t221-t3*t7*2.0+t4*t6+t2*t51*3.0-t17*t37*4.0-t4*t51*2.0-t19*t37*2.0-t14*t43*3.0-t4*t56+t23*t56+t8*t83-t4*t101*3.0-t4*t106+t23*t109+t19*t327-t33*tx*6.0+t38*tz+t39*tz-t40*tz*2.0-t41*tz*4.0;
    fdrv(1, 4) = t187-t197+t203+t237+t273+t419+t438+t459+t4*t5+t2*t13-t3*t13*2.0+t16*t34*3.0-t14*t37*2.0+t2*t54-t4*t54-t4*t55*2.0+t21*t38-t17*t43-t19*t43*3.0+t39*t69;
    fdrv(1, 5) = t188-t198+t201*3.0+t241+t275+t375+t429+t2*t12+t19*t34+t2*t52+t21*t34*3.0-t4*t52*2.0-t4*t57*3.0+t31*t39+t23*t57-t10*t85+t2*t95-t4*t93*2.0-t10*t89+t2*t99*3.0+t21*t150-t41*tx*2.0+t43*tx+t37*ty-t33*tz*4.0-t48*tz*6.0;
    fdrv(1, 6) = -t588*tdy+t592*tdz-tdx*(t32*-6.0+t33*1.2E+1-t49*2.0+t136+t154-t239+t259+t274+t313+t365+t462+t8*t25+t37*tx*1.2E+1-t34*tz*2.0);
    fdrv(1, 7) = -t588*tdx+t587*tdz+tdy*(t32*4.0+t49*4.0+t258-t259-t356-oy*t26-oy*t28+t2*t8*2.0+t10*t74+t44*tz*2.0);
    fdrv(1, 8) = t592*tdx+t587*tdy+tdz*(t48*-1.2E+1+t49*6.0+t125-t126+t153+t239+t258-t274+t317+t362+t10*t23-t37*tx*2.0-t43*ty*2.0+t44*tz*1.2E+1);
    fdrv(1, 9) = -tdx*(t84+t87*4.0-t206*3.0+t289+t334+t442+oy*t65+t15*t25-t4*ty+t161*ty+ox*t77*ty)+tdz*tx*(t78+t79+t83*3.0+t341-oy*tz*4.0+oz*ty)+t562*tdy*tx;
    fdrv(1, 10) = tdz*(t81+t85+t157+t205+t290+t291+t340+t384-t4*t14*2.0)+tdx*(t83+t166+t277+t283+t284+t336+t443+t472+t473)+tdy*(-t80+t86-t87-t88+t206+t385-t388+t4*t76+t2*ty*4.0+t4*ty*4.0);
    fdrv(1, 11) = -tdz*(t84-t86*4.0+t87-t206*2.0+t209*3.0+t285+t342-ox*t120*2.0+oy*t71+oz*t111-t2*ty)-tdx*tz*(t85*3.0+t89+t92+t335-ox*ty+oy*tx*4.0)+t562*tdy*tz;
    fdrv(1, 12) = tdx*(t58+t96*3.0+t102+t105+t168+t221+t281+t298+t399+t404)+tdz*(t55+t94+t103+t175+t217+t219+t302+t398+t405+t406+t411)+tdy*(t57+t93+t107+t173+t223+t280+t407+t410);
    fdrv(1, 13) = -tdy*(t96+t105+t168+t181)-tdx*(t95+t99*2.0+t171+t180+t297)-tdz*(t101*2.0+t106+t172+t177+t303);
    fdrv(1, 14) = -tdy*(t98-t109-t179+t222+t282+t343+t391+t400)+tdz*(t50+t96+t100*2.0-t104+t105*3.0+t181-t224-t281-t408+t26*t76)-tdx*(-t53+t103+t110-t175+t226+t228+t296+t390+t394+t401+t409); fdrv(1, 15) = ox*t574-oz*t576-oy*(t231+t510+t520);
    fdrv(1, 16) = ox*t568-oy*(t16*t62+t75*(t16-tx))-oz*(t113-t120*2.0+t349*(t14-tz)); fdrv(1, 17) = -oz*(t480+t349*(t15-ty))+ox*t584-oy*(t485+t523+t21*t75);
    fdrv(1, 18) = -rdx*(oy*(t59+t65+t69-ty*(t15-ty))-ox*(t120+t486+t502)+oz*(-t31+t64+t111+t454+t520))-rdz*(t163-ox*(t16+t306)+oz*(t14+t502+t60*ty+tx*(t15-ty)*2.0))-rdy*(-ox*(t66+t349+t60*tx*2.0)+oy*(t14+t120+t508)+oz*(t20+t499+t537));
    fdrv(1, 19) = t184+t186+t435+rdx*t336-rdy*t80+rdy*t86-rdy*t87-rdy*t88+rdy*t206+rdy*t385+rdz*t340-oz*rdx*t112*2.0-oz*rdy*t111*3.0+rdy*t4*t76-rdz*t4*t14*2.0+rdx*t4*tx+rdy*t2*ty*4.0+rdy*t4*ty*4.0+rdz*t2*tz+rdz*t159*tz;
    fdrv(1, 20) = -rdz*(-ox*(t29+t70+t120+t523+ty*(t16-tx))+oz*(t111+t496+t539)+oy*(t63+t71+t346+t61*ty))+rdx*(-t163+oz*(t14-t116*2.0)+ox*(t77+t496+t61*tz*2.0+ty*(t14-tz)))+rdy*(-oy*(t16+t454+t75*ty)+ox*(t20+t488+t499)+oz*(t66+t349-tz*(t14-tz)*2.0));
    fdrv(1, 24) = rdx*t574+rdy*t568+rdz*t584; fdrv(1, 25) = t213*-2.0-t214*2.0+t444+t446+t474+t475+t478+t479; fdrv(1, 26) = -rdz*(t480+t349*(t15-ty))-rdx*t576-rdy*(t113-t120*2.0+t349*(t14-tz)); fdrv(2, 0) = t15*t506+t77*t558+t348*t565;
    fdrv(2, 1) = t16*t505+t76*t564+t348*t560; fdrv(2, 2) = t76*t559+t77*t566-t348*t504;
    fdrv(2, 3) = t192*3.0+t196-t199+t234+t252+t374+t3*t8-t4*t10*2.0-t2*t50*3.0+t15*t41*3.0-t2*t58*2.0+t3*t58+t21*t48+t24*t50-t6*t79-t6*t83+t3*t96*3.0-t2*t100*2.0+t3*t105+t26*t88+t17*t126-t37*tx*6.0+t39*ty-t42*ty*2.0-t43*ty*4.0+t34*tz;
    fdrv(2, 4) = -t188+t198*3.0-t201+t242+t260+t380+t2*t11-t4*t12*2.0-t19*t34*4.0-t2*t52*2.0+t3*t52*3.0-t16*t39*3.0-t21*t34*2.0-t2*t57+t24*t57+t7*t80+t7*t88-t2*t95*3.0-t2*t99+t24*t93+t21*t324+t41*tx-t43*tx*2.0-t37*ty*4.0-t47*ty*6.0+t33*tz;
    fdrv(2, 5) = t187+t195-t200+t255+t272+t433+t439+t465+oy*t395+t3*t5-t4*t5*2.0+t2*t9-t16*t34*2.0-t2*t53*2.0-t2*t55+t3*t55+t17*t41-t19*t39-t21*t39*3.0+t21*t41*3.0;
    fdrv(2, 6) = t591*tdy+t585*tdz+tdx*(t35*6.0-t37*1.2E+1+t127+t147-t151-t233+t247+t268+t333+t363+t460+t6*t24+t33*tx*1.2E+1-t39*tz*2.0);
    fdrv(2, 7) = t591*tdx-t589*tdz-tdy*(t35*-2.0-t45*6.0+t47*1.2E+1+t132+t155+t233-t247+t266+t328+t361+t7*t23-t33*tx*2.0+t34*ty*1.2E+1-t41*tz*2.0);
    fdrv(2, 8) = t585*tdx-t589*tdy+tdz*(t35*4.0+t45*4.0-t266+t268-t355-oz*t26-oz*t27+t3*t7*2.0+t6*t72+t33*tx*2.0);
    fdrv(2, 9) = -tdx*(t81-t82*4.0+t92+t205*3.0-t207*2.0+t290+t335+ox*t119-oy*t115*2.0+oz*t65-t3*tz)-tdy*tx*(t78+t79*3.0+t83+t336-oy*tz+oz*ty*4.0)+t561*tdz*tx;
    fdrv(2, 10) = -tdy*(t81*4.0+t92-t207*3.0+t287+t340+t441+oz*t68+t14*t23-t2*tz+t159*tz+oy*t76*tz)+tdx*ty*(t80*3.0+t84+t88+t334+ox*tz-oz*tx*4.0)+t561*tdz*ty;
    fdrv(2, 11) = tdx*(t79+t90+t158+t208+t283+t286+t341+t387-t2*t16*2.0)+tdy*(t80+t156+t278+t288+t289+t342+t442+t468+t470)-tdz*(t81+t85+t89+t205+t384+t441+t469+t471-t2*tz*4.0-t3*tz*4.0);
    fdrv(2, 12) = tdx*(t51*2.0+t56-t97+t101*3.0+t106+t109*2.0-t220-t282-t393-t7*t19*2.0)-tdz*(-t93+t108-t180+t225+t280+t344+t403+t412)-tdy*(-t54+t94+t110-t176+t218+t227+t299+t392+t397+t402+t413);
    fdrv(2, 13) = tdy*(t51+t98+t101+t106*3.0+t177+t222+t282+t301+t391+t400)+tdx*(t53+t103+t110+t176+t226+t228+t296+t390+t394+t401+t409)+tdz*(t50+t100+t104+t182+t224+t281+t395+t408);
    fdrv(2, 14) = -tdz*(t101+t106+t170+t177)-tdx*(t95*2.0+t99+t173+t178+t297)-tdy*(t96*2.0+t105+t169+t181+t300); fdrv(2, 15) = -ox*(t485+t348*(t14-tz))+oy*t582-oz*(t483+t519+t15*t75); fdrv(2, 16) = -ox*t577+oy*t570-oz*(t230+t517+t521);
    fdrv(2, 17) = oy*t567-ox*(t115*-2.0+t121+t348*(t16-tx))-oz*(t15*t60+t77*(t15-ty));
    fdrv(2, 18) = -rdx*(-oy*(t31+t64+t115+t519+tz*(t15-ty))+ox*(t119+t497+t537)+oz*(t65+t66+t346+t62*tz))+rdy*(-t164+ox*(t16-t111*2.0)+oy*(t76+t497+t62*tx*2.0+tz*(t16-tx)))+rdz*(-oz*(t15+t455+t77*tz)+oy*(t22+t486+t500)+ox*(t69+t348-tx*(t16-tx)*2.0));
    fdrv(2, 19) = -rdy*(oz*(t59+t63+t68-tz*(t14-tz))-oy*(t115+t487+t495)+ox*(-t30+t67+t119+t455+t521))-rdx*(t164-oy*(t15+t305)+ox*(t16+t495+t61*tz+ty*(t14-tz)*2.0))-rdz*(-oy*(t69+t348+t61*ty*2.0)+ox*(t22+t500+t538)+oz*(t16+t115+t509));
    fdrv(2, 20) = t184+t185+t436+rdx*t341-rdz*t81+rdz*t82-rdz*t85+rdy*t342-rdz*t89+rdz*t207+rdz*t386-ox*rdy*t120*2.0-ox*rdz*t119*3.0-rdx*t2*t16*2.0+rdz*t2*t75+rdx*t3*tx+rdx*t160*tx+rdy*t2*ty+rdz*t2*tz*4.0+rdz*t3*tz*4.0;
    fdrv(2, 24) = -rdx*(t485+t348*(t14-tz))-rdy*t577-rdz*(t115*-2.0+t121+t348*(t16-tx)); fdrv(2, 25) = rdx*t582+rdy*t570+rdz*t567; fdrv(2, 26) = t211*-2.0-t212*2.0+t444+t445+t476+t477+t478+t479;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f27(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = ox*ty, t4 = oy*tx, t5 = ox*tz, t6 = oy*ty, t7 = oz*tx, t8 = oy*tz, t9 = oz*ty, t10 = oz*tz, t11 = tx*tx, t12 = tx*tx*tx, t13 = ty*ty, t14 = ty*ty*ty, t15 = tz*tz, t16 = tz*tz*tz, t26 = rx*tx*3.0, t27 = ry*ty*3.0;
    T t28 = rz*tz*3.0, t35 = rx*tx*ty, t36 = rx*tx*tz, t37 = ry*tx*ty, t38 = ry*ty*tz, t39 = rz*tx*tz, t40 = rz*ty*tz, t17 = t2*2.0, t18 = t3*2.0, t19 = t4*2.0, t20 = t5*2.0, t21 = t6*2.0, t22 = t7*2.0, t23 = t8*2.0, t24 = t9*2.0, t25 = t10*2.0;
    T t29 = t2*ty, t30 = t2*tz, t31 = t4*ty, t32 = t6*tz, t33 = t7*tz, t34 = t9*tz, t41 = t11*3.0, t42 = t13*3.0, t43 = t15*3.0, t44 = -t4, t46 = -t7, t48 = -t9, t50 = t2*tx, t51 = t2*t11, t52 = t3*ty, t53 = t4*tx, t54 = t5*tz, t55 = t6*ty;
    T t56 = t7*tx, t57 = t6*t13, t58 = t8*tz, t59 = t9*ty, t60 = t10*tz, t61 = t10*t15, t62 = rdx*t11, t63 = rdy*t13, t64 = rdz*t15, t65 = rx*t11, t66 = rx*t12, t67 = ry*t13, t68 = ry*t14, t69 = rz*t15, t70 = rz*t16, t77 = t35*2.0, t78 = t36*2.0;
    T t79 = t37*2.0, t80 = t38*2.0, t81 = t39*2.0, t82 = t40*2.0, t89 = ry*t5*t15, t90 = rz*t3*t13, t91 = rx*t8*t15, t92 = rz*t4*t11, t93 = rx*t9*t13, t94 = ry*t7*t11, t109 = rx*t3*tz*4.0, t112 = rx*t4*tz*4.0, t114 = rx*t7*ty*4.0;
    T t117 = rx*t4*tz*6.0, t118 = rx*t7*ty*6.0, t125 = ry*t3*tz*4.0, t127 = ry*t4*tz*4.0, t129 = ry*t7*ty*4.0, t130 = ry*t3*tz*6.0, t133 = ry*t7*ty*6.0, t138 = rz*t3*tz*4.0, t140 = rz*t4*tz*4.0, t143 = rz*t7*ty*4.0, t144 = rz*t3*tz*6.0;
    T t145 = rz*t4*tz*6.0, t159 = t35*tz*4.0, t161 = t37*tz*4.0, t163 = t39*ty*4.0, t182 = t11+t13, t183 = t11+t15, t184 = t13+t15, t189 = t4*t26, t193 = t3*t27, t196 = t7*t26, t215 = t5*t28, t218 = t9*t27, t222 = t8*t28, t239 = rx*t13*tx*2.0;
    T t240 = rx*t15*tx*2.0, t241 = ry*t11*ty*2.0, t242 = rx*t15*ty*2.0, t243 = rx*t13*tz*2.0, t244 = ry*t15*tx*2.0, t245 = ry*t11*tz*2.0, t246 = rz*t13*tx*2.0, t247 = rz*t11*ty*2.0, t248 = ry*t15*ty*2.0, t249 = rz*t11*tz*2.0, t250 = rz*t13*tz*2.0;
    T t255 = rx*t4*tz*-2.0, t256 = rx*t7*ty*-2.0, t259 = ry*t3*tz*-2.0, t264 = ry*t7*ty*-2.0, t266 = rz*t3*tz*-2.0, t267 = rz*t4*tz*-2.0, t299 = t3*t15*-2.0, t301 = t4*t15*-2.0, t303 = t7*t13*-2.0, t320 = t26+t27, t321 = t26+t28, t322 = t27+t28;
    T t45 = -t19, t47 = -t22, t49 = -t24, t71 = t17*ty, t72 = t17*tz, t73 = t19*ty, t74 = t21*tz, t75 = t22*tz, t76 = t24*tz, t83 = ry*t50, t84 = rx*t53, t85 = ry*t52, t86 = rz*t50, t87 = rx*t55, t88 = rx*t56, t95 = rz*t54, t96 = rz*t55;
    T t97 = rx*t60, t98 = ry*t59, t99 = rz*t58, t100 = ry*t60, t101 = rx*t29*4.0, t104 = rx*t30*4.0, t110 = ry*t30*4.0, t111 = rz*t29*4.0, t113 = ry*t31*4.0, t115 = ry*t30*6.0, t116 = rz*t29*6.0, t126 = rx*t32*4.0, t128 = rz*t31*4.0;
    T t131 = rx*t32*6.0, t132 = rz*t31*6.0, t139 = ry*t32*4.0, t141 = rx*t34*4.0, t142 = ry*t33*4.0, t146 = rx*t34*6.0, t147 = ry*t33*6.0, t150 = rz*t33*4.0, t151 = rz*t34*4.0, t153 = t29*tz*4.0, t155 = t31*tz*4.0, t157 = t33*ty*4.0;
    T t158 = t77*tz, t160 = t79*tz, t162 = t81*ty, t164 = t11*t17, t165 = t51*4.0, t166 = t13*t21, t167 = t57*4.0, t168 = t15*t25, t169 = t61*4.0, t170 = t66*2.0, t171 = t66*4.0, t172 = t68*2.0, t173 = t68*4.0, t174 = t70*2.0, t175 = t70*4.0;
    T t176 = -t77, t177 = -t78, t178 = -t79, t179 = -t80, t180 = -t81, t181 = -t82, t185 = t17*t65, t186 = rx*t50*6.0, t187 = rx*t18*ty, t190 = rx*t52*6.0, t191 = rx*t20*tz, t192 = ry*t19*tx, t197 = rx*t54*6.0, t198 = ry*t53*6.0, t199 = ry*t20*tz;
    T t200 = rz*t18*ty, t201 = rx*t23*tz, t202 = rz*t19*tx, t203 = rx*t24*ty, t204 = ry*t22*tx, t205 = ry*t54*3.0, t206 = rz*t52*3.0, t207 = rx*t58*3.0, t208 = t21*t67, t209 = rz*t53*3.0, t210 = rx*t59*3.0, t211 = ry*t56*3.0, t212 = ry*t55*6.0;
    T t219 = ry*t58*6.0, t220 = rz*t56*6.0, t224 = rz*t59*6.0, t225 = t25*t69, t226 = rz*t60*6.0, t227 = t13*t17, t228 = t15*t17, t230 = t15*t18, t231 = t18*ty*tz, t232 = t15*t19, t233 = t19*tx*tz, t234 = t13*t22, t235 = t22*tx*ty, t236 = t15*t21;
    T t251 = rx*t29*1.2E+1, t252 = rx*t30*1.2E+1, t253 = ry*t30*-2.0, t254 = rz*t29*-2.0, t257 = -t109, t258 = ry*t31*1.2E+1, t261 = rx*t32*-2.0, t262 = rz*t31*-2.0, t263 = rx*t33*-2.0, t265 = -t127, t268 = rx*t34*-2.0, t269 = ry*t33*-2.0;
    T t270 = -t143, t271 = ry*t32*1.2E+1, t274 = rz*t33*1.2E+1, t275 = rz*t34*1.2E+1, t277 = t17*t38, t279 = t17*t40, t282 = t19*t40, t283 = t22*t38, t285 = -t52, t286 = t44*tx, t287 = -t54, t288 = t46*tx, t289 = -t58, t290 = t48*ty;
    T t294 = rz*t56*-2.0, t300 = t52*tz*-2.0, t302 = t53*tz*-2.0, t304 = t56*ty*-2.0, t305 = -t242, t306 = -t243, t307 = -t244, t308 = -t245, t309 = -t246, t310 = -t247, t311 = rx*t182, t312 = rx*t183, t313 = ry*t182, t314 = ry*t184;
    T t315 = rz*t183, t316 = rz*t184, t317 = t3+t44, t318 = t5+t46, t319 = t8+t48, t323 = t15+t182, t327 = t320*tx, t328 = t320*ty, t329 = t321*tx, t330 = t321*tz, t331 = t322*ty, t332 = t322*tz, t333 = t2+t21+t25, t334 = t6+t17+t25;
    T t335 = t10+t17+t21, t336 = t41+t184, t337 = t42+t183, t338 = t43+t182, t102 = ry*t71, t103 = rx*t73, t106 = rz*t71, t120 = rz*t72, t121 = rx*t74, t123 = rx*t75, t137 = ry*t75, t152 = t71*tz, t154 = t73*tz, t156 = t75*ty, t188 = t83*3.0;
    T t194 = t86*3.0, t195 = t87*3.0, t216 = t96*3.0, t217 = t97*3.0, t223 = t100*3.0, t229 = t73*tx, t237 = t75*tx, t238 = t76*ty, t276 = t101*tz, t281 = t113*tz, t284 = t150*ty, t292 = -t212, t324 = t18+t45, t325 = t20+t47, t326 = t23+t49;
    T t339 = -t327, t340 = -t328, t341 = -t329, t342 = -t330, t343 = -t331, t344 = -t332, t345 = t33+t50+t73+t285, t346 = t31+t50+t75+t287, t347 = t34+t55+t71+t286, t348 = t29+t55+t76+t289, t349 = t32+t60+t72+t288, t350 = t30+t60+t74+t290;
    T t278 = t103*tz, t280 = t123*ty, t351 = t37+t180+t311+t312+t339, t352 = t39+t178+t311+t312+t341, t353 = t35+t181+t313+t314+t340, t354 = t40+t176+t313+t314+t343, t355 = t36+t179+t315+t316+t342, t356 = t38+t177+t315+t316+t344;
    T t360 = t83+t84+t101+t138+t141+t193+t195+t205+t207+t267+t269+t270, t361 = t86+t88+t104+t125+t126+t206+t210+t215+t217+t262+t264+t265, t362 = t85+t87+t113+t140+t142+t188+t189+t205+t207+t266+t268+t270;
    T t363 = t95+t97+t128+t129+t150+t194+t196+t206+t210+t259+t261+t265, t364 = t96+t98+t110+t112+t139+t209+t211+t222+t223+t254+t256+t257, t365 = t99+t100+t111+t114+t151+t209+t211+t216+t218+t253+t255+t257;
    
    fdrv(0, 0) = -t355*tx*ty+t353*tx*tz;
    fdrv(0, 1) = -t184*t356-t351*tx*tz; fdrv(0, 2) = t184*t354+t352*tx*ty;
    fdrv(0, 3) = t89-t90+t91+t281+t3*t38-t7*t35*3.0-t3*t69-t7*t67*2.0+t19*t69+t189*tz+rx*t13*t48+rx*t15*t48+ry*t15*t22-rz*t4*t13*2.0+ry*t30*tx*3.0+rx*t32*ty-rz*t29*tx*3.0-rz*t33*ty*4.0;
    fdrv(0, 4) = t277+t278-rz*t51-rz*t57*4.0-rz*t61*2.0-t7*t37*2.0-t2*t69-t9*t67*4.0+t46*t65+t262*tx+t192*tz-rx*t7*t13*3.0+rx*t15*t46+ry*t15*t23-rz*t2*t13*3.0+ry*t32*ty*6.0-rz*t33*tx*2.0-rz*t34*ty*6.0;
    fdrv(0, 5) = t208+ry*t51+ry*t169+t19*t37+t19*t39+t2*t67+t4*t65+t8*t69*4.0-t9*t69*6.0+t137*tx+t263*ty+t294*ty+t254*tz+rx*t4*t13+rx*t4*t43+ry*t6*t15*6.0+ry*t2*t43-rz*t9*t13*2.0;
    fdrv(0, 6) = tdx*(t99*2.0+t100*2.0+t115-t116+t117-t118+t139-t151+ry*t49*ty-rz*t21*ty)-t363*tdy+t362*tdz;
    fdrv(0, 7) = -t363*tdx-tdz*(-t102-t219+t224+t226+t292+rx*t33*2.0+rz*t30*2.0+rz*t56*2.0+rx*t45*ty+ry*t45*tx)-tdy*(t96*1.2E+1+t98*1.2E+1+t116+t118+t202+t204+t253+t255-t271+t275);
    fdrv(0, 8) = t362*tdx-tdy*(-t102-t219+t224+t226+t292+rx*t33*2.0+rz*t30*2.0+rz*t56*2.0+rx*t45*ty+ry*t45*tx)+tdz*(t99*1.2E+1+t100*1.2E+1+t115+t117+t202+t204+t254+t256+t271-t275);
    fdrv(0, 9) = -tdy*tx*(t32*-2.0+t56+t59*3.0+t60)+tdz*tx*(t34*-2.0+t53+t55+t58*3.0)+t319*t336*tdx;
    fdrv(0, 10) = tdx*(t155+t303+t5*t15+t15*t22+t30*tx*3.0+t52*tz)+tdz*(t51+t166+t169+t229+t237+t2*t13+t6*t15*6.0+t2*t43)+tdy*(t152+t233+t304-t9*t13*4.0+t15*t23+t32*ty*6.0);
    fdrv(0, 11) = -tdx*(t157+t301+t3*t13+t3*t15+t13*t19+t29*tx*3.0)-tdy*(t51+t167+t168+t229+t237+t2*t15+t2*t42+t34*ty*6.0)-tdz*(t152+t235+t302-t8*t15*4.0+t9*t15*6.0+t13*t24);
    fdrv(0, 12) = tdz*tx*(t67+t181+ry*t11+ry*t43)-tdy*tx*(t69+t179+rz*t11+rz*t42)+t336*tdx*(ry*tz-rz*ty);
    fdrv(0, 13) = tdx*(t161+t309+rx*t16+t69*tx*2.0+rx*t13*tz+rx*t41*tz)+tdz*(t66+t172+t175+t241+t249+t15*t26+rx*t13*tx+ry*t15*ty*6.0)+tdy*(t158+t245+t310+ry*t16*2.0-rz*t14*4.0+t67*tz*6.0);
    fdrv(0, 14) = -tdx*(t163+t307+rx*t14+t67*tx*2.0+rx*t15*ty+rx*t41*ty)-tdy*(t66+t173+t174+t241+t249+t13*t26+rx*t15*tx+rz*t13*tz*6.0)-tdz*(t158+t247+t308-ry*t16*4.0+rz*t14*2.0+t69*ty*6.0); fdrv(0, 15) = t319*t323*tx; fdrv(0, 16) = t323*t350;
    fdrv(0, 17) = -t323*t348; fdrv(0, 18) = t62*t319*2.0+rdy*t5*t323+rdx*t319*t323-rdz*t3*t323+rdy*t350*tx*2.0-rdz*t348*tx*2.0; fdrv(0, 19) = rdx*t46*t323+rdy*t323*t326-rdz*t323*t333+rdy*t350*ty*2.0-rdz*t348*ty*2.0+rdx*t319*tx*ty*2.0;
    fdrv(0, 20) = rdx*t4*t323+rdy*t323*t333+rdz*t323*t326+rdy*t350*tz*2.0-rdz*t348*tz*2.0+rdx*t319*tx*tz*2.0; fdrv(0, 24) = -t323*tx*(rdz*ty-rdy*tz); fdrv(0, 25) = t323*(t64-rdz*t13+rdx*tx*tz+rdy*ty*tz*2.0);
    fdrv(0, 26) = -t323*(t63-rdy*t15+rdx*tx*ty+rdz*ty*tz*2.0); fdrv(1, 0) = t183*t355+t353*ty*tz; fdrv(1, 1) = t356*tx*ty-t351*ty*tz; fdrv(1, 2) = -t183*t352-t354*tx*ty;
    fdrv(1, 3) = t225+rx*t234+rx*t300+rz*t57+rz*t165+rz*t227+t24*t40+t7*t65*4.0+t6*t69+t9*t67+t27*t56-rx*t5*t15*2.0+ry*t9*t15-rx*t30*tx*6.0-rx*t31*tz*2.0+rz*t31*tx*3.0+rz*t33*tx*6.0-ry*t29*tz*2.0;
    fdrv(1, 4) = -t89-t91+t92+t94+t284-t3*t38*3.0+t22*t35-t3*t69*2.0+t4*t69+t36*t44+t106*tx-rx*t9*t15*2.0+ry*t7*t15+ry*t7*t42+rz*t4*t42-ry*t30*tx-rx*t32*ty*3.0-rx*t29*tz*4.0;
    fdrv(1, 5) = t282+t283-rx*t51*2.0-rx*t57-rx*t61*4.0+rz*t234-t3*t40*2.0-t3*t67-t5*t69*4.0+t7*t69*6.0+t268*ty-rx*t2*t13*2.0-rx*t2*t15*6.0-rx*t6*t15*3.0-ry*t3*t15*3.0+rz*t11*t22-rx*t31*tx-ry*t29*tx;
    fdrv(1, 6) = t365*tdy+tdx*(t86*1.2E+1+t88*1.2E+1+t132+t133+t200+t203-t252+t259+t261+t274)+tdz*(-t102-t186-t187-t197+t220+t226+ry*t34*2.0+rz*t32*2.0+rz*t59*2.0+rx*t45*ty);
    fdrv(1, 7) = -tdy*(t95*2.0+t97*2.0+t104+t130+t131-t132-t133-t150+rx*t47*tx-rz*t17*tx)+t365*tdx-t360*tdz;
    fdrv(1, 8) = -t360*tdy-tdz*(t95*1.2E+1+t97*1.2E+1+t130+t131+t200+t203+t252+t262+t264-t274)+tdx*(-t102-t186-t187-t197+t220+t226+ry*t34*2.0+rz*t32*2.0+rz*t59*2.0+rx*t45*ty);
    fdrv(1, 9) = -tdy*(t153+t304+t8*t15+t15*t24+t32*ty*3.0+t53*tz)-tdz*(t57+t164+t169+t227+t238+t2*t15*6.0+t6*t43+t31*tx)-tdx*(t154+t231+t303-t7*t11*4.0+t15*t20+t30*tx*6.0);
    fdrv(1, 10) = tdx*ty*(t30*-2.0+t56*3.0+t59+t60)-tdz*ty*(t33*-2.0+t50+t52+t54*3.0)-t318*t337*tdy;
    fdrv(1, 11) = tdy*(t157+t299+t4*t11+t4*t15+t4*t42+t71*tx)+tdx*(t57+t165+t168+t227+t238+t6*t15+t31*tx*3.0+t33*tx*6.0)+tdz*(t154+t234+t300-t5*t15*4.0+t7*t15*6.0+t11*t22);
    fdrv(1, 12) = -tdy*(t159+t310+ry*t16+t69*ty*2.0+ry*t11*tz+ry*t42*tz)-tdz*(t68+t170+t175+t239+t250+t15*t27+rx*t15*tx*6.0+ry*t11*ty)-tdx*(t160+t243+t309+rx*t16*2.0-rz*t12*4.0+t65*tz*6.0);
    fdrv(1, 13) = -tdz*ty*(t65+t180+rx*t13+rx*t43)+tdx*ty*(t69+t177+rz*t13+rz*t41)-t337*tdy*(rx*tz-rz*tx);
    fdrv(1, 14) = tdy*(t163+t305+ry*t12+t65*ty*2.0+ry*t15*tx+ry*t42*tx)+tdx*(t68+t171+t174+t239+t250+t11*t27+ry*t15*ty+rz*t11*tz*6.0)+tdz*(t160+t246+t306-rx*t16*4.0+rz*t12*2.0+t69*tx*6.0); fdrv(1, 15) = -t323*t349; fdrv(1, 16) = -t318*t323*ty;
    fdrv(1, 17) = t323*t346; fdrv(1, 18) = rdy*t9*t323-rdx*t323*t325+rdz*t323*t334-rdx*t349*tx*2.0+rdz*t346*tx*2.0-rdy*t318*tx*ty*2.0; fdrv(1, 19) = t63*t318*-2.0-rdx*t8*t323+rdz*t4*t323-rdy*t318*t323-rdx*t349*ty*2.0+rdz*t346*ty*2.0;
    fdrv(1, 20) = -rdy*t3*t323-rdx*t323*t334-rdz*t323*t325-rdx*t349*tz*2.0+rdz*t346*tz*2.0-rdy*t318*ty*tz*2.0; fdrv(1, 24) = -t323*(t64-rdz*t11+rdx*tx*tz*2.0+rdy*ty*tz); fdrv(1, 25) = t323*ty*(rdz*tx-rdx*tz);
    fdrv(1, 26) = t323*(t62-rdx*t15+rdy*tx*ty+rdz*tx*tz*2.0); fdrv(2, 0) = -t182*t353-t355*ty*tz; fdrv(2, 1) = t182*t351+t356*tx*tz; fdrv(2, 2) = -t354*tx*tz+t352*ty*tz;
    fdrv(2, 3) = t279+t280+rx*t230+rx*t301-ry*t51*4.0-ry*t57*2.0-ry*t61-t4*t39*3.0-t4*t65*4.0-t8*t69+rx*t13*t18-ry*t2*t15*2.0-ry*t6*t15*2.0+rx*t29*tx*6.0-ry*t31*tx*6.0-ry*t33*tx*3.0-ry*t34*ty-rz*t32*ty;
    fdrv(2, 4) = t185+rx*t61+rx*t167+rx*t228+rx*t236+ry*t230+ry*t301+t3*t67*4.0-t4*t67*6.0+t5*t69+t28*t52+t269*ty+t262*tz+rx*t2*t13*6.0-ry*t4*t11*2.0+rx*t33*tx+rx*t34*ty*3.0+rz*t30*tx;
    fdrv(2, 5) = t90+t93+t276-t4*t36*2.0+t7*t35+t18*t38-t4*t69*3.0+t46*t67+t253*tx+t121*ty+rx*t9*t43-ry*t7*t15*3.0+ry*t11*t46+rz*t3*t43+rz*t11*t44+rz*t13*t44+rz*t29*tx-ry*t31*tz*4.0;
    fdrv(2, 6) = -t364*tdz-tdx*(t83*1.2E+1+t84*1.2E+1+t145+t147+t199+t201-t251+t258+t266+t268)-tdy*(-t120-t186-t190-t191+t198+t212+ry*t34*2.0+ry*t58*2.0+rz*t32*2.0+rx*t47*tz);
    fdrv(2, 7) = t361*tdz+tdy*(t85*1.2E+1+t87*1.2E+1+t144+t146+t199+t201+t251-t258+t267+t269)-tdx*(-t120-t186-t190-t191+t198+t212+ry*t34*2.0+ry*t58*2.0+rz*t32*2.0+rx*t47*tz);
    fdrv(2, 8) = tdz*(t85*2.0+t87*2.0+t101-t113+t144-t145+t146-t147+rx*t45*tx-ry*t17*tx)-t364*tdx+t361*tdy;
    fdrv(2, 9) = tdz*(t153+t302+t9*t13+t9*t43+t56*ty+t74*ty)+tdy*(t61+t164+t167+t228+t236+t2*t13*6.0+t33*tx+t34*ty*3.0)+tdx*(t156+t230+t301-t4*t11*4.0+t13*t18+t29*tx*6.0);
    fdrv(2, 10) = -tdz*(t155+t300+t7*t11+t7*t13+t7*t43+t72*tx)-tdx*(t61+t165+t166+t228+t236+t31*tx*6.0+t33*tx*3.0+t34*ty)-tdy*(t156+t232+t299-t3*t13*4.0+t4*t13*6.0+t11*t19);
    fdrv(2, 11) = -tdx*tz*(t29*-2.0+t53*3.0+t55+t58)+tdy*tz*(t31*-2.0+t50+t52*3.0+t54)+t317*t338*tdz;
    fdrv(2, 12) = tdz*(t159+t308+rz*t14+t67*tz*2.0+rz*t11*ty+rz*t43*ty)+tdy*(t70+t170+t173+t240+t248+t13*t28+rx*t13*tx*6.0+rz*t11*tz)+tdx*(t162+t242+t307+rx*t14*2.0-ry*t12*4.0+t65*ty*6.0);
    fdrv(2, 13) = -tdz*(t161+t306+rz*t12+t65*tz*2.0+rz*t13*tx+rz*t43*tx)-tdx*(t70+t171+t172+t240+t248+t11*t28+ry*t11*ty*6.0+rz*t13*tz)-tdy*(t162+t244+t305-rx*t14*4.0+ry*t12*2.0+t67*tx*6.0);
    fdrv(2, 14) = tdy*tz*(t65+t178+rx*t15+rx*t42)-tdx*tz*(t67+t176+ry*t15+ry*t41)+t338*tdz*(rx*ty-ry*tx); fdrv(2, 15) = t323*t347; fdrv(2, 16) = -t323*t345; fdrv(2, 17) = t317*t323*tz;
    fdrv(2, 18) = rdx*t323*t324-rdz*t8*t323-rdy*t323*t335+rdx*t347*tx*2.0-rdy*t345*tx*2.0+rdz*t317*tx*tz*2.0; fdrv(2, 19) = rdx*t323*t335+rdz*t5*t323+rdy*t323*t324+rdx*t347*ty*2.0-rdy*t345*ty*2.0+rdz*t317*ty*tz*2.0;
    fdrv(2, 20) = t64*t317*2.0+rdx*t9*t323+rdy*t46*t323+rdz*t317*t323+rdx*t347*tz*2.0-rdy*t345*tz*2.0; fdrv(2, 24) = t323*(t63-rdy*t11+rdx*tx*ty*2.0+rdz*ty*tz); fdrv(2, 25) = -t323*(t62-rdx*t13+rdy*tx*ty*2.0+rdz*tx*tz);
    fdrv(2, 26) = -t323*tz*(rdy*tx-rdx*ty);
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f28(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = ox*ty, t4 = oy*tx, t5 = ox*tz, t6 = oy*ty, t7 = oz*tx, t8 = oy*tz, t9 = oz*ty, t10 = oz*tz, t11 = rdx*tx, t12 = rdy*ty, t13 = rdz*tz, t14 = rx*tx, t15 = ry*ty, t16 = rz*tz, t17 = tx*tx, t18 = tx*tx*tx, t20 = ty*ty;
    T t21 = ty*ty*ty, t23 = tz*tz, t24 = tz*tz*tz, t41 = rx*ty*tz, t42 = ry*tx*tz, t43 = rz*tx*ty, t19 = t17*t17, t22 = t20*t20, t25 = t23*t23, t26 = t3*2.0, t27 = t4*2.0, t28 = t5*2.0, t29 = t7*2.0, t30 = t8*2.0, t31 = t9*2.0, t32 = t2*ty;
    T t33 = t2*tz, t34 = t4*ty, t35 = t6*tz, t36 = t7*tz, t37 = t9*tz, t38 = t14*ty, t39 = t14*tz, t40 = t15*tx, t44 = t15*tz, t45 = t16*tx, t46 = t16*ty, t47 = -t4, t49 = -t7, t51 = -t9, t53 = t2*t17, t54 = t3*ty, t55 = t4*tx, t56 = t3*t20;
    T t57 = t4*t17, t58 = t5*tz, t59 = t7*tx, t60 = t5*t23, t61 = t6*t20, t62 = t7*t17, t63 = t8*tz, t64 = t9*ty, t65 = t8*t23, t66 = t9*t20, t67 = t10*t23, t68 = t14*t17, t69 = rx*t20, t70 = ry*t17, t71 = rx*t23, t72 = rz*t17, t73 = t15*t20;
    T t74 = ry*t23, t75 = rz*t20, t76 = t16*t23, t83 = t2*t20, t84 = t2*t23, t86 = t3*t23, t88 = t4*t23, t90 = t7*t20, t92 = t6*t23, t95 = t14*t20, t96 = t14*t23, t97 = t15*t17, t104 = t15*t23, t105 = t16*t17, t106 = t16*t20, t122 = t14*tx*3.0;
    T t124 = t15*ty*3.0, t126 = t16*tz*3.0, t134 = -t42, t135 = -t43, t136 = t17+t20, t137 = t17+t23, t138 = t20+t23, t140 = t2*t14*tx*4.0, t154 = t6*t15*ty*4.0, t162 = t10*t16*tz*4.0, t173 = t4*t20*3.0, t187 = t7*t23*3.0, t191 = t9*t23*3.0;
    T t205 = t3*t21*-2.0, t206 = t4*t18*-2.0, t209 = t5*t24*-2.0, t210 = t7*t18*-2.0, t213 = t8*t24*-2.0, t214 = t9*t21*-2.0, t217 = t14*t18*1.0E+1, t218 = t15*t21*1.0E+1, t219 = t16*t24*1.0E+1, t220 = t2+t6, t221 = t2+t10, t222 = t6+t10;
    T t223 = t14+t15, t224 = t14+t16, t225 = t15+t16, t227 = t2*t14*tx*8.0, t238 = t6*t15*ty*8.0, t249 = t10*t16*tz*8.0, t254 = t4*t20*-2.0, t264 = t7*t23*-2.0, t266 = t9*t23*-2.0, t268 = t14*t21*8.0, t269 = t14*t24*8.0, t270 = t15*t18*8.0;
    T t277 = t15*t24*8.0, t278 = t16*t18*8.0, t279 = t16*t21*8.0, t312 = t7*t15*ty*4.0, t324 = t7*t15*ty*6.0, t326 = t3*t16*tz*4.0, t330 = t4*t16*tz*4.0, t336 = t3*t16*tz*6.0, t338 = t4*t16*tz*6.0, t410 = t3*t41*8.0, t422 = t4*t42*8.0;
    T t429 = t7*t15*ty*2.4E+1, t436 = t7*t43*8.0, t439 = t3*t16*tz*2.4E+1, t441 = t4*t16*tz*2.4E+1, t484 = t11+t12+t13, t48 = -t27, t50 = -t29, t52 = -t31, t77 = t38*2.0, t78 = t39*2.0, t79 = t40*2.0, t80 = t44*2.0, t81 = t45*2.0, t82 = t46*2.0;
    T t85 = t34*tx, t87 = t54*tz, t89 = t55*tz, t91 = t59*ty, t93 = t36*tx, t94 = t37*ty, t98 = t71*ty, t99 = t69*tz, t100 = t74*tx, t101 = t70*tz, t102 = t75*tx, t103 = t72*ty, t107 = t32*tz*2.0, t108 = t27*ty*tz, t109 = t29*ty*tz;
    T t110 = t21*t26, t111 = t18*t27, t112 = t56*4.0, t113 = t57*4.0, t114 = t24*t28, t115 = t18*t29, t116 = t60*4.0, t117 = t62*4.0, t118 = t24*t30, t119 = t21*t31, t120 = t65*4.0, t121 = t66*4.0, t123 = t68*4.0, t125 = t73*4.0, t127 = t76*4.0;
    T t139 = t14*t53*2.0, t141 = ry*t53*2.0, t142 = t14*t27*tx, t143 = t15*t26*ty, t144 = rz*t53*2.0, t145 = rx*t61*2.0, t146 = t14*t29*tx, t147 = t28*t74, t148 = t26*t75, t149 = t30*t71, t150 = t27*t72, t151 = t31*t69, t152 = t29*t70;
    T t153 = t15*t61*2.0, t155 = t16*t28*tz, t156 = rz*t61*2.0, t157 = rx*t67*2.0, t158 = t15*t31*ty, t159 = t16*t30*tz, t160 = ry*t67*2.0, t161 = t16*t67*2.0, t163 = t83*2.0, t164 = t32*tx*2.0, t165 = t83*3.0, t166 = t32*tx*3.0, t167 = t84*2.0;
    T t168 = t33*tx*2.0, t171 = t84*3.0, t172 = t33*tx*3.0, t175 = t86*4.0, t177 = t88*4.0, t179 = t90*4.0, t181 = t92*2.0, t182 = t35*ty*2.0, t185 = t92*3.0, t186 = t35*ty*3.0, t193 = t95*2.0, t194 = t95*3.0, t195 = t96*2.0, t196 = t97*2.0;
    T t197 = t96*3.0, t198 = t97*3.0, t199 = t104*2.0, t200 = t105*2.0, t201 = t104*3.0, t202 = t105*3.0, t203 = t106*2.0, t204 = t106*3.0, t228 = rx*t56*8.0, t229 = t14*t55*8.0, t230 = rx*t60*8.0, t231 = t15*t54*8.0, t232 = ry*t57*8.0;
    T t233 = t14*t59*8.0, t235 = ry*t60*8.0, t236 = rz*t56*8.0, t237 = rx*t65*8.0, t239 = rz*t57*8.0, t240 = rx*t66*8.0, t241 = ry*t62*8.0, t242 = t16*t58*8.0, t243 = ry*t65*8.0, t244 = t15*t64*8.0, t245 = rz*t62*8.0, t246 = t16*t63*8.0;
    T t247 = rz*t66*8.0, t280 = t2*t69*4.0, t281 = t14*t32*4.0, t282 = t2*t71*4.0, t283 = t14*t33*4.0, t284 = t15*t32*6.0, t285 = t2*t40*6.0, t286 = t4*t69*6.0, t287 = t14*t34*6.0, t288 = t2*t74*4.0, t289 = ry*t33*tx*4.0, t290 = t2*t75*4.0;
    T t291 = rz*t32*tx*4.0, t292 = t4*t71*4.0, t293 = t4*t39*4.0, t294 = t15*t34*4.0, t295 = t4*t40*4.0, t296 = t7*t69*4.0, t297 = t7*t38*4.0, t298 = t2*t74*6.0, t299 = ry*t33*tx*6.0, t300 = t2*t75*6.0, t301 = rz*t32*tx*6.0, t302 = t4*t71*6.0;
    T t303 = t4*t39*6.0, t304 = t7*t69*6.0, t305 = t7*t38*6.0, t306 = t3*t74*4.0, t307 = t3*t44*4.0, t308 = t6*t71*4.0, t309 = rx*t35*ty*4.0, t310 = t4*t75*4.0, t313 = t7*t40*4.0, t314 = t3*t74*6.0, t315 = t3*t44*6.0, t316 = t16*t33*6.0;
    T t317 = t2*t45*6.0, t318 = t6*t71*6.0, t319 = rx*t35*ty*6.0, t320 = t4*t75*6.0, t322 = t7*t71*6.0, t323 = t14*t36*6.0, t325 = t7*t40*6.0, t327 = t3*t46*4.0, t328 = t6*t74*4.0, t329 = t15*t35*4.0, t331 = t4*t45*4.0, t332 = t9*t71*4.0;
    T t334 = t7*t74*4.0, t337 = t3*t46*6.0, t339 = t4*t45*6.0, t340 = t9*t71*6.0, t342 = t7*t74*6.0, t344 = t16*t36*4.0, t345 = t7*t45*4.0, t346 = t16*t35*6.0, t347 = t6*t46*6.0, t348 = t9*t74*6.0, t349 = t15*t37*6.0, t350 = t16*t37*4.0;
    T t351 = t9*t46*4.0, t352 = t23*t32*4.0, t353 = t20*t33*4.0, t354 = t23*t34*4.0, t356 = t20*t36*4.0, t358 = rx*t32*tz*8.0, t359 = rx*t32*tz*1.6E+1, t360 = t15*t33*8.0, t361 = rx*t34*tz*8.0, t362 = t15*t33*1.2E+1, t363 = rx*t34*tz*1.2E+1;
    T t364 = t16*t32*8.0, t365 = t4*t44*8.0, t366 = rx*t36*ty*8.0, t367 = t16*t32*1.2E+1, t368 = rx*t36*ty*1.2E+1, t369 = t4*t44*1.6E+1, t370 = t16*t34*8.0, t371 = t15*t36*8.0, t372 = t16*t34*1.2E+1, t373 = t15*t36*1.2E+1, t374 = t7*t46*8.0;
    T t375 = t7*t46*1.6E+1, t388 = t83*tx*6.0, t389 = t84*tx*6.0, t390 = t20*t55*6.0, t391 = t92*ty*6.0, t392 = t23*t59*6.0, t393 = t23*t64*6.0, t394 = t23*t69*2.0, t395 = t23*t70*2.0, t396 = t20*t72*2.0, t397 = t2*t69*1.2E+1;
    T t398 = t14*t32*1.2E+1, t399 = t2*t71*1.2E+1, t400 = t14*t33*1.2E+1, t401 = t15*t32*1.2E+1, t402 = t2*t40*1.2E+1, t403 = t4*t69*1.2E+1, t404 = t14*t34*1.2E+1, t409 = t3*t71*8.0, t411 = t15*t34*1.2E+1, t412 = t4*t40*1.2E+1;
    T t413 = t4*t71*2.4E+1, t414 = t4*t39*2.4E+1, t415 = t7*t69*2.4E+1, t416 = t7*t38*2.4E+1, t421 = t4*t74*8.0, t423 = t16*t33*1.2E+1, t424 = t2*t45*1.2E+1, t425 = t7*t71*1.2E+1, t426 = t14*t36*1.2E+1, t427 = t3*t74*2.4E+1, t428 = t3*t44*2.4E+1;
    T t430 = t7*t40*2.4E+1, t435 = t7*t75*8.0, t437 = t6*t74*1.2E+1, t438 = t15*t35*1.2E+1, t440 = t3*t46*2.4E+1, t442 = t4*t45*2.4E+1, t443 = t16*t35*1.2E+1, t444 = t6*t46*1.2E+1, t445 = t9*t74*1.2E+1, t446 = t15*t37*1.2E+1;
    T t447 = t16*t36*1.2E+1, t448 = t7*t45*1.2E+1, t449 = t16*t37*1.2E+1, t450 = t9*t46*1.2E+1, t466 = rx*t136, t467 = rx*t137, t468 = ry*t136, t469 = ry*t138, t470 = rz*t137, t471 = rz*t138, t475 = t32*t39*4.0, t476 = t15*t84*4.0;
    T t477 = t34*t71*4.0, t478 = t16*t83*4.0, t479 = t34*t44*4.0, t480 = t36*t69*4.0, t481 = t34*t45*4.0, t482 = t36*t40*4.0, t483 = t36*t46*4.0, t485 = t223*tx*tz, t486 = t224*tx*ty, t487 = t223*ty*tz, t488 = t225*tx*ty, t489 = t224*ty*tz;
    T t490 = t225*tx*tz, t500 = t23*t54*1.2E+1, t501 = t23*t55*1.2E+1, t502 = t20*t59*1.2E+1, t511 = -t410, t516 = -t422, t520 = -t429, t522 = -t436, t524 = -t439, t525 = -t441, t530 = t23+t136, t534 = t14*t83*6.0, t535 = t14*t84*6.0;
    T t536 = t32*t40*6.0, t537 = t4*t95*6.0, t538 = t34*t40*6.0, t539 = t33*t45*6.0, t540 = t7*t96*6.0, t541 = t15*t92*6.0, t542 = t35*t46*6.0, t543 = t9*t104*6.0, t544 = t36*t45*6.0, t545 = t37*t46*6.0, t552 = t41+t134, t553 = t41+t135;
    T t554 = t42+t135, t567 = t136*t223, t568 = t137*t224, t569 = t138*t225, t174 = t85*3.0, t176 = t87*4.0, t178 = t89*4.0, t180 = t91*4.0, t188 = t93*3.0, t192 = t94*3.0, t207 = -t112, t208 = -t113, t211 = -t116, t212 = -t117, t215 = -t120;
    T t216 = -t121, t226 = -t139, t234 = -t153, t248 = -t161, t250 = -t163, t251 = -t164, t252 = -t167, t253 = -t168, t255 = t85*-2.0, t256 = -t175, t258 = -t177, t260 = -t179, t262 = -t181, t263 = -t182, t265 = t93*-2.0, t267 = t94*-2.0;
    T t271 = -t98, t272 = -t99, t273 = -t100, t274 = -t101, t275 = -t102, t276 = -t103, t311 = rz*t85*4.0, t321 = rz*t85*6.0, t333 = rx*t94*4.0, t335 = ry*t93*4.0, t341 = rx*t94*6.0, t343 = ry*t93*6.0, t355 = t85*tz*4.0, t357 = t93*ty*4.0;
    T t376 = -t228, t377 = -t230, t378 = -t232, t379 = -t235, t380 = -t236, t381 = -t237, t382 = -t239, t383 = -t240, t384 = -t241, t385 = -t243, t386 = -t245, t387 = -t247, t405 = -t288, t406 = -t289, t407 = -t290, t408 = -t291, t417 = -t308;
    T t418 = -t309, t419 = -t310, t431 = -t332, t433 = -t334, t451 = -t359, t452 = -t360, t453 = -t361, t454 = -t362, t455 = -t363, t456 = -t364, t457 = -t366, t458 = -t367, t459 = -t368, t460 = -t369, t461 = -t370, t462 = -t371, t463 = -t372;
    T t464 = -t373, t465 = -t375, t472 = t3+t48, t473 = t5+t50, t474 = t8+t52, t491 = t467*ty, t492 = t466*tz, t493 = t469*tx, t494 = t468*tz, t495 = t471*tx, t496 = t470*ty, t497 = -t388, t498 = -t389, t499 = -t390, t503 = -t391, t504 = -t392;
    T t505 = -t393, t506 = -t398, t507 = -t400, t508 = -t401, t509 = -t404, t510 = -t409, t512 = -t411, t513 = -t414, t514 = -t416, t515 = -t421, t517 = -t423, t518 = -t426, t519 = -t428, t521 = -t435, t523 = -t438, t526 = -t443, t527 = -t446;
    T t528 = -t447, t529 = -t449, t546 = -t475, t547 = -t479, t548 = -t483, t549 = -t500, t550 = -t501, t551 = -t502, t555 = t552*tx, t556 = t553*tx, t557 = t552*ty, t558 = t554*ty, t559 = t553*tz, t560 = t554*tz, t561 = t40+t467, t562 = t45+t466;
    T t563 = t38+t469, t564 = t46+t468, t565 = t39+t471, t566 = t44+t470, t588 = t484*t530*tx*ty*2.0, t589 = t484*t530*tx*tz*2.0, t590 = t484*t530*ty*tz*2.0, t594 = t69+t71+t79+t81+t122, t595 = t70+t74+t77+t82+t124, t596 = t72+t75+t78+t80+t126;
    T t612 = t68+t73+t95+t97+t127+t197+t200+t201+t203, t613 = t68+t76+t96+t105+t125+t194+t196+t199+t204, t614 = t73+t76+t104+t106+t123+t193+t195+t198+t202, t257 = -t176, t259 = -t178, t261 = -t180, t420 = -t311, t432 = -t333, t434 = -t335;
    T t570 = -t557, t571 = -t559, t572 = -t560, t573 = t561*tx, t574 = t562*tx, t575 = t563*tx, t576 = t561*ty, t577 = t563*ty, t578 = t565*tx, t579 = t562*tz, t580 = t564*ty, t581 = t566*ty, t582 = t564*tz, t583 = t565*tz, t584 = t566*tz;
    T t591 = -t588, t592 = -t589, t593 = -t590, t597 = t594*tdx*ty*tz*2.0, t598 = t595*tdy*tx*tz*2.0, t599 = t596*tdz*tx*ty*2.0, t603 = t56+t86+t109+t166+t208+t254+t258, t604 = t57+t88+t109+t173+t207+t251+t256;
    T t605 = t60+t87+t108+t172+t212+t260+t264, t609 = t53+t61+t83+t85+t171+t185+t265+t267, t610 = t53+t67+t84+t93+t165+t192+t255+t262, t611 = t61+t67+t92+t94+t174+t188+t250+t252, t624 = t613*tdy*tx*2.0, t625 = t612*tdz*tx*2.0;
    T t626 = t614*tdx*ty*2.0, t627 = t612*tdz*ty*2.0, t628 = t614*tdx*tz*2.0, t629 = t613*tdy*tz*2.0, t636 = t150+t152+t293+t297+t320+t324+t338+t342+t365+t374+t379+t380+t406+t408+t451+t519+t524;
    T t637 = t148+t151+t301+t305+t307+t312+t336+t340+t358+t374+t381+t382+t418+t419+t460+t513+t525, t638 = t147+t149+t299+t303+t315+t319+t326+t330+t358+t365+t383+t384+t431+t433+t465+t514+t520;
    T t642 = t155+t157+t227+t231+t282+t306+t317+t323+t337+t341+t344+t371+t378+t397+t402+t417+t461+t509+t512+t515, t644 = t159+t160+t229+t238+t292+t328+t339+t343+t347+t349+t350+t366+t376+t403+t405+t412+t456+t506+t508+t510;
    T t646 = t156+t158+t233+t249+t296+t321+t325+t329+t346+t348+t351+t361+t377+t407+t425+t448+t452+t507+t511+t517, t600 = -t597, t601 = -t598, t602 = -t599, t606 = t62+t90+t108+t187+t211+t253+t257, t607 = t65+t89+t107+t186+t216+t261+t266;
    T t608 = t66+t91+t107+t191+t215+t259+t263, t615 = t276+t487+t496+t555+t582, t616 = t272+t490+t492+t558+t578, t617 = t274+t489+t494+t556+t581, t618 = t273+t486+t493+t571+t576, t619 = t275+t485+t495+t570+t579, t620 = t271+t488+t491+t572+t575;
    T t621 = t105+t106+t567+t574+t580, t622 = t97+t104+t568+t573+t584, t623 = t95+t96+t569+t577+t583, t630 = -t624, t631 = -t625, t632 = -t626, t633 = -t627, t634 = -t628, t635 = -t629;
    T t643 = t143+t145+t227+t242+t280+t285+t287+t294+t314+t318+t327+t370+t386+t399+t424+t432+t462+t518+t521+t528, t645 = t141+t142+t238+t246+t281+t284+t286+t295+t298+t302+t331+t364+t387+t434+t437+t444+t457+t522+t527+t529;
    T t647 = t144+t146+t244+t249+t283+t300+t304+t313+t316+t322+t345+t360+t385+t420+t445+t450+t453+t516+t523+t526, t639 = t600+t633+t635, t640 = t601+t631+t634, t641 = t602+t630+t632;
    
    fdrv(0, 0) = t138*t623+t618*tx*ty+t619*tx*tz;
    fdrv(0, 1) = -t138*t620-t622*tx*ty+t615*tx*tz; fdrv(0, 2) = -t138*t616+t617*tx*ty-t621*tx*tz;
    fdrv(0, 3) = t234+t248+t476-t477+t478-t480+t534+t535-t538-t544+rx*t110+rx*t114-t36*t40*6.0-t16*t61*2.0-t34*t45*6.0-t6*t76*2.0-t37*t46*2.0-t14*t85*8.0-t14*t93*8.0-t15*t92*2.0+t54*t71*4.0+t2*t125+t2*t127+t15*t267-rx*t4*t21*4.0-rx*t7*t24*4.0-ry*t9*t24*2.0;
    fdrv(0, 4) = t536-t537+t548+ry*t114-t14*t57*2.0+t15*t56*1.0E+1-t15*t57*4.0+t16*t56*8.0-t16*t57*2.0-t36*t38*4.0-t4*t73*8.0+t32*t45*4.0+t3*t76*8.0-t4*t76*2.0+t15*t86*1.2E+1+t2*t100*2.0-t14*t88*2.0-t15*t88*4.0+t32*t71*8.0-t36*t70*2.0-t4*t106*6.0+t281*tx+rx*t2*t21*8.0-ry*t7*t24*2.0-t15*t36*ty*6.0;
    fdrv(0, 5) = t539-t540+t547+rz*t110+t33*t40*4.0-t34*t39*4.0-t14*t62*2.0+t16*t60*1.0E+1-t15*t62*2.0-t16*t62*4.0-t7*t73*2.0-t7*t76*8.0+t44*t54*8.0+t33*t69*8.0+t2*t102*2.0-t14*t90*2.0-t16*t90*4.0-t34*t72*2.0-t7*t104*6.0+t283*tx+rx*t2*t24*8.0+ry*t3*t24*8.0-rz*t4*t21*2.0+t3*t46*tz*1.2E+1-t16*t34*tz*6.0;
    fdrv(0, 6) = -t644*tdy-t646*tdz-tdx*(-t306+t308-t327+t333+t372+t373-t397-t399+t411+t447+rx*t61*4.0+rx*t67*4.0+t14*t34*2.4E+1+t14*t36*2.4E+1-t15*t54*4.0-t16*t58*4.0);
    fdrv(0, 7) = -t644*tdx-t636*tdz+tdy*(t140+t242-t344+t402+t427+t440+t463+t464+t509-ry*t57*4.0+t2*t45*4.0-t15*t34*2.4E+1-t14*t36*4.0+t15*t54*4.0E+1+t2*t69*2.4E+1+t2*t71*8.0-t4*t74*4.0);
    fdrv(0, 8) = -t646*tdx-t636*tdy+tdz*(t140+t231-t294+t424+t427+t440+t463+t464+t518-rz*t62*4.0+t2*t40*4.0-t14*t34*4.0-t16*t36*2.4E+1+t2*t69*8.0+t2*t71*2.4E+1+t16*t58*4.0E+1-t7*t75*4.0);
    fdrv(0, 9) = -tdx*(t205+t209+t354+t356+t497+t498+t4*t21*4.0+t7*t24*4.0+t17*t34*8.0+t17*t36*8.0-t23*t54*4.0)-t604*tdy*tx*2.0-t606*tdz*tx*2.0;
    fdrv(0, 10) = -tdy*(t209+t354+t497+t549-t3*t21*1.0E+1+t4*t21*8.0+t17*t34*4.0+t24*t29+t20*t36*6.0-t84*tx*2.0+t17*t29*tz)-t611*tdx*ty*2.0-t606*tdz*ty*2.0;
    fdrv(0, 11) = -tdz*(t205+t356+t498+t549-t5*t24*1.0E+1+t7*t24*8.0+t21*t27+t17*t36*4.0+t23*t34*6.0-t83*tx*2.0+t17*t27*ty)-t611*tdx*tz*2.0-t604*tdy*tz*2.0;
    fdrv(0, 12) = tdy*(t218+t268+t279+t395+ry*t25*2.0+t17*t38*4.0+t23*t38*8.0+t17*t46*4.0+t23*t46*8.0+t97*ty*6.0+t104*ty*1.2E+1)+tdz*(t219+t269+t277+t396+rz*t22*2.0+t17*t39*4.0+t20*t39*8.0+t17*t44*4.0+t20*t44*8.0+t105*tz*6.0+t106*tz*1.2E+1)+t138*t594*tdx*2.0;
    fdrv(0, 13) = t641; fdrv(0, 14) = t640; fdrv(0, 15) = t530*tx*(t34+t36-t54-t58)*-2.0; fdrv(0, 16) = t530*ty*(t34+t36-t54-t58)*-2.0; fdrv(0, 17) = t530*tz*(t34+t36-t54-t58)*-2.0;
    fdrv(0, 18) = rdx*t530*(t34+t36-t54-t58)*-2.0-t11*tx*(t34+t36-t54-t58)*4.0-t12*tx*(t34+t36-t54-t58)*4.0-t13*tx*(t34+t36-t54-t58)*4.0-t11*t222*t530*2.0-t12*t222*t530*2.0-t13*t222*t530*2.0;
    fdrv(0, 19) = rdy*t530*(t34+t36-t54-t58)*-2.0-t11*ty*(t34+t36-t54-t58)*4.0-t12*ty*(t34+t36-t54-t58)*4.0-t13*ty*(t34+t36-t54-t58)*4.0-t11*t530*(t4-t26)*2.0-t12*t530*(t4-t26)*2.0-t13*t530*(t4-t26)*2.0;
    fdrv(0, 20) = rdz*t530*(t34+t36-t54-t58)*-2.0-t11*tz*(t34+t36-t54-t58)*4.0-t12*tz*(t34+t36-t54-t58)*4.0-t13*tz*(t34+t36-t54-t58)*4.0-t11*t530*(t7-t28)*2.0-t12*t530*(t7-t28)*2.0-t13*t530*(t7-t28)*2.0; fdrv(0, 24) = t138*t484*t530*2.0;
    fdrv(0, 25) = t591; fdrv(0, 26) = t592; fdrv(1, 0) = -t137*t618-t623*tx*ty+t619*ty*tz; fdrv(1, 1) = t137*t622+t620*tx*ty+t615*ty*tz; fdrv(1, 2) = -t137*t617+t616*tx*ty-t621*ty*tz;
    fdrv(1, 3) = -t536+t537+t548+rx*t118+t14*t57*1.0E+1-t15*t56*2.0+t15*t57*8.0-t16*t56*2.0+t16*t57*8.0-t36*t38*6.0-t32*t45*6.0-t3*t76*2.0+t4*t76*8.0-t15*t86*2.0+t14*t88*1.2E+1+t15*t88*8.0-t32*t71*4.0+t6*t98*2.0-t37*t69*2.0+t4*t106*4.0+t4*t125-rx*t2*t21*4.0-rx*t9*t24*2.0-t14*t32*tx*8.0-t15*t36*ty*4.0;
    fdrv(1, 4) = t226+t248-t476+t477+t481-t482-t534+t538+t541-t545+ry*t111+ry*t118-t15*t53*4.0-t16*t53*2.0-t2*t73*8.0-t2*t76*2.0-t36*t45*2.0-t14*t84*2.0+t14*t85*4.0-t16*t83*6.0-t36*t69*6.0-t15*t94*8.0+t55*t74*4.0+t6*t127+t14*t265-rx*t7*t24*2.0-ry*t9*t24*4.0;
    fdrv(1, 5) = t542-t543+t546+rz*t111-t32*t44*4.0-t15*t66*2.0+t16*t65*1.0E+1-t16*t66*4.0-t9*t76*8.0+t39*t55*8.0-t38*t59*2.0+t4*t99*4.0-t32*t72*2.0-t7*t98*6.0+t27*t102+t329*ty+rx*t4*t24*8.0-rx*t7*t21*2.0+ry*t6*t24*8.0-rz*t2*t21*2.0-t7*t40*ty*2.0-t7*t45*ty*4.0+t4*t40*tz*8.0-t16*t32*tz*6.0+t4*t45*tz*1.2E+1;
    fdrv(1, 6) = -t642*tdy-t637*tdz+tdx*(t154+t246-t350+t403+t413+t442+t458+t459+t508-rx*t56*4.0+t4*t40*2.4E+1-t14*t32*2.4E+1+t6*t46*4.0-t15*t37*4.0+t14*t55*4.0E+1-t3*t71*4.0+t6*t74*8.0);
    fdrv(1, 7) = -t642*tdx-t647*tdz-tdy*(t288-t292-t331+t335+t367+t368+t398-t412-t437+t449+ry*t53*4.0+ry*t67*4.0+t15*t32*2.4E+1+t15*t37*2.4E+1-t14*t55*4.0-t16*t63*4.0);
    fdrv(1, 8) = -t637*tdx-t647*tdy+tdz*(t154+t229-t281+t413+t442+t444+t458+t459+t527-rz*t66*4.0+t4*t40*8.0-t15*t32*4.0-t7*t43*4.0-t16*t37*2.4E+1+t4*t69*4.0+t16*t63*4.0E+1+t6*t74*2.4E+1);
    fdrv(1, 9) = -tdx*(t213+t352+t499+t550-t4*t18*1.0E+1+t2*t21*4.0+t17*t32*8.0+t24*t31-t92*ty*2.0+t93*ty*6.0+t20*t31*tz)-t610*tdy*tx*2.0-t608*tdz*tx*2.0;
    fdrv(1, 10) = -tdy*(t206+t213+t352+t357+t499+t503+t2*t21*8.0+t9*t24*4.0+t17*t32*4.0+t20*t37*8.0-t23*t55*4.0)-t603*tdx*ty*2.0-t608*tdz*ty*2.0;
    fdrv(1, 11) = -tdz*(t206+t357+t503+t550+t2*t21*2.0-t8*t24*1.0E+1+t9*t24*8.0+t17*t32*2.0+t23*t32*6.0+t20*t37*4.0-t20*t55*2.0)-t603*tdx*tz*2.0-t610*tdy*tz*2.0; fdrv(1, 12) = t641;
    fdrv(1, 13) = tdx*(t217+t270+t278+t394+rx*t25*2.0+t20*t40*4.0+t23*t40*8.0+t20*t45*4.0+t23*t45*8.0+t95*tx*6.0+t96*tx*1.2E+1)+tdz*(t219+t269+t277+t396+rz*t19*2.0+t17*t39*8.0+t20*t39*4.0+t17*t44*8.0+t20*t44*4.0+t105*tz*1.2E+1+t106*tz*6.0)+t137*t595*tdy*2.0;
    fdrv(1, 14) = t639; fdrv(1, 15) = t530*tx*(t32+t37-t63+t47*tx)*-2.0; fdrv(1, 16) = t530*ty*(t32+t37-t63+t47*tx)*-2.0; fdrv(1, 17) = t530*tz*(t32+t37-t63+t47*tx)*-2.0;
    fdrv(1, 18) = rdx*t530*(t32+t37-t63+t47*tx)*-2.0-t11*tx*(t32+t37-t63+t47*tx)*4.0-t12*tx*(t32+t37-t63+t47*tx)*4.0-t13*tx*(t32+t37-t63+t47*tx)*4.0-t11*t472*t530*2.0-t12*t472*t530*2.0-t13*t472*t530*2.0;
    fdrv(1, 19) = rdy*t530*(t32+t37-t63+t47*tx)*-2.0-t11*ty*(t32+t37-t63+t47*tx)*4.0-t12*ty*(t32+t37-t63+t47*tx)*4.0-t13*ty*(t32+t37-t63+t47*tx)*4.0-t11*t221*t530*2.0-t12*t221*t530*2.0-t13*t221*t530*2.0;
    fdrv(1, 20) = rdz*t530*(t32+t37-t63+t47*tx)*-2.0-t11*tz*(t32+t37-t63+t47*tx)*4.0-t12*tz*(t32+t37-t63+t47*tx)*4.0-t13*tz*(t32+t37-t63+t47*tx)*4.0-t11*t530*(t9-t30)*2.0-t12*t530*(t9-t30)*2.0-t13*t530*(t9-t30)*2.0; fdrv(1, 24) = t591;
    fdrv(1, 25) = t137*t484*t530*2.0; fdrv(1, 26) = t593; fdrv(2, 0) = -t136*t619-t623*tx*tz+t618*ty*tz; fdrv(2, 1) = -t136*t615+t620*tx*tz-t622*ty*tz; fdrv(2, 2) = t136*t621+t616*tx*tz+t617*ty*tz;
    fdrv(2, 3) = -t539+t540+t547+rx*t119-t33*t40*6.0-t34*t39*6.0+t14*t62*1.0E+1-t16*t60*2.0+t15*t62*8.0+t16*t62*8.0+t7*t73*8.0-t44*t54*2.0-t33*t69*4.0+t14*t90*1.2E+1-t35*t69*2.0+t16*t90*8.0+t7*t104*4.0+t31*t98+t7*t127-rx*t2*t24*4.0-rx*t6*t24*2.0-ry*t3*t24*2.0-t14*t33*tx*8.0-t3*t46*tz*2.0-t16*t34*tz*4.0;
    fdrv(2, 4) = -t542+t543+t546+ry*t115-t32*t44*6.0+t15*t66*1.0E+1-t16*t65*2.0+t16*t66*8.0-t39*t55*2.0+t38*t59*8.0-t4*t99*6.0-t33*t70*2.0+t7*t98*4.0+t29*t100+t9*t127-rx*t4*t24*2.0+rx*t7*t21*8.0-ry*t2*t24*2.0-ry*t6*t24*4.0+t7*t40*ty*1.2E+1-t15*t35*ty*8.0+t7*t45*ty*8.0-t4*t40*tz*4.0-t16*t32*tz*4.0-t4*t45*tz*2.0;
    fdrv(2, 5) = t226+t234-t478+t480-t481+t482-t535-t541+t544+t545+rz*t115+rz*t119-t15*t53*2.0-t16*t53*4.0-t34*t40*2.0-t2*t73*2.0-t16*t61*4.0-t2*t76*8.0-t6*t76*8.0-t14*t83*2.0-t15*t84*6.0-t34*t71*6.0+t14*t93*4.0+t15*t94*4.0+t59*t75*4.0+t14*t255-rx*t4*t21*2.0;
    fdrv(2, 6) = -t638*tdy-t643*tdz+tdx*(t162+t244-t329+t415+t425+t430+t454+t455+t517-rx*t60*4.0-t3*t41*4.0-t14*t33*2.4E+1-t16*t35*4.0+t7*t45*2.4E+1+t9*t46*8.0+t14*t59*4.0E+1+t9*t74*4.0);
    fdrv(2, 7) = -t638*tdx-t645*tdz+tdy*(t162+t233-t283+t415+t430+t445+t454+t455+t526-ry*t65*4.0-t4*t42*4.0-t16*t33*4.0-t15*t35*2.4E+1+t7*t45*8.0+t9*t46*2.4E+1+t7*t71*4.0+t15*t64*4.0E+1);
    fdrv(2, 8) = -t643*tdx-t645*tdy-tdz*(t290-t296+t311-t313+t362+t363+t400+t438-t448-t450+rz*t53*4.0+rz*t61*4.0+t16*t33*2.4E+1+t16*t35*2.4E+1-t14*t59*4.0-t15*t64*4.0);
    fdrv(2, 9) = -tdx*(t214+t353+t504+t551-t7*t18*1.0E+1+t2*t24*4.0+t6*t24*2.0+t17*t33*8.0+t20*t35*2.0-t23*t64*2.0+t85*tz*6.0)-t607*tdy*tx*2.0-t609*tdz*tx*2.0;
    fdrv(2, 10) = -tdy*(t210+t355+t505+t551+t2*t24*2.0+t6*t24*4.0-t9*t21*1.0E+1+t17*t33*2.0+t20*t33*6.0+t20*t35*8.0-t23*t59*2.0)-t605*tdx*ty*2.0-t609*tdz*ty*2.0;
    fdrv(2, 11) = -tdz*(t210+t214+t353+t355+t504+t505+t2*t24*8.0+t6*t24*8.0+t17*t33*4.0+t20*t35*4.0-t20*t59*4.0)-t605*tdx*tz*2.0-t607*tdy*tz*2.0; fdrv(2, 12) = t640; fdrv(2, 13) = t639;
    fdrv(2, 14) = tdx*(t217+t270+t278+t394+rx*t22*2.0+t20*t40*8.0+t23*t40*4.0+t20*t45*8.0+t23*t45*4.0+t95*tx*1.2E+1+t96*tx*6.0)+tdy*(t218+t268+t279+t395+ry*t19*2.0+t17*t38*8.0+t23*t38*4.0+t17*t46*8.0+t23*t46*4.0+t97*ty*1.2E+1+t104*ty*6.0)+t136*t596*tdz*2.0;
    fdrv(2, 15) = t530*tx*(t33+t35+t49*tx+t51*ty)*-2.0; fdrv(2, 16) = t530*ty*(t33+t35+t49*tx+t51*ty)*-2.0; fdrv(2, 17) = t530*tz*(t33+t35+t49*tx+t51*ty)*-2.0;
    fdrv(2, 18) = rdx*t530*(t33+t35+t49*tx+t51*ty)*-2.0-t11*tx*(t33+t35+t49*tx+t51*ty)*4.0-t12*tx*(t33+t35+t49*tx+t51*ty)*4.0-t13*tx*(t33+t35+t49*tx+t51*ty)*4.0-t11*t473*t530*2.0-t12*t473*t530*2.0-t13*t473*t530*2.0;
    fdrv(2, 19) = rdy*t530*(t33+t35+t49*tx+t51*ty)*-2.0-t11*ty*(t33+t35+t49*tx+t51*ty)*4.0-t12*ty*(t33+t35+t49*tx+t51*ty)*4.0-t13*ty*(t33+t35+t49*tx+t51*ty)*4.0-t11*t474*t530*2.0-t12*t474*t530*2.0-t13*t474*t530*2.0;
    fdrv(2, 20) = rdz*t530*(t33+t35+t49*tx+t51*ty)*-2.0-t11*tz*(t33+t35+t49*tx+t51*ty)*4.0-t12*tz*(t33+t35+t49*tx+t51*ty)*4.0-t13*tz*(t33+t35+t49*tx+t51*ty)*4.0-t11*t220*t530*2.0-t12*t220*t530*2.0-t13*t220*t530*2.0; fdrv(2, 24) = t592;
    fdrv(2, 25) = t593; fdrv(2, 26) = t136*t484*t530*2.0;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f29(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = oy*ty, t4 = oz*tz, t5 = rdx*tx, t6 = rdy*ty, t7 = rdz*tz, t8 = rx*tx, t9 = ry*ty, t10 = rz*tz, t11 = tx*tx, t12 = ty*ty, t13 = tz*tz, t16 = rx*ty*2.0, t17 = ry*tx*2.0, t18 = rx*tz*2.0, t20 = rz*tx*2.0, t22 = ry*tz*2.0;
    T t23 = rz*ty*2.0, t29 = ox*ry*tz, t30 = ox*rz*ty, t31 = oy*rx*tz, t32 = oy*rz*tx, t33 = oz*rx*ty, t34 = oz*ry*tx, t41 = ox*ty*tz, t42 = oy*tx*tz, t43 = oz*tx*ty, t50 = rx*ty*tz, t51 = ry*tx*tz, t52 = rz*tx*ty, t119 = ox*rx*ty*-2.0;
    T t120 = ox*rx*tz*-2.0, t121 = oy*ry*tx*-2.0, t122 = oy*ry*tz*-2.0, t123 = oz*rz*tx*-2.0, t124 = oz*rz*ty*-2.0, t14 = t8*2.0, t15 = t8*6.0, t19 = t9*2.0, t21 = t9*6.0, t24 = t10*2.0, t25 = t10*6.0, t26 = ry*t2, t27 = rz*t2, t28 = rx*t3;
    T t35 = rz*t3, t36 = rx*t4, t37 = ry*t4, t38 = t2*ty, t39 = t2*tz, t40 = t3*tx, t44 = t3*tz, t45 = t4*tx, t46 = t4*ty, t47 = t8*ty, t48 = t8*tz, t49 = t9*tx, t53 = t9*tz, t54 = t10*tx, t55 = t10*ty, t56 = t50*tdx, t57 = t51*tdy, t58 = t52*tdz;
    T t59 = t11*3.0, t60 = t12*3.0, t61 = t13*3.0, t62 = t2*tx, t63 = ox*t12, t64 = oy*t11, t65 = ox*t13, t66 = t3*ty, t67 = oz*t11, t68 = oy*t13, t69 = oz*t12, t70 = t4*tz, t71 = t8*tx, t72 = t9*ty, t73 = t10*tz, t74 = rx*t2*2.0, t78 = ry*t3*2.0;
    T t82 = rz*t4*2.0, t86 = t41*2.0, t87 = t42*2.0, t88 = t43*2.0, t98 = t2*t8, t99 = t3*t9, t100 = t4*t10, t101 = rx*t12*tdx, t102 = ry*t11*tdy, t103 = rx*t13*tdx, t104 = rz*t11*tdz, t105 = ry*t13*tdy, t106 = rz*t12*tdz, t139 = -t51;
    T t140 = -t52, t141 = t11+t12, t142 = t11+t13, t143 = t12+t13, t150 = t2+t3, t151 = t2+t4, t152 = t3+t4, t168 = t16+t17, t169 = t18+t20, t170 = t22+t23, t171 = t5+t6+t7, t83 = t38*2.0, t84 = t39*2.0, t85 = t40*2.0, t89 = t44*2.0;
    T t90 = t45*2.0, t91 = t46*2.0, t92 = t14*ty, t93 = t14*tz, t94 = t19*tx, t95 = t19*tz, t96 = t24*tx, t97 = t24*ty, t125 = -t38, t127 = -t39, t128 = -t40, t132 = -t88, t133 = -t44, t134 = -t45, t137 = -t46, t144 = t71*tdx*3.0;
    T t145 = t72*tdy*3.0, t146 = t73*tdz*3.0, t147 = -t56, t148 = -t57, t149 = -t58, t153 = -t98, t154 = -t99, t155 = -t100, t156 = t8+t19, t157 = t9+t14, t158 = t8+t24, t159 = t10+t14, t160 = t9+t24, t161 = t10+t19, t162 = rx*t141;
    T t163 = rx*t142, t164 = ry*t141, t165 = ry*t143, t166 = rz*t142, t167 = rz*t143, t178 = oz*t168, t179 = oy*t169, t180 = ox*t170, t181 = t50+t139, t182 = t50+t140, t183 = t51+t140, t187 = t14+t19+t25, t188 = t14+t21+t24, t189 = t15+t19+t24;
    T t190 = t171*tx*ty, t191 = t171*tx*tz, t192 = t171*ty*tz, t107 = rdx*t83, t108 = rdx*t84, t109 = rdy*t85, t110 = rdy*t89, t111 = rdz*t90, t112 = rdz*t91, t113 = t92*tdy, t114 = t94*tdx, t115 = t93*tdz, t116 = t96*tdx, t117 = t95*tdz;
    T t118 = t97*tdy, t126 = -t83, t129 = -t84, t130 = -t85, t135 = -t89, t136 = -t90, t138 = -t91, t172 = t156*tx, t173 = t157*ty, t174 = t158*tx, t175 = t159*tz, t176 = t160*ty, t177 = t161*tz, t184 = t24+t156, t185 = t24+t157, t186 = t19+t159;
    T t193 = -t178, t194 = -t179, t195 = -t180, t196 = t49+t163, t197 = t54+t162, t198 = t47+t165, t199 = t55+t164, t200 = t48+t167, t201 = t53+t166, t202 = -t190, t203 = -t191, t204 = -t192, t205 = t54+t172, t206 = t49+t174, t207 = t55+t173;
    T t208 = t47+t176, t209 = t53+t175, t210 = t48+t177, t211 = t29+t31+t193, t212 = t30+t33+t194, t213 = t32+t34+t195;
    
    fdrv(0, 0) = t198*ty+t200*tz; fdrv(0, 1) = -t198*tx+t183*tz; fdrv(0, 2) = -t200*tx-t183*ty;
    fdrv(0, 3) = t154+t155+rx*t63+rx*t65-t3*t8*2.0-t4*t8*2.0-t3*t10-t4*t9; fdrv(0, 4) = -oy*t205-t37*tx+ox*(t97+t198+t156*ty); fdrv(0, 5) = -oz*t206-t35*tx+ox*(t95+t200+t158*tz);
    fdrv(0, 6) = -tdy*(t37+t119+oy*t186)-tdz*(t35+t120+oz*t185)-tdx*(t28*2.0+t36*2.0); fdrv(0, 7) = -tdx*(t37+t78+t119+oy*t10+oy*t14)-t213*tdz+tdy*(t121+ox*t188); fdrv(0, 8) = -tdx*(t35+t82+t120+oz*t9+oz*t14)-t213*tdy+tdz*(t123+ox*t187);
    fdrv(0, 9) = tdx*(t63+t65+t130+t136)-tdy*(t64+t126)-tdz*(t67+t129); fdrv(0, 10) = -tdx*(t46+t66)-tdy*(t45+t85-ox*(t13+t60))-tdz*(t43-t86); fdrv(0, 11) = -tdx*(t44+t70)-tdz*(t40+t90-ox*(t12+t61))-tdy*(t42-t86);
    fdrv(0, 12) = t101+t103+t105+t106+t113+t115+t117+t118+t145+t146; fdrv(0, 13) = t149-t205*tdy-tdx*(t55+t72+t92); fdrv(0, 14) = t148-t206*tdz-tdx*(t53+t73+t93); fdrv(0, 15) = -t3*t11-t4*t11+ox*(t12*tx+t13*tx); fdrv(0, 16) = -ty*(t40+t45-t63-t65);
    fdrv(0, 17) = -tz*(t40+t45-t63-t65); fdrv(0, 18) = -t6*t152-t7*t152-rdx*(t85+t90-ox*t143); fdrv(0, 19) = t107-rdy*t40*2.0+rdy*t65+rdy*t134-oy*t5*tx+ox*t6*ty*3.0+ox*t7*ty*2.0-oy*t7*tx;
    fdrv(0, 20) = t108-rdz*t45*2.0+rdz*t63+rdz*t128-oz*t5*tx+ox*t6*tz*2.0-oz*t6*tx+ox*t7*tz*3.0; fdrv(0, 24) = t143*t171; fdrv(0, 25) = t202; fdrv(0, 26) = t203; fdrv(1, 0) = -t196*ty+t182*tz; fdrv(1, 1) = t196*tx+t201*tz; fdrv(1, 2) = -t182*tx-t201*ty;
    fdrv(1, 3) = -ox*t207-t36*ty+oy*(t96+t196+t157*tx); fdrv(1, 4) = t153+t155+ry*t64+ry*t68-t2*t9*2.0-t2*t10-t4*t8-t4*t9*2.0; fdrv(1, 5) = -oz*t208-t27*ty+oy*(t93+t201+t160*tz);
    fdrv(1, 6) = -tdy*(t36+t74+t121+ox*t10+ox*t19)-t212*tdz+tdx*(t119+oy*t189); fdrv(1, 7) = -tdx*(t36+t121+ox*t186)-tdz*(t27+t122+oz*t184)-tdy*(t26*2.0+t37*2.0); fdrv(1, 8) = -tdy*(t27+t82+t122+oz*t8+oz*t19)-t212*tdx+tdz*(t124+oy*t187);
    fdrv(1, 9) = -tdy*(t45+t62)-tdx*(t46+t83-oy*(t13+t59))-tdz*(t43-t87); fdrv(1, 10) = tdy*(t64+t68+t126+t138)-tdx*(t63+t130)-tdz*(t69+t135); fdrv(1, 11) = -tdy*(t39+t70)-tdz*(t38+t91-oy*(t11+t61))-tdx*(t41-t87);
    fdrv(1, 12) = t149-t207*tdx-tdy*(t54+t71+t94); fdrv(1, 13) = t102+t103+t104+t105+t114+t115+t116+t117+t144+t146; fdrv(1, 14) = t147-t208*tdz-tdy*(t48+t73+t95); fdrv(1, 15) = -tx*(t38+t46-t64-t68); fdrv(1, 16) = -t2*t12-t4*t12+oy*(t11*ty+t13*ty);
    fdrv(1, 17) = -tz*(t38+t46-t64-t68); fdrv(1, 18) = t109-rdx*t38*2.0+rdx*t68+rdx*t137+oy*t5*tx*3.0-ox*t6*ty-ox*t7*ty+oy*t7*tx*2.0; fdrv(1, 19) = -t5*t151-t7*t151-rdy*(t83+t91-oy*t142);
    fdrv(1, 20) = t110-rdz*t46*2.0+rdz*t64+rdz*t125+oy*t5*tz*2.0-oz*t5*ty-oz*t6*ty+oy*t7*tz*3.0; fdrv(1, 24) = t202; fdrv(1, 25) = t142*t171; fdrv(1, 26) = t204; fdrv(2, 0) = t181*ty-t197*tz; fdrv(2, 1) = -t181*tx-t199*tz; fdrv(2, 2) = t197*tx+t199*ty;
    fdrv(2, 3) = -ox*t209-t28*tz+oz*(t94+t197+t159*tx); fdrv(2, 4) = -oy*t210-t26*tz+oz*(t92+t199+t161*ty); fdrv(2, 5) = t153+t154+rz*t67+rz*t69-t2*t9-t3*t8-t2*t10*2.0-t3*t10*2.0;
    fdrv(2, 6) = -tdz*(t28+t74+t123+ox*t9+ox*t24)-t211*tdy+tdx*(t120+oz*t189); fdrv(2, 7) = -tdz*(t26+t78+t124+oy*t8+oy*t24)-t211*tdx+tdy*(t122+oz*t188); fdrv(2, 8) = -tdx*(t28+t123+ox*t185)-tdy*(t26+t124+oy*t184)-tdz*(t27*2.0+t35*2.0);
    fdrv(2, 9) = -tdy*(t42+t132)-tdz*(t40+t62)-tdx*(t44+t84-oz*(t12+t59)); fdrv(2, 10) = -tdx*(t41+t132)-tdz*(t38+t66)-tdy*(t39+t89-oz*(t11+t60)); fdrv(2, 11) = tdz*(t67+t69+t129+t135)-tdx*(t65+t136)-tdy*(t68+t138);
    fdrv(2, 12) = t148-t209*tdx-tdz*(t49+t71+t96); fdrv(2, 13) = t147-t210*tdy-tdz*(t47+t72+t97); fdrv(2, 14) = t101+t102+t104+t106+t113+t114+t116+t118+t144+t145; fdrv(2, 15) = -tx*(t39+t44-t67-t69); fdrv(2, 16) = -ty*(t39+t44-t67-t69);
    fdrv(2, 17) = -t2*t13-t3*t13+oz*(t11*tz+t12*tz); fdrv(2, 18) = t111-rdx*t39*2.0+rdx*t69+rdx*t133+oz*t5*tx*3.0-ox*t6*tz+oz*t6*tx*2.0-ox*t7*tz; fdrv(2, 19) = t112-rdy*t44*2.0+rdy*t67+rdy*t127-oy*t5*tz+oz*t5*ty*2.0+oz*t6*ty*3.0-oy*t7*tz;
    fdrv(2, 20) = -t5*t150-t6*t150-rdz*(t84+t89-oz*t141); fdrv(2, 24) = t203; fdrv(2, 25) = t204; fdrv(2, 26) = t141*t171;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f30(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = ox*ty, t4 = oy*tx, t5 = ox*tz, t6 = oy*ty, t7 = oz*tx, t8 = oy*tz, t9 = oz*ty, t10 = oz*tz, t11 = rdx*tx, t12 = rdy*ty, t13 = rdz*tz, t14 = rx*tx, t15 = rx*ty, t16 = ry*tx, t17 = rx*tz, t18 = ry*ty, t19 = rz*tx;
    T t20 = ry*tz, t21 = rz*ty, t22 = rz*tz, t23 = ox*2.0, t24 = oy*2.0, t25 = oz*2.0, t26 = tx*tx, t27 = tx*tx*tx, t29 = ty*ty, t30 = ty*ty*ty, t32 = tz*tz, t33 = tz*tz*tz, t35 = t3*2.0, t36 = t4*2.0, t37 = t5*2.0, t38 = t7*2.0, t39 = t8*2.0;
    T t40 = t9*2.0, t41 = t14*2.0, t42 = t18*2.0, t43 = t22*2.0, t44 = t2*ty, t45 = t2*tz, t46 = t4*ty, t47 = t6*tz, t48 = t7*tz, t49 = t9*tz, t50 = t14*ty, t51 = t14*tz, t52 = t16*ty, t53 = t18*tz, t54 = t19*tz, t55 = t21*tz, t56 = t27*2.0;
    T t57 = t30*2.0, t58 = t33*2.0, t59 = -t4, t61 = -t5, t64 = -t9, t66 = t2*t26, t67 = t3*ty, t68 = t4*tx, t69 = t3*t29, t70 = t4*t26, t71 = t3*t30, t72 = t4*t27, t73 = t5*tz, t74 = t7*tx, t75 = t5*t32, t76 = t6*t29, t77 = t7*t26, t78 = t5*t33;
    T t79 = t7*t27, t80 = t8*tz, t81 = t9*ty, t82 = t8*t32, t83 = t9*t29, t84 = t8*t33, t85 = t9*t30, t86 = t10*t32, t87 = t14*t27, t88 = t18*t30, t89 = t22*t33, t90 = t30*tx, t91 = t27*ty, t92 = t33*tx, t93 = t27*tz, t94 = t33*ty, t95 = t30*tz;
    T t99 = t3*tz*4.0, t100 = t4*tz*4.0, t101 = t7*ty*4.0, t106 = t2*t16*tx, t109 = t2*t19*tx, t110 = t6*t15*ty, t120 = t6*t21*ty, t121 = t10*t17*tz, t124 = t10*t20*tz, t126 = t2*t29, t127 = t2*t32, t129 = t3*t32, t131 = t4*t32, t133 = t7*t29;
    T t135 = t6*t32, t138 = t14*t30, t140 = t14*t33, t142 = t16*t30, t144 = t15*t33, t145 = t15*t29*tz, t146 = t16*t33, t147 = t16*t26*tz, t148 = t19*t30, t149 = t19*t26*ty, t150 = t18*t33, t152 = t19*t33, t154 = t21*t33, t156 = t32*tx*ty;
    T t157 = t29*tx*tz, t158 = t26*ty*tz, t159 = t2*t15*4.0, t160 = t2*t17*4.0, t161 = t2*t18*4.0, t162 = t4*t15*4.0, t163 = t3*t17*4.0, t164 = t2*t20*4.0, t165 = t2*t21*4.0, t166 = t4*t18*4.0, t167 = t2*t22*4.0, t168 = t6*t17*4.0;
    T t169 = t4*t20*4.0, t170 = t4*t21*4.0, t171 = t7*t17*4.0, t172 = t6*t20*4.0, t173 = t9*t17*4.0, t174 = t7*t20*4.0, t175 = t7*t21*4.0, t176 = t6*t22*4.0, t177 = t9*t20*4.0, t178 = t7*t22*4.0, t179 = t9*t22*4.0, t186 = t2*tx*2.0;
    T t196 = t6*ty*2.0, t212 = t10*tz*2.0, t215 = t15*t29*2.0, t216 = t16*t26*2.0, t217 = t17*t32*2.0, t219 = t19*t26*2.0, t221 = t20*t32*2.0, t222 = t21*t29*2.0, t225 = t29*tx*2.0, t226 = t26*ty*2.0, t227 = t32*tx*2.0, t228 = t26*tz*2.0;
    T t229 = t32*ty*2.0, t230 = t29*tz*2.0, t245 = t26+t29, t246 = t26+t32, t247 = t29+t32, t249 = t2*t14*tx*4.0, t250 = t2*t14*6.0, t252 = t2*t16*2.0, t255 = t4*t14*4.0, t258 = t3*t15*6.0, t259 = t4*t14*6.0, t261 = t2*t19*2.0, t262 = t6*t15*2.0;
    T t268 = t3*t18*4.0, t269 = t7*t14*4.0, t274 = t5*t17*6.0, t275 = t3*t18*6.0, t276 = t4*t16*6.0, t277 = t7*t14*6.0, t282 = t6*t18*ty*4.0, t286 = t5*t20*6.0, t287 = t3*t21*6.0, t288 = t8*t17*6.0, t289 = t6*t18*6.0, t290 = t4*t19*6.0;
    T t291 = t9*t15*6.0, t292 = t7*t16*6.0, t294 = t6*t21*2.0, t295 = t10*t17*2.0, t301 = t5*t22*4.0, t302 = t9*t18*4.0, t307 = t5*t22*6.0, t308 = t8*t20*6.0, t309 = t9*t18*6.0, t310 = t7*t19*6.0, t311 = t10*t20*2.0, t315 = t8*t22*4.0;
    T t318 = t8*t22*6.0, t319 = t9*t21*6.0, t321 = t10*t22*tz*4.0, t322 = t10*t22*6.0, t333 = t4*t29*3.0, t347 = t7*t32*3.0, t351 = t9*t32*3.0, t355 = t14*t29*4.0, t359 = t14*t29*6.0, t365 = t14*t32*4.0, t367 = t16*t29*4.0, t373 = t14*t32*6.0;
    T t375 = t16*t29*6.0, t377 = t15*t32*2.0, t378 = t15*ty*tz*2.0, t379 = t16*t32*2.0, t380 = t16*tx*tz*2.0, t381 = t19*t29*2.0, t382 = t19*tx*ty*2.0, t387 = t18*t32*4.0, t389 = t19*t32*4.0, t395 = t18*t32*6.0, t397 = t19*t32*6.0;
    T t401 = t21*t32*4.0, t405 = t21*t32*6.0, t408 = t2*t15*1.2E+1, t410 = t2*t18*8.0, t411 = t4*t15*8.0, t412 = t2*t17*1.2E+1, t415 = t4*t17*8.0, t416 = t7*t15*8.0, t417 = t4*t17*1.2E+1, t418 = t4*t18*1.2E+1, t419 = t7*t15*1.2E+1;
    T t423 = t3*t20*8.0, t424 = t2*t22*8.0, t425 = t7*t17*8.0, t426 = t7*t18*8.0, t427 = t3*t20*1.2E+1, t428 = t7*t18*1.2E+1, t432 = t3*t22*8.0, t433 = t4*t22*8.0, t434 = t3*t22*1.2E+1, t435 = t6*t20*1.2E+1, t436 = t4*t22*1.2E+1;
    T t439 = t6*t22*8.0, t440 = t9*t20*8.0, t441 = t7*t22*1.2E+1, t442 = t9*t22*1.2E+1, t453 = t4*t15*tz*6.0, t483 = t14*t26*8.0, t484 = t18*t29*8.0, t485 = t22*t32*8.0, t486 = t2+t6, t487 = t2+t10, t488 = t6+t10, t489 = t14+t18, t490 = t14+t22;
    T t491 = t18+t22, t492 = t2*t14*tx*-2.0, t494 = t3*t15*-2.0, t508 = t6*t18*ty*-2.0, t519 = t8*t20*-2.0, t522 = t7*t19*-2.0, t532 = t10*t22*tz*-2.0, t537 = t4*t29*-2.0, t549 = t9*t32*-2.0, t558 = t14*t29*tx, t559 = t14*t32*tx;
    T t560 = t16*t29*tx, t561 = t15*t32*ty, t562 = t16*t32*tx, t563 = t19*t29*tx, t564 = t18*t32*ty, t565 = t19*t32*tx, t566 = t21*t32*ty, t601 = t4*t17*tz*3.0, t603 = t7*t15*ty*3.0, t621 = t3*t20*tz*3.0, t631 = t7*t18*ty*3.0;
    T t651 = t3*t22*tz*3.0, t653 = t4*t22*tz*3.0, t697 = t16*t29*tz*3.0, t702 = t19*t32*ty*3.0, t707 = t4*t15*tz*-4.0, t740 = t3*t17*tz*-4.0, t741 = t3*t15*tz*-4.0, t750 = t4*t20*tz*-2.0, t754 = t4*t20*tz*-4.0, t755 = t4*t16*tz*-4.0;
    T t767 = t7*t21*ty*-4.0, t768 = t7*t19*ty*-4.0, t781 = t16*t29*tz*-2.0, t782 = t19*t32*ty*-2.0, t804 = t11+t12+t13, t817 = t7*t18*ty*-1.2E+1, t818 = t3*t22*tz*-1.2E+1, t819 = t4*t22*tz*-1.2E+1, t60 = -t36, t62 = -t37, t63 = -t38, t65 = -t40;
    T t96 = t44*4.0, t97 = t45*4.0, t98 = t46*4.0, t102 = t47*4.0, t103 = t48*4.0, t104 = t49*4.0, t105 = t14*t66, t107 = t14*t68, t108 = t18*t67, t111 = t14*t74, t112 = t20*t73, t113 = t21*t67, t114 = t17*t80, t115 = t19*t68, t116 = t15*t81;
    T t117 = t16*t74, t118 = t18*t76, t119 = t22*t73, t122 = t18*t81, t123 = t22*t80, t125 = t22*t86, t128 = t46*tx, t130 = t67*tz, t132 = t68*tz, t134 = t74*ty, t136 = t48*tx, t137 = t49*ty, t139 = t26*t50, t141 = t26*t51, t143 = t26*t52;
    T t151 = t29*t53, t153 = t26*t54, t155 = t29*t55, t180 = t44*tz*2.0, t181 = t36*ty*tz, t182 = t38*ty*tz, t183 = t50*tz*4.0, t184 = t52*tz*4.0, t185 = t54*ty*4.0, t187 = t35*ty, t188 = t36*tx, t189 = t29*t35, t190 = t26*t36, t191 = t69*4.0;
    T t192 = t70*4.0, t193 = t67*6.0, t194 = t68*6.0, t195 = t37*tz, t197 = t38*tx, t199 = t26*t38, t200 = t75*4.0, t201 = t77*4.0, t202 = t73*6.0, t203 = t74*6.0, t204 = t39*tz, t205 = t40*ty, t206 = t32*t39, t207 = t29*t40, t208 = t82*4.0;
    T t209 = t83*4.0, t210 = t80*6.0, t211 = t81*6.0, t213 = t26*t41, t214 = t87*5.0, t218 = t29*t42, t220 = t88*5.0, t223 = t32*t43, t224 = t89*5.0, t231 = -t44, t235 = -t99, t236 = -t100, t237 = -t101, t240 = -t49, t242 = -t51, t243 = -t52;
    T t244 = -t55, t248 = t2*t41*tx, t256 = t15*t67*4.0, t260 = t17*t37, t263 = t16*t36, t270 = t17*t73*4.0, t272 = t16*t68*4.0, t278 = t6*t42*ty, t304 = t20*t80*4.0, t306 = t19*t74*4.0, t312 = t21*t40, t317 = t21*t81*4.0, t320 = t10*t43*tz;
    T t323 = t126*2.0, t324 = t44*tx*2.0, t325 = t126*3.0, t326 = t44*tx*3.0, t327 = t127*2.0, t328 = t45*tx*2.0, t329 = t29*t36, t331 = t127*3.0, t332 = t45*tx*3.0, t335 = t129*4.0, t337 = t131*4.0, t339 = t133*4.0, t341 = t135*2.0;
    T t342 = t47*ty*2.0, t343 = t32*t38, t345 = t135*3.0, t346 = t47*ty*3.0, t349 = t32*t40, t353 = t29*t41, t354 = t30*t41, t356 = t50*tx*4.0, t357 = t138*4.0, t360 = t50*tx*6.0, t361 = t32*t41, t362 = t52*tx*2.0, t363 = t33*t41;
    T t366 = t51*tx*4.0, t368 = t52*tx*4.0, t369 = t140*4.0, t371 = t142*4.0, t374 = t51*tx*6.0, t376 = t52*tx*6.0, t383 = t32*t42, t384 = t54*tx*2.0, t385 = t33*t42, t388 = t53*ty*4.0, t390 = t54*tx*4.0, t391 = t150*4.0, t393 = t152*4.0;
    T t396 = t53*ty*6.0, t398 = t54*tx*6.0, t399 = t55*ty*2.0, t402 = t55*ty*4.0, t403 = t154*4.0, t406 = t55*ty*6.0, t407 = -t159, t409 = -t161, t413 = -t163, t414 = -t164, t420 = -t169, t421 = -t170, t422 = -t171, t429 = -t172, t430 = -t173;
    T t431 = -t175, t437 = -t176, t438 = -t178, t444 = t46*tz*-2.0, t445 = t48*ty*-2.0, t451 = t15*t100, t452 = t18*t45*6.0, t455 = t4*t53*4.0, t457 = t21*t45*6.0, t458 = t15*t48*6.0, t459 = t4*t55*4.0, t461 = t4*t55*6.0, t462 = t18*t48*6.0;
    T t464 = -t67, t465 = t26*t59, t466 = -t71, t467 = t70*-2.0, t468 = t27*t59, t470 = t61*tz, t471 = -t74, t472 = t32*t61, t473 = t75*-2.0, t474 = t33*t61, t475 = -t79, t477 = t64*ty, t478 = t29*t64, t479 = -t84, t480 = t83*-2.0, t481 = t30*t64;
    T t493 = -t250, t495 = -t252, t496 = -t255, t498 = t17*t73*-2.0, t500 = t16*t68*-2.0, t504 = -t274, t505 = -t275, t506 = -t276, t507 = -t277, t515 = -t286, t516 = -t289, t517 = -t290, t518 = -t291, t520 = -t294, t521 = -t295, t523 = -t301;
    T t524 = -t302, t528 = t21*t81*-2.0, t530 = -t318, t531 = -t319, t533 = -t322, t540 = -t333, t542 = t32*t59, t551 = -t351, t552 = -t411, t553 = -t415, t554 = -t424, t555 = -t426, t556 = -t432, t557 = -t440, t567 = t15*t44*2.0, t568 = t41*t44;
    T t569 = t15*t44*6.0, t570 = t14*t44*6.0, t571 = t17*t45*2.0, t572 = t41*t45, t573 = t16*t44*2.0, t574 = t15*t36*ty, t575 = t18*t44*3.0, t576 = t16*t44*3.0, t577 = t15*t46*3.0, t578 = t14*t46*3.0, t579 = t17*t45*6.0, t580 = t14*t45*6.0;
    T t581 = t18*t44*6.0, t582 = t16*t44*6.0, t583 = t15*t46*6.0, t584 = t14*t46*6.0, t585 = t17*t35*tz, t587 = t20*t45*2.0, t588 = t16*t45*2.0, t589 = t21*t44*2.0, t590 = t19*t44*2.0, t591 = t17*t36*tz, t592 = t36*t51, t593 = t18*t36*ty;
    T t594 = t36*t52, t595 = t15*t38*ty, t596 = t38*t50, t597 = t20*t45*3.0, t598 = t16*t45*3.0, t599 = t21*t44*3.0, t600 = t19*t44*3.0, t602 = t4*t51*3.0, t604 = t7*t50*3.0, t607 = t18*t46*6.0, t608 = t16*t46*6.0, t609 = t20*t35*tz;
    T t610 = t35*t53, t611 = t19*t45*2.0, t612 = t17*t47*2.0, t613 = t15*t47*2.0, t617 = t19*t36*ty, t619 = t18*t38*ty, t620 = t38*t52, t622 = t3*t53*3.0, t623 = t22*t45*3.0, t624 = t19*t45*3.0, t625 = t17*t47*3.0, t626 = t15*t47*3.0;
    T t627 = t21*t46*3.0, t628 = t19*t46*3.0, t629 = t17*t48*3.0, t630 = t14*t48*3.0, t632 = t7*t52*3.0, t635 = t22*t45*6.0, t636 = t19*t45*6.0, t637 = t17*t48*6.0, t638 = t14*t48*6.0, t639 = t22*t35*tz, t640 = t35*t55, t641 = t20*t47*2.0;
    T t642 = t42*t47, t643 = t22*t36*tz, t644 = t36*t54, t646 = t15*t40*tz, t647 = t20*t38*tz, t648 = t16*t38*tz, t649 = t21*t38*ty, t652 = t3*t55*3.0, t654 = t4*t54*3.0, t655 = t17*t49*3.0, t656 = t15*t49*3.0, t657 = t20*t48*3.0;
    T t658 = t16*t48*3.0, t661 = t20*t47*6.0, t662 = t18*t47*6.0, t663 = t21*t47*2.0, t664 = t20*t40*tz, t665 = t22*t38*tz, t666 = t38*t54, t667 = t22*t47*3.0, t668 = t21*t47*3.0, t669 = t20*t49*3.0, t670 = t18*t49*3.0, t671 = t22*t47*6.0;
    T t672 = t21*t47*6.0, t673 = t20*t49*6.0, t674 = t18*t49*6.0, t675 = t22*t48*6.0, t676 = t19*t48*6.0, t677 = t22*t40*tz, t678 = t40*t55, t679 = t22*t49*6.0, t680 = t21*t49*6.0, t681 = t32*t44*2.0, t682 = t29*t45*2.0, t683 = t32*t36*ty;
    T t685 = t29*t38*tz, t689 = t41*tx*ty*tz, t690 = t32*t50*3.0, t691 = t29*t51*3.0, t692 = t50*tx*tz*3.0, t693 = t32*t52*2.0, t694 = t16*t230, t696 = t32*t52*3.0, t698 = t52*tx*tz*3.0, t699 = t19*t229, t700 = t29*t54*2.0, t703 = t29*t54*3.0;
    T t704 = t54*tx*ty*3.0, t705 = t15*t45*8.0, t706 = t18*t45*-4.0, t708 = t21*t45*-4.0, t709 = t15*t48*-4.0, t710 = t4*t53*8.0, t712 = t18*t48*-4.0, t713 = t21*t48*8.0, t716 = t29*t68*3.0, t721 = t32*t74*3.0, t722 = t32*t81*3.0, t723 = t558*3.0;
    T t724 = t559*3.0, t725 = t560*3.0, t726 = t564*3.0, t727 = t565*3.0, t728 = t566*3.0, t735 = t494*tz, t744 = t417*tz, t745 = t4*t51*1.2E+1, t746 = t419*ty, t747 = t7*t50*1.2E+1, t751 = t21*t46*-2.0, t752 = t19*t46*-2.0, t753 = t17*t48*-2.0;
    T t758 = t427*tz, t759 = t3*t53*1.2E+1, t761 = t7*t52*1.2E+1, t762 = t17*t49*-2.0, t763 = t15*t49*-2.0, t764 = t20*t48*-2.0, t765 = t16*t48*-2.0, t766 = t522*ty, t771 = t3*t55*1.2E+1, t773 = t4*t54*1.2E+1, t780 = t50*tx*tz*-2.0;
    T t786 = t489*tx, t787 = t489*ty, t788 = t490*tx, t789 = t490*tz, t790 = t491*ty, t791 = t491*tz, t792 = t41+t42, t793 = t41+t43, t794 = t42+t43, t796 = t42*t127, t797 = t15*t32*t36, t798 = t44*t55*2.0, t799 = t36*t53*ty, t805 = t126*tx*-3.0;
    T t806 = t127*tx*-3.0, t808 = t32*t67*-6.0, t809 = t32*t68*-6.0, t810 = t29*t74*-6.0, t811 = t135*ty*-3.0, t820 = t32+t245, t824 = t14*t333, t825 = t16*t333, t827 = t14*t347, t830 = t18*t351, t831 = t19*t347, t832 = t21*t351;
    T t845 = t25+t35+t59, t846 = t24+t38+t61, t847 = t23+t39+t64, t232 = -t96, t233 = -t97, t234 = -t98, t238 = -t102, t239 = -t103, t241 = -t104, t253 = t15*t187, t254 = t14*t188, t257 = t107*4.0, t271 = t108*4.0, t273 = t111*4.0;
    T t279 = t112*4.0, t280 = t113*4.0, t281 = t114*4.0, t283 = t115*4.0, t284 = t116*4.0, t285 = t117*4.0, t297 = t22*t195, t298 = t20*t204, t299 = t18*t205, t300 = t19*t197, t303 = t119*4.0, t305 = t122*4.0, t316 = t123*4.0, t334 = t128*3.0;
    T t336 = t130*4.0, t338 = t132*4.0, t340 = t134*4.0, t348 = t136*3.0, t352 = t137*3.0, t358 = t139*4.0, t364 = t143*2.0, t370 = t141*4.0, t372 = t143*4.0, t386 = t153*2.0, t392 = t151*4.0, t394 = t153*4.0, t400 = t155*2.0, t404 = t155*4.0;
    T t443 = -t180, t446 = -t183, t447 = -t184, t448 = -t185, t449 = t15*t97, t450 = t18*t97, t454 = t21*t97, t456 = t15*t103, t460 = t18*t103, t463 = t21*t103, t469 = -t192, t476 = -t200, t482 = -t209, t497 = -t256, t499 = t108*-2.0;
    T t501 = t111*-2.0, t502 = -t270, t503 = -t272, t525 = -t304, t526 = -t306, t527 = t123*-2.0, t529 = -t317, t534 = -t323, t535 = -t327, t536 = -t328, t538 = t128*-2.0, t539 = -t332, t541 = -t130, t543 = -t134, t545 = -t337, t547 = -t341;
    T t548 = t136*-2.0, t550 = t137*-2.0, t615 = t263*tz, t684 = t181*tx, t686 = t182*tx, t687 = t361*ty, t688 = t353*tz, t695 = t362*tz, t701 = t384*ty, t711 = -t459, t729 = -t570, t730 = -t573, t731 = -t579, t732 = -t580, t733 = -t581;
    T t734 = -t584, t736 = -t587, t737 = -t588, t738 = -t589, t739 = -t590, t742 = -t607, t743 = -t608, t748 = -t612, t749 = -t613, t756 = -t635, t757 = -t638, t769 = -t662, t774 = -t663, t775 = -t671, t776 = -t674, t777 = -t675, t778 = -t679;
    T t779 = -t680, t783 = -t705, t784 = -t710, t785 = -t713, t795 = t568*tz, t800 = t15*t182, t801 = t644*ty, t802 = t620*tz, t803 = t21*t343, t807 = -t716, t812 = -t721, t813 = -t722, t814 = -t745, t815 = -t747, t816 = -t759, t821 = t14*t325;
    T t822 = t14*t331, t823 = t16*t325, t826 = t19*t331, t828 = t18*t345, t829 = t21*t345, t833 = -t787, t834 = -t788, t835 = -t791, t836 = t792*tx*tz, t837 = t793*tx*ty, t838 = t792*ty*tz, t839 = t794*tx*ty, t840 = t793*ty*tz, t841 = t794*tx*tz;
    T t842 = t3+t25+t60, t843 = t7+t24+t62, t844 = t8+t23+t65, t848 = t245*t792, t849 = t246*t793, t850 = t247*t794, t851 = t15+t16+t53+t789, t852 = t17+t19+t50+t790, t853 = t20+t21+t54+t786, t878 = t35+t45+t47+t60+t471+t477;
    T t879 = t37+t63+t68+t80+t231+t240, t880 = t39+t46+t48+t65+t464+t470, t509 = -t279, t510 = -t280, t511 = -t281, t512 = -t283, t513 = -t284, t514 = -t285, t544 = -t336, t546 = -t340, t854 = t851*tx*ty, t855 = t852*tx*ty, t856 = t852*tx*tz;
    T t857 = t853*tx*tz, t858 = t851*ty*tz, t859 = t853*ty*tz, t860 = t15+t16+t242+t835, t861 = t17+t19+t244+t833, t862 = t20+t21+t243+t834, t872 = t245*t853, t873 = t246*t851, t874 = t247*t852;
    T t884 = t66+t76+t99+t126+t128+t236+t331+t345+t548+t550, t885 = t66+t86+t101+t127+t136+t235+t325+t352+t538+t547, t886 = t76+t86+t100+t135+t137+t237+t334+t348+t534+t535, t896 = t69+t129+t182+t203+t205+t212+t233+t326+t469+t537+t545;
    T t899 = t194+t196+t201+t204+t232+t339+t343+t444+t472+t539+t541, t900 = t191+t197+t211+t212+t238+t324+t335+t445+t465+t540+t542, t901 = t186+t187+t202+t208+t239+t338+t342+t443+t478+t543+t551;
    T t902 = t119+t121+t179+t249+t271+t292+t309+t311+t413+t414+t416+t460+t503+t569+t571+t582+t609+t624+t630+t652+t656+t665+t711+t734+t742+t748+t754;
    T t903 = t106+t107+t160+t261+t282+t287+t307+t316+t420+t421+t423+t454+t529+t568+t575+t577+t594+t597+t601+t644+t661+t672+t709+t765+t768+t776+t778;
    T t904 = t120+t122+t166+t259+t262+t273+t288+t321+t430+t431+t433+t451+t502+t595+t628+t632+t637+t642+t667+t669+t676+t678+t706+t732+t738+t741+t756;
    T t905 = t108+t110+t163+t165+t249+t303+t429+t459+t517+t520+t526+t530+t553+t567+t576+t578+t579+t593+t621+t625+t636+t640+t712+t757+t763+t767+t777;
    T t906 = t123+t124+t168+t169+t257+t282+t438+t456+t497+t507+t518+t521+t555+t583+t591+t608+t641+t654+t658+t668+t670+t677+t708+t729+t733+t736+t740;
    T t907 = t109+t111+t174+t175+t305+t321+t407+t450+t495+t505+t515+t525+t556+t572+t599+t603+t620+t623+t629+t666+t673+t680+t707+t752+t755+t769+t775, t863 = -t855, t864 = -t857, t865 = -t858, t866 = t860*tx*ty, t867 = t862*tx*ty, t868 = t860*tx*tz;
    T t869 = t861*tx*tz, t870 = t861*ty*tz, t871 = t862*ty*tz, t881 = t245*t861, t882 = t246*t862, t883 = t247*t860, t897 = t82+t132+t180+t186+t193+t195+t234+t346+t482+t546+t549, t898 = t77+t133+t181+t188+t196+t210+t241+t347+t476+t536+t544;
    T t908 = t112+t114+t167+t250+t258+t260+t410+t437+t449+t455+t506+t513+t514+t516+t519+t552+t598+t602+t622+t626+t639+t643+t762+t764+t785+t815+t817;
    T t909 = t113+t116+t177+t310+t312+t322+t409+t425+t449+t463+t493+t494+t504+t511+t512+t554+t600+t604+t610+t619+t651+t655+t749+t751+t784+t814+t819;
    T t910 = t115+t117+t162+t263+t289+t308+t422+t439+t455+t463+t509+t510+t522+t531+t533+t557+t592+t596+t627+t631+t653+t657+t737+t739+t783+t816+t818, t875 = -t867, t876 = -t868, t877 = -t870, t887 = t848+t856+t871, t888 = t849+t859+t866;
    T t889 = t850+t854+t869, t890 = t838+t863+t882, t893 = t837+t864+t883, t894 = t841+t865+t881, t891 = t836+t874+t875, t892 = t840+t872+t876, t895 = t839+t873+t877;
    
    fdrv(0, 0) = -t891*ty+t893*tz; fdrv(0, 1) = t891*tx+t889*tz;
    fdrv(0, 2) = -t893*tx-t889*ty;
    fdrv(0, 3) = t116*-2.0+t118+t125+t455+t613+t762+t797-t798+t800+t825+t831+t4*t51*6.0-t7*t50*6.0-t21*t48*4.0-t15*t69+t46*t54*3.0+t48*t52*3.0+t22*t100-t14*t126*3.0-t14*t127*3.0-t15*t129*2.0-t18*t126*2.0-t18*t127*2.0-t22*t127*2.0+t18*t135+t18*t137+t22*t135+t17*t204+t15*t329+t17*t343+t17*t472+t9*t20*t32+t9*t21*t32+t14*t98*tx+t14*t103*tx-t7*t18*ty*4.0+t21*t47*ty;
    fdrv(0, 4) = t122*-8.0+t298+t451+t501+t532+t615+t662+t753+t779+t803+t824-t7*t52*4.0-t19*t48*2.0-t20*t49*4.0+t14*t70-t18*t69*5.0-t44*t54*2.0+t46*t55*3.0+t54*t68-t55*t67*4.0+t22*t102-t15*t126*4.0-t15*t127*4.0-t16*t126*3.0-t16*t127+t14*t131-t18*t129*6.0-t22*t129*4.0+t16*t136+t22*t131+t29*t166+t52*t188+t20*t472+t596*tz+t7*t20*t32+t18*t32*t36-t14*t44*tx*2.0-t7*t15*ty*6.0+t18*t48*ty*3.0;
    fdrv(0, 5) = t123*8.0+t254+t278+t528+t574+t594+t661+t709+t766+t778+t799+t827+t4*t54*4.0-t18*t49*4.0-t21*t69+t14*t77-t22*t75*5.0-t53*t67*4.0+t21*t102+t52*t74-t17*t127*4.0-t19*t126-t19*t127*3.0+t14*t133+t19*t128-t20*t129*4.0-t21*t129*6.0+t18*t133+t21*t131*3.0+t32*t178+t54*t197+t18*t347+t4*t21*t29-t14*t45*tx*2.0+t38*t55*ty+t4*t17*tz*6.0-t15*t44*tz*4.0-t16*t44*tz*2.0+t36*t50*tz;
    fdrv(0, 6) = tdx*(t110*2.0-t119*2.0+t121*2.0+t172-t179+t315+t417-t419+t461+t462+t499+t524-t569+t607+t612+t646+t675+t731-t3*t55*2.0+t14*t46*1.2E+1+t14*t48*1.2E+1-t3*t20*tz*2.0)+t906*tdy+t904*tdz;
    fdrv(0, 7) = t906*tdx+t910*tdz-tdy*(t108*2.0E+1+t248+t303-t315+t419-t435+t442-t461-t462+t500+t582+t611+t734+t750+t758+t771-t4*t17*4.0+t7*t16*4.0+t9*t18*2.4E+1+t10*t20*4.0+t15*t44*1.2E+1-t14*t48*2.0-t18*t46*1.2E+1-t22*t48*2.0+t17*t97);
    fdrv(0, 8) = tdz*(t119*-2.0E+1-t271+t300+t417+t435-t442+t461+t462+t492+t524+t593-t636+t638+t649+t730-t771-t7*t15*4.0+t4*t19*4.0+t6*t21*4.0+t8*t22*2.4E+1-t15*t44*4.0-t17*t45*1.2E+1+t22*t48*1.2E+1+t36*t50-t3*t20*tz*1.2E+1)+t904*tdx+t910*tdy;
    fdrv(0, 9) = tdx*(t132*6.0-t134*6.0+t206+t342+t466+t474+t480+t549+t683+t685+t805+t806+t30*t36+t33*t38-t32*t67*2.0+t26*t98+t26*t103)-t900*tdy*tx+t898*tdz*tx;
    fdrv(0, 10) = tdy*(t71*-5.0-t83*8.0+t206+t474+t546+t683+t805+t808+t4*t30*4.0+t7*t33-t9*t32*4.0+t26*t48+t29*t48*3.0-t127*tx+t47*ty*6.0+t190*ty+t188*tz)+t886*tdx*ty+t898*tdz*ty;
    fdrv(0, 11) = tdz*(t78*-5.0+t82*8.0-t134*2.0+t338+t466+t480+t685+t806+t808+t4*t30+t7*t33*4.0-t9*t32*6.0+t26*t46+t32*t46*3.0-t126*tx+t102*ty+t199*tz)+t886*tdx*tz-t900*tdy*tz;
    fdrv(0, 12) = -tdy*(t220+t357+t403+t404+t562+t701+t725+t20*t33+t32*t50*4.0+t213*ty+t395*ty)-tdz*(t224+t369+t391+t392+t563+t695+t727+t21*t30+t29*t51*4.0+t405*ty+t213*tz)-t247*tdx*(t52*2.0+t54*2.0+t14*tx*3.0+t15*ty+t17*tz);
    fdrv(0, 13) = tdx*(t88+t154+t155+t184+t217+t354+t358+t374+t378+t389+t564+t687+t704+t725)+tdy*(t87+t152+t153+t183+t221+t364+t371+t380+t396+t401+t559+t693+t703+t723)+tdz*(t148+t149+t213+t218+t353+t362+t373+t390+t395+t402+t485+t689+t694+t702);
    fdrv(0, 14) = tdx*(t89+t150+t151-t215-t360+t363-t367+t370-t377+t448+t566+t688+t698+t727)+tdz*(t87+t142+t143-t222-t382+t386-t388+t393-t405+t446+t558+t696+t700+t724)-tdy*(-t146-t147+t213+t223+t359+t361+t368+t384+t387+t406+t484-t697+t780+t782);
    fdrv(0, 15) = t820*t880*tx; fdrv(0, 16) = t820*t880*ty; fdrv(0, 17) = t820*t880*tz; fdrv(0, 18) = rdx*t820*t880+t11*t488*t820+t12*t488*t820+t13*t488*t820+t11*t880*tx*2.0+t12*t880*tx*2.0+t13*t880*tx*2.0;
    fdrv(0, 19) = rdy*t820*t880-t11*t820*t845-t12*t820*t845-t13*t820*t845+t11*t880*ty*2.0+t12*t880*ty*2.0+t13*t880*ty*2.0; fdrv(0, 20) = rdz*t820*t880+t11*t820*t843+t12*t820*t843+t13*t820*t843+t11*t880*tz*2.0+t12*t880*tz*2.0+t13*t880*tz*2.0;
    fdrv(0, 24) = -t247*t804*t820; fdrv(0, 25) = t804*(t58+t90+t91+t156+t228+t230); fdrv(0, 26) = -t804*(t57-t92-t93-t157+t226+t229); fdrv(1, 0) = -t890*ty-t888*tz; fdrv(1, 1) = t890*tx-t895*tz; fdrv(1, 2) = t888*tx+t895*ty;
    fdrv(1, 3) = t111*8.0+t299+t320+t498+t664+t676+t678+t706+t732+t735+t803+t823+t7*t52*6.0-t22*t45*4.0-t14*t70*5.0+t18*t69+t44*t54*3.0+t48*t50*3.0-t17*t82-t46*t55*2.0+t15*t101+t17*t103-t54*t68*4.0+t55*t67-t16*t128*4.0-t14*t131*6.0+t18*t129-t18*t131*4.0-t15*t135+t22*t129+t15*t137-t22*t131*4.0+t15*t323+t15*t327+t18*t537-t4*t14*t29*3.0+t9*t17*t32+t14*t96*tx+t38*t53*ty;
    fdrv(1, 4) = t105-t112*2.0+t125+t463+t647+t737+t796+t802+t821+t832-t3*t53*6.0+t7*t50*4.0-t15*t45*4.0+t44*t55*3.0-t46*t54*2.0-t20*t82+t14*t127+t18*t126*4.0-t15*t131*2.0-t16*t131*2.0+t22*t127+t14*t136-t18*t135*3.0-t22*t135*2.0+t16*t197+t16*t324+t20*t349+t16*t465+t14*t538-t4*t16*t29*3.0+t7*t17*t32+t7*t19*t32+t19*t45*tx+t7*t18*ty*6.0+t15*t48*ty*3.0+t18*t104*ty-t3*t22*tz*4.0;
    fdrv(1, 5) = t119*-8.0+t300+t460+t492+t499-t567+t649+t675+t730+t731+t795+t830-t3*t55*4.0-t19*t45*4.0+t18*t83-t22*t82*5.0+t14*t103-t51*t68*4.0+t50*t74+t21*t126+t15*t133-t17*t131*4.0+t16*t133-t19*t131*6.0-t20*t135*4.0-t21*t135*3.0+t32*t179+t55*t205+t21*t331+t15*t347+t15*t444+t19*t465+t666*ty+t19*t29*t59+t19*t44*tx-t18*t47*ty*2.0-t3*t20*tz*6.0-t16*t46*tz*4.0+t42*t44*tz;
    fdrv(1, 6) = tdx*(t107*-2.0E+1+t253-t316-t412+t428+t441+t457+t458+t508+t523+t581-t583+t585+t677-t773+t774+t7*t14*2.4E+1-t3*t20*4.0+t9*t15*4.0+t10*t17*4.0+t14*t44*1.2E+1-t16*t46*1.2E+1-t20*t47*4.0+t40*t53-t4*t17*tz*1.2E+1)+t902*tdy+t909*tdz;
    fdrv(1, 7) = tdy*(t106*2.0-t107*2.0+t124*2.0-t160+t178+t269-t427+t428+t457+t458+t523+t527+t570+t587+t648-t661+t679+t743-t4*t54*2.0+t18*t44*1.2E+1+t18*t49*1.2E+1-t4*t17*tz*2.0)+t902*tdx+t907*tdz;
    fdrv(1, 8) = t909*tdx+t907*tdy-tdz*(t123*2.0E+1+t257-t269+t278+t412+t427-t441-t457-t458+t528+t574+t672+t744+t766+t773+t776+t2*t19*4.0+t3*t21*4.0-t7*t18*4.0+t5*t22*2.4E+1-t14*t44*2.0-t18*t44*2.0+t20*t47*1.2E+1-t22*t49*1.2E+1+t16*t98);
    fdrv(1, 9) = tdx*(t72*-5.0+t77*8.0-t130*2.0+t339+t473+t479+t681+t807+t809+t7*t32*4.0+t9*t33+t2*t57+t29*t49+t26*t96-t45*tx*6.0-t135*ty+t348*ty)+t885*tdy*tx-t901*tdz*tx;
    fdrv(1, 10) = tdy*(t130*-6.0+t133*6.0+t199+t343+t468+t473+t479+t536+t681+t686+t807+t811+t2*t30*4.0+t26*t44*2.0+t33*t40-t32*t68*2.0+t29*t104)+t896*tdx*ty-t901*tdz*ty;
    fdrv(1, 11) = tdz*(t75*-8.0-t84*5.0+t199+t468+t544+t686+t809+t811+t2*t30+t7*t32*6.0+t9*t33*4.0+t29*t38+t26*t44+t32*t44*3.0-t45*tx*4.0+t207*tz+t29*t59*tx)+t896*tdx*tz+t885*tdy*tz;
    fdrv(1, 12) = tdx*(t88+t154+t155-t217+t354+t358-t374-t378-t389+t447+t564+t687+t704+t725)+tdy*(t87+t152+t153-t221+t364+t371-t380-t396-t401+t446+t559+t693+t703+t723)-tdz*(-t148-t149+t213+t218+t353+t362+t373+t390+t395+t402+t485-t702+t780+t781);
    fdrv(1, 13) = -tdx*(t214+t372+t393+t394+t561+t700+t723+t17*t33+t16*t57+t32*t52*4.0+t373*tx)-tdz*(t224+t369+t370+t391+t563+t688+t728+t19*t27+t184*tx+t397*tx+t218*tz)-t246*tdy*(t55*2.0+t16*tx+t18*ty*3.0+t41*ty+t20*tz);
    fdrv(1, 14) = tdx*(t144+t145+t218+t223+t355+t365+t376+t383+t398+t399+t483+t692+t694+t699)+tdy*(t89+t140+t141+t185+t216+t356+t375+t379+t385+t392+t565+t691+t695+t728)+tdz*(t88+t138+t139+t184+t219+t366+t381+t397+t400+t403+t560+t690+t701+t726);
    fdrv(1, 15) = -t820*t879*tx; fdrv(1, 16) = -t820*t879*ty; fdrv(1, 17) = -t820*t879*tz; fdrv(1, 18) = -rdx*t820*t879+t11*t820*t842+t12*t820*t842+t13*t820*t842-t11*t879*tx*2.0-t12*t879*tx*2.0-t13*t879*tx*2.0;
    fdrv(1, 19) = -rdy*t820*t879+t11*t487*t820+t12*t487*t820+t13*t487*t820-t11*t879*ty*2.0-t12*t879*ty*2.0-t13*t879*ty*2.0; fdrv(1, 20) = -rdz*t820*t879-t11*t820*t847-t12*t820*t847-t13*t820*t847-t11*t879*tz*2.0-t12*t879*tz*2.0-t13*t879*tz*2.0;
    fdrv(1, 24) = -t804*(t58-t90-t91-t156+t228+t230); fdrv(1, 25) = -t246*t804*t820; fdrv(1, 26) = t804*(t56+t94+t95+t158+t225+t227); fdrv(2, 0) = t887*ty+t892*tz; fdrv(2, 1) = -t887*tx-t894*tz; fdrv(2, 2) = -t892*tx+t894*ty;
    fdrv(2, 3) = t107*-8.0+t253+t454+t508+t527+t570+t585-t641+t743+t774+t799+t826-t4*t54*6.0-t15*t46*4.0-t14*t77*5.0+t22*t75+t46*t51*3.0+t18*t96+t53*t67-t52*t74*4.0-t14*t133*6.0+t20*t129+t21*t129-t18*t133*4.0+t17*t135-t19*t136*4.0+t15*t180+t17*t327+t15*t478+t576*tz-t7*t14*t32*3.0-t7*t18*t32*2.0-t7*t22*t32*2.0+t21*t32*t36+t15*t32*t64+t14*t97*tx+t15*t47*ty-t21*t48*ty*4.0-t4*t17*tz*4.0;
    fdrv(2, 4) = t108*8.0+t248+t297+t500+t569+t571+t611+t711+t742+t750+t795+t829+t3*t55*6.0-t14*t46*4.0-t16*t77+t44*t53*3.0-t18*t83*5.0+t22*t82+t16*t96+t20*t99+t51*t68-t50*t74*4.0+t20*t127-t15*t133*4.0+t17*t131-t16*t133*6.0+t19*t131-t21*t137*4.0+t21*t327+t20*t341+t22*t549+t577*tz+t594*tz-t7*t15*t32*2.0-t7*t16*t32-t9*t18*t32*3.0+t16*t45*tx-t19*t48*ty*4.0+t18*t102*ty;
    fdrv(2, 5) = t105-t115*2.0+t118+t449-t455+t590+t751+t798+t801+t822+t828-t4*t51*4.0+t3*t53*4.0-t19*t77-t48*t52*2.0+t14*t126+t14*t128+t18*t126+t15*t131*3.0+t22*t127*4.0-t19*t133*2.0+t22*t135*4.0+t21*t187+t19*t328+t18*t331+t21*t342+t15*t445+t21*t478+t14*t548+t18*t550+t4*t15*t29+t4*t16*t29-t7*t19*t32*3.0-t9*t21*t32*3.0+t16*t44*tx+t3*t22*tz*6.0-t4*t22*tz*6.0;
    fdrv(2, 6) = t908*tdy+t905*tdz-tdx*(t111*2.0E+1-t268+t305+t320-t408+t418+t436-t452-t453+t498+t637+t664+t735+t746+t756+t761+t4*t14*2.4E+1+t6*t15*4.0-t3*t22*4.0+t8*t17*4.0-t14*t45*1.2E+1-t18*t47*2.0+t19*t48*1.2E+1-t22*t47*2.0+t21*t104);
    fdrv(2, 7) = tdy*(t122*-2.0E+1-t273+t298+t408-t418+t434+t452+t453+t496+t532+t572+t615+t671-t673+t753-t761+t2*t16*4.0+t3*t18*2.4E+1+t5*t20*4.0-t4*t22*4.0+t18*t47*1.2E+1-t19*t48*4.0-t21*t49*1.2E+1+t43*t45-t7*t15*ty*1.2E+1)+t908*tdx+t903*tdz;
    fdrv(2, 8) = tdz*(t109*2.0+t120*2.0-t122*2.0+t159-t166+t268+t434-t436+t452+t453+t496+t501+t580+t589+t617+t662-t676+t779-t7*t52*2.0+t22*t45*1.2E+1+t22*t47*1.2E+1-t7*t15*ty*2.0)+t905*tdx+t903*tdy;
    fdrv(2, 9) = tdx*(t70*-8.0-t79*5.0+t189+t481+t545+t682+t810+t812-t4*t29*4.0+t6*t33+t2*t58+t32*t35+t29*t47+t26*t97+t32*t477+t44*tx*6.0+t334*tz)+t897*tdy*tx+t884*tdz*tx;
    fdrv(2, 10) = tdy*(t69*8.0-t85*5.0-t131*2.0+t335+t467+t475+t684+t810+t813-t4*t29*6.0+t2*t33+t6*t58+t26*t45+t29*t45*3.0+t29*t102+t32*t471+t96*tx)-t899*tdx*ty+t884*tdz*ty;
    fdrv(2, 11) = tdz*(t129*6.0-t131*6.0+t189+t324+t467+t475+t481+t537+t682+t684+t812+t813+t2*t33*4.0+t6*t33*4.0+t26*t45*2.0+t29*t47*2.0-t29*t74*2.0)-t899*tdx*tz+t897*tdy*tz;
    fdrv(2, 12) = tdx*(t89+t150+t151+t185+t215+t360+t363+t367+t370+t377+t566+t688+t698+t727)+tdy*(t146+t147+t213+t223+t359+t361+t368+t384+t387+t406+t484+t689+t697+t699)+tdz*(t87+t142+t143+t183+t222+t382+t386+t388+t393+t405+t558+t696+t700+t724);
    fdrv(2, 13) = tdy*(t89+t140+t141-t216-t356-t375-t379+t385+t392+t448+t565+t691+t695+t728)+tdz*(t88+t138+t139-t219-t366-t381-t397+t400+t403+t447+t560+t690+t701+t726)-tdx*(-t144-t145+t218+t223+t355+t365+t376+t383+t398+t399+t483-t692+t781+t782);
    fdrv(2, 14) = -tdx*(t214+t371+t372+t394+t561+t693+t724+t15*t30+t19*t58+t29*t54*4.0+t359*tx)-tdy*(t220+t357+t358+t404+t562+t687+t726+t16*t27+t21*t58+t185*tx+t375*tx)-t245*tdz*(t19*tx+t21*ty+t22*tz*3.0+t41*tz+t42*tz); fdrv(2, 15) = t820*t878*tx;
    fdrv(2, 16) = t820*t878*ty; fdrv(2, 17) = t820*t878*tz; fdrv(2, 18) = rdx*t820*t878-t11*t820*t846-t12*t820*t846-t13*t820*t846+t11*t878*tx*2.0+t12*t878*tx*2.0+t13*t878*tx*2.0;
    fdrv(2, 19) = rdy*t820*t878+t11*t820*t844+t12*t820*t844+t13*t820*t844+t11*t878*ty*2.0+t12*t878*ty*2.0+t13*t878*ty*2.0; fdrv(2, 20) = rdz*t820*t878+t11*t486*t820+t12*t486*t820+t13*t486*t820+t11*t878*tz*2.0+t12*t878*tz*2.0+t13*t878*tz*2.0;
    fdrv(2, 24) = t804*(t57+t92+t93+t157+t226+t229); fdrv(2, 25) = -t804*(t56-t94-t95-t158+t225+t227); fdrv(2, 26) = -t245*t804*t820;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f31(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = ox*ty, t4 = oy*tx, t5 = ox*tz, t6 = oy*ty, t7 = oz*tx, t8 = oy*tz, t9 = oz*ty, t10 = oz*tz, t11 = rdx*tx, t12 = rdy*ty, t13 = rdz*tz, t14 = tx*tx, t15 = tx*tx*tx, t17 = ty*ty, t18 = ty*ty*ty, t20 = tz*tz, t21 = tz*tz*tz;
    T t29 = rx*tx*3.0, t30 = ry*ty*3.0, t31 = rz*tz*3.0, t38 = rx*tx*ty, t39 = rx*tx*tz, t40 = ry*tx*ty, t41 = ry*ty*tz, t42 = rz*tx*tz, t43 = rz*ty*tz, t16 = t14*t14, t19 = t17*t17, t22 = t20*t20, t23 = t3*2.0, t24 = t4*2.0, t25 = t5*2.0;
    T t26 = t7*2.0, t27 = t8*2.0, t28 = t9*2.0, t32 = t2*ty, t33 = t2*tz, t34 = t4*ty, t35 = t6*tz, t36 = t7*tz, t37 = t9*tz, t44 = -t4, t46 = -t7, t48 = -t9, t50 = t2*t14, t51 = t3*ty, t52 = t4*tx, t53 = t3*t17, t54 = t4*t14, t55 = t3*t18;
    T t56 = t4*t15, t57 = t5*tz, t58 = t7*tx, t59 = t5*t20, t60 = t6*t17, t61 = t7*t14, t62 = t5*t21, t63 = t7*t15, t64 = t8*tz, t65 = t9*ty, t66 = t8*t20, t67 = t9*t17, t68 = t8*t21, t69 = t9*t18, t70 = t10*t20, t71 = rx*t15, t72 = rx*t17;
    T t73 = ry*t14, t74 = rx*t20, t75 = rz*t14, t76 = ry*t18, t77 = ry*t20, t78 = rz*t17, t79 = rz*t21, t80 = t38*2.0, t81 = t39*2.0, t82 = t40*2.0, t83 = t41*2.0, t84 = t42*2.0, t85 = t43*2.0, t107 = t2*t17, t108 = t2*t20, t110 = t3*t20;
    T t112 = t4*t20, t114 = t7*t17, t116 = t6*t20, t134 = rx*t14*3.0, t137 = ry*t17*3.0, t140 = rz*t20*3.0, t155 = t14+t17, t156 = t14+t20, t157 = t17+t20, t192 = t4*t17*3.0, t206 = t7*t20*3.0, t210 = t9*t20*3.0, t213 = t17*t29;
    T t214 = rx*t18*tx*4.0, t217 = t20*t29, t218 = t14*t30, t219 = rx*t21*tx*4.0, t220 = ry*t15*ty*4.0, t223 = t20*t30, t224 = t14*t31, t225 = ry*t21*ty*4.0, t226 = rz*t15*tz*4.0, t228 = t17*t31, t229 = rz*t18*tz*4.0, t257 = t2+t6, t258 = t2+t10;
    T t259 = t6+t10, t279 = t4*t17*-2.0, t289 = t7*t20*-2.0, t291 = t9*t20*-2.0, t303 = t2*t30*tx, t327 = t4*t29*tz, t329 = t7*t29*ty, t332 = ry*t4*t17*6.0, t343 = t3*t30*tz, t345 = t2*t31*tx, t385 = t6*t31*ty, t392 = rz*t7*t20*6.0;
    T t396 = rz*t9*t20*6.0, t440 = t4*t39*1.2E+1, t442 = t7*t38*1.2E+1, t452 = t3*t41*1.2E+1, t454 = t7*t40*1.2E+1, t463 = t3*t43*1.2E+1, t465 = t4*t42*1.2E+1, t482 = t29+t30, t483 = t29+t31, t484 = t30+t31, t494 = t11+t12+t13, t45 = -t24;
    T t47 = -t26, t49 = -t28, t86 = t2*t71, t87 = ry*t50, t88 = rx*t54, t89 = ry*t53, t90 = rz*t50, t91 = rx*t60, t92 = rx*t61, t93 = ry*t59, t94 = rz*t53, t95 = rx*t66, t96 = rz*t54, t97 = rx*t67, t98 = ry*t61, t99 = t6*t76, t100 = rz*t59;
    T t101 = rz*t60, t102 = rx*t70, t103 = ry*t67, t104 = rz*t66, t105 = ry*t70, t106 = t10*t79, t109 = t34*tx, t111 = t51*tz, t113 = t52*tz, t115 = t58*ty, t117 = t36*tx, t118 = t37*ty, t119 = t72*tx, t120 = t74*tx, t121 = t73*ty, t122 = t77*ty;
    T t123 = t75*tz, t124 = t78*tz, t125 = t32*tz*2.0, t126 = t24*ty*tz, t127 = t26*ty*tz, t128 = t53*4.0, t129 = t54*4.0, t130 = t59*4.0, t131 = t61*4.0, t132 = t66*4.0, t133 = t67*4.0, t135 = t71*4.0, t136 = rx*t16*5.0, t138 = t76*4.0;
    T t139 = ry*t19*5.0, t141 = t79*4.0, t142 = rz*t22*5.0, t143 = -t32, t144 = -t33, t145 = -t34, t146 = -t35, t147 = -t36, t148 = -t37, t149 = -t80, t150 = -t81, t151 = -t82, t152 = -t83, t153 = -t84, t154 = -t85, t158 = rx*t50*2.0;
    T t159 = rx*t50*4.0, t166 = ry*t60*2.0, t170 = ry*t60*4.0, t180 = rz*t70*2.0, t181 = rz*t70*4.0, t182 = t107*2.0, t183 = t32*tx*2.0, t184 = t107*3.0, t185 = t32*tx*3.0, t186 = t108*2.0, t187 = t33*tx*2.0, t188 = t17*t24, t189 = t24*tx*ty;
    T t190 = t108*3.0, t191 = t33*tx*3.0, t194 = t110*4.0, t196 = t112*4.0, t198 = t114*4.0, t200 = t116*2.0, t201 = t35*ty*2.0, t202 = t20*t26, t203 = t26*tx*tz, t204 = t116*3.0, t205 = t35*ty*3.0, t208 = t20*t28, t209 = t28*ty*tz;
    T t230 = rx*t32*tz*4.0, t231 = ry*t32*tz*4.0, t232 = rx*t34*tz*4.0, t233 = ry*t32*tz*6.0, t234 = rx*t34*tz*6.0, t235 = rz*t32*tz*4.0, t236 = ry*t34*tz*4.0, t237 = rx*t36*ty*4.0, t238 = rz*t32*tz*6.0, t239 = rx*t36*ty*6.0, t240 = rz*t34*tz*4.0;
    T t241 = ry*t36*ty*4.0, t242 = rz*t34*tz*6.0, t243 = ry*t36*ty*6.0, t244 = rz*t36*ty*4.0, t245 = -t55, t246 = t15*t44, t249 = -t62, t250 = t15*t46, t253 = -t68, t254 = t18*t48, t261 = rx*t53*-4.0, t262 = rx*t59*-4.0, t263 = ry*t54*-4.0;
    T t271 = ry*t66*-4.0, t272 = rz*t61*-4.0, t273 = rz*t67*-4.0, t293 = t20*t72, t294 = t20*t73, t295 = t17*t75, t296 = t2*t72*2.0, t298 = t2*t72*6.0, t299 = rx*t32*tx*6.0, t300 = t2*t74*2.0, t304 = t4*t72*3.0, t305 = t29*t34, t306 = t2*t74*6.0;
    T t307 = rx*t33*tx*6.0, t308 = ry*t107*6.0, t309 = ry*t32*tx*6.0, t310 = t4*t72*6.0, t312 = t2*t77*2.0, t314 = t2*t78*2.0, t316 = t24*t74, t317 = t24*t39, t319 = t24*t40, t320 = t26*t72, t321 = t26*t38, t322 = t2*t77*3.0, t324 = t2*t78*3.0;
    T t326 = t4*t74*3.0, t328 = t7*t72*3.0, t330 = t3*t74*4.0, t334 = t23*t77, t335 = t23*t41, t336 = t6*t74*2.0, t340 = ry*t17*t26, t341 = t26*t40, t342 = t3*t77*3.0, t346 = t6*t74*3.0, t348 = t4*t78*3.0, t350 = t7*t74*3.0, t351 = t29*t36;
    T t352 = ry*t114*3.0, t353 = t30*t58, t354 = t4*t77*4.0, t356 = rz*t108*6.0, t357 = rz*t33*tx*6.0, t358 = t7*t74*6.0, t360 = rz*t20*t23, t361 = t23*t43, t362 = t6*t77*2.0, t364 = rz*t20*t24, t365 = t24*t42, t370 = rz*t110*3.0, t371 = t31*t51;
    T t372 = rz*t112*3.0, t373 = t31*t52, t374 = t9*t74*3.0, t376 = t7*t77*3.0, t378 = t7*t78*4.0, t380 = t6*t77*6.0, t381 = ry*t35*ty*6.0, t383 = t26*t42, t386 = t9*t77*3.0, t387 = t30*t37, t388 = rz*t116*6.0, t389 = rz*t35*ty*6.0;
    T t390 = t9*t77*6.0, t395 = t28*t43, t398 = t20*t32*2.0, t399 = t17*t33*2.0, t400 = t20*t24*ty, t402 = t17*t26*tz, t404 = rx*t32*tz*8.0, t413 = ry*t34*tz*8.0, t418 = rz*t36*ty*8.0, t421 = t17*t52*3.0, t422 = t20*t51*6.0, t423 = t20*t52*6.0;
    T t424 = t17*t58*6.0, t426 = t20*t58*3.0, t427 = t20*t65*3.0, t433 = ry*t33*tx*-2.0, t435 = rz*t32*tx*-2.0, t438 = -t332, t439 = t4*t74*1.2E+1, t441 = t7*t72*1.2E+1, t444 = rx*t35*ty*-2.0, t445 = t4*t78*-2.0, t451 = t3*t77*1.2E+1;
    T t453 = ry*t114*1.2E+1, t455 = t9*t74*-2.0, t457 = t7*t77*-2.0, t462 = rz*t110*1.2E+1, t464 = rz*t112*1.2E+1, t468 = -t392, t469 = -t396, t473 = rx*t155, t474 = rx*t156, t475 = ry*t155, t476 = ry*t157, t477 = rz*t156, t478 = rz*t157;
    T t486 = t32*t77*2.0, t488 = t33*t78*2.0, t495 = t107*tx*-3.0, t496 = t108*tx*-3.0, t501 = t116*ty*-3.0, t504 = -t440, t505 = -t442, t506 = -t452, t510 = t20+t155, t514 = t29*t107, t515 = t29*t108, t517 = t4*t213, t518 = t52*t137;
    T t520 = t7*t217, t521 = t30*t116, t523 = t9*t223, t524 = t58*t140, t525 = t65*t140, t526 = t32*t39*-2.0, t527 = ry*t279*tz, t528 = rz*t289*ty, t529 = t482*tx, t530 = t482*ty, t531 = t483*tx, t532 = t483*tz, t533 = t484*ty, t534 = t484*tz;
    T t550 = t72+t74+t82+t84+t134, t551 = t73+t77+t80+t85+t137, t552 = t75+t78+t81+t83+t140, t161 = t88*4.0, t163 = t89*4.0, t165 = t92*4.0, t167 = t93*4.0, t168 = t94*4.0, t169 = t95*4.0, t171 = t96*4.0, t172 = t97*4.0, t173 = t98*4.0;
    T t174 = t100*4.0, t176 = t103*4.0, t178 = t104*4.0, t193 = t109*3.0, t195 = t111*4.0, t197 = t113*4.0, t199 = t115*4.0, t207 = t117*3.0, t211 = t118*3.0, t212 = t119*2.0, t215 = t120*2.0, t216 = t121*2.0, t221 = t122*2.0, t222 = t123*2.0;
    T t227 = t124*2.0, t247 = -t128, t248 = -t129, t251 = -t130, t252 = -t131, t255 = -t132, t256 = -t133, t260 = -t86, t264 = -t99, t274 = -t106, t275 = -t182, t276 = -t183, t277 = -t186, t278 = -t187, t280 = t109*-2.0, t281 = -t194;
    T t283 = -t196, t285 = -t198, t287 = -t200, t288 = -t201, t290 = t117*-2.0, t292 = t118*-2.0, t297 = rx*t183, t301 = rx*t187, t302 = ry*t184, t311 = rx*t109*6.0, t318 = ry*t188, t323 = ry*t191, t325 = rz*t185, t333 = ry*t109*6.0;
    T t339 = rz*t189, t344 = rz*t190, t347 = rx*t205, t359 = rx*t117*6.0, t363 = ry*t201, t367 = rx*t209, t369 = ry*t203, t382 = rz*t202, t384 = rz*t204, t391 = ry*t118*6.0, t393 = rz*t117*6.0, t394 = rz*t208, t397 = rz*t118*6.0, t401 = t126*tx;
    T t403 = t127*tx, t405 = -t231, t406 = -t232, t407 = -t233, t408 = -t234, t409 = -t235, t410 = -t237, t411 = -t238, t412 = -t239, t414 = -t240, t415 = -t241, t416 = -t242, t417 = -t243, t428 = -t299, t429 = -t307, t430 = -t308, t432 = -t312;
    T t434 = -t314, t436 = -t330, t437 = rx*t111*-4.0, t443 = -t336, t447 = -t354, t448 = ry*t113*-4.0, t449 = -t356, t459 = -t378, t460 = rz*t115*-4.0, t461 = -t381, t466 = -t388, t470 = -t404, t471 = -t413, t472 = -t418, t479 = t3+t45;
    T t480 = t5+t47, t481 = t8+t49, t487 = t316*ty, t490 = t320*tz, t491 = t365*ty, t492 = t341*tz, t497 = -t421, t498 = -t422, t499 = -t423, t500 = -t424, t502 = -t426, t503 = -t427, t507 = -t453, t508 = -t462, t509 = -t464, t535 = -t529;
    T t536 = -t530, t537 = -t531, t538 = -t532, t539 = -t533, t540 = -t534, t544 = t494*t510*tx*ty, t545 = t494*t510*tx*tz, t546 = t494*t510*ty*tz, t553 = t550*tdx*ty*tz, t554 = t551*tdy*tx*tz, t555 = t552*tdz*tx*ty, t265 = -t167, t266 = -t168;
    T t267 = -t169, t268 = -t171, t269 = -t172, t270 = -t173, t282 = -t195, t284 = -t197, t286 = -t199, t349 = rz*t193, t375 = rx*t211, t377 = ry*t207, t431 = -t311, t446 = rz*t280, t450 = -t359, t456 = rx*t292, t458 = ry*t290, t467 = -t391;
    T t516 = t302*tx, t519 = t344*tx, t522 = t384*ty, t547 = -t544, t548 = -t545, t549 = -t546, t556 = -t553, t557 = -t554, t558 = -t555, t559 = t40+t153+t473+t474+t535, t560 = t42+t151+t473+t474+t537, t561 = t38+t154+t475+t476+t536;
    T t562 = t43+t149+t475+t476+t539, t563 = t39+t152+t477+t478+t538, t564 = t41+t150+t477+t478+t540, t580 = t53+t110+t127+t185+t248+t279+t283, t581 = t54+t112+t127+t192+t247+t276+t281, t582 = t59+t111+t126+t191+t252+t285+t289;
    T t592 = t50+t60+t107+t109+t190+t204+t290+t292, t593 = t50+t70+t108+t117+t184+t211+t280+t287, t594 = t60+t70+t116+t118+t193+t207+t275+t277, t595 = t71+t76+t119+t121+t141+t217+t222+t223+t227, t596 = t71+t79+t120+t123+t138+t213+t216+t221+t228;
    T t597 = t76+t79+t122+t124+t135+t212+t215+t218+t224, t565 = t559*tx*tz, t566 = t560*tx*ty, t567 = t559*ty*tz, t568 = t560*ty*tz, t569 = t561*tx*tz, t570 = t561*ty*tz, t571 = t562*tx*ty, t572 = t562*tx*tz, t573 = t563*tx*ty, t574 = t564*tx*ty;
    T t575 = t563*ty*tz, t576 = t564*tx*tz, t583 = t61+t114+t126+t206+t251+t278+t282, t584 = t66+t113+t125+t205+t256+t286+t291, t585 = t67+t115+t125+t210+t255+t284+t288, t586 = t155*t559, t587 = t156*t560, t588 = t155*t561, t589 = t157*t562;
    T t590 = t156*t563, t591 = t157*t564, t598 = t596*tdy*tx, t599 = t595*tdz*tx, t600 = t597*tdx*ty, t601 = t595*tdz*ty, t602 = t597*tdx*tz, t603 = t596*tdy*tz;
    T t619 = t96+t98+t236+t244+t265+t266+t317+t321+t348+t352+t372+t376+t433+t435+t470+t506+t508, t620 = t94+t97+t230+t244+t267+t268+t325+t329+t335+t340+t370+t374+t444+t445+t471+t504+t509;
    T t621 = t93+t95+t230+t236+t269+t270+t323+t327+t343+t347+t360+t364+t455+t457+t472+t505+t507, t622 = t100+t102+t159+t163+t241+t263+t298+t300+t309+t334+t345+t351+t371+t375+t382+t414+t431+t438+t443+t447;
    T t623 = t89+t91+t159+t174+t240+t272+t296+t303+t305+t306+t318+t342+t346+t357+t361+t415+t450+t456+t459+t468, t624 = t104+t105+t161+t170+t237+t261+t310+t316+t333+t362+t373+t377+t385+t387+t394+t409+t428+t430+t432+t436;
    T t625 = t87+t88+t170+t178+t235+t273+t297+t302+t304+t319+t322+t326+t365+t380+t389+t410+t458+t460+t467+t469, t626 = t101+t103+t165+t181+t232+t262+t320+t349+t353+t358+t363+t384+t386+t393+t395+t405+t429+t434+t437+t449;
    T t627 = t90+t92+t176+t181+t231+t271+t301+t324+t328+t341+t344+t350+t383+t390+t397+t406+t446+t448+t461+t466, t577 = -t572, t578 = -t573, t579 = -t574, t604 = -t598, t605 = -t599, t606 = -t600, t607 = -t601, t608 = -t602, t609 = -t603;
    T t613 = t571+t587, t614 = t566+t589, t615 = t576+t586, t616 = t565+t591, t617 = t575+t588, t618 = t570+t590, t610 = t568+t577, t611 = t567+t579, t612 = t569+t578, t628 = t556+t607+t609, t629 = t557+t605+t608, t630 = t558+t604+t606;
    
    fdrv(0, 0) = -t614*ty-t616*tz; fdrv(0, 1) = t614*tx-t612*tz; fdrv(0, 2) = t616*tx+t612*ty;
    fdrv(0, 3) = t264+t274+t486+t488+t514+t515+rx*t55+rx*t62-t34*t42*3.0-t36*t40*3.0+t2*t76*2.0+t2*t79*2.0-t6*t79-t34*t74*2.0-t36*t72*2.0-t6*t122+t78*t146-rx*t4*t18*2.0-rx*t7*t21*2.0-rx*t14*t34*4.0-rx*t14*t36*4.0-ry*t17*t52*3.0+ry*t21*t48+ry*t17*t148-rz*t20*t58*3.0+t23*t74*ty+rz*t20*t48*ty;
    fdrv(0, 4) = t516+t528+ry*t55*5.0+ry*t62-t36*t38*2.0-t4*t76*4.0+t32*t74*4.0-t34*t73*2.0-t34*t77*2.0+t44*t71+t32*t84+t44*t79-t52*t72*3.0-t4*t124*3.0+t51*t77*6.0+t3*t141+t44*t120+t44*t123+t73*t147+t168*tz+rx*t2*t18*4.0+rx*t14*t32*2.0-ry*t17*t36*3.0+ry*t21*t46+t2*t77*tx;
    fdrv(0, 5) = t519+t527+rz*t55+rz*t62*5.0+rz*t422-t34*t39*2.0-t7*t79*4.0+t33*t72*4.0-t36*t75*2.0-t36*t78*2.0+t46*t71+t46*t76-t7*t122*3.0-t58*t74*3.0+t46*t119+t46*t121+t75*t145+t163*tz+rx*t2*t21*4.0+rx*t14*t33*2.0+ry*t3*t21*4.0-rz*t20*t34*3.0+rz*t18*t44+ry*t125*tx+t2*t78*tx;
    fdrv(0, 6) = -t624*tdy-t626*tdz-tdx*(t89*-2.0+t91*2.0-t100*2.0+t102*2.0+t242+t243-t298-t306+t332+t336+t367+t392+rx*t109*1.2E+1+rx*t117*1.2E+1-t3*t43*2.0-t3*t77*2.0);
    fdrv(0, 7) = -t624*tdx-t619*tdz+tdy*(t89*2.0E+1+t158+t174+t309+t416+t417+t431+t451+t463+rx*t290-ry*t54*2.0+rz*t187+rz*t289+t2*t72*1.2E+1+t2*t74*4.0-t4*t77*2.0-ry*t4*t17*1.2E+1);
    fdrv(0, 8) = -t626*tdx-t619*tdy+tdz*(t100*2.0E+1+t158+t163+t357+t416+t417+t450+t451+t463+rx*t280+ry*t183+ry*t279-rz*t61*2.0+t2*t72*4.0+t2*t74*1.2E+1-t7*t78*2.0-rz*t7*t20*1.2E+1);
    fdrv(0, 9) = -tdx*(t245+t249+t400+t402+t495+t496+t18*t24+t21*t26+t14*t34*4.0+t14*t36*4.0-t20*t51*2.0)-t581*tdy*tx-t583*tdz*tx;
    fdrv(0, 10) = -tdy*(t55*-5.0+t249+t400+t495+t498+t4*t18*4.0+t7*t21+t14*t36+t17*t36*3.0-t108*tx+t14*t24*ty)-t594*tdx*ty-t583*tdz*ty;
    fdrv(0, 11) = -tdz*(t62*-5.0+t245+t402+t496+t498+t4*t18+t7*t21*4.0+t14*t34+t20*t34*3.0-t107*tx+t14*t26*tz)-t594*tdx*tz-t581*tdy*tz;
    fdrv(0, 12) = tdy*(t139+t214+t229+t294+ry*t22+t20*t38*4.0+t17*t73*3.0+t17*t77*6.0+t14*t85+t71*ty*2.0+t141*ty)+tdz*(t142+t219+t225+t295+rz*t19+t17*t39*4.0+t20*t75*3.0+t14*t83+t20*t78*6.0+t71*tz*2.0+t138*tz)+t157*t550*tdx; fdrv(0, 13) = t630;
    fdrv(0, 14) = t629; fdrv(0, 15) = -t510*tx*(t34+t36-t51-t57); fdrv(0, 16) = -t510*ty*(t34+t36-t51-t57); fdrv(0, 17) = -t510*tz*(t34+t36-t51-t57);
    fdrv(0, 18) = -rdx*t510*(t34+t36-t51-t57)-t11*tx*(t34+t36-t51-t57)*2.0-t12*tx*(t34+t36-t51-t57)*2.0-t13*tx*(t34+t36-t51-t57)*2.0-t11*t259*t510-t12*t259*t510-t13*t259*t510;
    fdrv(0, 19) = -rdy*t510*(t34+t36-t51-t57)-t11*ty*(t34+t36-t51-t57)*2.0-t12*ty*(t34+t36-t51-t57)*2.0-t13*ty*(t34+t36-t51-t57)*2.0-t11*t510*(t4-t23)-t12*t510*(t4-t23)-t13*t510*(t4-t23);
    fdrv(0, 20) = -rdz*t510*(t34+t36-t51-t57)-t11*tz*(t34+t36-t51-t57)*2.0-t12*tz*(t34+t36-t51-t57)*2.0-t13*tz*(t34+t36-t51-t57)*2.0-t11*t510*(t7-t25)-t12*t510*(t7-t25)-t13*t510*(t7-t25); fdrv(0, 24) = t157*t494*t510; fdrv(0, 25) = t547;
    fdrv(0, 26) = t548; fdrv(1, 0) = t613*ty-t611*tz; fdrv(1, 1) = -t613*tx-t618*tz; fdrv(1, 2) = t611*tx+t618*ty;
    fdrv(1, 3) = t517+t528+rx*t56*5.0+rx*t68+ry*t245+ry*t495-t32*t42*3.0-t36*t38*3.0-t3*t79+t24*t76-t32*t74*2.0+t34*t73*4.0+t34*t77*4.0+t52*t74*6.0-t51*t77+t4*t141+t24*t124+t72*t148-t94*tz+t171*tz-rx*t2*t18*2.0-rx*t14*t32*4.0+rx*t21*t48-ry*t17*t36*2.0+t6*t74*ty;
    fdrv(1, 4) = t260+t274-t486+t487+t491+t518+t521+ry*t56+ry*t68-t36*t40*2.0-t2*t76*4.0-t2*t79+t6*t79*2.0-t32*t73*2.0-t36*t72*3.0-t33*t78*3.0-t2*t119*3.0-t2*t120+t75*t144+rx*t21*t46+rx*t14*t147-ry*t9*t21*2.0-ry*t17*t37*4.0-rz*t20*t65*3.0+t24*t77*tx+rx*t14*t24*ty+rz*t20*t46*tx;
    fdrv(1, 5) = t522+t526+rz*t56+rz*t68*5.0+rz*t423-t9*t79*4.0-t37*t78*2.0+t48*t76+t52*t78-t65*t77*3.0+t75*t143+t161*tz+rx*t4*t21*4.0+rx*t18*t46+ry*t6*t21*4.0-ry*t17*t33*2.0+ry*t17*t35*2.0-rz*t2*t18-rz*t20*t32*3.0+ry*t109*tz*4.0+rz*t290*ty-t7*t74*ty*3.0+t24*t72*tz+rx*t14*t46*ty+ry*t17*t46*tx;
    fdrv(1, 6) = -t622*tdy-t620*tdz+tdx*(t88*2.0E+1+t166+t178+t310+t411+t412+t430+t439+t465-rx*t53*2.0+ry*t109*1.2E+1+ry*t292+rz*t201+rz*t291-t3*t74*2.0+t6*t77*4.0-rx*t32*tx*1.2E+1);
    fdrv(1, 7) = -t622*tdx-t627*tdz-tdy*(t87*2.0-t88*2.0-t104*2.0+t105*2.0+t238+t239+t299+t312-t333+t369-t380+t396+ry*t107*1.2E+1+ry*t118*1.2E+1-t4*t42*2.0-t4*t74*2.0);
    fdrv(1, 8) = -t620*tdx-t627*tdy+tdz*(t104*2.0E+1+t161+t166+t389+t411+t412+t439+t465+t467-ry*t107*2.0+ry*t109*4.0-rz*t67*2.0-rz*t115*2.0+t6*t77*1.2E+1+t24*t72-rz*t9*t20*1.2E+1-rx*t32*tx*2.0);
    fdrv(1, 9) = -tdx*(t56*-5.0+t253+t398+t497+t499+t2*t18*2.0+t9*t21+t14*t32*4.0+t17*t37-t116*ty+t207*ty)-t593*tdy*tx-t585*tdz*tx;
    fdrv(1, 10) = -tdy*(t246+t253+t398+t403+t497+t501+t2*t18*4.0+t14*t32*2.0+t21*t28+t17*t37*4.0-t20*t52*2.0)-t580*tdx*ty-t585*tdz*ty;
    fdrv(1, 11) = -tdz*(t68*-5.0+t246+t403+t499+t501+t2*t18+t9*t21*4.0+t14*t32+t20*t32*3.0+t17*t44*tx+t17*t28*tz)-t580*tdx*tz-t593*tdy*tz; fdrv(1, 12) = t630;
    fdrv(1, 13) = tdx*(t136+t220+t226+t293+rx*t22+t20*t40*4.0+t14*t72*3.0+t14*t74*6.0+t17*t84+t76*tx*2.0+t141*tx)+tdz*(t142+t219+t225+t295+rz*t16+t14*t41*4.0+t20*t75*6.0+t17*t81+t20*t78*3.0+t76*tz*2.0+t135*tz)+t156*t551*tdy; fdrv(1, 14) = t628;
    fdrv(1, 15) = -t510*tx*(t32+t37-t64+t44*tx); fdrv(1, 16) = -t510*ty*(t32+t37-t64+t44*tx); fdrv(1, 17) = -t510*tz*(t32+t37-t64+t44*tx);
    fdrv(1, 18) = -rdx*t510*(t32+t37-t64+t44*tx)-t11*tx*(t32+t37-t64+t44*tx)*2.0-t12*tx*(t32+t37-t64+t44*tx)*2.0-t13*tx*(t32+t37-t64+t44*tx)*2.0-t11*t479*t510-t12*t479*t510-t13*t479*t510;
    fdrv(1, 19) = -rdy*t510*(t32+t37-t64+t44*tx)-t11*ty*(t32+t37-t64+t44*tx)*2.0-t12*ty*(t32+t37-t64+t44*tx)*2.0-t13*ty*(t32+t37-t64+t44*tx)*2.0-t11*t258*t510-t12*t258*t510-t13*t258*t510;
    fdrv(1, 20) = -rdz*t510*(t32+t37-t64+t44*tx)-t11*tz*(t32+t37-t64+t44*tx)*2.0-t12*tz*(t32+t37-t64+t44*tx)*2.0-t13*tz*(t32+t37-t64+t44*tx)*2.0-t11*t510*(t9-t27)-t12*t510*(t9-t27)-t13*t510*(t9-t27); fdrv(1, 24) = t547; fdrv(1, 25) = t156*t494*t510;
    fdrv(1, 26) = t549; fdrv(2, 0) = -t610*ty+t615*tz; fdrv(2, 1) = t610*tx+t617*tz; fdrv(2, 2) = -t615*tx-t617*ty;
    fdrv(2, 3) = t520+t527+rx*t63*5.0+rx*t69+rz*t249+rz*t496-t34*t39*3.0+t26*t79-t33*t72*2.0+t36*t75*4.0+t36*t78*4.0+t58*t72*6.0+t65*t74+t7*t138+t26*t122+t72*t146+t173*ty-t89*tz-rx*t2*t21*2.0-rx*t6*t21-rx*t14*t33*4.0-ry*t3*t21-rz*t20*t34*2.0-rz*t20*t51-ry*t32*tx*tz*3.0;
    fdrv(2, 4) = t523+t526+ry*t63+ry*t69*5.0+ry*t424+rz*t253+rz*t501+t28*t79+t37*t78*4.0+t58*t77+t73*t144+t165*ty+rx*t7*t18*4.0+rx*t21*t44-ry*t2*t21-ry*t6*t21*2.0-ry*t17*t33*3.0-ry*t17*t35*4.0-rz*t20*t32*2.0+rz*t117*ty*4.0+ry*t280*tz+t26*t74*ty-t4*t72*tz*3.0+rx*t14*t44*tz+rz*t20*t44*tx;
    fdrv(2, 5) = t260+t264-t488+t490+t492+t524+t525+rz*t63+rz*t69-t34*t42*2.0-t2*t76-t2*t79*4.0-t6*t79*4.0-t33*t75*2.0-t34*t74*3.0-t32*t77*3.0-t35*t78*2.0-t2*t119-t2*t120*3.0-t6*t122*3.0+t73*t143+rx*t18*t44+rx*t14*t145+t26*t78*tx+ry*t17*t44*tx+rx*t14*t26*tz+ry*t17*t28*tz;
    fdrv(2, 6) = -t621*tdy-t623*tdz+tdx*(t92*2.0E+1+t176+t180+t358+t407+t408+t441+t449+t454-rx*t59*2.0-rx*t111*2.0-rz*t116*2.0+rz*t117*1.2E+1+rz*t118*4.0+t28*t77-rx*t33*tx*1.2E+1-ry*t35*ty*2.0);
    fdrv(2, 7) = -t621*tdx-t625*tdz+tdy*(t103*2.0E+1+t165+t180+t390+t407+t408+t441+t454+t466-ry*t66*2.0-ry*t113*2.0-rz*t108*2.0+rz*t117*4.0+rz*t118*1.2E+1+t26*t74-rx*t33*tx*2.0-ry*t35*ty*1.2E+1);
    fdrv(2, 8) = -t623*tdx-t625*tdy-tdz*(t90*2.0-t92*2.0+t101*2.0-t103*2.0+t233+t234+t307+t314+t339+t381-t393-t397+rz*t108*1.2E+1+rz*t116*1.2E+1-t7*t40*2.0-t7*t72*2.0);
    fdrv(2, 9) = -tdx*(t63*-5.0+t254+t399+t500+t502+t2*t21*2.0+t6*t21+t14*t33*4.0+t17*t35+t193*tz+t20*t48*ty)-t584*tdy*tx-t592*tdz*tx;
    fdrv(2, 10) = -tdy*(t69*-5.0+t250+t401+t500+t503+t2*t21+t6*t21*2.0+t14*t33+t17*t33*3.0+t17*t35*4.0+t20*t46*tx)-t582*tdx*ty-t592*tdz*ty;
    fdrv(2, 11) = -tdz*(t250+t254+t399+t401+t502+t503+t2*t21*4.0+t6*t21*4.0+t14*t33*2.0+t17*t35*2.0-t17*t58*2.0)-t582*tdx*tz-t584*tdy*tz; fdrv(2, 12) = t629; fdrv(2, 13) = t628;
    fdrv(2, 14) = tdx*(t136+t220+t226+t293+rx*t19+t17*t42*4.0+t14*t72*6.0+t14*t74*3.0+t20*t82+t79*tx*2.0+t138*tx)+tdy*(t139+t214+t229+t294+ry*t16+t14*t43*4.0+t17*t73*6.0+t17*t77*3.0+t20*t80+t79*ty*2.0+t135*ty)+t155*t552*tdz;
    fdrv(2, 15) = -t510*tx*(t33+t35+t46*tx+t48*ty); fdrv(2, 16) = -t510*ty*(t33+t35+t46*tx+t48*ty); fdrv(2, 17) = -t510*tz*(t33+t35+t46*tx+t48*ty);
    fdrv(2, 18) = -rdx*t510*(t33+t35+t46*tx+t48*ty)-t11*tx*(t33+t35+t46*tx+t48*ty)*2.0-t12*tx*(t33+t35+t46*tx+t48*ty)*2.0-t13*tx*(t33+t35+t46*tx+t48*ty)*2.0-t11*t480*t510-t12*t480*t510-t13*t480*t510;
    fdrv(2, 19) = -rdy*t510*(t33+t35+t46*tx+t48*ty)-t11*ty*(t33+t35+t46*tx+t48*ty)*2.0-t12*ty*(t33+t35+t46*tx+t48*ty)*2.0-t13*ty*(t33+t35+t46*tx+t48*ty)*2.0-t11*t481*t510-t12*t481*t510-t13*t481*t510;
    fdrv(2, 20) = -rdz*t510*(t33+t35+t46*tx+t48*ty)-t11*tz*(t33+t35+t46*tx+t48*ty)*2.0-t12*tz*(t33+t35+t46*tx+t48*ty)*2.0-t13*tz*(t33+t35+t46*tx+t48*ty)*2.0-t11*t257*t510-t12*t257*t510-t13*t257*t510; fdrv(2, 24) = t548; fdrv(2, 25) = t549;
    fdrv(2, 26) = t155*t494*t510;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f32(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*ty, t3 = oy*tx, t4 = ox*tz, t5 = oz*tx, t6 = oy*tz, t7 = oz*ty, t8 = rdx*tx, t9 = rdy*ty, t10 = rdz*tz, t11 = rx*tx, t12 = ry*ty, t13 = rz*tz, t14 = tx*tx, t16 = ty*ty, t18 = tz*tz, t26 = rx*ty*tz, t27 = ry*tx*tz, t28 = rz*tx*ty;
    T t20 = t11*4.0, t21 = t12*4.0, t22 = t13*4.0, t23 = t11*ty, t24 = t11*tz, t25 = t12*tx, t29 = t12*tz, t30 = t13*tx, t31 = t13*ty, t32 = -t3, t33 = -t5, t34 = -t7, t35 = ox*t14, t36 = t2*ty, t37 = t3*tx, t38 = t4*tz, t39 = oy*t16, t40 = t5*tx;
    T t41 = t6*tz, t42 = t7*ty, t43 = oz*t18, t44 = t11*t14, t45 = rx*t16, t46 = ry*t14, t47 = rx*t18, t48 = rz*t14, t49 = t12*t16, t50 = ry*t18, t51 = rz*t16, t52 = t13*t18, t53 = t2*tx*4.0, t54 = t4*tx*4.0, t55 = t3*ty*4.0, t56 = t6*ty*4.0;
    T t57 = t5*tz*4.0, t58 = t7*tz*4.0, t86 = t11*t16, t87 = t11*t18, t88 = t12*t14, t95 = t12*t18, t96 = t13*t14, t97 = t13*t16, t100 = rx*t2*tz*2.0, t101 = rx*t3*tz*4.0, t102 = rx*t5*ty*4.0, t103 = ry*t3*tz*2.0, t104 = ry*t2*tz*4.0;
    T t108 = rz*t5*ty*2.0, t119 = t11*tx*5.0, t121 = t12*ty*5.0, t123 = t13*tz*5.0, t131 = -t27, t132 = -t28, t133 = t14+t16, t134 = t14+t18, t135 = t16+t18, t136 = ox*t11*tx*3.0, t141 = oy*t12*ty*3.0, t146 = oz*t13*tz*3.0, t160 = t2*t11*1.0E+1;
    T t161 = t4*t11*1.0E+1, t163 = t3*t12*1.0E+1, t165 = rx*t5*tz*-4.0, t167 = t6*t12*1.0E+1, t168 = ry*t7*tz*-4.0, t169 = t5*t13*1.0E+1, t170 = t7*t13*1.0E+1, t176 = t2*t26, t189 = t2*t27*4.0, t190 = t3*t26*4.0, t192 = t5*t26*4.0, t195 = t11+t12;
    T t196 = t11+t13, t197 = t12+t13, t208 = oz*t13*tz*-5.0, t257 = t2*t11*tx*-5.0, t260 = t4*t11*tx*-5.0, t263 = t3*t12*ty*-5.0, t270 = t6*t12*ty*-5.0, t273 = t5*t13*tz*-5.0, t274 = t7*t13*tz*-5.0, t284 = t8+t9+t10, t59 = t20*ty, t60 = t20*tz;
    T t61 = t21*tx, t62 = t21*tz, t63 = t22*tx, t64 = t22*ty, t65 = t11*t35, t66 = rx*t36, t67 = t2*t45, t68 = t11*t37, t69 = rx*t38, t70 = ry*t37, t71 = t4*t47, t72 = t12*t36, t74 = t11*t40, t75 = t12*t39, t76 = ry*t41, t77 = rz*t40;
    T t78 = t13*t38, t79 = t6*t50, t80 = t12*t42, t82 = rz*t42, t83 = t13*t41, t85 = t13*t43, t89 = t47*ty, t90 = t45*tz, t91 = t50*tx, t92 = t46*tz, t93 = t51*tx, t94 = t48*ty, t98 = ry*t53, t99 = rx*t55, t105 = rz*t54, t107 = t5*t21;
    T t109 = t2*t22, t110 = t3*t22, t111 = rz*t56, t113 = t36*5.0, t114 = t37*5.0, t115 = t38*5.0, t116 = t40*5.0, t117 = t41*5.0, t118 = t42*5.0, t120 = t44*6.0, t122 = t49*6.0, t124 = t52*6.0, t125 = -t53, t126 = -t54, t127 = -t55, t128 = -t56;
    T t129 = -t57, t130 = -t58, t137 = ox*t119, t142 = oy*t121, t148 = t86*2.0, t149 = t86*5.0, t150 = t87*2.0, t151 = t88*2.0, t152 = t87*5.0, t153 = t88*5.0, t154 = t95*2.0, t155 = t96*2.0, t156 = t95*5.0, t157 = t96*5.0, t158 = t97*2.0;
    T t159 = t97*5.0, t162 = -t100, t164 = -t103, t166 = -t108, t171 = t2*t23, t172 = t4*t24, t173 = t2*t46, t174 = t3*t45, t175 = t2*t47, t177 = t3*t25, t178 = t4*t48, t181 = t5*t47, t182 = t6*t29, t185 = t6*t51, t186 = t7*t50, t187 = t5*t30;
    T t188 = t7*t31, t200 = t32*t46, t202 = rz*t33*tx, t204 = t33*t48, t205 = rz*t34*ty, t206 = t34*t51, t207 = -t146, t223 = t3*t47*2.0, t224 = t5*t45*2.0, t225 = t3*t47*5.0, t228 = t5*t45*5.0, t229 = t2*t50*2.0, t232 = t5*t25*2.0;
    T t235 = t2*t50*5.0, t236 = t5*t25*5.0, t237 = t2*t31*2.0, t238 = t3*t30*2.0, t239 = t2*t31*5.0, t242 = t3*t30*5.0, t251 = -t189, t252 = -t190, t253 = t2*t30*-4.0, t254 = -t192, t255 = t3*t31*-4.0, t256 = t5*t29*-4.0, t258 = t2*t25*-4.0;
    T t259 = t3*t23*-4.0, t262 = -t176, t264 = t32*t50, t265 = t27*t32, t266 = t4*t30*-4.0, t267 = t5*t24*-4.0, t268 = t33*t51, t269 = t28*t33, t271 = t6*t31*-4.0, t272 = t7*t29*-4.0, t275 = rx*t133, t276 = rx*t134, t277 = ry*t133, t278 = ry*t135;
    T t279 = rz*t134, t280 = rz*t135, t281 = t2+t32, t282 = t4+t33, t283 = t6+t34, t285 = t195*tx*tz, t286 = t196*tx*ty, t287 = t195*ty*tz, t288 = t197*tx*ty, t289 = t196*ty*tz, t290 = t197*tx*tz, t297 = t18+t133, t298 = t26+t131, t299 = t26+t132;
    T t300 = t27+t132, t314 = t133*t195, t315 = t134*t196, t316 = t135*t197, t138 = t68*6.0, t139 = t72*6.0, t140 = t74*6.0, t143 = t78*6.0, t144 = t80*6.0, t145 = t83*6.0, t198 = -t67, t199 = -t71, t201 = -t76, t203 = -t79, t209 = -t89;
    T t210 = -t90, t211 = -t91, t212 = -t92, t213 = -t93, t214 = -t94, t215 = t171*5.0, t217 = t173*2.0, t218 = t174*2.0, t221 = t172*5.0, t227 = t177*5.0, t230 = t178*2.0, t231 = t181*2.0, t240 = t182*5.0, t243 = t185*2.0, t244 = t186*2.0;
    T t248 = t187*5.0, t250 = t188*5.0, t261 = -t175, t291 = t276*ty, t292 = t275*tz, t293 = t278*tx, t294 = t277*tz, t295 = t280*tx, t296 = t279*ty, t301 = t297*t297, t302 = t298*tx, t303 = t299*tx, t304 = t298*ty, t305 = t300*ty, t306 = t299*tz;
    T t307 = t300*tz, t308 = t25+t276, t309 = t30+t275, t310 = t23+t278, t311 = t31+t277, t312 = t24+t280, t313 = t29+t279, t341 = t35+t38+t113+t127, t342 = t35+t36+t115+t129, t343 = t39+t41+t114+t125, t344 = t37+t39+t117+t130;
    T t345 = t42+t43+t116+t126, t346 = t40+t43+t118+t128, t350 = t45+t47+t61+t63+t119, t351 = t46+t50+t59+t64+t121, t352 = t48+t51+t60+t62+t123, t359 = t44+t49+t86+t88+t124+t152+t155+t156+t158, t360 = t44+t52+t87+t96+t122+t149+t151+t154+t159;
    T t361 = t49+t52+t95+t97+t120+t148+t150+t153+t157, t317 = -t304, t318 = -t306, t319 = -t307, t320 = t308*tx, t321 = t309*tx, t322 = t310*tx, t323 = t308*ty, t324 = t310*ty, t325 = t312*tx, t326 = t309*tz, t327 = t311*ty, t328 = t313*ty;
    T t329 = t311*tz, t330 = t312*tz, t331 = t313*tz, t332 = ox*t8*t301*2.0, t333 = rdx*t3*t301*2.0, t334 = rdy*t2*t301*2.0, t335 = rdx*t5*t301*2.0, t336 = oy*t9*t301*2.0, t337 = rdz*t4*t301*2.0, t338 = rdy*t7*t301*2.0, t339 = rdz*t6*t301*2.0;
    T t340 = oz*t10*t301*2.0, t347 = t284*t301*tx*2.0, t348 = t284*t301*ty*2.0, t349 = t284*t301*tz*2.0, t353 = t297*t350*tdx*ty*2.0, t354 = t297*t350*tdx*tz*2.0, t355 = t297*t351*tdy*tx*2.0, t356 = t297*t351*tdy*tz*2.0;
    T t357 = t297*t352*tdz*tx*2.0, t358 = t297*t352*tdz*ty*2.0, t404 = t297*t361*tdx*2.0, t405 = t297*t360*tdy*2.0, t406 = t297*t359*tdz*2.0, t407 = t65+t78+t139+t172+t178+t200+t215+t217+t229+t239+t255+t259+t263+t264;
    T t408 = t75+t83+t138+t182+t185+t198+t218+t223+t227+t242+t253+t257+t258+t261, t409 = t65+t72+t143+t171+t173+t204+t221+t230+t235+t237+t256+t267+t268+t273, t410 = t80+t85+t140+t186+t188+t199+t224+t231+t236+t248+t251+t260+t262+t266;
    T t411 = t68+t75+t145+t174+t177+t206+t225+t238+t240+t243+t254+t269+t272+t274, t412 = t74+t85+t144+t181+t187+t203+t228+t232+t244+t250+t252+t265+t270+t271, t362 = t214+t287+t296+t302+t329, t363 = t210+t290+t292+t305+t325;
    T t364 = t212+t289+t294+t303+t328, t365 = t211+t286+t293+t318+t323, t366 = t213+t285+t295+t317+t326, t367 = t209+t288+t291+t319+t322, t368 = t96+t97+t314+t321+t327, t369 = t88+t95+t315+t320+t331, t370 = t86+t87+t316+t324+t330;
    T t371 = t363*tx*ty, t372 = t364*tx*ty, t373 = t362*tx*tz, t374 = t363*tx*tz, t375 = t362*ty*tz, t376 = t364*ty*tz, t377 = t365*tx*ty, t378 = t367*tx*ty, t379 = t366*tx*tz, t380 = t367*tx*tz, t381 = t365*ty*tz, t382 = t366*ty*tz;
    T t386 = t369*tx*ty, t387 = t368*tx*tz, t388 = t370*tx*ty, t389 = t368*ty*tz, t390 = t370*tx*tz, t391 = t369*ty*tz, t395 = t133*t362, t396 = t134*t364, t397 = t135*t363, t398 = t133*t366, t399 = t134*t365, t400 = t135*t367, t401 = t133*t368;
    T t402 = t134*t369, t403 = t135*t370, t383 = -t371, t384 = -t372, t385 = -t373, t392 = -t380, t393 = -t381, t394 = -t382, t413 = t374+t376+t401, t414 = t375+t378+t402, t417 = t377+t379+t403, t415 = t383+t389+t396, t416 = t384+t387+t397;
    T t418 = t391+t392+t395, t419 = t385+t386+t400, t420 = t390+t393+t398, t421 = t388+t394+t399;
    
    fdrv(0, 0) = t416*ty-t419*tz; fdrv(0, 1) = -t416*tx-t417*tz; fdrv(0, 2) = t419*tx+t417*ty; fdrv(0, 3) = t283*t297*t350*-2.0; fdrv(0, 4) = t297*t412*2.0;
    fdrv(0, 5) = t297*t411*-2.0;
    fdrv(0, 6) = t297*tdy*(t107+t164+rx*t43+rx*t118+t5*t11*3.0+t5*t13*2.0-rx*t6*ty*4.0)*2.0-t297*tdz*(t110+t166+rx*t39+rx*t117+t3*t11*3.0+t3*t12*2.0-rx*t7*tz*4.0)*2.0+t412*tdy*tx*4.0-t411*tdz*tx*4.0-t283*t350*tdx*tx*4.0-t283*t297*tdx*(t11*1.0E+1+t21+t22)*2.0;
    fdrv(0, 7) = t297*tdz*(t70+t76*5.0-t82*3.0+t111+t141+t165+t202+t208+rx*t3*ty*2.0-ry*t7*tz*8.0)*-2.0+t297*tdy*(-t101-t167+t170+ry*t40*2.0+ry*t43*2.0-t6*t13*4.0+t7*t12*1.8E+1+rx*t5*ty*1.0E+1)*2.0+t412*tdy*ty*4.0-t411*tdz*ty*4.0+oz*t297*t350*tdx*2.0-t283*t350*tdx*ty*4.0-t283*t297*tdx*(rx*ty*2.0+ry*tx*4.0)*2.0;
    fdrv(0, 8) = t297*tdy*(t70+t76*3.0-t82*5.0+t99+t142+t168+t202+t207-rx*t5*tz*2.0+rz*t6*ty*8.0)*-2.0-t297*tdz*(-t102+t167-t170+rz*t37*2.0+rz*t39*2.0+t6*t13*1.8E+1-t7*t12*4.0+rx*t3*tz*1.0E+1)*2.0+t412*tdy*tz*4.0-t411*tdz*tz*4.0-oy*t297*t350*tdx*2.0-t283*t350*tdx*tz*4.0-t283*t297*tdx*(rx*tz*2.0+rz*tx*4.0)*2.0;
    fdrv(0, 9) = t297*t346*tdy*tx*2.0-t297*t344*tdz*tx*2.0-t283*t297*tdx*(t14*3.0+t133+t134)*2.0; fdrv(0, 10) = t297*tdy*(t6*t16*5.0-t7*t16*6.0+t6*t18-t7*t18*2.0-t40*ty*2.0+t37*tz)*-2.0-t297*t344*tdz*ty*2.0-t283*t297*tdx*tx*ty*8.0;
    fdrv(0, 11) = t297*tdz*(t6*t16*2.0+t6*t18*6.0-t7*t18*5.0+t16*t34+t37*tz*2.0+t33*tx*ty)*-2.0+t297*t346*tdy*tz*2.0-t283*t297*tdx*tx*tz*8.0; fdrv(0, 13) = -t354-t356-t406; fdrv(0, 14) = t353+t358+t405; fdrv(0, 15) = t283*t301*tx*-2.0;
    fdrv(0, 16) = t283*t301*ty*-2.0; fdrv(0, 17) = t283*t301*tz*-2.0; fdrv(0, 18) = t283*t297*(rdx*t16+rdx*t18+t8*tx*5.0+t9*tx*4.0+t10*tx*4.0)*-2.0;
    fdrv(0, 19) = t335+t338+t340-rdy*t283*t301*2.0-t8*t283*t297*ty*8.0-t9*t283*t297*ty*8.0-t10*t283*t297*ty*8.0; fdrv(0, 20) = -t333-t336-t339-rdz*t283*t301*2.0-t8*t283*t297*tz*8.0-t9*t283*t297*tz*8.0-t10*t283*t297*tz*8.0; fdrv(0, 25) = -t349;
    fdrv(0, 26) = t348; fdrv(1, 0) = t415*ty+t414*tz; fdrv(1, 1) = -t415*tx+t421*tz; fdrv(1, 2) = -t414*tx-t421*ty; fdrv(1, 3) = t297*t410*-2.0; fdrv(1, 4) = t282*t297*t351*2.0; fdrv(1, 5) = t297*t409*2.0;
    fdrv(1, 6) = t297*tdx*(-t104-t161+t169+rx*t42*2.0+rx*t43*2.0+t5*t11*1.8E+1-t4*t13*4.0+t5*t12*1.0E+1)*-2.0+t297*tdz*(t66+t69*5.0-t77*3.0+t105+t136+t168+t205+t208+ry*t2*tx*2.0-rx*t5*tz*8.0)*2.0-t410*tdx*tx*4.0+t409*tdz*tx*4.0-oz*t297*t351*tdy*2.0+t282*t351*tdy*tx*4.0+t282*t297*tdy*(rx*ty*4.0+ry*tx*2.0)*2.0;
    fdrv(1, 7) = t297*tdx*(t102+t162+ry*t43+ry*t116+t7*t12*3.0+t7*t13*2.0-ry*t4*tx*4.0)*-2.0+t297*tdz*(t109+t166+ry*t35+ry*t115+t2*t11*2.0+t2*t12*3.0-ry*t5*tz*4.0)*2.0-t410*tdx*ty*4.0+t409*tdz*ty*4.0+t282*t351*tdy*ty*4.0+t282*t297*tdy*(t12*1.0E+1+t20+t22)*2.0;
    fdrv(1, 8) = t297*tdz*(t161-t169+rz*t35*2.0+rz*t36*2.0-t5*t11*4.0+t4*t13*1.8E+1-t5*t12*4.0+ry*t2*tz*1.0E+1)*2.0+t297*tdx*(t66+t69*3.0-t77*5.0+t98+t137+t165+t205+t207+rz*t4*tx*8.0-ry*t7*tz*2.0)*2.0-t410*tdx*tz*4.0+t409*tdz*tz*4.0+ox*t297*t351*tdy*2.0+t282*t351*tdy*tz*4.0+t282*t297*tdy*(ry*tz*2.0+rz*ty*4.0)*2.0;
    fdrv(1, 9) = t297*tdx*(t4*t14*5.0-t5*t14*6.0-t5*t16*2.0+t4*t18-t5*t18*2.0+t36*tz)*2.0+t297*t342*tdz*tx*2.0+t282*t297*tdy*tx*ty*8.0; fdrv(1, 10) = t297*t345*tdx*ty*-2.0+t297*t342*tdz*ty*2.0+t282*t297*tdy*(t16*3.0+t133+t135)*2.0;
    fdrv(1, 11) = t297*tdz*(t4*t14*2.0+t4*t18*6.0-t5*t18*5.0+t14*t33+t16*t33+t36*tz*2.0)*2.0-t297*t345*tdx*tz*2.0+t282*t297*tdy*ty*tz*8.0; fdrv(1, 12) = t354+t356+t406; fdrv(1, 14) = -t355-t357-t404; fdrv(1, 15) = t282*t301*tx*2.0;
    fdrv(1, 16) = t282*t301*ty*2.0; fdrv(1, 17) = t282*t301*tz*2.0; fdrv(1, 18) = -t335-t338-t340+rdx*t282*t301*2.0+t8*t282*t297*tx*8.0+t9*t282*t297*tx*8.0+t10*t282*t297*tx*8.0;
    fdrv(1, 19) = t282*t297*(rdy*t14+rdy*t18+t8*ty*4.0+t9*ty*5.0+t10*ty*4.0)*2.0; fdrv(1, 20) = t332+t334+t337+rdz*t282*t301*2.0+t8*t282*t297*tz*8.0+t9*t282*t297*tz*8.0+t10*t282*t297*tz*8.0; fdrv(1, 24) = t349; fdrv(1, 26) = -t347;
    fdrv(2, 0) = -t413*ty-t418*tz; fdrv(2, 1) = t413*tx+t420*tz; fdrv(2, 2) = t418*tx-t420*ty; fdrv(2, 3) = t297*t408*2.0; fdrv(2, 4) = t297*t407*-2.0; fdrv(2, 5) = t281*t297*t352*-2.0;
    fdrv(2, 6) = t297*tdy*(t66*5.0+t69-t70*3.0+t98+t136+t201-oy*t12*ty*5.0-rx*t3*ty*8.0+rz*t4*tx*2.0-rz*t6*ty*4.0)*-2.0+t297*tdx*(-t160+t163+rx*t39*2.0+rx*t41*2.0-t2*t12*4.0+t3*t11*1.8E+1-t2*t13*4.0+t3*t13*1.0E+1)*2.0+t408*tdx*tx*4.0-t407*tdy*tx*4.0+oy*t297*t352*tdz*2.0-t281*t352*tdz*tx*4.0-t281*t297*tdz*(rx*tz*4.0+rz*tx*2.0)*2.0;
    fdrv(2, 7) = t297*tdx*(t66*3.0+t69-t70*5.0+t105+t137-t141+t201+ry*t2*tx*8.0-rx*t3*ty*4.0-rz*t6*ty*2.0)*-2.0-t297*tdy*(t160-t163+ry*t35*2.0+ry*t38*2.0+t2*t12*1.8E+1-t3*t11*4.0+t2*t13*1.0E+1-t3*t13*4.0)*2.0+t408*tdx*ty*4.0-t407*tdy*ty*4.0-ox*t297*t352*tdz*2.0-t281*t352*tdz*ty*4.0-t281*t297*tdz*(ry*tz*4.0+rz*ty*2.0)*2.0;
    fdrv(2, 8) = t297*tdx*(t101+t162+rz*t39+rz*t114+t6*t12*2.0+t6*t13*3.0-rz*t2*tx*4.0)*2.0-t297*tdy*(t104+t164+rz*t35+rz*t113+t4*t11*2.0+t4*t13*3.0-rz*t3*ty*4.0)*2.0+t408*tdx*tz*4.0-t407*tdy*tz*4.0-t281*t352*tdz*tz*4.0-t281*t297*tdz*(t13*1.0E+1+t20+t21)*2.0;
    fdrv(2, 9) = t297*tdx*(t2*t14*5.0-t3*t14*6.0+t2*t16-t3*t16*2.0+t2*t18-t3*t18*2.0)*-2.0-t297*t341*tdy*tx*2.0-t281*t297*tdz*tx*tz*8.0;
    fdrv(2, 10) = t297*tdy*(t2*t14*2.0+t2*t16*6.0-t3*t16*5.0+t2*t18*2.0+t14*t32+t18*t32)*-2.0+t297*t343*tdx*ty*2.0-t281*t297*tdz*ty*tz*8.0; fdrv(2, 11) = t297*t343*tdx*tz*2.0-t297*t341*tdy*tz*2.0-t281*t297*tdz*(t18*4.0+t297)*2.0;
    fdrv(2, 12) = -t353-t358-t405; fdrv(2, 13) = t355+t357+t404; fdrv(2, 15) = t281*t301*tx*-2.0; fdrv(2, 16) = t281*t301*ty*-2.0; fdrv(2, 17) = t281*t301*tz*-2.0;
    fdrv(2, 18) = t333+t336+t339-rdx*t281*t301*2.0-t8*t281*t297*tx*8.0-t9*t281*t297*tx*8.0-t10*t281*t297*tx*8.0; fdrv(2, 19) = -t332-t334-t337-rdy*t281*t301*2.0-t8*t281*t297*ty*8.0-t9*t281*t297*ty*8.0-t10*t281*t297*ty*8.0;
    fdrv(2, 20) = t281*t297*(rdz*t14+rdz*t16+t8*tz*4.0+t9*tz*4.0+t10*tz*5.0)*-2.0; fdrv(2, 24) = -t348; fdrv(2, 25) = t347;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f33(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*ty, t3 = oy*tx, t4 = ox*tz, t5 = oz*tx, t6 = oy*tz, t7 = oz*ty, t8 = rdx*tx, t9 = rdy*ty, t10 = rdz*tz, t11 = tx*tx, t12 = tx*tx*tx, t13 = ty*ty, t14 = ty*ty*ty, t15 = tz*tz, t16 = tz*tz*tz, t17 = rx*tx*2.0, t18 = rx*ty*2.0;
    T t19 = ry*tx*2.0, t20 = rx*tz*2.0, t21 = ry*ty*2.0, t22 = rz*tx*2.0, t23 = ry*tz*2.0, t24 = rz*ty*2.0, t25 = rz*tz*2.0, t26 = rx*tx*ty, t27 = rx*tx*tz, t28 = ry*tx*ty, t29 = rx*ty*tz, t30 = ry*tx*tz, t31 = rz*tx*ty, t32 = ry*ty*tz;
    T t33 = rz*tx*tz, t34 = rz*ty*tz, t35 = -t3, t36 = -t5, t37 = -t7, t38 = ox*t11, t39 = t2*ty, t40 = t3*tx, t41 = t4*tz, t42 = oy*t13, t43 = t5*tx, t44 = t6*tz, t45 = t7*ty, t46 = oz*t15, t47 = rx*t12, t48 = rx*t13, t49 = ry*t11, t50 = rx*t15;
    T t51 = rz*t11, t52 = ry*t14, t53 = ry*t15, t54 = rz*t13, t55 = rz*t16, t56 = t2*tx*2.0, t57 = t4*tx*2.0, t58 = t3*ty*2.0, t59 = t6*ty*2.0, t60 = t5*tz*2.0, t61 = t7*tz*2.0, t62 = t17*ty, t63 = t17*tz, t64 = t19*ty, t65 = t21*tz, t66 = t22*tz;
    T t67 = t24*tz, t83 = rx*t2*tx*6.0, t84 = t2*t19, t85 = t3*t18, t86 = ry*t2*tx*4.0, t87 = rx*t3*ty*4.0, t88 = rx*t4*tx*6.0, t90 = ry*t3*ty*6.0, t91 = t4*t22, t94 = rz*t4*tx*4.0, t95 = rx*t5*tz*4.0, t97 = ry*t6*ty*6.0, t100 = rz*t6*ty*4.0;
    T t101 = ry*t7*tz*4.0, t102 = rz*t5*tz*6.0, t103 = rz*t7*tz*6.0, t110 = rx*t11*3.0, t112 = ry*t13*3.0, t114 = rz*t15*3.0, t122 = -t30, t123 = -t31, t124 = t11+t13, t125 = t11+t15, t126 = t13+t15, t136 = t13*t17, t138 = t15*t17, t139 = t11*t21;
    T t142 = t15*t21, t143 = t11*t25, t146 = t13*t25, t149 = rx*t2*tz*-2.0, t150 = ry*t3*tz*-2.0, t151 = rx*t5*tz*-2.0, t153 = rz*t5*ty*-2.0, t154 = rz*t6*ty*-2.0, t155 = ry*t7*tz*-2.0, t174 = t18+t19, t175 = t20+t22, t176 = t23+t24;
    T t177 = t8+t9+t10, t68 = ox*t47, t69 = rx*t39, t70 = rx*t41, t71 = ry*t40, t72 = oy*t52, t73 = ry*t44, t74 = rz*t43, t75 = rz*t45, t76 = oz*t55, t77 = t48*tx, t78 = t50*tx, t79 = t49*ty, t80 = t53*ty, t81 = t51*tz, t82 = t54*tz;
    T t104 = t39*3.0, t105 = t40*3.0, t106 = t41*3.0, t107 = t43*3.0, t108 = t44*3.0, t109 = t45*3.0, t111 = t47*4.0, t113 = t52*4.0, t115 = t55*4.0, t116 = -t56, t117 = -t57, t118 = -t58, t119 = -t59, t120 = -t60, t121 = -t61, t127 = rx*t38*3.0;
    T t131 = ry*t42*3.0, t135 = rz*t46*3.0, t148 = -t87, t152 = -t95, t156 = -t101, t160 = rz*t36*tx, t162 = rz*t37*ty, t165 = rx*t124, t166 = rx*t125, t167 = ry*t124, t168 = ry*t126, t169 = rz*t125, t170 = rz*t126, t171 = t2+t35, t172 = t4+t36;
    T t173 = t6+t37, t178 = t15+t124, t179 = t29+t122, t180 = t29+t123, t181 = t30+t123, t206 = t48+t50+t64+t66+t110, t207 = t49+t53+t62+t67+t112, t208 = t51+t54+t63+t65+t114, t128 = t69*3.0, t129 = t70*3.0, t130 = t71*3.0, t132 = t73*3.0;
    T t133 = t74*3.0, t134 = t75*3.0, t137 = t77*3.0, t140 = t78*3.0, t141 = t79*3.0, t144 = t80*3.0, t145 = t81*3.0, t147 = t82*3.0, t158 = -t131, t159 = -t73, t164 = -t135, t182 = ox*t8*t178, t183 = rdx*t3*t178, t184 = rdy*t2*t178;
    T t185 = rdx*t5*t178, t186 = oy*t9*t178, t187 = rdz*t4*t178, t188 = rdy*t7*t178, t189 = rdz*t6*t178, t190 = oz*t10*t178, t191 = t28+t166, t192 = t33+t165, t193 = t26+t168, t194 = t34+t167, t195 = t27+t170, t196 = t32+t169, t197 = t177*t178*tx;
    T t198 = t177*t178*ty, t199 = t177*t178*tz, t200 = t38+t41+t104+t118, t201 = t38+t39+t106+t120, t202 = t42+t44+t105+t116, t203 = t40+t42+t108+t121, t204 = t45+t46+t107+t117, t205 = t43+t46+t109+t119, t209 = t206*tdx*ty, t210 = t206*tdx*tz;
    T t211 = t207*tdy*tx, t212 = t207*tdy*tz, t213 = t208*tdz*tx, t214 = t208*tdz*ty, t157 = -t130, t161 = -t133, t163 = -t134, t215 = t47+t52+t77+t79+t115+t140+t143+t144+t146, t216 = t47+t55+t78+t81+t113+t137+t139+t142+t147;
    T t217 = t52+t55+t80+t82+t111+t136+t138+t141+t145, t218 = t217*tdx, t219 = t216*tdy, t220 = t215*tdz, t221 = t70+t86+t91+t127+t128+t148+t154+t157+t158+t159, t222 = t69+t84+t94+t127+t129+t152+t155+t161+t162+t164;
    T t223 = t71+t85+t100+t131+t132+t151+t156+t160+t163+t164;
    
    fdrv(0, 0) = t126*t181+t195*tx*ty-t193*tx*tz; fdrv(0, 1) = -t125*t195-t181*tx*ty-t193*ty*tz; fdrv(0, 2) = t124*t193-t181*tx*tz+t195*ty*tz; fdrv(0, 3) = -t173*t206;
    fdrv(0, 4) = t76-t3*t29*2.0+t5*t33-t6*t34*2.0+t7*t34*3.0+t5*t48*3.0+t5*t50-t6*t53+t7*t53*2.0+t30*t35+t5*t64+rx*t5*t11-ry*t6*t13*3.0+ry*t7*t13*4.0;
    fdrv(0, 5) = -t72-t3*t33*2.0+t5*t31-t6*t32*3.0-t3*t50*3.0-t6*t54*2.0+t7*t54+t28*t35+t7*t65+t35*t48+t7*t114+rx*t11*t35-rz*t6*t15*4.0+t5*t18*tz;
    fdrv(0, 6) = tdy*(t150+rx*t46+rx*t107+rx*t109+t5*t25-rx*t6*ty*2.0+ry*t5*ty*4.0)-tdz*(t153+rx*t42+rx*t105+rx*t108+t3*t21-rx*t7*tz*2.0+rz*t3*tz*4.0)-t173*tdx*(t21+t25+rx*tx*6.0);
    fdrv(0, 7) = tdy*(-t97+t103+ry*t45*1.2E+1+ry*t46*2.0-rz*t44*2.0+t5*t19+rx*t5*ty*6.0-rx*t3*tz*2.0)-t223*tdz+oz*t206*tdx-t173*t174*tdx;
    fdrv(0, 8) = -tdz*(t97-t103-ry*t45*2.0+rz*t42*2.0+rz*t44*1.2E+1+t3*t22-rx*t5*ty*2.0+rx*t3*tz*6.0)-t223*tdy-oy*t206*tdx-t173*t175*tdx; fdrv(0, 9) = -t173*tdx*(t11+t124+t125)+t205*tdy*tx-t203*tdz*tx;
    fdrv(0, 10) = -tdy*(t6*t13*3.0-t7*t13*4.0+t6*t15-t7*t15*2.0-t43*ty*2.0+t40*tz)-t203*tdz*ty-t173*tdx*tx*ty*2.0; fdrv(0, 11) = -tdz*(t6*t13*2.0+t6*t15*4.0-t7*t15*3.0+t13*t37+t40*tz*2.0+t36*tx*ty)+t205*tdy*tz-t173*tdx*tx*tz*2.0;
    fdrv(0, 13) = -t210-t212-t220; fdrv(0, 14) = t209+t214+t219; fdrv(0, 15) = -t173*t178*tx; fdrv(0, 16) = -t173*t178*ty; fdrv(0, 17) = -t173*t178*tz; fdrv(0, 18) = -t173*(rdx*t13+rdx*t15+t8*tx*3.0+t9*tx*2.0+t10*tx*2.0);
    fdrv(0, 19) = t185+t188+t190-rdy*t173*t178-t8*t173*ty*2.0-t9*t173*ty*2.0-t10*t173*ty*2.0; fdrv(0, 20) = -t186-t189+rdx*t35*t178-rdz*t173*t178-t8*t173*tz*2.0-t9*t173*tz*2.0-t10*t173*tz*2.0; fdrv(0, 25) = -t199; fdrv(0, 26) = t198;
    fdrv(1, 0) = t126*t196+t180*tx*ty+t191*tx*tz; fdrv(1, 1) = -t125*t180-t196*tx*ty+t191*ty*tz; fdrv(1, 2) = -t124*t191-t196*tx*tz+t180*ty*tz;
    fdrv(1, 3) = -t76+t2*t29-t5*t28*3.0-t5*t33*3.0-t5*t48*2.0+t4*t50-t5*t50*2.0+t22*t41+t34*t37+t37*t53+t4*t110+t84*tz-rx*t5*t11*4.0+ry*t13*t37; fdrv(1, 4) = t172*t207;
    fdrv(1, 5) = t68+t2*t26+t4*t27*3.0-t5*t27*2.0-t5*t32*2.0+t2*t49+t2*t53*3.0+t4*t51*2.0+t2*t67+t36*t51+t36*t54+ry*t2*t13+rz*t4*t15*4.0-rz*t5*t15*3.0;
    fdrv(1, 6) = -tdx*(-t88+t102+rx*t43*1.2E+1+rx*t46*2.0-rz*t41*2.0+t7*t18+ry*t5*ty*6.0-ry*t2*tz*2.0)+t222*tdz-oz*t207*tdy+t172*t174*tdy;
    fdrv(1, 7) = -tdx*(t149+ry*t46+ry*t107+ry*t109+t7*t25-ry*t4*tx*2.0+rx*t5*ty*4.0)+tdz*(t153+ry*t38+ry*t104+ry*t106+t2*t17-ry*t5*tz*2.0+rz*t2*tz*4.0)+t172*tdy*(t17+t25+ry*ty*6.0);
    fdrv(1, 8) = tdz*(t88-t102-rx*t43*2.0+rz*t38*2.0+rz*t41*1.2E+1+t2*t24-ry*t5*ty*2.0+ry*t2*tz*6.0)+t222*tdx+ox*t207*tdy+t172*t176*tdy; fdrv(1, 9) = tdx*(t4*t11*3.0-t5*t11*4.0-t5*t13*2.0+t4*t15-t5*t15*2.0+t39*tz)+t201*tdz*tx+t172*tdy*tx*ty*2.0;
    fdrv(1, 10) = t172*tdy*(t13+t124+t126)-t204*tdx*ty+t201*tdz*ty; fdrv(1, 11) = tdz*(t4*t11*2.0+t4*t15*4.0-t5*t15*3.0+t11*t36+t13*t36+t39*tz*2.0)-t204*tdx*tz+t172*tdy*ty*tz*2.0; fdrv(1, 12) = t210+t212+t220; fdrv(1, 14) = -t211-t213-t218;
    fdrv(1, 15) = t172*t178*tx; fdrv(1, 16) = t172*t178*ty; fdrv(1, 17) = t172*t178*tz; fdrv(1, 18) = -t190+rdx*t36*t178+rdx*t172*t178+rdy*t37*t178+t8*t172*tx*2.0+t9*t172*tx*2.0+t10*t172*tx*2.0;
    fdrv(1, 19) = t172*(rdy*t11+rdy*t15+t8*ty*2.0+t9*ty*3.0+t10*ty*2.0); fdrv(1, 20) = t182+t184+t187+rdz*t172*t178+t8*t172*tz*2.0+t9*t172*tz*2.0+t10*t172*tz*2.0; fdrv(1, 24) = t199; fdrv(1, 26) = -t197; fdrv(2, 0) = -t126*t194-t192*tx*ty-t179*tx*tz;
    fdrv(2, 1) = t125*t192+t194*tx*ty-t179*ty*tz; fdrv(2, 2) = t124*t179+t194*tx*tz-t192*ty*tz; fdrv(2, 3) = t72-t2*t28*2.0+t3*t28*3.0-t2*t33*2.0+t3*t33*3.0+t6*t32-t2*t48+t3*t48*2.0-t2*t50+t3*t50*2.0+t6*t54-rx*t2*t11*3.0+rx*t3*t11*4.0+rz*t6*t15;
    fdrv(2, 4) = -t68-t2*t26*3.0-t4*t27-t2*t34*3.0-t2*t49*2.0+t3*t49-t2*t53*2.0-t4*t51+t3*t53+t3*t62+t3*t67+t3*t112-ry*t2*t13*4.0-rz*t4*t15; fdrv(2, 5) = -t171*t208;
    fdrv(2, 6) = tdx*(-t83+t90+rx*t40*1.2E+1+rx*t42*2.0-ry*t39*2.0+t6*t20-rz*t2*tz*2.0+rz*t3*tz*6.0)-t221*tdy+oy*t208*tdz-t171*t175*tdz;
    fdrv(2, 7) = -tdy*(t83-t90-rx*t40*2.0+ry*t38*2.0+ry*t39*1.2E+1+t4*t23+rz*t2*tz*6.0-rz*t3*tz*2.0)-t221*tdx-ox*t208*tdz-t171*t176*tdz;
    fdrv(2, 8) = tdx*(t149+rz*t42+rz*t105+rz*t108+t6*t21-rz*t2*tx*2.0+rx*t3*tz*4.0)-tdy*(t150+rz*t38+rz*t104+rz*t106+t4*t17+ry*t2*tz*4.0-rz*t3*ty*2.0)-t171*tdz*(t17+t21+rz*tz*6.0);
    fdrv(2, 9) = -tdx*(t2*t11*3.0-t3*t11*4.0+t2*t13-t3*t13*2.0+t2*t15-t3*t15*2.0)-t200*tdy*tx-t171*tdz*tx*tz*2.0; fdrv(2, 10) = -tdy*(t2*t11*2.0+t2*t13*4.0-t3*t13*3.0+t2*t15*2.0+t11*t35+t15*t35)+t202*tdx*ty-t171*tdz*ty*tz*2.0;
    fdrv(2, 11) = t202*tdx*tz-t200*tdy*tz-t171*tdz*(t15*2.0+t178); fdrv(2, 12) = -t209-t214-t219; fdrv(2, 13) = t211+t213+t218; fdrv(2, 15) = -t171*t178*tx; fdrv(2, 16) = -t171*t178*ty; fdrv(2, 17) = -t171*t178*tz;
    fdrv(2, 18) = t183+t186+t189-rdx*t171*t178-t8*t171*tx*2.0-t9*t171*tx*2.0-t10*t171*tx*2.0; fdrv(2, 19) = -t182-t184-t187-rdy*t171*t178-t8*t171*ty*2.0-t9*t171*ty*2.0-t10*t171*ty*2.0;
    fdrv(2, 20) = -t171*(rdz*t11+rdz*t13+t8*tz*2.0+t9*tz*2.0+t10*tz*3.0); fdrv(2, 24) = -t198; fdrv(2, 25) = t197;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f34(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = rdx*tx, t3 = rdy*ty, t4 = rdz*tz, t5 = rx*tx, t6 = rx*ty, t7 = ry*tx, t8 = rx*tz, t9 = ry*ty, t10 = rz*tx, t11 = ry*tz, t12 = rz*ty, t13 = rz*tz, t14 = tx*tx, t15 = tx*tx*tx, t17 = tx*tx*tx*tx*tx, t18 = ty*ty, t19 = ty*ty*ty;
    T t21 = ty*ty*ty*ty*ty, t22 = tz*tz, t23 = tz*tz*tz, t25 = tz*tz*tz*tz*tz, t26 = ox*tx*2.0, t27 = ox*ty*2.0, t28 = oy*tx*2.0, t29 = ox*ty*4.0, t30 = oy*tx*4.0, t31 = ox*tz*2.0, t32 = oy*ty*2.0, t33 = oz*tx*2.0, t34 = ox*tz*4.0;
    T t35 = oz*tx*4.0, t36 = oy*tz*2.0, t37 = oz*ty*2.0, t38 = oy*tz*4.0, t39 = oz*ty*4.0, t40 = oz*tz*2.0, t203 = ox*tx*ty*-2.0, t204 = ox*tx*tz*-2.0, t205 = oy*tx*ty*-2.0, t206 = ox*ty*tz*-2.0, t207 = oy*tx*tz*-2.0, t208 = oz*tx*ty*-2.0;
    T t209 = oy*ty*tz*-2.0, t210 = oz*tx*tz*-2.0, t211 = oz*ty*tz*-2.0, t374 = ox*tx*ty*tz*-4.0, t375 = oy*tx*ty*tz*-4.0, t376 = oz*tx*ty*tz*-4.0, t16 = t14*t14, t20 = t18*t18, t24 = t22*t22, t41 = t5*2.0, t42 = t9*2.0, t43 = t13*2.0, t44 = t5*ty;
    T t45 = t5*tz, t46 = t7*ty, t47 = t9*tz, t48 = t10*tz, t49 = t12*tz, t50 = -t27, t51 = -t29, t52 = -t33, t53 = -t35, t54 = -t36, t55 = -t38, t56 = ox*t14, t57 = ox*t15, t59 = ox*t18, t60 = oy*t14, t61 = ox*t19, t62 = oy*t15, t65 = ox*t21;
    T t66 = oy*t17, t67 = ox*t22, t68 = oy*t18, t69 = oz*t14, t70 = ox*t23, t71 = oy*t19, t72 = oz*t15, t76 = ox*t25, t77 = oz*t17, t78 = oy*t22, t79 = oz*t18, t80 = oy*t23, t81 = oz*t19, t84 = oy*t25, t85 = oz*t21, t86 = oz*t22, t87 = oz*t23;
    T t104 = t26*ty, t107 = t27*tz, t108 = t28*tz, t109 = t33*ty, t112 = t37*tz, t150 = t6*t19*tz, t152 = t7*t15*tz, t154 = t10*t15*ty, t158 = t29*tx*tz, t159 = t30*ty*tz, t160 = t35*ty*tz, t197 = t19*tx*2.0, t198 = t15*ty*2.0, t199 = t23*tx*2.0;
    T t200 = t15*tz*2.0, t201 = t23*ty*2.0, t202 = t19*tz*2.0, t215 = t14+t18, t216 = t14+t22, t217 = t18+t22, t275 = t18*t26, t277 = t14*t29, t279 = t15*t29, t281 = t22*t26, t285 = t14*t34, t286 = t18*t30, t288 = t15*t34, t289 = t19*t30;
    T t294 = t18*t31, t295 = t22*t28, t298 = t14*t37, t299 = t23*t27, t300 = t19*t31, t301 = t23*t28, t302 = t15*t36, t303 = t19*t33, t304 = t15*t37, t305 = t22*t32, t309 = t18*t38, t310 = t22*t35, t312 = t19*t38, t313 = t23*t35, t319 = t22*t39;
    T t320 = t23*t39, t326 = t5*t19*4.0, t332 = t7*t19*2.0, t336 = t5*t23*4.0, t344 = t6*t23*2.0, t345 = t6*t18*tz*2.0, t346 = t7*t23*2.0, t347 = t7*t14*tz*2.0, t348 = t10*t19*2.0, t349 = t10*t14*ty*2.0, t352 = t10*t23*2.0, t356 = t9*t23*4.0;
    T t364 = t12*t23*2.0, t371 = t22*tx*ty*2.0, t372 = t18*tx*tz*2.0, t373 = t14*ty*tz*2.0, t398 = t5*t15*-2.0, t399 = t5*t15*1.0E+1, t400 = t9*t19*-2.0, t401 = t9*t19*1.0E+1, t402 = t13*t23*-2.0, t403 = t13*t23*1.0E+1, t404 = t5+t9;
    T t405 = t5+t13, t406 = t9+t13, t407 = t14*t19*2.0, t408 = t15*t18*2.0, t409 = t14*t23*2.0, t410 = t15*t22*2.0, t411 = t18*t23*2.0, t412 = t19*t22*2.0, t503 = t5*t19*8.0, t505 = t5*t23*8.0, t507 = t7*t19*8.0, t509 = t9*t23*8.0;
    T t511 = t10*t23*8.0, t513 = t12*t23*8.0, t516 = t5*t29*tx, t520 = t5*t34*tx, t532 = t7*t34*tx, t542 = t6*t18*t34, t550 = ox*t7*tx*tz*6.0, t552 = ox*t10*tx*ty*6.0, t562 = t10*t30*ty, t570 = t7*t14*t38, t580 = oy*t6*ty*tz*6.0;
    T t582 = oy*t10*tx*ty*6.0, t590 = t9*t38*ty, t594 = t6*t39*tz, t596 = t7*t35*tz, t604 = t10*t14*t39, t610 = oz*t6*ty*tz*6.0, t612 = oz*t7*tx*tz*6.0, t630 = t22*t29*tx, t631 = t18*t34*tx, t633 = t22*t30*ty, t634 = t14*t38*ty, t636 = t18*t35*tz;
    T t637 = t14*t39*tz, t648 = t7*t18*tz*4.0, t651 = t7*t19*tz*4.0, t654 = t7*t18*tz*6.0, t656 = t10*t22*ty*4.0, t659 = t10*t23*ty*4.0, t662 = t10*t22*ty*6.0, t719 = t5*t18*tx*6.0, t720 = t5*t19*tx*6.0, t721 = t5*t14*t18*6.0;
    T t723 = t7*t18*tx*2.0, t727 = t7*t14*t18*4.0, t728 = t5*t22*tx*6.0, t729 = t7*t18*tx*6.0, t730 = t5*t23*tx*6.0, t731 = t5*t14*t22*6.0, t732 = t7*t19*tx*6.0, t733 = t7*t14*t18*6.0, t734 = t6*t22*ty*2.0, t735 = t7*t22*tx*2.0;
    T t736 = t10*t18*tx*2.0, t738 = t6*t18*t22*2.0, t740 = t7*t14*t22*2.0, t742 = t10*t14*t18*2.0, t744 = t10*t22*tx*2.0, t748 = t10*t14*t22*4.0, t749 = t9*t22*ty*6.0, t750 = t10*t22*tx*6.0, t751 = t9*t23*ty*6.0, t752 = t9*t18*t22*6.0;
    T t753 = t10*t23*tx*6.0, t754 = t10*t14*t22*6.0, t755 = t12*t22*ty*2.0, t757 = t12*t18*t22*4.0, t758 = t12*t22*ty*6.0, t759 = t12*t23*ty*6.0, t760 = t12*t18*t22*6.0, t761 = t18*t22*tx*2.0, t762 = t14*t22*ty*2.0, t763 = t14*t18*tz*2.0;
    T t782 = ox*t7*tx*tz*-4.0, t784 = ox*t10*tx*ty*-4.0, t792 = ox*t6*ty*tz*8.0, t804 = oy*t6*ty*tz*-4.0, t806 = oy*t10*tx*ty*-4.0, t814 = oy*t7*tx*tz*8.0, t832 = oz*t6*ty*tz*-4.0, t834 = oz*t7*tx*tz*-4.0, t841 = oz*t10*tx*ty*8.0;
    T t900 = t5*t23*t29, t901 = t5*t19*t34, t903 = t7*t22*t29, t905 = t10*t18*t34, t917 = t2+t3+t4, t927 = t5*t14*t18*8.0, t928 = t5*t14*t22*8.0, t930 = t9*t18*t22*8.0, t997 = t5*t18*t22*4.0, t1000 = t5*t18*t22*6.0, t1003 = t7*t18*t22*4.0;
    T t1006 = t7*t18*t22*6.0, t1009 = t10*t18*t22*4.0, t1012 = t10*t18*t22*6.0, t58 = ox*t16, t63 = ox*t20, t64 = oy*t16, t73 = ox*t24, t74 = oy*t20, t75 = oz*t16, t82 = oy*t24, t83 = oz*t20, t88 = oz*t24, t89 = t5*t16, t90 = t6*t20, t91 = t7*t16;
    T t92 = t8*t24, t93 = t9*t20, t94 = t10*t16, t95 = t11*t24, t96 = t12*t20, t97 = t13*t24, t98 = t20*tx, t99 = t16*ty, t100 = t24*tx, t101 = t16*tz, t102 = t24*ty, t103 = t20*tz, t114 = t6*t61, t115 = t7*t57, t116 = t8*t70, t117 = t10*t57;
    T t118 = t6*t71, t119 = t7*t62, t121 = t11*t80, t122 = t12*t71, t123 = t8*t87, t124 = t10*t72, t125 = t11*t87, t126 = t12*t81, t128 = t59*tx, t129 = t56*ty, t130 = t67*tx, t131 = t56*tz, t132 = t68*tx, t133 = t60*ty, t134 = t67*ty;
    T t135 = t59*tz, t136 = t78*tx, t137 = t60*tz, t138 = t79*tx, t139 = t69*ty, t140 = t78*ty, t141 = t68*tz, t142 = t86*tx, t143 = t69*tz, t144 = t86*ty, t145 = t79*tz, t146 = t5*t20, t147 = t5*t24, t148 = t15*t46, t149 = t6*t24, t151 = t7*t24;
    T t153 = t10*t20, t155 = t9*t24, t156 = t15*t48, t157 = t19*t49, t161 = t59*2.0, t162 = t60*2.0, t163 = t59*3.0, t164 = t61*2.0, t165 = t60*3.0, t166 = t62*2.0, t171 = t67*2.0, t172 = t69*2.0, t173 = t67*3.0, t174 = t70*2.0, t175 = t69*3.0;
    T t176 = t72*2.0, t181 = t78*2.0, t182 = t79*2.0, t183 = t78*3.0, t184 = t80*2.0, t185 = t79*3.0, t186 = t81*2.0, t191 = t15*t41, t193 = t19*t42, t195 = t23*t43, t212 = -t45, t213 = -t46, t214 = -t49, t219 = t5*t56*4.0, t220 = t5*t57*5.0;
    T t221 = t7*t56*2.0, t222 = t41*t60, t223 = t5*t62*4.0, t225 = t5*t62*5.0, t226 = t42*t59, t227 = t10*t56*2.0, t228 = t6*t68*2.0, t229 = t41*t69, t230 = t9*t61*4.0, t231 = t5*t72*4.0, t233 = t9*t61*5.0, t235 = t5*t72*5.0, t249 = t9*t68*4.0;
    T t250 = t11*t70*5.0, t251 = t12*t61*5.0, t252 = t8*t80*5.0, t253 = t9*t71*5.0, t254 = t10*t62*5.0, t255 = t6*t81*5.0, t256 = t7*t72*5.0, t257 = t43*t67, t258 = t12*t68*2.0, t259 = t8*t86*2.0, t260 = t42*t79, t261 = t13*t70*4.0;
    T t262 = t9*t81*4.0, t263 = t13*t70*5.0, t265 = t9*t81*5.0, t267 = t43*t78, t268 = t11*t86*2.0, t269 = t13*t80*4.0, t270 = t13*t80*5.0, t273 = t13*t86*4.0, t274 = t13*t87*5.0, t278 = t61*tx*4.0, t287 = t70*tx*4.0, t290 = t62*ty*4.0;
    T t311 = t80*ty*4.0, t314 = t72*tz*4.0, t321 = t81*tz*4.0, t323 = t19*t41, t324 = t14*t41*ty, t325 = t20*t41, t327 = t15*t44*4.0, t329 = t15*t44*5.0, t330 = t23*t41, t331 = t14*t41*tz, t333 = t14*t46*2.0, t334 = t24*t41, t337 = t14*t46*4.0;
    T t338 = t15*t45*4.0, t339 = t7*t20*4.0, t341 = t15*t45*5.0, t342 = t7*t20*5.0, t350 = t23*t42, t351 = t18*t42*tz, t353 = t14*t48*2.0, t354 = t24*t42, t357 = t14*t48*4.0, t358 = t19*t47*4.0, t359 = t10*t24*4.0, t361 = t19*t47*5.0;
    T t362 = t10*t24*5.0, t365 = t18*t49*2.0, t367 = t18*t49*4.0, t368 = t12*t24*4.0, t369 = t12*t24*5.0, t377 = -t62, t380 = -t65, t382 = t61*8.0, t383 = t62*8.0, t388 = -t77, t389 = t70*8.0, t390 = t72*8.0, t391 = -t81, t394 = -t84;
    T t396 = t80*8.0, t397 = t81*8.0, t413 = t5*t57*-2.0, t415 = t5*t56*8.0, t418 = t6*t59*8.0, t419 = t5*t60*8.0, t426 = t8*t67*8.0, t427 = t9*t59*8.0, t428 = t7*t60*8.0, t429 = t5*t69*8.0, t430 = t9*t71*-2.0, t435 = t11*t67*8.0;
    T t436 = t12*t59*8.0, t437 = t8*t78*8.0, t438 = t9*t68*8.0, t439 = t10*t60*8.0, t440 = t6*t79*8.0, t441 = t7*t69*8.0, t446 = t13*t67*8.0, t447 = t11*t78*8.0, t448 = t9*t79*8.0, t449 = t10*t69*8.0, t453 = t13*t78*8.0, t454 = t12*t79*8.0;
    T t455 = t13*t87*-2.0, t457 = t13*t86*8.0, t461 = t57*ty*-4.0, t468 = t57*tz*-4.0, t469 = t71*tx*-4.0, t477 = t70*ty*-2.0, t478 = t61*tz*-2.0, t479 = t80*tx*-2.0, t480 = t62*tz*-2.0, t481 = t81*tx*-2.0, t482 = t72*ty*-2.0, t494 = t71*tz*-4.0;
    T t495 = t87*tx*-4.0, t501 = t87*ty*-4.0, t504 = t14*t44*8.0, t506 = t14*t45*8.0, t508 = t14*t46*8.0, t510 = t18*t47*8.0, t512 = t14*t48*8.0, t514 = t18*t49*8.0, t515 = t5*t59*4.0, t517 = t5*t61*4.0, t518 = t5*t277, t519 = t5*t67*4.0;
    T t521 = t5*t70*4.0, t522 = t5*t285, t523 = t7*t61*4.0, t524 = t7*t277, t525 = t5*t71*4.0, t526 = t44*t60*4.0, t527 = t7*t59*6.0, t528 = ox*t46*tx*6.0, t529 = t5*t68*6.0, t530 = oy*t44*tx*6.0, t531 = t7*t67*4.0, t533 = t10*t59*4.0;
    T t535 = t5*t78*4.0, t536 = t30*t45, t537 = t7*t68*4.0, t538 = t30*t46, t539 = t5*t79*4.0, t540 = t35*t44, t541 = t6*t70*4.0, t543 = t7*t70*4.0, t544 = t7*t285, t545 = t10*t61*4.0, t547 = t7*t71*4.0, t548 = t46*t60*4.0, t549 = t7*t67*6.0;
    T t551 = t10*t59*6.0, t553 = t5*t78*6.0, t554 = oy*t45*tx*6.0, t555 = t5*t79*6.0, t556 = oz*t44*tx*6.0, t557 = t9*t67*4.0, t558 = t29*t47, t559 = t6*t78*4.0, t561 = t10*t68*4.0, t563 = t7*t79*4.0, t564 = t35*t46, t565 = t10*t70*4.0;
    T t567 = t6*t80*4.0, t569 = t7*t80*4.0, t571 = t10*t71*4.0, t573 = t5*t87*4.0, t574 = t45*t69*4.0, t575 = t9*t67*6.0, t576 = ox*t47*ty*6.0, t577 = t10*t67*6.0, t578 = ox*t48*tx*6.0, t579 = t6*t78*6.0, t581 = t10*t68*6.0, t583 = t5*t86*6.0;
    T t584 = oz*t45*tx*6.0, t585 = t7*t79*6.0, t586 = oz*t46*tx*6.0, t587 = t12*t67*4.0, t588 = t29*t49, t589 = t9*t78*4.0, t591 = t10*t78*4.0, t592 = t30*t48, t593 = t6*t86*4.0, t595 = t7*t86*4.0, t597 = t9*t80*4.0, t598 = t9*t309;
    T t599 = t6*t87*4.0, t601 = t7*t87*4.0, t603 = t10*t81*4.0, t605 = t12*t67*6.0, t606 = ox*t49*ty*6.0, t607 = t10*t78*6.0, t608 = oy*t48*tx*6.0, t609 = t6*t86*6.0, t611 = t7*t86*6.0, t613 = t10*t86*4.0, t614 = t35*t48, t615 = t12*t80*4.0;
    T t616 = t12*t309, t617 = t9*t87*4.0, t618 = t47*t79*4.0, t619 = t10*t87*4.0, t620 = t48*t69*4.0, t621 = t12*t78*6.0, t622 = oy*t49*ty*6.0, t623 = t9*t86*6.0, t624 = oz*t47*ty*6.0, t625 = t12*t86*4.0, t626 = t39*t49, t627 = t12*t87*4.0;
    T t628 = t49*t79*4.0, t629 = t14*t107, t632 = t18*t108, t635 = t22*t109, t638 = t22*t44*4.0, t639 = t18*t45*4.0, t640 = t44*tx*tz*4.0, t641 = t23*t44*4.0, t642 = t19*t45*4.0, t643 = t14*t44*tz*4.0, t644 = t22*t44*6.0, t645 = t18*t45*6.0;
    T t646 = t44*tx*tz*6.0, t647 = t22*t46*4.0, t649 = t46*tx*tz*4.0, t650 = t23*t46*4.0, t653 = t22*t46*6.0, t655 = t46*tx*tz*6.0, t657 = t18*t48*4.0, t658 = t48*tx*ty*4.0, t660 = t19*t48*4.0, t663 = t18*t48*6.0, t664 = t48*tx*ty*6.0;
    T t665 = ox*t44*tz*8.0, t666 = ox*t44*tz*1.6E+1, t667 = ox*t46*tz*8.0, t668 = oy*t44*tz*8.0, t669 = ox*t46*tz*1.2E+1, t670 = oy*t44*tz*1.2E+1, t671 = ox*t48*ty*8.0, t672 = oy*t46*tz*8.0, t673 = oz*t44*tz*8.0, t674 = ox*t48*ty*1.2E+1;
    T t675 = oz*t44*tz*1.2E+1, t676 = oy*t46*tz*1.6E+1, t677 = oy*t48*ty*8.0, t678 = oz*t46*tz*8.0, t679 = oy*t48*ty*1.2E+1, t680 = oz*t46*tz*1.2E+1, t681 = oz*t48*ty*8.0, t682 = oz*t48*ty*1.6E+1, t701 = t18*t56*2.0, t702 = t18*t56*6.0;
    T t703 = t22*t56*2.0, t705 = t22*t56*6.0, t706 = t18*t60*6.0, t707 = t22*t59*6.0, t708 = t22*t60*6.0, t709 = t18*t69*6.0, t710 = t22*t68*2.0, t712 = t22*t68*6.0, t713 = t22*t69*6.0, t715 = t22*t79*6.0, t716 = t18*t41*tx, t717 = t14*t18*t41;
    T t718 = t326*tx, t722 = t22*t41*tx, t724 = t14*t22*t41, t725 = t7*t197, t726 = t336*tx, t737 = t6*t201, t739 = t7*t199, t741 = t10*t197, t743 = t22*t42*ty, t745 = t18*t22*t42, t746 = t10*t199, t747 = t356*ty, t756 = t12*t201;
    T t765 = t5*t59*1.2E+1, t766 = ox*t44*tx*1.2E+1, t768 = t45*t56*-4.0, t771 = t46*t56*8.0, t772 = t5*t71*8.0, t773 = t5*t67*1.2E+1, t774 = ox*t45*tx*1.2E+1, t775 = t7*t59*1.2E+1, t776 = ox*t46*tx*1.2E+1, t777 = t5*t68*1.2E+1;
    T t778 = oy*t44*tx*1.2E+1, t779 = t7*t61*1.6E+1, t780 = t44*t60*1.6E+1, t791 = t6*t67*8.0, t795 = t7*t68*1.2E+1, t796 = oy*t46*tx*1.2E+1, t797 = t45*t60*1.6E+1, t798 = t44*t69*1.6E+1, t799 = t5*t78*2.4E+1, t800 = oy*t45*tx*2.4E+1;
    T t801 = t5*t79*2.4E+1, t802 = oz*t44*tx*2.4E+1, t808 = t48*t56*-4.0, t813 = t7*t78*8.0, t816 = t48*t56*8.0, t817 = t5*t87*8.0, t818 = t46*t69*8.0, t819 = t10*t67*1.2E+1, t820 = ox*t48*tx*1.2E+1, t821 = t5*t86*1.2E+1, t822 = oz*t45*tx*1.2E+1;
    T t823 = t47*t59*1.6E+1, t824 = t10*t70*1.6E+1, t825 = t45*t69*1.6E+1, t826 = t7*t81*1.6E+1, t827 = t9*t67*2.4E+1, t828 = ox*t47*ty*2.4E+1, t829 = t7*t79*2.4E+1, t830 = oz*t46*tx*2.4E+1, t840 = t10*t79*8.0, t842 = t49*t59*8.0;
    T t843 = t48*t60*8.0, t844 = t9*t78*1.2E+1, t845 = oy*t47*ty*1.2E+1, t846 = t12*t70*1.6E+1, t847 = t10*t80*1.6E+1, t848 = t12*t67*2.4E+1, t849 = ox*t49*ty*2.4E+1, t850 = t10*t78*2.4E+1, t851 = oy*t48*tx*2.4E+1, t855 = t49*t68*8.0;
    T t856 = t9*t87*8.0, t857 = t12*t78*1.2E+1, t858 = oy*t49*ty*1.2E+1, t859 = t9*t86*1.2E+1, t860 = oz*t47*ty*1.2E+1, t861 = t10*t86*1.2E+1, t862 = oz*t48*tx*1.2E+1, t863 = t12*t80*1.6E+1, t864 = t47*t79*1.6E+1, t867 = t12*t86*1.2E+1;
    T t868 = oz*t49*ty*1.2E+1, t879 = -t648, t880 = -t656, t890 = t404*tx, t891 = t404*ty, t892 = t405*tx, t893 = t405*tz, t894 = t406*ty, t895 = t406*tz, t896 = t41+t42, t897 = t41+t43, t898 = t42+t43, t902 = t45*t277, t904 = t44*t78*4.0;
    T t907 = t45*t79*4.0, t908 = t46*t80*4.0, t909 = t7*t312, t910 = t14*t38*t46, t914 = t10*t320, t915 = t48*t81*4.0, t916 = t14*t39*t48, t921 = t22*t59*1.2E+1, t922 = t22*t60*1.2E+1, t923 = t18*t69*1.2E+1, t929 = t507*tx, t931 = t511*tx;
    T t932 = t513*ty, t940 = -t792, t941 = t5*t81*-8.0, t947 = -t814, t948 = t9*t70*-8.0, t957 = -t841, t969 = t22+t215, t1005 = t648*tx, t1008 = t654*tx, t1010 = t656*tx, t1013 = t662*tx, t1015 = ox*t44*tx*tz*-4.0, t1016 = t44*t67*1.2E+1;
    T t1017 = t45*t59*1.2E+1, t1019 = t46*t67*8.0, t1022 = t44*t78*8.0, t1023 = t45*t68*8.0, t1025 = t46*t67*1.2E+1, t1028 = t44*t78*1.2E+1, t1029 = t45*t68*1.2E+1, t1033 = t48*t59*8.0, t1035 = t44*t86*8.0, t1036 = t45*t79*8.0;
    T t1039 = t48*t59*1.2E+1, t1041 = t46*t78*1.2E+1, t1044 = t44*t86*1.2E+1, t1045 = t45*t79*1.2E+1, t1048 = t48*t68*8.0, t1050 = t46*t86*8.0, t1054 = t48*t68*1.2E+1, t1056 = t46*t86*1.2E+1, t1061 = t48*t79*1.2E+1, t1107 = ox*t44*tx*tz*-1.2E+1;
    T t1110 = ox*t46*tx*tz*-8.0, t1113 = oy*t44*tx*tz*-1.2E+1, t1116 = oz*t44*tx*tz*-8.0, t1119 = ox*t48*tx*ty*-1.2E+1, t1122 = oy*t46*tx*tz*-1.2E+1, t1125 = oy*t48*tx*ty*-8.0, t1128 = oz*t46*tx*tz*-1.2E+1, t1131 = oz*t48*tx*ty*-1.2E+1;
    T t1147 = t26+t32+t108+t206, t1148 = t26+t40+t107+t208, t1149 = t32+t40+t109+t207, t113 = t5*t58, t120 = t9*t74, t127 = t13*t88, t167 = t63*2.0, t168 = t64*2.0, t169 = t63*5.0, t170 = t64*5.0, t177 = t73*2.0, t178 = t75*2.0, t179 = t73*5.0;
    T t180 = t75*5.0, t187 = t82*2.0, t188 = t83*2.0, t189 = t82*5.0, t190 = t83*5.0, t192 = t89*6.0, t194 = t93*6.0, t196 = t97*6.0, t224 = t114*5.0, t232 = t116*5.0, t234 = t119*5.0, t236 = t11*t171, t237 = t12*t161, t238 = t8*t181;
    T t239 = t10*t162, t240 = t6*t182, t241 = t7*t172, t242 = t11*t174, t243 = t12*t164, t244 = t8*t184, t246 = t10*t166, t247 = t6*t186, t248 = t7*t176, t264 = t121*5.0, t266 = t124*5.0, t271 = t126*5.0, t276 = t128*3.0, t280 = t129*6.0;
    T t283 = t130*3.0, t284 = t133*3.0, t291 = t131*6.0, t292 = t132*6.0, t307 = t140*3.0, t308 = t143*3.0, t315 = t141*6.0, t316 = t142*6.0, t318 = t145*3.0, t322 = t144*6.0, t328 = t146*5.0, t335 = t148*2.0, t340 = t147*5.0, t343 = t148*5.0;
    T t355 = t156*2.0, t360 = t155*5.0, t363 = t156*5.0, t366 = t157*2.0, t370 = t157*5.0, t378 = -t164, t385 = -t176, t392 = -t184, t414 = -t220, t417 = -t225, t420 = -t116, t421 = -t117, t422 = -t118, t423 = -t119, t424 = -t230, t425 = -t231;
    T t431 = -t251, t432 = -t252, t433 = -t253, t434 = -t256, t442 = -t263, t444 = -t265, t450 = -t125, t451 = -t126, t452 = -t269, t456 = -t274, t458 = t128*-2.0, t459 = t129*-4.0, t460 = -t278, t464 = -t132, t465 = t130*-2.0, t466 = t133*-2.0;
    T t467 = -t287, t470 = -t290, t472 = -t136, t473 = -t139, t474 = t134*-2.0, t475 = t137*-2.0, t476 = t138*-2.0, t483 = t134*8.0, t484 = t135*8.0, t485 = t136*8.0, t486 = t137*8.0, t487 = t138*8.0, t488 = t139*8.0, t489 = t140*-2.0;
    T t490 = t143*-2.0, t491 = t141*-4.0, t492 = t142*-4.0, t493 = -t311, t496 = -t314, t499 = -t144, t500 = t145*-2.0, t502 = -t321, t572 = t10*t133*4.0, t600 = t6*t145*4.0, t602 = t7*t143*4.0, t652 = t337*tz, t661 = t357*ty, t683 = -t382;
    T t684 = -t390, t685 = -t396, t686 = -t418, t687 = -t426, t688 = -t428, t689 = -t435, t690 = -t436, t691 = -t437, t692 = -t439, t693 = -t440, t694 = -t441, t695 = -t447, t696 = -t449, t697 = -t454, t704 = t18*t162, t711 = t22*t172;
    T t714 = t22*t182, t764 = -t517, t767 = -t521, t769 = -t525, t770 = -t526, t781 = -t531, t783 = -t533, t785 = -t541, t786 = t6*t135*-4.0, t787 = -t545, t789 = -t547, t790 = -t548, t793 = t5*t396, t794 = t5*t397, t803 = -t559, t805 = -t561;
    T t807 = -t565, t809 = -t567, t811 = -t569, t812 = t7*t137*-4.0, t815 = t9*t389, t831 = -t593, t833 = -t595, t835 = -t597, t836 = -t601, t838 = -t603, t839 = t10*t139*-4.0, t852 = -t617, t853 = -t618, t854 = -t620, t865 = -t627, t866 = -t628;
    T t869 = t129*tz*-2.0, t870 = t130*ty*-4.0, t871 = t128*tz*-4.0, t872 = t132*tz*-2.0, t873 = t136*ty*-4.0, t874 = t133*tz*-4.0, t875 = t142*ty*-2.0, t876 = t138*tz*-4.0, t877 = t139*tz*-4.0, t878 = -t640, t881 = -t666, t882 = -t667;
    T t883 = -t668, t884 = -t671, t885 = -t673, t886 = -t676, t887 = -t677, t888 = -t678, t889 = -t682, t911 = t592*ty, t912 = t564*tz, t918 = -t702, t919 = -t705, t920 = -t706, t924 = -t712, t925 = -t713, t926 = -t715, t933 = -t766, t934 = -t771;
    T t935 = -t774, t936 = -t775, t937 = -t778, t938 = -t779, t939 = -t791, t942 = -t795, t943 = -t798, t944 = -t800, t945 = -t802, t946 = -t813, t949 = -t817, t950 = -t819, t951 = -t822, t952 = -t823, t953 = -t825, t954 = -t828, t955 = -t829;
    T t956 = -t840, t958 = -t843, t959 = -t845, t960 = -t847, t961 = -t848, t962 = -t850, t963 = -t855, t964 = -t857, t965 = -t860, t966 = -t861, t967 = -t863, t968 = -t867, t970 = t5*t128*6.0, t971 = t5*t130*6.0, t972 = t7*t128*6.0;
    T t974 = t6*t134*6.0, t975 = t7*t130*6.0, t976 = t10*t128*6.0, t978 = t10*t130*6.0, t979 = t6*t140*6.0, t980 = t7*t136*6.0, t983 = t9*t140*6.0, t986 = t10*t138*6.0, t987 = t12*t140*6.0, t994 = t18*t137*6.0, t995 = t22*t138*6.0;
    T t996 = t22*t139*6.0, t998 = t638*tx, t999 = t639*tx, t1001 = t644*tx, t1002 = t645*tx, t1004 = t647*tx, t1007 = t653*tx, t1011 = t657*tx, t1014 = t663*tx, t1018 = t766*tz, t1024 = t668*tx, t1026 = t7*t135*1.2E+1, t1027 = t669*tx;
    T t1034 = t671*tx, t1038 = t10*t134*1.2E+1, t1040 = t674*tx, t1042 = t7*t141*1.2E+1, t1043 = t796*tz, t1046 = t675*tx, t1047 = t10*t140*8.0, t1051 = t7*t145*8.0, t1052 = t678*tx, t1053 = t10*t140*1.2E+1, t1055 = t679*tx;
    T t1057 = t7*t145*1.2E+1, t1059 = t10*t144*-4.0, t1060 = t10*t144*1.2E+1, t1062 = t862*ty, t1063 = t5*t707, t1064 = t46*t136*6.0, t1065 = t48*t138*6.0, t1066 = -t921, t1067 = -t922, t1068 = -t923, t1069 = -t891, t1070 = -t892, t1071 = -t895;
    T t1072 = t5*t128*1.8E+1, t1074 = t5*t132*-6.0, t1075 = t5*t130*1.8E+1, t1078 = t7*t132*-6.0, t1079 = t5*t136*1.2E+1, t1080 = t5*t138*1.2E+1, t1081 = t5*t136*1.8E+1, t1082 = t7*t132*1.8E+1, t1083 = t5*t138*1.8E+1, t1087 = t9*t134*1.2E+1;
    T t1088 = t7*t138*1.2E+1, t1089 = t9*t134*1.8E+1, t1090 = t7*t138*1.8E+1, t1091 = t7*t142*-6.0, t1093 = t12*t134*1.2E+1, t1094 = t10*t136*1.2E+1, t1095 = t12*t134*1.8E+1, t1096 = t9*t140*1.8E+1, t1097 = t10*t136*1.8E+1, t1098 = t9*t144*-6.0;
    T t1099 = t10*t142*1.8E+1, t1100 = t12*t144*-6.0, t1101 = t12*t144*1.8E+1, t1102 = t22*t129*-6.0, t1105 = -t1016, t1106 = -t1017, t1108 = -t1019, t1109 = t7*t135*-8.0, t1111 = -t1028, t1112 = -t1029, t1114 = -t1035, t1115 = -t1036;
    T t1118 = -t1039, t1120 = -t1041, t1124 = -t1048, t1126 = -t1056, t1130 = -t1061, t1132 = t896*tx*tz, t1133 = t897*tx*ty, t1134 = t896*ty*tz, t1135 = t898*tx*ty, t1136 = t897*ty*tz, t1137 = t898*tx*tz, t1144 = t215*t896, t1145 = t216*t897;
    T t1146 = t217*t898, t1150 = t6+t7+t47+t893, t1151 = t8+t10+t44+t894, t1152 = t11+t12+t48+t890, t1177 = t31+t53+t68+t78+t165+t203, t1178 = t39+t54+t56+t67+t163+t205, t1179 = t30+t50+t79+t86+t175+t204, t1180 = t37+t55+t56+t59+t173+t210;
    T t1181 = t28+t51+t69+t86+t185+t209, t1182 = t34+t52+t60+t68+t183+t211, t379 = -t167, t381 = -t168, t386 = -t177, t387 = -t178, t393 = -t187, t395 = -t188, t416 = -t224, t443 = -t264, t445 = -t266, t462 = -t280, t497 = -t315, t498 = -t316;
    T t698 = -t483, t699 = -t486, t700 = -t487, t788 = t10*t459, t810 = t6*t491, t837 = -t602, t973 = t5*t292, t977 = t7*t292, t981 = t10*t292, t982 = t5*t316, t984 = t6*t322, t988 = t9*t322, t989 = t10*t316, t990 = t12*t322, t992 = t18*t291;
    T t993 = t22*t292, t1031 = t7*t491, t1032 = t10*t483, t1073 = -t971, t1076 = -t974, t1077 = -t976, t1084 = -t978, t1085 = -t979, t1086 = -t980, t1092 = -t986, t1103 = -t994, t1104 = -t995, t1117 = -t1038, t1121 = -t1042, t1123 = -t1047;
    T t1127 = -t1057, t1129 = -t1060, t1138 = -t1072, t1139 = -t1081, t1140 = -t1090, t1141 = -t1095, t1142 = -t1096, t1143 = -t1099, t1153 = t1150*tx*ty, t1154 = t1151*tx*ty, t1155 = t1151*tx*tz, t1156 = t1152*tx*tz, t1157 = t1150*ty*tz;
    T t1158 = t1152*ty*tz, t1159 = t6+t7+t212+t1071, t1160 = t8+t10+t214+t1069, t1161 = t11+t12+t213+t1070, t1171 = t215*t1152, t1172 = t216*t1150, t1173 = t217*t1151, t1186 = t61+t129+t134+t172+t182+t204+t209+t377+t464+t472;
    T t1188 = t80+t137+t141+t161+t171+t205+t210+t391+t473+t499, t1198 = t57+t71+t128+t133+t283+t301+t302+t307+t477+t478+t490+t500+t632+t869, t1199 = t57+t87+t130+t143+t276+t299+t300+t318+t466+t481+t482+t489+t629+t875;
    T t1200 = t71+t87+t140+t145+t284+t303+t304+t308+t458+t465+t479+t480+t635+t872, t1162 = -t1154, t1163 = -t1156, t1164 = -t1157, t1165 = t1159*tx*ty, t1166 = t1161*tx*ty, t1167 = t1159*tx*tz, t1168 = t1160*tx*tz, t1169 = t1160*ty*tz;
    T t1170 = t1161*ty*tz, t1183 = t215*t1160, t1184 = t216*t1161, t1185 = t217*t1159, t1201 = t74+t82+t159+t170+t174+t291+t294+t460+t461+t492+t684+t700+t706+t708+t710+t870;
    T t1202 = t58+t63+t158+t179+t186+t298+t322+t491+t495+t496+t685+t699+t701+t705+t707+t876, t1203 = t75+t88+t160+t166+t190+t292+t295+t459+t493+t494+t683+t698+t709+t711+t715+t874;
    T t1204 = t58+t73+t169+t319+t374+t392+t397+t469+t470+t475+t488+t497+t702+t703+t707+t873, t1205 = t83+t88+t180+t286+t376+t378+t383+t462+t467+t468+t474+t485+t709+t713+t714+t871;
    T t1206 = t64+t74+t189+t285+t375+t385+t389+t476+t484+t498+t501+t502+t704+t708+t712+t877;
    T t1207 = t122+t226+t228+t254+t270+t415+t446+t515+t528+t530+t537+t575+t579+t588+t597+t598+t677+t696+t773+t785+t786+t787+t788+t793+t797+t820+t832+t888+t951+t956+t966+t981+t987+t1023+t1043+t1097+t1107+t1109+t1117;
    T t1208 = t115+t227+t229+t233+t250+t448+t457+t517+t518+t520+t551+t555+t564+t577+t583+t614+t667+t695+t806+t836+t837+t838+t839+t842+t846+t859+t868+t883+t947+t959+t964+t972+t975+t1016+t1034+t1089+t1116+t1127+t1129;
    T t1209 = t123+t235+t255+t267+t268+t419+t438+t535+t589+t608+t612+t619+t620+t622+t624+t625+t673+t686+t777+t781+t796+t809+t810+t811+t812+t818+t826+t884+t933+t936+t939+t982+t984+t1050+t1061+t1083+t1113+t1121+t1123;
    T t1210 = t221+t222+t421+t431+t438+t442+t453+t516+t527+t529+t538+t549+t553+t569+t570+t571+t572+t592+t671+t697+t767+t768+t834+t844+t858+t885+t948+t952+t957+t965+t968+t1024+t1042+t1053+t1077+t1084+t1106+t1110+t1141;
    T t1211 = t257+t259+t415+t427+t434+t444+t450+t519+t541+t542+t543+t544+t557+t578+t584+t606+t610+t613+t678+t688+t765+t776+t803+t865+t866+t887+t937+t941+t942+t943+t946+t1018+t1026+t1032+t1091+t1098+t1114+t1131+t1140;
    T t1212 = t258+t260+t417+t422+t429+t432+t457+t539+t582+t586+t590+t599+t600+t603+t604+t621+t623+t626+t668+t687+t783+t789+t790+t821+t862+t882+t935+t940+t950+t958+t960+t1046+t1051+t1060+t1074+t1085+t1120+t1124+t1139;
    T t1213 = t121+t234+t236+t238+t253+t414+t416+t420+t550+t554+t576+t580+t587+t591+t615+t616+t665+t672+t693+t694+t772+t780+t807+t808+t831+t833+t889+t934+t938+t945+t955+t980+t983+t1022+t1055+t1073+t1076+t1082+t1108+t1118+t1138;
    T t1214 = t114+t220+t232+t237+t240+t445+t451+t456+t523+t524+t552+t556+t558+t563+t605+t609+t665+t681+t691+t692+t804+t805+t816+t824+t852+t853+t886+t944+t949+t953+t962+t970+t974+t1025+t1033+t1075+t1092+t1100+t1115+t1128+t1143;
    T t1215 = t124+t239+t241+t271+t274+t423+t433+t443+t536+t540+t573+t574+t581+t585+t607+t611+t672+t681+t689+t690+t769+t770+t782+t784+t856+t864+t881+t954+t961+t963+t967+t986+t989+t1045+t1052+t1078+t1086+t1101+t1111+t1125+t1142, t1174 = -t1166;
    T t1175 = -t1167, t1176 = -t1169, t1189 = t1144+t1155+t1170, t1190 = t1145+t1158+t1165, t1191 = t1146+t1153+t1168, t1192 = t1134+t1162+t1184, t1195 = t1133+t1163+t1185, t1196 = t1137+t1164+t1183, t1193 = t1132+t1173+t1174;
    T t1194 = t1136+t1171+t1175, t1197 = t1135+t1172+t1176;
    
    fdrv(0, 0) = t217*t1191+t1195*tx*ty+t1193*tx*tz; fdrv(0, 1) = -t216*t1195-t1191*tx*ty+t1193*ty*tz; fdrv(0, 2) = -t215*t1193-t1191*tx*tz+t1195*ty*tz;
    fdrv(0, 3) = t114*2.0+t116*2.0+t430+t455+t523+t565-t573+t769+t903-t904+t905-t907+t908+t909+t910-t915+t970+t971+t1078-t6*t83-t7*t83*4.0+t8*t82+t10*t82*4.0-t12*t80*2.0-t6*t88-t9*t87*2.0-t44*t60*8.0+t45*t62*5.0-t45*t69*8.0-t44*t72*5.0-t49*t68*2.0-t47*t79*2.0-t10*t142*6.0-t12*t144*2.0-t44*t142*6.0-t48*t139*4.0+t45*t292+t9*t489+t10*t501+t22*t561+t118*tz+t6*t22*t29+t6*t23*t32+t10*t22*t60*4.0-t7*t18*t69*4.0-t6*t22*t79*2.0-t7*t22*t79*4.0+t5*t80*tx*6.0-t5*t81*tx*6.0-oy*t48*tx*ty*6.0-oz*t46*tx*tz*6.0;
    fdrv(0, 4) = -t127+t242+t518+t790+t842+t972+t1059-t1065+t1074+t1087-t5*t62*2.0+t9*t61*1.0E+1-t7*t71*8.0-t5*t75-t5*t83*5.0-t10*t80*2.0-t9*t83*6.0-t5*t88+t11*t82-t7*t87*2.0+t12*t82*4.0-t9*t88*2.0-t48*t60*2.0+t44*t67*8.0-t48*t68*6.0-t46*t72*2.0+t47*t71*5.0-t48*t72+t44*t80*4.0-t46*t78*4.0-t49*t81*5.0-t5*t136*2.0-t7*t145*6.0+t7*t281+t7*t301+t5*t312+t5*t382+t12*t389+t7*t490+t46*t492+t22*t562+t119*tz+t977*tz-t5*t18*t69*6.0-t5*t22*t69*2.0+t14*t38*t44+t12*t22*t68*4.0-t5*t22*t79*6.0-t9*t22*t79*8.0+t29*t48*tx-t7*t81*tx*8.0-t10*t87*tx*2.0+t9*t80*ty*6.0-t12*t87*ty*6.0-oz*t44*tx*tz*4.0;
    fdrv(0, 5) = t120+t243+t522+t815+t854+t978+t1031+t1064+t1093+t5*t64-t5*t72*2.0+t5*t74-t10*t71*2.0+t13*t70*1.0E+1-t7*t81*2.0-t12*t83+t13*t82*6.0-t10*t87*8.0-t12*t88*5.0+t45*t59*8.0+t47*t59*8.0+t46*t62-t46*t69*2.0-t45*t81*4.0-t48*t79*4.0-t47*t81*4.0-t44*t87*4.0-t46*t86*6.0-t5*t142*6.0-t10*t140*6.0+t7*t158-t44*t143*4.0+t5*t189+t9*t189+t18*t222+t10*t275+t10*t302+t48*t286+t5*t389+t10*t466+t5*t476+t10*t481+t9*t501+t22*t529+t5*t708+t9*t712+t7*t876-t124*ty+t7*t19*t28+t12*t19*t36-t12*t22*t79*6.0+t10*t396*tx-t10*t142*ty*6.0+t12*t396*ty-oy*t44*tx*tz*4.0;
    fdrv(0, 6) = -t1209*tdy-t1212*tdz-tdx*(t262+t452-t557+t559+t594+t627+t628+t679+t680-t765-t773+t795+t835+t861+t1044+t1062+t1088-t1094+t1112+t1122-t9*t59*4.0+t6*t68*4.0-t13*t67*4.0-t5*t80*1.2E+1+t5*t81*1.2E+1+t8*t86*4.0-t45*t60*2.0E+1+t44*t69*2.0E+1-t47*t68*4.0-t12*t140*4.0+t9*t319+oy*t44*tx*2.4E+1-ox*t49*ty*4.0+oz*t45*tx*2.4E+1);
    fdrv(0, 7) = -tdy*(t125*2.0-t219+t248-t446+t452+t613+t679+t680-t776+t778-t827-t849+t1044+t1062+t1112+t1122-t5*t59*2.4E+1+t7*t60*4.0-t9*t59*4.0E+1-t5*t67*8.0+t7*t68*2.4E+1+t35*t45-t5*t80*4.0+t7*t78*4.0+t5*t81*2.0E+1-t9*t80*1.2E+1+t9*t81*3.0E+1+t12*t87*1.2E+1-t45*t60*4.0+t44*t69*1.2E+1-t47*t68*2.0E+1+t49*t79*2.0E+1+t7*t138*2.4E+1-t10*t136*4.0-t12*t140*1.2E+1+t9*t144*2.4E+1+t7*t310-ox*t48*tx*4.0)-t1209*tdx-t1215*tdz;
    fdrv(0, 8) = -t1212*tdx-t1215*tdy+tdz*(t122*2.0+t219+t246-t262+t427-t537-t679-t680+t820+t827+t849+t951+t1029+t1043-t1044+t1131+t5*t59*8.0+t5*t67*2.4E+1-t10*t69*4.0+t13*t67*4.0E+1+t5*t80*2.0E+1-t5*t81*4.0+t9*t80*2.0E+1-t10*t79*4.0+t13*t80*3.0E+1-t10*t86*2.4E+1-t12*t87*2.0E+1+t45*t60*1.2E+1-t44*t69*4.0+t47*t68*1.2E+1-t49*t79*1.2E+1-t7*t138*4.0+t10*t136*2.4E+1+t12*t140*2.4E+1-t9*t144*1.2E+1+t10*t286-oy*t44*tx*4.0+t7*t29*tx);
    fdrv(0, 9) = -tdx*(t85+t289+t313+t379+t386+t394+t633+t636+t918+t919+t996+t1103-t22*t59*4.0-t23*t60*6.0+t19*t69*6.0-t23*t68*2.0+t22*t186+t88*ty+t180*ty+t383*ty-t64*tz*5.0-t74*tz+t390*tz)-t1203*tdy*tx+t1206*tdz*tx;
    fdrv(0, 10) = -tdy*(t63*-1.0E+1+t85*6.0+t290+t386+t394+t633-t703+t918+t1066+t1103+t16*t37+t15*t40+t23*t33+t24*t37-t23*t60*2.0+t19*t69*8.0-t23*t68*6.0+t14*t319+t22*t397+t71*tx*8.0-t64*tz-t74*tz*5.0+t138*tz*6.0)-t1200*tdx*ty*2.0+t1206*tdz*ty;
    fdrv(0, 11) = -tdz*(t73*-1.0E+1-t84*6.0+t85+t314+t379+t636-t701+t919+t996+t1066+t15*t32+t19*t28-t23*t60*8.0-t23*t68*8.0+t22*t81*6.0-t18*t137*4.0+t19*t172+t87*tx*8.0+t75*ty+t88*ty*5.0+t136*ty*6.0-t64*tz*2.0-t74*tz*2.0)-t1200*tdx*tz*2.0-t1203*tdy*tz;
    fdrv(0, 12) = tdy*(t401+t503+t513+t514+t658+t729+t735+t11*t23*2.0+t14*t44*4.0+t22*t44*8.0+t9*t22*ty*1.2E+1)+tdz*(t403+t505+t509+t510+t649+t736+t750+t12*t19*2.0+t14*t45*4.0+t18*t45*8.0+t12*t22*ty*1.2E+1)+t217*tdx*(t46*2.0+t48*2.0+t5*tx*3.0+t6*ty+t8*tz)*2.0;
    fdrv(0, 13) = tdy*(t95+t152-t337-t352-t353+t361+t368+t398-t507+t641+t642+t643-t647-t663-t719+t739+t751+t757+t1008+t1010-t5*t22*tx*2.0)+tdx*(t92+t150-t326+t341+t359-t364-t365+t400-t504-t638+t650+t651+t652-t664-t729+t730+t737+t748+t1002+t1009-t9*t22*ty*2.0)+tdz*(t89+t93+t146+t148+t196+t340-t348-t349+t355+t360+t366-t662+t717+t725+t731+t752+t878+t879+t931+t932+t1000+t1007+t1011);
    fdrv(0, 14) = -tdy*(t89+t97+t147+t156+t194+t328+t335+t346+t347+t354+t370+t640+t654+t656+t721+t724+t746+t759+t929+t930+t1000+t1004+t1014)-tdx*(t90+t149+t195+t329+t336+t339+t350+t351+t506+t639+t655+t659+t660+t661+t720+t727+t738+t750+t755+t1001+t1003)-tdz*(t96+t154+t191+t332+t333+t357+t358+t369+t511+t641+t642+t643+t653+t657+t716+t728+t741+t747+t760+t1005+t1013);
    fdrv(0, 15) = t969*t1188*tx; fdrv(0, 16) = t969*t1188*ty; fdrv(0, 17) = t969*t1188*tz; fdrv(0, 18) = rdx*t969*t1188-t2*t969*t1149-t3*t969*t1149-t4*t969*t1149+t2*t1188*tx*2.0+t3*t1188*tx*2.0+t4*t1188*tx*2.0;
    fdrv(0, 19) = rdy*t969*t1188-t2*t969*t1181-t3*t969*t1181-t4*t969*t1181+t2*t1188*ty*2.0+t3*t1188*ty*2.0+t4*t1188*ty*2.0; fdrv(0, 20) = rdz*t969*t1188+t2*t969*t1182+t3*t969*t1182+t4*t969*t1182+t2*t1188*tz*2.0+t3*t1188*tz*2.0+t4*t1188*tz*2.0;
    fdrv(0, 24) = t217*t917*t969*2.0; fdrv(0, 25) = t917*(t25+t101+t103-t197-t198-t371+t409+t411+t763); fdrv(0, 26) = -t917*(t21+t99+t102+t199+t200+t372+t407+t412+t762); fdrv(1, 0) = -t217*t1197-t1190*tx*ty+t1192*tx*tz;
    fdrv(1, 1) = t216*t1190+t1197*tx*ty+t1192*ty*tz; fdrv(1, 2) = -t215*t1192+t1197*tx*tz-t1190*ty*tz;
    fdrv(1, 3) = t127+t244+t547+t764+t843-t972+t973+t1059+t1065+t1079+t5*t62*1.0E+1-t9*t61*2.0+t5*t75*6.0-t8*t73-t12*t70*2.0-t10*t73*4.0+t9*t83-t6*t87*2.0+t9*t88-t44*t56*8.0-t45*t57*5.0+t46*t60*8.0-t49*t59*2.0-t44*t67*4.0-t46*t70*4.0+t46*t72*5.0+t48*t72*5.0+t41*t83+t46*t78*8.0+t41*t88+t49*t81-t7*t145*4.0-t45*t128*6.0-t46*t131*4.0+t22*t260+t6*t305+t10*t309+t46*t316+t10*t396+t18*t429+t22*t429+t6*t477+t9*t474+t6*t500+t22*t539-t114*tz+t12*t23*t37-t10*t22*t56*4.0-t10*t22*t59*4.0-t5*t70*tx*6.0+t7*t81*tx*6.0+t10*t87*tx*6.0-t7*t61*tz*4.0-ox*t48*tx*ty*6.0-oz*t44*tx*tz*6.0;
    fdrv(1, 4) = t119*2.0+t121*2.0+t413+t455+t526+t615+t852+t904+t911+t914+t915+t916-t970+t977+t983+t1100-t7*t61*8.0-t10*t70*2.0+t7*t75-t11*t73-t12*t73*4.0-t5*t87*2.0+t7*t88-t46*t56*4.0-t48*t56*2.0-t45*t61*4.0-t48*t59*6.0-t47*t61*5.0-t46*t67*4.0-t44*t70*4.0-t45*t69*2.0-t45*t79*6.0-t47*t79*8.0-t10*t142*2.0-t44*t131*4.0+t7*t190+t22*t241+t44*t310+t5*t465+t22*t585+t7*t709+t10*t870-t115*tz+t5*t15*t39+t5*t19*t35+t7*t22*t30-t12*t22*t59*4.0-t7*t70*tx*2.0-t9*t70*ty*6.0-t7*t128*tz*6.0-oz*t46*tx*tz*4.0;
    fdrv(1, 5) = -t113+t246+t598+t793+t866+t987+t1015+t1094+t1098-t5*t63-t10*t61*2.0-t9*t63-t5*t73*5.0-t9*t73*5.0+t10*t75-t5*t81*2.0-t13*t73*6.0-t9*t81*2.0+t10*t83+t13*t80*1.0E+1+t10*t88*5.0-t12*t87*8.0+t45*t60*8.0-t48*t57*2.0-t49*t61*2.0-t44*t69*2.0+t45*t72*4.0-t44*t86*6.0-t10*t129*2.0-t7*t135*4.0-t10*t134*6.0-t46*t130*6.0-t48*t128*4.0+t57*t213+t5*t309+t5*t313+t7*t320+t7*t321+t9*t396+t7*t476+t7*t637+t10*t713+t10*t715+t672*tx+t10*t18*t28-t5*t18*t56*2.0-t5*t22*t56*6.0-t5*t22*t59*6.0-t9*t22*t59*6.0+t18*t35*t45+t10*t18*t172-t7*t61*tx*2.0-t10*t70*tx*8.0-t12*t70*ty*8.0-oz*t48*tx*ty*4.0;
    fdrv(1, 6) = -t1211*tdy-t1214*tdz+tdx*(t123*2.0+t247+t249-t261+t453-t625-t674-t675+t777+t799+t851+t936+t1056+t1061+t1106+t5*t60*4.0E+1-t6*t59*4.0-t6*t67*4.0-t5*t70*1.2E+1+t5*t72*3.0E+1-t9*t70*4.0+t9*t78*8.0+t7*t81*1.2E+1+t10*t87*1.2E+1-t45*t56*2.0E+1-t47*t59*4.0+t46*t69*2.0E+1+t48*t69*2.0E+1-t10*t130*1.2E+1+t5*t138*2.4E+1-t12*t134*4.0+t5*t142*2.4E+1+t6*t319-ox*t44*tx*2.4E+1+oy*t46*tx*2.4E+1-oz*t47*ty*4.0+t12*t38*ty-ox*t46*tx*tz*1.2E+1);
    fdrv(1, 7) = -tdy*(t261+t425+t521+t522+t531-t535+t596-t619+t674+t675+t766-t796-t844+t854+t867+t1017+t1027-t1080+t1093+t1126+t1130+t7*t56*4.0-t5*t60*4.0+t7*t59*2.4E+1+t9*t70*1.2E+1-t7*t81*2.0E+1-t13*t78*4.0+t11*t86*4.0+t47*t59*2.0E+1-t46*t69*1.2E+1+t10*t130*4.0+t5*t492-oy*t48*tx*4.0+oz*t47*ty*2.4E+1)-t1211*tdx-t1208*tdz;
    fdrv(1, 8) = -t1214*tdx-t1208*tdy-tdz*(t117*2.0+t243-t249-t419+t425+t516+t674+t675-t799-t851-t858+t860+t1017+t1027+t1126+t1130+t7*t59*4.0-t5*t68*4.0+t5*t70*2.0E+1+t9*t70*2.0E+1+t13*t70*3.0E+1-t9*t78*2.4E+1-t7*t81*4.0+t12*t79*4.0-t13*t78*4.0E+1-t10*t87*2.0E+1+t12*t86*2.4E+1+t45*t56*1.2E+1+t47*t59*1.2E+1-t46*t69*4.0-t48*t69*1.2E+1+t10*t128*4.0+t10*t130*2.4E+1-t5*t138*4.0+t12*t134*2.4E+1-t5*t142*1.2E+1-oy*t46*tx*8.0+t10*t35*ty);
    fdrv(1, 9) = -tdx*(t64*-1.0E+1+t76-t77*6.0+t278+t393+t630-t710+t920+t992+t1067+t19*t40+t23*t37+t23*t56*6.0-t18*t72*8.0-t22*t72*8.0-t22*t138*4.0+t23*t161-t83*tx*2.0-t88*tx*2.0+t57*ty*8.0+t58*tz*5.0+t63*tz+t139*tz*6.0)-t1199*tdy*tx*2.0-t1202*tdz*tx;
    fdrv(1, 10) = -tdy*(t76+t279+t320+t381+t388+t393+t630+t637+t920+t924+t992+t1104+t23*t56*2.0-t22*t60*4.0+t23*t59*6.0-t18*t72*6.0-t22*t72*2.0-t83*tx*5.0-t88*tx+t382*tx+t58*tz+t169*tz+t397*tz)+t1205*tdx*ty-t1202*tdz*ty;
    fdrv(1, 11) = -tdz*(t76*6.0-t82*1.0E+1+t321+t381+t388+t637+t924+t1067+t1104+t15*t27+t19*t26+t16*t31+t20*t31-t18*t60*2.0+t23*t56*8.0+t23*t59*8.0-t18*t72*2.0-t22*t72*6.0+t18*t285-t83*tx-t88*tx*5.0+t87*ty*8.0+t130*ty*6.0)+t1205*tdx*tz-t1199*tdy*tz*2.0;
    fdrv(1, 12) = -tdz*(t89+t93+t146+t148+t196+t340+t348+t349+t355+t360+t366+t640+t648+t662+t717+t725+t731+t752+t931+t932+t1000+t1007+t1011)-tdx*(t92+t150+t193+t326+t341+t359+t364+t365+t504+t638+t650+t651+t652+t664+t729+t730+t737+t743+t748+t1002+t1009)-tdy*(t95+t152+t191+t337+t352+t353+t361+t368+t507+t641+t642+t643+t647+t663+t719+t722+t739+t751+t757+t1008+t1010);
    fdrv(1, 13) = tdz*(t403+t505+t506+t509+t639+t736+t758+t10*t15*2.0+t18*t47*4.0+t10*t22*tx*1.2E+1+t46*tx*tz*8.0)+tdx*(t399+t508+t511+t512+t657+t719+t734+t7*t19*4.0+t8*t23*2.0+t22*t46*8.0+t5*t22*tx*1.2E+1)+t216*tdy*(t49*2.0+t7*tx+t9*ty*3.0+t41*ty+t11*tz)*2.0;
    fdrv(1, 14) = tdy*(t91+t151+t327+t342-t356+t402-t510-t645-t649+t659+t660+t661+t718+t733+t740-t744-t758+t998+t1006-t5*t23*2.0-t14*t45*2.0)+tdz*(t94+t153+t338+t362-t367+t400-t513-t644+t650+t651+t652-t658-t723+t726+t742-t749+t754+t999+t1012-t5*t19*2.0-t14*t44*2.0)+tdx*(t93+t97+t155+t157+t192+t325+t334+t343-t344-t345+t363-t646+t732+t745+t753+t756+t879+t880+t927+t928+t997+t1007+t1014);
    fdrv(1, 15) = -t969*tx*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181); fdrv(1, 16) = -t969*ty*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181); fdrv(1, 17) = -t969*tz*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181);
    fdrv(1, 18) = -rdx*t969*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181)-t2*tx*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181)*2.0-t3*tx*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181)*2.0-t4*tx*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181)*2.0+t2*t969*t1179+t3*t969*t1179+t4*t969*t1179;
    fdrv(1, 19) = -rdy*t969*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181)-t2*ty*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181)*2.0-t3*ty*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181)*2.0-t4*ty*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181)*2.0-t2*t969*t1148-t3*t969*t1148-t4*t969*t1148;
    fdrv(1, 20) = -rdz*t969*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181)-t2*tz*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181)*2.0-t3*tz*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181)*2.0-t4*tz*(t70-t72+t104+t112+t131+t135-t138-t142-t162-t181)*2.0-t2*t969*t1180-t3*t969*t1180-t4*t969*t1180;
    fdrv(1, 24) = -t917*(t25+t101+t103+t197+t198+t371+t409+t411+t763); fdrv(1, 25) = t216*t917*t969*2.0; fdrv(1, 26) = t917*(t17+t98+t100-t201-t202-t373+t408+t410+t761); fdrv(2, 0) = -t217*t1196+t1194*tx*ty-t1189*tx*tz;
    fdrv(2, 1) = -t216*t1194+t1196*tx*ty-t1189*ty*tz; fdrv(2, 2) = t215*t1189+t1196*tx*tz+t1194*ty*tz;
    fdrv(2, 3) = -t120+t247+t619+t767+t818+t982+t1031-t1064+t1080+t1084-t5*t64*6.0+t6*t63+t7*t63*4.0+t5*t72*1.0E+1-t5*t74*2.0+t6*t73-t9*t70*2.0-t13*t70*2.0-t6*t80*2.0-t5*t82*2.0-t9*t82-t13*t82+t44*t57*5.0-t45*t56*8.0-t45*t59*4.0-t47*t59*2.0-t46*t62*5.0-t48*t62*5.0+t48*t69*8.0+t48*t79*8.0-t6*t141*2.0-t10*t140*4.0+t44*t130*6.0-t48*t132*6.0+t71*t214+t48*t277+t7*t319+t7*t397+t12*t474+t10*t23*t29+t10*t19*t34+t6*t22*t37+t7*t18*t56*4.0-t5*t18*t60*8.0-t5*t22*t60*8.0+t7*t22*t59*4.0-t5*t22*t68*4.0-t9*t22*t68*2.0+t6*t22*t161+t5*t61*tx*6.0-t7*t71*tx*6.0-t10*t80*tx*6.0-t12*t80*ty*2.0-ox*t46*tx*tz*6.0-oy*t44*tx*tz*6.0;
    fdrv(2, 4) = t113+t248+t627+t794+t835-t987+t988+t1015+t1063+t1088-t7*t64+t9*t63*6.0-t7*t70*2.0+t5*t73-t7*t74*5.0-t5*t80*2.0+t13*t73-t7*t82+t9*t81*1.0E+1-t13*t80*2.0-t45*t60*2.0+t48*t57-t44*t62*4.0+t49*t61*5.0+t44*t69*8.0-t45*t68*6.0+t42*t73-t47*t68*8.0-t48*t71*4.0+t49*t79*8.0-t7*t131*2.0-t7*t135*6.0-t10*t134*4.0-t10*t136*2.0+t5*t169+t48*t128*6.0-t44*t136*4.0-t48*t133*4.0+t5*t319+t22*t427+t5*t469+t7*t630+t5*t702+t681*tx+t7*t15*t27+t10*t23*t26+t7*t22*t33-t7*t18*t60*6.0-t7*t22*t60*2.0-t7*t22*t68*6.0+t22*t41*t56+t7*t382*tx+t12*t70*ty*6.0-t10*t80*ty*4.0-oy*t46*tx*tz*4.0;
    fdrv(2, 5) = t124*2.0+t126*2.0+t413+t430+t574+t618+t808+t900+t901+t902+t907-t908+t912-t983+t989+t990+t1073-t7*t61*2.0-t10*t64+t12*t63-t5*t71*2.0-t10*t70*8.0-t10*t74-t10*t82*5.0-t12*t80*8.0-t46*t56*2.0-t44*t60*2.0-t45*t62*4.0-t48*t59*4.0-t46*t67*6.0-t49*t68*4.0-t44*t78*6.0-t7*t132*2.0-t45*t132*4.0-t46*t137*4.0+t12*t179+t5*t458+t7*t494+t18*t532+t12*t707+t117*ty+t978*ty+t10*t19*t26+t9*t23*t29+t9*t19*t34+t10*t18*t35-t10*t18*t60*2.0-t10*t22*t60*6.0-t10*t22*t68*6.0-t5*t80*tx*4.0-oy*t48*tx*ty*4.0;
    fdrv(2, 6) = -t1213*tdy-t1207*tdz-tdx*(t118*2.0+t244-t273+t424-t448+t590+t669+t670-t801+t819-t821-t830+t1041+t1054+t1105+t1119-t5*t61*1.2E+1+t5*t62*3.0E+1-t5*t69*4.0E+1+t8*t67*4.0+t7*t71*1.2E+1-t12*t70*4.0+t10*t80*1.2E+1+t12*t78*4.0-t9*t86*4.0-t44*t56*2.0E+1+t46*t60*2.0E+1+t48*t60*2.0E+1-t49*t59*4.0-t7*t128*1.2E+1+t5*t132*2.4E+1+t5*t136*2.4E+1-t9*t134*4.0+t6*t140*4.0+ox*t45*tx*2.4E+1-oz*t48*tx*2.4E+1-oz*t49*ty*8.0+t6*t29*tz);
    fdrv(2, 7) = -t1213*tdx-t1210*tdz+tdy*(t115*2.0-t223+t242+t273+t429-t669-t670+t801+t830+t859+t964+t1016+t1040-t1054+t1120+t5*t61*2.0E+1+t9*t61*3.0E+1-t10*t67*4.0-t7*t71*2.0E+1+t12*t70*1.2E+1+t9*t79*4.0E+1-t11*t78*4.0-t10*t80*4.0+t5*t86*4.0+t44*t56*1.2E+1-t46*t60*1.2E+1-t48*t60*4.0+t49*t59*2.0E+1+t7*t128*2.4E+1-t5*t132*1.2E+1+t7*t130*4.0-t5*t136*4.0+t9*t134*2.4E+1-ox*t45*tx*4.0-oy*t47*ty*2.4E+1+oz*t48*tx*8.0+oz*t49*ty*2.4E+1-oy*t7*tx*tz*4.0);
    fdrv(2, 8) = -tdz*(t223+t424+t533-t539+t547+t548+t562+t669+t670+t764+t774+t845-t862-t868+t1041+t1054+t1079-t1087+t1105+t1119+t10*t56*4.0-t5*t69*4.0+t10*t67*2.4E+1+t12*t68*4.0-t12*t70*2.0E+1-t9*t79*4.0+t10*t80*2.0E+1+t12*t78*2.4E+1-t44*t56*4.0+t48*t60*1.2E+1-t49*t59*1.2E+1-t7*t128*4.0+t5*t286-oz*t46*tx*4.0)-t1207*tdx-t1210*tdy;
    fdrv(2, 9) = -tdx*(t66*6.0-t75*1.0E+1+t287+t380+t395+t631+t925+t1068+t1102+t20*t28+t24*t28+t19*t36+t23*t32-t19*t56*6.0-t22*t61*2.0-t22*t79*2.0+t22*t286+t18*t383+t22*t383-t58*ty*5.0-t73*ty+t57*tz*8.0+t133*tz*6.0)+t1204*tdy*tx-t1198*tdz*tx*2.0;
    fdrv(2, 10) = -tdy*(t65*-6.0+t66-t83*1.0E+1+t311+t387+t634+t926+t993+t1068+t15*t31+t23*t26-t19*t56*8.0+t18*t62*6.0-t22*t61*8.0-t22*t69*2.0+t22*t166+t22*t459+t74*tx*5.0+t82*tx-t58*ty*2.0-t73*ty*2.0+t71*tz*8.0+t128*tz*6.0)-t1201*tdx*ty-t1198*tdz*ty*2.0;
    fdrv(2, 11) = -tdz*(t66+t288+t312+t380+t387+t395+t631+t634+t925+t926+t993+t1102-t19*t56*2.0-t22*t61*6.0+t22*t62*6.0-t18*t69*4.0+t18*t166+t74*tx+t189*tx+t389*tx-t58*ty-t73*ty*5.0+t396*ty)-t1201*tdx*tz+t1204*tdy*tz;
    fdrv(2, 12) = tdx*(t90+t149+t329-t336+t339+t402-t506-t639-t655+t659+t660+t661+t720+t727+t738-t750-t755+t1001+t1003-t9*t23*2.0-t18*t47*2.0)+tdz*(t96+t154-t332-t333-t357+t358+t369+t398-t511+t641+t642+t643-t653-t657-t728+t741+t747+t760+t1005+t1013-t5*t18*tx*2.0)+tdy*(t89+t97+t147+t156+t194+t328+t335-t346-t347+t354+t370-t654+t721+t724+t746+t759+t878+t880+t929+t930+t1000+t1004+t1014);
    fdrv(2, 13) = -tdx*(t93+t97+t155+t157+t192+t325+t334+t343+t344+t345+t363+t646+t648+t656+t732+t745+t753+t756+t927+t928+t997+t1007+t1014)-tdy*(t91+t151+t195+t327+t330+t331+t342+t356+t510+t645+t649+t659+t660+t661+t718+t733+t740+t744+t758+t998+t1006)-tdz*(t94+t153+t193+t323+t324+t338+t362+t367+t513+t644+t650+t651+t652+t658+t723+t726+t742+t749+t754+t999+t1012);
    fdrv(2, 14) = tdy*(t401+t503+t504+t514+t638+t735+t749+t7*t15*2.0+t12*t23*4.0+t7*t18*tx*1.2E+1+t48*tx*ty*8.0)+tdx*(t399+t507+t508+t512+t647+t728+t734+t6*t19*2.0+t10*t23*4.0+t18*t48*8.0+t5*t18*tx*1.2E+1)+t215*tdz*(t10*tx+t12*ty+t13*tz*3.0+t41*tz+t42*tz)*2.0;
    fdrv(2, 15) = t969*t1186*tx; fdrv(2, 16) = t969*t1186*ty; fdrv(2, 17) = t969*t1186*tz; fdrv(2, 18) = rdx*t969*t1186-t2*t969*t1177-t3*t969*t1177-t4*t969*t1177+t2*t1186*tx*2.0+t3*t1186*tx*2.0+t4*t1186*tx*2.0;
    fdrv(2, 19) = rdy*t969*t1186+t2*t969*t1178+t3*t969*t1178+t4*t969*t1178+t2*t1186*ty*2.0+t3*t1186*ty*2.0+t4*t1186*ty*2.0; fdrv(2, 20) = rdz*t969*t1186-t2*t969*t1147-t3*t969*t1147-t4*t969*t1147+t2*t1186*tz*2.0+t3*t1186*tz*2.0+t4*t1186*tz*2.0;
    fdrv(2, 24) = t917*(t21+t99+t102-t199-t200-t372+t407+t412+t762); fdrv(2, 25) = -t917*(t17+t98+t100+t201+t202+t373+t408+t410+t761); fdrv(2, 26) = t215*t917*t969*2.0;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f35(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*ty, t3 = oy*tx, t4 = ox*tz, t5 = oz*tx, t6 = oy*tz, t7 = oz*ty, t8 = rdx*tx, t9 = rdy*ty, t10 = rdz*tz, t11 = tx*tx, t12 = tx*tx*tx, t13 = ty*ty, t14 = ty*ty*ty, t15 = tz*tz, t16 = tz*tz*tz, t17 = rx*tx*3.0, t18 = rx*tx*4.0;
    T t19 = ry*ty*3.0, t20 = ry*ty*4.0, t21 = rz*tz*3.0, t22 = rz*tz*4.0, t23 = rx*tx*ty, t24 = rx*tx*tz, t25 = ry*tx*ty, t26 = ry*ty*tz, t27 = rz*tx*tz, t28 = rz*ty*tz, t29 = -t3, t30 = -t5, t31 = -t7, t32 = ox*t11, t33 = t2*ty, t34 = t3*tx;
    T t35 = t4*tz, t36 = oy*t13, t37 = t5*tx, t38 = t6*tz, t39 = t7*ty, t40 = oz*t15, t41 = rx*t12, t42 = rx*t13, t43 = ry*t11, t44 = rx*t15, t45 = rz*t11, t46 = ry*t14, t47 = ry*t15, t48 = rz*t13, t49 = rz*t16, t50 = t2*tx*4.0, t51 = t4*tx*4.0;
    T t52 = t3*ty*4.0, t53 = t6*ty*4.0, t54 = t5*tz*4.0, t55 = t7*tz*4.0, t56 = t23*2.0, t57 = t18*ty, t58 = t24*2.0, t59 = t25*2.0, t60 = t18*tz, t61 = t20*tx, t62 = t26*2.0, t63 = t27*2.0, t64 = t20*tz, t65 = t22*tx, t66 = t28*2.0, t67 = t22*ty;
    T t71 = rx*t3*t11, t75 = ry*t2*t13, t77 = rx*t5*t11, t81 = rz*t4*t15, t83 = ry*t7*t13, t86 = rz*t6*t15, t97 = rx*t2*tz*2.0, t98 = rx*t3*tz*4.0, t99 = rx*t5*ty*4.0, t100 = ry*t3*tz*2.0, t101 = ry*t2*tz*4.0, t104 = t5*t20, t105 = rz*t5*ty*2.0;
    T t106 = t2*t22, t107 = t3*t22, t116 = rx*t11*5.0, t118 = ry*t13*5.0, t120 = rz*t15*5.0, t134 = t11+t13, t135 = t11+t15, t136 = t13+t15, t161 = rx*t2*tx*1.0E+1, t162 = rx*t4*tx*1.0E+1, t164 = ry*t3*ty*1.0E+1, t166 = rx*t5*tz*-4.0;
    T t168 = ry*t6*ty*1.0E+1, t169 = ry*t7*tz*-4.0, t170 = rz*t5*tz*1.0E+1, t171 = rz*t7*tz*1.0E+1, t172 = t2*t23, t173 = t4*t24, t178 = t3*t25, t183 = t6*t26, t188 = t5*t27, t189 = t7*t28, t228 = t5*t25*5.0, t231 = t2*t28*5.0, t234 = t3*t27*5.0;
    T t243 = ry*t2*tx*tz*-4.0, t244 = rx*t3*ty*tz*-4.0, t245 = t2*t27*-4.0, t247 = t3*t28*-4.0, t248 = t5*t26*-4.0, t249 = rx*t2*t11*-5.0, t250 = t2*t25*-4.0, t251 = t3*t23*-4.0, t252 = rx*t4*t11*-5.0, t255 = ry*t3*t13*-5.0, t258 = t4*t27*-4.0;
    T t259 = t5*t24*-4.0, t262 = ry*t6*t13*-5.0, t263 = t6*t28*-4.0, t264 = t7*t26*-4.0, t265 = rz*t5*t15*-5.0, t266 = rz*t7*t15*-5.0, t276 = t17+t19, t277 = t17+t21, t278 = t19+t21, t279 = t8+t9+t10, t68 = ox*t41, t69 = rx*t33, t70 = t2*t42;
    T t72 = rx*t35, t73 = ry*t34, t74 = t4*t44, t78 = oy*t46, t79 = ry*t38, t80 = rz*t37, t82 = t6*t47, t85 = rz*t39, t88 = oz*t49, t89 = t42*tx, t90 = t44*tx, t91 = t43*ty, t92 = t47*ty, t93 = t45*tz, t94 = t48*tz, t95 = ry*t50, t96 = rx*t52;
    T t102 = rz*t51, t108 = rz*t53, t110 = t33*5.0, t111 = t34*5.0, t112 = t35*5.0, t113 = t37*5.0, t114 = t38*5.0, t115 = t39*5.0, t117 = t41*6.0, t119 = t46*6.0, t121 = t49*6.0, t122 = -t50, t123 = -t51, t124 = -t52, t125 = -t53, t126 = -t54;
    T t127 = -t55, t128 = -t56, t129 = -t58, t130 = -t59, t131 = -t62, t132 = -t63, t133 = -t66, t137 = rx*t32*3.0, t138 = rx*t32*5.0, t139 = t71*6.0, t140 = t75*6.0, t141 = t77*6.0, t142 = ry*t36*3.0, t143 = ry*t36*5.0, t144 = t81*6.0;
    T t145 = t83*6.0, t146 = t86*6.0, t147 = rz*t40*3.0, t148 = rz*t40*5.0, t163 = -t97, t165 = -t100, t167 = -t105, t174 = t2*t43, t175 = t3*t42, t176 = t2*t44, t179 = t4*t45, t182 = t5*t44, t186 = t6*t48, t187 = t7*t47, t198 = t29*t43;
    T t200 = rz*t30*tx, t202 = t30*t45, t203 = rz*t31*ty, t204 = t31*t48, t207 = t172*5.0, t213 = t173*5.0, t215 = t3*t44*2.0, t216 = t5*t42*2.0, t217 = t3*t44*5.0, t219 = t178*5.0, t220 = t5*t42*5.0, t221 = t2*t47*2.0, t224 = t5*t59;
    T t227 = t2*t47*5.0, t229 = t2*t66, t230 = t3*t63, t232 = t183*5.0, t240 = t188*5.0, t242 = t189*5.0, t246 = t166*ty, t256 = t29*t47, t257 = ry*t29*tx*tz, t260 = t30*t48, t267 = rx*t134, t268 = rx*t135, t269 = ry*t134, t270 = ry*t136;
    T t271 = rz*t135, t272 = rz*t136, t273 = t2+t29, t274 = t4+t30, t275 = t6+t31, t280 = t15+t134, t281 = t276*tx, t282 = t276*ty, t283 = t277*tx, t284 = t277*tz, t285 = t278*ty, t286 = t278*tz, t312 = t42+t44+t61+t65+t116;
    T t313 = t43+t47+t57+t67+t118, t314 = t45+t48+t60+t64+t120, t149 = t89*2.0, t150 = t89*5.0, t151 = t90*2.0, t152 = t91*2.0, t153 = t90*5.0, t154 = t91*5.0, t155 = t92*2.0, t156 = t93*2.0, t157 = t92*5.0, t158 = t93*5.0, t159 = t94*2.0;
    T t160 = t94*5.0, t177 = t69*tz, t196 = -t70, t197 = -t74, t199 = -t79, t201 = -t82, t205 = -t147, t206 = -t148, t209 = t174*2.0, t210 = t175*2.0, t222 = t179*2.0, t223 = t182*2.0, t235 = t186*2.0, t236 = t187*2.0, t253 = -t176;
    T t261 = t200*ty, t287 = t280*t280, t288 = -t281, t289 = -t282, t290 = -t283, t291 = -t284, t292 = -t285, t293 = -t286, t303 = t32+t35+t110+t124, t304 = t32+t33+t112+t126, t305 = t36+t38+t111+t122, t306 = t34+t36+t114+t127;
    T t307 = t39+t40+t113+t123, t308 = t37+t40+t115+t125, t315 = t280*t312*tdx*ty, t316 = t280*t312*tdx*tz, t317 = t280*t313*tdy*tx, t318 = t280*t313*tdy*tz, t319 = t280*t314*tdz*tx, t320 = t280*t314*tdz*ty, t254 = -t177, t294 = ox*t8*t287;
    T t295 = rdx*t3*t287, t296 = rdy*t2*t287, t297 = rdx*t5*t287, t298 = oy*t9*t287, t299 = rdz*t4*t287, t300 = rdy*t7*t287, t301 = rdz*t6*t287, t302 = oz*t10*t287, t309 = t279*t287*tx, t310 = t279*t287*ty, t311 = t279*t287*tz;
    T t321 = t25+t132+t267+t268+t288, t322 = t27+t130+t267+t268+t290, t323 = t23+t133+t269+t270+t289, t324 = t28+t128+t269+t270+t292, t325 = t24+t131+t271+t272+t291, t326 = t26+t129+t271+t272+t293, t348 = t41+t46+t89+t91+t121+t153+t156+t157+t159;
    T t349 = t41+t49+t90+t93+t119+t150+t152+t155+t160, t350 = t46+t49+t92+t94+t117+t149+t151+t154+t158, t363 = t68+t81+t140+t173+t179+t198+t207+t209+t221+t231+t247+t251+t255+t256;
    T t364 = t78+t86+t139+t183+t186+t196+t210+t215+t219+t234+t245+t249+t250+t253, t365 = t68+t75+t144+t172+t174+t202+t213+t222+t227+t229+t248+t259+t260+t265, t367 = t71+t78+t146+t175+t178+t204+t217+t230+t232+t235+t246+t261+t264+t266;
    T t368 = t77+t88+t145+t182+t188+t201+t220+t224+t236+t242+t244+t257+t262+t263, t327 = t321*tx*tz, t328 = t322*tx*ty, t329 = t321*ty*tz, t330 = t322*ty*tz, t331 = t323*tx*tz, t332 = t323*ty*tz, t333 = t324*tx*ty, t334 = t324*tx*tz;
    T t335 = t325*tx*ty, t336 = t326*tx*ty, t337 = t325*ty*tz, t338 = t326*tx*tz, t342 = t134*t321, t343 = t135*t322, t344 = t134*t323, t345 = t136*t324, t346 = t135*t325, t347 = t136*t326, t351 = t280*t350*tdx, t352 = t280*t349*tdy;
    T t353 = t280*t348*tdz, t366 = t83+t88+t141+t187+t189+t197+t216+t223+t228+t240+t243+t252+t254+t258, t339 = -t334, t340 = -t335, t341 = -t336, t357 = t333+t343, t358 = t328+t345, t359 = t338+t342, t360 = t327+t347, t361 = t337+t344;
    T t362 = t332+t346, t354 = t330+t339, t355 = t329+t341, t356 = t331+t340;
    
    fdrv(0, 0) = -t136*t356-t360*tx*ty+t358*tx*tz; fdrv(0, 1) = t135*t360+t356*tx*ty+t358*ty*tz; fdrv(0, 2) = -t134*t358+t356*tx*tz-t360*ty*tz; fdrv(0, 3) = -t275*t280*t312;
    fdrv(0, 4) = t280*t368; fdrv(0, 5) = -t280*t367;
    fdrv(0, 6) = t368*tdy*tx*2.0-t367*tdz*tx*2.0-t280*tdz*(t107+t167+rx*t36+rx*t114+t3*t17+ry*t3*ty*2.0-rx*t7*tz*4.0)+t280*tdy*(t104+t165+rx*t40+rx*t115+t5*t17-rx*t6*ty*4.0+rz*t5*tz*2.0)-t275*t312*tdx*tx*2.0-t275*t280*tdx*(t20+t22+rx*tx*1.0E+1);
    fdrv(0, 7) = -t280*tdz*(t73+t79*5.0-t85*3.0+t108+t142+t166+t200+t206+rx*t3*ty*2.0-ry*t7*tz*8.0)+t280*tdy*(-t98-t168+t171+ry*t37*2.0+ry*t39*1.8E+1+ry*t40*2.0-rz*t38*4.0+rx*t5*ty*1.0E+1)+t368*tdy*ty*2.0-t367*tdz*ty*2.0+oz*t280*t312*tdx-t275*t312*tdx*ty*2.0-t275*t280*tdx*(rx*ty*2.0+ry*tx*4.0);
    fdrv(0, 8) = -t280*tdy*(t73+t79*3.0-t85*5.0+t96+t143+t169+t200+t205-rx*t5*tz*2.0+rz*t6*ty*8.0)-t280*tdz*(-t99+t168-t171-ry*t39*4.0+rz*t34*2.0+rz*t36*2.0+rz*t38*1.8E+1+rx*t3*tz*1.0E+1)+t368*tdy*tz*2.0-t367*tdz*tz*2.0-oy*t280*t312*tdx-t275*t312*tdx*tz*2.0-t275*t280*tdx*(rx*tz*2.0+rz*tx*4.0);
    fdrv(0, 9) = t280*t308*tdy*tx-t280*t306*tdz*tx-t275*t280*tdx*(t11*3.0+t134+t135); fdrv(0, 10) = -t280*tdy*(t6*t13*5.0-t7*t13*6.0+t6*t15-t7*t15*2.0-t37*ty*2.0+t34*tz)-t280*t306*tdz*ty-t275*t280*tdx*tx*ty*4.0;
    fdrv(0, 11) = -t280*tdz*(t6*t13*2.0+t6*t15*6.0-t7*t15*5.0+t13*t31+t34*tz*2.0+t30*tx*ty)+t280*t308*tdy*tz-t275*t280*tdx*tx*tz*4.0; fdrv(0, 13) = -t316-t318-t353; fdrv(0, 14) = t315+t320+t352; fdrv(0, 15) = -t275*t287*tx; fdrv(0, 16) = -t275*t287*ty;
    fdrv(0, 17) = -t275*t287*tz; fdrv(0, 18) = -t275*t280*(rdx*t13+rdx*t15+t8*tx*5.0+t9*tx*4.0+t10*tx*4.0); fdrv(0, 19) = t297+t300+t302-rdy*t275*t287-t8*t275*t280*ty*4.0-t9*t275*t280*ty*4.0-t10*t275*t280*ty*4.0;
    fdrv(0, 20) = -t298-t301+rdx*t29*t287-rdz*t275*t287-t8*t275*t280*tz*4.0-t9*t275*t280*tz*4.0-t10*t275*t280*tz*4.0; fdrv(0, 25) = -t311; fdrv(0, 26) = t310; fdrv(1, 0) = -t136*t362-t355*tx*ty-t357*tx*tz; fdrv(1, 1) = t135*t355+t362*tx*ty-t357*ty*tz;
    fdrv(1, 2) = t134*t357+t362*tx*tz-t355*ty*tz; fdrv(1, 3) = -t280*t366; fdrv(1, 4) = t274*t280*t313; fdrv(1, 5) = t280*t365;
    fdrv(1, 6) = t280*tdz*(t69+t72*5.0-t80*3.0+t102+t137+t169+t203+t206+ry*t2*tx*2.0-rx*t5*tz*8.0)-t280*tdx*(-t101-t162+t170+rx*t37*1.8E+1+rx*t39*2.0+rx*t40*2.0-rz*t35*4.0+ry*t5*ty*1.0E+1)-t366*tdx*tx*2.0+t365*tdz*tx*2.0-oz*t280*t313*tdy+t274*t313*tdy*tx*2.0+t274*t280*tdy*(rx*ty*4.0+ry*tx*2.0);
    fdrv(1, 7) = t366*tdx*ty*-2.0+t365*tdz*ty*2.0+t280*tdz*(t106+t167+ry*t32+ry*t112+t2*t19+rx*t2*tx*2.0-ry*t5*tz*4.0)-t280*tdx*(t99+t163+ry*t40+ry*t113+t7*t19-ry*t4*tx*4.0+rz*t7*tz*2.0)+t274*t313*tdy*ty*2.0+t274*t280*tdy*(t18+t22+ry*ty*1.0E+1);
    fdrv(1, 8) = t280*tdx*(t69+t72*3.0-t80*5.0+t95+t138+t166+t203+t205+rz*t4*tx*8.0-ry*t7*tz*2.0)+t280*tdz*(t162-t170-rx*t37*4.0+rz*t32*2.0+rz*t33*2.0+rz*t35*1.8E+1-ry*t5*ty*4.0+ry*t2*tz*1.0E+1)-t366*tdx*tz*2.0+t365*tdz*tz*2.0+ox*t280*t313*tdy+t274*t313*tdy*tz*2.0+t274*t280*tdy*(ry*tz*2.0+rz*ty*4.0);
    fdrv(1, 9) = t280*tdx*(t4*t11*5.0-t5*t11*6.0-t5*t13*2.0+t4*t15-t5*t15*2.0+t33*tz)+t280*t304*tdz*tx+t274*t280*tdy*tx*ty*4.0; fdrv(1, 10) = -t280*t307*tdx*ty+t280*t304*tdz*ty+t274*t280*tdy*(t13*3.0+t134+t136);
    fdrv(1, 11) = t280*tdz*(t4*t11*2.0+t4*t15*6.0-t5*t15*5.0+t11*t30+t13*t30+t33*tz*2.0)-t280*t307*tdx*tz+t274*t280*tdy*ty*tz*4.0; fdrv(1, 12) = t316+t318+t353; fdrv(1, 14) = -t317-t319-t351; fdrv(1, 15) = t274*t287*tx; fdrv(1, 16) = t274*t287*ty;
    fdrv(1, 17) = t274*t287*tz; fdrv(1, 18) = -t302+rdx*t30*t287+rdx*t274*t287+rdy*t31*t287+t8*t274*t280*tx*4.0+t9*t274*t280*tx*4.0+t10*t274*t280*tx*4.0; fdrv(1, 19) = t274*t280*(rdy*t11+rdy*t15+t8*ty*4.0+t9*ty*5.0+t10*ty*4.0);
    fdrv(1, 20) = t294+t296+t299+rdz*t274*t287+t8*t274*t280*tz*4.0+t9*t274*t280*tz*4.0+t10*t274*t280*tz*4.0; fdrv(1, 24) = t311; fdrv(1, 26) = -t309; fdrv(2, 0) = t136*t361+t359*tx*ty+t354*tx*tz; fdrv(2, 1) = -t135*t359-t361*tx*ty+t354*ty*tz;
    fdrv(2, 2) = -t134*t354-t361*tx*tz+t359*ty*tz; fdrv(2, 3) = t280*t364; fdrv(2, 4) = -t280*t363; fdrv(2, 5) = -t273*t280*t314;
    fdrv(2, 6) = -t280*tdy*(t69*5.0+t72-t73*3.0+t95+t137-t143+t199-rx*t3*ty*8.0+rz*t4*tx*2.0-rz*t6*ty*4.0)+t280*tdx*(-t161+t164+rx*t34*1.8E+1+rx*t36*2.0+rx*t38*2.0-ry*t33*4.0-rz*t2*tz*4.0+rz*t3*tz*1.0E+1)+t364*tdx*tx*2.0-t363*tdy*tx*2.0+oy*t280*t314*tdz-t273*t314*tdz*tx*2.0-t273*t280*tdz*(rx*tz*4.0+rz*tx*2.0);
    fdrv(2, 7) = -t280*tdx*(t69*3.0+t72-t73*5.0+t102+t138-t142+t199+ry*t2*tx*8.0-rx*t3*ty*4.0-rz*t6*ty*2.0)-t280*tdy*(t161-t164-rx*t34*4.0+ry*t32*2.0+ry*t33*1.8E+1+ry*t35*2.0+rz*t2*tz*1.0E+1-rz*t3*tz*4.0)+t364*tdx*ty*2.0-t363*tdy*ty*2.0-ox*t280*t314*tdz-t273*t314*tdz*ty*2.0-t273*t280*tdz*(ry*tz*4.0+rz*ty*2.0);
    fdrv(2, 8) = t364*tdx*tz*2.0-t363*tdy*tz*2.0-t280*tdy*(t101+t165+rz*t32+rz*t110+t4*t21+rx*t4*tx*2.0-rz*t3*ty*4.0)+t280*tdx*(t98+t163+rz*t36+rz*t111+t6*t21-rz*t2*tx*4.0+ry*t6*ty*2.0)-t273*t314*tdz*tz*2.0-t273*t280*tdz*(t18+t20+rz*tz*1.0E+1);
    fdrv(2, 9) = -t280*tdx*(t2*t11*5.0-t3*t11*6.0+t2*t13-t3*t13*2.0+t2*t15-t3*t15*2.0)-t280*t303*tdy*tx-t273*t280*tdz*tx*tz*4.0;
    fdrv(2, 10) = -t280*tdy*(t2*t11*2.0+t2*t13*6.0-t3*t13*5.0+t2*t15*2.0+t11*t29+t15*t29)+t280*t305*tdx*ty-t273*t280*tdz*ty*tz*4.0; fdrv(2, 11) = t280*t305*tdx*tz-t280*t303*tdy*tz-t273*t280*tdz*(t15*4.0+t280); fdrv(2, 12) = -t315-t320-t352;
    fdrv(2, 13) = t317+t319+t351; fdrv(2, 15) = -t273*t287*tx; fdrv(2, 16) = -t273*t287*ty; fdrv(2, 17) = -t273*t287*tz; fdrv(2, 18) = t295+t298+t301-rdx*t273*t287-t8*t273*t280*tx*4.0-t9*t273*t280*tx*4.0-t10*t273*t280*tx*4.0;
    fdrv(2, 19) = -t294-t296-t299-rdy*t273*t287-t8*t273*t280*ty*4.0-t9*t273*t280*ty*4.0-t10*t273*t280*ty*4.0; fdrv(2, 20) = -t273*t280*(rdz*t11+rdz*t13+t8*tz*4.0+t9*tz*4.0+t10*tz*5.0); fdrv(2, 24) = -t310; fdrv(2, 25) = t309;
    
    return fdrv;
}

template <typename T>
Matrix<T, 3, 27> SE3Qp<T>::f36(const Vec3T &The, const Vec3T &Rho, const Vec3T &Thed, const Vec3T &Rhod, const Vec3T &Omg)
{                                                  
    T tx = The(0), ty = The(1), tz = The(2);       
    T rx = Rho(0), ry = Rho(1), rz = Rho(2);       
    T ox = Omg(0), oy = Omg(1), oz = Omg(2);       
                                                   
    T tdx = Thed(0), tdy = Thed(1), tdz = Thed(2); 
    T rdx = Rhod(0), rdy = Rhod(1), rdz = Rhod(2); 
                                                   
    Matrix<T, 3, 27> fdrv; fdrv.setZero();         

    T t2 = ox*tx, t3 = ox*ty, t4 = oy*tx, t5 = ox*tz, t6 = oy*ty, t7 = oz*tx, t8 = oy*tz, t9 = oz*ty, t10 = oz*tz, t11 = rdx*tx, t12 = rdy*ty, t13 = rdz*tz, t14 = rx*tx, t15 = ry*ty, t16 = rz*tz, t17 = tx*tx, t18 = tx*tx*tx, t21 = ty*ty;
    T t22 = ty*ty*ty, t25 = tz*tz, t26 = tz*tz*tz, t44 = rx*ty*tz, t45 = ry*tx*tz, t46 = rz*tx*ty, t19 = t17*t17, t23 = t21*t21, t27 = t25*t25, t29 = t3*2.0, t30 = t4*2.0, t31 = t5*2.0, t32 = t7*2.0, t33 = t8*2.0, t34 = t9*2.0, t35 = t2*ty;
    T t36 = t2*tz, t37 = t4*ty, t38 = t6*tz, t39 = t7*tz, t40 = t9*tz, t41 = t14*ty, t42 = t14*tz, t43 = t15*tx, t47 = t15*tz, t48 = t16*tx, t49 = t16*ty, t50 = -t4, t52 = -t7, t54 = -t9, t56 = t2*t17, t57 = t3*ty, t58 = t4*tx, t59 = t3*t21;
    T t60 = t4*t17, t61 = t3*t22, t62 = t4*t18, t63 = t5*tz, t64 = t7*tx, t65 = t5*t25, t66 = t6*t21, t67 = t7*t17, t68 = t5*t26, t69 = t7*t18, t70 = t8*tz, t71 = t9*ty, t72 = t8*t25, t73 = t9*t21, t74 = t8*t26, t75 = t9*t22, t76 = t10*t25;
    T t77 = t14*t17, t78 = rx*t21, t79 = ry*t17, t80 = rx*t25, t81 = rz*t17, t82 = t15*t21, t83 = ry*t25, t84 = rz*t21, t85 = t16*t25, t95 = t2*t21, t96 = t2*t25, t98 = t3*t25, t100 = t4*t25, t102 = t7*t21, t104 = t6*t25, t107 = t14*t21;
    T t108 = t14*t25, t109 = t15*t17, t116 = t15*t25, t117 = t16*t17, t118 = t16*t21, t128 = t14*tx*5.0, t130 = t14*t18*7.0, t131 = t15*ty*5.0, t133 = t15*t22*7.0, t134 = t16*tz*5.0, t136 = t16*t26*7.0, t143 = -t45, t144 = -t46, t145 = t17+t21;
    T t146 = t17+t25, t147 = t21+t25, t158 = t4*t21*5.0, t172 = t7*t25*5.0, t176 = t9*t25*5.0, t180 = t14*t22*6.0, t185 = t14*t26*6.0, t186 = t15*t18*6.0, t191 = t15*t26*6.0, t192 = t16*t18*6.0, t195 = t16*t22*6.0, t208 = t2+t6, t209 = t2+t10;
    T t210 = t6+t10, t211 = t14+t15, t212 = t14+t16, t213 = t15+t16, t217 = t4*t21*-2.0, t227 = t7*t25*-2.0, t230 = t9*t25*-2.0, t247 = t2*t14*t18*tdx*1.2E+1, t248 = t6*t15*t22*tdy*1.2E+1, t249 = t10*t16*t26*tdz*1.2E+1, t262 = t11+t12+t13;
    T t290 = t4*t15*t22*tdz*8.0, t292 = t7*t16*t26*tdy*8.0, t293 = t9*t16*t26*tdx*8.0, t300 = rx*t4*t22*tdz*tz*1.6E+1, t305 = t4*t15*t26*tdy*1.6E+1, t307 = rx*t7*t26*tdy*ty*1.6E+1, t312 = t4*t16*t22*tdx*1.6E+1, t313 = t7*t15*t26*tdx*1.6E+1;
    T t318 = t7*t16*t22*tdz*1.6E+1, t328 = t2*t14*t22*tdx*4.8E+1, t330 = t2*t14*t22*tdz*8.0, t332 = t2*t14*t26*tdx*4.8E+1, t333 = t2*t14*t26*tdy*8.0, t348 = t6*t15*t26*tdx*8.0, t354 = t6*t15*t26*tdy*4.8E+1, t51 = -t30, t53 = -t32, t55 = -t34;
    T t86 = t41*4.0, t87 = t42*4.0, t88 = t43*4.0, t89 = t47*4.0, t90 = t48*4.0, t91 = t49*4.0, t92 = t14*t56, t93 = t15*t66, t94 = t16*t76, t97 = t37*tx, t99 = t57*tz, t101 = t58*tz, t103 = t64*ty, t105 = t39*tx, t106 = t40*ty, t110 = t80*ty;
    T t111 = t78*tz, t112 = t83*tx, t113 = t79*tz, t114 = t84*tx, t115 = t81*ty, t119 = t35*tz*4.0, t120 = t37*tz*4.0, t121 = t39*ty*4.0, t122 = t59*6.0, t123 = t60*6.0, t124 = t65*6.0, t125 = t67*6.0, t126 = t72*6.0, t127 = t73*6.0;
    T t129 = t77*6.0, t132 = t82*6.0, t135 = t85*6.0, t148 = t35*tx*2.0, t149 = t95*4.0, t150 = t95*5.0, t151 = t35*tx*5.0, t152 = t36*tx*2.0, t154 = t96*4.0, t156 = t96*5.0, t157 = t36*tx*5.0, t160 = t98*6.0, t162 = t100*6.0, t164 = t102*6.0;
    T t166 = t38*ty*2.0, t168 = t104*4.0, t170 = t104*5.0, t171 = t38*ty*5.0, t178 = t107*2.0, t179 = t107*5.0, t181 = t108*2.0, t182 = t109*2.0, t183 = t108*5.0, t184 = t109*5.0, t187 = t116*2.0, t188 = t117*2.0, t189 = t116*5.0, t190 = t117*5.0;
    T t193 = t118*2.0, t194 = t118*5.0, t196 = -t61, t197 = t18*t50, t200 = -t68, t201 = t18*t52, t204 = -t74, t205 = t22*t54, t238 = t25*t78, t239 = t25*t79, t240 = t21*t81, t241 = t25*t35*2.0, t242 = t21*t36*2.0, t243 = t25*t30*ty;
    T t244 = t30*tx*ty*tz, t245 = t21*t32*tz, t246 = t32*tx*ty*tz, t250 = rx*t145, t251 = rx*t146, t252 = ry*t145, t253 = ry*t147, t254 = rz*t146, t255 = rz*t147, t263 = t211*tx*tz, t264 = t212*tx*ty, t265 = t211*ty*tz, t266 = t213*tx*ty;
    T t267 = t212*ty*tz, t268 = t213*tx*tz, t275 = t25*t57*8.0, t276 = t25*t58*8.0, t277 = t21*t64*8.0, t278 = t25+t145, t284 = t37*t43*5.0, t286 = t39*t48*5.0, t287 = t40*t49*5.0, t288 = t35*t77*tdz*8.0, t289 = t36*t77*tdy*8.0;
    T t291 = t38*t82*tdx*8.0, t294 = rx*t26*t35*tdx*1.6E+1, t295 = rx*t22*t36*tdx*1.6E+1, t296 = t35*t42*tdx*tx*3.2E+1, t297 = t35*t42*tdy*tx*3.2E+1, t298 = t35*t42*tdz*tx*3.2E+1, t299 = t36*t109*tdz*1.6E+1, t301 = t36*t82*tdz*3.2E+1;
    T t303 = t4*t21*t47*tdx*3.2E+1, t304 = t35*t117*tdy*1.6E+1, t306 = t47*t60*tdy*1.6E+1, t308 = t35*t85*tdy*3.2E+1, t309 = t4*t21*t47*tdy*3.2E+1, t311 = t4*t21*t47*tdz*3.2E+1, t314 = t37*t85*tdx*3.2E+1, t315 = t39*t82*tdx*3.2E+1;
    T t316 = t7*t25*t49*tdx*3.2E+1, t317 = t7*t25*t49*tdy*3.2E+1, t319 = t49*t67*tdz*1.6E+1, t320 = t7*t25*t49*tdz*3.2E+1, t324 = t44+t143, t325 = t44+t144, t326 = t45+t144, t329 = t14*t95*tdy*tx*4.8E+1, t331 = t14*t95*tdz*tx*1.6E+1;
    T t334 = t14*t96*tdy*tx*1.6E+1, t335 = t14*t96*tdz*tx*4.8E+1, t336 = t4*t21*t43*tdx*4.8E+1, t337 = t56*t83*tdy*8.0, t338 = t58*t108*tdy*1.6E+1, t339 = t37*t109*tdy*4.8E+1, t340 = t56*t84*tdz*8.0, t341 = t37*t109*tdz*8.0;
    T t342 = t4*t21*t43*tdz*1.6E+1, t343 = t64*t107*tdz*1.6E+1, t344 = t66*t80*tdx*8.0, t345 = t57*t116*tdx*1.6E+1, t346 = rz*t22*t58*tdz*8.0, t347 = t43*t102*tdz*1.6E+1, t349 = rx*t26*t71*tdx*8.0, t350 = t49*t98*tdx*1.6E+1;
    T t351 = t15*t104*tdx*ty*1.6E+1, t352 = ry*t26*t64*tdy*8.0, t353 = t48*t100*tdy*1.6E+1, t355 = t15*t104*tdz*ty*4.8E+1, t356 = t7*t25*t48*tdx*4.8E+1, t357 = t39*t117*tdy*8.0, t358 = t7*t25*t48*tdy*1.6E+1, t359 = t39*t117*tdz*4.8E+1;
    T t360 = t40*t118*tdx*8.0, t361 = t9*t25*t49*tdx*1.6E+1, t362 = t9*t25*t49*tdy*4.8E+1, t363 = t40*t118*tdz*4.8E+1, t370 = t78*t96*tdx*2.4E+1, t371 = t35*t108*tdx*4.8E+1, t372 = t36*t107*tdx*4.8E+1, t373 = t35*t108*tdy*2.4E+1;
    T t374 = t36*t107*tdy*2.4E+1, t375 = t35*t108*tdz*2.4E+1, t376 = t36*t107*tdz*2.4E+1, t377 = t43*t96*tdx*2.4E+1, t378 = t37*t108*tdx*4.8E+1, t379 = t78*t100*tdy*2.4E+1, t380 = t35*t116*tdy*4.8E+1, t381 = t48*t95*tdx*2.4E+1;
    T t382 = t37*t116*tdx*2.4E+1, t383 = t37*t43*tdx*tz*2.4E+1, t384 = t39*t107*tdx*4.8E+1, t385 = t43*t100*tdy*2.4E+1, t386 = t37*t116*tdy*4.8E+1, t387 = t37*t43*tdy*tz*4.8E+1, t388 = t37*t116*tdz*2.4E+1, t389 = t37*t43*tdz*tz*2.4E+1;
    T t391 = t36*t118*tdz*4.8E+1, t392 = t4*t21*t48*tdy*2.4E+1, t393 = t39*t43*tdy*ty*4.8E+1, t394 = t7*t25*t43*tdz*2.4E+1, t395 = t37*t48*tdz*tz*4.8E+1, t396 = t39*t118*tdx*2.4E+1, t397 = t39*t48*tdx*ty*2.4E+1, t398 = t39*t118*tdy*2.4E+1;
    T t399 = t39*t48*tdy*ty*2.4E+1, t400 = t48*t102*tdz*2.4E+1, t401 = t39*t118*tdz*4.8E+1, t402 = t39*t48*tdz*ty*4.8E+1, t409 = t145*t211, t410 = t146*t212, t411 = t147*t213, t155 = t97*4.0, t159 = t97*5.0, t161 = t99*6.0, t163 = t101*6.0;
    T t165 = t103*6.0, t169 = t105*4.0, t173 = t105*5.0, t175 = t106*4.0, t177 = t106*5.0, t198 = -t122, t199 = -t123, t202 = -t124, t203 = -t125, t206 = -t126, t207 = -t127, t214 = -t148, t215 = -t149, t216 = -t152, t218 = -t154, t220 = -t160;
    T t222 = -t162, t224 = -t164, t226 = -t166, t228 = -t168, t232 = -t110, t233 = -t111, t234 = -t112, t235 = -t113, t236 = -t114, t237 = -t115, t256 = t3+t51, t257 = t5+t53, t258 = t8+t55, t259 = t35*t87, t260 = t37*t89, t261 = t39*t91;
    T t269 = t251*ty, t270 = t250*tz, t271 = t253*tx, t272 = t252*tz, t273 = t255*tx, t274 = t254*ty, t282 = t14*t150, t283 = t14*t156, t285 = t15*t170, t302 = t42*t97*tdz*3.2E+1, t310 = t41*t105*tdy*3.2E+1, t321 = -t275, t322 = -t276;
    T t323 = -t277, t327 = t278*t278, t364 = t324*tx, t365 = t325*tx, t366 = t324*ty, t367 = t326*ty, t368 = t325*tz, t369 = t326*tz, t390 = t7*t238*tdz*2.4E+1, t403 = t43+t251, t404 = t48+t250, t405 = t41+t253, t406 = t49+t252, t407 = t42+t255;
    T t408 = t47+t254, t433 = t78+t80+t88+t90+t128, t434 = t79+t83+t86+t91+t131, t435 = t81+t84+t87+t89+t134, t448 = t77+t82+t107+t109+t135+t183+t188+t189+t193, t449 = t77+t85+t108+t117+t132+t179+t182+t187+t194;
    T t450 = t82+t85+t116+t118+t129+t178+t181+t184+t190, t219 = -t155, t221 = -t161, t223 = -t163, t225 = -t165, t229 = -t169, t231 = -t175, t412 = -t366, t413 = -t368, t414 = -t369, t415 = t403*tx, t416 = t404*tx, t417 = t405*tx, t418 = t403*ty;
    T t419 = t405*ty, t420 = t407*tx, t421 = t404*tz, t422 = t406*ty, t423 = t408*ty, t424 = t406*tz, t425 = t407*tz, t426 = t408*tz, t430 = t262*t327*tx*ty*2.0, t431 = t262*t327*tx*tz*2.0, t432 = t262*t327*ty*tz*2.0;
    T t436 = t278*t433*tdx*ty*tz*2.0, t437 = t278*t434*tdy*tx*tz*2.0, t438 = t278*t435*tdz*tx*ty*2.0, t439 = t59+t98+t121+t151+t199+t217+t222, t440 = t60+t100+t121+t158+t198+t214+t220, t441 = t65+t99+t120+t157+t203+t224+t227;
    T t447 = t66+t76+t104+t106+t159+t173+t215+t218, t493 = t278*t449*tdy*tx*2.0, t494 = t278*t448*tdz*tx*2.0, t495 = t278*t450*tdx*ty*2.0, t496 = t278*t448*tdz*ty*2.0, t497 = t278*t450*tdx*tz*2.0, t498 = t278*t449*tdy*tz*2.0;
    T t442 = t67+t102+t120+t172+t202+t216+t221, t443 = t72+t101+t119+t171+t207+t225+t230, t444 = t73+t103+t119+t176+t206+t223+t226, t445 = t56+t66+t95+t97+t156+t170+t229+t231, t446 = t56+t76+t96+t105+t150+t177+t219+t228;
    T t451 = t237+t265+t274+t364+t424, t452 = t233+t268+t270+t367+t420, t453 = t235+t267+t272+t365+t423, t454 = t234+t264+t271+t413+t418, t455 = t236+t263+t273+t412+t421, t456 = t232+t266+t269+t414+t417, t457 = t117+t118+t409+t416+t422;
    T t458 = t109+t116+t410+t415+t426, t459 = t107+t108+t411+t419+t425, t499 = t436+t496+t498, t500 = t437+t494+t497, t501 = t438+t493+t495, t460 = t452*tx*ty, t461 = t453*tx*ty, t462 = t451*tx*tz, t463 = t452*tx*tz, t464 = t451*ty*tz;
    T t465 = t453*ty*tz, t466 = t454*tx*ty, t467 = t456*tx*ty, t468 = t455*tx*tz, t469 = t456*tx*tz, t470 = t454*ty*tz, t471 = t455*ty*tz, t475 = t458*tx*ty, t476 = t457*tx*tz, t477 = t459*tx*ty, t478 = t457*ty*tz, t479 = t459*tx*tz;
    T t480 = t458*ty*tz, t484 = t145*t451, t485 = t146*t453, t486 = t147*t452, t487 = t145*t455, t488 = t146*t454, t489 = t147*t456, t490 = t145*t457, t491 = t146*t458, t492 = t147*t459, t472 = -t460, t473 = -t461, t474 = -t462, t481 = -t469;
    T t482 = -t470, t483 = -t471, t502 = t463+t465+t490, t503 = t464+t467+t491, t506 = t466+t468+t492, t504 = t472+t478+t485, t505 = t473+t476+t486, t507 = t480+t481+t484, t508 = t474+t475+t489, t509 = t479+t482+t487, t510 = t477+t483+t488;
    
    fdrv(0, 0) = -t147*t506-t508*tx*ty-t505*tx*tz; fdrv(0, 1) = t146*t508+t506*tx*ty-t505*ty*tz; fdrv(0, 2) = t145*t505+t506*tx*tz-t508*ty*tz;
    fdrv(0, 3) = t278*(t93+t94+t284+t286+rx*t196+rx*t200+t16*t66+t39*t43*5.0-t2*t82*4.0+t37*t48*5.0-t2*t85*4.0+t40*t49+t6*t85-t14*t95*5.0-t14*t96*5.0+t14*t97*6.0-t15*t96*4.0-t16*t95*4.0+t14*t105*6.0+t15*t104+t15*t106-t57*t80*2.0+t30*t110+t32*t111+rx*t22*t30+rx*t26*t32+ry*t9*t26)*2.0;
    fdrv(0, 4) = t278*(t261+ry*t200+t14*t60-t15*t59*7.0-t16*t59*6.0+t16*t60-t35*t43*3.0-t35*t48*2.0-t3*t85*6.0+t4*t85-t15*t98*8.0+t14*t100-t35*t80*6.0+t39*t79+t39*t86+t4*t132+t30*t109+t30*t116+t39*t131+t4*t179+t4*t194+t2*t234-rx*t2*t22*6.0+ry*t7*t26-t14*t35*tx*2.0)*2.0;
    fdrv(0, 5) = t278*(t260+rz*t196-t36*t43*2.0+t14*t67-t16*t65*7.0+t15*t67-t36*t48*3.0+t7*t82-t47*t57*6.0-t36*t78*6.0+t14*t102+t37*t81+t37*t87+t7*t135+t32*t117+t32*t118+t37*t134+t7*t183+t7*t189+t2*t236-rx*t2*t26*6.0-ry*t3*t26*6.0+rz*t4*t22-t14*t36*tx*2.0-t3*t49*tz*8.0)*2.0;
    fdrv(0, 6) = t248+t249-t299+t300-t301+t302-t304+t307-t308+t310-t337+t338-t340+t343+t344-t345+t349-t350-t377+t378+t379-t380-t381+t382+t384+t385+t389+t390-t391+t396+t399+t400-t15*t61*tdx*8.0-t16*t61*tdx*8.0-t16*t68*tdx*8.0+t37*t77*tdx*6.0E+1+t39*t77*tdx*6.0E+1+t39*t82*tdx*2.4E+1+t37*t85*tdx*2.4E+1-t43*t95*tdx*2.4E+1-t48*t96*tdx*2.4E+1+t37*t109*tdx*4.0E+1+t39*t109*tdx*4.0E+1+t37*t117*tdx*4.0E+1+t39*t117*tdx*4.0E+1-t78*t96*tdx*4.8E+1+t14*t62*tdy*1.2E+1+t15*t62*tdy*2.0E+1+t16*t62*tdy*1.0E+1-t35*t77*tdy*2.0E+1+t40*t82*tdy*1.0E+1-t59*t80*tdy*2.4E+1-t35*t108*tdy*4.8E+1-t35*t109*tdy*2.4E+1+t48*t100*tdy*1.2E+1+t49*t104*tdy*1.2E+1+t40*t118*tdy*8.0+t58*t107*tdy*4.8E+1+t14*t69*tdz*1.2E+1+t15*t69*tdz*1.0E+1+t16*t69*tdz*2.0E+1+t16*t75*tdz*4.0-t36*t77*tdz*2.0E+1+t38*t82*tdz*8.0-t36*t107*tdz*4.8E+1+t43*t102*tdz*1.2E+1-t36*t117*tdz*2.4E+1+t38*t118*tdz*1.2E+1+t64*t108*tdz*4.8E+1+t71*t116*tdz*1.2E+1-rx*t2*t23*tdx*2.4E+1-rx*t2*t27*tdx*2.4E+1+rx*t6*t23*tdx*4.0+rx*t6*t27*tdx*4.0+rx*t10*t27*tdx*4.0+rx*t22*t40*tdx*4.0-rx*t3*t23*tdy*1.2E+1+rx*t4*t23*tdy*2.0E+1-rx*t3*t27*tdy*1.2E+1+rx*t4*t27*tdy*4.0+rx*t22*t39*tdy*1.6E+1+rx*t7*t23*tdz*4.0-rx*t5*t27*tdz*1.2E+1+rx*t7*t27*tdz*2.0E+1+rx*t26*t37*tdz*1.6E+1-rx*t26*t57*tdz*2.4E+1-ry*t3*t27*tdx*8.0-ry*t2*t27*tdy*8.0+ry*t6*t27*tdy*4.0+ry*t10*t27*tdy*2.0+ry*t18*t39*tdy*1.0E+1+ry*t26*t64*tdy*1.2E+1+ry*t9*t27*tdz*1.0E+1-rz*t2*t23*tdz*8.0+rz*t6*t23*tdz*2.0+rz*t18*t37*tdz*1.0E+1+rz*t22*t58*tdz*1.2E+1-rx*t61*tdz*tz*1.2E+1+t4*t14*t22*tdx*4.8E+1+t4*t15*t22*tdx*2.4E+1+t4*t16*t22*tdx*2.4E+1+t7*t14*t26*tdx*4.8E+1+t7*t15*t26*tdx*2.4E+1+t7*t16*t26*tdx*2.4E+1-t2*t14*t22*tdy*4.8E+1-t2*t15*t22*tdy*4.0E+1-t2*t16*t22*tdy*3.2E+1+t6*t16*t22*tdy*1.0E+1+t9*t15*t26*tdy*1.2E+1+t9*t16*t26*tdy*8.0+t4*t21*t43*tdy*4.8E+1+t4*t21*t48*tdy*3.6E+1+t16*t26*t33*tdy-t2*t14*t26*tdz*4.8E+1-t2*t15*t26*tdz*3.2E+1-t2*t16*t26*tdz*4.0E+1+t6*t15*t26*tdz*8.0+t6*t16*t26*tdz*1.0E+1+t15*t22*t34*tdz+t7*t25*t43*tdz*3.6E+1+t7*t25*t48*tdz*4.8E+1+t9*t25*t49*tdz*1.6E+1-t14*t95*tdx*tx*4.0E+1-t14*t96*tdx*tx*4.0E+1+t39*t43*tdy*ty*3.6E+1+t15*t104*tdy*ty*1.6E+1+t37*t48*tdz*tz*3.6E+1;
    fdrv(0, 7) = t292+t293-t298+t311+t318+t319+t320-t328-t329-t334+t336+t339+t351+t357+t360-t371+t386+t397+t398+t14*t62*tdx*1.2E+1+t15*t62*tdx*2.0E+1+t16*t62*tdx*1.0E+1-t35*t77*tdx*2.0E+1-t35*t85*tdx*3.2E+1+t40*t82*tdx*1.0E+1-t56*t83*tdx*8.0-t59*t80*tdx*2.4E+1+t43*t100*tdx*2.4E+1-t35*t109*tdx*2.4E+1+t41*t105*tdx*3.2E+1+t48*t100*tdx*1.2E+1-t35*t116*tdx*4.8E+1-t35*t117*tdx*1.6E+1+t49*t104*tdx*1.2E+1+t58*t107*tdx*4.8E+1+t58*t108*tdx*1.6E+1+t78*t100*tdx*2.4E+1-t15*t61*tdy*8.4E+1-t16*t61*tdy*6.0E+1-t16*t68*tdy*1.2E+1+t37*t77*tdy*2.4E+1+t39*t77*tdy*8.0+t39*t82*tdy*4.0E+1+t37*t85*tdy*2.4E+1-t43*t95*tdy*8.0E+1-t43*t96*tdy*4.8E+1-t48*t95*tdy*4.8E+1+t60*t83*tdy*8.0-t48*t96*tdy*1.6E+1+t37*t108*tdy*2.4E+1+t39*t107*tdy*2.4E+1-t49*t98*tdy*7.2E+1+t39*t109*tdy*2.4E+1+t37*t117*tdy*2.4E+1-t57*t116*tdy*1.2E+2-t78*t96*tdy*7.2E+1+t42*t60*tdz*8.0-t47*t59*tdz*6.0E+1+t47*t60*tdz*1.6E+1+t41*t67*tdz*8.0+t60*t84*tdz*1.2E+1+t67*t83*tdz*1.2E+1-rx*t3*t23*tdx*1.2E+1+rx*t4*t23*tdx*2.0E+1-rx*t3*t27*tdx*1.2E+1+rx*t4*t27*tdx*4.0+rx*t22*t39*tdx*1.6E+1-rx*t2*t23*tdy*6.0E+1-rx*t2*t27*tdy*1.2E+1-rx*t22*t36*tdz*4.8E+1-rx*t26*t35*tdz*4.8E+1-ry*t2*t27*tdx*8.0+ry*t6*t27*tdx*4.0+ry*t10*t27*tdx*2.0+ry*t18*t39*tdx*1.0E+1+ry*t26*t64*tdx*1.2E+1+ry*t4*t19*tdy*4.0-ry*t3*t27*tdy*3.6E+1+ry*t4*t27*tdy*4.0-ry*t5*t27*tdz*1.2E+1+ry*t7*t27*tdz*1.0E+1+ry*t19*t32*tdz-ry*t18*t36*tdz*4.0-rz*t3*t23*tdz*1.2E+1+rz*t4*t23*tdz*1.0E+1+rz*t19*t30*tdz-rz*t18*t35*tdz*4.0-t2*t15*t22*tdx*4.0E+1-t2*t16*t22*tdx*3.2E+1+t6*t15*t22*tdx*1.2E+1+t6*t16*t22*tdx*1.0E+1+t9*t15*t26*tdx*1.2E+1+t4*t21*t48*tdx*3.6E+1+t16*t26*t33*tdx-t2*t14*t18*tdy*4.0-t2*t15*t18*tdy*1.2E+1-t2*t16*t18*tdy*4.0+t4*t14*t22*tdy*4.0E+1+t4*t15*t22*tdy*6.0E+1+t4*t16*t22*tdy*4.0E+1+t7*t14*t26*tdy*8.0+t7*t15*t26*tdy*2.4E+1+t7*t14*t22*tdz*8.0-t3*t15*t26*tdz*7.2E+1+t4*t14*t26*tdz*8.0+t7*t15*t22*tdz*1.0E+1-t3*t16*t26*tdz*6.0E+1+t4*t15*t26*tdz*1.6E+1+t4*t16*t26*tdz*1.0E+1+t4*t21*t42*tdz*2.4E+1+t7*t25*t41*tdz*2.4E+1+t39*t43*tdx*ty*3.6E+1+t15*t67*tdz*ty*1.2E+1+t7*t116*tdz*ty*3.6E+1-t16*t59*tdz*tz*7.2E+1+t16*t60*tdz*tz*1.2E+1-t35*t43*tdz*tz*4.8E+1-t35*t48*tdz*tz*4.8E+1+t4*t118*tdz*tz*3.6E+1+rx*t7*t26*tdx*ty*1.6E+1-ry*t2*t26*tdz*tx*1.6E+1-rz*t2*t22*tdz*tx*1.6E+1;
    fdrv(0, 8) = t290+t291-t297+t305+t306+t309+t317-t331-t332-t335+t341+t348+t356+t359+t361-t372+t383+t388+t401+t14*t69*tdx*1.2E+1+t15*t69*tdx*1.0E+1+t16*t69*tdx*2.0E+1+t16*t75*tdx*4.0-t36*t77*tdx*2.0E+1-t36*t82*tdx*3.2E+1+t42*t97*tdx*3.2E+1-t56*t84*tdx*8.0-t36*t109*tdx*1.6E+1+t43*t102*tdx*1.2E+1+t48*t102*tdx*2.4E+1-t36*t117*tdx*2.4E+1-t36*t118*tdx*4.8E+1+t38*t118*tdx*1.2E+1+t64*t107*tdx*1.6E+1+t64*t108*tdx*4.8E+1+t71*t116*tdx*1.2E+1+t7*t238*tdx*2.4E+1+t42*t60*tdy*8.0-t47*t59*tdy*6.0E+1+t41*t67*tdy*8.0+t49*t67*tdy*1.6E+1+t60*t84*tdy*1.2E+1+t67*t83*tdy*1.2E+1-t15*t61*tdz*1.2E+1-t16*t61*tdz*3.6E+1-t16*t68*tdz*8.4E+1+t37*t77*tdz*8.0+t39*t77*tdz*2.4E+1+t39*t82*tdz*2.4E+1+t37*t85*tdz*4.0E+1-t43*t95*tdz*1.6E+1-t43*t96*tdz*4.8E+1-t48*t95*tdz*4.8E+1-t48*t96*tdz*8.0E+1+t37*t108*tdz*2.4E+1+t39*t107*tdz*2.4E+1-t49*t98*tdz*1.2E+2+t39*t109*tdz*2.4E+1+t67*t84*tdz*8.0+t37*t117*tdz*2.4E+1-t57*t116*tdz*7.2E+1-t78*t96*tdz*7.2E+1+rx*t7*t23*tdx*4.0-rx*t5*t27*tdx*1.2E+1+rx*t7*t27*tdx*2.0E+1+rx*t26*t37*tdx*1.6E+1-rx*t26*t57*tdx*2.4E+1-rx*t22*t36*tdy*4.8E+1-rx*t26*t35*tdy*4.8E+1-rx*t2*t23*tdz*1.2E+1-rx*t2*t27*tdz*6.0E+1+ry*t9*t27*tdx*1.0E+1-ry*t5*t27*tdy*1.2E+1+ry*t7*t27*tdy*1.0E+1+ry*t19*t32*tdy-ry*t18*t36*tdy*4.0-ry*t3*t27*tdz*6.0E+1-rz*t2*t23*tdx*8.0+rz*t6*t23*tdx*2.0+rz*t18*t37*tdx*1.0E+1+rz*t22*t58*tdx*1.2E+1-rz*t3*t23*tdy*1.2E+1+rz*t4*t23*tdy*1.0E+1+rz*t19*t30*tdy-rz*t18*t35*tdy*4.0+rz*t7*t19*tdz*4.0+rz*t7*t23*tdz*4.0-rx*t61*tdx*tz*1.2E+1-t2*t15*t26*tdx*3.2E+1-t2*t16*t26*tdx*4.0E+1+t6*t16*t26*tdx*1.0E+1+t10*t16*t26*tdx*1.2E+1+t15*t22*t34*tdx+t7*t25*t43*tdx*3.6E+1+t7*t14*t22*tdy*8.0-t3*t15*t26*tdy*7.2E+1+t4*t14*t26*tdy*8.0+t7*t15*t22*tdy*1.0E+1-t3*t16*t26*tdy*6.0E+1+t7*t16*t22*tdy*1.6E+1+t4*t16*t26*tdy*1.0E+1+t4*t21*t42*tdy*2.4E+1+t7*t25*t41*tdy*2.4E+1-t2*t14*t18*tdz*4.0-t2*t15*t18*tdz*4.0-t2*t16*t18*tdz*1.2E+1+t4*t14*t22*tdz*8.0+t4*t16*t22*tdz*2.4E+1+t7*t14*t26*tdz*4.0E+1+t7*t15*t26*tdz*4.0E+1+t7*t16*t26*tdz*6.0E+1+t15*t67*tdy*ty*1.2E+1+t7*t116*tdy*ty*3.6E+1+t37*t48*tdx*tz*3.6E+1-t16*t59*tdy*tz*7.2E+1+t16*t60*tdy*tz*1.2E+1-t35*t43*tdy*tz*4.8E+1-t35*t48*tdy*tz*4.8E+1+t4*t118*tdy*tz*3.6E+1-ry*t2*t26*tdy*tx*1.6E+1+rx*t4*t22*tdx*tz*1.6E+1-rz*t2*t22*tdy*tx*1.6E+1;
    fdrv(0, 9) = t278*tdx*(t196+t200+t243+t245+t22*t30+t17*t37*6.0+t17*t39*6.0+t26*t32-t25*t57*2.0-t95*tx*5.0-t96*tx*5.0)*2.0+t278*t440*tdy*tx*2.0+t278*t442*tdz*tx*2.0;
    fdrv(0, 10) = t278*tdy*(t61*-7.0+t200+t243+t321+t4*t22*6.0+t7*t26+t17*t39+t21*t39*5.0-t95*tx*3.0-t96*tx+t17*t30*ty)*2.0+t278*t447*tdx*ty*2.0+t278*t442*tdz*ty*2.0;
    fdrv(0, 11) = t278*tdz*(t68*-7.0+t196+t245+t321+t4*t22+t7*t26*6.0+t17*t37+t25*t37*5.0-t95*tx-t96*tx*3.0+t17*t32*tz)*2.0+t278*t447*tdx*tz*2.0+t278*t440*tdy*tz*2.0;
    fdrv(0, 12) = t278*tdy*(t133+t180+t195+t239+ry*t27+t17*t41*2.0+t17*t49*2.0+t25*t41*6.0+t25*t49*6.0+t109*ty*3.0+t116*ty*8.0)*-2.0-t278*tdz*(t136+t185+t191+t240+rz*t23+t17*t42*2.0+t21*t42*6.0+t17*t47*2.0+t21*t47*6.0+t117*tz*3.0+t118*tz*8.0)*2.0-t147*t278*t433*tdx*2.0;
    fdrv(0, 13) = t501; fdrv(0, 14) = t500; fdrv(0, 15) = t327*tx*(t37+t39-t57-t63)*2.0; fdrv(0, 16) = t327*ty*(t37+t39-t57-t63)*2.0; fdrv(0, 17) = t327*tz*(t37+t39-t57-t63)*2.0;
    fdrv(0, 18) = rdx*t327*(t37+t39-t57-t63)*2.0+t11*t210*t327*2.0+t12*t210*t327*2.0+t13*t210*t327*2.0+t11*t278*tx*(t37+t39-t57-t63)*8.0+t12*t278*tx*(t37+t39-t57-t63)*8.0+t13*t278*tx*(t37+t39-t57-t63)*8.0;
    fdrv(0, 19) = rdy*t327*(t37+t39-t57-t63)*2.0+t11*t327*(t4-t29)*2.0+t12*t327*(t4-t29)*2.0+t13*t327*(t4-t29)*2.0+t11*t278*ty*(t37+t39-t57-t63)*8.0+t12*t278*ty*(t37+t39-t57-t63)*8.0+t13*t278*ty*(t37+t39-t57-t63)*8.0;
    fdrv(0, 20) = rdz*t327*(t37+t39-t57-t63)*2.0+t11*t327*(t7-t31)*2.0+t12*t327*(t7-t31)*2.0+t13*t327*(t7-t31)*2.0+t11*t278*tz*(t37+t39-t57-t63)*8.0+t12*t278*tz*(t37+t39-t57-t63)*8.0+t13*t278*tz*(t37+t39-t57-t63)*8.0;
    fdrv(0, 24) = t147*t262*t327*-2.0; fdrv(0, 25) = t430; fdrv(0, 26) = t431; fdrv(1, 0) = t147*t510+t503*tx*ty-t504*tx*tz; fdrv(1, 1) = -t146*t503-t510*tx*ty-t504*ty*tz; fdrv(1, 2) = t145*t504-t510*tx*tz+t503*ty*tz;
    fdrv(1, 3) = t278*(t261+rx*t204-t14*t60*7.0+t15*t59-t15*t60*6.0+t16*t59-t16*t60*6.0+t35*t43*5.0+t39*t41*5.0+t35*t48*5.0-t4*t82*2.0+t3*t85-t4*t85*6.0-t4*t107*3.0+t15*t98-t14*t100*8.0-t15*t100*6.0+t35*t80*2.0+t40*t78-t4*t118*2.0+t15*t121+t6*t232+rx*t2*t22*2.0+rx*t9*t26+t14*t35*tx*6.0)*2.0;
    fdrv(1, 4) = t278*(t92+t94+t282-t284+t287+ry*t197+ry*t204+t15*t56*2.0+t16*t56-t37*t48*4.0+t2*t85+t39*t48-t6*t85*4.0+t14*t96-t14*t97*4.0+t15*t96*2.0-t37*t80*4.0+t39*t78*5.0+t14*t105-t15*t104*5.0+t15*t106*6.0+t2*t132-t58*t83*2.0+t16*t150+rx*t7*t26+ry*t26*t34+t32*t43*tz)*2.0;
    fdrv(1, 5) = t278*(t259+rz*t197-t38*t49*3.0+t15*t73-t16*t72*7.0-t42*t58*6.0+t41*t64-t4*t111*2.0+t35*t81+t7*t110*5.0+t35*t89+t9*t135+t34*t118+t50*t114+t35*t134+t9*t189+rx*t7*t22-rx*t4*t26*6.0-ry*t6*t26*6.0+rz*t2*t22+t7*t43*ty-t15*t38*ty*2.0+t32*t48*ty-t4*t43*tz*6.0-t4*t48*tz*8.0)*2.0;
    fdrv(1, 6) = t292+t293+t298-t311+t318+t319+t320+t328+t329+t334-t336-t339-t351+t357+t360+t371-t386+t397+t398-t14*t62*tdx*8.4E+1-t15*t62*tdx*6.0E+1-t16*t62*tdx*6.0E+1-t16*t74*tdx*1.2E+1+t35*t77*tdx*6.0E+1+t35*t85*tdx*2.4E+1+t40*t82*tdx*8.0+t59*t80*tdx*8.0-t43*t100*tdx*7.2E+1+t35*t109*tdx*4.0E+1+t41*t105*tdx*4.0E+1-t48*t100*tdx*7.2E+1+t35*t116*tdx*2.4E+1+t35*t117*tdx*4.0E+1-t49*t104*tdx*1.6E+1-t58*t107*tdx*8.0E+1-t58*t108*tdx*1.2E+2-t78*t100*tdx*4.8E+1+t15*t61*tdy*1.2E+1+t16*t61*tdy*1.0E+1-t37*t77*tdy*4.0E+1+t39*t77*tdy*1.0E+1+t39*t82*tdy*3.2E+1-t37*t85*tdy*3.2E+1+t43*t95*tdy*4.8E+1+t43*t96*tdy*2.4E+1+t48*t95*tdy*3.6E+1-t60*t83*tdy*2.4E+1+t48*t96*tdy*1.2E+1-t37*t108*tdy*4.8E+1+t39*t107*tdy*3.6E+1-t66*t80*tdy*8.0+t49*t98*tdy*1.2E+1+t39*t109*tdy*1.6E+1-t37*t117*tdy*3.2E+1+t57*t116*tdy*1.6E+1+t78*t96*tdy*2.4E+1-t42*t60*tdz*6.0E+1+t47*t59*tdz*8.0-t47*t60*tdz*4.8E+1+t41*t67*tdz*1.0E+1-t60*t84*tdz*1.6E+1+t73*t80*tdz*1.2E+1+rx*t3*t23*tdx*4.0-rx*t4*t23*tdx*1.2E+1+rx*t3*t27*tdx*4.0-rx*t4*t27*tdx*3.6E+1+rx*t22*t39*tdx*2.4E+1+rx*t2*t23*tdy*2.0E+1+rx*t2*t27*tdy*4.0-rx*t6*t27*tdy*8.0+rx*t10*t27*tdy*2.0+rx*t22*t40*tdy*1.0E+1+rx*t26*t71*tdy*1.2E+1-rx*t8*t27*tdz*1.2E+1+rx*t9*t27*tdz*1.0E+1+rx*t23*t34*tdz+rx*t22*t36*tdz*1.6E+1-rx*t22*t38*tdz*4.0+rx*t26*t35*tdz*1.6E+1-ry*t6*t27*tdx*1.2E+1-ry*t4*t19*tdy*1.2E+1+ry*t3*t27*tdy*4.0-ry*t4*t27*tdy*1.2E+1-rz*t4*t19*tdz*1.2E+1-rz*t4*t23*tdz*4.0+rz*t23*t29*tdz+rz*t18*t35*tdz*1.0E+1+t2*t15*t22*tdx*2.4E+1+t2*t16*t22*tdx*2.4E+1-t6*t15*t22*tdx*4.0-t6*t16*t22*tdx*4.0+t9*t15*t26*tdx*8.0-t4*t21*t48*tdx*4.8E+1+t2*t14*t18*tdy*1.2E+1+t2*t15*t18*tdy*2.0E+1+t2*t16*t18*tdy*1.0E+1-t4*t14*t22*tdy*2.4E+1-t4*t15*t22*tdy*2.0E+1-t4*t16*t22*tdy*1.6E+1+t7*t14*t26*tdy*1.2E+1+t7*t15*t26*tdy*1.6E+1+t16*t26*t31*tdy+t7*t14*t22*tdz*1.2E+1+t3*t15*t26*tdz*8.0-t4*t14*t26*tdz*7.2E+1+t7*t15*t22*tdz*8.0+t3*t16*t26*tdz*1.0E+1-t4*t15*t26*tdz*4.8E+1-t4*t16*t26*tdz*6.0E+1-t4*t21*t42*tdz*4.8E+1+t7*t25*t41*tdz*3.6E+1+t39*t43*tdx*ty*2.4E+1+t15*t67*tdz*ty*8.0+t7*t116*tdz*ty*2.4E+1+t16*t59*tdz*tz*1.2E+1-t16*t60*tdz*tz*7.2E+1+t35*t43*tdz*tz*2.4E+1+t35*t48*tdz*tz*3.6E+1-t4*t118*tdz*tz*4.8E+1+rx*t7*t26*tdx*ty*2.4E+1-rx*t6*t26*tdz*ty*1.6E+1+rz*t2*t22*tdz*tx*1.2E+1;
    fdrv(1, 7) = t247+t249+t299-t300+t301-t302-t312+t313-t314+t315+t337-t338-t344+t345-t346+t347+t352-t353+t370+t373+t376+t377-t378-t379+t380-t392+t393+t394-t395+t396+t399+t400+t15*t61*tdx*1.2E+1+t16*t61*tdx*1.0E+1-t37*t77*tdx*4.0E+1+t39*t77*tdx*1.0E+1+t43*t95*tdx*4.8E+1+t48*t95*tdx*3.6E+1-t60*t83*tdx*2.4E+1+t48*t96*tdx*1.2E+1-t37*t109*tdx*4.8E+1+t39*t107*tdx*3.6E+1+t49*t98*tdx*1.2E+1+t39*t109*tdx*1.6E+1-t37*t116*tdx*4.8E+1-t37*t117*tdx*3.2E+1+t39*t117*tdx*8.0-t14*t62*tdy*8.0-t15*t62*tdy*2.4E+1-t16*t62*tdy*8.0-t16*t74*tdy*8.0+t35*t77*tdy*2.4E+1+t35*t85*tdy*2.4E+1+t40*t82*tdy*6.0E+1-t43*t100*tdy*4.8E+1+t35*t109*tdy*4.8E+1+t41*t105*tdy*2.4E+1+t35*t117*tdy*2.4E+1-t49*t104*tdy*2.4E+1+t40*t118*tdy*4.0E+1-t58*t107*tdy*2.4E+1+t15*t69*tdz*4.0+t16*t69*tdz*4.0+t15*t75*tdz*1.2E+1+t16*t75*tdz*2.0E+1+t36*t77*tdz*8.0-t38*t82*tdz*2.0E+1+t56*t84*tdz*1.2E+1+t36*t117*tdz*1.2E+1+t36*t118*tdz*3.6E+1-t38*t118*tdz*2.4E+1+t64*t107*tdz*1.2E+1+t64*t108*tdz*1.2E+1+t71*t116*tdz*4.8E+1+t7*t238*tdz*3.6E+1+rx*t2*t23*tdx*2.0E+1+rx*t2*t27*tdx*4.0-rx*t6*t27*tdx*8.0+rx*t10*t27*tdx*2.0+rx*t22*t40*tdx*1.0E+1+rx*t26*t71*tdx*1.2E+1-rx*t4*t27*tdy*8.0+rx*t22*t39*tdy*4.0E+1+rx*t7*t23*tdz*1.0E+1+rx*t7*t27*tdz*1.0E+1-rx*t26*t37*tdz*3.2E+1-ry*t4*t19*tdx*1.2E+1+ry*t3*t27*tdx*4.0-ry*t4*t27*tdx*1.2E+1+ry*t2*t19*tdy*4.0+ry*t2*t27*tdy*4.0-ry*t6*t27*tdy*2.4E+1+ry*t10*t27*tdy*4.0+ry*t18*t39*tdy*4.0-ry*t8*t27*tdz*1.2E+1+ry*t9*t27*tdz*2.0E+1-ry*t26*t58*tdz*2.4E+1+rz*t2*t19*tdz*2.0+rz*t2*t23*tdz*1.0E+1-rz*t18*t37*tdz*8.0-ry*t62*tdz*tz*1.2E+1+t2*t15*t18*tdx*2.0E+1+t2*t16*t18*tdx*1.0E+1-t4*t14*t22*tdx*2.4E+1-t4*t15*t22*tdx*2.0E+1+t7*t14*t26*tdx*1.2E+1+t7*t16*t26*tdx*8.0+t16*t26*t31*tdx+t2*t14*t22*tdy*4.0E+1+t2*t15*t22*tdy*6.0E+1+t2*t16*t22*tdy*4.0E+1+t9*t15*t26*tdy*4.8E+1+t9*t16*t26*tdy*2.4E+1-t4*t21*t43*tdy*4.0E+1+t2*t14*t26*tdz*8.0+t2*t15*t26*tdz*1.6E+1+t2*t16*t26*tdz*1.0E+1-t6*t15*t26*tdz*4.8E+1-t6*t16*t26*tdz*4.0E+1+t14*t18*t32*tdz+t7*t25*t48*tdz*1.6E+1+t9*t25*t49*tdz*4.8E+1+t14*t95*tdx*tx*4.8E+1+t14*t96*tdx*tx*1.6E+1-t15*t104*tdy*ty*4.0E+1-t37*t43*tdz*tz*4.8E+1+rx*t7*t26*tdy*ty*2.4E+1;
    fdrv(1, 8) = t288+t289+t294+t295+t296-t303+t316+t330+t333-t342-t354-t355+t358+t362+t363+t374+t375-t387+t402-t42*t60*tdx*6.0E+1+t47*t59*tdx*8.0-t47*t60*tdx*4.8E+1+t41*t67*tdx*1.0E+1+t49*t67*tdx*1.6E+1-t60*t84*tdx*1.6E+1+t73*t80*tdx*1.2E+1+t15*t69*tdy*4.0+t16*t69*tdy*4.0+t15*t75*tdy*1.2E+1+t16*t75*tdy*2.0E+1+t36*t82*tdy*3.2E+1-t38*t82*tdy*2.0E+1-t42*t97*tdy*3.2E+1+t56*t84*tdy*1.2E+1+t36*t109*tdy*1.6E+1+t43*t102*tdy*1.6E+1+t48*t102*tdy*2.4E+1+t36*t117*tdy*1.2E+1+t36*t118*tdy*3.6E+1-t38*t118*tdy*2.4E+1+t64*t107*tdy*1.2E+1+t64*t108*tdy*1.2E+1+t71*t116*tdy*4.8E+1+t7*t238*tdy*3.6E+1-t14*t62*tdz*1.2E+1-t15*t62*tdz*1.2E+1-t16*t62*tdz*3.6E+1-t16*t74*tdz*8.4E+1+t35*t85*tdz*4.0E+1+t40*t82*tdz*2.4E+1-t43*t100*tdz*7.2E+1+t35*t109*tdz*8.0+t41*t105*tdz*2.4E+1-t48*t100*tdz*1.2E+2+t35*t116*tdz*2.4E+1+t35*t117*tdz*2.4E+1-t49*t104*tdz*8.0E+1-t58*t107*tdz*1.6E+1-t58*t108*tdz*7.2E+1-t78*t100*tdz*4.8E+1-rx*t8*t27*tdx*1.2E+1+rx*t9*t27*tdx*1.0E+1+rx*t23*t34*tdx-rx*t22*t38*tdx*4.0+rx*t7*t23*tdy*1.0E+1+rx*t7*t27*tdy*1.0E+1-rx*t26*t37*tdy*3.2E+1-rx*t4*t23*tdz*4.0-rx*t4*t27*tdz*6.0E+1+rx*t22*t39*tdz*2.4E+1-ry*t8*t27*tdy*1.2E+1+ry*t9*t27*tdy*2.0E+1-ry*t26*t58*tdy*2.4E+1-ry*t6*t27*tdz*6.0E+1-rz*t4*t19*tdx*1.2E+1-rz*t4*t23*tdx*4.0+rz*t23*t29*tdx+rz*t18*t35*tdx*1.0E+1+rz*t2*t19*tdy*2.0+rz*t2*t23*tdy*1.0E+1-rz*t18*t37*tdy*8.0-rz*t22*t58*tdy*8.0+rz*t9*t23*tdz*4.0+rz*t22*t64*tdz*8.0-ry*t62*tdy*tz*1.2E+1+rz*t69*tdz*ty*4.0+t7*t14*t22*tdx*1.2E+1+t3*t15*t26*tdx*8.0-t4*t14*t26*tdx*7.2E+1+t7*t15*t22*tdx*8.0+t3*t16*t26*tdx*1.0E+1-t4*t15*t26*tdx*4.8E+1+t7*t16*t22*tdx*1.6E+1-t4*t16*t26*tdx*6.0E+1-t4*t21*t42*tdx*4.8E+1+t7*t25*t41*tdx*3.6E+1+t2*t15*t26*tdy*1.6E+1+t2*t16*t26*tdy*1.0E+1-t6*t16*t26*tdy*4.0E+1+t10*t16*t26*tdy*1.2E+1+t14*t18*t32*tdy+t7*t25*t43*tdy*2.4E+1+t2*t15*t22*tdz*8.0+t2*t16*t22*tdz*2.4E+1-t6*t15*t22*tdz*4.0-t6*t16*t22*tdz*1.2E+1+t9*t15*t26*tdz*4.0E+1+t9*t16*t26*tdz*6.0E+1-t4*t21*t48*tdz*4.8E+1+t15*t67*tdx*ty*8.0+t7*t116*tdx*ty*2.4E+1+t39*t43*tdz*ty*2.4E+1+t16*t59*tdx*tz*1.2E+1-t16*t60*tdx*tz*7.2E+1+t35*t43*tdx*tz*2.4E+1+t35*t48*tdx*tz*3.6E+1-t4*t118*tdx*tz*4.8E+1-t37*t48*tdy*tz*4.8E+1-rx*t6*t26*tdx*ty*1.6E+1+rx*t7*t26*tdz*ty*4.0E+1+rz*t2*t22*tdx*tx*1.2E+1-rx*t4*t22*tdy*tz*1.6E+1;
    fdrv(1, 9) = t278*tdx*(t62*-7.0+t204+t241+t322+t2*t22*2.0+t9*t26+t17*t35*6.0+t21*t40-t21*t58*3.0-t104*ty+t173*ty)*2.0+t278*t446*tdy*tx*2.0+t278*t444*tdz*tx*2.0;
    fdrv(1, 10) = t278*tdy*(t197+t204+t241+t246+t2*t22*6.0+t17*t35*2.0+t26*t34+t21*t40*6.0-t21*t58*5.0-t25*t58*2.0-t104*ty*5.0)*2.0+t278*t439*tdx*ty*2.0+t278*t444*tdz*ty*2.0;
    fdrv(1, 11) = t278*tdz*(t74*-7.0+t197+t246+t322+t2*t22+t9*t26*6.0+t17*t35+t25*t35*5.0-t104*ty*3.0+t21*t50*tx+t21*t34*tz)*2.0+t278*t439*tdx*tz*2.0+t278*t446*tdy*tz*2.0; fdrv(1, 12) = t501;
    fdrv(1, 13) = t278*tdx*(t130+t186+t192+t238+rx*t27+t21*t43*2.0+t25*t43*6.0+t21*t48*2.0+t25*t48*6.0+t107*tx*3.0+t108*tx*8.0)*-2.0-t278*tdz*(t136+t185+t191+t240+rz*t19+t17*t42*6.0+t21*t42*2.0+t17*t47*6.0+t21*t47*2.0+t117*tz*8.0+t118*tz*3.0)*2.0-t146*t278*t434*tdy*2.0;
    fdrv(1, 14) = t499; fdrv(1, 15) = t327*tx*(t35+t40-t70+t50*tx)*2.0; fdrv(1, 16) = t327*ty*(t35+t40-t70+t50*tx)*2.0; fdrv(1, 17) = t327*tz*(t35+t40-t70+t50*tx)*2.0;
    fdrv(1, 18) = rdx*t327*(t35+t40-t70+t50*tx)*2.0+t11*t256*t327*2.0+t12*t256*t327*2.0+t13*t256*t327*2.0+t11*t278*tx*(t35+t40-t70+t50*tx)*8.0+t12*t278*tx*(t35+t40-t70+t50*tx)*8.0+t13*t278*tx*(t35+t40-t70+t50*tx)*8.0;
    fdrv(1, 19) = rdy*t327*(t35+t40-t70+t50*tx)*2.0+t11*t209*t327*2.0+t12*t209*t327*2.0+t13*t209*t327*2.0+t11*t278*ty*(t35+t40-t70+t50*tx)*8.0+t12*t278*ty*(t35+t40-t70+t50*tx)*8.0+t13*t278*ty*(t35+t40-t70+t50*tx)*8.0;
    fdrv(1, 20) = rdz*t327*(t35+t40-t70+t50*tx)*2.0+t11*t327*(t9-t33)*2.0+t12*t327*(t9-t33)*2.0+t13*t327*(t9-t33)*2.0+t11*t278*tz*(t35+t40-t70+t50*tx)*8.0+t12*t278*tz*(t35+t40-t70+t50*tx)*8.0+t13*t278*tz*(t35+t40-t70+t50*tx)*8.0; fdrv(1, 24) = t430;
    fdrv(1, 25) = t146*t262*t327*-2.0; fdrv(1, 26) = t432; fdrv(2, 0) = t147*t509-t507*tx*ty+t502*tx*tz; fdrv(2, 1) = t146*t507-t509*tx*ty+t502*ty*tz; fdrv(2, 2) = -t145*t502-t509*tx*tz-t507*ty*tz;
    fdrv(2, 3) = t278*(t260+rx*t205+t36*t43*5.0+t37*t42*5.0-t14*t67*7.0+t16*t65-t15*t67*6.0-t16*t67*6.0+t36*t48*5.0-t7*t82*6.0-t7*t85*2.0+t47*t57+t36*t78*2.0-t7*t108*3.0-t14*t102*8.0+t38*t78-t16*t102*6.0-t7*t116*2.0+t16*t120+t54*t110+rx*t2*t26*2.0+rx*t6*t26+ry*t3*t26+t14*t36*tx*6.0+t3*t49*tz)*2.0;
    fdrv(2, 4) = t278*(t259+ry*t201+t35*t47*5.0+t38*t49*5.0-t15*t73*7.0+t16*t72-t16*t73*6.0-t9*t85*2.0+t42*t58-t41*t64*6.0+t4*t111*5.0+t36*t79-t7*t110*2.0-t9*t116*3.0+t16*t119+t52*t112-rx*t7*t22*6.0+rx*t4*t26+ry*t2*t26+ry*t6*t26*2.0-t7*t43*ty*8.0+t15*t38*ty*6.0-t7*t48*ty*6.0+t4*t48*tz+t30*t43*tz)*2.0;
    fdrv(2, 5) = t278*(t92+t93+t283+t285-t286-t287+rz*t201+rz*t205+t15*t56+t16*t56*2.0+t37*t43+t16*t66*2.0-t39*t43*4.0+t2*t82+t14*t95+t14*t97+t16*t95*2.0+t37*t80*5.0-t39*t78*4.0-t14*t105*4.0-t15*t106*4.0+t2*t135+t6*t135-t64*t84*2.0+t15*t156+rx*t4*t22+t30*t48*ty)*2.0;
    fdrv(2, 6) = t290+t291+t297+t305+t306+t309-t317+t331+t332+t335+t341+t348-t356-t359-t361+t372+t383+t388-t401-t14*t69*tdx*8.4E+1-t15*t69*tdx*6.0E+1-t16*t69*tdx*6.0E+1-t15*t75*tdx*1.2E+1-t16*t75*tdx*1.2E+1+t36*t77*tdx*6.0E+1+t36*t82*tdx*2.4E+1+t42*t97*tdx*4.0E+1+t36*t109*tdx*4.0E+1-t43*t102*tdx*7.2E+1-t48*t102*tdx*7.2E+1+t36*t117*tdx*4.0E+1+t36*t118*tdx*2.4E+1+t38*t118*tdx*8.0-t64*t107*tdx*1.2E+2-t64*t108*tdx*8.0E+1-t71*t116*tdx*1.6E+1-t7*t238*tdx*4.8E+1+t42*t60*tdy*1.0E+1+t47*t59*tdy*1.0E+1-t41*t67*tdy*6.0E+1-t49*t67*tdy*4.8E+1-t67*t83*tdy*1.6E+1-t73*t80*tdy*1.6E+1+t16*t61*tdz*4.0+t16*t68*tdz*1.2E+1+t37*t77*tdz*1.0E+1-t39*t77*tdz*4.0E+1-t39*t82*tdz*3.2E+1+t37*t85*tdz*3.2E+1+t43*t95*tdz*1.2E+1+t43*t96*tdz*3.6E+1+t48*t95*tdz*2.4E+1+t48*t96*tdz*4.8E+1+t37*t108*tdz*3.6E+1-t39*t107*tdz*4.8E+1+t66*t80*tdz*1.2E+1+t49*t98*tdz*1.6E+1-t39*t109*tdz*3.2E+1-t67*t84*tdz*2.4E+1+t37*t117*tdz*1.6E+1+t57*t116*tdz*1.2E+1+t78*t96*tdz*2.4E+1-rx*t7*t23*tdx*3.6E+1+rx*t5*t27*tdx*4.0-rx*t7*t27*tdx*1.2E+1+rx*t26*t37*tdx*2.4E+1+rx*t26*t57*tdx*8.0-rx*t9*t23*tdy*1.2E+1-rx*t9*t27*tdy*4.0+rx*t22*t36*tdy*1.6E+1+rx*t22*t38*tdy*1.0E+1+rx*t27*t33*tdy+rx*t26*t35*tdy*1.6E+1+rx*t2*t23*tdz*4.0+rx*t2*t27*tdz*2.0E+1+rx*t6*t23*tdz*2.0+rx*t6*t27*tdz*1.0E+1-rx*t22*t40*tdz*8.0-rx*t26*t71*tdz*8.0-ry*t9*t27*tdx*4.0-ry*t7*t19*tdy*1.2E+1-ry*t7*t27*tdy*4.0+ry*t18*t36*tdy*1.0E+1+ry*t27*t31*tdy+ry*t3*t27*tdz*1.0E+1-rz*t7*t19*tdz*1.2E+1-rz*t7*t23*tdz*1.2E+1+rx*t61*tdx*tz*4.0+t2*t15*t26*tdx*2.4E+1+t2*t16*t26*tdx*2.4E+1+t6*t16*t26*tdx*8.0-t10*t16*t26*tdx*4.0-t7*t25*t43*tdx*4.8E+1-t7*t14*t22*tdy*7.2E+1+t3*t15*t26*tdy*1.2E+1+t4*t14*t26*tdy*1.2E+1-t7*t15*t22*tdy*6.0E+1+t3*t16*t26*tdy*8.0-t7*t16*t22*tdy*4.8E+1+t4*t16*t26*tdy*8.0+t4*t21*t42*tdy*3.6E+1-t7*t25*t41*tdy*4.8E+1+t2*t14*t18*tdz*1.2E+1+t2*t15*t18*tdz*1.0E+1+t2*t16*t18*tdz*2.0E+1+t4*t14*t22*tdz*1.2E+1+t4*t16*t22*tdz*1.6E+1-t7*t14*t26*tdz*2.4E+1-t7*t15*t26*tdz*1.6E+1-t7*t16*t26*tdz*2.0E+1+t15*t22*t29*tdz-t15*t67*tdy*ty*7.2E+1-t7*t116*tdy*ty*4.8E+1+t37*t48*tdx*tz*2.4E+1+t16*t59*tdy*tz*8.0+t16*t60*tdy*tz*8.0+t35*t43*tdy*tz*3.6E+1+t35*t48*tdy*tz*2.4E+1+t4*t118*tdy*tz*2.4E+1+ry*t2*t26*tdy*tx*1.2E+1+rx*t6*t26*tdy*ty*1.2E+1+rx*t4*t22*tdx*tz*2.4E+1;
    fdrv(2, 7) = t288+t289+t294+t295+t296+t303-t316+t330+t333+t342+t354+t355-t358-t362-t363+t374+t375+t387-t402+t42*t60*tdx*1.0E+1+t47*t59*tdx*1.0E+1+t47*t60*tdx*1.6E+1-t41*t67*tdx*6.0E+1-t49*t67*tdx*4.8E+1-t67*t83*tdx*1.6E+1-t73*t80*tdx*1.6E+1-t14*t69*tdy*1.2E+1-t15*t69*tdy*3.6E+1-t16*t69*tdy*1.2E+1-t15*t75*tdy*8.4E+1-t16*t75*tdy*6.0E+1+t36*t82*tdy*4.0E+1+t38*t82*tdy*6.0E+1+t42*t97*tdy*2.4E+1+t36*t109*tdy*2.4E+1-t43*t102*tdy*1.2E+2-t48*t102*tdy*7.2E+1+t36*t117*tdy*8.0+t36*t118*tdy*2.4E+1+t38*t118*tdy*4.0E+1-t64*t107*tdy*7.2E+1-t64*t108*tdy*1.6E+1-t71*t116*tdy*8.0E+1-t7*t238*tdy*4.8E+1+t15*t62*tdz*4.0+t16*t62*tdz*4.0+t16*t74*tdz*1.2E+1+t35*t85*tdz*3.2E+1-t40*t82*tdz*4.0E+1+t56*t83*tdz*1.2E+1+t43*t100*tdz*2.4E+1+t35*t109*tdz*1.2E+1-t41*t105*tdz*3.2E+1+t48*t100*tdz*1.6E+1+t35*t116*tdz*3.6E+1+t35*t117*tdz*1.6E+1+t49*t104*tdz*4.8E+1+t58*t107*tdz*1.2E+1+t58*t108*tdz*1.2E+1+t78*t100*tdz*3.6E+1-rx*t9*t23*tdx*1.2E+1-rx*t9*t27*tdx*4.0+rx*t22*t38*tdx*1.0E+1+rx*t27*t33*tdx-rx*t7*t23*tdy*6.0E+1-rx*t7*t27*tdy*4.0+rx*t26*t37*tdy*2.4E+1+rx*t4*t23*tdz*1.0E+1+rx*t4*t27*tdz*1.0E+1-rx*t22*t39*tdz*3.2E+1-ry*t7*t19*tdx*1.2E+1-ry*t7*t27*tdx*4.0+ry*t18*t36*tdx*1.0E+1+ry*t27*t31*tdx+ry*t8*t27*tdy*4.0-ry*t9*t27*tdy*1.2E+1+ry*t26*t58*tdy*8.0+ry*t2*t19*tdz*2.0+ry*t2*t27*tdz*1.0E+1+ry*t6*t27*tdz*2.0E+1-ry*t18*t39*tdz*8.0-ry*t26*t64*tdz*8.0-rz*t9*t23*tdz*1.2E+1-rz*t22*t64*tdz*2.4E+1+ry*t62*tdy*tz*4.0-rz*t69*tdz*ty*1.2E+1-t7*t14*t22*tdx*7.2E+1+t3*t15*t26*tdx*1.2E+1+t4*t14*t26*tdx*1.2E+1-t7*t15*t22*tdx*6.0E+1+t3*t16*t26*tdx*8.0+t4*t15*t26*tdx*1.6E+1-t7*t16*t22*tdx*4.8E+1+t4*t16*t26*tdx*8.0+t4*t21*t42*tdx*3.6E+1-t7*t25*t41*tdx*4.8E+1+t2*t15*t26*tdy*2.4E+1+t2*t16*t26*tdy*8.0+t6*t16*t26*tdy*2.4E+1-t10*t16*t26*tdy*4.0-t7*t25*t43*tdy*4.8E+1+t2*t15*t22*tdz*1.0E+1+t2*t16*t22*tdz*1.6E+1+t6*t15*t22*tdz*1.2E+1+t6*t16*t22*tdz*2.0E+1-t9*t15*t26*tdz*2.4E+1-t9*t16*t26*tdz*2.0E+1+t14*t18*t30*tdz+t4*t21*t48*tdz*2.4E+1-t15*t67*tdx*ty*7.2E+1-t7*t116*tdx*ty*4.8E+1-t39*t43*tdz*ty*4.8E+1+t16*t59*tdx*tz*8.0+t16*t60*tdx*tz*8.0+t35*t43*tdx*tz*3.6E+1+t35*t48*tdx*tz*2.4E+1+t4*t118*tdx*tz*2.4E+1+t37*t48*tdy*tz*2.4E+1+ry*t2*t26*tdx*tx*1.2E+1+rx*t6*t26*tdx*ty*1.2E+1-rx*t7*t26*tdz*ty*1.6E+1+rx*t4*t22*tdy*tz*4.0E+1;
    fdrv(2, 8) = t247+t248+t304-t307+t308-t310+t312-t313+t314-t315+t340-t343+t346-t347-t349+t350-t352+t353+t370+t373+t376+t381+t382-t384+t385+t389-t390+t391+t392-t393-t394+t395+t16*t61*tdx*4.0+t16*t68*tdx*1.2E+1+t37*t77*tdx*1.0E+1-t39*t77*tdx*4.0E+1+t43*t95*tdx*1.2E+1+t43*t96*tdx*3.6E+1+t48*t96*tdx*4.8E+1+t37*t108*tdx*3.6E+1+t37*t109*tdx*8.0+t66*t80*tdx*1.2E+1-t39*t109*tdx*3.2E+1-t67*t84*tdx*2.4E+1+t37*t117*tdx*1.6E+1-t39*t117*tdx*4.8E+1-t39*t118*tdx*4.8E+1+t57*t116*tdx*1.2E+1+t15*t62*tdy*4.0+t16*t62*tdy*4.0+t16*t74*tdy*1.2E+1+t35*t77*tdy*8.0-t40*t82*tdy*4.0E+1+t56*t83*tdy*1.2E+1+t35*t109*tdy*1.2E+1+t35*t116*tdy*3.6E+1+t49*t104*tdy*4.8E+1-t40*t118*tdy*4.8E+1+t58*t107*tdy*1.2E+1+t58*t108*tdy*1.2E+1+t78*t100*tdy*3.6E+1-t14*t69*tdz*8.0-t15*t69*tdz*8.0-t16*t69*tdz*2.4E+1-t15*t75*tdz*8.0-t16*t75*tdz*2.4E+1+t36*t77*tdz*2.4E+1+t36*t82*tdz*2.4E+1+t38*t82*tdz*2.4E+1+t42*t97*tdz*2.4E+1+t36*t109*tdz*2.4E+1-t48*t102*tdz*4.8E+1+t36*t117*tdz*4.8E+1+t38*t118*tdz*4.8E+1-t64*t108*tdz*2.4E+1-t71*t116*tdz*2.4E+1+rx*t2*t23*tdx*4.0+rx*t2*t27*tdx*2.0E+1+rx*t6*t23*tdx*2.0+rx*t6*t27*tdx*1.0E+1-rx*t22*t40*tdx*8.0+rx*t4*t23*tdy*1.0E+1+rx*t4*t27*tdy*1.0E+1-rx*t22*t39*tdy*3.2E+1-rx*t7*t23*tdz*8.0+rx*t26*t37*tdz*4.0E+1+ry*t3*t27*tdx*1.0E+1+ry*t2*t19*tdy*2.0+ry*t2*t27*tdy*1.0E+1+ry*t6*t27*tdy*2.0E+1-ry*t18*t39*tdy*8.0-rz*t7*t19*tdx*1.2E+1-rz*t7*t23*tdx*1.2E+1-rz*t9*t23*tdy*1.2E+1-rz*t22*t64*tdy*2.4E+1+rz*t2*t19*tdz*4.0+rz*t2*t23*tdz*4.0+rz*t6*t23*tdz*4.0+rz*t18*t37*tdz*4.0-rz*t69*tdy*ty*1.2E+1+t2*t15*t18*tdx*1.0E+1+t2*t16*t18*tdx*2.0E+1+t4*t14*t22*tdx*1.2E+1+t4*t15*t22*tdx*8.0-t7*t14*t26*tdx*2.4E+1-t7*t16*t26*tdx*2.0E+1+t15*t22*t29*tdx+t2*t14*t22*tdy*8.0+t2*t15*t22*tdy*1.0E+1+t2*t16*t22*tdy*1.6E+1+t6*t16*t22*tdy*2.0E+1-t9*t15*t26*tdy*2.4E+1-t9*t16*t26*tdy*2.0E+1+t14*t18*t30*tdy+t4*t21*t43*tdy*1.6E+1+t2*t14*t26*tdz*4.0E+1+t2*t15*t26*tdz*4.0E+1+t2*t16*t26*tdz*6.0E+1+t6*t15*t26*tdz*4.0E+1+t6*t16*t26*tdz*6.0E+1-t7*t25*t48*tdz*4.0E+1-t9*t25*t49*tdz*4.0E+1+t14*t95*tdx*tx*1.6E+1+t14*t96*tdx*tx*4.8E+1-t39*t48*tdy*ty*4.8E+1+t15*t104*tdy*ty*4.8E+1+rx*t4*t22*tdz*tz*2.4E+1;
    fdrv(2, 9) = t278*tdx*(t69*-7.0+t205+t242+t323+t2*t26*2.0+t6*t26+t17*t36*6.0+t21*t38-t25*t64*3.0+t159*tz+t25*t54*ty)*2.0+t278*t443*tdy*tx*2.0+t278*t445*tdz*tx*2.0;
    fdrv(2, 10) = t278*tdy*(t75*-7.0+t201+t244+t323+t2*t26+t6*t26*2.0+t17*t36+t21*t36*5.0+t21*t38*6.0-t25*t71*3.0+t25*t52*tx)*2.0+t278*t441*tdx*ty*2.0+t278*t445*tdz*ty*2.0;
    fdrv(2, 11) = t278*tdz*(t201+t205+t242+t244+t2*t26*6.0+t6*t26*6.0+t17*t36*2.0+t21*t38*2.0-t21*t64*2.0-t25*t64*5.0-t25*t71*5.0)*2.0+t278*t441*tdx*tz*2.0+t278*t443*tdy*tz*2.0; fdrv(2, 12) = t500; fdrv(2, 13) = t499;
    fdrv(2, 14) = t278*tdx*(t130+t186+t192+t238+rx*t23+t21*t43*6.0+t25*t43*2.0+t21*t48*6.0+t25*t48*2.0+t107*tx*8.0+t108*tx*3.0)*-2.0-t278*tdy*(t133+t180+t195+t239+ry*t19+t17*t41*6.0+t17*t49*6.0+t25*t41*2.0+t25*t49*2.0+t109*ty*8.0+t116*ty*3.0)*2.0-t145*t278*t435*tdz*2.0;
    fdrv(2, 15) = t327*tx*(t36+t38+t52*tx+t54*ty)*2.0; fdrv(2, 16) = t327*ty*(t36+t38+t52*tx+t54*ty)*2.0; fdrv(2, 17) = t327*tz*(t36+t38+t52*tx+t54*ty)*2.0;
    fdrv(2, 18) = rdx*t327*(t36+t38+t52*tx+t54*ty)*2.0+t11*t257*t327*2.0+t12*t257*t327*2.0+t13*t257*t327*2.0+t11*t278*tx*(t36+t38+t52*tx+t54*ty)*8.0+t12*t278*tx*(t36+t38+t52*tx+t54*ty)*8.0+t13*t278*tx*(t36+t38+t52*tx+t54*ty)*8.0;
    fdrv(2, 19) = rdy*t327*(t36+t38+t52*tx+t54*ty)*2.0+t11*t258*t327*2.0+t12*t258*t327*2.0+t13*t258*t327*2.0+t11*t278*ty*(t36+t38+t52*tx+t54*ty)*8.0+t12*t278*ty*(t36+t38+t52*tx+t54*ty)*8.0+t13*t278*ty*(t36+t38+t52*tx+t54*ty)*8.0;
    fdrv(2, 20) = rdz*t327*(t36+t38+t52*tx+t54*ty)*2.0+t11*t208*t327*2.0+t12*t208*t327*2.0+t13*t208*t327*2.0+t11*t278*tz*(t36+t38+t52*tx+t54*ty)*8.0+t12*t278*tz*(t36+t38+t52*tx+t54*ty)*8.0+t13*t278*tz*(t36+t38+t52*tx+t54*ty)*8.0; fdrv(2, 24) = t431;
    fdrv(2, 25) = t432; fdrv(2, 26) = t145*t262*t327*-2.0;
    
    return fdrv;
}


template class SE3Q<double>;
template class SE3Q<ceres::Jet<double, 4>>;
template class SE3Qp<double>;
template class SE3Qp<ceres::Jet<double, 4>>;