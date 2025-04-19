#pragma once

#include <cmath>
#include <functional>

namespace levmar
{
    using Real = double;
    using RealPtr = Real*;
    using ConstPtr = const Real*;

    /* block size for cache-friendly matrix-matrix multiply. It should be
     * such that __BLOCKSZ__^2*sizeof(LM_REAL) is smaller than the CPU (L1)
     * data cache size. Notice that a value of 32 when LM_REAL=double assumes
     * an 8Kb L1 data cache (32*32*8=8K). This is a concervative choice since
     * newer Pentium 4s have a L1 data cache of size 16K, capable of holding
     * up to 45x45 double blocks.
    */
    constexpr int MEMORY_BLOCK_SIZE = 32;
    constexpr int MEMORY_BLOCK_SIZE_SQ = MEMORY_BLOCK_SIZE * MEMORY_BLOCK_SIZE;

    constexpr Real LM_REAL_EPSILON = DBL_EPSILON;
    constexpr Real LM_REAL_MAX = DBL_MAX;
    constexpr Real LM_REAL_MIN = - DBL_MAX;

    constexpr Real EPSILON = 1E-12;
    constexpr Real ONE_THIRD = 0.3333333334; /* 1.0/3.0 */
    //constexpr Real ONE_THIRD = 1.0 / 3.0;

    /* blocking-based matrix multiply */
    void dlevmar_trans_mat_mat_mult(RealPtr a, RealPtr b, int numPoints, int numParams);

    /* forward finite differences */
    void dlevmar_fdif_forw_jac_approx(
        std::function<void(RealPtr, RealPtr, int numParams, int numPoints)> func,
        RealPtr p, RealPtr hx, RealPtr hxx, Real delta,
        RealPtr jac, int numParams, int numPoints);

    /* central finite differences */
    void dlevmar_fdif_cent_jac_approx(
        std::function<void(RealPtr, RealPtr, int numParams, int numPoints)> func,
        RealPtr p, RealPtr hxm, RealPtr hxp, Real delta,
        RealPtr jac, int numParams, int numPoints);

    /* e=x-y and ||e|| */
    double dlevmar_L2nrmxmy(RealPtr e, ConstPtr x, ConstPtr y, int numPoints);

    /* covariance of LS fit */
    int dlevmar_covar(double* JtJ, double* C, double sumsq, int numParams, int numPoints);

    /* This function computes the inverse of A in B. A and B can coincide */
    int dlevmar_LUinverse(Real* A, Real* B, int numParams);

}
