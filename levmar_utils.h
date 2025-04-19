#pragma once

namespace levmar
{
    using Real = double;

    /* block size for cache-friendly matrix-matrix multiply. It should be
     * such that __BLOCKSZ__^2*sizeof(LM_REAL) is smaller than the CPU (L1)
     * data cache size. Notice that a value of 32 when LM_REAL=double assumes
     * an 8Kb L1 data cache (32*32*8=8K). This is a concervative choice since
     * newer Pentium 4s have a L1 data cache of size 16K, capable of holding
     * up to 45x45 double blocks.
    */
    constexpr int MEMORY_BLOCK_SIZE = 32;
    constexpr int MEMORY_BLOCK_SIZE_SQ = MEMORY_BLOCK_SIZE * MEMORY_BLOCK_SIZE;

    /* blocking-based matrix multiply */
    void dlevmar_trans_mat_mat_mult(double* a, double* b, int n, int m);

    /* forward finite differences */
    void dlevmar_fdif_forw_jac_approx(void (*func)(double* p, double* hx, int m, int n, void* adata),
        double* p, double* hx, double* hxx, double delta,
        double* jac, int m, int n, void* adata);

    /* central finite differences */
    void dlevmar_fdif_cent_jac_approx(void (*func)(double* p, double* hx, int m, int n, void* adata),
        double* p, double* hxm, double* hxp, double delta,
        double* jac, int m, int n, void* adata);

    /* e=x-y and ||e|| */
    double dlevmar_L2nrmxmy(double* e, double* x, double* y, int n);

}
