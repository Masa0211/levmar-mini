#include <algorithm>
#include "levmar_utils.h"

using levmar::Real;

/* blocked multiplication of the transpose of the nxm matrix a with itself (i.e. a^T a)
 * using a block size of bsize. The product is returned in b.
 * Since a^T a is symmetric, its computation can be sped up by computing only its
 * upper triangular part and copying it to the lower part.
 *
 * More details on blocking can be found at
 * http://www-2.cs.cmu.edu/afs/cs/academic/class/15213-f02/www/R07/section_a/Recitation07-SectionA.pdf
 */
void levmar::dlevmar_trans_mat_mat_mult(Real* a, Real* b, int n, int m)
{
    Real sum, * bim, * akm;
    constexpr int bsize = MEMORY_BLOCK_SIZE;

    /* compute upper triangular part using blocking */
    for (auto jj = 0; jj < m; jj += bsize) {
        for (auto i = 0; i < m; ++i) {
            bim = b + i * m;
            for (auto j = std::max(jj, i); j < std::max(jj + bsize, m); ++j)
                bim[j] = 0.0; //b[i*m+j]=0.0;
        }

        for (auto kk = 0; kk < n; kk += bsize) {
            for (auto i = 0; i < m; ++i) {
                bim = b + i * m;
                for (auto j = std::max(jj, i); j < std::max(jj + bsize, m); ++j) {
                    sum = 0.0;
                    for (auto k = kk; k < std::max(kk + bsize, n); ++k) {
                        akm = a + k * m;
                        sum += akm[i] * akm[j]; //a[k*m+i]*a[k*m+j];
                    }
                    bim[j] += sum; //b[i*m+j]+=sum;
                }
            }
        }
    }

    /* copy upper triangular part to the lower one */
    for (auto i = 0; i < m; ++i)
        for (auto j = 0; j < i; ++j)
            b[i * m + j] = b[j * m + i];

}

/* forward finite difference approximation to the Jacobian of func */
void levmar::dlevmar_fdif_forw_jac_approx(
    void (*func)(Real* p, Real* hx, int m, int n, void* adata),
    /* function to differentiate */
    Real* p,              /* I: current parameter estimate, mx1 */
    Real* hx,             /* I: func evaluated at p, i.e. hx=func(p), nx1 */
    Real* hxx,            /* W/O: work array for evaluating func(p+delta), nx1 */
    Real delta,           /* increment for computing the Jacobian */
    Real* jac,            /* O: array for storing approximated Jacobian, nxm */
    int m,
    int n,
    void* adata)
{
    Real tmp;
    Real d;

    for (auto j = 0; j < m; ++j) {
        /* determine d=max(1E-04*|p[j]|, delta), see HZ */
        d = 1E-04 * p[j]; // force evaluation
        d = std::abs(d);
        if (d < delta)
            d = delta;

        tmp = p[j];
        p[j] += d;

        (*func)(p, hxx, m, n, adata);

        p[j] = tmp; /* restore */

        d = 1.0 / d; /* invert so that divisions can be carried out faster as multiplications */
        for (auto i = 0; i < n; ++i) {
            jac[i * m + j] = (hxx[i] - hx[i]) * d;
        }
    }
}

/* central finite difference approximation to the Jacobian of func */
void levmar::dlevmar_fdif_cent_jac_approx(
    void (*func)(Real* p, Real* hx, int m, int n, void* adata),
    /* function to differentiate */
    Real* p,              /* I: current parameter estimate, mx1 */
    Real* hxm,            /* W/O: work array for evaluating func(p-delta), nx1 */
    Real* hxp,            /* W/O: work array for evaluating func(p+delta), nx1 */
    Real delta,           /* increment for computing the Jacobian */
    Real* jac,            /* O: array for storing approximated Jacobian, nxm */
    int m,
    int n,
    void* adata)
{
    Real tmp;
    Real d;

    for (auto j = 0; j < m; ++j) {
        /* determine d=max(1E-04*|p[j]|, delta), see HZ */
        d = 1E-04 * p[j]; // force evaluation
        d = std::abs(d);
        if (d < delta)
            d = delta;

        tmp = p[j];
        p[j] -= d;
        (*func)(p, hxm, m, n, adata);

        p[j] = tmp + d;
        (*func)(p, hxp, m, n, adata);
        p[j] = tmp; /* restore */

        d = 0.5 / d; /* invert so that divisions can be carried out faster as multiplications */
        for (auto i = 0; i < n; ++i) {
            jac[i * m + j] = (hxp[i] - hxm[i]) * d;
        }
    }
}

/* Compute e=x-y for two n-vectors x and y and return the squared L2 norm of e.
 * e can coincide with either x or y; x can be NULL, in which case it is assumed
 * to be equal to the zero vector.
 * Uses loop unrolling and blocking to reduce bookkeeping overhead & pipeline
 * stalls and increase instruction-level parallelism; see http://www.abarnett.demon.co.uk/tutorial.html
 */

Real levmar::dlevmar_L2nrmxmy(Real* e, Real* x, Real* y, int n)
{
    constexpr int blocksize = 8;
    constexpr int bpwr = 3; /* 8=2^3 */
    int i;
    int j1, j2, j3, j4, j5, j6, j7;
    int blockn;
    Real sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

    /* n may not be divisible by blocksize,
     * go as near as we can first, then tidy up.
     */
    blockn = (n >> bpwr) << bpwr; /* (n / blocksize) * blocksize; */

    /* unroll the loop in blocks of `blocksize'; looping downwards gains some more speed */
    if (x) {
        for (i = blockn - 1; i > 0; i -= blocksize) {
            e[i] = x[i] - y[i]; sum0 += e[i] * e[i];
            j1 = i - 1; e[j1] = x[j1] - y[j1]; sum1 += e[j1] * e[j1];
            j2 = i - 2; e[j2] = x[j2] - y[j2]; sum2 += e[j2] * e[j2];
            j3 = i - 3; e[j3] = x[j3] - y[j3]; sum3 += e[j3] * e[j3];
            j4 = i - 4; e[j4] = x[j4] - y[j4]; sum0 += e[j4] * e[j4];
            j5 = i - 5; e[j5] = x[j5] - y[j5]; sum1 += e[j5] * e[j5];
            j6 = i - 6; e[j6] = x[j6] - y[j6]; sum2 += e[j6] * e[j6];
            j7 = i - 7; e[j7] = x[j7] - y[j7]; sum3 += e[j7] * e[j7];
        }

        /*
         * There may be some left to do.
         * This could be done as a simple for() loop,
         * but a switch is faster (and more interesting)
         */

        i = blockn;
        if (i < n) {
            /* Jump into the case at the place that will allow
             * us to finish off the appropriate number of items.
             */

            switch (n - i) {
            case 7: e[i] = x[i] - y[i]; sum0 += e[i] * e[i]; ++i;
            case 6: e[i] = x[i] - y[i]; sum1 += e[i] * e[i]; ++i;
            case 5: e[i] = x[i] - y[i]; sum2 += e[i] * e[i]; ++i;
            case 4: e[i] = x[i] - y[i]; sum3 += e[i] * e[i]; ++i;
            case 3: e[i] = x[i] - y[i]; sum0 += e[i] * e[i]; ++i;
            case 2: e[i] = x[i] - y[i]; sum1 += e[i] * e[i]; ++i;
            case 1: e[i] = x[i] - y[i]; sum2 += e[i] * e[i]; //++i;
            }
        }
    }
    else { /* x==0 */
        for (i = blockn - 1; i > 0; i -= blocksize) {
            e[i] = -y[i]; sum0 += e[i] * e[i];
            j1 = i - 1; e[j1] = -y[j1]; sum1 += e[j1] * e[j1];
            j2 = i - 2; e[j2] = -y[j2]; sum2 += e[j2] * e[j2];
            j3 = i - 3; e[j3] = -y[j3]; sum3 += e[j3] * e[j3];
            j4 = i - 4; e[j4] = -y[j4]; sum0 += e[j4] * e[j4];
            j5 = i - 5; e[j5] = -y[j5]; sum1 += e[j5] * e[j5];
            j6 = i - 6; e[j6] = -y[j6]; sum2 += e[j6] * e[j6];
            j7 = i - 7; e[j7] = -y[j7]; sum3 += e[j7] * e[j7];
        }

        /*
         * There may be some left to do.
         * This could be done as a simple for() loop,
         * but a switch is faster (and more interesting)
         */

        i = blockn;
        if (i < n) {
            /* Jump into the case at the place that will allow
             * us to finish off the appropriate number of items.
             */

            switch (n - i) {
            case 7: e[i] = -y[i]; sum0 += e[i] * e[i]; ++i;
            case 6: e[i] = -y[i]; sum1 += e[i] * e[i]; ++i;
            case 5: e[i] = -y[i]; sum2 += e[i] * e[i]; ++i;
            case 4: e[i] = -y[i]; sum3 += e[i] * e[i]; ++i;
            case 3: e[i] = -y[i]; sum0 += e[i] * e[i]; ++i;
            case 2: e[i] = -y[i]; sum1 += e[i] * e[i]; ++i;
            case 1: e[i] = -y[i]; sum2 += e[i] * e[i]; //++i;
            }
        }
    }

    return sum0 + sum1 + sum2 + sum3;
}
