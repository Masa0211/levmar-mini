/////////////////////////////////////////////////////////////////////////////////
// 
//  Levenberg - Marquardt non-linear minimization algorithm
//  Copyright (C) 2004-05  Manolis Lourakis (lourakis at ics forth gr)
//  Institute of Computer Science, Foundation for Research & Technology - Hellas
//  Heraklion, Crete, Greece.
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
/////////////////////////////////////////////////////////////////////////////////

/******************************************************************************** 
 * Miscelaneous functions for Levenberg-Marquardt nonlinear minimization. The
 * same core code is used with appropriate #defines to derive single and double
 * precision versions, see also misc_core.c
 ********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <algorithm>

#include "levmar.h"
#include "misc.h"

/* double precision definitions */
#define LM_REAL double

#define LM_REAL_EPSILON DBL_EPSILON

// ------------------------------------ misc_core.cpp ------------------------------------ //

static int dlevmar_LUinverse_noLapack(LM_REAL* A, LM_REAL* B, int m);

/* blocked multiplication of the transpose of the nxm matrix a with itself (i.e. a^T a)
 * using a block size of bsize. The product is returned in b.
 * Since a^T a is symmetric, its computation can be sped up by computing only its
 * upper triangular part and copying it to the lower part.
 *
 * More details on blocking can be found at
 * http://www-2.cs.cmu.edu/afs/cs/academic/class/15213-f02/www/R07/section_a/Recitation07-SectionA.pdf
 */
void dlevmar_trans_mat_mat_mult(LM_REAL* a, LM_REAL* b, int n, int m)
{
    int i, j, k, jj, kk;
    LM_REAL sum, * bim, * akm;
    const int bsize = __BLOCKSZ__;

    /* compute upper triangular part using blocking */
    for (jj = 0; jj < m; jj += bsize) {
        for (i = 0; i < m; ++i) {
            bim = b + i * m;
            for (j = std::max(jj, i); j < std::max(jj + bsize, m); ++j)
                bim[j] = 0.0; //b[i*m+j]=0.0;
        }

        for (kk = 0; kk < n; kk += bsize) {
            for (i = 0; i < m; ++i) {
                bim = b + i * m;
                for (j = std::max(jj, i); j < std::max(jj + bsize, m); ++j) {
                    sum = 0.0;
                    for (k = kk; k < std::max(kk + bsize, n); ++k) {
                        akm = a + k * m;
                        sum += akm[i] * akm[j]; //a[k*m+i]*a[k*m+j];
                    }
                    bim[j] += sum; //b[i*m+j]+=sum;
                }
            }
        }
    }

    /* copy upper triangular part to the lower one */
    for (i = 0; i < m; ++i)
        for (j = 0; j < i; ++j)
            b[i * m + j] = b[j * m + i];

}

/* forward finite difference approximation to the Jacobian of func */
void dlevmar_fdif_forw_jac_approx(
    void (*func)(LM_REAL* p, LM_REAL* hx, int m, int n, void* adata),
    /* function to differentiate */
    LM_REAL* p,              /* I: current parameter estimate, mx1 */
    LM_REAL* hx,             /* I: func evaluated at p, i.e. hx=func(p), nx1 */
    LM_REAL* hxx,            /* W/O: work array for evaluating func(p+delta), nx1 */
    LM_REAL delta,           /* increment for computing the Jacobian */
    LM_REAL* jac,            /* O: array for storing approximated Jacobian, nxm */
    int m,
    int n,
    void* adata)
{
    int i, j;
    LM_REAL tmp;
    LM_REAL d;

    for (j = 0; j < m; ++j) {
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
        for (i = 0; i < n; ++i) {
            jac[i * m + j] = (hxx[i] - hx[i]) * d;
        }
    }
}

/* central finite difference approximation to the Jacobian of func */
void dlevmar_fdif_cent_jac_approx(
    void (*func)(LM_REAL* p, LM_REAL* hx, int m, int n, void* adata),
    /* function to differentiate */
    LM_REAL* p,              /* I: current parameter estimate, mx1 */
    LM_REAL* hxm,            /* W/O: work array for evaluating func(p-delta), nx1 */
    LM_REAL* hxp,            /* W/O: work array for evaluating func(p+delta), nx1 */
    LM_REAL delta,           /* increment for computing the Jacobian */
    LM_REAL* jac,            /* O: array for storing approximated Jacobian, nxm */
    int m,
    int n,
    void* adata)
{
    int i, j;
    LM_REAL tmp;
    LM_REAL d;

    for (j = 0; j < m; ++j) {
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

        d = 0.5/ d; /* invert so that divisions can be carried out faster as multiplications */
        for (i = 0; i < n; ++i) {
            jac[i * m + j] = (hxp[i] - hxm[i]) * d;
        }
    }
}

/*
 * Check the Jacobian of a n-valued nonlinear function in m variables
 * evaluated at a point p, for consistency with the function itself.
 *
 * Based on fortran77 subroutine CHKDER by
 * Burton S. Garbow, Kenneth E. Hillstrom, Jorge J. More
 * Argonne National Laboratory. MINPACK project. March 1980.
 *
 *
 * func points to a function from R^m --> R^n: Given a p in R^m it yields hx in R^n
 * jacf points to a function implementing the Jacobian of func, whose correctness
 *     is to be tested. Given a p in R^m, jacf computes into the nxm matrix j the
 *     Jacobian of func at p. Note that row i of j corresponds to the gradient of
 *     the i-th component of func, evaluated at p.
 * p is an input array of length m containing the point of evaluation.
 * m is the number of variables
 * n is the number of functions
 * adata points to possible additional data and is passed uninterpreted
 *     to func, jacf.
 * err is an array of length n. On output, err contains measures
 *     of correctness of the respective gradients. if there is
 *     no severe loss of significance, then if err[i] is 1.0 the
 *     i-th gradient is correct, while if err[i] is 0.0 the i-th
 *     gradient is incorrect. For values of err between 0.0 and 1.0,
 *     the categorization is less certain. In general, a value of
 *     err[i] greater than 0.5 indicates that the i-th gradient is
 *     probably correct, while a value of err[i] less than 0.5
 *     indicates that the i-th gradient is probably incorrect.
 *
 *
 * The function does not perform reliably if cancellation or
 * rounding errors cause a severe loss of significance in the
 * evaluation of a function. therefore, none of the components
 * of p should be unusually small (in particular, zero) or any
 * other value which may cause loss of significance.
 */


/*
 * This function computes the inverse of A in B. A and B can coincide
 *
 * The function employs LAPACK-free LU decomposition of A to solve m linear
 * systems A*B_i=I_i, where B_i and I_i are the i-th columns of B and I.
 *
 * A and B are mxm
 *
 * The function returns 0 in case of error, 1 if successful
 *
 */
static int dlevmar_LUinverse_noLapack(LM_REAL* A, LM_REAL* B, int m)
{
    void* buf = NULL;
    int buf_sz = 0;

    int i, j, k, l;
    int* idx, maxi = -1, idx_sz, a_sz, x_sz, work_sz, tot_sz;
    LM_REAL* a, * x, * work, max, sum, tmp;

    /* calculate required memory size */
    idx_sz = m;
    a_sz = m * m;
    x_sz = m;
    work_sz = m;
    tot_sz = (a_sz + x_sz + work_sz) * sizeof(LM_REAL) + idx_sz * sizeof(int); /* should be arranged in that order for proper doubles alignment */

    buf_sz = tot_sz;
    buf = (void*)malloc(tot_sz);
    if (!buf) {
        fprintf(stderr, "memory allocation in dlevmar_LUinverse_noLapack() failed!\n");
        return 0; /* error */
    }

    a = reinterpret_cast<double*>(buf);
    x = a + a_sz;
    work = x + x_sz;
    idx = (int*)(work + work_sz);

    /* avoid destroying A by copying it to a */
    for (i = 0; i < a_sz; ++i) a[i] = A[i];

    /* compute the LU decomposition of a row permutation of matrix a; the permutation itself is saved in idx[] */
    for (i = 0; i < m; ++i) {
        max = 0.0;
        for (j = 0; j < m; ++j)
            if ((tmp = std::abs(a[i * m + j])) > max)
                max = tmp;
        if (max == 0.0) {
            fprintf(stderr, "Singular matrix A in dlevmar_LUinverse_noLapack()!\n");
            free(buf);

            return 0;
        }
        work[i] = 1.0 / max;
    }

    for (j = 0; j < m; ++j) {
        for (i = 0; i < j; ++i) {
            sum = a[i * m + j];
            for (k = 0; k < i; ++k)
                sum -= a[i * m + k] * a[k * m + j];
            a[i * m + j] = sum;
        }
        max = 0.0;
        for (i = j; i < m; ++i) {
            sum = a[i * m + j];
            for (k = 0; k < j; ++k)
                sum -= a[i * m + k] * a[k * m + j];
            a[i * m + j] = sum;
            if ((tmp = work[i] * std::abs(sum)) >= max) {
                max = tmp;
                maxi = i;
            }
        }
        if (j != maxi) {
            for (k = 0; k < m; ++k) {
                tmp = a[maxi * m + k];
                a[maxi * m + k] = a[j * m + k];
                a[j * m + k] = tmp;
            }
            work[maxi] = work[j];
        }
        idx[j] = maxi;
        if (a[j * m + j] == 0.0)
            a[j * m + j] = LM_REAL_EPSILON;
        if (j != m - 1) {
            tmp = 1.0 / (a[j * m + j]);
            for (i = j + 1; i < m; ++i)
                a[i * m + j] *= tmp;
        }
    }

    /* The decomposition has now replaced a. Solve the m linear systems using
     * forward and back substitution
     */
    for (l = 0; l < m; ++l) {
        for (i = 0; i < m; ++i) x[i] = 0.0;
        x[l] = 1.0;

        for (i = k = 0; i < m; ++i) {
            j = idx[i];
            sum = x[j];
            x[j] = x[i];
            if (k != 0)
                for (j = k - 1; j < i; ++j)
                    sum -= a[i * m + j] * x[j];
            else
                if (sum != 0.0)
                    k = i + 1;
            x[i] = sum;
        }

        for (i = m - 1; i >= 0; --i) {
            sum = x[i];
            for (j = i + 1; j < m; ++j)
                sum -= a[i * m + j] * x[j];
            x[i] = sum / a[i * m + i];
        }

        for (i = 0; i < m; ++i)
            B[i * m + l] = x[i];
    }

    free(buf);

    return 1;
}

/*
 * This function computes in C the covariance matrix corresponding to a least
 * squares fit. JtJ is the approximate Hessian at the solution (i.e. J^T*J, where
 * J is the Jacobian at the solution), sumsq is the sum of squared residuals
 * (i.e. goodnes of fit) at the solution, m is the number of parameters (variables)
 * and n the number of observations. JtJ can coincide with C.
 *
 * if JtJ is of full rank, C is computed as sumsq/(n-m)*(JtJ)^-1
 * otherwise and if LAPACK is available, C=sumsq/(n-r)*(JtJ)^+
 * where r is JtJ's rank and ^+ denotes the pseudoinverse
 * The diagonal of C is made up from the estimates of the variances
 * of the estimated regression coefficients.
 * See the documentation of routine E04YCF from the NAG fortran lib
 *
 * The function returns the rank of JtJ if successful, 0 on error
 *
 * A and C are mxm
 *
 */
int dlevmar_covar(LM_REAL* JtJ, LM_REAL* C, LM_REAL sumsq, int m, int n)
{
    int i;
    int rnk;
    LM_REAL fact;

    rnk = dlevmar_LUinverse_noLapack(JtJ, C, m);
    if (!rnk) return 0;

    rnk = m; /* assume full rank */

    fact = sumsq / (LM_REAL)(n - rnk);
    for (i = 0; i < m * m; ++i)
        C[i] *= fact;

    return rnk;
}

/* Compute e=x-y for two n-vectors x and y and return the squared L2 norm of e.
 * e can coincide with either x or y; x can be NULL, in which case it is assumed
 * to be equal to the zero vector.
 * Uses loop unrolling and blocking to reduce bookkeeping overhead & pipeline
 * stalls and increase instruction-level parallelism; see http://www.abarnett.demon.co.uk/tutorial.html
 */

LM_REAL dlevmar_L2nrmxmy(LM_REAL* e, LM_REAL* x, LM_REAL* y, int n)
{
    const int blocksize = 8, bpwr = 3; /* 8=2^3 */
    int i;
    int j1, j2, j3, j4, j5, j6, j7;
    int blockn;
    LM_REAL sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

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

// ------------------------------------ misc_core.cpp ------------------------------------ //

#undef LM_REAL
#undef LM_REAL_EPSILON
