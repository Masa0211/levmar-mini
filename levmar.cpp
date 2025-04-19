

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cmath>

#include "levmar.h"
#include "levmar_utils.h"

using levmar::Real;

levmar::LevMar::LevMar(int m, int n)
    : m_(m)
    , n_(n)
    , luBuffer_(m * m + m)
    , luIdx_(m)
    , lmWork_(workSize(m, n))
{
}

/*
 * This function seeks the parameter vector p that best describes the measurements vector x.
 * More precisely, given a vector function  func : R^m --> R^n with n>=m,
 * it finds p s.t. func(p) ~= x, i.e. the squared second order (i.e. L2) norm of
 * e=x-func(p) is minimized.
 *
 * This function requires an analytic Jacobian. In case the latter is unavailable,
 * use LEVMAR_DIF() bellow
 *
 * Returns the number of iterations (>=0) if successful, LM_ERROR if failed
 *
 * For more details, see K. Madsen, H.B. Nielsen and O. Tingleff's lecture notes on
 * non-linear least squares at http://www.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf
 */

 /* Secant version of the dlevmar_der() function above: the Jacobian is approximated with
  * the aid of finite differences (forward or central, see the comment for the opts argument)
  */
int levmar::LevMar::dlevmar_dif(
    std::function<void(Real*, Real*, int m, int n)> func,
    Real* p,         /* I/O: initial parameter estimates. On output has the estimated solution */
    Real* x,         /* I: measurement vector. NULL implies a zero vector */
    int m,              /* I: parameter vector dimension (i.e. #unknowns) */
    int n,              /* I: measurement vector dimension */
    int itmax,          /* I: maximum number of iterations */
    Real opts[5],    /* I: opts[0-4] = minim. options [\mu, \epsilon1, \epsilon2, \epsilon3, \delta]. Respectively the
                         * scale factor for initial \mu, stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2 and
                         * the step used in difference approximation to the Jacobian. Set to NULL for defaults to be used.
                         * If \delta<0, the Jacobian is approximated with central differences which are more accurate
                         * (but slower!) compared to the forward differences employed by default.
                         */
    Real info[LM_INFO_SZ],
    /* O: information regarding the minimization. Set to NULL if don't care
    * info[0]= ||e||_2 at initial p.
    * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
    * info[5]= # iterations,
    * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
    *                                 2 - stopped by small Dp
    *                                 3 - stopped by itmax
    *                                 4 - singular matrix. Restart from current p with increased mu
    *                                 5 - no further error reduction is possible. Restart with increased mu
    *                                 6 - stopped by small ||e||_2
    *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
    * info[7]= # function evaluations
    * info[8]= # Jacobian evaluations
    * info[9]= # linear systems solved, i.e. # attempts for reducing error
    */
    Real* covar)    /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
{
    int i, j, k, l;
    int issolved;
    /* temp work arrays */
    Real* e,          /* nx1 */
        * hx,         /* \hat{x}_i, nx1 */
        * jacTe,      /* J^T e_i mx1 */
        * jac,        /* nxm */
        * jacTjac,    /* mxm */
        * Dp,         /* mx1 */
        * diag_jacTjac,   /* diagonal of J^T J, mx1 */
        * pDp,        /* p + Dp, mx1 */
        * wrk,        /* nx1 */
        * wrk2;       /* nx1, used only for holding a temporary e vector and when differentiating with central differences */

    int using_ffdif = 1;

    Real mu;  /* damping constant */
    Real tmp; /* mainly used in matrix & vector multiplications */
    Real p_eL2, jacTe_inf, pDp_eL2; /* ||e(p)||_2, ||J^T e||_inf, ||e(p+Dp)||_2 */
    Real p_L2, Dp_L2 = LM_REAL_MAX, dF, dL;
    Real tau, eps1, eps2, eps2_sq, eps3, delta;
    Real init_p_eL2;
    int nu, nu2, stop = 0, nfev, njap = 0, nlss = 0, K = (m >= 10) ? m : 10, updjac, updp = 1, newjac;
    const int nm = n * m;

    mu = jacTe_inf = p_L2 = 0.0; /* -Wall */
    updjac = newjac = 0; /* -Wall */

    if (n < m) {
        fprintf(stderr, "dlevmar_dif(): cannot solve a problem with fewer measurements [%d] than unknowns [%d]\n", n, m);
        return LM_ERROR;
    }

    if (opts) {
        tau = opts[0];
        eps1 = opts[1];
        eps2 = opts[2];
        eps2_sq = opts[2] * opts[2];
        eps3 = opts[3];
        delta = opts[4];
        if (delta < 0.0) {
            delta = -delta; /* make positive */
            using_ffdif = 0; /* use central differencing */
        }
    }
    else { // use default values
        tau = LM_INIT_MU;
        eps1 = LM_STOP_THRESH;
        eps2 = LM_STOP_THRESH;
        eps2_sq = LM_STOP_THRESH * LM_STOP_THRESH;
        eps3 = LM_STOP_THRESH;
        delta = LM_DIFF_DELTA;
    }

    /* set up work arrays */
    e = lmWork_.data();
    hx = e + n;
    jacTe = hx + n;
    jac = jacTe + m;
    jacTjac = jac + nm;
    Dp = jacTjac + m * m;
    diag_jacTjac = Dp + m;
    pDp = diag_jacTjac + m;
    wrk = pDp + m;
    wrk2 = wrk + n;

    /* compute e=x - f(p) and its L2 norm */
    func(p, hx, m, n); nfev = 1;
    /* ### e=x-hx, p_eL2=||e|| */
    p_eL2 = dlevmar_L2nrmxmy(e, x, hx, n);
    init_p_eL2 = p_eL2;
    if (!std::isfinite(p_eL2)) stop = 7;

    nu = 20; /* force computation of J */

    for (k = 0; k < itmax && !stop; ++k) {
        /* Note that p and e have been updated at a previous iteration */

        if (p_eL2 <= eps3) { /* error is small */
            stop = 6;
            break;
        }

        /* Compute the Jacobian J at p,  J^T J,  J^T e,  ||J^T e||_inf and ||p||^2.
         * The symmetry of J^T J is again exploited for speed
         */

        if ((updp && nu > 16) || updjac == K) { /* compute difference approximation to J */
            if (using_ffdif) { /* use forward differences */
                dlevmar_fdif_forw_jac_approx(func, p, hx, wrk, delta, jac, m, n);
                ++njap; nfev += m;
            }
            else { /* use central differences */
                dlevmar_fdif_cent_jac_approx(func, p, wrk, wrk2, delta, jac, m, n);
                ++njap; nfev += 2 * m;
            }
            nu = 2; updjac = 0; updp = 0; newjac = 1;
        }

        if (newjac) { /* Jacobian has changed, recompute J^T J, J^t e, etc */
            newjac = 0;

            /* J^T J, J^T e */
            if (nm <= MEMORY_BLOCK_SIZE_SQ) { // this is a small problem
                /* J^T*J_ij = \sum_l J^T_il * J_lj = \sum_l J_li * J_lj.
                 * Thus, the product J^T J can be computed using an outer loop for
                 * l that adds J_li*J_lj to each element ij of the result. Note that
                 * with this scheme, the accesses to J and JtJ are always along rows,
                 * therefore induces less cache misses compared to the straightforward
                 * algorithm for computing the product (i.e., l loop is innermost one).
                 * A similar scheme applies to the computation of J^T e.
                 * However, for large minimization problems (i.e., involving a large number
                 * of unknowns and measurements) for which J/J^T J rows are too large to
                 * fit in the L1 cache, even this scheme incures many cache misses. In
                 * such cases, a cache-efficient blocking scheme is preferable.
                 *
                 * Thanks to John Nitao of Lawrence Livermore Lab for pointing out this
                 * performance problem.
                 *
                 * Note that the non-blocking algorithm is faster on small
                 * problems since in this case it avoids the overheads of blocking.
                 */
                int l;
                Real alpha, * jaclm, * jacTjacim;

                /* looping downwards saves a few computations */
                for (i = m * m; i-- > 0; )
                    jacTjac[i] = 0.0;
                for (i = m; i-- > 0; )
                    jacTe[i] = 0.0;

                for (l = n; l-- > 0; ) {
                    jaclm = jac + l * m;
                    for (i = m; i-- > 0; ) {
                        jacTjacim = jacTjac + i * m;
                        alpha = jaclm[i]; //jac[l*m+i];
                        for (j = i + 1; j-- > 0; ) /* j<=i computes lower triangular part only */
                            jacTjacim[j] += jaclm[j] * alpha; //jacTjac[i*m+j]+=jac[l*m+j]*alpha

                        /* J^T e */
                        jacTe[i] += alpha * e[l];
                    }
                }

                for (i = m; i-- > 0; ) /* copy to upper part */
                    for (j = i + 1; j < m; ++j)
                        jacTjac[i * m + j] = jacTjac[j * m + i];
            }
            else { // this is a large problem
                /* Cache efficient computation of J^T J based on blocking
                 */
                dlevmar_trans_mat_mat_mult(jac, jacTjac, n, m);

                /* cache efficient computation of J^T e */
                for (i = 0; i < m; ++i)
                    jacTe[i] = 0.0;

                for (i = 0; i < n; ++i) {
                    Real* jacrow;

                    for (l = 0, jacrow = jac + i * m, tmp = e[i]; l < m; ++l)
                        jacTe[l] += jacrow[l] * tmp;
                }
            }

            /* Compute ||J^T e||_inf and ||p||^2 */
            for (i = 0, p_L2 = jacTe_inf = 0.0; i < m; ++i) {
                if (jacTe_inf < (tmp = std::abs(jacTe[i]))) jacTe_inf = tmp;

                diag_jacTjac[i] = jacTjac[i * m + i]; /* save diagonal entries so that augmentation can be later canceled */
                p_L2 += p[i] * p[i];
            }
            //p_L2=sqrt(p_L2);
        }

        /* check for convergence */
        if ((jacTe_inf <= eps1)) {
            Dp_L2 = 0.0; /* no increment for p in this case */
            stop = 1;
            break;
        }

        /* compute initial damping factor */
        if (k == 0) {
            for (i = 0, tmp = LM_REAL_MIN; i < m; ++i)
                if (diag_jacTjac[i] > tmp) tmp = diag_jacTjac[i]; /* find max diagonal element */
            mu = tau * tmp;
        }

        /* determine increment using adaptive damping */

        /* augment normal equations */
        for (i = 0; i < m; ++i)
            jacTjac[i * m + i] += mu;

        /* solve augmented equations */
        /* use the LU included with levmar */
        issolved = dAx_eq_b_LU(jacTjac, jacTe, Dp, m); ++nlss;

        if (issolved) {
            /* compute p's new estimate and ||Dp||^2 */
            for (i = 0, Dp_L2 = 0.0; i < m; ++i) {
                pDp[i] = p[i] + (tmp = Dp[i]);
                Dp_L2 += tmp * tmp;
            }
            //Dp_L2=sqrt(Dp_L2);

            if (Dp_L2 <= eps2_sq * p_L2) { /* relative change in p is small, stop */
                //if(Dp_L2<=eps2*(p_L2 + eps2)){ /* relative change in p is small, stop */
                stop = 2;
                break;
            }

            if (Dp_L2 >= (p_L2 + eps2) / (EPSILON * EPSILON)) { /* almost singular */
                //if(Dp_L2>=(p_L2+eps2)/EPSILON/* almost singular */
                stop = 4;
                break;
            }

            func(pDp, wrk, m, n); ++nfev; /* evaluate function at p + Dp */
            /* compute ||e(pDp)||_2 */
            /* ### wrk2=x-wrk, pDp_eL2=||wrk2|| */
            pDp_eL2 = dlevmar_L2nrmxmy(wrk2, x, wrk, n);
            if (!std::isfinite(pDp_eL2)) { /* sum of squares is not finite, most probably due to a user error.
                                      * This check makes sure that the loop terminates early in the case
                                      * of invalid input. Thanks to Steve Danauskas for suggesting it
                                      */

                stop = 7;
                break;
            }

            dF = p_eL2 - pDp_eL2;
            if (updp || dF > 0) { /* update jac */
                for (i = 0; i < n; ++i) {
                    for (l = 0, tmp = 0.0; l < m; ++l)
                        tmp += jac[i * m + l] * Dp[l]; /* (J * Dp)[i] */
                    tmp = (wrk[i] - hx[i] - tmp) / Dp_L2; /* (f(p+dp)[i] - f(p)[i] - (J * Dp)[i])/(dp^T*dp) */
                    for (j = 0; j < m; ++j)
                        jac[i * m + j] += tmp * Dp[j];
                }
                ++updjac;
                newjac = 1;
            }

            for (i = 0, dL = 0.0; i < m; ++i)
                dL += Dp[i] * (mu * Dp[i] + jacTe[i]);

            if (dL > 0.0 && dF > 0.0) { /* reduction in error, increment is accepted */
                tmp = (2.0 * dF / dL - 1.0);
                tmp = 1.0 - tmp * tmp * tmp;
                mu = mu * ((tmp >= ONE_THIRD) ? tmp : ONE_THIRD);
                nu = 2;

                for (i = 0; i < m; ++i) /* update p's estimate */
                    p[i] = pDp[i];

                for (i = 0; i < n; ++i) { /* update e, hx and ||e||_2 */
                    e[i] = wrk2[i]; //x[i]-wrk[i];
                    hx[i] = wrk[i];
                }
                p_eL2 = pDp_eL2;
                updp = 1;
                continue;
            }
        }

        /* if this point is reached, either the linear system could not be solved or
         * the error did not reduce; in any case, the increment must be rejected
         */

        mu *= nu;
        nu2 = nu << 1; // 2*nu;
        if (nu2 <= nu) { /* nu has wrapped around (overflown). Thanks to Frank Jordan for spotting this case */
            stop = 5;
            break;
        }
        nu = nu2;

        for (i = 0; i < m; ++i) /* restore diagonal J^T J entries */
            jacTjac[i * m + i] = diag_jacTjac[i];
    }

    if (k >= itmax) stop = 3;

    for (i = 0; i < m; ++i) /* restore diagonal J^T J entries */
        jacTjac[i * m + i] = diag_jacTjac[i];

    if (info) {
        info[0] = init_p_eL2;
        info[1] = p_eL2;
        info[2] = jacTe_inf;
        info[3] = Dp_L2;
        for (i = 0, tmp = LM_REAL_MIN; i < m; ++i)
            if (tmp < jacTjac[i * m + i]) tmp = jacTjac[i * m + i];
        info[4] = mu / tmp;
        info[5] = (Real)k;
        info[6] = (Real)stop;
        info[7] = (Real)nfev;
        info[8] = (Real)njap;
        info[9] = (Real)nlss;
    }

    /* covariance matrix */
    if (covar) {
        dlevmar_covar(jacTjac, covar, p_eL2, m, n);
    }

    return (stop != 4 && stop != 7) ? k : LM_ERROR;

} /* dlevmar_dif() */

// ------------------------------ axb.cpp ----------------------------

/*
 * This function returns the solution of Ax = b
 *
 * The function employs LU decomposition followed by forward/back substitution (see
 * also the LAPACK-based LU solver above)
 *
 * A is mxm, b is mx1
 *
 * The function returns 0 in case of error, 1 if successful
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */

int levmar::LevMar::dAx_eq_b_LU(Real* A, Real* B, Real* x, int m)
{

    if (!A)
        return 1; /* NOP */

    /* calculate required memory size */
    const auto a_sz = m * m;
    const auto a = luBuffer_.data();
    const auto work = a + a_sz;
    const auto idx = luIdx_.data();

    /* avoid destroying A, B by copying them to a, x resp. */
    memcpy(a, A, a_sz * sizeof(Real));
    memcpy(x, B, m * sizeof(Real));

    Real tmp = 0.0;

    /* compute the LU decomposition of a row permutation of matrix a; the permutation itself is saved in idx[] */
    for (auto i = 0; i < m; ++i) {
        Real max = 0.0;
        for (auto j = 0; j < m; ++j)
            if ((tmp = std::abs(a[i * m + j])) > max)
                max = tmp;
        if (max == 0.0) {
            fprintf(stderr, "Singular matrix A in dAx_eq_b_LU()!\n");
            return 0;
        }
        work[i] = 1.0 / max;
    }

    int maxi = -1;
    for (auto j = 0; j < m; ++j) {
        for (auto i = 0; i < j; ++i) {
            auto sum = a[i * m + j];
            for (auto k = 0; k < i; ++k)
                sum -= a[i * m + k] * a[k * m + j];
            a[i * m + j] = sum;
        }
        Real max = 0.0;
        for (auto i = j; i < m; ++i) {
            auto sum = a[i * m + j];
            for (auto k = 0; k < j; ++k)
                sum -= a[i * m + k] * a[k * m + j];
            a[i * m + j] = sum;
            if ((tmp = work[i] * std::abs(sum)) >= max) {
                max = tmp;
                maxi = i;
            }
        }
        if (j != maxi) {
            for (auto k = 0; k < m; ++k) {
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
            for (auto i = j + 1; i < m; ++i)
                a[i * m + j] *= tmp;
        }
    }

    /* The decomposition has now replaced a. Solve the linear system using
     * forward and back substitution
     */
    for (auto i = 0, k = 0; i < m; ++i) {
        auto j = idx[i];
        auto sum = x[j];
        x[j] = x[i];
        if (k != 0)
            for (auto j = k - 1; j < i; ++j)
                sum -= a[i * m + j] * x[j];
        else
            if (sum != 0.0)
                k = i + 1;
        x[i] = sum;
    }

    for (auto i = m - 1; i >= 0; --i) {
        auto sum = x[i];
        for (auto j = i + 1; j < m; ++j)
            sum -= a[i * m + j] * x[j];
        x[i] = sum / a[i * m + i];
    }
    return 1;
}


// ------------------------------ misc.cpp ---------------------------- //



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


