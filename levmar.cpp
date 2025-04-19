#include <iostream>
#include <cmath>

#include "levmar.h"
#include "levmar_utils.h"

using levmar::Real;

levmar::LevMar::LevMar(int numParams, int numPoints)
    : numParams_(numParams)
    , numPoints_(numPoints)
    , luBuffer_(numParams * numParams + numParams)
    , luIdx_(numParams)
    , lmWork_(workSize(numParams, numPoints))
    , info_()
{
    if (numParams > numPoints)
    {
        std::cerr << "LevMar: cannot solve a problem with fewer measurements " << numPoints << " than unknowns " << numParams << std::endl;
        throw;
    }
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
    std::function<void(Real*, Real*, int numParams, int numPoints)> func,
    Real* p,             /* I/O: initial parameter estimates. On output has the estimated solution */
    Real* x,             /* I: measurement vector. NULL implies a zero vector */
    int itmax,           /* I: maximum number of iterations */
    const Options& opts, /* I: options for the minimization */
    bool updateInfo)     /* I: if true, update the info_ structure with the results of the minimization */
    // Real* covar)         /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
{
    const int numParams = numParams_; // parameter vector dimension (i.e. #unknowns)
    const int numPoints = numPoints_; // measurement vector dimension

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

    Real mu;  /* damping constant */
    Real tmp; /* mainly used in matrix & vector multiplications */
    Real p_eL2, jacTe_inf, pDp_eL2; /* ||e(p)||_2, ||J^T e||_inf, ||e(p+Dp)||_2 */
    Real p_L2, Dp_L2 = LM_REAL_MAX, dF, dL;
    Real init_p_eL2;
    int nu, nu2, stop = 0, nfev, njap = 0, nlss = 0, updjac, updp = 1, newjac;
    const int K = (numParams >= 10) ? numParams : 10;
    const int nm = numPoints * numParams;

    mu = jacTe_inf = p_L2 = 0.0; /* -Wall */
    updjac = newjac = 0; /* -Wall */

    const Real tau = opts.mu;
    const Real eps1 = opts.eps1;
    const Real eps2 = opts.eps2;
    const Real eps2_sq = eps2 * eps2;
    const Real eps3 = opts.eps3;
    const Real delta = std::abs(opts.delta);
    // use central differencing if delta < 0.0
    const int using_ffdif = (delta >= 0.0) ? 1 : 0; // use forward differences by default

    /* set up work arrays */
    e = lmWork_.data();
    hx = e + numPoints;
    jacTe = hx + numPoints;
    jac = jacTe + numParams;
    jacTjac = jac + nm;
    Dp = jacTjac + numParams * numParams;
    diag_jacTjac = Dp + numParams;
    pDp = diag_jacTjac + numParams;
    wrk = pDp + numParams;
    wrk2 = wrk + numPoints;

    /* compute e=x - f(p) and its L2 norm */
    func(p, hx, numParams, numPoints); nfev = 1;
    /* ### e=x-hx, p_eL2=||e|| */
    p_eL2 = dlevmar_L2nrmxmy(e, x, hx, numPoints);
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
                dlevmar_fdif_forw_jac_approx(func, p, hx, wrk, delta, jac, numParams, numPoints);
                ++njap; nfev += numParams;
            }
            else { /* use central differences */
                dlevmar_fdif_cent_jac_approx(func, p, wrk, wrk2, delta, jac, numParams, numPoints);
                ++njap; nfev += 2 * numParams;
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
                for (i = numParams * numParams; i-- > 0; )
                    jacTjac[i] = 0.0;
                for (i = numParams; i-- > 0; )
                    jacTe[i] = 0.0;

                for (l = numPoints; l-- > 0; ) {
                    jaclm = jac + l * numParams;
                    for (i = numParams; i-- > 0; ) {
                        jacTjacim = jacTjac + i * numParams;
                        alpha = jaclm[i]; //jac[l*numParams+i];
                        for (j = i + 1; j-- > 0; ) /* j<=i computes lower triangular part only */
                            jacTjacim[j] += jaclm[j] * alpha; //jacTjac[i*numParams+j]+=jac[l*numParams+j]*alpha

                        /* J^T e */
                        jacTe[i] += alpha * e[l];
                    }
                }

                for (i = numParams; i-- > 0; ) /* copy to upper part */
                    for (j = i + 1; j < numParams; ++j)
                        jacTjac[i * numParams + j] = jacTjac[j * numParams + i];
            }
            else { // this is a large problem
                /* Cache efficient computation of J^T J based on blocking
                 */
                dlevmar_trans_mat_mat_mult(jac, jacTjac, numPoints, numParams);

                /* cache efficient computation of J^T e */
                for (i = 0; i < numParams; ++i)
                    jacTe[i] = 0.0;

                for (i = 0; i < numPoints; ++i) {
                    Real* jacrow;

                    for (l = 0, jacrow = jac + i * numParams, tmp = e[i]; l < numParams; ++l)
                        jacTe[l] += jacrow[l] * tmp;
                }
            }

            /* Compute ||J^T e||_inf and ||p||^2 */
            for (i = 0, p_L2 = jacTe_inf = 0.0; i < numParams; ++i) {
                if (jacTe_inf < (tmp = std::abs(jacTe[i]))) jacTe_inf = tmp;

                diag_jacTjac[i] = jacTjac[i * numParams + i]; /* save diagonal entries so that augmentation can be later canceled */
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
            for (i = 0, tmp = LM_REAL_MIN; i < numParams; ++i)
                if (diag_jacTjac[i] > tmp) tmp = diag_jacTjac[i]; /* find max diagonal element */
            mu = tau * tmp;
        }

        /* determine increment using adaptive damping */

        /* augment normal equations */
        for (i = 0; i < numParams; ++i)
            jacTjac[i * numParams + i] += mu;

        /* solve augmented equations */
        /* use the LU included with levmar */
        issolved = dAx_eq_b_LU(jacTjac, jacTe, Dp, numParams); ++nlss;

        if (issolved) {
            /* compute p's new estimate and ||Dp||^2 */
            for (i = 0, Dp_L2 = 0.0; i < numParams; ++i) {
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

            func(pDp, wrk, numParams, numPoints); ++nfev; /* evaluate function at p + Dp */
            /* compute ||e(pDp)||_2 */
            /* ### wrk2=x-wrk, pDp_eL2=||wrk2|| */
            pDp_eL2 = dlevmar_L2nrmxmy(wrk2, x, wrk, numPoints);
            if (!std::isfinite(pDp_eL2)) { /* sum of squares is not finite, most probably due to a user error.
                                      * This check makes sure that the loop terminates early in the case
                                      * of invalid input. Thanks to Steve Danauskas for suggesting it
                                      */

                stop = 7;
                break;
            }

            dF = p_eL2 - pDp_eL2;
            if (updp || dF > 0) { /* update jac */
                for (i = 0; i < numPoints; ++i) {
                    for (l = 0, tmp = 0.0; l < numParams; ++l)
                        tmp += jac[i * numParams + l] * Dp[l]; /* (J * Dp)[i] */
                    tmp = (wrk[i] - hx[i] - tmp) / Dp_L2; /* (f(p+dp)[i] - f(p)[i] - (J * Dp)[i])/(dp^T*dp) */
                    for (j = 0; j < numParams; ++j)
                        jac[i * numParams + j] += tmp * Dp[j];
                }
                ++updjac;
                newjac = 1;
            }

            for (i = 0, dL = 0.0; i < numParams; ++i)
                dL += Dp[i] * (mu * Dp[i] + jacTe[i]);

            if (dL > 0.0 && dF > 0.0) { /* reduction in error, increment is accepted */
                tmp = (2.0 * dF / dL - 1.0);
                tmp = 1.0 - tmp * tmp * tmp;
                mu = mu * ((tmp >= ONE_THIRD) ? tmp : ONE_THIRD);
                nu = 2;

                for (i = 0; i < numParams; ++i) /* update p's estimate */
                    p[i] = pDp[i];

                for (i = 0; i < numPoints; ++i) { /* update e, hx and ||e||_2 */
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

        for (i = 0; i < numParams; ++i) /* restore diagonal J^T J entries */
            jacTjac[i * numParams + i] = diag_jacTjac[i];
    }

    if (k >= itmax) stop = 3;

    for (i = 0; i < numParams; ++i) /* restore diagonal J^T J entries */
        jacTjac[i * numParams + i] = diag_jacTjac[i];

    if (updateInfo)
    {
        for (i = 0, tmp = LM_REAL_MIN; i < numParams; ++i)
            if (tmp < jacTjac[i * numParams + i]) tmp = jacTjac[i * numParams + i];

        Info infoOut = {
            init_p_eL2, p_eL2, jacTe_inf, Dp_L2, mu / tmp,
            k, stop, nfev, njap, nlss
        };
        info_ = infoOut;
    }

    /* covariance matrix */
    // not supported at the moment
    // if (covar) {
    //     dlevmar_covar(jacTjac, covar, p_eL2, numParams, numPoints);
    // }

    return (stop != 4 && stop != 7) ? k : LM_ERROR;

} /* dlevmar_dif() */

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

int levmar::LevMar::dAx_eq_b_LU(Real* A, Real* B, Real* x, int numParams)
{
    if (!A)
        return 1; /* NOP */

    /* calculate required memory size */
    const auto a_sz = numParams * numParams;
    const auto a = luBuffer_.data();
    const auto work = a + a_sz;
    const auto idx = luIdx_.data();

    /* avoid destroying A, B by copying them to a, x resp. */
    memcpy(a, A, a_sz * sizeof(Real));
    memcpy(x, B, numParams * sizeof(Real));

    Real tmp = 0.0;

    /* compute the LU decomposition of a row permutation of matrix a; the permutation itself is saved in idx[] */
    for (auto i = 0; i < numParams; ++i) {
        Real max = 0.0;
        for (auto j = 0; j < numParams; ++j)
            if ((tmp = std::abs(a[i * numParams + j])) > max)
                max = tmp;
        if (max == 0.0) {
            std::cerr << "Singular matrix A in dAx_eq_b_LU()!" << std::endl;
            return 0;
        }
        work[i] = 1.0 / max;
    }

    int maxi = -1;
    for (auto j = 0; j < numParams; ++j) {
        for (auto i = 0; i < j; ++i) {
            auto sum = a[i * numParams + j];
            for (auto k = 0; k < i; ++k)
                sum -= a[i * numParams + k] * a[k * numParams + j];
            a[i * numParams + j] = sum;
        }
        Real max = 0.0;
        for (auto i = j; i < numParams; ++i) {
            auto sum = a[i * numParams + j];
            for (auto k = 0; k < j; ++k)
                sum -= a[i * numParams + k] * a[k * numParams + j];
            a[i * numParams + j] = sum;
            if ((tmp = work[i] * std::abs(sum)) >= max) {
                max = tmp;
                maxi = i;
            }
        }
        if (j != maxi) {
            for (auto k = 0; k < numParams; ++k) {
                tmp = a[maxi * numParams + k];
                a[maxi * numParams + k] = a[j * numParams + k];
                a[j * numParams + k] = tmp;
            }
            work[maxi] = work[j];
        }
        idx[j] = maxi;
        if (a[j * numParams + j] == 0.0)
            a[j * numParams + j] = LM_REAL_EPSILON;
        if (j != numParams - 1) {
            tmp = 1.0 / (a[j * numParams + j]);
            for (auto i = j + 1; i < numParams; ++i)
                a[i * numParams + j] *= tmp;
        }
    }

    /* The decomposition has now replaced a. Solve the linear system using
     * forward and back substitution
     */
    for (auto i = 0, k = 0; i < numParams; ++i) {
        auto j = idx[i];
        auto sum = x[j];
        x[j] = x[i];
        if (k != 0)
            for (auto j = k - 1; j < i; ++j)
                sum -= a[i * numParams + j] * x[j];
        else
            if (sum != 0.0)
                k = i + 1;
        x[i] = sum;
    }

    for (auto i = numParams - 1; i >= 0; --i) {
        auto sum = x[i];
        for (auto j = i + 1; j < numParams; ++j)
            sum -= a[i * numParams + j] * x[j];
        x[i] = sum / a[i * numParams + i];
    }
    return 1;
}
