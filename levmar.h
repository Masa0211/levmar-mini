/* 
////////////////////////////////////////////////////////////////////////////////////
// 
//  Prototypes and definitions for the Levenberg - Marquardt minimization algorithm
//  Copyright (C) 2004  Manolis Lourakis (lourakis at ics forth gr)
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
////////////////////////////////////////////////////////////////////////////////////
*/

#ifndef _LEVMAR_H_
#define _LEVMAR_H_


/************************************* Start of configuration options *************************************/
/* Note that when compiling with CMake, this configuration section is automatically generated
 * based on the user's input, see levmar.h.in
 */
                      
/* to avoid the overhead of repeated mallocs(), routines in axb.cpp can be instructed to
 * retain working memory between calls. Such a choice, however, renders these routines
 * non-reentrant and is not safe in a shared memory multiprocessing environment.
 * Bellow, an attempt is made to issue a warning if this option is turned on and OpenMP
 * is being used (note that this will work only if omp.h is included before levmar.h)
 */
#define LINSOLVERS_RETAIN_MEMORY
#if (defined(_OPENMP))
# ifdef LINSOLVERS_RETAIN_MEMORY
#  ifdef _MSC_VER
#  pragma message("LINSOLVERS_RETAIN_MEMORY is not safe in a multithreaded environment and should be turned off!")
#  else
#  warning LINSOLVERS_RETAIN_MEMORY is not safe in a multithreaded environment and should be turned off!
#  endif /* _MSC_VER */
# endif /* LINSOLVERS_RETAIN_MEMORY */
#endif /* _OPENMP */

/****************** End of configuration options, no changes necessary beyond this point ******************/


#ifdef __cplusplus
extern "C" {
#endif

/* work arrays size for ?levmar_der and ?levmar_dif functions.
 * should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
 */
#define LM_DER_WORKSZ(npar, nmeas) (2*(nmeas) + 4*(npar) + (nmeas)*(npar) + (npar)*(npar))
#define LM_DIF_WORKSZ(npar, nmeas) (4*(nmeas) + 4*(npar) + (nmeas)*(npar) + (npar)*(npar))

/* work arrays size for ?levmar_bc_der and ?levmar_bc_dif functions.
 * should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
 */
#define LM_BC_DER_WORKSZ(npar, nmeas) (2*(nmeas) + 4*(npar) + (nmeas)*(npar) + (npar)*(npar))
#define LM_BC_DIF_WORKSZ(npar, nmeas) LM_BC_DER_WORKSZ((npar), (nmeas)) /* LEVMAR_BC_DIF currently implemented using LEVMAR_BC_DER()! */

/* work arrays size for ?levmar_lec_der and ?levmar_lec_dif functions.
 * should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
 */
#define LM_LEC_DER_WORKSZ(npar, nmeas, nconstr) LM_DER_WORKSZ((npar)-(nconstr), (nmeas))
#define LM_LEC_DIF_WORKSZ(npar, nmeas, nconstr) LM_DIF_WORKSZ((npar)-(nconstr), (nmeas))

/* work arrays size for ?levmar_blec_der and ?levmar_blec_dif functions.
 * should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
 */
#define LM_BLEC_DER_WORKSZ(npar, nmeas, nconstr) LM_LEC_DER_WORKSZ((npar), (nmeas)+(npar), (nconstr))
#define LM_BLEC_DIF_WORKSZ(npar, nmeas, nconstr) LM_LEC_DIF_WORKSZ((npar), (nmeas)+(npar), (nconstr))

/* work arrays size for ?levmar_bleic_der and ?levmar_bleic_dif functions.
 * should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
 */
#define LM_BLEIC_DER_WORKSZ(npar, nmeas, nconstr1, nconstr2) LM_BLEC_DER_WORKSZ((npar)+(nconstr2), (nmeas)+(nconstr2), (nconstr1)+(nconstr2))
#define LM_BLEIC_DIF_WORKSZ(npar, nmeas, nconstr1, nconstr2) LM_BLEC_DIF_WORKSZ((npar)+(nconstr2), (nmeas)+(nconstr2), (nconstr1)+(nconstr2))

#define LM_OPTS_SZ    	 5 /* max(4, 5) */
#define LM_INFO_SZ    	 10
#define LM_ERROR         -1
#define LM_INIT_MU    	 1E-03
#define LM_STOP_THRESH	 1E-17
#define LM_DIFF_DELTA    1E-06
#define LM_VERSION       "2.6 (November 2011)"

/* double precision LM, with & without Jacobian */
/* unconstrained minimization */
extern int dlevmar_der(
      void (*func)(double *p, double *hx, int m, int n, void *adata),
      void (*jacf)(double *p, double *j, int m, int n, void *adata),
      double *p, double *x, int m, int n, int itmax, double *opts,
      double *info, double *work, double *covar, void *adata);

extern int dlevmar_dif(
      void (*func)(double *p, double *hx, int m, int n, void *adata),
      double *p, double *x, int m, int n, int itmax, double *opts,
      double *info, double *work, double *covar, void *adata);

/* box-constrained minimization */
extern int dlevmar_bc_der(
       void (*func)(double *p, double *hx, int m, int n, void *adata),
       void (*jacf)(double *p, double *j, int m, int n, void *adata),  
       double *p, double *x, int m, int n, double *lb, double *ub, double *dscl,
       int itmax, double *opts, double *info, double *work, double *covar, void *adata);

extern int dlevmar_bc_dif(
       void (*func)(double *p, double *hx, int m, int n, void *adata),
       double *p, double *x, int m, int n, double *lb, double *ub, double *dscl,
       int itmax, double *opts, double *info, double *work, double *covar, void *adata);


extern int dAx_eq_b_LU_noLapack(double *A, double *B, double *x, int n);


/* Jacobian verification, double & single precision */
extern void dlevmar_chkjac(
    void (*func)(double *p, double *hx, int m, int n, void *adata),
    void (*jacf)(double *p, double *j, int m, int n, void *adata),
    double *p, int m, int n, void *adata, double *err);


/* miscellaneous: standard deviation, coefficient of determination (R2),
 *                Pearson's correlation coefficient for best-fit parameters
 */
extern double dlevmar_stddev( double *covar, int m, int i);
extern double dlevmar_corcoef(double *covar, int m, int i, int j);
extern double dlevmar_R2(void (*func)(double *p, double *hx, int m, int n, void *adata), double *p, double *x, int m, int n, void *adata);


#ifdef __cplusplus
}
#endif

#endif /* _LEVMAR_H_ */
