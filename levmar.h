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

/* work arrays size for dlevmar_der and dlevmar_dif functions.
 * should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
 */
#define LM_DER_WORKSZ(npar, nmeas) (2*(nmeas) + 4*(npar) + (nmeas)*(npar) + (npar)*(npar))
#define LM_DIF_WORKSZ(npar, nmeas) (4*(nmeas) + 4*(npar) + (nmeas)*(npar) + (npar)*(npar))

#define LM_OPTS_SZ    	 5 /* max(4, 5) */
#define LM_INFO_SZ    	 10
#define LM_ERROR         -1
#define LM_INIT_MU    	 1E-03
#define LM_STOP_THRESH	 1E-17
#define LM_DIFF_DELTA    1E-06
#define LM_VERSION       "2.6 (November 2011)"

/* double precision LM, with & without Jacobian */
/* unconstrained minimization */
//int dlevmar_der(
//      void (*func)(double *p, double *hx, int m, int n, void *adata),
//      void (*jacf)(double *p, double *j, int m, int n, void *adata),
//      double *p, double *x, int m, int n, int itmax, double *opts,
//      double *info, double *work, double *covar, void *adata);

int dlevmar_dif(
      void (*func)(double *p, double *hx, int m, int n, void *adata),
      double *p, double *x, int m, int n, int itmax, double *opts,
      double *info, double *work, double *covar, void *adata);

int dAx_eq_b_LU_noLapack(double *A, double *B, double *x, int n);

#endif /* _LEVMAR_H_ */
