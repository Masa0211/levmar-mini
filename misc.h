/////////////////////////////////////////////////////////////////////////////////
// 
//  Levenberg - Marquardt non-linear minimization algorithm
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
/////////////////////////////////////////////////////////////////////////////////

#ifndef _MISC_H_
#define _MISC_H_

#define __BLOCKSZ__       32 /* block size for cache-friendly matrix-matrix multiply. It should be
                              * such that __BLOCKSZ__^2*sizeof(LM_REAL) is smaller than the CPU (L1)
                              * data cache size. Notice that a value of 32 when LM_REAL=double assumes
                              * an 8Kb L1 data cache (32*32*8=8K). This is a concervative choice since
                              * newer Pentium 4s have a L1 data cache of size 16K, capable of holding
                              * up to 45x45 double blocks.
                              */
#define __BLOCKSZ__SQ    (__BLOCKSZ__)*(__BLOCKSZ__)

/* blocking-based matrix multiply */
void dlevmar_trans_mat_mat_mult(double *a, double *b, int n, int m);

/* forward finite differences */
void dlevmar_fdif_forw_jac_approx(void (*func)(double *p, double *hx, int m, int n, void *adata),
					double *p, double *hx, double *hxx, double delta,
					double *jac, int m, int n, void *adata);

/* central finite differences */
void dlevmar_fdif_cent_jac_approx(void (*func)(double *p, double *hx, int m, int n, void *adata),
          double *p, double *hxm, double *hxp, double delta,
          double *jac, int m, int n, void *adata);

/* e=x-y and ||e|| */
double dlevmar_L2nrmxmy(double *e, double *x, double *y, int n);

/* covariance of LS fit */
int dlevmar_covar(double *JtJ, double *C, double sumsq, int m, int n);


#endif /* _MISC_H_ */
