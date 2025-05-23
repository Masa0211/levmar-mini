/*
////////////////////////////////////////////////////////////////////////////////////
//
//  Utility functions for the Levenberg - Marquardt minimization algorithm
//  Copyright (C) 2004  Manolis Lourakis (lourakis at ics forth gr)
//  Institute of Computer Science, Foundation for Research & Technology - Hellas
//  Heraklion, Crete, Greece.
//
//  This C++ implementation Copyright (C) 2025 Masahiro Ohta
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
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
//
////////////////////////////////////////////////////////////////////////////////////
*/

#pragma once

#include <cmath>
#include <functional>

namespace levmar
{
    using Real = double;
    using RealPtr = Real*;
    using ConstPtr = const Real*;
    using RealVector = std::vector<Real>;
    using RealVectorRef = std::vector<Real>&;
    using ConstRealVectorRef = const std::vector<Real>&;

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
    constexpr Real LM_REAL_MIN = -DBL_MAX;

    constexpr Real EPSILON = 1E-12;
    constexpr Real ONE_THIRD = 0.3333333334; /* 1.0/3.0 */
    //constexpr Real ONE_THIRD = 1.0 / 3.0;

    /* blocking-based matrix multiply */
    void dlevmar_trans_mat_mat_mult(RealPtr a, RealPtr b, int numPoints, int numParams);

    /* forward finite differences */
    void dlevmar_fdif_forw_jac_approx(
        std::function<void(RealVectorRef, RealVectorRef, int numParams, int numPoints)> func,
        RealVectorRef p, RealPtr hx, RealVectorRef hxx, Real delta,
        RealPtr jac, int numParams, int numPoints);

    /* central finite differences */
    void dlevmar_fdif_cent_jac_approx(
        std::function<void(RealVectorRef, RealVectorRef, int numParams, int numPoints)> func,
        RealVectorRef p, RealVectorRef hxm, RealVectorRef hxp, Real delta,
        RealPtr jac, int numParams, int numPoints);

    /* e=x-y and ||e|| */
    Real dlevmar_L2nrmxmy(RealPtr e, ConstPtr x, ConstPtr y, int numPoints);

    /* covariance of LS fit */
    int dlevmar_covar(RealPtr JtJ, RealPtr C, Real sumsq, int numParams, int numPoints);

    /* This function computes the inverse of A in B. A and B can coincide */
    int dlevmar_LUinverse(RealPtr A, RealPtr B, int numParams);

}
