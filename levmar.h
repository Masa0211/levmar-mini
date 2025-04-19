#pragma once
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

#include <cmath>
#include <vector>
#include <memory>

namespace levmar
{
    using Real = double;

    class LevMar
    {
    public:
        constexpr static double LMA_OPTS_SZ = 5;
        constexpr static double LMA_INFO_SZ = 1;
        constexpr static double LMA_ERROR = -1;
        constexpr static double LMA_INIT_MU = 1.e-03;
        constexpr static double LMA_STOP_THRESH = 1.e-17;
        constexpr static double LMA_DIFF_DELTA = 1.e-06;

        LevMar(
            int n, // parameter vector dimension (i.e. #unknowns)
            int m  // measurement vector dimension
        );

        int dlevmar_dif(
            void (*func)(double* p, double* hx, int m, int n, void* adata),
            double* p, double* x, int m, int n, int itmax, double* opts,
            double* info, double* work, double* covar, void* adata);

        int dAx_eq_b_LU_noLapack(double* A, double* B, double* x, int n);

    private:
        /* work arrays size for dlevmar_der and dlevmar_dif functions.
         * should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
         */
        int inline workSize(int npar, int nmeas) const noexcept
        {
            return 4 * (nmeas)+4 * (npar)+(nmeas) * (npar)+(npar) * (npar);
        }

        /* covariance of LS fit */
        int dlevmar_covar(double* JtJ, double* C, double sumsq, int m, int n);

        int n_;
        int m_;
    };
} // levmar
