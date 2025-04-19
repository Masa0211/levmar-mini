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
#include <functional>

namespace levmar
{
    using Real = double;
    constexpr int LM_OPTS_SZ = 5;
    constexpr int LM_INFO_SZ = 10;
    constexpr int LM_ERROR = -1;
    constexpr Real LM_INIT_MU = 1.e-03;
    constexpr Real LM_STOP_THRESH = 1.e-17;
    constexpr Real LM_DIFF_DELTA = 1.e-06;

    class LevMar
    {
    public:

        LevMar(
            int numParams, // (= m) number of optimization parameters (i.e. #unknowns)
            int numPoints  // (= n) number of data points/observations
        );

        int dlevmar_dif(
            std::function<void(Real*, Real*, int numParams, int numPoints)> func,
            double* p, double* x, int itmax, double* opts,
            double* info, double* covar);


    private:
        /* work arrays size for dlevmar_der and dlevmar_dif functions.
         * should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
         */
        int inline workSize(
            int numParams, // (= m) parameter vector dimension (i.e. #unknowns)
            int numPoints // (= n) measurement vector dimension
        ) const noexcept
        {
            return 4 * (numPoints) + 4 * (numParams) + (numPoints) * (numParams) + (numParams) * (numParams);
        }

        int dAx_eq_b_LU(double* A, double* B, double* x, int numPoints);

        int numParams_;
        int numPoints_;
        std::vector<double> luBuffer_; // m * m matrix and m vector
        std::vector<int> luIdx_; // m vector
        std::vector<double> lmWork_; // memory for levmar main algorithm
    };
} // levmar
