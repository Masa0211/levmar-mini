/*
////////////////////////////////////////////////////////////////////////////////////
//
//  Prototypes and definitions for the Levenberg - Marquardt minimization algorithm
//  Copyright (C) 2004  Manolis Lourakis (lourakis at ics forth gr)
//  Institute of Computer Science, Foundation for Research & Technology - Hellas
//  Heraklion, Crete, Greece.
//
//  This implementation Copyright (C) 2025 Masahiro Ohta
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
#include <vector>
#include <memory>
#include <functional>
#include <optional>

namespace levmar
{
    using Real = double;
    using RealPtr = Real*;
    using ConstPtr = const Real*;

    constexpr int LM_OPTS_SZ = 5;
    constexpr int LM_INFO_SZ = 10;
    constexpr int LM_ERROR = -1;
    constexpr Real LM_INIT_MU = 1.e-03;
    constexpr Real LM_STOP_THRESH = 1.e-17;
    constexpr Real LM_DIFF_DELTA = 1.e-06;

    class LevMar
    {
    public:
        /* opts[0-4] = minim. options [\mu, \epsilon1, \epsilon2, \epsilon3, \delta]. Respectively the
         * scale factor for initial \mu, stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2 and
         * the step used in difference approximation to the Jacobian. Set to NULL for defaults to be used.
         * If \delta<0, the Jacobian is approximated with central differences which are more accurate
         * (but slower!) compared to the forward differences employed by default.
        */
        struct Options
        {
            Real mu = LM_INIT_MU; // initial value for the damping factor
            Real eps1 = LM_STOP_THRESH; // stopping threshold for ||J^T e||_inf
            Real eps2 = LM_STOP_THRESH; // stopping threshold for ||Dp||_2
            Real eps3 = LM_STOP_THRESH; // stopping threshold for ||e||_2
            Real delta = LM_DIFF_DELTA; // step used in difference approximation to the Jacobian
        };

        /* information regarding the minimization. Set to NULL if don't care
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
        struct Info
        {
            Real eL2 = 0.0; // ||e||_2 at initial p
            Real p_eL2 = 0.0; // ||e||_2 at estimated p
            Real jacTe_inf = 0.0; // ||J^T e||_inf at estimated p
            Real Dp_L2 = 0.0; // ||Dp||_2 at estimated p
            Real mu_max_diag_jacTjac = 0.0; // mu/max[J^T J]_ii at estimated p
            int k = 0; // # iterations
            int stop = 0; // reason for terminating
            int nfev = 0; // # function evaluations
            int njap = 0; // # Jacobian evaluations
            int nlss = 0; // # linear systems solved, i.e. # attempts for reducing error
        };

        LevMar(
            int numParams, // (= m) number of optimization parameters (i.e. #unknowns)
            int numPoints  // (= n) number of data points/observations
        );

        int dlevmar_dif(
            std::function<void(RealPtr, RealPtr, int numParams, int numPoints)> func,
            std::vector<Real>& params,
            const std::vector<Real>& samplePoints,
            int itmax,
            const Options& opts,
            bool updateInfo = false);
            // double* covar = nullptr);

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

        int dAx_eq_b_LU(ConstPtr A, ConstPtr B, RealPtr x, int numPoints);

        int numParams_;
        int numPoints_;
        std::vector<Real> luBuffer_; // m * m matrix and m vector
        std::vector<int> luIdx_; // m vector
        std::vector<Real> lmWork_; // memory for levmar main algorithm

        Info info_;
    };
} // levmar
