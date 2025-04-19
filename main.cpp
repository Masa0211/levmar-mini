#include <iostream>
#include <iomanip>
#include <chrono>

#include "levmar.h"
#include "levmar_utils.h"

constexpr double ROSD = 105.0;

/* Rosenbrock function, global minimum at (1, 1) */
void ros(double* p, double* x, int numParams, int numPoints, void* data)
{
    for (auto i = 0; i < numPoints; ++i)
        x[i] = ((1.0 - p[0]) * (1.0 - p[0]) + ROSD * (p[1] - p[0] * p[0]) * (p[1] - p[0] * p[0]));
}

void jacros(double* p, double* jac, int numParams, int numPoints, void* data)
{
    for (auto i = 0, j = 0; i < numPoints; ++i) {
        jac[j++] = (-2 + 2 * p[0] - 4 * ROSD * (p[1] - p[0] * p[0]) * p[0]);
        jac[j++] = (2 * ROSD * (p[1] - p[0] * p[0]));
    }
}

void check_value(double value, double expected, double tol = 3.5e-12)
{
    auto diff = value / expected - 1.0;
    if (std::abs(diff) > tol)
    {
        std::cout << "TEST FAILED: diff = " << diff << " : " << expected << " was expected but " << value << std::endl;
    }
    else
    {
        std::cout << "TEST PASSED" << std::endl;
    }
}

void test_ross()
{
    using namespace levmar;

    double info[LM_INFO_SZ];
    levmar::LevMar::Options opts = { LM_INIT_MU, 1E-15, 1E-15, 1E-20, LM_DIFF_DELTA };

    int numParams = 2;
    int numPoints = 2;
    double x[] = { 0.0, 0.0 };

    {
        double p[] = { -1.2, 1.0 };
        const double expected[] = { 0.553823321867, 0.305535016043 };
        levmar::LevMar levmar(numParams, numPoints);

        auto targetFunc = [](double* p, double* x, int numParams, int numPoints) {
            ros(p, x, numParams, numPoints, nullptr);
        };

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000; ++i)
        {
            p[0] = -1.2; p[1] = 1.0;
            int ret = levmar.dlevmar_dif(targetFunc, p, x, 1000, opts, info, NULL);  // no Jacobianlevmar
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;

        std::cout << std::setprecision(12);
        std::cout << "differencial Jacobian" << std::endl;
        for (auto i = 0; i < numParams; ++i)
            //std::cout << "p[" << i << "] = " << p[i] << ", diff = " << p[i] - expected[i] << std::endl;
            check_value(p[i], expected[i]);
    }

}


/* Osborne's problem, minimum at (0.3754, 1.9358, -1.4647, 0.0129, 0.0221) */
void osborne(double* p, double* x, int numParams, int numPoints, void* data)
{
    for (auto i = 0; i < numPoints; ++i) {
        const double t = 10 * i;
        x[i] = p[0] + p[1] * std::exp(-p[3] * t) + p[2] * std::exp(-p[4] * t);
    }
}

void jacosborne(double* p, double* jac, int numParams, int numPoints, void* data)
{
    for (auto i = 0, j = 0; i < numPoints; ++i) {
        const double t = 10 * i;
        const double tmp1 = std::exp(-p[3] * t);
        const double tmp2 = std::exp(-p[4] * t);

        jac[j++] = 1.0;
        jac[j++] = tmp1;
        jac[j++] = tmp2;
        jac[j++] = -p[1] * t * tmp1;
        jac[j++] = -p[2] * t * tmp2;
    }
}

void test_osborne()
{
    using namespace levmar;

    double x33[] = {
      8.44E-1, 9.08E-1, 9.32E-1, 9.36E-1, 9.25E-1, 9.08E-1, 8.81E-1,
      8.5E-1, 8.18E-1, 7.84E-1, 7.51E-1, 7.18E-1, 6.85E-1, 6.58E-1,
      6.28E-1, 6.03E-1, 5.8E-1, 5.58E-1, 5.38E-1, 5.22E-1, 5.06E-1,
      4.9E-1, 4.78E-1, 4.67E-1, 4.57E-1, 4.48E-1, 4.38E-1, 4.31E-1,
      4.24E-1, 4.2E-1, 4.14E-1, 4.11E-1, 4.06E-1 };

    levmar::LevMar::Options opts = { LM_INIT_MU, 1E-15, 1E-15, 1E-20, LM_DIFF_DELTA };
    double info[LM_INFO_SZ];

    int numParams = 5;
    int numPoints = 33;

    const double theoretical[] = { 0.3754, 1.9358, -1.4647, 0.0129, 0.0221 };

    {
        std::cout << "Osborne differencial Jacobian" << std::endl;
        double p[] = { 0.5, 1.5, -1.0, 1.0E-2, 2.0E-2 };
        const double expected[] = { 0.375410053359,1.93584689416,-1.4646871224,0.0128675347011,0.0221227000131 };
        levmar::LevMar levmar(numParams, numPoints);

        auto targetFunc = [](double* p, double* x, int numParams, int numPoints) {
            osborne(p, x, numParams, numPoints, nullptr);
            };

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000; ++i)
        {
            p[0] = 0.5; p[1] = 1.5; p[2] = -1.0; p[3] = 1.0E-2; p[4] = 2.0E-2;
            int ret = levmar.dlevmar_dif(targetFunc, p, x33, 1000, opts, info, NULL);  // no Jacobian
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;

        for (auto i = 0; i < numParams; ++i)
            //std::cout << "p[" << i << "] = " << p[i] << ", diff = " << p[i] - expected[i] << std::endl;
            check_value(p[i], expected[i]);
    }
}

int main()
{
    test_ross();
    test_osborne();
    return 0;
}