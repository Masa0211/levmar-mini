#include <iostream>
#include <iomanip>


#include "levmar.h"


constexpr double ROSD = 105.0;

/* Rosenbrock function, global minimum at (1, 1) */
void ros(double* p, double* x, int m, int n, void* data)
{
    for (auto i = 0; i < n; ++i)
        x[i] = ((1.0 - p[0]) * (1.0 - p[0]) + ROSD * (p[1] - p[0] * p[0]) * (p[1] - p[0] * p[0]));
}

void jacros(double* p, double* jac, int m, int n, void* data)
{
    for (auto i = 0, j = 0; i < n; ++i) {
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

#define LM_INFO_SZ    	 10
#define LM_ERROR         -1
#define LM_INIT_MU    	 1E-03
#define LM_STOP_THRESH	 1E-17
#define LM_DIFF_DELTA    1E-06

void test_ross()
{
    double info[LM_INFO_SZ];
    double opts[] = { LM_INIT_MU, 1E-15, 1E-15, 1E-20, LM_DIFF_DELTA };

    int m = 2;
    int n = 2;
    double x[] = { 0.0, 0.0 };

    {
        double p[] = { -1.2, 1.0 };
        const double expected[] = { 0.553823321867, 0.305535016043 };
        int ret = dlevmar_dif(ros, p, x, m, n, 1000, opts, info, NULL, NULL, NULL);  // no Jacobian
        std::cout << std::setprecision(12);
        std::cout << "differencial Jacobian" << std::endl;
        for (auto i = 0; i < m; ++i)
            //std::cout << "p[" << i << "] = " << p[i] << ", diff = " << p[i] - expected[i] << std::endl;
            check_value(p[i], expected[i]);

    }

}


/* Osborne's problem, minimum at (0.3754, 1.9358, -1.4647, 0.0129, 0.0221) */
void osborne(double* p, double* x, int m, int n, void* data)
{
    for (auto i = 0; i < n; ++i) {
        const double t = 10 * i;
        x[i] = p[0] + p[1] * std::exp(-p[3] * t) + p[2] * std::exp(-p[4] * t);
    }
}

void jacosborne(double* p, double* jac, int m, int n, void* data)
{
    for (auto i = 0, j = 0; i < n; ++i) {
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
    double x33[] = {
      8.44E-1, 9.08E-1, 9.32E-1, 9.36E-1, 9.25E-1, 9.08E-1, 8.81E-1,
      8.5E-1, 8.18E-1, 7.84E-1, 7.51E-1, 7.18E-1, 6.85E-1, 6.58E-1,
      6.28E-1, 6.03E-1, 5.8E-1, 5.58E-1, 5.38E-1, 5.22E-1, 5.06E-1,
      4.9E-1, 4.78E-1, 4.67E-1, 4.57E-1, 4.48E-1, 4.38E-1, 4.31E-1,
      4.24E-1, 4.2E-1, 4.14E-1, 4.11E-1, 4.06E-1 };

    double opts[] = { LM_INIT_MU, 1E-15, 1E-15, 1E-20, LM_DIFF_DELTA };
    double info[LM_INFO_SZ];

    int m = 5;
    int n = 33;

    const double theoretical[] = { 0.3754, 1.9358, -1.4647, 0.0129, 0.0221 };

    {
        std::cout << "Osborne differencial Jacobian" << std::endl;
        double p[] = { 0.5, 1.5, -1.0, 1.0E-2, 2.0E-2 };
        const double expected[] = { 0.375410053359,1.93584689416,-1.4646871224,0.0128675347011,0.0221227000131 };
        int ret = dlevmar_dif(osborne, p, x33, m, n, 1000, opts, info, NULL, NULL, NULL);  // no Jacobian
        for (auto i = 0; i < m; ++i)
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