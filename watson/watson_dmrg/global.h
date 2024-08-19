/* Global environment set and pre-definition */

#ifndef global_H
#define global_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fstream>
#include <functional>
#include <getopt.h>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#ifdef MKL_INT
#undef MKL_INT
#endif
#define MKL_INT int64_t
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
#include <mkl_spblas.h>
#include <mkl_solvers_ee.h>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace KylinVib
{
    /* some common types in codes */
    typedef MKL_INT INT;
    typedef MKL_Complex16 Complex;
    typedef std::initializer_list<INT> Brace;

    INT int_rand(INT high)
    {
        std::random_device rd; 
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0,high);
        return dis(gen);
    }
    double real_rand(double high)
    {
        std::random_device rd; 
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.*high,high);
        return dis(gen);
    }
}
#endif
