/* Global environment set and pre-definition */

#ifndef global_H
#define global_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
//#include </public/software/compiler/gcc-9.3.0/include/c++/9.3.0/complex>
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
#define MKL_INT int
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
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
}
#endif
