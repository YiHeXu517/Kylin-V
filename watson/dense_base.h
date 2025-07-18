/* Base for dense tensor class */

#pragma once

#include <iostream>
#include <algorithm>
#include <complex>
#include <numeric>
#include <cstring>
#include <utility>
#define MKL_INT size_t
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
#include <mkl_spblas.h>
#include <mkl_solvers_ee.h>

namespace KylinVib
{
    using std::size_t;
    using std::cout;
    using std::malloc;
    using std::calloc;
    using std::free;
    using std::endl;
    using std::memcpy;
    using std::move;
    namespace Watson
    {
        template<typename ScalarType>
        class DenseBase
        {
            public:

            DenseBase() : size_(0), ptr_(nullptr)
            {

            }

            DenseBase(DenseBase<ScalarType> const & r)
            : size_(r.size_)
            {
                ptr_ = (ScalarType *)malloc(size_*sizeof(ScalarType));
                memcpy(ptr_,r.ptr_,size_*sizeof(ScalarType));
            }

            DenseBase(DenseBase<ScalarType> && r)
            : size_(r.size_)
            {
                ptr_ = r.ptr_;
                r.ptr_ = nullptr;
            }

            DenseBase(size_t Size)
            : size_(Size)
            {
                ptr_ = (ScalarType *)calloc(size_,sizeof(ScalarType));
            }

            ~DenseBase()
            {
                free(ptr_);
                ptr_ = nullptr;
            }

            DenseBase<ScalarType> & operator=(DenseBase<ScalarType> const & r)
            {
                free(ptr_);
                size_ = r.size_;
                ptr_ = (ScalarType *)malloc(size_*sizeof(ScalarType));
                memcpy(ptr_,r.ptr_,size_*sizeof(ScalarType));
                return *this;
            }

            DenseBase<ScalarType> & operator=(DenseBase<ScalarType> && r)
            {
                free(ptr_);
                size_ = r.size_;
                ptr_ = r.ptr_;
                r.ptr_ = nullptr;
                return *this;
            }

            size_t size() const
            {
                return size_;
            }

            ScalarType * ptr()
            {
                return ptr_;
            }

            ScalarType const * ptr() const
            {
                return ptr_;
            }

            void print() const
            {
                cout << "[ ";
                for(size_t i=0;i<size_;++i)
                {
                    cout << ptr_[i] << ' ';
                }
                cout << ']' << endl;
            }

            void initialize(size_t Sz)
            {
                free(ptr_);
                size_ = Sz;
                ptr_  = (ScalarType *)calloc(Sz,sizeof(ScalarType));
            }

            private:

            // size of allocation
            size_t size_;

            // saved data
            ScalarType * ptr_;
        };
    }
}
