/* Base for sparse tensor class */

#pragma once

#include <array>
#include <initializer_list>
#include <vector>
#include "timer.h"

namespace KylinVib
{
    using std::array;
    using std::initializer_list;
    using std::copy;
    using std::pow;
    using std::vector;
    namespace Watson
    {
        template<typename ScalarType, unsigned Rank>
        class SparseBase
        {
            public:

            SparseBase() = default;

            SparseBase(SparseBase<ScalarType,Rank> const & r)
            : shape_(r.shape_), indices_(r.indices_),values_(r.values_)
            {

            }

            SparseBase(SparseBase<ScalarType,Rank> && r)
            : shape_(move(r.shape_)), indices_(move(r.indices_)),values_(move(r.values_))
            {

            }
            SparseBase(array<size_t,Rank> const & Sp)
            : shape_(Sp)
            {

            }

            SparseBase(initializer_list<size_t> Sp)
            {
                copy(Sp.begin(),Sp.end(),shape_.begin());
            }

            SparseBase(array<size_t,Rank> const & Sp, size_t NNZ)
            : shape_(Sp), indices_(NNZ), values_(NNZ)
            {

            }

            SparseBase(initializer_list<size_t> Sp, size_t NNZ)
            : indices_(NNZ), values_(NNZ)
            {
                copy(Sp.begin(),Sp.end(),shape_.begin());
            }

            ~SparseBase() = default;

            SparseBase<ScalarType,Rank> & operator=(SparseBase<ScalarType,Rank> const & r)
            {
                shape_ = r.shape_;
                indices_ = r.indices_;
                values_ = r.values_;
                return *this;
            }

            SparseBase<ScalarType,Rank> & operator=(SparseBase<ScalarType,Rank> && r)
            {
                shape_ = move(r.shape_);
                indices_ = move(r.indices_);
                values_ = move(r.values_);
                return *this;
            }
            SparseBase<ScalarType,Rank> & operator*=(double coeff)
            {
                for(size_t idx=0;idx<this->nnz();++idx)
                {
                    values_[idx] *= coeff;
                }
                return *this;
            }
            SparseBase<ScalarType,Rank> operator*(double coeff)
            {
                SparseBase<ScalarType,Rank> res(*this);
                res *= coeff;
                return res;
            }

            size_t nnz() const
            {
                return indices_.size();
            }

            void print() const
            {
                cout << "Shape: [ ";
                for(unsigned i=0;i<Rank-1;++i)
                {
                    cout << shape_[i] << ',';
                }
                cout << shape_[Rank-1] <<  ']' << endl;
                for(size_t i=0;i<this->nnz();++i)
                {
                    cout << "[ ";
                    for(unsigned j=0;j<Rank-1;++j)
                    {
                        cout << indices_[i][j] << ',';
                    }
                    cout << indices_[i][Rank-1] << "] | " << std::scientific
                    << setprecision(8) << values_[i]  << endl;
                }
            }
            array<size_t,Rank> const & shape() const
            {
                return shape_;
            }
            array<size_t,Rank> & shape()
            {
                return shape_;
            }
            array<size_t,Rank> const & indices(size_t idx) const
            {
                return indices_[idx];
            }
            double const & values(size_t idx) const
            {
                return values_[idx];
            }
            array<size_t,Rank> & indices(size_t idx)
            {
                return indices_[idx];
            }
            double & values(size_t idx)
            {
                return values_[idx];
            }
            void push_back(array<size_t,Rank> const & Idx, double Val)
            {
                indices_.push_back(Idx);
                values_.push_back(Val);
            }
            void push_back(initializer_list<size_t> Idx, double Val)
            {
                array<size_t,Rank> Idxa;
                copy(Idx.begin(),Idx.end(),Idxa.begin());
                indices_.push_back(Idxa);
                values_.push_back(Val);
            }
            double norm() const
            {
                double res(0.0);
                for(size_t i=0;i<this->nnz();++i)
                {
                    res += pow(values_[i],2.0);
                }
                return sqrt(res);
            }
            ScalarType overlap(SparseBase<ScalarType,Rank> const & r) const
            {
                ScalarType res(0.0);
                for(size_t i=0;i<this->nnz();++i)
                {
                    for(size_t j=0;j<r.nnz();++j)
                    {
                        if( indices_[i] == r.indices_[j] )
                        {
                            res += values_[i] * r.values_[j];
                        }
                    }
                }
                return res;
            }

            private:

            // total dense shape
            array<size_t,Rank> shape_;

            // saved indices
            vector<array<size_t,Rank>> indices_;

            // saved data
            vector<double> values_;
        };
    }
}
