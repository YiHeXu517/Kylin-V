/* Base for sparse tensor class */

#pragma once

#include <initializer_list>
#include "timer.h"

namespace KylinVib
{
    using std::initializer_list;
    using std::copy;
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
            array<size_t,Rank> const & indices(size_t idx) const
            {
                return indices_[idx];
            }
            double const & values(size_t idx) const
            {
                return values_[idx];
            }

            void push_back(array<size_t,Rank> const & Idx, double Val)
            {
                indices_.push_back(Idx);
                values_.push_back(Val);
            }
            void puah_back(size_t pos, initializer_list<size_t> Idx, double Val)
            {
                array<size_t,Rank> Idxa;
                copy(Idx.begin(),Idx.end(),Idxa.begin());
                indices_.push_back(Idxa);
                values_.push_back(Val);
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
