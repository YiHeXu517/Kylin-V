/* tools for sparse algebras for watson model */

#pragma once

#include "operator.h"

namespace KylinVib
{
    using std::min;
    namespace Watson
    {
        typedef vector<SparseBase<double,4>> NaiveMPO;

        class MPS : public vector<DenseBase<double>>
        {
            public:
            MPS() = default;
            MPS(MPS const & r) : shapes_(r.shapes_), vector<DenseBase<double>>(r)
            {
            
            }
            MPS & operator=(MPS const & r)
            {
                shapes_ = r.shapes_;
                vector<DenseBase<double>>::operator=(r);
                return *this;
            }
            MPS(MPS && r) : shapes_(move(r.shapes_)), vector<DenseBase<double>>(move(r))
            {
            
            }
            MPS & operator=(MPS && r)
            {
                shapes_ = move(r.shapes_);
                vector<DenseBase<double>>::operator=(move(r));
                return *this;
            }
            MPS(size_t nsite) : shapes_(nsite), vector<DenseBase<double>>(nsite)
            {
            
            }
            array<size_t,3> const & get_shapes(size_t site) const 
            {
                return shapes_[site];
            }
            array<size_t,3> & set_shapes(size_t site)
            {
                return shapes_[site];
            }
            ~MPS() = default;

            void mps_svd(size_t site, size_t MaxBond, char l2r = 'r')
            {
                double alp(1.0),bet(0.0);
                if(l2r=='r') 
                {
                    size_t nrows = shapes_[site][0] * shapes_[site][1];
                    size_t ncols = shapes_[site][2];
                    size_t ldu = min(nrows,ncols);
                    DenseBase<double> U(nrows*ldu), s(ldu), Vt(ldu*ncols),sp(ldu-1);
                    int ifsd = LAPACKE_dgesvd( LAPACK_ROW_MAJOR, 'S', 'S',
                    nrows, ncols, (*this)[site].ptr(), ncols, s.ptr(), U.ptr(), ldu, Vt.ptr(), ncols, sp.ptr());
                    if(ldu<=MaxBond)
                    {
                        (*this)[site] = move(U);
                        for(size_t i=0;i<ldu;++i)
                        {
                            cblas_dscal(ncols,s.cptr()[i],Vt.ptr()+i*ncols,1);
                        }
                        nrows = ldu;
                        ncols = shapes_[site+1][1] * shapes_[site+1][2];
                        size_t nK = shapes_[site+1][0];
                        size_t ld1 = nK, ld2 = ncols, ldr = ncols;
                        DenseBase<double> res(nrows*ncols);
                        cblas_dgemm ( CblasRowMajor, CblasNoTrans, CblasNoTrans, nrows, ncols, nK,
                        alp, Vt.cptr(), ld1, (*this)[site+1].cptr(), ld2, bet, res.ptr(), ldr);
                        (*this)[site+1] = move(res);
                    }
                    else
                    {
                        DenseBase<double> Us(nrows*MaxBond),Vts(MaxBond*ncols);
                        for(size_t i=0;i<MaxBond;++i)
                        {
                            cblas_dcopy(nrows,U.cptr()+i,ldu,Us.ptr()+i,MaxBond);
                            cblas_daxpy(ncols,s.cptr()[i],Vt.cptr()+i*ncols,1,Vts.ptr()+i*ncols,1);
                        }
                        (*this)[site] = move(Us);
                        nrows = ldu;
                        ncols = shapes_[site+1][1] * shapes_[site+1][2];
                        size_t nK = shapes_[site+1][0];
                        size_t ld1 = nK, ld2 = ncols, ldr = ncols;
                        DenseBase<double> res(nrows*ncols);
                        cblas_dgemm ( CblasRowMajor, CblasNoTrans, CblasNoTrans, nrows, ncols, nK,
                        alp, Vts.cptr(), ld1, (*this)[site+1].cptr(), ld2, bet, res.ptr(), ldr);
                        (*this)[site+1] = move(res);
                    }
                }
                else
                {
                    size_t nrows = shapes_[site][0];
                    size_t ncols = shapes_[site][1]*shapes_[site][2];
                    size_t ldu = min(nrows,ncols);
                    DenseBase<double> U(nrows*ldu), s(ldu), Vt(ldu*ncols),sp(ldu-1);
                    int ifsd = LAPACKE_dgesvd( LAPACK_ROW_MAJOR, 'S', 'S',
                    nrows, ncols, (*this)[site].ptr(), ncols, s.ptr(), U.ptr(), ldu, Vt.ptr(), ncols, sp.ptr());
                    double alp(1.0),bet(0.0);
                    if(ldu<=MaxBond)
                    {
                        (*this)[site] = move(Vt);
                        for(size_t i=0;i<ldu;++i)
                        {
                            cblas_dscal(nrows,s.cptr()[i],U.ptr()+i,ldu);
                        }
                        ncols = ldu;
                        nrows = shapes_[site-1][0] * shapes_[site-1][1];
                        size_t nK = shapes_[site-1][2];
                        size_t ld1 = nK, ld2 = ncols, ldr = ncols;
                        DenseBase<double> res(nrows*ncols);
                        cblas_dgemm ( CblasRowMajor, CblasNoTrans, CblasNoTrans, nrows, ncols, nK,
                        alp, (*this)[site-1].cptr(), ld1, U.cptr(), ld2, bet, res.ptr(), ldr);
                        (*this)[site-1] = move(res);
                    }
                    else
                    {
                        DenseBase<double> Us(nrows*MaxBond),Vts(MaxBond*ncols);
                        for(size_t i=0;i<MaxBond;++i)
                        {
                            cblas_daxpy(nrows,s.cptr()[i],U.cptr()+i,ldu,Us.ptr()+i,MaxBond);
                            cblas_dcopy(ncols,Vt.cptr()+i*ncols,1,Vts.ptr()+i*ncols,1);
                        }
                        (*this)[site] = move(Vts);
                        ncols = ldu;
                        nrows = shapes_[site-1][0] * shapes_[site-1][1];
                        size_t nK = shapes_[site-1][2];
                        size_t ld1 = nK, ld2 = ncols, ldr = ncols;
                        DenseBase<double> res(nrows*ncols);
                        cblas_dgemm ( CblasRowMajor, CblasNoTrans, CblasNoTrans, nrows, ncols, nK,
                        alp, (*this)[site-1].cptr(), ld1, Us.cptr(), ld2, bet, res.ptr(), ldr);
                        (*this)[site-1] = move(res);
                    }
                } 
            }

            void canon(size_t MaxBond)
            {
                size_t ns = this->size();
                for(size_t i=0;i<ns-1;++i)
                {
                    mps_svd(i,MaxBond,'r');
                }
                for(size_t i=ns-1;i>0;--i)
                {
                    mps_svd(i,MaxBond,'l');
                }
            }

            void print() const
            {
                size_t ns = this->size();
                for(size_t i=0;i<ns;++i)
                {
                    cout << "Site-" << i+1 << endl;
                    cout << "Shape:[" << shapes_[i][0] << ","
                    << shapes_[i][1] << "," << shapes_[i][2]
                    << "]" << endl;
                    (*this)[i].print();
                }
            }

            private:
            vector<array<size_t,3>> shapes_;
        };
    }
}
