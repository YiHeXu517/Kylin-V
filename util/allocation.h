/* Memory allocation implemention.
 * Used for all enabled type.
 * Some functions for special type are defined inside with
 * constexpr assert.
 * No iterators.
 * As father for mant different structures.
 */

#ifndef allocation_H
#define allocation_H

#include "global.h"

namespace KylinVib
{
    using std::malloc;
    using std::calloc;
    using std::free;
    using std::memcpy;
    using std::exit;
    using std::cout;
    using std::endl;
    using std::min;
    using std::swap;
    using std::random_device;
    using std::mt19937;
    typedef std::uniform_real_distribution<double> AverRan;

    template<typename T>
    class Alloc
    {
    private:
        // size
        INT size_;

        // data
        T * ptr_ = nullptr;

    public:
        /* ctor & dtor & assignment */

        Alloc() = default;

        Alloc(const Alloc<T> & t) : size_(t.size_)
        {
            ptr_ = (T *)malloc( t.size_*sizeof(T) );
            memcpy( ptr_, t.ptr_, t.size_*sizeof(T) );
        }

        Alloc(Alloc<T> && t) : size_(t.size_)
        {
            ptr_ = t.ptr_;
            t.ptr_ = nullptr;
        }

        Alloc(INT Size) : size_(Size)
        {
            ptr_ = (T *)calloc( Size, sizeof(T) );
        }

        ~Alloc()
        {
            if(ptr_) { free(ptr_); ptr_ = nullptr;  }
        }

        Alloc<T> & operator=(const Alloc<T> & rhs)
        {
            if(ptr_) {free(ptr_);}
            size_ = rhs.size_;
            ptr_ = (T *)malloc( rhs.size_*sizeof(T) );
            memcpy( ptr_, rhs.ptr_, rhs.size_*sizeof(T) );
            return *this;
        }

        Alloc<T> & operator=(Alloc<T> && rhs)
        {
            if(ptr_) {free(ptr_);}
            size_ = rhs.size_;
            ptr_ = rhs.ptr_;
            rhs.ptr_ = nullptr;
            return *this;
        }

        T & operator[](INT pos) 
        {
            if(pos>=size_) { cout << "Overflow by " << pos << " " << size_ << endl; exit(1);  }
            return ptr_[pos];  
        }

        /* visiting members */

        INT size() const
        {
            return size_;
        }

        T * ptr()
        {
            return ptr_;
        }

        const T * cptr() const
        {
            return ptr_;
        }

        void print() const
        {
            cout << "Built-in Array: " << endl;
            for(INT i(0);i<size_;++i) { cout << ptr_[i] << endl;  }
        }

        /* modification */

        // non-ctor initialization without pointer
        void init_size ( INT Size )
        {
            if(ptr_) { free(ptr_); }
            ptr_ = (T *)calloc( Size, sizeof(T) );
        }

        // non-ctor initialization with copying pointer
        void init_cp ( INT Size, const T * oth )
        {
            if(ptr_) { free(ptr_); }
            size_ = Size;
            ptr_  = (T *)malloc( Size*sizeof(T) );
            memcpy( ptr_, oth, Size*sizeof(T) );
        }

        // non-ctor initialization with copying pointer
        void init_mv ( INT Size, T * oth )
        {
            if(ptr_) { free(ptr_); }
            size_ = Size;
            ptr_  = oth;
            oth   = nullptr;
        }

        /* member functions for special type */
        double norm() const;
        T inner( const Alloc<T> & oth ) const;
        T inner( Alloc<T> && oth ) const;
    };

    /* member functions for special type */

    // norm of an real array
    template<> inline double Alloc<double>::norm() const
    {
        return cblas_dnrm2(this->size(), this->cptr(), 1);
    }

    // norm of an complex array
    template<> inline double Alloc<Complex>::norm() const
    {
        return cblas_dznrm2(this->size(), this->cptr(), 1);
    }

    // inner product of two complex array
    template<> inline Complex Alloc<Complex>::inner( const Alloc<Complex> & oth ) const
    {
        Complex res;
        cblas_zdotc_sub( this->size(), this->cptr(), 1, oth.cptr(), 1 , &res);
        return res;
    }

    // inner product of two complex array
    template<> inline Complex Alloc<Complex>::inner( Alloc<Complex> && oth ) const
    {
        Complex res;
        cblas_zdotc_sub( this->size(), this->cptr(), 1, oth.cptr(), 1 , &res);
        return res;
    }

    /* non-member functions */

    // x = x - a * y (mkl zaxpy interface)
    void self_minus (  const Alloc<Complex> & y, const Complex & alp, Alloc<Complex> & x )
    {
        Complex alpm = -1.0*alp;
        cblas_zaxpy ( y.size(), &alpm, y.cptr(), 1, x.ptr(), 1 );
    }

    // dot of two complex array
    void zdot ( const Alloc<Complex> & m1, const Alloc<Complex> & m2, INT nrow, INT ncol,
    INT nK, Alloc<Complex> & res, const CBLAS_TRANSPOSE T1, const CBLAS_TRANSPOSE T2 )
    {
        //mkl_set_num_threads(1);
        Complex alp(1,0),bet(0,0);
        INT ld1,ld2,ldr(ncol);
        if(T1==CblasNoTrans) { ld1 = nK; }
        else { ld1 = nrow; }
        if(T2==CblasNoTrans) { ld2 = ncol; }
        else { ld2 = nK; }

        cblas_zgemm ( CblasRowMajor, T1, T2,
            nrow, ncol, nK,
            &alp,
            m1.cptr(), ld1,
            m2.cptr(), ld2,
            &bet,
            res.ptr(), ldr
        );
    }

    // naive svd for complex
    void naive_svd ( Alloc<Complex> & a, INT nrow, INT ncol, Alloc<Complex> & U,
    Alloc<double> & s, Alloc<Complex> & Vt )
    {
        INT ldu = min(nrow,ncol);
        Alloc<double> sp(ldu-1);
        int ifsd = LAPACKE_zgesvd( LAPACK_ROW_MAJOR, 'S', 'S',
            nrow, ncol, a.ptr(), ncol,
            s.ptr(), U.ptr(), ldu, Vt.ptr(), ncol,
            sp.ptr()
        );

        if(ifsd!=0) { cout << "Problems in SVD" << endl; exit(1); }
    }

    // naive eigen for complex square matrix
    void naive_eig ( Alloc<Complex> & a, INT na, Alloc<double> & ega )
    {
        int ifsd = LAPACKE_zheevd( LAPACK_ROW_MAJOR, 'V', 'U',
            na, a.ptr(), na, ega.ptr()
        );

        if(ifsd!=0) { cout << "Problems in eigen" << endl; exit(1); }
    }

    // quasi-Gaussian function
    double quasi_gauss(double x, double bet)
    {
        double r = std::exp(-0.25*bet*pow(x,2.0)) * std::abs(0.25*x/bet);
        r += (0.5/bet - 0.25*std::pow(x,2.0)) * std::erfc(0.5*std::pow(bet,0.5)*abs(x))*0.5*std::pow(3.1415926/bet,0.5);
        r *= std::exp(-0.25*bet*std::pow(x,2.0))*4.0*bet*std::pow(bet/3.1415926,0.5);
        return r;
    }
    // inverse quasi-gaussian bisection
    double inv_quasi_gauss(double y, double bet)
    {
        double ULim=2.0, DLim=0.0, eps=1e-8,res;
        while((ULim-DLim)>eps)
        {
            res = 0.5*(ULim+DLim);
            if( (quasi_gauss(res,bet)-y)*(quasi_gauss(DLim,bet)-y) < 0 )
            {
                ULim = res;
            }
            else
            {
                DLim = res;
            }
        }
        return res;
    }

    // naive gram-schmidt
    void gs_ortho (
        INT nrow,
        INT ncol,
        Alloc<Complex> & a,
        bool extendR,
        INT increment,
        bool FillZero = false
    )
    {
        if(extendR) // row extend
        {
            Alloc<Complex> asw((nrow+increment)*ncol);
            cblas_zcopy(nrow*ncol,a.cptr(),1,asw.ptr(),1);
            swap(a,asw);
            if(FillZero) { return;  }
            for(INT i(0);i<increment;++i)
            {
                random_device rd;
                mt19937 gen(rd());
                AverRan dis(-1,1);
                for(INT jr(0);jr<ncol;++jr)
                {
                    a[(i+nrow)*ncol+jr] = dis(gen);
                }
                double nrm =  cblas_dznrm2 (ncol,a.cptr()+(i+nrow)*ncol,1);
                cblas_zdscal (ncol,1/nrm,a.ptr()+(i+nrow)*ncol,1 );
                for(INT j(0);j<i+nrow;++j)
                {
                    Complex dotc;
                    cblas_zdotc_sub (ncol, a.cptr()+j*ncol,1,a.cptr()+(i+nrow)*ncol,1, &dotc); dotc *= -1.0;
                    cblas_zaxpy (ncol,&dotc,a.cptr()+j*ncol,1,a.ptr()+(i+nrow)*ncol,1);

                }
                nrm =  cblas_dznrm2 (ncol,a.cptr()+(i+nrow)*ncol,1);
                cblas_zdscal (ncol,1/nrm,a.ptr()+(i+nrow)*ncol,1 );
            }
        }
        else
        {
            Alloc<Complex> asw((ncol+increment)*nrow);
            for(INT i(0);i<ncol;++i)
            {
                cblas_zcopy(nrow,a.cptr()+i,ncol,asw.ptr()+i,ncol+increment);
            }
            swap(a,asw);
            if(FillZero) { return;  }
            for(INT i(0);i<increment;++i)
            {
                random_device rd;
                mt19937 gen(rd());
                AverRan dis(-1,1);
                for(INT jr(0);jr<nrow;++jr)
                {
                    a[i+ncol+jr*(ncol+increment)] = dis(gen);
                }
                double nrm =  cblas_dznrm2 (nrow,a.cptr()+i+ncol,ncol+increment);
                cblas_zdscal (nrow,1/nrm,a.ptr()+i+ncol,ncol+increment);
                for(INT j(0);j<i+ncol;++j)
                {
                    Complex dotc;
                    cblas_zdotc_sub (nrow, a.cptr()+j,ncol+increment,a.cptr()+i+ncol,ncol+increment, &dotc); dotc *= -1.0;
                    cblas_zaxpy (nrow,&dotc,a.cptr()+j,ncol+increment,a.ptr()+i+ncol,ncol+increment);
                }
                nrm =  cblas_dznrm2 (nrow,a.cptr()+i+ncol,ncol+increment);
                cblas_zdscal (nrow,1/nrm,a.ptr()+i+ncol,ncol+increment);
            }
        }
    }
}
#endif
