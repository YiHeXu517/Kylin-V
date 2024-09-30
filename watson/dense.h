/* dense tensor class */

#pragma once

#include <complex>
#include <array>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iomanip>
#include <random>

#define MKL_INT size_t
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
#include <mkl_spblas.h>
#include <mkl_solvers_ee.h>

#include "allocator.h"

namespace KylinVib
{
  template<typename Type, size_t Rank>
  class Dense : public Allocator<Type>
  {
    public:
    Dense() = default;
    Dense(Dense<Type,Rank> const & r)
    : shape_(r.shape_), dist_(r.dist_), Allocator<Type>(r)
    {
    
    }
    Dense(Dense<Type,Rank> && r)
    : shape_(std::move(r.shape_)), dist_(std::move(r.dist_)), 
      Allocator<Type>(std::move(r))
    {
    
    }
    Dense(std::array<size_t,Rank> const & Sp)
    : shape_(Sp), Allocator<Type>(std::accumulate(Sp.begin(),Sp.end(),1,std::multiplies<size_t>()))
    {
      dist_[Rank-1] = 1;
      for(size_t i=1;i<Rank;++i)
      {
        dist_[Rank-i-1] = dist_[Rank-i] * shape_[Rank-i];
      }
    }
    Dense(std::initializer_list<size_t> Sp)
    : Allocator<Type>(std::accumulate(Sp.begin(),Sp.end(),1,std::multiplies<size_t>()))
    {
      std::copy(Sp.begin(),Sp.end(),shape_.begin());
      dist_[Rank-1] = 1;
      for(size_t i=1;i<Rank;++i)
      {
        dist_[Rank-i-1] = dist_[Rank-i] * shape_[Rank-i];
      }
    }
    ~Dense() = default;
    Dense<Type,Rank> & operator=(Dense<Type,Rank> const & r)
    {
      shape_ = r.shape_;
      dist_  = r.dist_;
      Allocator<Type>::operator=(r);
      return *this;
    }
    Dense<Type,Rank> & operator=(Dense<Type,Rank> && r)
    {
      shape_ = std::move(r.shape_);
      dist_  = std::move(r.dist_);
      Allocator<Type>::operator=(std::move(r));
      return *this;
    }
    Type & operator()(std::array<size_t,Rank> const & idx)
    {
      size_t pos = std::inner_product(idx.begin(),idx.end(),dist_.begin(),0);
      if(pos>this->size())
      {
        std::cout << "Overflow!" << std::endl;
      }
      return this->ptr()[pos];
    }
    Type & operator()(std::initializer_list<size_t> idx)
    {
      size_t pos = std::inner_product(idx.begin(),idx.end(),dist_.begin(),0);
      if(pos>this->size())
      {
        std::cout << "Overflow!" << std::endl;
      }
      return this->ptr()[pos];
    }
    Type const & operator()(std::array<size_t,Rank> const & idx) const
    {
      size_t pos = std::inner_product(idx.begin(),idx.end(),dist_.begin(),0);
      if(pos>this->size())
      {
        std::cout << "Overflow!" << std::endl;
      }
      return this->ptr()[pos];
    }
    Type const & operator()(std::initializer_list<size_t> idx) const
    {
      size_t pos = std::inner_product(idx.begin(),idx.end(),dist_.begin(),0);
      if(pos>this->size())
      {
        std::cout << "Overflow!" << std::endl;
      }
      return this->ptr()[pos];
    }
    std::array<size_t,Rank> const & shape() const
    {
      return shape_;
    }
    std::array<size_t,Rank> const & dist() const
    {
      return dist_;
    }
    std::array<size_t,Rank> make_indices(size_t idx) const
    {
      std::array<size_t,Rank> res;
      for(size_t i=0;i<Rank;++i)
      {
        res[i] = idx / dist_[i] % shape_[i];
      }
      return res;
    }
    void rand_fill()
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<double> dis(-1,1);
      for(size_t i=0;i<this->size();++i)
      {
        this->ptr()[i] = dis(gen);  
      }
    }
    void print(double tol = 1e-14) const
    {
      std::cout << "Shape:[";
      for(size_t i=0;i<Rank-1;++i)
      {
        std::cout << shape_[i] << ",";
      }
      std::cout << shape_[Rank-1] << "]" << std::endl;
      for(size_t i=0;i<this->size();++i)
      {
        if( std::abs(this->ptr()[i]) <= tol  ) { continue; }
        std::array<size_t,Rank> idx = make_indices(i);
        std::cout << "[";
        for(size_t j=0;j<Rank-1;++j)
        {
          std::cout << idx[j]  << ",";
        }
        std::cout << idx[Rank-1] << "] | " << std::scientific 
        << std::setprecision(8) << this->ptr()[i] << std::endl;
      }
    }
    // arithmetic
    Dense<Type,Rank> & operator+=(Dense<Type,Rank> const & r)
    {
      Type ones(1.0);
      if constexpr (std::is_same<Type,double>::value)
      {
        cblas_daxpy(this->size(),ones,r.ptr(),1,this->ptr(),1);
      }
      else if constexpr (std::is_same<Type,MKL_Complex16>::value)
      {
        cblas_zaxpy(this->size(),&ones,r.ptr(),1,this->ptr(),1);
      }
      return *this;
    }
    Dense<Type,Rank> operator+(Dense<Type,Rank> const & r) const
    {
      Dense<Type,Rank> res(*this);
      res += r;
      return res;
    }
    Dense<Type,Rank> & operator-=(Dense<Type,Rank> const & r)
    {
      Type ones(-1.0);
      if constexpr (std::is_same<Type,double>::value)
      {
        cblas_daxpy(this->size(),ones,r.ptr(),1,this->ptr(),1);
      }
      else if constexpr (std::is_same<Type,MKL_Complex16>::value)
      {
        cblas_zaxpy(this->size(),&ones,r.ptr(),1,this->ptr(),1);
      }
      return *this;
    }
    Dense<Type,Rank> operator-(Dense<Type,Rank> const & r) const
    {
      Dense<Type,Rank> res(*this);
      res -= r;
      return res;
    }
    Dense<Type,Rank> & operator*=(double val)
    {
      if constexpr (std::is_same<Type,double>::value)
      {
        cblas_dscal(this->size(),val,this->ptr(),1);
      }
      else if constexpr (std::is_same<Type,MKL_Complex16>::value)
      {
        cblas_zdscal(this->size(),val,this->ptr(),1);
      }
      return *this;
    }
    Dense<Type,Rank> operator*(Type const & val) const
    {
      Dense<Type,Rank> res(*this);
      if constexpr (std::is_same<Type,double>::value)
      {
        cblas_dscal(this->size(),val,res.ptr(),1);
      }
      else if constexpr (std::is_same<Type,MKL_Complex16>::value)
      {
        cblas_zscal(this->size(),&val,res.ptr(),1);
      }
      return res;
    }
    Dense<Type,Rank> & operator/=(double val)
    {
      double valc(1.0/val);
      if constexpr (std::is_same<Type,double>::value)
      {
        cblas_dscal(this->size(),valc,this->ptr(),1);
      }
      else if constexpr (std::is_same<Type,MKL_Complex16>::value)
      {
        cblas_zdscal(this->size(),valc,this->ptr(),1);
      }
      return *this;
    }
    Dense<Type,Rank> operator/(Type const & val) const
    {
      Dense<Type,Rank> res(*this);
      Type valc(1.0/val);
      if constexpr (std::is_same<Type,double>::value)
      {
        cblas_dscal(this->size(),valc,res.ptr(),1);
      }
      else if constexpr (std::is_same<Type,MKL_Complex16>::value)
      {
        cblas_zscal(this->size(),&valc,res.ptr(),1);
      }
      return res;
    }
    double norm() const
    {
      double res = 0.0;
      if constexpr (std::is_same<Type,double>::value)
      {
        res = cblas_dnrm2(this->size(),this->ptr(),1);
      }
      else if constexpr (std::is_same<Type,MKL_Complex16>::value)
      {
        res = cblas_dznrm2(this->size(),this->ptr(),1);
      }
      return res;
    }
    Type overlap(Dense<Type,Rank> const & r) const
    {
      Type res(0.0);
      if constexpr (std::is_same<Type,double>::value)
      {
        res = cblas_ddot(this->size(),this->ptr(),1,r.ptr(),1);
      }
      else if constexpr (std::is_same<Type,MKL_Complex16>::value)
      {
        cblas_zdotc_sub(this->size(),this->ptr(),1,r.ptr(),1,&res);
      }
      return res;
    }
    private:
    std::array<size_t,Rank> shape_;
    std::array<size_t,Rank> dist_;
  };
}
