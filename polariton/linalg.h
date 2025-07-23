/* dense tensor class */

#pragma once

#include <cmath>
#include "hash.h"

namespace KylinVib
{
  template<typename Type, int Rank>
  Dense<Type,Rank> conj(Dense<Type,Rank> const & r)
  {
    Dense<Type,Rank> res(r);
    if constexpr (std::is_same<Type,MKL_Complex16>::value)
    {
      #pragma omp parallel for
      for(int i=0;i<r.size();++i)
      {
        res.ptr()[i] = std::conj(r.ptr()[i]);
      }
    }
    return res;
  }
  template<typename Type, int Ri, int Rr>
  Dense<Type,Rr> reshape(Dense<Type,Ri> const & r, std::array<int,Rr> const & sp)
  {
    Dense<Type,Rr> res(sp);
    if(res.size()!=r.size()) { std::cout << "Reshape sizes mismatch!" << std::endl; std::exit(1); }
    std::memcpy(res.ptr(), r.ptr(), r.size()*sizeof(Type));
    return res;
  }
  // transpose
  template<typename Type, int Rank>
  Dense<Type,Rank> transpose(Dense<Type,Rank> const & r, std::array<int,Rank> const & ax)
  {
    char UnTransposed = 'y';
    for(int i=0;i<Rank-1;++i)
    {
      if(ax[i+1] != ax[i]+1)
      {
        UnTransposed = 'n';
        break;
      }
    }
    if(UnTransposed=='y')
    {
      Dense<Type,Rank> res(r);
      return res;
    }
    std::array<int,Rank> rsp;
    for(int i=0;i<Rank;++i)
    {
      rsp[i] = r.shape()[ax[i]];
    } 
    Dense<Type,Rank> res(rsp);
    #pragma omp parallel for
    for(int i=0;i<r.size();++i)
    {
      std::array<int,Rank> idx;
      for(int j=0;j<Rank;++j)
      {
        idx[j] = i / r.dist()[ax[j]] % r.shape()[ax[j]];
      }
      res(idx) = r.ptr()[i];
    }
    return res;
  }
  template<typename Type, int Rank>
  Dense<Type,Rank> transpose(Dense<Type,Rank> const & r, std::initializer_list<int> ax)
  {
    std::array<int,Rank> axr;
    std::copy(ax.begin(),ax.end(),axr.begin());
    return transpose<Type,Rank>(r,axr);
  }
  // stack
  template<typename Type, int Rank>
  Dense<Type,Rank> stack(Dense<Type,Rank> const & r1, Dense<Type,Rank> const & r2, std::initializer_list<int> ax)
  {
    std::array<int,Rank> rsp(r1.shape());
    for(auto it = ax.begin(); it!= ax.end(); ++it)
    {
      rsp[*it] += r2.shape()[*it];
    }
    Dense<Type,Rank> res(rsp);
    for(int i=0;i<r1.size();++i)
    {
      std::array<int,Rank> idx;
      for(int j=0;j<Rank;++j)
      {
        idx[j] = i / r1.dist()[j] % r1.shape()[j];
      }
      res(idx) = r1.ptr()[i];
    }
    for(int i=0;i<r2.size();++i)
    {
      std::array<int,Rank> idx;
      for(int j=0;j<Rank;++j)
      {
        idx[j] = i / r2.dist()[j] % r2.shape()[j];
      }
      for(auto it = ax.begin(); it!= ax.end(); ++it)
      {
        idx[*it] += r1.shape()[*it];
      }
      res(idx) = r2.ptr()[i];
    }
    return res;
  }
  // eigensolver
  template<typename Type>
  Dense<Type,2> eig(Dense<Type,2> & Hm, char rep = 'r')
  {
      int Ns = Hm.shape()[0];
      Dense<double,1> ega({Ns});
      Dense<Type,2> egr({Ns,Ns});
      if constexpr (std::is_same<Type,double>::value)
      {
          int ifeg = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', Ns, Hm.ptr(), Ns, ega.ptr());
          if(ifeg!=0) { std::cout << "Problems in eig! " << Ns << std::endl; std::exit(1);}
      }
      else if constexpr (std::is_same<Type,MKL_Complex16>::value)
      {
          int ifeg = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', Ns, Hm.ptr(), Ns, ega.ptr());
          if(ifeg!=0) { std::cout << "Problems in eig! " << Ns << std::endl; std::exit(1);}
      }
      for(int i=0;i<Ns;++i)
      {
          egr({i,i}) = ega({i});
      }
      return egr;                                                                                                                                                         
  }
  // generate transpose shape
  template<typename Type, int Rank, int rc>
  Dense<Type,Rank> prod_trans( Dense<Type,Rank> const & r, 
  std::initializer_list<int> ax, char l2r)
  {
    std::array<int,Rank> AxTrans;
    if(l2r=='l')
    {
      int anchor = 0;
      for(int i=0;i<Rank;++i)
      {
        if(std::find(ax.begin(),ax.end(),i)==ax.end())
        {
          AxTrans[anchor] = i;
          anchor++;
        }
      }
      std::copy(ax.begin(),ax.end(),AxTrans.begin()+Rank-rc);
    }
    else
    {
      int anchor = 0;
      for(int i=0;i<Rank;++i)
      {
        if(std::find(ax.begin(),ax.end(),i)==ax.end())
        {
          AxTrans[anchor+rc] = i;
          anchor++;
        }
      }
      std::copy(ax.begin(),ax.end(),AxTrans.begin());
    }
    return transpose<Type,Rank>(r,AxTrans);
  }
  // combine resulting shapes
  template<typename Type, int R1, int R2, int rc>
  std::array<Type,R1+R2-2*rc> prod_shape(std::array<Type,R1> const & sp1,
  std::array<Type,R2> const & sp2, std::initializer_list<int> ax1, 
  std::initializer_list<int> ax2)
  {
    std::array<Type,R1+R2-2*rc> res;
    int anchor = 0;
    for(int i=0;i<R1;++i)
    {
      if(std::find(ax1.begin(),ax1.end(),i)==ax1.end())
      {
        res[anchor] = sp1[i];
        anchor++;
      }
    }
    for(int i=0;i<R2;++i)
    {
      if(std::find(ax2.begin(),ax2.end(),i)==ax2.end())
      {
        res[anchor] = sp2[i];
        anchor++;
      }
    }
    return res;
  }
  // calculate the needed parameters for gemm_
  template<typename Type, int R1, int R2, int rc>
  std::array<int,3> prod_params(Dense<Type,R1> const & r1t, Dense<Type,R2> const & r2t)
  {
    std::array<int,3> rck;
    rck[0] = std::accumulate(r1t.shape().begin(),r1t.shape().begin()+R1-rc,1,std::multiplies<int>());
    rck[1] = std::accumulate(r2t.shape().begin()+rc,r2t.shape().end(),1,std::multiplies<int>());
    rck[2] = r1t.size() / rck[0];
    if(rck[2] != r2t.size()/rck[1] )
    {
      std::cout << "r1@r2 shapes mismatch!" << std::endl;
      //std::cout << R1 << ":" << R2 << ":" << rc << std::endl;
      r1t.print(1e9); r2t.print(1e9);
      std::exit(1);
    }
    return rck;
  }
  // product main implementation
  template<typename Type, int R1, int R2, int rc>
  Dense<Type,R1+R2-2*rc> prod(Dense<Type,R1> const & r1, Dense<Type,R2> const & r2,
  std::initializer_list<int> ax1, std::initializer_list<int> ax2)
  {
    Dense<Type,R1> r1t = prod_trans<Type,R1,rc>(r1,ax1,'l');
    Dense<Type,R2> r2t = prod_trans<Type,R2,rc>(r2,ax2,'r');
    std::array<int,R1+R2-2*rc> rsp = prod_shape<int,R1,R2,rc>(r1.shape(),r2.shape(),ax1,ax2);
    Dense<Type,R1+R2-2*rc> res(rsp);
    std::array<int,3> rcks = prod_params<Type,R1,R2,rc>(r1t,r2t);
    Type ones(1.0),zeros(0.0);
    if constexpr (std::is_same<Type,double>::value)
    {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rcks[0], rcks[1], rcks[2], ones, 
      r1t.ptr(), rcks[2], r2t.ptr(), rcks[1], zeros, res.ptr(), rcks[1]);
    }
    else if constexpr (std::is_same<Type,MKL_Complex16>::value)
    {
      cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rcks[0], rcks[1], rcks[2], &ones, 
      r1t.ptr(), rcks[2], r2t.ptr(), rcks[1], &zeros, res.ptr(), rcks[1]);
    }
    return res;
  }
  // matrix product for labelled array
  template<typename Type>
  LabArr<Type,2> mat_prod(LabArr<Type,2> const & c1, LabArr<Type,2> const & c2)
  {
    LabArr<Type,2> res({c1.shape()[0],c2.shape()[1]});
    res.labs = c1.labs;
    res.labs.insert(res.labs.end(), c2.labs.begin(), c2.labs.end());
    Type ones(1.0),zeros(0.0);
    std::array<int,3> rcks = {c1.shape()[0],c2.shape()[1],c1.shape()[1]};
    if constexpr (std::is_same<Type,double>::value)
    {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rcks[0], rcks[1], rcks[2], ones, 
      c1.ptr(), rcks[2], c2.ptr(), rcks[1], zeros, res.ptr(), rcks[1]);
    }
    else if constexpr (std::is_same<Type,MKL_Complex16>::value)
    {
      cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rcks[0], rcks[1], rcks[2], &ones, 
      c1.ptr(), rcks[2], c2.ptr(), rcks[1], &zeros, res.ptr(), rcks[1]);
    }
    return res;
  }
  // svd return (u,s,vt)
  template<typename Type, int R1, int R2>
  std::tuple<Dense<Type,R1+1>,Dense<Type,R2+1>> svd(Dense<Type,R1+R2> & m, char l2r = 'r', double tol = 1e-14, int maxdim = 1000,
  char ReS = 'n', char PrintErr = 'n')
  {
    int nrow = std::accumulate(m.shape().begin(),m.shape().begin()+R1,1,std::multiplies<int>());
    int ncol = m.size() / nrow;
    int ldu = std::min(nrow,ncol);
    Dense<Type,R1+R2> mc(m);
    Dense<Type,2> u({nrow,ldu}),vt({ldu,ncol});
    Dense<double,1> s({ldu}),sp({ldu-1});
    std::array<int,R1+1> lsp; std::copy(m.shape().begin(),m.shape().begin()+R1,lsp.begin());
    std::array<int,R2+1> rsp; std::copy(m.shape().begin()+R1,m.shape().end(),rsp.begin()+1);
    if constexpr (std::is_same<Type,double>::value)
    {
      int ifsv = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.ptr(), ncol, s.ptr(), u.ptr(), ldu, vt.ptr(), ncol, sp.ptr());
      if(ifsv!=0) { std::cout << "Problems in SVD! " << R1 << " : " << R2 << std::endl; std::exit(1);}
    }
    else if constexpr (std::is_same<Type,MKL_Complex16>::value)
    {
      int ifsv = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.ptr(), ncol, s.ptr(), u.ptr(), ldu, vt.ptr(), ncol, sp.ptr());
      if(ifsv!=0) { std::cout << "Problems in SVD! " << R1 << " : " << R2 << std::endl; std::exit(1);}
    }
    int nstate = 1; double nrms = 0;
    for(int i=0;i<ldu;++i)
    {
        nrms += s.ptr()[i];
        if(s.ptr()[i]<=tol)
        {
            break;
        }
        nstate = i + 1;
    }
    nstate = std::min(maxdim,nstate);
    lsp[R1] = nstate; rsp[0] = nstate;
    Dense<Type,R1+1> lef(lsp); Dense<Type,R2+1> rig(rsp);
    if constexpr (std::is_same<Type,double>::value)
    {
      if(l2r=='r')
      {
          for(int i=0;i<nstate;++i)
          {
              cblas_dcopy(nrow, u.ptr()+i, ldu, lef.ptr()+i, nstate);
              cblas_daxpy(ncol, s.ptr()[i], vt.ptr()+i*ncol, 1, rig.ptr()+i*ncol, 1);
          }
          if(ReS=='r') { lef *= std::sqrt(nrms); rig *= (1/std::sqrt(nrms));}
      }
      else
      {
          for(int i=0;i<nstate;++i)
          {
              cblas_daxpy(nrow, s.ptr()[i],  u.ptr()+i, ldu, lef.ptr()+i, nstate);
              cblas_dcopy(ncol, vt.ptr()+i*ncol, 1, rig.ptr()+i*ncol, 1);
          }
          if(ReS=='r') { lef *= (1/std::sqrt(nrms)); rig *= std::sqrt(nrms); }
      }
    }
    else if constexpr (std::is_same<Type,MKL_Complex16>::value)
    {
      if(l2r=='r')
      {
          for(int i=0;i<nstate;++i)
          {
              Type sc(s.ptr()[i]);
              cblas_zcopy(nrow, u.ptr()+i, ldu, lef.ptr()+i, nstate);
              cblas_zaxpy(ncol, &sc, vt.ptr()+i*ncol, 1, rig.ptr()+i*ncol, 1);
          }
          if(ReS=='r') { lef *= std::sqrt(nrms); rig *= (1/std::sqrt(nrms));}
      }
      else
      {
          for(int i=0;i<nstate;++i)
          {
              Type sc(s.ptr()[i]);
              cblas_zaxpy(nrow, &sc, u.ptr()+i, ldu, lef.ptr()+i, nstate);
              cblas_zcopy(ncol, vt.ptr()+i*ncol, 1, rig.ptr()+i*ncol, 1);
          }
          if(ReS=='r') { lef *= (1/std::sqrt(nrms)); rig *= std::sqrt(nrms); }
      }
    }
    if(PrintErr=='y')
    {
        Dense<Type,R1+R2> Err = mc-prod<Type,R1+1,R2+1,1>(lef,rig,{R1},{0});
        std::cout << "Err. of SVD = " << Err.norm() << std::endl;
        ////Err.print();
    }
    return std::make_tuple(lef,rig);
  }
  template<typename Type>
  Dense<Type,2> svd_pinv(Dense<Type,2> & m)
  {
    int nrow = m.shape()[0];
    int ncol = m.shape()[1];
    int ldu = std::min(nrow,ncol);
    int ifsv;
    ///m.print();
    Dense<Type,2> u({nrow,ldu}),vt({ldu,ncol}),diags({ldu,ldu});
    Dense<double,1> s({ldu}),sp({ldu-1});
    if constexpr (std::is_same<Type,double>::value)
    {
    ifsv = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.ptr(), ncol,
            s.ptr(), u.ptr(), ldu, vt.ptr(), ncol, sp.ptr());
    }
    else if constexpr (std::is_same<Type,MKL_Complex16>::value)
    {
    ifsv = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.ptr(), ncol,
            s.ptr(), u.ptr(), ldu, vt.ptr(), ncol, sp.ptr());
    }
    ///s.print();
    if(ifsv!=0) { std::cout << "Problems in SVDINV! " << std::endl; std::exit(1);}
    for(int i=0;i<ldu;++i)
    {
        if(s.ptr()[i]<=1e-15)
        {
          break;
        }
        diags({i,i}) = 1.0 / s.ptr()[i];
    }
    Dense<Type,2> res = prod<Type,2,2,1>(vt,diags,{0},{0});
    res = prod<Type,2,2,1>(res,u,{1},{1});
    return res;
  }
}
