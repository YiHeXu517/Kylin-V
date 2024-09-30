/* MPO class with dense tensor */

#pragma once

#include "dense_mps.h"

namespace KylinVib
{
    template<typename Type>
    class MPO : public std::vector<Dense<Type,4>>
    {
        public:
        MPO() = default;
        MPO(MPO<Type> const & r)
        : std::vector<Dense<Type,4>>(r)
        {

        }
        MPO(MPO<Type> && r)
        : std::vector<Dense<Type,4>>(std::move(r))
        {

        }
        MPO(size_t ns)
        : std::vector<Dense<Type,4>>(ns)
        {

        }
        ~MPO() = default;
        MPO<Type> & operator=(MPO<Type> const & r)
        {
            std::vector<Dense<Type,4>>::operator=(r);
            return *this;
        }
        MPO<Type> & operator=(MPO<Type> && r)
        {
            std::vector<Dense<Type,4>>::operator=(r);
            return *this;
        }
        MPO<Type> & operator+=(MPO<Type> const & r)
        {
            size_t ns = this->size();
            (*this)[0] = stack<Type,4>((*this)[0],r[0],{3});
            (*this)[ns-1] = stack<Type,4>((*this)[ns-1],r[ns-1],{0});
            for(size_t i=1;i<ns-1;++i)
            {
                (*this)[i] = stack<Type,4>((*this)[i],r[i],{0,3});
            }
            return *this;
        }
        MPO<Type> operator+(MPO<Type> const & r) const
        {
            MPO<Type> res(*this);
            res += r;
            return res;
        }
        MPO<Type> & operator*=(double val)
        {
            (*this)[0] *= val;
            return *this;
        }
        MPO<Type> operator*(Type const & val) const
        {
            MPO<Type> res(*this);
            res[0] = res[0] * val;
            return res;
        }
        void print(double tol = 1e-14) const
        {
            for(size_t site=0;site<this->size();++site)
            {
                std::cout << "Site: " << site+1 << std::endl;
                (*this)[site].print(tol);
            }
        }
        void canon(double tol = 1e-8, size_t maxdim = 1000)
        {
            size_t ns = this->size();
            for(size_t i=0;i<ns-1;++i)
            {
                auto[lef,rig] = svd<Type,3,1>((*this)[i],'r',tol,maxdim,'r');
                (*this)[i] = std::move(lef);
                (*this)[i+1] = prod<Type,2,4,1>(rig,(*this)[i+1],{1},{0});
            }
            for(size_t i=ns-1;i>0;--i)
            {
                auto[lef,rig] = svd<Type,1,3>((*this)[i],'l',tol,maxdim,'r');
                (*this)[i] = std::move(rig);
                (*this)[i-1] = prod<Type,4,2,1>((*this)[i-1],lef,{3},{0});
            }
        }
        Type join(MPS<Type> const & s1, MPS<Type> const & s2) const
        {
            size_t ns = this->size();
            Dense<Type,3> env({1,1,1});
            env.ptr()[0] = 1.0;
            for(size_t i=0;i<ns;++i)
            {
                env = sweep(env,(*this)[i],s1[i],s2[i]);
            }
            return env.ptr()[0];
        }
        MPS<Type> apply_op(MPS<Type> const & s, double tol = 1e-14, size_t maxdim = 1000) const
        {
            size_t ns = this->size();
            Dense<double,3> env({1,1,1});
            env.ptr()[0] = 1.0;
            MPS<Type> res(ns);
            for(size_t i=0;i<ns;++i)
            {
                Dense<Type,4> lss = prod<Type,3,3,1>(env,s[i],{2},{0});
                Dense<Type,4> lso = prod<Type,4,4,2>(lss,(*this)[i],{1,2},{0,2});
                lso = transpose<Type,4>(lso,{0,2,3,1});
                if(i!=ns-1)
                {
                    auto[lef,rig] = svd<Type,2,2>(lso,'r',tol,maxdim);
                    res[i] = std::move(lef);
                    env = std::move(rig);
                }
                else
                {
                    Dense<Type,3> lsoc({lso.shape()[0],lso.shape()[1],lso.shape()[2]});
                    #pragma omp parallel for
                    for(size_t j=0;j<lsoc.size();++j)
                    {
                        lsoc.ptr()[j] = lso.ptr()[j];
                    }
                    res[i] = std::move(lsoc);
                }
            }
            for(size_t i=ns-1;i>0;--i)
            {
                auto[lef,rig] = svd<Type,1,2>(res[i],'l',tol,maxdim);
                res[i] = std::move(rig);
                res[i-1] = prod<Type,3,2,1>(res[i-1],lef,{2},{0});
            }
            return res;
        }
        // make diagonal 
        MPS<Type> diag_state() const
        {
            size_t ns = this->size();
            MPS<Type> res(ns);
            for(size_t i=0;i<ns;++i)
            {
                size_t bl = (*this)[i].shape()[0], br = (*this)[i].shape()[3],
                d = (*this)[i].shape()[1];
                Dense<Type,3> si({bl,d,br});
                for(size_t j=0;j<si.size();++j)
                {
                    std::array<size_t,3> sidx = si.make_indices(j);
                    si.ptr()[j] = (*this)[i]({sidx[0],sidx[1],sidx[1],sidx[2]});
                }
                res[i] = si;
            }
            res.canon();
            return res;
        }
        // product tools
        static Dense<Type,3> sweep(Dense<Type,3> const & env, Dense<Type,4> const & mpo,
        Dense<Type,3> const & mps1, Dense<Type,3> const & mps2, char l2r = 'r')
        {
            if(l2r=='r')
            {
                Dense<Type,4> lss = prod<Type,3,3,1>(env,mps2,{2},{0});
                Dense<Type,4> lso = prod<Type,4,4,2>(lss,mpo,{1,2},{0,2});
                Dense<Type,3> res = prod<Type,3,4,2>(conj<Type,3>(mps1),lso,{0,1},{0,2});
                return transpose<Type,3>(res,{0,2,1});
            }
            else
            {
                Dense<Type,4> lss = prod<Type,3,3,1>(mps2,env,{2},{2});
                Dense<Type,4> lso = prod<Type,4,4,2>(mpo,lss,{2,3},{1,3});
                Dense<Type,3> res = prod<Type,4,3,2>(lso,conj<Type,3>(mps1),{1,3},{1,2});
                return transpose<Type,3>(res,{2,0,1});
            }
        }
        // make_diagonal
        static Dense<Type,3> make_diag(Dense<Type,3> const & envl, Dense<Type,4> const & mpo
        , Dense<Type,3> const & envr)
        {
            Dense<Type,2> diagL({envl.shape()[0],envl.shape()[1]});
            for(size_t i=0;i<diagL.size();++i)
            {
                std::array<size_t,2> idx = diagL.make_indices(i);
                diagL.ptr()[i] = envl({idx[0],idx[1],idx[0]});
            }
            Dense<Type,3> diago({mpo.shape()[0],mpo.shape()[1],mpo.shape()[3]});
            for(size_t i=0;i<diago.size();++i)
            {
                std::array<size_t,3> idx = diago.make_indices(i);
                diago.ptr()[i] = mpo({idx[0],idx[1],idx[1],idx[2]});
            }
            Dense<Type,2> diagR({envr.shape()[1],envr.shape()[2]});
            for(size_t i=0;i<diagR.size();++i)
            {
                std::array<size_t,2> idx = diagR.make_indices(i);
                diagR.ptr()[i] = envr({idx[1],idx[0],idx[1]});
            }
            diago = prod<double,2,3,1>(diagL,diago,{1},{0});
            diago = prod<double,3,2,1>(diago,diagR,{2},{0});
            return diago;
        }
    };
}
