/* MPS class with dense tensor */

#pragma once

#include <map>
#include "linalg.h"

namespace KylinVib
{
    template<typename Type>
    class MPS : public std::vector<Dense<Type,3>>
    {
        public:
        MPS() = default;
        MPS(MPS<Type> const & r)
        : std::vector<Dense<Type,3>>(r)
        {

        }
        MPS(MPS<Type> && r)
        : std::vector<Dense<Type,3>>(std::move(r))
        {

        }
        MPS(size_t ns)
        : std::vector<Dense<Type,3>>(ns)
        {

        }
        MPS(size_t ns, size_t nphys, std::map<size_t,size_t> const & cfg)
        : std::vector<Dense<Type,3>>(ns)
        {
            for(size_t i=0;i<ns;++i)
            {
                Dense<Type,3> si({1,nphys,1});
                (*this)[i] = si;
            }
            for(const auto & [key,value] : cfg)
            {
                (*this)[key]({0,value,0}) = 1.0;
            }
        }
        MPS(size_t ns, size_t nphys, std::vector<int> const & cfg)
        : std::vector<Dense<Type,3>>(ns)
        {
            for(size_t i=0;i<ns;++i)
            {
                Dense<Type,3> si({1,nphys,1});
                si({0,(size_t)cfg[i],0}) = 1.0;
                (*this)[i] = si;
            }
        }
        MPS(size_t ns, size_t nphys, size_t bd)
        : std::vector<Dense<Type,3>>(ns)
        {
            Dense<Type,3> s1({1,nphys,bd});
            s1.rand_fill();
            (*this)[0] = s1;
            Dense<Type,3> s2({bd,nphys,1});
            s2.rand_fill();
            (*this)[ns-1] = s2;
            for(size_t i=1;i<ns-1;++i)
            {
                Dense<Type,3> si({bd,nphys,bd});
                si.rand_fill();
                (*this)[i] = si;
            }
        }
        ~MPS() = default;
        MPS<Type> & operator=(MPS<Type> const & r)
        {
            std::vector<Dense<Type,3>>::operator=(r);
            return *this;
        }
        MPS<Type> & operator=(MPS<Type> && r)
        {
            std::vector<Dense<Type,3>>::operator=(r);
            return *this;
        }
        MPS<Type> & operator+=(MPS<Type> const & r)
        {
            size_t ns = this->size();
            (*this)[0] = stack<Type,3>((*this)[0],r[0],{2});
            (*this)[ns-1] = stack<Type,3>((*this)[ns-1],r[ns-1],{0});
            for(size_t i=1;i<ns-1;++i)
            {
                (*this)[i] = stack<Type,3>((*this)[i],r[i],{0,2});
            }
            return *this;
        }
        MPS<Type> operator+(MPS<Type> const & r) const
        {
            MPS<Type> res(*this);
            res += r;
            return res;
        }
        MPS<Type> & operator*=(double val)
        {
            (*this)[0] *= val;
            return *this;
        } 
        MPS<Type> operator*(Type const & val) const
        {
            MPS<Type> res(*this);
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
        void canon(double tol = 1e-14, size_t maxdim = 1000)
        {
            size_t ns = this->size();
            for(size_t i=0;i<ns-1;++i)
            {
                auto[lef,rig] = svd<Type,2,1>((*this)[i],'r',tol,maxdim);
                (*this)[i] = std::move(lef);
                (*this)[i+1] = prod<Type,2,3,1>(rig,(*this)[i+1],{1},{0});
            }
            for(size_t i=ns-1;i>0;--i)
            {
                auto[lef,rig] = svd<Type,1,2>((*this)[i],'l',tol,maxdim);
                (*this)[i] = std::move(rig);
                (*this)[i-1] = prod<Type,3,2,1>((*this)[i-1],lef,{2},{0});
            }
        }
        Type overlap(MPS<Type> const & s) const
        {
            size_t ns = this->size();
            Dense<Type,2> env({1,1});
            env.ptr()[0] = 1.0;
            for(size_t i=0;i<ns;++i)
            {
                env = sweep(env,(*this)[i],s[i]);
            }
            return env.ptr()[0];
        }
        // extract dominant cfgs
        std::vector<LabArr<Type,2>> dominant(size_t nst) const
        {
            std::vector<LabArr<Type,2>> levs = all_levels((*this)[0]), kepts;
            kepts = truncate_levels(levs,nst);
            for(size_t site=1;site<this->size();++site)
            {
                levs = all_levels((*this)[site]);
                std::vector<LabArr<Type,2>> envs(levs.size()*kepts.size());
                #pragma omp parallel for
                for(size_t i=0;i<envs.size();++i)
                {
                    size_t i1 = i / levs.size(), i2 = i % levs.size();
                    envs[i] = mat_prod<Type>(kepts[i1],levs[i2]);
                }
                kepts = truncate_levels(envs,nst);
            }
            return kepts;
        }
        // product tools
        static Dense<Type,2> sweep(Dense<Type,2> const & env, Dense<Type,3> const & mps1, 
        Dense<Type,3> const & mps2, char l2r = 'r')
        {
            if(l2r=='r')
            {
                Dense<Type,3> ls2 = prod<Type,2,3,1>(env,mps2,{1},{0});
                Dense<Type,2> res = prod<Type,3,3,2>(conj<Type,3>(mps1),ls2,{0,1},{0,1});
                return res;
            }
            else
            {
                Dense<Type,3> ls2 = prod<Type,3,2,1>(mps2,env,{2},{1});
                Dense<Type,2> res = prod<Type,3,3,2>(conj<Type,3>(mps1),ls2,{1,2},{1,2});
                return res;
            }
        }
        // two-site overlap
        static Dense<Type,4> make_overlap(Dense<Type,2> const & envl, Dense<Type,3> const & mps1, 
        Dense<Type,3> const & mps2, Dense<Type,2> const & envr)
        {
            Dense<Type,3> ls1 = prod<Type,2,3,1>(envl,mps1,{1},{0});
            Dense<Type,3> rs2 = prod<Type,3,2,1>(mps2,envr,{2},{1});
            Dense<Type,4> res = prod<Type,3,3,1>(ls1,rs2,{2},{0});
            return res;
        } 
        static std::vector<LabArr<Type,2>> all_levels(Dense<Type,3> const & mps)
        {
            std::vector<LabArr<Type,2>> res(mps.shape()[1],{mps.shape()[0],mps.shape()[2]});
            #pragma omp parallel for
            for(size_t i=0;i<mps.size();++i)
            {
                std::array<size_t,3> sidx = mps.make_indices(i);
                res[sidx[1]]({sidx[0],sidx[2]}) = mps.ptr()[i];
            }
            int d = res.size();
            for(int i=0;i<d;++i)
            {
                res[i].labs.push_back(i);
            }
            return res;
        }
        static std::vector<LabArr<Type,2>> truncate_levels(std::vector<LabArr<Type,2>> const & r, size_t nst)
        {
            if(r.size()<=nst)
            {
                std::vector<LabArr<Type,2>> res(r);
                return res;
            }
            std::vector<std::tuple<size_t,double>> tups;
            for(size_t i=0;i<r.size();++i)
            {
                tups.push_back( std::make_tuple(i,r[i].norm()) );
            }
            std::sort(tups.begin(),tups.end(),
            [&tups](std::tuple<size_t,double> x1, std::tuple<size_t,double> x2){
            return std::get<1>(x1)>std::get<1>(x2);});
            std::vector<LabArr<Type,2>> res(nst);
            #pragma omp parallel for
            for(size_t i=0;i<nst;++i)
            {
                res[i] = r[std::get<0>(tups[i])];
            }
            return res;
        }
    };
}
