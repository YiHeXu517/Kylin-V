/* DMRG implementation */

#pragma once

#include <deque>
#include "dense_mpo.h"
//#include "timer.h"

namespace KylinVib
{
    class DMRG
    {
        public:
        DMRG(MPO<double> const & op, MPS<double> const & s, double tol = 1e-8, size_t maxdim = 20)
        : ham_(op), state_(s),  tol_(tol), MaxBond_(maxdim)
        {
            Dense<double,3> edges({1,1,1});
            edges({0,0,0})= 1.0;
            envl_.push_back(edges);
            envr_.push_back(edges);
            size_t ns = op.size();
            for(size_t i=ns-1;i>1;--i)
            {
                envr_.push_back(MPO<double>::sweep(envr_.back(),ham_[i],state_[i],state_[i],'l'));
            }
        }
        ~DMRG() = default;

        // Hv
        static Dense<double,3> one_site_apply(Dense<double,5> const & vlo, Dense<double,3> const & s, Dense<double,3> const & envr)
        {
            Dense<double,4> los = prod<double,5,3,2>(vlo,s,{3,4},{0,1});
            Dense<double,3> res = prod<double,4,3,2>(los,envr,{2,3},{1,2});
            return res;
        }
        static Dense<double,4> two_site_apply(Dense<double,5> const & vlo, Dense<double,4> const & s2, Dense<double,5> const & vro)
        {
            Dense<double,5> los = prod<double,5,4,2>(vlo,s2,{3,4},{0,1});
            Dense<double,4> res = prod<double,5,5,3>(los,vro,{2,3,4},{0,1,2});
            return res;
        }
        Dense<double,3> one_site_eigen(Dense<double,3> const & s, size_t site, size_t MacroIter, size_t MicroIter, double convs)
        {
            std::vector<Dense<double,3>> Vk(MicroIter,s.shape()), Wk(MicroIter,s.shape());
            Vk[0] = s * (1.0 / s.norm());
            Dense<double,3> fk,fm,res;
            Dense<double,5> vlo = prod<double,3,4,1>(envl_.back(),ham_[site],{1},{0});
            vlo = transpose<double,5>(vlo,{0,2,4,1,3});
            // main program
            for(size_t mac=0;mac<MacroIter;++mac)
            {
                for(size_t i=0;i<MicroIter;++i)
                {
                    for(size_t j=0;j<i;++j)
                    {
                        double ov = -1.0 * Vk[j].overlap(Vk[i]);
                        Vk[i] += Vk[j] * ov;
                    }
                    Vk[i] *= (1.0/Vk[i].norm());
                    Dense<double,3> hv = one_site_apply(vlo,Vk[i],envr_.back());
                    Wk[i] = hv;
                    for(size_t j=0;j<i+1;++j)
                    {
                        double ov = -1.0 * Vk[j].overlap(hv);
                        hv += Vk[j] * ov;
                    }
                    if(i!=MicroIter-1)
                    {
                        hv *= (1.0/hv.norm());
                        Vk[i+1] = std::move(hv);
                    }
                    else
                    {
                        fk = std::move(hv);
                    }
                }
                // eigensolver
                Dense<double,2> Hk({MicroIter,MicroIter});
                for(size_t i=0;i<MicroIter;++i)
                {
                    for(size_t j=i;j<MicroIter;++j)
                    {
                        Hk({i,j}) = Vk[i].overlap(Wk[j]);
                        Hk({j,i}) = Hk({i,j});
                    }
                }
                Dense<double,2> Hkc(Hk);
                Dense<double,2> EigHk = eig<double>(Hkc);
                // jacobi davidson
                Dense<double,3> Xk = Vk[0] * Hkc({0,0}),
                WXk = Wk[0] * Hkc({0,0});
                for(size_t i=1;i<MicroIter;++i)
                {
                    Xk += Vk[i] * Hkc({i,0});
                    WXk += Wk[i] * Hkc({i,0});
                }
                Dense<double,3> fk = (WXk * -1.0) + Xk * EigHk({0,0});
                double nfk = fk.norm();
                std::cout << "Residual of iter " << mac+1 << " = " << nfk << std::endl; 
                std::cout << "Energy of iter " << mac+1 << " = " << EigHk({0,0}) << std::endl;
                if( std::abs(nfk / EigHk({0,0})) < convs || mac == MacroIter-1 )
                {
                    res = Xk;
                    break;
                }
                Vk[0] = Xk;
            }
            return res;
        }
        Dense<double,4> two_site_eigen(Dense<double,4> const & s2, size_t site, size_t MacroIter, size_t MicroIter, double convs)
        {
            std::vector<Dense<double,4>> Vk(MicroIter,s2.shape()), Wk(MicroIter,s2.shape());
            Vk[0] = s2 * (1.0 / s2.norm());
            Dense<double,4> fk,fm,res;
            Dense<double,5> vlo = prod<double,3,4,1>(envl_.back(),ham_[site],{1},{0});
            vlo = transpose<double,5>(vlo,{0,2,4,1,3});
            Dense<double,5> vro = prod<double,4,3,1>(ham_[site+1],envr_.back(),{3},{1});
            vro = transpose<double,5>(vro,{0,2,4,1,3});
            // main program
            for(size_t mac=0;mac<MacroIter;++mac)
            {
                for(size_t i=0;i<MicroIter;++i)
                {
                    for(size_t j=0;j<i;++j)
                    {
                        double ov = -1.0 * Vk[j].overlap(Vk[i]);
                        Vk[i] += Vk[j] * ov;
                    }
                    Vk[i] *= (1.0/Vk[i].norm());
                    Dense<double,4> hv = two_site_apply(vlo,Vk[i],vro);
                    Wk[i] = hv;
                    for(size_t j=0;j<i+1;++j)
                    {
                        double ov = -1.0 * Vk[j].overlap(hv);
                        hv += Vk[j] * ov;
                    }
                    if(i!=MicroIter-1)
                    {
                        hv *= (1.0/hv.norm());
                        Vk[i+1] = std::move(hv);
                    }
                    else
                    {
                        fk = std::move(hv);
                    }
                }
                // eigensolver
                Dense<double,2> Hk({MicroIter,MicroIter});
                for(size_t i=0;i<MicroIter;++i)
                {
                    for(size_t j=i;j<MicroIter;++j)
                    {
                        Hk({i,j}) = Vk[i].overlap(Wk[j]);
                        Hk({j,i}) = Hk({i,j});
                    }
                }
                Dense<double,2> Hkc(Hk);
                Dense<double,2> EigHk = eig<double>(Hkc);
                // jacobi davidson
                Dense<double,4> Xk = Vk[0] * Hkc({0,0}),
                WXk = Wk[0] * Hkc({0,0});
                for(size_t i=1;i<MicroIter;++i)
                {
                    Xk += Vk[i] * Hkc({i,0});
                    WXk += Wk[i] * Hkc({i,0});
                }
                Dense<double,4> fk = (WXk * -1.0) + Xk * EigHk({0,0});
                double nfk = fk.norm();
                std::cout << "Residual of iter " << mac+1 << " = " << nfk << std::endl; 
                std::cout << "Energy of iter " << mac+1 << " = " << EigHk({0,0}) << std::endl;
                if( std::abs(nfk / EigHk({0,0})) < convs || mac == MacroIter-1 )
                {
                    res = Xk;
                    break;
                }
                Vk[0] = Xk;
            }
            return res;
        }
        void one_site(size_t NumSwps = 100, size_t MacroIter = 50, size_t MicroIter = 50, double convs = 1e-4)
        {
            size_t ns = ham_.size();
            double Ene = 0.0, Ovp = 0.0;
            for(size_t swp=0;swp<NumSwps;++swp)
            {
                for(size_t i=0;i<ns;++i)
                {
                    std::cout << "Site " << i+1 << std::endl;
                    Dense<double,3> s1 = one_site_eigen(state_[i], i, MacroIter, MicroIter, convs);
                    if(i!=ns-1)
                    {
                        auto[lef,rig] = svd<double,2,1>(s1,'r',tol_, MaxBond_);
                        state_[i] = std::move(lef);
                        state_[i+1] = prod<double,2,3,1>(rig,state_[i+1],{1},{0});
                        envl_.push_back(MPO<double>::sweep(envl_.back(),ham_[i],state_[i],state_[i],'r'));
                        envr_.pop_back();
                    }
                    else
                    {
                        state_[i] = std::move(s1);
                    }
                }
                for(size_t i=ns;i>0;--i)
                {
                    std::cout << "Site " << i << std::endl;
                    Dense<double,3> s1 = one_site_eigen(state_[i-1], i-1, MacroIter, MicroIter, convs);
                    if(i!=1)
                    {
                        auto[lef,rig] = svd<double,1,2>(s1,'l',tol_, MaxBond_);
                        state_[i-1] = std::move(rig);
                        state_[i-2] = prod<double,3,2,1>(state_[i-2],lef,{2},{0});
                        envr_.push_back(MPO<double>::sweep(envr_.back(),ham_[i-1],state_[i-1],state_[i-1],'l'));
                        envl_.pop_back();
                    }
                    else
                    {
                        state_[i-1] = std::move(s1);
                    }
                }
                double CurEne = ham_.join(state_,state_);
                std::cout << "Current energy  = " << CurEne << std::endl;
                if(std::abs(CurEne-Ene)<0.1)
                {
                    break;
                }
                else
                {
                    Ene = CurEne;
                }
            }
        }
        void two_site(size_t NumSwps = 100, size_t MacroIter = 50, size_t MicroIter = 50, double convs = 1e-4)
        {
            size_t ns = ham_.size();
            double Ene = 0.0, Ovp = 0.0;
            for(size_t swp=0;swp<NumSwps;++swp)
            {
                for(size_t i=0;i<ns-1;++i)
                {
                    std::cout << "Site " << i+1 << std::endl;
                    Dense<double,4> s2 = prod<double,3,3,1>(state_[i],state_[i+1],{2},{0});
                    s2 = two_site_eigen(s2,i,MacroIter, MicroIter, convs);
                    auto[lef,rig] = svd<double,2,2>(s2,'r',tol_, MaxBond_);
                    state_[i] = std::move(lef);
                    state_[i+1] = std::move(rig);
                    if(i!=ns-2)
                    {
                        envl_.push_back(MPO<double>::sweep(envl_.back(),ham_[i],state_[i],state_[i],'r'));
                        envr_.pop_back();
                    }
                }
                for(size_t i=ns-1;i>0;--i)
                {
                    std::cout << "Site " << i+1 << std::endl;
                    Dense<double,4> s2 = prod<double,3,3,1>(state_[i-1],state_[i],{2},{0});
                    s2 = two_site_eigen(s2, i-1, MacroIter, MicroIter, convs);
                    auto[lef,rig] = svd<double,2,2>(s2,'l',tol_, MaxBond_);
                    state_[i] = std::move(rig);
                    state_[i-1] = std::move(lef);
                    if(i!=1)
                    {
                        envr_.push_back(MPO<double>::sweep(envr_.back(),ham_[i],state_[i],state_[i],'l'));
                        envl_.pop_back();
                    }
                }
                double CurEne = ham_.join(state_,state_);
                std::cout << "Current energy  = " << CurEne << std::endl;
                if(std::abs(CurEne-Ene)<0.1)
                {
                    break;
                }
                else
                {
                    Ene = CurEne;
                }
            }
            envr_.push_back(MPO<double>::sweep(envr_.back(),ham_[1],state_[1],state_[1],'l'));
        }
        MPS<double> get_mps() const { return state_; }
        MPO<double> get_mpo() const { return ham_; }

        protected:
        MPO<double> ham_;
        MPS<double> state_;
        double tol_;
        size_t MaxBond_;
        std::vector<Dense<double,3>> envl_;
        std::vector<Dense<double,3>> envr_;
    };
}