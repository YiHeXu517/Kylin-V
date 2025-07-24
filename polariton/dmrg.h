/* DMRG implementation */

#pragma once

#include <deque>
#include "dense_mpo.h"

namespace KylinVib
{
    template<typename Type>
    class DMRG
    {
        public:
        DMRG(MPO<Type> const & op, MPS<Type> const & s, double tol = 1e-8, int maxdim = 20)
        : ham_(op), state_(s), state0_(s), tol_(tol), MaxBond_(maxdim)
        {
            Dense<Type,3> edges({1,1,1});
            edges({0,0,0})= 1.0;
            envl_.push_back(edges);
            envr_.push_back(edges);
            int ns = op.size();
            for(int i=ns-1;i>1;--i)
            {
                envr_.push_back(MPO<Type>::sweep(envr_.back(),ham_[i],state_[i],state_[i],'l'));
            }
            Dense<Type,2> ovs({1,1});
            ovs({0,0})= 1.0;
            ovl_.push_back(ovs);
            ovr_.push_back(ovs);
            for(int i=ns-1;i>1;--i)
            {
                ovr_.push_back(MPS<Type>::sweep(ovr_.back(),state_[i],state0_[i],'l'));
            }
        }
        ~DMRG() = default;

        // Hv
        static Dense<Type,3> one_site_apply(Dense<Type,5> const & vlo, Dense<Type,3> const & s, Dense<Type,3> const & envr)
        {
            Dense<Type,4> los = prod<Type,5,3,2>(vlo,s,{3,4},{0,1});
            Dense<Type,3> res = prod<Type,4,3,2>(los,envr,{2,3},{1,2});
            return res;
        }
        static Dense<Type,4> two_site_apply(Dense<Type,5> const & vlo, Dense<Type,4> const & s2, Dense<Type,5> const & vro)
        {
            Dense<Type,5> los = prod<Type,5,4,2>(vlo,s2,{3,4},{0,1});
            Dense<Type,4> res = prod<Type,5,5,3>(los,vro,{2,3,4},{0,1,2});
            return res;
        }
        static Dense<Type,3> one_site_overlap_apply(Dense<Type,2> const & ovl, Dense<Type,3> const & s0, Dense<Type,2> const & ovr)
        {
            Dense<Type,3> ls0 = prod<Type,2,3,1>(ovl,s0,{1},{0});
            Dense<Type,3> res = prod<Type,3,2,1>(ls0,ovr,{2},{1});
            return res;
        }
        static Dense<Type,4> two_site_overlap_apply(Dense<Type,2> const & ovl, Dense<Type,3> const & s01, Dense<Type,3> const & s02, Dense<Type,2> const & ovr)
        {
            Dense<Type,3> ls1 = prod<Type,2,3,1>(ovl,s01,{1},{0});
            Dense<Type,3> ls2 = prod<Type,3,2,1>(s02,ovr,{2},{1});
            Dense<Type,4> res = prod<Type,3,3,1>(ls1,ls2,{2},{0});
            return res;
        }
        Dense<Type,4> two_site_gmres(Dense<Type,4> const & x0, int site, Dense<Type,4> const & f, Type const & eta, int MicroIter)
        {
            std::vector<Dense<Type,4>> Vk(MicroIter,x0.shape()), Wk(MicroIter,x0.shape());
            Dense<Type,5> vlo = prod<Type,3,4,1>(envl_.back(),ham_[site],{1},{0});
            vlo = transpose<Type,5>(vlo,{0,2,4,1,3});
            Dense<Type,5> vro = prod<Type,4,3,1>(ham_[site+1],envr_.back(),{3},{1});
            vro = transpose<Type,5>(vro,{0,2,4,1,3});
            Dense<Type,4> r0 = f - two_site_apply(vlo,x0,vro);
            r0 += x0 * eta;
            double beta = r0.norm();
            Vk[0] = r0 / beta;
            Dense<Type,4> res(x0);
            std::vector<std::tuple<int,int,MKL_Complex16>> HkTps;
            ////Dense<Type,2> Hk({MicroIter+1,MicroIter}),Eyek({MicroIter+1,1});
            // main program
            for(int i=0;i<MicroIter;++i)
            {
                for(int j=0;j<i;++j)
                {
                    Type ov = -1.0 * Vk[j].overlap(Vk[i]);
                    Vk[i] += Vk[j] * ov;
                }
                Vk[i] *= (1.0/Vk[i].norm());
                Dense<Type,4> hv = two_site_apply(vlo,Vk[i],vro);
                hv -= Vk[i] * eta;
                for(int j=0;j<i+1;++j)
                {
                    Type ov = -1.0 * Vk[j].overlap(hv);
                    HkTps.push_back(std::make_tuple(j,i,-1.0*ov));
                    ////Hk({j,i}) = -1.0*ov;
                    hv += Vk[j] * ov;
                }
                ///Hk({i+1,i}) = hv.norm();
                double nmhv = hv.norm();
                HkTps.push_back(std::make_tuple(i+1,i,(MKL_Complex16)nmhv));
                if(i!=MicroIter-1 && nmhv>1e-12)
                {
                    hv *= 1.0/nmhv;
                    Vk[i+1] = std::move(hv);
                }
                else
                {
                    break;
                }
            }
            int RealIter = Vk.size();
            Dense<Type,2> Hk({RealIter+1,RealIter}),Eyek({RealIter+1,1});
            Eyek({0,0}) = beta;
            for(auto const & tp : HkTps)
            {
                Hk({std::get<0>(tp),std::get<1>(tp)}) = std::get<2>(tp);
            }
            Dense<Type,2> Hk1 = svd_pinv<Type>(Hk);
            Eyek = prod<Type,2,2,1>(Hk1,Eyek,{1},{0});
            for(int i=0;i<RealIter;++i)
            {
                res += Vk[i] * Eyek({i,0});
            }
            res /= res.norm();
            return res;
        }
        Dense<Type,3> one_site_gmres(Dense<Type,3> const & x0, int site, Dense<Type,3> const & f, Type const & eta, int MicroIter)
        {
            Dense<Type,5> vlo = prod<Type,3,4,1>(envl_.back(),ham_[site],{1},{0});
            vlo = transpose<Type,5>(vlo,{0,2,4,1,3});
            Dense<Type,3> r0 = f - one_site_apply(vlo,x0,envr_.back());
            r0 += x0 * eta;
            double beta = r0.norm();
            std::vector<Dense<Type,3>> Vk(MicroIter,x0.shape());
            Vk[0] = r0 / beta;
            Dense<Type,3> res(x0);
            std::vector<std::tuple<int,int,MKL_Complex16>> HkTps;
            ////Dense<Type,2> Hk({MicroIter+1,MicroIter}),Eyek({MicroIter+1,1});
            // main program
            for(int i=0;i<MicroIter;++i)
            {
                for(int j=0;j<i;++j)
                {
                    Type ov = -1.0 * Vk[j].overlap(Vk[i]);
                    Vk[i] += Vk[j] * ov;
                }
                Vk[i] *= (1.0/Vk[i].norm());
                Dense<Type,3> hv = one_site_apply(vlo,Vk[i],envr_.back());
                hv -= Vk[i] * eta;
                for(int j=0;j<i+1;++j)
                {
                    Type ov = -1.0 * Vk[j].overlap(hv);
                    HkTps.push_back(std::make_tuple(j,i,-1.0*ov));
                    /////Hk({j,i}) = -1.0*ov;
                    hv += Vk[j] * ov;
                }
                ////Hk({i+1,i}) = hv.norm();
                double nmhv = hv.norm();
                HkTps.push_back(std::make_tuple(i+1,i,(MKL_Complex16)nmhv));
                if(i!=MicroIter-1 && nmhv>1e-12)
                {
                    hv *= 1.0/nmhv;
                    Vk[i+1] = std::move(hv);
                }
                else
                {
                    break;
                }
            }
            int RealIter = Vk.size();
            Dense<Type,2> Hk({RealIter+1,RealIter}),Eyek({RealIter+1,1});
            for(auto const & tp : HkTps)
            {
                Hk({std::get<0>(tp),std::get<1>(tp)}) = std::get<2>(tp);
            }
            Eyek({0,0}) = beta;
            Dense<Type,2> Hk1 = svd_pinv<Type>(Hk);
            Eyek = prod<Type,2,2,1>(Hk1,Eyek,{1},{0});
            for(int i=0;i<RealIter;++i)
            {
                res += Vk[i] * Eyek({i,0});
            }
            res /= res.norm();
            return res;
        }
        void two_site_SI(Type const & eta, int NumSwps, int MicroIter)
        {
            int ns = ham_.size();
            Type Ene = 0.0, Ovp = 0.0;
            double Var = 1e19;
            for(int swp=0;swp<NumSwps;++swp)
            {
                for(int i=0;i<ns-1;++i)
                {
                    Dense<Type,4> s2 = prod<Type,3,3,1>(state_[i],state_[i+1],{2},{0});
                    s2 = two_site_gmres(s2,i,s2,eta,MicroIter);
                    auto[lef,rig] = svd<Type,2,2>(s2,'r',tol_, MaxBond_);
                    state_[i] = std::move(lef);
                    state_[i+1] = std::move(rig);
                    if(i!=ns-2)
                    {
                        envl_.push_back(MPO<Type>::sweep(envl_.back(),ham_[i],state_[i],state_[i],'r'));
                        envr_.pop_back();
                    }
                }
                for(int i=ns-1;i>0;--i)
                {
                    Dense<Type,4> s2 = prod<Type,3,3,1>(state_[i-1],state_[i],{2},{0});
                    s2 = two_site_gmres(s2,i-1,s2,eta,MicroIter);
                    auto[lef,rig] = svd<Type,2,2>(s2,'l',tol_, MaxBond_);
                    state_[i] = std::move(rig);
                    state_[i-1] = std::move(lef);
                    if(i!=1)
                    {
                        envr_.push_back(MPO<Type>::sweep(envr_.back(),ham_[i],state_[i],state_[i],'l'));
                        envl_.pop_back();
                    }
                }
                Type CurEne = ham_.join(state_,state_);
                MPS<Type> hs0 = ham_.apply_op(state_);
                Type CurH2 = hs0.overlap(hs0);
                double CurVar = std::real(CurH2 - std::pow(CurEne,2.0));
                std::cout << std::scientific << "<H>  = " << CurEne << std::endl; 
                std::cout << std::scientific << "<H^2>-<H>^2  = " << Var << std::endl; 
                if(CurVar<=0.01 || CurVar>Var)
                {
                    break;
                }
                else
                {
                    Var = CurVar;
                }
            }
            envr_.push_back(MPO<Type>::sweep(envr_.back(),ham_[1],state_[1],state_[1],'l'));
        }
        void one_site_SI(Type const & eta, int NumSwps, int MicroIter)
        {
            int ns = ham_.size();
            Type Ene = ham_.join(state_,state_);
            MPS<Type> hs0 = ham_.apply_op(state_);
            Type CurH2 = hs0.overlap(hs0);
            double Var = std::real(CurH2 - std::pow(Ene,2.0));
            for(int swp=0;swp<NumSwps;++swp)
            {
                for(int i=0;i<ns;++i)
                {
                    Dense<Type,3> s1 = one_site_gmres(state_[i], i, state_[i], eta, MicroIter);
                    if(i!=ns-1)
                    {
                        auto[lef,rig] = svd<Type,2,1>(s1,'r',tol_, MaxBond_);
                        state_[i] = std::move(lef);
                        state_[i+1] = prod<Type,2,3,1>(rig,state_[i+1],{1},{0});
                        envl_.push_back(MPO<Type>::sweep(envl_.back(),ham_[i],state_[i],state_[i],'r'));
                        envr_.pop_back();
                    }
                    else
                    {
                        state_[i] = std::move(s1);
                    }
                }
                for(int i=ns;i>0;--i)
                {
                    Dense<Type,3> s1 = one_site_gmres(state_[i-1], i-1, state_[i-1],eta, MicroIter);
                    if(i!=1)
                    {
                        auto[lef,rig] = svd<Type,1,2>(s1,'l',tol_, MaxBond_);
                        state_[i-1] = std::move(rig);
                        state_[i-2] = prod<Type,3,2,1>(state_[i-2],lef,{2},{0});
                        envr_.push_back(MPO<Type>::sweep(envr_.back(),ham_[i-1],state_[i-1],state_[i-1],'l'));
                        envl_.pop_back();
                    }
                    else
                    {
                        state_[i-1] = std::move(s1);
                    }
                }
                Type CurEne = ham_.join(state_,state_);
                MPS<Type> hs0 = ham_.apply_op(state_);
                Type CurH2 = hs0.overlap(hs0);
                double CurVar = std::real(CurH2 - std::pow(CurEne,2.0));
                std::cout << std::scientific << "<H>  = " << CurEne << std::endl; 
                std::cout << std::scientific << "<H^2>-<H>^2  = " << Var << std::endl; 
                if(CurVar<=0.01 || CurVar>Var)
                {
                    break;
                }
                else
                {
                    Var = CurVar;
                }
            }
        }

        static char check_conv(std::vector<Type> const & E1, std::vector<Type> const & E2, double gap = 0.1)
        {
            char conv = 'y';
            if(E1.size()!=E2.size())
            {
              conv = 'n';
              return conv;
            }
            for(int i=0;i<E1.size();++i)
            {
              if(std::abs(E1[i]-E2[i])>=gap)
              {
                conv = 'n';
                break;
              }
            }
            return conv;
        }
 
        MPS<Type> get_mps() const { return state_; }
        MPO<Type> get_mpo() const { return ham_; }

        protected:
        MPO<Type> ham_;
        MPS<Type> state_;
        MPS<Type> state0_;
        double tol_;
        int MaxBond_;
        std::vector<Dense<Type,3>> envl_;
        std::vector<Dense<Type,3>> envr_;
        std::vector<Dense<Type,2>> ovl_;
        std::vector<Dense<Type,2>> ovr_;
    };

    class TDVP
    {
        public:
        TDVP(MPO<MKL_Complex16> const & op, MPS<MKL_Complex16> const & s, double tol = 1e-8, int maxdim = 20)
        : ham_(op), state_(s), state0_(s), tol_(tol), MaxBond_(maxdim)
        {
            Dense<MKL_Complex16,3> edges({1,1,1});
            edges({0,0,0})= 1.0;
            envl_.push_back(edges);
            envr_.push_back(edges);
            int ns = op.size();
            for(int i=ns-1;i>1;--i)
            {
                envr_.push_back(MPO<MKL_Complex16>::sweep(envr_.back(),ham_[i],state_[i],state_[i],'l'));
            }
        }
        ~TDVP() = default;

        // Hv
        static Dense<MKL_Complex16,2> zero_site_apply(Dense<MKL_Complex16,3> const & envl, Dense<MKL_Complex16,2> const & s, Dense<MKL_Complex16,3> const & envr)
        {
            Dense<MKL_Complex16,3> lss = prod<MKL_Complex16,3,2,1>(envl,s,{2},{0});
            Dense<MKL_Complex16,2> res = prod<MKL_Complex16,3,3,2>(lss,envr,{1,2},{1,2});
            return res;
        }
        static Dense<MKL_Complex16,3> one_site_apply(Dense<MKL_Complex16,5> const & vlo, Dense<MKL_Complex16,3> const & s, Dense<MKL_Complex16,3> const & envr)
        {
            Dense<MKL_Complex16,4> los = prod<MKL_Complex16,5,3,2>(vlo,s,{3,4},{0,1});
            Dense<MKL_Complex16,3> res = prod<MKL_Complex16,4,3,2>(los,envr,{2,3},{1,2});
            return res;
        }
        static Dense<MKL_Complex16,4> two_site_apply(Dense<MKL_Complex16,5> const & vlo, Dense<MKL_Complex16,4> const & s2, Dense<MKL_Complex16,5> const & vro)
        {
            Dense<MKL_Complex16,5> los = prod<MKL_Complex16,5,4,2>(vlo,s2,{3,4},{0,1});
            Dense<MKL_Complex16,4> res = prod<MKL_Complex16,5,5,3>(los,vro,{2,3,4},{0,1,2});
            return res;
        }
        Dense<MKL_Complex16,4> two_site_lanczos(Dense<MKL_Complex16,4> const & x0, int site, double dt, int MicroIter)
        {
            std::vector<Dense<MKL_Complex16,4>> Vk(MicroIter,x0.shape()), Wk(MicroIter,x0.shape());
            Dense<MKL_Complex16,5> vlo = prod<MKL_Complex16,3,4,1>(envl_.back(),ham_[site],{1},{0});
            vlo = transpose<MKL_Complex16,5>(vlo,{0,2,4,1,3});
            Dense<MKL_Complex16,5> vro = prod<MKL_Complex16,4,3,1>(ham_[site+1],envr_.back(),{3},{1});
            vro = transpose<MKL_Complex16,5>(vro,{0,2,4,1,3});
            double beta = x0.norm();
            Vk[0] = x0 / beta;
            Dense<MKL_Complex16,4> res(x0.shape());
            Dense<MKL_Complex16,2> Hk({MicroIter,MicroIter});
            // main program
            for(int i=0;i<MicroIter;++i)
            {
                Dense<MKL_Complex16,4> hv = two_site_apply(vlo,Vk[i],vro);
                hv *= dt;
                for(int j=0;j<i+1;++j)
                {
                    MKL_Complex16 ov = -1.0 * Vk[j].overlap(hv);
                    Hk({j,i}) = -1.0*ov;
                    hv += Vk[j] * ov;
                }
                if(i!=MicroIter-1)
                {
                    Hk({i+1,i}) = hv.norm();
                    hv = hv / Hk({i+1,i});
                    Vk[i+1] = std::move(hv);
                }
            }
            Dense<MKL_Complex16,2> Egak = eig<MKL_Complex16>(Hk);
            Dense<MKL_Complex16,1> UeU({MicroIter});
            MKL_Complex16 ones(0,-1.0);
            for(int i=0;i<MicroIter;++i)
            {
                Egak({i,i}) = std::exp(ones*Egak({i,i}));
                for(int j=0;j<MicroIter;++j)
                {
                    UeU({j}) += std::conj(Hk({0,i})) * Egak({i,i}) * Hk({j,i});
                }
            }
            for(int i=0;i<MicroIter;++i)
            {
                res += Vk[i] * UeU({i});
            }
            res /= res.norm();
            return res;
        }
        Dense<MKL_Complex16,3> one_site_lanczos(Dense<MKL_Complex16,3> const & x0, int site, double dt, int MicroIter)
        {
            Dense<MKL_Complex16,5> vlo = prod<MKL_Complex16,3,4,1>(envl_.back(),ham_[site],{1},{0});
            vlo = transpose<MKL_Complex16,5>(vlo,{0,2,4,1,3});
            double beta = x0.norm();
            std::vector<Dense<MKL_Complex16,3>> Vk(MicroIter,x0.shape());
            Vk[0] = x0 / beta;
            Dense<MKL_Complex16,3> res(x0);
            Dense<MKL_Complex16,2> Hk({MicroIter+1,MicroIter});
            // main program
            for(int i=0;i<MicroIter;++i)
            {
                Dense<MKL_Complex16,3> hv = one_site_apply(vlo,Vk[i],envr_.back());
                hv *= dt;
                for(int j=0;j<i+1;++j)
                {
                    MKL_Complex16 ov = -1.0 * Vk[j].overlap(hv);
                    Hk({j,i}) = -1.0*ov;
                    hv += Vk[j] * ov;
                }
                if(i!=MicroIter-1)
                {
                    Hk({i+1,i}) = hv.norm();
                    hv = hv / Hk({i+1,i});
                    Vk[i+1] = std::move(hv);
                }
            }
            Dense<MKL_Complex16,2> Egak = eig<MKL_Complex16>(Hk);
            Dense<MKL_Complex16,1> UeU({MicroIter});
            MKL_Complex16 ones(0,-1.0);
            for(int i=0;i<MicroIter;++i)
            {
                Egak({i,i}) = std::exp(ones*Egak({i,i}));
                for(int j=0;j<MicroIter;++j)
                {
                    UeU({j}) += std::conj(Hk({0,i})) * Egak({i,i}) * Hk({j,i});
                }
            }
            for(int i=0;i<MicroIter;++i)
            {
                res += Vk[i] * UeU({i});
            }
            res /= res.norm();
            return res;
        }
        Dense<MKL_Complex16,2> zero_site_lanczos(Dense<MKL_Complex16,2> const & x0, double dt, int MicroIter)
        {
            double beta = x0.norm();
            std::vector<Dense<MKL_Complex16,2>> Vk(MicroIter,x0.shape());
            Vk[0] = x0 / beta;
            Dense<MKL_Complex16,2> res(x0);
            Dense<MKL_Complex16,2> Hk({MicroIter+1,MicroIter});
            // main program
            for(int i=0;i<MicroIter;++i)
            {
                Dense<MKL_Complex16,2> hv = zero_site_apply(envl_.back(),Vk[i],envr_.back());
                hv *= dt;
                for(int j=0;j<i+1;++j)
                {
                    MKL_Complex16 ov = -1.0 * Vk[j].overlap(hv);
                    Hk({j,i}) = -1.0*ov;
                    hv += Vk[j] * ov;
                }
                if(i!=MicroIter-1)
                {
                    Hk({i+1,i}) = hv.norm();
                    hv = hv / Hk({i+1,i});
                    Vk[i+1] = std::move(hv);
                }
            }
            Dense<MKL_Complex16,2> Egak = eig<MKL_Complex16>(Hk);
            Dense<MKL_Complex16,1> UeU({MicroIter});
            MKL_Complex16 ones(0,-1.0);
            for(int i=0;i<MicroIter;++i)
            {
                Egak({i,i}) = std::exp(ones*Egak({i,i}));
                for(int j=0;j<MicroIter;++j)
                {
                    UeU({j}) += std::conj(Hk({0,i})) * Egak({i,i}) * Hk({j,i});
                }
            }
            for(int i=0;i<MicroIter;++i)
            {
                res += Vk[i] * UeU({i});
            }
            res /= res.norm();
            return res;
        }
        void two_site(double dt, int NumSwps, int MicroIter)
        {
            int ns = ham_.size();
            MKL_Complex16 Ene = 0.0, Ovp = 0.0;
            for(int swp=0;swp<NumSwps;++swp)
            {
                for(int i=0;i<ns-1;++i)
                {
                    Dense<MKL_Complex16,4> s2 = prod<MKL_Complex16,3,3,1>(state_[i],state_[i+1],{2},{0});
                    s2 = two_site_lanczos(s2,i,dt,MicroIter);
                    auto[lef,rig] = svd<MKL_Complex16,2,2>(s2,'r',tol_, MaxBond_);
                    state_[i] = std::move(lef);
                    if(i!=ns-2)
                    {
                        envl_.push_back(MPO<MKL_Complex16>::sweep(envl_.back(),ham_[i],state_[i],state_[i],'r'));
                        state_[i+1] = one_site_lanczos(rig,i+1,-1.0*dt,MicroIter);
                        envr_.pop_back();
                    }
                    else
                    {
                        state_[i+1] = std::move(rig);
                    }
                }
                for(int i=ns-1;i>0;--i)
                {
                    Dense<MKL_Complex16,4> s2 = prod<MKL_Complex16,3,3,1>(state_[i-1],state_[i],{2},{0});
                    s2 = two_site_lanczos(s2,i-1,dt,MicroIter);
                    auto[lef,rig] = svd<MKL_Complex16,2,2>(s2,'l',tol_, MaxBond_);
                    state_[i] = std::move(rig);
                    if(i!=1)
                    {
                        envr_.push_back(MPO<MKL_Complex16>::sweep(envr_.back(),ham_[i],state_[i],state_[i],'l'));
                        state_[i-1] = one_site_lanczos(lef,i-1,-1.0*dt,MicroIter);
                        envl_.pop_back();
                    }
                    else
                    {
                        state_[i-1] = std::move(lef);
                    }
                }
            }
            envr_.push_back(MPO<MKL_Complex16>::sweep(envr_.back(),ham_[1],state_[1],state_[1],'l'));
        }
        void one_site(double dt, int NumSwps, int MicroIter)
        {
            int ns = ham_.size();
            MKL_Complex16 Ene = 0.0, Ovp = 0.0;
            for(int swp=0;swp<NumSwps;++swp)
            {
                for(int i=0;i<ns;++i)
                {
                    Dense<MKL_Complex16,3> s1 = one_site_lanczos(state_[i], i, dt, MicroIter);
                    if(i!=ns-1)
                    {
                        auto[lef,rig] = svd<MKL_Complex16,2,1>(s1,'r',tol_, MaxBond_);
                        state_[i] = std::move(lef);
                        envl_.push_back(MPO<MKL_Complex16>::sweep(envl_.back(),ham_[i],state_[i],state_[i],'r'));
                        rig = zero_site_lanczos(rig,-1.0*dt,MicroIter);
                        state_[i+1] = prod<MKL_Complex16,2,3,1>(rig,state_[i+1],{1},{0});
                        envr_.pop_back();
                    }
                    else
                    {
                        state_[i] = std::move(s1);
                    }
                }
                for(int i=ns;i>0;--i)
                {
                    Dense<MKL_Complex16,3> s1 = one_site_lanczos(state_[i-1], i-1, dt, MicroIter);
                    if(i!=1)
                    {
                        auto[lef,rig] = svd<MKL_Complex16,1,2>(s1,'l',tol_, MaxBond_);
                        state_[i-1] = std::move(rig);
                        envr_.push_back(MPO<MKL_Complex16>::sweep(envr_.back(),ham_[i-1],state_[i-1],state_[i-1],'l'));
                        lef = zero_site_lanczos(lef,-1.0*dt,MicroIter);
                        state_[i-2] = prod<MKL_Complex16,3,2,1>(state_[i-2],lef,{2},{0});
                        envl_.pop_back();
                    }
                    else
                    {
                        state_[i-1] = std::move(s1);
                    }
                }
            }
        }
 
        MPS<MKL_Complex16> get_mps() const { return state_; }
        MPO<MKL_Complex16> get_mpo() const { return ham_; }

        protected:
        MPO<MKL_Complex16> ham_;
        MPS<MKL_Complex16> state_;
        MPS<MKL_Complex16> state0_;
        double tol_;
        int MaxBond_;
        std::vector<Dense<MKL_Complex16,3>> envl_;
        std::vector<Dense<MKL_Complex16,3>> envr_;
    };

    // naive feast implementation
    class FEAST
    {
        public:
        FEAST(MPO<MKL_Complex16> const & op, std::vector<MPS<MKL_Complex16>> const & s, double tol, int maxdim)
        : Vm_(s),ham_(op), tol_(tol), MaxBond_(maxdim)
        {
            
        }
        ~FEAST() = default;

        void naive_impl(double Emin, double Emax,std::vector<MKL_Complex16> & Ene)
        {
            double r0 = (Emax - Emin) / 2.0, z0 = (Emax + Emin) / 2.0;
            double zj[8] = {-0.96028986, -0.79666648, -0.52553241, -0.18343464, 0.18343464, 0.52553241, 0.79666648, 0.96028986};
            MKL_Complex16 cpi(0,3.141592535*2.0);
            for(int mac=0;mac<1;++mac)
            {
                std::cout << "Macro Iteration " << mac+1 << std::endl;
                int Nm = Vm_.size();
                std::vector<MPS<MKL_Complex16>> VTmp(8*Nm);
                for(int i=0;i<Nm;++i)
                {
                    for(int node=0;node<8;++node)
                    {
                        std::cout << "Node " << node+1 << std::endl;
                        MKL_Complex16 znode = z0 + r0 * std::exp(zj[node]*cpi);
                        //MKL_Complex16 znode(Emin+node*r0/4,0.0);
                        std::cout << "Eta = " << znode << std::endl;
                        DMRG dnode(ham_,Vm_[i],tol_,MaxBond_);
                        dnode.two_site_SI(znode,5,5);
                        dnode.one_site_SI(znode,5,5);
                        VTmp[i*8+node] = dnode.get_mps();
                    }
                }
                Nm = VTmp.size();
                Dense<MKL_Complex16,2> Hm({Nm,Nm}), Sm({Nm,Nm});
                for(int i=0;i<Nm;++i)
                {
                    for(int j=i;j<Nm;++j)
                    {
                        Hm({i,j}) = ham_.join(VTmp[i],VTmp[j]);
                        Sm({i,j}) = VTmp[i].overlap(VTmp[j]);
                        Hm({j,i}) = std::conj(Hm({i,j}));
                        Sm({j,i}) = std::conj(Sm({i,j}));
                    }
                }
                Dense<MKL_Complex16,2> Smc(Sm), XSm(Sm.shape());
                Dense<MKL_Complex16,2> EigSm = eig<MKL_Complex16>(Smc);
                for(int im=0;im<Nm;++im)
                {
                    XSm({im,im}) = std::pow(EigSm({im,im}),-0.5);
                }
                XSm = prod<MKL_Complex16,2,2,1>(Smc,XSm,{1},{0});
                Dense<MKL_Complex16,2> Hog = prod<MKL_Complex16,2,2,1>(Hm,XSm,{1},{0});
                Hog = prod<MKL_Complex16,2,2,1>(conj<MKL_Complex16>(XSm),Hog,{0},{0});

                Dense<MKL_Complex16,2> EigHog = eig<MKL_Complex16>(Hog);
                Dense<MKL_Complex16,2> RealEvc = prod<MKL_Complex16,2,2,1>(XSm,Hog,{1},{0});

                std::vector<MKL_Complex16> CurEne;
                std::vector<MPS<MKL_Complex16>> inners;
                for(int i=0;i<Nm;++i)
                {
                    if( std::real(EigHog({i,i})) < Emax && std::real(EigHog({i,i})) > Emin  )
                    {
                        std::cout << "Energy " << i+1 << " : " << EigHog({i,i})
                        << std::endl;
                        CurEne.push_back(EigHog({i,i}));
                        MPS<MKL_Complex16> ins = VTmp[0] * RealEvc({0,i});
                        for(int j=1;j<Nm;++j)
                        {
                            ins += VTmp[j] * RealEvc({j,i});
                            ins.canon();
                        }
                        ins *= 1.0 / ins[0].norm();
                        std::vector<LabArr<MKL_Complex16,2>> inslabs = ins.dominant(0.1);
                        for(auto const & lab : inslabs)
                        {
                            lab.print_special(0.0);
                        }
                        inners.push_back(ins);
                    }
                }
                if( DMRG<MKL_Complex16>::check_conv(Ene,CurEne,0.01)=='y')
                {
                    break;
                }
                else
                {
                    Ene = CurEne;
                    Vm_ = inners;
                }
            }
        }

        std::vector<MPS<MKL_Complex16>> const & get_mps() const
        {
            return Vm_;
        }

        private:
        MPO<MKL_Complex16> ham_;
        std::vector<MPS<MKL_Complex16>> Vm_;
        double tol_;
        int MaxBond_;
    };
}
