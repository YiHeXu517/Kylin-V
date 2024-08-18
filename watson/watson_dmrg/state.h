/* All kinds of quantum states */

#pragma once

#include "algebra.h"

namespace KylinVib
{
    namespace WatsonDMRG
    {
        // double-version obc matrix product state
        class xMPS
        {
            private:

            // tensor train
            std::vector<ArrR<3>> tt_;

            public:

            // default tors
            xMPS() = default;

            ~xMPS() = default;

            // copy
            xMPS(xMPS const & r)
            : tt_(r.tt_)
            {

            }

            xMPS & operator=(xMPS const & r)
            {
                tt_ = r.tt_;
                return *this;
            }

            // move
            xMPS(xMPS && r)
            : tt_(r.tt_)
            {

            }

            xMPS & operator=(xMPS && r)
            {
                tt_ = std::move(r.tt_);
                return *this;
            }

            // allocation
            xMPS(INT nsite)
            : tt_(nsite)
            {

            }

            // double allocation without initialization
            xMPS(INT nsite, Brace Sp)
            : tt_(nsite, Sp)
            {

            }

            // plus
            xMPS & operator+=(xMPS const & r)
            {
                INT Ns = tt_.size();
                tt_[0] = Alg::stack<3>(tt_[0],r.tt_[0],{2});
                tt_[Ns-1] = Alg::stack<3>(tt_[Ns-1],r.tt_[Ns-1],{0});
                for(INT i=1;i<Ns-1;++i)
                {
                    tt_[i] = Alg::stack<3>(tt_[i],r.tt_[i],{0,2});
                }
                return *this;
            }
            xMPS operator+(xMPS const & r) const
            {
                xMPS res(*this);
                INT Ns = tt_.size();
                res.tt_[0] = Alg::stack<3>(tt_[0],r.tt_[0],{2});
                res.tt_[Ns-1] = Alg::stack<3>(tt_[Ns-1],r.tt_[Ns-1],{0});
                for(INT i=1;i<Ns-1;++i)
                {
                    res.tt_[i] = Alg::stack<3>(tt_[i],r.tt_[i],{0,2});
                }
                return res;
            }
            xMPS & add_right_stack(xMPS const & r)
            {
                INT Ns = tt_.size();
                tt_[0] = Alg::stack<3>(tt_[0],r.tt_[0],{2});
                for(INT i=1;i<Ns;++i)
                {
                    tt_[i] = Alg::stack<3>(tt_[i],r.tt_[i],{0,2});
                }
                return *this;
            }
            xMPS & add_left_stack(xMPS const & r)
            {
                INT Ns = tt_.size();
                tt_[Ns-1] = Alg::stack<3>(tt_[Ns-1],r.tt_[Ns-1],{0});
                for(INT i=0;i<Ns-1;++i)
                {
                    tt_[i] = Alg::stack<3>(tt_[i],r.tt_[i],{0,2});
                }
                return *this;
            }

            xMPS slice(INT Starts, INT Ends)
            {
                xMPS res(Ends-Starts);
                std::copy(tt_.begin()+Starts, tt_.begin()+Ends, res.tt_.begin());
                return res;
            }

            // self multiplication
            xMPS & operator*=(double val)
            {
                tt_[0] *= val;
                return *this;
            }

            xMPS operator*(double const & val) const
            {
                xMPS res(*this);
                res.tt_[0] = res.tt_[0] * val;
                return res;
            }

            // index
            ArrR<3> & operator[](INT idx)
            {
                return tt_[idx];
            }

            const ArrR<3> & at(INT idx) const
            {
                return tt_[idx];
            }

            // number of site
            INT nsite() const
            {
                return tt_.size();
            }

            // print out
            void print(INT ncol = 1) const
            {
                std::cout << tt_.size() << "-site " <<  " xMPS:" << std::endl;
                for(INT i=0;i<tt_.size();++i)
                {
                    std::cout << "Site " << i << ":" << std::endl;
                    tt_[i].print(ncol);
                }
            }

            // canonicalize
            void canon(char ReN = 'r', INT MaxBond = 1000)
            {
                INT Ns = tt_.size();
                for(INT i=0;i<Ns-1;++i)
                {
                    auto[lef,rig] = Alg::svd<2,1>(tt_[i],'r',1e-14,MaxBond,'n');
                    tt_[i] = std::move(lef);
                    tt_[i+1] = Alg::gmm<2,3,1>(rig,tt_[i+1]);
                }
                for(INT i=Ns-1;i>0;--i)
                {
                    auto[lef,rig] = Alg::svd<1,2>(tt_[i],'l',1e-14,MaxBond,'n');
                    tt_[i] = std::move(rig);
                    tt_[i-1] = Alg::gmm<3,2,1>(tt_[i-1],lef);
                }
                if(ReN=='r')
                {
                    double r2 = this->norm();
                    tt_[0] /= r2;
                }
            }
            void change_site_shape(INT site, INT nlevel)
            {
                ArrR<3> res({1,nlevel,1});
                tt_[site] = std::move(res);
            }

            // norm
            double norm() const
            {
                return Alg::norm(tt_[0]);
            }

            // overlap
            double overlap(xMPS const & r) const
            {
                INT Ns = tt_.size();
                ArrR<2> env({1,1}); env({0,0}) = 1.0;
                for(INT i=0;i<Ns;++i)
                {
                    ArrR<3> lss = Alg::gmm<2,3,1>(env,r.tt_[i]);
                    env = Alg::gmm<3,3,2>(tt_[i],lss,CblasTrans,CblasNoTrans);
                }
                return env.ptr()[0];
            }
            std::vector<std::vector<int>> dominant(double tol = 1e-1) const
            {
                //this->print(5);
                INT Ns = tt_.size(), d = tt_[0].shape()[1];
                std::vector<ArrR<2>> LevelComp = Alg::all_levels(tt_[0]);
                std::vector<std::vector<int>> res;
                int sig = 0;
                for(auto it=LevelComp.begin();it!=LevelComp.end();)
                {
                    double nrms = Alg::norm<2>(*it);
                    //std::cout << sig << " : " << nrms << std::endl;
                    if(nrms<=tol)
                    {
                        it = LevelComp.erase(it);
                    }
                    else
                    {
                        std::vector<int> cfs = {sig};
                        res.push_back(cfs);
                        it++;
                    }
                    sig++;
                }
                ArrR<3> env = Alg::comb_levels(LevelComp);
                for(INT i(1);i<Ns;++i)
                {
                    ArrR<4> AT4 = Alg::gmm<3,3,1>(env,tt_[i]);
                    ArrR<3> AT3 = Alg::reshape<4,3>(AT4,{AT4.shape()[0],
                    AT4.shape()[1]*AT4.shape()[2],AT4.shape()[3]});
                    LevelComp = Alg::all_levels(AT3);
                    std::vector<int> cfsi; sig = 0;
                    for(auto it=LevelComp.begin();it!=LevelComp.end();)
                    {
                        double nrms = Alg::norm<2>(*it);
                        std::cout << i << ":" << sig << " : " << nrms << std::endl;
                        if(nrms<=tol)
                        {
                            it = LevelComp.erase(it);
                        }
                        else
                        {
                            cfsi.push_back(sig);
                            it++;
                        }
                        sig++;
                    }
                    std::vector<std::vector<int>> nres(cfsi.size());
                    for(INT j(0);j<cfsi.size();++j)
                    {
                        INT xi = cfsi[j] / d;
                        int yi = cfsi[j] % d;
                        std::vector<int> resi = res[xi];
                        resi.push_back(yi);
                        nres[j] = resi;
                    }
                    res = std::move(nres);
                    env = Alg::comb_levels(LevelComp);
                }
                std::cout << res.size() << " configurations." << std::endl;
                //env.print_sp();
                for(INT i(0);i<res.size();++i)
                {
                    std::for_each(res[i].begin(),res[i].end(),[](int & x)
                    {std::cout << x << ",";});
                    std::cout << Alg::norm<2>(LevelComp[i]) << std::endl;
                }
                return res;
            }
            // initialize MPS with excitation map
            void rand_initialize(INT Nphys, INT bd)
            {
                INT Ns = tt_.size();
                ArrR<2> FilterHigh({Nphys,Nphys});
                for(INT i(0);i<Nphys/2;++i)
                {
                    FilterHigh({i,i}) = 1.0;
                }
                for(INT i=0;i<Ns;++i)
                {
                    INT Blef = (i==0) ? 1 : bd;
                    INT Brig = (i==Ns-1) ? 1 : bd;
                    ArrR<3> ri({Blef,Brig,Nphys});
                    ri.rand_fill();
                    ri = Alg::gmm<3,2,1>(ri,FilterHigh);
                    tt_[i] = std::move(Alg::transpose<3>(ri,{0,2,1}));
                }
            }
            void one_initialize(INT Nphys, INT bd)
            {
                INT Ns = tt_.size();
                for(INT i=0;i<Ns;++i)
                {
                    INT Blef = (i==0) ? 1 : bd;
                    INT Brig = (i==Ns-1) ? 1 : bd;
                    ArrR<3> ri({Blef,Nphys,Brig});
                    for(INT j=0;j<ri.size();++j)
                    {
                        ri.ptr()[j] = 1.0;
                    }
                    tt_[i] = std::move(ri);
                }
            }
            // initialize MPS with excitation map
            void initialize(std::map<INT,INT> const & excs, INT Nphys)
            {
                INT Ns = tt_.size();
                for(INT i=0;i<Ns;++i)
                {
                    ArrR<3> ri({1,Nphys,1});
                    auto it = excs.find(i);
                    if(it==excs.end())
                    {
                        ri({0,0,0}) = 1.0;
                    }
                    else
                    {
                        ri({0,it->second,0}) = 1.0;
                    }
                    tt_[i] = std::move(ri);
                }
            }
            void initialize(std::vector<INT> const & excs, INT Nphys)
            {
                INT Ns = tt_.size();
                for(INT i=0;i<Ns;++i)
                {
                    ArrR<3> ri({1,Nphys,1});
                    ri({0,excs[i],0}) = 1.0;
                    tt_[i] = std::move(ri);
                }
            }
 
            void initialize(std::map<INT,INT> const & excs)
            {
                INT Ns = tt_.size();
                for(INT i=0;i<Ns;++i)
                {
                    auto it = excs.find(i);
                    if(it==excs.end())
                    {
                        tt_[i]({0,0,0}) = 1.0;
                    }
                    else
                    {
                        tt_[i]({0,it->second,0}) = 1.0;
                    }
                }
            }
 
            // maximal bond dimension
            INT max_bond() const
            {
                INT Ns = tt_.size();
                INT res = 1;
                for(INT i=0;i<Ns;++i)
                {
                    res = std::max(tt_[i].shape()[2],res);
                }
                return res;
            }
        };
    }
}
