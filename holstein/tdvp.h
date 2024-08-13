/* TDVP implemention */

#ifndef hols_TDVP_H
#define hols_TDVP_H

#include "lattice.h"
#include "krylov.h"

namespace KylinVib
{
    namespace Holstein
    {
        using std::real;
        using std::stringstream;
        using std::getline;
        using std::for_each;

        // parse the stage
        array<INT,4> parse_stage(const string & s)
        {
            array<INT,4> r;
            stringstream ss;
            string tmp;
            ss.str( s.substr(1,s.size()-1) );
            for(INT i(0);i<4;++i)
            {
                getline(ss,tmp,',');
                r[i] = stol(tmp);
            }
            ss.str("");
            return r;
        }

        class TDVP
        {
        private:
            // SVD input
            SVDConfig SVDCfg_;

            // Krylov input
            KrylovConfig KryCfg_;

            // state
            State mps_;

            // Hamiltonian
            Operator ham_;

            // left env tensor
            vector<Tensor<3>> EnvL_;

            // right env tensor
            vector<Tensor<3>> EnvR_;

        public:
            /* ctor & dtor & assign */
            TDVP() = default;
            ~TDVP() = default;

            TDVP( State & MPS, Operator & MPO, double tolSVD, double tolExp, INT maxDim,
             INT maxIter, double timeStep)
            : mps_(move(MPS)), ham_(move(MPO))
            {
                SVDCfg_.SVDThres = tolSVD;
                SVDCfg_.MaxDim   = maxDim;

                KryCfg_.KrylovThres = tolExp;
                KryCfg_.MaxIter     = maxIter;
                KryCfg_.TimeStep    = timeStep;

                cout << KryCfg_ << SVDCfg_ << endl;
                ofstream ofs("SVS");
                ofs << "Total Site = " << mps_.nsite() << endl;
                ofs.close();

            }
            // one site for one step
            void one_site(Timer & tm)
            {
                INT ns = mps_.nsite();
                Tensor<4> DummyOp1,DummyOp2;
                SVDCfg_.Left2Right = true;
                for(INT i(0);i<ns;++i)
                {
                      start_module(tm,"Evl-1");
                    KrylovExit KE1 = local_krylov<3>(EnvL_.back(),mps_[i], ham_.at(i), DummyOp2, EnvR_.back(),KryCfg_);
                      end_module(tm,"Evl-1");
                    if(i<ns-1)
                    {
                          start_module(tm,"SVD");
                        auto[cur,nex] = svd_impl<2,1>( mps_[i], SVDCfg_,-1);
                          end_module(tm,"SVD");

                        mps_[i] = move(cur);

                          start_module(tm,"Update");
                        EnvL_.push_back( env_sweep(EnvL_.back(),mps_.at(i), ham_.at(i), SVDCfg_.Left2Right) );
                          end_module(tm,"Update");

                          start_module(tm,"Evl-0");
                        KrylovExit KE0 = local_krylov<2>(EnvL_.back(), nex, DummyOp1, DummyOp2, EnvR_.back(), KryCfg_,true);
                          end_module(tm,"Evl-0");

                          start_module(tm,"Sweep");
                        mps_[i+1] = product<2,3,1>(nex,mps_[i+1],CblasNoTrans,CblasNoTrans);
                          end_module(tm,"Sweep");

                        EnvR_.pop_back();
                    }
                }
                SVDCfg_.Left2Right = false;
                for(INT i(ns-1);i>=0;--i)
                {
                      start_module(tm,"Evl-1");
                    KrylovExit KE1 = local_krylov<3>(EnvL_.back(),mps_[i], ham_.at(i), DummyOp2, EnvR_.back(),KryCfg_);
                      end_module(tm,"Evl-1");

                    if(i>0)
                    {
                          start_module(tm,"SVD");
                        auto[nex,cur] = svd_impl<1,2>( mps_[i], SVDCfg_,-1);
                          end_module(tm,"SVD");

                        mps_[i] = move(cur);

                          start_module(tm,"Update");
                        EnvR_.push_back( env_sweep(EnvR_.back(),mps_.at(i), ham_.at(i),SVDCfg_.Left2Right) );
                          end_module(tm,"Update");

                          start_module(tm,"Evl-0");
                        KrylovExit KE0 = local_krylov<2>(EnvL_.back(), nex, DummyOp1, DummyOp2, EnvR_.back(), KryCfg_,true);
                          end_module(tm,"Evl-0");

                          start_module(tm,"Sweep");
                        mps_[i-1] = product<3,2,1>(mps_[i-1],nex,CblasNoTrans,CblasNoTrans);
                          end_module(tm,"Sweep");

                        EnvL_.pop_back();
                    }
                }
            }

            // two site for one step
            void two_site(  Timer & tm)
            {

                struct sysinfo sif;
                int err = sysinfo(&sif);
                //cout << "Bef  Total: " << sif.totalram / 1024/1024/1024 << " GB" << endl;
                //cout << "Bef  Free:  " << sif.freeram / 1024/1024/1024 << " GB" << endl;

                INT ns = mps_.nsite();
                Tensor<4> DummyOp;
                SVDCfg_.Left2Right = true;
                EnvR_.pop_back();

                for(INT i(0);i<ns-1;++i)
                {

                      start_module(tm,"Evl-2");
                    Tensor<4> mm = product<3,3,1>(mps_.at(i),mps_.at(i+1),CblasNoTrans,CblasNoTrans);
                    KrylovExit KE2 = local_krylov<4>(EnvL_.back(),mm, ham_.at(i),ham_.at(i+1), EnvR_.back(),KryCfg_);
                      end_module(tm,"Evl-2");

                      start_module(tm,"SVD");
                    auto[cur,nex] = svd_impl<2,2>(mm, SVDCfg_, -1, false, true);
                      end_module(tm,"SVD");


                    if(i<ns-2)
                    {
                        mps_[i] = move(cur);

                          start_module(tm,"Update");
                        EnvL_.push_back( env_sweep(EnvL_.back(),mps_.at(i), ham_.at(i), SVDCfg_.Left2Right) );
                          end_module(tm,"Update");


                          start_module(tm,"Evl-1");
                        KrylovExit KE1 = local_krylov<3>(EnvL_.back(), nex, ham_.at(i+1), DummyOp, EnvR_.back(), KryCfg_,true);
                          end_module(tm,"Evl-1");


                        mps_[i+1] = move(nex);
                        EnvR_.pop_back();
                    }
                    else
                    {
                        mps_[i] = move(cur);
                        mps_[i+1] = move(nex);
                    }
                }
                SVDCfg_.Left2Right = false;
                for(INT i(ns-1);i>0;--i)
                {
                      start_module(tm,"Evl-2");
                    Tensor<4> mm = product<3,3,1>(mps_.at(i-1),mps_.at(i),CblasNoTrans,CblasNoTrans);
                    KrylovExit KE2 = local_krylov<4>(EnvL_.back(),mm, ham_.at(i-1), ham_.at(i), EnvR_.back(),KryCfg_);
                      end_module(tm,"Evl-2");

                      start_module(tm,"SVD");
                    auto[nex,cur] = svd_impl<2,2>(mm, SVDCfg_,-1, false,true);
                      end_module(tm,"SVD");

                    if(i>1)
                    {
                        mps_[i] = move(cur);

                          start_module(tm,"Update");
                        EnvR_.push_back( env_sweep(EnvR_.back(), mps_.at(i), ham_.at(i), SVDCfg_.Left2Right) );
                          end_module(tm,"Update");

                          start_module(tm,"Evl-1");
                        KrylovExit KE1 = local_krylov<3>(EnvL_.back(), nex, ham_.at(i-1), DummyOp, EnvR_.back(), KryCfg_,true);
                          end_module(tm,"Evl-1");
                        mps_[i-1] = move(nex);

                        EnvL_.pop_back();
                    }
                    else
                    {
                        mps_[i] = move(cur);
                        mps_[i-1] = move(nex);
                    }
                }
                EnvR_.push_back( env_sweep(EnvR_.back(),mps_.at(1),ham_.at(1),false) );
            }

            // sa-one-site for one step
            void sa_site(Timer & tm, vector<double> & betaSet)
            {
                INT ns = mps_.nsite();
                Tensor<4> DummyOp1,DummyOp2;
                SVDCfg_.Left2Right = true;
                for(INT i(0);i<ns;++i)
                {
                      start_module(tm,"Evl-1");
                    KrylovExit KE1 = local_krylov<3>(EnvL_.back(),mps_[i], ham_.at(i), DummyOp2, EnvR_.back(),KryCfg_);
                      end_module(tm,"Evl-1");
                    if(i<ns-1)
                    {
                          start_module(tm,"SVD");
                        auto[cur,nex] = sa_svd_impl<2,1>( mps_[i], SVDCfg_,-1, betaSet[i]);
                          end_module(tm,"SVD");

                        mps_[i] = move(cur);

                          start_module(tm,"Update");
                        EnvL_.push_back( env_sweep(EnvL_.back(),mps_.at(i), ham_.at(i), SVDCfg_.Left2Right) );
                          end_module(tm,"Update");

                          start_module(tm,"Evl-0");
                        KrylovExit KE0 = local_krylov<2>(EnvL_.back(), nex, DummyOp1, DummyOp2, EnvR_.back(), KryCfg_,true);
                          end_module(tm,"Evl-0");

                          start_module(tm,"Sweep");
                        mps_[i+1] = product<2,3,1>(nex,mps_[i+1],CblasNoTrans,CblasNoTrans);
                          end_module(tm,"Sweep");

                        EnvR_.pop_back();
                    }
                }
                SVDCfg_.Left2Right = false;
                for(INT i(ns-1);i>=0;--i)
                {
                      start_module(tm,"Evl-1");
                    KrylovExit KE1 = local_krylov<3>(EnvL_.back(),mps_[i], ham_.at(i), DummyOp2, EnvR_.back(),KryCfg_);
                      end_module(tm,"Evl-1");

                    if(i>0)
                    {
                          start_module(tm,"SVD");
                        auto[nex,cur] = sa_svd_impl<1,2>( mps_[i], SVDCfg_,-1,betaSet[i-1]);
                          end_module(tm,"SVD");

                        mps_[i] = move(cur);

                          start_module(tm,"Update");
                        EnvR_.push_back( env_sweep(EnvR_.back(),mps_.at(i), ham_.at(i),SVDCfg_.Left2Right) );
                          end_module(tm,"Update");

                          start_module(tm,"Evl-0");
                        KrylovExit KE0 = local_krylov<2>(EnvL_.back(), nex, DummyOp1, DummyOp2, EnvR_.back(), KryCfg_,true);
                          end_module(tm,"Evl-0");

                          start_module(tm,"Sweep");
                        mps_[i-1] = product<3,2,1>(mps_[i-1],nex,CblasNoTrans,CblasNoTrans);
                          end_module(tm,"Sweep");

                        EnvL_.pop_back();
                    }
                }

            }

            // total evolution stage
            void total_start(const array<INT,4> & r)
            {
                INT swp;
                Timer tm;

                cout << "\n ---------------- Start initialize stage ----------------" << endl;
                Tensor<3> envl({1,-1,-1});
                Block<3> ri({1,1,1},{0,0,0}); ri({0,0,0}) = 1.0;
                envl.push_back(ri); envl.check_legs(); EnvL_.push_back(envl);

                INT ns = mps_.nsite();
                INT totSym = mps_.at(ns-1).at(0).qn()[2];
                Tensor<3> envr({-1,1,1});
                Block<3> rf({1,1,1},{totSym,0,totSym}); rf({0,0,0}) = 1.0;
                envr.push_back(rf); envr.check_legs(); EnvR_.push_back(envr);

                for(INT i(ns-1);i>0;--i)
                {
                    EnvR_.push_back( env_sweep(EnvR_.back(),mps_.at(i),ham_.at(i),false) );
                }

                cout << "\n ---------------- Start Krylov stage ----------------" << endl;
                Operator Hc(ham_);
                Hc *= KryCfg_.TimeStep;
                for(swp=0;swp<r[0];++swp)
                {
                    global_main(Hc,mps_,KryCfg_,SVDCfg_,tm);
                    mps_.max_bond();
                    string fn = "State/T-" + to_string(swp) + ".mps";
                    ofstream ofs(fn);
                    mps_.save(ofs);
                }

                cout << "\n ---------------- Start 2TDVP stage ----------------" << endl;
                for(swp=r[0];swp<r[0]+r[1];++swp)
                {
                    cout << "\nSweep " << swp+1 << ":" << endl;
                    two_site(tm);
                    mps_.max_bond();
                    string fn = "State/T-" + to_string(swp) + ".mps";
                    ofstream ofs(fn);
                    mps_.save(ofs);
                }

                if(r[2]!=0) 
                {
                    vector<double> betaSet;
                    int ifos = system("nls.py");
                    cout << "Fitting OK!" << endl;

                    ifstream ifsBeta("beta");
                    if(!ifsBeta.is_open()) { cout << "Fail to open beta file" << endl; }
                    while (!ifsBeta.eof())
                    {
                        double bt; string tmp;
                        getline(ifsBeta,tmp);
                        if( tmp.size()>3)
                        {
                            bt = 1.0/stod(tmp);
                            betaSet.push_back(bt);
                        }
                    }
                   ifsBeta.close();
                   cout << "Read betas OK!" << endl;
                

                   cout << "\n ---------------- Start SA-1TDVP stage -------------" << endl;
                   for(swp=r[0]+r[1];swp<r[0]+r[1]+r[2];++swp)
                   {
                    cout << "\nSweep " << swp+1 << ":" << endl;
                    sa_site(tm,betaSet);
                    mps_.max_bond();
                    string fn = "State/T-" + to_string(swp) + ".mps";
                    ofstream ofs(fn);
                    mps_.save(ofs);
                   }
                }
                cout << "\n ---------------- Start 1TDVP stage ----------------" << endl;
                for(swp=r[0]+r[1]+r[2];swp<r[0]+r[1]+r[2]+r[3];++swp)
                {
                    cout << "\nSweep " << swp+1 << ":" << endl;
                    one_site(tm);
                    mps_.max_bond();
                    string fn = "State/T-" + to_string(swp) + ".mps";
                    ofstream ofs(fn);
                    mps_.save(ofs);
                }
                cout << tm;
            }
        };
    }
}

#endif
