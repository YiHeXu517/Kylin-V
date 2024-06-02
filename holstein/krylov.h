/* global krylov implemention  */

#ifndef hols_krylov_H
#define hols_krylov_H

#include "exponential.h"

namespace klnX
{
    namespace Holstein
    {
        /* f(N,L[j],L[j-1]) -> M */
        Tensor<4> apply_ortho_site ( const Tensor<4> & Nh, const Tensor<4> & La, const Tensor<4> & Lb )
        {
            Complex LNa = overlap(La,Nh); 
            Complex LNb = overlap(Lb,Nh);
            Complex aa  = pow(norm(La),2.0);
            Complex bb  = pow(norm(Lb),2.0);
            Complex ab  = overlap(La,Lb);
            Complex ba  = conj(ab);

            Complex cc  = ab*ba-aa*bb;
            Tensor<4> res(Nh);
            if(abs(cc)<1e-17) { 
                if(abs(aa)<1e-17)
                {
                    return res;
                }
                else
                {
                    Tensor<4> Lac(La);
                    Complex Betaa = -1.0 * LNa / aa;
                    res += Lac * Betaa;
                }
            }
            else
            {
                Complex Betaa = ( bb * LNa - ab * LNb ) / cc;
                Complex Betab = ( aa * LNb - ba * LNa ) / cc;
                Tensor<4> Lac(La),Lbc(Lb);
                res += Lac * Betaa;
                res += Lbc * Betab;
            }
            return res;
        }

        /* initialize env-3 tensor pair */
        void init_env_op ( const Operator & H, const State & s1, const State & s2,
        vector<Tensor<3>> & EnvL, vector<Tensor<3>> & EnvR)
        {
            Tensor<3> envl({1,-1,-1});
            Block<3> ri({1,1,1},{0,0,0}); ri({0,0,0}) = 1.0;
            envl.push_back(ri); envl.check_legs(); EnvL.push_back(envl);

            INT ns = s1.nsite();
            INT totSym = s1.at(ns-1).at(0).qn()[2];
            Tensor<3> envr({-1,1,1});
            Block<3> rf({1,1,1},{totSym,0,totSym}); rf({0,0,0}) = 1.0;
            envr.push_back(rf); envr.check_legs(); EnvR.push_back(envr);

            for(INT i(ns-1);i>1;--i)
            {
                EnvR.push_back( env_sweep(EnvR.back(),s1.at(i),H.at(i),s2.at(i),false) );
            } 
        }

        /* initialize env-2 tensor pair */
        void init_env_state ( const State & s1, const State & s2,
        vector<Tensor<2>> & EnvL, vector<Tensor<2>> & EnvR)
        {
            Tensor<2> envl({1,-1});
            Block<2> ri({1,1},{0,0}); ri({0,0}) = 1.0;
            envl.push_back(ri); envl.check_legs(); EnvL.push_back(envl);

            INT ns = s1.nsite();
            INT totSym = s1.at(ns-1).at(0).qn()[2];
            Tensor<2> envr({-1,1});
            Block<2> rf({1,1},{totSym,totSym}); rf({0,0}) = 1.0;
            envr.push_back(rf); envr.check_legs(); EnvR.push_back(envr);

            for(INT i(ns-1);i>1;--i)
            {
                Tensor<3> s1t = transpose<3>(s1.at(i),{2,1,0});
                Tensor<3> s2t = transpose<3>(s2.at(i),{2,1,0});
                Tensor<3> lm  = product<2,3,1>(EnvR.back(),s2t,CblasNoTrans,CblasNoTrans);
                EnvR.push_back( product<3,3,2>(s1t,lm,CblasConjTrans,CblasNoTrans) );
            }
        }

        /* update env-3 tensor pair */
        void update_env_op ( const Tensor<3> & M, const Tensor<4> & W, const Tensor<3> & La,
        vector<Tensor<3>> & EnvL, vector<Tensor<3>> & EnvR, bool l2r)
        {
            if(l2r)
            {
                EnvL.push_back( env_sweep(EnvL.back(),M,W,La,l2r));
                EnvR.pop_back();
            }
            else
            {
                EnvR.push_back( env_sweep(EnvR.back(),M,W,La,l2r));
                EnvL.pop_back();
            }
        }

        /* update env-2 tensor pair */
        void update_env_state ( const Tensor<3> & M,  const Tensor<3> & La,
        vector<Tensor<2>> & EnvL, vector<Tensor<2>> & EnvR, bool l2r)
        {
            if(l2r)
            {
                Tensor<3> lm = product<2,3,1>(EnvL.back(),La,CblasNoTrans,CblasNoTrans);
                EnvL.push_back( product<3,3,2>(M,lm,CblasConjTrans,CblasNoTrans)  );
                EnvR.pop_back();
            }
            else
            {
                Tensor<3> s1t = transpose<3>(M,{2,1,0});
                Tensor<3> Lat = transpose<3>(La,{2,1,0});
                Tensor<3> lm = product<2,3,1>(EnvR.back(),Lat,CblasNoTrans,CblasNoTrans);
                EnvR.push_back( product<3,3,2>(s1t,lm,CblasConjTrans,CblasNoTrans) );
                EnvL.pop_back();
            }
        }
 
        Tensor<4> apply_double_state ( const Tensor<2> & envl, const Tensor<4> & M ,
        const Tensor<2> & envr )
        {
            Tensor<4> lm = product<2,4,1>(envl,M,CblasNoTrans,CblasNoTrans);
            Tensor<4> res = product<4,2,1>(lm,envr,CblasNoTrans,CblasTrans);
            return res;
        }

        State apply_ortho ( const Operator & H, const State & Va, SVDConfig & rs)
        {
            State gss = apply_operator(H,Va,rs.SVDThres,rs.MaxDim);
            Complex ovp = overlap(gss,Va);
            State Vac(Va);
            ovp *= -1.0;
            gss += Vac*ovp;
            gss.canonicalize();
            return gss;
        }

        /* f(H,v[j],v_[j-1]) -> v[j+1] */
        State apply_ortho ( const Operator & H, const State & Va, const State & Vb, SVDConfig & rs)
        {
            vector<Tensor<3>> EnvLH,EnvRH;
            vector<Tensor<2>> EnvLa,EnvRa,EnvLb,EnvRb;

            State gss = apply_operator(H,Va,rs.SVDThres,rs.MaxDim);
            
            init_env_op(H,gss,Va,EnvLH,EnvRH);
            init_env_state(gss,Va,EnvLa,EnvRa);
            init_env_state(gss,Vb,EnvLb,EnvRb);
 
            INT ns = H.nsite();
            rs.Left2Right =  true;
            for(INT i(0);i<ns-1;++i)
            {
                Tensor<4> Ta = product<3,3,1>(Va.at(i),Va.at(i+1),CblasNoTrans,CblasNoTrans); 
                Tensor<4> Tb = product<3,3,1>(Vb.at(i),Vb.at(i+1),CblasNoTrans,CblasNoTrans);

                Tensor<4> Nh = apply_double(EnvLH.back(),Ta,H.at(i),H.at(i+1),EnvRH.back());
                Tensor<4> La = apply_double_state(EnvLa.back(),Ta,EnvRa.back());
                Tensor<4> Lb = apply_double_state(EnvLb.back(),Tb,EnvRb.back());

                Tensor<4> Mgs = apply_ortho_site(Nh,La,Lb);
                auto[Lef,Rig] = svd_impl<2,2>(Mgs,rs,-1);
    
                gss[i] = move(Lef);
                gss[i+1] = move(Rig);
                if(i<ns-2)
                {
                    update_env_op(gss.at(i),H.at(i),Va.at(i),EnvLH,EnvRH,rs.Left2Right);
                    update_env_state(gss.at(i),Va.at(i),EnvLa,EnvRa,rs.Left2Right);
                    update_env_state(gss.at(i),Vb.at(i),EnvLb,EnvRb,rs.Left2Right);
                }
            }
            rs.Left2Right =  false;
            for(INT i(ns-1);i>0;--i)
            {
                Tensor<4> Ta = product<3,3,1>(Va.at(i-1),Va.at(i),CblasNoTrans,CblasNoTrans); 
                Tensor<4> Tb = product<3,3,1>(Vb.at(i-1),Vb.at(i),CblasNoTrans,CblasNoTrans);

                Tensor<4> Nh = apply_double(EnvLH.back(),Ta,H.at(i-1),H.at(i),EnvRH.back());
                Tensor<4> La = apply_double_state(EnvLa.back(),Ta,EnvRa.back());
                Tensor<4> Lb = apply_double_state(EnvLb.back(),Tb,EnvRb.back());

                Tensor<4> Mgs = apply_ortho_site(Nh,La,Lb);
                auto[Lef,Rig] = svd_impl<2,2>(Mgs,rs,-1);
                
                gss[i-1] = move(Lef);
                gss[i] = move(Rig);
                if(i>1)
                {
                    update_env_op(gss.at(i),H.at(i),Va.at(i),EnvLH,EnvRH,rs.Left2Right);
                    update_env_state(gss.at(i),Va.at(i),EnvLa,EnvRa,rs.Left2Right);
                    update_env_state(gss.at(i),Vb.at(i),EnvLb,EnvRb,rs.Left2Right);
                }
            }
            return gss; 
        }

        /* main function for global krylov  */
        KrylovExit global_main ( const Operator & H, State & mps, KrylovConfig & r, SVDConfig & rs
        ,Timer & tm )
        {
            double nrms = norm(mps);
            mps *= 1/nrms;
            vector<State> Vm = {mps};
            
            INT k = 0;           // current iter
            double err = 1.0;    // current error

            KrylovExit exitCfg;

            start_module(tm,"Krylov-Iter");
            while (k<r.MaxIter-1 && err > r.KrylovThres)
            {
                State hv;
                if(k>0) 
                {
                    hv = apply_ortho(H,Vm[k],Vm[k-1],rs);
                }
                else
                {
                    hv = apply_ortho(H,Vm[k],rs);
                }
               
                double nmhv = norm(hv);
                err *= nmhv;
                cout << "Circle : " << k << " Err : " << std::scientific << setprecision(6) << err << endl;
                if ( err < r.KrylovThres || nmhv > 1.0)
                {
                    exitCfg.KrylovErr = err;
                    exitCfg.RealIter  = k+1;
                    exitCfg.reason    = KrylovExitReason::AchKryThr;
                    break;
                }
                if ( k == r.MaxIter -1 )
                {
                    exitCfg.KrylovErr = err;
                    exitCfg.RealIter  = k+1;
                    exitCfg.reason    = KrylovExitReason::AchMaxItr;
                    break;
                }
 
                hv *= 1/nmhv;
                Vm.push_back(hv);
                ++k;
            }
            end_module(tm,"Krylov-Iter");

            INT ldhv = Vm.size();               // size of effect Hamiltonian
            Complex img(0,-1);
            Alloc<Complex> Hm(ldhv*ldhv);
            Alloc<Complex> UeiTU(ldhv);
            Alloc<double> ega(ldhv);

            start_module(tm,"Krylov-Diag");
            for(INT i(0);i<ldhv;++i)
            {
                Hm.ptr()[i*ldhv+i] = H.join(Vm[i],Vm[i]);
                if(i<ldhv-1) { Hm.ptr()[i*ldhv+i+1] = H.join(Vm[i],Vm[i+1]); }
            }

            naive_eig(Hm,ldhv,ega);              // eigen-decomp

            for(INT i(0);i<ldhv;++i)
            {
                for(INT j(0);j<ldhv;++j)
                {
                    UeiTU.ptr()[i] +=  Hm.ptr()[i*ldhv+j] * exp(img*ega.ptr()[j]) * conj(Hm.ptr()[j]);
                }
            }
            mps = Vm[0] * UeiTU.ptr()[0];
            for(INT i(1);i<ldhv;++i)
            { mps += Vm[i] * UeiTU.ptr()[i]; mps.canonicalize(rs.SVDThres,rs.MaxDim); }
            exitCfg.LossNorm = 1.0 - norm(mps);
            end_module(tm,"Krylov-Diag");
            return exitCfg;
        }
    }
}

#endif
