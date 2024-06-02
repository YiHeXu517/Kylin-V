/* Matrix and matrix product exponential calculation */

#ifndef exponential_H
#define exponential_H

#include "product.h"

namespace klnX
{
    namespace Holstein
    {
 
        using std::exp;
        using std::conj;     
 
        /* exit reason */
        enum class KrylovExitReason
        {
            /* achieve the krylov thres */
            AchKryThr,
      
            /* achieve the maximal iteraction */
            AchMaxItr,
        };
 
        ostream & operator<<(ostream & os, KrylovExitReason r)
        {
            switch(r)
            {
            case KrylovExitReason::AchKryThr :
            os << "Krylov error converges." << endl;
            break;
            case KrylovExitReason::AchMaxItr :
            os << "Maximal iteration reached." << endl;
            break;
            default:
            os << "Unknown reason" << endl;
            break;
            }
            return os;
        } 
 
        /* krylov input configuration */
        struct KrylovConfig
        {
            /* threshold of Krylov iteration error */
            double KrylovThres = 1.0e-9;

            /* maximal iteration numbers  */
            INT MaxIter = 15;
            
            /* time step */
            double TimeStep;
        };

        ostream & operator<<(ostream & os, KrylovConfig & r)
        {
            os << "Krylov iteration threshold = " << r.KrylovThres << endl;
            os << "Krylov maximal iteration   = " << r.MaxIter << endl;
            os << "Time step                  = " << r.TimeStep << endl;
            return os;
        };
        /* krylov exit configration */
        struct KrylovExit
        {
            /* exit reason */
            KrylovExitReason reason;
 
            /* krylov error */
            double KrylovErr;

            /* iterations */
            INT RealIter;

            /* loss of normalisation */
            double LossNorm;
        };

        ostream & operator<<(ostream & os, KrylovExit & k)
        {
            if(k.reason==KrylovExitReason::AchKryThr) 
            { os << "Krylov iteration converges." << endl; }
            else
            { os << "Limited iteration achieved." << endl; }
            os << "Evolutionary error: " << std::scientific << setprecision(10) << k.KrylovErr << endl;
            os << "Iteractions: " << k.RealIter << endl;
            os << "Discard norm: " << k.LossNorm << endl;
            return os;
        }
        
        /* local site krylov iteration */
        template<INT Rank>
        KrylovExit local_krylov( const Tensor<3> & envl, Tensor<Rank> & mps, 
        const Tensor<4> & mpo1, const Tensor<4> & mpo2, const Tensor<3> & envr, 
        KrylovConfig & r, bool ReverseEvl = false )
        {
            double nrms = norm(mps);
            mps *= 1/nrms;
            vector<Tensor<Rank>> Vm = {mps};
            vector<Complex> HAlp,HBetUpper, HBetLower;
            
            INT k = 0;           // current iter
            double err = 1.0;    // current error

            KrylovExit exitCfg;

            // int erg; cout << "Bef iter " << Rank << "  Free:  " << sif.freeram*1.0 / 1024/1024/1024 << " GB" << endl;

            while (k<r.MaxIter-1 && err > r.KrylovThres)
            {
                Tensor<Rank> hv;

                if constexpr (Rank==2)
                {
                    hv = apply_zero(envl,Vm[k],envr);
                }
                else if constexpr (Rank==3)
                {
                    hv = apply_single(envl,Vm[k],mpo1,envr);
                }
                else
                {
                    hv = apply_double(envl,Vm[k],mpo1,mpo2,envr);
                }
                hv *= r.TimeStep*0.5;
                if(k>0) 
                {
                    Complex betk = overlap(Vm[k-1],hv);
                    HBetUpper.push_back(betk);
                    betk *= -1.0;
                    hv += Vm[k-1] * betk;
               }

                Complex alpk = overlap(Vm[k],hv);
                HAlp.push_back(alpk);
                alpk *= -1.0;
                hv += Vm[k] * alpk;

                double nmhv = norm(hv);
                err *= nmhv;
                if ( err < r.KrylovThres )
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
                HBetLower.push_back(nmhv);
                Vm.push_back(hv);
                ++k;
            }


            INT ldhv = HAlp.size();               // size of effect Hamiltonian
            Complex img(0,-1);
            if(ReverseEvl) { img *= -1.0; }
            Alloc<Complex> Hm(ldhv*ldhv);
            Alloc<Complex> UeiTU(ldhv);
            Alloc<double> ega(ldhv);
            for(INT i(0);i<ldhv;++i)
            {
                Hm.ptr()[i*ldhv+i] = HAlp[i];
                if(i<ldhv-1) { Hm.ptr()[i*ldhv+i+1] = HBetUpper[i]; }
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
            { mps += Vm[i] * UeiTU.ptr()[i]; }
            exitCfg.LossNorm = 1.0 - norm(mps);
            
            return exitCfg;
        }
    }
}

#endif
