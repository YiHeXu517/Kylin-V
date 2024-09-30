/* all kinds of hamiltonian defined here  */

#pragma once

#include <fstream>
#include <omp.h>
#include "dense_mpo.h"
#include "local.h"

namespace KylinVib
{
    namespace Vibronic
    {
        class Lattice
        {
            private:
            size_t NumEle_;
            size_t NumVib_;
            Dense<double,2> HamEle_;
            Dense<double,1> HamVib_;
            Dense<double,3> HamLVC_;
            Dense<double,4> HamQVC_;
            std::vector<int> SiteToLab_;
            std::vector<size_t> SiteOccs_;
            std::map<int,size_t> LabToSite_;

            public:
            Lattice()  = default;
            ~Lattice() = default;

            Lattice(const char * filename);
        };
    }
    // spin-1/2
    namespace Heisenberg2
    {
        class Lattice
        {
            private:
            size_t L_;
            std::vector<MPO<double>> ops_;

            public:
            Lattice(size_t len) : L_(len), ops_(3*L_-3)
            {
            
            }
            ~Lattice() = default;

            MPO<double> gen_total()
            {
                MPO<double> res(L_);
                for(size_t j=0;j<L_-1;++j)
                {
                    MPO<double> opp(L_),opm(L_),opz(L_);
                    for(size_t k=0;k<L_;++k)
                    {
                        if(k==j) 
                        { 
                            opp[k] = to_matrix(LocalOp::Sp);
                            opp[k+1] = to_matrix(LocalOp::Sm);
                            opm[k] = to_matrix(LocalOp::Sm);
                            opm[k+1] = to_matrix(LocalOp::Sp);
                            opz[k] = to_matrix(LocalOp::Sz);
                            opz[k+1] = to_matrix(LocalOp::Sz);
                            opp[k] *= 0.5;
                            opm[k] *= 0.5;
                        }
                        else if(k!=j && k!=j+1)
                        {
                            opp[k] = to_matrix(LocalOp::I);
                            opm[k] = to_matrix(LocalOp::I);
                            opz[k] = to_matrix(LocalOp::I);
                        }
                    }
                    if(j==0) { res = opp; }
                    else { res += opp;  }
                    res += opm;
                    res += opz;
                    res.canon();
                }
                return res;
            }
        };
    }
    namespace Watson
    {
        class Lattice
        {
            private:
            // maximal level , number of sites and integrals
            size_t ndof_ = 0;
            size_t nexc_ = 0;
            std::vector<std::vector<LocalOp>> Lab_;
            std::vector<double> Val_;
            public:
            // tors
            Lattice() = default;
            ~Lattice() = default;

            // reads
            size_t ndof() const
            {
            return ndof_;
            }
            size_t nexc() const
            {
            return nexc_;
            }
            size_t num_term() const
            {
            return Lab_.size();
            }
            std::vector<LocalOp> const & lab_at(size_t idx) const
            {
            return Lab_[idx];
            }
            double val_at(size_t idx) const
            {
            return Val_[idx];
            }

            // orders of force field ff: 4->QFF 6->SFF
            Lattice(size_t nmod, size_t maxl, const char * fdump, size_t ff = 6)
            : ndof_(nmod), nexc_(maxl)
            {
                size_t KTerm = 0;
                std::ifstream ifs(fdump);
                if(!ifs.is_open()) { std::cout << "Fail to open dump!" << std::endl; std::exit(1);}
                while(!ifs.eof())
                {
                    std::vector<LocalOp> labs(ndof_); double val;
                    std::for_each(labs.begin(),labs.end(),[](LocalOp & x){ x = LocalOp::I;});
                    int LapTmp = 0;
                    for(size_t i(0);i<ff;++i)
                    {
                        ifs >> LapTmp;
                        if(LapTmp>0) { labs[LapTmp-1] = apply_lop(labs[LapTmp-1],LocalOp::Q); }
                        else if(LapTmp<0) { labs[-1-LapTmp] = apply_lop(labs[-1-LapTmp],LocalOp::P);}
                    }
                    ifs >> val;
                    if(ifs.eof()) {break;}
                    Lab_.push_back(labs);
                    Val_.push_back(val);
                    KTerm++;
                }
                ifs.close();
            }
            // generate total Hamiltonian
            MPO<double> gen_total()
            {
                size_t nTerm = Val_.size();
                MPO<double> ham(ndof_);
                for(size_t i=0;i<nTerm;++i)
                {
                    MPO<double> tmpi(ndof_);
                    for(size_t j=0;j<ndof_;++j)
                    {
                        tmpi[j] = to_matrix<Dense<double,4>>(Lab_[i][j],nexc_);
                    }
                    tmpi *= Val_[i];
                    if(i==0) { ham = tmpi; }
                    else { ham += tmpi; }
                    std::cout << "Contract term " << i+1 << " / " << nTerm << std::endl;
                    if((i/50)!=(i+1)/50)
                    {
                        ham.canon();
                    }
                }
                return ham;
            }
            // generate total Hamiltonian parallel version
            MPO<double> gen_total_para()
            {
                size_t nTerm = Val_.size(), nth;
                #pragma omp parallel
                nth = omp_get_num_threads();

                std::vector<MPO<double>> hams(nth,ndof_);
                #pragma omp parallel for
                for(size_t i=0;i<nTerm;++i)
                {
                    MPO<double> tmpi(ndof_);
                    for(size_t j=0;j<ndof_;++j)
                    {
                        tmpi[j] = to_matrix<Dense<double,4>>(Lab_[i][j],nexc_);
                    }
                    tmpi *= Val_[i];
                    size_t ThreadID = omp_get_thread_num();
                    if(hams[ThreadID][0].shape()[1]!=tmpi[0].shape()[1]) { hams[ThreadID] = tmpi; }
                    else { hams[ThreadID] += tmpi; }
                    //std::cout << "Contract term " << i+1 << " / " << nTerm << std::endl;
                    hams[ThreadID].canon();
                }
                for(size_t i=1;i<nth;++i)
                {
                    hams[0] += hams[i];
                    hams[0].canon();
                }
                return hams[0];
            }
            MPO<MKL_Complex16> gen_ctotal_para()
            {
                size_t nTerm = Val_.size(), nth;
                #pragma omp parallel
                nth = omp_get_num_threads();

                std::vector<MPO<MKL_Complex16>> hams(nth,ndof_);
                #pragma omp parallel for
                for(size_t i=0;i<nTerm;++i)
                {
                    MPO<MKL_Complex16> tmpi(ndof_);
                    for(size_t j=0;j<ndof_;++j)
                    {
                        tmpi[j] = to_matrix<Dense<MKL_Complex16,4>>(Lab_[i][j],nexc_);
                    }
                    tmpi *= Val_[i];
                    size_t ThreadID = omp_get_thread_num();
                    if(hams[ThreadID][0].shape()[1]!=tmpi[0].shape()[1]) { hams[ThreadID] = tmpi; }
                    else { hams[ThreadID] += tmpi; }
                    //std::cout << "Contract term " << i+1 << " / " << nTerm << std::endl;
                    hams[ThreadID].canon();
                }
                for(size_t i=1;i<nth;++i)
                {
                    hams[0] += hams[i];
                    hams[0].canon();
                }
                return hams[0];
            }
 
            MPO<double> gen_eye()
            {
                MPO<double> ham(ndof_);
                std::vector<LocalOp> labs(ndof_);
                std::for_each(labs.begin(),labs.end(),[](LocalOp & x){ x = LocalOp::I;});
                for(size_t j=0;j<ndof_;++j)
                {
                    ham[j] = to_matrix<Dense<double,4>>(labs[j],nexc_);
                }
                return ham;
            }
        };
    }
}
