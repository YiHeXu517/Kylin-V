/* watson hamiltonian  */

#pragma once

#include "local.h"

namespace KylinVib
{
    namespace WatsonDMRG
    {
        class Lattice
        {
        private:
            // maximal level , number of sites and integrals
            INT ndof_ = 0;
            INT nexc_ = 0;
            std::vector<std::vector<LocalOp>> INTLab_;
            std::vector<double> INTVal_;
        public:
            // tors
            Lattice() = default;
            ~Lattice() = default;

            // reads
            INT ndof() const
            {
            return ndof_;
            }
            INT nexc() const
            {
            return nexc_;
            }
            INT num_term() const
            {
            return INTLab_.size();
            }
            std::vector<LocalOp> const & lab_at(INT idx) const
            {
            return INTLab_[idx];
            }
            double val_at(INT idx) const
            {
            return INTVal_[idx];
            }

            // orders of force field ff: 4->QFF 6->SFF
            Lattice(INT nmod, INT maxl, const char * fdump, INT ff = 6)
            : ndof_(nmod), nexc_(maxl)
            {
                INT KTerm = 0;
                std::ifstream ifs(fdump);
                if(!ifs.is_open()) { std::cout << "Fail to open dump!" << std::endl; std::exit(1);}
                while(!ifs.eof())
                {
                    std::vector<LocalOp> labs(ndof_); double val;
                    std::for_each(labs.begin(),labs.end(),[](LocalOp & x){ x = LocalOp::I;});
                    INT LapTmp = 0;
                    for(INT i(0);i<ff;++i)
                    {
                        ifs >> LapTmp;
                        if(LapTmp>0) { labs[LapTmp-1] = apply_lop(labs[LapTmp-1],LocalOp::Q); }
                        else if(LapTmp<0) { labs[-1-LapTmp] = apply_lop(labs[-1-LapTmp],LocalOp::P);}
                    }
                    ifs >> val;
                    if(ifs.eof()) {break;}
                    INTLab_.push_back(labs);
                    INTVal_.push_back(val);
                    KTerm++;
                }
                ifs.close();
            }
            // generate total Hamiltonian
            xMPO gen_ctotal()
            {
                INT nTerm = INTVal_.size();
                xMPO ham(ndof_);
                for(INT i=0;i<nTerm;++i)
                {
                    xMPO tmpi(ndof_);
                    for(INT j=0;j<ndof_;++j)
                    {
                        tmpi[j] = to_matrix<ArrR<4>>(INTLab_[i][j],nexc_);
                    }
                    tmpi *= INTVal_[i];
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
        };
    }
}
