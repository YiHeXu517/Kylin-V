/* all kinds of hamiltonian defined here  */

#pragma once

#include <fstream>
#include <omp.h>
#include "dense_mpo.h"
#include "local.h"
#define Nmol 4
#define VibCut 16

namespace KylinVib
{
    namespace Polariton
    {
        class NaiveHTCLattice
        {
            private:
            std::vector<std::vector<LocalOp>> Lab_;
            std::vector<double> Val_;
            public:
            // tors
            NaiveHTCLattice() = default;
            ~NaiveHTCLattice() = default;

            // reads
            int num_term() const
            {
                return Lab_.size();
            }
            std::vector<LocalOp> const & lab_at(int idx) const
            {
                return Lab_[idx];
            }
            double val_at(int idx) const
            {
                return Val_[idx];
            }

            // position with cavity in center
            static void ref_position(int mol, char type, std::vector<int> & SiteDims, 
            std::map<int,int> & ExLabs, std::map<int,int> & VibLabs)
            {
                int res = 0;
                switch (type)
                {
                case 'E':
                    res = (mol>=Nmol/2) ? 2*mol+1 : 2*mol;
                    ExLabs[mol] = res;
                    SiteDims[res] = 2;
                    break;
                case 'V':
                    res = (mol>=Nmol/2) ? 2*mol+2 : 2*mol+1;
                    VibLabs[mol] = res;
                    SiteDims[res] = VibCut;
                    break;
                case 'C':
                    res = Nmol;
                    SiteDims[res] = VibCut;
                    break;
                default:
                    break;
                }
            }

            // orders of force field ff: 4->QFF 6->SFF
            NaiveHTCLattice(std::vector<int> & SiteDims, 
            std::map<int,int> & ExLabs, std::map<int,int> & VibLabs)
            {
                int nsite = 2 * Nmol + 1;
                for(int mol=0;mol<Nmol;++mol)
                {
                    std::vector<LocalOp> labs(nsite); double val;
                    std::for_each(labs.begin(),labs.end(),[](LocalOp & x){ x = LocalOp::I;});
                    labs[ExLabs[mol]] = LocalOp::N;
                    Lab_.push_back(labs);
                    Val_.push_back(2.0);
                }
                for(int mol=Nmol/2;mol<Nmol/2+1;++mol)
                {
                    std::vector<LocalOp> labs(nsite); double val;
                    std::for_each(labs.begin(),labs.end(),[](LocalOp & x){ x = LocalOp::I;});
                    labs[Nmol] = LocalOp::N;
                    Lab_.push_back(labs);
                    Val_.push_back(2.0);
                }
                for(int mol=0;mol<Nmol;++mol)
                {
                    std::vector<LocalOp> labs(nsite); double val;
                    std::for_each(labs.begin(),labs.end(),[](LocalOp & x){ x = LocalOp::I;});
                    labs[VibLabs[mol]] = LocalOp::N;
                    Lab_.push_back(labs);
                    Val_.push_back(0.15);
                }
                for(int mol=0;mol<Nmol;++mol)
                {
                    std::vector<LocalOp> labs(nsite); double val;
                    std::for_each(labs.begin(),labs.end(),[](LocalOp & x){ x = LocalOp::I;});
                    labs[ExLabs[mol]] = LocalOp::Upper;
                    labs[Nmol] = LocalOp::Lower;
                    Lab_.push_back(labs);
                    Val_.push_back(0.07);
                }
                for(int mol=0;mol<Nmol;++mol)
                {
                    std::vector<LocalOp> labs(nsite); double val;
                    std::for_each(labs.begin(),labs.end(),[](LocalOp & x){ x = LocalOp::I;});
                    labs[ExLabs[mol]] = LocalOp::Lower;
                    labs[Nmol] = LocalOp::Upper;
                    Lab_.push_back(labs);
                    Val_.push_back(0.07);
                }
                /*
                for(int mol=0;mol<Nmol;++mol)
                {
                    std::vector<LocalOp> labs(nsite); double val;
                    std::for_each(labs.begin(),labs.end(),[](LocalOp & x){ x = LocalOp::I;});
                    labs[ExLabs[mol]] = LocalOp::N;
                    labs[VibLabs[mol]] = LocalOp::Q;
                    Lab_.push_back(labs);
                    Val_.push_back(-0.15);
                }
                */
            }
            // generate total Hamiltonian
            MPO<double> gen_total(std::vector<int> & SiteDims)
            {
                int nTerm = Val_.size(), ndof_ = 2*Nmol+1;
                MPO<double> ham(ndof_);
                for(int i=0;i<nTerm;++i)
                {
                    MPO<double> tmpi(ndof_);
                    for(int j=0;j<ndof_;++j)
                    {
                        tmpi[j] = to_matrix<Dense<double,4>>(Lab_[i][j],SiteDims[j]);
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
            MPO<double> gen_total_para(std::vector<int> & SiteDims)
            {
                int nTerm = Val_.size(), nth, ndof_ = 2*Nmol+1;
                #pragma omp parallel
                nth = omp_get_num_threads();

                std::vector<MPO<double>> hams(nth,ndof_);
                #pragma omp parallel for
                for(int i=0;i<nTerm;++i)
                {
                    MPO<double> tmpi(ndof_);
                    for(int j=0;j<ndof_;++j)
                    {
                        tmpi[j] = to_matrix<Dense<double,4>>(Lab_[i][j],SiteDims[j]);
                    }
                    tmpi *= Val_[i];
                    tmpi.print();
                    int ThreadID = omp_get_thread_num();
                    if(hams[ThreadID][0].shape()[1]!=tmpi[0].shape()[1]) { hams[ThreadID] = tmpi; }
                    else { hams[ThreadID] += tmpi; }
                    hams[ThreadID].canon();
                }
                for(int i=1;i<nth;++i)
                {
                    hams[0] += hams[i];
                    hams[0].canon();
                }
                return hams[0];
            }
            MPO<MKL_Complex16> gen_ctotal_para(std::vector<int> & SiteDims)
            {
                int nTerm = Val_.size(), nth, ndof_ = 2*Nmol+1;
                #pragma omp parallel
                nth = omp_get_num_threads();

                std::vector<MPO<MKL_Complex16>> hams(nth,ndof_);
                #pragma omp parallel for
                for(int i=0;i<nTerm;++i)
                {
                    MPO<MKL_Complex16> tmpi(ndof_);
                    for(int j=0;j<ndof_;++j)
                    {
                        tmpi[j] = to_matrix<Dense<MKL_Complex16,4>>(Lab_[i][j],SiteDims[j]);
                    }
                    tmpi *= Val_[i];
                    int ThreadID = omp_get_thread_num();
                    if(hams[ThreadID][0].shape()[1]!=tmpi[0].shape()[1]) { hams[ThreadID] = tmpi; }
                    else { hams[ThreadID] += tmpi; }
                    hams[ThreadID].canon();
                }
                for(int i=1;i<nth;++i)
                {
                    hams[0] += hams[i];
                    hams[0].canon();
                }
                return hams[0];
            }
 
            MPO<double> gen_eye(std::vector<int> & SiteDims)
            {
                int ndof_ = 2*Nmol+1;
                MPO<double> ham(ndof_);
                std::vector<LocalOp> labs(ndof_);
                std::for_each(labs.begin(),labs.end(),[](LocalOp & x){ x = LocalOp::I;});
                for(int j=0;j<ndof_;++j)
                {
                    ham[j] = to_matrix<Dense<double,4>>(labs[j],SiteDims[j]);
                }
                return ham;
            }
            MPO<MKL_Complex16> gen_upper(std::vector<int> & SiteDims,int site)
            {
                int ndof_ = 2*Nmol+1;
                MPO<MKL_Complex16> ham(ndof_);
                std::vector<LocalOp> labs(ndof_);
                std::for_each(labs.begin(),labs.end(),[](LocalOp & x){ x = LocalOp::I;});
                labs[site] = LocalOp::Upper;
                for(int j=0;j<ndof_;++j)
                {
                    ham[j] = to_matrix<Dense<MKL_Complex16,4>>(labs[j],SiteDims[j]);
                }
                return ham;
            }
            MPO<MKL_Complex16> gen_lower(std::vector<int> & SiteDims,int site)
            {
                int ndof_ = 2*Nmol+1;
                MPO<MKL_Complex16> ham(ndof_);
                std::vector<LocalOp> labs(ndof_);
                std::for_each(labs.begin(),labs.end(),[](LocalOp & x){ x = LocalOp::I;});
                labs[site] = LocalOp::Lower;
                for(int j=0;j<ndof_;++j)
                {
                    ham[j] = to_matrix<Dense<MKL_Complex16,4>>(labs[j],SiteDims[j]);
                }
                return ham;
            }
        };
    }
}
