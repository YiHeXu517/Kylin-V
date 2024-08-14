/* DMRG implementation for watson models */

#pragma once

#include "state.h"

namespace KylinVib
{
    using std::ifstream;
    namespace Watson
    {
        class DMRG
        {
            public:

            DMRG() = default;

            /// read hamiltonian terms from input file
            DMRG(char const * fname, size_t nmode, size_t nocc)
            : State_(nmode)
            {
                ifstream ifs(fname);
                vector<SparseBase<double,4>> Tmps(nmode,{1,nocc,nocc,1});
                size_t TermIdx = 0;
                if(!ifs.is_open())
                {
                    cout << "Fail to open input file." << endl;
                    exit(1);
                }
                while(!ifs.eof())
                {
                    vector<LocalOp> Term(nmode,LocalOp::I);
                    int tmps;
                    double tmpv;
                    for(size_t pos=0;pos<6;++pos)
                    {
                        ifs >> tmps;
                        if(tmps>0)
                        {
                            Term[tmps-1] = apply_lop(Term[tmps-1],LocalOp::Q);
                        }
                        else if(tmps<0)
                        {
                            Term[-1*tmps-1] = apply_lop(Term[-1*tmps-1],LocalOp::P);
                        }
                    }
                    ifs >> tmpv;
                    if(TermIdx!=0)
                    {
                        Tmps[0] = stack(Tmps[0], to_matrix(Term[0],nocc)*tmpv, 'r' );
                        Tmps[nmode-1] = stack(Tmps[nmode-1], to_matrix(Term[nmode-1],nocc), 'l' );
                        for(size_t site=1;site<nmode-1;++site)
                        {
                            Tmps[site] = stack(Tmps[site], to_matrix(Term[site],nocc), 'm' );
                        }
                    }
                    else
                    {
                        Tmps[0] = to_matrix(Term[0],nocc)*tmpv;
                        for(size_t site=1;site<nmode;++site)
                        {
                            Tmps[site] = to_matrix(Term[site],nocc);
                        }
                    }
                    TermIdx++;
                    if(ifs.eof()) { break; }
                }
                ifs.close();
                Ham_ = move(Tmps);
                for(size_t i=0;i<nmode;++i)
                {
                    State_.set_shapes(i)[0] = 1;
                    State_.set_shapes(i)[1] = 1;
                    State_.set_shapes(i)[2] = 1;
                    DenseBase<double> ri(1); ri.ptr()[0] = 1.0;
                    State_[i] = move(ri);
                }
                State_.print();
            }

            ~DMRG() = default;

            void compress()
            {
                for(size_t site=0;site<Ham_.size()-1;++site)
                {
                    auto[cur,nex] = deparallel(Ham_[site],'r');
                    Ham_[site] = move(cur);
                    Ham_[site+1] = contract(nex,Ham_[site+1],'r');
                }
                for(size_t site=Ham_.size()-1;site>0;--site)
                {
                    auto[cur,nex] = deparallel(Ham_[site],'l');
                    Ham_[site] = move(cur);
                    Ham_[site-1] = contract(nex,Ham_[site-1],'l');
                }
            }

            void print_ham() const
            {
                for(size_t site=0;site<Ham_.size();++site)
                {
                    cout << "Site " << site+1 << ":" << endl;
                    Ham_[site].print();
                }
            }

            private:

            NaiveMPO Ham_;
            MPS State_;
        };
    }
}
