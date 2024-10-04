/* iCI implementation for watson models */

#pragma once

#include <fstream>
#include <tuple>
#include <vector>
#include <complex>
#include <unordered_set>
#include <unordered_map>
#include <omp.h>
#include <cmath>
#include <bitset>
#include "timer.h"

namespace KylinVib
{
    using std::ifstream;
    using std::ostream;
    using std::vector;
    using std::hash;
    using std::unordered_set;
    using std::unordered_map;
    using std::swap;
    using std::tuple;
    using std::make_tuple;
    using std::get;
    using std::max;
    using std::clock_t;
    using std::clock;
    using std::bitset;
    namespace Heisenberg
    {
        /// apply a S+/S-/Sz operator
        template<size_t Ns>
        void apply_PQ(bitset<Ns> const & r, double rcf, size_t site, char mode,
        unordered_map<bitset<Ns>,double> & groups)
        {
            bitset<Ns> rup(r);
            double rupcf(rcf);
            if(mode=='+' && (!rup[site]) )
            {
                rup[site] = true;
                if(groups.find(rup)==groups.end())
                {
                    groups[rup] = rupcf;
                }
                else
                {
                    groups[rup] += rupcf;
                }
            }
            else if(mode=='-' && (rup[site]))
            {
                rup[site] = false;
                if(groups.find(rup)==groups.end())
                {
                    groups[rup] = rupcf;
                }
                else
                {
                    groups[rup] += rupcf;
                }
            }
            else if(mode=='z')
            {
                if(rup[site]) { groups[rup] += rupcf*0.5; } 
                else { groups[rup] -= rupcf*0.5; }
            }
        }

        template<size_t Ns>
        class iCI
        {
            public:
            /// read hamiltonian terms from input file
            iCI()
            {
                for(size_t i=0;i<Ns-1;++i)
                {
                    vector<size_t> SpmLab = {i,i+1}, SmpLab = {i,i+1}, SzLab = {i,i+1};
                    vector<char> SpmOp = {'+','-'}, SmpOp = {'-','+'}, SzOp = {'z','z'};
                    labels_.push_back(SpmLab);
                    ops_.push_back(SpmOp);
                    coeff_.push_back(0.5);

                    labels_.push_back(SmpLab);
                    ops_.push_back(SmpOp);
                    coeff_.push_back(0.5);

                    labels_.push_back(SzLab);
                    ops_.push_back(SzOp);
                    coeff_.push_back(1.0);
                }
            }

            ~iCI() = default;

            /// print hamiltonian terms
            void print_ham() const
            {
                size_t ntm = labels_.size();
                cout << Ns << "-site spin-1/2 chain:" << endl;
                for(size_t i=0;i<ntm;++i)
                {
                    cout << "[" << labels_[i][0] << ops_[i][0] << ","
                    << labels_[i][1] << ops_[i][1] << "] | " << coeff_[i] << endl;
                }
            }

            /// print saved basis
            void print_basis() const
            {
                size_t ntm = OrdRef_.size();
                cout << ntm << " basis saved:" << endl;
                for(size_t i=0;i<ntm;++i)
                {
                    cout << OrdRef_[i] << endl;
                }
            }

            /// print triplet entries
            void print_trip() const
            {
                size_t ntm = trips_.size();
                for(size_t i=0;i<ntm;++i)
                {
                    cout << get<0>(trips_[i]) << ',' << get<1>(trips_[i])
                    << " | " << get<2>(trips_[i]) << endl;
                }
            }

            /// add a new basis after checking
            void add_basis(bitset<Ns> const & r)
            {
                Ord_[r] = OrdRef_.size();
                OrdRef_.push_back(r);
            }

            /// generate total Hamiltonian to triplets
            void make_total()
            {
                size_t nref = OrdRef_.size(), ntm = labels_.size();
                #pragma omp parallel for
                for(size_t ref=0;ref<nref;++ref)
                {
                    // calculate
                    unordered_map<bitset<Ns>,double> Hr;
                    for(size_t tm=0;tm<ntm;++tm)
                    {
                        double coeff = coeff_[tm];
                        unordered_map<bitset<Ns>,double> tmr;
                        tmr[OrdRef_[ref]] = coeff;
                        for(size_t s12=0;s12<2;++s12)
                        {
                            unordered_map<bitset<Ns>,double> tmps;
                            for(const auto & [bas,cf] : tmr)
                            {
                                apply_PQ<Ns>(bas,cf,labels_[tm][s12],ops_[tm][s12],tmps);
                            }
                            swap(tmr,tmps);
                        }
                        for(const auto & [bas,cf] : tmr)
                        {
                            if(Hr.find(bas)==Hr.end())
                            {
                                Hr[bas] = cf;
                            }
                            else
                            {
                                Hr[bas] += cf;
                            }
                        }
                    }
                    #pragma omp critical
                    {
                        for(const auto & [bas,cf] : Hr)
                        {
                            if(abs(cf)>1e-8 && Ord_.find(bas)!=Ord_.end())
                            {
                                trips_.push_back(make_tuple(Ord_[bas],ref,cf));
                            }
                        }
                    }
                }
                cout << "Num. basis = " << nref << endl;
                cout << "Num. NZ    = " << trips_.size() << endl;
                cout << "Sparsity : " << std::scientific << setprecision(11)
                << (trips_.size() / nref) * 100.0 / nref
                << " %." << endl;
            }
            /// make total Hamiltonian with PHP already built
            void make_total_increment(size_t LastNref)
            {
                size_t nref = OrdRef_.size(), ntm = labels_.size();
                #pragma omp parallel for
                for(size_t ref=LastNref;ref<nref;++ref)
                {
                    // calculate
                    unordered_map<bitset<Ns>,double> Hr;
                    for(size_t tm=0;tm<ntm;++tm)
                    {
                        double coeff = coeff_[tm];
                        unordered_map<bitset<Ns>,double> tmr;
                        tmr[OrdRef_[ref]] = coeff;
                        for(size_t s12=0;s12<2;++s12)
                        {
                            unordered_map<bitset<Ns>,double> tmps;
                            for(const auto & [bas,cf] : tmr)
                            {
                                apply_PQ<Ns>(bas,cf,labels_[tm][s12],ops_[tm][s12],tmps);
                            }
                            swap(tmr,tmps);
                        }
                        for(const auto & [bas,cf] : tmr)
                        {
                            if(Hr.find(bas)==Hr.end())
                            {
                                Hr[bas] = cf;
                            }
                            else
                            {
                                Hr[bas] += cf;
                            }
                        }
                    }
                    // mapping
                    #pragma omp critical
                    {
                        for(const auto & [bas,cf] : Hr)
                        {
                            if(abs(cf)>1e-8 && Ord_.find(bas)!=Ord_.end())
                            {
                                trips_.push_back(make_tuple(Ord_[bas],ref,cf));
                                if(Ord_[bas]<LastNref)
                                {
                                    trips_.push_back(make_tuple(ref,Ord_[bas],cf));
                                }
                            }
                        }
                    }
                }
                cout << "Num. basis = " << nref << endl;
                cout << "Num. NZ    = " << trips_.size() << endl;
                cout << "Sparsity : " << std::scientific << setprecision(11)
                << (trips_.size() / nref) * 100.0 / nref
                << " %." << endl;
            }
            /// sparse eigen solver
            void eigen(size_t k0, DenseBase<double> & E, DenseBase<double> & Prim)
            {
                size_t nref = OrdRef_.size(), nnz = trips_.size();
                k0 = std::min(nref,k0);
                size_t k,info;
                DenseBase<size_t> rws(nnz),cms(nnz),vaci(1);
                DenseBase<double> vus(nnz),res(k0),vacd(1);
                E.initialize(k0);
                Prim.initialize(k0*nref);
                #pragma omp parallel for
                for(size_t em=0;em<nnz;++em)
                {
                    rws.ptr()[em] = get<0>(trips_[em])+1;
                    cms.ptr()[em] = get<1>(trips_[em])+1;
                    vus.ptr()[em] = get<2>(trips_[em]);
                    //cout << rws.ptr()[em] << "," << cms.ptr()[em] << "," << vus.ptr()[em] << endl;
                }
                char which = 'S';
                sparse_matrix_t A = NULL, B = NULL;
                struct matrix_descr descr;
                descr.type = SPARSE_MATRIX_TYPE_GENERAL;
                mkl_sparse_d_create_coo ( &A, SPARSE_INDEX_BASE_ONE, nref, nref, nnz,
                rws.ptr(), cms.ptr(),  vus.ptr());
                size_t pm[128];
                mkl_sparse_ee_init(pm);
                pm[1] = 8;
                pm[6] = 1;
                info = mkl_sparse_convert_csr ( A, SPARSE_OPERATION_NON_TRANSPOSE, &B);
                info = mkl_sparse_d_ev(&which, pm, B, descr, k0, &k, E.ptr(), Prim.ptr(), res.ptr());
                mkl_sparse_destroy(A);
                mkl_sparse_destroy(B);
                cout << "Rel. Energies:" << endl;
                for(size_t st=0;st<k0;++st)
                {
                    double val = (st==0) ? E.cptr()[st] : E.cptr()[st] - E.cptr()[0];
                    cout << "Eigval " << st+1 << " = " << val << endl;
                }
            }
            /// expand the external space basis according to PT2
            void expand(DenseBase<double> const & E, DenseBase<double> const & Prim)
            {
                size_t Np = E.size(), nref = OrdRef_.size(), ntm = labels_.size(), nth;
                #pragma omp parallel
                nth = omp_get_num_threads();

                // save the results of single process
                vector<unordered_set<bitset<Ns>>> TmpPots(nth);

                // calculate Hmn
                #pragma omp parallel for
                for(size_t ref=0;ref<nref;++ref)
                {
                    // build m from every n
                    unordered_map<bitset<Ns>,double> Hr;
                    for(size_t tm=0;tm<ntm;++tm)
                    {
                        double coeff = coeff_[tm];
                        unordered_map<bitset<Ns>,double> tmr;
                        tmr[OrdRef_[ref]] = coeff;
                        for(size_t s12=0;s12<2;++s12)
                        {
                            unordered_map<bitset<Ns>,double> tmps;
                            for(const auto & [bas,cf] : tmr)
                            {
                                apply_PQ<Ns>(bas,cf,labels_[tm][s12],ops_[tm][s12],tmps);
                            }
                            swap(tmr,tmps);
                        }
                        for(const auto & [bas,cf] : tmr)
                        {
                            if(Hr.find(bas)==Hr.end() && Ord_.find(bas)==Ord_.end())
                            {
                                Hr[bas] = cf;
                            }
                            else if(Hr.find(bas)!=Hr.end() && Ord_.find(bas)==Ord_.end())
                            {
                                Hr[bas] += cf;
                            }
                        }
                    }
                    // select
                    size_t MaxPrimEnt = cblas_idamax(Np,Prim.cptr()+ref,nref);
                    size_t ThreadID   = omp_get_thread_num();
                    for(const auto & [bas,cf] : Hr)
                    {
                        if(abs(cf*Prim.cptr()[MaxPrimEnt*nref+ref])>thres_ &&
                        Ord_.find(bas)==Ord_.end())
                        {
                            TmpPots[ThreadID].insert(bas);
                        }
                    }
                }
                // add basis
                for(size_t th=0;th<nth;++th)
                {
                    for(const auto & bas : TmpPots[th])
                    {
                        if(Ord_.find(bas)==Ord_.end())
                        {
                            add_basis(bas);
                        }
                    }
                }
            }

            void enpt2(DenseBase<double> const & E, DenseBase<double> const & Prim, double eps = 0.02)
            {
                size_t Np = E.size(), nref = OrdRef_.size(), ncol = Prim.size() / Np,
                ntm = labels_.size(), nth;
                DenseBase<double> Prim2(Np*nref);
                for(size_t p=0;p<Np;++p)
                {
                    cblas_dcopy(ncol,Prim.cptr()+p*ncol,1,Prim2.ptr()+p*nref,1);
                }
                #pragma omp parallel
                nth = omp_get_num_threads();

                vector<bitset<Ns>> PTVec;
                unordered_map<bitset<Ns>,size_t> PTOrds,VacPTs;

                #pragma omp parallel for
                for(size_t ref=0;ref<nref;++ref)
                {
                    unordered_map<bitset<Ns>,double> Hr;
                    for(size_t tm=0;tm<ntm;++tm)
                    {
                        double coeff = coeff_[tm];
                        unordered_map<bitset<Ns>,double> tmr;
                        tmr[OrdRef_[ref]] = coeff;
                        for(size_t s12=0;s12<2;++s12)
                        {
                            unordered_map<bitset<Ns>,double> tmps;
                            for(const auto & [bas,cf] : tmr)
                            {
                                apply_PQ<Ns>(bas,cf,labels_[tm][s12],ops_[tm][s12],tmps);
                            }
                            swap(tmr,tmps);
                        }
                        for(const auto & [bas,cf] : tmr)
                        {
                            if(Hr.find(bas)==Hr.end() && Ord_.find(bas)==Ord_.end())
                            {
                                Hr[bas] = cf;
                            }
                            else if(Hr.find(bas)!=Hr.end() && Ord_.find(bas)==Ord_.end())
                            {
                                Hr[bas] += cf;
                            }
                        }
                    }

                    size_t MaxPrimEnt = cblas_idamax(Np,Prim2.cptr()+ref,nref);
                    size_t ThreadID = omp_get_thread_num();
                    #pragma omp critical
                    {
                        for(const auto & [bas,cf] : Hr)
                        {
                            if(abs(cf*Prim2.cptr()[MaxPrimEnt*nref+ref])>eps &&
                            Ord_.find(bas)==Ord_.end())
                            {
                                if(PTOrds.find(bas)==PTOrds.end())
                                {
                                    PTOrds[bas] = PTVec.size();
                                    PTVec.push_back(bas);
                                }
                            }
                        }
                    }
                }
                PTOrds = VacPTs;
                DenseBase<double> DeltaE(Np*nth);
                size_t npt2 = PTVec.size();
                cout << npt2 << " ENPT2 basis." << endl;

                /// correction
                #pragma omp parallel for
                for(size_t m=0;m<npt2;++m)
                {
                    unordered_map<bitset<Ns>,double> Hr;
                    double Hmm = 0.0;
                    for(size_t tm=0;tm<ntm;++tm)
                    {
                        double coeff = coeff_[tm];
                        unordered_map<bitset<Ns>,double> tmr;
                        tmr[PTVec[m]] = coeff;
                        for(size_t s12=0;s12<2;++s12)
                        {
                            unordered_map<bitset<Ns>,double> tmps;
                            for(const auto & [bas,cf] : tmr)
                            {
                                apply_PQ<Ns>(bas,cf,labels_[tm][s12],ops_[tm][s12],tmps);
                            }
                            swap(tmr,tmps);
                        }
                        for(const auto & [bas,cf] : tmr)
                        {
                            if(Hr.find(bas)==Hr.end() && Ord_.find(bas)!=Ord_.end())
                            {
                                Hr[bas] = cf;
                            }
                            else if(Hr.find(bas)!=Hr.end() && Ord_.find(bas)!=Ord_.end())
                            {
                                Hr[bas] += cf;
                            }
                            if(bas==PTVec[m])
                            {
                                Hmm += cf;
                            }
                        }
                    }
                    for(size_t p=0;p<Np;++p)
                    {
                        double Hmn = 0.0;
                        for(const auto & [bas,cf] : Hr)
                        {
                            if(Ord_[bas]>=nref) {continue;}
                            Hmn += cf * Prim2.cptr()[p*nref+Ord_[bas]];
                        }
                        size_t ThreadID = omp_get_thread_num();
                        DeltaE.ptr()[p*nth+ThreadID] += pow(Hmn,2.0) / (E.cptr()[p] - Hmm);
                    }
                }
                DenseBase<double> DeltaE2(Np);
                for(size_t st=0;st<Np;++st)
                {
                    for(size_t th=0;th<nth;++th)
                    {
                        DeltaE2.ptr()[st] += DeltaE.cptr()[st*nth+th];
                    }
                }
 
                cout << "ENPT2 correction:" << endl;
                for(size_t st=0;st<Np;++st)
                {
                    double val = (st==0) ? E.cptr()[st] + DeltaE2.cptr()[st] : E.cptr()[st] + DeltaE2.cptr()[st]
                    - E.cptr()[0] - DeltaE2.cptr()[0];
                    cout << "Eigval " << st+1 << " = " << val << endl;
                }
            }

            size_t num_ref() const
            {
                return OrdRef_.size();
            }

            bitset<Ns> const & get_ref_basis(size_t idx) const
            {
                return OrdRef_[idx];
            }

            void set_tol(double tol)
            {
                thres_ = tol;
            }

            private:

            /// label-coefficients form Hamiltonian
            vector<vector<size_t>> labels_;
            vector<vector<char>> ops_;
            vector<double> coeff_;

            /// saved ordered reference basis group
            vector<bitset<Ns>> OrdRef_;

            /// saved orders
            unordered_map<bitset<Ns>,size_t> Ord_;

            /// triplets-form coo matrix
            vector<tuple<size_t,size_t,double>> trips_;

            /// threshold for all places
            double thres_ = 0.5;
        };
    }
}
