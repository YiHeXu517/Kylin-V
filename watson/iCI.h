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
    namespace Watson
    {
        /// normal mode basis
        typedef vector<int> Basis;

        /// print out terms
        ostream & operator<<(ostream & os, Basis const & r)
        {
            os << '[';
            for(size_t md = 0; md < r.size() - 1; ++md)
            {
                os << r[md] << ',';
            }
            os << r.back() << ']';
            return os;
        }

        // mapping
        struct HashBasis
        {
            uint64_t operator()(Basis const & myVector) const
            {
                hash<int> hasher;
                uint64_t answer = 0;
                for (int i : myVector)
                {
                    answer ^= hasher(i) + 0x9e3779b9 +
                    (answer << 6) + (answer >> 2);
                }
                return answer;
            }
        };

        /// apply a Pm/Qm operator
        void apply_PQ(Basis const & r, double rcf, int mode,
        unordered_map<Basis,double,HashBasis> & groups)
        {
            Basis rup(r);
            double rupcf(rcf);
            size_t Qmode = abs(mode);
            rup[Qmode-1] += 1;
            rupcf *= sqrt(r[Qmode-1]*0.5+0.5);
            if(groups.find(rup)==groups.end())
            {
                groups[rup] = rupcf;
            }
            else
            {
                groups[rup] += rupcf;
            }
            // P or Q ?
            double qorp = (mode>0) ? 1.0 : -1.0;
            if(r[Qmode-1]!=0)
            {
                Basis rdn(r);
                double rdncf(rcf);
                rdn[Qmode-1] -= 1;
                rdncf *= sqrt(r[Qmode-1]*0.5) * qorp;
                if(groups.find(rdn)==groups.end())
                {
                    groups[rdn] = rdncf;
                }
                else
                {
                    groups[rdn] += rdncf;
                }
            }
        }
        class iCI
        {
            public:

            iCI() = default;

            /// read hamiltonian terms from input file
            iCI(char const * fname)
            {
                ifstream ifs(fname);
                if(!ifs.is_open())
                {
                    cout << "Fail to open input file." << endl;
                    exit(1);
                }
                while(!ifs.eof())
                {
                    vector<int> labs;
                    int tmps;
                    double tmpv;
                    for(size_t pos=0;pos<6;++pos)
                    {
                        ifs >> tmps;
                        if(tmps!=0)
                        {
                            labs.push_back(tmps);
                        }
                        size_t um = abs(tmps);
                        NumMode_ = max(NumMode_,um);
                    }
                    ifs >> tmpv;
                    if(ifs.eof()) { break; }
                    labels_.push_back(labs);
                    vals_.push_back(tmpv);
                }
                ifs.close();
                Basis vac(NumMode_,0);
                add_basis(vac);
            }

            ~iCI() = default;

            void plus_one()
            {
                unordered_set<Basis,HashBasis> rp;
                for(auto & bas : OrdRef_)
                {
                    for(int md = 0; md < NumMode_; ++md)
                    {
                        Basis bc(bas);
                        bc[md] += 1;
                        rp.insert(bc);
                    }
                }
                for(auto & bas : rp)
                {
                    if(Ord_.find(bas)==Ord_.end())
                    {
                        add_basis(bas);
                    }
                }
            }

            /// print hamiltonian terms
            void print_ham() const
            {
                size_t ntm = labels_.size();
                cout << NumMode_ << "-mode system:" << endl;
                for(size_t i=0;i<ntm;++i)
                {
                    cout << labels_[i] << " | " << vals_[i] << endl;
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
            void add_basis(Basis const & r)
            {
                Ord_[r] = OrdRef_.size();
                OrdRef_.push_back(r);
            }
            void clear_basis()
            {
                Ord_.clear();
                OrdRef_.clear();
            }

            /// generate total Hamiltonian to triplets
            void make_total()
            {
                size_t nref = OrdRef_.size(), ntm = labels_.size();
                #pragma omp parallel for
                for(size_t ref=0;ref<nref;++ref)
                {
                    // calculate
                    unordered_map<Basis,double,HashBasis> Hr;
                    for(size_t tm=0;tm<ntm;++tm)
                    {
                        double coeff = vals_[tm];
                        unordered_map<Basis,double,HashBasis> tmr;
                        tmr[OrdRef_[ref]] = coeff;
                        for(int lab : labels_[tm])
                        {
                            unordered_map<Basis,double,HashBasis> tmps;
                            for(const auto & [bas,cf] : tmr)
                            {
                                apply_PQ(bas,cf,lab,tmps);
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
                    //cout << "Hr-size: " << Hr.size() << endl;
                    // mapping
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
                    unordered_map<Basis,double,HashBasis> Hr;
                    for(size_t tm=0;tm<ntm;++tm)
                    {
                        double coeff = vals_[tm];
                        unordered_map<Basis,double,HashBasis> tmr;
                        tmr[OrdRef_[ref]] = coeff;
                        for(int lab : labels_[tm])
                        {
                            unordered_map<Basis,double,HashBasis> tmps;
                            for(const auto & [bas,cf] : tmr)
                            {
                                apply_PQ(bas,cf,lab,tmps);
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

            /// expectation <s|H|s> with same basis
            double expectation(Basis const & r)
            {
                size_t ntm = labels_.size();
                vector<double> Eacc(ntm,0.0);
                #pragma omp parallel for
                for(size_t tm=0;tm<ntm;++tm)
                {
                    if(labels_[tm].size()%2!=0)
                    {
                        continue;
                    }
                    unordered_map<Basis,double,HashBasis> tmr;
                    tmr[r] = vals_[tm];
                    for(int lab : labels_[tm])
                    {
                        unordered_map<Basis,double,HashBasis> tmps;
                        for(const auto & [bas,cf] : tmr)
                        {
                            apply_PQ(bas,cf,lab,tmps);
                        }
                        swap(tmr,tmps);
                    }
                    if(tmr.find(r)!=tmr.end())
                    {
                        Eacc[tm] = tmr[r];
                    }
                }
                return accumulate(Eacc.begin(),Eacc.end(),0.0);
            }
            /// expectation <s1|H|s2> with same basis
            double expectation(Basis const & r1, Basis const & r2)
            {
                size_t ntm = labels_.size();
                vector<double> Eacc(ntm,0.0);
                #pragma omp parallel for
                for(size_t tm=0;tm<ntm;++tm)
                {
                    unordered_map<Basis,double,HashBasis> tmr;
                    tmr[r2] = vals_[tm];
                    for(int lab : labels_[tm])
                    {
                        unordered_map<Basis,double,HashBasis> tmps;
                        for(const auto & [bas,cf] : tmr)
                        {
                            apply_PQ(bas,cf,lab,tmps);
                        }
                        swap(tmr,tmps);
                    }
                    if(tmr.find(r1)!=tmr.end())
                    {
                        Eacc[tm] = tmr[r1];
                    }
                }
                return accumulate(Eacc.begin(),Eacc.end(),0.0);
            }

            /// expand the external space basis according to PT2
            void expand(DenseBase<double> const & E, DenseBase<double> const & Prim)
            {
                size_t Np = E.size(), nref = OrdRef_.size(), ntm = labels_.size(), nth;
                #pragma omp parallel
                nth = omp_get_num_threads();

                // save the results of single process
                vector<unordered_set<Basis,HashBasis>> TmpPots(nth);

                // calculate Hmn
                #pragma omp parallel for
                for(size_t ref=0;ref<nref;++ref)
                {
                    // build m from every n
                    unordered_map<Basis,double,HashBasis> Hr;
                    for(size_t tm=0;tm<ntm;++tm)
                    {
                        double coeff = vals_[tm];
                        unordered_map<Basis,double,HashBasis> tmr;
                        tmr[OrdRef_[ref]] = coeff;
                        for(int lab : labels_[tm])
                        {
                            unordered_map<Basis,double,HashBasis> tmps;
                            for(const auto & [bas,cf] : tmr)
                            {
                                apply_PQ(bas,cf,lab,tmps);
                            }
                            swap(tmr,tmps);
                        }
                        for(const auto & [bas,cf] : tmr)
                        {
                            if(accumulate(bas.begin(),bas.end(),0)>MaxQn_) {continue;}
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

                vector<Basis> PTVec;
                unordered_map<Basis,size_t,HashBasis> PTOrds,VacPTs;

                #pragma omp parallel for
                for(size_t ref=0;ref<nref;++ref)
                {
                    unordered_map<Basis,double,HashBasis> Hr;
                    for(size_t tm=0;tm<ntm;++tm)
                    {
                        double coeff = vals_[tm];
                        unordered_map<Basis,double,HashBasis> tmr;
                        tmr[OrdRef_[ref]] = coeff;
                        for(int lab : labels_[tm])
                        {
                            unordered_map<Basis,double,HashBasis> tmps;
                            for(const auto & [bas,cf] : tmr)
                            {
                                apply_PQ(bas,cf,lab,tmps);
                            }
                            swap(tmr,tmps);
                        }
                        for(const auto & [bas,cf] : tmr)
                        {
                            if(accumulate(bas.begin(),bas.end(),0)>MaxQn_) { continue; }
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
                    unordered_map<Basis,double,HashBasis> Hr;
                    double Hmm = 0.0;
                    for(size_t tm=0;tm<ntm;++tm)
                    {
                        double coeff = vals_[tm];
                        unordered_map<Basis,double,HashBasis> tmr;
                        tmr[PTVec[m]] = coeff;
                        for(int lab : labels_[tm])
                        {
                            unordered_map<Basis,double,HashBasis> tmps;
                            for(const auto & [bas,cf] : tmr)
                            {
                                apply_PQ(bas,cf,lab,tmps);
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


            void enpt2_bkp(DenseBase<double> const & E, DenseBase<double> const & Prim, double eps = 0.02)
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

                vector<Basis> PTVec;
                unordered_map<Basis,size_t,HashBasis> PTOrds,VacPTs;

                #pragma omp parallel for
                for(size_t ref=0;ref<nref;++ref)
                {
                    unordered_map<Basis,double,HashBasis> Hr;
                    for(size_t tm=0;tm<ntm;++tm)
                    {
                        double coeff = vals_[tm];
                        unordered_map<Basis,double,HashBasis> tmr;
                        tmr[OrdRef_[ref]] = coeff;
                        for(int lab : labels_[tm])
                        {
                            unordered_map<Basis,double,HashBasis> tmps;
                            for(const auto & [bas,cf] : tmr)
                            {
                                apply_PQ(bas,cf,lab,tmps);
                            }
                            swap(tmr,tmps);
                        }
                        for(const auto & [bas,cf] : tmr)
                        {
                            if(accumulate(bas.begin(),bas.end(),0)>MaxQn_) { continue; }
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
                DenseBase<double> DeltaE(Np);
                size_t npt2 = PTVec.size();
                cout << npt2 << " ENPT2 basis." << endl;

                /// correction
                #pragma omp parallel for
                for(size_t m=0;m<npt2;++m)
                {
                    unordered_map<Basis,double,HashBasis> Hr;
                    double Hmm = 0.0;
                    for(size_t tm=0;tm<ntm;++tm)
                    {
                        double coeff = vals_[tm];
                        unordered_map<Basis,double,HashBasis> tmr;
                        tmr[PTVec[m]] = coeff;
                        for(int lab : labels_[tm])
                        {
                            unordered_map<Basis,double,HashBasis> tmps;
                            for(const auto & [bas,cf] : tmr)
                            {
                                apply_PQ(bas,cf,lab,tmps);
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
                    #pragma omp critical
                    {
                    for(size_t p=0;p<Np;++p)
                    {
                        double Hmn = 0.0;
                        for(const auto & [bas,cf] : Hr)
                        {
                            if(Ord_[bas]>=nref) {continue;}
                            Hmn += cf * Prim2.cptr()[p*nref+Ord_[bas]];
                        }
                        DeltaE.ptr()[p] += pow(Hmn,2.0) / (E.cptr()[p] - Hmm);
                    }
                    }
                }

                cout << "ENPT2 correction:" << endl;
                for(size_t st=0;st<Np;++st)
                {
                    double val = (st==0) ? E.cptr()[st] + DeltaE.cptr()[st] : E.cptr()[st] + DeltaE.cptr()[st]
                    - E.cptr()[0] - DeltaE.cptr()[0];
                    cout << "Eigval " << st+1 << " = " << val << endl;
                }
            }

            size_t num_ref() const
            {
                return OrdRef_.size();
            }

            Basis const & get_ref_basis(size_t idx) const
            {
                return OrdRef_[idx];
            }

            void set_tol(double tol)
            {
                thres_ = tol;
            }

            void set_max_qn(size_t qns)
            {
                MaxQn_ = qns;
            }

            private:

            /// number of vibrational modes
            size_t NumMode_ = 0;

            /// label-coefficients form Hamiltonian
            vector<vector<int>> labels_;

            /// coefficients
            vector<double> vals_;

            /// saved ordered reference basis group
            vector<Basis> OrdRef_;

            /// saved orders
            unordered_map<Basis,size_t,HashBasis> Ord_;

            /// triplets-form coo matrix
            vector<tuple<size_t,size_t,double>> trips_;

            /// threshold for all places
            double thres_ = 0.5;

            /// maximal total quantum number 
            size_t MaxQn_ = 10;

        };
    }
}
