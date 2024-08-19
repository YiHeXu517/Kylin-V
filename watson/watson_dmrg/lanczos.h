/* All kinds of Lanczos eigen solvers */

#pragma once

#include "operator.h"

#define PI std::acos(-1)

namespace KylinVib
{
    namespace WatsonDMRG
    {
		class Lanczos
		{
			public:
			Lanczos(xMPO const & op, xMPS const & s, double tl, INT mb, INT ns)
			: tol_(tl), MaxBond_(mb), Nstates_(ns), mps_(op.nsite()), mpo_(op.nsite())
			{
				INT L = op.nsite();
				for(INT i=0;i<L;++i)
				{
					SparR<4> oi(op.at(i)); 
					SparR<3> si(s.at(i));
					mpo_[i] = std::move(oi);
					mps_[i] = std::move(si);
				}
				SparR<3> edges({1,1,1});
				edges.add_elem({0,0,0},1.0);
				envl_.push_back(edges);
				envr_.push_back(edges);
				for(INT i=L-1;i>0;--i)
				{
					envr_.push_back(sweep(envr_.back(),mpo_[i],mps_[i],'l'));
				}
                std::cout << "Environment initialized." << std::endl;
			}
			~Lanczos() = default;

			static SparR<3> sweep(SparR<3> const & env, SparR<4> const & op, SparR<3> const & mps,
			char l2r = 'r')
			{
				if(l2r=='r')
				{
					SparR<4> vss = Alg::gmm<3,3,1>(env,mps);
					SparR<4> opt = Alg::transpose<4>(op,{0,2,1,3});
					SparR<4> vsst = Alg::transpose<4>(vss,{0,3,1,2});
					SparR<4> vso = Alg::transpose<4>(Alg::gmm<4,4,2>(vsst,opt),{1,3,0,2});
					SparR<3> res = Alg::transpose<3>(Alg::gmm<4,3,2>(vso,mps),{2,1,0});
					return res;
				}
				else
				{
					SparR<3> mpst = Alg::transpose<3>(mps,{2,1,0});
					SparR<4> mpot = Alg::transpose<4>(op,{3,1,2,0});
					return sweep(env,mpot,mpst,'r');
				}
			}

			static ArrR<4> eigen(SparR<3> const & vl, SparR<4> const & op, SparR<3> const & vr, INT nst)
			{
				SparR<3> vlt = Alg::transpose<3>(vl,{0,2,1}), vrt = Alg::transpose<3>(vr,{1,0,2});
				SparR<5> vlo = Alg::gmm<3,4,1>(vlt,op);
				SparR<6> lor = Alg::gmm<5,3,1>(vlo,vrt);

                INT nref = lor.shape()[0]*lor.shape()[2]*lor.shape()[4], nnz = lor.size();
                INT k,info,k0 = std::min(nref,nst);
                INT * rws = (INT *)std::malloc(nnz*sizeof(INT)),
				* cms = (INT *)std::malloc(nnz*sizeof(INT)), * vaci = (INT *)std::malloc(1*sizeof(INT));
                ArrR<1> vus({nnz}),res({k0}),vacd({1}),E({k0});
                ArrR<4> Prim({k0,lor.shape()[0],lor.shape()[2],lor.shape()[4]});
				std::array<INT,3> Dists = {lor.shape()[2]*lor.shape()[4], lor.shape()[4], 1};

                #pragma omp parallel for
                for(INT em=0;em<nnz;++em)
                {
                    rws[em] = lor[em][0] * Dists[0] + lor[em][2] * Dists[1] + lor[em][4] +1;
                    cms[em] = lor[em][1] * Dists[0] + lor[em][3] * Dists[1] + lor[em][5] +1;
                    vus({em}) = lor.values()[em];
                }

                char which = 'S';
                sparse_matrix_t A = NULL, B = NULL;
                struct matrix_descr descr;
                descr.type = SPARSE_MATRIX_TYPE_GENERAL;
                mkl_sparse_d_create_coo ( &A, SPARSE_INDEX_BASE_ONE, nref, nref, nnz,
                rws, cms, vus.ptr());
                INT pm[128];
                mkl_sparse_ee_init(pm);
                pm[1] = 8;
                pm[6] = 1;
                info = mkl_sparse_convert_csr ( A, SPARSE_OPERATION_NON_TRANSPOSE, &B);

                
                info = mkl_sparse_d_ev(&which, pm, B, descr, k0, &k, E.ptr(), Prim.ptr(), res.ptr());
                mkl_sparse_destroy(A);
                mkl_sparse_destroy(B);
                std::cout << "Rel. Energies:" << std::endl;
                for(INT st=0;st<k0;++st)
                {
                    double val = (st==0) ? E.cptr()[st] : E.cptr()[st] - E.cptr()[0];
                    std::cout << "Eigval " << st+1 << " = " << val << std::endl;
                }
				std::free(rws);
				std::free(cms);
				std::free(vaci);
				return Prim;
			} 

			static bool TupCmp(std::tuple<INT,INT,double> const & c1, std::tuple<INT,INT,double> const & c2)
			{
				return std::get<2>(c1) > std::get<2>(c2);
			}
			static SparR<3> split(ArrR<4> const & Prim, double tol, INT mb, char l2r = 'r')
			{
                // a_j-1,sig_j,a_j,I
				if(l2r=='r')
				{
					ArrR<4> pmt = Alg::transpose<4>(Prim,{1,2,3,0});
					std::vector<std::tuple<INT,INT,double>> nrms;
					for(INT i=0;i<pmt.shape()[0]*pmt.shape()[1];++i)
					{
						double nmi = cblas_dnrm2(pmt.dist()[1],pmt.cptr()+i*pmt.dist()[1],1);
						if(nmi<tol)
						{
							continue;
						}
						nrms.push_back(std::make_tuple(i/pmt.shape()[1],i%pmt.shape()[1],nmi));
					}
					std::sort(nrms.begin(),nrms.end(),TupCmp);
					if(nrms.size()>mb)
					{
						for(INT i=0;i<nrms.size()-mb;++i)
						{
							nrms.pop_back();
						}
					}
					SparR<3> res({pmt.shape()[0],pmt.shape()[1],nrms.size()},nrms.size());
					for(INT i=0;i<nrms.size();++i)
					{
						std::array<INT,3> idx = {std::get<0>(nrms[i]),std::get<1>(nrms[i]),i};
						res[i] = std::move(idx);
						res.values()[i] = 1.0;
					}
					return res;
				}
                // I, a_j-1, sig_j, a_j
				else
				{
					std::vector<std::tuple<INT,INT,double>> nrms;
					INT sp01 = Prim.shape()[0] * Prim.shape()[1];
					for(INT i=0;i<Prim.dist()[1];++i)
					{
						double nmi = cblas_dnrm2(sp01,Prim.cptr()+i,Prim.dist()[1]);
						if(nmi<tol)
						{
							continue;
						}
						nrms.push_back(std::make_tuple(i/Prim.shape()[3],i%Prim.shape()[3],nmi));
					}
					std::sort(nrms.begin(),nrms.end(),TupCmp);
					if(nrms.size()>mb)
					{
						for(INT i=0;i<nrms.size()-mb;++i)
						{
							nrms.pop_back();
						}
					}
					SparR<3> res({nrms.size(),Prim.shape()[2],Prim.shape()[3]},nrms.size());
					for(INT i=0;i<nrms.size();++i)
					{
						std::array<INT,3> idx = {i,std::get<0>(nrms[i]),std::get<1>(nrms[i])};
						res[i] = std::move(idx);
						res.values()[i] = 1.0;
					}
                    std::cout << "4" << std::endl;
					return res;
				}
			}

			void impl_dmrg(INT MaxSweeps = 10)
			{
				INT Ns = mps_.size();
				for(INT swp = 0; swp < MaxSweeps; swp++)
				{
					for(INT i=0;i<Ns;++i)
					{
                        std::cout << "Site " << i+1 << std::endl;
						ArrR<4> Vm = eigen(envl_.back(),mpo_[i],envr_.back(),Nstates_);
						mps_[i] = split(Vm,tol_,MaxBond_,'r');
                        if(i!=Ns-1)
                        {
                        std::cout << "Bond dimension " << Vm.shape()[3] << "->"  << mps_[i].shape()[2] << std::endl;
						envl_.push_back(sweep(envl_.back(),mpo_[i],mps_[i],'r'));
						envr_.pop_back();
                        }
					}
					for(INT i=Ns-1;i>=0;--i)
					{
                        std::cout << "Site " << i+1 << std::endl;
						ArrR<4> Vm = eigen(envl_.back(),mpo_[i],envr_.back(),Nstates_);
						mps_[i] = split(Vm,tol_,MaxBond_,'l');
                        if(i!=0)
                        {
                        std::cout << "Bond dimension " << Vm.shape()[1] << "->"  << mps_[i].shape()[0] << std::endl;
						envr_.push_back(sweep(envr_.back(),mpo_[i],mps_[i],'l'));
						envl_.pop_back();
                        }
					}
				}
			}
			
			private:
			double tol_;
			INT MaxBond_;
			INT Nstates_;
			std::vector<SparR<3>> mps_;
			std::vector<SparR<4>> mpo_;
			std::vector<SparR<3>> envl_;
			std::vector<SparR<3>> envr_;
		};
    }
}
