/* All kinds of Lanczos eigen solvers */

#pragma once

#include "operator.h"

#define PI std::acos(-1)

namespace KylinVib
{
    namespace WatsonDMRG
    {
		/*
        class Lanczos
        {
			public:

			Lanczos(INT ma, INT mi, INT ns, INT mb, xMPO const & op)
			: Macro_(ma), Micro_(mi), Nstates_(ns), MaxBond_(mb), mpo_(op), Vm_(mi), Wm_(mi)
			{
				INT L = op.nsite(), d = op.at(0).shape()[1];
				std::map<INT,INT> excs;
				//xMPS v0(L); v0.initialize(excs,d); v0.canon();
				xMPS v0(L); v0.ones_initialize(d,1); v0.canon('r');
				Vm_[0] = v0;
			}

			~Lanczos() = default;

			void impl()
			{
				INT L = mpo_.nsite();
                ArrR<1> Enes({Nstates_}), CurEnes({Nstates_});
				for(INT mac=0;mac<Macro_;++mac)
				{
					std::cout << "Lanczos Macro Iteration " << mac+1 << std::endl;
					for(INT mic=0;mic<Micro_;++mic)
					{
						std::cout << "Iteration " << mic+1 << std::endl;
						Wm_[mic] = mpo_.apply_op(Vm_[mic]);
						if(mic==Micro_-1)
						{
							break;
						}
						xMPS hv = Wm_[mic];
						for(INT j=0;j<mic+1;++j)
						{
							double ovj = -1.0*Vm_[j].overlap(hv);
							hv += Vm_[j] * ovj;
							hv.canon('n');
						}
						hv.canon('r',MaxBond_);
						Vm_[mic+1] = std::move(hv);
					}
					ArrR<2> Hm({Micro_,Micro_}), Sm({Micro_,Micro_});
					for(size_t i=0;i<Micro_;++i)
					{
						for(size_t j=0;j<Micro_;++j)
						{
							Hm({i,j}) = Vm_[i].overlap(Wm_[j]);
							Sm({i,j}) = Vm_[i].overlap(Vm_[j]);
						}
					}
					ArrR<2> Smc(Sm), Hmc(Hm);
					ArrR<2> EigSm = Alg::eig(Smc);
					for(size_t i=0;i<Micro_;++i)
					{
						EigSm({i,i}) = std::sqrt(EigSm({i,i}));
					}
					ArrR<2> Us2 = Alg::gmm<2,2,1>(Smc,EigSm);
					ArrR<2> Us2c(Us2);
					Alg::inv<1,1>(Us2c);
					Hmc = Alg::gmm<2,2,1>(Us2c,Hmc);
					Hmc = Alg::gmm<2,2,1>(Hmc,Us2c,CblasNoTrans,CblasTrans);
					ArrR<2> EigHm = Alg::eig(Hmc);
					ArrR<2> StateAverageCoeff({Micro_,1});
					for(size_t i=0;i<Nstates_;++i)
					{
						double Valc = (i==0) ? EigHm({i,i}) : EigHm({i,i}) - EigHm({0,0});
						std::cout << "Eig-state " << i+1
						<< " = " << Valc << std::endl;
						StateAverageCoeff.ptr()[i] = 1.0;
                        CurEnes({i}) = EigHm({i,i});
					}
                    // whether to converge
                    double EneDiff = Alg::norm<1>(CurEnes-Enes) / Nstates_;
                    if(EneDiff<0.1)
                    {
                        std::cout << "<E> converged." << std::endl;
                        break;
                    }
                    else
                    {
                        Enes = CurEnes;
                    }
					ArrR<2> EigVecHam = Alg::gmm<2,2,1>(Us2c,Hmc,CblasTrans,CblasNoTrans);
					StateAverageCoeff = Alg::gmm<2,2,1>(EigVecHam,StateAverageCoeff);
					Vm_[0] *= StateAverageCoeff.cptr()[0];
					for(size_t i=1;i<Micro_;++i)
					{
						Vm_[0] += Vm_[i] * StateAverageCoeff.cptr()[i];
					}
					Vm_[0].canon('r',MaxBond_);
				}
			}

			void impl2(std::vector<xMPO> const & ops)
			{
				INT Nop = ops.size();
				ArrR<2> Hm({Micro_,Micro_}), Sm({Micro_,Micro_});
				for(size_t i=0;i<Micro_;+i)
				{

				}
				for(size_t i=0;i<Micro_;++i)
				{
					for(size_t j=0;j<Micro_;++j)
					{
						Hm({i,j}) = mpo_.join(Vm_[i],Vm_[j]);
						Sm({i,j}) = Vm_[i].overlap(Vm_[j]);
					}
				}
				ArrR<2> Smc(Sm), Hmc(Hm);
				ArrR<2> EigSm = Alg::eig(Smc);
				for(size_t i=0;i<Micro_;++i)
				{
					EigSm({i,i}) = std::sqrt(EigSm({i,i}));
				}
				ArrR<2> Us2 = Alg::gmm<2,2,1>(Smc,EigSm);
				ArrR<2> Us2c(Us2);
				Alg::inv<1,1>(Us2c);
				Hmc = Alg::gmm<2,2,1>(Us2c,Hmc);
				Hmc = Alg::gmm<2,2,1>(Hmc,Us2c,CblasNoTrans,CblasTrans);
				ArrR<2> EigHm = Alg::eig(Hmc);
				ArrR<2> StateAverageCoeff({Micro_,1});
				for(size_t i=0;i<Nstates_;++i)
				{
					double Valc = (i==0) ? EigHm({i,i}) : EigHm({i,i}) - EigHm({0,0});
					std::cout << "Eig-state " << i+1
					<< " = " << Valc << std::endl;
					StateAverageCoeff.ptr()[i] = 1.0;
				}
			}
			
			private:

			INT Macro_;   // macro iteraction

			INT Micro_;   // micro iteraction

			INT Nstates_; // number of eigenstates

			INT MaxBond_; // maximal bond

			std::vector<xMPS> Vm_; // saved Krylov vectors

			std::vector<xMPS> Wm_; // saved Krylov vectors

			xMPO mpo_; 
        };
		
        class Lanczos
        {
			public:

			Lanczos(xMPO const & op, INT nst, INT mb, INT Nop)
			: mpo_(op), Nstates_(nst), MaxBond_(mb), Vm_(Nop*nst+nst)
			{
				INT L = op.nsite(), d = op.at(0).shape()[1];
				std::map<INT,INT> excs;
				for(INT i=0;i<nst;++i)
				{
					xMPS v0(L); v0.rand_initialize(d,MaxBond_); v0.canon('r');
					Vm_[i] = v0;
				}
			}

			~Lanczos() = default;

			void impl(std::vector<xMPO> const & ops)
			{
				INT Nop = ops.size();
				ArrR<2> Hm({Nstates_*(Nop+1),Nstates_*(Nop+1)}), 
				Sm({Nstates_*(Nop+1),Nstates_*(Nop+1)});
//std::time_t now = std::time(nullptr);
//std::cout << "1: " << std::ctime(&now) << std::endl;
				for(size_t j=0;j<Nstates_*Nop;++j)
				{
					INT ks = j / Nop, ko = j % Nop;
					Vm_[j+Nstates_] = ops[ko].apply_op_single(Vm_[ks]);
					Vm_[j+Nstates_].canon('r');
				}
//now = std::time(nullptr);
//std::cout << "2: " << std::ctime(&now) << std::endl;
				INT Nm = Nstates_*(Nop+1);
				for(size_t j=0;j<Nm*Nm;++j)
				{
					INT j1 = j / Nm, j2 = j % Nm;
					Hm({j1,j2}) = mpo_.join(Vm_[j1],Vm_[j2]);
					Sm({j1,j2}) = Vm_[j1].overlap(Vm_[j2]);
				}
//now = std::time(nullptr);
//std::cout << "3: " << std::ctime(&now) << std::endl;
				//Hm.print_sp(); Sm.print_sp();
				ArrR<2> Smc(Sm), Hmc(Hm);
				ArrR<2> EigSm = Alg::eig(Smc);
				for(size_t i=0;i<Nm;++i)
				{
					EigSm({i,i}) = std::sqrt(EigSm({i,i}));
				}
				ArrR<2> Us2 = Alg::gmm<2,2,1>(Smc,EigSm);
				ArrR<2> Us2c(Us2);
				Alg::inv<1,1>(Us2c);
				Hmc = Alg::gmm<2,2,1>(Us2c,Hmc);
				Hmc = Alg::gmm<2,2,1>(Hmc,Us2c,CblasNoTrans,CblasTrans);
				ArrR<2> EigHm = Alg::eig(Hmc);
				for(size_t i=0;i<Nstates_;++i)
				{
					double Valc = (i==0) ? EigHm({i,i}) : EigHm({i,i}) - EigHm({0,0});
					std::cout << "Eig-state " << i+1
					<< " = " << std::scientific << Valc << std::endl;
				}
//now = std::time(nullptr);
//std::cout << "4: " << std::ctime(&now) << std::endl;
			}

			private:

			INT Nstates_;

			INT MaxBond_;

			std::vector<xMPS> Vm_; // saved Krylov vectors

			xMPO mpo_; 
        };
		*/
	    
		class Lanczos
		{
			public:
			Lanczos(xMPO const & op, xMPS const & s, double tl, INt mb, INT ns)
			: tol_(tl), MaxBond_(mb), Nstates_(ns), mps_(op.nsite()), mpo(op.nsite())
			{
				INT L = op.nsite();
				for(INT i=0;i<L;++i)
				{
					SparR<4> oi(op[i]); 
					SparR<3> si(s[i]);
					mpo_[i] = std::move(oi);
					mps_[i] = std::move(si);
				}
				SparR<3> edges({1,1,1});
				edges.add_elem({0,0,0},1.0);
				envl_.push_back(edges);
				envr_.push_back(edges);
				for(INT i=L-1;i>1;--i)
				{
					envr_.push_back(sweep(envr_.back(),mpo_[i],mps_[i],'l'));
				}
			}
			~Lanczos() = default;

			static SparR<3> sweep(SparR<3> const & env, SparR<4> const & op, SparR<3> const & mps,
			char l2r = 'r')
			{
				if(l2r=='r')
				{
					SparR<4> vss = Alg::gmm<3,3,1>(env,mps,{0,1,2},{0,1,2},{0,3,1,2});
					SparR<4> opt = Alg::transpose<4>(op,{0,2,1,3}),
					vsst = Alg::transpose<4>(vss,{0,3,1,2});
					SparR<4> vso = Alg::transpose<4>(Alg::gmm<3,3,1>(vsst,opt),{1,3,0,2});
					SparR<3> res = Alg::transpose<3>(Alg::gmm<3,4,2>(vso,mps),{2,1,0});
					return res;
				}
				else
				{
					SparR<3> mpst = Alg::transpose<3>(mps,{2,1,0});
					SparR<4> mpot = Alg::transpose<4>(mpo,{3,1,2,0});
					return sweep(env,mpot,mpst,'r');
				}
			}

			static ArrR<4> eigen(SparR<3> const & vl, SparR<4> const & op, SparR<3> const & vr)
			{
				SparR<3> vlt = Alg::transpose<3>(vl,{0,2,1}), vrt = Alg::transpose<3>(vr,{1,0,2});
				SparR<5> vlo = Alg::gmm<3,4,1>(vlt,op);
				SparR<6> lor = Alg::gmm<5,3,1>(vlo,vrt);

                INT nref = lor.shape()[0]*lor.shape()[2]*lor.shape()[4], nnz = lor.size();
                INT k,info,k0 = std::min(nref,Nstates_);
                INT * rws = (INT *)std::malloc(nnz*sizeof(INT)),
				cms = (INT *)std::malloc(nnz*sizeof(INT)),vaci = (INT *)std::malloc(1*sizeof(INT));
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
                rws, cms, vus);
                INT pm[128];
                mkl_sparse_ee_init(pm);
                pm[1] = 8;
                pm[6] = 1;
                info = mkl_sparse_convert_csr ( A, SPARSE_OPERATION_NON_TRANSPOSE, &B);
                info = mkl_sparse_d_ev(&which, pm, B, descr, k0, &k, E.ptr(), Prim.ptr(), res.ptr());
                mkl_sparse_destroy(A);
                mkl_sparse_destroy(B);
                cout << "Rel. Energies:" << endl;
                for(INT st=0;st<k0;++st)
                {
                    double val = (st==0) ? E.cptr()[st] : E.cptr()[st] - E.cptr()[0];
                    cout << "Eigval " << st+1 << " = " << val << endl;
                }
				std::free(rws);
				std::free(cms);
				std::free(vus); 
				std::free(vaci);
				return Prim;
			} 

			static bool TupCmp(std::tuple<INT,INT,double> const & c1, std::tuple<INT,INT,double> const & c2)
			{
				return std::get<2>(c1) > std::get<2>(c2);
			}
			static SparR<3> split(ArrR<4> const & Prim, double tol, INT mb, char l2r = 'r')
			{
				if(l2r=='r')
				{
					ArrR<4> pmt = Alg::transpose<4>(Prim,{1,2,3,0});
					std::vector<std::tuple<INT,INT,double>> nrms;
					for(INT i=0;i<pmt.shape()[0]*pmt.shape()[1];++i)
					{
						double nmi = cblas_dnrm2(pmt.dist()[1],pmt.cptr()+i*pmy.dist()[1],1);
						if(nmi<tol)
						{
							continue;
						}
						nrms.push_back(std::make_tuple(i/pmt.shape()[1],i%pmt.shape()[1],nmi));
					}
					std::sort(nrms.begin(),nrms.end(),TupComp);
					if(nmrs.size()>mb)
					{
						for(INT i=0;i<nmrs.size()-mb;++i)
						{
							nrms.pop_back();
						}
					}
					SparR<3> res({pmt.shape()[0],pmt.shape()[1],nrms.size()},nrms.size());
					for(INT i=0;i<nrms.size();++i)
					{
						std::array<INT,3> idx = {get<0>(nrms[i]),get<1>(nrms[i]),i};
						res[i] = std::move(idx);
						res.values[i] = 1.0;
					}
					return res;
				}
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
					std::sort(nrms.begin(),nrms.end(),TupComp);
					if(nmrs.size()>mb)
					{
						for(INT i=0;i<nmrs.size()-mb;++i)
						{
							nrms.pop_back();
						}
					}
					SparR<3> res({nrms.size(),Prim.shape()[1],Prim.shape()[0]},nrms.size());
					for(INT i=0;i<nrms.size();++i)
					{
						std::array<INT,3> idx = {i,get<0>(nrms[i]),get<1>(nrms[i])};
						res[i] = std::move(idx);
						res.values[i] = 1.0;
					}
					return res;
				}
			}

			void impl_dmrg(INT MaxSweeps = 10)
			{
				INT Ns = mps_.size();
				for(INT swp = 0; swp < MaxSweep; swp++)
				{
					for(INT i=0;i<Ns-1;++i)
					{
						ArrR<4> Vm = eigen(envl_.back(),mpo_[i],envr_.back());
						mps_[i] = split(Vm,tol_,MaxBond_,'r');
						envl_.push_back(envl_.back(),mpo_[i],mps_[i],'r');
						envr_.pop_back();
					}
					for(INT i=Ns-1;i>0;++i)
					{
						ArrR<4> Vm = eigen(envl_.back(),mpo_[i],envr_.back());
						mps_[i] = split(Vm,tol_,MaxBond_,'l');
						envr_.push_back(envr_.back(),mpo_[i],mps_[i],'r');
						envl_.pop_back();
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
