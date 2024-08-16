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

			Lanczos(INT ma, INT mi, INT ns, INT mb, xMPO const & op)
			: Macro_(ma), Micro_(mi), Nstates_(ns), MaxBond_(mb), mpo_(op), Vm_(mi), Wm_(mi)
			{
				INT L = op.nsite(), d = op.at(0).shape()[1];
				std::map<INT,INT> excs;
				xMPS v0(L); v0.initialize(excs,d); v0.canon();
				Vm_[0] = v0;
			}

			~Lanczos() = default;

			void impl()
			{
				INT L = mpo_.nsite();
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

			private:

			INT Macro_;   // macro iteraction

			INT Micro_;   // micro iteraction

			INT Nstates_; // number of eigenstates

			INT MaxBond_; // maximal bond

			std::vector<xMPS> Vm_; // saved Krylov vectors

			std::vector<xMPS> Wm_; // saved Krylov vectors

			xMPO mpo_; 
        };
    }
}
