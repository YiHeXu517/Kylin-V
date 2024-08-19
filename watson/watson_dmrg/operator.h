/* All kinds of quantum states */

#pragma once

#include "state.h"

namespace KylinVib
{
    namespace WatsonDMRG
    {
        // float-version matrix product state
        class xMPO
        {
            private:

            // tensor train
            std::vector<ArrR<4>> tt_;

            public:

            // default tors
            xMPO() = default;

            ~xMPO() = default;

            // copy
            xMPO(xMPO const & r)
            : tt_(r.tt_)
            {

            }

            xMPO & operator=(xMPO const & r)
            {
                tt_ = r.tt_;
                return *this;
            }

            // move
            xMPO(xMPO && r)
            : tt_(std::move(r.tt_))
            {

            }

            xMPO & operator=(xMPO && r)
            {
                tt_ = std::move(r.tt_);
                return *this;
            }

            // allocation
            xMPO(INT nsite)
            : tt_(nsite)
            {

            }

            // double allocation
            xMPO(INT nsite, Brace Sp)
            : tt_(nsite, Sp)
            {

            }

            // plus
            xMPO & operator+=(xMPO const & r)
            {
                INT Ns = tt_.size();
                tt_[0] = Alg::stack<4>(tt_[0],r.tt_[0],{3});
                tt_[Ns-1] = Alg::stack<4>(tt_[Ns-1],r.tt_[Ns-1],{0});
                for(INT i=1;i<Ns-1;++i)
                {
                    tt_[i] = Alg::stack<4>(tt_[i],r.tt_[i],{0,3});
                }
                return *this;
            }
            xMPO operator+(xMPO const & r)
            {
                xMPO res(*this);
                INT Ns = tt_.size();
                res.tt_[0] = Alg::stack<4>(tt_[0],r.tt_[0],{3});
                res.tt_[Ns-1] = Alg::stack<4>(tt_[Ns-1],r.tt_[Ns-1],{0});
                for(INT i=1;i<Ns-1;++i)
                {
                    res.tt_[i] = Alg::stack<4>(tt_[i],r.tt_[i],{0,3});
                }
                return res;
            }
            // self multiplication
            xMPO & operator*=(double val)
            {
                tt_[0] *= val;
                return *this;
            }

            // index
            ArrR<4> & operator[](INT idx)
            {
                return tt_[idx];
            }

            const ArrR<4> & at(INT idx) const
            {
                return tt_[idx];
            }

            xMPO slice(INT Starts, INT Ends)
            {
                xMPO res(Ends-Starts);
                std::copy(tt_.begin()+Starts, tt_.begin()+Ends, res.tt_.begin());
                return res;
            }

            // number of site
            INT nsite() const
            {
                return tt_.size();
            }

            // print out
            void print(INT ncol = 1) const
            {
                std::cout << tt_.size() << "-site " <<  " MPO:" << std::endl;
                for(INT i=0;i<tt_.size();++i)
                {
                    std::cout << "Site " << i << ":" << std::endl;
                    tt_[i].print(ncol);
                }
            }
            // canonicalize
            void canon()
            {
                INT Ns = tt_.size();
                for(INT i=0;i<Ns-1;++i)
                {
                    auto[lef,rig] = Alg::svd<3,1>(tt_[i],'r',1e-14,1000,'r');
                    tt_[i] = std::move(lef);
                    tt_[i+1] = Alg::gmm<2,4,1>(rig,tt_[i+1]);
                }
                for(INT i=Ns-1;i>0;--i)
                {
                    auto[lef,rig] = Alg::svd<1,3>(tt_[i],'l',1e-14,1000,'r');
                    tt_[i] = std::move(rig);
                    tt_[i-1] = Alg::gmm<4,2,1>(tt_[i-1],lef);
                }
            }
            void canon_with_mps(xMPS const & s)
            {
                INT Ns = tt_.size();
                for(INT i=0;i<Ns-1;++i)
                {
                    auto[lef,rig] = Alg::svd<3,1>(tt_[i],'r',1e-14,1000,'r');
                    tt_[i] = std::move(lef);
                    tt_[i+1] = Alg::gmm<2,4,1>(rig,tt_[i+1]);
                    std::cout << "Site " << i+1 << " = " << join(s,s) << std::endl;
                }
                for(INT i=Ns-1;i>0;--i)
                {
                    auto[lef,rig] = Alg::svd<1,3>(tt_[i],'l',1e-14,1000,'r');
                    tt_[i] = std::move(rig);
                    tt_[i-1] = Alg::gmm<4,2,1>(tt_[i-1],lef);
                    std::cout << "Site " << i+1 << " = " << join(s,s) << std::endl;
                }
            }

            xMPO transpose(Brace ax) const
            {
                xMPO res(tt_.size());
                for(INT i=0;i<tt_.size();++i)
                { res[i] = Alg::transpose<4>(tt_[i],ax);  }
                return res;
            }

            // local basis truncation
            void local_trunc(INT NeoLevel)
            {
                INT Ns = tt_.size();
                for(INT i=0;i<Ns;++i)
                {
                    tt_[i] = tt_[i].slice({0,0,0,0},
                    {tt_[i].shape()[0],NeoLevel,NeoLevel,tt_[i].shape()[3]});
                }
            }

            // expectation
            double join(xMPS const & s1, xMPS const & s2)
            {
                INT Ns = tt_.size();
                ArrR<3> env({1,1,1}); env({0,0,0}) = 1.0;
                for(INT i=0;i<Ns;++i)
                {
                    env = sweep(env,tt_[i],s1.at(i),s2.at(i));
                }
                return env.ptr()[0];
            }

            // operator application
            xMPS apply_op(xMPS const & s, double tol = 1e-14, INT maxdim = 1000)
            {
                INT Ns = tt_.size();
                xMPS res(Ns);
                ArrR<3> env({1,1,1}); env({0,0,0}) = 1.0;
                for(INT i=0;i<Ns;++i)
                {
                    //std::cout << "app" << i << std::endl;
                    ArrR<4> ssi = env_mps_mpo(env,tt_[i],s.at(i),'r');
                    ssi = Alg::transpose<4>(ssi,{2,0,1,3});
                    if(i!=Ns-1)
                    {
                        auto[lef,rig] = Alg::svd<2,2>(ssi,'r',tol,maxdim);
                        res[i] = std::move(lef);
                        env    = std::move(rig);
                    }
                    else
                    {
                        res[i] = Alg::reshape<4,3>(ssi,{ssi.shape()[0],ssi.shape()[1],ssi.shape()[3]});
                    }
                }
                for(INT i=Ns-1;i>0;--i)
                {
                    //std::cout << "app" << i << std::endl;
                    auto[lef,rig] = Alg::svd<1,2>(res[i],'l',tol,maxdim,'n');
                    res[i] = std::move(rig);
                    res[i-1] = Alg::gmm<3,2,1>(res.at(i-1),lef);
                }
                return res;
            }
            // apply bond-1 single operator to wavefunction
            xMPS apply_op_single(xMPS const & s) const
            {
                INT Ns = tt_.size();
                xMPS res(Ns);
                for(INT i=0;i<Ns;++i)
                {
                    INT aj1 = s.at(i).shape()[0], sig = s.at(i).shape()[1], aj = s.at(i).shape()[2];
                    ArrR<3> ri(s.at(i).shape());
                    for(INT j=0;j<aj1;++j)
                    {
                        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,sig,aj,sig,
                        1.0,tt_[i].cptr(),sig,s.at(i).cptr()+j*sig*aj,aj,0.0,ri.ptr()+j*sig*aj,aj);
                    }
                    res[i] = std::move(ri);
                }
                return res;
            }

            INT max_bond() const
            {
                INT Ns = tt_.size();
                INT res = 1;
                for(INT i=0;i<Ns;++i)
                {
                    res = std::max(tt_[i].shape()[3],res);
                }
                return res;
            }

            // save the Hamiltonian
            void save(const char * fn) const
            {
                INT Ns = tt_.size();
                std::ofstream ofs(fn);
                ofs.write(reinterpret_cast<const char*>(&Ns), sizeof(INT));
                for(INT i=0;i<Ns;++i)
                {
                    ofs.write(reinterpret_cast<const char*>(tt_[i].shape().data()), sizeof(INT)*4);
                    ofs.write(reinterpret_cast<const char*>(tt_[i].cptr()), sizeof(double)*tt_[i].size());
                }
                ofs.close();
            }

            // load Hamiltonian
            static xMPO load(const char * fn)
            {
                INT Ns;
                std::ifstream ifs(fn);
                if(!ifs.is_open()) { std::cout << "Fail to open MPO file!" << std::endl; std::exit(1); }
                ifs.read(reinterpret_cast<char*>(&Ns), sizeof(INT));
                xMPO res(Ns);
                for(INT i=0;i<Ns;++i)
                {
                    std::array<INT,4> spi;
                    ifs.read(reinterpret_cast<char*>(spi.data()), sizeof(INT)*4);
                    ArrR<4> ri(spi);
                    ifs.read(reinterpret_cast<char*>(ri.ptr()), sizeof(double)*ri.size());
                    res[i] = std::move(ri);
                }
                ifs.close();
                return res;
            }

            // mpo dot product with bond=1
            static xMPO mpo_dot(xMPO const & h1, xMPO const & h2)
            {
                INT Ns = h1.nsite();
                xMPO res(Ns);
                for(INT i=0;i<Ns;++i)
                {
                    res[i] = Alg::gmm<4,4,2>(h1.at(i),h2.at(i));
                }
                return res;
            }
            // mpo dot product with bond!=1
            static xMPO mpo_dot_pl(xMPO const & h1, xMPO const & h2)
            {
                INT Ns = h1.nsite();
                xMPO res(Ns);
                ArrR<3> env({1,1,1}); env({0,0,0}) = 1.0;
                for(INT i=0;i<Ns;++i)
                {
                    ArrR<5> EO = Alg::gmm<3,4,1>(env,h2.at(i));
                    ArrR<5> OEO = Alg::gtmm<4,5,2>(h1.at(i),EO,{1,3,0,2},{0,2,1,3,4});
                    ArrR<5> OEOt = Alg::transpose<5>(OEO,{2,0,3,1,4});
                    auto[lef,rig] = Alg::svd<3,2>(OEOt,'r',1e-14);
                    res.tt_[i] = std::move(lef);
                    env = Alg::transpose<3>(rig,{1,0,2});
                    if(i==Ns-1)
                    {
                        res[i] = res[i] * env({0,0,0});
                    }
                }
                return res;
            }

            // l-w-m product
            static ArrR<4> env_mps_mpo(ArrR<3> const & envl, ArrR<4> const & mpo, ArrR<3> const & mps,char l2r='r')
            {
                if(l2r=='r') // sig_j * b_j * a_j-1 * a_j'
                {
                    ArrR<4> lss = Alg::gmm<3,3,1>(envl,mps);
                    ArrR<4> lso = Alg::gtmm<4,4,2>(mpo,lss,{1,3,0,2},{1,2,0,3});
                    return lso;
                }
                else // b_j-1 * sig_j * a_j'-1 * a_j
                {
                    ArrR<4> lss = Alg::gmm<3,3,1>(mps,envl,CblasNoTrans,CblasTrans);
                    ArrR<4> lsst = Alg::transpose<4>(lss,{1,3,0,2});
                    ArrR<4> lso = Alg::gmm<4,4,2>(mpo,lsst);
                    return lso;
                }
            }

            // sweep along mps
            static ArrR<3> sweep(ArrR<3> const & envl, ArrR<4> const & mpo, ArrR<3> const & mps, char l2r = 'r')
            {
                if(l2r=='r')
                {
                    ArrR<4> lso = Alg::transpose<4>(env_mps_mpo(envl,mpo,mps,l2r),{2,0,1,3});
                    ArrR<3> res = Alg::gmm<3,4,2>(mps,lso,CblasTrans,CblasNoTrans);
                    return res;
                }
                else
                {
                    ArrR<4> lso = Alg::transpose<4>(env_mps_mpo(envl,mpo,mps,l2r),{1,3,0,2});
                    ArrR<3> res = Alg::gmm<3,4,2>(mps,lso,CblasNoTrans,CblasNoTrans);
                    return res;
                }
            }
            // <2|1>
            static ArrR<2> sweep(ArrR<2> const & envl, ArrR<3> const & mps1, ArrR<3> const & mps2, char l2r = 'r')
            {
                if(l2r=='r')
                {
                    ArrR<3> lso = Alg::gmm<2,3,1>(envl,mps1);
                    ArrR<2> res = Alg::gmm<3,3,2>(mps2,lso,CblasTrans,CblasNoTrans);
                    return res;
                }
                else
                {
                    ArrR<3> lso = Alg::gmm<3,2,1>(mps1,envl,CblasNoTrans,CblasTrans);
                    ArrR<2> res = Alg::gmm<3,3,2>(mps2,lso,CblasNoTrans,CblasTrans);
                    return res;
                }
            }

            static ArrR<3> sweep(ArrR<3> const & envl, ArrR<4> const & mpo, ArrR<3> const & mps1, ArrR<3> const & mps2, char l2r = 'r')
            {
                if(l2r=='r')
                {
                    ArrR<4> lso = Alg::transpose<4>(env_mps_mpo(envl,mpo,mps2,l2r),{2,0,1,3});
                    ArrR<3> res = Alg::gmm<3,4,2>(mps1,lso,CblasTrans,CblasNoTrans);
                    return res;
                }
                else
                {
                    ArrR<4> lso = Alg::transpose<4>(env_mps_mpo(envl,mpo,mps2,l2r),{1,3,0,2});
                    ArrR<3> res = Alg::gmm<3,4,2>(mps1,lso,CblasNoTrans,CblasNoTrans);
                    return res;
                }
            }
            // 2-MPS application
            static ArrR<3> apply_single(ArrR<2> const & envl, ArrR<3> const & mps, ArrR<2> const & envr)
            {
                ArrR<3> lss = Alg::gmm<2,3,1>(envl,mps);
                ArrR<3> res = Alg::gmm<3,2,1>(lss,envr,CblasNoTrans,CblasTrans);
                return res;
            }
            // one-site application
            static ArrR<3> apply_single(ArrR<3> const & envl, ArrR<4> const & mpo, ArrR<3> const & mps, ArrR<3> const & envr)
            {
                ArrR<4> lso = Alg::transpose<4>(env_mps_mpo(envl,mpo,mps,'r'),{2,0,1,3});
                ArrR<3> res = Alg::gmm<4,3,2>(lso,envr,CblasNoTrans,CblasTrans);
                return res;
            }
            static ArrR<3> apply_single(ArrR<3> const & envl, ArrR<4> const & mpo, ArrR<3> const & mps, ArrR<3> const & envr,
	    ArrR<3> const & u)
            {
                ArrR<4> lso = Alg::transpose<4>(env_mps_mpo(envl,mpo,mps,'r'),{2,0,1,3});
                ArrR<3> res = Alg::gmm<4,3,2>(lso,envr,CblasNoTrans,CblasTrans);
		double ovp = Alg::overlap<3>(u,res);
		res -= u * ovp;
                return res;
            }
	
            // one-site application
            // Vm : (I,a_j-1,sig_j,a_j)
            // @retrun (a_j-1,sig_j,a_j,I)
            static ArrR<4> apply_single(ArrR<3> const & envl, ArrR<4> const & mpo, ArrR<4> const & Vm, ArrR<3> const & envr)
            {
                ArrR<4> res(Vm.shape());
                INT EleSize = Vm.size() / Vm.shape()[0];
                for(INT i=0;i<Vm.shape()[0];++i)
                {
                    ArrR<3> Vmi({Vm.shape()[1],Vm.shape()[2],Vm.shape()[3]});
                    cblas_zcopy(EleSize,Vm.cptr()+i*EleSize,1,Vmi.ptr(),1);
                    Vmi = apply_single(envl,mpo,Vmi,envr);
                    cblas_zcopy(EleSize,Vmi.cptr(),1,res.ptr()+i*EleSize,1);
                }
                return res;
            }

            // one-site application (avoid copy in lanczos)
            static ArrR<3> apply_single(ArrR<3> const & envl, ArrR<4> const & mpo, ArrR<4> const & Vm, ArrR<3> const & envr, INT idx)
            {
                ArrR<4> lss({envl.shape()[0],envl.shape()[1],Vm.shape()[2],Vm.shape()[3]});
                double ones(1.0),zeros(0.0);
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, envl.shape()[0]*envl.shape()[1],
                Vm.shape()[2]*Vm.shape()[3], envl.shape()[2], &ones, envl.cptr(), envl.shape()[2],
                Vm.cptr()+idx*Vm.shape()[1]*Vm.shape()[2]*Vm.shape()[3], Vm.shape()[2]*Vm.shape()[3],
                &zeros, lss.ptr(), Vm.shape()[2]*Vm.shape()[3]);
                ArrR<4> lso = Alg::gtmm<4,4,2>(mpo,lss,{1,3,0,2},{1,2,0,3});
                lso = Alg::transpose<4>(lso,{2,0,1,3});
                ArrR<3> res = Alg::gmm<4,3,2>(lso,envr,CblasNoTrans,CblasTrans);
                return res;
            }
            // @return same
            static ArrR<4> apply_double(ArrR<3> const & envl, ArrR<4> const & mpo1, ArrR<4> const & mpo2, ArrR<4> const & mps2, ArrR<3> const & envr)
            {
                ArrR<5> lss = Alg::gmm<3,4,1>(envl,mps2);
                ArrR<5> lso1 = Alg::gtmm<5,4,2>(lss,mpo1,{0,3,4,1,2},{0,2,1,3});
                ArrR<5> lso2 = Alg::gtmm<5,4,2>(lso1,mpo2,{0,2,3,4,1},{0,2,1,3});
                lso2 = Alg::transpose<5>(lso2,{0,2,3,4,1});
                ArrR<4> hsas2 = Alg::gmm<5,3,2>(lso2,envr,CblasNoTrans,CblasTrans);
                return hsas2;
            }
            static ArrR<5> apply_double(ArrR<3> const & envl, ArrR<4> const & mpo1, ArrR<4> const & mpo2, ArrR<5> const & Vm, ArrR<3> const & envr)
            {
                ArrR<5> res(Vm.shape());
                INT EleSize = Vm.size() / Vm.shape()[0];
                for(INT i=0;i<Vm.shape()[0];++i)
                {
                    ArrR<4> Vmi({Vm.shape()[1],Vm.shape()[2],Vm.shape()[3],Vm.shape()[4]});
                    cblas_zcopy(EleSize,Vm.cptr()+i*EleSize,1,Vmi.ptr(),1);
                    Vmi = apply_double(envl,mpo1,mpo2,Vmi,envr);
                    cblas_zcopy(EleSize,Vmi.cptr(),1,res.ptr()+i*EleSize,1);
                }
                return res;
            }
            static ArrR<4> apply_double(ArrR<3> const & envl, ArrR<4> const & mpo1, ArrR<4> const & mpo2, ArrR<5> const & Vm, ArrR<3> const & envr, INT idx)
            {
                INT EleSize = Vm.size() / Vm.shape()[0];
                ArrR<4> res({Vm.shape()[1],Vm.shape()[2],Vm.shape()[3],Vm.shape()[4]});
                cblas_zcopy(EleSize,Vm.cptr()+idx*EleSize,1,res.ptr(),1);
                ArrR<4> Vmi = apply_double(envl,mpo1,mpo2,res,envr);
                cblas_zcopy(EleSize,Vmi.cptr(),1,res.ptr(),1);
                return res;
            }
            // diagonal elements of effective H
            static ArrR<3> diag_heff(ArrR<3> const & envl, ArrR<4> const & mpo, ArrR<3> const & envr)
            {
                ArrR<2> diagL({envl.shape()[0],envl.shape()[1]});
                ArrR<3> diagW({mpo.shape()[0],mpo.shape()[1],mpo.shape()[3]});
                ArrR<2> diagR({envr.shape()[1],envr.shape()[2]});
                for(INT i(0);i<diagL.size();++i)
                {
                    INT i1(i/diagL.dist()[0]%diagL.shape()[0]), i2(i%diagL.shape()[1]);
                    diagL.ptr()[i] = envl({i1,i2,i1});
                }
                for(INT i(0);i<diagR.size();++i)
                {
                    INT i1(i/diagR.dist()[0]%diagR.shape()[0]), i2(i%diagR.shape()[1]);
                    diagR.ptr()[i] = envr({i2,i1,i2});
                }
                for(INT i(0);i<diagW.size();++i)
                {
                    INT i1(i/diagW.dist()[0]%diagW.shape()[0]), i2(i/diagW.dist()[1]%diagW.shape()[1]),
                    i3(i%diagW.shape()[2]);
                    diagW.ptr()[i] = mpo({i1,i2,i2,i3});
                }
                diagW = Alg::gmm<2,3,1>(diagL,diagW);
                ArrR<3> res = Alg::gmm<3,2,1>(diagW,diagR);
                return res;
            }
        };
    }
}
