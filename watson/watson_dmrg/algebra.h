/* Linear algebra tools */

#pragma once

#include "dense.h"

namespace KylinVib
{
    namespace WatsonDMRG
    {
        class Alg
        {
            public:

            // ctors
            Alg() = default;

            // dtors
            ~Alg() = default;

            // norm
            template<INT N>
            static double norm(ArrR<N> const & m)
            {
                return cblas_dnrm2(m.size(), m.cptr(), 1);
            }
            template<INT N>
            static double norm(ArrC<N> const & m)
            {
                return cblas_dznrm2(m.size(), m.cptr(), 1);
            }

            // overlap
            template<INT N>
            static double overlap(ArrR<N> const & m1, ArrR<N> const & m2)
            {
                return cblas_ddot(m1.size(), m1.cptr(), 1, m2.cptr(), 1);
            }
            template<INT N>
            static Complex overlap(ArrC<N> const & m1, ArrC<N> const & m2)
            {
                Complex val;
                cblas_zdotc_sub(m1.size(), m1.cptr(), 1, m2.cptr(), 1, &val);
                return val;
            }

            // reshape
            template<INT Ni, INT Nr>
            static ArrR<Nr> reshape(ArrR<Ni> const & m, Brace rSp)
            {
                ArrR<Nr> res(rSp);
                std::memcpy(res.ptr(), m.cptr(), res.size()*sizeof(double));
                return res;
            }
            template<INT Ni, INT Nr>
            static ArrC<Nr> reshape(ArrC<Ni> const & m, Brace rSp)
            {
                ArrC<Nr> res(rSp);
                std::memcpy(res.ptr(), m.cptr(), res.size()*sizeof(Complex));
                return res;
            }

            template<INT Ni, INT Nr>
            static ArrR<Nr> reshape(ArrR<Ni> const & m, std::array<INT,Nr> const & rSp)
            {
                ArrR<Nr> res(rSp);
                std::memcpy(res.ptr(), m.cptr(), res.size()*sizeof(double));
                return res;
            }
            template<INT Ni, INT Nr>
            static ArrC<Nr> reshape(ArrC<Ni> const & m, std::array<INT,Nr> const & rSp)
            {
                ArrC<Nr> res(rSp);
                std::memcpy(res.ptr(), m.cptr(), res.size()*sizeof(Complex));
                return res;
            }
            // conjugated
            template<INT N>
            static ArrC<N> conj(ArrC<N> const & m)
            {
                ArrC<N> res(m.shape());
                #pragma omp parallel for schedule(static)
                for(INT i=0;i<res.size();++i) { res.ptr()[i] = std::conj(m.cptr()[i]); }
                return res;
            }
            // power of tensors
            template<INT N>
            static ArrR<N> tensor_power(ArrR<N> const & r, double Pow)
            {
                ArrR<N> res(r);
                # pragma omp parallel for schedule(static)
                for(INT i=0;i<r.size();++i)
                {
                    res.ptr()[i] = std::pow(r.cptr()[i], Pow);
                }
                return res;
            }
            // remove all nans
            // power of tensors
            template<INT N>
            static void rm_nan(ArrR<N> & r)
            {
                # pragma omp parallel for schedule(static)
                for(INT i=0;i<r.size();++i)
                {
                    if ( std::isnan(r.ptr()[i]) )
                    {
                        r.ptr()[i] = 0.0;
                    }
                }
            }
            // Hadamard product
            template<INT N>
            static ArrR<N> hadamard(ArrR<N> const & r1, ArrR<N> const & r2)
            {
                ArrR<N> res(r1);
                # pragma omp parallel for schedule(static)
                for(INT i=0;i<res.size();++i)
                {
                    res.ptr()[i] = r1.cptr()[i] * r2.cptr()[i];
                }
                return res;
            }

            template<INT N>
            static ArrR<N> transpose(ArrR<N> const & r, Brace ax)
            {
                std::array<INT,N> rsp;
                //auto axit = ax.begin();
                //std::for_each(rsp.begin(),rsp.end(),[&axit, &r](INT & x){x=r.shape()[*axit]; ++axit;});
                for(INT i=0;i<N;++i)
                {
                    rsp[i] = r.shape()[*(ax.begin()+i)];
                }
                ArrR<N> res(rsp);
                //#pragma omp parallel for schedule(static)
                for(INT i=0;i<r.size();++i)
                {
                    std::array<INT,N> idx;
                    for(INT j=0;j<N;++j)
                    {
                        idx[j] = i / r.dist()[*(ax.begin()+j)] % r.shape()[*(ax.begin()+j)];
                    }
                    res(idx) = r.cptr()[i];
                }
                return res;
            }
            template<INT N>
            static ArrC<N> transpose(ArrC<N> const & r, Brace ax)
            {
                std::array<INT,N> rsp;
                //auto axit = ax.begin();
                //std::for_each(rsp.begin(),rsp.end(),[&axit, &r](INT & x){x=r.shape()[*axit]; ++axit;});
                for(INT i=0;i<N;++i)
                {
                    rsp[i] = r.shape()[*(ax.begin()+i)];
                }
                ArrC<N> res(rsp);
                //#pragma omp parallel for schedule(static)
                for(INT i=0;i<r.size();++i)
                {
                    std::array<INT,N> idx;
                    for(INT j=0;j<N;++j)
                    {
                        idx[j] = i / r.dist()[*(ax.begin()+j)] % r.shape()[*(ax.begin()+j)];
                    }
                    res(idx) = r.cptr()[i];
                }
                return res;
            }

            // stack two arrays
            template<INT N>
            static ArrR<N> stack(ArrR<N> const & r1, ArrR<N> const & r2, Brace ax)
            {
                std::array<INT,N> rsp(r1.shape());
                INT leg = 0;
                for(auto it=ax.begin();it!=ax.end();++it)
                {
                    rsp[*it] += r2.shape()[*it];
                }
                ArrR<N> res(rsp);
                //#pragma omp parallel for schedule(static)
                for(INT i=0;i<r1.size();++i)
                {
                    std::array<INT,N> idx;
                    for(INT j=0;j<N;++j)
                    {
                        idx[j] = i / r1.dist()[j] % r1.shape()[j];
                    }
                    res(idx) = r1.cptr()[i];
                }
                //#pragma omp parallel for schedule(static)
                for(INT i=0;i<r2.size();++i)
                {
                    std::array<INT,N> idx;
                    for(INT j=0;j<N;++j)
                    {
                        idx[j] = i / r2.dist()[j] % r2.shape()[j];
                        if(std::find(ax.begin(),ax.end(),j)!=ax.end())
                        {
                            idx[j] += r1.shape()[j];
                        }
                    }
                    res(idx) = r2.cptr()[i];
                }
                return res;
            }
            // stack two arrays
            template<INT N>
            static ArrC<N> stack(ArrC<N> const & r1, ArrC<N> const & r2, Brace ax)
            {
                std::array<INT,N> rsp(r1.shape());
                INT leg = 0;
                for(auto it=ax.begin();it!=ax.end();++it)
                {
                    rsp[*it] += r2.shape()[*it];
                }
                ArrC<N> res(rsp);
                //#pragma omp parallel for schedule(static)
                for(INT i=0;i<r1.size();++i)
                {
                    std::array<INT,N> idx;
                    for(INT j=0;j<N;++j)
                    {
                        idx[j] = i / r1.dist()[j] % r1.shape()[j];
                    }
                    res(idx) = r1.cptr()[i];
                }
                //#pragma omp parallel for schedule(static)
                for(INT i=0;i<r2.size();++i)
                {
                    std::array<INT,N> idx;
                    for(INT j=0;j<N;++j)
                    {
                        idx[j] = i / r2.dist()[j] % r2.shape()[j];
                        if(std::find(ax.begin(),ax.end(),j)!=ax.end())
                        {
                            idx[j] += r1.shape()[j];
                        }
                    }
                    res(idx) = r2.cptr()[i];
                }
                return res;
            }
            template<INT N>
            static ArrR<N> ortho(ArrR<N> const & h, ArrR<N> const & u)
            {
                ArrR<N> res(h);
                double ov = Alg::overlap<N>(u,res);
                res -= u * ov;
                return res;
            }
            template<INT N>
            static ArrC<N> ortho(ArrC<N> const & h, ArrC<N> const & u)
            {
                ArrC<N> res(h);
                Complex ov = Alg::overlap<N>(u,res);
                res -= u * ov;
                return res;
            }

            // eigen solver
            // @param rep: replace the H by eigen vectors
            // @return : eigenvalues
            static ArrR<2> eig(ArrR<2> & Hm, char rep = 'r')
            {
                INT Ns = Hm.shape()[0];
                ArrR<1> ega({Ns});
                ArrR<2> egr({Ns,Ns});
                if(rep=='r')
                {
                    int ifeg = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', Ns, Hm.ptr(), Ns, ega.ptr());
                    if(ifeg!=0) { std::cout << "Problems in eig! " << Ns << std::endl; std::exit(1);}
                }
                else
                {
                    ArrR<2> Hmc = Hm;
                    int ifeg = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', Ns, Hmc.ptr(), Ns, ega.ptr());
                    if(ifeg!=0) { std::cout << "Problems in eig! " << Ns << std::endl; std::exit(1);}
                }
                for(INT i=0;i<Ns;++i)
                {
                    egr({i,i}) = ega({i});
                }
                return egr;
            }
            static ArrC<2> eig(ArrC<2> & Hm, char rep = 'r')
            {
                INT Ns = Hm.shape()[0];
                ArrR<1> ega({Ns});
                ArrC<2> egr({Ns,Ns});
                if(rep=='r')
                {
                    int ifeg = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', Ns, Hm.ptr(), Ns, ega.ptr());
                    if(ifeg!=0) { std::cout << "Problems in eig! " << Ns << std::endl; std::exit(1);}
                }
                else
                {
                    ArrC<2> Hmc = Hm;
                    int ifeg = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', Ns, Hmc.ptr(), Ns, ega.ptr());
                    if(ifeg!=0) { std::cout << "Problems in eig! " << Ns << std::endl; std::exit(1);}
                }
                for(INT i=0;i<Ns;++i)
                {
                    egr({i,i}) = ega({i});
                }
                return egr;
            }
            // basic QR decomposition
            template<INT N1, INT N2>
            static std::tuple<ArrR<N1+1>,ArrR<N2+1>> qr(ArrR<N1+N2> & A)
            {
                INT nrow = std::accumulate(A.shape().begin(),A.shape().begin()+N1,1,std::multiplies<INT>());
                INT ncol = A.size() / nrow;
                INT ldu = std::min(nrow,ncol);
                int ifqr1,ifqr2;
                ArrR<1> tau({ldu});
                std::array<INT,N1+1> lsp; std::copy(A.shape().begin(),A.shape().begin()+N1,lsp.begin());
                std::array<INT,N2+1> rsp; std::copy(A.shape().begin()+N1,A.shape().end(),rsp.begin()+1);
                lsp[N1] = ldu; rsp[0] = ldu;
                ArrR<N1+1> Q(lsp); ArrR<N2+1> R(rsp);
                ifqr1 = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, nrow, ncol, A.ptr(), ncol, tau.ptr());
                for(INT i=0;i<ldu;++i)
                {
                    for(INT j=i;j<ncol;++j)
                    {
                        R.ptr()[i*ldu+j] = A.ptr()[i*ncol+j];
                    }
                }
                ifqr2 = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, nrow, ldu, ldu, A.ptr(), ncol, tau.ptr());
                if(ifqr2!=0 || ifqr1!=0) { std::cout << "Problems in QR! " << N1 << " : " << N2 << std::endl; std::exit(1);}
                for(INT i=0;i<ldu;++i) { cblas_dcopy(nrow, A.cptr()+i, ncol, Q.ptr()+i, ldu);}
                return std::make_tuple(Q,R);
            }
            // basic QR decomposition
            template<INT N1, INT N2>
            static std::tuple<ArrC<N1+1>,ArrC<N2+1>> qr(ArrC<N1+N2> & A)
            {
                INT nrow = std::accumulate(A.shape().begin(),A.shape().begin()+N1,1,std::multiplies<INT>());
                INT ncol = A.size() / nrow;
                INT ldu = std::min(nrow,ncol);
                int ifqr1,ifqr2;
                ArrC<1> tau({ldu});
                std::array<INT,N1+1> lsp; std::copy(A.shape().begin(),A.shape().begin()+N1,lsp.begin());
                std::array<INT,N2+1> rsp; std::copy(A.shape().begin()+N1,A.shape().end(),rsp.begin()+1);
                lsp[N1] = ldu; rsp[0] = ldu;
                ArrC<N1+1> Q(lsp); ArrC<N2+1> R(rsp);
                ifqr1 = LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, nrow, ncol, A.ptr(), ncol, tau.ptr());
                for(INT i=0;i<ldu;++i)
                {
                    for(INT j=i;j<ncol;++j)
                    {
                        R.ptr()[i*ldu+j] = A.ptr()[i*ncol+j];
                    }
                }
                ifqr2 = LAPACKE_zungqr(LAPACK_ROW_MAJOR, nrow, ldu, ldu, A.ptr(), ncol, tau.ptr());
                if(ifqr2!=0 || ifqr1!=0) { std::cout << "Problems in QR! " << N1 << " : " << N2 << std::endl; std::exit(1);}
                for(INT i=0;i<ldu;++i) { cblas_zcopy(nrow, A.cptr()+i, ncol, Q.ptr()+i, ldu);}
                return std::make_tuple(Q,R);
            }
            // LQ decomposition
            template<INT N1, INT N2>
            static std::tuple<ArrR<N1+1>,ArrR<N2+1>> lq(ArrR<N1+N2> & A)
            {
                INT nrow = std::accumulate(A.shape().begin(),A.shape().begin()+N1,1,std::multiplies<INT>());
                INT ncol = A.size() / nrow;
                INT ldu = std::min(nrow,ncol);
                int ifqr1,ifqr2;
                ArrR<1> tau({ldu});
                std::array<INT,N1+1> lsp; std::copy(A.shape().begin(),A.shape().begin()+N1,lsp.begin());
                std::array<INT,N2+1> rsp; std::copy(A.shape().begin()+N1,A.shape().end(),rsp.begin()+1);
                lsp[N1] = ldu; rsp[0] = ldu;
                ArrR<N1+1> R(lsp); ArrR<N2+1> Q(rsp);
                ifqr1 = LAPACKE_dgelqf(LAPACK_ROW_MAJOR, nrow, ncol, A.ptr(), ncol, tau.ptr());
                for(INT i=0;i<ldu;++i)
                {
                    for(INT j=i;j<nrow;++j)
                    {
                        R.ptr()[j*ldu+i] = A.ptr()[j*ncol+i];
                    }
                }
                ifqr2 = LAPACKE_dorglq(LAPACK_ROW_MAJOR, ldu, ncol, ldu, A.ptr(), ncol, tau.ptr());
                if(ifqr2!=0 || ifqr1!=0) { std::cout << "Problems in LQ! " << N1 << " : " << N2 << std::endl; std::exit(1);}
                cblas_dcopy(nrow*ncol, A.cptr(), 1, Q.ptr(), 1);
                return std::make_tuple(R,Q);
            }
            // LQ decomposition
            template<INT N1, INT N2>
            static std::tuple<ArrC<N1+1>,ArrC<N2+1>> lq(ArrC<N1+N2> & A)
            {
                INT nrow = std::accumulate(A.shape().begin(),A.shape().begin()+N1,1,std::multiplies<INT>());
                INT ncol = A.size() / nrow;
                INT ldu = std::min(nrow,ncol);
                int ifqr1,ifqr2;
                ArrC<1> tau({ldu});
                std::array<INT,N1+1> lsp; std::copy(A.shape().begin(),A.shape().begin()+N1,lsp.begin());
                std::array<INT,N2+1> rsp; std::copy(A.shape().begin()+N1,A.shape().end(),rsp.begin()+1);
                lsp[N1] = ldu; rsp[0] = ldu;
                ArrC<N1+1> R(lsp); ArrC<N2+1> Q(rsp);
                ifqr1 = LAPACKE_zgelqf(LAPACK_ROW_MAJOR, nrow, ncol, A.ptr(), ncol, tau.ptr());
                for(INT i=0;i<ldu;++i)
                {
                    for(INT j=i;j<nrow;++j)
                    {
                        R.ptr()[j*ldu+i] = A.ptr()[j*ncol+i];
                    }
                }
                ifqr2 = LAPACKE_zunglq(LAPACK_ROW_MAJOR, ldu, ncol, ldu, A.ptr(), ncol, tau.ptr());
                if(ifqr2!=0 || ifqr1!=0) { std::cout << "Problems in LQ! " << N1 << " : " << N2 << std::endl; std::exit(1);}
                cblas_zcopy(nrow*ncol, A.cptr(), 1, Q.ptr(), 1);
                return std::make_tuple(R,Q);
            }
            // matrix inverse
            template<INT N1, INT N2>
            static void inv(ArrR<N1+N2> & m)
            {
                INT nrow = std::accumulate(m.shape().begin(),m.shape().begin()+N1,1,std::multiplies<INT>());
                INT ncol = m.size() / nrow;
                INT * ipiv = (INT *)std::malloc(sizeof(INT)*ncol);
                int iffc = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, nrow, ncol, m.ptr(), ncol, ipiv);
                int ifiv = LAPACKE_dgetri(LAPACK_ROW_MAJOR, ncol, m.ptr(), ncol, ipiv);
                if(iffc!=0 || ifiv!=0) { std::cout << "Problems in Inv! " << N1 << " : " << N2 << std::endl; std::exit(1);}
                std::free(ipiv);
                ipiv = nullptr;
            }
            template<INT N1, INT N2>
            static void inv(ArrC<N1+N2> & m)
            {
                INT nrow = std::accumulate(m.shape().begin(),m.shape().begin()+N1,1,std::multiplies<INT>());
                INT ncol = m.size() / nrow;
                INT * ipiv = (INT *)std::malloc(sizeof(INT)*ncol);
                int iffc = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, nrow, ncol, m.ptr(), ncol, ipiv);
                int ifiv = LAPACKE_zgetri(LAPACK_ROW_MAJOR, ncol, m.ptr(), ncol, ipiv);
                if(iffc!=0 || ifiv!=0) { std::cout << "Problems in Inv! " << N1 << " : " << N2 << std::endl; std::exit(1);}
                std::free(ipiv);
                ipiv = nullptr;
            }
            template<INT N1, INT N2>
            static ArrR<N1+N2> svd_pinv(ArrR<N1+N2> & m)
            {
                INT nrow = std::accumulate(m.shape().begin(),m.shape().begin()+N1,1,std::multiplies<INT>());
                INT ncol = m.size() / nrow;
                INT ldu = std::min(nrow,ncol);
                ArrR<2> u({nrow,ldu}),vt({ldu,ncol}),diags({ldu,ldu});
                ArrR<1> s({ldu}),sp({ldu-1});

                int ifsv = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.ptr(), ncol,
                s.ptr(), u.ptr(), ldu, vt.ptr(), ncol, sp.ptr());
                if(ifsv!=0) { std::cout << "Problems in SVD! " << N1 << " : " << N2 << std::endl; std::exit(1);}
                for(INT i=0;i<ldu;++i)
                {
                    diags({i,i}) = 1.0 / s.cptr()[i];
                }
                ArrR<N2+1> res = Alg::gmm<2,N2+1,1>(vt,diags,CblasTrans,CblasNoTrans);
                res = Alg::gmm<N2+1,N1+1,1>(res,u,CblasNoTrans,CblasTrans);
                return res;
            }
            template<INT N1, INT N2>
            static ArrC<N1+N2> svd_pinv(ArrC<N1+N2> & m)
            {
                INT nrow = std::accumulate(m.shape().begin(),m.shape().begin()+N1,1,std::multiplies<INT>());
                INT ncol = m.size() / nrow;
                INT ldu = std::min(nrow,ncol);
                ArrC<2> u({nrow,ldu}),vt({ldu,ncol}),diags({ldu,ldu});
                ArrR<1> s({ldu}),sp({ldu-1});

                int ifsv = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.ptr(), ncol,
                s.ptr(), u.ptr(), ldu, vt.ptr(), ncol, sp.ptr());
                if(ifsv!=0) { std::cout << "Problems in SVD! " << N1 << " : " << N2 << std::endl; std::exit(1);}
                for(INT i=0;i<ldu;++i)
                {
                    diags({i,i}) = 1.0 / s.cptr()[i];
                }
                ArrC<N2+1> res = Alg::gmm<2,N2+1,1>(vt,diags,CblasTrans,CblasNoTrans);
                res = Alg::gmm<N2+1,N1+1,1>(res,u,CblasNoTrans,CblasTrans);
                return res;
            }
            // singular value decomposition with contraction
            // ReS : rescaled or not
            template<INT N1, INT N2>
            static std::tuple<ArrR<N1+1>,ArrR<N2+1>> svd(ArrR<N1+N2> & m, char l2r, double tol = 0.0, INT md = 1000, char ReS = 'n')
            {
                INT nrow = std::accumulate(m.shape().begin(),m.shape().begin()+N1,1,std::multiplies<INT>());
                INT ncol = m.size() / nrow;
                INT ldu = std::min(nrow,ncol);
                ArrR<N1+N2> mc(m);
                ArrR<2> u({nrow,ldu}),vt({ldu,ncol});
                ArrR<1> s({ldu}),sp({ldu-1});
                std::array<INT,N1+1> lsp; std::copy(m.shape().begin(),m.shape().begin()+N1,lsp.begin());
                std::array<INT,N2+1> rsp; std::copy(m.shape().begin()+N1,m.shape().end(),rsp.begin()+1);

                int ifsv = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.ptr(), ncol, s.ptr(), u.ptr(), ldu, vt.ptr(), ncol, sp.ptr());
                if(ifsv!=0) { std::cout << "Problems in SVD! " << N1 << " : " << N2 << std::endl; std::exit(1);}
                INT nstate = 1; double nrms = 0;
                for(INT i=0;i<ldu;++i)
                {
                    nrms += s.ptr()[i];
                    if(s.ptr()[i]<=tol)
                    {
                        break;
                    }
                    nstate = i + 1;
                }
                nstate = std::min(md,nstate);
                lsp[N1] = nstate; rsp[0] = nstate;
                ArrR<N1+1> lef(lsp); ArrR<N2+1> rig(rsp);
                if(l2r=='r')
                {
                    for(INT i=0;i<nstate;++i)
                    {
                        cblas_dcopy(nrow, u.cptr()+i, ldu, lef.ptr()+i, nstate);
                        cblas_daxpy(ncol, s.cptr()[i], vt.cptr()+i*ncol, 1, rig.ptr()+i*ncol, 1);
                    }
                    /*
                    ArrR<N1+N2> ResDiff = mc-Alg::gmm<N1+1,N2+1,1>(lef,rig);
                    if(Alg::norm<N1+N2>(ResDiff)>=1e-5)
                    {
                        auto[lefc,rigc] = qr<N1,N2>(mc);
                        lef = std::move(lefc);
                        rig = std::move(rigc);
                    }
                    */
                    if(ReS=='r') { lef *= std::sqrt(nrms); rig /= std::sqrt(nrms);}
                }
                else
                {
                    for(INT i=0;i<nstate;++i)
                    {
                        cblas_daxpy(nrow, s.cptr()[i],  u.cptr()+i, ldu, lef.ptr()+i, nstate);
                        cblas_dcopy(ncol, vt.cptr()+i*ncol, 1, rig.ptr()+i*ncol, 1);
                    }
                    // whether svd converges
                    /*
                    ArrR<N1+N2> ResDiff = mc-Alg::gmm<N1+1,N2+1,1>(lef,rig);
                    if(Alg::norm<N1+N2>(ResDiff)>=1e-5)
                    {
                        auto[lefc,rigc] = lq<N1,N2>(mc);
                        lef = std::move(lefc);
                        rig = std::move(rigc);
                    }
                    */
                    if(ReS=='r') { lef /= std::sqrt(nrms); rig *= std::sqrt(nrms); }
                }
                return std::make_tuple(lef,rig);
            }
            template<INT N1, INT N2>
            static std::tuple<ArrC<N1+1>,ArrC<N2+1>> svd(ArrC<N1+N2> & m, char l2r, double tol = 0.0, INT md = 1000, char ReS = 'n')
            {
                INT nrow = std::accumulate(m.shape().begin(),m.shape().begin()+N1,1,std::multiplies<INT>());
                INT ncol = m.size() / nrow;
                INT ldu = std::min(nrow,ncol);
                ArrC<N1+N2> mc(m);
                ArrC<2> u({nrow,ldu}),vt({ldu,ncol});
                ArrR<1> s({ldu}),sp({ldu-1});
                std::array<INT,N1+1> lsp; std::copy(m.shape().begin(),m.shape().begin()+N1,lsp.begin());
                std::array<INT,N2+1> rsp; std::copy(m.shape().begin()+N1,m.shape().end(),rsp.begin()+1);

                int ifsv = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.ptr(), ncol, s.ptr(), u.ptr(), ldu, vt.ptr(), ncol, sp.ptr());
                if(ifsv!=0) { std::cout << "Problems in SVD! " << N1 << " : " << N2 << std::endl; std::exit(1);}
                INT nstate = 1; double nrms = 0;
                for(INT i=0;i<ldu;++i)
                {
                    nrms += s.ptr()[i];
                    if(s.ptr()[i]<=tol)
                    {
                        break;
                    }
                    nstate = i + 1;
                }
                nstate = std::min(md,nstate);
                lsp[N1] = nstate; rsp[0] = nstate;
                ArrC<N1+1> lef(lsp); ArrC<N2+1> rig(rsp);
                if(l2r=='r')
                {
                    for(INT i=0;i<nstate;++i)
                    {
                        Complex si(s.cptr()[i]);
                        cblas_zcopy(nrow, u.cptr()+i, ldu, lef.ptr()+i, nstate);
                        cblas_zaxpy(ncol, &si, vt.cptr()+i*ncol, 1, rig.ptr()+i*ncol, 1);
                    }

                    ArrC<N1+N2> ResDiff = mc-Alg::gmm<N1+1,N2+1,1>(lef,rig);
		    /*
                    if(Alg::norm<N1+N2>(ResDiff)>=1e-5)
                    {
                        auto[lefc,rigc] = qr<N1,N2>(mc);
                        lef = std::move(lefc);
                        rig = std::move(rigc);
                    }
		    */
                    if(ReS=='r') { lef *= std::sqrt(nrms); rig /= std::sqrt(nrms);}
                }
                else
                {
                    for(INT i=0;i<nstate;++i)
                    {
                        Complex si(s.cptr()[i]);
                        cblas_zaxpy(nrow, &si,  u.cptr()+i, ldu, lef.ptr()+i, nstate);
                        cblas_zcopy(ncol, vt.cptr()+i*ncol, 1, rig.ptr()+i*ncol, 1);
                    }
                    // whether svd converges
                    ArrC<N1+N2> ResDiff = mc-Alg::gmm<N1+1,N2+1,1>(lef,rig);
		    /*
                    if(Alg::norm<N1+N2>(ResDiff)>=1e-5)
                    {
                        auto[lefc,rigc] = lq<N1,N2>(mc);
                        lef = std::move(lefc);
                        rig = std::move(rigc);
                    }
		    */
                    if(ReS=='r') { lef /= std::sqrt(nrms); rig *= std::sqrt(nrms); }
                }
                return std::make_tuple(lef,rig);
            }
            // svd without contraction
            template<INT N1, INT N2>
            static std::tuple<ArrR<N1+1>,ArrR<N2+1>> svd_onlybasis(ArrR<N1+N2> & m)
            {
                INT nrow = std::accumulate(m.shape().begin(),m.shape().begin()+N1,1,std::multiplies<INT>());
                INT ncol = m.size() / nrow;
                INT ldu = std::min(nrow,ncol);
                ArrR<N1+N2> mc(m);
                ArrR<2> u({nrow,ldu}),vt({ldu,ncol});
                ArrR<1> s({ldu}),sp({ldu-1});
                std::array<INT,N1+1> lsp; std::copy(m.shape().begin(),m.shape().begin()+N1,lsp.begin());
                std::array<INT,N2+1> rsp; std::copy(m.shape().begin()+N1,m.shape().end(),rsp.begin()+1);

                int ifsv = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.ptr(), ncol, s.ptr(), u.ptr(), ldu, vt.ptr(), ncol, sp.ptr());
                if(ifsv!=0) { std::cout << "Problems in SVD! " << N1 << " : " << N2 << std::endl; std::exit(1);}
                lsp[N1] = ldu; rsp[0] = ldu;
                ArrR<N1+1> lef(lsp); cblas_dcopy(lef.size(),u.cptr(),1,lef.ptr(),1);
                ArrR<N2+1> rig(rsp); cblas_dcopy(rig.size(),vt.cptr(),1,rig.ptr(),1);
                return std::make_tuple(lef,rig);
            }

            // tensor product without transpose
            template<INT N1, INT N2, INT nc>
            static ArrR<N1+N2-2*nc> gmm(ArrR<N1> const & m1, ArrR<N2> const & m2, CBLAS_TRANSPOSE t1 = CblasNoTrans, CBLAS_TRANSPOSE t2 = CblasNoTrans)
            {
                std::array<INT,N1+N2-2*nc> rsp;
                INT nrow, ncol, nK, lda, ldb;
                if(t1==CblasNoTrans)
                {
                    std::copy(m1.shape().begin(), m1.shape().begin()+N1-nc, rsp.begin());
                    nrow = std::accumulate(m1.shape().begin(), m1.shape().begin()+N1-nc, 1, std::multiplies<INT>());
                    lda = m1.size() / nrow;
                }
                else
                {
                    std::copy(m1.shape().begin()+nc, m1.shape().end(), rsp.begin());
                    nrow = std::accumulate(m1.shape().begin()+nc, m1.shape().end(), 1, std::multiplies<INT>());
                    lda = nrow;
                }
                nK = m1.size() / nrow;
                if(t2==CblasNoTrans)
                {
                    std::copy(m2.shape().begin()+nc, m2.shape().end(), rsp.begin()+N1-nc);
                    ncol = std::accumulate(m2.shape().begin()+nc, m2.shape().end(), 1, std::multiplies<INT>());
                    ldb = ncol;
                }
                else
                {
                    std::copy(m2.shape().begin(), m2.shape().begin()+N2-nc, rsp.begin()+N1-nc);
                    ncol = std::accumulate(m2.shape().begin(), m2.shape().begin()+N2-nc, 1, std::multiplies<INT>());
                    ldb = m2.size() / ncol;
                }
                if(nK!=m2.size()/ncol) { std::cout << N1 << ":" << N2 << ":" << nc << " Gmm product shapes mismatch!" << std::endl;
                m1.print_sp(1e9); m2.print_sp(1e9);
                std::exit(1); }
                double ones(1.0),zeros(0.0);
                ArrR<N1+N2-2*nc> res(rsp);
                cblas_dgemm(CblasRowMajor, t1, t2, nrow, ncol, nK, ones, m1.cptr(), lda, m2.cptr(), ldb, zeros, res.ptr(), ncol);
                return res;
            }
            template<INT N1, INT N2, INT nc>
            static ArrC<N1+N2-2*nc> gmm(ArrC<N1> const & m1, ArrC<N2> const & m2, CBLAS_TRANSPOSE t1 = CblasNoTrans, CBLAS_TRANSPOSE t2 = CblasNoTrans)
            {
                std::array<INT,N1+N2-2*nc> rsp;
                INT nrow, ncol, nK, lda, ldb;
                if(t1==CblasNoTrans)
                {
                    std::copy(m1.shape().begin(), m1.shape().begin()+N1-nc, rsp.begin());
                    nrow = std::accumulate(m1.shape().begin(), m1.shape().begin()+N1-nc, 1, std::multiplies<INT>());
                    lda = m1.size() / nrow;
                }
                else
                {
                    std::copy(m1.shape().begin()+nc, m1.shape().end(), rsp.begin());
                    nrow = std::accumulate(m1.shape().begin()+nc, m1.shape().end(), 1, std::multiplies<INT>());
                    lda = nrow;
                }
                nK = m1.size() / nrow;
                if(t2==CblasNoTrans)
                {
                    std::copy(m2.shape().begin()+nc, m2.shape().end(), rsp.begin()+N1-nc);
                    ncol = std::accumulate(m2.shape().begin()+nc, m2.shape().end(), 1, std::multiplies<INT>());
                    ldb = ncol;
                }
                else
                {
                    std::copy(m2.shape().begin(), m2.shape().begin()+N2-nc, rsp.begin()+N1-nc);
                    ncol = std::accumulate(m2.shape().begin(), m2.shape().begin()+N2-nc, 1, std::multiplies<INT>());
                    ldb = m2.size() / ncol;
                }
                if(nK!=m2.size()/ncol) { std::cout << N1 << ":" << N2 << ":" << nc << " Gmm product shapes mismatch!" << std::endl;
                m1.print_sp(1e9); m2.print_sp(1e9);
                std::exit(1); }
                Complex ones(1.0),zeros(0.0);
                ArrC<N1+N2-2*nc> res(rsp);
                cblas_zgemm(CblasRowMajor, t1, t2, nrow, ncol, nK, &ones, m1.cptr(), lda, m2.cptr(), ldb, &zeros, res.ptr(), ncol);
                return res;
            }
            // tensor product with transpose
            template<INT N1, INT N2, INT nc>
            static ArrR<N1+N2-2*nc> gtmm(ArrR<N1> const & m1, ArrR<N2> const & m2, Brace ax1, Brace ax2, CBLAS_TRANSPOSE t1 = CblasNoTrans,
            CBLAS_TRANSPOSE t2 = CblasNoTrans)
            {
                ArrR<N1> m1t = transpose(m1, ax1);
                ArrR<N2> m2t = transpose(m2, ax2);
                return gmm<N1,N2,nc>(m1t,m2t,t1,t2);
            }
            template<INT N1, INT N2, INT nc>
            static ArrC<N1+N2-2*nc> gtmm(ArrC<N1> const & m1, ArrC<N2> const & m2, Brace ax1, Brace ax2, CBLAS_TRANSPOSE t1 = CblasNoTrans,
            CBLAS_TRANSPOSE t2 = CblasNoTrans)
            {
                ArrC<N1> m1t = transpose(m1, ax1);
                ArrC<N2> m2t = transpose(m2, ax2);
                return gmm<N1,N2,nc>(m1t,m2t,t1,t2);
            }

            // extract all [sig] (aj-1,aj) from mps tensor
            static std::vector<ArrR<2>> all_levels(ArrR<3> const & mps)
            {
                std::vector<ArrR<2>> res(mps.shape()[1]);
                ArrR<3> mpst = transpose<3>(mps,{1,0,2});
                for(INT i(0);i<res.size();++i)
                {
                    ArrR<2> ri({mps.shape()[0],mps.shape()[2]});
                    cblas_dcopy(ri.size(), mpst.cptr()+i*ri.size(),1, ri.ptr(),1);
                    res[i] = std::move(ri);
                }
                return res;
            }
            static std::vector<ArrC<2>> all_levels(ArrC<3> const & mps)
            {
                std::vector<ArrC<2>> res(mps.shape()[1]);
                ArrC<3> mpst = transpose<3>(mps,{1,0,2});
                for(INT i(0);i<res.size();++i)
                {
                    ArrC<2> ri({mps.shape()[0],mps.shape()[2]});
                    cblas_zcopy(ri.size(), mpst.cptr()+i*ri.size(),1, ri.ptr(),1);
                    res[i] = std::move(ri);
                }
                return res;
            }
            // stack all levels together
            static ArrR<3> comb_levels(std::vector<ArrR<2>> const & vecs)
            {
                INT Nmat = vecs.size();
                INT nrow(vecs[0].shape()[0]), ncol(vecs[0].shape()[1]);
                ArrR<3> res({Nmat,nrow,ncol});
                for(INT i(0);i<Nmat;++i)
                {
                    cblas_dcopy(nrow*ncol, vecs[i].cptr(), 1, res.ptr()+i*nrow*ncol,1);
                }
                res = Alg::transpose<3>(res,{1,0,2});
                return res;
            }
            // stack all levels together
            static ArrC<3> comb_levels(std::vector<ArrC<2>> const & vecs)
            {
                INT Nmat = vecs.size();
                INT nrow(vecs[0].shape()[0]), ncol(vecs[0].shape()[1]);
                ArrC<3> res({Nmat,nrow,ncol});
                for(INT i(0);i<Nmat;++i)
                {
                    cblas_zcopy(nrow*ncol, vecs[i].cptr(), 1, res.ptr()+i*nrow*ncol,1);
                }
                res = Alg::transpose<3>(res,{1,0,2});
                return res;
            }

            template<INT N>
            SparR<N> transpose(SparR<N> const & r, Brace ax)
            {
                std::array<INT,N> rsp;
                for(INT i=0;i<N;++i)
                {
                    rsp[i] = r.shape()[*(ax.begin()+i)];
                }
                SparR<N> res(rsp,r.size());
                for(INT i=0;i<r.size();++i)
                {
                    std::array<INT,N> idx;
                    for(INT j=0;j<N;++j)
                    {
                        idx[j] = r[i][*(ax.begin()+j)];
                    }
                    res[i] = idx;
                    res.values()[i] = r.values()[i];
                }
                return res;
            }

            // mm product of two sparses
            template<INT N1, INT N2, INT nc>
            SparR<N1+N2-2*nc> gmm(SparR<N1> const & m1, SparR<N2> const & m2)
            {
                INT nz1 = m1.size(), nz2 = m2.size(), nth;
                std::array<INT,N1+N2-2*nc> rsp;
                for(INT i=0;i<N1-nc;++i)
                {
                    rsp[i] = m1.shape()[i];
                }
                for(INT i=0;i<N2-nc;++i)
                {
                    rsp[ii+N1-nc] = m2.shape()[i+nc];
                }
                for(INT i=0;i<nc;++i)
                {
                    if(m1.shape()[i+N1-nc] != m2.shape()[i])
                    {
                        std::cout << "Sparse shape mismatch!" << std::endl;
                        std::exit(1);
                    }
                }
                #pragma omp parallel
                nth = omp_get_num_threads()

                std::vector<SparR<N1+N2-2*nc>> ress(nth,rsp);

                #pragma omp parallel for
                for(INT i=0;i<nz1*nz2;++i)
                {
                    INT i1 = i / nz2, i2 = i % nz2;
                    char IsMatch = 'y';
                    for(INT j=0;j<nc;++j)
                    {
                        if(m1[i1][j+N1-nc] != m2[i2][j])
                        {
                            IsMatch = 'n';
                            break;
                        }
                    }
                    std::array<INT,N1+N2-nc> idx;
                    if(IsMatch=='y')
                    {
                        for(INT j=0;j<N1-nc;++j)
                        {
                            idx[j] = m1.shape()[j];
                        }
                        for(INT j=0;j<N2-nc;++j)
                        {
                            idx[j+N1-nc] = m2.shape()[j+nc];
                        }
                        INT ThreadID = omp_get_thread_num();
                        ress[ThreadID].add_elem(idx,m1.values()[i1]*m2.values()[i2]);
                    }
                }
                SparR<N1+N2-2*nc> res(rsp);
                for(INT i=0;i<nth;++i)
                {
                    res += ress[i];
                }
                return res;
            }
        };
    }
}
