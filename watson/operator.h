/* watson hamiltonian local operators with exact high-lying configurations */

#pragma once

#include <tuple>
#include <unordered_map>
#include "sparse_base.h"

namespace KylinVib
{
    using std::sqrt;
    using std::tuple;
    using std::pow;
    using std::make_tuple;
    using std::hash;
    using std::unordered_map;
    using std::get;
    using std::inner_product;
    namespace Watson
    {
        enum class LocalOp
        {
            I,Q,P,QP,PQ,Q2,P2,
            Q3,Q4,Q5,Q6
        };
        // output
        ostream & operator<<(ostream & os, LocalOp op)
        {
            switch (op)
            {
            case LocalOp::I:
                os << "I";
                break;
            case LocalOp::Q:
                os << "Q";
                break;
            case LocalOp::P:
                os << "P";
                break;
            case LocalOp::QP:
                os << "QP";
                break;
            case LocalOp::PQ:
                os << "PQ";
                break;
            case LocalOp::Q2:
                os << "Q2";
                break;
            case LocalOp::P2:
                os << "P2";
                break;
            case LocalOp::Q3:
                os << "Q3";
                break;
            case LocalOp::Q4:
                os << "Q4";
                break;
            case LocalOp::Q5:
                os << "Q5";
                break;
            case LocalOp::Q6:
                os << "Q6";
                break;
            default:
                break;
            }
            return os;
        }
        // transform to mpo-matrix
        SparseBase<double,4> to_matrix(LocalOp op, size_t dim)
        {
            SparseBase<double,4> res({1,dim,dim,1});
            size_t lev = 0;
            switch (op)
            {
            case LocalOp::I:
                for(lev=0;lev<dim;++lev)
                {
                    res.push_back({0,lev,lev,0}, 1.0);
                }
                break;
            case LocalOp::Q:
                for(lev=0;lev<dim-1;++lev)
                {
                    res.push_back({0,lev,lev+1,0}, sqrt(0.5*lev+0.5));
                    res.push_back({0,lev+1,lev,0}, sqrt(0.5*lev+0.5));
                }
                break;
            case LocalOp::P:
                for(lev=0;lev<dim-1;++lev)
                {
                    res.push_back({0,lev,lev+1,0}, sqrt(0.5*lev+0.5));
                    res.push_back({0,lev+1,lev,0}, -1.0*sqrt(0.5*lev+0.5));
                }
                break;
            case LocalOp::QP:
                for(lev=0;lev<dim-2;++lev)
                {
                    res.push_back({0,lev,lev+2,0}, sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0));
                    res.push_back({0,lev+2,lev,0}, -1.0*sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0));
                }
                for(lev=0;lev<dim;++lev)
                {
                    res.push_back({0,lev,lev,0}, -0.5);
                }
                break;
            case LocalOp::PQ:
                for(lev=0;lev<dim-2;++lev)
                {
                    res.push_back({0,lev,lev+2,0},sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0));
                    res.push_back({0,lev+2,lev,0}, -1.0*sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0));
                }
                for(lev=0;lev<dim;++lev)
                {
                    res.push_back({0,lev,lev,0}, 0.5);
                }
                break;
            case LocalOp::Q2:
                for(lev=0;lev<dim-2;++lev)
                {
                    res.push_back({0,lev,lev+2,0}, sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0));
                    res.push_back({0,lev+2,lev,0}, sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0));
                }
                for(lev=0;lev<dim;++lev)
                {
                    res.push_back({0,lev,lev,0}, 0.5*(2*lev+1.0));
                }
                break;
            case LocalOp::P2:
                for(lev=0;lev<dim-2;++lev)
                {
                    res.push_back({0,lev,lev+2,0}, sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0));
                    res.push_back({0,lev+2,lev,0}, sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0));
                }
                for(lev=0;lev<dim;++lev)
                {
                    res.push_back({0,lev,lev,0}, -0.5*(2*lev+1.0));
                }
                break;
            case LocalOp::Q3:
                for(lev=0;lev<dim-3;++lev)
                {
                    res.push_back({0,lev,lev+3,0}, sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0)*
                    sqrt(0.5*lev+1.5));
                    res.push_back({0,lev+3,lev,0}, sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0)*
                    sqrt(0.5*lev+1.5));
                }
                for(lev=0;lev<dim-1;++lev)
                {
                    res.push_back({0,lev,lev+1,0}, 0.5*(3*lev+3.0)*sqrt(lev*0.5+0.5));
                    res.push_back({0,lev+1,lev,0}, 0.5*(3*lev+3.0)*sqrt(lev*0.5+0.5));
                }
                break;
            case LocalOp::Q4:
                for(lev=0;lev<dim-4;++lev)
                {
                    res.push_back({0,lev,lev+4,0},  sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0)*
                    sqrt(0.5*lev+1.5)*sqrt(0.5*lev+2.0));
                    res.push_back({0,lev+4,lev,0},  sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0)*
                    sqrt(0.5*lev+1.5)*sqrt(0.5*lev+2.0));
                }
                for(lev=0;lev<dim-2;++lev)
                {
                    res.push_back({0,lev,lev+2,0}, (2*lev+3.0)*sqrt(lev*0.5+0.5)*sqrt(lev*0.5+1.0));
                    res.push_back({0,lev+2,lev,0}, (2*lev+3.0)*sqrt(lev*0.5+0.5)*sqrt(lev*0.5+1.0));
                }
                for(lev=0;lev<dim;++lev)
                {
                    res.push_back({0,lev,lev,0}, 0.25*(6*lev*lev+6*lev+3.0));
                }
                break;
            case LocalOp::Q5:
                for(lev=0;lev<dim-5;++lev)
                {
                    res.push_back({0,lev,lev+5,0}, sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0)*
                    sqrt(0.5*lev+1.5)*sqrt(0.5*lev+2.0)*sqrt(0.5*lev+2.5));
                    res.push_back({0,lev+5,lev,0}, sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0)*
                    sqrt(0.5*lev+1.5)*sqrt(0.5*lev+2.0)*sqrt(0.5*lev+2.5));
                }
                for(lev=0;lev<dim-3;++lev)
                {
                    res.push_back({0,lev,lev+3,0}, (2.5*lev+5.0)*sqrt(lev*0.5+0.5)*sqrt(lev*0.5+1.0)
                    *sqrt(lev*0.5+1.5));
                    res.push_back({0,lev+3,lev,0}, (2.5*lev+5.0)*sqrt(lev*0.5+0.5)*sqrt(lev*0.5+1.0)
                    *sqrt(lev*0.5+1.5));
                }
                for(lev=0;lev<dim-1;++lev)
                {
                    res.push_back({0,lev,lev+1,0}, 0.25*(10*lev*lev+20*lev+15.0)*sqrt(0.5*lev+0.5));
                    res.push_back({0,lev+1,lev,0}, 0.25*(10*lev*lev+20*lev+15.0)*sqrt(0.5*lev+0.5));
                }
                break;
            case LocalOp::Q6:
                for(lev=0;lev<dim-6;++lev)
                {
                    res.push_back({0,lev,lev+6,0}, sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0)*
                    sqrt(0.5*lev+1.5)*sqrt(0.5*lev+2.0)*sqrt(0.5*lev+2.5)*
                    sqrt(0.5*lev+3.0));
                    res.push_back({0,lev+6,lev,0}, sqrt(0.5*lev+0.5)*sqrt(0.5*lev+1.0)*
                    sqrt(0.5*lev+1.5)*sqrt(0.5*lev+2.0)*sqrt(0.5*lev+2.5)*
                    sqrt(0.5*lev+3.0));
                }
                for(lev=0;lev<dim-4;++lev)
                {
                    res.push_back({0,lev,lev+4,0}, (3.0*lev+7.5)*sqrt(lev*0.5+0.5)*sqrt(lev*0.5+1.0)
                    *sqrt(lev*0.5+1.5)*sqrt(lev*0.5+2.0));
                    res.push_back({0,lev+4,lev,0}, (3.0*lev+7.5)*sqrt(lev*0.5+0.5)*sqrt(lev*0.5+1.0)
                    *sqrt(lev*0.5+1.5)*sqrt(lev*0.5+2.0));
                }
                for(lev=0;lev<dim-2;++lev)
                {
                    res.push_back({0,lev,lev+2,0}, 0.25*(15*lev*lev+45*lev+45.0)*sqrt(0.5*lev+0.5)
                    *sqrt(0.5*lev+1.0));
                    res.push_back({0,lev+2,lev,0}, 0.25*(15*lev*lev+45*lev+45.0)*sqrt(0.5*lev+0.5)
                    *sqrt(0.5*lev+1.0));
                }
                for(lev=0;lev<dim;++lev)
                {
                    res.push_back({0,lev,lev,0}, 0.125*(20*lev*lev*lev+30*lev*lev+40*lev+15.0));
                }
                break;
            default:
                break;
            }
            return res;
        }
        // define multiplication
        LocalOp apply_lop(LocalOp lhs, LocalOp rhs)
        {
            LocalOp res = LocalOp::I;
            switch (rhs)
            {
            case LocalOp::I:
                res = lhs;
                break;
            case LocalOp::Q:
                switch (lhs)
                {
                case LocalOp::I:
                    res = LocalOp::Q;
                    break;
                case LocalOp::Q:
                    res = LocalOp::Q2;
                    break;
                case LocalOp::P:
                    res = LocalOp::QP;
                    break;
                case LocalOp::Q2:
                    res = LocalOp::Q3;
                    break;
                case LocalOp::Q3:
                    res = LocalOp::Q4;
                    break;
                case LocalOp::Q4:
                    res = LocalOp::Q5;
                    break;
                case LocalOp::Q5:
                    res = LocalOp::Q6;
                    break;
                default:
                    break;
                };
                break;
            case LocalOp::P:
                switch (lhs)
                {
                case LocalOp::I:
                    res = LocalOp::P;
                    break;
                case LocalOp::Q:
                    res = LocalOp::PQ;
                    break;
                case LocalOp::P:
                    res = LocalOp::P2;
                    break;
                default:
                    break;
                };
                break;
            default:
                break;
            }
            return res;
        }
        // stack two sparse mpo tensors
        SparseBase<double,4> stack(SparseBase<double,4> const & r1, SparseBase<double,4> const & r2, char l2r = 'r')
        {
            SparseBase<double,4> res(r1);
            if(l2r=='r')
            {
                res.shape()[3] += r2.shape()[3];
                for(size_t x2=0;x2<r2.nnz();++x2)
                {
                    res.push_back({r2.indices(x2)[0],r2.indices(x2)[1],r2.indices(x2)[2],r2.indices(x2)[3]+r1.shape()[3]},
                    r2.values(x2));
                }
            }
            else if(l2r=='l')
            {
                res.shape()[0] += r2.shape()[0];
                for(size_t x2=0;x2<r2.nnz();++x2)
                {
                    res.push_back({r2.indices(x2)[0]+r1.shape()[0],r2.indices(x2)[1],r2.indices(x2)[2],r2.indices(x2)[3]},
                    r2.values(x2));
                }
            }
            else
            {
                res.shape()[0] += r2.shape()[0];
                res.shape()[3] += r2.shape()[3];
                for(size_t x2=0;x2<r2.nnz();++x2)
                {
                    res.push_back({r2.indices(x2)[0]+r1.shape()[0],r2.indices(x2)[1],r2.indices(x2)[2],r2.indices(x2)[3]+r1.shape()[3]}, r2.values(x2));
                }
            }
            return res;
        }
        // slice
        vector<SparseBase<double,3>> slice(SparseBase<double,4> const & mpo, char l2r = 'r')
        {
            if(l2r=='r')
            {
                vector<SparseBase<double,3>> res(mpo.shape()[3],{mpo.shape()[0],mpo.shape()[1],mpo.shape()[2]});
                for(size_t i=0;i<mpo.nnz();++i)
                {
                    res[mpo.indices(i)[3]].push_back({mpo.indices(i)[0],mpo.indices(i)[1],mpo.indices(i)[2]}, mpo.values(i));
                }
                return res;
            }
            else
            {
                vector<SparseBase<double,3>> res(mpo.shape()[0],{mpo.shape()[1],mpo.shape()[2],mpo.shape()[3]});
                for(size_t i=0;i<mpo.nnz();++i)
                {
                    res[mpo.indices(i)[0]].push_back({mpo.indices(i)[1],mpo.indices(i)[2],mpo.indices(i)[3]}, mpo.values(i));
                }
                return res;
            }
        }
        // deparallel
        tuple<SparseBase<double,4>,SparseBase<double,2>> deparallel(SparseBase<double,4> const & mpo, char l2r = 'r')
        {
            vector<SparseBase<double,3>> slc = slice(mpo,l2r), slct;
            vector<tuple<size_t,double>> mpairs;
            if(l2r=='r')
            {
                for(size_t i=0;i<slc.size();++i)
                {
                    char fd = 'n';
                    double ns1 = slc[i].norm();
                    for(size_t j=0;j<slct.size();++j)
                    {
                        double ns2  = slct[j].norm();
                        double ov12 = slc[i].overlap(slct[j]);
                        if(abs(1.0-ov12/ns1/ns2)<=1e-12)
                        {
                            fd = 'y';
                            mpairs.push_back(make_tuple(j,ns1/ns2));
                            break;
                        }
                    }
                    if(fd=='n')
                    {
                        mpairs.push_back(make_tuple(slct.size(),1.0));
                        slct.push_back(slc[i]);
                    }
                }
                SparseBase<double,4> cur({mpo.shape()[0],mpo.shape()[1],mpo.shape()[2],slct.size()});
                SparseBase<double,2> nex({slct.size(),mpo.shape()[3]});
                for(size_t i=0;i<slct.size();++i)
                {
                    for(size_t j=0;j<slct[i].nnz();++j)
                    {
                        cur.push_back({slct[i].indices(j)[0],slct[i].indices(j)[1],slct[i].indices(j)[2],i},
                        slct[i].values(j));
                    }
                }
                for(size_t i=0;i<mpairs.size();++i)
                {
                    nex.push_back({get<0>(mpairs[i]),i}, get<1>(mpairs[i]));
                }
                return make_tuple(cur,nex);
            }
            else
            {
                for(size_t i=0;i<slc.size();++i)
                {
                    char fd = 'n';
                    double ns1 = slc[i].norm();
                    for(size_t j=0;j<slct.size();++j)
                    {
                        double ns2  = slct[j].norm();
                        double ov12 = slc[i].overlap(slct[j]);
                        if(abs(1.0-ov12/ns1/ns2)<=1e-12)
                        {
                            fd = 'y';
                            mpairs.push_back(make_tuple(j,ns1/ns2));
                            break;
                        }
                    }
                    if(fd=='n')
                    {
                        mpairs.push_back(make_tuple(slct.size(),1.0));
                        slct.push_back(slc[i]);
                    }
                }
                SparseBase<double,4> cur({slct.size(),mpo.shape()[1],mpo.shape()[2],mpo.shape()[3]});
                SparseBase<double,2> nex({mpo.shape()[0],slct.size()});
                for(size_t i=0;i<slct.size();++i)
                {
                    for(size_t j=0;j<slct[i].nnz();++j)
                    {
                        cur.push_back({i,slct[i].indices(j)[0],slct[i].indices(j)[1],slct[i].indices(j)[2]},
                        slct[i].values(j));
                    }
                }
                for(size_t i=0;i<mpairs.size();++i)
                {
                    nex.push_back({i,get<0>(mpairs[i])}, get<1>(mpairs[i]));
                }
                return make_tuple(cur,nex);
            }
        }
        // contract bond-site in mpo
        SparseBase<double,4> contract(SparseBase<double,2> const & bond, SparseBase<double,4> const & nex, char l2r = 'r')
        {
            if(l2r=='r')
            {
                SparseBase<double,4> res({bond.shape()[0],nex.shape()[1],nex.shape()[2],nex.shape()[3]});
                array<size_t,4> Dist = { nex.shape()[1]*nex.shape()[2]*nex.shape()[3], nex.shape()[2]*nex.shape()[3],
                nex.shape()[3], 1};
                unordered_map<size_t,size_t> refs;
                for(size_t bi=0;bi<bond.nnz();++bi)
                {
                    for(size_t ni=0;ni<nex.nnz();++ni)
                    {
                        if( bond.indices(bi)[1] == nex.indices(ni)[0] )
                        {
                            array<size_t,4> ridx = { bond.indices(bi)[0], nex.indices(ni)[1], nex.indices(ni)[2], nex.indices(ni)[3] };
                            size_t pos = inner_product(ridx.begin(), ridx.end(), Dist.begin(), 0);
                            if( refs.find(pos) == refs.end() )
                            {
                                refs[pos] = res.nnz();
                                res.push_back(ridx, bond.values(bi)*nex.values(ni));
                            }
                            else
                            {
                                res.values(refs[pos]) += bond.values(bi)*nex.values(ni);
                            }
                        }
                    }
                }
                return res;
            }
            else
            {
                SparseBase<double,4> res({nex.shape()[0],nex.shape()[1],nex.shape()[2],bond.shape()[1]});
                array<size_t,4> Dist = { nex.shape()[1]*nex.shape()[2]*bond.shape()[1], nex.shape()[2] * bond.shape()[1],
                bond.shape()[1], 1};
                unordered_map<size_t,size_t> refs;
                for(size_t bi=0;bi<bond.nnz();++bi)
                {
                    for(size_t ni=0;ni<nex.nnz();++ni)
                    {
                        if( bond.indices(bi)[0] == nex.indices(ni)[3] )
                        {
                            array<size_t,4> ridx = {nex.indices(ni)[0], nex.indices(ni)[1], nex.indices(ni)[2], bond.indices(bi)[1]};
                            size_t pos = inner_product(ridx.begin(), ridx.end(), Dist.begin(), 0);
                            if( refs.find(pos) == refs.end() )
                            {
                                refs[pos] = res.nnz();
                                res.push_back(ridx, bond.values(bi)*nex.values(ni));
                            }
                            else
                            {
                                res.values(refs[pos]) += bond.values(bi)*nex.values(ni);
                            }
                        }
                    }
                }
                return res;
            }
        }
    }
}
