/* All kinds of local operators defined here */

#pragma once

#include "linalg.h"

namespace KylinVib
{
    namespace Vibronic
    {
        enum class LocalOp
        {
            I,Upper,Lower,N,Q,Q2
        };
        // output
        std::ostream & operator<<(std::ostream & os, LocalOp op)
        {
            switch (op)
            {
            case LocalOp::I:
                os << "I";
                break;
            case LocalOp::Upper:
                os << "a^{+}";
                break;
            case LocalOp::Lower:
                os << "a";
                break;
            case LocalOp::Q:
                os << "Q";
                break;
            case LocalOp::Q2:
                os << "Q2";
                break;
            case LocalOp::N:
                os << "N";
                break;
            default:
                break; 
            }
            return os;
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
                default:
                    break;
                }
                break;
            case LocalOp::Upper:
                switch (lhs)
                {
                case LocalOp::I:
                    res = LocalOp::Q;
                    break;
                case LocalOp::Lower:
                    res = LocalOp::N;
                    break;
                default:
                    break;
                }
                break;
            default:
                break;
            }
            return res;
        }
    }
    namespace Heisenberg2
    {
        enum class LocalOp
        {
            I,Sp,Sm,Sz,
        };
        // output
        std::ostream & operator<<(std::ostream & os, LocalOp op)
        {
            switch (op)
            {
            case LocalOp::I:
                os << "I";
                break;
            case LocalOp::Sp:
                os << "Sp";
                break;
            case LocalOp::Sm:
                os << "Sm";
                break;
            case LocalOp::Sz:
                os << "Sz";
                break;
            default:
                break;
            }
            return os;
        }
        // transform to mpo-matrix
        template<typename ArrType = Dense<double,4>>
        ArrType to_matrix(LocalOp op)
        {
            ArrType res({1,2,2,1});
            size_t lev = 0;
            switch (op)
            {
            case LocalOp::I:
                res({0,0,0,0}) = 1.0;
                res({0,1,1,0}) = 1.0;
                break;
            case LocalOp::Sp:
                res({0,1,0,0}) = 1.0;
                break;
            case LocalOp::Sm:
                res({0,0,1,0}) = 1.0;
                break;
            case LocalOp::Sz:
                res({0,0,0,0}) = 0.5;
                res({0,1,1,0}) = -0.5;
                break;
            default:
                break;
            }
            return res;
        }
    }
    namespace Watson
    {
        enum class LocalOp
        {
            I,Q,P,QP,PQ,Q2,P2,
            Q3,Q4,Q5,Q6
        };
        // output
        std::ostream & operator<<(std::ostream & os, LocalOp op)
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
        template<typename ArrType = Dense<double,4>>
        ArrType to_matrix(LocalOp op, size_t dim)
        {
            ArrType res({1,dim,dim,1});
            size_t lev = 0;
            switch (op)
            {
            case LocalOp::I:
                for(lev=0;lev<dim;++lev)
                {
                    res({0,lev,lev,0}) = 1.0;
                }
                break;
            case LocalOp::Q:
                for(lev=0;lev<dim-1;++lev)
                {
                    res({0,lev,lev+1,0}) = std::sqrt(0.5*lev+0.5);
                    res({0,lev+1,lev,0}) = std::sqrt(0.5*lev+0.5);
                }
                break;
            case LocalOp::P:
                for(lev=0;lev<dim-1;++lev)
                {
                    res({0,lev,lev+1,0}) = std::sqrt(0.5*lev+0.5);
                    res({0,lev+1,lev,0}) = -1.0*std::sqrt(0.5*lev+0.5);
                }
                break;
            case LocalOp::QP:
                for(lev=0;lev<dim-2;++lev)
                {
                    res({0,lev,lev+2,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0);
                    res({0,lev+2,lev,0}) = -1.0*std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0);
                }
                for(lev=0;lev<dim;++lev)
                {
                    res({0,lev,lev,0}) = -0.5;
                }
                break;
            case LocalOp::PQ:
                for(lev=0;lev<dim-2;++lev)
                {
                    res({0,lev,lev+2,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0);
                    res({0,lev+2,lev,0}) = -1.0*std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0);
                }
                for(lev=0;lev<dim;++lev)
                {
                    res({0,lev,lev,0}) = 0.5;
                }
                break;
            case LocalOp::Q2:
                for(lev=0;lev<dim-2;++lev)
                {
                    res({0,lev,lev+2,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0);
                    res({0,lev+2,lev,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0);
                }
                for(lev=0;lev<dim;++lev)
                {
                    res({0,lev,lev,0}) = 0.5*(2*lev+1.0);
                }
                break;
            case LocalOp::P2:
                for(lev=0;lev<dim-2;++lev)
                {
                    res({0,lev,lev+2,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0);
                    res({0,lev+2,lev,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0);
                }
                for(lev=0;lev<dim;++lev)
                {
                    res({0,lev,lev,0}) = -0.5*(2*lev+1.0);
                }
                break;
            case LocalOp::Q3:
                for(lev=0;lev<dim-3;++lev)
                {
                    res({0,lev,lev+3,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0)*
                    std::sqrt(0.5*lev+1.5);
                    res({0,lev+3,lev,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0)*
                    std::sqrt(0.5*lev+1.5);
                }
                for(lev=0;lev<dim-1;++lev)
                {
                    res({0,lev,lev+1,0}) = 0.5*(3*lev+3.0)*std::sqrt(lev*0.5+0.5);
                    res({0,lev+1,lev,0}) = 0.5*(3*lev+3.0)*std::sqrt(lev*0.5+0.5);
                }
                break;
            case LocalOp::Q4:
                for(lev=0;lev<dim-4;++lev)
                {
                    res({0,lev,lev+4,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0)*
                    std::sqrt(0.5*lev+1.5)*std::sqrt(0.5*lev+2.0);
                    res({0,lev+4,lev,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0)*
                    std::sqrt(0.5*lev+1.5)*std::sqrt(0.5*lev+2.0);
                }
                for(lev=0;lev<dim-2;++lev)
                {
                    res({0,lev,lev+2,0}) = (2*lev+3.0)*std::sqrt(lev*0.5+0.5)*std::sqrt(lev*0.5+1.0);
                    res({0,lev+2,lev,0}) = (2*lev+3.0)*std::sqrt(lev*0.5+0.5)*std::sqrt(lev*0.5+1.0);
                }
                for(lev=0;lev<dim;++lev)
                {
                    res({0,lev,lev,0}) = 0.25*(6*lev*lev+6*lev+3.0);
                }
                break;
            case LocalOp::Q5:
                for(lev=0;lev<dim-5;++lev)
                {
                    res({0,lev,lev+5,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0)*
                    std::sqrt(0.5*lev+1.5)*std::sqrt(0.5*lev+2.0)*std::sqrt(0.5*lev+2.5);
                    res({0,lev+5,lev,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0)*
                    std::sqrt(0.5*lev+1.5)*std::sqrt(0.5*lev+2.0)*std::sqrt(0.5*lev+2.5);
                }
                for(lev=0;lev<dim-3;++lev)
                {
                    res({0,lev,lev+3,0}) = (2.5*lev+5.0)*std::sqrt(lev*0.5+0.5)*std::sqrt(lev*0.5+1.0)
                    *std::sqrt(lev*0.5+1.5);
                    res({0,lev+3,lev,0}) = (2.5*lev+5.0)*std::sqrt(lev*0.5+0.5)*std::sqrt(lev*0.5+1.0)
                    *std::sqrt(lev*0.5+1.5);
                }
                for(lev=0;lev<dim-1;++lev)
                {
                    res({0,lev,lev+1,0}) = 0.25*(10*lev*lev+20*lev+15.0)*std::sqrt(0.5*lev+0.5);
                    res({0,lev+1,lev,0}) = 0.25*(10*lev*lev+20*lev+15.0)*std::sqrt(0.5*lev+0.5);
                }
                break;
            case LocalOp::Q6:
                for(lev=0;lev<dim-6;++lev)
                {
                    res({0,lev,lev+6,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0)*
                    std::sqrt(0.5*lev+1.5)*std::sqrt(0.5*lev+2.0)*std::sqrt(0.5*lev+2.5)*
                    std::sqrt(0.5*lev+3.0);
                    res({0,lev+6,lev,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0)*
                    std::sqrt(0.5*lev+1.5)*std::sqrt(0.5*lev+2.0)*std::sqrt(0.5*lev+2.5)*
                    std::sqrt(0.5*lev+3.0);
                }
                for(lev=0;lev<dim-4;++lev)
                {
                    res({0,lev,lev+4,0}) = (3.0*lev+7.5)*std::sqrt(lev*0.5+0.5)*std::sqrt(lev*0.5+1.0)
                    *std::sqrt(lev*0.5+1.5)*std::sqrt(lev*0.5+2.0);
                    res({0,lev+4,lev,0}) = (3.0*lev+7.5)*std::sqrt(lev*0.5+0.5)*std::sqrt(lev*0.5+1.0)
                    *std::sqrt(lev*0.5+1.5)*std::sqrt(lev*0.5+2.0);
                }
                for(lev=0;lev<dim-2;++lev)
                {
                    res({0,lev,lev+2,0}) = 0.25*(15*lev*lev+45*lev+45.0)*std::sqrt(0.5*lev+0.5)
                    *std::sqrt(0.5*lev+1.0);
                    res({0,lev+2,lev,0}) = 0.25*(15*lev*lev+45*lev+45.0)*std::sqrt(0.5*lev+0.5)
                    *std::sqrt(0.5*lev+1.0);
                }
                for(lev=0;lev<dim;++lev)
                {
                    res({0,lev,lev,0}) = 0.125*(20*lev*lev*lev+30*lev*lev+40*lev+15.0);
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
    }
}
