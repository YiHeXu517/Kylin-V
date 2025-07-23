/* All kinds of local operators defined here */

#pragma once

#include "linalg.h"

namespace KylinVib
{
    namespace Polariton
    {
        enum class LocalOp
        {
            I,Upper,Lower,Q,N
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
            case LocalOp::N:
                os << "N";
                break;
            default:
                break; 
            }
            return os;
        }
        // transform to mpo-matrix
        template<typename ArrType = Dense<double,4>>
        ArrType to_matrix(LocalOp op, int dim)
        {
            ArrType res({1,dim,dim,1});
            int lev = 0;
            switch (op)
            {
            case LocalOp::I:
                for(lev=0;lev<dim;++lev)
                {
                    res({0,lev,lev,0}) = 1.0;
                }
                break;
            case LocalOp::N:
                for(lev=0;lev<dim;++lev)
                {
                    res({0,lev,lev,0}) = 1.0*lev;
                }
                break;
            case LocalOp::Q:
                for(lev=0;lev<dim-1;++lev)
                {
                    res({0,lev,lev+1,0}) = std::sqrt(0.5*lev+0.5);
                    res({0,lev+1,lev,0}) = std::sqrt(0.5*lev+0.5);
                }
                break;
            case LocalOp::Upper:
                for(lev=0;lev<dim-1;++lev)
                {
                    res({0,lev+1,lev,0}) = std::sqrt(1.0*lev+1.0);
                }
                break;
            case LocalOp::Lower:
                for(lev=0;lev<dim-1;++lev)
                {
                    res({0,lev,lev+1,0}) = std::sqrt(1.0*lev+1.0);
                }
                break;
            default:
                break;
            }
            return res;
        }
    }
}
