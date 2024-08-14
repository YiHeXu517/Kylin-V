/* tools for sparse algebras for watson model */

#pragma once

#include "operator.h"

namespace KylinVib
{
    namespace Watson
    {
        // sweep 1 : env-mps product
        SparseBase<double,4> env_mps(SparseBase<double,4> const & env, SparseBase<double,4> const & mps,
        char l2r = 'r')
        {
            size_t nz1 = env.nnz(), nz2 = mps.nnz(), nth;
            if(l2r=='r')
            {
                #pragma omp parallel
                nth = omp_get_num_threads();

                vector<SparseBase<double,4>> ress(nth,{env.shape()[0],env.shape()[1],
                mps.shape()[1],mps.shape()[2]});

                #pragma omp parallel for
                for(size_t i=0;i<nz1*nz2;++i)
                {
                    size_t i1 = i / nz2, i2 = i % nz2;
                    if( env.indices(i1)[2] == mps.indices(i2)[0] )
                    {
                        size_t ThreadID = omp_get_thread_num();
                        array<size_t,4> rind = {env.indices(i1)[0], env.indices(i1)[1],
                        mps.indices(i2)[1], mps.indices(i2)[2]};
                        double rval = env.values(i1) * mps.values(i2);
                        ress[ThreadID].push_back(rind,rval);
                    }
                }

                SparseBase<double,4> res({env.shape()[0],env.shape()[1],
                mps.shape()[1],mps.shape()[2]});
            }
        }
    }
}
