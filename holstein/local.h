/* Local operators in pp-Holstein model */

#ifndef local_H
#define local_H

#include "product.h"
#include "operator.h"

namespace KylinVib
{
    namespace Holstein
    {
        
        enum class LocalOperator
        {
            /* eye operator */
            Eye,

            /* physical raising operator */
            PhyRaise,

            /* physical lowering operator */
            PhyLower,

            /* auxillary raising operator */
            AuxRaise,

            /* auxillary lowering operator */
            AuxLower,
        };

        void gen_local_op ( Tensor<4> & op, LocalOperator s, INT phys, INT lastB )
        {
            if (s==LocalOperator::Eye)
            {
                for(INT i(0);i<phys;++i)
                {
                    Block<4> ri ( {1,1,1,1},{i,lastB,i,lastB});
                    ri({0,0,0,0}) = 1.0;
                    op.push_back(ri);
                }
            }
            else if (s==LocalOperator::PhyRaise)
            {
                for(INT i(0);i<phys-1;++i)
                {
                    Block<4> ri ( {1,1,1,1},{i+1,lastB,i,lastB+1});
                    ri({0,0,0,0}) = sqrt(i*1.0+1.0);
                    op.push_back(ri);
                }
                op.leg(0)[0] = 1;
                op.leg(2)[phys-1] = 1;
            }
            else if (s==LocalOperator::PhyLower)
            {
                for(INT i(0);i<phys-1;++i)
                {
                    Block<4> ri ( {1,1,1,1},{i,lastB,i+1,lastB-1});
                    ri({0,0,0,0}) = sqrt(i*1.0+1.0);
                    op.push_back(ri);
                }
                op.leg(0)[phys-1] = 1;
                op.leg(2)[0] = 1;
            }
            else if (s==LocalOperator::AuxRaise)
            {
                for(INT i(0);i<phys-1;++i)
                {
                    Block<4> ri ( {1,1,1,1},{i+1,lastB,i,lastB+1});
                    ri({0,0,0,0}) = 1.0;
                    op.push_back(ri);
                }
                op.leg(0)[0] = 1;
                op.leg(2)[phys-1] = 1;
            }
            else if (s==LocalOperator::AuxLower)
            {
                for(INT i(0);i<phys-1;++i)
                {
                    Block<4> ri ( {1,1,1,1},{i,lastB,i+1,lastB-1});
                    ri({0,0,0,0}) = 1.0;
                    op.push_back(ri);
                }
                op.leg(0)[phys-1] = 1;
                op.leg(2)[0] = 1;
            }
            else { return; }
            op.check_legs();
        }
    }
}

#endif
