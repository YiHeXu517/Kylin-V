/* Local U(1) symmetry operator class implemention */

#ifndef hols_operator_H
#define hols_operator_H

#include "state.h"

namespace KylinVib
{
    namespace Holstein
    {
        class Operator
        {
        private:
            /* tensor train */
            vector<Tensor<4>> tt_;    

        public:
            /* ctor & dtor & assign */
            Operator() = default;
            ~Operator() = default;
    
            Operator(const Operator & t)
            : tt_(t.tt_)
            {

            }

            Operator(Operator && t)
            : tt_(move(t.tt_))
            {

            }

            Operator(INT Nsites) 
            : tt_(Nsites,{1,1,-1,-1})
            {
   
            }
           
            Operator & operator=(const Operator & rhs)
            {
                tt_    = rhs.tt_;
                return *this;
            }

            Operator & operator=(Operator && rhs)
            {
                tt_ = move(rhs.tt_);
                return *this;
            }
            
            /* --------------- arithmetic operators -------------------  */
            Operator & operator+=(Operator & r)
            {
                INT ns = nsite();
                stack<4>(tt_[0],r.tt_[0],{3});
                for(INT i(1);i<ns-1;++i)
                {
                    stack<4>(tt_[i],r.tt_[i],{1,3});
                }
                stack<4>(tt_[ns-1],r.tt_[ns-1],{1});
                return *this;
            }

            Operator & operator*=(double val)
            {
                tt_[0] *= val;
                return *this;
            }

            /* --------------------- IO ----------------------------- */

            friend ostream & operator<<(ostream & os, const Operator & r)
            {
                for(INT i(0);i<r.nsite();++i)
                {
                    os << "Site " << i+1 << endl;
                    os << r.at(i);
                }
                return os;
            }

            void print_leg(INT axis)
            {
                for(INT i(0);i<nsite();++i)
                {
                    cout << "Site " << i+1 << endl;
                    tt_[i].print_leg(axis);
                }
            } 


            void save(ofstream & ofs) const
            {
                INT labelMPO =  112;
                ofs.write(reinterpret_cast<char*>(&labelMPO),sizeof(INT));
                
                INT ns = nsite();
                ofs.write(reinterpret_cast<char*>(&ns),sizeof(INT));

                for(INT i(0);i<ns;++i)
                {
                    INT nb = tt_[i].num_blocks();
                    ofs.write(reinterpret_cast<char*>(&nb),sizeof(INT));
                    for(INT j(0);j<nb;++j)
                    {
                        ofs.write(reinterpret_cast<const char*>(tt_[i].at(j).shape().data()),sizeof(INT)*4);
                        ofs.write(reinterpret_cast<const char*>(tt_[i].at(j).qn().data()),sizeof(INT)*4);
                        ofs.write(reinterpret_cast<const char*>(tt_[i].at(j).cptr()),sizeof(Complex)*tt_[i].at(j).size());
                    }
                }
                ofs << endl;
                ofs.close();
            }

            static Operator load(ifstream & ifs)
            {
                INT labelMPO;
                ifs.read(reinterpret_cast<char*>(&labelMPO),sizeof(INT));
                if(labelMPO!=112) { cout << "Illegal MPO file!" << endl; exit(1); }
                INT ns;
                ifs.read(reinterpret_cast<char*>(&ns),sizeof(INT));
                Operator res(ns);
                for(INT i(0);i<ns;++i)
                {
                    INT nb;
                    Tensor<4> ri({1,1,-1,-1});
                    ifs.read(reinterpret_cast<char*>(&nb),sizeof(INT));
                    for(INT j(0);j<nb;++j)
                    {
                        array<INT,4> Sp, Qn;
                        ifs.read(reinterpret_cast<char*>(Sp.data()),sizeof(INT)*4);
                        ifs.read(reinterpret_cast<char*>(Qn.data()),sizeof(INT)*4);
                        Block<4> rij(Sp,Qn);
                        ifs.read(reinterpret_cast<char*>(rij.ptr()),sizeof(Complex)*rij.size());
                        ri.push_back(rij);
                    }
                    ri.check_legs();
                    res[i] = move(ri);
                }
                return res;
            }

            /* ----------------- visiting members -------------------- */

            INT nsite() const { return tt_.size(); }
            Tensor<4> & operator[](INT i) { return tt_[i]; }
            const Tensor<4> & at(INT i) const { return tt_[i]; }
            double op_mem_cost() const { return mem_cost<4>(tt_);  }
           
            /* ----------------  change members ----------------------- */
            void truncate(double tol = 1.0e-14)
            {
                SVDConfig r;
                r.Left2Right = true;
                r.SVDThres = tol;
                r.EnableRescale = true;
                for(INT i(0);i<nsite()-1;++i)
                {
                    auto[cur,nex] = svd_impl<3,1>( tt_[i], r, -1 );
                    tt_[i] = move(cur);
                    Tensor<4> lt = transpose<4>(tt_[i+1],{1,0,2,3});
                    Tensor<4> nlt = product<2,4,1>( nex, lt, CblasNoTrans, CblasNoTrans );
                    tt_[i+1] = transpose<4>(nlt,{1,0,2,3});
                }
                r.Left2Right = false;
                for(INT i(nsite()-1);i>0;--i)
                {
                    Tensor<4> rt = transpose<4>(tt_[i],{1,0,2,3});
                    auto[nex,cur] = svd_impl<1,3>( rt, r, -1 );
                    tt_[i] = transpose<4>(cur,{1,0,2,3});
                    tt_[i-1] = product<4,2,1>( tt_[i-1], nex, CblasNoTrans, CblasNoTrans );
                }
            }

            // join two mps
            Complex join( const State & s1, const State & s2 ) const
            {
                Tensor<3> envl({1,-1,-1});
                Tensor<3> envr({1,1,1});
                Block<3> ri({1,1,1},{0,0,0}); ri({0,0,0}) = 1.0;
                envl.push_back(ri); envl.check_legs();
                INT ns = nsite();
                for(INT i(0);i<ns;++i)
                {
                    envl = env_sweep(envl,s1.at(i),tt_[i],s2.at(i));
                }
                INT totSym = 0;
                if(envl.num_blocks()>0) 
                { 
                    totSym = envl.at(0).qn()[0];
                    Block<3> rf({1,1,1},{totSym,0,totSym}); rf({0,0,0}) = 1.0;
                    envr.push_back(rf);
                    return overlap(envl,envr);
                }
                else { Complex res(0,0); return res;  }
            }
        };

        /* non-member functions  */

        // operator dot
        Operator mpo_dot(const Operator & op1, const Operator & op2 )
        {
            Tensor<3> envl({1,-1,-1});
            Block<3> ri({1,1,1},{0,0,0}); ri({0,0,0}) = 1.0; envl.push_back(ri);
            envl.check_legs();
            SVDConfig r;
            r.Left2Right = true;
            r.SVDThres   = 1e-14;
            r.EnableRescale = true;
            INT ns = op1.nsite();
            Operator res(ns);
            for(INT i(0);i<ns;++i)
            {
                auto[cur,nex] = apply_mpo ( envl, op1.at(i), op2.at(i), r);
                envl = move(nex);
                if(i==ns-1) 
                {  
                    Tensor<3> envr({-1,-1,1});
                    Block<3> rf({1,1,1},{0,envl.at(0).qn()[1],envl.at(0).qn()[2]}); rf({0,0,0}) = 1.0; envr.push_back(rf);
                    envr.check_legs();
                    res[i] = cur * overlap(envr,envl);
                }
                else { res[i] = move(cur); }
            }
            return res;
        }
      
        // operator-state application
        State apply_operator( const Operator & op, const State & wf, double tolSVD, INT maxD)
        {
            Tensor<3> envl({1,-1,-1});
            Block<3> ri({1,1,1},{0,0,0}); ri({0,0,0}) = 1.0; envl.push_back(ri);
            envl.check_legs();
            SVDConfig r;
            r.Left2Right = true;
            r.SVDThres   = tolSVD;
            r.MaxDim     = maxD;
            r.EnableRescale = false;
            INT ns = op.nsite();
            State res(ns);
            INT lastB = wf.at(ns-1).at(0).qn()[2];
            INT lastw = op.at(ns-1).at(0).qn()[3];
            for(INT i(0);i<ns;++i)
            {
                Tensor<4> lmo = env_mps_mpo(envl, wf.at(i),op.at(i),true);
                if(i==ns-1) 
                {  
                    Tensor<3> envr({-1,1,1});
                    Block<3> rf({1,1,1},{lastB,lastw,lastB}); rf({0,0,0}) = 1.0; envr.push_back(rf);
                    envr.check_legs();
                    res[i] = product<4,3,2>(lmo,envr,CblasNoTrans,CblasTrans);
                }
                else { auto[cur,nex] = svd_impl<2,2> (lmo, r, -1); res[i] = move(cur);  envl = move(nex);
                }
            }
            r.Left2Right = false;
            
            for(INT i(op.nsite()-1);i>0;--i)
            {
                auto[nex,cur] = svd_impl<1,2>( res[i], r, -1 );
                res[i] = move(cur);
                res[i-1] = product<3,2,1>( res[i-1], nex, CblasNoTrans, CblasNoTrans );
            }
            
            return res;
        }
    }
}

#endif
