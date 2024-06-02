/* Local U(1) symmetry state class implemention */

#ifndef hols_state_H
#define hols_state_H

#include "product.h"

namespace klnX
{
    namespace Holstein
    {
        class State
        {
        private:
            /* tensor train */
            vector<Tensor<3>> tt_;    

        public:
            /* ctor & dtor & assign */
            State() = default;
            ~State() = default;
    
            State(const State & t)
            : tt_(t.tt_)
            {

            }

            State(State && t)
            : tt_(move(t.tt_))
            {

            }

            State(INT Nsites) 
            : tt_(Nsites,{1,1,-1})
            {
   
            }
           
            State & operator=(const State & rhs)
            {
                tt_    = rhs.tt_;
                return *this;
            }

            State & operator=(State && rhs)
            {
                tt_ = move(rhs.tt_);
                return *this;
            }
            
            /* --------------- arithmetic operators -------------------  */
            State & operator+=(State && r)
            {
                INT ns = nsite();
                stack<3>(tt_[0],r.tt_[0],{2});
                for(INT i(1);i<ns-1;++i)
                {
                    stack(tt_[i],r.tt_[i],{0,2});
                }
                stack<3>(tt_[ns-1],r.tt_[ns-1],{0});
                return *this;
            }
 
            State operator*(const Complex & val)
            {
                State res(*this);
                res[0] = tt_[0] * val;
                return res;
            }
         
            State operator*(double val)
            {
                State res(*this);
                Complex valc(val,0);
                res[0] = tt_[0] * valc;
                return res;
            }

            State & operator*=(double val)
            {
                tt_[0] *= val;
                return *this;
            }

            friend ostream & operator<<(ostream & os, const State & r)
            {
                for(INT i(0);i<r.nsite();++i)
                {
                    os << "Site " << i+1 << endl;
                    os << r.at(i);
                }
                return os;
            }       
            
            // print bond only
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
                INT labelMPS =  111;
                ofs.write(reinterpret_cast<char*>(&labelMPS),sizeof(INT));

                INT ns = nsite();
                ofs.write(reinterpret_cast<char*>(&ns),sizeof(INT));

                for(INT i(0);i<ns;++i)
                {
                    INT nb = tt_[i].num_blocks();
                    ofs.write(reinterpret_cast<char*>(&nb),sizeof(INT));
                    for(INT j(0);j<nb;++j)
                    {
                        ofs.write(reinterpret_cast<const char*>(tt_[i].at(j).shape().data()),sizeof(INT)*3);
                        ofs.write(reinterpret_cast<const char*>(tt_[i].at(j).qn().data()),sizeof(INT)*3);
                        ofs.write(reinterpret_cast<const char*>(tt_[i].at(j).cptr()),sizeof(Complex)*tt_[i].at(j).size());
                    }
                }
                ofs << endl;
                ofs.close();
            }

            static State load(ifstream & ifs)
            {
                INT labelMPS;
                ifs.read(reinterpret_cast<char*>(&labelMPS),sizeof(INT));
                if(labelMPS!=111) { cout << "Illegal MPS file!" << endl; exit(1); }
                INT ns;
                ifs.read(reinterpret_cast<char*>(&ns),sizeof(INT));
                State res(ns);
                for(INT i(0);i<ns;++i)
                {
                    INT nb;
                    Tensor<3> ri({1,1,-1});
                    ifs.read(reinterpret_cast<char*>(&nb),sizeof(INT));
                    for(INT j(0);j<nb;++j)
                    {
                        array<INT,3> Sp, Qn;
                        ifs.read(reinterpret_cast<char*>(Sp.data()),sizeof(INT)*3);
                        ifs.read(reinterpret_cast<char*>(Qn.data()),sizeof(INT)*3);
                        Block<3> rij(Sp,Qn);
                        ifs.read(reinterpret_cast<char*>(rij.ptr()),sizeof(Complex)*rij.size());
                        ri.push_back(rij);
                    }
                    ri.check_legs();
                    res[i] = move(ri);
                }
                ifs.close();
                return res;
            }
            /* ----------------- visiting members -------------------- */

            INT nsite() const { return tt_.size(); }
            Tensor<3> & operator[](INT i) { return tt_[i]; }
            const Tensor<3> & at(INT i) const { return tt_[i]; }
            double s_mem_cost() const { return mem_cost(tt_);  }
           
            /* ----------------  change members ----------------------- */
            void canonicalize(double tolS = 0.0, INT mD = 1000)
            {
                SVDConfig r;
                r.SVDThres = tolS;
                r.MaxDim   = mD;
                r.Left2Right = true;
                for(INT i(0);i<nsite()-1;++i)
                {
                    auto[cur,nex] = svd_impl<2,1>( tt_[i], r, -1 );
                    tt_[i] = move(cur);
                    tt_[i+1] = product<2,3,1>( nex, tt_[i+1], CblasNoTrans, CblasNoTrans );
                }
                r.Left2Right = false;
                for(INT i(nsite()-1);i>0;--i)
                {
                    auto[nex,cur] = svd_impl<1,2>( tt_[i], r, -1 );
                    tt_[i] = move(cur);
                    tt_[i-1] = product<3,2,1>( tt_[i-1], nex, CblasNoTrans, CblasNoTrans );
                }
            }
            /* ---------------- print max bond dimension and its position ------------------- */
            void max_bond() const
            {
                INT mb(0), mbp(0);
                for(INT i(0);i<nsite()-1;++i)
                {
                    INT CurBond = 0;
                    for(auto it = tt_[i].cleg(2).begin(); it!=tt_[i].cleg(2).end(); ++it)
                    {
                        CurBond += it->second;
                    }
                    if(mbp < CurBond) { mb = i; mbp = CurBond; }
                }
                cout << "Max bond dimension = " << mbp << " on bond " << mb << "-" << mb+1 << endl;
            }
        };

        /* non-member functions  */
        double norm(const State & s) { return norm(s.at(0)); }

        Complex overlap(const State & s1, const State & s2)
        {
            Tensor<2> envl({1,-1});
            Tensor<2> envr({-1,1});
            Block<2> ri({1,1},{0,0}); ri({0,0}) = 1.0;
            envl.push_back(ri);
            INT ns = s1.nsite();
            for(INT i(0);i<ns;++i)
            {
                Tensor<3> lm = product<2,3,1>(envl,s2.at(i),CblasNoTrans,CblasNoTrans);
                envl = product<3,3,2>(s1.at(i),lm,CblasConjTrans,CblasNoTrans);
            }
            INT totSym = 0;
            if(envl.num_blocks()>0)
            {
                totSym = envl.at(0).qn()[0];
                Block<2> rf({1,1},{totSym,totSym}); rf({0,0}) = 1.0;
                envr.push_back(rf);
                return overlap(envl,envr);
            }
            else { Complex res(0,0); return res;  }
        }
    }
}

#endif
