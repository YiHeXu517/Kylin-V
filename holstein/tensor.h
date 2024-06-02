/* Local U(1) symmetry dense tensor class implemention */

#ifndef hols_tensor_H
#define hols_tensor_H

#include "block.h"

namespace klnX
{
    using std::vector;
    using std::sqrt;
    using std::pow;

    namespace Holstein
    {
        template<INT Rank>
        class Tensor
        {
        private:
            /* symmetry legs */
            array<map<INT,INT>,Rank> leg_;

            /* directions of each leg */
            array<INT,Rank> dirs_;

            /* saved blocks */
            vector<Block<Rank>> blocks_;    

        public:
            /* ctor & dtor & assign */
            Tensor() = default;
            ~Tensor() = default;
    
            Tensor(const Tensor<Rank> & t)
            : leg_(t.leg_), dirs_(t.dirs_), blocks_(t.blocks_)
            {

            }

            Tensor(Tensor<Rank> && t)
            : leg_(move(t.leg_)), dirs_(move(t.dirs_)), blocks_(move(t.blocks_))
            {

            }

            Tensor(const array<map<INT,INT>,Rank> & Legs, const array<INT,Rank> & Di) 
            : leg_(Legs), dirs_(Di)
            {
   
            }
      
            Tensor(const array<map<INT,INT>,Rank> & Legs, const array<INT,Rank> & Di, INT NumBks) 
            : leg_(Legs), dirs_(Di), blocks_(NumBks)
            {
   
            }
 
   
            Tensor(const array<INT,Rank> & Di) 
            : dirs_(Di)
            {
   
            }
   
            Tensor(Brace Di)
            {
                copy(Di.begin(),Di.end(),dirs_.begin());
            }
           
            Tensor<Rank> & operator=(const Tensor<Rank> & rhs)
            {
                leg_ = rhs.leg_;
                dirs_ = rhs.dirs_;
                blocks_    = rhs.blocks_;
                return *this;
            }

            Tensor<Rank> & operator=(Tensor<Rank> && rhs)
            {
                leg_ = move(rhs.leg_);
                dirs_ = move(rhs.dirs_);
                blocks_ = move(rhs.blocks_);
                return *this;
            }
            
            /* --------------- arithmetic operators -------------------  */

            Tensor<Rank> & operator+=(const Tensor<Rank> & rhs)
            {
                for(INT i(0);i<rhs.num_blocks();++i)
                {
                    push_back_ch(rhs.at(i));
                }
                return *this;
            }
            Tensor<Rank> & operator+=(Tensor<Rank> && rhs)
            {
                for(INT i(0);i<rhs.num_blocks();++i)
                {
                    push_back_ch(rhs.at(i));
                }
                return *this;
            }


            // multiply
            Tensor<Rank> operator*(const Complex & val)
            {
                Tensor<Rank> res(leg_,dirs_); 
                for(INT i(0);i<num_blocks();++i)
                {
                    res.push_back( blocks_[i] * val );
                }
                return res;
            }
            Tensor<Rank> & operator*=(double val)
            {
                for(INT i(0);i<num_blocks();++i)
                {
                    blocks_[i] *= val;
                }
                return *this;
            }

            friend ostream & operator<<(ostream & os, const Tensor<Rank> & r)
            {
                for(INT i(0);i<Rank;++i)
                {
                    string_view Dstr = (r.dirs_[i]==1)?"  [In]: ": " [Out]: "; 
                    os << "LEG " << i+1 << Dstr << " <";
                    for(auto it=r.cleg(i).begin();it!=r.cleg(i).end();++it)
                    {
                        os << "[" << it->first << ":" << it->second << "],";
                    }
                    os << "\b>" << endl;
                }
                for(INT i(0);i<r.num_blocks();++i)
                {
                    os << "Tensor block " << i+1 << endl;
                    os << r.at(i);
                }
                return os;
            } 

            // only bond legs
            void print_leg(INT axis)
            {
                string_view Dstr = (dirs_[axis]==1)?"  [In]: ": " [Out]: ";
                cout << "LEG " << axis+1 << Dstr << " <";
                for(auto it=cleg(axis).begin();it!=cleg(axis).end();++it)
                {
                    cout << "[" << it->first << ":" << it->second << "],";
                }
                cout << "\b>" << endl;
            }     

            /* ----------------- visiting members -------------------- */

            const map<INT,INT> & cleg(INT axis) const { return leg_[axis]; }
            const array<map<INT,INT>,Rank> & clegs() const { return leg_; }
            map<INT,INT> & leg(INT axis) { return leg_[axis]; }
            INT num_blocks() const { return blocks_.size(); }
            const array<INT,Rank> & dirs() const { return dirs_; }
            Block<Rank> & operator[](INT i) { return blocks_[i]; }
            const Block<Rank> & at(INT i) const { return blocks_[i]; }
           
            /* ----------------  change members ----------------------- */

            // push back without check and stack
            void push_back( const Block<Rank> & t ) { blocks_.push_back(t); }
            
            // push back with check
            void push_back_ch ( const Block<Rank> & t )
            {
                bool found = false;
                for(INT i(0);i<num_blocks();++i)
                {
                    if( at(i).shape() == t.shape() && at(i).qn() == t.qn()  )
                    {
                        blocks_[i] += t;
                        found = true;
                        break;
                    }
                }
                if(!found) { blocks_.push_back(t); }
            }

            // make legs from contained blocks
            void check_legs ( )
            {
                for(INT i(0);i<num_blocks();++i)
                {
                    for(INT j(0);j<Rank;++j)
                    {
                        leg_[j][ blocks_[i].qn()[j] ] = blocks_[i].shape()[j];
                    }
                }
            }

            // check u1-symmetry
            void check_sym ( const char * var ) const
            {
                string_view vars(var);
                for(INT i(0);i<num_blocks();++i)
                {
                    INT tSy = inner_product(dirs_.begin(),dirs_.end(),blocks_[i].qn().begin(),0);
                    if(tSy!=0) 
                    { cout << "Tensor " << var << " has symmetry-destroyed blocks at index " 
                      << i << ":" << endl; cout << (*this); break; }
                }
            }
        };

        template<INT Rank>
        double norm(const Tensor<Rank> & ori)
        {
            double nrm = 0.0;
            for(INT i(0);i<ori.num_blocks();++i)
            {
                nrm += pow(ori.at(i).norm(),2.0);
            }
            return sqrt(nrm);
        }

        template<INT Rank>
        Complex overlap(const Tensor<Rank> & ori, const Tensor<Rank> & oth)
        {
            Complex nrm = 0.0;
            for(INT i(0);i<ori.num_blocks();++i)
            {
                for(INT j(0);j<oth.num_blocks();++j)
                {
                    if( ori.at(i).shape() == oth.at(j).shape() &&
                        ori.at(i).qn() == oth.at(j).qn() )
                    {
                        nrm += oth.at(j).inner (ori.at(i));
                    }
                }
            }
            return nrm;
        }
 
        // transpose
        template<INT Rank> 
        Tensor<Rank> transpose( const Tensor<Rank> & ori, Brace ax )
        {
            array<map<INT,INT>,Rank> rLegs;
            array<INT,Rank> Dis;
            for(INT i(0);i<Rank;++i)
            {
                rLegs[i] = ori.cleg(*(ax.begin()+i)); 
                Dis[i] = ori.dirs()[ *(ax.begin()+i) ];
            }
            Tensor<Rank> res(rLegs,Dis,ori.num_blocks());
            for(INT i(0);i<ori.num_blocks();++i)
            {
                res[i] = transpose_para<Rank>(ori.at(i), ax);
            }
            return res;
        }

        //stack
        template<INT Rank>
        void stack ( Tensor<Rank> & om1, Tensor<Rank> & om2, Brace ax )
        {
            // combine legs
            for(INT i(0);i<Rank;++i)
            {
                if(find(ax.begin(),ax.end(),i)!=ax.end())
                {
                    for(auto it=om2.cleg(i).begin();it!=om2.cleg(i).end();++it)
                    {
                        if(om1.cleg(i).find(it->first)==om1.cleg(i).end())   // not found
                        {
                            om1.leg(i)[it->first] = it->second;
                        }
                        else
                        {
                            om1.leg(i)[it->first] += it->second;
                        }
                    }
                }
                else { continue; }
            }           
  
            // extend om1 itself
            for(INT i(0);i<om1.num_blocks();++i)
            {
                for(auto it=ax.begin();it!=ax.end();++it)
                {
                    if ( om1.at(i).shape()[*it] < om1.leg(*it)[om1.at(i).qn()[*it]] )   // need to extend
                    {
                        array<INT,Rank> Sp(om1.at(i).shape()), Qn(om1.at(i).qn()), Pos;
                        Pos.fill(0); Sp[*it] = om1.leg(*it)[om1.at(i).qn()[*it]];
                        Block<Rank> r1(Sp,Qn);
                        stack<Rank>(om1.at(i),Pos,r1);
                        om1[i] = move(r1);
                    }
                }
            }
            // extend om2
            for(INT i(0);i<om2.num_blocks();++i)
            {
                for(auto it=ax.begin();it!=ax.end();++it)
                {
                    if ( om2.at(i).shape()[*it] < om1.leg(*it)[om2.at(i).qn()[*it]] )   // need to extend
                    {
                        array<INT,Rank> Sp(om2.at(i).shape()), Qn(om2.at(i).qn()), Pos;
                        Pos.fill(0); Pos[*it] = om1.leg(*it)[om2.at(i).qn()[*it]] - om2.at(i).shape()[*it];
                        Sp[*it] = om1.leg(*it)[om2.at(i).qn()[*it]];
                        Block<Rank> r2(Sp,Qn);
                        stack<Rank>(om2.at(i),Pos,r2);
                        om2[i] = move(r2);
                    }
                }
                om1.push_back_ch(om2[i]);
            }
        }

        // memory cost of tensor
        template<INT Rank>
        double mem_cost ( const Tensor<Rank> & t1 )
        {
            double cst = 1.0 * Rank * sizeof(INT);
            cst += sizeof(INT) * 2.0 * t1.clegs().size();
            cst /= pow(1024.0,3.0);
            for(INT i(0);i<t1.num_blocks();++i) { cst += mem_cost<Rank>(t1.at(i));  }
            return cst;
        }

        template<INT Rank>
        double mem_cost ( const vector<Tensor<Rank>> & t1 )
        {
            double cst = 0.0;
            for(INT i(0);i<t1.size();++i) { cst += mem_cost<Rank>(t1[i]);  }
            return cst;
        }  
    }
}

#endif
