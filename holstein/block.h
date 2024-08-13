/* Local U(1) symmetry dense tensor block class implemention */

#ifndef hols_block_H
#define hols_block_H

#include "../util/timer.h"

namespace KylinVib
{
    using std::array;
    using std::accumulate;
    using std::multiplies;
    using std::inner_product;
    using std::copy;
    using std::vector;
    using std::ref;
    using std::thread;
    using std::move;
    using std::find;

    namespace Holstein
    {
        template<INT Rank>
        class Block : public Alloc<Complex>
        {
        private:
            /* shape */
            array<INT,Rank> shape_;

            /* distance */
            array<INT,Rank> dist_;

            /* quantum numbers */
            array<INT,Rank> qn_;            

        public:
            /* ctor & dtor & assign */
            Block() = default;
            ~Block() = default;
    
            Block(const Block<Rank> & t)
            : shape_(t.shape_), dist_(t.dist_),
            qn_(t.qn_), Alloc<Complex>(t)
            {

            }

            Block(Block<Rank> && t)
            : shape_(move(t.shape_)), dist_(move(t.dist_)),
            qn_(move(t.qn_)), Alloc<Complex>(move(t))
            {

            }
            
            Block(const array<INT,Rank> & Sp, const array<INT,Rank> & Qn)
            : shape_(Sp), qn_(Qn), Alloc<Complex>( accumulate(Sp.begin(),Sp.end(),(INT)1,multiplies<INT>() ) )
            {
                make_distance();
            }

            Block(Brace Sp, Brace Qn)
            : Alloc<Complex>( accumulate(Sp.begin(),Sp.end(),(INT)1,multiplies<INT>() ) )
            {
                copy(Sp.begin(),Sp.end(),shape_.begin());
                copy(Qn.begin(),Qn.end(),qn_.begin());
                make_distance();
            }

            Block<Rank> & operator=(const Block<Rank> & rhs)
            {
                shape_ = rhs.shape_;
                dist_  = rhs.dist_;
                qn_    = rhs.qn_;
                Alloc<Complex>::operator=(rhs);
                return *this;
            }

            Block<Rank> & operator=(Block<Rank> && rhs)
            {
                shape_ = move(rhs.shape_);
                dist_  = move(rhs.dist_);
                qn_    = move(rhs.qn_);
                Alloc<Complex>::operator=(move(rhs));
                return *this;
            }
            
            /* --------------- arithmetic operators -------------------  */

            // addition
            Block<Rank> & operator+=(const Block<Rank> & rhs)
            {
                Complex alp(1,0);
                cblas_zaxpy ( this->size(), &alp, rhs.cptr(), 1, this->ptr(), 1 );
                return *this;
            }
 
            // multiply
            Block<Rank> operator*(const Complex & val)
            {
                Block<Rank> res(*this); 
                cblas_zscal ( res.size(), &val, res.ptr(), 1 );
                return res;
            }

            Block<Rank> & operator*=(double val)
            {
                cblas_zdscal ( this->size(), val, this->ptr(), 1 );
                return *this;
            }

            // index operators
            Complex & operator()(const array<INT,Rank> & idx)
            {
                INT pos = inner_product(idx.begin(),idx.end(),dist_.begin(),(INT)0);
                if(pos>=this->size()) { cout << "Overflow!" << endl; exit(1); }
                return this->ptr()[pos];
            }

            Complex & operator()(Brace idx)
            {
                INT pos = inner_product(idx.begin(),idx.end(),dist_.begin(),(INT)0);
                if(pos>=this->size()) { cout << "Overflow!" << endl; exit(1); }
                return this->ptr()[pos];
            }

            friend ostream & operator<<(ostream & os, const Block<Rank> & r)
            {
                for(INT j(0);j<Rank;++j)
                {
                    os << "LEG: " << j+1 << " LEN: " << r.shape()[j] << " QN: " 
                       << r.qn()[j] << endl;
                }
                os << setw(5*Rank+35) << setfill('=') << "=" << endl;
                for(INT i(0);i<r.size();++i)
                {
                    if(abs(r.cptr()[i])<1e-17) { continue; }
                    for(INT j(0);j<Rank;++j)
                    {
                        os << setw(5) << setfill(' ') << i / r.dist()[j] % r.shape()[j];
                    }
                    os << " | " << setw(30) << setprecision(4) << std::scientific << r.cptr()[i] << endl;
                }
                os << setw(5*Rank+35) << setfill('=') << "=" << endl;
                return os;
            }            

            /* ----------------- visiting members -------------------- */

            const array<INT,Rank> & shape() const { return shape_; }
            const array<INT,Rank> & dist() const { return dist_; }
            const array<INT,Rank> & qn() const { return qn_; }

            /* make distance */
            void make_distance()
            {
                dist_[Rank-1] = 1;
                for(INT i(1);i<Rank;++i)
                {
                    dist_[Rank-i-1] = dist_[Rank-i] * shape_[Rank-i];
                }
            }
        };

        /* non-member functions */
        
        // atomic transpose function
        template<INT Rank>
        void transpose ( const Block<Rank> & ori, Brace ax, Block<Rank> &res )
        {
            for(INT i(0);i<ori.size();++i)
            {
                array<INT,Rank> idx;
                for(INT j(0);j<Rank;++j)
                {
                    idx[j] = i / ori.dist()[*(ax.begin()+j)] % ori.shape()[*(ax.begin()+j)];
                }
                res(idx) = ori.cptr()[i];
            }
        }

        // atomic stack function
        template<INT Rank>
        void stack ( const Block<Rank> & ori, const array<INT,Rank> & pos, Block<Rank> &res )
        {
            for(INT i(0);i<ori.size();++i)
            {
                array<INT,Rank> idx;
                for(INT j(0);j<Rank;++j)
                {
                    idx[j] = i / ori.dist()[j] % ori.shape()[j];
                    idx[j] += *(pos.begin()+j);
                }
                res(idx) = ori.cptr()[i];
            }
        }

        // parallel transpose
        template<INT Rank>
        Block<Rank> transpose_para ( const Block<Rank> & ori,  Brace ax)
        {
            array<INT,Rank> sps,qns;
            for(INT i(0);i<Rank;++i) 
            { 
                sps[i] = ori.shape()[*(ax.begin()+i)];  
                qns[i] = ori.qn()[*(ax.begin()+i)];
            }
            Block<Rank> res(sps,qns);
	    transpose( ori, ax ,res);
            return res;
        }
      
        // total stack
        template<INT Rank>
        Block<Rank> stack_pair ( const Block<Rank> & m1, const Block<Rank> & m2, Brace ax )
        {
            array<INT,Rank> sps,qns,pos1,pos2;
            pos1.fill(0); pos2.fill(0);
            for(INT i(0);i<Rank;++i) 
            {
                sps[i] = m1.shape()[i];
                qns[i] = m1.qn()[i];
                if(find(ax.begin(),ax.end(),i)!=ax.end())
                {
                    sps[i] += m2.shape()[i];
                    pos2[i] = m1.shape()[i];
                }
            }
            Block<Rank> res(sps,qns);
            stack<Rank>(m1,pos1,res);
            stack<Rank>(m2,pos2,res);
            return res; 
        }
        // memory cost of one block (GB)
        template<INT Rank>
        double mem_cost ( const Block<Rank> & m1  )
        {
            double cst =  3.0 * Rank * sizeof(INT);
            cst += m1.size() * sizeof(Complex);
            return cst / pow(1024.0,3.0);
        }
    }
}

#endif
