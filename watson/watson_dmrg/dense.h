/* multi-dimension dense array class */

#pragma once

#include "global.h"

namespace KylinVib
{
    namespace WatsonDMRG
    {
        // float-version
        template<INT N>
        class ArrR
        {
            private:

            // shape
            std::array<INT,N> shape_;

            // distance
            std::array<INT,N> dist_;

            // number of elements
            INT size_ = 0;

            // data saved
            double * ptr_ = nullptr;

            // make distance from shape
            void make_distance()
            {
                dist_[N-1] = 1;
                for(INT i=1;i<N;++i)
                {
                    dist_[N-i-1] = dist_[N-i] * shape_[N-i];
                }
            }

            public:

            // default
            ArrR() = default;

            // copy
            ArrR(ArrR<N> const & r)
            : shape_(r.shape_), dist_(r.dist_),size_(r.size_)
            {
                ptr_ = (double *)std::malloc(size_*sizeof(double));
                std::memcpy(ptr_,r.ptr_,size_*sizeof(double));
            }

            ArrR<N> & operator=(ArrR<N> const & r)
            {
                std::free(ptr_);
                shape_ = r.shape_;
                dist_  = r.dist_;
                size_  = r.size_;
                ptr_ = (double *)std::malloc(size_*sizeof(double));
                std::memcpy(ptr_,r.ptr_,size_*sizeof(double));
                return *this;
            }

            // move
            ArrR(ArrR<N> && r)
            : shape_(std::move(r.shape_)), dist_(std::move(r.dist_)),size_(r.size_)
            {
                ptr_ = r.ptr_;
                r.ptr_ = nullptr;
            }

            ArrR<N> & operator=(ArrR<N> && r)
            {
                std::free(ptr_);
                shape_ = std::move(r.shape_);
                dist_  = std::move(r.dist_);
                size_  = r.size_;
                ptr_ = r.ptr_;
                r.ptr_ = nullptr;
                return *this;
            }

            // allocation
            ArrR(std::array<INT,N> const & sp)
            : shape_(sp), size_(std::accumulate(sp.begin(),sp.end(),1,std::multiplies<INT>()))
            {
                make_distance();
                ptr_ = (double *)std::calloc(size_,sizeof(double));
            }

            ArrR(Brace sp)
            : size_(std::accumulate(sp.begin(),sp.end(),1,std::multiplies<INT>()))
            {
                std::copy(sp.begin(),sp.end(),shape_.begin());
                make_distance();
                ptr_ = (double *)std::calloc(size_,sizeof(double));
            }

            // dtors
            ~ArrR()
            {
                std::free(ptr_);
                ptr_ = nullptr;
            }

            /* members */

            INT size() const
            {
                return size_;
            }

            const std::array<INT,N> & shape() const
            {
                return shape_;
            }

            const std::array<INT,N> & dist() const
            {
                return dist_;
            }

            double * ptr()
            {
                return ptr_;
            }

            const double * cptr() const
            {
                return ptr_;
            }

            /* arithmetics */

            // plus
            ArrR<N> & operator+=(ArrR<N> const & r)
            {
                cblas_daxpy(size_,1.0,r.ptr_,1,ptr_,1);
                return *this;
            }

            ArrR<N> operator+(ArrR<N> const & r) const
            {
                ArrR<N> res(*this);
                cblas_daxpy(size_,1.0,r.ptr_,1,res.ptr_,1);
                return res;
            }

            // minus
            ArrR<N> & operator-=(ArrR<N> const & r)
            {
                cblas_daxpy(size_,-1.0,r.ptr_,1,ptr_,1);
                return *this;
            }

            ArrR<N> operator-(ArrR<N> const & r) const
            {
                ArrR<N> res(*this);
                cblas_daxpy(size_,-1.0,r.ptr_,1,res.ptr_,1);
                return res;
            }

            // multiplication
            ArrR<N> & operator*=(double val)
            {
                cblas_dscal(size_,val,ptr_,1);
                return *this;
            }

            ArrR<N> operator*(double val) const
            {
                ArrR<N> res(*this);
                cblas_dscal(size_,val,res.ptr_,1);
                return res;
            }

            // division
            ArrR<N> & operator/=(double val)
            {
                double vald = 1/val;
                cblas_dscal(size_,vald,ptr_,1);
                return *this;
            }

            ArrR<N> operator/(double val) const
            {
                double vald = 1/val;
                ArrR<N> res(*this);
                cblas_dscal(size_,vald,res.ptr_,1);
                return res;
            }

            /* modifiers */

            // index
            double & operator()(std::array<INT,N> const & idx)
            {
                INT pos = std::inner_product(idx.begin(),idx.end(),dist_.begin(),0);
                if(pos>=size_)
                {
                    std::cout << "Overflow!" << std::endl;
                    std::exit(1);
                }
                return this->ptr()[pos];
            }

            double & operator()(Brace idx)
            {
                INT pos = std::inner_product(idx.begin(),idx.end(),dist_.begin(),0);
                if(pos>=size_)
                {
                    std::cout << "Overflow!" << std::endl;
                    std::exit(1);
                }
                return this->ptr()[pos];
            }

            const double & operator()(std::array<INT,N> const & idx) const
            {
                INT pos = std::inner_product(idx.begin(),idx.end(),dist_.begin(),0);
                if(pos>=size_)
                {
                    std::cout << "Overflow!" << std::endl;
                    std::exit(1);
                }
                return this->cptr()[pos];
            }

            const double & operator()(Brace idx) const
            {
                INT pos = std::inner_product(idx.begin(),idx.end(),dist_.begin(),0);
                if(pos>=size_)
                {
                    std::cout << "Overflow!" << std::endl;
                    std::exit(1);
                }
                return this->cptr()[pos];
            }
            /* IO */

            // print out the array
            void print(INT ncol) const
            {
                INT leg = 1;
                std::for_each(shape_.begin(),shape_.end(),[&leg](INT x){std::cout << "LEN " << leg << " : " << x << std::endl; ++leg;});
                INT Nw = 10, Nr = size_ / ncol;
                for(INT i=0;i<Nr;++i)
                {
                    for(INT j=0;j<ncol;++j)
                    {
                        std::cout << std::scientific << std::setw(Nw) << ptr_[i*ncol+j] << " ";
                    }
                    std::cout << std::endl;
                }
            }
            void print_sp(double tol = 1e-15) const
            {
                INT leg = 1;
                std::for_each(shape_.begin(),shape_.end(),[&leg](INT x){std::cout << "LEN " << leg << " : " << x << std::endl; ++leg;});
                for(INT i=0;i<size_;++i)
                {
                    if(std::abs(ptr_[i])<=tol)
                    {
                        continue;
                    }
                    std::cout << '[';
                    for(INT j=0;j<N;++j)
                    {
                        std::cout << i / dist_[j] % shape_[j];
                        if(j!=N-1)
                        {
                            std::cout << " ";
                        }
                    }
                    std::cout << "] | " << ptr_[i] << std::endl;
                }
            }

            // randomly fill
            void rand_fill()
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(-1,1);
                for(INT i=0;i<this->size();++i)
                {
                    this->ptr()[i] = dis(gen);
                }
            }
            // fill with 1
            void ones(double val = 1.0)
            {
                for(INT i=0;i<this->size();++i)
                {
                    ptr_[i] = val;
                }
            }
            // eye operator
            static ArrR<N> eye(INT Ns)
            {
                std::array<INT,N> rsp;
                rsp.fill(Ns);
                ArrR<N> res(rsp);
                for(INT i=0;i<Ns;++i)
                {
                    std::array<INT,N> idx;
                    idx.fill(i);
                    res(idx) = 1.0;
                }
                return res;
            }

            // slice operation
            ArrR<N> slice(Brace Ahead, Brace ATail)
            {
                std::array<INT,N> rsp;
                for(INT i=0;i<N;++i)
                {
                    rsp[i] = *(ATail.begin()+i) - *(Ahead.begin()+i);
                }
                ArrR<N> res(rsp);
                for(INT i=0;i<res.size_;++i)
                {
                    std::array<INT,N> idx;
                    for(INT j=0;j<N;++j)
                    {
                        idx[j] = i / res.dist_[j] % res.shape_[j] + *(Ahead.begin()+j);
                    }
                    res.ptr_[i] = (*this)(idx);
                }
                return res;
            }

            // return the maximal absolute number and index
            std::tuple<INT,double> arg_min()
            {
                INT ResIdx=0; double Res=ptr_[0];
                for(INT i=0;i<size_;++i)
                {
                    if(ptr_[i]<Res)
                    {
                        Res = ptr_[i];
                        ResIdx = i;
                    }
                }
                return std::make_tuple(ResIdx,Res);
            }
            std::tuple<INT,double> arg_abs_max()
            {
                INT ResIdx=0; double Res=ptr_[0];
                for(INT i=0;i<size_;++i)
                {
                    if(std::abs(ptr_[i])>std::abs(Res))
                    {
                        Res = ptr_[i];
                        ResIdx = i;
                    }
                }
                return std::make_tuple(ResIdx,Res);
            }
            void arg_max_above(std::vector<std::tuple<INT,double>> & ArgPairs, double tol=1e-1)
            {
                for(INT i=0;i<size_;++i)
                {
                    if(std::abs(ptr_[i])>tol)
                    {
                        ArgPairs.push_back( std::make_tuple(i,ptr_[i]) );
                    }
                }
            }
        };
        // complex-version
        template<INT N>
        class ArrC
        {
            private:

            // shape
            std::array<INT,N> shape_;

            // distance
            std::array<INT,N> dist_;

            // number of elements
            INT size_ = 0;

            // data saved
            Complex * ptr_ = nullptr;

            // make distance from shape
            void make_distance()
            {
                dist_[N-1] = 1;
                for(INT i=1;i<N;++i)
                {
                    dist_[N-i-1] = dist_[N-i] * shape_[N-i];
                }
            }

            public:

            // default
            ArrC() = default;

            // copy
            ArrC(ArrC<N> const & r)
            : shape_(r.shape_), dist_(r.dist_),size_(r.size_)
            {
                ptr_ = (Complex *)std::malloc(size_*sizeof(Complex));
                std::memcpy(ptr_,r.ptr_,size_*sizeof(Complex));
            }

            ArrC<N> & operator=(ArrC<N> const & r)
            {
                std::free(ptr_);
                shape_ = r.shape_;
                dist_  = r.dist_;
                size_  = r.size_;
                ptr_ = (Complex *)std::malloc(size_*sizeof(Complex));
                std::memcpy(ptr_,r.ptr_,size_*sizeof(Complex));
                return *this;
            }

            // move
            ArrC(ArrC<N> && r)
            : shape_(std::move(r.shape_)), dist_(std::move(r.dist_)),size_(r.size_)
            {
                ptr_ = r.ptr_;
                r.ptr_ = nullptr;
            }

            ArrC<N> & operator=(ArrC<N> && r)
            {
                std::free(ptr_);
                shape_ = std::move(r.shape_);
                dist_  = std::move(r.dist_);
                size_  = r.size_;
                ptr_ = r.ptr_;
                r.ptr_ = nullptr;
                return *this;
            }

            // allocation
            ArrC(std::array<INT,N> const & sp)
            : shape_(sp), size_(std::accumulate(sp.begin(),sp.end(),1,std::multiplies<INT>()))
            {
                make_distance();
                ptr_ = (Complex *)std::calloc(size_,sizeof(Complex));
            }

            ArrC(Brace sp)
            : size_(std::accumulate(sp.begin(),sp.end(),1,std::multiplies<INT>()))
            {
                std::copy(sp.begin(),sp.end(),shape_.begin());
                make_distance();
                ptr_ = (Complex *)std::calloc(size_,sizeof(Complex));
            }

            // dtors
            ~ArrC()
            {
                std::free(ptr_);
                ptr_ = nullptr;
            }

            /* members */

            INT size() const
            {
                return size_;
            }

            const std::array<INT,N> & shape() const
            {
                return shape_;
            }

            const std::array<INT,N> & dist() const
            {
                return dist_;
            }

            Complex * ptr()
            {
                return ptr_;
            }

            const Complex * cptr() const
            {
                return ptr_;
            }

            /* arithmetics */

            // plus
            ArrC<N> & operator+=(ArrC<N> const & r)
            {
                Complex ones(1.0);
                cblas_zaxpy(size_,&ones,r.ptr_,1,ptr_,1);
                return *this;
            }

            ArrC<N> operator+(ArrC<N> const & r) const
            {
                ArrC<N> res(*this);
                Complex ones(1.0);
                cblas_zaxpy(size_,&ones,r.ptr_,1,res.ptr_,1);
                return res;
            }

            // minus
            ArrC<N> & operator-=(ArrC<N> const & r)
            {
                Complex ones(-1.0);
                cblas_zaxpy(size_,&ones,r.ptr_,1,ptr_,1);
                return *this;
            }

            ArrC<N> operator-(ArrC<N> const & r) const
            {
                ArrC<N> res(*this);
                Complex ones(-1.0);
                cblas_zaxpy(size_,&ones,r.ptr_,1,res.ptr_,1);
                return res;
            }
            // multiplication
            ArrC<N> & operator*=(double val)
            {
                cblas_zdscal(size_,val,ptr_,1);
                return *this;
            }

            ArrC<N> operator*(Complex const & val) const
            {
                ArrC<N> res(*this);
                cblas_zscal(size_,&val,res.ptr_,1);
                return res;
            }

            // division
            ArrC<N> & operator/=(double val)
            {
                double vald = 1.0/val;
                cblas_zdscal(size_,vald,ptr_,1);
                return *this;
            }

            ArrC<N> operator/(Complex const & val) const
            {
                Complex vald = 1.0/val;
                ArrC<N> res(*this);
                cblas_zscal(size_,&vald,res.ptr_,1);
                return res;
            }

            /* modifiers */

            // index
            Complex & operator()(std::array<INT,N> const & idx)
            {
                INT pos = std::inner_product(idx.begin(),idx.end(),dist_.begin(),0);
                if(pos>=size_)
                {
                    std::cout << "Overflow!" << std::endl;
                    std::exit(1);
                }
                return this->ptr()[pos];
            }

            Complex & operator()(Brace idx)
            {
                INT pos = std::inner_product(idx.begin(),idx.end(),dist_.begin(),0);
                if(pos>=size_)
                {
                    std::cout << "Overflow!" << std::endl;
                    std::exit(1);
                }
                return this->ptr()[pos];
            }

            const Complex & operator()(std::array<INT,N> const & idx) const
            {
                INT pos = std::inner_product(idx.begin(),idx.end(),dist_.begin(),0);
                if(pos>=size_)
                {
                    std::cout << "Overflow!" << std::endl;
                    std::exit(1);
                }
                return this->cptr()[pos];
            }

            const Complex & operator()(Brace idx) const
            {
                INT pos = std::inner_product(idx.begin(),idx.end(),dist_.begin(),0);
                if(pos>=size_)
                {
                    std::cout << "Overflow!" << std::endl;
                    std::exit(1);
                }
                return this->cptr()[pos];
            }
            /* IO */

            // print out the array
            void print(INT ncol) const
            {
                INT leg = 1;
                std::for_each(shape_.begin(),shape_.end(),[&leg](INT x){std::cout << "LEN " << leg << " : " << x << std::endl; ++leg;});
                INT Nw = 10, Nr = size_ / ncol;
                for(INT i=0;i<Nr;++i)
                {
                    for(INT j=0;j<ncol;++j)
                    {
                        std::cout << std::scientific << std::setw(Nw) << ptr_[i*ncol+j] << " ";
                    }
                    std::cout << std::endl;
                }
            }
            void print_sp(double tol = 1e-15) const
            {
                INT leg = 1;
                std::for_each(shape_.begin(),shape_.end(),[&leg](INT x){std::cout << "LEN " << leg << " : " << x << std::endl; ++leg;});
                for(INT i=0;i<size_;++i)
                {
                    if(std::abs(ptr_[i])<=tol)
                    {
                        continue;
                    }
                    std::cout << '[';
                    for(INT j=0;j<N;++j)
                    {
                        std::cout << i / dist_[j] % shape_[j];
                        if(j!=N-1)
                        {
                            std::cout << " ";
                        }
                    }
                    std::cout << "] | " << ptr_[i] << std::endl;
                }
            }

            // randomly fill
            void rand_fill()
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(-1,1);
                for(INT i=0;i<this->size();++i)
                {
                    this->ptr()[i] = dis(gen);
                }
            }
            // fill with 1
            void ones(double val = 1.0)
            {
                for(INT i=0;i<this->size();++i)
                {
                    ptr_[i] = val;
                }
            }
            // eye operator
            static ArrC<N> eye(INT Ns)
            {
                std::array<INT,N> rsp;
                rsp.fill(Ns);
                ArrC<N> res(rsp);
                for(INT i=0;i<Ns;++i)
                {
                    std::array<INT,N> idx;
                    idx.fill(i);
                    res(idx) = 1.0;
                }
                return res;
            }

            // slice operation
            ArrC<N> slice(Brace Ahead, Brace ATail)
            {
                std::array<INT,N> rsp;
                for(INT i=0;i<N;++i)
                {
                    rsp[i] = *(ATail.begin()+i) - *(Ahead.begin()+i);
                }
                ArrC<N> res(rsp);
                for(INT i=0;i<res.size_;++i)
                {
                    std::array<INT,N> idx;
                    for(INT j=0;j<N;++j)
                    {
                        idx[j] = i / res.dist_[j] % res.shape_[j] + *(Ahead.begin()+j);
                    }
                    res.ptr_[i] = (*this)(idx);
                }
                return res;
            }
        };

        // sparse tensor class
        template<INT N>
        class SparR : public std::vector<std::array<INT,N>>
        {
            public:
            SparR() = default;
            SparR(SparR<N> const & r)
            : shape_(r.shape_), values_(r.values_), std::vector<std::array<INT,N>>(r)
            {

            }
            SparR<N> & operator=(SparR<N> const & r)
            {
                shape_ = r.shape_;
                values_ = r.values_;
                std::vector<std::array<INT,N>>::operator=(r);
                return *this;
            }
            SparR(SparR<N> && r)
            : shape_(std::move(r.shape_)), values_(std::move(r.values_)), 
            std::vector<std::array<INT,N>>(std::move(r))
            {

            }
            SparR<N> & operator=(SparR<N> && r)
            {
                shape_ = std::move(r.shape_);
                values_ = std::move(r.values_);
                std::vector<std::array<INT,N>>::operator=(std::move(r));
                return *this;
            }
            SparR(std::array<INT,N> const & Sp, INT nnz = 0)
            : shape_(Sp), values_(nnz), std::vector<std::array<INT,N>>(nnz)
            {

            }
            SparR(std::initializer_list<INT> Sp, INT nnz = 0)
            : values_(nnz), std::vector<std::array<INT,N>>(nnz)
            {
                std::copy(Sp.begin(),Sp.end(),shape_.begin());
            }
            SparR(ArrR<N> const & r)
            : shapes_(r.shape())
            {
                for(INT i=0;i<r.size();++i)
                {
                    if(std::abs(r.cptr()[i])<=1e-13)
                    {
                        continue;
                    }
                    std::array<INT,N> idx;
                    for(INT j=0;j<N;++j)
                    {
                        idx[j] = i / r.dist()[j] % r.shape()[j];
                    }
                    this->push_back(idx);
                    values_.push_back(r.cptr()[i]);
                }
            }
            std::array<INT,N> const & shape() const
            {
                return shape_;
            }
            std::vector<double> const & values() const
            {
                return values_;
            }
            std::vector<double> & values()
            {
                return values_;
            }
            SparR<N> & operator+=(SparR<N> const & r)
            {
                for(INT i=0;i<r.size();++i)
                {
                    auto fd = std::find(this->begin(),this->end(),r[i]);
                    if(fd==this->end())
                    {
                        this->push_back(r[i]);
                        values_.push_back(r.values_[i]);
                    }
                    else
                    {
                        INT pos = std::distance(this->begin(),fd);
                        values_[pos] += r.values_[i];
                    }
                }
                return *this;
            }
            SparR<N> & operator*=(double Val)
            {
                for(INT i=0;i<this->size();++i)
                {
                    values_[i] *= Val;
                }
                return *this;
            }
            void add_elem(std::array<INT,N> const & Idx, double Val)
            {
                this->push_back(Idx);
                values_.push_back(Val);
            }
            void add_elem(std::initializer_list<INT> Idx, double Val)
            {
                this->emplace_back(Idx);
                values_.push_back(Val);
            }
            ~SparR() = default;

            private:
            std::array<INT,N> shape_;
            std::vector<double> values_;
        };
    }
}
