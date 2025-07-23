/* memory allocator class */

#pragma once

#include <iostream>
#include <cstring>
#include <utility>

namespace KylinVib
{
  template<typename Type>
  class Allocator
  {
    public:
    Allocator() : size_(0), ptr_(nullptr)
    {

    }
    Allocator(Allocator<Type> const & r) : size_(r.size_)
    {
      ptr_ = (Type *)std::malloc(size_*sizeof(Type));
      std::memcpy(ptr_,r.ptr_,size_*sizeof(Type));
    }
    Allocator(Allocator<Type> && r) : size_(r.size_)
    {
      ptr_ = r.ptr_;
      r.ptr_ = nullptr;
    }
    Allocator(int sz) : size_(sz)
    {
      ptr_ = (Type *)std::calloc(size_,sizeof(Type));
    }
    ~Allocator()
    {
      std::free(ptr_);
      ptr_ = nullptr;
    }
    Allocator<Type> & operator=(Allocator<Type> const & r)
    {
      std::free(ptr_);
      size_ = r.size_;
      ptr_ = (Type *)std::malloc(size_*sizeof(Type));
      std::memcpy(ptr_,r.ptr_,size_*sizeof(Type));
      return *this;
    }
    Allocator<Type> & operator=(Allocator<Type> && r)
    {
      std::free(ptr_);
      size_ = r.size_;
      ptr_ = r.ptr_;
      r.ptr_ = nullptr;
      return *this;
    }
    int size() const 
    {
      return size_;
    } 
    Type * ptr()
    {
      return ptr_;
    }
    Type const * ptr() const
    {
      return ptr_;
    }
    void print() const
    {
      std::cout << "[";
      for(int i=0;i<size_-1;++i)
      {
        std::cout << ptr_[i] << ",";
      }
      std::cout << ptr_[size_-1] << "]" << std::endl;
    }
    private:
    int size_;
    Type * ptr_;
  };
}
