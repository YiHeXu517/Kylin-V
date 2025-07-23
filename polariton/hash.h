/* hash container class */

#pragma once

#include <vector>
#include <unordered_map>
#include "dense.h"

namespace KylinVib
{
  struct HashIntVec
  {
    int operator()(std::vector<int> const & myVector) const
    {
      std::hash<int> hasher;
      int answer = 0;
      for (int i : myVector)
      {
        answer ^= hasher(i) + 0x9e3779b9 +
        (answer << 6) + (answer >> 2);
      }
      return answer;
    }
  };
  template<int N>
  struct HashIntArr
  {
    int operator()(std::array<int,N> const & myArr) const
    {
      std::hash<int> hasher;
      int answer = 0;
      for (int i : myArr)
      {
        answer ^= hasher(i) + 0x9e3779b9 +
        (answer << 6) + (answer >> 2);
      }
      return answer;
    }
  };
  template<typename Type, int Rank>
  struct LabArr : public Dense<Type,Rank>
  {
    std::vector<int> labs;

    LabArr() = default;
    LabArr(LabArr<Type,Rank> const & r) : labs(r.labs),
    Dense<Type,Rank>(r)
    {

    }
    LabArr(LabArr<Type,Rank> && r) : labs(std::move(r.labs)),
    Dense<Type,Rank>(std::move(r))
    {
      
    }
    LabArr(std::initializer_list<int> Sp)
    : Dense<Type,Rank>(Sp)
    {

    }
    ~LabArr() = default;
    LabArr<Type,Rank> & operator=(LabArr<Type,Rank> const & r)
    {
      labs = r.labs;
      Dense<Type,Rank>::operator=(r);
      return *this;
    }
    LabArr<Type,Rank> & operator=(LabArr<Type,Rank> && r)
    {
      labs = std::move(r.labs);
      Dense<Type,Rank>::operator=(std::move(r));
      return *this;
    }
    void print(double tol = 1e-15) const
    {
      std::cout << "Labels:[";
      for(int i=0;i<labs.size()-1;++i)
      {
        std::cout << labs[i] << ",";
      }
      std::cout << labs.back() << "]" << std::endl;
      Dense<Type,Rank>::print(tol);
    }
    void print_special(double EShift) const
    {
      std::cout << "Labels:[";
      for(int i=0;i<labs.size()-1;++i)
      {
        std::cout << labs[i] << ",";
      }
      std::cout << labs.back() << "] | " 
      <<  this->ptr()[0]+EShift << std::endl;
    }
  };
  
}
