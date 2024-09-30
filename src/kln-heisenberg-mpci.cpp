/* holstein model construction */

#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../watson/lattice.h"
#include "../watson/gl_dmrg.h"
#include "../watson/timer.h"
using namespace KylinVib;
using namespace KylinVib::Heisenberg2;

int main( int argc, char ** argv )
{
  //Timer tm;
  std::string Nsstr(argv[1]), Ncstr(argv[2]);
  size_t Ns = std::stoul(Nsstr), Ncfg = std::stoul(Ncstr);
  Lattice lat(Ns);
  MPO<double> ham = lat.gen_total();
  std::vector<MPS<double>> ss;
  for(size_t j=0;j<Ncfg;++j)
  {
    MPS<double> s(Ns,2,1);
    s.canon();
    s *= 1.0 / s[0].norm();
    double MaxE = ham.join(s,s);
    ss.push_back(s);
    std::cout << std::scientific << "Init Energy = " << MaxE / Ns << std::endl;
  }
  GLDMRG driver(ham,ss,Ncfg,20,1e-8,20);
  driver.naive_impl(100);
  return 0;
}
