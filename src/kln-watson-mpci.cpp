/* holstein model construction */

#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../watson/lattice.h"
#include "../watson/si_dmrg.h"
#include "../watson/timer.h"
using namespace KylinVib;
using namespace KylinVib::Watson;

int main( int argc, char ** argv )
{
  mkl_set_num_threads(1);
  omp_set_num_threads(4);
  std::string Nsstr(argv[2]), dstr(argv[3]), Ncfgstr(argv[4]), bdstr(argv[5]);
  size_t Ns = std::stoul(Nsstr), d = std::stoul(dstr), Ncfg = std::stoul(Ncfgstr);
  size_t nbd = std::stoul(bdstr);

  std::string Eminstr(argv[6]), Emaxstr(argv[7]);
  double Emin = std::stod(Eminstr), Emax = std::stod(Emaxstr);

  Lattice lat(Ns,d,argv[1]);
  MPO<MKL_Complex16> ham = lat.gen_ctotal_para();
  MPS<MKL_Complex16> si(Ns,d,nbd);
  si.canon();
  si *= 1.0 / si[0].norm();

  FEAST driver(ham,si,1e-8,nbd);
  driver.naive_impl(Emin,Emax);
  return 0;
}
