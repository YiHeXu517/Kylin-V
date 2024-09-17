/* holstein model construction */

#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../watson/lattice.h"
#include "../watson/dmrg.h"
#include "../watson/timer.h"
#include "../watson/iCI.h"
using namespace KylinVib;
using namespace KylinVib::Watson;

int main( int argc, char ** argv )
{
  Timer tm;
  std::string Nsstr(argv[2]), dstr(argv[3]), Ncfgstr(argv[4]), Npstr(argv[5]);
  size_t Ns = std::stoul(Nsstr), d = std::stoul(dstr), Ncfg = std::stoul(Ncfgstr),
  Np = std::stoul(Npstr);
  Lattice lat(Ns,d,argv[1]);
  MPO<double> ham = lat.gen_total_para();
  std::map<size_t,size_t> c;
  for(size_t i=0;i<Ns;++i)
  {
    c[i] = d-1;
  }
  MPS<double> s(Ns,d,c);
  double MaxE = ham.join(s,s);
  std::cout << std::scientific << "Maximal guess energy = " << MaxE << std::endl;
  MPO<double> Eyes = lat.gen_eye();
  Eyes *= -1.0*MaxE;
  MPO<double> ham_shift = ham + Eyes;
  ham_shift.canon();
  MPS<double> hamst = ham_shift.diag_state();
  std::vector<LabArr<double,2>> liss = hamst.dominant(Ncfg);

iCI cores(argv[1]);
std::string tolstr(argv[6]), enptstr(argv[7]);
cores.clear_basis();
for(size_t i=0;i<Ncfg;++i)
{
    cores.add_basis(liss[i].labs);
}
size_t iter = 0;
DenseBase<double> E,Prim;
double TolHCI = stod(tolstr), TolPT2 = stod(enptstr);
cores.set_tol(TolHCI);
cores.set_max_qn(d);
    while(iter<=500)
    {
        if(iter==0)
        {
start_module(tm,"make-H");
        cores.make_total();
end_module(tm,"make-H");
        }

        size_t nref = cores.num_ref();

start_module(tm,"eigen");
        cores.eigen(Np,E,Prim);
end_module(tm,"eigen");

start_module(tm,"Expand-PT2");
        cores.expand(E,Prim);
end_module(tm,"Expand-PT2");

        double incre = (cores.num_ref() - nref) * 100.0 / cores.num_ref();
        cout << "Space enlarged by " << incre << "%" << endl;
        if(iter!=0 && incre < 1.0  )
        {
            cout << "Dominant basis:" << endl;
            for(size_t st=0;st<Np;++st)
            {
               size_t MaxPrimSt = cblas_idamax(nref,Prim.cptr()+st*nref,1);
               cout << "Eig-state " << st+1 << ": " << cores.get_ref_basis(MaxPrimSt)
               << " | " << Prim.cptr()[st*nref+MaxPrimSt] << endl;
            }
start_module(tm,"ENPT2");
            cores.enpt2(E,Prim,TolPT2);
end_module(tm,"ENPT2");
            break;
        }
start_module(tm,"make-Hex");
        cores.make_total_increment(nref);
end_module(tm,"make-Hex");
 
        iter++;
    }
    cout << tm << endl;
    return 0;
}
