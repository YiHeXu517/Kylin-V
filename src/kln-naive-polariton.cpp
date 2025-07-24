/* naive HTC model construction */

#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../polariton/lattice.h"
#include "../polariton/dmrg.h"

using namespace KylinVib;
using namespace KylinVib::Polariton;

int main( int argc, char ** argv )
{
    
    int nsite = 2*Nmol+1;
    std::vector<int> SiteDims(nsite);
    std::map<int,int> ExLabs;
    std::map<int,int> VibLabs;
    for(int mol=0;mol<Nmol;++mol)
    {
        NaiveHTCLattice::ref_position(mol,'E',SiteDims,ExLabs,VibLabs);
        NaiveHTCLattice::ref_position(mol,'V',SiteDims,ExLabs,VibLabs);
        NaiveHTCLattice::ref_position(mol,'C',SiteDims,ExLabs,VibLabs);
    }
    NaiveHTCLattice latt(SiteDims,ExLabs,VibLabs);
    MPO<MKL_Complex16> ham = latt.gen_ctotal_para(SiteDims);
    std::vector<MPO<MKL_Complex16>> LO(nsite),UO(nsite);
    for(int site=0;site<nsite;++site)
    {
        LO[site] = latt.gen_lower(SiteDims,site);
        UO[site] = latt.gen_upper(SiteDims,site);
    }
    int Nm0 = Nmol;

    std::vector<MPS<MKL_Complex16>> state0;
    for(int m=0;m<Nm0;++m)
    {
        state0.emplace_back(nsite,SiteDims,1);
        for(int mol=0;mol<Nmol;++mol)
        {
            for(int d=3;d<VibCut;++d)
            {
                state0.back()[VibLabs[mol]]({0,d,0}) *= 0.0;
            }
        }
        state0.back().canon();
        state0.back()[0] *= 1.0/state0.back()[0].norm();
    }
    
    std::vector<MKL_Complex16> Eg,Ee,Ef;

    FEAST driver_g(ham,state0,1e-8,64);
    double Emin = 0, Emax = 0.5;
    driver_g.naive_impl(Emin,Emax,Eg);
    std::vector<MPS<MKL_Complex16>> Vmg = driver_g.get_mps();
    int Ntg = Vmg.size();

    FEAST driver_e(ham,state0,1e-8,64);
    Emin = 1.5, Emax = 2.5;
    driver_e.naive_impl(Emin,Emax,Ee);
    std::vector<MPS<MKL_Complex16>> Vme = driver_e.get_mps();
    int Nte = Vme.size();

    FEAST driver_f(ham,state0,1e-8,64);
    Emin = 3.5, Emax = 4.5;
    driver_f.naive_impl(Emin,Emax,Ef);
    std::vector<MPS<MKL_Complex16>> Vmf = driver_f.get_mps();
    int Ntf = Vmf.size();

    /* effective matrix */
    Dense<MKL_Complex16,2> dipole_ge({Ntg,Nte}),dipole_ef({Nte,Ntf}),adown_ge({Ntg,Nte})
    ,adown_ef({Nte,Ntf});
    for(int e=0;e<Nte;++e)
    {
        for(int g=0;g<Ntg;++g)
        {
            for(auto const & [key,val] : ExLabs)
            {
                dipole_ge({g,e}) += LO[val].join(Vmg[g],Vme[e]);
            }
            adown_ge({g,e}) = LO[Nmol/2].join(Vmg[g],Vme[e]);
        }
        for(int f=0;f<Ntf;++f)
        {
            for(auto const & [key,val] : ExLabs)
            {
                dipole_ef({e,f}) += LO[val].join(Vme[e],Vmf[f]);
            }
             adown_ef({e,f}) = LO[Nmol/2].join(Vme[e],Vmf[f]);
        }
    }

    /* signals */

}