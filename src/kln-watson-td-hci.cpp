/* iCI main program */
#include "../watson/watson_hci.h"
using namespace KylinVib;
using namespace KylinVib::Watson;
using std::stoul;
using std::stod;
using std::string;
int main(int argc, char ** argv)
{
    string qnstr(argv[2]), tolstr(argv[3]), stepsstr(argv[4]), dtstr(argv[5]);
    cout << "================  Time-independent stage =============" << endl;
    Timer tm;
    size_t iter = 0;
    DenseBase<double> E,Prim;
    size_t Np = 1, Qn = stoul(qnstr), TimeStep = stoul(stepsstr);
    double TolHCI = stod(tolstr), TimeInterval = stod(dtstr);
    iCI cores(argv[1]);
    for(size_t i=0;i<3;++i)
    {
        cores.plus_one();
    }						    
    cores.set_tol(TolHCI);
    cores.set_max_qn(Qn);
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
               size_t MaxPrimSt = cblas_idamax(nref,Prim.ptr()+st*nref,1);
               cout << "Eig-state " << st+1 << ": " << cores.get_ref_basis(MaxPrimSt)
               << " | " << Prim.ptr()[st*nref+MaxPrimSt] << endl;
            }
            break;
        }
        start_module(tm,"make-Hex");
        cores.make_total_increment(nref);
        end_module(tm,"make-Hex");
        iter++;
    }
    size_t NrefPsi = cores.num_ref();
    DenseBase<MKL_Complex16> MuGS(NrefPsi);
    #pragma omp parallel for
    for(size_t ref=0;ref<NrefPsi;++ref)
    {
        MuGS.ptr()[ref] = Prim.ptr()[ref];
    }
    iCI dms("dipole.inp");
    dms.copy_basis(cores);
    dms.expand(MuGS);
    dms.make_total();
    dms.apply_hamiltonian(MuGS);
    cores.copy_basis(dms);
    cores.make_total_increment(dms.num_ref());
    double nmMuGS = cblas_dznrm2(MuGS.size(),MuGS.ptr(),1);
    cblas_zdscal(MuGS.size(),1.0/nmMuGS,MuGS.ptr(),1);
    array<DenseBase<MKL_Complex16>,3> MuGSEvery = {MuGS,MuGS,MuGS};
    cout << "================  Time-dependent stage =============" << endl;
    while(iter<=TimeStep)
    {
        cout << "Step " << iter << ":" << endl;
        size_t nref = cores.num_ref();
        start_module(tm,"Lanczos");
        cores.Lanczos(MuGS,TimeInterval);
        end_module(tm,"Lanczos");
        MuGSEvery[2] = MuGS;
        start_module(tm,"Expand-PT2");
        cores.expand(MuGS);
        end_module(tm,"Expand-PT2");
        double incre = (cores.num_ref() - nref) * 100.0 / cores.num_ref();
        cout << "Space enlarged by " << incre << "%" << endl;
        if(iter==TimeStep)
        {
            cout << "Dominant basis:" << endl;
            for(size_t st=0;st<50;++st)
            {
               size_t MaxPrimSt = cblas_izamax(nref,MuGS.ptr(),1);
               cout << "Basis " << st+1 << ": " << cores.get_ref_basis(MaxPrimSt)
               << " | " << MuGS.ptr()[MaxPrimSt] << endl;
               MuGS.ptr()[MaxPrimSt] = 0.0;
            }
            break;
        }
        start_module(tm,"make-Hex");
        cores.make_total_increment(nref);
        end_module(tm,"make-Hex");
        iter++;
         
        MKL_Complex16 Ct0;
        cblas_zdotc_sub(MuGSEvery[0].size(),MuGSEvery[0].ptr(),1,MuGSEvery[2].ptr(),1, &Ct0);
        cout << "< Psi_0 | Psi_t> = " << real(Ct0) << " " << imag(Ct0) << endl;

        if(iter!=0)
        {
            cblas_zdotc_sub(MuGSEvery[1].size(),MuGSEvery[1].ptr(),1,MuGSEvery[2].ptr(),1, &Ct0);
            cout << "< exp(-iH dt) > = " << real(Ct0) << " " << imag(Ct0) << endl;
        }

        MuGSEvery[1] = MuGSEvery[2];
    }
    cout << tm << endl;
    return 0;
}
