/* iCI main program */

#include "../watson/iCI.h"
using namespace KylinVib;
using namespace KylinVib::Watson;
int main(int argc, char ** argv)
{
    iCI cores(argv[1]);
    string_view npstr(argv[2]), qnstr(argv[3]), tolstr(argv[4]), enptstr(argv[5]);
    for(size_t i=0;i<3;++i)
    {
        cores.plus_one();
    }
    Timer tm;
    size_t iter = 0;
    DenseBase<double> E,Prim;
    size_t Np = stoul(npstr), Qn = stoul(qnstr);
    double TolHCI = stod(tolstr), TolPT2 = stod(enptstr);
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
