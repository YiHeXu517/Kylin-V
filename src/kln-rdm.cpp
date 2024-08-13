/* calculate expectation */

#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../holstein/lattice.h"

using namespace KylinVib;

static const option long_op[]={
    {"inputfile",required_argument,NULL,'i'},
    {"numberthesite",required_argument,NULL,'n'},
    {"help",no_argument,NULL,'h'},
    {0,0,0,0}
};


int main( int argc, char ** argv )
{
    int ch;
    int op_ind = 0;
    string fi,fn;
    INT CurSite;
    while((ch=getopt_long(argc,argv,"i:n:h",long_op,&op_ind))!=-1)
    {
        switch (ch) 
        {
            case 'i':
                fi = optarg; 
                break;
            case 'n':
                fn = optarg; 
                CurSite = stol(fn);
                break;
            case 'h':
                std::cout <<  "Calculate 2-RDM of MPS for assigned site \n"
                          <<  "Usage: kln-rdm.a -i [mps file] -n [site number]\n"
                          <<  "Options:\n"
                          <<  "--inputfile,-i                      MPS file a\n"
                          <<  "--numberthesite,-n                  Number of site\n"
                          <<  "--help,-h                           Show this message" << std::endl;
                return 0;
                break;
            case '?':
                std::cout << "Unknown Option!\n" << std::endl;
                break;
        }
    }
    ifstream ifi(fi);
     
    Holstein::State mpsa = Holstein::State::load(ifi);
    Holstein::SVDConfig r;
    r.SVDThres = 0.0;
    r.MaxDim   = 1000;
    r.Left2Right = true;
    for(INT i(0);i<CurSite;++i)
    {
        auto[cur,nex] = Holstein::svd_impl<2,1>( mpsa[i], r, -1 );
        mpsa[i] = move(cur);
        mpsa[i+1] = Holstein::product<2,3,1>( nex, mpsa[i+1], CblasNoTrans, CblasNoTrans );
    }
    Holstein::Tensor<4> X2 = Holstein::product<3,3,1>(mpsa[CurSite], mpsa[CurSite+1], CblasNoTrans, CblasNoTrans);
    Holstein::Tensor<4> X2Up = Holstein::transpose<4>(X2, {0,3,1,2});
    Holstein::Tensor<4> X2Down = X2Up;
    Holstein::Tensor<4> rdm2 = Holstein::product<4,4,2>(X2Up,X2Down,CblasConjTrans, CblasNoTrans);
    for(INT i(0);i<rdm2.num_blocks();++i)
    {
        cout << rdm2.at(i).qn()[0] << " "
             << rdm2.at(i).qn()[1] << " "
             << rdm2.at(i).qn()[2] << " "
             << rdm2.at(i).qn()[3] << " "
             << real(rdm2.at(i).cptr()[0]) << " "
             << imag(rdm2.at(i).cptr()[0]) << " "
             << endl;
    }
    return 0;
}
