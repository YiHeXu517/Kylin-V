/* holstein model construction */

#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>
#include "../watson/watson_dmrg/lattice.h"
#include "../watson/watson_dmrg/lanczos.h"
using namespace KylinVib;
using namespace KylinVib::WatsonDMRG;

static const option long_op[]={
    {"mpo-file",required_argument,NULL,'p'},
    {"num-state",required_argument,NULL,'s'},
    {"help",no_argument,NULL,'h'},
    {0,0,0,0}
};

int main( int argc, char ** argv )
{
    int ch;
    int op_ind = 0;
    std::string fmpo,fl,fr,fnst;
    while((ch=getopt_long(argc,argv,"p:s:h",long_op,&op_ind))!=-1)
    {
        switch (ch)
        {
            case 'p':
                fmpo = optarg;
                break;
            case 's':
                fnst = optarg;
                break;
            case 'h':
                std::cout <<  "Shift-inverse DMRG implementation \n"
                          <<  "Usage: kln-dmrg - \n"
                          <<  "Options:\n"
                          <<  "--help,-h                        Show this message" << std::endl;
                return 0;
                break;
            case '?':
                std::cout << "Unknown Option!\n" << std::endl;
                break;
        }
    }
    // electronic states
 
//    std::time_t now = std::time(nullptr);
//    std::cout << "Start is: " << std::ctime(&now) << std::endl;

    xMPO ham = xMPO::load(fmpo.c_str());
    INT L = ham.nsite();
    INT d = ham[0].shape()[1];

    std::vector<xMPO> ops(L);
    for(INT i=0;i<L;++i)
    {
        std::vector<LocalOp> labs(L);
        std::for_each(labs.begin(),labs.end(),[](LocalOp & x){ x = LocalOp::I;});
        labs[i] = LocalOp::Q;
        xMPO tmpi(L);
        for(INT j=0;j<L;++j)
        {
            tmpi[j] = to_matrix<ArrR<4>>(labs[j],d);
        }
        ops[i] = tmpi;
    }
    Lanczos driver(ham,std::stoi(fnst),5,ops.size());
    driver.impl(ops);
//    now = std::time(nullptr);
//    std::cout << "End is: " << std::ctime(&now) << std::endl;
    return 0;
}
