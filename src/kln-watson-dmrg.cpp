/* holstein model construction */

#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>
#include "../watson/watson_dmrg/lattice.h"
#include "../watson/watson_dmrg/lanczos.h"
using namespace KylinVib;

static const option long_op[]={
    {"num-state",required_argument,NULL,'s'},
    {"help",no_argument,NULL,'h'},
    {0,0,0,0}
};

int main( int argc, char ** argv )
{
    int ch;
    int op_ind = 0;
    std::string fl,fr,fnst;
    while((ch=getopt_long(argc,argv,"s:h",long_op,&op_ind))!=-1)
    {
        switch (ch)
        {
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
 
    std::time_t now = std::time(nullptr);
    std::cout << "Start is: " << std::ctime(&now) << std::endl;

    WatsonDMRG::xMPO ham = WatsonDMRG::xMPO::load("tot.H");
    INT L = ham.nsite();
    INT d = ham[0].shape()[1];
    WatsonDMRG::Lanczos driver(50,24,12,5,ham);
    driver.impl();
    now = std::time(nullptr);
    std::cout << "End is: " << std::ctime(&now) << std::endl;
    return 0;
}
