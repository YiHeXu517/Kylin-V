/* holstein model construction */

#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../watson/watson_dmrg/lattice.h"
using namespace KylinVib::WatsonDMRG;

static const option long_op[]={
    {"input",required_argument,NULL,'i'},
    {"num-vib",required_argument,NULL,'n'},
    {"num-occ",required_argument,NULL,'d'},
    {"help",no_argument,NULL,'h'},
    {0,0,0,0}
};

int main( int argc, char ** argv )
{
    int ch;
    int op_ind = 0;
    std::string fn,Ls,ds;
    while((ch=getopt_long(argc,argv,"i:n:d:h",long_op,&op_ind))!=-1)
    {
        switch (ch)
        {
            case 'i':
                fn = optarg;
                break;
            case 'n':
                Ls = optarg;
                break;
            case 'd':
                ds = optarg;
                break;
            case 'h':
                std::cout <<  "Construct watson operators \n"
                          <<  "Usage: wat-latt -i [file] -n [int] -d [int]\n"
                          <<  "Options:\n"
                          <<  "--input,-i                       Filename of input file\n"
                          <<  "--num-vibs,-n                    Number of vibrational modes\n"
                          <<  "--num-occ,-d                     Maximal vibrational quanta\n"
                          <<  "--help,-h                        Show this message" << std::endl;
                return 0;
                break;
            case '?':
                std::cout << "Unknown Option!\n" << std::endl;
                break;
        }
    }
    Lattice wt(std::stol(Ls),std::stol(ds),fn.c_str());
    xMPO xham = wt.gen_ctotal();
    std::cout << "Maxbond = " << xham.max_bond() << std::endl;
    xham.save("tot.H");
    return 0;
}
