/* holstein model construction */

#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../holstein/lattice.h"
#include "../holstein/svd.h"

using namespace klnX;

static const option long_op[]={
    {"input",required_argument,NULL,'i'},
    {"tolerance",required_argument,NULL,'t'},
    {"help",no_argument,NULL,'h'},
    {0,0,0,0}
};


int main( int argc, char ** argv )
{
    if(access("Op",F_OK)==-1) { int status = mkdir("Op",S_IRWXU | S_IRWXG | S_IROTH |S_IXOTH); 
    if(status!=0) { cout << "Fail to create New Dir" << endl; exit(1); } }
    if(access("State",F_OK)==-1) { int status = mkdir("State",S_IRWXU | S_IRWXG | S_IROTH |S_IXOTH); 
    if(status!=0) { cout << "Fail to create New Dir" << endl; exit(1); } }
 
    int ch;
    int op_ind = 0;
    string tole;
    Holstein::DumpConfig Ir;
    while((ch=getopt_long(argc,argv,"i:t:h",long_op,&op_ind))!=-1)
    {
        switch (ch) 
        {
            case 'i':    
                Holstein::read_dump(optarg, Ir);
                break;
            case 't':
                tole = optarg;
                break;
            case 'h':
                std::cout <<  "Construct U(1)-symmetry conserved operators with mapping\n"
                          <<  "Usage: kln-holstein-map.a -i [file]\n"
                          <<  "Options:\n"
                          <<  "--input,-i                       Filename of input file\n"
                          <<  "--tolerance,-t                   Truncation tolerance\n"
                          <<  "--help,-h                        Show this message" << std::endl;
                return 0;
                break;
            case '?':
                std::cout << "Unknown Option!\n" << std::endl;
                break;
        }
    }
    // electronic states
    Holstein::all_ele_states(Ir);

    // all elementary operators
    vector<Holstein::Operator> RPO,LPO,RAO,LAO;
    Holstein::non_eye_op(RPO,Ir,Holstein::LocalOperator::PhyRaise);
    Holstein::non_eye_op(LPO,Ir,Holstein::LocalOperator::PhyLower);
    Holstein::non_eye_op(RAO,Ir,Holstein::LocalOperator::AuxRaise);
    Holstein::non_eye_op(LAO,Ir,Holstein::LocalOperator::AuxLower);

    // density operators
    Holstein::all_density(RPO,LPO);
    // total Hamiltonian
    Holstein::totals(RPO,LPO,RAO,LAO,Ir,std::stod(tole));
    return 0;
}
