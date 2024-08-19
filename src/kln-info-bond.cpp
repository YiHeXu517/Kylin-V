#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../holstein/lattice.h"
#include "../holstein/svd.h"

using namespace KylinVib;

static const option long_op[]={
    {"input",required_argument,NULL,'i'},
    {"type",required_argument,NULL,'t'}, 
    {"help",no_argument,NULL,'h'},
    {0,0,0,0}
};


int main( int argc, char ** argv )
{
    int ch;
    int op_ind = 0;
    string fn,ftyp;
    while((ch=getopt_long(argc,argv,"i:t:h",long_op,&op_ind))!=-1)
    {
        switch (ch) 
        {
            case 'i':    
                fn = optarg;
                break;
            case 't':
                ftyp = optarg;
                break;
            case 'h':
                std::cout <<  "Print states or operators without detail blocks \n"
                          <<  "Usage: kln-info-bond.a -i [file] -t [type,mps or mpo]\n"
                          <<  "Options:\n"
                          <<  "--input,-i                       Filename of state/op file\n"
                          <<  "--help,-h                        Show this message" << std::endl;
                return 0;
                break;
            case '?':
                std::cout << "Unknown Option!\n" << std::endl;
                break;
        }
    }
    ifstream ifs(fn);
    if(ftyp.find("mps")!=string::npos) { Holstein::State::load(ifs).print_leg(2); }
    else if(ftyp.find("mpo")!=string::npos) { Holstein::Operator::load(ifs).print_leg(3); }
    else{ cout << "Unknown type!" << endl; }
    return 0;
}
