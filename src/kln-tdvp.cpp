/* TDVP program */

#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../holstein/tdvp.h"

using namespace KylinVib;
static const option long_op[]={
    {"wavefunc",required_argument,NULL,'w'},
    {"operator",required_argument,NULL,'o'},
    {"stage",required_argument,NULL,'s'},
    {"interval",required_argument,NULL,'v'},
    {"Taylor",required_argument,NULL,'Y'},
    {"max-bond",required_argument,NULL,'D'},
    {"tolerance",required_argument,NULL,'t'},
    {"help",no_argument,NULL,'h'},
    {0,0,0,0}
};

int main(int argc, char** argv){
    
    int ch;
    int op_ind = 0;
    string mpsFile, mpoFile, argiv, argss, argD, argt;
    INT maxdim=1000;
    double interval,tol=1.0e-8,tolExp=1.0e-8,maxIt = 15;
    while((ch=getopt_long(argc,argv,"w:o:s:v:D:t:h",long_op,&op_ind))!=-1)
    {
        switch (ch) 
        {
            case 'w':
                mpsFile = optarg;
                break;
            case 'o':
                mpoFile = optarg;
                break;
            case 's':
                argss   = optarg;
                break;
            case 'v':
                argiv   = optarg;
                interval= std::stod(argiv);
                break;
            case 'D':
                argD    = optarg;
                maxdim  = std::stol(argD);
                break;
            case 't':     
                argt    = optarg;
                tol     = std::stod(argt);
                break;
            case 'h':
                std::cout <<  "Program to calculate tDMRG (TDVP) \n"
                          <<  "Usage: kln-tdvp.a -w [MPS file] -o [MPO file] -s [n_K,n_2,n_sa,n_1] -D [real] -t [real] \n"
                          <<  "Options:\n"
                          <<  "--wavefunc,-w                    Filename of initial MPS\n"
                          <<  "--operator,-o                    Filename of operator for evolution\n"
                          <<  "--stage,-s                       Steps of every algos\n"
                          <<  "--interval,-v                    Time step\n"
                          <<  "--max-bond,-D                    Max bond dimension\n"
                          <<  "--tolerance,-t                   SVD trunctaion limit for mode 2\n"
                          <<  "                                 Cutoff and break limit for mode 3\n"
                          <<  "--help,-h                        Show this message" << std::endl;
                return 0;
                break;
            case '?':
                std::cout << "Unknown Option!\n" << std::endl;
                break;
        }
    } 
    ifstream ifss(mpsFile),ifso(mpoFile);
    Holstein::State mps0 = Holstein::State::load(ifss);
    Holstein::Operator mpoh = Holstein::Operator::load(ifso);
    array<INT,4> stg = Holstein::parse_stage(argss);
    Holstein::TDVP drive(mps0,mpoh,tol,tolExp,maxdim,maxIt,interval);
    drive.total_start(stg);
    return 0;
}

