/* calculate expectation */

#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../holstein/lattice.h"

using namespace klnX;

static const option long_op[]={
    {"afile",required_argument,NULL,'a'},
    {"bfile",required_argument,NULL,'b'},
    {"scalea",required_argument,NULL,'x'},
    {"scaleb",required_argument,NULL,'y'},
    {"ofile",required_argument,NULL,'o'},
    {"help",no_argument,NULL,'h'},
    {0,0,0,0}
};


int main( int argc, char ** argv )
{
    int ch;
    int op_ind = 0;
    string fa,fb,fx,fy,fo;
    double scla,sclb;
    while((ch=getopt_long(argc,argv,"a:b:x:y:o:h",long_op,&op_ind))!=-1)
    {
        switch (ch) 
        {
            case 'a':
                fa = optarg; 
                break;
            case 'b':
                fb = optarg; 
                break;
            case 'x':
                fx = optarg; 
                scla = std::stod(fx);
                break;
            case 'y':
                fy = optarg; 
                sclb = std::stod(fy);
                break;
            case 'o':
                fo = optarg;
                break;
            case 'h':
                std::cout <<  "Calculate (|a> + |b>)*s \n"
                          <<  "Usage: kln-add.a -a [file a] -b [file b] -s [scl] -o [ofile]\n"
                          <<  "Options:\n"
                          <<  "--afile,-a                       MPS file a\n"
                          <<  "--bfile,-b                       MPS file b\n"
                          <<  "--scalea,-x                      re-scale a\n"
                          <<  "--scaleb,-y                      re-scale b\n"
                          <<  "--ofile,-o                       result MPS file\n"
                          <<  "--help,-h                        Show this message" << std::endl;
                return 0;
                break;
            case '?':
                std::cout << "Unknown Option!\n" << std::endl;
                break;
        }
    }
    ifstream ifa(fa),ifb(fb);
    Holstein::Operator mpsa = Holstein::Operator::load(ifa);
    Holstein::Operator mpsb = Holstein::Operator::load(ifb);
    mpsa *= scla;
    mpsa += mpsb * sclb;
    ofstream ofo(fo);
    mpsa.save(ofo);
    return 0;
}
