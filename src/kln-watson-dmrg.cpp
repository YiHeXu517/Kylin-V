/* iCI main program */
#include "../watson/dmrg.h"
using namespace KylinVib;
using namespace KylinVib::Watson;
using std::stoul;
using std::stod;
using std::string;
int main(int argc, char ** argv)
{
    string nmodstr(argv[2]), noccstr(argv[3]);
    DMRG cores(argv[1],stoul(nmodstr),stoul(noccstr));
    cores.compress();
    cores.print_ham();
    return 0;
}
