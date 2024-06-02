# Kylin-V

#### Description
Easily-used and improved code for TD-DMRG (ex-ph coupling model) and Vibrational heat-bath CI.

#### Author's declaim 
The author is weak in both english and programming thus this package is as simple as he can. Any non-standard external library or
package is needed. The only required are Intel MKL and NumPy package.  

#### Installation
Ensure that you have installed MKL and NumPy. The dependence:
1.  C++ >= 7.5
2.  Intel MKL >= 2020.1 
3.  NumPy
4.  Linux system

Aftering getting the .zip package (Yes, although this package is open-source,  it is inconvenient for the author to use github or
other platforms), run the following commands:
1. cd kylin-v 
2. make

Before making, be sure that the Makefile is correctly written. The original Makefile is author's own. You may need to change the compiler/path inside. 

#### Instructions
After making, the directory '\bin' contains all binary files and a .py script 'nls.py'.

1.  Be sure to modify the file header of nls.py to make nls.py can run by ./nls.py.

For TD-DMRG users, the main steps are "Build MPO(Hamiltonian) - Build MPS(initial wave function) - Perform TD-DMRG - analyze"

1. Build MPO: 'kln-holstein.a -i [input file] -t [threshold]'. The format of input file is shown in example tests/py24.inp
2. Build MPS: after MPO building, all excitonic states with zero vibrational quanta are saved in 'State/Ele-'. If you want a coherent state you can use 'kln-add.a -a [s1] -b [s2] -x [c1] -y [c2] -o [r]' to get |r> = c1|s1> + c2|s2>.
3. TD-DMRG:   use the command 'kln-tdvp.a -w [wave function at time 0] -o [Hamiltonian] -s [Krylov,2TDVP,SS,1TDVP] -t [SVD threshold] -D [maximal MPS bond dimension] -v [time step]'. Wave functions at all steps are saved in 'State/'. Make sure that this directory exist.

For HM-TD-DMRG users, only difference is the MPO 

1. Build MPO: 'kln-holstein-map.a -i [input file] -t [threshold]'. The format of input file is shown in example tests/183-map.inp.

How to generate such a file? 

For VHCI users

1. 'kln-watson-hci.a [input file] [Number of targeted states] [Maximal total quanta] [HCI threshold] [ENPT2 threshold]' 

