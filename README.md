# Kylin-V

#### Description
Easily-used and improved code for TD-DMRG (ex-ph coupling model) and Vibrational heat-bath CI.

#### Author's declaim
The author is weak in english and making friends thus this package is as simple as he can. No unstable external library or
package is needed. The only required are Intel MKL and NumPy package.

#### Installation
Ensure that you have installed MKL and NumPy. The dependence must be checked:
1.  C++ >= 7.5
2.  Intel MKL >= 2020.1
3.  Python3/NumPy
4.  Linux system

Aftering getting the .zip package (Yes, although this package is open-source, you can either mail to the author or go to 'https://github/YiHeXu517/Kylin-V' for newest version), run the following commands:
1. cd Kylin-V
2. make

Before making, be sure that the Makefile is correctly written. The original Makefile is author's own. You may need to change the compiler/path inside.

#### Instructions
After making, the directory '\bin' contains all binary files and a .py script 'nls.py'.

1.  Be sure to modify the file header of nls.py to make nls.py can run by ./nls.py. For TD-DMRG users, the main steps are "Build MPO(Hamiltonian) - Build MPS(initial wave function) - Perform TD-DMRG - analyze".


2. Build MPO: 'kln-holstein.a -i [input file] -t [threshold]'. The format of input file is shown in example tests/py24.inp. The Hamiltonain terms are explained as:

electron part

N_el N_Term

i i E_{ii}

i j E_{ij}

...


vibration part

N_vib

1 w_1 d_1

...

K w_K d_K

...


1st-order coupling part

N_Term

K i j g_{ij}^{K}

...


2nd-order coupling part

N_Term

K L i j r_{ij}^{KL}

...

reordering

(Defaut: 0 1 2 ... N_el+Nvib-1, here you can reorder all sites)


3. Build MPS: after MPO building, all excitonic states with zero vibrational quanta are saved in 'State/Ele-'. If you want a coherent state you can use 'kln-add.a -a [s1] -b [s2] -x [c1] -y [c2] -o [r]' to get |r> = c1|s1> + c2|s2>.

4. TD-DMRG:   use the command 'kln-tdvp.a -w [wave function at time 0] -o [Hamiltonian] -s [Krylov,2TDVP,SS,1TDVP] -t [SVD threshold] -D [maximal MPS bond dimension] -v [time step]'. Wave functions at all steps are saved in 'State/'. Make sure that this directory exist.

5. Analyze:   use the command 'kln-join.a -a [s1] -b [s2] -o [operator O]' to calculate <s1|O|s2>. The wavfunctions and possibly useful operators can be found in 'State/' and 'Op' respectively.


For HM-TD-DMRG users, only difference is the MPO

1. Build MPO: 'kln-holstein-map.a -i [input file] -t [threshold]' then one can use TDVP subsequently.

How to generate such a file?

1. Just use the script "/bin/hm.py":  run "./hm.py [input file]" and then you get a new input file "map.input" which is used above.
2. All HM settings like Ndir, Nblock cannot set in hm.py.

For VHCI users

1. 'kln-watson-hci.a [input file] [Number of targeted states] [Maximal total quanta] [HCI threshold] [ENPT2 threshold]'. An example of input file is './tests/h2co.inp'.

Contact:

Any other problems you can contact the author by email : yihe_xu@smail.nju.edu.cn

