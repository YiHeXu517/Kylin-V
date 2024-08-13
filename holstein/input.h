/* Read parameters from input files */

#ifndef input_H
#define input_H

#include "../util/timer.h"

namespace KylinVib
{
    using std::vector;
    using std::ifstream;
    using std::ofstream;
    using std::tuple;
    using std::string;
    using std::getline;
    using std::make_tuple;

    namespace Holstein
    {
        /* including all input parameters */     
        struct InputConfig
        {
            /* number of electronic states */
            INT NumEle;

            /* number of vibrational modes */
            INT NumVib;

            /* sites to label */
            vector<INT> SiteToLab;

            /* label to sites */
            map<INT,INT> LabToSite;

            /* dimension of each site */
            map<INT,INT> LabToDim;

            /* excitonic Hamiltonian */
            vector<tuple<INT,INT,double>> exc;

            /* pure vibrational Hamiltonian */
            map<INT,double> LabToFreq;

            /* linear coupling  */
            vector<tuple<INT,INT,INT,double>> LinearCoupling;

            /* bilinear coupling */
            vector<tuple<INT,INT,INT,INT,double>> BilinearCoupling;
        };

        struct DumpConfig
        {
            /* num of ele sites */
            INT NumEle;
            /* num of vib sites */
            INT NumVib;
            /* occupation number  */
            INT Occ;
            /* number of terms  */
            INT nTerms;
            /* all terms */
            vector<tuple<INT,INT,INT,INT,double>> Terms;
        };

        ostream & operator<<(ostream & os, InputConfig & r)
        {
            os << "Holstein lattice:" << endl;
            os << "Num of electronic states : " << r.NumEle << endl;
            os << "Num of vibrational modes : " << r.NumVib << endl;
            os << "Rearranged ordering      : <";
            for(INT i(0);i<r.SiteToLab.size();++i)
            {
                char labType; INT lab;
                if(r.SiteToLab[i]<0) { labType = 'a'; lab = -1*r.SiteToLab[i] - r.NumEle+1; }
                else if(r.SiteToLab[i]>=r.NumEle) { labType = 'v'; lab = r.SiteToLab[i] - r.NumEle+1; }
                else {  labType = 'e'; lab = r.SiteToLab[i] + 1; }
        
                os << "[" << r.LabToDim[r.SiteToLab[i]] << "," << labType << "(" << lab << ")],";
            } 
            os << "\b>" << endl;
            return os;
        }

        /* read the raw parameter file */
        void read_raw ( const char * fn, InputConfig & r )
        {
            ifstream ifs(fn); if(!ifs.is_open()) { cout << "Fail to open input file!" << endl; exit(1); }
            string tmp;
            while (!ifs.eof())
            {
                getline(ifs,tmp);
                if(tmp.find("electron")!=string::npos)
                {
                    INT nTerm;
                    ifs >> r.NumEle >> nTerm;
                    for(INT i(0);i<nTerm;++i)
                    {
                        INT e1,e2; double val;
                        ifs >> e1 >> e2 >> val;
                        r.exc.push_back( make_tuple(e1-1,e2-1,val) );
                        r.LabToDim[e1-1] = 2;
                    }
                }
                else if(tmp.find("vibration")!=string::npos)
                {
                    ifs >> r.NumVib;
                    for(INT i(0);i<r.NumVib;++i)
                    {
                        INT vib, occ; double fq;
                        ifs >> vib >> fq >> occ;
                        r.LabToFreq[vib+r.NumEle-1] = fq;
                        r.LabToDim[vib+r.NumEle-1] = occ+1;
                        r.LabToDim[-1*(vib+r.NumEle-1)] = occ+1; 
                    }
                }
                else if(tmp.find("1st-order")!=string::npos)
                {
                    INT nTerm;
                    ifs >> nTerm;
                    for(INT i(0);i<nTerm;++i)
                    {
                        INT vib, e1, e2; double val;
                        ifs >> vib >> e1 >> e2 >> val;
                        r.LinearCoupling.push_back( make_tuple(vib+r.NumEle-1,e1-1,e2-1,val) );
                    }
                }
                else if(tmp.find("2nd-order")!=string::npos)
                {
                    INT nTerm;
                    ifs >> nTerm;
                    for(INT i(0);i<nTerm;++i)
                    {
                        INT vib1, vib2, e1, e2; double val;
                        ifs >> vib1 >> vib2 >> e1 >> e2 >> val;
                        r.BilinearCoupling.push_back( make_tuple(vib1+r.NumEle-1,vib2+r.NumEle-1,e1-1,e2-1,val) );
                    }
                }
                else if(tmp.find("reorder")!=string::npos)
                {
                    for(INT i(0);i<r.NumEle+r.NumVib;++i)
                    {
                        INT lab; ifs >> lab;
                        r.SiteToLab.push_back(lab);
                        if(lab>=r.NumEle)           // vib site
                        {  r.SiteToLab.push_back(-1*lab);  }
                    }
                    for(INT i(0);i<r.NumEle+2*r.NumVib;++i)
                    {
                        r.LabToSite[ r.SiteToLab[i] ] = i;
                    }
                }
            }
            ifs.close();
            cout << r;
        }
        /* read the dump parameter file */
        void read_dump ( const char * fn, DumpConfig & r )
        {
            ifstream ifs(fn); if(!ifs.is_open()) { cout << "Fail to open input file!" << endl; exit(1); }
            ifs >> r.NumEle >> r.NumVib >> r.Occ >> r.nTerms;
            
            for(INT i(0);i<r.nTerms;++i)
            {
                INT lab1, lab2, lab3, lab4;
                double val;
                ifs >> lab1 >> lab2 >> lab3 >> lab4 >> val;
                r.Terms.push_back(make_tuple(lab1,lab2,lab3,lab4,val)); 
            }
            ifs.close();
        }
    }
}

#endif
