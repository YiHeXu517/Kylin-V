/* Read parameters from lattice files */

#ifndef hols_lattice_H
#define hols_lattice_H

#include "input.h"
#include "local.h"

namespace KylinVib
{
    namespace Holstein
    {
        using std::string;
        using std::to_string;
        // generatr eye operator
        void eye_op ( Operator & EyeOps, InputConfig & s )
        {
            INT ns = s.SiteToLab.size();
            for(INT i(0);i<ns;++i)
            {
                gen_local_op(EyeOps[i],LocalOperator::Eye, s.LabToDim[s.SiteToLab[i]] ,0);
            }
        }
        void eye_op ( Operator & EyeOps, DumpConfig & s )
        {
            for(INT i(0);i<s.NumEle;++i)
            {
                gen_local_op(EyeOps[i],LocalOperator::Eye, 2  ,0);
            }
            for(INT i(0);i<2*s.NumVib;++i)
            {
                gen_local_op(EyeOps[i+s.NumEle],LocalOperator::Eye, s.Occ + 1 ,0);
            }
        }
 
 
        // generatr other local operators
        void non_eye_op ( vector<Operator> & Ops, InputConfig & s, LocalOperator lo )
        {
            INT ns = s.SiteToLab.size();
            for(INT i(0);i<ns;++i)
            {
                Operator opi(ns);
                for(INT j(0);j<ns;++j)
                {
                    INT lastB = (j==0)?0:opi.at(j-1).at(0).qn()[3];
                    if(i==j)
                    {
                        gen_local_op(opi[j], lo , s.LabToDim[s.SiteToLab[j]] ,lastB);
                    }
                    else
                    {
                        gen_local_op(opi[j], LocalOperator::Eye, s.LabToDim[s.SiteToLab[j]] ,lastB);
                    }
                }
                Ops.push_back(opi);
            }
        }
        void non_eye_op ( vector<Operator> & Ops, DumpConfig & s, LocalOperator lo )
        {
            INT ns = s.NumEle + 2*s.NumVib;
            for(INT i(0);i<ns;++i)
            {
                Operator opi(ns);
                for(INT j(0);j<ns;++j)
                {
                    INT lastB = (j==0)?0:opi.at(j-1).at(0).qn()[3];
                    INT occB  = (j<s.NumEle)?2:s.Occ+1;
                    if(i==j)
                    {
                        gen_local_op(opi[j], lo , occB  ,lastB);
                    }
                    else
                    {
                        gen_local_op(opi[j], LocalOperator::Eye, occB ,lastB);
                    }
                }
                Ops.push_back(opi);
            }
        }
        
        // generate all-ele-lab initial pure states
        void all_ele_states ( InputConfig & s )
        {
            INT ns = s.SiteToLab.size();
            INT nEl = s.NumEle;
            for(INT i(0);i<nEl;++i)
            {
                State esi(ns);
                for(INT j(0);j<ns;++j)
                {
                    INT lastB = (j==0)?0:esi.at(j-1).at(0).qn()[2];
                    INT labj = s.SiteToLab[j]; 
                    INT Dimj = s.LabToDim[labj];
                    if(labj==i)      // target ele sites
                    {
                        Block<3> rjs ( {1,1,1}, {lastB, 1, lastB+1} ); rjs({0,0,0}) = 1.0;
                        esi[j].push_back(rjs);
                        esi[j].check_legs();
                    }
                    else if(labj!=i && labj<nEl && labj>=0) // other ele sites
                    {
                        Block<3> rjs ( {1,1,1}, {lastB, 0, lastB} ); rjs({0,0,0}) = 1.0;
                        esi[j].push_back(rjs);
                        esi[j].check_legs();
                    }
                    else if(labj>=nEl)     // vib sites
                    {
                        Block<3> rjs ( {1,1,1}, {lastB, 0, lastB} ); rjs({0,0,0}) = 1.0;
                        esi[j].push_back(rjs);
                        esi[j].check_legs();
                    } 
                    else                   // aux sites
                    {
                        Block<3> rjs ( {1,1,1}, {lastB, Dimj-1, lastB+Dimj-1} ); rjs({0,0,0}) = 1.0;
                        esi[j].push_back(rjs);
                        esi[j].check_legs();
                    }
                }
                string fn = "State/Ele-" + to_string(i+1) + ".mps";
                ofstream ofs(fn);
                esi.save(ofs);
            }
        }
        void all_ele_states ( DumpConfig & s )
        {
            INT ns = s.NumEle + s.NumVib * 2;
            INT nEl = s.NumEle;
            for(INT i(0);i<nEl;++i)
            {
                State esi(ns);
                for(INT j(0);j<ns;++j)
                {
                    INT lastB = (j==0)?0:esi.at(j-1).at(0).qn()[2];
                    INT labj = j;
                    INT Dimj =  (j<s.NumEle)?2:s.Occ+1;
                    if(labj==i)      // target ele sites
                    {
                        Block<3> rjs ( {1,1,1}, {lastB, 1, lastB+1} ); rjs({0,0,0}) = 1.0;
                        esi[j].push_back(rjs);
                        esi[j].check_legs();
                    }
                    else if(labj!=i && labj<nEl) // other ele sites
                    {
                        Block<3> rjs ( {1,1,1}, {lastB, 0, lastB} ); rjs({0,0,0}) = 1.0;
                        esi[j].push_back(rjs);
                        esi[j].check_legs();
                    }
                    else if(labj>=nEl && (labj-nEl)%2==0)     // vib sites
                    {
                        Block<3> rjs ( {1,1,1}, {lastB, 0, lastB} ); rjs({0,0,0}) = 1.0;
                        esi[j].push_back(rjs);
                        esi[j].check_legs();
                    } 
                    else                   // aux sites
                    {
                        Block<3> rjs ( {1,1,1}, {lastB, Dimj-1, lastB+Dimj-1} ); rjs({0,0,0}) = 1.0;
                        esi[j].push_back(rjs);
                        esi[j].check_legs();
                    }
                }
                string fn = "State/Ele-" + to_string(i+1) + ".mps";
                ofstream ofs(fn);
                esi.save(ofs);
            }
        }


        // generate all density matrices
        void all_density( const vector<Operator> & RPO, const vector<Operator> & LPO )
        {
            INT ns = RPO.size();
            for(INT j(0);j<ns;++j)
            {
                Operator denj = mpo_dot(RPO.at(j),LPO.at(j));
                string fn = "Op/Density_" + to_string(j+1);
                ofstream ofs(fn);
                denj.save(ofs);
            }
        }

        // generate total Hamiltonian
        void totals ( const vector<Operator> & RPO, const vector<Operator> & LPO, const vector<Operator> & RAO,
        const vector<Operator> & LAO, InputConfig & s, vector<Operator> & totals, double tol )
        {
            // pure electronic Hamitoian
            INT ns = RPO.size();
            Operator ey(ns);
            eye_op(ey,s);
            ofstream ofey("Op/Eye");
            ey.save(ofey);

            Operator PureEle;
            for(INT i(0);i<s.exc.size();++i)
            {
                cout << "ele term " << i+1 << endl;
                INT e1 = get<0>(s.exc[i]);
                INT e2 = get<1>(s.exc[i]);
                double val = get<2>(s.exc[i]);
                INT site1 = s.LabToSite[e1];
                INT site2 = s.LabToSite[e2];
                Operator t12 = mpo_dot(RPO[site1],LPO[site2]);
                t12 *= val;
                if(i==0)
                {
                    PureEle = move(t12);
                }
                else
                {
                    PureEle += t12;
                } 
            } 
            PureEle.truncate();
            totals.push_back(PureEle);
            cout << "Electronic part finished." << endl;
            string fne = "Op/H_e";
            ofstream ofse(fne);
            PureEle.save(ofse);

            // pure vibrational Hamiltonian
            INT nTerm = 0;
            Operator PureVib;
            for(auto it=s.LabToFreq.begin();it!=s.LabToFreq.end();++it)
            {
                INT vib = it->first;
                cout << "vib term " << vib << endl;
                double val = it->second;
                INT site = s.LabToSite[vib];
                Operator w = mpo_dot(RPO[site],LPO[site]);
                w *= val;
                if(nTerm==0)
                {
                    PureVib = move(w);
                }
                else
                {
                    PureVib += w;
                } 
                ++nTerm;
            } 
            PureVib.truncate(tol);
            totals.push_back(PureVib);
            cout << "Vibrational part finished." << endl;
            string fnv = "Op/H_v";
            ofstream ofsv(fnv);
            PureVib.save(ofsv);


            // first order
            Operator Fir;
            for(INT i(0);i<s.LinearCoupling.size();++i)
            {
                cout << "1st-ord term " << i+1 << endl;
                INT vib = get<0>(s.LinearCoupling[i]);
                INT e1 = get<1>(s.LinearCoupling[i]);
                INT e2 = get<2>(s.LinearCoupling[i]);
                double val = get<3>(s.LinearCoupling[i]);
                INT sitev = s.LabToSite[vib];
                INT site1 = s.LabToSite[e1];
                INT site2 = s.LabToSite[e2];
                Operator t12 = mpo_dot(RPO[site1],LPO[site2]);
                Operator bp = mpo_dot(RPO[sitev],LAO[sitev+1]);
                Operator pb = mpo_dot(LPO[sitev],RAO[sitev+1]);
                bp += pb;
                Operator g3 = mpo_dot(t12,bp);
                g3 *= val*sqrt(0.5);
                if(i==0)
                {
                    Fir = move(g3);
                }
                else
                {
                    Fir += g3;
                } 
                if( i/10 != (i+1)/10 )   // truncate every n terms
                { Fir.truncate(); }
            }
            totals.push_back(Fir);
            cout << "1st-order part finished." << endl;
            string fn1 = "Op/H_1";
            ofstream ofs1(fn1);
            Fir.save(ofs1);

            // second order
            if(s.BilinearCoupling.size()!=0) 
            {
                Operator Sec;
                for(INT i(0);i<s.BilinearCoupling.size();++i)
                {
                    INT vib1 = get<0>(s.BilinearCoupling[i]);
                    INT vib2 = get<1>(s.BilinearCoupling[i]);
                    INT e1 = get<2>(s.BilinearCoupling[i]);
                    INT e2 = get<3>(s.BilinearCoupling[i]);
                    double val = get<4>(s.BilinearCoupling[i]);
     
                    INT sitev1 = s.LabToSite[vib1];
                    INT sitev2 = s.LabToSite[vib2];
                    INT site1 = s.LabToSite[e1];
                    INT site2 = s.LabToSite[e2];

                    Operator t12 = mpo_dot(RPO[site1],LPO[site2]);
   
                    Operator bp1 = mpo_dot(RPO[sitev1],LAO[sitev1+1]);
                    Operator pb1 = mpo_dot(LPO[sitev1],RAO[sitev1+1]);
                    bp1 += pb1;
       
                    Operator bp2 = mpo_dot(RPO[sitev2],LAO[sitev2+1]);
                    Operator pb2 = mpo_dot(LPO[sitev2],RAO[sitev2+1]);
                    bp2 += pb2;

                    Operator g3 = mpo_dot(t12,bp1);
                    Operator g4 = mpo_dot(g3,bp2);
                    g4 *= val*0.5;
                    if(i==0)
                    {
                        Sec = move(g4);
                    }
                    else
                    {
                        Sec += g4;
                    } 
                    if( i/10 != (i+1)/10 )   // truncate every 50 terms
                    { Sec.truncate(tol); }
                    cout << "2nd-ord term " << i+1 << endl;
                } 
                totals.push_back(Sec);
                cout << "2nd-order part finished." << endl;
                string fn2 = "Op/H_2";
                ofstream ofs2(fn2);
                Sec.save(ofs2);
            }
            // add together
            Operator res (PureEle);
            for(INT i(1);i<totals.size();++i) { res += totals[i]; }
            res.truncate(tol);           
            string fn = "Op/H_tot";
            ofstream ofs(fn);
            res.save(ofs);
        }

        // generate total Hamiltonian
        void totals ( const vector<Operator> & RPO, const vector<Operator> & LPO, const vector<Operator> & RAO,
        const vector<Operator> & LAO, DumpConfig & s, double tol )
        {
            // pure electronic Hamitoian
            INT ns = RPO.size();
            Operator ey(ns),res(ns);
            eye_op(ey,s);
            ofstream ofey("Op/Eye");
            ey.save(ofey);

            for(INT i(0);i<s.Terms.size();++i)
            {
                cout << "dump term " << i+1 << endl;
                INT lab1 = get<0>(s.Terms[i]);
                INT lab2 = get<1>(s.Terms[i]);
                INT lab3 = get<2>(s.Terms[i]);
                INT lab4 = get<3>(s.Terms[i]);
                double val = get<4>(s.Terms[i]);

                if(lab3==0 && lab4==0) // pure ele
                {
                    Operator t12 = mpo_dot(RPO[lab1-1],LPO[lab2-1]);
                    t12 *= val;
                    if(i==0) {res = move(t12);}
                    else { res += t12;  }
                }
                else if(lab1>s.NumEle && lab2>s.NumEle && lab3>s.NumEle && lab4>s.NumEle) // pure vib
                {
                    Operator o1 = mpo_dot(RPO[lab1-1],LAO[lab2-1]);
                    Operator o2 = mpo_dot(LPO[lab3-1],RAO[lab4-1]);
                    Operator o3 = mpo_dot(o1,o2);
                    o3 *= val;
                    res += o3;
                }
                else  // coupling
                {
                    Operator o1 = mpo_dot(RPO[lab1-1],LPO[lab2-1]);

                    Operator o2 = mpo_dot(RPO[lab3-1],LAO[lab4-1]);
                    Operator o2c = mpo_dot(LPO[lab3-1],RAO[lab4-1]);
                    
                    o2 += o2c;

                    Operator o3 = mpo_dot(o1,o2);

                    o3 *= val*sqrt(0.5);

                    res += o3;
                }

                if( i/10 != (i+1)/10 )   // truncate every 10 terms
                { res.truncate(tol); }
            }
            res.truncate(tol);           
            string fn = "Op/H_tot";
            ofstream ofs(fn);
            res.save(ofs);
        }
    }
}

#endif
