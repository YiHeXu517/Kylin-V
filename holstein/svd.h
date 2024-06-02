/* SVD for used kind of tensor */

#ifndef hols_SVD_H
#define hols_SVD_H

#include <float.h>
#include "tensor.h"

namespace klnX
{
    namespace Holstein
    {
        using std::plus;
        using std::abs;
        using std::sort;
        using std::tuple;
        using std::get;
        using std::make_tuple;
        using std::for_each;
        using std::find_if;
        using std::pow;
        using std::log10;

        /* SVD configuration */
        struct SVDConfig
        {
            /* singular value threshold */
            double SVDThres = 1.0e-9;
            /* max allowed states */
            INT MaxDim   = 2000;
            /* stochastic adaptive paramter */
            double SAModeBeta = 0.1;
            /* Us or sVt */
            bool Left2Right = true;
            /* renormalisation */
            bool EnableRenorm = false;
            /* rescaled */
            bool EnableRescale = false;
        };
 
        ostream & operator<<(ostream & os, SVDConfig & r)
        {
            os << "SVD threshold              = " << r.SVDThres << endl;
            os << "SVD maximal allowed        = " << r.MaxDim << endl;
            return os;
        };
 
        /* SVD sub-block with rank-R1 left tensor and rank-R2 right tensor */
        template<INT R>
        struct SVDSubBlock
        {
            /* saved shape */
            array<INT, R> SubShape;
            /* saved quantum numbers */
            array<INT, R> SubQuaNum;
            /* start positions for each individual shape inside a macro block */
            INT SubPosition;
            /* tensor blocks index matching parameters above */
            vector<INT> SubIndices;
        };

        template<INT R>
        ostream & operator<<(ostream & os, SVDSubBlock<R> & r)
        {
            os << "Shape: [";
            for_each(r.SubShape.begin(), r.SubShape.end(),[&os](INT x){os << x << ","; } );
            os << "\b]" << endl;
            os << "Sym: [";
            for_each(r.SubQuaNum.begin(), r.SubQuaNum.end(),[&os](INT x){os << x << ","; } );
            os << "\b]" << endl;
            os << "Including indices: [";
            for_each(r.SubIndices.begin(), r.SubIndices.end(),[&os](INT x){os << x << ","; } );
            os << "\b]" << endl;
            os << "Position: " << r.SubPosition << endl;
            return os;
        }

        /* SVD block with rank-2 matrix language. Ingnore the micro states */

        template<INT R>
        struct SVDBlock
        {
            /* total length N */
            INT TotLen;
            /* reduced total quantum number */
            INT TotQuaNum;
            /* sub-blocks */
            vector<SVDSubBlock<R>> Subs;
            /* allowed kept states */
            INT NumKept = 0;
        };

        template<INT R>
        ostream & operator<<(ostream & os, SVDBlock<R> & r)
        {
            os << "Length  : " << r.TotLen << endl;
            os << "Sym     : " << r.TotQuaNum << endl;
            os << "Kept    : " << r.NumKept << endl;
            os << "----------------------------------------" << endl;
            for(INT i(0);i<r.Subs.size();++i)
            {
                os << "Sub SVD block " << i+1 << ":" << endl;
                os << r.Subs[i];
            }
            os << "----------------------------------------" << endl;
            return os;
        }

        /* fill the SVD blocks by tensor A */
        template<INT R1, INT R2>
        void fill_block ( Tensor<R1+R2> & A, vector<SVDBlock<R1>> & RowInfo,
        vector<SVDBlock<R2>> & ColInfo )
        {
            for(INT k(0);k<A.num_blocks();++k)
            {
                // split
                INT qnr = inner_product(A.at(k).qn().begin(),A.at(k).qn().begin()+R1,A.dirs().begin(),0);
                INT qnc = inner_product(A.at(k).qn().begin()+R1,A.at(k).qn().end(),A.dirs().begin()+R1,0);
                INT spr = accumulate(A.at(k).shape().begin(),A.at(k).shape().begin()+R1,1,multiplies<INT>());
                INT spc = accumulate(A.at(k).shape().begin()+R1,A.at(k).shape().end(),1,multiplies<INT>());
       
                array<INT,R1> SPRS, QNRS;
                array<INT,R2> SPCS, QNCS;

                copy(A.at(k).qn().begin(),A.at(k).qn().begin()+R1,QNRS.begin());
                copy(A.at(k).shape().begin(),A.at(k).shape().begin()+R1,SPRS.begin());
                copy(A.at(k).qn().begin()+R1,A.at(k).qn().end(),QNCS.begin());
                copy(A.at(k).shape().begin()+R1,A.at(k).shape().end(),SPCS.begin());

                // fill left
                bool fdMacro = false;
                for(INT i(0);i<RowInfo.size();++i)
                {
                    if(RowInfo[i].TotQuaNum == qnr)  // found macro
                    {
                        fdMacro = true;
                        bool fdMicro = false;
                        for(INT j(0);j<RowInfo[i].Subs.size();++j)
                        {
                            if(  RowInfo[i].Subs[j].SubShape == SPRS && RowInfo[i].Subs[j].SubQuaNum == QNRS ) // found micro
                            {
                                fdMicro = true;
                                RowInfo[i].Subs[j].SubIndices.push_back(k);
                                break;
                            }
                        }
                        if(!fdMicro)   // not found micro
                        {
                            vector<INT> INDX = {k};
                            SVDSubBlock<R1> Mic = { SPRS, QNRS, RowInfo[i].TotLen, INDX };
                            RowInfo[i].TotLen += spr;
                            RowInfo[i].Subs.push_back(Mic);
                        }
                        break;
                    }
                }
                if(!fdMacro)
                {
                    SVDBlock<R1> Mac;
                    Mac.TotLen = spr;
                    Mac.TotQuaNum = qnr;
                    vector<INT> INDX = {k};
                    SVDSubBlock<R1> Mic = { SPRS, QNRS, 0, INDX };
                    Mac.Subs.emplace_back( Mic );
                    RowInfo.push_back(Mac);
                }
                // fill right
                fdMacro = false;
                for(INT i(0);i<ColInfo.size();++i)
                {
                    if(ColInfo[i].TotQuaNum == qnc)  // found macro
                    {
                        fdMacro = true;
                        bool fdMicro = false;
                        for(INT j(0);j<ColInfo[i].Subs.size();++j)
                        {
                            if(  ColInfo[i].Subs[j].SubShape == SPCS && ColInfo[i].Subs[j].SubQuaNum == QNCS ) // found micro
                            {
                                fdMicro = true;
                                ColInfo[i].Subs[j].SubIndices.push_back(k);
                                break;
                            }
                        }
                        if(!fdMicro)   // not found micro
                        {
                            vector<INT> INDX = {k};
                            SVDSubBlock<R2> Mic = { SPCS, QNCS, ColInfo[i].TotLen, INDX };
                            ColInfo[i].TotLen += spc;
                            ColInfo[i].Subs.push_back(Mic);
                        }
                        break;
                    }
                }
                if(!fdMacro)
                {
                    SVDBlock<R2> Mac;
                    Mac.TotLen = spc;
                    Mac.TotQuaNum = qnc;
                    vector<INT> INDX = {k};
                    SVDSubBlock<R2> Mic = { SPCS, QNCS, 0, INDX };
                    Mac.Subs.emplace_back( Mic );
                    ColInfo.push_back(Mac);
                }  
            }
        }
       
        void find_same( vector<INT> & v1, vector<INT> & v2, vector<INT> & vres )
        {
            for(auto & r1 : v1)
            {
                for(auto & r2 : v2)
                {
                    if(r1==r2) { vres.push_back(r1); }
                }
            }                 
        }

        /* copy blocks in A to dense matrix Ar */
        template<INT R1, INT R2>
        void copy_to_mat ( Tensor<R1+R2> & A, SVDBlock<R1> & RB,
        SVDBlock<R2> & CB, Alloc<Complex> & Ar )
        {
            INT Tor = RB.TotLen;
            INT Toc = CB.TotLen;
            for(auto & ra : RB.Subs)
            {
                for(auto & ca : CB.Subs)
                {
                    INT posr = ra.SubPosition;
                    INT posc = ca.SubPosition;
                    INT nrow = accumulate(ra.SubShape.begin(),ra.SubShape.end(),1,multiplies<INT>());
                    INT ncol = accumulate(ca.SubShape.begin(),ca.SubShape.end(),1,multiplies<INT>());
                    
                    vector<INT> SamIdx;
                    find_same( ra.SubIndices, ca.SubIndices, SamIdx );
                     
                    for(auto & idx : SamIdx)
                    {
                        for(INT i(0);i<ncol;++i)
                        {
                            cblas_zcopy( nrow, A.at(idx).cptr() + i, ncol, 
                            Ar.ptr() + posr*Toc + posc + i, Toc  );
                        }
                    }
                }
            } 
        }

        // redistribution
        template<INT R1>
        void redist_row ( Alloc<Complex> & U,  Tensor<R1+1> & res,
        SVDBlock<R1> & Indr, INT LeftDir)
        {
            INT nstate = Indr.NumKept;
            for(INT i(0);i<Indr.Subs.size();++i) // index in row - SVDBlockInfo
            {
                INT posr = Indr.Subs[i].SubPosition; 
                INT SubRow = accumulate( Indr.Subs[i].SubShape.begin(), Indr.Subs[i].SubShape.end(), 1, multiplies<INT>() );
                array<INT,R1+1> Sp,Qs;
                copy(Indr.Subs[i].SubShape.begin(),Indr.Subs[i].SubShape.end(),Sp.begin()); Sp[R1] = nstate;
                copy(Indr.Subs[i].SubQuaNum.begin(),Indr.Subs[i].SubQuaNum.end(),Qs.begin()); 
                Qs[R1] = -1*LeftDir*Indr.TotQuaNum;
                Block<R1+1> ri(Sp,Qs);
                INT colU = U.size() / Indr.TotLen;
                for(INT j(0);j<nstate;++j)
                {
                    cblas_zcopy(SubRow, U.cptr() + posr * colU + j, colU, ri.ptr() + j, nstate );
                }
                res.push_back_ch( ri );
            }
        }

        template<INT R2>
        void redist_col ( Alloc<Complex> & Vt,  Tensor<R2+1> & res,
        SVDBlock<R2> & Indc, INT RightDir)
        {
            INT nstate = Indc.NumKept;
            for(INT i(0);i<Indc.Subs.size();++i) // index in col - SVDBlockInfo
            {
                INT posc = Indc.Subs[i].SubPosition;    //
                INT SubCol = accumulate( Indc.Subs[i].SubShape.begin(), Indc.Subs[i].SubShape.end(), 1, multiplies<INT>() );
                array<INT,R2+1> Sp,Qs;
                copy(Indc.Subs[i].SubShape.begin(),Indc.Subs[i].SubShape.end(),Sp.begin()+1); Sp[0] = nstate;
                copy(Indc.Subs[i].SubQuaNum.begin(),Indc.Subs[i].SubQuaNum.end(),Qs.begin()+1);
                Qs[0] = -1*RightDir*Indc.TotQuaNum;
                Block<R2+1> ri(Sp,Qs);
                for(INT j(0);j<SubCol;++j)
                {
                    cblas_zcopy(nstate, Vt.cptr()+posc+j, Indc.TotLen, ri.ptr() + j, SubCol );
                }
                res.push_back_ch( ri );
            }
        }

        /* Generic implemention bondS: save the singular values */
        template<INT R1, INT R2>
        tuple<Tensor<R1+1>,Tensor<R2+1>> svd_impl ( Tensor<R1+R2> & A, SVDConfig & r , INT LeftDir,
        bool PrintTrunc = false , bool SaveFile = false
         )
        {
            // build blocks
            vector<SVDBlock<R1>> RowInfo;
            vector<SVDBlock<R2>> ColInfo;
            fill_block<R1,R2>(A,RowInfo,ColInfo); 

            vector<tuple<INT,double>> collectS;
            vector<Alloc<Complex>> collectU, collectVt; 

            // copy
            for(auto & rw : RowInfo)
            {
                for(auto & cl : ColInfo)
                {
                    if(rw.TotQuaNum + cl.TotQuaNum == 0)
                    {
                        INT nrow = rw.TotLen;
                        INT ncol = cl.TotLen;
                        INT ldu = min(nrow,ncol);
                        rw.NumKept = ldu;
                        cl.NumKept = ldu;
                        Alloc<Complex> Ar(nrow*ncol), U(nrow*ldu), Vt(ldu*ncol);
                        Alloc<double> s(ldu);
                        copy_to_mat<R1,R2>(A,rw,cl,Ar);
                        naive_svd(Ar,nrow,ncol,U,s,Vt);

                        collectU.push_back(move(U));
                        collectVt.push_back(move(Vt));

                        INT nstate = 1;
                        if(r.Left2Right)
                        {
                            for(INT i(0);i<ldu;++i)
                            {
                                cblas_zdscal(ncol, s.ptr()[i], collectVt.back().ptr() + i * ncol,1);
                                collectS.push_back(make_tuple( rw.TotQuaNum, s.ptr()[i])); 
                            }
                        }
                        else
                        {
                            for(INT i(0);i<ldu;++i)
                            {
                                cblas_zdscal(nrow, s.ptr()[i], collectU.back().ptr() + i ,ldu);
                                collectS.push_back(make_tuple( cl.TotQuaNum, s.ptr()[i]));
                            }
                        }
                    }
                }
            }
 
            // order singular values 
            sort(collectS.begin(),collectS.end(),
            [&collectS](tuple<INT,double> x1, tuple<INT,double> x2){
            return get<1>(x1)>get<1>(x2);});

            INT TotalNumS = collectS.size();

            // truncate and save singular values
            ofstream ofs( "SVS", ofstream::app );
            if(!ofs.is_open()) { cout << "SVS file not exist!" << endl;  } 

            for(INT i(TotalNumS-1);i>0;--i)
            {
                INT qni = get<0>(collectS[i]);
                double svi = get<1>(collectS[i]);
                if(i>=r.MaxDim || svi<=r.SVDThres)
                {
                    if(r.Left2Right)
                    {
                        auto itr = find_if(RowInfo.begin(), RowInfo.end(),
                        [&qni](SVDBlock<R1> &x){ return x.TotQuaNum==qni; } );
                        auto itc = find_if(ColInfo.begin(), ColInfo.end(),
                        [&qni](SVDBlock<R2> &x){ return x.TotQuaNum==-1*qni; } );
                        itr->NumKept -= 1;
                        itc->NumKept -= 1;
                    }
                    else
                    {
                        auto itr = find_if(RowInfo.begin(), RowInfo.end(),
                        [&qni](SVDBlock<R1> &x){ return x.TotQuaNum==-1*qni; } );
                        auto itc = find_if(ColInfo.begin(), ColInfo.end(),
                        [&qni](SVDBlock<R2> &x){ return x.TotQuaNum==qni; } );
                        itr->NumKept -= 1;
                        itc->NumKept -= 1;
                    }
                    collectS.pop_back();
                }
                else { break; }  // end the truncation
            }

            if(SaveFile)
            {
                ofs << collectS.size() << endl;
                for(INT i(0);i<collectS.size();++i)
                {
                    if(i>=2)
                    {
                        ofs << std::left << setw(20) << std::scientific << setprecision(11) << log10(get<1>(collectS[i])*get<1>(collectS[i-2])/pow(get<1>(collectS[i-1]),2.0)) << endl; 
                    }
                }
            }
 
            ofs.close();
            if (PrintTrunc) { cout << TotalNumS << " -> " << collectS.size() << endl; }

            // redist
            array<INT,R1+1> Dirr; copy( A.dirs().begin(), A.dirs().begin()+R1, Dirr.begin() ); 
            array<INT,R2+1> Dicc; copy( A.dirs().begin()+R1, A.dirs().end(), Dicc.begin()+1 ); 
            Dirr[R1] = LeftDir; Dicc[0] = -1*LeftDir;
            Tensor<R1+1> Lef(Dirr);
            Tensor<R2+1> Rig(Dicc);

            INT QnOrd = 0;
            for(auto & row : RowInfo)
            {
                for(auto & col : ColInfo)
                {
                    if(row.TotQuaNum + col.TotQuaNum == 0)
                    {
                        if( row.NumKept<=0 || col.NumKept<=0 ) { ++QnOrd; continue; }
                        redist_row ( collectU[QnOrd], Lef, row, Dirr[R1]);
                        redist_col ( collectVt[QnOrd], Rig, col, Dicc[0]);
                        ++QnOrd ;
                    }
                    else  {continue;}
                }
            }
            Lef.check_legs();
            Rig.check_legs();
            if(r.EnableRenorm)
            {
                if(r.Left2Right) { Rig *= 1.0/norm(Rig); }
                else { Lef *= 1.0/norm(Lef); }
            }
            if(r.EnableRescale)
            { 
                if(r.Left2Right) { double nsid = norm(Rig); Rig *= pow(nsid,-0.5); Lef *= pow(nsid,0.5); }
                else {  double nsid = norm(Lef); Lef *= pow(nsid,-0.5); Rig *= pow(nsid,0.5); }
            }
            return make_tuple(Lef,Rig);
        }

        /* sa implemention  */
        template<INT R1, INT R2>
        tuple<Tensor<R1+1>,Tensor<R2+1>> sa_svd_impl ( Tensor<R1+R2> & A, SVDConfig & r , INT LeftDir, double beta )
        {
            // build blocks
            vector<SVDBlock<R1>> RowInfo;
            vector<SVDBlock<R2>> ColInfo;
            fill_block<R1,R2>(A,RowInfo,ColInfo); 

            vector<tuple<INT,double>> collectS;
            vector<Alloc<Complex>> collectU, collectVt; 

            // copy
            for(auto & rw : RowInfo)
            {
                for(auto & cl : ColInfo)
                {
                    if(rw.TotQuaNum + cl.TotQuaNum == 0)
                    {
                        INT nrow = rw.TotLen;
                        INT ncol = cl.TotLen;
                        INT ldu = min(nrow,ncol);
                        rw.NumKept = ldu;
                        cl.NumKept = ldu;
                        Alloc<Complex> Ar(nrow*ncol), U(nrow*ldu), Vt(ldu*ncol);
                        Alloc<double> s(ldu);
                        copy_to_mat<R1,R2>(A,rw,cl,Ar);
                        naive_svd(Ar,nrow,ncol,U,s,Vt);

                        collectU.push_back(move(U));
                        collectVt.push_back(move(Vt));

                        INT nstate = 1;
                        if(r.Left2Right)
                        {
                            for(INT i(0);i<ldu;++i)
                            {
                                cblas_zdscal(ncol, s.ptr()[i], collectVt.back().ptr() + i * ncol,1);
                                collectS.push_back(make_tuple( rw.TotQuaNum, s.ptr()[i])); 
                            }
                        }
                        else
                        {
                            for(INT i(0);i<ldu;++i)
                            {
                                cblas_zdscal(nrow, s.ptr()[i], collectU.back().ptr() + i ,ldu);
                                collectS.push_back(make_tuple( cl.TotQuaNum, s.ptr()[i]));
                            }
                        }
                    }
                }
            }
            
            // order singular values 
            sort(collectS.begin(),collectS.end(),
            [&collectS](tuple<INT,double> x1, tuple<INT,double> x2){
            return get<1>(x1)>get<1>(x2);});

            INT TotalNumS = collectS.size();

            // extension
            INT TotalAddS = 0;
            vector<double> tmps = {get<1>(collectS[collectS.size()-2]), get<1>(collectS[collectS.size()-1])};
            if(TotalNumS < r.MaxDim && TotalNumS > 5)
            {
                for(INT i(TotalNumS);i<r.MaxDim;++i)
                {
                    random_device rd; 
                    mt19937 gen{rd()};
                    std::uniform_real_distribution<double> d(0,1);
                    double inv_rds = d(gen);
                    double rds = inv_quasi_gauss(inv_rds, beta);
                    if((i+TotalNumS)%2==0)
                    {
                        tmps.push_back( pow(10,rds) * pow(tmps[tmps.size()-1],2.0) / tmps[tmps.size()-2] );
                    } 
                    else 
                    {
                        tmps.push_back( pow(10,-1.0*rds) * pow(tmps[tmps.size()-1],2.0) / tmps[tmps.size()-2] );
                    }
                    if(tmps[tmps.size()-1] <= r.SVDThres )
                    {
                        tmps.pop_back();
                        break;
                    }
                    if(tmps[tmps.size()-1] > tmps[tmps.size()-2])
                    {
                        tmps[tmps.size()-1] = tmps[tmps.size()-2];
                    }
                    TotalAddS = tmps.size() - 2;
                }
            }

            // redist
            array<INT,R1+1> Dirr; copy( A.dirs().begin(), A.dirs().begin()+R1, Dirr.begin() ); 
            array<INT,R2+1> Dicc; copy( A.dirs().begin()+R1, A.dirs().end(), Dicc.begin()+1 ); 
            Dirr[R1] = LeftDir; Dicc[0] = -1*LeftDir;
            Tensor<R1+1> Lef(Dirr);
            Tensor<R2+1> Rig(Dicc);

            INT QnOrd = 0;
            for(auto & row : RowInfo)
            {
                for(auto & col : ColInfo)
                {
                    if(row.TotQuaNum + col.TotQuaNum == 0)
                    {
                        if( row.NumKept<=0 || col.NumKept<=0 ) { ++QnOrd; continue; }
  
                        if(TotalAddS!=0)    // have not completed yet
                        {
                            if(r.Left2Right)
                            {
                                if(TotalAddS <= row.TotLen - row.NumKept)
                                {
                                    gs_ortho( row.TotLen, row.NumKept, collectU[QnOrd], false, TotalAddS);
                                    gs_ortho( col.NumKept, col.TotLen, collectVt[QnOrd], true, TotalAddS,true);
                                    row.NumKept += TotalAddS;
                                    col.NumKept += TotalAddS;
                                    TotalAddS = 0;
                                }
                                else if (TotalAddS > row.TotLen - row.NumKept && row.TotLen - row.NumKept > 0)
                                {
                                    INT addS = row.TotLen - row.NumKept;
                                    gs_ortho( row.TotLen, row.NumKept, collectU[QnOrd], false, addS);
                                    gs_ortho( col.NumKept, col.TotLen, collectVt[QnOrd], true, addS, true);
                                    row.NumKept = row.TotLen;
                                    col.NumKept += addS;
                                    TotalAddS -= addS;
                                }
                            }
                            else
                            {
                                if(TotalAddS <= col.TotLen - col.NumKept)
                                {
                                    gs_ortho( row.TotLen, row.NumKept, collectU[QnOrd], false, TotalAddS, true);
                                    gs_ortho( col.NumKept, col.TotLen, collectVt[QnOrd], true, TotalAddS);
                                    row.NumKept += TotalAddS;
                                    col.NumKept += TotalAddS;
                                    TotalAddS = 0;
                                }
                                else if (TotalAddS > col.TotLen - col.NumKept && col.TotLen - col.NumKept > 0)
                                {
                                    INT addS = col.TotLen - col.NumKept;
                                    gs_ortho( row.TotLen, row.NumKept, collectU[QnOrd], false, addS, true);
                                    gs_ortho( col.NumKept, col.TotLen, collectVt[QnOrd], true, addS);
                                    col.NumKept = col.TotLen;
                                    row.NumKept += addS;
                                    TotalAddS -= addS;
                                }
                            }
                        }
                        redist_row ( collectU[QnOrd], Lef, row, Dirr[R1]);
                        redist_col ( collectVt[QnOrd], Rig, col, Dicc[0]);
                        ++QnOrd ;
                    }
                    else  {continue;}
                }
            }
            Lef.check_legs();
            Rig.check_legs();
            if(r.EnableRenorm)
            {
                if(r.Left2Right) { Rig *= 1.0/norm(Rig); }
                else { Lef *= 1.0/norm(Lef); }
            }
            if(r.EnableRescale)
            { 
                if(r.Left2Right) { double nsid = norm(Rig); Rig *= pow(nsid,-0.5); Lef *= pow(nsid,0.5); }
                else {  double nsid = norm(Lef); Lef *= pow(nsid,-0.5); Rig *= pow(nsid,0.5); }
            }
            return make_tuple(Lef,Rig); 
        }
 
    }
}

#endif
