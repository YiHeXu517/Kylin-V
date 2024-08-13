/* Generic product tools for tensors */

#ifndef product_H
#define product_H

#include "svd.h"

namespace KylinVib
{
    namespace Holstein
    {
        /* work pair */
        template<INT N1, INT N2, INT nc>
        struct WorkPair
        {
            /* label of factor tensor 1 */
            INT xLab;
            /* label of factor tensor 2 */
            INT yLab;
            /* uncontracted legs' shape for result  */
            array<INT,N1+N2-2*nc> UncontrShape;
            /* uncontracted legs' qns   for result  */
            array<INT,N1+N2-2*nc> UncontrQns;
            /* rows  */
            INT nrows;
            /* cols  */
            INT ncols;
            /* k-contraction */
            INT nK;
            /* position of work groups (same when shape and qn are the same) */
            INT pos;

            WorkPair() = default;
            WorkPair(const WorkPair<N1,N2,nc> & t)
            : xLab(t.xLab),yLab(t.yLab),UncontrShape(t.UncontrShape),
            UncontrQns(t.UncontrQns), nrows(t.nrows), ncols(t.ncols),
            nK(t.nK), pos(t.pos)
            { }
            WorkPair<N1,N2,nc> & operator=(const WorkPair<N1,N2,nc> & t)
            {
                xLab = t.xLab;
                yLab = t.yLab;
                UncontrShape = t.UncontrShape;
                UncontrQns   = t.UncontrQns;
                nrows        = t.nrows;
                ncols        = t.ncols;
                nK           = t.nK;
                pos          = t.pos;
                return *this;
            }
            WorkPair( INT XLab, INT YLab, const array<INT,N1+N2-2*nc> & USP, 
                      const array<INT,N1+N2-2*nc> & UQS, INT nR, INT nC, INT K, INT Pos  )
            : xLab(XLab),yLab(YLab),UncontrShape(USP),
            UncontrQns(UQS), nrows(nR), ncols(nC),
            nK(K), pos(Pos)
            { }
            ~WorkPair() = default;
        };

        /* Generic product for simplest case */        
        template<INT N1, INT N2, INT nc>
        Tensor<N1+N2-2*nc> product ( const Tensor<N1> & m1, const Tensor<N2> & m2,
        CBLAS_TRANSPOSE T1, CBLAS_TRANSPOSE T2 )
        { 
            /* initilaize the shape */
            array<map<INT,INT>,N1+N2-2*nc> rLegs;
            array<INT,N1+N2-2*nc> rDis;

            if(T1==CblasNoTrans)
            {
                copy( m1.clegs().begin(), m1.clegs().begin()+N1-nc, rLegs.begin() );
                copy( m1.dirs().begin(), m1.dirs().begin()+N1-nc, rDis.begin() );
            }
            else
            {
                copy( m1.clegs().begin()+nc, m1.clegs().end(), rLegs.begin() );
                copy( m1.dirs().begin()+nc, m1.dirs().end(), rDis.begin() );
                if(T1==CblasConjTrans) 
                {  
                    for(INT i(0);i<N1-nc;++i) { rDis[i] *= -1; }
                }
            }
            if(T2==CblasNoTrans)
            {
                copy( m2.clegs().begin()+nc, m2.clegs().end(), rLegs.begin()+N1-nc );
                copy( m2.dirs().begin()+nc, m2.dirs().end(), rDis.begin()+N1-nc );
            }
            else
            {
                copy( m2.clegs().begin(), m2.clegs().begin()+N2-nc, rLegs.begin()+N1-nc );
                copy( m2.dirs().begin(), m2.dirs().begin()+N2-nc, rDis.begin()+N1-nc );
                if(T2==CblasConjTrans) 
                {  
                    for(INT i(0);i<N2-nc;++i) { rDis[N1-nc+i] *= -1; }
                }
            }

            INT iax1 = (T1==CblasNoTrans)?N1-nc:0;
            INT iax2 = (T2==CblasNoTrans)?0:N2-nc;
            INT fac1 = (T1==CblasConjTrans)?-1:1;
            INT fac2 = (T2==CblasConjTrans)?-1:1;
            for(INT ax(0);ax<nc;++ax)
            {
                if(m1.dirs()[iax1+ax]*fac1 != -1*m2.dirs()[iax2+ax]*fac2) 
                {
                    cout << "Contracted legs' directions mismatch!" << endl;
                    cout << N1 << " : " << N2 << " : " << nc << endl;
                    exit(1); 
                }
            }
            /* check and extract match pairs */
            vector<WorkPair<N1,N2,nc>> wpr;
            INT RealSize(0);

            for(INT i(0);i<m1.num_blocks();++i)
            {
                for(INT j(0);j<m2.num_blocks();++j)
                {
                    /* all nc axis match */
                    bool MatchAllAxis = true;
                    for(INT ax(0);ax<nc;++ax)
                    {
                        if( m1.at(i).shape()[iax1+ax] != m2.at(j).shape()[iax2+ax] || 
                            m1.at(i).qn()[iax1+ax] != m2.at(j).qn()[iax2+ax] )
                        {
                            MatchAllAxis = false;
                            break;
                        } 
                    }
              
                    /* product */
                    if(MatchAllAxis)
                    {
                        array<INT,N1+N2-2*nc> Sp,Qn;
                        INT nrow,ncol,nK;
                        if(T1==CblasNoTrans) 
                        { 
                            copy( m1.at(i).shape().begin(), m1.at(i).shape().begin()+N1-nc, Sp.begin() );
                            nrow = accumulate( m1.at(i).shape().begin(), 
                                   m1.at(i).shape().begin()+N1-nc,(INT)1, multiplies<INT>() );
                            copy( m1.at(i).qn().begin(), m1.at(i).qn().begin()+N1-nc, Qn.begin() ); 
                        }
                        else
                        { 
                            copy( m1.at(i).shape().begin()+nc, m1.at(i).shape().end(), Sp.begin() );
                            nrow = accumulate( m1.at(i).shape().begin()+nc, 
                                   m1.at(i).shape().end(),(INT)1, multiplies<INT>() );
                            copy( m1.at(i).qn().begin()+nc, m1.at(i).qn().end(), Qn.begin() ); 
                        }
                        if(T2!=CblasNoTrans) 
                        { 
                            copy( m2.at(j).shape().begin(), m2.at(j).shape().begin()+N2-nc, Sp.begin()+N1-nc );
                            ncol = accumulate( m2.at(j).shape().begin(), 
                                  m2.at(j).shape().begin()+N2-nc,(INT)1, multiplies<INT>() );
                            copy( m2.at(j).qn().begin(), m2.at(j).qn().begin()+N2-nc, Qn.begin()+N1-nc ); 
                        }
                        else
                        { 
                            copy( m2.at(j).shape().begin()+nc, m2.at(j).shape().end(), Sp.begin()+N1-nc );
                            ncol = accumulate( m2.at(j).shape().begin()+nc, 
                                  m2.at(j).shape().end(),(INT)1, multiplies<INT>() );
                            copy( m2.at(j).qn().begin()+nc, m2.at(j).qn().end(), Qn.begin()+N1-nc ); 
                        }
                        nK = m1.at(i).size() / nrow;
                        

                        /* find same shape / qn */
                        bool fd = false;
                        for(INT spr(0);spr<wpr.size();++spr)
                        {
                            if(wpr[spr].UncontrShape == Sp && wpr[spr].UncontrQns == Qn)
                            {
                                fd = true;
                                wpr.emplace_back( i,j,Sp,Qn,nrow,ncol,nK,wpr[spr].pos);
                                break;
                            }
                        }
                        if(!fd) 
                        { 
                            wpr.emplace_back( i,j,Sp,Qn,nrow,ncol,nK,RealSize);  
                            RealSize += 1;
                        }
                    }
                }
            }

            /* core product */
	    Tensor<N1+N2-2*nc> res(rLegs,rDis,RealSize);
            for(INT spr = 0;spr<wpr.size();++spr)
            {
                //cout << RealSize << "  " << wpr[spr].pos << endl;
                Block<N1+N2-2*nc> rij(wpr[spr].UncontrShape, wpr[spr].UncontrQns);
                zdot( m1.at(wpr[spr].xLab), m2.at(wpr[spr].yLab), 
                      wpr[spr].nrows, wpr[spr].ncols, wpr[spr].nK,
                      rij, T1, T2 );
                if( res.at(wpr[spr].pos).cptr() )
                { res[wpr[spr].pos] += rij;  }
                else { res[wpr[spr].pos] = move(rij);  }
            }
            return res;
        }
        /* atomic env-mps-mpo product */
        void env_mps_mpo_atom ( const Tensor<4> & lm, const Tensor<4> & mpo, INT alp, INT zet,
        const vector<tuple<INT,INT>> & MatchPair, Tensor<4> & res  )
        {
            for(INT m(alp);m<zet;++m)
            {
                INT i = get<0>(MatchPair[m]);
                INT j = get<1>(MatchPair[m]);
                Block<4> rij (
                    { lm.at(i).shape()[0], mpo.at(j).shape()[0],  
                      mpo.at(j).shape()[3], lm.at(i).shape()[3] },
                    { lm.at(i).qn()[0], mpo.at(j).qn()[0],  
                      mpo.at(j).qn()[3], lm.at(i).qn()[3] }
                );
                Complex alp(1,0), bet(0,0);
                for(INT k(0);k<lm.at(i).shape()[0];++k)
                {
                  for(INT k2(0);k2<mpo.at(j).shape()[0];++k2)
                  {
                    cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    mpo.at(j).shape()[3], lm.at(i).shape()[3], lm.at(i).shape()[1]*lm.at(i).shape()[2],
                    &alp, mpo.at(j).cptr() + k2 * mpo.at(j).dist()[0], mpo.at(j).shape()[3],
                    lm.at(i).cptr() + k * lm.at(i).dist()[0], lm.at(i).shape()[3],
                    &bet, rij.ptr() + k * rij.dist()[0] + k2 * rij.dist()[1], lm.at(i).shape()[3] );
                  }
                }
                res[m] = move(rij);
            }
        }
        void env_mps_mpo_match ( const Tensor<4> & lm, const Tensor<4> & mpo, vector<tuple<INT,INT>> & MatchPair  )
        {
            for(INT i(0);i<lm.num_blocks();++i)
            {
                for(INT j(0);j<mpo.num_blocks();++j)
                {
                    if( 
                        lm.at(i).shape()[1] == mpo.at(j).shape()[1] &&
                        lm.at(i).shape()[2] == mpo.at(j).shape()[2] &&
                        lm.at(i).qn()[1] == mpo.at(j).qn()[1] &&
                        lm.at(i).qn()[2] == mpo.at(j).qn()[2]
                    )
                    {
                        MatchPair.push_back(make_tuple(i,j));
                    }
                }
            }
        }

        Tensor<4> env_mps_mpo ( const Tensor<3> & env, const Tensor<3> & mps, const Tensor<4> & mpo)
        {
            Tensor<4> lm = product<3,3,1>(env,mps,CblasNoTrans, CblasNoTrans);
            array<map<INT,INT>,4> rLegs = { lm.cleg(0), mpo.cleg(0), mpo.cleg(3), lm.cleg(3) };
            array<INT,4> rDi = { lm.dirs()[0], mpo.dirs()[0], mpo.dirs()[3], lm.dirs()[3] };
            vector<tuple<INT,INT>> MatPair;
            env_mps_mpo_match(lm,mpo,MatPair);
            Tensor<4> res(rLegs,rDi,MatPair.size());
            INT ncore = thread::hardware_concurrency();
            if(MatPair.size()<ncore || true)
            {   env_mps_mpo_atom(lm,mpo,0,MatPair.size(),MatPair,res);  } 
            else
            {
                vector<thread> nth(ncore-1);
                INT nBlock = MatPair.size() / (ncore-2); INT lastBlock = MatPair.size() % (ncore - 2);
                for(INT i(0);i<ncore-2;++i)
                {
                   nth[i] = thread(&env_mps_mpo_atom, ref(lm),ref(mpo),
                            i*nBlock,(i+1)*nBlock,ref(MatPair),ref(res));
                }
                nth[ncore-2] = thread(&env_mps_mpo_atom, ref(lm),ref(mpo),
                            (ncore-2)*nBlock,MatPair.size(),ref(MatPair),ref(res));
                for(INT i(0);i<ncore-1;++i) { nth[i].join();  }
            }
            return res;
        }

        /* commomly used product */
        Tensor<4> env_mps_mpo ( const Tensor<3> & env, const Tensor<3> & mps, const Tensor<4> & mpo, bool Chk )
        {
            Tensor<4> lm = product<3,3,1>(env,mps,CblasNoTrans, CblasNoTrans);
            array<map<INT,INT>,4> rLegs = { lm.cleg(0), mpo.cleg(0), mpo.cleg(3), lm.cleg(3) };
            array<INT,4> rDi = { lm.dirs()[0], mpo.dirs()[0], mpo.dirs()[3], lm.dirs()[3] };
            Tensor<4> res(rLegs,rDi);
            for(INT i(0);i<lm.num_blocks();++i)
            {
                for(INT j(0);j<mpo.num_blocks();++j)
                {
                    if( 
                        lm.at(i).shape()[1] == mpo.at(j).shape()[1] &&
                        lm.at(i).shape()[2] == mpo.at(j).shape()[2] &&
                        lm.at(i).qn()[1] == mpo.at(j).qn()[1] &&
                        lm.at(i).qn()[2] == mpo.at(j).qn()[2]
                    )
                    {
                        Block<4> rij (
                            { lm.at(i).shape()[0], mpo.at(j).shape()[0],  
                              mpo.at(j).shape()[3], lm.at(i).shape()[3] },
                             { lm.at(i).qn()[0], mpo.at(j).qn()[0],  
                              mpo.at(j).qn()[3], lm.at(i).qn()[3] }
                        );
                        Complex alp(1,0), bet(0,0);
                        for(INT k(0);k<lm.at(i).shape()[0];++k)
                        {
                          for(INT k2(0);k2<mpo.at(j).shape()[0];++k2)
                          {
                            cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            mpo.at(j).shape()[3], lm.at(i).shape()[3], lm.at(i).shape()[1]*lm.at(i).shape()[2],
                            &alp, mpo.at(j).cptr() + k2 * mpo.at(j).dist()[0], mpo.at(j).shape()[3],
                            lm.at(i).cptr() + k * lm.at(i).dist()[0], lm.at(i).shape()[3],
                            &bet, rij.ptr() + k * rij.dist()[0] + k2 * rij.dist()[1], lm.at(i).shape()[3] );
                          }
                        }
                        if(Chk) {res.push_back_ch(rij);}  // SVD need the check before
                        else { res.push_back(rij); }      // check will decrease parallel efficiency
                    }
                }
            }
            return res;
        } 

        void env_mps2_mpo1_atom(const Tensor<5> & lm, const Tensor<4> & mpo, INT alp, INT zet,
        const vector<tuple<INT,INT>> & MatchPair, Tensor<5> & res )
        {
            for(INT m(alp);m<zet;++m)
            {
                INT i = get<0>(MatchPair[m]);
                INT j = get<1>(MatchPair[m]);
                Block<5> rij (
                            { lm.at(i).shape()[0], mpo.at(j).shape()[0],  
                              mpo.at(j).shape()[3], lm.at(i).shape()[3], lm.at(i).shape()[4] },
                             { lm.at(i).qn()[0], mpo.at(j).qn()[0],  
                              mpo.at(j).qn()[3], lm.at(i).qn()[3], lm.at(i).qn()[4] }
                );
                Complex alp(1,0), bet(0,0);
                for(INT k(0);k<lm.at(i).shape()[0];++k)
                {
                  for(INT k2(0);k2<mpo.at(j).shape()[0];++k2)
                  {
                    cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    mpo.at(j).shape()[3], lm.at(i).shape()[4], lm.at(i).shape()[1]*lm.at(i).shape()[2],
                    &alp, mpo.at(j).cptr() + k2 * mpo.at(j).dist()[0], mpo.at(j).shape()[3],
                    lm.at(i).cptr() + k * lm.at(i).dist()[0], lm.at(i).shape()[4],
                    &bet, rij.ptr() + k * rij.dist()[0] + k2 * rij.dist()[1], lm.at(i).shape()[4] );
                  }
                }
                res[m] = move(rij);
            }
        }
        void env_mps2_mpo1_match(const Tensor<5> & lm, const Tensor<4> & mpo, vector<tuple<INT,INT>> & MatchPair)
        {
            for(INT i(0);i<lm.num_blocks();++i)
            {
                for(INT j(0);j<mpo.num_blocks();++j)
                {
                    if( 
                        lm.at(i).shape()[1] == mpo.at(j).shape()[1] &&
                        lm.at(i).shape()[2] == mpo.at(j).shape()[2] &&
                        lm.at(i).qn()[1] == mpo.at(j).qn()[1] &&
                        lm.at(i).qn()[2] == mpo.at(j).qn()[2]
                    )
                    {
                        MatchPair.push_back(make_tuple(i,j));
                    }
                }
            }
 
        }

        // (a-1,b-1,a-1') - (a-1', s', s+1' a+1') -> lm : (a-1,b-1,s',s+1',a+1')
        // (a-1,b-1,s',s+1',a+1') - ( s, b-1, s', b ) -> lmo1: ( a-1,s,b,s+1',a+1')
        // ( a-1,s,b,s+1',a+1') - ( s+1,b,s+1',b+1 ) -> lmo1o2 : (a-1,s,s+1,b+1,a+1')
        Tensor<5> env_mps2_mpo1 ( const Tensor<3> & env, const Tensor<4> & mps, const Tensor<4> & mpo)
        {
            Tensor<5> lm = product<3,4,1>(env,mps,CblasNoTrans, CblasNoTrans);
            array<map<INT,INT>,5> rLegs = { lm.cleg(0), mpo.cleg(0), mpo.cleg(3), lm.cleg(3), lm.cleg(4) };
            array<INT,5> rDi = { lm.dirs()[0], mpo.dirs()[0], mpo.dirs()[3], lm.dirs()[3], lm.dirs()[4] };
            vector<tuple<INT,INT>> MatPair;
            env_mps2_mpo1_match(lm,mpo,MatPair);
            Tensor<5> res(rLegs,rDi,MatPair.size());
            INT ncore = thread::hardware_concurrency();
            if(MatPair.size()<ncore || true)
            {   env_mps2_mpo1_atom(lm,mpo,0,MatPair.size(),MatPair,res);  } 
            else
            {
                vector<thread> nth(ncore-1);
                INT nBlock = MatPair.size() / (ncore-2); INT lastBlock = MatPair.size() % (ncore - 2);
                for(INT i(0);i<ncore-2;++i)
                {
                   nth[i] = thread(&env_mps2_mpo1_atom, ref(lm),ref(mpo),
                            i*nBlock,(i+1)*nBlock,ref(MatPair),ref(res));
                }
                nth[ncore-2] = thread(&env_mps2_mpo1_atom, ref(lm),ref(mpo),
                            (ncore-2)*nBlock,MatPair.size(),ref(MatPair),ref(res));
                for(INT i(0);i<ncore-1;++i) { nth[i].join();  }
            }
            return res;
        } 

        void env_mps2_mpo2_atom(const Tensor<5> & lm2o1, const Tensor<4> & mpo, INT alp, INT zet,
        const vector<tuple<INT,INT>> & MatchPair, Tensor<5> & res )
        {
            //struct sysinfo sif;
            //int erg = sysinfo(&sif); cout << "Mem bef o2m2 : " << sif.freeram * 1.0 / pow(1024,3.0) << " GB" << endl;            
            for(INT m(alp);m<zet;++m)
            {
                INT i = get<0>(MatchPair[m]);
                INT j = get<1>(MatchPair[m]);
                Block<5> rij (
                            { lm2o1.at(i).shape()[0],  lm2o1.at(i).shape()[1], mpo.at(j).shape()[0],  
                              mpo.at(j).shape()[3],lm2o1.at(i).shape()[4] },
                             { lm2o1.at(i).qn()[0],  lm2o1.at(i).qn()[1], mpo.at(j).qn()[0],  
                              mpo.at(j).qn()[3], lm2o1.at(i).qn()[4] }
                );
                Complex alp(1,0), bet(0,0);
                for(INT k(0);k<lm2o1.at(i).shape()[0]*lm2o1.at(i).shape()[1];++k)
                {
                  for(INT k2(0);k2<mpo.at(j).shape()[0];++k2)
                  {
                    cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    mpo.at(j).shape()[3], lm2o1.at(i).shape()[4], lm2o1.at(i).shape()[3]*lm2o1.at(i).shape()[2],
                    &alp, mpo.at(j).cptr()+ k2 * mpo.at(j).dist()[0], mpo.at(j).shape()[3],
                    lm2o1.at(i).cptr() + k * lm2o1.at(i).dist()[1], lm2o1.at(i).shape()[4],
                    &bet, rij.ptr() + k * rij.dist()[1] + k2 * rij.dist()[2], lm2o1.at(i).shape()[4] );
                  }
                }
                res[m] = move(rij);
            }
            //erg = sysinfo(&sif); cout << "Mem aft o2m2 : " << sif.freeram * 1.0 / pow(1024,3.0) << " GB" << endl;
        }
        void env_mps2_mpo2_match(const Tensor<5> & lm, const Tensor<4> & mpo, vector<tuple<INT,INT>> & MatchPair)
        {
            for(INT i(0);i<lm.num_blocks();++i)
            {
                for(INT j(0);j<mpo.num_blocks();++j)
                {
                    if( 
                        lm.at(i).shape()[2] == mpo.at(j).shape()[1] &&
                        lm.at(i).shape()[3] == mpo.at(j).shape()[2] &&
                        lm.at(i).qn()[2] == mpo.at(j).qn()[1] &&
                        lm.at(i).qn()[3] == mpo.at(j).qn()[2]
                    )
                    {
                        MatchPair.push_back(make_tuple(i,j));
                    }
                }
            }
 
        }


        Tensor<5> env_mps2_mpo2 (const Tensor<5> & lm2o1, const Tensor<4> & mpo)
        {
            array<map<INT,INT>,5> rLegs = { lm2o1.cleg(0), lm2o1.cleg(1), mpo.cleg(0), mpo.cleg(3), lm2o1.cleg(4) };
            array<INT,5> rDi = { lm2o1.dirs()[0],  lm2o1.dirs()[1], mpo.dirs()[0], mpo.dirs()[3], lm2o1.dirs()[4] };
            vector<tuple<INT,INT>> MatPair;
            env_mps2_mpo2_match(lm2o1,mpo,MatPair);
            Tensor<5> res(rLegs,rDi,MatPair.size());
            INT ncore = thread::hardware_concurrency();
            if(MatPair.size()<ncore || true)
            {   env_mps2_mpo2_atom(lm2o1,mpo,0,MatPair.size(),MatPair,res);  } 
            else
            {
                vector<thread> nth(ncore-1);
                INT nBlock = MatPair.size() / (ncore-2); INT lastBlock = MatPair.size() % (ncore - 2);
                for(INT i(0);i<ncore-2;++i)
                {
                   nth[i] = thread(&env_mps2_mpo2_atom, ref(lm2o1),ref(mpo),
                            i*nBlock,(i+1)*nBlock,ref(MatPair),ref(res));
                }
                nth[ncore-2] = thread(&env_mps2_mpo2_atom, ref(lm2o1),ref(mpo),
                            (ncore-2)*nBlock,MatPair.size(),ref(MatPair),ref(res));
                for(INT i(0);i<ncore-1;++i) { nth[i].join();  }
            }
            //erg = sysinfo(&sif); cout << "aft all o2m2 : " << sif.freeram * 1.0 / pow(1024,3.0) << " GB" << endl;
 
            return res;
        } 
        Tensor<4> apply_double ( const Tensor<3> & envL, const Tensor<4> & mps, const Tensor<4> & mpo1,
        const Tensor<4> & mpo2, const Tensor<3> & envR )
        {
            Tensor<5> lm2o1 = env_mps2_mpo1(envL,mps,mpo1);
            Tensor<5> lm2o2 = env_mps2_mpo2(lm2o1, mpo2);
            Tensor<4> res   = product<5,3,2>(lm2o2,envR,CblasNoTrans,CblasTrans);
            //erg = sysinfo(&sif); cout << "aft all Free:  " << sif.freeram*1.0 / 1024/1024/1024 << " GB" << endl;
            return res;
        }
        
        Tensor<3> env_sweep( const Tensor<3> & env, const Tensor<3> & mps, const Tensor<4> & mpo, 
        bool Left2Right  )
        {
            if(Left2Right)
            {
                Tensor<4> lmo = env_mps_mpo(env,mps,mpo);
                Tensor<3> res = product<3,4,2>(mps, lmo, CblasConjTrans, CblasNoTrans);
                return res;
            }
            else
            {
                Tensor<3> mpst = transpose<3>(mps,{2,1,0});
                Tensor<4> mpot = transpose<4>(mpo,{0,3,2,1});
                return env_sweep(env,mpst,mpot,true);
            }
        }

        Tensor<3> env_sweep( const Tensor<3> & env, const Tensor<3> & mps1, const Tensor<4> & mpo,
        const Tensor<3> & mps2, bool Left2Right  = true)
        {
            if(Left2Right)
            {
                Tensor<4> lmo = env_mps_mpo(env,mps2,mpo);
                Tensor<3> res = product<3,4,2>(mps1, lmo, CblasConjTrans, CblasNoTrans);
                return res;
            }
            else
            {
                Tensor<3> m1t = transpose<3>(mps1,{2,1,0});
                Tensor<4> mpot = transpose<4>(mpo,{0,3,2,1});
                Tensor<3> m2t = transpose<3>(mps2,{2,1,0});
                return env_sweep(env,m1t,mpot,m2t,true);
            }
        }

        Tensor<2> env_sweep( const Tensor<2> & env, const Tensor<3> & mps1,
        const Tensor<3> & mps2, bool Left2Right  = true)
        {
            if(Left2Right)
            {
                Tensor<3> lm = product<2,3,1>(env,mps2,CblasNoTrans,CblasNoTrans);
                Tensor<2> res = product<3,3,2>(mps1, lm, CblasConjTrans, CblasNoTrans);
                return res;
            }
            else
            {
                Tensor<3> m1t = transpose<3>(mps1,{2,1,0});
                Tensor<3> m2t = transpose<3>(mps2,{2,1,0});
                return env_sweep(env,m1t,m2t,true);
            }
        }

        Tensor<2> apply_zero ( const Tensor<3> & envL,const Tensor<2> & C,const Tensor<3> & envR )
        {
            Tensor<3> lc = product<3,2,1>(envL,C,CblasNoTrans,CblasNoTrans);
            Tensor<2> res = product<3,3,2>(lc,envR,CblasNoTrans,CblasTrans);
            return res;
        }

        Tensor<3> apply_single ( const Tensor<3> & envL, const Tensor<3> & mps, const Tensor<4> & mpo,
        const Tensor<3> & envR )
        {
            Tensor<4> lmo = env_mps_mpo(envL,mps,mpo);
            Tensor<3> res = product<4,3,2>(lmo,envR,CblasNoTrans,CblasTrans);
            return res;
        }

        tuple<Tensor<4>,Tensor<3>> apply_mpo ( const Tensor<3> & env,  const Tensor<4> & mpo1, const Tensor<4> & mpo2 ,
        SVDConfig & r )
        {
            Tensor<5> lo2 = product <3,4,1> ( env, transpose<4>(mpo2,{1,0,2,3}), CblasNoTrans, CblasNoTrans );
            Tensor<5> lo1o2 = product<4,5,2> ( transpose<4>(mpo1,{0,3,1,2}), transpose<5>(lo2,{1,2,0,3,4}), CblasNoTrans, CblasNoTrans );  
            Tensor<5> res = transpose<5>(lo1o2, {0,2,3,1,4});
            return svd_impl<3,2>(res,r,-1);
        }
    }
}

#endif
