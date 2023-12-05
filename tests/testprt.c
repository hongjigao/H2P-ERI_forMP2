#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "TinyDFT.h"
#include "H2ERI.h"

void TestCOO(COOmat_p coomat)
{
    double maxv=0;
    int larger1e5=0;
    int larger1e9=0;
    int larger1e2=0;
    int lg0tst=0;
//    printf("%d\n",lg0tst);
    for(size_t i=0;i<coomat->nnz;i++)
    {
        if(fabs(coomat->cooval[i])>maxv)
        {
            maxv=fabs(coomat->cooval[i]);
        }
    }

    printf("The max value is %e\n",maxv);

    double norm=0;
    for(size_t i=0;i<coomat->nnz;i++)
    {
        if(fabs(coomat->cooval[i])>maxv*1e-2)
            larger1e2+=1;
        if(fabs(coomat->cooval[i])>=maxv*1e-5)
            larger1e5+=1;
        if(fabs(coomat->cooval[i])>maxv*1e-9)
            larger1e9+=1;
        if(fabs(coomat->cooval[i])>0)
            lg0tst+=1;
        norm+=coomat->cooval[i]*coomat->cooval[i];

    }
    printf("The number of values larger than 1e-2,1e-5 and 1e-9 are respectively %d,%d,%d\n",larger1e2,larger1e5,larger1e9);
    printf("The number of elements is %d\n",lg0tst);
    printf("The norm of the COO matrix is %f\n",norm);
}

void TestCSR(CSRmat_p csrmat)
{
    double maxv=0;
    double norm=0;
    size_t larger1e5=0;
    size_t larger1e9=0;
    size_t larger1e2=0;
    size_t lg0tst=0;
//    printf("%d\n",lg0tst);
    for(size_t i=0;i<csrmat->nnz;i++)
    {
        if(fabs(csrmat->csrval[i])>maxv)
        {
            maxv=fabs(csrmat->csrval[i]);
        }
    }

    printf("The max value is %e\n",maxv);
 
    for(size_t i=0;i<csrmat->nnz;i++)
    {
        if(fabs(csrmat->csrval[i])>maxv*1e-2)
            larger1e2+=1;
        if(fabs(csrmat->csrval[i])>=maxv*1e-5)
            larger1e5+=1;
        if(fabs(csrmat->csrval[i])>maxv*1e-9)
            larger1e9+=1;
        if(fabs(csrmat->csrval[i])>0)
            lg0tst+=1;
        norm += csrmat->csrval[i]*csrmat->csrval[i];

    }
    int nn0 = csrmat->nrow;
    int nlong = 0;
    for(int j=0;j<csrmat->nrow;j++)
    {
        if(csrmat->csrrow[j]==csrmat->csrrow[j+1])
            nn0 -= 1;
        else
        {
            if(csrmat->csrrow[j+1]-csrmat->csrrow[j]>nlong)
                nlong=csrmat->csrrow[j+1]-csrmat->csrrow[j];
        }
    }
    printf("The number of values larger than 1e-2,1e-5 and 1e-9 are respectively %lu,%lu,%lu\n",larger1e2,larger1e5,larger1e9);
    printf("The number of elements is %lu\n",lg0tst);
    printf("The norm of the csrmat is %f\n", norm);
    printf("The number of nonzero rows is %d, the totol rows is %d, the longest row is %d\n",nn0,csrmat->nrow,nlong);
    printf("Test the ascending order:\n");
    int tests=0;
    for(int i=0;i<csrmat->nrow;i++)
    {
        if(csrmat->csrrow[i]<csrmat->csrrow[i+1]-1)
        {
            for(size_t j=csrmat->csrrow[i];j<csrmat->csrrow[i+1]-1;j++)
            {
                if(csrmat->csrcol[j]>csrmat->csrcol[j+1])
                {
                    printf("Ascending order wrong!\n");
                    tests=1;
                    return;
                }
                if(csrmat->csrcol[j]==csrmat->csrcol[j+1])
                {
//                    printf("equal wrong\n");
                    tests+=1;
                }
            }
        }
    }
    if(tests==0)
        printf("Ascending order correct!\n");
    else
        printf("Same value %d\n",tests);
}


double Calc2norm(const double *mat, int siz) 
{
  double norms = 0;
  for (int i = 0; i < siz; i++)
    for (int j = 0; j < siz; j++) 
    {
      norms = norms + mat[i * siz + j] * mat[i * siz + j];
    }

  return norms;
}



int main(int argc, char **argv)
{
    int nbf=4;
    COOmat_p cooh2d;
    COOmat_init(&cooh2d,nbf,nbf);
    double * testmat;
    testmat = (double*) malloc(sizeof(double) * (256));
    memset(testmat, 0, sizeof(double) * (256));
    for(int i=0;i<16;i++)
        for(int j=0;j<16;j++)
        {
            if(i==j)
                testmat[i*16+j]=1;
            if(i-j==1)
                testmat[i*16+j]=0.01;
            if(i-j==-1)
                testmat[i*16+j]=0.01;
        }
    //den=np.array([[1,0.01,0,0],[0.01,1,0.1,0],[0,0.1,1,0],[0,0,0,0.5]])
    for(int i=0;i<32;i++)
    {
        printf("%f ",testmat[i]);
    }
    printf("0\n");
    printf("%f\n",testmat[16]);
    size_t nde =Extract_COO_DDCMat(16, 16, 0, testmat, cooh2d);
    printf("1\n");
    TestCOO(cooh2d);
    CSRmat_p csrh2d;
    CSRmat_init(&csrh2d,16,16);
    Double_COO_to_CSR( 16,  nde, cooh2d,csrh2d);
    printf("Build csr success\n");
    TestCSR(csrh2d);
    printf("2\n");
    COOmat_p cooden;
    COOmat_init(&cooden,4,4);
    double * den;
    den = (double*) malloc(sizeof(double) * (16));
    memset(den, 0, sizeof(double) * (16));
    den[0]=1;
    den[1]=0.01;
    den[4]=0.01;
    den[5]=1;
    den[6]=0.1;
    den[9]=0.1;
    den[10]=1;
    den[15]=0.5;
    double nm=Calc2norm(den,nbf);
    printf("The norm of Dmat is %f\n",nm);
    double thres = 0;
    size_t nden =Extract_COO_DDCMat(4, 4, thres, den, cooden);
     TestCOO(cooden);
    printf("The total elements of D are %d and the rate of survival by threshold %e is %ld \n",4*4,thres,nden);
    printf("Now print CSR Den Matrix info--------\n");
    CSRmat_p csrden;
    CSRmat_init(&csrden,4,4);
    Double_COO_to_CSR( 4,  nden, cooden,csrden);
    TestCSR(csrden);
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    //dc=np.array([[1,0.03,0.01,0],[0.03,1.01,0,-0.3],[0.01,0,0.8,0],[0,-0.3,0,1.2]])
    double * dc;
    dc = (double*) malloc(sizeof(double) * (16));
    memset(dc, 0, sizeof(double) * (16));
    dc[0]=1;
    dc[1]=0.03;
    dc[2]=0.01;
    dc[4]=0.03;
    dc[5]=1.01;
    dc[7]=-0.3;
    dc[8]=0.01;
    dc[10]=0.8;
    dc[13]=-0.3;
    dc[15]=1.2;
    nm=Calc2norm(dc,nbf);
    printf("The norm of DCmat is %f\n",nm);
    COOmat_p coodc;
    COOmat_init(&coodc,4,4);
    size_t ndc =Extract_COO_DDCMat(4, 4, thres, dc, coodc);
    printf("Now print COO DC Matrix info--------\n");
    TestCOO(coodc);
    printf("The total elements of DC are %d and the rate of survival by threshold %e is %ld \n",4*4,thres,ndc);

    CSRmat_p csrdc;
    CSRmat_init(&csrdc,4,4);
    Double_COO_to_CSR( 4,  ndc, coodc,csrdc);
    printf("Now print CSR DC Matrix info--------\n");
    TestCSR(csrdc);
    printf("Build energy weighted matrices success\n");
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("Now do index transformation\n");
    CSRmat_p gdle;
    CSRmat_init(&gdle,4*4,4*4);
    printf("test numbf is%d\n",4);
    double st1,et1;
    st1 = get_wtime_sec();
    Xindextransform1(4,csrh2d,csrden,gdle);
    et1 = get_wtime_sec();
    printf("The X Index transformation time is %.3lf (s)\n",et1-st1);
    TestCSR(gdle);
    printf("Xindex transformation finished\n");
    CSRmat_p gdls;
    CSRmat_init(&gdls,4*4,4*4);
    
    st1 = get_wtime_sec();
    Yindextransform1(4,gdle,csrdc,gdls);
    et1 = get_wtime_sec();
    
    TestCSR(gdls);
    printf("The Y Index transformation time is %.3lf (s)\n",et1-st1);
    
    printf("Now do energy calculation \n");
    st1 = get_wtime_sec();

//    double energy;
//    energy = Calc_S1energy(gdls);
//    printf("The energy is %f\n",energy);
    CSRmat_p colgdls;
    CSRmat_init(&colgdls,nbf*nbf,nbf*nbf);
    CSR_to_CSC(nbf*nbf, gdls,colgdls);
    TestCSR(colgdls);
    double energy;
    energy = Calc_S1energy(gdls,colgdls);
    printf("The energy is %f\n",energy);
    et1 = get_wtime_sec();
    printf("Energy computation time is %.3lf (s)\n",et1-st1);
    printf("%f\n",gdls->csrval[gdls->nnz-1]);


    return 0;
}
