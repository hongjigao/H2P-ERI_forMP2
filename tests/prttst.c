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

int main(int argc, char **argv)
{
    int n=5;
    int nnz=10;
    COOmat_p coomat;
    COOmat_init(&coomat,n,n);
    double * testmat;
    testmat = (double*) malloc(sizeof(double) * (n*n));
    memset(testmat, 0, sizeof(double) * (n*n));
    testmat[0]=1;
    testmat[1]=5;
    testmat[5]=3;
    testmat[6]=2;
    testmat[12]=3;
    testmat[13]=6;
    testmat[16]=2;
    testmat[17]=111;
    testmat[18]=4;
    testmat[24]=5;
    printf("0\n");
    printf("%f\n",testmat[24]);
    size_t nde =Extract_COO_DDCMat(5, 5, 0, testmat, coomat);
    printf("1\n");
    TestCOO(coomat);
    CSRmat_p csrmat;
    CSRmat_init(&csrmat,5,5);
    Double_COO_to_CSR( 5,  nde, coomat,csrmat);
    printf("Build csr success\n");
    TestCSR(csrmat);
    printf("2\n");
    CSRmat_p cscmat;
    CSRmat_init(&cscmat,5,5);
    CSR_to_CSC(5, csrmat,cscmat);
    TestCSR(cscmat);
    printf("3\n");
    double energy;
    energy = Calc_S1energy(csrmat,cscmat);
    printf("The energy is %f\n",energy);
    for(int i=0;i<nnz;i++)
    {
        printf("%d,%f\n",cscmat->csrcol[i],cscmat->csrval[i]);
    }
    for(int j=0;j<csrmat->nrow+1;j++)
    {
        printf("Row index %lu\n",csrmat->csrrow[j]);
    }
    return 0;
}
