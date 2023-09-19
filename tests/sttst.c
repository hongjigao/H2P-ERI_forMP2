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

    }
    printf("The number of values larger than 1e-2,1e-5 and 1e-9 are respectively %d,%d,%d\n",larger1e2,larger1e5,larger1e9);
    printf("The number of elements is %d\n",lg0tst);
}

void TestCSR(CSRmat_p csrmat)
{
    double maxv=0;
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

void Build_COO_from_densemat(const int nrow, const int ncol, double densemat[nrow][ncol], COOmat_p coomat)
{
    coomat->nnz=0;
    for(int i=0;i<nrow;i++)
        for(int j=0;j<ncol;j++)
        {
            if(densemat[i][j]!=0)
                coomat->nnz+=1;
        }
    coomat->coorow = (int*) malloc(sizeof(int) * (coomat->nnz));
    coomat->coocol = (int*) malloc(sizeof(int) * (coomat->nnz));
    coomat->cooval = (double*) malloc(sizeof(double) * (coomat->nnz));
    ASSERT_PRINTF(coomat->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->cooval    != NULL, "Failed to allocate arrays for D matrices indexing\n");
    int posit=0;
    for(int i=0;i<nrow;i++)
        for(int j=0;j<ncol;j++)
        {
            if(densemat[i][j]!=0)
            {
                coomat->coorow[posit]=i;
                coomat->coocol[posit]=j;
                coomat->cooval[posit]=densemat[i][j];
                posit+=1;
            }
        }
}


int main(int argc, char **argv)
{
    int nbf=4;
    printf("Initialize cooh2d\n");
    COOmat_p cooh2d;
    COOmat_init(&cooh2d,nbf*nbf,nbf*nbf);
    cooh2d->nnz=3*nbf*nbf-2;
    cooh2d->coorow = (int*) malloc(sizeof(int) * (cooh2d->nnz));
    cooh2d->coocol = (int*) malloc(sizeof(int) * (cooh2d->nnz));
    cooh2d->cooval = (double*) malloc(sizeof(double) * (cooh2d->nnz));
    
    for(int i=0;i<nbf*nbf;i++)
    {
        cooh2d->coorow[i]=i;
        cooh2d->coocol[i]=i;
        cooh2d->cooval[i]=1;
    }
    for(int i=0;i<nbf*nbf-1;i++)
    {
        cooh2d->coorow[2*i+nbf*nbf]=i;
        cooh2d->coocol[2*i+nbf*nbf]=i+1;
        cooh2d->cooval[2*i+nbf*nbf]=0.1;
        cooh2d->coorow[2*i+nbf*nbf+1]=i+1;
        cooh2d->coocol[2*i+nbf*nbf+1]=i;
        cooh2d->cooval[2*i+nbf*nbf+1]=0.1;
    }
    printf("test cooh2d\n");
    TestCOO(cooh2d);
    CSRmat_p csrh2d;
    CSRmat_init(&csrh2d,nbf*nbf,nbf*nbf);
    printf("Initialize csrh2d\n");
    Double_COO_to_CSR(nbf*nbf,cooh2d->nnz,cooh2d,csrh2d);
    printf("test csrh2d\n");
    TestCSR(csrh2d);
    printf("rows\n");
    for(int i=0;i<csrh2d->nrow+1;i++)
    {
        printf("%ld,  ",csrh2d->csrrow[i]);
    }
    printf("Now cols\n");
    for(int i=0;i<csrh2d->nnz;i++)
    {
        printf("%d,  ",csrh2d->csrcol[i]);
        printf("%f,  ",csrh2d->csrval[i]);
    }
    COOmat_p cooden;
    COOmat_init(&cooden,nbf,nbf);
    double denmat[4][4]={{1,0.01,0,0},{0.01,1,0.1,0},{0,0.1,1,0},{0,0,0,0.5}};
    Build_COO_from_densemat(4,4,denmat,cooden);
    printf("nnz is %ld\n",cooden->nnz);
    TestCOO(cooden);
    CSRmat_p csrden;
    CSRmat_init(&csrden,nbf,nbf);
    Double_COO_to_CSR(nbf,cooden->nnz,cooden,csrden);

    COOmat_p coodc;
    COOmat_init(&coodc,nbf,nbf);
    double dcmat[4][4]={{1,0.03,0.01,0},{0.03,1.01,0,-0.3},{0.01,0,0.8,0},{0,-0.3,0,1.2}};
    Build_COO_from_densemat(4,4,dcmat,coodc);
    printf("nnz is %ld\n",coodc->nnz);
    TestCOO(coodc);
    CSRmat_p csrdc;
    CSRmat_init(&csrdc,nbf,nbf);
    Double_COO_to_CSR(nbf,coodc->nnz,coodc,csrdc);
    TestCSR(csrdc);

    printf("Now do index transformation\n");
    CSRmat_p gdle;
    CSRmat_init(&gdle,nbf*nbf,nbf*nbf);
    printf("test numbf is%d\n",nbf);
    double st1,et1;
    st1 = get_wtime_sec();
    Xindextransform(nbf,csrh2d,csrden,gdle);
    et1 = get_wtime_sec();
    printf("The X Index transformation time is %.3lf (s)\n",et1-st1);
    TestCSR(gdle);
    printf("Xindex transformation finished\n");
    CSRmat_p gdls;
    CSRmat_init(&gdls,nbf*nbf,nbf*nbf);
    
    st1 = get_wtime_sec();
    Yindextransform1(nbf,gdle,csrdc,gdls);
    et1 = get_wtime_sec();
    
    TestCSR(gdls);
    printf("The Y Index transformation time is %.3lf (s)\n",et1-st1);
    CSRmat_p colgdls;
    CSRmat_init(&colgdls,nbf*nbf,nbf*nbf);
    CSR_to_CSC(nbf*nbf, gdls,colgdls);
    TestCSR(colgdls);
    double energy;
    energy = Calc_S1energy(gdls,colgdls);
    printf("The energy is %f\n",energy);

    COOmat_destroy(cooden);
    CSRmat_destroy(csrden);
    COOmat_destroy(coodc);
    CSRmat_destroy(csrdc);
    COOmat_destroy(cooh2d);
    CSRmat_destroy(csrh2d);
    CSRmat_destroy(gdls);
    CSRmat_destroy(gdle);
    CSRmat_destroy(colgdls);
    return 0;
}
