#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "TinyDFT.h"
#include "H2ERI.h"
/*
struct Sparsemat
{
    int nrow;                           // number of rows of the matrix
    int ncol;                           // number of columns of the matrix
    int nnz;                            // number of nonzero elements of the matrix
    int             * coorow;           // Array, size nnz, the row information in COOMatrix form.
    int             * coocol;           // Array, size nnz, the column information in COOMatrix form.
    double          * cooval;           // Array, size nnz, the value information in COOMatrix form.
    int             * csrrow;           // Array, size nrow+1, the row information in CSRMatrix form.
    int             * csrcol;           // Array, size nnz, the column information in CSRMatrix form.
    double          * csrval;           // Array, size nnz, the value information in CSRMatrix form.
};
typedef struct Sparsemat* Sparsemat_p;

void Sparsemat_init(Sparsemat_p *sparsemat_, const int nrow, const int ncol)
{
    Sparsemat_p sparsemat = (Sparsemat_p) malloc(sizeof(struct Sparsemat));
    assert(sparsemat != NULL);
    memset(sparsemat, 0, sizeof(struct Sparsemat));
    sparsemat->nrow=nrow;
    sparsemat->ncol=ncol;
    sparsemat->coocol  =  NULL;
    sparsemat->coorow  =  NULL;
    sparsemat->cooval  =  NULL;
    sparsemat->csrcol  =  NULL;
    sparsemat->csrrow  =  NULL;
    sparsemat->csrval  =  NULL;
    *sparsemat_ = sparsemat;
}
*/
int main(int argc, char **argv)
{
    double x = -0.002;
    double y =fabs(x);
    double lgx=log10(y);
    printf("%f,%f\n",y,lgx);
    double vals[10] = { 5.2, 5.6, 5.3, 5.4, 1.5, 8.5, -1.5, 9.5, 0.05, -15.0 };
    int key[10]={6,7,7,9,7,7,2,7,4,15};
    int key0[10]={1,1,1,1,1,3,4,5,5,4};
    Qsort_double_long0(key,vals, 0, 9);
    int j=0;
    for(j=0;j<=10;j++)
        printf("%d ",j);
    printf("%d\n",j);
    for(int i=0;i<10;i++)
        printf("index %d, value %e yes\n",key[i],vals[i]);
    
    printf("Yes\n");
    
    Qsort_double_long(key0,vals, 0, 9);
    for(int i=0;i<10;i++)
        printf("index %d, value %e \n",key0[i],vals[i]);
    
    int row[8]={0,0,0,1,3,2,3,2};
    int col[8]={2,3,0,3,1,1,0,2};
    double val[8]={1.2,1.4,3.1,-11.0,-5.7,2.5,1.13,-3.5};
   

    COOmat_p coomat;
    COOmat_init(&coomat,5,5);
    CSRmat_p csrmat;
    CSRmat_init(&csrmat,5,5);
    coomat->coorow = (int *) malloc(sizeof(int) * 8);
    coomat->coocol = (int *) malloc(sizeof(int) * 8);
    coomat->cooval = (double *) malloc(sizeof(double) * 8);
    csrmat->csrrow = (size_t *) malloc(sizeof(int) * 5);
    csrmat->csrcol = (int *) malloc(sizeof(int) * 8);
    csrmat->csrval = (double *) malloc(sizeof(double) * 8);
     for(int i=0;i<8;i++)
    {
        coomat->coorow[i]=row[i];
        coomat->coocol[i]=col[i];
        coomat->cooval[i]=val[i];
    }
    Double_COO_to_CSR( 5, 8, coomat,csrmat);
   for(int i=0;i<6;i++)
        printf("%lu ",csrmat->csrrow[i]);
    for(int i=0;i<8;i++)
    {
        printf("col:%d, val:%e\n",csrmat->csrcol[i],csrmat->csrval[i]);
    }
    for(int i=0;i<8;i++)
    {
        printf("Initial value:%e\n",val[i]);
    }
//    double energy;
//    energy=Calc_S1energy(csrmat);
//    printf("The energy is %f\n",energy);
    return 0;
}
