#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "TinyDFT.h"



int main(int argc, char **argv)
{
    double x = -0.002;
    double y =fabs(x);
    double lgx=log10(y);
    printf("%f,%f",y,lgx);
    double vals[10] = { 5.2, 5.6, 5.3, 5.4, 1.5, 8.5, -1.5, 9.5, 0.05, -15.0 };
    int key[10]={6,7,81,9,101,1,2,3,4,15};
    Qsort_double_key_val(key,vals, 0, 9);
    for(int i=0;i<10;i++)
        printf("index %d, value %e \n",key[i],vals[i]);
    
    int row[8]={0,0,1,4,3,2,3,2};
    int col[8]={2,4,0,2,3,1,4,3};
    double val[8]={1.2,1.4,3.1,-11.0,-5.7,2.5,1.13,-3.5};
    int rowcsr[6];
    int colcsr[8];
    double valcsr[8];
    Double_COO_to_CSR( 5, 8, row, col, 
   val, rowcsr, colcsr, valcsr);
   for(int i=0;i<6;i++)
        printf("%d  ",rowcsr[i]);
    for(int i=0;i<8;i++)
    {
        printf("col:%d, val:%e\n",colcsr[i],valcsr[i]);
    }
    for(int i=0;i<8;i++)
    {
        printf("Initial value:%e\n",val[i]);
    }
    return 0;
}
