#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "H2ERI.h"

int main(int argc, char **argv)
{
    simint_init();

    if (argc < 5)
    {
        printf("Usage: %s <mol file> <D mat bin file> <ref J mat bin file> <relerr>\n", argv[0]);
        return 255;
    }
    
    H2ERI_p h2eri;
    H2ERI_init(&h2eri, 1e-10, 1e-10, atof(argv[4]));
    
    // 1. Read molecular file
    CMS_read_mol_file(argv[1], &h2eri->natom, &h2eri->nshell, &h2eri->shells);
    
    // 2. Process input shells for H2 partitioning
    H2ERI_process_shells(h2eri);
    
    // 3. H2 partition of screened shell pair centers
    H2ERI_partition(h2eri);
    
    // 4. Build H2 representation for ERI tensor
    H2ERI_build_H2(h2eri, 0);

    // 5. Read the reference density and Coulomb matrix from binary file
    size_t nbf2 = h2eri->num_bf * h2eri->num_bf;
    double *D_mat = (double *) malloc(sizeof(double) * nbf2);
    double *J_mat = (double *) malloc(sizeof(double) * nbf2);
    double *J_ref = (double *) malloc(sizeof(double) * nbf2);
    FILE *D_file = fopen(argv[2], "r");
    FILE *J_file = fopen(argv[3], "r");
    fread(D_mat, nbf2, sizeof(double), D_file);
    fread(J_ref, nbf2, sizeof(double), J_file);
    fclose(D_file);
    fclose(J_file);
    
    // 6. Construct the Coulomb matrix and save it to file
    H2ERI_build_Coulomb(h2eri, D_mat, J_mat);  // Warm up
    h2eri->h2pack->n_matvec = 0;
    memset(h2eri->h2pack->timers + 4, 0, sizeof(double) * 5);
    for (int k = 0; k < 10; k++)
        H2ERI_build_Coulomb(h2eri, D_mat, J_mat);
    
    H2ERI_print_statistic(h2eri);
    printf("The BD_JIT mode is %d\n",h2eri->h2pack->BD_JIT);
    printf("n_D0_blk is %d\n",h2eri->h2pack->D_blk0->length - 1);
    printf("The largest value is %d\n",h2eri->h2pack->D_blk0->data[h2eri->h2pack->D_blk0->length]);

    // 7. Calculate the relative error
    double ref_l2 = 0.0, err_l2 = 0.0;
    for (int i = 0; i < nbf2; i++)
    {
        double diff = J_ref[i] - J_mat[i];
        ref_l2 += J_ref[i] * J_ref[i];
        err_l2 += diff * diff;
    }
    ref_l2 = sqrt(ref_l2);
    err_l2 = sqrt(err_l2);
    printf("||J_{H2} - J_{ref}||_2 / ||J_{ref}||_2 = %e\n", err_l2 / ref_l2);\
    printf("D0totalsize=%lu,D1totalsize=%lu\n",h2eri->nD0element,h2eri->nD1element);
    H2ERI_build_COO_Diamat(h2eri,1);
    int nnz=h2eri->nD0element+2*h2eri->nD1element;
    double maxv=0;
    long double detectsum=0;
    int larger1e5=0;
    int larger1e9=0;
    int larger1e2=0;
    int lg0tst=0;
    for(size_t i=0;i<nnz;i++)
    {
        if(fabs(h2eri->DataD[i])>maxv)
        {
            maxv=fabs(h2eri->DataD[i]);
            printf("%e\n",maxv);
        }
        detectsum=detectsum+h2eri->DataD[i];
    }
    printf("The max value is %e\n",maxv);
    printf("The sum is %Le \n",detectsum);
    for(size_t i=0;i<nnz;i++)
    {
        if(fabs(h2eri->DataD[i])>maxv*1e-2)
            larger1e2+=1;
        if(fabs(h2eri->DataD[i])>=maxv*1e-5)
            larger1e5+=1;
        if(fabs(h2eri->DataD[i])>maxv*1e-9)
            larger1e9+=1;
        if(fabs(h2eri->DataD[i])>=0)
            lg0tst+=1;

    }
    printf("The number of values larger than 1e-2,0 and 1e-9 are respectively %d,%d,%d\n",larger1e2,larger1e5,larger1e9);
    printf("The number of elements is %d\n",lg0tst);
    h2eri->rowDCSR = (int*) malloc(sizeof(int) * (h2eri->num_bf*h2eri->num_bf+1));
    h2eri->colDCSR = (int*) malloc(sizeof(int) * h2eri->nD0element+2*h2eri->nD1element);
    printf("Malloc success");
    h2eri->DataDCSR = (double*) malloc(sizeof(double) * h2eri->nD0element+2*h2eri->nD1element);

    maxv=0;
    detectsum=0;
    for(size_t i=0;i<nnz;i++)
    {
        if(fabs(h2eri->DataD[i])>maxv)
        {
            maxv=fabs(h2eri->DataD[i]);
            printf("%e\n",maxv);
        }
        detectsum=detectsum+h2eri->DataD[i];
    }
    printf("The max value is %e\n",maxv);
    printf("The sum is %Le \n",detectsum);

 //   Double_COO_to_CSR( h2eri->num_bf*h2eri->num_bf,  h2eri->nD0element+2*h2eri->nD1element, h2eri->rowD, h2eri->colD, 
  //  h2eri->DataD, h2eri->rowDCSR, h2eri->colDCSR, h2eri->DataDCSR);
  // Get the number of non-zeros in each row
  
  int nrow=h2eri->num_bf*h2eri->num_bf;
    memset(h2eri->rowDCSR, 0, sizeof(int) * (nrow + 1));
    for (int i = 0; i < nnz; i++) h2eri->rowDCSR[h2eri->rowD[i] + 1]++;
    // Calculate the displacement of 1st non-zero in each row
    for (int i = 2; i <= nrow; i++) h2eri->rowDCSR[i] += h2eri->rowDCSR[i - 1];
    // Use row_ptr to bucket sort col[] and val[]
    for (int i = 0; i < nnz; i++)
    {
        int idx = h2eri->rowDCSR[h2eri->rowD[i]];
        h2eri->colDCSR[idx] = h2eri->colD[i];
        h2eri->DataDCSR[idx] = h2eri->DataD[i];
        h2eri->rowDCSR[h2eri->rowD[i]]++;
    }
    // Reset row_ptr
    for (int i = nrow; i >= 1; i--) h2eri->rowDCSR[i] = h2eri->rowDCSR[i - 1];
    h2eri->rowDCSR[0] = 0;
    printf("Outer test ------------------\n");
    maxv=0;
    detectsum=0;
    larger1e5=0;
    larger1e9=0;
    larger1e2=0;
    lg0tst=0;
    for(size_t i=0;i<h2eri->nD0element+2*h2eri->nD1element;i++)
    {
        if(fabs(h2eri->DataD[i])>maxv)
        {
            maxv=fabs(h2eri->DataD[i]);
            printf("%e\n",maxv);
        }
        detectsum=detectsum+h2eri->DataD[i];
    }
    printf("The max value is %e\n",maxv);
    printf("The sum is %Le \n",detectsum);
    for(size_t i=0;i<h2eri->nD0element+2*h2eri->nD1element;i++)
    {
        if(fabs(h2eri->DataDCSR[i])>maxv*1e-2)
            larger1e2+=1;
        if(fabs(h2eri->DataDCSR[i])>=maxv*1e-5)
            larger1e5+=1;
        if(fabs(h2eri->DataDCSR[i])>maxv*1e-9)
            larger1e9+=1;
        if(fabs(h2eri->DataDCSR[i])>=0)
            lg0tst+=1;

    }
    printf("The number of values larger than 1e-2,1e-5 and 1e-9 are respectively %d,%d,%d\n",larger1e2,larger1e5,larger1e9);
    printf("The number of elements is %d\n",lg0tst);
    long double dtd=0;
    long double dtcsr=0;
    for(size_t i=0;i<h2eri->nD0element+2*h2eri->nD1element;i++)
    {
        dtd+=maxv;
        dtcsr+=h2eri->DataDCSR[i];
    }
    printf("Norm of DataD is %Le while norm of DataDCSR is %Le\n",dtd,dtcsr);
    printf("%d",h2eri->rowDCSR[h2eri->num_bf*h2eri->num_bf]);
    free(J_ref);
    free(J_mat);
    free(D_mat);
    
    simint_finalize();
    return 0;
}