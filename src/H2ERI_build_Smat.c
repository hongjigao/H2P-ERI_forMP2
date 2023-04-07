#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "CMS.h"
#include "H2ERI_typedef.h"
#include "H2Pack_build.h"
#include "H2Pack_utils.h"
#include "H2Pack_aux_structs.h"
#include "H2Pack_ID_compress.h"
#include "H2ERI_utils.h"
#include "linalg_lib_wrapper.h"

void H2ERI_build_COO_Diamat(H2ERI_p h2eri, int D1tst)
{
    H2Pack_p h2pack = h2eri->h2pack;
    int n_leaf_node      = h2pack->n_leaf_node;
    int n_r_inadm_pair   = h2pack->n_r_inadm_pair;
    int num_bf           = h2eri->num_bf;
    int *sp_bfp_sidx     = h2eri->sp_bfp_sidx;
    int *D_nrow          = h2pack->D_nrow;
    int *D_ncol          = h2pack->D_ncol;
    int num_sp = h2eri->num_sp;
    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int *sp_shell_idx  = h2eri->sp_shell_idx;
    H2P_dense_mat_p  *c_D_blks   = h2eri->c_D_blks;
    int *pt_cluster      = h2pack->pt_cluster;
    int    *leaf_nodes    = h2pack->height_nodes;
    int    *r_inadm_pairs = h2pack->r_inadm_pairs;
    h2eri->rowD = (int*) malloc(sizeof(int) * (h2eri->nD0element+2*h2eri->nD1element));
    h2eri->colD = (int*) malloc(sizeof(int) * (h2eri->nD0element+2*h2eri->nD1element));
    h2eri->DataD = (double*) malloc(sizeof(double) * (h2eri->nD0element+2*h2eri->nD1element));
    ASSERT_PRINTF(h2eri->rowD != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(h2eri->colD != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(h2eri->DataD    != NULL, "Failed to allocate arrays for D matrices indexing\n");
    int *rowD            = h2eri->rowD;
    int *colD            = h2eri->colD;
    double *DataD           = h2eri->DataD;
    int *rowspot;
    int *colspot;
    double *Dataspot;
    int rowspotupdate=0;
    int colspotupdate=0;
    int loopinrow;
    int loopval;
    size_t numdata=0;
    //Store the information in D0 into COO matrix
    for(int i=0;i<n_leaf_node;i++)
    {
        int node = leaf_nodes[i];
        int pt_s = pt_cluster[2 * node];
        int pt_e = pt_cluster[2 * node + 1];
        //number of shell pairs in the node
 //       int node_npts = pt_e - pt_s + 1;
//        int Di_nrow = D_nrow[i];
        int Di_ncol = D_ncol[i];
        H2P_dense_mat_p Di = c_D_blks[i];
        rowspot=rowD+numdata;
        colspot=colD+numdata;
        Dataspot=DataD+numdata;
        loopval=0;
        loopinrow=0;
        //Store the COO matrix element data sp by sp
        for(int rowsp=pt_s;rowsp<pt_e+1;rowsp++)
        {
            int rowshell0 = sp_shell_idx[rowsp];
            int rowshell1 = sp_shell_idx[rowsp+num_sp];

            for(int colsp=pt_s;colsp<pt_e+1;colsp++)
            {

                int colshell0 = sp_shell_idx[colsp];
                int colshell1 = sp_shell_idx[colsp+num_sp];
                //pt_Dblock is the pointer of C_D_bls[i]'s data of the corresponding first element in the row
                double *pt_Dblock=Di->data+rowspotupdate+colspotupdate;
                for(int mu=shell_bf_sidx[rowshell0];mu<shell_bf_sidx[rowshell0+1];mu++)
                    for(int nu=shell_bf_sidx[rowshell1];nu<shell_bf_sidx[rowshell1+1];nu++)
                    {
                        for(int lambda=shell_bf_sidx[colshell0];lambda<shell_bf_sidx[colshell0+1];lambda++)
                            for(int sigma=shell_bf_sidx[colshell1];sigma<shell_bf_sidx[colshell1+1];sigma++)
                            {
                                rowspot[loopval]=nu*num_bf+mu;
                                colspot[loopval]=sigma*num_bf+lambda;
                                Dataspot[loopval]=pt_Dblock[loopinrow];
                                loopinrow+=1;
                                loopval+=1;
                            }
                        pt_Dblock+=Di_ncol;
                        loopinrow=0;
                    }
                colspotupdate+=sp_bfp_sidx[colsp+1]=sp_bfp_sidx[colsp];
                        
            }
            rowspotupdate+=(sp_bfp_sidx[rowsp+1]=sp_bfp_sidx[rowsp])*Di_ncol;
            colspotupdate=0;
        }
        numdata+=loopval;
    }
    if(D1tst==0) return;
    //store the information in D1 in COO matrix
    for (int i = 0; i < n_r_inadm_pair; i++)
    {
        int node0 = r_inadm_pairs[2 * i];
        int node1 = r_inadm_pairs[2 * i + 1];
        int pt_s0 = pt_cluster[2 * node0];
        int pt_s1 = pt_cluster[2 * node1];
        int pt_e0 = pt_cluster[2 * node0 + 1];
        int pt_e1 = pt_cluster[2 * node1 + 1];
        int Di_ncol = D_ncol[i+n_leaf_node];
        H2P_dense_mat_p Di = c_D_blks[i+n_leaf_node];
        rowspot=rowD+numdata;
        colspot=colD+numdata;
        Dataspot=DataD+numdata;
        loopval=0;
        loopinrow=0;
        //Store the COO matrix element data sp by sp
        for(int rowsp=pt_s0;rowsp<pt_e0+1;rowsp++)
        {
            int rowshell0 = sp_shell_idx[rowsp];
            int rowshell1 = sp_shell_idx[rowsp+num_sp];

            for(int colsp=pt_s1;colsp<pt_e1+1;colsp++)
            {

                int colshell0 = sp_shell_idx[colsp];
                int colshell1 = sp_shell_idx[colsp+num_sp];
                //pt_Dblock is the pointer of C_D_bls[i]'s data of the corresponding first element in the row
                double *pt_Dblock=Di->data+rowspotupdate+colspotupdate;
                for(int mu=shell_bf_sidx[rowshell0];mu<shell_bf_sidx[rowshell0+1];mu++)
                    for(int nu=shell_bf_sidx[rowshell1];nu<shell_bf_sidx[rowshell1+1];nu++)
                    {
                        for(int lambda=shell_bf_sidx[colshell0];lambda<shell_bf_sidx[colshell0+1];lambda++)
                            for(int sigma=shell_bf_sidx[colshell1];sigma<shell_bf_sidx[colshell1+1];sigma++)
                            {
                                rowspot[loopval]=nu*num_bf+mu;
                                colspot[loopval]=sigma*num_bf+lambda;
                                Dataspot[loopval]=pt_Dblock[loopinrow];
                                rowspot[loopval+1]=sigma*num_bf+lambda;
                                colspot[loopval+1]=nu*num_bf+mu;
                                Dataspot[loopval+1]=pt_Dblock[loopinrow];
                                loopinrow+=1;
                                loopval+=2;
                            }
                        pt_Dblock+=Di_ncol;
                        loopinrow=0;
//                        printf("%d\n",loopval);
                    }
                colspotupdate+=sp_bfp_sidx[colsp+1]=sp_bfp_sidx[colsp];
                        
            }
            rowspotupdate+=(sp_bfp_sidx[rowsp+1]=sp_bfp_sidx[rowsp])*Di_ncol;
            colspotupdate=0;
        }
        numdata+=loopval;
    }
}


void Qsort_double_key_val(int *key, double *val, int l, int r)
{
    int i = l, j = r, tmp_key;
    double tmp_val;
    int mid_key = key[(l + r) / 2];
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            tmp_key = key[i]; key[i] = key[j]; key[j] = tmp_key;
            tmp_val = val[i]; val[i] = val[j]; val[j] = tmp_val;
            i++;  j--;
        }
    }
    if (i < r) Qsort_double_key_val(key, val, i, r);
    if (j > l) Qsort_double_key_val(key, val, l, j);
}



void Double_COO_to_CSR(
    const int nrow, const int nnz, const int *row, const int *col, 
    const double *val, int *row_ptr, int *col_idx, double *val_
)
{
    // Get the number of non-zeros in each row
    memset(row_ptr, 0, sizeof(int) * (nrow + 1));
    for (int i = 0; i < nnz; i++) row_ptr[row[i] + 1]++;
    // Calculate the displacement of 1st non-zero in each row
    for (int i = 2; i <= nrow; i++) row_ptr[i] += row_ptr[i - 1];
    // Use row_ptr to bucket sort col[] and val[]
    for (int i = 0; i < nnz; i++)
    {
        int idx = row_ptr[row[i]];
        col_idx[idx] = col[i];
        val_[idx] = val[i];
        row_ptr[row[i]]++;
    }
    // Reset row_ptr
    for (int i = nrow; i >= 1; i--) row_ptr[i] = row_ptr[i - 1];
    row_ptr[0] = 0;
    double maxv=0;
    long double detectsum=0;
    printf("Inner test for COOdata___-----\n");
    for(size_t i=0;i<nnz;i++)
    {
        if(fabs(val[i])>maxv)
        {
            maxv=fabs(val[i]);
            printf("%e\n",maxv);
        }
        detectsum=detectsum+val[i];
    }
    printf("The max value is %e\n",maxv);
    printf("The sum is %Le \n",detectsum);
    printf("Inner test for CSR data\n");
    maxv=0;
    detectsum=0;
    for(size_t i=0;i<nnz;i++)
    {
        if(fabs(val_[i])>maxv)
        {
            maxv=fabs(val_[i]);
            printf("%e\n",maxv);
        }
        detectsum=detectsum+val_[i];
    }
    printf("The max value is %e\n",maxv);
    printf("The sum is %Le \n",detectsum);
    // Sort the non-zeros in each row according to column indices
 //   #pragma omp parallel for
 //   for (int i = 0; i < nrow; i++)
  //      Qsort_double_key_val(col_idx, val_, row_ptr[i], row_ptr[i + 1] - 1);

    printf("Test after qsort:COO Data\n");
    maxv=0;
    detectsum=0;
    for(size_t i=0;i<nnz;i++)
    {
        if(fabs(val[i])>maxv)
        {
            maxv=fabs(val[i]);
            printf("%e\n",maxv);
        }
        detectsum=detectsum+val[i];
    }
    printf("The max value is %e\n",maxv);
    printf("The sum is %Le \n",detectsum);
    printf("Test after qsort:CSRData\n");
    maxv=0;
    detectsum=0;
    for(size_t i=0;i<nnz;i++)
    {
        if(fabs(val_[i])>maxv)
        {
            maxv=fabs(val_[i]);
            printf("%e\n",maxv);
        }
        detectsum=detectsum+val_[i];
    }
    printf("The max value is %e\n",maxv);
    printf("The sum is %Le \n",detectsum);
    
}


