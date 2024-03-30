#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h>

#include "H2ERI_typedef.h"
#include "H2ERI_build_Coulomb.h"
#include "H2ERI_matvec.h"
#include "H2ERI_build_S1.h"

// "Uncontract" the density matrix according to SSP and unroll 
// the result to a column for H2 matvec.
// Input parameters:
//   den_mat              : Symmetric density matrix, size h2eri->num_bf^2
//   h2eri->num_bf        : Number of basis functions in the system
//   h2eri->num_sp        : Number of screened shell pairs (SSP)
//   h2eri->shell_bf_sidx : Array, size nshell, indices of each shell's 
//                          first basis function
//   h2eri->sp_bfp_sidx   : Array, size num_sp+1, indices of each 
//                          SSP's first basis function pair
//   h2eri->sp_shell_idx  : Array, size 2 * num_sp, each row is 
//                          the contracted shell indices of a SSP
// Output parameter:
//   h2eri->unc_denmat_x  : Array, size num_sp_bfp, uncontracted density matrix
void H2ERI_uncontract_den_mat(H2ERI_p h2eri, const double *den_mat)
{
    int num_bf = h2eri->num_bf;
    int num_sp = h2eri->num_sp;
    int num_sp_bfp = h2eri->num_sp_bfp;
    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int *sp_bfp_sidx   = h2eri->sp_bfp_sidx;
    int *sp_shell_idx  = h2eri->sp_shell_idx;
    double *x = h2eri->unc_denmat_x;
    h2eri->bf1st = (int*) malloc(sizeof(int) * num_sp_bfp);
    h2eri->bf2nd = (int*) malloc(sizeof(int) * num_sp_bfp);
    h2eri->sameshell = (int*) malloc(sizeof(int) * num_sp_bfp);
    int *bf1st = h2eri->bf1st;
    int *bf2nd = h2eri->bf2nd;
    int *sameshell = h2eri->sameshell;
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < num_sp; i++)
    {
        int x_spos = sp_bfp_sidx[i];
        int shell_idx0 = sp_shell_idx[i];
        int shell_idx1 = sp_shell_idx[i + num_sp];
        int srow = shell_bf_sidx[shell_idx0];
        int erow = shell_bf_sidx[shell_idx0 + 1];
        int scol = shell_bf_sidx[shell_idx1];
        int ecol = shell_bf_sidx[shell_idx1 + 1];
        int nrow = erow - srow;
        int ncol = ecol - scol;
        double sym_coef = (shell_idx0 == shell_idx1) ? 1.0 : 2.0;
        int same_shell = (shell_idx0 == shell_idx1) ? 1 : 0;
        

        // Originally we need to store den_mat[srow:erow-1, scol:ecol-1]
        // column by column to x(x_spos:x_epos-1). Since den_mat is 
        // symmetric, we store den_mat[scol:ecol-1, srow:erow-1] row by 
        // row to x(x_spos:x_epos-1).
        for (int j = 0; j < ncol; j++)
        {
            const double *den_mat_ptr = den_mat + (scol + j) * num_bf + srow;
            double *x_ptr = x + x_spos + j * nrow;
            int *bf1st_ptr = bf1st + x_spos + j * nrow;
            int *bf2nd_ptr = bf2nd + x_spos + j * nrow;
            int *sameshell_ptr = sameshell + x_spos + j * nrow;

            #pragma omp simd 
            for (int k = 0; k < nrow; k++)
            {
                x_ptr[k] = sym_coef * den_mat_ptr[k];
                bf1st_ptr[k]=srow+k;
                bf2nd_ptr[k]=scol+j;
                sameshell_ptr[k]=same_shell;
            }
                
        }
    }
}

// "Contract" the H2 matvec result according to SSP and reshape
// the result to form a symmetric Coulomb matrix
// Input parameters:
//   h2eri->num_bf        : Number of basis functions in the system
//   h2eri->num_sp        : Number of SSP
//   h2eri->shell_bf_sidx : Array, size nshell, indices of each shell's 
//                          first basis function
//   h2eri->sp_bfp_sidx   : Array, size num_sp+1, indices of each 
//                          SSP's first basis function pair
//   h2eri->sp_shell_idx  : Array, size 2 * num_sp, each row is 
//                          the contracted shell indices of a SSP
//   h2eri->H2_matvec_y   : Array, size num_sp_bfp, H2 matvec result 
// Output parameter:
//   J_mat : Symmetric Coulomb matrix, size h2eri->num_bf^2
void H2ERI_contract_H2_matvec(H2ERI_p h2eri, double *J_mat)
{
    int num_bf = h2eri->num_bf;
    int num_sp = h2eri->num_sp;
    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int *sp_bfp_sidx   = h2eri->sp_bfp_sidx;
    int *sp_shell_idx  = h2eri->sp_shell_idx;
    double *y = h2eri->H2_matvec_y;
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_bf * num_bf; i++) J_mat[i] = 0.0;
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < num_sp; i++)
    {
        int y_spos = sp_bfp_sidx[i];
        int shell_idx0 = sp_shell_idx[i];
        int shell_idx1 = sp_shell_idx[i + num_sp];
        int srow = shell_bf_sidx[shell_idx0];
        int erow = shell_bf_sidx[shell_idx0 + 1];
        int scol = shell_bf_sidx[shell_idx1];
        int ecol = shell_bf_sidx[shell_idx1 + 1];
        int nrow = erow - srow;
        int ncol = ecol - scol;
        double sym_coef = (shell_idx0 == shell_idx1) ? 0.5 : 1.0;
        
        // Originally we need to reshape y(y_spos:y_epos-1) as a
        // nrow-by-ncol column-major matrix and add it to column-major
        // matrix J_mat[srow:erow-1, scol:ecol-1]. Since J_mat is 
        // symmetric, we reshape y(y_spos:y_epos-1) as a ncol-by-nrow
        // row-major matrix and add it to J_mat[scol:ecol-1, srow:erow-1].
        for (int j = 0; j < ncol; j++)
        {
            double *J_mat_ptr = J_mat + (scol + j) * num_bf + srow;
            double *y_ptr = y + y_spos + j * nrow;
            #pragma omp simd 
            for (int k = 0; k < nrow; k++) J_mat_ptr[k] += sym_coef * y_ptr[k];
        }
    }
    
    // Symmetrize the Coulomb matrix: J_mat = J_mat + J_mat^T
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < num_bf; i++)
    {
        for (int j = 0; j < i; j++)
        {
            int idx0 = i * num_bf + j;
            int idx1 = j * num_bf + i;
            double val = J_mat[idx0] + J_mat[idx1];
            J_mat[idx0] = val;
            J_mat[idx1] = val;
        }
        int idx_ii = i * num_bf + i;
        J_mat[idx_ii] += J_mat[idx_ii];
    }
}

// Build the Coulomb matrix using the density matrix, H2 representation
// of the ERI tensor, and H2 matvec
void H2ERI_build_Coulomb(H2ERI_p h2eri, const double *den_mat, double *J_mat)
{
    if (h2eri->unc_denmat_x == NULL)
    {
        size_t vec_msize = sizeof(double) * h2eri->num_sp_bfp;
        h2eri->unc_denmat_x = (double *) malloc(vec_msize);
        h2eri->H2_matvec_y  = (double *) malloc(vec_msize);
        assert(h2eri->unc_denmat_x != NULL && h2eri->H2_matvec_y != NULL);
    }
    
    H2ERI_uncontract_den_mat(h2eri, den_mat);
    
    H2ERI_matvec(h2eri, h2eri->unc_denmat_x, h2eri->H2_matvec_y);
    
    H2ERI_contract_H2_matvec(h2eri, J_mat);
}


void H2ERI_build_Coulombtest(H2ERI_p h2eri, const double *den_mat, double *J_mat)
{
    if (h2eri->unc_denmat_x == NULL)
    {
        size_t vec_msize = sizeof(double) * h2eri->num_sp_bfp;
        h2eri->unc_denmat_x = (double *) malloc(vec_msize);
        h2eri->H2_matvec_y  = (double *) malloc(vec_msize);
        assert(h2eri->unc_denmat_x != NULL && h2eri->H2_matvec_y != NULL);
    }
    
    H2ERI_uncontract_den_mat(h2eri, den_mat);
    
    H2ERI_matvectest(h2eri, h2eri->unc_denmat_x, h2eri->H2_matvec_y);
    
    H2ERI_contract_H2_matvec(h2eri, J_mat);
}

void H2ERI_build_Coulombtest2(H2ERI_p h2eri, const double *den_mat, double *J_mat)
{
    if (h2eri->unc_denmat_x == NULL)
    {
        size_t vec_msize = sizeof(double) * h2eri->num_sp_bfp;
        h2eri->unc_denmat_x = (double *) malloc(vec_msize);
        h2eri->H2_matvec_y  = (double *) malloc(vec_msize);
        assert(h2eri->unc_denmat_x != NULL && h2eri->H2_matvec_y != NULL);
    }
    
    H2ERI_uncontract_den_mat(h2eri, den_mat);
    
    H2ERI_matvectest2(h2eri, h2eri->unc_denmat_x, h2eri->H2_matvec_y);
    
    H2ERI_contract_H2_matvec(h2eri, J_mat);
}

void H2ERI_build_Coulombtest1(H2ERI_p h2eri, const double *den_mat, double *J_mat)
{
    if (h2eri->unc_denmat_x == NULL)
    {
        size_t vec_msize = sizeof(double) * h2eri->num_sp_bfp;
        h2eri->unc_denmat_x = (double *) malloc(vec_msize);
        h2eri->H2_matvec_y  = (double *) malloc(vec_msize);
        assert(h2eri->unc_denmat_x != NULL && h2eri->H2_matvec_y != NULL);
    }
    
    H2ERI_uncontract_den_mat(h2eri, den_mat);
    
    H2ERI_matvectest1(h2eri, h2eri->unc_denmat_x, h2eri->H2_matvec_y);
    
    H2ERI_contract_H2_matvec(h2eri, J_mat);
}




void H2ERI_build_Coulombdiagtest(H2ERI_p h2eri, const double *den_mat, double *J_mat)
{
    if (h2eri->unc_denmat_x == NULL)
    {
        size_t vec_msize = sizeof(double) * h2eri->num_sp_bfp;
        h2eri->unc_denmat_x = (double *) malloc(vec_msize);
        h2eri->H2_matvec_y  = (double *) malloc(vec_msize);
        assert(h2eri->unc_denmat_x != NULL && h2eri->H2_matvec_y != NULL);
    }
    
    H2ERI_uncontract_den_mat(h2eri, den_mat);
    
    H2ERI_matvectestdiag(h2eri, h2eri->unc_denmat_x, h2eri->H2_matvec_y);
    
    H2ERI_contract_H2_matvec(h2eri, J_mat);
}


void H2ERI_build_Coulombpointdiagtest(H2ERI_p h2eri, const double *den_mat, double *J_mat)
{
    if (h2eri->unc_denmat_x == NULL)
    {
        size_t vec_msize = sizeof(double) * h2eri->num_sp_bfp;
        h2eri->unc_denmat_x = (double *) malloc(vec_msize);
        h2eri->H2_matvec_y  = (double *) malloc(vec_msize);
        assert(h2eri->unc_denmat_x != NULL && h2eri->H2_matvec_y != NULL);
    }
    
    H2ERI_uncontract_den_mat(h2eri, den_mat);
    
    COOmat_p cooh2d;
    COOmat_init(&cooh2d,h2eri->num_sp_bfp,h2eri->num_sp_bfp);
    cooh2d->nnz=h2eri->nD0element;
    cooh2d->coorow = (int*) malloc(sizeof(int) * (cooh2d->nnz));
    cooh2d->coocol = (int*) malloc(sizeof(int) * (cooh2d->nnz));
    cooh2d->cooval = (double*) malloc(sizeof(double) * (cooh2d->nnz));
    ASSERT_PRINTF(cooh2d->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(cooh2d->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(cooh2d->cooval    != NULL, "Failed to allocate arrays for D matrices indexing\n");
    int *rowD            = cooh2d->coorow;
    int *colD            = cooh2d->coocol;
    double *DataD           = cooh2d->cooval;
    int *rowspot;
    int *colspot;
    double *Dataspot;
    int rowspotupdate=0;
    int colspotupdate=0;
    int mu;
    int nu;
    int lambda;
    int sigma;
    int rowidx;
    int rowvalue;
    int colidx;
    int colvalue;

    size_t numdata=0;
    int *pt_cluster      = h2eri->pt_cluster;
    int    *leaf_nodes    = h2eri->height_nodes;
    int    n_leaf_node    = h2eri->n_leaf_node;
    int *D_nrow          = h2eri->D_nrow;
    int *D_ncol          = h2eri->D_ncol;
    H2E_dense_mat_p  *c_D_blks   = h2eri->c_D_blks;
    //Store the information in D0 into COO matrix
    for(int i=0;i<n_leaf_node;i++)
    {
        int node = leaf_nodes[i];
        int pt_s = pt_cluster[2 * node];
        int pt_e = pt_cluster[2 * node + 1];
        //number of shell pairs in the node
    //       int node_npts = pt_e - pt_s + 1;
        int Di_nrow = D_nrow[i];
        int Di_ncol = D_ncol[i];
        int dspot=0;
        H2E_dense_mat_p Di = c_D_blks[i];
        rowspot=rowD+numdata;
        colspot=colD+numdata;
        Dataspot=DataD+numdata;
        int startpoint = h2eri->mat_cluster[2*node];
        for(int j=0;j<Di_nrow;j++)
        {
            rowvalue = startpoint+j;
            for(int k=0;k<Di_ncol;k++)
            {
                colvalue=startpoint+k;
                rowspot[dspot]=rowvalue;
                colspot[dspot]=colvalue;
                Dataspot[dspot]=Di->data[dspot];
    //                Di->data[dspot]=0;
                dspot+=1;
            }
        }
        if(dspot != Di->size) printf("Wrong\n");
        numdata += dspot;
        dspot = 0;
    }
    printf("numdata=%ld\n",numdata);
    printf("In comparison, nD0element=%ld\n",h2eri->nD0element);

    CSRmat_p csrh2d;
    CSRmat_init(&csrh2d,h2eri->num_sp_bfp,h2eri->num_sp_bfp);

    Double_COO_to_CSR( h2eri->num_sp_bfp,  cooh2d->nnz, cooh2d,csrh2d);
    double *x=h2eri->unc_denmat_x;
    double *y=h2eri->H2_matvec_y;
    memset(y,0,sizeof(double)*h2eri->num_sp_bfp);
    printf("memset success\n");
    for(int i=0;i<h2eri->num_sp_bfp;i++)
    {
        if(csrh2d->csrrow[i+1]-csrh2d->csrrow[i]!=0)
        {
            for(size_t j=csrh2d->csrrow[i];j<csrh2d->csrrow[i+1];j++)
            {
                y[i]+=csrh2d->csrval[j]*x[csrh2d->csrcol[j]];
            }
        }
    }
    printf("matvec success\n");
    H2ERI_contract_H2_matvec(h2eri, J_mat);
}