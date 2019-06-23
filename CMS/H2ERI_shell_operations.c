#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "mkl.h"

#include "CMS.h"

// Rotate shell coordinates for better hierarchical partitioning
void H2ERI_rotate_shells(const int nshell, shell_t *shells)
{
    size_t col_msize = sizeof(double) * nshell;
    double center[3] = {0.0, 0.0, 0.0};
    double eigval[3], eigvec[9];
    double *coord  = (double*) malloc(col_msize * 3);
    double *coord1 = (double*) malloc(col_msize * 3);
    assert(coord != NULL && coord1 != NULL);
    
    // 1. Rotate coordinates so the center is at the origin point & 
    // the minimal bounding box of center points is parallel to unit box
    for (int i = 0; i < nshell; i++)
    {
        coord[0 * nshell + i] = shells[i].x;
        coord[1 * nshell + i] = shells[i].y;
        coord[2 * nshell + i] = shells[i].z;
        center[0] += shells[i].x;
        center[1] += shells[i].y;
        center[2] += shells[i].z;
    }
    double d_nshell = (double) nshell;
    center[0] = center[0] / d_nshell;
    center[1] = center[1] / d_nshell;
    center[2] = center[2] / d_nshell;
    for (int i = 0; i < nshell; i++)
    {
        coord[0 * nshell + i] -= center[0];
        coord[1 * nshell + i] -= center[1];
        coord[2 * nshell + i] -= center[2];
    }
    cblas_dgemm(
        CblasColMajor, CblasTrans, CblasNoTrans, 3, 3, nshell,
        1.0, coord, nshell, coord, nshell, 0.0, eigvec, 3
    );
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', 3, eigvec, 3, eigval);
    cblas_dgemm(
        CblasColMajor, CblasNoTrans, CblasTrans, nshell, 3, 3, 
        1.0, coord, nshell, eigvec, 3, 0.0, coord1, nshell
    );
    
    // 2. Move the zero columns to the end
    int col_idx = 0;
    for (int i = 0; i < 3; i++)
    {
        double *col_ptr = coord1 + i * nshell;
        double max_col_val = fabs(col_ptr[0]);
        for (int j = 1; j < nshell; j++)
        {
            double val_j = fabs(col_ptr[j]);
            if (val_j > max_col_val) max_col_val = val_j;
        }
        if (max_col_val > 1e-15)
        {
            double *dst_col = coord1 + col_idx * nshell;
            if (col_idx != i) memcpy(dst_col, col_ptr, col_msize);
            col_idx++;
        }
    }
    
    // 3. Update the center coordinates of shells
    for (int i = 0; i < nshell; i++)
    {
        shells[i].x = coord1[0 * nshell + i];
        shells[i].y = coord1[1 * nshell + i];
        shells[i].z = coord1[2 * nshell + i];
    }
    
    free(coord1);
    free(coord);
}

// Fully uncontract all shells and screen uncontracted shell pairs
void H2ERI_uncontract_shell_pairs(
    const int nshell, shell_t *shells, const double scr_tol, 
    int *num_unc_sp_, shell_t **unc_sp_, double **unc_sp_center_
)
{
    // 1. Uncontract all shells
    int nshell_unc = 0;
    for (int i = 0; i < nshell; i++) nshell_unc += shells[i].nprim;
    int *shells_unc_idx = (int *) malloc(sizeof(int) * nshell_unc * 2);
    shell_t *shells_unc = (shell_t *) malloc(sizeof(shell_t) * nshell_unc);
    assert(shells_unc_idx != NULL && shells_unc != NULL);
    for (int i = 0; i < nshell_unc; i++)
    {
        simint_initialize_shell(&shells_unc[i]);
        simint_allocate_shell(1, &shells_unc[i]);
    }
    int unc_idx = 0;
    for (int i = 0; i < nshell; i++)
    {
        int am = shells[i].am;
        double x = shells[i].x;
        double y = shells[i].y;
        double z = shells[i].z;
        for (int j = 0; j < shells[i].nprim; j++)
        {
            shells_unc[unc_idx].am       = am;
            shells_unc[unc_idx].nprim    = 1;
            shells_unc[unc_idx].x        = x;
            shells_unc[unc_idx].y        = y;
            shells_unc[unc_idx].z        = z;
            shells_unc[unc_idx].alpha[0] = shells[i].alpha[j];
            shells_unc[unc_idx].coef[0]  = shells[i].coef[j];
            shells_unc_idx[2*unc_idx]    = i;
            shells_unc_idx[2*unc_idx+1]  = j;
            unc_idx++;
        }
    }
    
    // 2. Construct new shell pairs with uncontracted shells
    double *scr_vals = (double *) malloc(sizeof(double) * nshell_unc * nshell_unc);
    assert(scr_vals != NULL);
    double max_scr_val = CMS_get_Schwarz_scrval(nshell_unc, shells_unc, scr_vals);
    double scr_thres = scr_tol * scr_tol / max_scr_val;
    int num_unc_sp = 0;
    for (int i = 0; i < nshell_unc; i++)
    {
        double *src_vals_row = scr_vals + i * nshell_unc;
        for (int j = i; j < nshell_unc; j++)
            if (src_vals_row[j] >= scr_thres) num_unc_sp++;
    }
    double *unc_sp_center = (double *) malloc(sizeof(double) * num_unc_sp * 3);
    shell_t *unc_sp = (shell_t *) malloc(sizeof(shell_t) * num_unc_sp * 2);
    assert(unc_sp_center != NULL && unc_sp != NULL);
    for (int i = 0; i < num_unc_sp * 2; i++)
    {
        simint_initialize_shell(&unc_sp[i]);
        simint_allocate_shell(1, &unc_sp[i]);
    }
    int unc_sp_idx = 0;
    int cidx0 = 0, cidx1 = num_unc_sp, cidx2 = 2 * num_unc_sp;
    const double sqrt2 = sqrt(2.0);
    for (int i = 0; i < nshell_unc; i++)
    {
        double *src_vals_row = scr_vals + i * nshell_unc;
        double a_i = shells_unc[i].alpha[0];
        double x_i = shells_unc[i].x;
        double y_i = shells_unc[i].y;
        double z_i = shells_unc[i].z;
        for (int j = i; j < nshell_unc; j++)
        {
            if (src_vals_row[j] < scr_thres) continue;
            
            // Add a new shell pair
            double a_j = shells_unc[j].alpha[0];
            double x_j = shells_unc[j].x;
            double y_j = shells_unc[j].y;
            double z_j = shells_unc[j].z;
            double aij = a_i + a_j;
            unc_sp_center[cidx0] = (a_i * x_i + a_j * x_j) / aij;
            unc_sp_center[cidx1] = (a_i * y_i + a_j * y_j) / aij;
            unc_sp_center[cidx2] = (a_i * z_i + a_j * z_j) / aij;
            simint_copy_shell(&shells_unc[i], &unc_sp[unc_sp_idx]);
            simint_copy_shell(&shells_unc[j], &unc_sp[unc_sp_idx + 1]);
            
            // If two shell_uncs come from the same contracted shell but are
            // different primitive functions, multiple a sqrt(2) for symmetry.
            // Let shell A = a1 + a2, B = b1 + b2, (AB| = \sum \sum a_i b_j.
            // If A == B, due to symmetry, we only handle (a1a2| once.
            int shell_idx_i = shells_unc_idx[2 * i];
            int prim_idx_i  = shells_unc_idx[2 * i + 1];
            int shell_idx_j = shells_unc_idx[2 * j];
            int prim_idx_j  = shells_unc_idx[2 * j + 1];
            if ((shell_idx_i == shell_idx_j) && (prim_idx_i != prim_idx_j))
            {
                unc_sp[unc_sp_idx].coef[0]     *= sqrt2;
                unc_sp[unc_sp_idx + 1].coef[0] *= sqrt2;
            }
            
            unc_sp_idx += 2;
            cidx0++;
            cidx1++;
            cidx2++;
        }
    }
    
    // 3. Free temporary arrays and return
    *num_unc_sp_ = num_unc_sp;
    *unc_sp_ = unc_sp;
    *unc_sp_center_ = unc_sp_center;
    free(scr_vals);
    for (int i = 0; i < nshell_unc; i++)
        simint_free_shell(&shells_unc[i]);
    free(shells_unc);
    free(shells_unc_idx);
}
