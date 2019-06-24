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
    const int nshell, const shell_t *shells, const double scr_tol, 
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
        }  // End of j loop
    }  // End of i loop
    
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

// Estimate the extent of Gaussian function 
//       coef * exp(-alpha * x^2) * x^am
// Input parameters:
//   alpha, coef, am : Gaussian function parameter
//   tol : Tolerance of extent
// Output parameter:
//   <return> : Estimated extent of given Gaussian function
#define Gaussian_tol(x) (coef * exp(-alpha * (x) * (x)) * pow((x), am) - tol)
double H2ERI_calc_Gaussian_extent(
    const double alpha, const double coef, 
    const int _am,      const double tol
)
{
    double am = (double) _am;
    double lower = 0.0, upper = 20.0;
    
    // If am > 0, the maximum of Gaussian is obtained at x = sqrt(am / 2 / alpha)
    if (_am > 0) 
    {
        lower = sqrt(am * 0.5 / alpha);
        upper = 10.0 * lower;
    }
    
    if (Gaussian_tol(lower) <= 1e-8 * tol) return 0.0;
    while (Gaussian_tol(upper) > 0) upper *= 2.0;
    
    double extent   = (upper + lower) * 0.5;
    double G_extent = Gaussian_tol(extent);
    while (fabs(G_extent) > 2e-16)
    {
        if (G_extent > 0.0) lower = extent; 
        else                upper = extent;
        extent   = (upper + lower) * 0.5;
        G_extent = Gaussian_tol(extent);
    }
    
    return extent;
}

// Calculate the extent of each shell pair
void H2ERI_calc_shell_pair_extents(
    const int num_sp, const shell_t *sp, 
    const double ext_tol, double *sp_extent
)
{
    int max_nprim = 0;
    for (int i = 0; i < num_sp * 2; i++)
        max_nprim = MAX(max_nprim, sp[i].nprim);
    size_t tmp_msize = sizeof(double) * max_nprim * max_nprim;
    double *extent_tmp = (double *) malloc(tmp_msize);
    double *center_tmp = (double *) malloc(tmp_msize * 3);
    double *upper = (double *) malloc(tmp_msize * 3);
    double *lower = (double *) malloc(tmp_msize * 3);
    assert(extent_tmp != NULL && center_tmp != NULL);
    assert(upper != NULL && lower != NULL);
    
    for (int i = 0; i < num_sp; i++)
    {
        int i20 = i * 2;
        int i21 = i * 2 + 1;
        int am1 = sp[i20].am;
        int am2 = sp[i21].am;
        int nprim1  = sp[i20].nprim;
        int nprim2  = sp[i21].nprim;
        int nprim12 = nprim1 * nprim2;
        double x1 = sp[i20].x;
        double y1 = sp[i20].y;
        double z1 = sp[i20].z;
        double x2 = sp[i21].x;
        double y2 = sp[i21].y;
        double z2 = sp[i21].z;
        double dx = x1 - x2;
        double dy = y1 - y2;
        double dz = z1 - z2;
        double *alpha1 = sp[i20].alpha;
        double *alpha2 = sp[i21].alpha;
        double *coef1  = sp[i20].coef;
        double *coef2  = sp[i21].coef;
        double tol_i = ext_tol / (double) nprim12;
        double r12   = dx * dx + dy * dy + dz * dz;
        
        // 1. Calculate the center and extent of each primitive function pair
        int j_idx = 0;
        for (int j1 = 0; j1 < nprim1; j1++)
        {
            double aj1 = alpha1[j1];
            for (int j2 = 0; j2 < nprim2; j2++)
            {
                double aj2  = alpha2[j2];
                double aj12 = aj1 + aj2;
                double cj12 = coef1[j1] * coef2[j2];
                double exp_c = (aj1 * aj2 / aj12) * r12;
                double coef  = cj12 * exp(-exp_c);
                double *center_j = center_tmp + j_idx * 3;
                center_j[0] = (aj1 * x1 + aj2 * x2) / aj12;
                center_j[1] = (aj1 * y1 + aj2 * y2) / aj12;
                center_j[2] = (aj1 * z1 + aj2 * z2) / aj12;
                extent_tmp[j_idx] = H2ERI_calc_Gaussian_extent(aj12, coef, am1 + am2, tol_i);
                j_idx++;
            }
        }
        
        // 2. Find a large box to cover all extents
        double d12 = sqrt(r12);
        if (d12 < 5e-16)
        {
            double max_ext = 0.0;
            for (int j = 0; j < nprim12; j++)
                max_ext = MAX(max_ext, extent_tmp[j]);
            sp_extent[i] = max_ext;
        } else {
            dx /= d12;  dy /= d12;  dz /= d12;
            for (int j = 0; j < nprim12; j++)
            {
                double extent_j  = extent_tmp[j];
                double *center_j = center_tmp + j * 3;
                double *upper_j  = upper + j * 3;
                double *lower_j  = lower + j * 3;
                upper_j[0] = center_j[0] + extent_j * dx;
                upper_j[1] = center_j[1] + extent_j * dy;
                upper_j[2] = center_j[2] + extent_j * dz;
                lower_j[0] = center_j[0] - extent_j * dx;
                lower_j[1] = center_j[1] - extent_j * dy;
                lower_j[2] = center_j[2] - extent_j * dz;
            }
            // j1's upper bound of extent has the largest distance to coord2
            // j2's lower bound of extent has the largest distance to coord1
            int j1 = 0, j2 = 0;
            double jc1_max = 0.0, jc2_max = 0.0;
            for (int j = 0; j < nprim12; j++)
            {
                double *upper_j = upper + j * 3;
                double *lower_j = lower + j * 3;
                
                dx = upper_j[0] - x2;
                dy = upper_j[1] - y2;
                dz = upper_j[2] - z2;
                double dist_j_c1 = sqrt(dx * dx + dy * dy + dz * dz);
                if (dist_j_c1 > jc1_max)
                {
                    jc1_max = dist_j_c1;
                    j1 = j;
                }
                
                dx = lower_j[0] - x1;
                dy = lower_j[1] - y1;
                dz = lower_j[2] - z1;
                double dist_j_c2 = sqrt(dx * dx + dy * dy + dz * dz);
                if (dist_j_c2 > jc2_max)
                {
                    jc2_max = dist_j_c2;
                    j2 = j;
                }
            }
            j1 = 0; j2 = 0;
            double *upper_j1 = upper + j1 * 3;
            double *lower_j2 = lower + j2 * 3;
            dx = upper_j1[0] - lower_j2[0];
            dy = upper_j1[1] - lower_j2[1];
            dz = upper_j1[2] - lower_j2[2];
            sp_extent[i] = sqrt(dx * dx + dy * dy + dz * dz) * 0.5;
        }  // End of "if (d12 < 1e-15)"
    }  // End of i loop
    
    free(lower);
    free(upper);
    free(center_tmp);
    free(extent_tmp);
}


