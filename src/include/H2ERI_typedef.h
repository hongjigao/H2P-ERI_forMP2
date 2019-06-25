#ifndef __H2ERI_TYPEDEF_H__
#define __H2ERI_TYPEDEF_H__

// Shell operations used in H2P-ERI

#include "CMS.h"
#include "H2Pack_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

struct H2ERI
{
    int     nshell;           // Number of contracted shells (from input file)
    int     num_unc_sp;       // Number of fully uncontracted shell pairs (FUSP)
    int     *shell_bf_sidx;   // Array, size nshell+1, index of each shell's first basis function
    int     *unc_sp_bf_sidx;  // Array, size num_unc_sp+1, index of each FUSP first basis function 
    double  scr_tol;          // Tolerance of Schwarz screening
    double  ext_tol;          // Tolerance of shell pair extent
    double *unc_sp_center;    // Array, size 3 * num_unc_sp, center of FUSP
    double *unc_sp_extent;    // Array, size num_unc_sp, extent of FUSP
    shell_t *shells;          // Array, size nshell, contracted shells
    shell_t *unc_sp;          // Array, size num_unc_sp * 2, FUSP
    
    H2Pack_t h2pack;          // H2Pack data structure
};

typedef struct H2ERI* H2ERI_t;

// Initialize a H2ERI structure
// Input parameters:
//   scr_tol : Tolerance of Schwarz screening (typically 1e-10)
//   ext_tol : Tolerance of shell pair extent (typically 1e-10)
//   QR_tol  : Tolerance of column-pivoting QR (controls the overall accuracy)
// Output parameter:
//   h2eri_ : Initialized H2ERI structure
void H2ERI_init(H2ERI_t *h2eri_, const double scr_tol, const double ext_tol, const double QR_tol);

// Destroy a H2ERI structure
// Input parameter:
//   h2eri : H2ERI structure to be destroyed
void H2ERI_destroy(H2ERI_t h2eri);

#ifdef __cplusplus
}
#endif

#endif