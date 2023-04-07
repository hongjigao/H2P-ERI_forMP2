//#ifndef __H2ERI_BUILD_H2_H__
//#define __H2ERI_BUILD_H2_H__

#include "H2ERI_typedef.h"
#include "H2Pack_aux_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build H2 representation for ERI tensor
// Input parameter:
//   h2eri  : H2ERI structure with D matrix info

// Output parameter:
//   h2eri : COO formed D matrix elements
void H2ERI_build_COO_Diamat(H2ERI_p h2eri, int D1tst);




// Quick sorting a double key-value pair array by key
// Input parameters:
//   key, val : Array, size >= r+1, key-value pairs
//   l, r     : Sort range: [l, r-1]
// Output parameters:
//   key, val : Array, size >= r+1, sorted key-value pairs
void Qsort_double_key_val(int *key, double *val, int l, int r);



// Convert a double COO matrix to a CSR matrix 
// Input parameters:
//   nrow          : Number of rows
//   nnz           : Number of nonzeros in the matrix
//   row, col, val : Size nnz, COO matrix
// Output parameters:
//   row_ptr, col_idx, val_ : Size nrow+1, nnz, nnz, CSR matrix
void Double_COO_to_CSR(
    const int nrow, const int nnz, const int *row, const int *col, 
    const double *val, int *row_ptr, int *col_idx, double *val_
);

#ifdef __cplusplus
}
#endif

//#endif
