#ifndef __H2ERI_UTILS_H__
#define __H2ERI_UTILS_H__


#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>

#include "H2ERI_typedef.h"
#include "H2ERI_config.h"
#include "H2ERI_aux_structs.h"
#include "H2ERI_build_S1.h"
#include "linalg_lib_wrapper.h"
#include "TinyDFT.h"

#ifdef __cplusplus
extern "C" {
#endif

// Build the basis set for every node in the H2 partition level by level
// Input parameters
// h2eri: H2ERI data structure
// Output parameters
// h2eri: H2ERI data structure
// Urbasis: row basis set for every node in the H2 partition
void H2ERI_build_rowbs(H2ERI_p h2eri, H2E_dense_mat_p* Urbasis);


// Build the column basis set for every admissible block
// Input parameters
// h2eri: H2ERI data structure
// admpair1st: the first index of the admissible block
// admpair2nd: the second index of the admissible block
// Output parameters
// h2eri: H2ERI data structure
// Ucbasis: column basis set for every admissible block
void H2ERI_build_colbs(H2ERI_p h2eri, H2E_dense_mat_p* Ucbasis,int *admpair1st,int *admpair2nd,H2E_dense_mat_p *Urbasis);


// Extract near large elements in X and Y matrices
// Input parameters
// h2eri: H2ERI data structure
// TinyDFT: TinyDFT data structure
// r: the radius of the near field
// threshold: the threshold for large elements
// Output parameters
// csrd5: large and near elements in density matrix X in csr format
// csrdc5: large and near elements in density matrix Y in csr format
void H2ERI_extract_near_large_elements(H2ERI_p h2eri, TinyDFT_p TinyDFT, CSRmat_p csrd5, CSRmat_p csrdc5, double r, double threshold);
void H2ERI_divide_xy(H2ERI_p h2eri, TinyDFT_p TinyDFT, CSRmat_p csrd5, CSRmat_p csrdc5, CSRmat_p csrdrm, CSRmat_p csrdcrm, double r, double threshold);

// Build the pseudo inverse of every nonleaf node
// Input parameters
// h2eri: H2ERI data structure
// Output parameters
// h2eri: H2ERI data structure
// Upinv: pseudo inverse R matrix of every nonleaf node
void compute_pseudo_inverse(double* R, int nrow, int ncol, double* R_pinv);
void build_pinv_rmat(H2ERI_p h2eri, H2E_dense_mat_p* Upinv);

int Split_node(H2ERI_p h2eri, int node0, int leaf,  int *childstep, int *nodesidx, int *basisidx,H2E_int_vec_p *nodeadmpairs, H2E_int_vec_p *nodeadmpairidx);


int testadmpair(H2E_int_vec_p *nodeadmpairs, H2E_int_vec_p *nodeadmpairidx, int node0, int node1);
// Build the S5 matrix
// Input parameters
// h2eri: H2ERI data structure
// nodepairs: the node pairs in the H2 partition
// Urbasis: row basis set for every node in the H2 partition
// Ucbasis: column basis set for every admissible block
// Output parameters
// h2eri: H2ERI data structure
// S51cbasis; ordered in the same way as the pairs
// It needs to stress that S51cbasis is column major
void H2ERI_build_S5(H2ERI_p h2eri, H2E_dense_mat_p* Urbasis, H2E_dense_mat_p* Ucbasis, CSRmat_p csrd5, CSRmat_p csrdc5, int npairs, int *pair1st,
    int *pair2nd, H2E_int_vec_p *nodepairs, H2E_int_vec_p *nodeadmpairs, H2E_int_vec_p *nodeadmpairidx, H2E_dense_mat_p* S51cbasis,H2E_dense_mat_p* Upinv);


void H2ERI_build_S5test(H2ERI_p h2eri, H2E_dense_mat_p* Urbasis, H2E_dense_mat_p* Ucbasis, CSRmat_p csrd5, CSRmat_p csrdc5, int npairs, int *pair1st,
    int *pair2nd, H2E_int_vec_p *nodepairs, H2E_int_vec_p *nodeadmpairs, H2E_int_vec_p *nodeadmpairidx, H2E_dense_mat_p* S51cbasis,H2E_dense_mat_p* Upinv);

size_t H2ERI_build_S5_draft(H2ERI_p h2eri, H2E_dense_mat_p* Urbasis, H2E_dense_mat_p* Ucbasis, CSRmat_p csrd5, CSRmat_p csrdc5, int npairs, int *pair1st,
    int *pair2nd, H2E_int_vec_p *nodepairs, H2E_int_vec_p *nodeadmpairs, H2E_int_vec_p *nodeadmpairidx, H2E_dense_mat_p* S51cbasis,H2E_dense_mat_p* Upinv, double thr);
// Compute S1S51 interaction
// Input parameters
// Csrmat_p S1: the S1 matrix
// h2eri: H2ERI data structure
// Urbasis: row basis set for every node in the H2 partition
// S51cbasis: column basis set for every admissible block
// nodepairs: the node pairs in the H2 partition
// nodepairidx: the index of the node pairs in the H2 partition, which is the order that S51cbasis is ordered
// In S1S523, mnkl is column major while in S1S51 it is row major
double compute_eleval_S51(H2ERI_p h2eri, H2E_dense_mat_p* Urbasis,H2E_dense_mat_p* S51cbasis,H2E_int_vec_p *nodepairs, H2E_int_vec_p *nodepairidx, int row, int column);
double compute_eleval_Wlr(H2ERI_p h2eri, H2E_dense_mat_p* Urbasis,H2E_dense_mat_p* Ucbasis,H2E_int_vec_p *nodeadmpairs, H2E_int_vec_p *nodeadmpairidx, int row, int column);
double calc_S1S51(CSRmat_p S1, H2ERI_p h2eri, H2E_dense_mat_p* Urbasis, H2E_dense_mat_p *S51cbasis, H2E_int_vec_p *nodepairs, H2E_int_vec_p *nodepairidx);
double calc_S1S523(CSRmat_p mnke, H2ERI_p h2eri, H2E_dense_mat_p* Urbasis, H2E_dense_mat_p *Ucbasis, H2E_int_vec_p *nodeadmpairs, H2E_int_vec_p *nodeadmpairidx);


// Compute S51 self interaction
// Input parameters
// h2eri: H2ERI data structure
// Urbasis: row basis set for every node in the H2 partition
// S51cbasis: column basis set for every admissible block
// npairs: the number of node pairs in the H2 partition
// pair1st: the first index of the node pairs in the H2 partition
// pair2nd: the second index of the node pairs in the H2 partition
// Output parameters
// h2eri: H2ERI data structure
// energy: the energy of the S51 self interaction
double calc_S51_self_interaction(H2ERI_p h2eri, H2E_dense_mat_p* Urbasis, H2E_dense_mat_p* S51cbasis, int npairs, int *pair1st, int *pair2nd);


#ifdef __cplusplus
}
#endif

#endif
