//#ifndef __H2ERI_BUILD_H2_H__
//#define __H2ERI_BUILD_H2_H__

#include "H2ERI_typedef.h"
#include <omp.h>
#ifdef __cplusplus
extern "C" {
#endif

// Build H2 representation for ERI tensor
// Input parameter:
//   h2eri  : H2ERI structure with D matrix info
//   D1tst  : the parameter determining whether to use neighbor blocks. If D1tst==1, then use. If 0 then only diagonal ones.

// Output parameter:
//   coomat : Dmat information stored in COO format


void H2ERI_build_COO_fulldensetest(H2ERI_p h2eri , COOmat_p coomat);
void H2ERI_build_COO_noderow(H2ERI_p h2eri , COOmat_p coomat);



//Provided the COO matrix ERI tensor (\mu\nu|\lambda\sigma)
//Calculate the ERI tensor (\mu\sigma|\lambda\nu) in COO form
// Input parameters:
//   cooin : COO matrix information of (\mu\nu|\lambda\sigma)
// Output parameters:
//   cooout : COO matrix information of (\mu\sigma|\lambda\nu)
void H2ERI_build_COO_transpose(COOmat_p cooin, COOmat_p cooout, int nbf);

// Quick sorting a double key-value pair array by key
// Input parameters:
//   key, val : Array, size >= r+1, key-value pairs
//   l, r     : Sort range: [l, r-1]
// Output parameters:
//   key, val : Array, size >= r+1, sorted key-value pairs
void Qsort_double_long0(int *key, double *val, int l, int r);


void Qsort_double_long(int *key, double *val, size_t l, size_t r);

void Qsort_double_long1(int *key, double *val, size_t l, size_t r);
void Qsort_double_long2(int *key, double *val, size_t low, size_t high);
void insertionSort(int *key, double *val, size_t l, size_t r);

// Compress the COO matrices
// Input cooini and output coofinal. coofinal contains all the values whose absolute value is larger than thres*maximum
void compresscoo(COOmat_p cooini, COOmat_p coofinal, double thres);
void compresscsr(CSRmat_p cooini, CSRmat_p coofinal, double thres);


// Convert a double COO matrix to a CSR matrix 
// Input parameters:
//   nrow          : Number of rows
//   nnz           : Number of nonzeros in the matrix
//   coomat        : COO Matrix information
// Output parameters:
//   csrmat        : CSR matrix
void Double_COO_to_CSR(
    const int nrow, const size_t nnz, COOmat_p coomat, CSRmat_p csrmat
);


void Double_COO_to_CSR_nosort(
    const int nrow, const size_t nnz, COOmat_p coomat, CSRmat_p csrmat
);


// Provided a dense matrix, calculate the sparse large values of the matrix into COO matrix forms.
// of the matrices into sparse form. Typically it is used to extract out the large elements in density matrix and its complimentary.
// Input parameters:
//   nrow           : number of rows, in D and DC it equals h2eri->num+bf
//   ncol           : number of columns, in D and DC it equals h2eri->num+bf
//   thres          : Relative threshold of selection. The COO matrix only stores the elements of absolute
//                    values larger than the thres*Maxvalue.
//   mat            :The matrix to extract large elements. Size nbf*nbf, stored in row major.
//
//
// Output parameters:
//   return         : nnz, number of nonzero values of the matrix.
//   mat            : The matrix with remaining elements. The large values are extracted out and the values in their position are replaced by 0.
//   coomat         : The COO Matrix containing the information of large elements.

size_t Extract_COO_DDCMat(const int nrow, const int ncol, const double thres, double * mat, COOmat_p coomat);


// Do the X index transformation
// Input parameters:
// nbf              : number of basis functions
// csrh2d           : the h2mat diagonal ERI tensor to be transformed in CSR form
// csrden           : the density matrix in CSR form

// Output parameter:
// csrtrans         : the X index transformation result in csr form

void Xindextransform(int nbf, CSRmat_p csrh2d, CSRmat_p csrden, CSRmat_p csrtrans);

void Xindextransform1(int nbf, CSRmat_p csrh2d, CSRmat_p csrden, CSRmat_p csrtrans);
void Xindextransform2(int nbf, CSRmat_p csrh2d, CSRmat_p csrden, CSRmat_p csrtrans);
void Xindextransform3(int nbf, CSRmat_p csrh2d, CSRmat_p csrden, CSRmat_p csrtrans);
void Xindextransform4(int nbf, CSRmat_p csrh2d, CSRmat_p csrden, CSRmat_p csrtrans, double thres);
void Xindextransform5(int nbf, CSRmat_p csrh2d, CSRmat_p csrden, CSRmat_p csrtrans, double thres);

// Do the Y index transformation
// Input parameters:
// nbf              : number of basis functions
// csrh2d           :the matrix to be transformed in CSR form
// csrdc            : the density complimentary matrix in CSR form

// Output parameter:
// csrtrans         : the X index transformation result in csr form

void Yindextransform1(int nbf, CSRmat_p csrh2d, CSRmat_p csrdc, CSRmat_p csrtrans);
void Yindextransform2(int nbf, CSRmat_p csrh2d, CSRmat_p csrdc, CSRmat_p csrtrans);
void Yindextransform3(int nbf, CSRmat_p csrh2d, CSRmat_p csrdc, CSRmat_p csrtrans);
void Yindextransform4(int nbf, CSRmat_p csrh2d, CSRmat_p csrdc, CSRmat_p csrtrans,double thres);
void Yindextransform5(int nbf, CSRmat_p csrh2d, CSRmat_p csrdc, CSRmat_p csrtrans,double thres);
void Yindextransform6(int nbf, H2ERI_p h2eri, CSRmat_p csrh2d, CSRmat_p csrdc, CSRmat_p csrtrans,double thres);

void H2Pmp2_build_S1(int nbf,H2ERI_p h2eri, CSRmat_p csrh2d, CSRmat_p csrden, CSRmat_p csrdc, CSRmat_p csrtrans, double thres);
void H2Pmp2_build_S11(int nbf,H2ERI_p h2eri, CSRmat_p csrh2d, CSRmat_p csrden, CSRmat_p csrdc, CSRmat_p csrtrans, double thres);

// Provided the row oriented CSR matrix, compute the column oriented CSC matrix.
// Attension: The new CSC matrix is still applying the previous CSRmat data type.
// In cscmat, cscmat->csrrow is actually the column while cscmat->csrcol is actually the row.
// Need to pay attension to it to avoid confusion and errors.
// Input parameters:
// csrmat          : The row oriented CSR matrix
// Output parameters:
// cscmat          : The column oriented CSC matrix stored in CSRmat form

void CSR_to_CSC(const int ncol, CSRmat_p csrmat, CSRmat_p cscmat);



// Calculate the S1 energy
// Input parameters:
// csrs1            : S1 matrix in CSR form
// cscs1            : S1 matrix in CSC form but cscs1->csrrow stores the information of columns
// Output
// return: S1 energy
double Calc_S1energy(CSRmat_p csrs1, CSRmat_p cscs1);
double Calcs1self(H2ERI_p h2eri, CSRmat_p nodes1);
double Calcs1selfnode(H2ERI_p h2eri, CSRmat_p csrnode);





#ifdef __cplusplus
}
#endif

//#endif
