#ifndef __H2ERI_BUILD_COULOMB_H__
#define __H2ERI_BUILD_COULOMB_H__

#include "H2ERI_typedef.h"
#include "H2ERI_build_S1.h"

#ifdef __cplusplus
extern "C" {
#endif




void H2ERI_uncontract_den_mat(H2ERI_p h2eri, const double *den_mat);
void H2ERI_contract_H2_matvec(H2ERI_p h2eri, double *J_mat);

// Build the Coulomb matrix with the density matrix and H2 representation of the ERI tensor
// Input parameters:
//   h2eri   : H2ERI structure with H2 representation for ERI tensor
//   den_mat : Symmetric density matrix, size h2eri->num_bf * h2eri->num_bf
// Output parameters:
//   J_mat : Symmetric Coulomb matrix, size h2eri->num_bf * h2eri->num_bf
void H2ERI_build_Coulomb(H2ERI_p h2eri, const double *den_mat, double *J_mat);
void H2ERI_build_Coulombtest(H2ERI_p h2eri, const double *den_mat, double *J_mat);
void H2ERI_build_Coulombtest1(H2ERI_p h2eri, const double *den_mat, double *J_mat);
void H2ERI_build_Coulombtest2(H2ERI_p h2eri, const double *den_mat, double *J_mat);
void H2ERI_build_Coulombdiagtest(H2ERI_p h2eri, const double *den_mat, double *J_mat);
void H2ERI_build_Coulombpointdiagtest(H2ERI_p h2eri, const double *den_mat, double *J_mat);



#ifdef __cplusplus
}
#endif

#endif
