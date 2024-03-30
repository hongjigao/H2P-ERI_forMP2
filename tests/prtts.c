#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>

void compute_pseudo_inverse(double* R, int nrow, int ncol, double* R_pinv) {
    int lda = nrow;
    int ldu = nrow;
    int ldvt = ncol;
    int info;

    // Allocate memory for the decomposition
    double* S = (double*)malloc(sizeof(double) * ncol);
    double* U = (double*)malloc(sizeof(double) * nrow * nrow);
    double* VT = (double*)malloc(sizeof(double) * ncol * ncol);
    double* superb = (double*)malloc(sizeof(double) * (ncol - 1));

    // Compute the SVD of R
    info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', nrow, ncol, R, lda, S, U, ldu, VT, ldvt, superb);

    if (info > 0) {
        printf("The algorithm computing SVD failed to converge.\n");
        exit(1);
    }

    // Compute the pseudo-inverse from SVD results
    for (int i = 0; i < ncol; ++i) {
        for (int j = 0; j < nrow; ++j) {
            R_pinv[i * nrow + j] = 0.0;
            for (int k = 0; k < ncol; ++k) {
                if (S[k] > 1e-10) { // Threshold for numerical stability
                    R_pinv[i * nrow + j] += VT[i * ncol + k] * (1.0 / S[k]) * U[j * nrow + k];
                }
            }
        }
    }

    // Free allocated memory
    free(S);
    free(U);
    free(VT);
    free(superb);
}

int main() {
    // Example usage
    int nrow = 3;
    int ncol = 2;
    double R[] = {1, 2, 3, 4, 5, 6}; // 3x2 matrix
    double* R_pinv = (double*)malloc(sizeof(double) * ncol * nrow); // 2x3 matrix

    compute_pseudo_inverse(R, nrow, ncol, R_pinv);
    
    // Print the pseudo-inverse
    for (int i = 0; i < ncol; ++i) {
        for (int j = 0; j < nrow; ++j) {
            printf("%lf ", R_pinv[i * nrow + j]);
        }
        printf("\n");
    }
    
    free(R_pinv);

    return 0;
}