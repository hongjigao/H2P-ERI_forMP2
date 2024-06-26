#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "TinyDFT.h"
#include "H2ERI.h"

void TinyDFT_copy_shells_to_H2ERI(TinyDFT_p TinyDFT, H2ERI_p h2eri)
{
    h2eri->natom  = TinyDFT->natom;
    h2eri->nshell = TinyDFT->nshell;
    h2eri->shells = (shell_t *) malloc(sizeof(shell_t) * h2eri->nshell);
    assert(h2eri->shells != NULL);
    simint_initialize_shells(h2eri->nshell, h2eri->shells);
    
    shell_t *src_shells = (shell_t*) TinyDFT->simint->shells;
    shell_t *dst_shells = h2eri->shells;
    for (int i = 0; i < h2eri->nshell; i++)
    {
        simint_allocate_shell(src_shells[i].nprim, &dst_shells[i]);
        simint_copy_shell(&src_shells[i], &dst_shells[i]);
    }
}

void H2ERI_HFSCF(TinyDFT_p TinyDFT, H2ERI_p h2eri, const int max_iter)
{
    // Start SCF iterations
    printf("HFSCF iteration started...\n");
    printf("Nuclear repulsion energy = %.10lf\n", TinyDFT->E_nuc_rep);
    TinyDFT->iter = 0;
    TinyDFT->max_iter = max_iter;
    double E_prev, E_curr, E_delta = 19241112.0;
    
    int    mat_size       = TinyDFT->mat_size;
    double *D_mat         = TinyDFT->D_mat;
    double *J_mat         = TinyDFT->J_mat;
    double *K_mat         = TinyDFT->K_mat;
    double *F_mat         = TinyDFT->F_mat;
    double *X_mat         = TinyDFT->X_mat;
    double *S_mat         = TinyDFT->S_mat;
    double *Hcore_mat     = TinyDFT->Hcore_mat;
    double *Cocc_mat      = TinyDFT->Cocc_mat;
    double *E_nuc_rep     = &TinyDFT->E_nuc_rep;
    double *E_one_elec    = &TinyDFT->E_one_elec;
    double *E_two_elec    = &TinyDFT->E_two_elec;
    double *E_HF_exchange = &TinyDFT->E_HF_exchange;

    while ((TinyDFT->iter < TinyDFT->max_iter) && (fabs(E_delta) >= TinyDFT->E_tol))
    {
        printf("--------------- Iteration %d ---------------\n", TinyDFT->iter);
        
        double st0, et0, st1, et1, st2;
        st0 = get_wtime_sec();
        
        // Build the Fock matrix
        st1 = get_wtime_sec();
        TinyDFT_build_JKmat(TinyDFT, D_mat, NULL, K_mat);
        st2 = get_wtime_sec();
        H2ERI_build_Coulomb(h2eri, D_mat, J_mat);
        #pragma omp parallel for simd
        for (int i = 0; i < mat_size; i++)
            F_mat[i] = Hcore_mat[i] + 2 * J_mat[i] - K_mat[i];
        et1 = get_wtime_sec();
 //       printf("* Build Fock matrix     : %.3lf (s), H2ERI J mat used %.3lf (s)\n", et1 - st1, et1 - st2);
        
        // Calculate new system energy
        st1 = get_wtime_sec();
        TinyDFT_calc_HF_energy(
            mat_size, D_mat, Hcore_mat, J_mat, K_mat, 
            E_one_elec, E_two_elec, E_HF_exchange
        );
        E_curr = (*E_nuc_rep) + (*E_one_elec) + (*E_two_elec) + (*E_HF_exchange);
        et1 = get_wtime_sec();
//        printf("* Calculate energy      : %.3lf (s)\n", et1 - st1);
        E_delta = E_curr - E_prev;
        E_prev = E_curr;
        
        // CDIIS acceleration (Pulay mixing)
        st1 = get_wtime_sec();
        TinyDFT_CDIIS(TinyDFT, X_mat, S_mat, D_mat, F_mat);
        et1 = get_wtime_sec();
//        printf("* CDIIS procedure       : %.3lf (s)\n", et1 - st1);
        
        // Diagonalize and build the density matrix
        st1 = get_wtime_sec();
        TinyDFT_build_Dmat_eig(TinyDFT, F_mat, X_mat, D_mat, Cocc_mat);
        et1 = get_wtime_sec(); 
 //       printf("* Build density matrix  : %.3lf (s)\n", et1 - st1);
        
        et0 = get_wtime_sec();
        
        printf("* Iteration runtime     = %.3lf (s)\n", et0 - st0);
        printf("* Energy = %.10lf", E_curr);
        if (TinyDFT->iter > 0) 
        {
            printf(", delta = %e\n", E_delta);
            printf("The Eoe, Ete and Eex are respectively %f, %f and %f\n",*E_one_elec,*E_two_elec,*E_HF_exchange); 
        } else {
            printf("\n");
            E_delta = 19241112.0;  // Prevent the SCF exit after 1st iteration when no SAD initial guess
        }
        
        TinyDFT->iter++;
        fflush(stdout);
    }
    printf("--------------- SCF iterations finished ---------------\n");
}

void TestCOO(COOmat_p coomat)
{
    double maxv=0;
    double norm=0;
    int larger1e5=0;
    int larger1e9=0;
    int larger1e2=0;
    int lg0tst=0;
//    printf("%d\n",lg0tst);
    for(size_t i=0;i<coomat->nnz;i++)
    {
        if(fabs(coomat->cooval[i])>maxv)
        {
            maxv=fabs(coomat->cooval[i]);
        }
    }

    printf("The max value is %e\n",maxv);


    for(size_t i=0;i<coomat->nnz;i++)
    {
        if(fabs(coomat->cooval[i])>maxv*1e-2)
            larger1e2+=1;
        if(fabs(coomat->cooval[i])>=maxv*1e-5)
            larger1e5+=1;
        if(fabs(coomat->cooval[i])>maxv*1e-9)
            larger1e9+=1;
        if(fabs(coomat->cooval[i])>0)
            lg0tst+=1;
        norm+=coomat->cooval[i]*coomat->cooval[i];

    }
    printf("The number of values larger than 1e-2,1e-5 and 1e-9 are respectively %d,%d,%d\n",larger1e2,larger1e5,larger1e9);
    printf("The number of elements is %d\n",lg0tst);
    printf("The norm square of the COO matrix is %f\n",norm);
}

void TestCSR(CSRmat_p csrmat)
{
    double maxv=0;
    double norm=0;
    size_t larger1e5=0;
    size_t larger1e9=0;
    size_t larger1e2=0;
    size_t lg0tst=0;
//    printf("%d\n",lg0tst);
    for(size_t i=0;i<csrmat->nnz;i++)
    {
        if(fabs(csrmat->csrval[i])>maxv)
        {
            maxv=fabs(csrmat->csrval[i]);
        }
    }

    printf("The max value is %e\n",maxv);
 
    for(size_t i=0;i<csrmat->nnz;i++)
    {
        if(fabs(csrmat->csrval[i])>maxv*1e-2)
            larger1e2+=1;
        if(fabs(csrmat->csrval[i])>=maxv*1e-5)
            larger1e5+=1;
        if(fabs(csrmat->csrval[i])>maxv*1e-9)
            larger1e9+=1;
        if(fabs(csrmat->csrval[i])>0)
            lg0tst+=1;
        norm += csrmat->csrval[i]*csrmat->csrval[i];

    }
    int nn0 = csrmat->nrow;
    int nlong = 0;
    for(int j=0;j<csrmat->nrow;j++)
    {
        if(csrmat->csrrow[j]==csrmat->csrrow[j+1])
            nn0 -= 1;
        else
        {
            if(csrmat->csrrow[j+1]-csrmat->csrrow[j]>nlong)
                nlong=csrmat->csrrow[j+1]-csrmat->csrrow[j];
        }
    }
    printf("The number of values larger than 1e-2,1e-5 and 1e-9 are respectively %lu,%lu,%lu\n",larger1e2,larger1e5,larger1e9);
    printf("The number of elements is %lu\n",lg0tst);
    printf("The norm of the csrmat is %f\n", norm);
    printf("The number of nonzero rows is %d, the totol rows is %d, the longest row is %d\n",nn0,csrmat->nrow,nlong);
    printf("Test the ascending order:\n");
    int tests=0;
    for(int i=0;i<csrmat->nrow;i++)
    {
        if(csrmat->csrrow[i]<csrmat->csrrow[i+1]-1)
        {
            for(size_t j=csrmat->csrrow[i];j<csrmat->csrrow[i+1]-1;j++)
            {
                if(csrmat->csrcol[j]>csrmat->csrcol[j+1])
                {
                    printf("Ascending order wrong!\n");
                    tests=1;
                    return;
                }
                if(csrmat->csrcol[j]==csrmat->csrcol[j+1])
                {
//                    printf("equal wrong\n");
                    tests+=1;
                }
            }
        }
    }
    if(tests==0)
        printf("Ascending order correct!\n");
    else
        printf("Same value %d\n",tests);
}


double Calc2norm(const double *mat, int siz) 
{
  double norms = 0;
  for (int i = 0; i < siz; i++)
    for (int j = 0; j < siz; j++) 
    {
      norms = norms + mat[i * siz + j] * mat[i * siz + j];
    }

  return norms;
}

void compute_pseudo_inverse1(double* R, int nrow, int ncol, double* R_pinv) {
    int lda = ncol;
    int ldu = nrow;
    int ldvt = ncol;
    int info;

    // Allocate memory for the decomposition
    double* S = (double*)malloc(sizeof(double) * ncol);
    double* U6 = (double*)malloc(sizeof(double) * nrow * nrow);
    double* VT = (double*)malloc(sizeof(double) * ncol * ncol);
    double* superb = (double*)malloc(sizeof(double) * (ncol - 1));

    // Compute the SVD of R
    info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', nrow, ncol, R, lda, S, U6, ldu, VT, ldvt, superb);
    printf("The singular values are:\n");
    for (int i = 0; i < ncol; ++i) {
        printf("%lf ", S[i]);
    }
    printf("U\n");
    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < nrow; ++j) {
            printf("%lf ", U6[i * nrow + j]);
        }
        printf("\n");
    }
    printf("VT\n");
    for (int i = 0; i < ncol; ++i) {
        for (int j = 0; j < ncol; ++j) {
            printf("%lf ", VT[i * ncol + j]);
        }
        printf("\n");
    }
    if (info > 0) {
        printf("The algorithm computing SVD failed to converge.\n");
        exit(1);
    }
        int rank = ncol;  // Assuming rank is full unless proven otherwise
    for (int i = 0; i < rank; ++i) {
        printf("!%lf\n",S[i]);
        if (S[i] <= 1e-10) {  // Adjust rank if singular values are too small
            rank = i;
            break;
        }
    }

    // Initialize R_pinv to zero
    for (int i = 0; i < ncol * nrow; ++i) {
        R_pinv[i] = 0;
    }

    // Compute pseudo-inverse R_pinv = V * Sigma^+ * U^T
    for (int i = 0; i < ncol; ++i) {      // over columns of V and R_pinv
        for (int j = 0; j < nrow; ++j) {  // over rows of U and R_pinv
            for (int k = 0; k < rank; ++k) {  // sum over the singular values
                R_pinv[j + i * nrow] += VT[i + k * ncol] * (1.0 / S[k]) * U6[k + j * nrow];
            }
        }
    }

    // Free allocated memory
    free(S);
    free(U6);
    free(VT);
    free(superb);
}







int main(int argc, char **argv)
{
    if (argc < 5)
    {
        printf("Usage: %s <basis> <xyz> <niter> <QR_tol>\n", argv[0]);
        return 255;
    }
    
    printf("INFO: use H2ERI J (relerr %.2e), HF exchange K\n", atof(argv[4]));

    // Initialize TinyDFT
    TinyDFT_p TinyDFT;
    TinyDFT_init(&TinyDFT, argv[1], argv[2]);
    
    // Initialize H2P-ERI
    double st = get_wtime_sec();
    H2ERI_p h2eri;
    H2ERI_init(&h2eri, 1e-10, 1e-10, atof(argv[4]));
    TinyDFT_copy_shells_to_H2ERI(TinyDFT, h2eri);
    //printf("H2ERI initialization done, used %.3lf (s)\n", get_wtime_sec() - st);
    H2ERI_process_shells(h2eri);
    H2ERI_partition(h2eri);
    //printf("H2ERI partition done\n");
    H2ERI_build_H2(h2eri, 0);
    double et = get_wtime_sec();
    printf("H2ERI build H2 for J matrix done, used %.3lf (s)\n", et - st);
    
    // Compute constant matrices and get initial guess for D
    TinyDFT_build_Hcore_S_X_mat(TinyDFT, TinyDFT->Hcore_mat, TinyDFT->S_mat, TinyDFT->X_mat);
    TinyDFT_build_Dmat_SAD(TinyDFT, TinyDFT->D_mat);
    
    // Do HFSCF calculation
    H2ERI_HFSCF(TinyDFT, h2eri, atoi(argv[3]));
    
    // Print H2P-ERI statistic info
    H2ERI_print_statistic(h2eri);
    int nbf = h2eri->num_bf;
    int * tmpshellidx;
    tmpshellidx=(int *) malloc(sizeof(int) * (h2eri->nshell+1));
    for(int j=0;j<h2eri->nshell;j++)
    {
        tmpshellidx[j]=h2eri->shell_bf_sidx[j];
    }
    tmpshellidx[h2eri->nshell]=h2eri->num_bf;
    h2eri->shell_bf_sidx=(int *) malloc(sizeof(int) * (h2eri->nshell+1));
    memset(h2eri->shell_bf_sidx, 0, sizeof(int) * (h2eri->nshell+1));
    for(int j=0;j<h2eri->nshell+1;j++)
    {
        h2eri->shell_bf_sidx[j]=tmpshellidx[j];
    }

    free(tmpshellidx);

    printf("The number of basis functions is %d\n",nbf);
    printf("1The number of nodes is %d\n",h2eri->n_node);
    // Now we need to build the row basis set for every node
    H2E_dense_mat_p *Urbasis;
    Urbasis = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * h2eri->n_node);
    H2ERI_build_rowbs(h2eri,Urbasis);
    printf("2The number of nodes is %d\n",h2eri->n_node);
    for(int i=0;i<h2eri->n_node;i++)
    {
        printf("The %dth node has %d rows and %d columns\n",i,h2eri->U[i]->nrow,h2eri->U[i]->ncol);
    }
    for(int i=0;i<h2eri->n_node;i++)
    {
        if(Urbasis[i]!=NULL){
            printf("The %dth node has %d rows and %d columns\n",i,Urbasis[i]->nrow,Urbasis[i]->ncol);
        }
    }
    for (int i = 0; i < h2eri->n_r_adm_pair; i++)
    {
        int node0  = h2eri->r_adm_pairs[2 * i];
        int node1  = h2eri->r_adm_pairs[2 * i + 1];
        int level0 = h2eri->node_level[node0];
        int level1 = h2eri->node_level[node1];
        printf("The %dth adm pair is (%d,%d) with level (%d,%d)\n",i,node0,node1,level0,level1);
        printf("its B matrix has %d rows and %d columns\n",h2eri->c_B_blks[i]->nrow,h2eri->c_B_blks[i]->ncol);
    }
    
    int *admpair1st;
    admpair1st=(int *) malloc(sizeof(int) * 2 * h2eri->n_r_adm_pair);
    int *admpair2nd;
    admpair2nd=(int *) malloc(sizeof(int) * 2 * h2eri->n_r_adm_pair);
    H2E_int_vec_p *nodeadmpairs;
    nodeadmpairs = (H2E_int_vec_p *) malloc(sizeof(H2E_int_vec_p) * h2eri->n_node);
    for(int i=0;i<h2eri->n_node;i++)
    {
        H2E_int_vec_init(&nodeadmpairs[i],10);
    }
    for(int i=0;i<h2eri->n_r_adm_pair;i++)
    {
        admpair1st[i]=h2eri->r_adm_pairs[2*i];
        admpair2nd[i]=h2eri->r_adm_pairs[2*i+1];
        H2E_int_vec_push_back(nodeadmpairs[admpair1st[i]],admpair2nd[i]);
        H2E_int_vec_push_back(nodeadmpairs[admpair2nd[i]],admpair1st[i]);
    }
    for(int i=0;i<h2eri->n_r_adm_pair;i++)
    {
        admpair1st[i+h2eri->n_r_adm_pair]=h2eri->r_adm_pairs[2*i+1];
        admpair2nd[i+h2eri->n_r_adm_pair]=h2eri->r_adm_pairs[2*i];
    }
    // Now we need to build the column basis set for every admissible pair
    H2E_dense_mat_p *Ucbasis;
    Ucbasis = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * h2eri->n_r_adm_pair*2);
    H2ERI_build_colbs(h2eri,Ucbasis,admpair1st,admpair2nd,Urbasis);
    /*
    for(int i=0;i<2*h2eri->n_r_adm_pair;i++)
    {
        if(Ucbasis[i]!=NULL)
        {
        //    printf("In the admissible pair %d and %d ",admpair1st[i],admpair2nd[i]);
         //   printf("The %dth Ucbasis has %d rows and %d columns\n",i,Ucbasis[i]->nrow,Ucbasis[i]->ncol);
        }
    }
    */
    h2eri->leafidx= (int *) malloc(sizeof(int) * h2eri->num_bf*h2eri->num_bf);
    memset(h2eri->leafidx, -1, sizeof(int) * h2eri->num_bf*h2eri->num_bf);
    h2eri->bfpidx= (int *) malloc(sizeof(int) * h2eri->num_bf*h2eri->num_bf);
    memset(h2eri->bfpidx, -1, sizeof(int) * h2eri->num_bf*h2eri->num_bf);
    
    for(int i = 0; i < h2eri->n_leaf_node; i++)
    {
        int node = h2eri->height_nodes[i];
        printf("%d \n",node);
        int startp = h2eri->mat_cluster[2 * node];
        int endp = h2eri->mat_cluster[2 * node + 1];
        for(int j = startp; j <= endp; j++)
        {
            int bf1st = h2eri->bf1st[j];
            int bf2nd = h2eri->bf2nd[j];
            h2eri->leafidx[bf1st*h2eri->num_bf+bf2nd] = node;
            h2eri->leafidx[bf2nd*h2eri->num_bf+bf1st] = node;
            if(h2eri->sameshell[j]==1)
            {
                h2eri->bfpidx[bf1st*h2eri->num_bf+bf2nd] = j;
            }
            else if(h2eri->sameshell[j]==0)
            {
                h2eri->bfpidx[bf1st*h2eri->num_bf+bf2nd] = j;
                h2eri->bfpidx[bf2nd*h2eri->num_bf+bf1st] = j;
            }
        }
    }
    int npairs = 2*h2eri->n_r_adm_pair+2*h2eri->n_r_inadm_pair+h2eri->n_leaf_node;
    printf("The number of pairs is %d\n",npairs);
    int *pair1st;
    pair1st=(int *) malloc(sizeof(int) * npairs);
    int *pair2nd;
    pair2nd=(int *) malloc(sizeof(int) * npairs);
    for(int i=0;i<2*h2eri->n_r_adm_pair;i++)
    {
        pair1st[i]=admpair1st[i];
        pair2nd[i]=admpair2nd[i];
    }
    for(int i=0;i<h2eri->n_r_inadm_pair;i++)
    {
        pair1st[i+2*h2eri->n_r_adm_pair]=h2eri->r_inadm_pairs[2*i];
        pair2nd[i+2*h2eri->n_r_adm_pair]=h2eri->r_inadm_pairs[2*i+1];
        pair1st[i+2*h2eri->n_r_adm_pair+h2eri->n_r_inadm_pair]=h2eri->r_inadm_pairs[2*i+1];
        pair2nd[i+2*h2eri->n_r_adm_pair+h2eri->n_r_inadm_pair]=h2eri->r_inadm_pairs[2*i];
    }
    for(int i=0;i<h2eri->n_leaf_node;i++)
    {
        pair1st[i+2*h2eri->n_r_adm_pair+2*h2eri->n_r_inadm_pair]=h2eri->height_nodes[i];
        pair2nd[i+2*h2eri->n_r_adm_pair+2*h2eri->n_r_inadm_pair]=h2eri->height_nodes[i];
    }
    H2E_int_vec_p *nodepairs;
    nodepairs = (H2E_int_vec_p *) malloc(sizeof(H2E_int_vec_p) * h2eri->n_node);
    
    for(int i=0;i<h2eri->n_node;i++)
    {
        H2E_int_vec_init(&nodepairs[i],10);
    }
    for(int i=0;i<npairs;i++)
    {
        H2E_int_vec_push_back(nodepairs[pair1st[i]],pair2nd[i]);
        H2E_int_vec_push_back(nodepairs[pair2nd[i]],pair1st[i]);
    }
    int sum=0;
    for(int i=0;i<h2eri->n_node;i++)
    {
        printf("The %dth node has %d pairs\n",i,nodepairs[i]->length);
        sum+=nodepairs[i]->length;
    }
    printf("The total number of pairs is %d\n",sum);
    

    TinyDFT_build_MP2info_eig(TinyDFT, TinyDFT->F_mat,
                               TinyDFT->X_mat, TinyDFT->D_mat,
                               TinyDFT->Cocc_mat, TinyDFT->DC_mat,
                               TinyDFT->Cvir_mat, TinyDFT->orbitenergy_array);
    double talpha=0.0;
    double Fermie=0.0;
    TinyDFT_build_energyweightedDDC(TinyDFT, TinyDFT->Cocc_mat,TinyDFT->Cvir_mat,TinyDFT->orbitenergy_array,TinyDFT->D_mat,TinyDFT->DC_mat,Fermie,talpha);
    CSRmat_p csrd5;
    CSRmat_init(&csrd5, nbf, nbf);
    CSRmat_p csrdc5;
    CSRmat_init(&csrdc5, nbf, nbf);
    H2ERI_extract_near_large_elements(h2eri, TinyDFT, csrd5, csrdc5, 5, 1e-5);

    // Now we compute the pseudo inverse of the Unodes for every nonleaf node
    H2E_dense_mat_p *Upinv;
    Upinv = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * h2eri->n_node);


    // Now we need to build the column basis set for every node pair including the inadmissible and self
    //H2E_dense_mat_p *S51cbasis;
    //S51cbasis = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * npairs);

    int nrow = 3;
    int ncol = 2;
    double R[] = {1, 2, 3, 4, 5, 6}; // 3x2 matrix
    double R1[] = {1, 2, 3, 4, 5, 6};
    double* R_pinv = (double*)malloc(sizeof(double) * ncol * nrow); // 2x3 matrix
    compute_pseudo_inverse(R, nrow, ncol, R_pinv);
    
    // Print the pseudo-inverse
    for (int i = 0; i < ncol; ++i) {
        for (int j = 0; j < nrow; ++j) {
            printf("%lf ", R_pinv[i * nrow + j]);
        }
        printf("\n");
    }
    double C[2*2] = {0,0,0,0};
    CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                2, 2, 3, 1.0,
                R_pinv, 3, R1, 2,
                0.0, C, 2);
    
    free(R_pinv);
     printf("Resulting Matrix C:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%lf ", C[i*2 + j]);
        }
        printf("\n");
    }

    FILE *file1 = fopen("U2.txt", "w");
    if (file1 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < h2eri->U[2]->nrow; i++) 
    {
        for(int j=0; j<h2eri->U[2]->ncol; j++)
        {
            fprintf(file1, "%.16g ", h2eri->U[2]->data[i*h2eri->U[2]->ncol+j]);
            
        }
        fprintf(file1, "\n");
    }
    fclose(file1);

    H2E_dense_mat_p *U  = h2eri->U;
    H2E_dense_mat_p tmpr;
    H2E_dense_mat_init(&tmpr, 500, 500);
    H2E_dense_mat_p tmprinv;
    H2E_dense_mat_init(&tmprinv, 500, 500);
    for(int i=0;i<h2eri->n_node;i++)
    {
        printf("%d\n",i);
        if(U[i]->nrow==0||h2eri->n_child[i]==0)
        {
            H2E_dense_mat_init(&Upinv[i], 1, 1);
            Upinv[i]->nrow = 0;
            Upinv[i]->ncol = 0;
            Upinv[i]->ld   = 0;
        }
        else
        {
            printf("%d\n",i);
            int nrow = U[i]->nrow;
            int ncol = U[i]->ncol;
            printf("%d, %d\n", nrow, ncol);
            H2E_dense_mat_resize(tmpr, nrow, ncol);
            for(int j=0;j<nrow;j++)
                for(int k=0;k<ncol;k++)
                    tmpr->data[j*ncol+k]=U[i]->data[j*ncol+k];
            H2E_dense_mat_resize(tmprinv, ncol, nrow);
            memset(tmprinv->data, 0, sizeof(DTYPE) * ncol * nrow);
            compute_pseudo_inverse(tmpr->data, nrow, ncol, tmprinv->data);
            H2E_dense_mat_init(&Upinv[i], ncol, nrow);
            for(int j=0;j<ncol;j++)
                for(int k=0;k<nrow;k++)
                    Upinv[i]->data[j*nrow+k]=tmprinv->data[j*nrow+k];
            memset(tmpr->data, 0, sizeof(DTYPE) * nrow * ncol);
            memset(tmprinv->data, 0, sizeof(DTYPE) * ncol * nrow);
                       
        }
    }

    for(int i=0;i<h2eri->n_node;i++)
    {
        printf(".%d\n",i);
        if(U[i]->nrow==0||h2eri->n_child[i]==0)
        {
            continue;
        }
        else
        {
            printf("%d\n",i);
            int nrow = U[i]->nrow;
            int ncol = U[i]->ncol;
            H2E_dense_mat_resize(tmpr, ncol, ncol);
            printf("%d, %d\n", nrow, ncol);
            CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ncol, ncol, nrow, 1.0,
                Upinv[i]->data, nrow, U[i]->data, ncol,
                0.0, tmpr->data, ncol);
            printf("The %dth node is orthogonal\n",i);
            for(int j=0;j<ncol;j++)
                for(int k=0;k<ncol;k++)
                {
                    if(j==k)
                    {
                        if(tmpr->data[j*ncol+k]-1>1e-9)
                            printf("The %dth node is not orthogonal\n",i);
                        else
                            printf("The %dth node is orthogonal\n",i);
                    }
                    else
                    {
                        if(tmpr->data[j*ncol+k]>1e-9)
                            printf("The %dth node is not orthogonal\n",i);
                        //else
                            //printf("The %dth node is orthogonal\n",i);
                    
                    }
                }
            memset(tmpr->data, 0, sizeof(DTYPE) * ncol * ncol);

                       
        }
    }





    for(int i=0;i<h2eri->n_r_adm_pair;i++)
    {
        if(Ucbasis[i]!=NULL)
        {
            H2E_dense_mat_destroy(&Ucbasis[i]);
        }
    }


    // Free memory

    free(admpair1st);
    free(admpair2nd);
    for(int i=0;i<h2eri->n_node;i++)
    {
        if(Urbasis[i]!=NULL)
        {
            H2E_dense_mat_destroy(&Urbasis[i]);
        }
    }


    // Free TinyDFT and H2P-ERI
    TinyDFT_destroy(&TinyDFT);
    H2ERI_destroy(h2eri);
    
    return 0;
}
