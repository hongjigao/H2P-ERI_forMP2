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

    while ((TinyDFT->iter < TinyDFT->max_iter) && (fabs(E_delta) >= 1e-1))
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
        printf("* Calculate energy      : %.3lf (s)\n", et1 - st1);
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
        
 //       printf("* Iteration runtime     = %.3lf (s)\n", et0 - st0);
        printf("* Energy = %.10lf", E_curr);
        if (TinyDFT->iter > 0) 
        {
            printf(", delta = %e\n", E_delta); 
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

    }
    printf("The number of values larger than 1e-2,1e-5 and 1e-9 are respectively %d,%d,%d\n",larger1e2,larger1e5,larger1e9);
    printf("The number of elements is %d\n",lg0tst);
}

void TestCSR(CSRmat_p csrmat)
{
    double maxv=0;
    int larger1e5=0;
    int larger1e9=0;
    int larger1e2=0;
    int lg0tst=0;
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

    }
    printf("The number of values larger than 1e-2,1e-5 and 1e-9 are respectively %d,%d,%d\n",larger1e2,larger1e5,larger1e9);
    printf("The number of elements is %d\n",lg0tst);
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
                    printf("equal wrong\n");
                    tests=1;
                }
            }
        }
    }
    if(tests==0)
        printf("Ascending order correct!\n");
}


int count_unique(H2ERI_p h2eri, int *array, int length) {
    int count = 0;
    int* hashTable;  // Initialize hash table with 0
    hashTable = (int*) malloc(sizeof(int) * h2eri->num_bf);
    memset(hashTable, 0, sizeof(int) * h2eri->num_bf);
    for (int i = 0; i < length; i++) {
        if (!hashTable[array[i]]) {  // If the integer is not yet encountered
            hashTable[array[i]] = 1; // Mark it as encountered
            count++;                // Increment unique count
        }
    }

    return count;
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
    H2ERI_process_shells(h2eri);
    H2ERI_partition(h2eri);
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

    int *tmpptr1 = NULL;
    int *tmpptr2 = NULL;
    int length=0;
    for(int i=0;i<h2eri->n_node;i++)
    {
        tmpptr1=h2eri->bf1st+h2eri->mat_cluster[2*i];
        tmpptr2=h2eri->bf2nd+h2eri->mat_cluster[2*i];
        length=h2eri->mat_cluster[2*i+1]-h2eri->mat_cluster[2*i];
        int count1=count_unique(h2eri,tmpptr1,length);
        int count2=count_unique(h2eri,tmpptr2,length);
        printf("In the %d node of height %d, number of BFP is %d, number of unique 1st bf is%d, 2nd bf is %d\n",i,h2eri->node_height[i],length,count1,count2);

    }

    COOmat_p cooh2d;
    COOmat_init(&cooh2d,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    int numbfp=h2eri->num_sp_bfp;
    cooh2d->nnz=numbfp*numbfp;
    cooh2d->coorow = (int*) malloc(sizeof(int) * (cooh2d->nnz));
    cooh2d->coocol = (int*) malloc(sizeof(int) * (cooh2d->nnz));
    cooh2d->cooval = (double*) malloc(sizeof(double) * (cooh2d->nnz));
    ASSERT_PRINTF(cooh2d->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(cooh2d->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(cooh2d->cooval != NULL, "Failed to allocate arrays for D matrices indexing\n");

    int num=0;
    for(int i=0;i<numbfp;i++)
    {
        double* testdmat;
        double* testjmat;
        testdmat=(double*) malloc(sizeof(double) * numbfp);
        testjmat=(double*) malloc(sizeof(double) * numbfp);
        memset(testdmat, 0, sizeof(double) * numbfp);
        memset(testjmat, 0, sizeof(double) * numbfp);
        testdmat[i]=1;
        H2ERI_matvec(h2eri,testdmat, testjmat);

        for(int j=0;j<numbfp;j++)
        {
            size_t ptr=j*numbfp+i;
            cooh2d->coorow[ptr]=j;
            cooh2d->coocol[ptr]=i;
            cooh2d->cooval[ptr]=testjmat[j];
        }
//        printf("%d th column finished\n",i);
//        num+=1;
        free(testdmat);
        free(testjmat);
    }

    printf("Test COOH2D whole ERI\n");
    TestCOO(cooh2d);
    COOmat_destroy(cooh2d);


    COOmat_p cooh2d1;
    COOmat_init(&cooh2d1,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    cooh2d1->nnz=numbfp*numbfp;
    cooh2d1->coorow = (int*) malloc(sizeof(int) * (cooh2d1->nnz));
    cooh2d1->coocol = (int*) malloc(sizeof(int) * (cooh2d1->nnz));
    cooh2d1->cooval = (double*) malloc(sizeof(double) * (cooh2d1->nnz));
    ASSERT_PRINTF(cooh2d1->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(cooh2d1->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(cooh2d1->cooval != NULL, "Failed to allocate arrays for D matrices indexing\n");

    for(int i=0;i<numbfp;i++)
    {
        double* testdmat;
        double* testjmat;
        testdmat=(double*) malloc(sizeof(double) * numbfp);
        testjmat=(double*) malloc(sizeof(double) * numbfp);
        memset(testdmat, 0, sizeof(double) * numbfp);
        memset(testjmat, 0, sizeof(double) * numbfp);
        testdmat[i]=1;
        H2ERI_matvectest(h2eri,testdmat, testjmat);

        for(int j=0;j<numbfp;j++)
        {
            size_t ptr=j*numbfp+i;
            cooh2d1->coorow[ptr]=j;
            cooh2d1->coocol[ptr]=i;
            cooh2d1->cooval[ptr]=testjmat[j];
        }
//        printf("%d th column finished\n",i);
        num+=1;
        free(testdmat);
        free(testjmat);
    }




    
    printf("Test COOH2D1 dense ERI\n");
    TestCOO(cooh2d1);


    COOmat_destroy(cooh2d1);
   
    COOmat_p cooh2d2;
    COOmat_init(&cooh2d2,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    cooh2d2->nnz=numbfp*numbfp;
    cooh2d2->coorow = (int*) malloc(sizeof(int) * (cooh2d2->nnz));
    cooh2d2->coocol = (int*) malloc(sizeof(int) * (cooh2d2->nnz));
    cooh2d2->cooval = (double*) malloc(sizeof(double) * (cooh2d2->nnz));
    ASSERT_PRINTF(cooh2d2->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(cooh2d2->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(cooh2d2->cooval != NULL, "Failed to allocate arrays for D matrices indexing\n");

    for(int i=0;i<numbfp;i++)
    {
        double* testdmat;
        double* testjmat;
        testdmat=(double*) malloc(sizeof(double) * numbfp);
        testjmat=(double*) malloc(sizeof(double) * numbfp);
        memset(testdmat, 0, sizeof(double) * numbfp);
        memset(testjmat, 0, sizeof(double) * numbfp);
        testdmat[i]=1;
        H2ERI_matvectest2(h2eri,testdmat, testjmat);

        for(int j=0;j<numbfp;j++)
        {
            size_t ptr=j*numbfp+i;
            cooh2d2->coorow[ptr]=j;
            cooh2d2->coocol[ptr]=i;
            cooh2d2->cooval[ptr]=testjmat[j];
        }
//        printf("%d th column finished\n",i);
        num+=1;
        free(testdmat);
        free(testjmat);
    }

    printf("Test COOH2D2 offdiagonal ERI\n");
    TestCOO(cooh2d1);
    COOmat_destroy(cooh2d2);
    printf("3*numbfp:%d\n",num);


    // Free TinyDFT and H2P-ERI
    TinyDFT_destroy(&TinyDFT);
    H2ERI_destroy(h2eri);
    
    return 0;
}
