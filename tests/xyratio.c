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
    printf("The norm square of the COO matrix is %.16g\n",norm);
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
    //printf("The number of values larger than 1e-2,1e-5 and 1e-9 are respectively %lu,%lu,%lu\n",larger1e2,larger1e5,larger1e9);
    printf("The number of elements is %lu, ",lg0tst);
    printf("The norm of the csrmat is %.16g\n", norm);
    printf("The number of nonzero rows is %d, the totol rows is %d, the longest row is %d\n",nn0,csrmat->nrow,nlong);
    //printf("Test the ascending order:\n");
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
    //if(tests==0)
        //printf("Ascending order correct!\n");
    //else
    //    printf("Same value %d\n",tests);
}


double Calcmaxv(const double *mat, int siz) 
{
  double maxv = 0;
  for (int i = 0; i < siz; i++)
    for (int j = 0; j < siz; j++) 
    {
      if(fabs(mat[i * siz + j]) > maxv)
        maxv = fabs(mat[i * siz + j]);
    }

  return maxv;
}

char* format_double(double value) {
    char* result = malloc(10);  // Allocate memory for the result string
    if (result == NULL) {
        return NULL;  // Return NULL if memory allocation fails
    }

    if (value == 0) {
        sprintf(result, "0E0");
        return result;
    }

    int exponent = (int)floor(log10(fabs(value)));  // Find the exponent if value were expressed in scientific notation
    int most_significant_digit = (int)(value / pow(10, exponent));  // Extract the most significant digit

    sprintf(result, "%dE%d", most_significant_digit, exponent);  // Format the string as required

    return result;  // Return the formatted string
}

char* concatenate(const char* s1, const char* s2) {
    // Calculate the total length needed for the concatenated string
    int length = strlen(s1) + strlen(s2) + 1;  // +1 for the null terminator

    // Allocate memory for the concatenated string
    char* result = malloc(length);
    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // Copy the first string
    strcpy(result, s1);

    // Concatenate the second string
    strcat(result, s2);

    return result;
}



int main(int argc, char **argv)
{
    if (argc < 5)
    {
        printf("Usage: %s <basis> <xyz> <niter> <QR_tol>\n", argv[0]);
        return 255;
    }
    
    printf("INFO: use H2ERI J (relerr %.2e), HF exchange K\n", atof(argv[4]));
    double stt =get_wtime_sec();
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
    
    
    
    // Compute constant matrices and get initial guess for D
    TinyDFT_build_Hcore_S_X_mat(TinyDFT, TinyDFT->Hcore_mat, TinyDFT->S_mat, TinyDFT->X_mat);
    TinyDFT_build_Dmat_SAD(TinyDFT, TinyDFT->D_mat);
    
    // Do HFSCF calculation
    H2ERI_HFSCF(TinyDFT, h2eri, atoi(argv[3]));
    
    // Print H2P-ERI statistic info
    H2ERI_print_statistic(h2eri);
    int nbf = h2eri->num_bf;
    TinyDFT_build_MP2info_eig(TinyDFT, TinyDFT->F_mat,
                               TinyDFT->X_mat, TinyDFT->D_mat,
                               TinyDFT->Cocc_mat, TinyDFT->DC_mat,
                               TinyDFT->Cvir_mat, TinyDFT->orbitenergy_array);

    
    double Fermie=0.0;
    size_t mostx=0;
    size_t mosty=0;

    double maxx = Calcmaxv(TinyDFT->D_mat, nbf);
    double maxy = Calcmaxv(TinyDFT->DC_mat, nbf);
    printf("maxx and maxy are respectively %f and %f\n",maxx,maxy);
    int nxls[6]={0,0,0,0,0,0};
    int nyls[6]={0,0,0,0,0,0};
    for(int i=0;i<nbf*nbf;i++)
    {
        if(fabs(TinyDFT->D_mat[i])>maxx*1e-3)
            nxls[0]+=1;
        if(fabs(TinyDFT->D_mat[i])>maxx*3e-4)
            nxls[1]+=1;
        if(fabs(TinyDFT->D_mat[i])>maxx*1e-4)
            nxls[2]+=1;
        if(fabs(TinyDFT->D_mat[i])>maxx*3e-5)
            nxls[3]+=1;
        if(fabs(TinyDFT->D_mat[i])>maxx*1e-5)
            nxls[4]+=1;
        if(fabs(TinyDFT->D_mat[i])>maxx*3e-6)
            nxls[5]+=1;
        if(fabs(TinyDFT->DC_mat[i])>maxy*1e-3)
            nyls[0]+=1;
        if(fabs(TinyDFT->DC_mat[i])>maxy*3e-4)
            nyls[1]+=1;
        if(fabs(TinyDFT->DC_mat[i])>maxy*1e-4)
            nyls[2]+=1;
        if(fabs(TinyDFT->DC_mat[i])>maxy*3e-5)
            nyls[3]+=1;
        if(fabs(TinyDFT->DC_mat[i])>maxy*1e-5)
            nyls[4]+=1;
        if(fabs(TinyDFT->DC_mat[i])>maxy*3e-6)
            nyls[5]+=1;

    }

    int length = strlen(argv[1]) + 4; // 4 for ".txt" and null terminator
    char *outname = malloc(length);

    if (outname == NULL) {
        perror("Failed to allocate memory");
        return EXIT_FAILURE;
    }

    sprintf(outname, "%s.txt", argv[6]);
    double ett =get_wtime_sec();
    printf("The total time is %.3lf (s)\n", ett - stt);
    // Open the file
    FILE *fileou = fopen(outname, "w"); // Change "r" to "w" if you want to write to the file
    if (fileou == NULL) {
        perror("Failed to open file");
        free(outname);
        return EXIT_FAILURE;
    }
    fprintf(fileou, "nbf is %d\n",nbf);
    fprintf(fileou, "maxx and maxy are respectively %f and %f\n",maxx,maxy);
    fprintf(fileou, "The number of values larger than 1e-3 for D is %d\n",nxls[0]);
    fprintf(fileou, "The number of values larger than 3e-4 for D is %d\n",nxls[1]);
    fprintf(fileou, "The number of values larger than 1e-4 for D is %d\n",nxls[2]);
    fprintf(fileou, "The number of values larger than 3e-5 for D is %d\n",nxls[3]);
    fprintf(fileou, "The number of values larger than 1e-5 for D is %d\n",nxls[4]);
    fprintf(fileou, "The number of values larger than 3e-6 for D is %d\n",nxls[5]);
    fprintf(fileou, "The number of values larger than 1e-3 for DC is %d\n",nyls[0]);
    fprintf(fileou, "The number of values larger than 3e-4 for DC is %d\n",nyls[1]);
    fprintf(fileou, "The number of values larger than 1e-4 for DC is %d\n",nyls[2]);
    fprintf(fileou, "The number of values larger than 3e-5 for DC is %d\n",nyls[3]);
    fprintf(fileou, "The number of values larger than 1e-5 for DC is %d\n",nyls[4]);
    fprintf(fileou, "The number of values larger than 3e-6 for DC is %d\n",nyls[5]);
    fprintf(fileou, "The total time is %.3lf (s)\n", ett - stt);
    // Close the file
    printf("File '%s' opened successfully.\n", outname);

    // Close the file
    fclose(fileou);

    // Free the allocated memory for the filename
    free(outname);


    // Free TinyDFT and H2P-ERI
    TinyDFT_destroy(&TinyDFT);
    H2ERI_destroy(h2eri);
    
    return 0;
}
