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

    while ((TinyDFT->iter < TinyDFT->max_iter) && (fabs(E_delta) >= 0.00000001))
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

    int nbf = h2eri->num_bf;
    double *productmat;
    productmat = (double*) malloc_aligned(sizeof(double) * nbf * nbf,    64);
    memset(productmat,  0, sizeof(double) * nbf * nbf);
    H2ERI_build_Coulomb(h2eri, TinyDFT->D_mat, productmat);
    double normj=Calc2norm(productmat,nbf);
    printf("Initial. The norm square of J is %f\n",normj);


    
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



    COOmat_p cooh2d;
    COOmat_init(&cooh2d,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    H2ERI_build_COO_fulldensetest(h2eri,cooh2d);
//    size_t nnz=cooh2d->nnz;
    printf("arggfdsadgf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("Now print COO H2D Matrix info--------\n");
    TestCOO(cooh2d);
    
    CSRmat_p csrh2d;
    CSRmat_init(&csrh2d,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);


    Double_COO_to_CSR( h2eri->num_bf*h2eri->num_bf,  cooh2d->nnz, cooh2d,csrh2d);
    printf("Now print CSR H2D Matrix info--------\n");
    TestCSR(csrh2d);
    

    COOmat_p cooexc;
    COOmat_init(&cooexc,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    H2ERI_build_COO_transpose(cooh2d, cooexc, nbf);
//    size_t nnz=cooh2d->nnz;

    printf("Now print COO EXC Matrix info--------\n");
    TestCOO(cooexc);
    
    CSRmat_p csrexc;
    CSRmat_init(&csrexc,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);


    Double_COO_to_CSR( h2eri->num_bf*h2eri->num_bf,  cooexc->nnz, cooexc,csrexc);
    printf("Now print CSR EXC Matrix info--------\n");
    TestCSR(csrexc);

    //Now we have finished the Construction of the dense part of the ERI tensor. The next step is to compute J=W*D
    //Then we compute D.*J and compare with the final Eone
    
    memset(productmat,  0, sizeof(double) * nbf * nbf);
    H2ERI_build_Coulomb(h2eri, TinyDFT->D_mat, productmat);
    normj=Calc2norm(productmat,nbf);
    printf("Initial. The norm square of J is %f\n",normj);
    double *diffmat;
    diffmat = (double*) malloc_aligned(sizeof(double) * nbf * nbf,    64);
    memset(diffmat,  0, sizeof(double) * nbf * nbf);
    memset(productmat,  0, sizeof(double) * nbf * nbf);
    int tmpcol;
    //Now compute J=W*D
    for(int j=0;j<csrh2d->nrow;j++)
    {
        if(csrh2d->csrrow[j]!=csrh2d->csrrow[j+1])
        {
            for(size_t i=csrh2d->csrrow[j];i<csrh2d->csrrow[j+1];i++)
            {
                tmpcol=csrh2d->csrcol[i];
                productmat[j]+=TinyDFT->D_mat[tmpcol]*csrh2d->csrval[i];
            }
        }
    }
    //Now theoretically productmat is the J matrix

    normj=Calc2norm(productmat,nbf);
    printf("The dense csr extracted matrix computed J norm square is %f\n",normj);

    normj=Calc2norm(TinyDFT->J_mat,nbf);
    printf("The exact norm (in TinyDFT->J_mat) %f\n",normj);

    for(int i=0;i<nbf*nbf;i++)
    {
        diffmat[i]=productmat[i]-TinyDFT->J_mat[i];
    }
    normj=Calc2norm(diffmat,nbf);
    printf("The 2 norm of the difference matrix is %f\n",normj);
    double energy=0;
    for(int i=0;i<nbf*nbf;i++)
    {
        energy+=productmat[i]*TinyDFT->D_mat[i];
    }
    printf("And the energy computed is %f\n",energy);

    //Now we compute the total norm of the Dmat in H2 form. This is to make sure that the extraction step is correct
    double eone=0;

    memset(productmat,  0, sizeof(double) * nbf * nbf);
    eone = 0;
    for(int i=0;i<nbf*nbf;i++)
    {
//        printf(" %f ", productmat[i]);
        eone += TinyDFT->J_mat[i]*TinyDFT->D_mat[i];
    }
    printf("\nIt should be %f\n",eone);

    printf("Now test K part\n");
    memset(productmat,  0, sizeof(double) * nbf * nbf);

    //Now compute J=W*D
    for(int j=0;j<csrexc->nrow;j++)
    {
        if(csrexc->csrrow[j]!=csrexc->csrrow[j+1])
        {
            for(size_t i=csrexc->csrrow[j];i<csrexc->csrrow[j+1];i++)
            {
                tmpcol=csrexc->csrcol[i];
                productmat[j]+=TinyDFT->D_mat[tmpcol]*csrexc->csrval[i];
            }
        }
    }
    //Now theoretically productmat is the K matrix

    normj=Calc2norm(productmat,nbf);
    printf("The dense csr extracted matrix computed K norm square is %f\n",normj);

    normj=Calc2norm(TinyDFT->K_mat,nbf);
    printf("The exact norm (in TinyDFT->K_mat) %f\n",normj);

    energy=0;
    for(int i=0;i<nbf*nbf;i++)
    {
        energy+=productmat[i]*TinyDFT->D_mat[i];
    }
    printf("And the energy computed is %f\n",energy);



    /*
    printf("Added tests \n");
    double *weighteddmat;
    weighteddmat = (double*) malloc_aligned(sizeof(double) * nbf * nbf,    64);
    memset(weighteddmat,  0, sizeof(double) * nbf * nbf);
    for(int i=0;i<nbf*nbf;i++)
        weighteddmat[i]=2*TinyDFT->D_mat[i];
    
    printf("1\n");

    for(int i=0;i<h2eri->nshell;i++)
    {
        for(int j=h2eri->shell_bf_sidx[i];j<h2eri->shell_bf_sidx[i+1];j++)
            for(int k=h2eri->shell_bf_sidx[i];k<h2eri->shell_bf_sidx[i+1];k++)
            {
//                printf("%d<%d\n",j*nbf+k,nbf*nbf);
                weighteddmat[j*nbf+k]/=2;
            }
    }
    printf("2\n");
    memset(productmat,  0, sizeof(double) * nbf * nbf);
    for(int j=0;j<csrh2d->nrow;j++)
    {
        if(csrh2d->csrrow[j]!=csrh2d->csrrow[j+1])
        {
            for(size_t i=csrh2d->csrrow[j];i<csrh2d->csrrow[j+1];i++)
            {
                tmpcol=csrh2d->csrcol[i];
                productmat[j]+=TinyDFT->D_mat[tmpcol]*csrh2d->csrval[i];
            }
        }
    }
    printf("3\n");
    for(int i=0;i<h2eri->nshell;i++)
    {
        for(int j=h2eri->shell_bf_sidx[i];j<h2eri->shell_bf_sidx[i+1];j++)
            for(int k=h2eri->shell_bf_sidx[i];k<h2eri->shell_bf_sidx[i+1];k++)
            {
                productmat[j*nbf+k]/=2;
            }
    }

    normj=Calc2norm(productmat,nbf);
    printf("The diagonal computed J norm square is %f\n",normj);
    for(int i=0;i<nbf*nbf;i++)
    {
        diffmat[i]=productmat[i]-TinyDFT->J_mat[i];
    }
    normj=Calc2norm(diffmat,nbf);
    printf("The norm square of difference matrix is %f\n",normj);
    */
   

    COOmat_destroy(cooh2d);
    CSRmat_destroy(csrh2d);

    // Free TinyDFT and H2P-ERI
    TinyDFT_destroy(&TinyDFT);
    H2ERI_destroy(h2eri);
    return 0;
}
