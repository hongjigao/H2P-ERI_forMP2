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
    H2ERI_build_COO_Diamattest(h2eri,cooh2d,1,0);
//    size_t nnz=cooh2d->nnz;
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("Now print COO H2D Matrix info--------\n");
    TestCOO(cooh2d);
    
    double thres1=1e-7;
    COOmat_p cooh2d1;
    COOmat_init(&cooh2d1,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    compresscoo(cooh2d, cooh2d1, thres1);
    TestCOO(cooh2d1);
    printf("The number of values above threshold is %lu \n", cooh2d1->nnz);
    
    CSRmat_p csrh2d;
    CSRmat_init(&csrh2d,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);


    Double_COO_to_CSR( h2eri->num_bf*h2eri->num_bf,  cooh2d1->nnz, cooh2d1,csrh2d);
    printf("Now print CSR H2D Matrix info--------\n");
    TestCSR(csrh2d);
    
    double *productmat;
    productmat = (double*) malloc_aligned(sizeof(double) * nbf * nbf,    64);
    memset(productmat,  0, sizeof(double) * nbf * nbf);
    double *diffmat;
    diffmat = (double*) malloc_aligned(sizeof(double) * nbf * nbf,    64);
    memset(diffmat,  0, sizeof(double) * nbf * nbf);
    int tmpcol;
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
    double normj=Calc2norm(productmat,nbf);
    printf("The diagonal computed J norm square is %f\n",normj);
    normj=Calc2norm(TinyDFT->J_mat,nbf);
    printf("It should be %f\n",normj);
    for(int i=0;i<nbf*nbf;i++)
    {
        diffmat[i]=productmat[i]-TinyDFT->J_mat[i];
    }
    normj=Calc2norm(diffmat,nbf);
    printf("The 2 norm of the difference is %f\n",normj);
    double eone=0;
    for(int i=0;i<nbf*nbf;i++)
    {
//        printf(" %f ", productmat[i]);
        eone += productmat[i]*TinyDFT->D_mat[i];
    }
    printf("\nThe diagonal eone is %f\n",eone);
    eone = 0;
    for(int i=0;i<nbf*nbf;i++)
    {
//        printf(" %f ", productmat[i]);
        eone += TinyDFT->J_mat[i]*TinyDFT->D_mat[i];
    }
    printf("\nIt should be %f\n",eone);
    H2E_dense_mat_p  *c_D_blks   = h2eri->c_D_blks;
    double normd=0;

    for(int i=0;i<h2eri->n_leaf_node;i++)
    {
        H2E_dense_mat_p Di = c_D_blks[i];
        for(int j=0;j<Di->nrow*Di->ncol;j++)
        {
            normd+=Di->data[j]*Di->data[j];
        }
    }
    for(int i=0;i<h2eri->n_r_inadm_pair;i++)
    {
        H2E_dense_mat_p Di = c_D_blks[i+h2eri->n_leaf_node];
        for(int j=0;j<Di->size;j++)
        {
            normd+=2*Di->data[j]*Di->data[j];
        }
    }
    printf("The norm of the dense part is %f\n",normd);
    memset(productmat,  0, sizeof(double) * nbf * nbf);
    H2ERI_build_Coulombtest(h2eri, TinyDFT->D_mat, productmat);
    for(int i=0;i<nbf*nbf;i++)
    {
        diffmat[i]=productmat[i]-TinyDFT->J_mat[i];
    }
    normj=Calc2norm(productmat,nbf);
    printf("The norm square of diagonal J is %f\n",normj);
    normj=Calc2norm(diffmat,nbf);
    printf("The norm square of difference matrix is %f\n",normj);
    eone = 0;
    for(int i=0;i<nbf*nbf;i++)
    {
//        printf(" %f ", productmat[i]);
        eone += productmat[i]*TinyDFT->D_mat[i];
    }
    printf("\n D part computation eone is %f\n",eone);
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
    /*
    printf("1st\n");
    for(int i=0; i<h2eri->num_sp_bfp; i++)
    {
        printf("%d ",h2eri->bf1st[i]);
    }
    printf("2nd %d\n",h2eri->num_sp_bfp);
        for(int i=0; i<h2eri->num_sp_bfp; i++)
    {
        printf("%d ",h2eri->bf1st[i]*nbf+h2eri->bf2nd[i]);
    }
    */
    
    
   /*
    TinyDFT_build_MP2info_eig(TinyDFT, TinyDFT->F_mat,
                               TinyDFT->X_mat, TinyDFT->D_mat,
                               TinyDFT->Cocc_mat, TinyDFT->DC_mat,
                               TinyDFT->Cvir_mat, TinyDFT->orbitenergy_array);
    double Fermie =0;
    double talpha=1;
    printf("talpha is %f\n", talpha);
    double st0,et0;
    st0 = get_wtime_sec();

    TinyDFT_build_energyweightedDDC(TinyDFT, TinyDFT->Cocc_mat,TinyDFT->Cvir_mat,TinyDFT->orbitenergy_array,TinyDFT->D_mat,TinyDFT->DC_mat,Fermie,talpha);
    et0 = get_wtime_sec();
    printf("D/DC building time is %.3lf (s)\n",et0-st0);
    double nm=Calc2norm(TinyDFT->D_mat,nbf);
    printf("The norm of Dmat is %f\n",nm);
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    COOmat_p cooden;
    COOmat_init(&cooden,h2eri->num_bf,h2eri->num_bf);
    double thres = 1e-4;
    size_t nden =Extract_COO_DDCMat(h2eri->num_bf, h2eri->num_bf, thres, TinyDFT->D_mat, cooden);
    printf("Now print COO Den Matrix info--------\n");
    TestCOO(cooden);
    printf("The total elements of D are %d and the rate of survival by threshold %e is %ld \n",h2eri->num_bf*h2eri->num_bf,thres,nden);
    printf("Now print CSR Den Matrix info--------\n");
    CSRmat_p csrden;
    CSRmat_init(&csrden,h2eri->num_bf,h2eri->num_bf);
    Double_COO_to_CSR( h2eri->num_bf,  nden, cooden,csrden);
    TestCSR(csrden);
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    nm=Calc2norm(TinyDFT->DC_mat,nbf);
    printf("The norm of DCmat is %f\n",nm);
    COOmat_p coodc;
    COOmat_init(&coodc,h2eri->num_bf,h2eri->num_bf);
    size_t ndc =Extract_COO_DDCMat(h2eri->num_bf, h2eri->num_bf, thres, TinyDFT->DC_mat, coodc);
    printf("Now print COO DC Matrix info--------\n");
    TestCOO(coodc);
    printf("The total elements of DC are %d and the rate of survival by threshold %e is %ld \n",h2eri->num_bf*h2eri->num_bf,thres,ndc);

    CSRmat_p csrdc;
    CSRmat_init(&csrdc,h2eri->num_bf,h2eri->num_bf);
    Double_COO_to_CSR( h2eri->num_bf,  ndc, coodc,csrdc);
    printf("Now print CSR DC Matrix info--------\n");
    TestCSR(csrdc);
    printf("Build energy weighted matrices success\n");
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("Now do index transformation\n");
    CSRmat_p gdle;
    CSRmat_init(&gdle,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    printf("test numbf is%d\n",h2eri->num_bf);
    double st1,et1;
    st1 = get_wtime_sec();
    Xindextransform1(h2eri->num_bf,csrh2d,csrden,gdle);
    et1 = get_wtime_sec();
    printf("The X Index transformation time is %.3lf (s)\n",et1-st1);
    TestCSR(gdle);
    printf("Xindex transformation finished\n");
    CSRmat_p gdls;
    CSRmat_init(&gdls,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    
    st1 = get_wtime_sec();
    Yindextransform1(h2eri->num_bf,gdle,csrdc,gdls);
    et1 = get_wtime_sec();
    
    TestCSR(gdls);
    printf("The Y Index transformation time is %.3lf (s)\n",et1-st1);
    
    printf("Now do energy calculation \n");
    st1 = get_wtime_sec();
//    double energy;
//    energy = Calc_S1energy(gdls);
//    printf("The energy is %f\n",energy);
    CSRmat_p colgdls;
    CSRmat_init(&colgdls,nbf*nbf,nbf*nbf);
    CSR_to_CSC(nbf*nbf, gdls,colgdls);
    TestCSR(colgdls);
    double energy;
    energy = Calc_S1energy(gdls,colgdls);
    printf("The energy is %f\n",energy);
    et1 = get_wtime_sec();
    printf("Energy computation time is %.3lf (s)\n",et1-st1);

    printf("Test whether csrh2d product Dmat equals Jmat\n");
    

    
    COOmat_destroy(cooden);
    CSRmat_destroy(csrden);
    
    COOmat_destroy(coodc);
    CSRmat_destroy(csrdc);


    
    CSRmat_destroy(gdls);
    CSRmat_destroy(gdle);
    CSRmat_destroy(colgdls);
    */

    COOmat_destroy(cooh2d1);
    CSRmat_destroy(csrh2d);

    // Free TinyDFT and H2P-ERI
    TinyDFT_destroy(&TinyDFT);
    H2ERI_destroy(h2eri);
    
    COOmat_destroy(cooh2d1);
    CSRmat_destroy(csrh2d);
    return 0;
}
