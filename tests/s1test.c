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
    printf("The number of values larger than 1e-2,1e-5 and 1e-9 are respectively %lu,%lu,%lu\n",larger1e2,larger1e5,larger1e9);
    printf("The number of elements is %lu\n",lg0tst);
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
    
    
    
    TinyDFT_build_MP2info_eig(TinyDFT, TinyDFT->F_mat,
                               TinyDFT->X_mat, TinyDFT->D_mat,
                               TinyDFT->Cocc_mat, TinyDFT->DC_mat,
                               TinyDFT->Cvir_mat, TinyDFT->orbitenergy_array);
    double talpha=0.0;
    double Fermie=0.0;
    TinyDFT_build_energyweightedDDC(TinyDFT, TinyDFT->Cocc_mat,TinyDFT->Cvir_mat,TinyDFT->orbitenergy_array,TinyDFT->D_mat,TinyDFT->DC_mat,Fermie,talpha);
    double norm=0;
    for(int i=0;i<nbf*nbf;i++)
    {
        norm+=TinyDFT->D_mat[i]*TinyDFT->D_mat[i];
    }
    printf("The norm of the D matrix is %f\n",norm);
    
    double thresxy = atof(argv[6]);
    double threstsfm = atof(argv[7]);
    printf("Now build cooden\n");
    COOmat_p cooden;
    COOmat_init(&cooden,h2eri->num_bf,h2eri->num_bf);
    size_t nden =Extract_COO_DDCMat(h2eri->num_bf, h2eri->num_bf, thresxy, TinyDFT->D_mat, cooden);
    printf("Now build csrden\n");
    CSRmat_p csrden;
    CSRmat_init(&csrden,h2eri->num_bf,h2eri->num_bf);
    Double_COO_to_CSR( h2eri->num_bf,  nden, cooden,csrden);
    TestCSR(csrden);
    printf("Now build coodc\n");
    COOmat_p coodc;
    COOmat_init(&coodc,h2eri->num_bf,h2eri->num_bf);
    size_t ndc =Extract_COO_DDCMat(h2eri->num_bf, h2eri->num_bf, thresxy, TinyDFT->DC_mat, coodc);
    printf("Now build csrdc\n");
    CSRmat_p csrdc;
    CSRmat_init(&csrdc,h2eri->num_bf,h2eri->num_bf);
    Double_COO_to_CSR( h2eri->num_bf,  ndc, coodc,csrdc);
    TestCSR(csrdc);
    printf("Now build gdle\n");


    COOmat_p cooh2d;
    COOmat_init(&cooh2d,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    H2ERI_build_COO_fulldensetest(h2eri,cooh2d);
    size_t nnz=cooh2d->nnz;
    
    
    
    for(int i=0;i<1;i++)
    {
        double thres = atof(argv[5]);
        printf("The threshold is %e\n",thres);
        COOmat_p cooh2d1;
        COOmat_init(&cooh2d1,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
        compresscoo(cooh2d, cooh2d1, thres);

        CSRmat_p csrh2d;
        CSRmat_init(&csrh2d,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);

        Double_COO_to_CSR( h2eri->num_bf*h2eri->num_bf,  cooh2d1->nnz, cooh2d1,csrh2d);
        printf("TestCSRh2d\n");
        TestCSR(csrh2d);
        CSRmat_p gdle;
        CSRmat_init(&gdle,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
        double st1,et1;
        st1 = get_wtime_sec();
        Xindextransform3(h2eri->num_bf,csrh2d,csrden,gdle);
        printf("!!!!!!!!!!!testgdle");
        TestCSR(gdle);
        CSRmat_p gdle1;
        CSRmat_init(&gdle1,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
        compresscsr(gdle,gdle1,thres);
        printf("!!!!!!!!!!!testgdle1\n");
        TestCSR(gdle1);
        CSRmat_p gdle2;
        CSRmat_init(&gdle2,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
        Xindextransform4(h2eri->num_bf,csrh2d,csrden,gdle2,threstsfm);
        printf("!!!!!!!!!!!testgdle2\n");
        TestCSR(gdle2);

        printf("Now build gdls\n");
        et1 = get_wtime_sec();
        CSRmat_p gdls;
        CSRmat_init(&gdls,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
            
        st1 = get_wtime_sec();
        Yindextransform3(h2eri->num_bf,gdle,csrdc,gdls);
        et1 = get_wtime_sec();
            
        printf("The Y Index transformation time is %.3lf (s)\n",et1-st1);
        printf("@@@@@@@@@@GDLS\n");
        TestCSR(gdls);
        printf("Now do energy calculation \n");
        st1 = get_wtime_sec();
        //    double energy;
        //    energy = Calc_S1energy(gdls);
        //    printf("The energy is %f\n",energy);
        CSRmat_p colgdls;
        CSRmat_init(&colgdls,nbf*nbf,nbf*nbf);
        CSR_to_CSC(nbf*nbf, gdls,colgdls);
        printf("TestCOLGDLS\n");
        TestCSR(colgdls);
        //CSRmat_p gdls1;
        //CSRmat_init(&gdls1,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
        //Yindextransform3(h2eri->num_bf,gdle1,csrdc,gdls1);
        //printf("TestGDLS1\n");
        //TestCSR(gdls1);
        CSRmat_p gdls2;
        CSRmat_init(&gdls2,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
        Yindextransform4(h2eri->num_bf,gdle2,csrdc,gdls2,threstsfm);
        printf("@@@@@@@@@@@TestGDLS2\n");
        TestCSR(gdls2);
        CSRmat_p colgdls2;
        CSRmat_init(&colgdls2,nbf*nbf,nbf*nbf);
        CSR_to_CSC(nbf*nbf, gdls2,colgdls2);
        //TestCSR(colgdls2);


        double energy;
        energy = Calc_S1energy(gdls,colgdls);
        printf("The S1 energy is %.16g\n",energy);
        double energy2;
        energy2 = Calc_S1energy(gdls2,colgdls2);
        printf("The S1 energy 2 is %.16g\n",energy2);
        COOmat_destroy(cooh2d1);
        CSRmat_destroy(csrh2d);
        CSRmat_destroy(gdle);
        CSRmat_destroy(gdls);
        CSRmat_destroy(colgdls);
        //CSRmat_destroy(gdls1);
        CSRmat_destroy(gdls2);
        CSRmat_destroy(colgdls2);
        CSRmat_destroy(gdle1);

    
    }

    COOmat_destroy(cooden);
    CSRmat_destroy(csrden);
    COOmat_destroy(coodc);
    CSRmat_destroy(csrdc);
    COOmat_destroy(cooh2d);







    
    // Free TinyDFT and H2P-ERI
    TinyDFT_destroy(&TinyDFT);
    H2ERI_destroy(h2eri);
    
    return 0;
}
