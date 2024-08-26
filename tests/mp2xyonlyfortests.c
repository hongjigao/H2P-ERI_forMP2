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
    double thres=1e-5;
    double thres1=atof(argv[5]);
    double thres2=10*thres1;
    double et = get_wtime_sec();
    printf("The time for SCF is %.3lf (s)\n", et - st);
    double scftime = et - st;

    int nbf = h2eri->num_bf;
    int * tmpshellidx;
    tmpshellidx=(int *) malloc(sizeof(int) * (h2eri->nshell+1));
    for(int j=0;j<h2eri->nshell;j++)
    {
        tmpshellidx[j]=h2eri->shell_bf_sidx[j];
    }
    tmpshellidx[h2eri->nshell]=nbf;
    h2eri->shell_bf_sidx=(int *) malloc(sizeof(int) * (h2eri->nshell+1));
    memset(h2eri->shell_bf_sidx, 0, sizeof(int) * (h2eri->nshell+1));
    for(int j=0;j<h2eri->nshell+1;j++)
    {
        h2eri->shell_bf_sidx[j]=tmpshellidx[j];
    }

    free(tmpshellidx);

    printf("The number of basis functions is %d\n",nbf);
    printf("1The number of nodes is %d\n",h2eri->n_node);
    
    st = get_wtime_sec();
    //Step 1: Build the low rank ERI part
    st = get_wtime_sec();
    H2E_dense_mat_p *Urbasis;
    Urbasis = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * h2eri->n_node);
    H2ERI_build_rowbs(h2eri,Urbasis);
    
    printf("Now init pairwise information\n");
    int *admpair1st;
    admpair1st=(int *) malloc(sizeof(int) * 2 * h2eri->n_r_adm_pair);
    int *admpair2nd;
    admpair2nd=(int *) malloc(sizeof(int) * 2 * h2eri->n_r_adm_pair);
    H2E_int_vec_p *nodeadmpairs;
    nodeadmpairs = (H2E_int_vec_p *) malloc(sizeof(H2E_int_vec_p) * h2eri->n_node);
    H2E_int_vec_p *nodeadmpairidx;
    nodeadmpairidx = (H2E_int_vec_p *) malloc(sizeof(H2E_int_vec_p) * h2eri->n_node);
    for(int i=0;i<h2eri->n_node;i++)
    {
        H2E_int_vec_init(&nodeadmpairs[i],10);
        H2E_int_vec_init(&nodeadmpairidx[i],10);
    }
    for(int i=0;i<h2eri->n_r_adm_pair;i++)
    {
        admpair1st[i]=h2eri->r_adm_pairs[2*i];
        admpair2nd[i]=h2eri->r_adm_pairs[2*i+1];
        H2E_int_vec_push_back(nodeadmpairs[admpair1st[i]],admpair2nd[i]);
        H2E_int_vec_push_back(nodeadmpairs[admpair2nd[i]],admpair1st[i]);
        H2E_int_vec_push_back(nodeadmpairidx[admpair1st[i]],i);
        H2E_int_vec_push_back(nodeadmpairidx[admpair2nd[i]],i+h2eri->n_r_adm_pair);
    }
    for(int i=0;i<h2eri->n_r_adm_pair;i++)
    {
        admpair1st[i+h2eri->n_r_adm_pair]=h2eri->r_adm_pairs[2*i+1];
        admpair2nd[i+h2eri->n_r_adm_pair]=h2eri->r_adm_pairs[2*i];
    }
    // Now we need to build the column basis set for every admissible pair
    printf("Now we are going to build the Ucbasis\n");
    H2E_dense_mat_p *Ucbasis;
    Ucbasis = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * h2eri->n_r_adm_pair*2);
    H2ERI_build_colbs(h2eri,Ucbasis,admpair1st,admpair2nd,Urbasis);
    
    h2eri->leafidx= (int *) malloc(sizeof(int) * nbf*nbf);
    memset(h2eri->leafidx, -1, sizeof(int) * nbf*nbf);
    h2eri->bfpidx= (int *) malloc(sizeof(int) * nbf*nbf);
    memset(h2eri->bfpidx, -1, sizeof(int) * nbf*nbf);
    printf("Now we are going to build the leafidx\n");
    for(int i = 0; i < h2eri->n_leaf_node; i++)
    {
        int node = h2eri->height_nodes[i];
        //printf("%d \n",node);
        int startp = h2eri->mat_cluster[2 * node];
        int endp = h2eri->mat_cluster[2 * node + 1];
        for(int j = startp; j <= endp; j++)
        {
            int bf1st = h2eri->bf1st[j];
            int bf2nd = h2eri->bf2nd[j];
            h2eri->leafidx[bf1st*nbf+bf2nd] = node;
            h2eri->leafidx[bf2nd*nbf+bf1st] = node;
            if(h2eri->sameshell[j]==1)
            {
                h2eri->bfpidx[bf1st*nbf+bf2nd] = j;
            }
            else if(h2eri->sameshell[j]==0)
            {
                h2eri->bfpidx[bf1st*nbf+bf2nd] = j;
                h2eri->bfpidx[bf2nd*nbf+bf1st] = j;
            }
        }
    }
    int npairs = 2*h2eri->n_r_adm_pair+2*h2eri->n_r_inadm_pair+h2eri->n_leaf_node;
    printf("The number of pairs is %d\n",npairs);
    int *pair1st;
    pair1st=(int *) malloc(sizeof(int) * npairs);
    int *pair2nd;
    pair2nd=(int *) malloc(sizeof(int) * npairs);
    H2E_int_vec_p *nodepairs;
    nodepairs = (H2E_int_vec_p *) malloc(sizeof(H2E_int_vec_p) * h2eri->n_node);
    H2E_int_vec_p *nodepairidx;
    nodepairidx = (H2E_int_vec_p *) malloc(sizeof(H2E_int_vec_p) * h2eri->n_node);
    for(int i=0;i<h2eri->n_node;i++)
    {
        H2E_int_vec_init(&nodepairs[i],10);
        H2E_int_vec_init(&nodepairidx[i],10);
    }

    for(int i=0;i<h2eri->n_leaf_node;i++)
    {
        pair1st[i]=h2eri->height_nodes[i];
        pair2nd[i]=h2eri->height_nodes[i];
        H2E_int_vec_push_back(nodepairs[pair1st[i]],pair2nd[i]);
        H2E_int_vec_push_back(nodepairidx[pair1st[i]],i);
        
    }
    for(int i=0;i<h2eri->n_r_inadm_pair;i++)
    {
        pair1st[i+h2eri->n_leaf_node]=h2eri->r_inadm_pairs[2*i];
        pair2nd[i+h2eri->n_leaf_node]=h2eri->r_inadm_pairs[2*i+1];
        pair1st[i+h2eri->n_leaf_node+h2eri->n_r_inadm_pair]=h2eri->r_inadm_pairs[2*i+1];
        pair2nd[i+h2eri->n_leaf_node+h2eri->n_r_inadm_pair]=h2eri->r_inadm_pairs[2*i];
        H2E_int_vec_push_back(nodepairs[pair1st[i+h2eri->n_leaf_node]],pair2nd[i+h2eri->n_leaf_node]);
        H2E_int_vec_push_back(nodepairs[pair2nd[i+h2eri->n_leaf_node]],pair1st[i+h2eri->n_leaf_node]);
        H2E_int_vec_push_back(nodepairidx[pair1st[i+h2eri->n_leaf_node]],i+h2eri->n_leaf_node);
        H2E_int_vec_push_back(nodepairidx[pair2nd[i+h2eri->n_leaf_node]],i+h2eri->n_leaf_node+h2eri->n_r_inadm_pair);
    }


    for(int i=0;i<2*h2eri->n_r_adm_pair;i++)
    {
        pair1st[i+h2eri->n_leaf_node+2*h2eri->n_r_inadm_pair]=admpair1st[i];
        pair2nd[i+h2eri->n_leaf_node+2*h2eri->n_r_inadm_pair]=admpair2nd[i];
        H2E_int_vec_push_back(nodepairs[pair1st[i+h2eri->n_leaf_node+2*h2eri->n_r_inadm_pair]],pair2nd[i+h2eri->n_leaf_node+2*h2eri->n_r_inadm_pair]);
        H2E_int_vec_push_back(nodepairidx[pair1st[i+h2eri->n_leaf_node+2*h2eri->n_r_inadm_pair]],i+h2eri->n_leaf_node+2*h2eri->n_r_inadm_pair);
    }

    et = get_wtime_sec();
    printf("The time for building the nodepairs, Ur and Uc is %.3lf (s)\n", et - st);
    
    double buildnodepairs = et - st;
   
    
    
    
    st = get_wtime_sec();
    
    H2E_dense_mat_p *Upinv;
    Upinv = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * h2eri->n_node);
    printf("Now we are going to build the Upinv\n");
    build_pinv_rmat(h2eri,Upinv);
    et = get_wtime_sec();
    printf("The time for building the Upinv is %.3lf (s)\n", et - st);
    double buildpinv = et - st;

    // Step2: Build the dense ERI part
    
    COOmat_p cooh2d;
    COOmat_init(&cooh2d,nbf*nbf,nbf*nbf);
    H2ERI_build_COO_fulldensetest(h2eri,cooh2d);
    size_t nnz=cooh2d->nnz;
    
    
    COOmat_p cooh2d1;
    COOmat_init(&cooh2d1,nbf*nbf,nbf*nbf);
    compresscoo(cooh2d, cooh2d1, thres);
    CSRmat_p csrh2d;
    CSRmat_init(&csrh2d,nbf*nbf,nbf*nbf);

    Double_COO_to_CSR( nbf*nbf,  cooh2d1->nnz, cooh2d1,csrh2d);
    //printf("TestCSRh2d\n");
    //TestCSR(csrh2d);
    et = get_wtime_sec();
    printf("The time for construction of ERI is %.3lf (s)\n", et - st);
    
    TinyDFT_build_MP2info_eig(TinyDFT, TinyDFT->F_mat,
                               TinyDFT->X_mat, TinyDFT->D_mat,
                               TinyDFT->Cocc_mat, TinyDFT->DC_mat,
                               TinyDFT->Cvir_mat, TinyDFT->orbitenergy_array);

    
    double Fermie=0.0;
    size_t mostx=0;
    size_t mosty=0;

    double maxx = Calcmaxv(TinyDFT->D_mat, nbf);
    double maxy = Calcmaxv(TinyDFT->DC_mat, nbf);


    double gap = (TinyDFT->orbitenergy_array[TinyDFT->nbf-1]-TinyDFT->orbitenergy_array[0])/(TinyDFT->orbitenergy_array[TinyDFT->n_occ]-TinyDFT->orbitenergy_array[TinyDFT->n_occ-1]);
    printf("The gap factor is %.16g\n",gap);
    
    


    // Step 3: Find the quadrature points
    char directorypath[100]= "/gpfs/projects/JiaoGroup/hongjigao/gccmp2test/H2P-ERI_forMP2/tests/1_x/";
    //char *filename="1_xk08_3E1";
    char *format = format_double(gap);
    int n;
    char *num;
    if(gap>=10) 
    {
        num = "1_xk08_";
        n=8;
    }
    else
    {
        num = "1_xk07_";
        n=7;
    }    

    char * quadfile = concatenate(num,format);
    printf("quad file name: %s\n", quadfile);
    char full_path[150];  
    snprintf(full_path, sizeof(full_path), "%s%s", directorypath, quadfile);
    FILE *file = fopen(full_path, "r");
    if (file == NULL) {
        perror("Failed to open file");
        return 1;
    }

    
    double omega[n], alpha[n];
    int omega_count = 0, alpha_count = 0;
    char line[128];

    while (fgets(line, sizeof(line), file)) {
        double value;
        char label[50];

        if (sscanf(line, "%lf {%[^}]}", &value, label) > 0) {
            if (strstr(label, "omega") && omega_count < n) {
                omega[omega_count++] = value;
            } else if (strstr(label, "alpha") && alpha_count < n) {
                alpha[alpha_count++] = value;
            }

            // Stop reading if we have enough entries
            if (omega_count == n && alpha_count == n) {
                break;
            }
        }
    }

    fclose(file);
    // Output arrays to verify correctness
    printf("Omega Array:\n");
    for (int i = 0; i < omega_count; i++) {
        printf("%.15lf\n", omega[i]);
    }

    printf("Alpha Array:\n");
    for (int i = 0; i < alpha_count; i++) {
        printf("%.15lf\n", alpha[i]);
    }
    
    double buildxy=0;
    double denseindtrans=0;
    double calcs1=0;
    double builds5=0;
    double calcs51=0;
    size_t mostdense=0;
    size_t mosts5=0;
    size_t nflop = 0;
    size_t nflopx = 0;
    size_t nflopy = 0;
    double *Cocc_mat = TinyDFT->Cocc_mat;
    double *Cvir_mat = TinyDFT->Cvir_mat;
    double *orbitenergy_array = TinyDFT->orbitenergy_array;
    double thresprod = atof(argv[7]);
    // Step 4: Do the MP2 calculation in the for loop
    double sumenergy = 0;
    double sumenergyxy = 0;
    double largerate = 0;
    double sumx[n];
    memset(sumx, 0, sizeof(double) * n);
    double sumy[n];
    memset(sumy, 0, sizeof(double) * n);
    //#pragma omp parallel for
    for(int quad = 0;quad<n;quad++)
    {
        printf("Thread %d processing index %d\n", omp_get_thread_num(), quad);
        double omega_val = omega[quad];
        double alpha_val = alpha[quad];
        double st1,et1;
        printf("The %dth omega is %f, the %dth alpha is %f\n",quad,omega_val,quad,alpha_val);
        st1 = get_wtime_sec();
        double *tmpD;
        tmpD = (double *) malloc(sizeof(double) * nbf * nbf);
        ASSERT_PRINTF(tmpD, "Failed to allocate memory for tmpD\n");
        memset(tmpD, 0, sizeof(double) * nbf * nbf);
        double* tmpDC;
        tmpDC = (double *) malloc(sizeof(double) * nbf * nbf);
        ASSERT_PRINTF(tmpDC, "Failed to allocate memory for tmpDC\n");
        memset(tmpDC, 0, sizeof(double) * nbf * nbf);
        int n_occ=TinyDFT->n_occ;
        int n_vir=TinyDFT->n_vir;
        //energy difference
        double edif;
        double tmp_factor;
        double talpha = alpha_val;
        for(int i=0;i<n_occ;i++)
        {
            edif=orbitenergy_array[i]-Fermie;
            tmp_factor= exp(edif*talpha);
            for(int mu=0;mu<nbf;mu++)
                for(int nu=0;nu<nbf;nu++)
                {
                    tmpD[mu*nbf+nu] += Cocc_mat[mu*n_occ+i]*Cocc_mat[nu*n_occ+i]*tmp_factor;
                }
        }
        //Calculate Y
        for(int a=n_occ;a<nbf;a++)
        {
            edif=Fermie - orbitenergy_array[a];
            tmp_factor= exp(edif*talpha);
            for(int mu=0;mu<nbf;mu++)
                for(int nu=0;nu<nbf;nu++)
                {
                    tmpDC[mu*nbf+nu] += Cvir_mat[mu*n_vir+a-n_occ]*Cvir_mat[nu*n_vir+a-n_occ]*tmp_factor;
                }
        }
        COOmat_p cooden;
        COOmat_init(&cooden,nbf,nbf);
        cooden->maxv = maxx;
        size_t nden =Extract_COO_DDCMat(nbf, nbf, thres1, tmpD, cooden);
        if(nden>mostx) mostx=nden;
        CSRmat_p csrden;
        CSRmat_init(&csrden,nbf,nbf);
        Double_COO_to_CSR( nbf,  nden, cooden,csrden);
        COOmat_p coodc;
        COOmat_init(&coodc,nbf,nbf);
        coodc->maxv = maxy;
        size_t ndc =Extract_COO_DDCMat(nbf, nbf, thres1, tmpDC, coodc);
        if(ndc>mosty) mosty=ndc;
        CSRmat_p csrdc;
        CSRmat_init(&csrdc,nbf,nbf);
        Double_COO_to_CSR( nbf,  ndc, coodc,csrdc);
        printf("The useful csrden and csrdc are %ld and %ld\n",csrden->nnz,csrdc->nnz);
        

        COOmat_p cood5;
        COOmat_init(&cood5,nbf,nbf);
        cood5->maxv = maxx;
        size_t nden5 =Extract_COO_DDCMat(nbf, nbf, thres2, tmpD, cood5);
        
        printf("Thres1 = %.16g, Thres2 = %.16g\n", thres1, thres2);
        printf("The number of elements in the d5 of quad %d is %ld\n",quad,nden5);
        CSRmat_p csrd5;
        CSRmat_init(&csrd5,nbf,nbf);
        Double_COO_to_CSR( nbf,  nden5, cood5,csrd5);
        
        COOmat_p coodc5;
        COOmat_init(&coodc5,nbf,nbf);
        coodc5->maxv = maxy;
        size_t ndc5 =Extract_COO_DDCMat(nbf, nbf, thres2, tmpDC, coodc5);
        printf("The number of elements in the d5 of quad %d is %ld\n",quad,ndc5);
        CSRmat_p csrdc5;
        CSRmat_init(&csrdc5,nbf,nbf);
        Double_COO_to_CSR( nbf,  ndc5, coodc5,csrdc5);
        CSRmat_p gdle;
        CSRmat_init(&gdle,nbf*nbf,nbf*nbf);
        //printf("now do x index transformation\n");
        st1 = get_wtime_sec();
        Xindextransform3(nbf,csrh2d,csrden,gdle);
        //printf("Finished X index transformation\n");
        
        CSRmat_p gdls;
        CSRmat_init(&gdls,nbf*nbf,nbf*nbf);
            
        //printf("now do y index transformation\n");
        Yindextransform3(nbf,gdle,csrdc,gdls);
        et1 = get_wtime_sec();
        //printf("Dense Index transformation time is %.16g\n",et1-st1);    
        if(gdls->nnz>mostdense) mostdense=gdls->nnz;
        denseindtrans += et1-st1;
        st1 = get_wtime_sec();
        CSRmat_p colgdls;
        CSRmat_init(&colgdls,nbf*nbf,nbf*nbf);
        CSR_to_CSC(nbf*nbf, gdls,colgdls);
        double energy;
        energy = Calc_S1energy(gdls,colgdls);
        printf("The S1 energy is %.16g\n",energy);
        et1 = get_wtime_sec();
        //printf("The time for calculating the S1 energy is %.3lf (s)\n", et1 - st1);
        calcs1 += et1-st1;
        if(h2eri->n_r_adm_pair==0)
        {
            printf("The total energy in quadrature %d is %.16g\n",quad,energy);
            sumenergy += omega_val*energy;
            continue;
        }
        // Now we need to build the column basis set for every node pair including the inadmissible and self
        st1 = get_wtime_sec();
        //printf("Now we are going to build the S51cbasis\n");
        double s51energyxy=0;
        double s1s5xy = 0;
        sumenergyxy += omega_val*energy;

        free(tmpD);
        free(tmpDC);
        printf("Now we are going to build the S51cbasisxy in quad %d\n",quad);
        st1 = get_wtime_sec();
        H2E_dense_mat_p *S51cbasisx;
        S51cbasisx = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * npairs*2);
        
        size_t nfqx = H2ERI_build_S5_X(h2eri,Urbasis,Ucbasis,csrden,csrdc,npairs,pair1st,pair2nd,nodepairs,nodeadmpairs,nodeadmpairidx,S51cbasisx,Upinv);
        printf("Finish x build y in quad %d\n",quad);
        H2E_dense_mat_p *S51cbasisy;
        S51cbasisy = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * npairs);
        size_t nfqy = H2ERI_build_S5_Y(h2eri,Urbasis,S51cbasisx,csrdc,npairs,pair1st,pair2nd,nodepairs,nodepairidx,S51cbasisy,Upinv);
        et1 = get_wtime_sec();
        printf("build S5xy time in quad %d is %.16g\n",quad, et1-st1);
        s51energyxy = calc_S51_self_interaction(h2eri, Urbasis, S51cbasisy, npairs, pair1st, pair2nd);
        s1s5xy = calc_S1S51(gdls,h2eri, Urbasis,S51cbasisy, nodepairs, nodepairidx);
        printf("The S51xy energy in quad %d is %.16g\n",quad,s51energyxy);
        printf("The S1S51xy energy in quad %d is %.16g\n",quad,s1s5xy);
        sumenergyxy += omega_val*(2*s1s5xy+s51energyxy);
        nflopx += nfqx;
        nflopy += nfqy;

    
        
        printf("Stat Finish quad %d\n",quad);
        sumx[quad] = 0;
        for(int i=0;i<2*npairs;i++)
        {
            if(S51cbasisx[i]!=NULL)
            {
                for(int j=0;j<S51cbasisx[i]->nrow;j++)
                    for(int k=0;k<S51cbasisx[i]->ncol;k++)
                    {
                        sumx[quad] += S51cbasisx[i]->data[j*S51cbasisx[i]->ncol+k];
                    }
                H2E_dense_mat_destroy(&S51cbasisx[i]);
            }
        }
        for(int i=0;i<npairs;i++)
        {    
            if(S51cbasisy[i]!=NULL)
            {
                for(int j=0;j<S51cbasisy[i]->nrow;j++)
                    for(int k=0;k<S51cbasisy[i]->ncol;k++)
                    {
                        sumy[quad] += S51cbasisy[i]->data[j*S51cbasisy[i]->ncol+k];
                    }
                H2E_dense_mat_destroy(&S51cbasisy[i]);
            }
        }
        printf("Finish1 quad %d\n",quad);
        printf("sumx in quad %d is %.16g\n",quad,sumx[quad]);
        printf("sumy in quad %d is %.16g\n",quad,sumy[quad]);
        
        COOmat_destroy(cooden);
        CSRmat_destroy(csrden);
        COOmat_destroy(coodc);
        CSRmat_destroy(csrdc);
        COOmat_destroy(cood5);
        CSRmat_destroy(csrd5);
        COOmat_destroy(coodc5);
        CSRmat_destroy(csrdc5);
        CSRmat_destroy(gdle);
        CSRmat_destroy(gdls);
        CSRmat_destroy(colgdls);
        printf("Finish quad %d\n",quad);

        
    }
    
    
    printf("The total energy is %.16g\n",sumenergy);
    double tmpval=0;
    // Now write the admissible blocks of ERI tensor into a file
    
    double DTYPE_MB = (double) sizeof(DTYPE) / 1048576.0;
    double int_MB   = (double) sizeof(int)   / 1048576.0;

    int length = strlen(argv[1]) + 4; // 4 for ".txt" and null terminator
    char *outname = malloc(length);

    if (outname == NULL) {
        perror("Failed to allocate memory");
        return EXIT_FAILURE;
    }

    // Construct the filename
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
    fprintf(fileou, "The total energy is %.16g\n",sumenergy);
    fprintf(fileou, "The total energy with xy is %.16g\n",sumenergyxy);
    fprintf(fileou, "The number of basis functions is %d\n",nbf);
    fprintf(fileou, "The number of nodes is %d\n",h2eri->n_node);
    fprintf(fileou, "The scf time is %.16g\n",scftime);
    fprintf(fileou, "The number of pairs is %d\n",npairs);
    fprintf(fileou, "The number of admissible pairs is %d\n",h2eri->n_r_adm_pair);
    fprintf(fileou, "The number of inadmissible pairs is %d\n",h2eri->n_r_inadm_pair);
    fprintf(fileou, "The largest number of X is %ld\n",mostx);
    fprintf(fileou, "The largest number of Y is %ld\n",mosty);
    fprintf(fileou, "The gap factor is %.16g\n",gap);
    fprintf(fileou, "The number of quadrature points is %d\n",n);
    fprintf(fileou, "The threshold for X and Y is %.16g\n",thres1);
    fprintf(fileou, "The threshold for the product of X and Y is %.16g\n",thresprod);
    fprintf(fileou, "The largest rate of large values is %.16g\n",largerate);
    fprintf(fileou,"Running time for scf is %.16g\n",scftime);
    fprintf(fileou, "Running time for building nodepairs, Ur and Uc is %.16g\n",buildnodepairs);
    fprintf(fileou, "Running time for building Upinv is %.16g\n",buildpinv);
    fprintf(fileou, "Running time for building xy is %.16g\n",buildxy);
    fprintf(fileou, "Running time for dense index transformation is %.16g\n",denseindtrans);
    fprintf(fileou, "Running time for calculating S1 is %.16g\n",calcs1);
    fprintf(fileou, "The total running time is %.16g\n",ett-stt);
    fprintf(fileou, "memory cost for dense ERI is %ld\n",mostdense);
    fprintf(fileou, "Equals %.16g \n", mostdense*DTYPE_MB);
    fprintf(fileou, "The number of flops for S5x is %ld\n",nflopx);
    fprintf(fileou, "The number of flops for S5y is %ld\n",nflopy);
    for(int i=0;i<n;i++)
    {
        fprintf(fileou, "The %dth sumx is %.16g\n",i,sumx[i]);
        fprintf(fileou, "The %dth sumy is %.16g\n",i,sumy[i]);
    }
    // File operations here (e.g., reading or writing)
    printf("File '%s' opened successfully.\n", outname);

    // Close the file
    fclose(fileou);

    // Free the allocated memory for the filename
    free(outname);







    
    
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
