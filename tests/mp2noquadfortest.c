#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <errno.h>
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
        if(fabs(coomat->cooval[i])>=maxv*1e-3)
            larger1e5+=1;
        if(fabs(coomat->cooval[i])>maxv*1e-4)
            larger1e9+=1;
        if(fabs(coomat->cooval[i])>0)
            lg0tst+=1;
        norm+=coomat->cooval[i]*coomat->cooval[i];

    }
    printf("The number of values larger than 1e-2,1e-3 and 1e-4 are respectively %d,%d,%d\n",larger1e2,larger1e5,larger1e9);
    printf("The number of elements is %d\n",lg0tst);
    printf("The norm square of the COO matrix is %.16g\n",norm);
}

void TestCSR(CSRmat_p csrmat)
{
    double maxv= csrmat->maxv;
    double norm=0;
    size_t larger1e5=0;
    size_t larger1e9=0;
    size_t larger1e2=0;
    size_t lg0tst=0;


    printf("The max value is %e\n",maxv);
 
    for(size_t i=0;i<csrmat->nnz;i++)
    {
        if(fabs(csrmat->csrval[i])>maxv*1e-2)
            larger1e2+=1;
        if(fabs(csrmat->csrval[i])>=maxv*1e-3)
            larger1e5+=1;
        if(fabs(csrmat->csrval[i])>maxv*1e-4)
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
    printf("The number of values larger than 1e-2,1e-3 and 1e-4 are respectively %lu,%lu,%lu\n",larger1e2,larger1e5,larger1e9);
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
    double thres=1e-8;
    double thres1=atof(argv[5]);
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

    int length = strlen(argv[6]) + 4; // 4 for ".txt" and null terminator
    char *outname = malloc(length);

    if (outname == NULL) {
        perror("Failed to allocate memory");
        return EXIT_FAILURE;
    }

    // Construct the filename
    sprintf(outname, "%s.txt", argv[6]);

    // Step 0: Create the directory
    char dir_path[512];
    snprintf(dir_path, sizeof(dir_path), "/gpfs/scratch/hogao/%s/", argv[6]);

    // Create the directory with permissions (read, write, execute for everyone)
    if (mkdir(dir_path, 0777) != 0) {
        if (errno == EEXIST) {
            printf("Directory already exists.\n");
        } else {
            perror("Error creating directory");
            return 1;
        }
    } else {
        printf("Directory created: %s\n", dir_path);
    }


    //Step1: build low rank ERI matrix
    // Now we need to build the row basis set for every node
    H2E_dense_mat_p *Urbasis;
    Urbasis = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * h2eri->n_node);
    H2ERI_build_rowbs(h2eri,Urbasis);
    printf("2The number of nodes is %d\n",h2eri->n_node);
    FILE *fileur = fopen("ur.txt", "w");
    if (fileur == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<h2eri->n_node;i++)
    {
        if(Urbasis[i]!=NULL){
            fprintf(fileur,"The %dth node has %d rows and %d columns\n",i,Urbasis[i]->nrow,Urbasis[i]->ncol);
        }
    }
    fclose(fileur);
    /*
    for (int i = 0; i < h2eri->n_r_adm_pair; i++)
    {
        int node0  = h2eri->r_adm_pairs[2 * i];
        int node1  = h2eri->r_adm_pairs[2 * i + 1];
        int level0 = h2eri->node_level[node0];
        int level1 = h2eri->node_level[node1];
        printf("The %dth adm pair is (%d,%d) with level (%d,%d)\n",i,node0,node1,level0,level1);
        printf("its B matrix has %d rows and %d columns\n",h2eri->c_B_blks[i]->nrow,h2eri->c_B_blks[i]->ncol);
    }
    printf("Now init pairwise information\n");
    */
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
    FILE *fileuc = fopen("ucbasis.txt", "w");
    if (fileuc == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<2*h2eri->n_r_adm_pair;i++)
    {
        if(Ucbasis[i]!=NULL)
        {
        //    printf("In the admissible pair %d and %d ",admpair1st[i],admpair2nd[i]);
            fprintf(fileuc,"The %dth Ucbasis has %d rows and %d columns\n",i,Ucbasis[i]->nrow,Ucbasis[i]->ncol);
        }
    }
    fclose(fileuc);
    h2eri->leafidx= (int *) malloc(sizeof(int) * h2eri->num_bf*h2eri->num_bf);
    memset(h2eri->leafidx, -1, sizeof(int) * h2eri->num_bf*h2eri->num_bf);
    h2eri->bfpidx= (int *) malloc(sizeof(int) * h2eri->num_bf*h2eri->num_bf);
    memset(h2eri->bfpidx, -1, sizeof(int) * h2eri->num_bf*h2eri->num_bf);
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
    
    
    

    int sum=0;
    for(int i=0;i<h2eri->n_node;i++)
    {
        //printf("The %dth node has %d pairs\n",i,nodepairs[i]->length);
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
    double norm=0;
    for(int i=0;i<nbf*nbf;i++)
    {
        norm+=TinyDFT->D_mat[i]*TinyDFT->D_mat[i];
    }
    printf("The norm of the D matrix is %f\n",norm);
    
   
    COOmat_p cooden;
    COOmat_init(&cooden,h2eri->num_bf,h2eri->num_bf);
    size_t nden =Extract_COO_DDCMat(h2eri->num_bf, h2eri->num_bf, thres1, TinyDFT->D_mat, cooden);
    //printf("Now build csrden\n");
    CSRmat_p csrden;
    CSRmat_init(&csrden,h2eri->num_bf,h2eri->num_bf);
    Double_COO_to_CSR( h2eri->num_bf,  nden, cooden,csrden);
    printf("Test CSRden\n");
    TestCSR(csrden);
    //printf("Now build coodc\n");
    COOmat_p coodc;
    COOmat_init(&coodc,h2eri->num_bf,h2eri->num_bf);
    size_t ndc =Extract_COO_DDCMat(h2eri->num_bf, h2eri->num_bf, thres1, TinyDFT->DC_mat, coodc);
    //printf("Now build csrdc\n");
    CSRmat_p csrdc;
    CSRmat_init(&csrdc,h2eri->num_bf,h2eri->num_bf);
    Double_COO_to_CSR( h2eri->num_bf,  ndc, coodc,csrdc);
    printf("Test CSRdc\n");
    TestCSR(csrdc);

    
    FILE *file11 = fopen("outputCOCC.txt", "w");
    if (file11 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < nbf; i++) {
        for (int j = 0; j < TinyDFT->n_occ; j++) {
            fprintf(file11, "%.16g ", TinyDFT->Cocc_mat[i*TinyDFT->n_occ+j]);
        }
        fprintf(file11, "\n");
    }

    fclose(file11);
    
    FILE *file2 = fopen("outputCVIR.txt", "w");
    if (file2 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < nbf; i++) {
        for (int j = 0; j < TinyDFT->n_vir; j++) {
            fprintf(file2, "%.16g ", TinyDFT->Cvir_mat[i*TinyDFT->n_vir+j]);
        }
        fprintf(file2, "\n");
    }

    
    
    fclose(file2);

    FILE *file3 = fopen("outputenergy.txt", "w");
    if (file3 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < nbf; i++) 
    {
        fprintf(file3, "%.16g ", TinyDFT->orbitenergy_array[i]);
    }
    printf("The number of nocc is %d\n",TinyDFT->n_occ);
    printf("The number of nvir is %d\n",TinyDFT->n_vir);
    fclose(file3);

    FILE *file210 = fopen("outputsameshell.txt", "w");
    if (file210 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<h2eri->num_sp_bfp;i++)
    {
        fprintf(file210, "%d ", h2eri->sameshell[i]);
    }

    fclose(file210);



    COOmat_p cooh2d;
    COOmat_init(&cooh2d,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    H2ERI_build_COO_fulldensetest(h2eri,cooh2d);
    size_t nnz=cooh2d->nnz;
    
    
    COOmat_p cooh2d1;
    COOmat_init(&cooh2d1,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    //compresscoo(cooh2d, cooh2d1, thres);
    CSRmat_p csrh2d;
    CSRmat_init(&csrh2d,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);

    Double_COO_to_CSR( h2eri->num_bf*h2eri->num_bf,  cooh2d->nnz, cooh2d,csrh2d);
    printf("TestCSRh2d\n");
    TestCSR(csrh2d);
    size_t nnz1=csrh2d->nnz;

    FILE *file7 = fopen("outeri.txt", "w");
    if (file7 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < csrh2d->nrow; i++) {
        for (size_t j = csrh2d->csrrow[i]; j < csrh2d->csrrow[i+1]; j++) {
            fprintf(file7, "%d %d %d %d %.16g ", i%nbf,i/nbf,csrh2d->csrcol[j]%nbf,csrh2d->csrcol[j]/nbf,csrh2d->csrval[j]);
            fprintf(file7, "\n");
        }
    }
    fclose(file7);
    //printf("Now build cooden\n");
    
    
    FILE *filepairs = fopen("pairs.txt", "w");
    if (filepairs == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<npairs;i++)
    {
        fprintf(filepairs,"%d %d %d\n",i,pair1st[i],pair2nd[i]);
    }
    fclose(filepairs);
    FILE *filepairidx = fopen("pairsidx.txt", "w");
    if (filepairidx == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<h2eri->n_node;i++)
    {
        for(int j=0;j<nodepairs[i]->length;j++)
        {
            fprintf(filepairidx,"%d %d %d\n",i,nodepairs[i]->data[j],nodepairidx[i]->data[j]);
        }
    }
    fclose(filepairidx);

    FILE *fileadmp = fopen("admpairs.txt", "w");
    if (fileadmp == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<2 * h2eri->n_r_adm_pair;i++)
    {
        fprintf(fileadmp,"%d %d %d \n",i, admpair1st[i],admpair2nd[i]);
    }
    fclose(fileadmp);
    
    FILE *fileadmidx = fopen("admidx.txt", "w");
    if (fileadmidx == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<h2eri->n_node;i++)
    {
        for(int j=0;j<nodeadmpairs[i]->length;j++)
        {
            fprintf(fileadmidx,"%d %d %d\n ",i,nodeadmpairs[i]->data[j],nodeadmpairidx[i]->data[j]);
        }
    }
    fclose(fileadmidx);
    
    printf("Now build gdle\n");
    CSRmat_p gdle;
    CSRmat_init(&gdle,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    double st1,et1;
    st1 = get_wtime_sec();
    Xindextransform3(h2eri->num_bf,csrh2d,csrden,gdle);
    printf("GDLE\n");
    TestCSR(gdle);
    
    printf("Now build gdls\n");
    et1 = get_wtime_sec();
    CSRmat_p gdls;
    CSRmat_init(&gdls,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
        
    st1 = get_wtime_sec();
    Yindextransform3(h2eri->num_bf,gdle,csrdc,gdls);
    et1 = get_wtime_sec();
        
    printf("The Y Index transformation time is %.3lf (s)\n",et1-st1);
    printf("GDLS\n");
    TestCSR(gdls);
    printf("Now do energy calculation \n");
    st1 = get_wtime_sec();
    //    double energy;
    //    energy = Calc_S1energy(gdls);
    //    printf("The energy is %f\n",energy);
    CSRmat_p colgdls;
    CSRmat_init(&colgdls,nbf*nbf,nbf*nbf);
    CSR_to_CSC(nbf*nbf, gdls,colgdls);
    //TestCSR(colgdls);
    size_t mostdense= gdls->nnz;
    double energy;
    energy = Calc_S1energy(gdls,colgdls);
    printf("The S1 energy is %.16g\n",energy);

    
    H2E_dense_mat_p *Upinv;
    Upinv = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * h2eri->n_node);
    printf("Now we are going to build the Upinv\n");
    double usum=0;
    H2E_dense_mat_p *U=h2eri->U;
    FILE *fileu = fopen("umat.txt", "w");
    if (fileu == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<h2eri->n_node;i++)
    {
        usum=0;
        //printf("%d\n",i);
        if(U[i]->nrow==0||h2eri->n_child[i]==0)
        {
            usum+=0;
        }
        else
        {
            //printf("%d\n",i);
            int nrow = U[i]->nrow;
            int ncol = U[i]->ncol;
            for(int j=0;j<nrow;j++)
            {
                for(int k=0;k<ncol;k++)
                {
                    usum+=U[i]->data[j*ncol+k];
                }
            }
            fprintf(fileu,"The %dth U has %d rows and %d columns\n",i,U[i]->nrow,U[i]->ncol);
            fprintf(fileu," sum %.16g \n",usum);
        }
    }
    fclose(fileu);

    build_pinv_rmat(h2eri,Upinv);
    double invsum=0;
    FILE *fileinv = fopen("inv.txt", "w");
    if (fileinv == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<h2eri->n_node;i++)
    {
    if(Upinv[i]->nrow!=0)
        {
            fprintf(fileinv,"The %dth Upinv has %d rows and %d columns\n",i,Upinv[i]->nrow,Upinv[i]->ncol);
            for(int j=0;j<Upinv[i]->nrow;j++)
            {
                for(int k=0;k<Upinv[i]->ncol;k++)
                {
                    invsum+=Upinv[i]->data[j*Upinv[i]->ncol+k]*Upinv[i]->data[j*Upinv[i]->ncol+k];
                }
                fprintf(fileinv," sum %.16g \n",invsum);
            }
        }
    }
    fclose(fileinv);
    printf("finish filewriting\n");
    // Now we need to build the column basis set for every node pair including the inadmissible and self
    H2E_dense_mat_p *S51cbasisx;
    S51cbasisx = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * npairs*2);
    printf("Now we are going to build the S51cbasis\n");
    size_t nfqx = H2ERI_build_S5_X(h2eri,Urbasis,Ucbasis,csrden,csrdc,npairs,pair1st,pair2nd,nodepairs,nodeadmpairs,nodeadmpairidx,S51cbasisx,Upinv);
    printf("Now finish X build Y\n");

    H2E_dense_mat_p *S51cbasisy;
    S51cbasisy = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * npairs);
    size_t nfqy = H2ERI_build_S5_Y(h2eri,Urbasis,S51cbasisx,csrdc,npairs,pair1st,pair2nd,nodepairs,nodepairidx,S51cbasisy,Upinv);
    printf("Now finish Y build c\n");
    H2E_dense_mat_p *S51cbasis;
    S51cbasis = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * npairs*2);
    size_t nfq = H2ERI_build_S5_draft(h2eri,Urbasis,Ucbasis,csrden,csrdc,npairs,pair1st,pair2nd,nodepairs,nodeadmpairs,nodeadmpairidx,S51cbasis,Upinv,1e-6);
    
    
    double tmpval=0;
    // Now write the admissible blocks of ERI tensor into a file
    FILE *file = fopen("outadmeri.txt", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<2*h2eri->n_r_adm_pair;i++)
    {
        int node0 = admpair1st[i];
        int node1 = admpair2nd[i];
        int size0 = Ucbasis[i]->nrow;
        int ncol = Ucbasis[i]->ncol;
        int nrow = Urbasis[node0]->nrow;
        int startrow = h2eri->mat_cluster[2 * node0];
        int startcol = h2eri->mat_cluster[2 * node1];
        for(int j=0;j<nrow;j++)
        {
            int rowidx = startrow+j;
            for(int k=0;k<ncol;k++)
            {
                int colidx = startcol+k;
                tmpval = 0;
                for(int l=0;l<size0;l++)
                {
                    tmpval += Ucbasis[i]->data[l*ncol+k]*Urbasis[node0]->data[j*size0+l];
                }
                fprintf(file, "%d %d %.16g \n", rowidx, colidx, tmpval);
            }
        }
    }

    fclose(file);


    FILE *file1 = fopen("outputsp.txt", "w");
    if (file1 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < h2eri->num_sp_bfp; i++) 
    {
        fprintf(file1, "%d %d %d ", h2eri->sp_bfp_sidx[i],h2eri->bf1st[i],h2eri->bf2nd[i]);
        fprintf(file1, "\n");
    }

    fclose(file1);

    char s51path[512];
    char s51xpath[512];
    char s51ypath[512];
    char xbasispath[512];
    char ybasispath[512];
    const char *directory = "/gpfs/hogao/scratch/my_directory/";
    const char *s51name = "outputs51.txt";
    const char *s51xname = "outputs51x.txt";
    const char *s51yname = "outputs51y.txt";
    const char *xbasisname = "outputxbasis.txt";
    const char *ybasisname = "outputybasis.txt";

    // Concatenate directory path and filename
    snprintf(s51path, sizeof(s51path), "%s%s", dir_path, s51name);
    snprintf(s51xpath, sizeof(s51xpath), "%s%s", dir_path, s51xname);
    snprintf(s51ypath, sizeof(s51ypath), "%s%s", dir_path, s51yname);
    snprintf(xbasispath, sizeof(xbasispath), "%s%s", dir_path, xbasisname);
    snprintf(ybasispath, sizeof(ybasispath), "%s%s", dir_path, ybasisname);


    // Now the S51x information has nodeidx for rows and expanded bfp index for columns 
    
    FILE *file5 = fopen(s51path, "w");
    if (file5 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<npairs;i++)
    {
        int node0 = pair1st[i];
        int node1 = pair2nd[i];
        int startrow = h2eri->mat_cluster[2 * node0];
        int startcol = h2eri->mat_cluster[2 * node1];
        int nrow = Urbasis[node0]->nrow;
        int ncol = S51cbasis[i]->nrow;
        int size0 = S51cbasis[i]->ncol;
        for(int j=0;j<nrow;j++)
        {
            int rowidx = startrow+j;
            for(int k=0;k<ncol;k++)
            {
                int colidx = startcol+k;
                int sameshell = h2eri->sameshell[colidx];
                tmpval = 0;
                for(int l=0;l<size0;l++)
                {
                    tmpval += S51cbasis[i]->data[k*size0+l]*Urbasis[node0]->data[j*size0+l];
                }
                fprintf(file5, "%d %d %.16g \n", rowidx, colidx, tmpval);
            }
        }
    }

    fclose(file5);


    // Now the S51x information has nodeidx for rows and expanded bfp index for columns 
    
    FILE *file5x = fopen(s51xpath, "w");
    if (file5x == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<npairs;i++)
    {
        int node0 = pair1st[i];
        int node1 = pair2nd[i];
        int startrow = h2eri->mat_cluster[2 * node0];
        int startcol = h2eri->mat_cluster[2 * node1];
        int nrow = Urbasis[node0]->nrow;
        int ncol = S51cbasis[i]->nrow;
        int size0 = S51cbasis[i]->ncol;
        for(int j=0;j<nrow;j++)
        {
            int rowidx = startrow+j;
            for(int k=0;k<ncol;k++)
            {
                int colidx = startcol+k;
                int sameshell = h2eri->sameshell[colidx];
                int bf1st = h2eri->bf1st[colidx];
                int bf2nd = h2eri->bf2nd[colidx];
                tmpval = 0;
                if(sameshell==1)
                {
                    for(int l=0;l<size0;l++)
                    {
                        tmpval += S51cbasisx[i]->data[k*size0+l]*Urbasis[node0]->data[j*size0+l];
                    }
                    fprintf(file5x, "%d %d %.16g \n", rowidx, bf2nd*nbf+bf1st, tmpval);
                    
                }
                else if(sameshell==0)
                {
                    int bfsmall;
                    int bflarge;
                    if(bf1st<bf2nd)
                    {
                        bfsmall = bf1st;
                        bflarge = bf2nd;
                    }
                    else
                    {
                        bfsmall = bf2nd;
                        bflarge = bf1st;
                    }
                    
                    tmpval = 0;
                    for(int l=0;l<size0;l++)
                    {
                        tmpval += S51cbasisx[i]->data[k*size0+l]*Urbasis[node0]->data[j*size0+l];
                    }
                    fprintf(file5x, "%d %d %.16g \n", rowidx, bflarge*nbf+bfsmall, tmpval);
                    tmpval = 0;
                    for(int l=0;l<size0;l++)
                    {
                        tmpval += S51cbasisx[i+npairs]->data[k*size0+l]*Urbasis[node0]->data[j*size0+l];
                    }
                    fprintf(file5x, "%d %d %.16g \n", rowidx, bfsmall*nbf+bflarge, tmpval);
                }
                
            }
        }
    }

    fclose(file5);
    


    FILE *file51y = fopen(s51ypath, "w");
    if (file51y == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<npairs;i++)
    {
        int node0 = pair1st[i];
        int node1 = pair2nd[i];
        int startrow = h2eri->mat_cluster[2 * node0];
        int startcol = h2eri->mat_cluster[2 * node1];
        int nrow = Urbasis[node0]->nrow;
        int ncol = S51cbasisy[i]->nrow;
        int size0 = S51cbasisy[i]->ncol;
        for(int j=0;j<nrow;j++)
        {
            int rowidx = startrow+j;
            for(int k=0;k<ncol;k++)
            {
                int colidx = startcol+k;
                int sameshell = h2eri->sameshell[colidx];
                tmpval = 0;
                for(int l=0;l<size0;l++)
                {
                    tmpval += S51cbasisy[i]->data[k*size0+l]*Urbasis[node0]->data[j*size0+l];
                }
                fprintf(file51y, "%d %d %.16g \n", rowidx, colidx, tmpval);
            }
        }
    }

    fclose(file51y);

    FILE *filebasisy = fopen(ybasispath, "w");
    if (filebasisy == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<npairs;i++)
    {
        for(int j=0;j<S51cbasisy[i]->nrow;j++)
        for(int k=0;k<S51cbasisy[i]->ncol;k++)
        {
            fprintf(filebasisy, "%d %d %.16g \n",j, k, S51cbasisy[i]->data[j*S51cbasisy[i]->ncol+k]);
        }

    }

    fclose(filebasisy);


    FILE *filebasisx = fopen(xbasispath, "w");
    if (filebasisx == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<2*npairs;i++)
    {
        for(int j=0;j<S51cbasisx[i]->nrow;j++)
        for(int k=0;k<S51cbasisx[i]->ncol;k++)
        {
            fprintf(filebasisx, "%d %d %.16g \n", j, k, S51cbasisx[i]->data[j*S51cbasisx[i]->ncol+k]);
        }

    }

    fclose(filebasisx);

    double s51energy=0;
    printf("Now we are going to calculate the S51 energy\n");
    s51energy = calc_S51_self_interaction(h2eri, Urbasis, S51cbasis, npairs, pair1st, pair2nd);
    printf("The S51 energy is %.16g\n",s51energy);

    double s51yenergy=0;
    s51yenergy = calc_S51_self_interaction(h2eri, Urbasis, S51cbasisy, npairs, pair1st, pair2nd);
    printf("The S51y energy is %.16g\n",s51yenergy);
    double s1s5 = 0;
    s1s5 = calc_S1S51(gdls,h2eri, Urbasis,S51cbasis, nodepairs, nodepairidx);
    printf("The S1S51 energy is %.16g\n",s1s5);
    printf("The total energy is %.16g\n",energy+2*s1s5+s51energy);
    double sumenergy = energy+2*s1s5+s51energy;
    double s1s5y = 0;
    s1s5y = calc_S1S51(gdls,h2eri, Urbasis,S51cbasisy, nodepairs, nodepairidx);
    printf("The S1S51y energy is %.16g\n",s1s5y);
    printf("The total energyY is %.16g\n",energy+2*s1s5y+s51yenergy);
    double sumenergyxy = energy+2*s1s5y+s51yenergy;
    
    size_t mems51 = 0;
    size_t mems51xy = 0;
    double sum51=0;
    double sumx=0;
    double sumy=0;
    printf("now x\n");
    for(int i=0;i<2*npairs;i++)
    {
        if(S51cbasisx[i]!=NULL)
        {
            mems51xy+=S51cbasisx[i]->nrow*S51cbasisx[i]->ncol;
            for(int j=0;j<S51cbasisx[i]->nrow*S51cbasisx[i]->ncol;j++)
            {
                sumx+=S51cbasisx[i]->data[j];
            }
            
        }
    }
    printf("now y\n");
    for(int i=0;i<npairs;i++)
    {
        if(S51cbasisy[i]!=NULL)
        {
            
            mems51xy+=S51cbasisy[i]->nrow*S51cbasisy[i]->ncol;
            for(int j=0;j<S51cbasisy[i]->nrow*S51cbasisy[i]->ncol;j++)
            {
                sumy+=S51cbasisy[i]->data[j];
            }
            
        }
    }
    printf("now s\n");
    for(int i=0;i<npairs;i++)
    {
        if(S51cbasis[i]!=NULL)
        {
            mems51+=S51cbasis[i]->nrow*S51cbasis[i]->ncol;
            for(int j=0;j<S51cbasis[i]->nrow*S51cbasis[i]->ncol;j++)
            {
                sum51+=S51cbasis[i]->data[j];
            }
           
        }
    }
    
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
    fprintf(fileou, "The number of pairs is %d\n",npairs);
    fprintf(fileou, "The number of admissible pairs is %d\n",h2eri->n_r_adm_pair);
    fprintf(fileou, "The number of inadmissible pairs is %d\n",h2eri->n_r_inadm_pair);

    fprintf(fileou, "The threshold for X and Y is %.16g\n",thres1);
    fprintf(fileou,"The number of elements in ERI is %ld\n",nnz1);
    fprintf(fileou, "memory cost for dense ERI is %ld\n",mostdense);
    fprintf(fileou, "memory cost for S5 is %ld\n",mems51);
    fprintf(fileou, "memory cost for S5x and S5y is %ld\n",mems51xy);
    fprintf(fileou, "The number of flops for S5 is %ld\n",nfq);
    fprintf(fileou, "The number of flops for S5x is %ld\n",nfqx);
    fprintf(fileou, "The number of flops for S5y is %ld\n",nfqy);

    fprintf(fileou, "The sumx is %.16g\n",sumx);
    fprintf(fileou, "The sumy is %.16g\n",sumy);
    fprintf(fileou, "The sum51 is %.16g\n",sum51);
    
    // File operations here (e.g., reading or writing)
    printf("File '%s' opened successfully.\n", outname);

    // Close the file
    fclose(fileou);

    // Free the allocated memory for the filename
    free(outname);



    printf("free cbasis\n");
    for(int i=0;i<npairs;i++)
    {
        if(S51cbasis[i]!=NULL)
        {
            H2E_dense_mat_destroy(&S51cbasis[i]);
        }
    }
    printf("free cbasisx\n");
    for(int i=0;i<2*npairs;i++)
    {
        if(S51cbasisx[i]!=NULL)
        {
            if(i%100==0) printf("x%d\n",i);
            H2E_dense_mat_destroy(&S51cbasisx[i]);
        }
    }
    printf("free cbasisy\n");
    for(int i=0;i<npairs;i++)
    {
        if(S51cbasisy[i]!=NULL)
        {
            H2E_dense_mat_destroy(&S51cbasisy[i]);
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
    free(pair1st);
    free(pair2nd);
    for(int i=0;i<h2eri->n_node;i++)
    {
        if(Urbasis[i]!=NULL)
        {
            H2E_dense_mat_destroy(&Urbasis[i]);
        }
    }
    free(Urbasis);
    free(Ucbasis);
    free(S51cbasis);
    free(S51cbasisx);
    free(S51cbasisy);
    free(admpair1st);
    free(admpair2nd);
    for(int i=0;i<h2eri->n_node;i++)
    {
        if(Upinv[i]!=NULL)
        {
            H2E_dense_mat_destroy(&Upinv[i]);
        }
    }
    free(Upinv);
    for(int i=0;i<h2eri->n_node;i++)
    {
        H2E_int_vec_destroy(&nodepairs[i]);
        H2E_int_vec_destroy(&nodepairidx[i]);
    }
    free(nodepairs);
    free(nodepairidx);
    for(int i=0;i<h2eri->n_node;i++)
    {
        H2E_int_vec_destroy(&nodeadmpairs[i]);
        H2E_int_vec_destroy(&nodeadmpairidx[i]);
    }
    free(nodeadmpairs);
    free(nodeadmpairidx);
    



    // Free TinyDFT and H2P-ERI
    TinyDFT_destroy(&TinyDFT);
    H2ERI_destroy(h2eri);
    
    return 0;
}
//check the random function they use and determine the function that set the seed then in main function
//pass in the seed from command line
//test whether reproduce the same result with same seed
//set seed make seed and run many times 
//run in debugger when can reproduce it
//uninitiallized variables or reading out of bound