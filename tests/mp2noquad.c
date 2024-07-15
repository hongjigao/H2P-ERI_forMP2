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
    double thres=1e-6;
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

    FILE *file112 = fopen("outputsp.txt", "w");
    if (file112 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < h2eri->num_sp_bfp; i++) 
    {
        fprintf(file112, "%d %d %d ", h2eri->sp_bfp_sidx[i],h2eri->bf1st[i],h2eri->bf2nd[i]);
        fprintf(file112, "\n");
    }

    fclose(file112);

    
    H2E_dense_mat_p *Upinv;
    Upinv = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * h2eri->n_node);
    printf("Now we are going to build the Upinv\n");


    build_pinv_rmat(h2eri,Upinv);
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
        }
    }
    fclose(fileinv);

    // Now we need to build the column basis set for every node pair including the inadmissible and self
    H2E_dense_mat_p *S51cbasis;
    S51cbasis = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * npairs);
    printf("Now we are going to build the S51cbasis\n");
    H2ERI_build_S5(h2eri,Urbasis,Ucbasis,csrden,csrdc,npairs,pair1st,pair2nd,nodepairs,nodeadmpairs,nodeadmpairidx,S51cbasis,Upinv);
    FILE *files51bs = fopen("s51bs.txt", "w");
    if (files51bs == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<npairs;i++)
    {
        if(S51cbasis[i]!=NULL)
        {
            fprintf(files51bs,"The %dth S51cbasis has %d rows and %d columns\n",i,S51cbasis[i]->nrow,S51cbasis[i]->ncol);
        }
    }
    fclose(files51bs);
    
    

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




    
    double *s51sp;
    s51sp=(double*) malloc(sizeof(double)*h2eri->num_sp_bfp*h2eri->num_sp_bfp);
    memset(s51sp,0,sizeof(double)*h2eri->num_sp_bfp*h2eri->num_sp_bfp);
    FILE *file5 = fopen("outputs51.txt", "w");
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
                tmpval = 0;
                for(int l=0;l<size0;l++)
                {
                    tmpval += S51cbasis[i]->data[k*size0+l]*Urbasis[node0]->data[j*size0+l];
                }
                fprintf(file, "%d %d %.16g \n", rowidx, colidx, tmpval);
                s51sp[rowidx*h2eri->num_sp_bfp+colidx]=tmpval;
            }
        }
    }

    fclose(file5);



    


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
    
    double energy;
    energy = Calc_S1energy(gdls,colgdls);
    printf("The S1 energy is %.16g\n",energy);
    double s51energy=0;
    s51energy = calc_S51_self_interaction(h2eri, Urbasis, S51cbasis, npairs, pair1st, pair2nd);
    printf("The S51 energy is %.16g\n",s51energy);


    
    double s1s5 = 0;
    s1s5 = calc_S1S51(gdls,h2eri, Urbasis,S51cbasis, nodepairs, nodepairidx);
    printf("The S1S51 energy is %.16g\n",s1s5);
    printf("The total energy is %.16g\n",energy+2*s1s5+s51energy);

    int * childlist;
    childlist=(int *) malloc(sizeof(int) * h2eri->n_node);
    memset(childlist,0,sizeof(int) * h2eri->n_node);
    int * pairlist;
    pairlist=(int *) malloc(sizeof(int) * h2eri->n_node);
    memset(pairlist,0,sizeof(int) * h2eri->n_node);
    int childstep[5] = {22, 24, 28, 29, 30};

    int nchild = Split_node(h2eri,21,22,childstep,childlist,pairlist,nodeadmpairs,nodeadmpairidx);
    printf("The number of children is %d\n",nchild);
    for(int i=0;i<nchild;i++)
    {
        printf("The %dth child is %d\n",i,childlist[i]);
    }
    for(int i=0;i<nchild;i++)
    {
        printf("The %dth pair is %d\n",i,pairlist[i]);
    }

    FILE *file17 = fopen("urbasis0.txt", "w");
    if (file17 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Urbasis[0]->nrow; i++) 
    {
        for(int j=0;j<Urbasis[0]->ncol;j++)
        {
            fprintf(file17, "%.16g ", Urbasis[0]->data[i*Urbasis[0]->ncol+j]);            
        }
        fprintf(file17, "\n");
    }

    fclose(file17);

    FILE *file18 = fopen("urbasis1.txt", "w");
    if (file18 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Urbasis[1]->nrow; i++) 
    {
        for(int j=0;j<Urbasis[1]->ncol;j++)
        {
            fprintf(file18, "%.16g ", Urbasis[1]->data[i*Urbasis[1]->ncol+j]);            
        }
        fprintf(file18, "\n");
    }

    fclose(file18);
    

    FILE *file19 = fopen("urbasis2.txt", "w");
    if (file19 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Urbasis[2]->nrow; i++) 
    {
        for(int j=0;j<Urbasis[2]->ncol;j++)
        {
            fprintf(file19, "%.16g ", Urbasis[2]->data[i*Urbasis[2]->ncol+j]);            
        }
        fprintf(file19, "\n");
    }

    fclose(file19);

    FILE *file20 = fopen("urbasis6.txt", "w");
    if (file20 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Urbasis[6]->nrow; i++) 
    {
        for(int j=0;j<Urbasis[6]->ncol;j++)
        {
            fprintf(file20, "%.16g ", Urbasis[6]->data[i*Urbasis[6]->ncol+j]);            
        }
        fprintf(file20, "\n");
    }

    fclose(file20);

    FILE *file21 = fopen("urbasis21.txt", "w");
    if (file21 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Urbasis[21]->nrow; i++) 
    {
        for(int j=0;j<Urbasis[21]->ncol;j++)
        {
            fprintf(file21, "%.16g ", Urbasis[21]->data[i*Urbasis[21]->ncol+j]);            
        }
        fprintf(file21, "\n");
    }

    fclose(file21);



    FILE *file15 = fopen("ucbasis6.txt", "w");
    if (file15 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Ucbasis[6]->nrow; i++) 
    {
        for(int j=0;j<Ucbasis[6]->ncol;j++)
        {
            fprintf(file15, "%.16g ", Ucbasis[6]->data[i*Ucbasis[6]->ncol+j]);            
        }
        fprintf(file15, "\n");
    }

    fclose(file15);

    FILE *filec0 = fopen("ucbasis0.txt", "w");
    if (filec0 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Ucbasis[0]->nrow; i++) 
    {
        for(int j=0;j<Ucbasis[0]->ncol;j++)
        {
            fprintf(filec0, "%.16g ", Ucbasis[0]->data[i*Ucbasis[0]->ncol+j]);            
        }
        fprintf(filec0, "\n");
    }

    fclose(filec0);

    FILE *filec1 = fopen("ucbasis1.txt", "w");
    if (filec1 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Ucbasis[1]->nrow; i++) 
    {
        for(int j=0;j<Ucbasis[1]->ncol;j++)
        {
            fprintf(filec1, "%.16g ", Ucbasis[1]->data[i*Ucbasis[1]->ncol+j]);            
        }
        fprintf(filec1, "\n");
    }

    fclose(filec1);




    FILE *file16 = fopen("ucbasis7.txt", "w");
    if (file16 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Ucbasis[7]->nrow; i++) 
    {
        for(int j=0;j<Ucbasis[7]->ncol;j++)
        {
            fprintf(file16, "%.16g ", Ucbasis[7]->data[i*Ucbasis[7]->ncol+j]);            
        }
        fprintf(file16, "\n");
    }

    fclose(file16);

    FILE *file244 = fopen("ucbasis24.txt", "w");
    if (file244 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Ucbasis[24]->nrow; i++) 
    {
        for(int j=0;j<Ucbasis[24]->ncol;j++)
        {
            fprintf(file244, "%.16g ", Ucbasis[24]->data[i*Ucbasis[24]->ncol+j]);            
        }
        fprintf(file244, "\n");
    }

    fclose(file244);


    FILE *file255 = fopen("ucbasis25.txt", "w");
    if (file16 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Ucbasis[25]->nrow; i++) 
    {
        for(int j=0;j<Ucbasis[25]->ncol;j++)
        {
            fprintf(file255, "%.16g ", Ucbasis[25]->data[i*Ucbasis[25]->ncol+j]);            
        }
        fprintf(file255, "\n");
    }

    fclose(file255);



    FILE *file88 = fopen("u2.txt", "w");
    if (file88 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < h2eri->U[2]->nrow; i++) 
    {
        for(int j=0;j<h2eri->U[2]->ncol;j++)
        {
            fprintf(file88, "%.16g ", h2eri->U[2]->data[i*h2eri->U[2]->ncol+j]);            
        }
        fprintf(file88, "\n");
    }

    fclose(file88);

    FILE *file89 = fopen("u6.txt", "w");
    if (file89 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < h2eri->U[6]->nrow; i++) 
    {
        for(int j=0;j<h2eri->U[6]->ncol;j++)
        {
            fprintf(file89, "%.16g ", h2eri->U[6]->data[i*h2eri->U[6]->ncol+j]);            
        }
        fprintf(file89, "\n");
    }

    fclose(file89);







    for(int i=0;i<npairs;i++)
    {
        if(S51cbasis[i]!=NULL)
        {
            H2E_dense_mat_destroy(&S51cbasis[i]);
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
