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
    for(int i=0;i<2*h2eri->n_r_adm_pair;i++)
    {
        if(Ucbasis[i]!=NULL)
        {
        //    printf("In the admissible pair %d and %d ",admpair1st[i],admpair2nd[i]);
         //   printf("The %dth Ucbasis has %d rows and %d columns\n",i,Ucbasis[i]->nrow,Ucbasis[i]->ncol);
        }
    }
    h2eri->leafidx= (int *) malloc(sizeof(int) * h2eri->num_bf*h2eri->num_bf);
    memset(h2eri->leafidx, -1, sizeof(int) * h2eri->num_bf*h2eri->num_bf);
    h2eri->bfpidx= (int *) malloc(sizeof(int) * h2eri->num_bf*h2eri->num_bf);
    memset(h2eri->bfpidx, -1, sizeof(int) * h2eri->num_bf*h2eri->num_bf);
    printf("Now we are going to build the leafidx\n");
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
    norm=0;
    for(int i=0;i<nbf*nbf;i++)
    {
        norm+=TinyDFT->DC_mat[i]*TinyDFT->DC_mat[i];
    }
    printf("The norm of the DC matrix is %f\n",norm);
    CSRmat_p csrd5;
    CSRmat_init(&csrd5, nbf, nbf);
    CSRmat_p csrdc5;
    CSRmat_init(&csrdc5, nbf, nbf);
    H2ERI_extract_near_large_elements(h2eri, TinyDFT, csrd5, csrdc5, 15, 1e-6);

    printf("The number of nonzero elements in the D5 matrix is %ld\n",csrd5->nnz);
    printf("The number of nonzero elements in the DC5 matrix is %ld\n",csrdc5->nnz);
    norm=0;
    for(size_t i=0;i<csrd5->nnz;i++)
    {
        norm+=csrd5->csrval[i]*csrd5->csrval[i];
    }
    printf("The norm of the D5 matrix is %f\n",norm);
    norm=0;
    
    for(size_t i=0;i<csrdc5->nnz;i++)
    {
        norm+=csrdc5->csrval[i]*csrdc5->csrval[i];
    }
    printf("The norm of the DC5 matrix is %f\n",norm);
    double *x;
    x=(double*) malloc(sizeof(double)*nbf);
    double *y;
    y=(double*) malloc(sizeof(double)*nbf);
    double *z;
    z=(double*) malloc(sizeof(double)*nbf);
    for(int i=0;i<h2eri->nshell;i++)
    {
        for(int j=h2eri->shell_bf_sidx[i];j<h2eri->shell_bf_sidx[i+1];j++)
        {
            x[j]=h2eri->shells[i].x;
            y[j]=h2eri->shells[i].y;
            z[j]=h2eri->shells[i].z;
        }
    }
    //Here we don't know exactly about the math function so we directly use its square
    double * distance;
    distance=(double*) malloc(sizeof(double)*nbf*nbf);
    for(int i=0;i<nbf;i++)
        for(int j=0;j<nbf;j++)
        {
            distance[i*nbf+j]=(x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])+(z[i]-z[j])*(z[i]-z[j]);
            //printf("%f ",distance[i*nbf+j]);
            //printf("%f ",distance[i*nbf+j]);
        }

    double maxdistance=0;
    for(int i=0;i<nbf;i++)
    {
        for(int j=csrd5->csrrow[i];j<csrd5->csrrow[i+1];j++)
        {
            if(distance[i*nbf+csrd5->csrcol[j]]>maxdistance)
                maxdistance=distance[i*nbf+csrd5->csrcol[j]];
        }
    }
    printf("The max distance is %f\n",maxdistance);
    maxdistance=0;
    for(int i=0;i<nbf;i++)
    {
        for(int j=0;j<nbf;j++)
        {
            if(distance[i*nbf+j]>maxdistance)
                maxdistance=distance[i*nbf+j];
        }
    }
    printf("The max distance is %f\n",maxdistance);

        printf("Pairs\n");
    for(int i=0;i<npairs;i++)
    {
        printf("%d %d %d\n",i,pair1st[i],pair2nd[i]);
    }
    printf("Pairsidx\n");
    for(int i=0;i<h2eri->n_node;i++)
    {
        for(int j=0;j<nodepairs[i]->length;j++)
        {
            printf("%d %d %d\n",i,nodepairs[i]->data[j],nodepairidx[i]->data[j]);
        }
    }


    printf("Adm pairs\n");
    for(int i=0;i<2 * h2eri->n_r_adm_pair;i++)
    {
        printf("%d %d %d\n",i, admpair1st[i],admpair2nd[i]);
    }
    printf("Nodeadmpairs\n");
    for(int i=0;i<h2eri->n_node;i++)
    {
        for(int j=0;j<nodeadmpairs[i]->length;j++)
        {
            printf("%d %d %d\n",i,nodeadmpairs[i]->data[j],nodeadmpairidx[i]->data[j]);
        }
    }

    
    H2E_dense_mat_p *Upinv;
    Upinv = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * h2eri->n_node);
    printf("Now we are going to build the Upinv\n");
    build_pinv_rmat(h2eri,Upinv);
    for(int i=0;i<h2eri->n_node;i++)
    {
        if(Upinv[i]!=NULL)
        {
            printf("The %dth Upinv has %d rows and %d columns\n",i,Upinv[i]->nrow,Upinv[i]->ncol);
        }
    }

    // Now we need to build the column basis set for every node pair including the inadmissible and self
    H2E_dense_mat_p *S51cbasis;
    S51cbasis = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * npairs);
    H2ERI_build_S5_draft(h2eri,Urbasis,Ucbasis,csrd5,csrdc5,npairs,pair1st,pair2nd,nodepairs,nodeadmpairs,nodeadmpairidx,S51cbasis,Upinv);
    for(int i=0;i<npairs;i++)
    {
        if(S51cbasis[i]!=NULL)
        {
            printf("The %dth S51cbasis has %d rows and %d columns\n",i,S51cbasis[i]->nrow,S51cbasis[i]->ncol);
        }
    }



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

    FILE *file9 = fopen("urbasis10.txt", "w");
    if (file9 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Urbasis[9]->nrow; i++) 
    {
        for(int j=0;j<Urbasis[9]->ncol;j++)
        {
            fprintf(file9, "%.16g ", Urbasis[9]->data[i*Urbasis[9]->ncol+j]);            
        }
        fprintf(file9, "\n");
    }

    fclose(file9);

    FILE *file10 = fopen("urbasis10.txt", "w");
    if (file10 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Urbasis[10]->nrow; i++) 
    {
        for(int j=0;j<Urbasis[10]->ncol;j++)
        {
            fprintf(file10, "%.16g ", Urbasis[10]->data[i*Urbasis[10]->ncol+j]);            
        }
        fprintf(file10, "\n");
    }

    fclose(file10);

    FILE *file11 = fopen("urbasis11.txt", "w");
    if (file11 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Urbasis[11]->nrow; i++) 
    {
        for(int j=0;j<Urbasis[11]->ncol;j++)
        {
            fprintf(file11, "%.16g ",Urbasis[11]->data[i*Urbasis[11]->ncol+j]);            
        }
        fprintf(file11, "\n");
    }

    fclose(file11);

    FILE *file12 = fopen("urbasis12.txt", "w");
    if (file12 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Urbasis[12]->nrow; i++) 
    {
        for(int j=0;j<Urbasis[12]->ncol;j++)
        {
            fprintf(file12, "%.16g ", Urbasis[12]->data[i*Urbasis[12]->ncol+j]);            
        }
        fprintf(file12, "\n");
    }

    fclose(file12);

    FILE *file1e3 = fopen("urbasis13.txt", "w");
    if (file1e3 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Urbasis[13]->nrow; i++) 
    {
        for(int j=0;j<Urbasis[13]->ncol;j++)
        {
            fprintf(file1e3, "%.16g ", Urbasis[13]->data[i*Urbasis[13]->ncol+j]);            
        }
        fprintf(file1e3, "\n");
    }

    fclose(file1e3);


    FILE *file13 = fopen("upinv12.txt", "w");
    if (file13 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Upinv[12]->nrow; i++) 
    {
        for(int j=0;j<Upinv[12]->ncol;j++)
        {
            fprintf(file13, "%.16g ", Upinv[12]->data[i*Upinv[12]->ncol+j]);            
        }
        fprintf(file13, "\n");
    }

    fclose(file13);

    FILE *file14 = fopen("utrans.txt", "w");
    if (file14 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < h2eri->U[12]->nrow; i++) 
    {
        for(int j=0;j<h2eri->U[12]->ncol;j++)
        {
            fprintf(file14, "%.16g ", h2eri->U[12]->data[i*h2eri->U[12]->ncol+j]);            
        }
        fprintf(file14, "\n");
    }

    fclose(file14);

    FILE *file15 = fopen("ucbss.txt", "w");
    if (file15 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < Ucbasis[15]->nrow; i++) 
    {
        for(int j=0;j<Ucbasis[15]->ncol;j++)
        {
            fprintf(file15, "%.16g ",Ucbasis[15]->data[i*Ucbasis[15]->ncol+j]);            
        }
        fprintf(file15, "\n");
    }

    fclose(file15);


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


    // Now print the sameshell information
    FILE *file2 = fopen("outputsameshell.txt", "w");
    if (file2 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<h2eri->num_sp_bfp;i++)
    {
        fprintf(file2, "%d ", h2eri->sameshell[i]);
    }

    fclose(file2);

    // Now print X and Y matrix
    FILE *file3 = fopen("outputx.txt", "w");
    if (file3 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<nbf;i++)
    {
        for(size_t j=csrd5->csrrow[i];j<csrd5->csrrow[i+1];j++)
        {
            fprintf(file3, "%d %d %.16g ", i,csrd5->csrcol[j],csrd5->csrval[j]);
            fprintf(file3, "\n");
        }
    }
    fclose(file3);
    printf("3\n");
    FILE *file4 = fopen("outputy.txt", "w");
    if (file4 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for(int i=0;i<nbf;i++)
    {
        for(size_t j=csrdc5->csrrow[i];j<csrdc5->csrrow[i+1];j++)
        {
            fprintf(file4, "%d %d %.16g ", i,csrdc5->csrcol[j],csrdc5->csrval[j]);
            fprintf(file4, "\n");
        }
    }
    fclose(file4);

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

    printf("4\n");
    COOmat_p cooh2d;
    COOmat_init(&cooh2d,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    H2ERI_build_COO_fulldensetest(h2eri,cooh2d);
    size_t nnz=cooh2d->nnz;
    
    double thres1=1e-7;
    COOmat_p cooh2d1;
    COOmat_init(&cooh2d1,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    compresscoo(cooh2d, cooh2d1, thres1);
    CSRmat_p csrh2d;
    CSRmat_init(&csrh2d,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);

    Double_COO_to_CSR( h2eri->num_bf*h2eri->num_bf,  cooh2d1->nnz, cooh2d1,csrh2d);
    printf("TestCSR\n");
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
    printf("TESTCSRd5\n");
    TestCSR(csrd5);
    printf("TESTCSRdc5\n");
    TestCSR(csrdc5);
    CSRmat_p gdle;
    CSRmat_init(&gdle,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
    double st1,et1;
    st1 = get_wtime_sec();
    Xindextransform2(h2eri->num_bf,csrh2d,csrd5,gdle);
    printf("GDLE\n");
    TestCSR(gdle);
    
    et1 = get_wtime_sec();
    CSRmat_p gdls;
    CSRmat_init(&gdls,h2eri->num_bf*h2eri->num_bf,h2eri->num_bf*h2eri->num_bf);
        
    st1 = get_wtime_sec();
    Yindextransform2(h2eri->num_bf,gdle,csrdc5,gdls);
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
    TestCSR(colgdls);
    double energy;
    energy = Calc_S1energy(gdls,colgdls);
    printf("The S1 energy is %.16g\n",energy);
    double s51energy=0;
    s51energy = calc_S51_self_interaction(h2eri, Urbasis, S51cbasis, npairs, pair1st, pair2nd);
    printf("The S51 energy is %.16g\n",s51energy);


    
    double s1s5 = 0;
    s1s5 = calc_S1S51(gdls,h2eri, Urbasis,S51cbasis, nodepairs, nodepairidx);
    printf("The S1S5 energy is %.16g\n",s1s5);

    printf("%lf\n",compute_eleval_S51(h2eri, Urbasis,S51cbasis,nodepairs, nodepairidx, 51, 252));
    printf("%lf\n",compute_eleval_S51(h2eri, Urbasis,S51cbasis,nodepairs, nodepairidx, 51, 298));

    printf("%.16g\n",s51sp[13*h2eri->num_sp_bfp+62]);
    double nm=0;
    for(int i=0;i<h2eri->num_sp_bfp;i++)
    {
        for(int j=0;j<h2eri->num_sp_bfp;j++)
        {
            nm+=s51sp[i*h2eri->num_sp_bfp+j]*s51sp[j*h2eri->num_sp_bfp+i];
        }
    }
    printf("%.16g\n",nm);

    nm=0;
    double *s51out;
    s51out=(double*) malloc(sizeof(double)*nbf*nbf*nbf*nbf);
    memset(s51out,0,sizeof(double)*nbf*nbf*nbf*nbf);
    for(int i=0;i<h2eri->num_sp_bfp;i++)
    {
        for(int j=0;j<h2eri->num_sp_bfp;j++)
        {
            int bf1st = h2eri->bf1st[i];
            int bf2nd = h2eri->bf2nd[i];
            int bf3rd = h2eri->bf1st[j];
            int bf4th = h2eri->bf2nd[j];
            s51out[bf1st*nbf*nbf*nbf+bf2nd*nbf*nbf+bf3rd*nbf+bf4th]=s51sp[i*h2eri->num_sp_bfp+j];
            if(h2eri->sameshell[i]==0)
            {
                s51out[bf2nd*nbf*nbf*nbf+bf1st*nbf*nbf+bf3rd*nbf+bf4th]=s51sp[i*h2eri->num_sp_bfp+j];
            }
        }

    }
    double *s1out;
    s1out=(double*) malloc(sizeof(double)*nbf*nbf*nbf*nbf);
    memset(s1out,0,sizeof(double)*nbf*nbf*nbf*nbf);
    for(int i=0;i<csrh2d->nrow;i++)
    {
        int bf1st = i%nbf;
        int bf2nd = i/nbf;
        for(size_t j=csrh2d->csrrow[i];j<csrh2d->csrrow[i+1];j++)
        {
            int bf3rd = csrh2d->csrcol[j]%nbf;
            int bf4th = csrh2d->csrcol[j]/nbf;
            s1out[bf1st*nbf*nbf*nbf+bf2nd*nbf*nbf+bf3rd*nbf+bf4th]=csrh2d->csrval[j];
        }
    }
    nm=0;
    for(int i=0;i<nbf*nbf*nbf*nbf;i++)
    {
        nm+=s1out[i]*s1out[i];
    }
    printf("S1d %.16g\n",nm);
    double *x5;
    x5=(double*) malloc(sizeof(double)*nbf*nbf);
    memset(x5,0,sizeof(double)*nbf*nbf);
    for(int i=0;i<nbf;i++)
    {
        for(size_t j=csrd5->csrrow[i];j<csrd5->csrrow[i+1];j++)
        {
            x5[i*nbf+csrd5->csrcol[j]]=csrd5->csrval[j];
        }
    }
    double *y5;
    y5=(double*) malloc(sizeof(double)*nbf*nbf);
    memset(y5,0,sizeof(double)*nbf*nbf);
    for(int i=0;i<nbf;i++)
    {
        for(size_t j=csrdc5->csrrow[i];j<csrdc5->csrrow[i+1];j++)
        {
            y5[i*nbf+csrdc5->csrcol[j]]=csrdc5->csrval[j];
        }
    }




    double *z5;
    z5=(double*) malloc(sizeof(double)*nbf*nbf*nbf*nbf);
    double *zhalf;
    zhalf=(double*) malloc(sizeof(double)*nbf*nbf*nbf*nbf);
    memset(z5,0,sizeof(double)*nbf*nbf*nbf*nbf);
    memset(zhalf,0,sizeof(double)*nbf*nbf*nbf*nbf);
    double *yhalf;
    yhalf=(double*) malloc(sizeof(double)*nbf*nbf);
    memset(yhalf,0,sizeof(double)*nbf*nbf);
    for(int i=0;i<nbf;i++)
    {
            yhalf[i*nbf+i]=1;
    }
    for(int i=0;i<nbf;i++)
        for(int j=0;j<nbf;j++)
            for(int k=0;k<nbf;k++)
                for(int l=0;l<nbf;l++)
                {
                    z5[i*nbf*nbf*nbf+j*nbf*nbf+k*nbf+l]=x5[i*nbf+k]*y5[j*nbf+l];
                    zhalf[i*nbf*nbf*nbf+j*nbf*nbf+k*nbf+l]=x5[i*nbf+l]*yhalf[j*nbf+k];
                }

    nm=0;
    for(int i=0;i<nbf*nbf*nbf*nbf;i++)
    {
        nm+=z5[i]*z5[i];
    }
    printf("Z5 %.16g\n",nm);
    double *prod;
    prod=(double*) malloc(sizeof(double)*nbf*nbf*nbf*nbf);
    memset(prod,0,sizeof(double)*nbf*nbf*nbf*nbf);
    double *prodhalf;
    prodhalf=(double*) malloc(sizeof(double)*nbf*nbf*nbf*nbf);
    memset(prodhalf,0,sizeof(double)*nbf*nbf*nbf*nbf);
    for(int i=0;i<nbf*nbf;i++)
        for(int j=0;j<nbf*nbf;j++)
            for(int k=0;k<nbf*nbf;k++)
            {
                prod[i*nbf*nbf+j]+=s1out[i*nbf*nbf+k]*z5[k*nbf*nbf+j];
                prodhalf[i*nbf*nbf+j]+=s1out[i*nbf*nbf+k]*zhalf[k*nbf*nbf+j];
            }
    nm=0;
    printf("%.16g\n",prod[0]);
    for(int i=0;i<nbf*nbf*nbf*nbf;i++)
    {
        nm+=prod[i]*prod[i];
    }
    printf("Prod %.16g\n",nm);
    nm=0;
    for(int i=0;i<nbf*nbf;i++)
        for(int j=0;j<nbf*nbf;j++)
        {
            nm+=prod[i*nbf*nbf+j]*prod[j*nbf*nbf+i];
        }
    printf("S1self %.16g\n",nm);
    nm=0;
    for(int i=0;i<nbf*nbf*nbf*nbf;i++)
    {
        nm+=prodhalf[i]*prodhalf[i];
    }
    printf("Prodhalf norm %.16g\n",nm);
    nm=0;
    for(int i=0;i<nbf*nbf;i++)
        for(int j=0;j<nbf*nbf;j++)
        {
            nm+=prodhalf[i*nbf*nbf+j]*prodhalf[j*nbf*nbf+i];
        }
    printf("S1selfhalf %.16g\n",nm);

    printf("gdle err\n");
    for(int i=0;i<nbf*nbf;i++)
    {
        for(size_t j=gdle->csrrow[i];j<gdle->csrrow[i+1];j++)
        {
            if(fabs(gdle->csrval[j]-prodhalf[i*nbf*nbf+gdle->csrcol[j]])>1e-9)
            {
                printf("%d %d %.16g %.16g\n",i,gdle->csrcol[j],gdle->csrval[j],prodhalf[i*nbf+gdle->csrcol[j]]);
                break;
            }
        }
    }
    printf(" output S1half\n");
    /*
    for(int i=0;i<nbf*nbf;i++)
    {
        if(prodhalf[i]!=0)
        {
            printf("%d %.16g\n",i,prodhalf[i]);
        }
    }
    */
    printf("Finish the test of the D matrix\n");

    







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
