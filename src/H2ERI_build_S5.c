#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lapacke.h>

#include "H2ERI_utils.h"
#include "H2ERI_aux_structs.h"
#include "H2ERI_build_S5.h"
#include "H2ERI_build_S1.h"
#include "linalg_lib_wrapper.h"
#include "TinyDFT.h"

void H2ERI_build_rowbs(H2ERI_p h2eri, H2E_dense_mat_p* Urbasis)
{
    int n_thread       = h2eri->n_thread;
    int max_child      = h2eri->max_child;
    int n_leaf_node    = h2eri->n_leaf_node;
    int max_level      = h2eri->max_level;
    int min_adm_level  = h2eri->min_adm_level;
    int *children      = h2eri->children;
    int *n_child       = h2eri->n_child;
    int *level_n_node  = h2eri->level_n_node;
    int *level_nodes   = h2eri->level_nodes;
    int *mat_cluster   = h2eri->mat_cluster;
    H2E_thread_buf_p *thread_buf = h2eri->thread_buffs;
    H2E_dense_mat_p *U  = h2eri->U;
    //for (int i = max_level; i >= min_adm_level; i--)
    for (int i = max_level; i >= 0; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(level_i_n_node, n_thread);
        
        #pragma omp parallel num_threads(n_thread_i)
        {
            int tid = omp_get_thread_num();
            //printf("%d\n", level_i_n_node);
            thread_buf[tid]->timer = -get_wtime_sec();
            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < level_i_n_node; j++)
            {
                
                int node = level_i_nodes[j];
                int n_child_node = n_child[node];
                H2E_dense_mat_p U_node = U[node];
                //printf("nrow %d ncol %d\n", U_node->nrow, U_node->ncol);
                if(U_node->nrow == 0)
                {
                    // Empty node, the row basis set is empty
                    printf("Empty node %d, %d\n", j,node);
                    H2E_dense_mat_init(&Urbasis[node], 1, 1);
                    Urbasis[node]->nrow = 0;
                    Urbasis[node]->ncol = 0;
                    Urbasis[node]->ld   = 0;
                }
                else if (n_child_node == 0 && U_node->nrow != 0)
                {
                    //printf("Leaf node %d, %d\n",j, node);
                    // Leaf node, the row basis set is just the U matrix of this node
                    Urbasis[node] = U_node;
                } 
                else if (n_child_node != 0 && U_node->nrow != 0)
                {
                    //printf("Non-leaf node%d, %d\n",j, node);
                    // Non-leaf node, multiply the U matrices of the children nodes with R matrix
                    int *node_children = children + node * max_child;
                    int Unrow = 0;
                    int Uncol = U_node->ncol;
                    for (int k = 0; k < n_child_node; k++)
                    {
                        int child_k = node_children[k];
                        Unrow += U[child_k]->nrow;
                    }
                    H2E_dense_mat_init(&Urbasis[node], Unrow, Uncol);
                    memset(Urbasis[node]->data, 0, sizeof(DTYPE) * Unrow * Uncol);
                    int rowridx=0;
                    int rowuidx=0;
                    for (int k = 0; k < n_child_node; k++)
                    {
                        int child_k = node_children[k];
                        //Multiply U matrices of children nodes with R matrix
                        CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                                   U[child_k]->nrow, U_node->ncol, U[child_k]->ncol, 
                                   1.0, U[child_k]->data, U[child_k]->ncol, 
                                   U_node->data + rowridx * U_node->ncol, U_node->ncol, 
                                   0.0, Urbasis[node]->data + rowuidx * Urbasis[node]->ncol, Urbasis[node]->ncol);
                        rowuidx += U[child_k]->nrow;
                        rowridx += U[child_k]->ncol;
                    }
                }  // End of "if (n_child_node == 0)"
            }  // End of j loop
            thread_buf[tid]->timer += get_wtime_sec();
        }  // End of "pragma omp parallel"
        /*
        if (h2eri->print_timers == 1)
        {
            double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
            for (int i = 0; i < n_thread_i; i++)
            {
                double thread_i_timer = thread_buf[i]->timer;
                avg_t += thread_i_timer;
                max_t = MAX(max_t, thread_i_timer);
                min_t = MIN(min_t, thread_i_timer);
            }
            avg_t /= (double) n_thread_i;
            INFO_PRINTF("Matvec forward transformation: level %d, %d/%d threads, %d nodes\n", i, n_thread_i, n_thread, level_i_n_node);
            INFO_PRINTF("    min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
        }*/
    }  // End of i loop
}


void H2ERI_build_colbs(H2ERI_p h2eri, H2E_dense_mat_p* Ucbasis,int *admpair1st,int *admpair2nd,H2E_dense_mat_p *Urbasis)
{
    int n_thread       = h2eri->n_thread;
    H2E_thread_buf_p *thread_buf = h2eri->thread_buffs;
    #pragma omp parallel num_threads(n_thread)
        {
            int tid = omp_get_thread_num();
            //printf("%d\n", level_i_n_node);
            thread_buf[tid]->timer = -get_wtime_sec();
            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < h2eri->n_r_adm_pair; j++)
            {
                int node0 = admpair1st[j];
                int node1 = admpair2nd[j];
                if(Urbasis[node1]!=NULL && Urbasis[node0]!=NULL)
                {
                    int n1col = Urbasis[node1]->ncol;
                    int n0col = Urbasis[node0]->ncol;
                    int n0row = Urbasis[node0]->nrow;
                    int n1row = Urbasis[node1]->nrow;
                    H2E_dense_mat_init(&Ucbasis[j], n0col, n1row);
                    H2E_dense_mat_init(&Ucbasis[j+h2eri->n_r_adm_pair], n1col, n0row);
                    memset(Ucbasis[j]->data, 0, sizeof(DTYPE) * n0col * n1row);
                    memset(Ucbasis[j+h2eri->n_r_adm_pair]->data, 0, sizeof(DTYPE) * n1col * n0row);
                    CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasTrans, 
                               n0col, n1row, n1col, 
                               1.0, h2eri->c_B_blks[j]->data, n1col, 
                               Urbasis[node1]->data, n1col, 
                               0.0, Ucbasis[j]->data, n1row);
                    CBLAS_GEMM(CblasRowMajor, CblasTrans, CblasTrans,
                                 n1col, n0row, n0col, 
                                 1.0, h2eri->c_B_blks[j]->data, n1col, 
                                 Urbasis[node0]->data, n0col, 
                                 0.0, Ucbasis[j+h2eri->n_r_adm_pair]->data, n0row);
                }
            }  // End of j loop
            //thread_buf[tid]->timer += get_wtime_sec();
        }  // End of "pragma omp parallel"
}


void H2ERI_extract_near_large_elements(H2ERI_p h2eri, TinyDFT_p TinyDFT, CSRmat_p csrd5, CSRmat_p csrdc5, double r, double threshold)
{
    int nbf=h2eri->num_bf;
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
    COOmat_p cooden;
    COOmat_init(&cooden,h2eri->num_bf,h2eri->num_bf);
    double thres = threshold;
    size_t nden =Extract_COO_DDCMat(h2eri->num_bf, h2eri->num_bf, thres, TinyDFT->D_mat, cooden);
    COOmat_p cooden5;
    COOmat_init(&cooden5,h2eri->num_bf,h2eri->num_bf);
    size_t nden5 = 0;
    for(size_t i=0;i<nden;i++)
    {
        int row = cooden->coorow[i];
        int col = cooden->coocol[i];
        if(distance[row*nbf+col]<r*r)
        {
            nden5++;
        }
    }
    cooden5->nnz = nden5;
    cooden5->coorow = (int*) malloc(sizeof(int)*nden5);
    cooden5->coocol = (int*) malloc(sizeof(int)*nden5);
    cooden5->cooval = (DTYPE*) malloc(sizeof(DTYPE)*nden5);
    size_t index = 0;
    for(size_t i=0;i<nden;i++)
    {
        int row = cooden->coorow[i];
        int col = cooden->coocol[i];
        if(distance[row*nbf+col]<r*r)
        {
            cooden5->coorow[index] = row;
            cooden5->coocol[index] = col;
            cooden5->cooval[index] = cooden->cooval[i];
            index++;
        }
    }
    Double_COO_to_CSR( h2eri->num_bf,  nden5, cooden5,csrd5);
    COOmat_destroy(cooden);
    COOmat_destroy(cooden5);
    COOmat_p coodc;
    COOmat_init(&coodc,h2eri->num_bf,h2eri->num_bf);
    size_t ndc =Extract_COO_DDCMat(h2eri->num_bf, h2eri->num_bf, thres, TinyDFT->DC_mat, coodc);
    COOmat_p coodc5;
    COOmat_init(&coodc5,h2eri->num_bf,h2eri->num_bf);
    size_t ndc5 = 0;
    for(size_t i=0;i<ndc;i++)
    {
        int row = coodc->coorow[i];
        int col = coodc->coocol[i];
        if(distance[row*nbf+col]<r*r)
        {
            ndc5++;
        }
    }
    coodc5->nnz = ndc5;
    coodc5->coorow = (int*) malloc(sizeof(int)*ndc5);
    coodc5->coocol = (int*) malloc(sizeof(int)*ndc5);
    coodc5->cooval = (DTYPE*) malloc(sizeof(DTYPE)*ndc5);
    index = 0;
    for(size_t i=0;i<ndc;i++)
    {
        int row = coodc->coorow[i];
        int col = coodc->coocol[i];
        if(distance[row*nbf+col]<r*r)
        {
            coodc5->coorow[index] = row;
            coodc5->coocol[index] = col;
            coodc5->cooval[index] = coodc->cooval[i];
            index++;
        }
    }
    Double_COO_to_CSR( h2eri->num_bf,  ndc5, coodc5,csrdc5);
    COOmat_destroy(coodc);
    COOmat_destroy(coodc5);
    free(x);
    free(y);
    free(z);
    free(distance);

}

void compute_pseudo_inverse(double* R, int nrow, int ncol, double* R_pinv) {
    int lda = nrow;
    int ldu = nrow;
    int ldvt = ncol;
    int info;

    // Allocate memory for the decomposition
    double* S = (double*)malloc(sizeof(double) * ncol);
    double* U = (double*)malloc(sizeof(double) * nrow * nrow);
    double* VT = (double*)malloc(sizeof(double) * ncol * ncol);
    double* superb = (double*)malloc(sizeof(double) * (ncol - 1));

    // Compute the SVD of R
    info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', nrow, ncol, R, lda, S, U, ldu, VT, ldvt, superb);

    if (info > 0) {
        printf("The algorithm computing SVD failed to converge.\n");
        exit(1);
    }

    // Compute the pseudo-inverse from SVD results
    for (int i = 0; i < ncol; ++i) {
        for (int j = 0; j < nrow; ++j) {
            R_pinv[i * nrow + j] = 0.0;
            for (int k = 0; k < ncol; ++k) {
                if (S[k] > 1e-10) { // Threshold for numerical stability
                    R_pinv[i * nrow + j] += VT[i * ncol + k] * (1.0 / S[k]) * U[j * nrow + k];
                }
            }
        }
    }

    // Free allocated memory
    free(S);
    free(U);
    free(VT);
    free(superb);
}


void build_pinv_rmat(H2ERI_p h2eri, H2E_dense_mat_p* Upinv)
{
    H2E_dense_mat_p *U  = h2eri->U;
    H2E_dense_mat_p tmpr;
    H2E_dense_mat_init(&tmpr, 0, 0);
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
            int nrow = U[i]->nrow;
            int ncol = U[i]->ncol;
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
            
            
            
        }
    }
}

// return Ucbasis value
int testadmpair(H2E_int_vec_p *nodeadmpairs, H2E_int_vec_p *nodeadmpairidx, int node0, int node1)
{
    int n_admpair = nodeadmpairs[node0]->length;
    for(int i=0;i<n_admpair;i++)
    {
        if(nodeadmpairs[node0]->data[i]==node1)
        {
            return nodeadmpairidx[node0]->data[i];
        }
    }
    return -1;
}



void H2ERI_build_S5(H2ERI_p h2eri, H2E_dense_mat_p* Urbasis, H2E_dense_mat_p* Ucbasis, CSRmat_p csrd5, CSRmat_p csrdc5, int npairs, int *pair1st,
    int *pair2nd, H2E_int_vec_p *nodepairs, H2E_int_vec_p *nodeadmpairs, H2E_int_vec_p *nodeadmpairidx, H2E_dense_mat_p* S51cbasis)
{
    for(int i=0;i<npairs;i++)
    {
        int node0 = pair1st[i];
        int node1 = pair2nd[i];
        int startpoint=h2eri->mat_cluster[2*node1];
        H2E_dense_mat_init(&S51cbasis[i], Urbasis[node1]->nrow,Urbasis[node0]->ncol);
        memset(S51cbasis[i]->data, 0, sizeof(DTYPE) * Urbasis[node1]->nrow * Urbasis[node0]->ncol);
        if(h2eri->n_child[node0]==0)
        {
            for(int j=0;j<Urbasis[node1]->nrow;j++)
            {
                int idx=startpoint+j;
                int sameshell=h2eri->sameshell[idx];
                int bf1st = h2eri->bf1st[idx];
                int bf2nd = h2eri->bf2nd[idx];
                if(sameshell==1)
                //if(1)
                {
                    for(size_t k=csrd5->csrrow[bf1st];k<csrd5->csrrow[bf1st+1];k++)
                        for(size_t l=csrdc5->csrrow[bf2nd];l<csrdc5->csrrow[bf2nd+1];l++)
                            /*{
                                int gamma = csrd5->csrcol[k];
                                int delta = csrdc5->csrcol[l];
                                double value = csrd5->csrval[k]*csrdc5->csrval[l];
                                int tsfmnode = h2eri->leafidx[gamma*h2eri->num_bf+delta];
                                int colbfp = h2eri->bfpidx[gamma*h2eri->num_bf+delta];
                                if(colbfp!=-1)
                                {
                                    if(testadmpair(nodeadmpairs,nodeadmpairidx,node0,tsfmnode)!=-1)
                                    {
                                        int tsfmidx = testadmpair(nodeadmpairs,nodeadmpairidx,node0,tsfmnode);
                                    
                                    
                                        int ptr = colbfp - h2eri->mat_cluster[2*tsfmnode];
                                        for(int m=0;m<Ucbasis[tsfmidx]->nrow;m++)
                                        {
                                            S51cbasis[i]->data[j*Urbasis[node0]->ncol+m]+=value*Ucbasis[tsfmidx]->data[m*Ucbasis[tsfmidx]->ncol+ptr];
                                        }
                                    }
                                }
                                
                            
                                else if (testadmpair(nodeadmpairs,nodeadmpairidx,node0,h2eri->parent[tsfmnode])!=-1)
                                {
                                    int parent = h2eri->parent[tsfmnode];
                                    int tsfmidx = testadmpair(nodeadmpairs,nodeadmpairidx,node0,parent);
                                    int colbfp = h2eri->bfpidx[gamma*h2eri->num_bf+delta];
                                    int ptr = colbfp - h2eri->mat_cluster[2*parent];
                                    int ridx = 0;
                                    for(int s=0;s<h2eri->n_child[parent];s++)
                                        {
                                            if(h2eri->children[parent*h2eri->max_child+s]<tsfmnode)
                                            {
                                                ridx += h2eri->U[h2eri->children[parent*h2eri->max_child+s]]->ncol;
                                            }
                                        }
                                    for(int m=0;m<Ucbasis[tsfmidx]->nrow;m++)
                                        {
                                            for(int r=0;r<Urbasis[node0]->ncol;r++)
                                            {
                                                S51cbasis[i]->data[j*Urbasis[node0]->ncol+r]+=value*Ucbasis[tsfmidx]->data[m*Ucbasis[tsfmidx]->ncol+ptr]*h2eri->U[parent]->data[(ridx+r)*Urbasis[parent]->ncol+m];
                                            }
                                        }
                                }
                            }*/
                            {
                                int gamma = csrd5->csrcol[k];
                                int delta = csrdc5->csrcol[l];
                                double value = csrd5->csrval[k]*csrdc5->csrval[l];
                                int tsfmnode = h2eri->leafidx[gamma*h2eri->num_bf+delta];
                                if(tsfmnode==-1)
                                    continue;
                                if(testadmpair(nodeadmpairs,nodeadmpairidx,node0,tsfmnode)!=-1)
                                {
                                    int tsfmidx = testadmpair(nodeadmpairs,nodeadmpairidx,node0,tsfmnode);
                                    int colbfp = h2eri->bfpidx[gamma*h2eri->num_bf+delta];
                                    if(colbfp!=-1)
                                    {
                                        int ptr = colbfp - h2eri->mat_cluster[2*tsfmnode];
                                        for(int m=0;m<Ucbasis[tsfmidx]->nrow;m++)
                                        {
                                            S51cbasis[i]->data[j*Urbasis[node0]->ncol+m]+=value*Ucbasis[tsfmidx]->data[m*Ucbasis[tsfmidx]->ncol+ptr];
                                        }
                                    }
                                }
                                else if (testadmpair(nodeadmpairs,nodeadmpairidx,node0,h2eri->parent[tsfmnode])!=-1)
                                {
                                    printf("Error%d, %d, %d\n",node0,tsfmnode,testadmpair(nodeadmpairs,nodeadmpairidx,node0,h2eri->parent[tsfmnode]));
                                }
                            }
                        

                }
                else if(sameshell==0)
                {
                    for(int k=csrd5->csrrow[bf1st];k<csrd5->csrrow[bf1st+1];k++)
                        for(int l=csrdc5->csrrow[bf2nd];l<csrdc5->csrrow[bf2nd+1];l++)
                            {
                                int gamma = csrd5->csrcol[k];
                                int delta = csrdc5->csrcol[l];
                                double value = csrd5->csrval[k]*csrdc5->csrval[l];
                                int tsfmnode = h2eri->leafidx[gamma*h2eri->num_bf+delta];
                                if(tsfmnode==-1)
                                    continue;
                                if(testadmpair(nodeadmpairs,nodeadmpairidx,node0,tsfmnode)!=-1)
                                {
                                    int tsfmidx = testadmpair(nodeadmpairs,nodeadmpairidx,node0,tsfmnode);
                                    int colbfp = h2eri->bfpidx[gamma*h2eri->num_bf+delta];
                                    if(colbfp!=-1)
                                    {
                                        int ptr = colbfp - h2eri->mat_cluster[2*tsfmnode];
                                        for(int m=0;m<Ucbasis[tsfmidx]->nrow;m++)
                                        {
                                            S51cbasis[i]->data[j*Urbasis[node0]->ncol+m]+=value*Ucbasis[tsfmidx]->data[m*Ucbasis[tsfmidx]->ncol+ptr];
                                        }
                                    }
                                }
                                else if (testadmpair(nodeadmpairs,nodeadmpairidx,node0,h2eri->parent[tsfmnode])!=-1)
                                {
                                    printf("Error%d, %d, %d\n",node0,tsfmnode,testadmpair(nodeadmpairs,nodeadmpairidx,node0,h2eri->parent[tsfmnode]));
                                }
                            }
                    for(int k=csrd5->csrrow[bf2nd];k<csrd5->csrrow[bf2nd+1];k++)
                        for(int l=csrdc5->csrrow[bf1st];l<csrdc5->csrrow[bf1st+1];l++)
                            {
                                int gamma = csrd5->csrcol[k];
                                int delta = csrdc5->csrcol[l];
                                double value = csrd5->csrval[k]*csrdc5->csrval[l];
                                int tsfmnode = h2eri->leafidx[gamma*h2eri->num_bf+delta];
                                if(tsfmnode==-1)
                                    continue;
                                if(testadmpair(nodeadmpairs,nodeadmpairidx,node0,tsfmnode)!=-1)
                                {
                                    int tsfmidx = testadmpair(nodeadmpairs,nodeadmpairidx,node0,tsfmnode);
                                    int colbfp = h2eri->bfpidx[gamma*h2eri->num_bf+delta];
                                    if(colbfp!=-1)
                                    {
                                        int ptr = colbfp - h2eri->mat_cluster[2*tsfmnode];
                                        for(int m=0;m<Ucbasis[tsfmidx]->nrow;m++)
                                        {
                                            S51cbasis[i]->data[j*Urbasis[node0]->ncol+m]+=value*Ucbasis[tsfmidx]->data[m*Ucbasis[tsfmidx]->ncol+ptr];
                                        }
                                    }
                                }
                                else if (testadmpair(nodeadmpairs,nodeadmpairidx,node0,h2eri->parent[tsfmnode])!=-1)
                                {
                                    printf("Error%d, %d, %d\n",node0,tsfmnode,testadmpair(nodeadmpairs,nodeadmpairidx,node0,h2eri->parent[tsfmnode]));
                                }
                            }
                }
            }
        }
    }
}



void H2ERI_build_S5test(H2ERI_p h2eri, H2E_dense_mat_p* Urbasis, H2E_dense_mat_p* Ucbasis, CSRmat_p csrd5, CSRmat_p csrdc5, int npairs, int *pair1st,
    int *pair2nd, H2E_int_vec_p *nodepairs, H2E_int_vec_p *nodeadmpairs, H2E_int_vec_p *nodeadmpairidx, H2E_dense_mat_p* S51cbasis)
{
    for(int i=0;i<npairs;i++)
    {
        int node0 = pair1st[i];
        int node1 = pair2nd[i];
        int startpoint=h2eri->mat_cluster[2*node1];
        H2E_dense_mat_init(&S51cbasis[i], Urbasis[node1]->nrow,Urbasis[node0]->ncol);
        memset(S51cbasis[i]->data, 0, sizeof(DTYPE) * Urbasis[node1]->nrow * Urbasis[node0]->ncol);
            for(int j=0;j<Urbasis[node1]->nrow;j++)
            {
                int idx=startpoint+j;
                int sameshell=h2eri->sameshell[idx];
                int bf1st = h2eri->bf1st[idx];
                int bf2nd = h2eri->bf2nd[idx];
                    for(size_t k=csrd5->csrrow[bf1st];k<csrd5->csrrow[bf1st+1];k++)
                        for(size_t l=csrdc5->csrrow[bf2nd];l<csrdc5->csrrow[bf2nd+1];l++)
                            {
                                int gamma = csrd5->csrcol[k];
                                int delta = csrdc5->csrcol[l];
                                double value = csrd5->csrval[k]*csrdc5->csrval[l];
                                int tsfmnode = h2eri->leafidx[gamma*h2eri->num_bf+delta];
                                if(tsfmnode==-1)
                                    continue;
                                if(testadmpair(nodeadmpairs,nodeadmpairidx,node0,tsfmnode)!=-1)
                                {
                                    int tsfmidx = testadmpair(nodeadmpairs,nodeadmpairidx,node0,tsfmnode);
                                    int colbfp = h2eri->bfpidx[gamma*h2eri->num_bf+delta];
                                    if(colbfp!=-1)
                                    {
                                        int ptr = colbfp - h2eri->mat_cluster[2*tsfmnode];
                                        for(int m=0;m<Ucbasis[tsfmidx]->nrow;m++)
                                        {
                                            S51cbasis[i]->data[j*Urbasis[node0]->ncol+m]+=value*Ucbasis[tsfmidx]->data[m*Ucbasis[tsfmidx]->ncol+ptr];
                                        }
                                    }
                                }
                                else if (testadmpair(nodeadmpairs,nodeadmpairidx,node0,h2eri->parent[tsfmnode])!=-1)
                                {
                                    printf("Error%d, %d, %d\n",node0,tsfmnode,testadmpair(nodeadmpairs,nodeadmpairidx,node0,h2eri->parent[tsfmnode]));
                                }
                            }

                
            }
        
    }
}

double compute_eleval(H2ERI_p h2eri, H2E_dense_mat_p* Urbasis,H2E_dense_mat_p* S51cbasis,H2E_int_vec_p *nodepairs, H2E_int_vec_p *nodepairidx, int row, int column)
{
    //int nbf=h2eri->num_bf;
    int rowbfp=h2eri->bfpidx[row];
    int columnbfp=h2eri->bfpidx[column];
    int rowleaf = h2eri->leafidx[row];
    int columnleaf = h2eri->leafidx[column];
    double value = 0.0;
    if(rowbfp==-1||columnbfp==-1)
    {
        //printf("Error: No bfp index found\n");
        return 0.0;
    }  
    int node0 = rowleaf;
    int node1 = columnleaf;
    int pairidx = -1;
    int height = 0;
    for(int i=0;i<nodepairs[rowleaf]->length;i++)
    {
        if(nodepairs[rowleaf]->data[i]==columnleaf)
        {
            pairidx = nodepairidx[rowleaf]->data[i];
        }
    }
    if(pairidx==-1)
    {
        while(pairidx==-1 && height<h2eri->max_level-1)
        {
            node0 = h2eri->parent[node0];
            node1 = h2eri->parent[node1];
            for(int i=0;i<nodepairs[node0]->length;i++)
            {
                if(nodepairs[node0]->data[i]==node1)
                {
                    pairidx = nodepairidx[node0]->data[i];
                }
            }
            height++;
        }
    }
    if(pairidx==-1)
    {
        //printf("Error: No pair index found\n");
        return 0;
    }
    int rowptr = rowbfp - h2eri->mat_cluster[2*node0];
    int columnptr = columnbfp - h2eri->mat_cluster[2*node1];
    for(int j=0;j<Urbasis[node0]->ncol;j++)
    {
        value+=S51cbasis[pairidx]->data[columnptr*S51cbasis[pairidx]->ncol+j]*Urbasis[node0]->data[rowptr*Urbasis[node0]->ncol+j];
    }
    int gamma = h2eri->bf1st[columnbfp];
    int delta = h2eri->bf2nd[columnbfp];
    //printf("gamma: %d, delta: %d, sameshell:%d \n",gamma,delta,h2eri->sameshell[columnbfp]);
    if(gamma==column%h2eri->num_bf && h2eri->sameshell[columnbfp]==0)
    {
        value = 0.0;
    }
    if(row==573&&column==551)
    {
        printf("Computed value: %d, %d, %d, %f\n",node0,node1,pairidx,value);
    }
    return value;
}

double calc_S1S51(CSRmat_p S1, H2ERI_p h2eri, H2E_dense_mat_p* Urbasis, H2E_dense_mat_p *S51cbasis, H2E_int_vec_p *nodepairs, H2E_int_vec_p *nodepairidx)
{
    double value = 0.0;
    for(int i=0;i<S1->nrow;i++)
    {
        int bfpidx = h2eri->bfpidx[i];
        int mu = i%h2eri->num_bf;
        int nu = i/h2eri->num_bf;
        for(size_t j=S1->csrrow[i];j<S1->csrrow[i+1];j++)
            {
                value+=S1->csrval[j]*compute_eleval(h2eri, Urbasis, S51cbasis, nodepairs, nodepairidx, S1->csrcol[j],i);
            }
    }

    return value;
}


double calc_S51_self_interaction(H2ERI_p h2eri, H2E_dense_mat_p* Urbasis, H2E_dense_mat_p* S51cbasis, int npairs, int *pair1st, int *pair2nd)
{
    // Calculate leaf, inadmissible and admissible blocks seperately
    double vlf = 0.0;
    double via = 0.0;
    double vad = 0.0;
    double tmpvlf = 0.0;
    double tmpvia = 0.0;
    double tmpvad = 0.0;
    H2E_dense_mat_p tmpmat;
    H2E_dense_mat_init(&tmpmat, 0, 0);
    H2E_dense_mat_p tmpmat1;
    H2E_dense_mat_init(&tmpmat1, 0, 0);
    // Calculate leaf blocks
    for(int i=0;i<h2eri->n_leaf_node;i++)
    {
        tmpvlf = 0.0;
        int node = pair1st[i];
        int nrow = Urbasis[node]->nrow;
        int ncol = nrow;
        int nloop = Urbasis[node]->ncol;
        H2E_dense_mat_resize(tmpmat, nloop, nloop);
        memset(tmpmat->data, 0, sizeof(DTYPE) * nloop * nloop);
        for(int j=0;j<nloop;j++)
        {
            for(int k=0;k<nloop;k++)
            {
                for(int l=0;l<nrow;l++)
                {
                    tmpmat->data[j*nloop+k]+=S51cbasis[i]->data[l*nloop+j]*Urbasis[node]->data[l*nloop+k];
                }
            }
        }
        for(int j=0;j<nloop;j++)
            {
                for(int k=0;k<nloop;k++)
                {
                    tmpvlf+=tmpmat->data[j*nloop+k]*tmpmat->data[k*nloop+j];
                }
            }
        printf("tmpvlf:%d, %.16g\n",node, tmpvlf);
        vlf+=tmpvlf;
    } // end of leaf node loop
    // Calculate inadmissible blocks. Since Tr(AB)=Tr(BA), we only need to calculate the upper triangle part
    for(int i=0;i<h2eri->n_r_inadm_pair;i++)
    {
        tmpvia = 0.0;
        int node0 = pair1st[i+h2eri->n_leaf_node];
        int node1 = pair2nd[i+h2eri->n_leaf_node];
        int n1cbsidx = i+h2eri->n_leaf_node; // it means node1 column basis set index
        int n0cbsidx = i+h2eri->n_leaf_node+h2eri->n_r_inadm_pair;// it means node0 column basis set index
        int n0 = Urbasis[node0]->nrow;
        int n1 = Urbasis[node1]->nrow;
        int n0rc = Urbasis[node0]->ncol; // it means node0 row basis set columns
        int n1rc = Urbasis[node1]->ncol; // it means node1 row basis set columns
        H2E_dense_mat_resize(tmpmat1, n0rc, n1rc);
        memset(tmpmat1->data, 0, sizeof(DTYPE) * n0rc * n1rc);
        H2E_dense_mat_resize(tmpmat, n1rc, n0rc);
        memset(tmpmat->data, 0, sizeof(DTYPE) * n1rc * n0rc);
        for(int j=0;j<n1rc;j++)
        {
            for(int k=0;k<n0rc;k++)
            {
                for(int l=0;l<n0;l++)
                {
                    tmpmat->data[j*n0rc+k]+=S51cbasis[n0cbsidx]->data[l*n1rc+j]*Urbasis[node0]->data[l*n0rc+k];
                }
            }
        }
        for(int j=0;j<n0rc;j++)
        {
            for(int k=0;k<n1rc;k++)
            {
                for(int l=0;l<n1;l++)
                {
                    tmpmat1->data[j*n1rc+k]+=S51cbasis[n1cbsidx]->data[l*n0rc+j]*Urbasis[node1]->data[l*n1rc+k];
                }
            }
        }
        for(int j=0;j<n0rc;j++)
        {
            for(int k=0;k<n1rc;k++)
            {
                tmpvia+=tmpmat1->data[j*n1rc+k]*tmpmat->data[k*n0rc+j];
            }
        }
        printf("tmpvia:%d, %d, %.16g\n",node0, node1, tmpvia);
        via+=tmpvia;
    }
    via = 2.0*via;
    // Calculate admissible blocks
    //double tmpvad =
    for(int i=0;i<h2eri->n_r_adm_pair;i++)
    {
        tmpvad = 0.0;
        int node0 = pair1st[i+h2eri->n_leaf_node+2*h2eri->n_r_inadm_pair];
        int node1 = pair2nd[i+h2eri->n_leaf_node+2*h2eri->n_r_inadm_pair];
        int n1cbsidx = i+h2eri->n_leaf_node+2*h2eri->n_r_inadm_pair; // it means node1 column basis set index
        int n0cbsidx = i+h2eri->n_leaf_node+2*h2eri->n_r_inadm_pair+h2eri->n_r_adm_pair;
        int n0 = Urbasis[node0]->nrow;
        int n1 = Urbasis[node1]->nrow;
        int n0rc = Urbasis[node0]->ncol; // it means node0 row basis set columns
        int n1rc = Urbasis[node1]->ncol; // it means node1 row basis set columns
        H2E_dense_mat_resize(tmpmat, n1rc, n0rc);
        memset(tmpmat->data, 0, sizeof(DTYPE) * n1rc * n0rc);
        H2E_dense_mat_resize(tmpmat1, n0rc, n1rc);
        memset(tmpmat1->data, 0, sizeof(DTYPE) * n0rc * n1rc);
        for(int j=0;j<n1rc;j++)
        {
            for(int k=0;k<n0rc;k++)
            {
                for(int l=0;l<n0;l++)
                {
                    tmpmat->data[j*n0rc+k]+=S51cbasis[n0cbsidx]->data[l*n1rc+j]*Urbasis[node0]->data[l*n0rc+k];
                }
            }
        }
        for(int j=0;j<n0rc;j++)
        {
            for(int k=0;k<n1rc;k++)
            {
                for(int l=0;l<n1;l++)
                {
                    tmpmat1->data[j*n1rc+k]+=S51cbasis[n1cbsidx]->data[l*n0rc+j]*Urbasis[node1]->data[l*n1rc+k];
                }
            }
        }
        for(int j=0;j<n0rc;j++)
        {
            for(int k=0;k<n1rc;k++)
            {
                tmpvad+=tmpmat1->data[j*n1rc+k]*tmpmat->data[k*n0rc+j];
            }
        }

    printf("tmpvad:%d, %d, %d, %.16g\n",i, node0, node1, tmpvad);
    vad+=tmpvad; 
    }
    vad = 2.0*vad;
    double energy = vlf+via+vad;
    return energy;
}