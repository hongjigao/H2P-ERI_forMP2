#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "CMS.h"
#include "H2ERI_typedef.h"
#include "H2ERI_utils.h"
#include "linalg_lib_wrapper.h"
#include "H2ERI_build_Smat.h"

void H2ERI_build_COO_Diamat(H2ERI_p h2eri , COOmat_p coomat, int D1tst, int threstest)
{
    int n_leaf_node      = h2eri->n_leaf_node;
    int n_r_inadm_pair   = h2eri->n_r_inadm_pair;
    int num_bf           = h2eri->num_bf;
    int *sp_bfp_sidx     = h2eri->sp_bfp_sidx;
//    int *D_nrow          = h2eri->D_nrow;
    int *D_ncol          = h2eri->D_ncol;
    int num_sp = h2eri->num_sp;
    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int *sp_shell_idx  = h2eri->sp_shell_idx;
    H2E_dense_mat_p  *c_D_blks   = h2eri->c_D_blks;
    int *pt_cluster      = h2eri->pt_cluster;
    int    *leaf_nodes    = h2eri->height_nodes;
    int    *r_inadm_pairs = h2eri->r_inadm_pairs;
    coomat->nnz=h2eri->nD0element+2*h2eri->nD1element;
    if(D1tst==0) coomat->nnz=h2eri->nD0element;
    coomat->coorow = (int*) malloc(sizeof(int) * (coomat->nnz));
    coomat->coocol = (int*) malloc(sizeof(int) * (coomat->nnz));
    coomat->cooval = (double*) malloc(sizeof(double) * (coomat->nnz));
    ASSERT_PRINTF(coomat->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->cooval    != NULL, "Failed to allocate arrays for D matrices indexing\n");
    int *rowD            = coomat->coorow;
    int *colD            = coomat->coocol;
    double *DataD           = coomat->cooval;
    int *rowspot;
    int *colspot;
    double *Dataspot;
    int rowspotupdate=0;
    int colspotupdate=0;
    int loopinrow;
    int loopval;
    size_t numdata=0;
    //Store the information in D0 into COO matrix
    for(int i=0;i<n_leaf_node;i++)
    {
        int node = leaf_nodes[i];
        int pt_s = pt_cluster[2 * node];
        int pt_e = pt_cluster[2 * node + 1];
        //number of shell pairs in the node
 //       int node_npts = pt_e - pt_s + 1;
//        int Di_nrow = D_nrow[i];
        int Di_ncol = D_ncol[i];
        H2E_dense_mat_p Di = c_D_blks[i];
        rowspot=rowD+numdata;
        colspot=colD+numdata;
        Dataspot=DataD+numdata;
        loopval=0;
        loopinrow=0;
        //Store the COO matrix element data sp by sp
        for(int rowsp=pt_s;rowsp<pt_e+1;rowsp++)
        {
            int rowshell0 = sp_shell_idx[rowsp];
            int rowshell1 = sp_shell_idx[rowsp+num_sp];

            for(int colsp=pt_s;colsp<pt_e+1;colsp++)
            {

                int colshell0 = sp_shell_idx[colsp];
                int colshell1 = sp_shell_idx[colsp+num_sp];
                //pt_Dblock is the pointer of C_D_bls[i]'s data of the corresponding first element in the row
                double *pt_Dblock=Di->data+rowspotupdate+colspotupdate;
                for(int mu=shell_bf_sidx[rowshell0];mu<shell_bf_sidx[rowshell0+1];mu++)
                    for(int nu=shell_bf_sidx[rowshell1];nu<shell_bf_sidx[rowshell1+1];nu++)
                    {
                        for(int lambda=shell_bf_sidx[colshell0];lambda<shell_bf_sidx[colshell0+1];lambda++)
                            for(int sigma=shell_bf_sidx[colshell1];sigma<shell_bf_sidx[colshell1+1];sigma++)
                            {
                                rowspot[loopval]=nu*num_bf+mu;
                                //Here temperatory exchange and need to double check
                                colspot[loopval]=sigma*num_bf+lambda;
                                Dataspot[loopval]=pt_Dblock[loopinrow];
                                loopinrow+=1;
                                loopval+=1;
                            }
                        pt_Dblock+=Di_ncol;
                        loopinrow=0;
                    }
                colspotupdate+=sp_bfp_sidx[colsp+1]=sp_bfp_sidx[colsp];
                        
            }
            rowspotupdate+=(sp_bfp_sidx[rowsp+1]=sp_bfp_sidx[rowsp])*Di_ncol;
            colspotupdate=0;
        }
        numdata+=loopval;
    }
    if(D1tst==0) return;
    //store the information in D1 in COO matrix
    for (int i = 0; i < n_r_inadm_pair; i++)
    {
        int node0 = r_inadm_pairs[2 * i];
        int node1 = r_inadm_pairs[2 * i + 1];
        int pt_s0 = pt_cluster[2 * node0];
        int pt_s1 = pt_cluster[2 * node1];
        int pt_e0 = pt_cluster[2 * node0 + 1];
        int pt_e1 = pt_cluster[2 * node1 + 1];
        int Di_ncol = D_ncol[i+n_leaf_node];
        H2E_dense_mat_p Di = c_D_blks[i+n_leaf_node];
        rowspot=rowD+numdata;
        colspot=colD+numdata;
        Dataspot=DataD+numdata;
        loopval=0;
        loopinrow=0;
        //Store the COO matrix element data sp by sp
        for(int rowsp=pt_s0;rowsp<pt_e0+1;rowsp++)
        {
            int rowshell0 = sp_shell_idx[rowsp];
            int rowshell1 = sp_shell_idx[rowsp+num_sp];

            for(int colsp=pt_s1;colsp<pt_e1+1;colsp++)
            {

                int colshell0 = sp_shell_idx[colsp];
                int colshell1 = sp_shell_idx[colsp+num_sp];
                //pt_Dblock is the pointer of C_D_bls[i]'s data of the corresponding first element in the row
                double *pt_Dblock=Di->data+rowspotupdate+colspotupdate;
                for(int mu=shell_bf_sidx[rowshell0];mu<shell_bf_sidx[rowshell0+1];mu++)
                    for(int nu=shell_bf_sidx[rowshell1];nu<shell_bf_sidx[rowshell1+1];nu++)
                    {
                        for(int lambda=shell_bf_sidx[colshell0];lambda<shell_bf_sidx[colshell0+1];lambda++)
                            for(int sigma=shell_bf_sidx[colshell1];sigma<shell_bf_sidx[colshell1+1];sigma++)
                            {
                                rowspot[loopval]=nu*num_bf+mu;
                                colspot[loopval]=sigma*num_bf+lambda;
                                Dataspot[loopval]=pt_Dblock[loopinrow];
                                rowspot[loopval+1]=sigma*num_bf+lambda;
                                colspot[loopval+1]=nu*num_bf+mu;
                                Dataspot[loopval+1]=pt_Dblock[loopinrow];
                                loopinrow+=1;
                                loopval+=2;
                            }
                        pt_Dblock+=Di_ncol;
                        loopinrow=0;
//                        printf("%d\n",loopval);
                    }
                colspotupdate+=sp_bfp_sidx[colsp+1]=sp_bfp_sidx[colsp];
                        
            }
            rowspotupdate+=(sp_bfp_sidx[rowsp+1]=sp_bfp_sidx[rowsp])*Di_ncol;
            colspotupdate=0;
        }
        numdata+=loopval;
    }
    printf("The number of total elements is %lu\n",coomat->nnz);
    if (threstest==0)
        return;
    size_t nnz=coomat->nnz;
    double maxv=0;
    for(size_t i=0;i<nnz;i++)
    {
        if(fabs(coomat->cooval[i])>maxv)
        {
            maxv=fabs(coomat->cooval[i]);
        }
    }
    size_t newnz=0;
    for(size_t i=0;i<nnz;i++)
    {
        if(fabs(coomat->cooval[i])>maxv*1e-9)
        {
            newnz+=1;
        }
    }
    COOmat_p coonew;
    COOmat_init(&coonew,num_bf*num_bf,num_bf*num_bf);
    coonew->nnz=newnz;
    coonew->coorow = (int*) malloc(sizeof(int) * newnz);
    coonew->coocol = (int*) malloc(sizeof(int) * newnz);
    coonew->cooval = (double*) malloc(sizeof(double) * newnz);
    ASSERT_PRINTF(coonew->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coonew->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coonew->cooval != NULL, "Failed to allocate arrays for D matrices indexing\n");
    size_t newptr=0;
    for(size_t i=0;i<nnz;i++)
    {
        if(fabs(coomat->cooval[i])>maxv*1e-9)
        {
            coonew->coorow[newptr]=coomat->coorow[i];
            coonew->coocol[newptr]=coomat->coocol[i];
            coonew->cooval[newptr]=coomat->cooval[i];
            newptr+=1;
        }
    }
    printf("nnz after prescreening is %lu",newnz);
    coomat->nnz=newnz;
    coomat->coorow = (int*) malloc(sizeof(int) * newnz);
    coomat->coocol = (int*) malloc(sizeof(int) * newnz);
    coomat->cooval = (double*) malloc(sizeof(double) * newnz);
    ASSERT_PRINTF(coomat->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->cooval != NULL, "Failed to allocate arrays for D matrices indexing\n");
    for(size_t i=0;i<newnz;i++)
    {        
        coomat->coorow[i]=coonew->coorow[i];
        coomat->coocol[i]=coonew->coocol[i];
        coomat->cooval[i]=coonew->cooval[i];      
    }
    COOmat_destroy(coonew);
}


void H2ERI_build_COO_Diamattest(H2ERI_p h2eri , COOmat_p coomat, int D1tst, int threstest)
{
    int n_leaf_node      = h2eri->n_leaf_node;
    int n_r_inadm_pair   = h2eri->n_r_inadm_pair;
    int num_bf           = h2eri->num_bf;
    int *sp_bfp_sidx     = h2eri->sp_bfp_sidx;
    int *D_nrow          = h2eri->D_nrow;
    int *D_ncol          = h2eri->D_ncol;
    int num_sp = h2eri->num_sp;
    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int *sp_shell_idx  = h2eri->sp_shell_idx;
    H2E_dense_mat_p  *c_D_blks   = h2eri->c_D_blks;
    int *pt_cluster      = h2eri->pt_cluster;
    int    *leaf_nodes    = h2eri->height_nodes;
    int    *r_inadm_pairs = h2eri->r_inadm_pairs;
    coomat->nnz=h2eri->nD0element+2*h2eri->nD1element;
    if(D1tst==0) coomat->nnz=h2eri->nD0element;
    coomat->coorow = (int*) malloc(sizeof(int) * (coomat->nnz));
    coomat->coocol = (int*) malloc(sizeof(int) * (coomat->nnz));
    coomat->cooval = (double*) malloc(sizeof(double) * (coomat->nnz));
    ASSERT_PRINTF(coomat->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->cooval    != NULL, "Failed to allocate arrays for D matrices indexing\n");
    int *rowD            = coomat->coorow;
    int *colD            = coomat->coocol;
    double *DataD           = coomat->cooval;
    int *rowspot;
    int *colspot;
    double *Dataspot;
    int rowspotupdate=0;
    int colspotupdate=0;
    int mu;
    int nu;
    int lambda;
    int sigma;
    int rowidx;
    int rowvalue;
    int colidx;
    int colvalue;

    size_t numdata=0;
    //Store the information in D0 into COO matrix
    for(int i=0;i<n_leaf_node;i++)
    {
        int node = leaf_nodes[i];
        int pt_s = pt_cluster[2 * node];
        int pt_e = pt_cluster[2 * node + 1];
        //number of shell pairs in the node
 //       int node_npts = pt_e - pt_s + 1;
        int Di_nrow = D_nrow[i];
        int Di_ncol = D_ncol[i];
        int dspot=0;
        H2E_dense_mat_p Di = c_D_blks[i];
        rowspot=rowD+numdata;
        colspot=colD+numdata;
        Dataspot=DataD+numdata;
        int startpoint = h2eri->mat_cluster[2*node];
        for(int j=0;j<Di_nrow;j++)
        {
            rowidx=startpoint+j;
            mu = h2eri->bf1st[rowidx];
            nu = h2eri->bf2nd[rowidx];
            rowvalue = nu * num_bf + mu;
            for(int k=0;k<Di_ncol;k++)
            {
                colidx=startpoint+k;
                lambda = h2eri->bf1st[colidx];
                sigma = h2eri->bf2nd[colidx];
                colvalue = sigma * num_bf + lambda;
                rowspot[dspot]=rowvalue;
                colspot[dspot]=colvalue;
                Dataspot[dspot]=Di->data[dspot];
//                Di->data[dspot]=0;
                dspot+=1;
            }
        }
        if(dspot != Di->size) printf("Wrong\n");
        numdata += dspot;
        dspot = 0;
    }
    if(D1tst==0) return;
    //store the information in D1 in COO matrix
    for (int i = 0; i < n_r_inadm_pair; i++)
    {
        int node0 = r_inadm_pairs[2 * i];
        int node1 = r_inadm_pairs[2 * i + 1];
        int vec_s0 = h2eri->mat_cluster[2 * node0];
        int vec_s1 = h2eri->mat_cluster[2 * node1];
        int Di_ncol = D_ncol[i+n_leaf_node];
        int Di_nrow = D_nrow[i+n_leaf_node];
        H2E_dense_mat_p Di = c_D_blks[i+n_leaf_node];
        rowspot=rowD+numdata;
        colspot=colD+numdata;
        Dataspot=DataD+numdata;
        int mspot=0;
        int dspot=0;
        for(int j=0;j<Di_nrow;j++)
        {
            rowidx=vec_s0+j;
            mu = h2eri->bf1st[rowidx];
            nu = h2eri->bf2nd[rowidx];
            rowvalue = nu * num_bf + mu;
            for(int k=0;k<Di_ncol;k++)
            {
                colidx=vec_s1+k;
                lambda = h2eri->bf1st[colidx];
                sigma = h2eri->bf2nd[colidx];
                colvalue = sigma * num_bf + lambda;
                rowspot[dspot]=rowvalue;
                colspot[dspot]=colvalue;
                Dataspot[dspot]=Di->data[mspot];
                rowspot[dspot+1]=colvalue;
                colspot[dspot+1]=rowvalue;
                Dataspot[dspot+1]=Di->data[mspot];
                mspot+=1;
                dspot+=2;
            }
        }
        if(mspot != Di->size) printf("Wrong\n");
        numdata += dspot;
        dspot = 0;


    }
    printf("The number of total elements is %lu and %lu\n",coomat->nnz, numdata);

    if (threstest==0)
        return;
    size_t nnz=coomat->nnz;
    double maxv=0;
    for(size_t i=0;i<nnz;i++)
    {
        if(fabs(coomat->cooval[i])>maxv)
        {
            maxv=fabs(coomat->cooval[i]);
        }
    }
    size_t newnz=0;
    for(size_t i=0;i<nnz;i++)
    {
        if(fabs(coomat->cooval[i])>maxv*1e-9)
        {
            newnz+=1;
        }
    }
    COOmat_p coonew;
    COOmat_init(&coonew,num_bf*num_bf,num_bf*num_bf);
    coonew->nnz=newnz;
    coonew->coorow = (int*) malloc(sizeof(int) * newnz);
    coonew->coocol = (int*) malloc(sizeof(int) * newnz);
    coonew->cooval = (double*) malloc(sizeof(double) * newnz);
    ASSERT_PRINTF(coonew->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coonew->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coonew->cooval != NULL, "Failed to allocate arrays for D matrices indexing\n");
    size_t newptr=0;
    for(size_t i=0;i<nnz;i++)
    {
        if(fabs(coomat->cooval[i])>maxv*1e-9)
        {
            coonew->coorow[newptr]=coomat->coorow[i];
            coonew->coocol[newptr]=coomat->coocol[i];
            coonew->cooval[newptr]=coomat->cooval[i];
            newptr+=1;
        }
    }
    printf("nnz after prescreening is %lu",newnz);
    coomat->nnz=newnz;
    coomat->coorow = (int*) malloc(sizeof(int) * newnz);
    coomat->coocol = (int*) malloc(sizeof(int) * newnz);
    coomat->cooval = (double*) malloc(sizeof(double) * newnz);
    ASSERT_PRINTF(coomat->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->cooval != NULL, "Failed to allocate arrays for D matrices indexing\n");
    for(size_t i=0;i<newnz;i++)
    {        
        coomat->coorow[i]=coonew->coorow[i];
        coomat->coocol[i]=coonew->coocol[i];
        coomat->cooval[i]=coonew->cooval[i];      
    }
    COOmat_destroy(coonew);
}

void Qsort_double_long0(int *key, double *val, int l, int r)
{
    int i = l, j = r, tmp_key;
    double tmp_val;
    int mid_key = key[(l + r) / 2];
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            tmp_key = key[i]; key[i] = key[j]; key[j] = tmp_key;
            tmp_val = val[i]; val[i] = val[j]; val[j] = tmp_val;
            i++;  j--;
        }
    }
    if (i < r) Qsort_double_long0(key, val, i, r);
    if (j > l) Qsort_double_long0(key, val, l, j);
}


void Qsort_double_long(int *key, double *val, size_t l, size_t r)
{
    size_t i = l, j = r;
    int  tmp_key;
    double tmp_val;
    int mid_key = key[(l + r) / 2];
    while (i <= j)
    {
        while (key[i] < mid_key && i<=j) i++;
        while (key[j] > mid_key && j>=i) j--;
        if (i <= j)
        {
            tmp_key = key[i]; key[i] = key[j]; key[j] = tmp_key;
            tmp_val = val[i]; val[i] = val[j]; val[j] = tmp_val;
            i++;  j--;
        }
    }
    if(i == r-1)
    {
        if(key[i]>key[r])
        {
            tmp_key = key[i]; key[i] = key[r]; key[r] = tmp_key;
            tmp_val = val[i]; val[i] = val[r]; val[r] = tmp_val;
        }
    }
    
    if(j == l+1)
    {
        if(key[j]>key[l])
        {
            tmp_key = key[j]; key[j] = key[l]; key[l] = tmp_key;
            tmp_val = val[j]; val[j] = val[l]; val[l] = tmp_val;
        }
    }
    for(size_t k=l; k<i;k++)
    {
        if(key[k]>mid_key)
        {
            printf("sorting problem\n");
            return;
        }
    }
    for(size_t k=r; k>j;k--)
    {
        if(key[k]<mid_key)
        {
            printf("sorting problem\n");
            return;
        }
    }
    if (i < r-1) Qsort_double_long(key, val, i, r);
    if (j > l+1) Qsort_double_long(key, val, l, j);
}


void Qsort_double_long1(int *key, double *val, size_t l, size_t r)
{
    if(l==r)
        return;
    int midkey = key[(l + r) / 2];
    int * tmpkey = NULL;
    double * tmpval = NULL;
    tmpkey=(int*) malloc(sizeof(int) * (r-l+1));
    tmpval=(double*) malloc(sizeof(double) * (r-l+1));
    int nlest=0;
    int nlart=0;
    int neq=0;
    for(size_t i=l; i<=r;i++)
    {
        if(key[i]<midkey)
        {
            nlest++;
        }
        else if (key[i]>midkey)
        {
            nlart++;
        }
        else if (key[i]==midkey)
        {
            neq++;
        }  
    }
//    printf("%d and %lu\n",nlest+neq+nlart,r-l+1);
    int lsptr=0;
    int eqptr=nlest;
    int lgptr=nlest+neq;
    for(size_t i=l; i<=r;i++)
    {
        if(key[i]<midkey)
        {
            tmpkey[lsptr]=key[i];
            tmpval[lsptr]=val[i];
            lsptr++;
        }
        else if (key[i]>midkey)
        {
            tmpkey[lgptr]=key[i];
            tmpval[lgptr]=val[i];
            lgptr++;
        }
        else if (key[i]==midkey)
        {
            tmpkey[eqptr]=key[i];
            tmpval[eqptr]=val[i];
            eqptr++;
        }  
    }
//    printf("here\n");
    if(lsptr!=nlest||eqptr!=nlest+neq||lgptr!=r-l+1)
    {
        printf("Problematic\n");
    }
    for(int j=0;j<r-l+1;j++)
    {
        key[l+j]=tmpkey[j];
        val[l+j]=tmpval[j];
    }
//    printf("There\n");
    free(tmpkey);
    free(tmpval);
    if(nlest>1)
        Qsort_double_long1(key,val,l,l+nlest-1);
    if(nlart>1)
        Qsort_double_long1(key,val,r-nlart+1,r);
}

void compresscoo(COOmat_p cooini, COOmat_p coofinal, double thres)
{
    size_t nva=0;
    double max = 0;
    for(size_t i=0;i<cooini->nnz;i++)
    {
        if(fabs(cooini->cooval[i])>max)
            max=fabs(cooini->cooval[i]);
    }
    for(size_t i=0;i<cooini->nnz;i++)
    {
        if(fabs(cooini->cooval[i])>max*thres)
            nva+=1;
    }
    coofinal->nnz=nva;
    coofinal->coorow = (int*) malloc(sizeof(int) * (nva));
    coofinal->coocol = (int*) malloc(sizeof(int) * (nva));
    coofinal->cooval = (double*) malloc(sizeof(double) * (nva));
    ASSERT_PRINTF(coofinal->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coofinal->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coofinal->cooval != NULL, "Failed to allocate arrays for D matrices indexing\n");
    size_t pointer=0;
    for(size_t i=0;i<cooini->nnz;i++)
    {
        if(fabs(cooini->cooval[i])>max*thres)
        {
            coofinal->coorow[pointer]=cooini->coorow[i];
            coofinal->coocol[pointer]=cooini->coocol[i];
            coofinal->cooval[pointer]=cooini->cooval[i];
            pointer +=1;
        }
    }
    COOmat_destroy(cooini);
}

void Double_COO_to_CSR(
    const int nrow, const size_t nnz, COOmat_p coomat, CSRmat_p csrmat
)
{
    // Get the number of non-zeros in each row
    int *coorow = coomat->coorow;
    int *coocol = coomat->coocol;
    double *cooval = coomat->cooval;
    csrmat->csrrow = (size_t*) malloc(sizeof(size_t) * (nrow+1));
    csrmat->csrcol = (int*) malloc(sizeof(int) * (nnz));
    csrmat->csrval = (double*) malloc(sizeof(double) * (nnz));
    ASSERT_PRINTF(csrmat->csrrow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrmat->csrcol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrmat->csrval    != NULL, "Failed to allocate arrays for D matrices indexing\n");
    size_t * csrrow=csrmat->csrrow;
    int * csrcol=csrmat->csrcol;
    double * csrval=csrmat->csrval;
    printf("Alloc success\n");
    csrmat->nnz=nnz;
    memset(csrrow, 0, sizeof(size_t) * (nrow + 1));
    for (size_t i = 0; i < nnz; i++) csrrow[coorow[i] + 1]++;
    // Calculate the displacement of 1st non-zero in each row
    for (int j = 2; j <= nrow; j++) csrrow[j] += csrrow[j - 1];
    // Use csrrow to bucket sort col[] and val[]
    for (size_t i = 0; i < nnz; i++)
    {
        size_t idx = csrrow[coorow[i]];
        csrcol[idx] = coocol[i];
        csrval[idx] = cooval[i];
        csrrow[coorow[i]]++;
    }
    // Reset csrrow
    for (int i = nrow; i >= 1; i--) csrrow[i] = csrrow[i - 1];
    csrrow[0] = 0;
        // Sort the non-zeros in each row according to column indices
        //*
    #pragma omp parallel for
    for (int i = 0; i < nrow; i++)
    {
        if(csrrow[i]<csrrow[i+1])
            Qsort_double_long1(csrcol, csrval, csrrow[i], csrrow[i + 1] - 1);
    }
    //*/
}



void Double_COO_to_CSR_nosort(
    const int nrow, const size_t nnz, COOmat_p coomat, CSRmat_p csrmat
)
{
    // Get the number of non-zeros in each row
    int *coorow = coomat->coorow;
    int *coocol = coomat->coocol;
    double *cooval = coomat->cooval;
    csrmat->csrrow = (size_t*) malloc(sizeof(size_t) * (nrow+1));
    csrmat->csrcol = (int*) malloc(sizeof(int) * (nnz));
    csrmat->csrval = (double*) malloc(sizeof(double) * (nnz));
    ASSERT_PRINTF(csrmat->csrrow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrmat->csrcol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrmat->csrval    != NULL, "Failed to allocate arrays for D matrices indexing\n");
    size_t * csrrow=csrmat->csrrow;
    int * csrcol=csrmat->csrcol;
    double * csrval=csrmat->csrval;
    csrmat->nnz=nnz;
    memset(csrrow, 0, sizeof(size_t) * (nrow + 1));
    for (size_t i = 0; i < nnz; i++) csrrow[coorow[i] + 1]++;
    // Calculate the displacement of 1st non-zero in each row
    for (int i = 2; i <= nrow; i++) csrrow[i] += csrrow[i - 1];
    printf("Now CSR[nrow+1]is %lu\n",csrrow[nrow]);
    // Use csrrow to bucket sort col[] and val[]
    for (size_t i = 0; i < nnz; i++)
    {
        size_t idx = csrrow[coorow[i]];
        csrcol[idx] = coocol[i];
        csrval[idx] = cooval[i];
        csrrow[coorow[i]]++;
    }
    printf("Now Now CSR[nrow+1]is %lu , %lu\n",csrrow[nrow],csrrow[nrow-1]);
    // Reset csrrow
    for (int i = nrow; i >= 1; i--) csrrow[i] = csrrow[i - 1];
    csrrow[0] = 0;
    printf("Finished csr\n");

}

size_t Extract_COO_DDCMat(const int nrow, const int ncol, const double thres, double * mat, COOmat_p coomat)
{
    // Firstly, go through the elements of the matrix and find the maximum absolute value
    double maxval=0;
    for(int i=0;i<nrow*ncol;i++)
    {
        if(fabs(mat[i])>maxval)
        {
            maxval=fabs(mat[i]);
        }
        
    }
    printf("find maxv success, which is %f\n", maxval);
    // Then compute the number of elements that is no less than maxval*thres
    size_t nlarge=0;
    for(int i=0;i<nrow*ncol;i++)
    {
        if(fabs(mat[i])>maxval*thres)
        {
            nlarge +=1 ;
        }
    }
    
//    coomat->nnz=nlarge;
    // Allocate 
    coomat->coorow = (int*) malloc(sizeof(int) * (nlarge));
    coomat->coocol = (int*) malloc(sizeof(int) * (nlarge));
    coomat->cooval = (double*) malloc(sizeof(double) * (nlarge));
    ASSERT_PRINTF(coomat->coorow != NULL, "Failed to allocate arrays for COO matrices indexing\n");
    ASSERT_PRINTF(coomat->coocol != NULL, "Failed to allocate arrays for COO matrices indexing\n");
    ASSERT_PRINTF(coomat->cooval    != NULL, "Failed to allocate arrays for COO matrices indexing\n");
    //printf("alloc success\n");
    // Go through the elements and extract the large points into COO and delete the value and so only the remainder left.
    size_t pt_idx=0;
    for(int i=0;i<nrow;i++)
        for(int j=0;j<ncol;j++)
        {
            if(fabs(mat[i*ncol+j])>maxval*thres+1e-10)
            {
                coomat->coorow[pt_idx]=i ;
                coomat->coocol[pt_idx]=j ;
                coomat->cooval[pt_idx]=mat[i*ncol+j];
                pt_idx += 1;
                mat[i*ncol+j]=0;
            }
        }
    printf("finished extract\n");
    coomat->nnz=nlarge;
    if(nlarge==pt_idx)
        return nlarge;
    else
        {
            printf("Something wrong!\n");
            return 0;
        }
        
}


void Xindextransform(int nbf, CSRmat_p csrh2d, CSRmat_p csrden, CSRmat_p csrtrans)
{
    //1, Compute the number of elements without compression in each row.
    // nele is the number of initial elements. nelec is the number of elements after merging
    int* nele;
    int* nelec;
    nele = (int*) malloc(sizeof(int) * (nbf*nbf));
    nelec = (int*) malloc(sizeof(int) * (nbf*nbf));
    int maxnele=0;
    memset(nele, 0, sizeof(int) * (nbf*nbf));
    memset(nelec, 0, sizeof(int) * (nbf*nbf));
    double st1,et1;
    st1 = get_wtime_sec();
    for ( int i=0;i<nbf*nbf;i++)
    {
        for(size_t j=csrh2d->csrrow[i];j<csrh2d->csrrow[i+1];j++)
        {
            int colidx=csrh2d->csrcol[j];
            int kappa = colidx % nbf;
            nele[i] += (csrden->csrrow[kappa+1]-csrden->csrrow[kappa]);
        }
        if(nele[i]>maxnele)
            maxnele=nele[i];
    }
    et1 = get_wtime_sec();
    //2, compute the number of merged elements in each row
    int * tmpcol = (int*) malloc(sizeof(int) * (maxnele));
    double * tmpval = (double*) malloc(sizeof(double) * (maxnele));
    memset(tmpcol, 0, sizeof(int) * (maxnele));
    memset(tmpval, 0, sizeof(double) * (maxnele)); 
    printf("alc success, max length is %d\n",maxnele);
    st1 = get_wtime_sec();
//    #pragma omp parallel for
    for ( int i=0;i<nbf*nbf;i++)
    {
        if(nele[i]>0)
        {
            int ptr=0;
            for(size_t j=csrh2d->csrrow[i];j<csrh2d->csrrow[i+1];j++)
            {
                int colidx=csrh2d->csrcol[j];
                int epsilon=colidx / nbf;
                int kappa = colidx % nbf;
                for(int k=csrden->csrrow[kappa];k<csrden->csrrow[kappa+1];k++)
                {
                    tmpcol[ptr]=epsilon*nbf+csrden->csrcol[k];
                    tmpval[ptr]=csrh2d->csrval[j]*csrden->csrval[k];
                    ptr+=1;
                }
            }
            Qsort_double_long1(tmpcol,tmpval,0,nele[i]-1);
            nelec[i]=1;
            for(int l=0;l<nele[i]-1;l++)
            {
                if(tmpcol[l]!=tmpcol[l+1])
                    nelec[i]+=1;
            }
        }
    }
    printf("Compute nele success\n");
    et1 = get_wtime_sec();
    printf("Calculate nelec[i] time is %.3lf (s)\n",et1-st1);
    // Allocate memory
    size_t ntotal=0;
    for(int i=0; i<nbf*nbf;i++)
    {
        ntotal+=nelec[i];
    }

    csrtrans->csrrow = (size_t*) malloc(sizeof(size_t) * (nbf*nbf+1));
    csrtrans->csrcol = (int*) malloc(sizeof(int) * (ntotal));
    csrtrans->csrval = (double*) malloc(sizeof(double) * (ntotal));
    ASSERT_PRINTF(csrtrans->csrrow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrtrans->csrcol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrtrans->csrval != NULL, "Failed to allocate arrays for D matrices indexing\n");
    printf("Allocate csrtrans success\n");
    st1 = get_wtime_sec();
    csrtrans->nnz=ntotal;
    memset(csrtrans->csrrow, 0, sizeof(size_t) * (nbf*nbf+1));
    //Calculate the CSR matrix of transformed version
    for(int i=1; i<nbf*nbf+1;i++)
    {
        csrtrans->csrrow[i]+=nelec[i-1];
    }

    for(int i=2; i<nbf*nbf+1;i++)
    {
        csrtrans->csrrow[i]+=csrtrans->csrrow[i-1];
    }
    for(int i=0;i<nbf*nbf;i++)
    {
        if(nelec[i]>0)
        {
            int ptr=0;
            for(size_t j=csrh2d->csrrow[i];j<csrh2d->csrrow[i+1];j++)
            {
                int colidx=csrh2d->csrcol[j];
                int epsilon=colidx / nbf;
                int kappa = colidx % nbf;
                for(int k=csrden->csrrow[kappa];k<csrden->csrrow[kappa+1];k++)
                {
                    tmpcol[ptr]=epsilon*nbf+csrden->csrcol[k];
                    tmpval[ptr]=csrh2d->csrval[j]*csrden->csrval[k];
                    ptr+=1;
                }
            }
            Qsort_double_long1(tmpcol,tmpval,0,nele[i]-1);
            size_t posit = csrtrans->csrrow[i];
            csrtrans->csrcol[posit]=tmpcol[0];
            csrtrans->csrval[posit]=tmpval[0];
            for(int l=1;l<nele[i];l++)
            {
                if(tmpcol[l]==tmpcol[l-1])
                {
                    csrtrans->csrval[posit]+=tmpval[l];
                }
                else
                {
                    posit +=1 ;
                    csrtrans->csrcol[posit]=tmpcol[l];
                    csrtrans->csrval[posit]=tmpval[l];
                }
            }
        }
    }
    et1 = get_wtime_sec();
    printf("Calculate element time is %.3lf (s)\n",et1-st1);
    free(nele);
    free(nelec);
    free(tmpcol);
    free(tmpval);
}


void Xindextransform1(int nbf, CSRmat_p csrh2d, CSRmat_p csrden, CSRmat_p csrtrans)
{
    // Compute the number of elements without compression in each row.
    // nele is the number of initial elements. nelec is the number of elements after merging
    int* nele;
    int* nelec;
    nele = (int*) malloc(sizeof(int) * (nbf*nbf));
    nelec = (int*) malloc(sizeof(int) * (nbf*nbf));
    memset(nele, 0, sizeof(int) * (nbf*nbf));
    memset(nelec, 0, sizeof(int) * (nbf*nbf));
    printf("Memset success\n");
    double st1,et1;
    st1 = get_wtime_sec();
    int nn0=nbf*nbf;
    for ( int i=0;i<nbf*nbf;i++)
    {
        if(csrh2d->csrrow[i+1]-csrh2d->csrrow[i]!=0)
        {
            nn0 -=1;
            for(size_t j=csrh2d->csrrow[i];j<csrh2d->csrrow[i+1];j++)
            {
                int colidx=csrh2d->csrcol[j];
                int kappa = colidx % nbf;
                nele[i] += (csrden->csrrow[kappa+1]-csrden->csrrow[kappa]);
            }
        }
        
    }
    et1 = get_wtime_sec();
    printf("Calculate element time is %.3lf (s)\n",et1-st1);
    printf("elements computing success, the number of empty rows is %d\n", nn0);
    // compute the number of merged elements in each row
    size_t nto=0;
    for(int i=0;i<nbf*nbf;i++)
    {
        nto+=nele[i];
    }
    printf("The totol number of elements in coomat : nto = %lu\n",nto);
    COOmat_p coomat;

    COOmat_init(&coomat, nbf*nbf,nbf*nbf);
    coomat->nnz=nto;
    coomat->coorow = (int*) malloc(sizeof(int) * (nto));
    coomat->coocol = (int*) malloc(sizeof(int) * (nto));
    coomat->cooval = (double*) malloc(sizeof(double) * (nto));
    ASSERT_PRINTF(coomat->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->cooval != NULL, "Failed to allocate arrays for D matrices indexing\n");
    st1 = get_wtime_sec();
    size_t ptr=0;
    for ( int i=0;i<nbf*nbf;i++)
    {
        if(nele[i]>0)
        {
            for(size_t j=csrh2d->csrrow[i];j<csrh2d->csrrow[i+1];j++)
            {
                int colidx=csrh2d->csrcol[j];
                int epsilon=colidx / nbf;
                int kappa = colidx % nbf;
                for(int k=csrden->csrrow[kappa];k<csrden->csrrow[kappa+1];k++)
                {
                    coomat->coorow[ptr]=i;
                    coomat->coocol[ptr]=epsilon*nbf+csrden->csrcol[k];
                    coomat->cooval[ptr]=csrh2d->csrval[j]*csrden->csrval[k];
                    ptr+=1;
                }
            }
        }
    }
 //   printf("Compute coo success\n");
    et1 = get_wtime_sec();
    printf("Calculate coomat time is %.3lf (s)\n",et1-st1);
    
    st1 = get_wtime_sec();
    COOmat_p tmpcoo;
    COOmat_init(&tmpcoo,coomat->nrow,coomat->ncol);
    compresscoo(coomat, tmpcoo, 1e-7);
    CSRmat_p tmpcsr;
    CSRmat_init(&tmpcsr,tmpcoo->nrow,tmpcoo->ncol);
    Double_COO_to_CSR(nbf*nbf,tmpcoo->nnz,tmpcoo,tmpcsr);
     et1 = get_wtime_sec();
//    printf("Calculate transformed csr mat time is %.3lf (s)\n",et1-st1);
/*
   int tests=0;
   size_t teste=0;

    for(int i=0;i<tmpcsr->nrow;i++)
    {
        if(tmpcsr->csrrow[i]<tmpcsr->csrrow[i+1]-1)
        {
            for(size_t j=tmpcsr->csrrow[i];j<tmpcsr->csrrow[i+1]-1;j++)
            {
                if(tmpcsr->csrcol[j]>tmpcsr->csrcol[j+1])
                {
//                    printf("Ascending order wrong!\n");
                    tests=1;
                }
                if(tmpcsr->csrcol[j]==tmpcsr->csrcol[j+1])
                {
                    teste+=1;
                }
            }
        }
    }
    if(tests==0)
        printf("Ascending order correct!\n");
    if(tests==1)
        printf("Ascending order wrong\n");
  */  
//    printf("The same value is %lu\n",teste);
    size_t ntotal=0;
    for(int i=0; i<nbf*nbf;i++)
    {
        if(nele[i]!=0)
        {
            ntotal+=1;
            nelec[i]=1;
            for(size_t j=tmpcsr->csrrow[i];j<tmpcsr->csrrow[i+1]-1;j++)
            {
                if(tmpcsr->csrcol[j]!=tmpcsr->csrcol[j+1])
                {
                    ntotal+=1;
                    nelec[i]+=1;
                }
            }
        }
        
    }
    st1 = get_wtime_sec();
    csrtrans->csrrow = (size_t*) malloc(sizeof(size_t) * (nbf*nbf+1));
    csrtrans->csrcol = (int*) malloc(sizeof(int) * (ntotal));
    csrtrans->csrval = (double*) malloc(sizeof(double) * (ntotal));
    ASSERT_PRINTF(csrtrans->csrrow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrtrans->csrcol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrtrans->csrval != NULL, "Failed to allocate arrays for D matrices indexing\n");
//    printf("Allocate csrtrans success\n");
    csrtrans->nnz=ntotal;
    printf("The number of elements in csr is %lu \n",ntotal);
    memset(csrtrans->csrrow, 0, sizeof(size_t) * (nbf*nbf+1));
    //Calculate the CSR matrix of transformed version
    for(int i=1; i<nbf*nbf+1;i++)
    {
        csrtrans->csrrow[i]+=nelec[i-1];
    }

    for(int i=2; i<nbf*nbf+1;i++)
    {
        csrtrans->csrrow[i]+=csrtrans->csrrow[i-1];
    }
//    printf("Now last line is %lu\n",csrtrans->csrrow[csrtrans->nrow]);
    for(int i=0;i<nbf*nbf;i++)
    {
        if(nelec[i]>0)
        {
            size_t posit = csrtrans->csrrow[i];
            csrtrans->csrcol[posit]=tmpcsr->csrcol[tmpcsr->csrrow[i]];
            csrtrans->csrval[posit]=tmpcsr->csrval[tmpcsr->csrrow[i]];
            for(size_t l=tmpcsr->csrrow[i]+1;l<tmpcsr->csrrow[i+1];l++)
            {
                if(tmpcsr->csrcol[l]==tmpcsr->csrcol[l-1])
                {
                    csrtrans->csrval[posit]+=tmpcsr->csrval[l];
                }
                else
                {
                    posit +=1 ;
                    csrtrans->csrcol[posit]=tmpcsr->csrcol[l];
                    csrtrans->csrval[posit]=tmpcsr->csrval[l];
                }
            }
        }
    }
//    printf("The difference should be zero:%lu\n",teste+csrtrans->nnz-tmpcsr->nnz);
    et1 = get_wtime_sec();
    printf("Calculate transformed csr elements time is %.3lf (s)\n",et1-st1);
    free(nele);
    free(nelec);
    COOmat_destroy(tmpcoo);
    CSRmat_destroy(tmpcsr);
}


void Yindextransform1(int nbf, CSRmat_p csrh2d, CSRmat_p csrdc, CSRmat_p csrtrans)
{
    // Compute the number of elements without compression in each row.
    // nele is the number of initial elements. nelec is the number of elements after merging
    int* nele;
    int* nelec;
    nele = (int*) malloc(sizeof(int) * (nbf*nbf));
    nelec = (int*) malloc(sizeof(int) * (nbf*nbf));
    memset(nele, 0, sizeof(int) * (nbf*nbf));
    memset(nelec, 0, sizeof(int) * (nbf*nbf));
//    printf("Memset success\n");
    int nn0=nbf*nbf;
    for ( int i=0;i<nbf*nbf;i++)
    {
        if(csrh2d->csrrow[i+1]-csrh2d->csrrow[i]!=0)
        {
            nn0 -=1;
            for(size_t j=csrh2d->csrrow[i];j<csrh2d->csrrow[i+1];j++)
            {
                int colidx=csrh2d->csrcol[j];
                int epsilon = colidx / nbf;
                nele[i] += (csrdc->csrrow[epsilon+1]-csrdc->csrrow[epsilon]);
            }
        }
        
    }
//    printf("elements computing success, the number of empty rows is %d\n", nn0);
    // compute the number of merged elements in each row
    size_t nto=0;
    for(int i=0;i<nbf*nbf;i++)
    {
        nto+=nele[i];
    }
    printf("The totol number of elements in coomat : nto = %lu\n",nto);
    COOmat_p coomat;

    COOmat_init(&coomat, nbf*nbf,nbf*nbf);
    coomat->nnz=nto;
    coomat->coorow = (int*) malloc(sizeof(int) * (nto));
    coomat->coocol = (int*) malloc(sizeof(int) * (nto));
    coomat->cooval = (double*) malloc(sizeof(double) * (nto));
    ASSERT_PRINTF(coomat->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->cooval != NULL, "Failed to allocate arrays for D matrices indexing\n");

    size_t ptr=0;
    for ( int i=0;i<nbf*nbf;i++)
    {
        if(nele[i]>0)
        {
            for(size_t j=csrh2d->csrrow[i];j<csrh2d->csrrow[i+1];j++)
            {
                int colidx=csrh2d->csrcol[j];
                int epsilon=colidx / nbf;
                int kappa = colidx % nbf;
                for(int k=csrdc->csrrow[epsilon];k<csrdc->csrrow[epsilon+1];k++)
                {
                    coomat->coorow[ptr]=i;
                    coomat->coocol[ptr]=csrdc->csrcol[k]*nbf+kappa;
                    coomat->cooval[ptr]=csrh2d->csrval[j]*csrdc->csrval[k];
                    ptr+=1;
                }
            }
        }
    }
 //   printf("Compute coo success\n");
//    int maxr=0;
//    int maxc=0;
    COOmat_p tmpcoo;
    COOmat_init(&tmpcoo,coomat->nrow,coomat->ncol);
    compresscoo(coomat, tmpcoo, 1e-7);
    CSRmat_p tmpcsr;
    CSRmat_init(&tmpcsr,tmpcoo->nrow,tmpcoo->ncol);
    Double_COO_to_CSR(nbf*nbf,tmpcoo->nnz,tmpcoo,tmpcsr);
//    printf("Transform succes\n");
/*
   int tests=0;
   size_t teste=0;
    for(int i=0;i<tmpcsr->nrow;i++)
    {
        if(tmpcsr->csrrow[i]<tmpcsr->csrrow[i+1]-1)
        {
            for(size_t j=tmpcsr->csrrow[i];j<tmpcsr->csrrow[i+1]-1;j++)
            {
                if(tmpcsr->csrcol[j]>tmpcsr->csrcol[j+1])
                {
//                    printf("Ascending order wrong!\n");
                    tests=1;
                }
                if(tmpcsr->csrcol[j]==tmpcsr->csrcol[j+1])
                {
                    teste+=1;
                }
            }
        }
    }
    if(tests==0)
        printf("Ascending order correct!\n");
    if(tests==1)
        printf("Ascending order wrong\n");
    
    printf("The same value is %lu\n",teste);
    */
    size_t ntotal=0;
    for(int i=0; i<nbf*nbf;i++)
    {
        if(nele[i]!=0)
        {
            ntotal+=1;
            nelec[i]=1;
            for(size_t j=tmpcsr->csrrow[i];j<tmpcsr->csrrow[i+1]-1;j++)
            {
                if(tmpcsr->csrcol[j]!=tmpcsr->csrcol[j+1])
                {
                    ntotal+=1;
                    nelec[i]+=1;
                }
            }
        }
        
    }
    csrtrans->csrrow = (size_t*) malloc(sizeof(size_t) * (nbf*nbf+1));
    csrtrans->csrcol = (int*) malloc(sizeof(int) * (ntotal));
    csrtrans->csrval = (double*) malloc(sizeof(double) * (ntotal));
    ASSERT_PRINTF(csrtrans->csrrow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrtrans->csrcol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrtrans->csrval != NULL, "Failed to allocate arrays for D matrices indexing\n");
//    printf("Allocate csrtrans success\n");
    csrtrans->nnz=ntotal;
    printf("NNZ in csr is %lu \n",ntotal);
    memset(csrtrans->csrrow, 0, sizeof(size_t) * (nbf*nbf+1));
    //Calculate the CSR matrix of transformed version
    for(int i=1; i<nbf*nbf+1;i++)
    {
        csrtrans->csrrow[i]+=nelec[i-1];
    }

    for(int i=2; i<nbf*nbf+1;i++)
    {
        csrtrans->csrrow[i]+=csrtrans->csrrow[i-1];
    }
//    printf("Now last line is %lu\n",csrtrans->csrrow[csrtrans->nrow]);
    for(int i=0;i<nbf*nbf;i++)
    {
        if(nelec[i]>0)
        {
            size_t posit = csrtrans->csrrow[i];
            csrtrans->csrcol[posit]=tmpcsr->csrcol[tmpcsr->csrrow[i]];
            csrtrans->csrval[posit]=tmpcsr->csrval[tmpcsr->csrrow[i]];
            for(size_t l=tmpcsr->csrrow[i]+1;l<tmpcsr->csrrow[i+1];l++)
            {
                if(tmpcsr->csrcol[l]==tmpcsr->csrcol[l-1])
                {
                    csrtrans->csrval[posit]+=tmpcsr->csrval[l];
                }
                else
                {
                    posit +=1 ;
                    csrtrans->csrcol[posit]=tmpcsr->csrcol[l];
                    csrtrans->csrval[posit]=tmpcsr->csrval[l];
                }
            }
        }
    }
//    printf("The difference should be zero:%lu\n",teste+csrtrans->nnz-tmpcsr->nnz);
    free(nele);
    free(nelec);
    COOmat_destroy(tmpcoo);
    CSRmat_destroy(tmpcsr);
}

void CSR_to_CSC(const int ncol, CSRmat_p csrmat, CSRmat_p cscmat)
{
    cscmat->nnz=csrmat->nnz;
    cscmat->csrrow = (size_t*) malloc(sizeof(size_t) * (ncol+1));
    cscmat->csrcol = (int*) malloc(sizeof(int) * (cscmat->nnz));
    cscmat->csrval = (double*) malloc(sizeof(double) * (cscmat->nnz));
    ASSERT_PRINTF(cscmat->csrrow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(cscmat->csrcol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(cscmat->csrval != NULL, "Failed to allocate arrays for D matrices indexing\n");
    memset(cscmat->csrrow, 0, sizeof(size_t) * (ncol + 1));
    printf("Allocate csc matrix success!\n");
    for(size_t i=0;i<cscmat->nnz;i++)
    {
        cscmat->csrrow[csrmat->csrcol[i]+1]+=1;
    }
    //printf("1\n");
    for(int j=1;j<ncol+1;j++)
    {
        cscmat->csrrow[j]+=cscmat->csrrow[j-1];
    }
    //printf("2\n");
    size_t* posi;
    int tmpcol;
    posi=(size_t*) malloc(sizeof(size_t) * (ncol+1));
    //printf("3\n");
    ASSERT_PRINTF(posi != NULL, "Failed to allocate arrays for D matrices indexing\n");
    memset(posi, 0, sizeof(size_t) * (ncol));
    //printf("before\n");
    for(int j=0;j<csrmat->nrow;j++)
    {
        if(csrmat->csrrow[j]!=csrmat->csrrow[j+1])
        {
            for(size_t i=csrmat->csrrow[j];i<csrmat->csrrow[j+1];i++)
            {
                tmpcol=csrmat->csrcol[i];
                cscmat->csrcol[cscmat->csrrow[tmpcol]+posi[tmpcol]]=j;
                cscmat->csrval[cscmat->csrrow[tmpcol]+posi[tmpcol]]=csrmat->csrval[i];
                posi[tmpcol]+=1;
            }
        }
    }
}


double Calc_S1energy(CSRmat_p csrs1, CSRmat_p cscs1)
{
    double trace = 0;
//    #pragma omp parallel for
    if(csrs1->nrow!=cscs1->ncol)
        printf("Error~!\n");
    for(int i=0;i<csrs1->nrow;i++)
    {
        if(csrs1->csrrow[i]!=csrs1->csrrow[i+1])
            if(cscs1->csrrow[i]!=cscs1->csrrow[i+1])
            {
                double testtrace=trace;
                //Compare the elements in the i-th row of csr and i-th column of csc
                size_t j=csrs1->csrrow[i];
                size_t k=cscs1->csrrow[i];
                while(j<csrs1->csrrow[i+1] && k<cscs1->csrrow[i+1])
                {
                    if(csrs1->csrcol[j]==cscs1->csrcol[k])
                    {
                        trace += csrs1->csrval[j]*cscs1->csrval[k];
                        j+=1;
                        k+=1;
                        
                    }
                    else if (csrs1->csrcol[j]<cscs1->csrcol[k])
                    {
                        j+=1;
                    }
                    else if (csrs1->csrcol[j]>cscs1->csrcol[k])
                    {
                        k+=1;
                    }                   
                }
            }
        
    }
//    printf("The energy is %f\n", trace);
    return trace;
}


