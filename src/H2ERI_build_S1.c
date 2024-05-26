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
#include "H2ERI_build_S1.h"









void H2ERI_build_COO_fulldensetest(H2ERI_p h2eri , COOmat_p coomat)
{
    int *mat_cluster     = h2eri->mat_cluster;
    int n_leaf_node      = h2eri->n_leaf_node;
    int n_node           = h2eri->n_node;
    int n_r_inadm_pair   = h2eri->n_r_inadm_pair;
    int num_bf           = h2eri->num_bf;
    int *sp_bfp_sidx     = h2eri->sp_bfp_sidx;
    int *D_nrow          = h2eri->D_nrow;
    int *D_ncol          = h2eri->D_ncol;
    int *n_child         = h2eri->n_child;
    int num_sp = h2eri->num_sp;
    int *shell_bf_sidx = h2eri->shell_bf_sidx;
    int *sp_shell_idx  = h2eri->sp_shell_idx;
    H2E_dense_mat_p  *c_D_blks   = h2eri->c_D_blks;
    int *pt_cluster      = h2eri->pt_cluster;
    int    *leaf_nodes    = h2eri->height_nodes;
    int    *r_inadm_pairs = h2eri->r_inadm_pairs;
    size_t D0_total_size = 0;
    int * fullnrow = (int*) malloc(sizeof(int) * (n_node));
    memset(fullnrow, 0, sizeof(int) * (n_node));
    int countleaf=0;
    //Count the number of points in each node
    for (int i = 0; i < n_node; i++)
    {
        if(n_child[i]==0)
        {
            H2E_dense_mat_p Di = c_D_blks[countleaf];
            countleaf+=1;
            //size_t Di_size = Di->size;
            int startpoint = h2eri->mat_cluster[2*i];
            for(int j=0;j<Di->nrow;j++)
            {
                if(h2eri->sameshell[startpoint+j]==1)
                {
                    fullnrow[i]+=1;
                }
                else if(h2eri->sameshell[startpoint+j]==0)
                {
                    fullnrow[i]+=2;
                }
            }
        }
        else
        {
            int nchild = n_child[i];
            for(int j=0;j<nchild;j++)
            {
                fullnrow[i]+=fullnrow[h2eri->children[i*h2eri->max_child+j]];
            }
        }
    }
    size_t D1_total_size = 0;
    for (int i = 0; i < n_leaf_node; i++)
    {
        int node = leaf_nodes[i];
        D0_total_size += fullnrow[node]*fullnrow[node];
    }
    for (int i = 0; i < n_r_inadm_pair; i++)
    {
        int node0 = r_inadm_pairs[2 * i];
        int node1 = r_inadm_pairs[2 * i + 1];
        D1_total_size +=2* fullnrow[node0]*fullnrow[node1];
    }

    printf("The total number of full elements in D0 is %lu\n",D0_total_size);
    printf("The total number of elements in D0 is %lu\n",h2eri->nD0element);
    printf("The total number of full elements in D1 is %lu\n",D1_total_size);
    printf("The total number of elements in D1 is %lu\n",h2eri->nD1element);
    coomat->nnz=D0_total_size+D1_total_size;
    coomat->coorow = (int*) malloc(sizeof(int) * (coomat->nnz));
    coomat->coocol = (int*) malloc(sizeof(int) * (coomat->nnz));
    coomat->cooval = (double*) malloc(sizeof(double) * (coomat->nnz));
    ASSERT_PRINTF(coomat->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(coomat->cooval != NULL, "Failed to allocate arrays for D matrices indexing\n");
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
        //printf("The %dth leaf node\n",i);
        int node = leaf_nodes[i];
        //number of shell pairs in the node
        //       int node_npts = pt_e - pt_s + 1;
        int Di_nrow = D_nrow[i];
        int Di_ncol = D_ncol[i];
        int dspot=0;
        int pointspot=0;
        H2E_dense_mat_p Di = c_D_blks[i];
        rowspot=rowD+numdata;
        colspot=colD+numdata;
        Dataspot=DataD+numdata;
        int startpoint = h2eri->mat_cluster[2*node];
        for(int j=0;j<Di_nrow;j++)
        {
            if(h2eri->sameshell[startpoint+j]==1)
            //if(1)
            {   
                //printf("1");         
                rowidx=startpoint+j;
                mu = h2eri->bf1st[rowidx];
                nu = h2eri->bf2nd[rowidx];
                rowvalue = nu * num_bf + mu;
                for(int k=0;k<Di_ncol;k++)
                {
                    if(h2eri->sameshell[startpoint+k]==1)
                    //if(1)
                    {
                        colidx=startpoint+k;
                        lambda = h2eri->bf1st[colidx];
                        sigma = h2eri->bf2nd[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot]=rowvalue;
                        colspot[pointspot]=colvalue;
                        Dataspot[pointspot]=Di->data[dspot];
                        pointspot+=1;
                        dspot+=1;
                        //printf("1");
                    }
                    else
                    {
                        
                        colidx=startpoint+k;
                        //first value
                        lambda = h2eri->bf1st[colidx];
                        sigma = h2eri->bf2nd[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot]=rowvalue;
                        colspot[pointspot]=colvalue;
                        Dataspot[pointspot]=Di->data[dspot];
                        //second value
                        lambda = h2eri->bf2nd[colidx];
                        sigma = h2eri->bf1st[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot+1]=rowvalue;
                        colspot[pointspot+1]=colvalue;
                        Dataspot[pointspot+1]=Di->data[dspot];
                        //    Di->data[dspot]=0;
                        dspot+=1;
                        pointspot+=2;
                        //printf("2");
                    }

                }
            }
            else
            {
                //printf("2");
                for(int k=0;k<Di_ncol;k++)
                {
                    if(h2eri->sameshell[startpoint+k]==1)
                    {
                        rowidx=startpoint+j;
                        mu = h2eri->bf1st[rowidx];
                        nu = h2eri->bf2nd[rowidx];
                        rowvalue = nu * num_bf + mu;
                        colidx=startpoint+k;
                        lambda = h2eri->bf1st[colidx];
                        sigma = h2eri->bf2nd[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot]=rowvalue;
                        colspot[pointspot]=colvalue;
                        Dataspot[pointspot]=Di->data[dspot];
                        mu = h2eri->bf2nd[rowidx];
                        nu = h2eri->bf1st[rowidx];
                        rowvalue = nu * num_bf + mu;
                        rowspot[pointspot+1]=rowvalue;
                        colspot[pointspot+1]=colvalue;
                        Dataspot[pointspot+1]=Di->data[dspot];
                        pointspot+=2;
                        dspot+=1;
                        //printf("3");
                    }
                    else
                    {
                        rowidx=startpoint+j;
                        mu = h2eri->bf1st[rowidx];
                        nu = h2eri->bf2nd[rowidx];
                        rowvalue = nu * num_bf + mu;
                        colidx=startpoint+k;
                        //first value
                        lambda = h2eri->bf1st[colidx];
                        sigma = h2eri->bf2nd[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot]=rowvalue;
                        colspot[pointspot]=colvalue;
                        Dataspot[pointspot]=Di->data[dspot];
                        //second value
                        lambda = h2eri->bf2nd[colidx];
                        sigma = h2eri->bf1st[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot+1]=rowvalue;
                        colspot[pointspot+1]=colvalue;
                        Dataspot[pointspot+1]=Di->data[dspot];
                        //third value
                        mu = h2eri->bf2nd[rowidx];
                        nu = h2eri->bf1st[rowidx];
                        rowvalue = nu * num_bf + mu;
                        lambda = h2eri->bf1st[colidx];
                        sigma = h2eri->bf2nd[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot+2]=rowvalue;
                        colspot[pointspot+2]=colvalue;
                        Dataspot[pointspot+2]=Di->data[dspot];
                        lambda = h2eri->bf2nd[colidx];
                        sigma = h2eri->bf1st[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot+3]=rowvalue;
                        colspot[pointspot+3]=colvalue;
                        Dataspot[pointspot+3]=Di->data[dspot];
                        dspot+=1;
                        pointspot+=4;
                        //printf("4");
                    }

                }
            }
        }
        printf("!!");
        if(dspot != Di->size) printf("%d node Wrong\n",i);
        numdata += pointspot;
        dspot = 0;
        pointspot=0;
    }
    if(numdata != D0_total_size) printf(" whole data D0 Wrong\n");
    //if(numdata == D0_total_size) printf(" whole data D0 right\n");
    //printf("Start to store D1\n");
    int *rowN            = coomat->coorow+numdata;
    int *colN            = coomat->coocol+numdata;
    double *DataN           = coomat->cooval+numdata;
    size_t numdataN=0;
    //store the information in D1 in COO matrix
    for (int i = 0; i < n_r_inadm_pair; i++)
    {
        int node0 = r_inadm_pairs[2 * i];
        int node1 = r_inadm_pairs[2 * i + 1];
        int Di_ncol = D_ncol[i+n_leaf_node];
        int Di_nrow = D_nrow[i+n_leaf_node];
        //printf("Inadmissible pair %d has %d rows and %d cols\n",i,Di_nrow,Di_ncol);
        //printf("Full rows and columns are %d and %d\n",fullnrow[node0],fullnrow[node1]);
        //printf("The number of elements is %d\n",fullnrow[node0]*fullnrow[node1]);
        H2E_dense_mat_p Di = c_D_blks[i+n_leaf_node];
        int dspot=0;
        int pointspot=0;
        rowspot=rowN+numdataN;
        colspot=colN+numdataN;
        Dataspot=DataN+numdataN;
        int rowstartpoint = h2eri->mat_cluster[2*node0];
        int colstartpoint = h2eri->mat_cluster[2*node1];
        
        for(int j=0;j<Di_nrow;j++)
        {
            
            if(h2eri->sameshell[rowstartpoint+j]==1)
                {
                    rowidx=rowstartpoint+j;
                    mu = h2eri->bf1st[rowidx];
                    nu = h2eri->bf2nd[rowidx];
                    rowvalue = nu * num_bf + mu;
                    for(int k=0;k<Di_ncol;k++)
                    {
                        if(h2eri->sameshell[colstartpoint+k]==1)
                        {
                            colidx=colstartpoint+k;
                            lambda = h2eri->bf1st[colidx];
                            sigma = h2eri->bf2nd[colidx];
                            colvalue = sigma * num_bf + lambda;
                            rowspot[pointspot]=rowvalue;
                            colspot[pointspot]=colvalue;
                            Dataspot[pointspot]=Di->data[dspot];
                            rowspot[pointspot+1]=colvalue;
                            colspot[pointspot+1]=rowvalue;
                            Dataspot[pointspot+1]=Di->data[dspot];
                            pointspot+=2;
                            dspot+=1;
                            //printf("1");
                        }
                        else
                        {
                            colidx=colstartpoint+k;
                        //first value
                        lambda = h2eri->bf1st[colidx];
                        sigma = h2eri->bf2nd[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot]=rowvalue;
                        colspot[pointspot]=colvalue;
                        Dataspot[pointspot]=Di->data[dspot];
                        rowspot[pointspot+1]=colvalue;
                        colspot[pointspot+1]=rowvalue;
                        Dataspot[pointspot+1]=Di->data[dspot];
                        //second value
                        lambda = h2eri->bf2nd[colidx];
                        sigma = h2eri->bf1st[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot+2]=rowvalue;
                        colspot[pointspot+2]=colvalue;
                        Dataspot[pointspot+2]=Di->data[dspot];
                        rowspot[pointspot+3]=colvalue;
                        colspot[pointspot+3]=rowvalue;
                        Dataspot[pointspot+3]=Di->data[dspot];
                            pointspot+=4;
                            dspot+=1;
                            //printf("2");
                        }

                    }
                }
            else
            {
                for(int k=0;k<Di_ncol;k++)
                {
                    if(h2eri->sameshell[colstartpoint+k]==1)
                    {
                        rowidx=rowstartpoint+j;
                        mu = h2eri->bf1st[rowidx];
                        nu = h2eri->bf2nd[rowidx];
                        rowvalue = nu * num_bf + mu;
                        colidx=colstartpoint+k;
                        lambda = h2eri->bf1st[colidx];
                        sigma = h2eri->bf2nd[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot]=rowvalue;
                        colspot[pointspot]=colvalue;
                        Dataspot[pointspot]=Di->data[dspot];
                        rowspot[pointspot+1]=colvalue;
                        colspot[pointspot+1]=rowvalue;
                        Dataspot[pointspot+1]=Di->data[dspot];
                        mu = h2eri->bf2nd[rowidx];
                        nu = h2eri->bf1st[rowidx];
                        rowvalue = nu * num_bf + mu;
                        rowspot[pointspot+2]=rowvalue;
                        colspot[pointspot+2]=colvalue;
                        Dataspot[pointspot+2]=Di->data[dspot];
                        rowspot[pointspot+3]=colvalue;
                        colspot[pointspot+3]=rowvalue;
                        Dataspot[pointspot+3]=Di->data[dspot];
                        pointspot+=4;
                        dspot+=1;
                        //printf("3");
                    }
                    else
                    {
                        rowidx=rowstartpoint+j;
                        mu = h2eri->bf1st[rowidx];
                        nu = h2eri->bf2nd[rowidx];
                        rowvalue = nu * num_bf + mu;
                        colidx=colstartpoint+k;
                        //first value
                        lambda = h2eri->bf1st[colidx];
                        sigma = h2eri->bf2nd[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot]=rowvalue;
                        colspot[pointspot]=colvalue;
                        Dataspot[pointspot]=Di->data[dspot];
                        rowspot[pointspot+1]=colvalue;
                        colspot[pointspot+1]=rowvalue;
                        Dataspot[pointspot+1]=Di->data[dspot];
                        //second value
                        lambda = h2eri->bf2nd[colidx];
                        sigma = h2eri->bf1st[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot+2]=rowvalue;
                        colspot[pointspot+2]=colvalue;
                        Dataspot[pointspot+2]=Di->data[dspot];
                        rowspot[pointspot+3]=colvalue;
                        colspot[pointspot+3]=rowvalue;
                        Dataspot[pointspot+3]=Di->data[dspot];
                        //third value
                        mu = h2eri->bf2nd[rowidx];
                        nu = h2eri->bf1st[rowidx];
                        rowvalue = nu * num_bf + mu;
                        lambda = h2eri->bf1st[colidx];
                        sigma = h2eri->bf2nd[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot+4]=rowvalue;
                        colspot[pointspot+4]=colvalue;
                        Dataspot[pointspot+4]=Di->data[dspot];
                        rowspot[pointspot+5]=colvalue;
                        colspot[pointspot+5]=rowvalue;
                        Dataspot[pointspot+5]=Di->data[dspot];
                        lambda = h2eri->bf2nd[colidx];
                        sigma = h2eri->bf1st[colidx];
                        colvalue = sigma * num_bf + lambda;
                        rowspot[pointspot+6]=rowvalue;
                        colspot[pointspot+6]=colvalue;
                        Dataspot[pointspot+6]=Di->data[dspot];
                        rowspot[pointspot+7]=colvalue;
                        colspot[pointspot+7]=rowvalue;
                        Dataspot[pointspot+7]=Di->data[dspot];
                        pointspot+=8;
                        dspot+=1;
                        //printf("4");
                    }
                }
            
            }
        }
        
        

        //printf("!!");
        if(dspot != Di->size) printf("Wrong in nodes %d and %d\n",node0,node1);
        //if(dspot == Di->size) printf("Right in nodes %d and %d\n",node0,node1);
        //printf("The number of elements is %d\n",pointspot);
        numdataN += pointspot;
        dspot = 0;
        pointspot=0;
    }

    if(numdataN != D1_total_size) printf("Wrong in D1 total\n");
    //if(numdataN == D1_total_size) printf("Right in D1 total\n");
}









void H2ERI_build_COO_transpose(COOmat_p cooin, COOmat_p cooout, int nbf)
{
    int *rowin            = cooin->coorow;
    int *colin            = cooin->coocol;
    double *Datain           = cooin->cooval;
    size_t nnz = cooin->nnz;
    cooout->nnz=nnz;
    cooout->coorow = (int*) malloc(sizeof(int) * (nnz));
    cooout->coocol = (int*) malloc(sizeof(int) * (nnz));
    cooout->cooval = (double*) malloc(sizeof(double) * (nnz));
    ASSERT_PRINTF(cooout->coorow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(cooout->coocol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(cooout->cooval != NULL, "Failed to allocate arrays for D matrices indexing\n");
    int mu;
    int nu;
    int lambda;
    int sigma;
    for(size_t i=0;i<nnz;i++)
    {
        mu = rowin[i]%nbf;
        nu = rowin[i]/nbf;
        lambda = colin[i]%nbf;
        sigma = colin[i]/nbf;
        cooout->coorow[i]=sigma*nbf+mu;
        cooout->coocol[i]=nu*nbf+lambda;
        cooout->cooval[i]=Datain[i];
    }
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


void swap_int(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void swap_double(double *a, double *b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

size_t partition(int *key, double *val, size_t low, size_t high) {
    int pivot = key[high];
    size_t i = low - 1;
    for (size_t j = low; j <= high - 1; j++) {
        if (key[j] < pivot) {
            i++;
            swap_int(&key[i], &key[j]);
            swap_double(&val[i], &val[j]);
        }
    }
    swap_int(&key[i + 1], &key[high]);
    swap_double(&val[i + 1], &val[high]);
    return (i + 1);
}

void Qsort_double_long2(int *key, double *val, size_t low, size_t high) {
    if (low < high) {
        size_t pi = partition(key, val, low, high);

        Qsort_double_long(key, val, low, pi - 1);
        Qsort_double_long(key, val, pi + 1, high);
    }
}

void insertionSort(int *key, double *val, size_t l, size_t r) {
    size_t i, j;
    int keyToInsert;
    double valToInsert;
    for (i = l + 1; i < r; i++) {
        keyToInsert = key[i];
        valToInsert = val[i];
        j = i;
        // Move elements of key[l..i-1] and val[l..i-1], that are
        // greater than keyToInsert, to one position ahead
        // of their current position
        while (j > l && key[j - 1] > keyToInsert) {
            key[j] = key[j - 1];
            val[j] = val[j - 1];
            j--;
        }
        key[j] = keyToInsert;
        val[j] = valToInsert;
    }
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
    //printf("Alloc success\n");
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
    //printf("Finished csr\n");
    //*/
}

// Helper function to merge two halves
void merge(int *key, double *val, int *tempKey, double *tempVal, size_t l, size_t m, size_t r) {
    size_t i = l, j = m + 1, k = l;

    // Copy data to temporary arrays
    for (size_t idx = l; idx <= r; idx++) {
        tempKey[idx] = key[idx];
        tempVal[idx] = val[idx];
    }

    // Merge the temporary arrays back into the original arrays
    while (i <= m && j <= r) {
        if (tempKey[i] <= tempKey[j]) {
            key[k] = tempKey[i];
            val[k] = tempVal[i];
            i++;
        } else {
            key[k] = tempKey[j];
            val[k] = tempVal[j];
            j++;
        }
        k++;
    }

    // Copy remaining elements from left half, if any
    while (i <= m) {
        key[k] = tempKey[i];
        val[k] = tempVal[i];
        i++;
        k++;
    }

    // Right half elements are already in place
}

// Recursive mergesort function
void mergeSortRec(int *key, double *val, int *tempKey, double *tempVal, size_t l, size_t r) {
    if (l < r) {
        size_t m = l + (r - l) / 2; // Find the midpoint
        mergeSortRec(key, val, tempKey, tempVal, l, m);    // Sort first half
        mergeSortRec(key, val, tempKey, tempVal, m + 1, r); // Sort second half
        merge(key, val, tempKey, tempVal, l, m, r);         // Merge the sorted halves
    }
}

// Public API for mergesort
void mergeSort(int *key, double *val, size_t l, size_t r) {
    if (r > l) { // Ensure the range is valid
        size_t n = r - l + 1;
        int *tempKey = malloc(n * sizeof(int));
        double *tempVal = malloc(n * sizeof(double));

        if (tempKey && tempVal) {
            mergeSortRec(key, val, tempKey, tempVal, l, r);
        }

        free(tempKey);
        free(tempVal);
    }
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
    //printf("find maxv success, which is %f\n", maxval);
    // Then compute the number of elements that is no less than maxval*thres
    size_t nlarge=0;
    for(int i=0;i<nrow*ncol;i++)
    {
        if(fabs(mat[i])>maxval*thres+1e-10)
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
                //mat[i*ncol+j]=0;
            }
        }
    //printf("finished extract\n");
    coomat->nnz=nlarge;
    if(nlarge==pt_idx)
        return nlarge;
    else
        {
            printf("Something wrong! nlarge is %ld while ptidx is %ld\n");
            printf("nrow is %d, ncol is %d, maxv is %.16g\n",nrow,ncol,maxval);
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
    //printf("Memset success\n");
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
    //printf("Calculate element time is %.3lf (s)\n",et1-st1);
    //printf("elements computing success, the number of empty rows is %d\n", nn0);
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

void Xindextransform2(int nbf, CSRmat_p csrh2d, CSRmat_p csrden, CSRmat_p csrtrans)
{
    H2E_int_vec_p *eleidx;
    H2E_dense_mat_p *eleval;
    eleidx = (H2E_int_vec_p*) malloc(sizeof(H2E_int_vec_p) * (nbf*nbf));
    eleval = (H2E_dense_mat_p*) malloc(sizeof(H2E_dense_mat_p) * (nbf*nbf));
    int * nval;
    nval = (int*) malloc(sizeof(int) * (nbf*nbf));
    memset(nval, 0, sizeof(int) * (nbf*nbf));
    H2E_int_vec_p tmpidx;
    H2E_dense_mat_p tmpval;
    H2E_int_vec_init(&tmpidx, 64*nbf*nbf);
    memset(tmpidx->data, 0, sizeof(int) * 64*nbf*nbf);
    H2E_dense_mat_init(&tmpval, 1, 64*nbf*nbf);
    memset(tmpval->data, 0, sizeof(double) * 64*nbf*nbf);
    int tmplength=0;
    for(int i=0;i<nbf*nbf;i++)
    {
        if(csrh2d->csrrow[i+1]-csrh2d->csrrow[i]!=0)
        {
            tmplength=0;
            for(size_t j=csrh2d->csrrow[i];j<csrh2d->csrrow[i+1];j++)
            {
                int colidx=csrh2d->csrcol[j];
                //int epsilon = colidx / nbf;
                int kappa = colidx % nbf;
                tmplength += (csrden->csrrow[kappa+1]-csrden->csrrow[kappa]);
            }
            if(tmplength > 64*nbf*nbf)
            {
                printf("The length is too long!\n");
                exit(0);
            }
            //printf("In X i, erilength is %d tmplength is %lu,%lu\n",i,csrh2d->csrrow[i+1]-csrh2d->csrrow[i],tmplength);
            if(tmplength==0) continue;
            int ptr=0;
            for(size_t j=csrh2d->csrrow[i];j<csrh2d->csrrow[i+1];j++)
            {
                int colidx=csrh2d->csrcol[j];
                int epsilon = colidx / nbf;
                int kappa = colidx % nbf;
                for(int k=csrden->csrrow[kappa];k<csrden->csrrow[kappa+1];k++)
                {
                    tmpidx->data[ptr]=epsilon*nbf+csrden->csrcol[k];
                    tmpval->data[ptr]=csrh2d->csrval[j]*csrden->csrval[k];
                    ptr+=1;
                }
            }
            if(ptr!=tmplength)
            {
                printf("Something wrong!\n");
            }
            mergeSort(tmpidx->data, tmpval->data, 0, tmplength-1);
            
            int nunique=1;
            //printf("tmplength before loop: %ld\n", tmplength);
            for(int l=0;l<tmplength-1;l++)
            {
                if(tmpidx->data[l]>tmpidx->data[l+1])
                {
                    printf("Ascending order wrong! tmplength is %d\n",tmplength);
                    exit(0);
                }
                    
            }
            for(int l=0;l<tmplength-1;l++)
            {
                if(tmpidx->data[l]!=tmpidx->data[l+1])
                    nunique+=1;
            }
            //printf("The number of unique elements in row %d is %d\n",i,nunique);
            H2E_int_vec_init(&eleidx[i], nunique);
            eleidx[i]->length=nunique;
            nval[i]=nunique;
            H2E_dense_mat_init(&eleval[i], 1, nunique);
            memset(eleidx[i]->data, 0, sizeof(int) * nunique);
            memset(eleval[i]->data, 0, sizeof(double) * nunique);
            eleidx[i]->data[0]=tmpidx->data[0];
            eleval[i]->data[0]=tmpval->data[0];
            int ptrvec=0;
            for(int l=1;l<tmplength;l++)
            {
                if(tmpidx->data[l]==eleidx[i]->data[ptrvec])
                {
                    eleval[i]->data[ptrvec]+=tmpval->data[l];
                }
                else
                {
                    ptrvec+=1;
                    eleidx[i]->data[ptrvec]=tmpidx->data[l];
                    eleval[i]->data[ptrvec]=tmpval->data[l];
                }
            }
            if(ptrvec!=nunique-1)
            {
                printf("Something wrong with ptrvec!\n");
            }
            memset(tmpidx->data, 0, sizeof(int) * tmplength);
            memset(tmpval->data, 0, sizeof(double) * tmplength);
        }
        else
        {
            H2E_int_vec_init(&eleidx[i], 0);
            eleidx[i]->length=0;
            H2E_dense_mat_init(&eleval[i], 0, 0);
        }
    }
    //printf("Now destroy\n");
    H2E_int_vec_destroy(&tmpidx);
    H2E_dense_mat_destroy(&tmpval);
    //printf("Now allocate\n");
    size_t ntotal=0;
    for(int i=0; i<nbf*nbf;i++)
    {
        if(nval[i]!=0)
        {
            ntotal+=nval[i];
        }
    }
    //printf("The total number of elements in coomat : nto = %lu\n",ntotal);
    csrtrans->csrrow = (size_t*) malloc(sizeof(size_t) * (nbf*nbf+1));
    csrtrans->csrcol = (int*) malloc(sizeof(int) * (ntotal));
    csrtrans->csrval = (double*) malloc(sizeof(double) * (ntotal));
    ASSERT_PRINTF(csrtrans->csrrow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrtrans->csrcol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrtrans->csrval != NULL, "Failed to allocate arrays for D matrices indexing\n");
    csrtrans->nnz=ntotal;
    memset(csrtrans->csrrow, 0, sizeof(size_t) * (nbf*nbf+1));
    for(int i=0; i<nbf*nbf;i++)
    {
        //if(csrh2d->csrrow[i+1]-csrh2d->csrrow[i]!=0)
        {
            csrtrans->csrrow[i+1]=csrtrans->csrrow[i]+nval[i];
        }
    }
    //printf("Now fill\n");
    size_t totv=0;
    for(int i=0; i<nbf*nbf;i++)
    {
        if(nval[i]!=0)
        {
            //totv+=eleidx[i]->length;
            //printf("2The number of elements in row %d is %d, total is %lu,%lu\n",i,eleidx[i]->length,totv,csrtrans->csrrow[i+1]);
            for(size_t j=csrtrans->csrrow[i];j<csrtrans->csrrow[i+1];j++)
            {
                csrtrans->csrcol[j]=eleidx[i]->data[j-csrtrans->csrrow[i]];
                csrtrans->csrval[j]=eleval[i]->data[j-csrtrans->csrrow[i]];
            }
            
        }
    }
    //printf("Now destroy\n");
    for(int i=0; i<nbf*nbf;i++)
    {
        if(nval[i]!=0)
        {
            H2E_int_vec_destroy(&eleidx[i]);
            H2E_dense_mat_destroy(&eleval[i]);
        }
    }
    free(eleidx);
    free(eleval);
    free(nval);
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


void Yindextransform2(int nbf, CSRmat_p csrh2d, CSRmat_p csrdc, CSRmat_p csrtrans)
{
    H2E_int_vec_p *eleidx;
    H2E_dense_mat_p *eleval;
    eleidx = (H2E_int_vec_p*) malloc(sizeof(H2E_int_vec_p) * (nbf*nbf));
    eleval = (H2E_dense_mat_p*) malloc(sizeof(H2E_dense_mat_p) * (nbf*nbf));
    int * nval;
    nval = (int*) malloc(sizeof(int) * (nbf*nbf));
    memset(nval, 0, sizeof(int) * (nbf*nbf));
    H2E_int_vec_p tmpidx;
    H2E_dense_mat_p tmpval;
    H2E_int_vec_init(&tmpidx, 64*nbf*nbf);
    memset(tmpidx->data, 0, sizeof(int) * 64*nbf*nbf);
    H2E_dense_mat_init(&tmpval, 1, 64*nbf*nbf);
    memset(tmpval->data, 0, sizeof(double) * 64*nbf*nbf);
    size_t tmplength=0;
    for(int i=0;i<nbf*nbf;i++)
    {
        if(csrh2d->csrrow[i+1]-csrh2d->csrrow[i]!=0)
        {
            //printf("i %d",i);

            tmplength=0;
            for(size_t j=csrh2d->csrrow[i];j<csrh2d->csrrow[i+1];j++)
            {
                int colidx=csrh2d->csrcol[j];
                int epsilon = colidx / nbf;
                //int kappa = colidx % nbf;
                tmplength += (csrdc->csrrow[epsilon+1]-csrdc->csrrow[epsilon]);
            }

            //printf("tmplength %ld\n",tmplength);
            if(tmplength==0) continue;
            int ptr=0;
            for(size_t j=csrh2d->csrrow[i];j<csrh2d->csrrow[i+1];j++)
            {
                int colidx=csrh2d->csrcol[j];
                int epsilon = colidx / nbf;
                int kappa = colidx % nbf;
                for(int k=csrdc->csrrow[epsilon];k<csrdc->csrrow[epsilon+1];k++)
                {
                    tmpidx->data[ptr]=csrdc->csrcol[k]*nbf+kappa;
                    tmpval->data[ptr]=csrh2d->csrval[j]*csrdc->csrval[k];
                    ptr+=1;
                }
            }
            if(ptr!=tmplength)
            {
                printf("Something wrong!\n");
            }
            mergeSort(tmpidx->data, tmpval->data, 0, tmplength-1);
            int nunique=1;
            for(int l=0;l<tmplength-1;l++)
            {
                if(tmpidx->data[l]>tmpidx->data[l+1])
                {
                    printf("Ascending order wrong!\n");
                    exit(0);
                }
            }
            for(int l=0;l<tmplength-1;l++)
            {
                if(tmpidx->data[l]!=tmpidx->data[l+1])
                    nunique+=1;
            }
            //printf("The number of unique elements in row %d is %d\n",i,nunique);
            H2E_int_vec_init(&eleidx[i], nunique);
            eleidx[i]->length=nunique;
            nval[i]=nunique;
            H2E_dense_mat_init(&eleval[i], 1, nunique);
            memset(eleidx[i]->data, 0, sizeof(int) * nunique);
            memset(eleval[i]->data, 0, sizeof(double) * nunique);
            eleidx[i]->data[0]=tmpidx->data[0];
            eleval[i]->data[0]=tmpval->data[0];
            int ptrvec=0;
            for(int l=1;l<tmplength;l++)
            {
                if(tmpidx->data[l]==eleidx[i]->data[ptrvec])
                {
                    eleval[i]->data[ptrvec]+=tmpval->data[l];
                }
                else
                {
                    ptrvec+=1;
                    eleidx[i]->data[ptrvec]=tmpidx->data[l];
                    eleval[i]->data[ptrvec]=tmpval->data[l];
                }
            }
            if(ptrvec!=nunique-1)
            {
                printf("Something wrong with ptrvec!\n");
            }
            memset(tmpidx->data, 0, tmplength);
            memset(tmpval->data, 0, tmplength);
        }
        else
        {
            H2E_int_vec_init(&eleidx[i], 0);
            H2E_dense_mat_init(&eleval[i], 0, 0);
        }
    }
    size_t ntotal=0;
    for(int i=0; i<nbf*nbf;i++)
    {
        if(csrh2d->csrrow[i+1]-csrh2d->csrrow[i]!=0)
        {
            ntotal+=nval[i];
        }
    }
    //printf("The total number of elements in coomat : nto = %lu\n",ntotal);
    csrtrans->csrrow = (size_t*) malloc(sizeof(size_t) * (nbf*nbf+1));
    csrtrans->csrcol = (int*) malloc(sizeof(int) * (ntotal));
    csrtrans->csrval = (double*) malloc(sizeof(double) * (ntotal));
    ASSERT_PRINTF(csrtrans->csrrow != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrtrans->csrcol != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(csrtrans->csrval != NULL, "Failed to allocate arrays for D matrices indexing\n");
    csrtrans->nnz=ntotal;
    memset(csrtrans->csrrow, 0, sizeof(size_t) * (nbf*nbf+1));
    for(int i=0; i<nbf*nbf;i++)
    {
        //if(csrh2d->csrrow[i+1]-csrh2d->csrrow[i]!=0)
        {
            csrtrans->csrrow[i+1]=csrtrans->csrrow[i]+nval[i];
        }
    }

    size_t totv=0;
    for(int i=0; i<nbf*nbf;i++)
    {
        if(nval[i]!=0)
        {
            totv+=eleidx[i]->length;
            //printf("2The number of elements in row %d is %d, total is %lu,%lu\n",i,eleidx[i]->length,totv,csrtrans->csrrow[i+1]);
            for(size_t j=csrtrans->csrrow[i];j<csrtrans->csrrow[i+1];j++)
            {
                csrtrans->csrcol[j]=eleidx[i]->data[j-csrtrans->csrrow[i]];
                csrtrans->csrval[j]=eleval[i]->data[j-csrtrans->csrrow[i]];
            }
            
        }
    }
    for(int i=0; i<nbf*nbf;i++)
    {
        if(nval[i]!=0)
        {
            H2E_int_vec_destroy(&eleidx[i]);
            H2E_dense_mat_destroy(&eleval[i]);
        }
    }
    free(eleidx);
    free(eleval);
    free(nval);
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
    free(posi);
    printf("Transform csc success!\n");
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


