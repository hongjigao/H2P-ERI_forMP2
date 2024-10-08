#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "CMS.h"
#include "H2ERI_typedef.h"
#include "H2ERI_build_H2.h"
#include "H2ERI_utils.h"
#include "H2ERI_aux_structs.h"
#include "H2ERI_ID_compress.h"
#include "linalg_lib_wrapper.h"

// Partition the ring area (r1 < r < r2) using multiple layers of 
// box surface and generate the same number of uniformly distributed 
// proxy points on each box surface layer [-r, r]^3. 
// Input parameters:
//   r1, r2     : Radius of ring area
//   nlayer     : Number of layers
//   npts_layer : Minimum number of proxy points on each layer
// Output parameters:
//   pp : H2E_dense_mat structure, contains coordinates of proxy points
void H2ERI_generate_proxy_point_layers(
    const double r1, const double r2, const int nlayer, 
    int npts_layer, H2E_dense_mat_p pp
)
{
    // 1. Decide the number of proxy points on each layer
    int npts_face = npts_layer / 6;
    int npts_axis = (int) ceil(sqrt((double) npts_face));
    npts_layer = 6 * npts_axis * npts_axis;
    int npts_total = nlayer * npts_layer;
    H2E_dense_mat_resize(pp, 3, npts_total);
    
    // 2. Generate a layer of proxy points on a standard [-1, 1]^3 box surface
    double h = 2.0 / (double) (npts_axis + 1);
    double *x = pp->data;
    double *y = pp->data + npts_total;
    double *z = pp->data + npts_total * 2;
    int index = 0;
    for (int i = 0; i < npts_axis; i++)
    {
        double h_i = h * (i + 1) - 1.0;
        for (int j = 0; j < npts_axis; j++)
        {
            double h_j = h * (j + 1) - 1.0;
            
            x[index + 0] = h_i;
            y[index + 0] = h_j;
            z[index + 0] = -1.0;
            
            x[index + 1] = h_i;
            y[index + 1] = h_j;
            z[index + 1] = 1.0;
            
            x[index + 2] = h_i;
            y[index + 2] = -1.0;
            z[index + 2] = h_j;
            
            x[index + 3] = h_i;
            y[index + 3] = 1.0;
            z[index + 3] = h_j;
            
            x[index + 4] = -1.0;
            y[index + 4] = h_i;
            z[index + 4] = h_j;
            
            x[index + 5] = 1.0;
            y[index + 5] = h_i;
            z[index + 5] = h_j;
            
            index += 6;
        }
    }
    // Copy the proxy points on the standard [-1, 1]^3 box surface to each layer
    size_t layer_msize = sizeof(double) * npts_layer;
    for (int i = 1; i < nlayer; i++)
    {
        memcpy(x + i * npts_layer, x, layer_msize);
        memcpy(y + i * npts_layer, y, layer_msize);
        memcpy(z + i * npts_layer, z, layer_msize);
    }
    
    // 3. Scale each layer
    int nlayer1 = MAX(nlayer - 1, 1);
    double dr = ((r2 - r1) / r1) / (double) nlayer1;
    for (int i = 0; i < nlayer; i++)
    {
        double *x_i = x + i * npts_layer;
        double *y_i = y + i * npts_layer;
        double *z_i = z + i * npts_layer;
        double r = r1 * (1.0 + i * dr);
        #pragma omp simd
        for (int j = 0; j < npts_layer; j++)
        {
            x_i[j] *= r;
            y_i[j] *= r;
            z_i[j] *= r;
        }
    }
}

// For all nodes, find shell pairs in idx_in that:
//   1. Are admissible from i-th node;
//   2. Their extents overlap with i-th node's near field boxes (super cell).
// Input parameters:
//   h2eri->num_sp    : Number of screened shell pairs (SSP)
//   h2eri->sp_center : Array, size 3 * num_sp, centers of SSP, sorted
//   h2eri->sp_extent : Array, size num_sp, extents of SSP, sorted
// Output parameters:
//   h2eri->ovlp_ff_idx : Array, size h2eri->n_node, i-th vector contains
//                        SSP indices that satisfy the requirements.
void H2ERI_calc_ovlp_ff_idx(H2ERI_p h2eri)
{
    int    n_node         = h2eri->n_node;
    int    root_idx       = h2eri->root_idx;
    int    n_point        = h2eri->n_point;    // == h2eri->num_sp
    int    min_adm_level  = h2eri->min_adm_level; 
    int    max_level      = h2eri->max_level;  // level = [0, max_level], total max_level+1 levels
    int    max_child      = h2eri->max_child;
    int    n_leaf_node    = h2eri->n_leaf_node;
    int    *children      = h2eri->children;
    int    *n_child       = h2eri->n_child;
    int    *level_nodes   = h2eri->level_nodes;
    int    *level_n_node  = h2eri->level_n_node;
    double *enbox         = h2eri->enbox;
    double *center   = h2eri->sp_center;
    double *extent   = h2eri->sp_extent;
    double *center_x = center;
    double *center_y = center + n_point;
    double *center_z = center + n_point * 2;
    
    // 1. Initialize ovlp_ff_idx
    h2eri->ovlp_ff_idx = (H2E_int_vec_p *) malloc(sizeof(H2E_int_vec_p) * n_node);
    assert(h2eri->ovlp_ff_idx != NULL);
    H2E_int_vec_p *ovlp_ff_idx = h2eri->ovlp_ff_idx;
    for (int i = 0; i < n_node; i++)
        H2E_int_vec_init(&ovlp_ff_idx[i], n_point);  // Won't exceed n_point
    ovlp_ff_idx[root_idx]->length = n_point;
    for (int i = 0; i < n_point; i++)
        ovlp_ff_idx[root_idx]->data[i] = i;
    
    // 2. Hierarchical partition of all centers
    for (int i = 0; i < max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            int node_n_child = n_child[node];
            if (node_n_child == 0) continue;
            int *child_nodes = children + node * max_child;
            
            H2E_int_vec_p tmp_ff_idx = ovlp_ff_idx[node];
            int n_tmp_ff_idx = tmp_ff_idx->length;
            for (int k = 0; k < node_n_child; k++)
            {
                int child_k = child_nodes[k];
                double *enbox_k = enbox + 6 * child_k;
                double enbox_center[3];
                enbox_center[0] = enbox_k[0] + 0.5 * enbox_k[3];
                enbox_center[1] = enbox_k[1] + 0.5 * enbox_k[4];
                enbox_center[2] = enbox_k[2] + 0.5 * enbox_k[5];
                
                // Half width of current node's super cell
                double sup_coef = 0.5 + ALPHA_SUP;
                double sup_semi_L[3];
                sup_semi_L[0] = sup_coef * enbox_k[3];
                sup_semi_L[1] = sup_coef * enbox_k[4];
                sup_semi_L[2] = sup_coef * enbox_k[5];
                
                for (int l = 0; l < n_tmp_ff_idx; l++)
                {
                    int ff_idx_l = tmp_ff_idx->data[l];
                    double extent_l = extent[ff_idx_l];
                    
                    // Left corner of each center's extent box to the left 
                    // corner of current child node's super cell box
                    double rel_x = center_x[ff_idx_l] - enbox_center[0];
                    double rel_y = center_y[ff_idx_l] - enbox_center[1];
                    double rel_z = center_z[ff_idx_l] - enbox_center[2];
                    rel_x += sup_semi_L[0] - extent_l;
                    rel_y += sup_semi_L[1] - extent_l;
                    rel_z += sup_semi_L[2] - extent_l;
                    
                    int left_x  = (rel_x <  0);
                    int left_y  = (rel_y <  0);
                    int left_z  = (rel_z <  0);
                    int right_x = (rel_x >= 0);
                    int right_y = (rel_y >= 0);
                    int right_z = (rel_z >= 0);
                    int adm_left_x  = (fabs(rel_x) >= 2.0 * extent_l - 1e-8);
                    int adm_left_y  = (fabs(rel_y) >= 2.0 * extent_l - 1e-8);
                    int adm_left_z  = (fabs(rel_z) >= 2.0 * extent_l - 1e-8);
                    int adm_right_x = (fabs(rel_x) >= 2.0 * sup_semi_L[0] - 1e-8);
                    int adm_right_y = (fabs(rel_y) >= 2.0 * sup_semi_L[1] - 1e-8);
                    int adm_right_z = (fabs(rel_z) >= 2.0 * sup_semi_L[2] - 1e-8);
                    int adm_x = ((left_x && adm_left_x) || (right_x && adm_right_x));
                    int adm_y = ((left_y && adm_left_y) || (right_y && adm_right_y));
                    int adm_z = ((left_z && adm_left_z) || (right_z && adm_right_z));
                    int inadm = (!(adm_x || adm_y || adm_z));
                    if (inadm)
                    {
                        int tail = ovlp_ff_idx[child_k]->length;
                        ovlp_ff_idx[child_k]->data[tail] = ff_idx_l;
                        ovlp_ff_idx[child_k]->length++;
                    }
                }  // End of l loop
            }  // End of k loop
        }  // End of j loop
    }  // End of i loop
    
    // 3. Remove centers that are in each node's inadmissible neighbor nodes
    int *tmp_ff_idx = (int *) malloc(sizeof(int) * n_point);
    assert(tmp_ff_idx != NULL);
    for (int i = min_adm_level; i <= max_level; i++)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            double enbox_center[3], adm_semi_L[3];
            double *node_enbox = enbox + 6 * node;
            double adm_coef = 0.5 + ALPHA_H2;
            enbox_center[0] = node_enbox[0] + 0.5 * node_enbox[3];
            enbox_center[1] = node_enbox[1] + 0.5 * node_enbox[4];
            enbox_center[2] = node_enbox[2] + 0.5 * node_enbox[5];
            adm_semi_L[0] = adm_coef * node_enbox[3];
            adm_semi_L[1] = adm_coef * node_enbox[4];
            adm_semi_L[2] = adm_coef * node_enbox[5];
            
            int *ff_idx  = ovlp_ff_idx[node]->data;
            int n_ff_idx = ovlp_ff_idx[node]->length;
            int ff_cnt = 0;
            for (int l = 0; l < n_ff_idx; l++)
            {
                int ff_idx_l = ff_idx[l];
                double rel_x = fabs(center_x[ff_idx_l] - enbox_center[0]);
                double rel_y = fabs(center_y[ff_idx_l] - enbox_center[1]);
                double rel_z = fabs(center_z[ff_idx_l] - enbox_center[2]);
                int adm_x = (rel_x > adm_semi_L[0]);
                int adm_y = (rel_y > adm_semi_L[1]);
                int adm_z = (rel_z > adm_semi_L[2]);
                if (adm_x || adm_y || adm_z)
                {
                    tmp_ff_idx[ff_cnt] = ff_idx_l;
                    ff_cnt++;
                }
            }
            memcpy(ff_idx, tmp_ff_idx, sizeof(int) * ff_cnt);
            ovlp_ff_idx[node]->length = ff_cnt;
        }  // End of j loop
    }  // End of i loop
    free(tmp_ff_idx);
}

// Extract shell pair and row indices of a target row index set from
// a given set of FISP
// Input parameters:
//   sp       : Array, size num_sp, SSP
//   row_idx  : Vector, target row indices set
//   sp_idx   : Vector, given SSP set 
//   work_buf : Vector, work buffer
// Output parameters:
//   pair_idx    : Vector, SSP indices that contains target row indices set
//   row_idx_new : Vector, target row new indices in pair_idx SSP
void H2ERI_extract_shell_pair_idx(
    const multi_sp_t *sp, H2E_int_vec_p row_idx,
    H2E_int_vec_p sp_idx,   H2E_int_vec_p work_buf,
    H2E_int_vec_p pair_idx, H2E_int_vec_p row_idx_new
)
{
    int num_target = row_idx->length;
    int num_sp     = sp_idx->length;
    
    H2E_int_vec_set_capacity(work_buf, num_sp * 5 + num_target + 2);
    int *nbf1    = work_buf->data;
    int *nbf2    = nbf1 + num_sp;
    int *off12   = nbf2 + num_sp;
    int *sp_flag = off12 + (num_sp + 1);
    int *tmp_idx = sp_flag + num_sp;
    int *idx_off = tmp_idx + num_target;
    
    off12[0] = 0;
    for (int i = 0; i < num_sp; i++)
    {
        const multi_sp_t *sp_i = sp + sp_idx->data[i];
        nbf1[i] = NCART(sp_i->am1);
        nbf2[i] = NCART(sp_i->am2);
        off12[i + 1] = off12[i] + nbf1[i] * nbf2[i];
    }
    
    memset(sp_flag, 0, sizeof(int) * num_sp);
    for (int i = 0; i < num_target; i++)
    {
        int j = 0, x = row_idx->data[i];
        for (j = 0; j < num_sp; j++) 
            if (off12[j] <= x && x < off12[j + 1]) break;
        tmp_idx[i] = j;
        sp_flag[j] = 1;
    }
    
    H2E_int_vec_set_capacity(pair_idx, num_sp);
    int npair = 0;
    for (int i = 0; i < num_sp; i++)
    {
        if (sp_flag[i])
        {
            pair_idx->data[npair] = i;
            sp_flag[i] = npair;
            npair++;
        }
    }
    pair_idx->length = npair;
    
    idx_off[0] = 0;
    for (int i = 0; i < npair; i++) 
    {
        int spidx = pair_idx->data[i];
        idx_off[i + 1] = idx_off[i] + nbf1[spidx] * nbf2[spidx];
    }
    
    H2E_int_vec_set_capacity(row_idx_new, num_target);
    for (int i = 0; i < num_target; i++)
    {
        int sp_idx1 = tmp_idx[i];
        int sp_idx2 = sp_flag[sp_idx1];
        row_idx_new->data[i] = row_idx->data[i] - off12[sp_idx1] + idx_off[sp_idx2];
    }
    row_idx_new->length = num_target;
}

typedef enum
{
    U_BUILD_ERI_TIMER_IDX = 0,
    U_BUILD_NAI_TIMER_IDX,
    U_BUILD_SPMM_TIMER_IDX,
    U_BUILD_QRID_TIMER_IDX,
    U_BUILD_OTHER_TIMER_IDX
} u_build_timer_idx_t;

// Build H2 projection matrices using proxy points
// Input parameter:
//   h2eri : H2ERI structure with point partitioning & shell pair info
// Output parameter:
//   h2eri : H2ERI structure with H2 projection blocks
void H2ERI_build_UJ_proxy(H2ERI_p h2eri)
{
    int    n_thread       = h2eri->n_thread;
    int    n_point        = h2eri->n_point;
    int    n_node         = h2eri->n_node;
    int    n_leaf_node    = h2eri->n_leaf_node;
    int    min_adm_level  = h2eri->min_adm_level;
    int    max_level      = h2eri->max_level;
    int    max_child      = h2eri->max_child;
    int    num_sp         = h2eri->num_sp;
    int    pp_npts_layer  = h2eri->pp_npts_layer;
    int    pp_nlayer_ext  = h2eri->pp_nlayer_ext;
    int    *children      = h2eri->children;
    int    *n_child       = h2eri->n_child;
    int    *level_nodes   = h2eri->level_nodes;
    int    *level_n_node  = h2eri->level_n_node;
    int    *node_level    = h2eri->node_level;
    int    *leaf_nodes    = h2eri->height_nodes;
    int    *pt_cluster    = h2eri->pt_cluster;
    int    *sp_nbfp       = h2eri->sp_nbfp;
    int    *index_seq     = h2eri->index_seq;
    double *enbox         = h2eri->enbox;
    double *box_extent    = h2eri->box_extent;
    size_t *mat_size      = h2eri->mat_size;
    void   *stop_param    = &h2eri->QR_stop_tol;
    multi_sp_t *sp = h2eri->sp;
    shell_t *sp_shells = h2eri->sp_shells;
    H2E_thread_buf_p *thread_buf      = h2eri->thread_buffs;
    simint_buff_p    *simint_buffs    = h2eri->simint_buffs;
    eri_batch_buff_p *eri_batch_buffs = h2eri->eri_batch_buffs;
    
    // 1. Allocate U and J
    h2eri->n_UJ  = n_node;
    h2eri->U      = (H2E_dense_mat_p*) malloc(sizeof(H2E_dense_mat_p) * n_node);
    h2eri->J_pair = (H2E_int_vec_p*)   malloc(sizeof(H2E_int_vec_p)   * n_node);
    h2eri->J_row  = (H2E_int_vec_p*)   malloc(sizeof(H2E_int_vec_p)   * n_node);
    assert(h2eri->U != NULL && h2eri->J_pair != NULL && h2eri->J_row != NULL);
    for (int i = 0; i < h2eri->n_UJ; i++)
    {
        h2eri->U[i]      = NULL;
        h2eri->J_pair[i] = NULL;
        h2eri->J_row[i]  = NULL;
    }
    H2E_dense_mat_p *U      = h2eri->U;
    H2E_int_vec_p   *J_pair = h2eri->J_pair;
    H2E_int_vec_p   *J_row  = h2eri->J_row;
    
    // 2. Calculate overlapping far field (admissible) shell pairs
    //    and auxiliary information for updating skel_flag on each level
    // skel_flag  : Marks if a point is a skeleton point on the current level
    // lvl_leaf   : Leaf nodes above the i-th level
    // lvl_n_leaf : Number of leaf nodes above the i-th level
    int n_level = max_level + 1;
    int *skel_flag  = (int *) malloc(sizeof(int) * n_point);
    int *lvl_leaf   = (int *) malloc(sizeof(int) * n_leaf_node * n_level);
    int *lvl_n_leaf = (int *) malloc(sizeof(int) * n_level);
    assert(skel_flag != NULL && lvl_leaf != NULL && lvl_n_leaf != NULL);
    // At the leaf-node level, all points are skeleton points
    for (int i = 0; i < n_point; i++) skel_flag[i] = 1;
    memset(lvl_n_leaf, 0, sizeof(int) * n_level);
    for (int i = 0; i < n_leaf_node; i++)
    {
        int leaf_i  = leaf_nodes[i];
        int level_i = node_level[leaf_i];
        for (int j = level_i + 1; j <= max_level; j++)
        {
            int idx = lvl_n_leaf[j];
            lvl_leaf[j * n_leaf_node + idx] = leaf_i;
            lvl_n_leaf[j]++;
        }
    }
    H2ERI_calc_ovlp_ff_idx(h2eri);
    H2E_int_vec_p *ovlp_ff_idx = h2eri->ovlp_ff_idx;
    
    // 3. Allocate thread-local buffers
    H2E_int_vec_p   *tb_idx = (H2E_int_vec_p *)   malloc(sizeof(H2E_int_vec_p)   * n_thread * 10);
    H2E_dense_mat_p *tb_mat = (H2E_dense_mat_p *) malloc(sizeof(H2E_dense_mat_p) * n_thread * 4);
    assert(tb_idx != NULL && tb_mat != NULL);
    for (int i = 0; i < n_thread * 10; i++)
        H2E_int_vec_init(tb_idx + i, 1024);
    for (int i = 0; i < n_thread * 4; i++)
        H2E_dense_mat_init(tb_mat + i, 1024, 1);
    size_t U_timers_msize = sizeof(double) * n_thread * 8;
    double *U_timers = (double *) malloc_aligned(U_timers_msize, 64);
    assert(U_timers != NULL);
    
    // 4. Hierarchical construction level by level
    for (int i = max_level; i >= min_adm_level; i--)
    {
        int *level_i_nodes = level_nodes + i * n_leaf_node;
        int level_i_n_node = level_n_node[i];
        int n_thread_i = MIN(n_thread, level_i_n_node);
        
        memset(U_timers, 0, U_timers_msize);
        
        // A. Compress at the i-th level
        
        #pragma omp parallel num_threads(n_thread_i)
        {
            int tid = omp_get_thread_num();

            H2E_int_vec_p   *tid_idx = tb_idx + tid * 10;
            H2E_dense_mat_p *tid_mat = tb_mat + tid * 4;
            H2E_int_vec_p    pair_idx        = tid_idx[0];
            H2E_int_vec_p    row_idx         = tid_idx[1];
            H2E_int_vec_p    node_ff_idx     = tid_idx[2];
            H2E_int_vec_p    ID_buff         = tid_idx[2];
            H2E_int_vec_p    sub_idx         = tid_idx[3];
            H2E_int_vec_p    rndmatA_idx     = tid_idx[4];
            H2E_int_vec_p    sub_row_idx     = tid_idx[2];
            H2E_int_vec_p    sub_pair        = tid_idx[4];
            H2E_int_vec_p    work_buf1       = tid_idx[5];
            H2E_int_vec_p    work_buf2       = tid_idx[6];
            H2E_int_vec_p    rndmatA_idx_cup = tid_idx[7];
            H2E_int_vec_p    rndmatA_idx1    = tid_idx[8];
            H2E_int_vec_p    node_ff_idx1    = tid_idx[9];
            H2E_dense_mat_p  pp              = tid_mat[0];
            H2E_dense_mat_p  A_ff_pp         = tid_mat[1];
            H2E_dense_mat_p  A_block         = tid_mat[2];
            H2E_dense_mat_p  QR_buff         = tid_mat[0];
            H2E_dense_mat_p  rndmatA_val     = tid_mat[3];
            simint_buff_p    simint_buff     = simint_buffs[tid];
            eri_batch_buff_p eri_batch_buff  = eri_batch_buffs[tid];
            double *timers = U_timers + tid * 8;
            
            double st, et;
            
            thread_buf[tid]->timer = -get_wtime_sec();
            #pragma omp for schedule(dynamic) nowait
            for (int j = 0; j < level_i_n_node; j++)
            {
                int node = level_i_nodes[j];
                int node_n_child = n_child[node];
                int *child_nodes = children + node * max_child;
                
                // (1) Construct row subset for this node
                st = get_wtime_sec();
                if (node_n_child == 0)
                {
                    int pt_s = pt_cluster[2 * node];
                    int pt_e = pt_cluster[2 * node + 1];
                    int node_npts = pt_e - pt_s + 1;
                    H2E_int_vec_set_capacity(pair_idx, node_npts);
                    memcpy(pair_idx->data, index_seq + pt_s, sizeof(int) * node_npts);
                    pair_idx->length = node_npts;
                    
                    int nbfp = H2ERI_gather_sum_int(sp_nbfp, pair_idx->length, pair_idx->data);
                    H2E_int_vec_set_capacity(row_idx, nbfp);
                    for (int k = 0; k < nbfp; k++) row_idx->data[k] = k;
                    row_idx->length = nbfp;
                } else {
                    int row_idx_offset = 0;
                    pair_idx->length = 0;
                    row_idx->length  = 0;
                    for (int k = 0; k < node_n_child; k++)
                    {
                        int child_k = child_nodes[k];
                        H2E_int_vec_concatenate(pair_idx, J_pair[child_k]);
                        int row_idx_spos = row_idx->length;
                        int row_idx_epos = row_idx_spos + J_row[child_k]->length;
                        H2E_int_vec_concatenate(row_idx, J_row[child_k]);
                        for (int l = row_idx_spos; l < row_idx_epos; l++)
                            row_idx->data[l] += row_idx_offset;
                        row_idx_offset += H2ERI_gather_sum_int(sp_nbfp, J_pair[child_k]->length, J_pair[child_k]->data);
                    }
                }  // End of "if (node_n_child == 0)"
                
                // (2) Generate proxy points
                //st = get_wtime_sec();
                double *node_enbox = enbox + 6 * node;
                double width  = node_enbox[3];
                double extent = box_extent[node];
                double r1 = width * (0.5 + ALPHA_SUP);
                double r2 = width * (0.5 + extent);
                //changed! added max 
                double d_nlayer = MAX((extent - ALPHA_SUP) * (pp_nlayer_ext - 1),0);
                //double d_nlayer = (extent - ALPHA_SUP) * (pp_nlayer_ext - 1);
                int nlayer_node = 1 + ceil(d_nlayer);
                H2ERI_generate_proxy_point_layers(r1, r2, nlayer_node, pp_npts_layer, pp);
                int num_pp = pp->ncol;
                double *pp_x = pp->data;
                double *pp_y = pp->data + num_pp;
                double *pp_z = pp->data + num_pp * 2;
                double center_x = node_enbox[0] + 0.5 * node_enbox[3];
                double center_y = node_enbox[1] + 0.5 * node_enbox[4];
                double center_z = node_enbox[2] + 0.5 * node_enbox[5];
                #pragma omp simd
                for (int k = 0; k < num_pp; k++)
                {
                    pp_x[k] += center_x;
                    pp_y[k] += center_y;
                    pp_z[k] += center_z;
                }
                //et = get_wtime_sec();
                //timers[U_BUILD_OTHER_TIMER_IDX] += et - st;
                
                // (3) Prepare current node's overlapping far field point list
                //st = get_wtime_sec();
                int n_ff_idx0 = ovlp_ff_idx[node]->length;
                int *ff_idx0  = ovlp_ff_idx[node]->data;
                int n_ff_idx  = H2ERI_gather_sum_int(skel_flag, ovlp_ff_idx[node]->length, ovlp_ff_idx[node]->data);
                H2E_int_vec_set_capacity(node_ff_idx, n_ff_idx);
                n_ff_idx = 0;
                for (int k = 0; k < n_ff_idx0; k++)
                {
                    int l = ff_idx0[k];
                    if (skel_flag[l] == 1)
                    {
                        node_ff_idx->data[n_ff_idx] = l;
                        n_ff_idx++;
                    }
                }
                node_ff_idx->length = n_ff_idx;
                et = get_wtime_sec();
                timers[U_BUILD_OTHER_TIMER_IDX] += et - st;
                
                int A_blk_nrow = H2ERI_gather_sum_int(sp_nbfp, pair_idx->length, pair_idx->data);
                int A_ff_ncol  = H2ERI_gather_sum_int(sp_nbfp, node_ff_idx->length, node_ff_idx->data);
                int A_pp_ncol  = num_pp;
                int A_blk_ncol = 2 * A_blk_nrow;
                int max_nbfp   = NCART(5) * NCART(5);
                H2E_dense_mat_resize(A_block, A_blk_nrow, A_blk_ncol);
                double *A_blk_pp = A_block->data;
                double *A_blk_ff = A_block->data + A_blk_nrow;

                // (4.1) Construct the random sparse matrix for NAI block normalization
                st = get_wtime_sec();
                int max_nnz_col = 16;
                H2E_gen_rand_sparse_mat_trans(max_nnz_col, A_pp_ncol, A_blk_nrow, rndmatA_val, rndmatA_idx);
                // Find the union of all rndmatA_idx
                int rand_nnz_col = (max_nnz_col <= A_pp_ncol) ? max_nnz_col : A_pp_ncol;
                int rndmatA_nnz = A_blk_nrow * rand_nnz_col;
                int *rndmatA_col = rndmatA_idx->data + (A_blk_nrow + 1);
                H2E_int_vec_set_capacity(work_buf1, A_pp_ncol);
                for (int k = 0; k < A_pp_ncol; k++)
                    work_buf1->data[k] = -1;
                work_buf1->length = A_pp_ncol;
                for (int k = 0; k < rndmatA_nnz; k++)
                    work_buf1->data[rndmatA_col[k]] = 1;
                H2E_int_vec_set_capacity(rndmatA_idx_cup, A_pp_ncol);
                int Aidx_cup_cnt = 0;
                for (int k = 0; k < A_pp_ncol; k++)
                {
                    if (work_buf1->data[k] == -1) continue;
                    rndmatA_idx_cup->data[Aidx_cup_cnt] = k;
                    work_buf1->data[k] = Aidx_cup_cnt;
                    Aidx_cup_cnt++;
                }
                rndmatA_idx_cup->length = Aidx_cup_cnt;
                Aidx_cup_cnt = 0;
                for (int k = 0; k < A_pp_ncol; k++)
                {
                    if (work_buf1->data[k] == -1) continue;
                    if (Aidx_cup_cnt != k)
                    {
                        pp_x[Aidx_cup_cnt] = pp_x[k];
                        pp_y[Aidx_cup_cnt] = pp_y[k];
                        pp_z[Aidx_cup_cnt] = pp_z[k];
                    }
                    Aidx_cup_cnt++;
                }
                // Map the old rndmatA_idx to new one
                for (int k = 0; k < rndmatA_nnz; k++)
                {
                    int nnz_idx = work_buf1->data[rndmatA_col[k]];
                    assert(nnz_idx != -1);
                    rndmatA_col[k] = nnz_idx;
                }
                H2E_dense_mat_resize(A_ff_pp, A_blk_nrow, Aidx_cup_cnt + 1);
                et = get_wtime_sec();
                timers[U_BUILD_SPMM_TIMER_IDX] += et - st;
                // (4.2) Calculate the NAI block and use the random sparse matrix to normalize it
                double *A_pp = A_ff_pp->data;
                double *A_pp_buf = A_pp + A_blk_nrow * Aidx_cup_cnt;
                st = get_wtime_sec();
                H2ERI_calc_NAI_pairs_to_mat(
                    sp_shells, num_sp, pair_idx->length, pair_idx->data, 
                    Aidx_cup_cnt, pp_x, pp_y, pp_z, A_pp, Aidx_cup_cnt, A_pp_buf
                );
                et = get_wtime_sec();
                timers[U_BUILD_NAI_TIMER_IDX] += et - st;
                st = get_wtime_sec();
                H2E_calc_sparse_mm_trans(
                    A_blk_nrow, A_blk_nrow, Aidx_cup_cnt,
                    rndmatA_val, rndmatA_idx, 
                    A_pp, Aidx_cup_cnt, A_blk_pp, A_blk_ncol
                );
                et = get_wtime_sec();
                timers[U_BUILD_SPMM_TIMER_IDX] += et - st;

                // (5.1) Construct the random sparse matrix for ERI block normalization
                st = get_wtime_sec();
                H2E_gen_rand_sparse_mat_trans(max_nnz_col, A_ff_ncol, A_blk_nrow, rndmatA_val, rndmatA_idx);
                // Find the union of all rndmatA_idx
                // Note: the first (A_blk_nrow+1) elements in rndmatA_idx are row_ptr,
                //       the rest RAND_NNZ_COL * A_blk_nrow elements are col
                rand_nnz_col = (max_nnz_col <= A_ff_ncol) ? max_nnz_col : A_ff_ncol;
                rndmatA_nnz = A_blk_nrow * rand_nnz_col;
                rndmatA_col = rndmatA_idx->data + (A_blk_nrow + 1);
                H2E_int_vec_set_capacity(work_buf1, A_ff_ncol);
                for (int k = 0; k < A_ff_ncol; k++)
                    work_buf1->data[k] = -1;
                work_buf1->length = A_ff_ncol;
                for (int k = 0; k < rndmatA_nnz; k++)
                    work_buf1->data[rndmatA_col[k]] = 1;
                H2E_int_vec_set_capacity(rndmatA_idx_cup, A_ff_ncol);
                Aidx_cup_cnt = 0;
                for (int k = 0; k < A_ff_ncol; k++)
                {
                    if (work_buf1->data[k] == -1) continue;
                    rndmatA_idx_cup->data[Aidx_cup_cnt] = k;
                    work_buf1->data[k] = Aidx_cup_cnt;
                    Aidx_cup_cnt++;
                }
                rndmatA_idx_cup->length = Aidx_cup_cnt;
                // Find the new far field shell pairs 
                H2ERI_extract_shell_pair_idx(
                    sp, rndmatA_idx_cup, node_ff_idx, 
                    work_buf2, node_ff_idx1, rndmatA_idx1
                );
                for (int k = 0; k < node_ff_idx1->length; k++)
                {
                    int tmp = node_ff_idx1->data[k];
                    node_ff_idx1->data[k] = node_ff_idx->data[tmp];
                }
                // Map the old rndmatA_idx to new one
                for (int k = 0; k < rndmatA_nnz; k++)
                {
                    int nnz_idx = work_buf1->data[rndmatA_col[k]];
                    assert(nnz_idx != -1);
                    rndmatA_col[k] = rndmatA_idx1->data[nnz_idx];
                }
                int A_ff_ncol1 = H2ERI_gather_sum_int(sp_nbfp, node_ff_idx1->length, node_ff_idx1->data);
                assert(A_ff_ncol1 >= rndmatA_idx_cup->length);
                et = get_wtime_sec();
                timers[U_BUILD_SPMM_TIMER_IDX] += et - st;
                // (5.2) Calculate the ERI block strip by strip and use the random sparse
                //        matrix to normalize it
                H2E_dense_mat_resize(A_ff_pp, max_nbfp, A_ff_ncol1);
                double *A_ff = A_ff_pp->data;
                int nbfp_cnt = 0;
                for (int k = 0; k < pair_idx->length; k++)
                {
                    int nbfp_k = CMS_get_sp_nbfp(sp + pair_idx->data[k]);
                    assert(nbfp_k <= max_nbfp);
                    st = get_wtime_sec();
                    H2ERI_calc_ERI_pairs_to_mat(
                        sp, 1, node_ff_idx1->length, pair_idx->data + k,
                        node_ff_idx1->data, simint_buff, A_ff, A_ff_ncol1, eri_batch_buff
                    );
                    et = get_wtime_sec();
                    timers[U_BUILD_ERI_TIMER_IDX] += et - st;
                    st = get_wtime_sec();
                    H2E_calc_sparse_mm_trans(
                        nbfp_k, A_blk_nrow, A_ff_ncol1,
                        rndmatA_val, rndmatA_idx, 
                        A_ff, A_ff_ncol1, A_blk_ff + nbfp_cnt * A_blk_ncol, A_blk_ncol
                    );
                    et = get_wtime_sec();
                    timers[U_BUILD_SPMM_TIMER_IDX] += et - st;
                    nbfp_cnt += nbfp_k;
                }
                assert(nbfp_cnt == A_blk_nrow);

                // (6) ID compression
                st = get_wtime_sec();
                H2E_dense_mat_normalize_columns(A_block, rndmatA_val);
                H2E_dense_mat_select_rows(A_block, row_idx);
                H2E_dense_mat_resize(QR_buff, 1, A_block->nrow);
                H2E_int_vec_set_capacity(ID_buff, 4 * A_block->nrow);
                et = get_wtime_sec();
                timers[U_BUILD_OTHER_TIMER_IDX] += et - st;
                st = get_wtime_sec();
                H2E_ID_compress(
                    A_block, QR_REL_NRM, stop_param, &U[node], sub_idx, 
                    1, QR_buff->data, ID_buff->data, 1
                );
                et = get_wtime_sec();
                timers[U_BUILD_QRID_TIMER_IDX] += et - st;
                st = get_wtime_sec();
                H2E_int_vec_gather(row_idx, sub_idx, sub_row_idx);
                H2E_int_vec_init(&J_pair[node], pair_idx->length);
                H2E_int_vec_init(&J_row[node],  sub_row_idx->length);
                H2ERI_extract_shell_pair_idx(
                    sp, sub_row_idx, pair_idx, 
                    work_buf1, sub_pair, J_row[node]
                );
                H2E_int_vec_gather(pair_idx, sub_pair, J_pair[node]);
                et = get_wtime_sec();
                timers[U_BUILD_OTHER_TIMER_IDX] += et - st;
            }  // End of j loop
            thread_buf[tid]->timer += get_wtime_sec();
        }  // End of "#pragma omp parallel"
        
        #ifdef PROFILING_OUTPUT
        double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
        for (int i = 0; i < n_thread_i; i++)
        {
            double thread_i_timer = thread_buf[i]->timer;
            avg_t += thread_i_timer;
            max_t = MAX(max_t, thread_i_timer);
            min_t = MIN(min_t, thread_i_timer);
        }
        avg_t /= (double) n_thread_i;
        printf("[PROFILING] Build U: level %d, %d/%d threads, %d nodes, ", i, n_thread_i, n_thread, level_i_n_node);
        printf("min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
        printf("[PROFILING] Build U subroutine time consumption:\n");
        printf("tid, calc ERI, calc NAI, SpMM, ID compress, misc., total\n");
        for (int tid = 0; tid < n_thread_i; tid++)
        {
            double *timers = U_timers + 8 * tid;
            printf(
                "%3d, %6.3lf, %6.3lf, %6.3lf, %6.3lf, %6.3lf, %6.3lf\n", tid, 
                timers[U_BUILD_ERI_TIMER_IDX], 
                timers[U_BUILD_NAI_TIMER_IDX],
                timers[U_BUILD_SPMM_TIMER_IDX],
                timers[U_BUILD_QRID_TIMER_IDX],
                timers[U_BUILD_OTHER_TIMER_IDX],
                thread_buf[tid]->timer
            );
        }
        #endif
        
        // B. Update skeleton points after the compression at the i-th level.
        //    At the (i-1)-th level, only need to consider overlapping FF shell pairs
        //    inside the skeleton shell pairs at i-th level. Note that the skeleton
        //    shell pairs of leaf nodes at i-th level are all shell pairs in leaf nodes.
        memset(skel_flag, 0, sizeof(int) * n_point);
        for (int j = 0; j < level_i_n_node; j++)
        {
            int node = level_i_nodes[j];
            int *idx = J_pair[node]->data;
            for (int k = 0; k < J_pair[node]->length; k++) skel_flag[idx[k]] = 1;
        }
        for (int j = 0; j < lvl_n_leaf[i]; j++)
        {
            int leaf_j = lvl_leaf[i * n_leaf_node + j];
            int pt_s = pt_cluster[2 * leaf_j];
            int pt_e = pt_cluster[2 * leaf_j + 1];
            for (int k = pt_s; k <= pt_e; k++) skel_flag[k] = 1;
        }
    }  // End of i loop
    
    // 5. Initialize other not touched U J & add statistic info
    for (int i = 0; i < h2eri->n_UJ; i++)
    {
        if (U[i] == NULL)
        {
            H2E_dense_mat_init(&U[i], 1, 1);
            U[i]->nrow = 0;
            U[i]->ncol = 0;
            U[i]->ld   = 0;
        } else {
            mat_size[U_SIZE_IDX]      += U[i]->nrow * U[i]->ncol;
            mat_size[MV_FWD_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[MV_FWD_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
            mat_size[MV_BWD_SIZE_IDX] += U[i]->nrow * U[i]->ncol;
            mat_size[MV_BWD_SIZE_IDX] += U[i]->nrow + U[i]->ncol;
        }
        if (J_row[i]  == NULL) H2E_int_vec_init(&J_row[i], 1);
        if (J_pair[i] == NULL) H2E_int_vec_init(&J_pair[i], 1);
        //printf("%4d, %4d\n", U[i]->nrow, U[i]->ncol);
    }

    // 6. Free local buffers
    free(skel_flag);
    free(lvl_leaf);
    free(lvl_n_leaf);
    free_aligned(U_timers);
    for (int i = 0; i < n_thread; i++)
        H2E_thread_buf_reset(thread_buf[i]);
    for (int i = 0; i < n_thread * 10; i++)
        H2E_int_vec_destroy(&tb_idx[i]);
    for (int i = 0; i < n_thread * 4; i++)
        H2E_dense_mat_destroy(&tb_mat[i]);
    free(tb_idx);
    free(tb_mat);
    
    BLAS_SET_NUM_THREADS(n_thread);
}

// Compress a B or D block blk into low-rank form using ID approximation. 
// blk = U * V where V = blk(J, :). If the compressed rank is not small 
// enough, we will still use the original block.
// Input parameters:
//   blk     : B or D block to be compressed, will be overwritten 
//   blk0    : Used for temporarily storing the original blk
//   U_mat   : Used for temporarily storing the U matrix
//   QR_buff : Used for QR buffer in ID compression
//   J       : Used for temporarily storing the skeleton row indices
//   ID_buff : Used for ID buffer in ID compression
// Output parameter:
//   *res_blk_ : The output B or D block after compression. If *res_blk_->ld < 0, 
//   -*res_blk_->ld is rank of the low-rank approximation, U and V are stored 
//   contiguous in *res_blk_->data. Otherwise, *res_blk_ stores a dense block.
void H2ERI_compress_BD_blk(
    H2E_dense_mat_p blk,   H2E_dense_mat_p blk0, 
    H2E_dense_mat_p U_mat, H2E_dense_mat_p QR_buff,
    H2E_int_vec_p   J,     H2E_int_vec_p   ID_buff,
    void *stop_param, H2E_dense_mat_p *res_blk_
)
{
    int blk_nrow = blk->nrow;
    int blk_ncol = blk->ncol;

    // Backup the original block
    H2E_dense_mat_resize(blk0, blk_nrow, blk_ncol);
    memcpy(blk0->data, blk->data, sizeof(double) * blk_nrow * blk_ncol);

    // Perform ID compress on the original block
    H2E_dense_mat_resize(QR_buff, 1, blk_nrow);
    H2E_int_vec_set_capacity(ID_buff, 4 * blk_nrow);
    H2E_ID_compress(
        blk, QR_REL_NRM, stop_param, &U_mat, 
        J, 1, QR_buff->data, ID_buff->data, 1
    );

    // Check if we should keep the compressed form or use the original block
    int blk_rank = J->length;
    int old_size = blk_nrow * blk_ncol;
    int new_size = (blk_nrow + blk_ncol) * blk_rank;
    if (1)
    //if (new_size > (old_size * 4 / 5))
    {
        // The compressed form is not small enough, use the original block
        H2E_dense_mat_init(res_blk_, blk_nrow, blk_ncol);
        H2E_dense_mat_p res_blk = *res_blk_;
        memcpy(res_blk->data, blk0->data, sizeof(double) * blk_nrow * blk_ncol);
    }
}

// Build H2 generator matrices
// Input parameter:
//   h2eri : H2ERI structure with point partitioning & shell pair info
// Output parameter:
//   h2eri : H2ERI structure with H2 generator blocks
void H2ERI_build_B(H2ERI_p h2eri)
{
    int BD_JIT            = h2eri->BD_JIT;
    int n_thread          = h2eri->n_thread;
    int n_node            = h2eri->n_node;
    int n_r_adm_pair      = h2eri->n_r_adm_pair;
    int *r_adm_pairs      = h2eri->r_adm_pairs;
    int *node_level       = h2eri->node_level;
    int *pt_cluster       = h2eri->pt_cluster;
    int *sp_nbfp          = h2eri->sp_nbfp;
    int *sp_bfp_sidx      = h2eri->sp_bfp_sidx;
    int *index_seq        = h2eri->index_seq;
    void *stop_param      = &h2eri->QR_stop_tol;
    multi_sp_t    *sp     = h2eri->sp;
    H2E_int_vec_p B_blk   = h2eri->B_blk;
    H2E_int_vec_p *J_pair = h2eri->J_pair;
    H2E_int_vec_p *J_row  = h2eri->J_row;
    H2E_thread_buf_p *thread_buf      = h2eri->thread_buffs;
    simint_buff_p    *simint_buffs    = h2eri->simint_buffs;
    eri_batch_buff_p *eri_batch_buffs = h2eri->eri_batch_buffs;
    
    // 1. Allocate B
    h2eri->n_B = n_r_adm_pair;
    h2eri->B_nrow = (int*)    malloc(sizeof(int)    * n_r_adm_pair);
    h2eri->B_ncol = (int*)    malloc(sizeof(int)    * n_r_adm_pair);
    h2eri->B_ptr  = (size_t*) malloc(sizeof(size_t) * (n_r_adm_pair + 1));
    int    *B_nrow = h2eri->B_nrow;
    int    *B_ncol = h2eri->B_ncol;
    size_t *B_ptr  = h2eri->B_ptr;
    assert(h2eri->B_nrow != NULL && h2eri->B_ncol != NULL && h2eri->B_ptr != NULL);

    int B_pair_cnt = 0;
    int *B_pair_i = (int*) malloc(sizeof(int) * n_r_adm_pair * 2);
    int *B_pair_j = (int*) malloc(sizeof(int) * n_r_adm_pair * 2);
    int *B_pair_v = (int*) malloc(sizeof(int) * n_r_adm_pair * 2);
    ASSERT_PRINTF(
        B_pair_i != NULL && B_pair_j != NULL && B_pair_v != NULL,
        "Failed to allocate working buffer for B matrices indexing\n"
    );
    
    // 2. Partition B matrices into multiple blocks s.t. each block has approximately
    //    the same workload (total size of B matrices in a block)
    B_ptr[0] = 0;
    size_t B_total_size = 0;
    h2eri->node_n_r_adm = (int*) malloc(sizeof(int) * n_node);
    assert(h2eri->node_n_r_adm != NULL);
    int *node_n_r_adm = h2eri->node_n_r_adm;
    memset(node_n_r_adm, 0, sizeof(int) * n_node);
    for (int i = 0; i < n_r_adm_pair; i++)
    {
        int node0  = r_adm_pairs[2 * i];
        int node1  = r_adm_pairs[2 * i + 1];
        int level0 = node_level[node0];
        int level1 = node_level[node1];
        node_n_r_adm[node0]++;
        node_n_r_adm[node1]++;
        if(1)
        //if (level0 == level1)
        {
            B_nrow[i] = J_row[node0]->length;
            B_ncol[i] = J_row[node1]->length;
        }
        /*
        if (level0 > level1)
        {
            int pt_s1 = pt_cluster[2 * node1];
            int pt_e1 = pt_cluster[2 * node1 + 1];
            B_nrow[i] = J_row[node0]->length;
            B_ncol[i] = sp_bfp_sidx[pt_e1 + 1] - sp_bfp_sidx[pt_s1];
        }
        if (level0 < level1)
        {
            int pt_s0 = pt_cluster[2 * node0];
            int pt_e0 = pt_cluster[2 * node0 + 1];
            B_nrow[i] = sp_bfp_sidx[pt_e0 + 1] - sp_bfp_sidx[pt_s0];
            B_ncol[i] = J_row[node1]->length;
        }
        */
        size_t Bi_size = (size_t) B_nrow[i] * (size_t) B_ncol[i];
        B_total_size += Bi_size;
        B_ptr[i + 1] = Bi_size;
        B_pair_i[B_pair_cnt] = node0;
        B_pair_j[B_pair_cnt] = node1;
        B_pair_v[B_pair_cnt] = i + 1;
        B_pair_cnt++;
        B_pair_i[B_pair_cnt] = node1;
        B_pair_j[B_pair_cnt] = node0;
        B_pair_v[B_pair_cnt] = -(i + 1);
        B_pair_cnt++;
    }
    int BD_ntask_thread = (BD_JIT == 1) ? BD_NTASK_THREAD : 1;
    H2E_partition_workload(n_r_adm_pair, B_ptr + 1, B_total_size, n_thread * BD_ntask_thread, B_blk);
    for (int i = 1; i <= n_r_adm_pair; i++) B_ptr[i] += B_ptr[i - 1];

    // 2.1 Store pair-to-index relations in a CSR matrix for matvec, matmul
    h2eri->B_p2i_rowptr = (int*) malloc(sizeof(int) * (n_node + 1));
    h2eri->B_p2i_colidx = (int*) malloc(sizeof(int) * n_r_adm_pair * 2);
    h2eri->B_p2i_val    = (int*) malloc(sizeof(int) * n_r_adm_pair * 2);
    ASSERT_PRINTF(h2eri->B_p2i_rowptr != NULL, "Failed to allocate arrays for B matrices indexing\n");
    ASSERT_PRINTF(h2eri->B_p2i_colidx != NULL, "Failed to allocate arrays for B matrices indexing\n");
    ASSERT_PRINTF(h2eri->B_p2i_val    != NULL, "Failed to allocate arrays for B matrices indexing\n");
    H2E_int_COO_to_CSR(
        n_node, B_pair_cnt, B_pair_i, B_pair_j, B_pair_v, 
        h2eri->B_p2i_rowptr, h2eri->B_p2i_colidx, h2eri->B_p2i_val
    );
    free(B_pair_i);
    free(B_pair_j);
    free(B_pair_v);
    
    if (BD_JIT == 1) return;
    
    // 3. Generate B matrices
    h2eri->c_B_blks = (H2E_dense_mat_p*) malloc(sizeof(H2E_dense_mat_p) * h2eri->n_B);
    assert(h2eri->c_B_blks != NULL);
    H2E_dense_mat_p *c_B_blks = h2eri->c_B_blks;
    const int n_B_blk = B_blk->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        H2E_dense_mat_p  tmpB           = thread_buf[tid]->mat0;
        H2E_dense_mat_p  tmpB0          = thread_buf[tid]->mat1;
        H2E_int_vec_p    J              = thread_buf[tid]->idx0;
        H2E_int_vec_p    ID_buff        = thread_buf[tid]->idx1;
        simint_buff_p    simint_buff    = simint_buffs[tid];
        eri_batch_buff_p eri_batch_buff = eri_batch_buffs[tid];
        
        H2E_dense_mat_p U_mat, QR_buff;
        H2E_dense_mat_init(&U_mat,   1, 1024);
        H2E_dense_mat_init(&QR_buff, 1, 1024);

        thread_buf[tid]->timer = -get_wtime_sec();
        #pragma omp for schedule(dynamic) nowait
        for (int i_blk = 0; i_blk < n_B_blk; i_blk++)
        {
            int B_blk_s = B_blk->data[i_blk];
            int B_blk_e = B_blk->data[i_blk + 1];
            if (i_blk >= n_B_blk)
            {
                B_blk_s = 0; 
                B_blk_e = 0;
            }
            for (int i = B_blk_s; i < B_blk_e; i++)
            {
                int node0  = r_adm_pairs[2 * i];
                int node1  = r_adm_pairs[2 * i + 1];
                int level0 = node_level[node0];
                int level1 = node_level[node1];
                
                // (1) Two nodes are of the same level, compress on both sides
                if(1)
                //if (level0 == level1)
                {
                    int tmpB_nrow  = H2ERI_gather_sum_int(sp_nbfp, J_pair[node0]->length, J_pair[node0]->data);
                    int tmpB_ncol  = H2ERI_gather_sum_int(sp_nbfp, J_pair[node1]->length, J_pair[node1]->data);
                    int n_bra_pair = J_pair[node0]->length;
                    int n_ket_pair = J_pair[node1]->length;
                    int *bra_idx   = J_pair[node0]->data;
                    int *ket_idx   = J_pair[node1]->data;
                    H2E_dense_mat_resize(tmpB, tmpB_nrow, tmpB_ncol);
                    H2ERI_calc_ERI_pairs_to_mat(
                        sp, n_bra_pair, n_ket_pair, bra_idx, ket_idx, 
                        simint_buff, tmpB->data, tmpB->ncol, eri_batch_buff
                    );
                    H2E_dense_mat_select_rows   (tmpB, J_row[node0]);
                    H2E_dense_mat_select_columns(tmpB, J_row[node1]);
                }
                
                // (2) node1 is a leaf node and its level is higher than node0's level, 
                //     only compress on node0's side
                /*
                if (level0 > level1)
                {
                    int tmpB_nrow  = H2ERI_gather_sum_int(sp_nbfp, J_pair[node0]->length, J_pair[node0]->data);
                    int tmpB_ncol  = B_ncol[i];
                    int pt_s1      = pt_cluster[2 * node1];
                    int pt_e1      = pt_cluster[2 * node1 + 1];
                    int n_bra_pair = J_pair[node0]->length;
                    int n_ket_pair = pt_e1 - pt_s1 + 1;
                    int *bra_idx   = J_pair[node0]->data;
                    int *ket_idx   = index_seq + pt_s1;
                    H2E_dense_mat_resize(tmpB, tmpB_nrow, tmpB_ncol);
                    H2ERI_calc_ERI_pairs_to_mat(
                        sp, n_bra_pair, n_ket_pair, bra_idx, ket_idx, 
                        simint_buff, tmpB->data, tmpB->ncol, eri_batch_buff
                    );
                    H2E_dense_mat_select_rows(tmpB, J_row[node0]);
                }
                
                // (3) node0 is a leaf node and its level is higher than node1's level, 
                //     only compress on node1's side
                if (level0 < level1)
                {
                    int tmpB_nrow  = B_nrow[i];
                    int tmpB_ncol  = H2ERI_gather_sum_int(sp_nbfp, J_pair[node1]->length, J_pair[node1]->data);
                    int pt_s0      = pt_cluster[2 * node0];
                    int pt_e0      = pt_cluster[2 * node0 + 1];
                    int n_bra_pair = pt_e0 - pt_s0 + 1;
                    int n_ket_pair = J_pair[node1]->length;
                    int *bra_idx   = index_seq + pt_s0;
                    int *ket_idx   = J_pair[node1]->data;
                    H2E_dense_mat_resize(tmpB, tmpB_nrow, tmpB_ncol);
                    H2ERI_calc_ERI_pairs_to_mat(
                        sp, n_bra_pair, n_ket_pair, bra_idx, ket_idx, 
                        simint_buff, tmpB->data, tmpB->ncol, eri_batch_buff
                    );
                    H2E_dense_mat_select_columns(tmpB, J_row[node1]);
                }
                */
                H2ERI_compress_BD_blk(tmpB, tmpB0, U_mat, QR_buff, J, ID_buff, stop_param, &c_B_blks[i]);
            }  // End of i loop
        }  // End of i_blk loop
        thread_buf[tid]->timer += get_wtime_sec();

        H2E_dense_mat_destroy(&U_mat);
        H2E_dense_mat_destroy(&QR_buff);
    }  // End of "pragma omp parallel"

    // Recalculate the total size of the B blocks and re-partition B blocks for matvec
    h2eri->mat_size[MV_MID_SIZE_IDX] = 0;
    B_ptr[0] = 0;
    B_total_size = 0;
    for (int i = 0; i < h2eri->n_B; i++)
    {
        size_t Bi_size = c_B_blks[i]->size;
        h2eri->mat_size[MV_MID_SIZE_IDX] += Bi_size;
        h2eri->mat_size[MV_MID_SIZE_IDX] += (size_t) (B_nrow[i] + B_ncol[i]) * 2;
        B_ptr[i + 1] = Bi_size;
        B_total_size += Bi_size;
    }
    H2E_partition_workload(n_r_adm_pair, B_ptr + 1, B_total_size, n_thread * BD_ntask_thread, B_blk);
    for (int i = 1; i <= n_r_adm_pair; i++) B_ptr[i] += B_ptr[i - 1];
    h2eri->mat_size[B_SIZE_IDX] = B_total_size;

    #ifdef PROFILING_OUTPUT
    double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
    for (int i = 0; i < n_thread; i++)
    {
        double thread_i_timer = thread_buf[i]->timer;
        avg_t += thread_i_timer;
        max_t = MAX(max_t, thread_i_timer);
        min_t = MIN(min_t, thread_i_timer);
    }
    avg_t /= (double) n_thread;
    printf("[PROFILING] Build B: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    #endif
    
    BLAS_SET_NUM_THREADS(n_thread);
}

// Build dense blocks in the original matrices
// Input parameter:
//   h2eri : H2ERI structure with point partitioning & shell pair info
// Output parameter:
//   h2eri : H2ERI structure with H2 dense blocks
void H2ERI_build_D(H2ERI_p h2eri)
{
    int BD_JIT           = h2eri->BD_JIT;
    int n_thread         = h2eri->n_thread;
    int n_node           = h2eri->n_node;
    int n_leaf_node      = h2eri->n_leaf_node;
    int n_r_inadm_pair   = h2eri->n_r_inadm_pair;
    int *leaf_nodes      = h2eri->height_nodes;
    int *pt_cluster      = h2eri->pt_cluster;
    int *r_inadm_pairs   = h2eri->r_inadm_pairs;
    int *sp_bfp_sidx     = h2eri->sp_bfp_sidx;
    int *index_seq       = h2eri->index_seq;
    void *stop_param     = &h2eri->QR_stop_tol;
    H2E_int_vec_p D_blk0 = h2eri->D_blk0;
    H2E_int_vec_p D_blk1 = h2eri->D_blk1;
    multi_sp_t *sp       = h2eri->sp;
    H2E_thread_buf_p *thread_buf      = h2eri->thread_buffs;
    simint_buff_p    *simint_buffs    = h2eri->simint_buffs;
    eri_batch_buff_p *eri_batch_buffs = h2eri->eri_batch_buffs;
    
    // 1. Allocate D
    h2eri->n_D = n_leaf_node + n_r_inadm_pair;
    h2eri->D_nrow = (int*)    malloc(sizeof(int)    * h2eri->n_D);
    h2eri->D_ncol = (int*)    malloc(sizeof(int)    * h2eri->n_D);
    h2eri->D_ptr  = (size_t*) malloc(sizeof(size_t) * (h2eri->n_D + 1));
    int    *D_nrow = h2eri->D_nrow;
    int    *D_ncol = h2eri->D_ncol;
    size_t *D_ptr  = h2eri->D_ptr;
    assert(h2eri->D_nrow != NULL && h2eri->D_ncol != NULL && h2eri->D_ptr != NULL);

    int D_pair_cnt = 0;
    int n_Dij_pair = n_leaf_node + 2 * n_r_inadm_pair;
    int *D_pair_i = (int*) malloc(sizeof(int) * n_Dij_pair);
    int *D_pair_j = (int*) malloc(sizeof(int) * n_Dij_pair);
    int *D_pair_v = (int*) malloc(sizeof(int) * n_Dij_pair);
    ASSERT_PRINTF(
        D_pair_i != NULL && D_pair_j != NULL && D_pair_v != NULL,
        "Failed to allocate working buffer for D matrices indexing\n"
    );
    
    // 2. Partition D matrices into multiple blocks s.t. each block has approximately
    //    the same total size of D matrices in a block
    D_ptr[0] = 0;
    size_t D0_total_size = 0;
    for (int i = 0; i < n_leaf_node; i++)
    {
        int node = leaf_nodes[i];
        int pt_s = pt_cluster[2 * node];
        int pt_e = pt_cluster[2 * node + 1];
        int node_nbfp = sp_bfp_sidx[pt_e + 1] - sp_bfp_sidx[pt_s];
        size_t Di_size = (size_t) node_nbfp * (size_t) node_nbfp;
        D_nrow[i] = node_nbfp;
        D_ncol[i] = node_nbfp;
        D_ptr[i + 1] = Di_size;
        D0_total_size += Di_size;
        D_pair_i[D_pair_cnt] = node;
        D_pair_j[D_pair_cnt] = node;
        D_pair_v[D_pair_cnt] = i + 1;
        D_pair_cnt++;
    }
    int BD_ntask_thread = (BD_JIT == 1) ? BD_NTASK_THREAD : 1;
    H2E_partition_workload(n_leaf_node, D_ptr + 1, D0_total_size, n_thread * BD_ntask_thread, D_blk0);
    size_t D1_total_size = 0;
    for (int i = 0; i < n_r_inadm_pair; i++)
    {
        int ii = i + n_leaf_node;
        int node0 = r_inadm_pairs[2 * i];
        int node1 = r_inadm_pairs[2 * i + 1];
        int pt_s0 = pt_cluster[2 * node0];
        int pt_s1 = pt_cluster[2 * node1];
        int pt_e0 = pt_cluster[2 * node0 + 1];
        int pt_e1 = pt_cluster[2 * node1 + 1];
        int node0_nbfp = sp_bfp_sidx[pt_e0 + 1] - sp_bfp_sidx[pt_s0];
        int node1_nbfp = sp_bfp_sidx[pt_e1 + 1] - sp_bfp_sidx[pt_s1];
        size_t Di_size = (size_t) node0_nbfp * (size_t) node1_nbfp;
        D_nrow[i + n_leaf_node] = node0_nbfp;
        D_ncol[i + n_leaf_node] = node1_nbfp;
        D_ptr[n_leaf_node + 1 + i] = Di_size;
        D1_total_size += Di_size;
        D_pair_i[D_pair_cnt] = node0;
        D_pair_j[D_pair_cnt] = node1;
        D_pair_v[D_pair_cnt] = ii + 1;
        D_pair_cnt++;
        D_pair_i[D_pair_cnt] = node1;
        D_pair_j[D_pair_cnt] = node0;
        D_pair_v[D_pair_cnt] = -(ii + 1);
        D_pair_cnt++;
    }
    H2E_partition_workload(n_r_inadm_pair, D_ptr + n_leaf_node + 1, D1_total_size, n_thread * BD_ntask_thread, D_blk1);
    for (int i = 1; i <= n_leaf_node + n_r_inadm_pair; i++) D_ptr[i] += D_ptr[i - 1];

    // 2.1 Store pair-to-index relations in a CSR matrix for matvec, matmul, and SPDHSS construction
    h2eri->D_p2i_rowptr = (int*) malloc(sizeof(int) * (n_node + 1));
    h2eri->D_p2i_colidx = (int*) malloc(sizeof(int) * n_Dij_pair);
    h2eri->D_p2i_val    = (int*) malloc(sizeof(int) * n_Dij_pair);
    ASSERT_PRINTF(h2eri->D_p2i_rowptr != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(h2eri->D_p2i_colidx != NULL, "Failed to allocate arrays for D matrices indexing\n");
    ASSERT_PRINTF(h2eri->D_p2i_val    != NULL, "Failed to allocate arrays for D matrices indexing\n");
    H2E_int_COO_to_CSR(
        n_node, D_pair_cnt, D_pair_i, D_pair_j, D_pair_v, 
        h2eri->D_p2i_rowptr, h2eri->D_p2i_colidx, h2eri->D_p2i_val
    );
    free(D_pair_i);
    free(D_pair_j);
    free(D_pair_v);
    
    if (BD_JIT == 1) return;
    
    h2eri->c_D_blks = (H2E_dense_mat_p*) malloc(sizeof(H2E_dense_mat_p) * h2eri->n_D);
    assert(h2eri->c_D_blks != NULL);
    H2E_dense_mat_p *c_D_blks = h2eri->c_D_blks;
    const int n_D0_blk = D_blk0->length - 1;
    const int n_D1_blk = D_blk1->length - 1;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        
        H2E_dense_mat_p  tmpD           = thread_buf[tid]->mat0;
        H2E_dense_mat_p  tmpD0          = thread_buf[tid]->mat1;
        H2E_int_vec_p    J              = thread_buf[tid]->idx0;
        H2E_int_vec_p    ID_buff        = thread_buf[tid]->idx1;
        simint_buff_p    simint_buff    = simint_buffs[tid];
        eri_batch_buff_p eri_batch_buff = eri_batch_buffs[tid];
        
        H2E_dense_mat_p U_mat, QR_buff;
        H2E_dense_mat_init(&U_mat,   1, 1024);
        H2E_dense_mat_init(&QR_buff, 1, 1024);

        thread_buf[tid]->timer = -get_wtime_sec();
        
        // 3. Generate diagonal blocks (leaf node self interaction)
        #pragma omp for schedule(dynamic) nowait
        for (int i_blk0 = 0; i_blk0 < n_D0_blk; i_blk0++)
        {
            int D_blk0_s = D_blk0->data[i_blk0];
            int D_blk0_e = D_blk0->data[i_blk0 + 1];
            if (i_blk0 >= n_D0_blk)
            {
                D_blk0_s = 0;
                D_blk0_e = 0;
            }
            for (int i = D_blk0_s; i < D_blk0_e; i++)
            {
                int node = leaf_nodes[i];
                int pt_s = pt_cluster[2 * node];
                int pt_e = pt_cluster[2 * node + 1];
                int node_npts = pt_e - pt_s + 1;
                int Di_nrow = D_nrow[i];
                int Di_ncol = D_ncol[i];
                H2E_dense_mat_resize(tmpD, Di_nrow, Di_ncol);
                double *Di = tmpD->data;
                int *bra_idx = index_seq + pt_s;
                int *ket_idx = bra_idx;
                H2ERI_calc_ERI_pairs_to_mat(
                    sp, node_npts, node_npts, bra_idx, ket_idx, 
                    simint_buff, Di, Di_ncol, eri_batch_buff
                );

                H2ERI_compress_BD_blk(tmpD, tmpD0, U_mat, QR_buff, J, ID_buff, stop_param, &c_D_blks[i]);
            }
        }  // End of i_blk0 loop
        
        // 4. Generate off-diagonal blocks from inadmissible pairs
        #pragma omp for schedule(dynamic) nowait
        for (int i_blk1 = 0; i_blk1 < n_D1_blk; i_blk1++)
        {
            int D_blk1_s = D_blk1->data[i_blk1];
            int D_blk1_e = D_blk1->data[i_blk1 + 1];
            if (i_blk1 >= n_D1_blk)
            {
                D_blk1_s = 0;
                D_blk1_e = 0;
            }
            for (int i = D_blk1_s; i < D_blk1_e; i++)
            {
                int node0 = r_inadm_pairs[2 * i];
                int node1 = r_inadm_pairs[2 * i + 1];
                int pt_s0 = pt_cluster[2 * node0];
                int pt_s1 = pt_cluster[2 * node1];
                int pt_e0 = pt_cluster[2 * node0 + 1];
                int pt_e1 = pt_cluster[2 * node1 + 1];
                int node0_npts = pt_e0 - pt_s0 + 1;
                int node1_npts = pt_e1 - pt_s1 + 1;
                int Di_nrow = D_nrow[i + n_leaf_node];
                int Di_ncol = D_ncol[i + n_leaf_node];
                H2E_dense_mat_resize(tmpD, Di_nrow, Di_ncol);
                double *Di = tmpD->data;
                int *bra_idx = index_seq + pt_s0;
                int *ket_idx = index_seq + pt_s1;
                H2ERI_calc_ERI_pairs_to_mat(
                    sp, node0_npts, node1_npts, bra_idx, ket_idx, 
                    simint_buff, Di, Di_ncol, eri_batch_buff
                );

                H2ERI_compress_BD_blk(tmpD, tmpD0, U_mat, QR_buff, J, ID_buff, stop_param, &c_D_blks[i + n_leaf_node]);
            }
        }  // End of i_blk1 loop
        thread_buf[tid]->timer += get_wtime_sec();

        H2E_dense_mat_destroy(&U_mat);
        H2E_dense_mat_destroy(&QR_buff);
    }  // End of "pragma omp parallel"
    
    // Recalculate the total size of the D blocks and re-partition D blocks for matvec
    h2eri->mat_size[MV_DEN_SIZE_IDX] = 0;
    D0_total_size = 0;
    for (int i = 0; i < n_leaf_node; i++)
    {
        H2E_dense_mat_p Di = c_D_blks[i];
        size_t Di_size = Di->size;
        h2eri->mat_size[MV_DEN_SIZE_IDX] += Di_size;
        h2eri->mat_size[MV_DEN_SIZE_IDX] += (size_t) (Di->nrow + Di->ncol);
        D_ptr[i + 1] = Di_size;
        D0_total_size += Di_size;
    }
    H2E_partition_workload(n_leaf_node, D_ptr + 1, D0_total_size, n_thread * BD_ntask_thread, D_blk0);
    D1_total_size = 0;
    for (int i = 0; i < n_r_inadm_pair; i++)
    {
        H2E_dense_mat_p Di = c_D_blks[i + n_leaf_node];
        size_t Di_size = Di->size;
        h2eri->mat_size[MV_DEN_SIZE_IDX] += Di_size;
        h2eri->mat_size[MV_DEN_SIZE_IDX] += (size_t) (2 * (Di->nrow + Di->ncol));
        D_ptr[n_leaf_node + 1 + i] = Di_size;
        D1_total_size += Di_size;
    }

    h2eri->nD0element=D0_total_size;
    h2eri->nD1element=D1_total_size;
    
    H2E_partition_workload(n_r_inadm_pair, D_ptr + n_leaf_node + 1, D1_total_size, n_thread * BD_ntask_thread, D_blk1);
    for (int i = 1; i <= n_leaf_node + n_r_inadm_pair; i++) D_ptr[i] += D_ptr[i - 1];
    h2eri->mat_size[D_SIZE_IDX] = D0_total_size + D1_total_size;

    #ifdef PROFILING_OUTPUT
    double max_t = 0.0, avg_t = 0.0, min_t = 19241112.0;
    for (int i = 0; i < n_thread; i++)
    {
        double thread_i_timer = thread_buf[i]->timer;
        avg_t += thread_i_timer;
        max_t = MAX(max_t, thread_i_timer);
        min_t = MIN(min_t, thread_i_timer);
    }
    avg_t /= (double) n_thread;
    printf("[PROFILING] Build D: min/avg/max thread wall-time = %.3lf, %.3lf, %.3lf (s)\n", min_t, avg_t, max_t);
    #endif
    
    BLAS_SET_NUM_THREADS(n_thread);
}

// Build H2 representation for ERI tensor
void H2ERI_build_H2(H2ERI_p h2eri, const int BD_JIT)
{
    double st, et;

    if (BD_JIT == 1) h2eri->BD_JIT = 1;
    else h2eri->BD_JIT = 0;
    printf("111\n");
    // 1. Build projection matrices and skeleton row sets
    st = get_wtime_sec();
    H2ERI_build_UJ_proxy(h2eri);
    et = get_wtime_sec();
    h2eri->timers[U_BUILD_TIMER_IDX] = et - st;
    printf("222\n");
    // 2. Build generator matrices
    st = get_wtime_sec();
    H2ERI_build_B(h2eri);
    et = get_wtime_sec();
    h2eri->timers[B_BUILD_TIMER_IDX] = et - st;
    
    // 3. Build dense blocks
    st = get_wtime_sec();
    H2ERI_build_D(h2eri);
    et = get_wtime_sec();
    h2eri->timers[D_BUILD_TIMER_IDX] = et - st;
}
