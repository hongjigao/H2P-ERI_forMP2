#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#include "CMS.h"

// Read all shell information in a .mol file and normalize all these shells
void CMS_read_mol_file(const char *mol_fname, int *nshell_, shell_t **shells_)
{
    int AM_map[128];
    AM_map['S'] = 0;
    AM_map['P'] = 1;
    AM_map['D'] = 2;
    AM_map['F'] = 3;
    AM_map['G'] = 4;
    AM_map['H'] = 5;
    AM_map['I'] = 6;
    AM_map['J'] = 7;
    AM_map['K'] = 8;
    AM_map['L'] = 9;
    
    FILE *inf;
    
    inf = fopen(mol_fname, "r");
    if (inf == NULL)
    {
        printf("[FATAL] CMS cannot open mol file %s\n", mol_fname);
        assert(inf != NULL);
    }
    
    // 1. First pass, get the nshell_total
    int natom, nshell_total = 0;
    fscanf(inf, "%d", &natom);
    for (int i = 0; i < natom; i++)
    {
        char sym[8];
        int nshell, nprimall, nallprimg;
        double x, y, z;
        fscanf(inf, "%s %d %d %d", sym, &nshell, &nprimall, &nallprimg);
        fscanf(inf, "%lf %lf %lf", &x, &y, &z);
        for (int j = 0; j < nshell; j++)
        {
            char type[8];
            int nprim, ngen;
            fscanf(inf, "%s %d %d", type, &nprim, &ngen);
            nshell_total += ngen;
            for (int k = 0; k < nprim; k++)
            {
                double alpha;
                fscanf(inf, "%lf", &alpha);
                for (int l = 0; l < ngen; l++)
                {
                    double coef;
                    fscanf(inf, "%lf", &coef);
                }
            }
        }
    }
    fclose(inf);
    
    // 2. Second pass, create Simint shells
    shell_t *shells = (shell_t *) malloc(sizeof(shell_t) * nshell_total);
    assert(shells != NULL);
    int shell_idx = 0;
    inf = fopen(mol_fname, "r");
    fscanf(inf, "%d", &natom);
    for (int i = 0; i < natom; i++)
    {
        char sym[8], type[8];
        int nshell, nprimall, nallprimg;
        int nprim, ngen, sidx;
        double x, y, z;
        double alpha, coef;
        
        fscanf(inf, "%s %d %d %d", sym, &nshell, &nprimall, &nallprimg);
        fscanf(inf, "%lf %lf %lf", &x, &y, &z);
        
        for (int j = 0; j < nshell; j++)
        {
            fscanf(inf, "%s %d %d", type, &nprim, &ngen);
            
            for (int l = 0; l < ngen; l++)
            {
                sidx = shell_idx + l;
                simint_initialize_shell(&shells[sidx]);
                simint_allocate_shell(nprim, &shells[sidx]);
                shells[sidx].am    = AM_map[(char) type[l]];;
                shells[sidx].nprim = nprim;
                shells[sidx].x     = x;
                shells[sidx].y     = y;
                shells[sidx].z     = z;
            }
            
            for (int k = 0; k < nprim; k++)
            {
                fscanf(inf, "%lf", &alpha);
                for (int l = 0; l < ngen; l++)
                {
                    fscanf(inf, "%lf", &coef);
                    sidx = shell_idx + l;
                    shells[sidx].alpha[k] = alpha;
                    shells[sidx].coef[k]  = coef;
                }
            }
            
            shell_idx += ngen;
        }
    }
    fclose(inf);
    
    // 3. Normalize all shells
    simint_normalize_shells(nshell_total, shells);
    
    *nshell_ = nshell_total;
    *shells_ = shells;
}

// Destroy all Simint shells
void CMS_destroy_shells(const int nshell, shell_t *shells)
{
    for (int i = 0; i < nshell; i++)
        simint_free_shell(&shells[i]);
}

// Print all shell information, for debugging
void CMS_print_shells(const int nshell, shell_t *shells)
{
    printf("%d Shells:\n", nshell);
    for (int i = 0; i < nshell; i++)
    {
        printf(
            "%d, %2d, %.3lf, %.3lf, %.3lf, ", shells[i].am, 
            shells[i].nprim, shells[i].x, shells[i].y, shells[i].z
        );
        int nprim = shells[i].nprim;
        for (int j = 0; j < nprim; j++) printf("%.3lf, ", shells[i].alpha[j]);
        for (int j = 0; j < nprim; j++) printf("%.3lf, ", shells[i].coef[j]);
        printf("\n");
    }
}

// Get the Schwarz screening value from a given set of shells
double CMS_get_Schwarz_scrval(const int nshell, shell_t *shells, double *scr_vals)
{
    // 1. Calculate the size of each shell and prepare Simint buffer
    int *shell_bf_num = (int*) malloc(sizeof(int) * nshell);
    int max_am = 0, max_am_ncart = 0;
    assert(shell_bf_num != NULL);
    for (int i = 0; i < nshell; i++)
    {
        int am = shells[i].am;
        int am_ncart = NCART(am);
        max_am = MAX(max_am, am);
        max_am_ncart = MAX(max_am_ncart, am_ncart); 
        shell_bf_num[i] = am_ncart;
    }
    size_t work_msize = simint_ostei_workmem(0, max_am);
    size_t ERI_msize  = sizeof(double) * (max_am_ncart * max_am_ncart * max_am_ncart * max_am_ncart);
    ERI_msize += sizeof(double) * 8;
    
    // 2. Calculate (MN|MN) and find the Schwarz screening value
    double global_max_scrval = 0.0;
    #pragma omp parallel
    {    
        struct simint_multi_shellpair MN_pair;
        simint_initialize_multi_shellpair(&MN_pair);
        
        double *work_mem = SIMINT_ALLOC(work_msize);
        double *ERI_mem  = SIMINT_ALLOC(ERI_msize);
        assert(work_mem != NULL && ERI_mem != NULL);
        
        #pragma omp for schedule(dynamic) reduction(max:global_max_scrval)
        for (int M = 0; M < nshell; M++)
        {
            int dimM = shell_bf_num[M];
            for (int N = 0; N < nshell; N++)
            {
                int dimN = shell_bf_num[N];
                
                simint_create_multi_shellpair(1, &shells[M], 1, &shells[N], &MN_pair, 0);
                int ERI_size = simint_compute_eri(&MN_pair, &MN_pair, 0.0, work_mem, ERI_mem);
                if (ERI_size <= 0) continue;
                
                int ld_MNM_M = (dimM * dimN * dimM + dimM);
                int ld_NM_1  = (dimN * dimM + 1);
                double max_val = 0.0;
                for (int iM = 0; iM < dimM; iM++)
                {
                    for (int iN = 0; iN < dimN; iN++)
                    {
                        int    idx = iN * ld_MNM_M + iM * ld_NM_1;
                        double val = fabs(ERI_mem[idx]);
                        max_val = MAX(max_val, val);
                    }
                }
                global_max_scrval = MAX(global_max_scrval, max_val);
                scr_vals[M * nshell + N] = max_val;
                scr_vals[N * nshell + M] = max_val;
            }
        }
        
        SIMINT_FREE(ERI_mem);
        SIMINT_FREE(work_mem);
        simint_free_multi_shellpair(&MN_pair);
    }
    
    free(shell_bf_num);
    return global_max_scrval;
}

// Initialize a Simint buffer structure
void CMS_init_Simint_buff(const int max_am, simint_buff_t *buff_)
{
    simint_buff_t buff = (simint_buff_t) malloc(sizeof(struct simint_buff));
    assert(buff != NULL);
    
    int max_ncart = NCART(max_am);
    int max_int = max_ncart * max_ncart * max_ncart * max_ncart;
    buff->work_msize = simint_ostei_workmem(0, max_am);
    buff->ERI_msize  = sizeof(double) * max_int * NPAIR_SIMD;
    buff->work_mem = SIMINT_ALLOC(buff->work_msize);
    buff->ERI_mem  = SIMINT_ALLOC(buff->ERI_msize);
    assert(buff->work_mem != NULL && buff->ERI_mem != NULL);
    
    simint_initialize_shell(&buff->NAI_shell1);
    simint_initialize_shell(&buff->NAI_shell2);
    simint_initialize_multi_shellpair(&buff->bra_pair);
    simint_initialize_multi_shellpair(&buff->ket_pair);
    
    *buff_ = buff;
}

// Destroy a Simint buffer structure
void CMS_destroy_Simint_buff(simint_buff_t buff)
{
    buff->work_msize = 0;
    buff->ERI_msize  = 0;
    SIMINT_FREE(buff->work_mem);
    SIMINT_FREE(buff->ERI_mem);
    simint_free_shell(&buff->NAI_shell1);
    simint_free_shell(&buff->NAI_shell2);
    simint_free_multi_shellpair(&buff->bra_pair);
    simint_free_multi_shellpair(&buff->ket_pair);
}

// Calculate shell quartet pairs (M_i N_i|P_j Q_j)
void CMS_calc_ERI_pairs(
    shell_t *shells, const int n_bra_pair, const int n_ket_pair,
    int *M_list, int *N_list, int *P_list, int *Q_list, simint_buff_t buff
)
{
    // 1. Check if we need to reallocate ERI result buffer
    int total_ints = 0;
    for (int i = 0; i < n_bra_pair; i++)
    {
        int am_M = shells[M_list[i]].am;
        int am_N = shells[N_list[i]].am;
        int ncart_MN = NCART(am_M) * NCART(am_N);
        for (int j = 0; j < n_ket_pair; j++)
        {
            int am_P = shells[P_list[j]].am;
            int am_Q = shells[Q_list[j]].am;
            int ncart_PQ = NCART(am_P) * NCART(am_Q);
            total_ints += ncart_MN * ncart_PQ;
        }
    }
    if (sizeof(double) * total_ints > buff->ERI_msize)
    {
        buff->ERI_msize = sizeof(double) * total_ints;
        SIMINT_FREE(buff->ERI_mem);
        buff->ERI_mem = SIMINT_ALLOC(buff->ERI_msize);
        assert(buff->ERI_mem != NULL);
    }
    
    // 2. Calculate shell quartets one by one (not batched yet)
    double *work_mem = buff->work_mem;
    double *ERI_mem  = buff->ERI_mem;
    multi_sp_t *bra_pair = &buff->bra_pair;
    multi_sp_t *ket_pair = &buff->ket_pair;
    total_ints = 0;
    for (int i = 0; i < n_bra_pair; i++)
    {
        shell_t *M_shell = shells + M_list[i];
        shell_t *N_shell = shells + N_list[i];
        int am_M = shells[M_list[i]].am;
        int am_N = shells[N_list[i]].am;
        int ncart_MN = NCART(am_M) * NCART(am_N);
        simint_create_multi_shellpair(1, M_shell, 1, N_shell, bra_pair, 0);
        for (int j = 0; j < n_ket_pair; j++)
        {
            shell_t *P_shell = shells + P_list[j];
            shell_t *Q_shell = shells + Q_list[j];
            int am_P = shells[P_list[j]].am;
            int am_Q = shells[Q_list[j]].am;
            int ncart_PQ = NCART(am_P) * NCART(am_Q);
            simint_create_multi_shellpair(1, P_shell, 1, Q_shell, ket_pair, 0);
            int ret = simint_compute_eri(bra_pair, ket_pair, 0.0, work_mem, ERI_mem + total_ints);
            if (ret > 0) total_ints += ncart_MN * ncart_PQ;
        }
    }
}

// Calculate shell quartet pairs (N_i M_i|Q_j P_j) and unfold all ERI 
// results to form a matrix.
void CMS_calc_ERI_pairs_to_mat(
    shell_t *shells, const int n_bra_pair, const int n_ket_pair,
    int *M_list, int *N_list, int *P_list, int *Q_list,
    simint_buff_t buff, H2P_dense_mat_t mat
)
{
    // 1. Calculate all shell quartets
    CMS_calc_ERI_pairs(
        shells, n_bra_pair, n_ket_pair,  
        N_list, M_list, Q_list, P_list, buff
    );
    
    // 2. Adjust output matrix size
    int nrow = 0, ncol = 0;
    for (int i = 0; i < n_bra_pair; i++)
    {
        int am_M = shells[M_list[i]].am;
        int am_N = shells[N_list[i]].am;
        int ncart_MN = NCART(am_M) * NCART(am_N);
        nrow += ncart_MN;
    }
    for (int j = 0; j < n_ket_pair; j++)
    {
        int am_P = shells[P_list[j]].am;
        int am_Q = shells[Q_list[j]].am;
        int ncart_PQ = NCART(am_P) * NCART(am_Q);
        ncol += ncart_PQ;
    }
    H2P_dense_mat_resize(mat, nrow, ncol);
    
    // 3. Unfold ERI results to a matrix
    double *ERI_mem = buff->ERI_mem;
    int total_ints = 0;
    int row_idx = 0;
    for (int i = 0; i < n_bra_pair; i++)
    {
        int am_M = shells[M_list[i]].am;
        int am_N = shells[N_list[i]].am;
        int ncart_MN = NCART(am_M) * NCART(am_N);
        
        int col_idx = 0;
        for (int j = 0; j < n_ket_pair; j++)
        {
            int am_P = shells[P_list[j]].am;
            int am_Q = shells[Q_list[j]].am;
            int ncart_PQ = NCART(am_P) * NCART(am_Q);
            
            double *NMQP_ERI = ERI_mem + total_ints;
            double *mat_blk  = mat->data + row_idx * mat->ld + col_idx;
            
            for (int k = 0; k < ncart_MN; k++)
            {
                double *mat_blk_row = mat_blk  + k * mat->ld;
                double *ERI_ket_row = NMQP_ERI + k * ncart_PQ;
                memcpy(mat_blk_row, ERI_ket_row, sizeof(double) * ncart_PQ);
            }
            
            col_idx += ncart_PQ;
            total_ints += ncart_MN * ncart_PQ;
        }
        
        row_idx += ncart_MN;
    }
}

