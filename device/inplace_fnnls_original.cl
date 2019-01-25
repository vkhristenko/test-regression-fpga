//
// inplace fnnls.
// for simplicity assume row major storage of all the matrices
//

//
// definitions of constants
//
#define NUM_TIME_SAMPLES 10
#define NUM_TIME_SAMPLES_SQ 100 
// to easily switch between row major and column major matrix representations
#define M_LINEAR_ACCESS(M, row, col) M[(row) * (NUM_TIME_SAMPLES) + (col)]
typedef float data_type;
#define SIMPLE_SQRT(x) sqrt(x)

#define NNLS_LOCAL
#define NNLS_GLOBAL __global
#define NNLS_DEBUG
#undef NNLS_DEBUG

//
// Various Matrix and Vector Multiplications 
// C = A * A_T -> the result is symmetric matrix.
// the restult matrix C will be stored in column major
//
inline void transpose_multiply_m_m(__global data_type const* restrict A, 
                            NNLS_LOCAL data_type *restrict result);
inline void transpose_multiply_m_v_v(__global data_type const * restrict M, 
                              __global data_type const * restrict v, 
                              NNLS_LOCAL data_type * restrict result);
inline void multiply_m_v_v(NNLS_LOCAL data_type const * restrict M, 
                    NNLS_LOCAL data_type const * restrict v, 
                    NNLS_LOCAL data_type *restrict result);

inline void transpose_multiply_m_m(__global data_type const* restrict A, 
                            NNLS_LOCAL data_type *restrict result) {
#pragma loop_coalesce 2
#pragma ivdep
    for (int i=0; i<NUM_TIME_SAMPLES; ++i) {
#pragma ivdep
#pragma unroll 1
        for (int j=i; j<NUM_TIME_SAMPLES; ++j) {
            data_type tmp = 0.0f;
#pragma unroll 1
            for (int k=0; k<NUM_TIME_SAMPLES; ++k)
                tmp += M_LINEAR_ACCESS(A, k, i) * M_LINEAR_ACCESS(A, k, j);
            M_LINEAR_ACCESS(result, i, j) = tmp;
            M_LINEAR_ACCESS(result, j, i) = tmp;

/*
#pragma unroll 1
            for (int k=0; k<NUM_TIME_SAMPLES; ++k)
                M_LINEAR_ACCESS(result, i, j) += M_LINEAR_ACCESS(A, k, i) * 
                    M_LINEAR_ACCESS(A, k, j);
            M_LINEAR_ACCESS(result, j, i) = M_LINEAR_ACCESS(result, i, j);
            */
       }
    }
}

inline void transpose_multiply_m_v_v(__global data_type const *M, 
                              __global data_type const *v, 
                              NNLS_LOCAL data_type *restrict result) {
#pragma loop_coalesce 2
#pragma ivdep
    for (int i=0; i<NUM_TIME_SAMPLES; ++i) {
        data_type tmp = 0.0f;
#pragma unroll 1
#pragma ivdep
        for (int k=0; k<NUM_TIME_SAMPLES; ++k) {
            tmp += M_LINEAR_ACCESS(M, k, i) * v[k];
        }

        result[i] = tmp;
    }
}

inline void multiply_m_v_v(NNLS_LOCAL data_type const *M, 
                    NNLS_LOCAL data_type const *v, 
                    NNLS_LOCAL data_type *restrict result) {
#pragma loop_coalesce 2
#pragma ivdep
    for (int i=0; i<NUM_TIME_SAMPLES; ++i) {
        data_type tmp = 0.0f;
#pragma unroll 1
#pragma ivdep
        for (int k=0; k<NUM_TIME_SAMPLES; ++k) 
            tmp += M_LINEAR_ACCESS(M, i, k) * v[k];

        result[i] = tmp;
    }
}

//
// Swap Functions
//
inline void swap_permm(NNLS_LOCAL data_type *permutation, int idx1, int idx2);
inline void swap_row_column(NNLS_LOCAL data_type *pM, 
                     int i, int j, int full_size, int view_size);
inline void swap_element(NNLS_LOCAL data_type *pv, int i, int j);
inline void swap_perm_element(NNLS_LOCAL int *pv, int i, int j);

inline void swap_element(NNLS_LOCAL data_type *pv,
                  int i, int j) {
    data_type tmp = pv[i];
    pv[i] = pv[j];
    pv[j] = tmp;
}

inline void swap_perm_element(NNLS_LOCAL int *pv,
                       int i, int j) {
    int tmp = pv[i];
    pv[i] = pv[j];
    pv[j] = tmp;
}

// same precondition: index_j > index_i
#define SWAP_LOOP(start, finish, matrix, index_i, index_j) \
    for (int elem=start; elem<finish; ++elem) { \
        data_type tmp = M_LINEAR_ACCESS(matrix, index_i, elem); \
        M_LINEAR_ACCESS(matrix, index_i, elem) = \
            M_LINEAR_ACCESS(matrix, index_j, elem); \
        M_LINEAR_ACCESS(matrix, index_j, elem) = tmp; \
        M_LINEAR_ACCESS(matrix, elem, index_i) = \
            M_LINEAR_ACCESS(matrix, elem, index_j); \
        M_LINEAR_ACCESS(matrix, elem, index_j) = tmp; \
    }

// precondition: j > i
inline void swap_row_column(NNLS_LOCAL data_type *pM, 
                     int i, int j, int full_size, int view_size) {
    // diagnoal
    data_type tmptmp = M_LINEAR_ACCESS(pM, i, i);
    M_LINEAR_ACCESS(pM, i, i) = M_LINEAR_ACCESS(pM, j, j);
    M_LINEAR_ACCESS(pM, j, j) = tmptmp;

#pragma ivdep
    for (int elem=0; elem<NUM_TIME_SAMPLES; ++elem) {
        if (elem==i || elem==j)
            continue;

        data_type tmp = M_LINEAR_ACCESS(pM, i, elem);
        M_LINEAR_ACCESS(pM, i, elem) =
            M_LINEAR_ACCESS(pM, j, elem);
        M_LINEAR_ACCESS(pM, j, elem) = tmp;
        M_LINEAR_ACCESS(pM, elem, i) =
            M_LINEAR_ACCESS(pM, elem, j);
        M_LINEAR_ACCESS(pM, elem, j) = tmp;
    }

/*
#pragma unroll 1
    SWAP_LOOP(0, i, pM, i, j)
#pragma unroll 1
    SWAP_LOOP(i+1, j, pM, i, j)
#pragma unroll 1
    SWAP_LOOP(j+1, NUM_TIME_SAMPLES, pM, i, j)
*/
}

//
// Cholesky Decomposition + Forward/Backward Substituion Solvers
//
inline void cholesky_decomp(NNLS_LOCAL data_type const* restrict pM,
                     NNLS_LOCAL data_type * restrict pL,
                     int full_size, int view_size);
inline void fused_cholesky_forward_substitution_solver_rcadd(
        NNLS_LOCAL data_type const * restrict pM, 
        NNLS_LOCAL data_type *restrict pL, 
        NNLS_LOCAL data_type const * restrict pb,
        NNLS_LOCAL data_type *restrict py, 
        int full_size, int view_size);
inline void fused_cholesky_forward_substitution_solver_inbetween_removal(
        NNLS_LOCAL data_type const * restrict pM, 
        NNLS_LOCAL data_type *restrict pL, 
        NNLS_LOCAL data_type const * restrict pb, 
        NNLS_LOCAL data_type *restrict py, 
        int position, int full_size, int view_size);

inline void cholesky_decomp(NNLS_LOCAL data_type const* restrict pM,  
                     NNLS_LOCAL data_type *restrict pL, 
                     int full_size, int view_size) {
    for (int i=0; i<view_size; ++i) {

        // first compute elements to the left of the diagoanl
        data_type sumsq = 0.0f;
        for (int j=0; j<i; ++j) {

            data_type sumsq2 = 0.0f;
            for (int k=0; k<j; ++k) {
                sumsq2 += M_LINEAR_ACCESS(pL, i, k) *
                    M_LINEAR_ACCESS(pL, j, k);
            }

            // compute the i,j : i>j. elements to the left of the diagonal
            M_LINEAR_ACCESS(pL, i, j) =
                (M_LINEAR_ACCESS(pM, i, j) - sumsq2)
                / M_LINEAR_ACCESS(pL, j, j);

            // needed to compute diagonal element
            sumsq += M_LINEAR_ACCESS(pL, i, j) *
                M_LINEAR_ACCESS(pL, i, j);
        }

        // second, compute the diagonal element
        M_LINEAR_ACCESS(pL, i, i) =
            SIMPLE_SQRT(M_LINEAR_ACCESS(pM, i, i) - sumsq);
    }
}

inline void fused_cholesky_forward_substitution_solver_rcadd(
        NNLS_LOCAL data_type const *pM, 
        NNLS_LOCAL data_type *restrict pL, 
        NNLS_LOCAL data_type const *pb,
        NNLS_LOCAL data_type *restrict py, 
        int full_size, int view_size) {
    /*
     * cholesky decomposition using partial computation
     * only compute values in the row that was just added
     */
    data_type sum = 0.0f;
    int row = view_size - 1;
    data_type total = pb[row];
    for (int col=0; col<row; ++col) {
        data_type sum2 = 0;
        for (int j=0; j<col; ++j) {
            sum2 += M_LINEAR_ACCESS(pL, row, j) *
                M_LINEAR_ACCESS(pL, col, j);
        }

        // compute the row,col : row > col. elements to the left of the diagonal
        data_type value_row_col = (M_LINEAR_ACCESS(pM, row, col) - sum2)
            / M_LINEAR_ACCESS(pL, col, col);

        // update the sum needed for forward substitution
        total -= value_row_col * py[col];

        // needed to compute the diagonal element
        sum += value_row_col * value_row_col;

        // set the value
        M_LINEAR_ACCESS(pL, row, col) = value_row_col;
    }

    // compute the diagonal value
    data_type diag_value = SIMPLE_SQRT(M_LINEAR_ACCESS(pM, row, row)
        - sum);
    M_LINEAR_ACCESS(pL, row, row) = diag_value;

    // compute the new value (view_size - 1 ) of the result of forward substitution
//    __local data_type y_last = total / diag_value;
    py[row] = total / diag_value;
}

inline void fused_cholesky_forward_substitution_solver_inbetween_removal(
        NNLS_LOCAL data_type const *pM, 
        NNLS_LOCAL data_type *restrict pL, 
        NNLS_LOCAL data_type const *pb, 
        NNLS_LOCAL data_type *restrict py, 
        int position, int full_size, int view_size) {
    // only for elements with >= position
    for (int i=position; i<view_size; ++i) {

        // first compute elements to the left of the diagoanl
        data_type sumsq = 0.0f;
        data_type total = pb[i];
        for (int j=0; j<i; ++j) {

            data_type sumsq2 = 0.0f;
            for (int k=0; k<j; ++k) {
                sumsq2 += M_LINEAR_ACCESS(pL, i, k) *
                    M_LINEAR_ACCESS(pL, j, k);
            }

            // compute the i,j : i>j. elements to the left of the diagonal
            data_type value_i_j =
                (M_LINEAR_ACCESS(pM, i, j) - sumsq2)
                / M_LINEAR_ACCESS(pL, j, j);
            M_LINEAR_ACCESS(pL, i, j) = value_i_j;
            total -= value_i_j * py[j];

            // needed to compute diagonal element
            sumsq += value_i_j * value_i_j;
        }

        // second, compute the diagonal element
        data_type value_i_i =
            SIMPLE_SQRT(M_LINEAR_ACCESS(pM, i, i) - sumsq);
        M_LINEAR_ACCESS(pL, i, i) = value_i_i;

        // compute the i-th solution value for forward sub
        py[i] = total / value_i_i;
    }
}

inline void solve_forward_substitution(NNLS_LOCAL data_type const *pM, 
                                NNLS_LOCAL data_type const* pb, 
                                NNLS_LOCAL data_type *restrict psolution,
                                int full_size, int view_size);
inline void solve_forward_substitution(NNLS_LOCAL data_type const *pM, 
                                NNLS_LOCAL data_type const* pb, 
                                NNLS_LOCAL data_type *restrict psolution,
                                int full_size, int view_size) {
    // very first element is trivial
    psolution[0] = pb[0] / pM[0];

    // for the rest
    for (int i=1; i<view_size; ++i) {
        data_type total = pb[i];
        for (int j=0; j<i; j++) {
            total -= M_LINEAR_ACCESS(pM, i, j) * psolution[j];
        }

        // set the value of the i-solution
        psolution[i] = total / M_LINEAR_ACCESS(pM, i, i);
    }
}

/*
 * Solving Ax=b using backward substitution
 * pM - Lower triangular matrix.
 * Note: we take lower triangular (not upper triangular) to remove the transposition
 * step required otherwise.
 */
inline void solve_backward_substitution(NNLS_LOCAL data_type const * restrict pM, 
                                 NNLS_LOCAL data_type const * restrict pb, 
                                 NNLS_LOCAL data_type *restrict psolution,
                                 int full_size, int view_size);
inline void solve_backward_substitution(NNLS_LOCAL data_type const * restrict pM, 
                                 NNLS_LOCAL data_type const * restrict pb, 
                                 NNLS_LOCAL data_type *restrict psolution,
                                 int full_size, int view_size) {
    // very last element is trivial
    psolution[view_size-1] = pb[view_size-1] /
        M_LINEAR_ACCESS(pM, view_size-1, view_size-1);

    // for the rest
    for (int i=view_size-2; i>=0; --i) {
//        data_type total = pb[i];
        data_type total = 0.0f;
        for (int j=i+1; j<view_size; ++j) {
            total += M_LINEAR_ACCESS(pM, j, i) * psolution[j];
        }

        psolution[i] = (pb[i] - total) / M_LINEAR_ACCESS(pM, i, i);
    }
}

inline void permutation_identity(NNLS_LOCAL int *permutation);
inline void permutation_identity(NNLS_LOCAL int *permutation) {
#pragma unroll 1
    for (int i=0; i<NUM_TIME_SAMPLES; ++i)
        permutation[i] = i;
}

//
// debug functions
//
#ifdef NNLS_DEBUG

#define PRINT_FLOAT(value) printf("%f\n", value)
#define PRINT_DOUBLE(value) printf("%e\n", value)

void print_vector(data_type const* pv, int size);
void print_matrix(data_type const *pm, int size);
void print_permutation(int const *pv, int size);
void print_gmatrix(__global data_type const *pm, int size);
void print_vector(data_type const* pv, int size) {
    for (int i=0; i<size; ++i)
        PRINT_FLOAT(pv[i]);
}

void print_matrix(data_type const *pm, int size) {
    for (int i=0; i<size; ++i) {
        for (int j=0; j<size; ++j) {
            printf("%f  ", pm[i*NUM_TIME_SAMPLES + j]);
        }
        printf("\n");
    }
}

void print_gmatrix(__global data_type const *pm, int size) {
    for (int i=0; i<size; ++i) {
        for (int j=0; j<size; ++j) {
            printf("%f  ", pm[i*NUM_TIME_SAMPLES + j]);
        }
        printf("\n");
    }
}

void print_permutation(int const *pv, int size) {
    for (int i=0; i<size; ++i)
        printf("%d\n", pv[i]);
}
#endif

__attribute__((max_global_work_dim(0)))
__kernel
void inplace_fnnls(__global data_type const * restrict As,
                   __global data_type const * restrict bs,
                   __global data_type *restrict xs,
                   unsigned int size,
                   double const epsilon,
                   unsigned int const max_iterations) {

#pragma ivdep
    for (unsigned int idx=0; idx<size; ++idx) {
        __global data_type const *A = As + idx * NUM_TIME_SAMPLES_SQ;
        __global data_type const *b = bs + idx * NUM_TIME_SAMPLES;
        __global data_type *x = xs + idx * NUM_TIME_SAMPLES;

#pragma unroll 1
        for (int i=0; i<NUM_TIME_SAMPLES; ++i)
            x[i] = 0;

#ifdef NNLS_DEBUG
        printf("A = \n");
        print_gmatrix(A, NUM_TIME_SAMPLES);
#endif

        // initial setup
        int npassive = 0;
        NNLS_LOCAL data_type AtA[NUM_TIME_SAMPLES_SQ];
        NNLS_LOCAL data_type Atb[NUM_TIME_SAMPLES];
        NNLS_LOCAL data_type s[NUM_TIME_SAMPLES];
        NNLS_LOCAL data_type w[NUM_TIME_SAMPLES];
        NNLS_LOCAL data_type final_s[NUM_TIME_SAMPLES];
        NNLS_LOCAL int permutation[NUM_TIME_SAMPLES];
        NNLS_LOCAL data_type AtAx[NUM_TIME_SAMPLES];
        NNLS_LOCAL data_type pL[NUM_TIME_SAMPLES_SQ];
        NNLS_LOCAL data_type py[NUM_TIME_SAMPLES];
        transpose_multiply_m_m(A, AtA);
        transpose_multiply_m_v_v(A, b, Atb);
        permutation_identity(permutation);

#pragma unroll 1
        for (int i=0; i<NUM_TIME_SAMPLES; ++i)
            final_s[i] = x[i];

#ifdef NNLS_DEBUG
        printf("permutaion = \n");
        print_permutation(permutation, NUM_TIME_SAMPLES);
        printf("AtA = \n");
        print_matrix(AtA, NUM_TIME_SAMPLES);
        printf("Atb = \n");
        print_vector(Atb, NUM_TIME_SAMPLES);
#endif

        // loop over all iterations. 
#pragma unroll 1
        for (unsigned int iter = 0; iter < max_iterations; ++iter) {
            int nactive = NUM_TIME_SAMPLES - npassive;
#ifdef NNLS_DEBUG
            printf("*************\n");
            printf("iteration = %d\n", iter);
            printf("nactive = %d\n", nactive);
                printf("final_s = \n");
                print_vector(final_s, NUM_TIME_SAMPLES);
#endif

            // exit condition
            if (!nactive)
                break;

            // update the gradient vector but only for the active constraints
            multiply_m_v_v(AtA, final_s, AtAx);
#ifdef NNLS_DEBUG
                printf("AtAx = \n");
                print_vector(AtAx, NUM_TIME_SAMPLES);
#endif
            data_type max_w_value = FLT_MIN; int max_w_idx; 
            int iii = 0;
#pragma unroll 1
            for (int i=npassive; i<NUM_TIME_SAMPLES; ++i) {
                w [ i ] = Atb [ i ] - AtAx [ i ];

                if (iii == 0 || w [ i ] > max_w_value) {
                    max_w_value = w [ i ];
                    max_w_idx = i;
                }
                ++iii;
            }

#ifdef NNLS_DEBUG
                printf("w = \n");
                print_vector(w, NUM_TIME_SAMPLES);
                printf("max_w_value = %f\n", max_w_value);
                printf("max_w_idx = %d\n", max_w_idx);
#endif

            // convergence check
            if (max_w_value < epsilon)
                break;

            // note, max_w_idx already points to the right location in the vector
            // run swaps
            swap_row_column(AtA, npassive, max_w_idx, 
                NUM_TIME_SAMPLES, NUM_TIME_SAMPLES);
            swap_element(Atb, npassive, max_w_idx);
            swap_element(final_s, npassive, max_w_idx);
            swap_perm_element(permutation, npassive, max_w_idx);

#ifdef NNLS_DEBUG
                printf("after swap AtA = \n");
                print_matrix(AtA, NUM_TIME_SAMPLES);
                printf("after swap Atb = \n");
                print_vector(Atb, NUM_TIME_SAMPLES);
                printf("after swap final_s = \n");
                print_vector(final_s, NUM_TIME_SAMPLES);
                printf("after swap permutation = \n");
                print_permutation(permutation, NUM_TIME_SAMPLES);
#endif

            ++npassive;

            // inner loop
            int inner_iteration = 0;
            int position_removed = -1;
            while (npassive > 0) {
                if (npassive == 1) {
                    // scalar case
                    data_type l_0_0 = sqrt(AtA[0]);
                    data_type y_0_0 = Atb[0] / l_0_0;
                    pL[0] = l_0_0;
                    py[0] = y_0_0;
                    s[0] = y_0_0 / l_0_0;
                    position_removed = -1;
#ifdef NNLS_DEBUG
                        printf("npassive == 1 branch\n");
#endif
                } else {
                    if (inner_iteration == 0) {
                    fused_cholesky_forward_substitution_solver_rcadd(
                        AtA, pL, Atb, py, NUM_TIME_SAMPLES, npassive);
                    //solve_backward_substitution(pL, py, s, NUM_TIME_SAMPLES, npassive);
#ifdef NNLS_DEBUG
                        printf("npassive != 1 else inner_iteration == 0 branch\n");
#endif
                    } else {
                    fused_cholesky_forward_substitution_solver_inbetween_removal(
                        AtA, pL, Atb, py, position_removed, NUM_TIME_SAMPLES, npassive);
                    //solve_backward_substitution(pL, py, s, NUM_TIME_SAMPLES, npassive);
#ifdef NNLS_DEBUG
                        printf("npassive != 1 else inner_iteration!=0 else branch\n");
#endif
                    }
                    solve_backward_substitution(pL, py, s, NUM_TIME_SAMPLES, npassive);
                }

#ifdef NNLS_DEBUG
                    printf("***** inner iteration %d *****\n", inner_iteration);
                    printf("L = \n");
                    print_matrix(pL, NUM_TIME_SAMPLES);
                    printf("s = \n");
                    print_vector(s, NUM_TIME_SAMPLES);
                    printf("py = \n");
                    print_vector(py, NUM_TIME_SAMPLES);
#endif

                // update the elements from the passive set
                data_type min_s_value = s [ 0 ];
#pragma unroll 1
                for (int i=1; i<npassive; ++i) {
                    if (s [ i ] < min_s_value)
                        min_s_value = s [ i ];
                }

#ifdef NNLS_DEBUG
                    printf("min_s_value = %f\n", min_s_value);
                    printf("npassive = %d\n", npassive);
#endif

                // if elements of the passive set are all positive
                // set the solution vector and break from the inner loop
                if (min_s_value > 0.0f) {
#ifdef NNLS_DEBUG
                    printf("min_s_value = %f and branching here\n", min_s_value);
#endif
#pragma unroll 1
                    for (int i=0; i<npassive; ++i)
                        final_s [ i ] = s [ i ];
                    break;
                }

#ifdef NNLS_DEBUG
                if (iter%100 == 0) {
                    printf("final_s = \n");
                    print_vector(final_s, NUM_TIME_SAMPLES);
                }
#endif

                // 
                data_type alpha = FLT_MAX;
                int alpha_idx = 0;
#pragma unroll 1
                for (int i=0; i<npassive; ++i) {
                    if (s [ i ] <= 0.0f) {
                        data_type const ratio = final_s [ i ] / 
                            ( final_s [ i ] - s [ i ] );
                        if (ratio < alpha) {
                            alpha = ratio;
                            alpha_idx = i;
                        }
                    }
                }

                if (alpha == FLT_MAX) {
#pragma unroll 1
                    for (int i=0; i<npassive; ++i) 
                        final_s [ i ] = s [ i ];
                    break;
                }

                // update solution using alpha
#pragma unroll 1
                for (int i=0; i<npassive; ++i) {
                    final_s [ i ] += alpha * (s [ i ] - final_s [ i ]);
                }
                final_s [ alpha_idx] = 0;
                --npassive;

                // run swaps
                swap_row_column(AtA, alpha_idx, npassive,
                    NUM_TIME_SAMPLES, NUM_TIME_SAMPLES);
                swap_element(Atb, npassive, alpha_idx);
                swap_element(final_s, npassive, alpha_idx);
                swap_perm_element(permutation, npassive, alpha_idx);
                inner_iteration++;
                position_removed = alpha_idx;

#ifdef NNLS_DEBUG
                printf("final_s = \n");
                print_vector(final_s, NUM_TIME_SAMPLES);
#endif
            }
        }

#ifdef NNLS_DEBUG
        printf("final_s = \n");
        print_vector(final_s, NUM_TIME_SAMPLES);
        printf("permutation = \n");
        print_permutation(permutation, NUM_TIME_SAMPLES);
#endif

        // permute the solution vector back to have x[i] sit at the original position
#pragma unroll 1
        for (int i=0; i<NUM_TIME_SAMPLES; ++i) {
            x [ permutation[i] ] = final_s [i];
        }
    }
}
