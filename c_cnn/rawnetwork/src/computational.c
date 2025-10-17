#include <stdio.h>
#include <string.h>
#include "types.h"
#include "functional.h"
#include "computational.h"

/*--------------------------------------------------------------------------------------------------------------------*/

static elm_t linear_sum_(const elm_t *vec, const elm_t *opp, const Tensor *t_opp, const size_t col) {
    elm_t res = 0;
    for (size_t elm = 0; elm < t_opp->m; elm ++) {
        // sum accumulation
        res += vec[elm] * opp[elm * t_opp->n + col];
    }
    return res;
}

static elm_t cdot_(const elm_t *mat, const Tensor *t_mat, const elm_t *kernel, const Kernel *k_kernel,
    const size_t row, const size_t col) {
    // dot product
    elm_t res = 0;
    for (size_t row_k = 0; row_k < k_kernel->m; row_k++) {
        for (size_t col_k = 0; col_k < k_kernel->n; col_k++) {
            // dot product accumulation
            res += kernel[row_k * k_kernel->n + col_k] * mat[(row + row_k) * t_mat->n + (col + col_k)];
        }
    }
    return res;
}

static elm_t max_pool_(const elm_t *mat, const Tensor *t_mat, const Pooler *pooler,
    const size_t row, const size_t col) {
    // max pool
    elm_t res = mat[row * t_mat->n + col];
    for (size_t row_k = 0; row_k < pooler->m; row_k++) {
        for (size_t col_k = 0; col_k < pooler->n; col_k++) {
            // max_pool
            const elm_t elm = mat[(row + row_k) * t_mat->n + (col + col_k)];
            if (elm > res) res = elm;
        }
    }
    return res;
}

static void matmul_(elm_t *targ, const elm_t *main, const Tensor *t_main, const elm_t *opp, const Tensor *t_opp) {
    // lone matmul operation
    for (size_t row = 0; row < t_main->m; row++) {
        for (size_t col = 0; col < t_opp->n; col++) {
            // sum row and col
            targ[row * t_opp->n + col] = linear_sum_(&main[row * t_main->n], opp, t_opp, col);
        }
    }
}

static void conv_(const Tensor *targ, const elm_t *main, const Tensor *t_main,
    const elm_t *kernel, const Kernel *k_kernel) {
    // lone convolution operation
    for (size_t row = 0; row < targ->m; row++) {
        for (size_t col = 0; col < targ->n; col++) {
            // dot product
            targ->arr[row * targ->n + col] += cdot_(main, t_main, kernel, k_kernel, row, col) + k_kernel->bias;
        }
    }
}

static void pool_(elm_t *targ, const Tensor *t_targ, const elm_t *main, const Tensor *t_main, const Pooler *pooler) {
    // lone pooling operation
    for (size_t row = 0; row < t_targ->m; row++) {
        for (size_t col = 0; col < t_targ->n; col++) {
            // max pool
            targ[row * t_targ->n + col] = max_pool_(main, t_main, pooler, row, col);
        }
    }
}

/*--------------------------------------------------------------------------------------------------------------------*/

/**
 * Sums an array of same-shaped tensors together.
 * Caller is responsible for freeing returned tensor & array.
 *
 * @param tensors: array of tensors to be summed.
 * @param num: number of tensors to be summed.
 *
 * @return: Pointer to summed tensors. NULL with any misshaped tensors or malloc fail.
 */
Tensor *sum(Tensor **tensors, const size_t num) {
    // dimension setup
    const size_t m = tensors[0]->m;
    const size_t n = tensors[0]->n;
    const size_t o = tensors[0]->o;

    // dimensionality check
    for (size_t tens = 1; tens < num; tens++) {
        if (m != tensors[tens]->m || n != tensors[tens]->n || o != tensors[tens]->o) {
            fprintf(stderr, "Dimension mismatch: m (%zu) != m (%zu) || n (%zu) != n (%zu) || o (%zu) != o (%zu).\n",
                m, tensors[tens]->m, n, tensors[tens]->n, o, tensors[tens]->o);
            return NULL;
        }
    }

    // malloc
    const size_t out_size = m * n * o;
    elm_t *res_arr = malloc(out_size * sizeof(elm_t));
    Tensor *res = malloc(sizeof(Tensor));
    if (res_arr == NULL || res == NULL) {
        // malloc fail
        fprintf(stderr, "Failed malloc: Tensor sized %zu x %zu x %zu.\n", m, n, o);
        free(res_arr); free_tensor(res);
        return NULL;
    }

    // sum operation
    memcpy(res_arr, tensors[0]->arr, out_size * sizeof(elm_t));
    for (size_t itm = 1; itm < num; itm++) {
        for (size_t elm = 0; elm < out_size; elm++) {
            // value accumulation
            res_arr[elm] += tensors[itm]->arr[elm];
        }
    }

    // struct setup
    res->m = m; res->n = n; res->o = o;
    res->arr = res_arr;
    return res;
}

/**
 * Matrix multiplication of two tensors, treating the 3rd dimension as a batch.
 * Caller is responsible for freeing returned tensor & array.
 *
 * @param main: main tensor.
 * @param opp: opposite tensor.
 *
 * @return: matmul of tensors. NULL with any dimensional mismatch, failed operation, or malloc fail.
 */
Tensor *matmul(const Tensor *main, const Tensor *opp) {
    // dimension setup
    const size_t m = main->m;
    const size_t t = main->n;
    const size_t n = opp->n;
    const size_t o = main->o;

    // dimensionality check
    if (o != opp->o || t != opp->m) {
        fprintf(stderr, "Dimensional mismatch: a_o (%zu) != b_o: (%zu) || a_n (%zu) != b_m (%zu).\n",
            o, opp->o, t, opp->m);
        return NULL;
    }

    // malloc
    const size_t out_size = m * n * o;
    elm_t *res_arr = malloc(out_size * sizeof(elm_t));
    Tensor *res = malloc(sizeof(Tensor));
    if (res_arr == NULL || res == NULL) {
        // malloc fail
        fprintf(stderr, "Failed malloc: Tensor sized %zu x %zu x %zu.\n", m, n, o);
        free(res_arr); free_tensor(res);
        return NULL;
    }

    // struct setup
    res->m = m; res->n = n; res->o = o;
    res->arr = res_arr;

    // matmul operation
    for (size_t mat = 0; mat < o; mat++) {
        matmul_(&res->arr[m * mat * n], &main->arr[mat * m * t], main, &opp->arr[mat * t * n],  opp);
    }
    return res;
}

/**
 * Convolution of a batch of tensors with a same-sized batch of kernels.
 * Caller is responsible for freeing returned tensor & array.
 *
 * @param channels: tensors to be convolved.
 * @param kernels: convolutional kernels.
 *
 * @return: convolved tensors. NULL with any dimensional mismatch, failed operation, or malloc fail.
 */
Tensor *conv(const Tensor *channels, const Kernel *kernels) {
    // dimension setup
    const size_t m = channels->m;
    const size_t n = channels->n;
    const size_t m_k = kernels->m;
    const size_t n_k = kernels->n;
    const size_t o = channels->o;

    // dimensionality check
    if (o != kernels->o || m < kernels->m || n < kernels->n) {
        fprintf(stderr, "Invalid convolution: oversized kernel or channels (%zu) != kernels (%zu).\n", o, kernels->o);
        return NULL;
    }

    // result dimension setup
    const size_t m_res = (m - m_k) / kernels->m_stride + 1;
    const size_t n_res = (n - n_k) / kernels->n_stride + 1;

    // malloc
    const size_t out_size = m_res * n_res;
    elm_t *res_arr = calloc(out_size * sizeof(elm_t), sizeof(elm_t));
    Tensor *res = malloc(sizeof(Tensor));
    if (res_arr == NULL || res == NULL) {
        // malloc fail
        fprintf(stderr, "Failed malloc: Tensor sized %zu x %zu x %zu.\n", m_res, n_res, (size_t)1);
        free(res_arr); free_tensor(res);
        return NULL;
    }

    // struct setup
    res->m = m_res; res->n = n_res; res->o = o;
    res->arr = res_arr;

    // convolution operation
    for (size_t pair = 0; pair < o; pair++) {
        conv_(res, &channels->arr[pair * m * n], channels, &kernels->arr[pair * m_k * n_k], kernels);
    }
    return res;
}

/**
 * Max pooling of tensors.
 * Caller is responsible for freeing returned tensor & array.
 *
 * @param main: tensor to be pooled.
 * @param pooler: pooling kernel.
 *
 * @return: pooled tensors. NULL with any dimensional mismatch, failed operation, or malloc fail.
 */
Tensor *pool(const Tensor *main, const Pooler *pooler) {
    // dimension setup
    const size_t m = main->m;
    const size_t n = main->n;
    const size_t o = main->o;

    // dimensionality check
    if (main->m < pooler->m || main->n < pooler->n) {
        fprintf(stderr, "Invalid pooling: oversized pooling kernel.\n");
        return NULL;
    }

    // result dimension setup
    const size_t m_res = (m - pooler->m) / pooler->m_stride + 1;
    const size_t n_res = (n - pooler->n) / pooler->n_stride + 1;

    // malloc
    const size_t out_size = m_res * n_res * o;
    elm_t *res_arr = calloc(out_size * sizeof(elm_t), sizeof(elm_t));
    Tensor *res = malloc(sizeof(Tensor));
    if (res_arr == NULL || res == NULL) {
        // malloc fail
        fprintf(stderr, "Failed malloc: Tensor sized %zu x %zu x %zu.\n", m_res, n_res, (size_t)1);
        free(res_arr); free_tensor(res);
        return NULL;
    }

    // struct setup
    res->m = m_res; res->n = n_res; res->o = o;
    res->arr = res_arr;

    // pooling operation
    for (size_t mat = 0; mat < o; mat++) {
        pool_(&res->arr[mat * m_res * n_res], res, &main->arr[mat * m_res * n_res], main, pooler);
    }
    return res;
}
