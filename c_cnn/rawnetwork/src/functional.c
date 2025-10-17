#include <stdio.h>
#include <string.h>
#include "types.h"
#include "functional.h"

/**
 * Frees all memory associated with a tensor. If tensor is NULL, passes.
 *
 * @param tensor: tensor to be freed.
 */
void free_tensor(Tensor *tensor) {
    if (tensor == NULL) return;
    free(tensor->arr);
    free(tensor);
}

/**
 * Frees all memory associated with a kernel. If kernel is NULL, passes.
 *
 * @param kernel: kernel to be freed.
 */
void free_kernel(Kernel *kernel) {
    if (kernel == NULL) return;
    free(kernel->arr);
    free(kernel);
}

/**
 * Frees all memory associated with a dense layer. If dense is NULL, passes.
 *
 * @param dense: dense layer to be freed.
 */
void free_dense(Dense *dense) {
    if (dense == NULL) return;
    free_tensor(dense->weights);
    free_tensor(dense->biases);
    free(dense);
}

/**
 * Frees all memory associated with a convolutional layer. If convolutional is NULL, passes.
 *
 * @param convolutional: convolutional layer to be freed.
 */
void free_convolutional(Convolutional *convolutional) {
    if (convolutional == NULL) return;
    for (size_t kernel = 0; kernel < convolutional->num; kernel++) {
        free_kernel(convolutional->kernels[kernel]);
    }
    free(convolutional->kernels);
    free(convolutional);
}

/**
 * Combines an array of tensors with same-sized matrices into a single tensor.
 * Frees combined tensors.
 * Caller is responsible for freeing returned tensor & array.
 *
 * @param tensors: tensors to be combined.
 * @param num: number of tensors.
 *
 * @return: combined tensor. NULL for any misshaped tensors or malloc fail.
 */
Tensor *combine(Tensor **tensors, const size_t num) {
    // dimension setup
    const size_t m = tensors[0]->m;
    const size_t n = tensors[0]->n;

    // dimensionality check and setup
    size_t o = tensors[0]->o;
    for (size_t tens = 1; tens < num; tens++) {
        if (m != tensors[tens]->m || n != tensors[tens]->n) {
            // dimension mismatch
            fprintf(stderr, "Dimension mismatch: m (%zu) != m (%zu) || n (%zu) != n (%zu).\n",
                m, tensors[tens]->m, n, tensors[tens]->n);
            return NULL;
        }
        // o accumulation
        o += tensors[tens]->o;
    }

    // malloc
    elm_t *res_arr = realloc(tensors[0]->arr, m * n * o * sizeof(elm_t));
    if (res_arr == NULL) {
        // malloc fail
        fprintf(stderr, "Failed malloc: Tensor sized %zu x %zu x %zu.\n", m, n, o);
        return NULL;
    }

    // tensor combination
    elm_t *dst = res_arr + m * n * tensors[0]->o;
    for (size_t tens = 1; tens < num; tens++) {
        const size_t len = m * n * tensors[tens]->o;
        memcpy(dst, tensors[tens]->arr, len * sizeof(elm_t));
        dst += len;
        // free added tensors
        free_tensor(tensors[tens]);
        tensors[tens] = NULL;
    }

    // struct setup and return
    tensors[0]->arr = res_arr;
    tensors[0]->o = o;
    return tensors[0];
}

/**
 * Flattens a tensor. If tensor is NULL, passes.
 *
 * @param tensor: tensor to be flattened.
 */
void flatten(Tensor *tensor) {
    if (tensor == NULL) return;
    tensor->n = tensor->m * tensor->n * tensor->o;
    tensor->m = 1; tensor->o = 1;
}

/**
 * Finds the argmax of a flat tensor.
 *
 * @param tensor: tensor to be run through argmax.
 *
 * @return: argmax of flat tensor. (size_t) - 1 for invalid tensor.
 */
size_t argmax(const Tensor *tensor) {
    if (tensor == NULL || tensor->m != 1 || tensor->o != 1) {
        // invalid tensor
        fprintf(stderr, "Invalid tensor: argmax requires a flat tensor.\n");
        return (size_t) - 1;
    }

    // find argmax
    elm_t max_val = tensor->arr[0];
    size_t max_idx = 0;
    for (size_t idx = 1; idx < tensor->n; idx++) {
        if (tensor->arr[idx] > max_val) {
            max_val = tensor->arr[idx];
            max_idx = idx;
        }
    }
    return max_idx;
}
