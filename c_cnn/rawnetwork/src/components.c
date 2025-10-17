#include <stdio.h>
#include "types.h"
#include "computational.h"
#include "functional.h"
#include "components.h"

/**
 * Dense layer function.
 * Automatically frees any intermediate values.
 * Caller is responsible for freeing returned tensor & array.
 *
 * @param input: activations.
 * @param dense: dense layer.
 * @param fn: activation function.
 *
 * @return: next layer activations. NULL for any failed operation or malloc fail.
 */
Tensor *dense(const Tensor *input, const Dense *dense, void (*fn)(const Tensor*)) {
    // matmul
    Tensor *alpha = matmul(input, dense->weights);

    // malloc
    Tensor **tensors = malloc(2 * sizeof(Tensor*));
    if (alpha == NULL || tensors == NULL) {
        // malloc fail
        fprintf(stderr, "Failed malloc: Tensor* or internal matmul fail.\n");
        free_tensor(alpha);
        if (tensors != NULL) {
            free_tensor(tensors[0]);
            free_tensor(tensors[1]);
        }
        free(tensors);
        return NULL;
    }

    // bias
    tensors[0] = alpha; tensors[1] = dense->biases;
    Tensor *res = sum(tensors, 2);

    // free
    free_tensor(alpha); free(tensors);

    if (res == NULL) {
        // sum fail
        fprintf(stderr, "Failed operation: internal sum fail.\n");
        return NULL;
    }

    // activation function
    fn(res);

    // return
    return res;
}

/**
 * Convolutional layer function.
 * Automatically frees any intermediate values.
 * Caller is responsible for freeing returned tensor & array.
 *
 * @param input: channels.
 * @param kernels: convolutional kernels.
 * @param fn: activation function.
 *
 * @return: next layer channels. NULL for any failed operation or malloc fail.
 */
Tensor *convolution(const Tensor *input, const Convolutional *kernels, void (*fn)(const Tensor*)) {
    // malloc
    Tensor **alpha = malloc(kernels->num * sizeof(Tensor*));
    if (alpha == NULL) {
        // malloc fail
        fprintf(stderr, "Failed malloc: Tensor*.\n");
        return NULL;
    }

    for (size_t kernel = 0; kernel < kernels->num; kernel++) {
        // conv
        Tensor *out = conv(input, kernels->kernels[kernel]);
        if (out == NULL) {
            // conv fail
            fprintf(stderr, "Failed operation: internal conv fail.\n");
            // dump memory
            for (size_t idx = 0; idx < kernel; idx++) {
                free_tensor(alpha[idx]);
            }
            free(alpha);
            return NULL;
        }
        // conv accumulation
        alpha[kernel] = out;
    }

    // combine outputs
    Tensor *res = combine(alpha, kernels->num);
    if (res == NULL) {
        // combine fail
        fprintf(stderr, "Failed operation: internal combine fail.\n");
        // dump memory
        for (size_t idx = 0; idx < kernels->num; idx++) {
            free_tensor(alpha[idx]);
        }
        free(alpha);
        return NULL;
    }

    // activation function
    fn(res);

    // free and return
    free(alpha);
    return res;
}
