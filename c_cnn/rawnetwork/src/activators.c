#include <math.h>
#include "types.h"
#include "activators.h"

#include <stdio.h>

/**
 * Rectified Linear Unit (ReLU) applied element-wise on a tensor.
 *
 * @param tens: tensor to have ReLU applied.
 */
void relu(const Tensor *tens) {
    for (size_t elm = 0; elm < tens->m * tens->n * tens->o; elm++) {
        if (tens->arr[elm] < 0.0) tens->arr[elm] = (elm_t)0.0;
    }
}

/**
 *  Sigmoid applied element-wise on a tensor.
 *
 * @param tens: tensor to have sigmoid applied.
 */
void sigmoid(const Tensor *tens) {
    for (size_t elm = 0; elm < tens->m * tens->n * tens->o; elm++) {
        tens->arr[elm] = (elm_t)(1.0 / (1.0 + exp(-tens->arr[elm])));
    }
}

/**
 * Softmax applied element-wise on a tensor.
 * elements spanning the o-th dimension are treated separately, i.e. the o-th dimension is the batch dimension.
 *
 * @param tens: tensor to have softmax applied.
 */
void softmax(const Tensor *tens) {
    for (size_t mat = 0; mat < tens->o; mat++) {
        elm_t sum = 0;
        for (size_t elm = 0; elm < tens->m * tens->n; elm++) {
            sum += (elm_t)exp(tens->arr[mat * tens->m * tens->n + elm]);
        }
        for (size_t elm = 0; elm < tens->m * tens->n; elm++) {
            tens->arr[mat * tens->m * tens->n + elm] = (elm_t)(exp(tens->arr[mat * tens->m * tens->n + elm]) / sum);
        }
    }
}
