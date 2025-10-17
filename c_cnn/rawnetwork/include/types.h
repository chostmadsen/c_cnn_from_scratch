#ifndef TYPES_H
#define TYPES_H

#include <stdlib.h>

typedef float elm_t;

typedef struct {
    size_t m;
    size_t n;
    size_t o;
    elm_t *arr;
} Tensor;

typedef struct {
    size_t m;
    size_t n;
    size_t o;
    size_t m_stride;
    size_t n_stride;
    elm_t bias;
    elm_t *arr;
} Kernel;

typedef struct {
    size_t m;
    size_t n;
    size_t m_stride;
    size_t n_stride;
} Pooler;

typedef struct {
    Tensor *weights;
    Tensor *biases;
} Dense;

typedef struct {
    Kernel **kernels;
    size_t num;
} Convolutional;

#endif // TYPES_H
