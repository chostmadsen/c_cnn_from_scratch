#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include "types.h"

void free_tensor(Tensor *tensor);

void free_kernel(Kernel *kernel);

void free_dense(Dense *dense);

void free_convolutional(Convolutional *convolutional);

Tensor *combine(Tensor **tensors, size_t num);

void flatten(Tensor *tensor);

size_t argmax(const Tensor *tensor);

#endif // FUNCTIONAL_H
