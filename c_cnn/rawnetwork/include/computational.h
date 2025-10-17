#ifndef COMPUTATIONAL_H
#define COMPUTATIONAL_H

#include "types.h"

Tensor *sum(Tensor **tensors, size_t num);

Tensor *matmul(const Tensor *main, const Tensor *opp);

Tensor *conv(const Tensor *channels, const Kernel *kernels);

Tensor *pool(const Tensor *main, const Pooler *pooler);

#endif // COMPUTATIONAL_H
