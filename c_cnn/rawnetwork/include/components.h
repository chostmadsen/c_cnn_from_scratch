#ifndef COMPONENTS_H
#define COMPONENTS_H

Tensor *dense(const Tensor *input, const Dense *dense, void (*fn)(const Tensor *));

Tensor *convolution(const Tensor *input, const Convolutional *kernels, void (*fn)(const Tensor*));

#endif // COMPONENTS_H
