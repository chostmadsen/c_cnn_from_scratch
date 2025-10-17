#ifndef HELPERS_H
#define HELPERS_H

void print_tensor(const Tensor *tens);

void print_kernel(const Kernel *kernel);

void print_pooler(const Pooler *pooler);

void print_dense(const Dense *dense);

void print_convolutional(const Convolutional *conv);

Tensor *read_tensor(const char *filename);

Kernel *read_kernel(const char *filename);

Pooler *read_pool(const char *filename);

Dense *read_dense(const char *filename);

Convolutional *read_convolutional(const char *filename);

size_t read_label(const char *filename);

#endif // HELPERS_H
