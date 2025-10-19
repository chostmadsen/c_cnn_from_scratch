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

void vis_tensor(const Tensor *tens, const char *label, size_t h_stretch, size_t v_stretch);

void vis_dense(const Dense *dense, size_t h_stretch, size_t v_stretch);

void vis_conv(const Convolutional *conv, size_t h_stretch, size_t v_stretch);

#endif // HELPERS_H
