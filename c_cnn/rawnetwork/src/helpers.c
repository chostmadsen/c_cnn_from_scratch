#include <stdio.h>
#include "types.h"
#include "functional.h"
#include "helpers.h"

/*--------------------------------------------------------------------------------------------------------------------*/

static void print_arr_(const elm_t *arr, const size_t m, const size_t n, const size_t o, const size_t shift) {
    // delimiter
    printf("{\n");
    for (size_t mat = 0; mat < o; mat++) {
        // matrix
        for (size_t _ = 0; _ < shift; _++) printf(" ");
        printf(" [");
        for (size_t row = 0; row < m; row++) {
            // row
            if (row != 0) printf("  ");
            printf("[");
            for (size_t elm = 0; elm < n; elm++) {
                // spaced elm
                printf("%f", arr[mat * m * n + row * n + elm]);
                if (elm + 1 != n) printf("  \t");
            }
            printf("]");
            if (row + 1 != m) {
                printf("\n");
                for (size_t _ = 0; _ < shift; _++) printf(" ");
            }
        }
        printf("]\n");
        if (mat + 1 != o) printf("\n");
    }
    // delimiter
    for (size_t _ = 0; _ < shift; _++) printf(" ");
    printf("}");
}

static elm_t *read_arr_(FILE *f, const size_t size) {
    // malloc
    elm_t *arr = malloc(size * sizeof(elm_t));
    if (arr == NULL) {
        fprintf(stderr, "Failed malloc: array sized %zu.\n", size);
        return NULL;
    }

    // read array
    if (fread(arr, sizeof(elm_t), size, f) != size) {
        fprintf(stderr, "Unexpected EOF: nitems (%zu) not reached.\n", size);
        free(arr);
        return NULL;
    }
    return arr;
}

static Tensor *read_tensor_(FILE *fp) {
    // size definition
    const size_t metadata_size = 3;

    // read metadata
    size_t metadata[metadata_size];
    if (fread(metadata, sizeof(size_t), metadata_size, fp) != metadata_size) {
        // invalid metadata
        fprintf(stderr, "Invalid metadata.\n");
        return NULL;
    }

    // metadata setup
    const size_t m = metadata[0], n = metadata[1], o = metadata[2];

    // malloc
    Tensor *tensor = malloc(sizeof(Tensor));
    elm_t *arr = read_arr_(fp, m * n * o);
    if (arr == NULL || tensor == NULL) {
        fprintf(stderr, "Failed malloc: Tensor sized %zu x %zu x %zu.\n", m, n, o);
        free(arr); free(tensor);
        return NULL;
    }

    // struct setup
    tensor->m = m; tensor->n = n; tensor->o = o;
    tensor->arr = arr;
    return tensor;
}

Kernel *read_kernel_(FILE *fp) {
    // size definition
    const size_t metadata_size = 5;

    // read metadata
    size_t metadata[metadata_size];
    if (fread(metadata, sizeof(size_t), metadata_size, fp) != metadata_size) {
        // invalid metadata
        fprintf(stderr, "Invalid metadata.\n");
        return NULL;
    }

    // metadata setup
    const size_t m = metadata[0], n = metadata[1], o = metadata[2];
    const size_t m_stride = metadata[3], n_stride = metadata[4];

    // read bias
    elm_t bias;
    if (fread(&bias, sizeof(elm_t), 1, fp) != 1) {
        fprintf(stderr, "Failed kernel bias read.\n");
        return NULL;
    }

    // malloc
    Kernel *kernel = malloc(sizeof(Kernel));
    elm_t *arr = read_arr_(fp, m * n * o);
    if (arr == NULL || kernel == NULL) {
        fprintf(stderr, "Failed malloc: Kernel sized %zu x %zu x %zu.\n", m, n, o);
        free(arr); free(kernel);
        return NULL;
    }

    // struct setup
    kernel->m = m; kernel->n = n; kernel->o = o;
    kernel->m_stride = m_stride; kernel->n_stride = n_stride; kernel->bias = bias;
    kernel->arr = arr;
    return kernel;
}

/*--------------------------------------------------------------------------------------------------------------------*/

/**
 * Prints a tensor in the terminal with spacing.
 *
 * @param tens: tensor to be printed.
 */
void print_tensor(const Tensor *tens) {
    if (tens == NULL) return;
    // tensor
    print_arr_(tens->arr, tens->m, tens->n, tens->o, 0);
    // metadata
    printf(" %zu x %zu x %zu;\n", tens->m, tens->n, tens->o);
}

/**
 * Prints a kernel in the terminal with spacing.
 *
 * @param kernel: kernel to be printed.
 */
void print_kernel(const Kernel *kernel) {
    if (kernel == NULL) return;
    // kernel
    print_arr_(kernel->arr, kernel->m, kernel->n, kernel->o, 0);
    // metadata
    printf(" %zu x %zu x %zu;", kernel->m, kernel->n, kernel->o);
    printf(" b %f;", kernel->bias);
    printf(" s %zu x %zu;\n", kernel->m_stride, kernel->n_stride);
}

/**
 * Prints a pooling kernel in the terminal with spacing.
 *
 * @param pooler: pooler to be printed.
 */
void print_pooler(const Pooler *pooler) {
    if (pooler == NULL) return;
    // metadata
    printf("{\n %zu x %zu;", pooler->m, pooler->n);
    printf(" s %zu x %zu;\n}\n", pooler->m_stride, pooler->n_stride);
}

/**
 * Prints a dense layer in the terminal with spacing.
 *
 * @param dense: dense layer to be printed.
 */
void print_dense(const Dense *dense) {
    if (dense == NULL) return;
    printf("{\n");
    // weights
    printf(" w: ");
    const Tensor *weights = dense->weights;
    const Tensor *biases = dense->biases;
    print_arr_(weights->arr, weights->m, weights->n, weights->o, 1);
    // metadata
    printf(" %zu x %zu x %zu;\n\n", weights->m, weights->n, weights->o);
    // biases
    printf(" b: ");
    print_arr_(biases->arr, biases->m, biases->n, biases->o, 1);
    // metadata
    printf(" %zu x %zu x %zu;\n}\n", biases->m, biases->n, biases->o);
}

/**
 * Prints a convolutional layer in the terminal with spacing.
 *
 * @param conv: convolutional layer to be printed.
 */
void print_convolutional(const Convolutional *conv) {
    if (conv == NULL) return;
    for (size_t k = 0; k < conv->num; k++) {
        // kernel
        printf(" k_%zu: ", k);
        const Kernel *kernel = conv->kernels[k];
        print_arr_(kernel->arr, kernel->m, kernel->n, kernel->o, 1);
        // metadata
        printf(" %zu x %zu x %zu;", kernel->m, kernel->n, kernel->o);
        printf(" b %f;", kernel->bias);
        printf(" s %zu x %zu;\n", kernel->m_stride, kernel->n_stride);
        if (k + 1 == conv->num) printf("}");
        printf("\n");
    }
}

/**
 * Reads a tensor from a bin file.
 * Caller is responsible for freeing returned tensor.
 * Sets up tensor from metadata stored within the bin file.
 *
 * @param filename: filename.
 *
 * @return: read tensor. NULL for an invalid file or internal reading fail.
 */
Tensor *read_tensor(const char *filename) {
    // get file ptr
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed opening file: %s.\n", filename);
        return NULL;
    }

    // read tensor
    Tensor *tensor = read_tensor_(fp);
    fclose(fp);
    if (tensor == NULL) {
        fprintf(stderr, "Failed reading Tensor.\n");
        return NULL;
    }
    return tensor;
}

/**
 * Reads a kernel from a bin file.
 * Caller is responsible for freeing returned kernel.
 * Sets up kernel from metadata stored within the bin file.
 *
 * @param filename: filename.
 *
 * @return: read kernel. NULL for any invalid file or internal reading fail.
 */
Kernel *read_kernel(const char *filename) {
    // get file ptr
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed opening file: %s.\n", filename);
        return NULL;
    }

    // read kernel
    Kernel *kernel = read_kernel_(fp);
    fclose(fp);
    if (kernel == NULL) {
        fprintf(stderr, "Failed reading Kernel.\n");
        return NULL;
    }
    return kernel;
}

/**
 * Reads a pooling kernel from a bin file.
 * Caller is responsible for freeing returned pooling kernel.
 * Sets up pooling kernel from metadata stored within the bin file.
 *
 * @param filename: filename.
 *
 * @return: read pooling kernel. NULL for any invalid file or internal reading fail.
 */
Pooler *read_pool(const char *filename) {
    // size definition
    const size_t metadata_size = 4;

    // get file ptr
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed opening file: %s.\n", filename);
        return NULL;
    }

    // read metadata
    size_t metadata[metadata_size];
    if (fread(metadata, sizeof(size_t), metadata_size, fp) != metadata_size) {
        // invalid metadata
        fprintf(stderr, "Invalid metadata: %s.\n", filename);
        fclose(fp);
        return NULL;
    }

    // metadata setup
    const size_t m = metadata[0], n = metadata[1];
    const size_t m_stride = metadata[2], n_stride = metadata[3];

    // malloc
    Pooler *pooler = malloc(sizeof(Pooler));
    fclose(fp);
    if (pooler == NULL) {
        fprintf(stderr, "Failed malloc: Pooling kernel sized %zu x %zu.\n", m, n);
        return NULL;
    }

    // struct setup
    pooler->m = m; pooler->n = n;
    pooler->m_stride = m_stride; pooler->n_stride = n_stride;
    return pooler;
}

/**
 * Reads a dense layer from a bin file.
 * Caller is responsible for freeing returned dense layer.
 * Sets up a dense layer from metadata stored within the bin file.
 *
 * @param filename: filename.
 *
 * @return: read dense layer. NULL for any invalid file or internal reading fail.
 */
Dense *read_dense(const char *filename) {
    // get file ptr
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed opening file: %s.\n", filename);
        return NULL;
    }

    // read weights
    Tensor *weights = read_tensor_(fp);
    if (weights == NULL) {
        fprintf(stderr, "Failed reading weights.\n");
        fclose(fp);
        return NULL;
    }
    Tensor *biases = read_tensor_(fp);
    if (biases == NULL) {
        fprintf(stderr, "Failed reading biases.\n");
        free_tensor(weights); fclose(fp);
        return NULL;
    }
    fclose(fp);

    // struct setup
    Dense *dense = malloc(sizeof(Dense));
    if (dense == NULL) {
        fprintf(stderr, "Failed malloc: dense layer.\n");
        free_tensor(weights); free_tensor(biases); fclose(fp);
        return NULL;
    }
    dense->weights = weights;
    dense->biases = biases;
    return dense;
}

/**
 * Reads a convolutional layer from a bin file.
 * Caller is responsible for freeing returned convolutional layer.
 * Sets up a convolutional layer from metadata stored within the bin file.
 *
 * @param filename: filename.
 *
 * @return: read convolutional layer. NULL for any invalid file or internal reading fail.
 */
Convolutional *read_convolutional(const char *filename) {
    // size definition
    size_t num;

    // get file ptr
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed opening file: %s.\n", filename);
        return NULL;
    }

    // read metadata
    if (fread(&num, sizeof(size_t), 1, fp) != 1) {
        // invalid metadata
        fprintf(stderr, "Invalid metadata.\n");
        fclose(fp);
        return NULL;
    }

    // malloc
    Convolutional *convolutional = malloc(sizeof(Convolutional));
    Kernel **kernels = malloc(num * sizeof(Kernel*));
    if (convolutional == NULL) {
        fprintf(stderr, "Failed malloc: convolutional layer.\n");
        fclose(fp);
        return NULL;
    }

    // read kernels
    for (size_t kern = 0; kern < num; kern++) {
        Kernel *kernel = read_kernel_(fp);
        if (kernel == NULL) {
            // kernel read error
            fprintf(stderr, "Failed reading kernel.\n");
            // dump memory
            for (size_t k = 0; k < kern; k++) {
                free_kernel(kernels[k]);
            }
            free(kernels); free(convolutional);
            fclose(fp);
            return NULL;
        }
        kernels[kern] = kernel;
    }

    // struct setup
    convolutional->num = num;
    convolutional->kernels = kernels;
    fclose(fp);
    return convolutional;
}

/**
 * Reads a label from a bin file.
 *
 * @param filename: filename.
 *
 * @return: label layer. NULL for any invalid file or internal reading fail.
 */
size_t read_label(const char *filename) {
    // get file ptr
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed opening file: %s.\n", filename);
        fclose(fp); return 0;
    }

    // read label
    size_t value;
    if (fread(&value, sizeof(size_t), 1, fp) != 1) {
        fprintf(stderr, "Failed reading label.\n");
        fclose(fp); return 0;
    }
    fclose(fp); return value;
}
