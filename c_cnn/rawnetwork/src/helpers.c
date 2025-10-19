#include <stdio.h>
#include <stdbool.h>
#include "types.h"
#include "functional.h"
#include "helpers.h"

#include <string.h>

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
                if (elm + 1 != n) printf(" \t");
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

elm_t max_val_(const elm_t *arr, const size_t size) {
    elm_t max_val = arr[0];
    for (size_t elm = 1; elm < size; elm++) {
        if (arr[elm] > max_val) max_val = arr[elm];
    }
    return max_val;
}

elm_t min_val_(const elm_t *arr, const size_t size) {
    elm_t min_val = arr[0];
    for (size_t elm = 1; elm < size; elm++) {
        if (arr[elm] < min_val) min_val = arr[elm];
    }
    return min_val;
}

char *vis_mat_(const elm_t *mat, const Tensor *tens) {
    // setup out str
    char *out_str = malloc(tens->m * tens->n * sizeof(char) + 1);
    if (out_str == NULL) {
        fprintf(stderr, "Failed malloc: str of size %zu.\n", tens->m * tens->n + 1);
        return NULL;
    }
    out_str[tens->m * tens->n] = '\0';

    // setup reference
    const elm_t max_val = max_val_(mat, tens->m * tens->n);
    elm_t min_val = min_val_(mat, tens->m * tens->n);
    if (min_val < 0.0) min_val = -min_val;
    const elm_t max_abs_val = max_val > min_val ? max_val : min_val;

    // create out str
    for (size_t elm = 0; elm < tens->m * tens->n; elm++) {
        const char ref[] = " .,:-+=%$#";
        // abs elm
        const elm_t ref_elm = 0 < mat[elm] ? mat[elm] : -mat[elm];
        // scale elm
        const elm_t elm_adj = (elm_t)(ref_elm / max_abs_val * 10 - 1e-06);
        out_str[elm] = ref[(int)elm_adj];
    }
    return out_str;
}

bool *sgn_mat_(const elm_t *mat, const Tensor *tens) {
    // setup out arr
    bool *sgn_arr = malloc(tens->m * tens->n * sizeof(bool));
    if (sgn_arr == NULL) {
        fprintf(stderr, "Failed malloc: arr of size %zu.\n", tens->m * tens->n);
        return NULL;
    }

    // create sgn arr
    for (size_t elm = 0; elm < tens->m * tens->n; elm++) {
        sgn_arr[elm] = 0 < mat[elm];
    }
    return sgn_arr;
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
 * @return: label layer. (size_t) - 1 for any invalid file or internal reading fail.
 */
size_t read_label(const char *filename) {
    // get file ptr
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed opening file: %s.\n", filename);
        fclose(fp); return (size_t) - 1;
    }

    // read label
    size_t value;
    if (fread(&value, sizeof(size_t), 1, fp) != 1) {
        fprintf(stderr, "Failed reading label.\n");
        fclose(fp); return (size_t) - 1;
    }
    fclose(fp); return value;
}

/**
 * Visualizes a tensor with an image.
 *
 * @param tens: tensor.
 * @param label: image label.
 * @param h_stretch: image horizontal stretch.
 * @param v_stretch: image vertical stretch.
 */
void vis_tensor(const Tensor *tens, const char *label, const size_t h_stretch, const size_t v_stretch) {
    // print top
    printf("+");
    for (size_t j = 0; j < h_stretch * tens->n; j++) printf("-");
    printf("+\n");

    // print tensors
    for (size_t mat = 0; mat < tens->o; mat++) {
        // setup arrs
        char *str_arr = vis_mat_(&tens->arr[mat * tens->m * tens->n], tens);
        bool *sgn_arr = sgn_mat_(&tens->arr[mat * tens->m * tens->n], tens);

        for (size_t row = 0; row < tens->m; row++) { for (size_t _v = 0; _v < v_stretch; _v++) {
            // border
            printf("|");
            for (size_t col = 0; col < tens->n; col++) { for (size_t _h = 0; _h < h_stretch; _h++) {
                const size_t elm = row * tens->n + col;
                // color
                sgn_arr[elm] ? printf("\x1b[0m") : printf("\x1b[37m");
                // element
                printf("%c\x1b[0m", str_arr[elm]);
            }}
            // border
            printf("|\n");
        }}

        // seperator
        if (mat + 1 != tens->o) {
            printf("+");
            for (size_t j = 0; j < h_stretch * tens->n; j++) printf("-");
            printf("+\n");
        }

        // free arrs
        free(str_arr); free(sgn_arr);
    }

    // bottom
    const size_t label_size = strlen(label);
    printf("+");
    const size_t center = (h_stretch * tens->n - label_size) / 2;
    for (size_t b = 0; b < center; b++) printf("-");
    // label
    printf("%s", label);
    for (size_t b = 0; b < h_stretch * tens->n - center - label_size; b++) printf("-");
    printf("+\n");
}

/**
 * Visualizes a dense layer with an image.
 *
 * @param dense: dense layer.
 * @param h_stretch: image horizontal stretch.
 * @param v_stretch: image vertical stretch.
 */
void vis_dense(const Dense *dense, const size_t h_stretch, const size_t v_stretch) {
    // vis weights and biases
    vis_tensor(dense->weights, "w", h_stretch, v_stretch);
    vis_tensor(dense->biases, "b", h_stretch, v_stretch);
}

/**
 * Visualizes a convolutional layer with an image.
 *
 * @param conv: convolutional layer.
 * @param h_stretch: image horizontal stretch.
 * @param v_stretch: image vertical stretch.
 */
void vis_conv(const Convolutional *conv, const size_t h_stretch, const size_t v_stretch) {
    // vis kernels
    const size_t m_k = conv->kernels[0]->m, n_k = conv->kernels[0]->n, o_k = conv->kernels[0]->o;
    for (size_t elm = 0; elm < conv->num; elm++) {
        // setup tensors
        elm_t *kern_arr = conv->kernels[elm]->arr;
        Tensor kern_t = {.m=m_k, .n=n_k, .o=o_k, .arr=kern_arr};
        // setup label
        char img_label[8];
        snprintf(img_label, sizeof(img_label), "k%zu", elm + 1);
        vis_tensor(&kern_t, img_label, h_stretch, v_stretch);
    }

    // vis biases
    elm_t biases[conv->num];
    for (size_t b = 0; b < conv->num; b++) biases[b] = conv->kernels[b]->bias;
    const Tensor kern_b = {.m=1, .n=conv->num, .o=1, .arr=biases};
    vis_tensor(&kern_b, "b", 1, 1);
}
