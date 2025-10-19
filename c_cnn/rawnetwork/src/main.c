#include "types.h"
#include "functional.h"
#include "helpers.h"
#include "computational.h"
#include "components.h"
#include "activators.h"
#include <stdio.h>

/*--------------------------------------------------------------------------------------------------------------------*/

void print_img_(const Tensor *tensor, const char *label, const size_t label_size) {
    const size_t stretch = 2;

    if (tensor->o != 1 || tensor->n - 2 < label_size) {
        // invalid image
        fprintf(stderr, "Image error: invalid image.\n");
        return;
    }

    // set up image vis
    char img_vis[tensor->m * stretch * tensor->n + 1];
    img_vis[tensor->m * stretch * tensor->n] = '\0';
    // interpret image
    for (size_t i = 0; i < tensor->m; i++) {
        for (size_t j = 0; j < tensor->n; j++) {
            // reference brightness
            const char ref[] = " .,:-+=%$#";
            // char reference
            const char img_char = ref[(int)(tensor->arr[i * tensor->n + j] * 10 - 1e-4)];
            // set img
            for (size_t k = 0; k < stretch; k++) {
                img_vis[i * stretch * tensor->n + stretch * j + k] = img_char;
            }
        }
    }
    // set label
    for (size_t idx = 0; idx < label_size; idx++) {
        img_vis[(tensor->m - 1) * stretch * tensor->n + idx + (stretch * tensor->n - label_size)] = label[idx];
    }

    // print image
    // box
    printf("+");
    for (size_t j = 0; j < stretch * tensor->n; j++) printf("-");
    printf("+\n");
    // image
    for (size_t i = 0; i < tensor->m; i++) {
        printf("|");
        for (size_t j = 0; j < stretch * tensor->n; j++) {
            printf("%c", img_vis[i * stretch * tensor->n + j]);
        }
        printf("|\n");
    }
    // box
    printf("+");
    for (size_t j = 0; j < stretch * tensor->n; j++) printf("-");
    printf("+\n");
}

/*--------------------------------------------------------------------------------------------------------------------*/

/**
 * Main program. Runs forward pass for DATAPTS datapoints.
 *
 * @param argc: num args.
 * @param argv: two arguments. mode to execute: n = normal, d = debug, i = images; and number of points.
 *
 * @return: exit code: -1 for model load fail; 1 for run fail; 2 for start fail; 0 for complete run.
 */
int main(const int argc, const char *argv[]) {
    // arguments
    if (argc != 3) {
        printf("Usage: %s <mode> <number>\n", argv[0]);
        return 2;
    }
    // get arguments (we ignore strtol errors here)
    const char mode = argv[1][0];
    char *ptr;
    const long val = strtol(argv[2], &ptr, 10);
    const size_t number = (size_t)val;

    // read parameters
    Convolutional *conv1 = read_convolutional("parameters/conv1.bin");
    Pooler *pool1 = read_pool("parameters/pool1.bin");
    Convolutional *conv2 = read_convolutional("parameters/conv2.bin");
    Pooler *pool2 = read_pool("parameters/pool2.bin");
    Dense *dense1 = read_dense("parameters/dense1.bin");
    // check errors
    if (conv1 == NULL || pool1 == NULL || conv2 == NULL || pool2 == NULL || dense1 == NULL) {
        fprintf(stderr, "Error reading network parameters.\n");
        return -1;
    }

    // testing loop
    size_t correct = 0;
    for (size_t pt = 0; pt < number; pt++) {
        // setup image and label location
        char pt_filename[64];
        char label_filename[64];
        snprintf(pt_filename, sizeof(pt_filename), "../data/images/img_%zu.bin", pt);
        snprintf(label_filename, sizeof(label_filename), "../data/labels/img_%zu.bin", pt);
        // read image and label
        Tensor *img = read_tensor(pt_filename);
        const size_t label = read_label(label_filename);
        if (img == NULL || label == (size_t) - 1) {
            // error reading img or label
            fprintf(stderr, "Error reading image data.\n");
            return 1;
        }

        // forward pass
        // conv1, pool1
        Tensor *a1_t = convolution(img, conv1, noop);
        Tensor *a1 = pool(a1_t, pool1);
        free_tensor(a1_t);
        // conv2, pool2
        Tensor *a2_t = convolution(a1, conv2, sigmoid);
        free_tensor(a1);
        Tensor *a2 = pool(a2_t, pool2);
        free_tensor(a2_t);
        // flatten
        flatten(a2);
        // dense1
        Tensor *yhat = dense(a2, dense1, softmax);
        free_tensor(a2);
        if (yhat == NULL) {
            // forward pass fail
            fprintf(stderr, "Failed forward pass.\n");
            return 1;
        }
        // determine accuracy
        if (argmax(yhat) == label) correct++;

        // terminal outputs
        // debug
        if (mode == 'n') {
            // print current progress
            const float acc = (float)correct / (float)pt;
            printf("\r%zu/%zu points; %zu/%zu correct; %.4g accuracy;", pt, number, correct, pt, acc);
        } else if (mode == 'd') {
            // print output
            printf("expected %zu; raw output [", label);
            for (size_t idx = 0; idx < yhat->n - 1; idx++) {
                printf("%f  ", yhat->arr[idx]);
            }
            printf("%f];\n", yhat->arr[yhat->n - 1]);
        } else if (mode == 'i') {
            // print image
            size_t label_size = snprintf(NULL, 0, "[yhat %zu | y %zu]", argmax(yhat), label);
            char img_label[label_size];
            snprintf(img_label, label_size + 1, "[yhat %zu | y %zu]", argmax(yhat), label);
            printf("\n");
            print_img_(img, img_label, label_size);
        }

        // free
        free_tensor(img);
        free_tensor(yhat);
    }

    // print final results
    printf("\nend; %zu correct; %zu total; %.8g accuracy;\n", correct, number, (float)correct/(float)number);
    // free memory and end program
    free_convolutional(conv1); free(pool1);
    free_convolutional(conv2); free(pool2);
    free_dense(dense1);
    return 0;
}
