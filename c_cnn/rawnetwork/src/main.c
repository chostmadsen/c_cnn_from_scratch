#include "types.h"
#include "functional.h"
#include "helpers.h"
#include "computational.h"
#include "components.h"
#include "activators.h"
#include <stdio.h>

/**
 * Main program. Runs forward pass for DATAPTS datapoints.
 *
 * @param argc: num args.
 * @param argv: two arguments. mode to execute: n=normal, d=debug, i=images, f=full images; and number of points.
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

        // full loop label
        if (mode == 'f') {
            char loop_label[32];
            snprintf(loop_label, sizeof(loop_label), "\niteration %zu\n", pt + 1);
            printf("%s", loop_label);
            char img_label[8];
            snprintf(img_label, sizeof(img_label), "[%zu]", label);
            vis_tensor(img, img_label, 2, 1);
        }

        // forward pass
        // conv1, pool1
        Tensor *a1_t = convolution(img, conv1, relu);
        Tensor *a1 = pool(a1_t, pool1);
        // full vis
        if (mode == 'f') vis_tensor(a1_t, "a1_t", 2, 1);
        if (mode == 'f') vis_tensor(a1, "a1", 2, 1);
        free_tensor(a1_t);
        // conv2, pool2
        Tensor *a2_t = convolution(a1, conv2, sigmoid);
        free_tensor(a1);
        // full vis
        Tensor *a2 = pool(a2_t, pool2);
        if (mode == 'f') vis_tensor(a2_t, "a2_t", 2, 1);
        if (mode == 'f') vis_tensor(a2, "a2", 2, 1);
        free_tensor(a2_t);
        // flatten
        flatten(a2);
        if (mode == 'f') vis_tensor(a2, "a2_f", 1, 1);
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
        if (mode == 'n') {
            // print current progress
            const float acc = (float)correct / (float)(pt + 1);
            printf("\r%zu/%zu points; %zu/%zu correct; %.4g%% accuracy;", pt + 1, number, correct, pt + 1, 100 * acc);
        } else if (mode == 'd') {
            // print output
            printf("expected %zu; raw output [", label);
            for (size_t idx = 0; idx < yhat->n - 1; idx++) {
                printf("%f  ", yhat->arr[idx]);
            }
            printf("%f];\n", yhat->arr[yhat->n - 1]);
        } else if (mode == 'i') {
            // print image
            char img_label[64];
            snprintf(img_label, sizeof(img_label), "[yhat %zu | y %zu]", argmax(yhat), label);
            printf("\n");
            vis_tensor(img, img_label, 2, 1);
        }
        if (mode == 'f') vis_tensor(yhat, "0123456789", 1, 1);

        // free
        free_tensor(img);
        free_tensor(yhat);
    }

    if (mode == 'f') {
        printf("\nparameter visualization\n");

        // vis conv1
        printf("\nconv1\n");
        vis_conv(conv1, 2, 1);

        // vis conv2
        printf("\nconv2\n");
        vis_conv(conv2, 2, 1);

        // vis dense1
        printf("\ndense1\n");
        vis_dense(dense1, 1, 1);
    }

    // print final results
    const float acc = (float)correct / (float)number;
    printf("\nend: %zu correct; %zu total; %.4g%% accuracy;\n", correct, number, 100 * acc);
    // free memory and end program
    free_convolutional(conv1); free(pool1);
    free_convolutional(conv2); free(pool2);
    free_dense(dense1);
    return 0;
}
