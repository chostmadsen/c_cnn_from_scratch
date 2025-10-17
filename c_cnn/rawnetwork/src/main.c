#include "types.h"
#include "functional.h"
#include "helpers.h"
#include "computational.h"
#include "components.h"
#include "activators.h"
#include <stdio.h>
#include <stdbool.h>

void print_tensor_s(const Tensor *tensor) {
    print_tensor(tensor);
    printf("\n");
}

// settings for run parameters
const size_t DATAPTS = 1;
const bool DEBUG = true;

/**
 * Main program. Runs forward pass for DATAPTS datapoints.
 *
 * @return: exit code: -1 for model load fail; 1 for run fail; 0 for complete run.
 */
int main(void) {
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
    for (size_t pt = 0; pt < DATAPTS; pt++) {
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
        Tensor *a1_t = convolution(img, conv1, relu);
        print_tensor_s(a1_t); // todo: check
        free_tensor(img);
        Tensor *a1 = pool(a1_t, pool1);
        print_tensor_s(a1); // todo: check
        free_tensor(a1_t);
        // conv2, pool2
        Tensor *a2_t = convolution(a1, conv2, sigmoid);
        print_tensor_s(a2_t); // todo: check
        free_tensor(a1);
        Tensor *a2 = pool(a2_t, pool2);
        print_tensor_s(a2); // todo: check
        free_tensor(a2_t);
        // flatten
        flatten(a2);
        print_tensor_s(a2); // todo: check
        // dense1
        Tensor *yhat = dense(a2, dense1, softmax);
        print_tensor_s(yhat); // todo: check
        free_tensor(a2);
        if (yhat == NULL) {
            // forward pass fail
            fprintf(stderr, "Failed forward pass.\n");
            return 1;
        }

        // determine accuracy
        if (argmax(yhat) == label) correct++;
        // print current progress
        const float acc = (float)correct / (float)DATAPTS;
        printf("\r%zu/%zu pts; %zu/%zu correct; %.4g acc;", pt, DATAPTS, correct, pt, acc);
        if (DEBUG) {
            // print output
            printf("\n\npt %zu\npred: ", pt);
            print_tensor(yhat);
            printf("exp: %zu\n\n\n", label);
        }
        free_tensor(yhat);
    }

    // print final results
    printf("\ncorrect %zu; total %zu; acc %.8g;\n", correct, DATAPTS, (float)correct/(float)DATAPTS);
    // free memory and end program
    free_convolutional(conv1); free(pool1);
    free_convolutional(conv2); free(pool2);
    free_dense(dense1);
    return 0;
}
