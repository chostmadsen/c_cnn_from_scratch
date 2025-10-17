#include "types.h"
#include "functional.h"
#include "helpers.h"
#include "computational.h"
#include "components.h"
#include "activators.h"
#include <stdio.h>
#include <stdbool.h>

// settings for run parameters
const size_t DATAPTS = 1;
const bool DEBUG = true;

/**
 * Main program. Runs forward pass for DATAPTS datapoints.
 *
 * @return: exit code: -1 for model load fail; 1 for run fail; 0 for complete run.
 */
// todo: check if this memory leaks over lifetime (i do not give a shit about program end memory leaks)
int main(void) {
    // read parameters
    const Convolutional *conv1 = read_convolutional("../parameters/conv1.bin");
    const Pooler *pool1 = read_pool("../parameters/pool1.bin");
    const Convolutional *conv2 = read_convolutional("../parameters/conv2.bin");
    const Pooler *pool2 = read_pool("../parameters/pool2.bin");
    const Dense *dense1 = read_dense("../parameters/dense1.bin");
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
        snprintf(pt_filename, sizeof(pt_filename), "../../tests/img_%zu.bin", pt);
        snprintf(label_filename, sizeof(label_filename), "../../tests/img_%zu.bin", pt);
        // read image and label
        const Tensor *img = read_tensor(pt_filename);
        const size_t label = read_label(label_filename);
        if (img == NULL || label == 0) {
            // error reading img or label
            fprintf(stderr, "Error reading image data.\n");
            return 1;
        }

        // forward pass
        // conv1, pool1
        Tensor *a1_t = convolution(img, conv1, relu);
        const Tensor *a1 = pool(a1_t, pool1);
        free_tensor(a1_t);
        // conv2, pool2
        Tensor *a2_t = convolution(a1, conv2, sigmoid);
        Tensor *a2 = pool(a2_t, pool2);
        free_tensor(a2_t);
        // flatten
        flatten(a2);
        // dense1
        const Tensor *yhat = dense(a2, dense1, softmax);
        if (yhat == NULL) {
            // forward pass fail
            fprintf(stderr, "Failed forward pass.\n");
            return 1;
        }

        // determine accuracy
        if (argmax(yhat) == label) correct++;
        if (DEBUG) {
            // print output
            printf("pt %zu\npred: ", pt);
            print_tensor(yhat);
            printf("exp: %zu\n\n\n\n", label);
        }
    }

    // print results and end program
    printf("correct %zu; total %zu; acc %f\n", correct, DATAPTS, (float)correct/(float)DATAPTS);
    return 0;
}
