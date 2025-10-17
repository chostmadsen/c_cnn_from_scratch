#ifndef ACTIVATORS_H
#define ACTIVATORS_H

#include "types.h"

void relu(const Tensor *tens);

void sigmoid(const Tensor *tens);

void softmax(const Tensor *tens);

#endif // ACTIVATORS_H
