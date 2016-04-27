//
// Created by vuzi on 26/04/2016.
//

#include <plugin.h>

void perceptronDispose(perceptron* p) {
    delete p;
}

perceptron* perceptronCreate(unsigned n, int type) {
    return new perceptron(n, static_cast<perceptronType>(type));
}

int perceptronGetType(perceptron* p) {
    return p->getType();
}

double perceptronClassify(perceptron* p, double* x) {
    return p->classify(x);
}

double perceptronTrain(perceptron* p, double a, double* x, double* y, unsigned k, unsigned max) {
    p->train(a, x, y, k, max);
}
