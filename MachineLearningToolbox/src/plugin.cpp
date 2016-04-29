//
// Created by vuzi on 26/04/2016.
//

#include <plugin.h>

// Perceptron
perceptron* perceptronCreate(unsigned n, int type) {
    return new perceptron(n, static_cast<perceptronType>(type));
}

void perceptronDispose(perceptron* p) {
    delete p;
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

// Multi layer perceptron
perceptronMultiLayer* perceptronMLCreate(unsigned n, unsigned *l, unsigned ln, int type) {
    return new perceptronMultiLayer(n, l, ln);
}

void perceptronMLDispose(perceptronMultiLayer* p) {
    delete p;
}

double* perceptronMLClassify(perceptronMultiLayer* p, double* x) {
    return p->classify(x);
}

double perceptronMLTrain(perceptronMultiLayer* p, double a, double* x, double* y, unsigned k, unsigned max) {
    p->train(a, x, y, k, max);
}