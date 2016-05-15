//
// Created by vuzi on 26/04/2016.
//

#include <plugin.h>

void init() {
    srand((unsigned) time(NULL));
}

// Perceptron
perceptron* perceptronCreate(int n, int type) {
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

void perceptronTrain(perceptron* p, double a, double* x, double* y, int k, int max) {
    p->train(a, x, y, k, max);
}


// Multi layer perceptron
multiLayerPerceptron* multiLayerPerceptronCreate(int *l, int ln, int type) {
    return new multiLayerPerceptron(l, ln);
}

void multiLayerPerceptronDispose(multiLayerPerceptron* p) {
    delete p;
}

int multiLayerPerceptronGetType(multiLayerPerceptron* p) {
    return MLP_TYPE_CLASSIFICATION;
}

double* multiLayerPerceptronPropagate(multiLayerPerceptron* p, double* x) {
    return p->propagate(x);
}

void multiLayerPerceptronTrain(multiLayerPerceptron* p, double a, double* x, double* y, int k, int max) {
    p->train(a, 0.1, x, y, k, max);
}