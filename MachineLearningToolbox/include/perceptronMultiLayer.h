//
// Created by vuzi on 28/04/2016.
//

#ifndef PERCEPTRONMULTILAYER_H
#define PERCEPTRONMULTILAYER_H

#include <cmath>
#include <utils.h>

class perceptronMultiLayer {

    public:
        perceptronMultiLayer(unsigned n, unsigned *l, unsigned ln);
        ~perceptronMultiLayer();

        double* classify(double* x);

        void train(double a, double *x, double *y, unsigned k, unsigned max);

        double*** w; // Weights, by layer and by neuron
        unsigned* l; // Dimensions of layers

        double** computedValues;
        double** trainValues;

        unsigned n;  // Number of inputs
        unsigned ln; // Number of layers
};



#endif //PERCEPTRONMULTILAYER_H
