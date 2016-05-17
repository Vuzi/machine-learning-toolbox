//
// Created by vuzi on 28/04/2016.
//

#ifndef MULTI_LAYER_PERCEPTRON_H
#define MULTI_LAYER_PERCEPTRON_H

#include <cmath>
#include <utils.h>

/**
 * Perceptron type
 */
enum multiLayerPerceptronType {
    MLP_TYPE_CLASSIFICATION = 0, // Linear
    MLP_TYPE_REGRESSION     = 1  // Regression
};

/**
 * Neural network class using multiple layer of perceptron. For now, only the classification is supported
 */
class multiLayerPerceptron {

    public:
        /**
         * Multi layer perceptron constructor
         * @param  l  The layers of the perceptron. The number of layer must be specified in this order :
         *            input layer (number of inputs), hidden layers size and then the number of outputs, for
         *            example { 2, 3, 1 } means a perceptron with two inputs, an hidden layer of 3 neurons, and
         *            one output. At least 3 values should be passed
         * @param  ln The size of the l parameter
         */
        multiLayerPerceptron(int *l, int ln);

        /**
         * Neural network destructor
         */
        ~multiLayerPerceptron();

        /**
         * Propagate the provided values into the neural network, and return the output of the last layer
         * @param  x  The values provided to the input layer. The size of the array must match the size of the
         *            input layer of the multi layer perceptron
         * @return   The output layer value after propagation
         */
        double* propagate(double* x);

        /**
         * Train the multi layer perceptron using the provided values. For now, max * k iteration will be performed
         * @param a   The learning step
         * @param x   The test values, in an k * n size array
         * @param y   The k expected classification results
         * @param k   The number of examples
         * @param max The maximum number of tests to performs
         */
        void train(double a, double *x, double *y, int k, int max);

    private:
        double*** weights; // Weights, by layer and by neuron
        int* layers;       // Dimensions of layers

        int ln;            // Number of layers

        double** computedValues; // Value of each layer after propagation
        double** delta;    // Error of each layer after backward propagation
};



#endif //MULTI_LAYER_PERCEPTRON_H
