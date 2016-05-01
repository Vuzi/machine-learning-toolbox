//
// Created by vuzi on 27/04/2016.
//

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <utils.h>
#include <Eigen/Dense>

/**
 * Perceptron type
 */
enum perceptronType {
    TYPE_HEAVISIDE  = 0, // Heaviside, or PLA (linear)
    TYPE_ROSENBLATT = 1, // Rosenblatt (linear)
    TYPE_REGRESSION = 2  // Regression
};

typedef enum perceptronType perceptronType;

/**
 * Perceptron class
 */
class perceptron {
    public:
        /**
         * Perceptron constructor
         * @param  _n    The number of input of the perceptron. Note that every provided values
         *               in training or classification should be of the same size
         * @param  _type The type of algorithm to use internally. Note that this will condition
         *               which value can be returned by the classification and which values should
         *               be used with the training
         */
        perceptron(unsigned _n, perceptronType _type);

        /**
         * Perceptron Destructor
         */
        ~perceptron();

        /**
         * Return the type of the perceptron (see perceptronType)
         * @return   The type of perceptron
         */
        perceptronType getType();

        /**
         * Classify the given values, and return the result
         * @param  x The values to be classified. They should as many as input on
         *           the perceptron, otherwise the behavior is undefined
         * @return   Return the classification value according to the
         *           perceptron internal algorithm defined
         */
        double classify(double* x);

        /**
         * Train the perceptron with the provided values, using the defined internal algorithm
         * @param a   The learning step
         * @param x   The test values, in an k * n size array
         * @param y   The k expected classification results
         * @param k   The number of examples
         * @param max The maximum number of tests to performs
         */
        void train(double a, double* x, double* y, unsigned k, unsigned max);

    private:
        void trainLinear(double a, double* x, double* y, unsigned k, unsigned max);
        void trainRegression(double a, double* x, double* y, unsigned k, unsigned max);

        double updateModelHeaviside(double a, double y, double* x);
        double updateModelRosenblatt(double a, double yk, double y, double* x);

        unsigned n;
        double* w;
        enum perceptronType type;
};

#endif //PERCEPTRON_H
