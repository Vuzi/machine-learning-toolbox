//
// Created by vuzi on 28/04/2016.
//

#include <multiLayerPerceptron.h>
#include <cstring>

multiLayerPerceptron::multiLayerPerceptron(int* _l, int _ln) {
    // Layers dimensions
    ln = _ln;
    layers = new int[ln];
    memcpy(layers, _l, ln * sizeof(int));
    layers[0] += 1; // Bias neuron

    // Weights
    weights = new double**[ln - 1]; // ln - 1 because the first layer doesn't have weights
    for(int l = 1; l < ln; l++) {
        weights[l - 1] = new double*[layers[l]];

        for(int n = 0; n < layers[l]; n++) {
            weights[l - 1][n] = new double[layers[l - 1]];

            for(int w = 0; w < layers[l - 1]; w++)
                weights[l - 1][n][w] = randValue(1.0, -1.0);
        }
    }

    // Signal
    computedValues = new double*[ln];
    for(int l = 0; l < ln; l++)
        computedValues[l] = new double[layers[l]];

    // Errors
    trainValues = new double*[ln];
    for(int l = 0; l < ln; l++)
        trainValues[l] = new double[layers[l]];
}

multiLayerPerceptron::~multiLayerPerceptron() {
    delete [] layers;

    for(int l = 1; l < ln; l++) {
        for (int n = 0; n < layers[l]; n++)
            delete [] weights[l - 1][n];

        delete [] weights[l - 1];
    }
    delete [] weights;

    for(int l = 0; l < ln; l++) {
        delete [] computedValues[l];
        delete [] trainValues[l];
    }

    delete [] computedValues;
    delete [] trainValues;
}

double* multiLayerPerceptron::propagate(double *x) {
    // Input layer
    computedValues[0][0] = 1.0; // Bias neuron always to 1
    for(int n = 1; n < layers[0]; n++)
        computedValues[0][n] = x[n - 1];

    // Propagate to the hidden and output layers
    for(int l = 1; l < ln; l++) {
        for(int n = 0; n < layers[l]; n++) {
            double sum = 0.0;

            for(int w = 0; w < layers[l - 1]; w++) {
                sum += weights[l - 1][n][w] * computedValues[l - 1][w];
            }

            computedValues[l][n] = sigmoid(sum);
        }
    }

    return computedValues[ln - 1]; // Return output computed value
}

void multiLayerPerceptron::train(double a, double *x, double *y, int k, int max) {
    int count = 0;

    // TODO compute error, and quit when the error is near zero
    while(count++ < max) {
        for (int ex = 0; ex < k; ex++) {
            double *inputs = x + ((layers[0] - 1) * ex); // Input values
            double *outputs = y + (layers[ln - 1] * ex); // Output values
            propagate(inputs);

            // Error on the output layer
            for (int n = 0; n < layers[ln - 1]; n++) {
                double xVal = computedValues[ln - 1][n];

                trainValues[ln - 1][n] = dsigmoid(xVal) * (xVal - outputs[n]);
            }

            // Error on hidden layers
            for (int l = ln - 2; l > 0; l--) {
                for (int n = 0; n < layers[l]; n++) {
                    double sum = 0.0;

                    for (int nPrev = 0; nPrev < layers[l + 1]; nPrev++) {
                        sum += weights[l][nPrev][n] * trainValues[l + 1][nPrev];
                    }

                    trainValues[l][n] = dsigmoid(computedValues[l][n]) * sum;
                }
            }

            // TODO use momentum ?
            // Update weights
            for (int l = 1; l < ln; l++) {
                for (int n = 0; n < layers[l]; n++) {
                    for (int nPrev = 0; nPrev < layers[l - 1]; nPrev++) {
                        weights[l - 1][n][nPrev] -= a * computedValues[l - 1][nPrev] * trainValues[l][n];
                    }
                }
            }

            // TODO use to decide to keep training or quit
            //double error = 0.0;
            //for(int n = 0; n < layers[ln - 1]; n++) {
            //    double diff = outputs[n] - computedValues[ln - 1][n];
            //    error += 0.5 * (diff * diff);
            //}
        }
    }
}
