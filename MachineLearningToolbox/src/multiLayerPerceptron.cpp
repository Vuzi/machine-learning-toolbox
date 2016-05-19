//
// Created by vuzi on 28/04/2016.
//

#include <multiLayerPerceptron.h>


multiLayerPerceptron::multiLayerPerceptron(int* _l, int _ln, multiLayerPerceptronType _type) {
    type = _type;

    // Layers dimensions
    ln = _ln;
    layers = new int[ln];
    for(int l = 0; l < ln; l++)
        layers[l] = _l[l] + 1;

    // Weights
    weights = new double**[ln];
    for(int l = 1; l < ln; l++) {
        weights[l] = new double*[layers[l]];

        for(int n = 0; n < layers[l]; n++) {
            weights[l][n] = new double[layers[l - 1]];

            for(int w = 0; w < layers[l - 1]; w++) {
                weights[l][n][w] = randValue(1, -1);
            }
        }
    }

    // Signal
    computedValues = new double*[ln];
    for(int l = 0; l < ln; l++)
        computedValues[l] = new double[layers[l]];

    // Errors
    delta = new double*[ln];
    for(int l = 0; l < ln; l++)
        delta[l] = new double[layers[l]];
}

multiLayerPerceptron::~multiLayerPerceptron() {
    for(int l = 1; l < ln; l++) {
        for (int n = 0; n < layers[l]; n++) {
            delete[] weights[l][n];
        }

        delete [] weights[l];
    }
    delete [] weights;
    delete [] layers;

    for(int l = 0; l < ln; l++) {
        delete [] computedValues[l];
        delete [] delta[l];
    }

    delete [] computedValues;
    delete [] delta;
}

enum multiLayerPerceptronType multiLayerPerceptron::getType() {
    return type;
}

double* multiLayerPerceptron::propagate(double *x) {
    // Input layer
    computedValues[0][0] = 1.0; // Bias neuron
    for(int n = 1; n < layers[0]; n++)
        computedValues[0][n] = x[n - 1];

    // Propagate to the hidden and output layers
    for(int l = 1; l < ln; l++) {
        computedValues[l][0] = 1.0; // Bias neuron

        for(int n = 1; n < layers[l]; n++) {
            double sum = 0.0;

            for(int w = 0; w < layers[l - 1]; w++) {
                sum += weights[l][n][w] * computedValues[l - 1][w];
            }

            if(l == ln - 1) {
                // Output layer
                if (type == MLP_TYPE_CLASSIFICATION)
                    computedValues[l][n] = sigmoid(sum);
                else if (type == MLP_TYPE_REGRESSION)
                    computedValues[l][n] = sum;
            }else
                computedValues[l][n] = sigmoid(sum);
        }
    }

    return computedValues[ln - 1] + 1; // Return output computed value
}

void multiLayerPerceptron::train(double a, double *x, double *y, int k, int max) {
    int count = 0;

    while(count++ < max) {
        for (int ex = 0; ex < k; ex++) {
            double *inputs = x + ((layers[0] - 1) * ex); // Input values
            double *outputs = y + ((layers[ln - 1] - 1) * ex); // Output values
            propagate(inputs);

            // Error on the output layer
            for (int n = 1; n < layers[ln - 1]; n++) {
                double xVal = computedValues[ln - 1][n];

                if (type == MLP_TYPE_CLASSIFICATION)
                    delta[ln - 1][n] = dsigmoid(xVal) * (xVal - outputs[n - 1]);
                else if (type == MLP_TYPE_REGRESSION)
                    delta[ln - 1][n] = xVal - outputs[n - 1];
            }

            // Error on hidden layers
            for (int l = ln - 2; l >= 0; l--) {
                for (int n = 0; n < layers[l]; n++) {
                    double sum = 0.0;

                    for (int nPrev = 1; nPrev < layers[l + 1]; nPrev++) {
                        sum += weights[l + 1][nPrev][n] * delta[l + 1][nPrev];
                    }

                    delta[l][n] = dsigmoid(computedValues[l][n]) * sum;
                }
            }

            // Update weights
            for (int l = 1; l < ln; l++) {
                for (int n = 0; n < layers[l]; n++) {
                    for (int nPrev = 0; nPrev < layers[l - 1]; nPrev++) {
                        double change = computedValues[l - 1][nPrev] * delta[l][n];
                        weights[l][n][nPrev] = weights[l][n][nPrev] - (a * change);
                    }
                }
            }

        }
    }
}


