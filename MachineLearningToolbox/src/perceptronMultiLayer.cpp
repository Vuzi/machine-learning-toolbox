//
// Created by vuzi on 28/04/2016.
//

#include <perceptronMultiLayer.h>

perceptronMultiLayer::perceptronMultiLayer(unsigned _n, unsigned* _l, unsigned _ln) {
    n = _n;
    l = _l;
    ln = _ln;

    // Prepare weight values
    w = new double**[ln];

    // For each layer
    for(unsigned i = 0; i < ln; i++) {

        unsigned wSize = (i == 0 ? n : l[i - 1]) + 1;
        w[i] = new double*[l[i]];

        // For each neuron
        for(unsigned j = 0; j < l[i]; j++) {
            w[i][j] = new double[wSize];

            // For each weight
            for(unsigned k = 0; k < wSize; k++)
                w[i][j][k] = randValue(1, -1); // Init between -1 and 1
        }
    }

    // Also prepare values used for training and classification
    trainValues = new double*[ln];
    computedValues = new double*[ln];
    for(unsigned i = 0; i < ln; i++) {
        trainValues[i] = new double[l[i]];
        computedValues[i] = new double[l[i]];
    }
}

perceptronMultiLayer::~perceptronMultiLayer() {
    for(unsigned i = 0; i < ln; i++) {
        for(unsigned j = 0; j < l[i]; j++)
            delete [] w[i][j];

        delete [] w[i];
        delete [] trainValues[i];
        delete [] computedValues[i];
    }

    delete [] w;
    delete [] trainValues;
    delete [] computedValues;
}

double* perceptronMultiLayer::classify(double *x) {

    for(unsigned j = 0; j < l[0]; j++) {
        double sum = 0.0;

        sum += w[0][j][0] * 1.0;
        for(unsigned k = 0; k < n; k++) {
            sum += w[0][j][k + 1] * x[k];
        }

        computedValues[0][j] = tan(sum);
    }

    for(unsigned i = 1; i < ln; i++) {
        for(unsigned j = 0; j < l[i]; j++) {
            double sum = 0.0;

            sum += w[i][j][0] * 1.0;
            for(unsigned k = 0; k < l[i - 1]; k++) {
                sum += w[i][j][k + 1] * computedValues[i - 1][k];
            }

            computedValues[i][j] = tanh(sum);
        }
    }

    return computedValues[ln - 1];

    /*
    // For each neuron in the first layer
    for(unsigned j = 0; j < l[0]; j++) {
        double sum = 0.0;

        sum += w[0][j][0] * 1.0;
        for (unsigned k = 0; k < n; k++)
            sum += w[0][j][k + 1] * x[k]; // Sum using the provided input

        // Save the value
        computedValues[0][j] = tanh(sum); // TODO handle 1 layer ?
    }

    // For each other layer
    for(unsigned i = 1; i < ln; i++) {
        // For each neuron
        for(unsigned j = 0; j < l[i]; j++) {
            double sum = 0.0;

            sum += w[i][j][0] * 1.0;
            for(unsigned k = 0; k < l[i - 1]; k++)
                sum += w[i][j][k + 1] * computedValues[i - 1][k]; // Sum using the previous outputs

            // Save the value
            //if(i == (ln - 1))
            //    computedValues[i][j] = sign(sum); // sign ? TODO Regression + Enum
            //else
                computedValues[i][j] = tanh(sum);
        }
    }

    return computedValues[ln - 1];*/
}

void perceptronMultiLayer::train(double a, double *x, double *y, unsigned k, unsigned max) {
    unsigned trainCount = 0;

    while(trainCount++ < max) {
        for(unsigned ex = 0; ex < k; ex++) {
            double* xk = x + ex * n;
            double* yk = y + ex * l[ln - 1];
            classify(xk);

            for(unsigned j = 0; j < l[ln - 1]; j++) {
                double xVal  = computedValues[ln - 1][j];
                trainValues[ln - 1][j] = (1 - (xVal * xVal)) * (xVal - yk[j]);
            }

            for(int i = ln - 2; i >= 0; i--) {
                for(unsigned j = 0; j < l[i]; j++) {
                    double xVal = computedValues[i][j];
                    double sum = 0.0;

                    for(unsigned m = 0; m < l[i + 1]; m++) {
                        sum += w[i + 1][m][j + 1] * trainValues[i + 1][m];
                    }

                    trainValues[i][j] = (1 - (xVal * xVal)) * sum;
                }
            }

            for(unsigned i = 1; i < ln; i++) {
                for(unsigned j = 0; j < l[i]; j++) {

                    w[i][j][0] -= a * 1.0 * trainValues[i][j];
                    for(unsigned m = 0; m < l[i - 1]; m++) {
                        w[i][j][m + 1] -= a * computedValues[i - 1][m] * trainValues[i][j];
                    }
                }
            }
        }
    }

    /*
    while(trainCount++ < max) {
        // Iter through each example
        for(unsigned ex = 0; ex < k; ex++) {
            double* xk = x + ex * n;         // Input array for example j
            double* yk = y + ex * l[ln - 1]; // Expected outputs for example j
            classify(xk); // Observed outputs for example j

            // For each neuron of the last layer
            for(unsigned j = 0; j < l[ln - 1]; j++) {
                double xVal = computedValues[ln - 1][j];

                //trainValues[ln - 1][j] = xVal * (1.0 - xVal) * (yk[j] - xVal);
                trainValues[ln - 1][j] = (1 - (xVal * xVal)) * (xVal - yk[j]);
            }

            // For every layer (excepted the last)
            for(int i = ln - 2; i >= 0; i--) {
                // For every neuron of the layer
                for(unsigned j = 0; j < l[i]; j++) {
                    double xVal = computedValues[i][j];
                    double sum = 0.0;

                    // Sum of all the weights * sigma connected to the actual neuron
                    for(unsigned m = 0; m < l[i + 1]; m++) {
                        sum += w[i + 1][m][j + 1] * trainValues[i + 1][m];
                    }

                    // Save the value
                     trainValues[i][j] = (1 - (xVal * xVal)) * sum;
                    //trainValues[i][j] = xVal * (1.0 - xVal) * sum;
                }
            }

            // TODO
            // Update the weights
            for(unsigned i = 1; i < ln; i++) {
                for(unsigned j = 0; j < l[i]; j++) {

                    w[i][j][0] = w[i][j][0] - a * computedValues[i - 1][0] * trainValues[i][j];
                    for(unsigned m = 0; m < l[i - 1]; m++) {

                        w[i][j][m + 1] = w[i][j][m + 1] - a * computedValues[i - 1][m] * trainValues[i][j];
                    }
                }
            }
        }
    }*/
}
