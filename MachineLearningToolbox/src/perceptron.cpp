//
// Created by vuzi on 27/04/2016.
//

#include <perceptron.h>

perceptron::perceptron(unsigned _n, perceptronType _type) {
    n = _n;
    w = new double[n + 1];
    type = _type;

    // Init every weight with a rand between 0 an 1
    for(unsigned i = 0; i <= n ; i++)
        w[i] = randValue(1, -1);
}

perceptron::~perceptron() {
    delete [] w;
}

perceptronType perceptron::getType() {
    return type;
}

double perceptron::classify(double *x) {
    double sum = 0.0;

    sum += w[0] * 1.0;
    for(unsigned i = 0; i < n; i++)
        sum += w[i + 1] * x[i];

    switch (type) {
        case TYPE_HEAVISIDE:
            return sign(sum);
        case TYPE_ROSENBLATT:
            return sum > 0.0 ? 1.0 : 0.0;
        case TYPE_REGRESSION:
        default:
            return sum;
    }
}

void perceptron::train(double a, double *x, double *y, unsigned k, unsigned max) {
    switch (type) {
        case TYPE_HEAVISIDE:
        case TYPE_ROSENBLATT:
            trainLinear(a, x, y, k, max);
            break;
        case TYPE_REGRESSION:
            trainRegression(a, x,y, k, max);
            break;
    }
}

void perceptron::trainLinear(double a, double *x, double *y, unsigned k, unsigned max) {
    unsigned i = 0;

    while(i++ < max) {
        bool error = false;

        // Iter through each example
        for(unsigned j = 0; j < k; j++) {
            double* xk = x + j * n;   // Input array for example j
            double yk = y[j];         // Expected output for example j
            double yt = classify(xk); // Observed output for example j

            if(yt != yk) {
                // Train the perceptron
                if(type == TYPE_ROSENBLATT)
                    updateModelRosenblatt(a, yk, yt, xk);
                else
                    updateModelHeaviside(a, yk, xk);
                error = true; break;
            }
        }

        if(!error) break; // No error, end the training
    }
}

void perceptron::trainRegression(double a, double *x, double *y, unsigned k, unsigned max) {
    // Prepare y matrix
    Eigen::MatrixXd yMat(k, 1);
    for(int i = 0; i < k; i++)
        yMat(i, 0) = y[i];

    // Prepare x matrix (unfold the provided array)
    Eigen::MatrixXd xMat(k, n + 1);
    for(int i = 0; i < k; i++) {
        xMat(i, 0) = 1.0;
        for(int j = 0; j < n; j++) {
            xMat(i, j + 1) = x[n * i + j];
        }
    }

    Eigen::MatrixXd xMatT = xMat.transpose();
    Eigen::MatrixXd wMat = ((xMatT * xMat).inverse() * xMatT) * yMat; // W = ((Xt * X)^-1 * Xt) * Y

    // Map back to w
    Eigen::Map<Eigen::MatrixXd>(w, n + 1, 1) = wMat;
}

double perceptron::updateModelHeaviside(double a, double y, double *x) {
    w[0] = w[0] + a * y;
    for(unsigned i = 0; i < n; i++)
        w[i + 1] = w[i + 1] + a * (y * x[i]);
}

double perceptron::updateModelRosenblatt(double a, double yk, double y, double *x) {
    w[0] = w[0] + a * (yk - y);
    for(unsigned i = 0; i < n; i++)
        w[i + 1] = w[i + 1] + a * ((yk - y) * x[i]);
}
