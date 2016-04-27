#include <iostream>
#include <Eigen/Dense>
#include <time.h>

#include <plugin.h>

using namespace std;

/**
 * Performs a training using Rosenblatt rule using the provided examples
 * a   => Training step
 * x*  => Input matrix (array of n * k values)
 * y*  => Expected output
 * k   => Number of provided examples
 * max => Max training iterations
 * p   => Perceptron to train
 */
void linearClassificationTrainRosenblatt(double a, double* x, int* y, unsigned k, unsigned max, perceptron* p) {
    unsigned i = 0;

    while(i++ < max) {
        bool error = false;

        // Iter through each example
        for(unsigned j = 0; j < k; j++) {
            double* xk = x + j * p->n; // Input array for example j
            int yk = y[j];             // Expected output for example j
            int yt = linearClassification(xk, p);

            if(yt != yk) {
                // Train the perceptron
                linearClassificationRosenblatt(a, yt, yk, xk, p);
                error = true;
            }
        }

        if(!error)
            break; // No error, end the training
    }
}

/**
 * Performs a training using PLA using the provided examples
 * a   => Training step
 * x*  => Input matrix (array of n * k values)
 * y*  => Expected output
 * k   => Number of provided examples
 * max => Max training iterations
 * p   => Perceptron to train
 */
void linearClassificationTrainPLA(double a, double* x, int* y, unsigned k, unsigned max, perceptron* p) {
    unsigned i = 0;

    while(i++ < max) {
        bool error = false;

        // Iter through each example
        for(unsigned j = 0; j < k; j++) {
            double* xk = x + j * p->n; // Input array for example j
            int yk = y[j];             // Expected output for example j

            if(linearClassification(xk, p) != yk) {
                // Train the perceptron
                linearClassificationPLA(a, yk, xk, p);
                error = true;
            }
        }

        if(!error)
            break; // No error, end the training

    }
}

/**
 * Performs a PLA for the provided perceptron
 */
void linearClassificationPLA(double a, int y, double* x, perceptron* p) {
    p->w[0] = p->w[0] + a * y;
    for(unsigned i = 0; i < p->n; i++)
        p->w[i + 1] = p->w[i + 1] + a * (y * x[i]);
}

/**
 * Performs a Rosenblatt training for the provided perceptron
 */
void linearClassificationRosenblatt(double a, int yt, int y, double* x, perceptron* p) {
    p->w[0] = p->w[0] + a * (yt - y);
    for(unsigned i = 0; i < p->n; i++)
        p->w[i] = p->w[i] + a * ((yt - y) * x[i]);
}

/**
 * Performs a linear classification on the provided perceptron with the provided input
 * x* => input matrix
 * w* => perceptron
 * n  => size
 */
int linearClassification(double* x, perceptron* p) {
    double sum = 0.0;

    sum += p->w[0] * 1.0;
    for(unsigned i = 0; i < p->n; i++)
        sum += p->w[i + 1] * x[i];

    return sign(sum);
}

/**
 * Delete and free the provided perceptron
 */
void deleteModel(perceptron* p) {
    delete [] p->w;
    delete p;
}

/**
 * Init and return a perceptron of size n
 * n  => size
 */
perceptron* createModel(unsigned n) {
    perceptron* p = new perceptron;
    p->w = new double[n + 1];
    p->n = n;

    // Init every weight with a rand between 0 an 1
    for(unsigned i = 0; i <= p->n ; i++)
        p->w[i] = randValue(1, -1);

    return p;
}

/**
 * Return a random double between the specified bounds
 */
double randValue(double max, double min) {
    return (max - min) * ((double)rand() / (double)RAND_MAX) + min;
}

/**
 * Return -1 if negative, otherwise 1
 */
int sign(double x) {
    return (x < 0) ? -1 : 1;
}

static void testRand() {
    cout << "Rand tests: ";
    for (int n = 0; n < 1000; ++n) {
        double v = randValue(1, -1);
        assert(v >= -1 && v <= 1);
    }
    cout << "passed" << endl;
}

static void testPerceptron() {
    cout << "Tests perceptron: " << endl;

    for(int i = 0; i < 100; i++) {
        perceptron *p = createModel(2);

        // Training values
        double *x = new double[8]{-1, 1,
                                   1, 1,
                                  -1, -1,
                                   1, -1};
        int *y = new int[4]{-1,
                            -1,
                             1,
                             1};

        // Train our model
        linearClassificationTrainPLA(0.1, x, y, 4, 5000, p);

        double *xTest1 = new double[2]{-1, 1}; // -1
        double *xTest2 = new double[2]{1, 1}; // -1
        double *xTest3 = new double[2]{-1, -1}; // 1
        double *xTest4 = new double[2]{0, -2}; // Unknown (should be 1)
        double *xTest5 = new double[2]{0, 1.5}; // Unknown (should be -1)

        assert(linearClassification(xTest1, p) == -1);
        assert(linearClassification(xTest2, p) == -1);
        assert(linearClassification(xTest3, p) == 1);
        assert(linearClassification(xTest4, p) == 1);
        assert(linearClassification(xTest5, p) == -1);

        cout << "\tTest " << (i + 1) << " passed" << endl;

        // Clear
        delete[] x;
        delete[] y;
        delete[] xTest1;
        delete[] xTest2;
        delete[] xTest3;
        delete[] xTest4;
        delete[] xTest5;

        deleteModel(p);
    }
    cout << "passed" << endl;
}

int main() {
    cout << "Tests" << endl;

    // TODO in some init method
    srand((unsigned) time(NULL));

    testRand();
    testPerceptron();

    cout << "All test passed" << endl;

    /*
    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;*/

    return 0;
}