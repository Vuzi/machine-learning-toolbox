#include <iostream>
#include <Eigen/Dense>
#include <time.h>

#include <plugin.h>

using namespace std;

// Tests
const char* getHelloWorld() {
    return "Hello world!";
}

int add(int a, int b) {
    return a + b;
}

int mult(int a, int b) {
    return a * b;
}
// End of tests

/**
 * Performs a Rosenblatt for the provided perceptron
 * a  => learning step
 * yt => observed result
 * y  => expected result
 * x* => input matrix
 * w* => perceptron
 */
double* linearClassificationRosenblatt(double a, int yt, int y, double* x, double* w, unsigned n) {
    for(unsigned i = 0; i < n; i++)
        w[i] = w[i] + a * ((yt - y) * x[i]);

    return w;
}

/**
 * Performs a PLA for the provided perceptron
 * a  => learning step
 * y  => expected result
 * x* => input matrix
 * w* => perceptron
 * n  => size
 */
double* linearClassificationPLA(double a, int y, double* x, double* w, unsigned n) {
    for(unsigned i = 0; i < n; i++)
        w[i] = w[i] + a * (y * x[i]);

    return w;
}

/**
 * Performs a linear classification on the provided perceptron with the provided input
 * x* => input matrix
 * w* => perceptron
 * n  => size
 */
int linearClassification(double* x, double* w, unsigned n) {
    double sum = 0;

    for(unsigned i = 0; i < n; i++)
        sum += w[i] * x[i];

    return sign(sum);
}

/**
 * Delete and free the provided perceptron
 */
void delPerceptron(double* w) {
    delete [] w;
}

/**
 * Init and return a perceptron of size n
 * n  => size
 */
double* initPerceptron(unsigned n) {
    double* m = new double[n];

    for(unsigned i = 0; i < n; i++)
        m[i] = randValue(1, -1);

    return m;
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
    if(x < 0)
        return -1;
    else
        return 1;
}

int main() {
    cout << "Hello, World!" << endl;

    // TODO in some init method
    srand((unsigned) time(NULL));

    for (int n = 0; n < 10; ++n) {
        std::cout << randValue(1, -1) << ' ';
    }
    std::cout << '\n';

    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;

    return 0;
}