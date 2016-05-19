//
// Created by vuzi on 27/04/2016.
//

#include <iostream>
#include <cassert>

#include <plugin.h>

using namespace std;

/**
 * Test on the rand
 */
static void testRand() {
    cout << "Rand tests: ";
    for (int n = 0; n < 1000; ++n) {
        double v = randValue(1, -1);
        assert(v >= -1 && v <= 1);
    }
    cout << "passed" << endl;
}

/**
 * Test on the perceptron
 */
static void testPerceptron() {
    cout << "Tests perceptron: " << endl;

    for(int i = 0; i < 100; i++) {
        perceptron p(2, TYPE_HEAVISIDE);

        // Training values
        double *x = new double[8]{-1,  1,
                                   1,  1,
                                  -1, -1,
                                   1, -1};
        double *y = new double[4]{-1,
                                  -1,
                                   1,
                                   1};

        // Train our model
        p.train(0.1, x, y, 4, 5000);

        double *xTest1 = new double[2]{-1, 1}; // -1
        double *xTest2 = new double[2]{1, 1}; // -1
        double *xTest3 = new double[2]{-1, -1}; // 1
        double *xTest4 = new double[2]{0, -2}; // Unknown (should be 1)
        double *xTest5 = new double[2]{0, 1.5}; // Unknown (should be -1)

        assert(p.classify(xTest1) == -1);
        assert(p.classify(xTest2) == -1);
        assert(p.classify(xTest3) ==  1);
        assert(p.classify(xTest4) ==  1);
        assert(p.classify(xTest5) == -1);

        cout << "\tTest " << (i + 1) << " passed" << endl;

        // Clear
        delete[] x;
        delete[] y;
        delete[] xTest1;
        delete[] xTest2;
        delete[] xTest3;
        delete[] xTest4;
        delete[] xTest5;
    }
    cout << "passed" << endl;
}

static void testPerceptronMultiLayer() {
    cout << "Tests perceptron mulit layer: " << endl;

    int passed = 0;

    for(int i = 0; i < 100; i++) {

        bool hasFailed = false;
        int *l = new int[3]{2, 3, 1};

        multiLayerPerceptron p(l, 3, MLP_TYPE_CLASSIFICATION);

        cout << "init" << endl;

        // Training values (XOR)
        double *x = new double[8]{ 1,  1,
                                   1, -1,
                                  -1,  1,
                                  -1, -1};

        double *y = new double[4]{-1, 1, 1, -1};

        // Train our model
        p.train(0.1, x, y, 4, 500);

        cout << "train" << endl;

        double *xTest1 = new double[2]{-1, 1}; // 1
        double *xTest2 = new double[2]{1, -1}; // 1
        double *xTest3 = new double[2]{1, 1}; // -1
        double *xTest4 = new double[2]{-1, -1}; // -1

        p.propagate(xTest3);

        cout << "propagate" << endl;

        cout << p.propagate(xTest1)[0] << endl; // 1
        cout << p.propagate(xTest2)[0] << endl; // 1
        cout << p.propagate(xTest3)[0] << endl; // -1
        cout << p.propagate(xTest4)[0] << endl; // -1

        if(!(p.propagate(xTest1)[0] > 0)) hasFailed = true;
        if(!(p.propagate(xTest2)[0] > 0)) hasFailed = true;
        if(!(p.propagate(xTest3)[0] < 0)) hasFailed = true;
        if(!(p.propagate(xTest4)[0] < 0)) hasFailed = true;

        if(!hasFailed) {
            passed++;
            cout << "\tTest " << (i + 1) << " passed" << endl;
        } else {
            cout << "\tTest " << (i + 1) << " failed" << endl;
        };

        delete[] x;
        delete[] y;
        delete[] xTest1;
        delete[] xTest2;
        delete[] xTest3;
        delete[] xTest4;

        cout << "clean done" << endl;
    }

    cout << passed << "/100 passed" << endl;
}

int main() {
    cout << "Tests" << endl;

    init();

    //srand(42);
    testRand();
    testPerceptron();
    testPerceptronMultiLayer();

    return 0;
}
