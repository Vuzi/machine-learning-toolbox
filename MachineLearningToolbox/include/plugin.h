//
// Created by vuzi on 26/04/2016.
//

#ifndef PLUGIN_H
#define PLUGIN_H

#ifdef _WIN32
    // Windows DLL export
    #define EXPORT extern "C" __declspec(dllexport)
#else
    // Unix and OSX
    #define EXPORT
#endif

/**
 * Perceptron structure
 */
struct perceptron {
    unsigned n;
    double* w;
};

typedef struct perceptron perceptron;

// Exported methods
EXPORT void linearClassificationTrainRosenblatt(double a, double* x, int* y, unsigned k, unsigned max, perceptron* p);
EXPORT void linearClassificationTrainPLA(double a, double* x, int* y, unsigned k, unsigned max, perceptron* p);

EXPORT void linearClassificationPLA(double a, int y, double* x, perceptron* p);
EXPORT void linearClassificationRosenblatt(double a, int yt, int y, double* x, perceptron* p);

EXPORT int linearClassification(double* x, perceptron* p);

EXPORT void deleteModel(perceptron* p);
EXPORT perceptron* createModel(unsigned n);

// 'private' methods
double randValue(double max, double min);
int sign(double x);

#endif //PLUGIN_H
