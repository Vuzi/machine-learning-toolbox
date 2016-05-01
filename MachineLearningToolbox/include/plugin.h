//
// Created by vuzi on 26/04/2016.
//

#ifndef PLUGIN_H
#define PLUGIN_H

#include <ctime>

#include <perceptron.h>
#include <multiLayerPerceptron.h>

#ifdef _WIN32
    // Windows DLL export
    #define EXPORT extern "C" __declspec(dllexport)
#else
    // Unix and OSX
    #define EXPORT extern "C"
#endif

// Exported methods
EXPORT void init();

// Perceptron
EXPORT perceptron* perceptronCreate(unsigned n, int type);
EXPORT void perceptronDispose(perceptron* p);

EXPORT int perceptronGetType(perceptron* p);

EXPORT double perceptronClassify(perceptron* p, double* x);
EXPORT double perceptronTrain(perceptron* p, double a, double* x, double* y, unsigned k, unsigned max);

// Perceptron multi layer
EXPORT multiLayerPerceptron* multiLayerPerceptronCreate(int *l, int ln, int type);
EXPORT void multiLayerPerceptronDispose(multiLayerPerceptron* p);

EXPORT int multiLayerPerceptronGetType(multiLayerPerceptron* p);

EXPORT double* multiLayerPerceptronPropagate(multiLayerPerceptron* p, double* x);
EXPORT double multiLayerPerceptronTrain(multiLayerPerceptron* p, double a, double* x, double* y, unsigned k, unsigned max);

#endif //PLUGIN_H
