//
// Created by vuzi on 26/04/2016.
//

#ifndef PLUGIN_H
#define PLUGIN_H

#include <perceptron.h>
#include <perceptronMultiLayer.h>

#ifdef _WIN32
    // Windows DLL export
    #define EXPORT extern "C" __declspec(dllexport)
#else
    // Unix and OSX
    #define EXPORT
#endif

// Exported methods
// Perceptron
EXPORT perceptron* perceptronCreate(unsigned n, int type);
EXPORT void perceptronDispose(perceptron* p);

EXPORT int perceptronGetType(perceptron* p);

EXPORT double perceptronClassify(perceptron* p, double* x);
EXPORT double perceptronTrain(perceptron* p, double a, double* x, double* y, unsigned k, unsigned max);

// Perceptron multi layer
EXPORT perceptronMultiLayer* perceptronMLCreate(unsigned n, unsigned *l, unsigned ln, int type);
EXPORT void perceptronMLDispose(perceptronMultiLayer* p);

EXPORT double* perceptronMLClassify(perceptronMultiLayer* p, double* x);
EXPORT double perceptronMLTrain(perceptronMultiLayer* p, double a, double* x, double* y, unsigned k, unsigned max);

#endif //PLUGIN_H
