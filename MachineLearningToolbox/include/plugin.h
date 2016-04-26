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

// Tests
EXPORT const char* getHelloWorld();
EXPORT int add(int a, int b);
EXPORT int mult(int a, int b);
// End of tests

// Exported methods
EXPORT double* linearClassificationRosenblatt(double a, int yt, int y, double* x, double* w, unsigned n);
EXPORT double* linearClassificationPLA(double a, int y, double* x, double* w, unsigned n);
EXPORT int linearClassification(double* x, double* w, unsigned n);
EXPORT void delPerceptron(double* w);
EXPORT double* initPerceptron(unsigned n);

// 'private' methods
double randValue(double max, double min);
int sign(double x);

#endif //PLUGIN_H
