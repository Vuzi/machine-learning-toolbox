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

EXPORT const char* getHelloWorld();
EXPORT int add(int a, int b);
EXPORT int mult(int a, int b);

#endif //PLUGIN_H
