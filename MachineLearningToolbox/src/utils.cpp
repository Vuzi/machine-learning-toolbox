//
// Created by vuzi on 27/04/2016.
//

#include <utils.h>
#include <cstdlib>

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