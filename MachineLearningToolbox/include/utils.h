//
// Created by vuzi on 27/04/2016.
//

#ifndef UTILS_H
#define UTILS_H

#define sigmoid(x) (tanh(x))
#define dsigmoid(x) (1.0 - (x * x))

double randValue(double max, double min);
int sign(double x);

#endif //UTILS_H
