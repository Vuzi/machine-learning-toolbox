using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System;

public class PerceptronMultiLayer {
    [DllImport("libMachineLearningToolbox")]
    static extern IntPtr perceptronMLCreate(uint n, uint[] l, uint ln, int type);
    [DllImport("libMachineLearningToolbox")]
    static extern void perceptronMLDispose(IntPtr p);

    [DllImport("libMachineLearningToolbox")]
    static extern double[] perceptronMLClassify(IntPtr p, double[] x);
    [DllImport("libMachineLearningToolbox")]
    static extern double perceptronMLTrain(IntPtr p, double a, double[] x, double[] y, uint k, uint max);

    // Perceptron multi layer C++ object
    private IntPtr p;

    public PerceptronMultiLayer(uint inputSize, uint[] layers, PerceptronType type) {
        p = perceptronMLCreate(inputSize, layers, (uint)layers.Length, (int)type);
    }

    ~PerceptronMultiLayer() {
        perceptronMLDispose(p);
    }

    public double[] Classify(double[] values) {
        // TODO copy returned array
        return perceptronMLClassify(p, values);
    }

    public void Train(double trainingStep, double[,] values, double[] expectedResults, uint max) {
        double[] tmpValues = new double[values.Length];
        System.Buffer.BlockCopy(values, 0, tmpValues, 0, sizeof(double) * values.Length);

        perceptronMLTrain(p, trainingStep, tmpValues, expectedResults, (uint)expectedResults.Length, max);
    }
}

