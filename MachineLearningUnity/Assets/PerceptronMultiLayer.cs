using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System;

public class PerceptronMultiLayer {
    [DllImport("libMachineLearningToolbox")]
    static extern void init();

    [DllImport("libMachineLearningToolbox")]
    static extern IntPtr perceptronMLCreate(uint n, uint[] l, uint ln, int type);
    [DllImport("libMachineLearningToolbox")]
    static extern void perceptronMLDispose(IntPtr p);

    [DllImport("libMachineLearningToolbox")]
    static extern IntPtr perceptronMLClassify(IntPtr p, double[] x);
    [DllImport("libMachineLearningToolbox")]
    static extern void perceptronMLTrain(IntPtr p, double a, double[] x, double[] y, uint k, uint max);

    // Perceptron multi layer C++ object
    private IntPtr p;
    private uint outSize;

    public PerceptronMultiLayer(uint inputSize, uint[] layers, PerceptronType type) {
        init();

        p = perceptronMLCreate(inputSize, layers, (uint)layers.Length, (int)type);
        outSize = layers[layers.Length - 1];
    }

    ~PerceptronMultiLayer() {
        perceptronMLDispose(p);
    }

    public double[] Classify(double[] values) {
        // TODO copy returned array
        IntPtr ptr = perceptronMLClassify(p, values);
        double[] result = new double[outSize];
        Marshal.Copy(ptr, result, 0, (int)outSize);

        return result;
    }

    public void Train(double trainingStep, double[,] values, double[] expectedResults, uint max) {
        double[] tmpValues = new double[values.Length];
        System.Buffer.BlockCopy(values, 0, tmpValues, 0, sizeof(double) * values.Length);

        perceptronMLTrain(p, trainingStep, tmpValues, expectedResults, (uint)expectedResults.Length, max);
    }
}

