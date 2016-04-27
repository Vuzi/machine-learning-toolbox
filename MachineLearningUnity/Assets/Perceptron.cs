using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System;

public enum PerceptronType : int {
    HEAVISIDE = 0,
    ROSENBLATT = 1,
    REGRESSION = 2
}

public class Perceptron {
    [DllImport("libMachineLearningToolbox")]
    static extern IntPtr perceptronCreate(uint n, int type);
    [DllImport("libMachineLearningToolbox")]
    static extern void perceptronDispose(IntPtr p);

    [DllImport("libMachineLearningToolbox")]
    static extern int perceptronGetType(IntPtr p);

    [DllImport("libMachineLearningToolbox")]
    static extern double perceptronClassify(IntPtr p, double[] x);
    [DllImport("libMachineLearningToolbox")]
    static extern double perceptronTrain(IntPtr p, double a, double[] x, double[] y, uint k, uint max);

    // Perceptron C++ object
    private IntPtr p;

    public Perceptron(uint size, PerceptronType type) {
        p = perceptronCreate(size, (int)type);
    }

    ~Perceptron() {
        perceptronDispose(p);
    }

    public double Classify(double[] values) {
        return perceptronClassify(p, values);
    }

    public void Train(double trainingStep, double[,] values, double[] expectedResults, uint max) {
        double[] tmpValues = new double[values.Length];
        System.Buffer.BlockCopy(values, 0, tmpValues, 0, sizeof(double) * values.Length);

        perceptronTrain(p, trainingStep, tmpValues, expectedResults, (uint)expectedResults.Length, max);
    }

    public PerceptronType GetPerceptronType() {
        return (PerceptronType)perceptronGetType(p);
    }
}
