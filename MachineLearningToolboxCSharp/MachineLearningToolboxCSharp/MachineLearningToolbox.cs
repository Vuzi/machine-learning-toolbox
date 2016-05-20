using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningToolbox {

    public enum PerceptronType : int {
        HEAVISIDE = 0,
        ROSENBLATT = 1,
        REGRESSION = 2
    }

    public enum PerceptronMultiLayerType : int {
        CLASSIFICATION = 0,
        REGRESSION = 1
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

    public class PerceptronMultiLayer {
        [DllImport("libMachineLearningToolbox")]
        static extern void init();

        [DllImport("libMachineLearningToolbox")]
        static extern IntPtr multiLayerPerceptronCreate(int[] l, int ln, int type);
        [DllImport("libMachineLearningToolbox")]
        static extern void multiLayerPerceptronDispose(IntPtr p);

        [DllImport("libMachineLearningToolbox")]
        static extern IntPtr multiLayerPerceptronPropagate(IntPtr p, double[] x);
        [DllImport("libMachineLearningToolbox")]
        static extern void multiLayerPerceptronTrain(IntPtr p, double a, double[] x, double[] y, int k, int max);

        // Perceptron multi layer C++ object
        private IntPtr p;
        private int outSize;

        public PerceptronMultiLayer(int[] layers, PerceptronMultiLayerType type) {
            init();

            p = multiLayerPerceptronCreate(layers, layers.Length, (int)type);
            outSize = layers[layers.Length - 1];
        }

        ~PerceptronMultiLayer() {
            multiLayerPerceptronDispose(p);
        }

        public double[] Propagate(double[] values) {
            IntPtr ptr = multiLayerPerceptronPropagate(p, values);
            double[] result = new double[outSize];
            Marshal.Copy(ptr, result, 0, (int)outSize);

            return result;
        }

        public void Train(double trainingStep, double[,] values, double[,] expectedResults, int max) {
            double[] tmpValues = new double[values.Length];
            System.Buffer.BlockCopy(values, 0, tmpValues, 0, sizeof(double) * values.Length);
            double[] tmpExpectedResults = new double[expectedResults.Length];
            System.Buffer.BlockCopy(expectedResults, 0, tmpExpectedResults, 0, sizeof(double) * expectedResults.Length);

            multiLayerPerceptronTrain(p, trainingStep, tmpValues, tmpExpectedResults, expectedResults.Length / outSize, max);
        }

        public void Train(double trainingStep, double[] values, double[] expectedResults, int max) {
            multiLayerPerceptronTrain(p, trainingStep, values, expectedResults, expectedResults.Length / outSize, max);
        }
    }

}
