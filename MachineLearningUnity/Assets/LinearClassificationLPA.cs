using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System;

public class LinearClassificationLPA : MonoBehaviour {

    public class Perceptron
    {
        [DllImport("libMachineLearningToolbox")]
        static extern void linearClassificationTrainRosenblatt(double a, double[] x, int[] y, uint k, uint max, IntPtr p);

        [DllImport("libMachineLearningToolbox")]
        static extern void linearClassificationTrainPLA(double a, double[] x, int[] y, uint k, uint max, IntPtr p);

        [DllImport("libMachineLearningToolbox")]
        static extern void linearClassificationPLA(double a, int y, double[] x, IntPtr p);

        [DllImport("libMachineLearningToolbox")]
        static extern void linearClassificationRosenblatt(double a, int yt, int y, double[] x, IntPtr p);

        [DllImport("libMachineLearningToolbox")]
        static extern int linearClassification(double[] x, IntPtr p);

        [DllImport("libMachineLearningToolbox")]
        static extern void deleteModel(IntPtr p);

        [DllImport("libMachineLearningToolbox")]
        static extern IntPtr createModel(uint n);

        // Perceptron C++ object
        private IntPtr p;

        public Perceptron(uint size) {
            p = createModel(size);
        }

        ~Perceptron() {
            deleteModel(p);
        }

        public int Classify(double[] values) {
            return linearClassification(values, p);
        }

        public void TrainPLA(double trainingStep, double[] values, int[] expectedResults, uint max) {
            linearClassificationTrainPLA(trainingStep, values, expectedResults, (uint) expectedResults.Length, max, p);
        }
    }

    // Sphere to classify
    public GameObject[] transforms;
    
    // Materials used
    public Material blue;
    public Material red;

    // Training set
    public Transform[] reds;
    public Transform[] blues;

    // Use this for initialization
    void Start () {
        // Create the perceptron
        Perceptron model = new Perceptron(2);

        // Create the training values
        double[] values = new double[(reds.Length + blues.Length) * 2];
        int[] expectedValues = new int[reds.Length + blues.Length];

        for (int i = 0; i < reds.Length; i++) {
            values[i * 2] = reds[i].position.x;
            values[(i * 2) + 1] = reds[i].position.z;
            expectedValues[i] = -1;
        }

        for (int i = 0; i < blues.Length; i++) {
            values[i * 2 + reds.Length * 2] = blues[i].position.x;
            values[(i * 2) + 1 + reds.Length * 2] = blues[i].position.z;
            expectedValues[i + reds.Length] = 1;
        }

        model.TrainPLA(0.1, values, expectedValues, 5000);

        // Use the perceptron
        foreach(GameObject transform in transforms)
        {
            if (model.Classify(new double[] { transform.GetComponent<Transform>().position.x, transform.GetComponent<Transform>().position.z }) < 0)
                transform.GetComponent<Renderer>().material.color = Color.red;
            else
                transform.GetComponent<Renderer>().material.color = Color.blue;
        }
    }

	// Update is called once per frame
	void Update () {}
}
