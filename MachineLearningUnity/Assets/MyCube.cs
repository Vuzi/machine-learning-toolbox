using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System;

public class MyCube : MonoBehaviour {

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

        public int classify(double[] values) {
            return linearClassification(values, p);
        }

        public void TrainPLA(double trainingStep, double[] values, int[] expectedResults, uint max) {
            linearClassificationTrainPLA(trainingStep, values, expectedResults, (uint) expectedResults.Length, max, p);
        }
    }

    // Test Unity
    Transform cubeTransform;
    
    public Material blue;
    
    public Material red;

    // Use this for initialization
    void Start () {
        // Test DLL

        Debug.Log("Tests perceptron:");

        for (int i = 0; i < 100; i++) {
            Perceptron model = new Perceptron(2);

            // Training values
            double[] x = new double[8]{-1, 1,
                                        1, 1,
                                       -1, -1,
                                        1, -1 };
            int[] y = new int[4]{-1,
                                 -1,
                                  1,
                                  1};

            // Train our model
            model.TrainPLA(0.1, x, y, 5000);

            // Tests
            double[][] tests = new double[5][]
            {
                new double[2] {
                    -1, 1 // -1
                },
                new double[2] {
                    1, 1 // -1
                },
                new double[2] {
                    -1, -1 // 1
                },
                new double[2] {
                    0, -2 // Unknown (should be 1)
                },
                new double[2] {
                    0, 1.5 // Unknown (should be -1)
                }
            };

            Debug.Assert(model.classify(tests[0]) == -1);
            Debug.Assert(model.classify(tests[1]) == -1);
            Debug.Assert(model.classify(tests[2]) == 1);
            Debug.Assert(model.classify(tests[3]) == 1);
            Debug.Assert(model.classify(tests[4]) == -1);

            Debug.Log("\tTest " + (i + 1) + "passed");
        }
        Debug.Log("Passed");

        // Tests Unity
        GetComponent<Renderer>().material = blue;
    }
	
    private bool down = false;

	// Update is called once per frame
	void Update () {
        if (!down && transform.position.y > 5)
        {
            GetComponent<Renderer>().material = red;
            down = true;
        } else if (down && transform.position.y < 0)
        {
            GetComponent<Renderer>().material = blue;
            down = false;
        }

        if (down)
            transform.Translate(Vector3.down * Time.deltaTime);
        else
            transform.Translate(Vector3.up * Time.deltaTime);
    }
}
