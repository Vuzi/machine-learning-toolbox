using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System;

public class LinearClassificationLPA : MonoBehaviour {

    // Sphere to classify
    public GameObject[] toClassify;
    
    // Materials used
    public Material blue;
    public Material red;

    // Training set
    public GameObject[] reds;
    public GameObject[] blues;

    private void HeavisideTest() {
        // Create the perceptron
        Perceptron model = new Perceptron(2, PerceptronType.HEAVISIDE);

        // Create the training values
        double[,] values = new double[reds.Length + blues.Length, 2];
        double[] expectedValues = new double[reds.Length + blues.Length];

        for(int i = 0; i < reds.Length; i++) {
            Transform t = reds[i].GetComponent<Transform>();
            values[i, 0] = t.position.x;
            values[i, 1] = t.position.z;
            expectedValues[i] = -1;
        }

        for(int i = 0; i < blues.Length; i++) {
            Transform t = blues[i].GetComponent<Transform>();
            values[i + reds.Length, 0] = t.position.x;
            values[i + reds.Length, 1] = t.position.z;
            expectedValues[i + reds.Length] = 1;
        }

        model.Train(0.1, values, expectedValues, 5000);

        // Use the perceptron
        foreach(GameObject gameObject in toClassify) {
            Transform t = gameObject.GetComponent<Transform>();
            Renderer r = gameObject.GetComponent<Renderer>();

            if(model.Classify(new double[] { t.position.x, t.position.z }) < 0)
                r.material.color = Color.red;
            else
                r.material.color = Color.blue;
        }
    }

    private void RosenblattTest() {
        // Create the perceptron
        Perceptron model = new Perceptron(2, PerceptronType.ROSENBLATT);

        // Create the training values
        double[,] values = new double[reds.Length + blues.Length, 2];
        double[] expectedValues = new double[reds.Length + blues.Length];

        for(int i = 0; i < reds.Length; i++) {
            Transform t = reds[i].GetComponent<Transform>();
            values[i, 0] = t.position.x;
            values[i, 1] = t.position.z;
            expectedValues[i] = 0.0;
        }

        for(int i = 0; i < blues.Length; i++) {
            Transform t = blues[i].GetComponent<Transform>();
            values[i + reds.Length, 0] = t.position.x;
            values[i + reds.Length, 1] = t.position.z;
            expectedValues[i + reds.Length] = 1.0;
        }

        model.Train(0.1, values, expectedValues, 5000);

        // Use the perceptron
        foreach(GameObject gameObject in toClassify) {
            Transform t = gameObject.GetComponent<Transform>();
            Renderer r = gameObject.GetComponent<Renderer>();
            
            if(model.Classify(new double[] { t.position.x, t.position.z }) == 0)
                r.material.color = Color.red;
            else
                r.material.color = Color.blue;
        }
    }

    // Use this for initialization
    void Start () {
        //HeavisideTest();
        RosenblattTest();
    }

	// Update is called once per frame
	void Update () {}
}
