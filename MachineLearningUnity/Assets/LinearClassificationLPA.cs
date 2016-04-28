using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System;
using UnityEngine.UI;

public class LinearClassificationLPA : MonoBehaviour {

    public Text timerText;

    // Sphere to classify
    public GameObject[] toClassify;

    // Training set
    public GameObject[] reds;
    public GameObject[] blues;

    public void ResetClassification() {
        foreach(GameObject gameObject in toClassify) {
            Transform t = gameObject.GetComponent<Transform>();
            Renderer r = gameObject.GetComponent<Renderer>();

            r.material.color = Color.white;
            t.position = new Vector3(t.position.x, elementPositionRef.y, t.position.z);
        }
        GetComponent<Transform>().position = startPosition;
        GetComponent<Transform>().rotation = startRotation;
        timerText.text = "<none> ms";
    }

    public void HeavisideClassification() {
        ResetClassification();
        var watch = System.Diagnostics.Stopwatch.StartNew();

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

        watch.Stop();
        var elapsedMs = watch.ElapsedMilliseconds;
        timerText.text = "" + elapsedMs + " ms";
    }

    public void RosenblattClassification() {
        ResetClassification();
        var watch = System.Diagnostics.Stopwatch.StartNew();

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

        watch.Stop();
        var elapsedMs = watch.ElapsedMilliseconds;
        timerText.text = "" + elapsedMs + " ms";
    }

    public void Regression() {
        ResetClassification();
        var watch = System.Diagnostics.Stopwatch.StartNew();

        // Create the perceptron
        Perceptron model = new Perceptron(2, PerceptronType.REGRESSION);

        // Create the training values
        double[,] values = new double[reds.Length + blues.Length, 2];
        double[] expectedValues = new double[reds.Length + blues.Length];

        for(int i = 0; i < reds.Length; i++) {
            Transform t = reds[i].GetComponent<Transform>();
            values[i, 0] = t.position.x;
            values[i, 1] = t.position.z;
            expectedValues[i] = 1.0; // #ff0000
        }

        for(int i = 0; i < blues.Length; i++) {
            Transform t = blues[i].GetComponent<Transform>();
            values[i + reds.Length, 0] = t.position.x;
            values[i + reds.Length, 1] = t.position.z;
            expectedValues[i + reds.Length] = -1.0; // #0000ff
        }

        model.Train(0.001, values, expectedValues, 500000);

        // Use the perceptron
        foreach(GameObject gameObject in toClassify) {
            Transform t = gameObject.GetComponent<Transform>();
            Renderer r = gameObject.GetComponent<Renderer>();

            double val = model.Classify(new double[] { t.position.x, t.position.z });

            val = (val + 1) / 2;

            r.material.color = new Color((float)val, 0f, (float)(1.0 - val));

            /*if(val < 0) {
                r.material.color = new Color(0f + 1 - , 0f, (float)Math.Abs(val));
            } else {
                r.material.color = new Color((float)val, 0f, 0f);
            }*/

            //r.material.color = new Color(val < 0 ? 0f : (float)val, 0f, val > 0 ? 0f : (float)Math.Abs(val));
            t.Translate(Vector3.up * (float)val);
        }

        watch.Stop();
        var elapsedMs = watch.ElapsedMilliseconds;
        timerText.text = "" + elapsedMs + " ms";

        // Move camera
        GetComponent<Transform>().position = new Vector3(-6.3f, 2.8f, -5.3f);
        GetComponent<Transform>().rotation = Quaternion.Euler(4.5f, 46f, -1f);
    }

    private Vector3 startPosition;
    private Quaternion startRotation;
    private Vector3 elementPositionRef;

    // Use this for initialization
    void Start () {
        startPosition = GetComponent<Transform>().position;
        startRotation = GetComponent<Transform>().rotation;
        elementPositionRef = toClassify[0].GetComponent<Transform>().position;
    }

	// Update is called once per frame
	void Update () {}
}
