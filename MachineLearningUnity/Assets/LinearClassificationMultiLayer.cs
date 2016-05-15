using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System;
using UnityEngine.UI;

public class LinearClassificationMultiLayer : MonoBehaviour {

    public Text timerText;

    // Sphere to classify
    GameObject[] toClassify;

    // Training set
    GameObject[] reds;
    GameObject[] blues;

    public void Reset() {
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

    public void Classification() {
        Reset();
        var watch = System.Diagnostics.Stopwatch.StartNew();

        // Create the perceptron
        PerceptronMultiLayer model = new PerceptronMultiLayer(new int[] { 2, 10, 10, 1 }, PerceptronType.HEAVISIDE);

        // Create the training values
        double[,] values = new double[reds.Length + blues.Length, 2];
        double[] expectedValues = new double[reds.Length + blues.Length];

        for(int i = 0; i < reds.Length; i++) {
            Transform t = reds[i].GetComponent<Transform>();
            values[i, 0] = t.position.x;
            values[i, 1] = t.position.z;
            expectedValues[i] = 0;
        }

        for(int i = 0; i < blues.Length; i++) {
            Transform t = blues[i].GetComponent<Transform>();
            values[i + reds.Length, 0] = t.position.x;
            values[i + reds.Length, 1] = t.position.z;
            expectedValues[i + reds.Length] = 1;
        }

        model.Train(0.1, values, expectedValues, 50000);

        // Use the perceptron
        foreach(GameObject gameObject in toClassify) {
            Transform t = gameObject.GetComponent<Transform>();
            Renderer r = gameObject.GetComponent<Renderer>();

            double[] val = model.Propagate(new double[] { t.position.x, t.position.z });

            Debug.Log(val[0]);

            if(val[0] < 0.5)
                r.material.color = Color.red;
            else
                r.material.color = Color.blue;
        }

        watch.Stop();
        var elapsedMs = watch.ElapsedMilliseconds;
        timerText.text = "" + elapsedMs + " ms";
    }

    private Vector3 startPosition;
    private Quaternion startRotation;
    private Vector3 elementPositionRef;

    // Use this for initialization
    void Start () {
        toClassify = GameObject.FindGameObjectsWithTag("white");
        reds = GameObject.FindGameObjectsWithTag("red");
        blues = GameObject.FindGameObjectsWithTag("blue");

        startPosition = GetComponent<Transform>().position;
        startRotation = GetComponent<Transform>().rotation;
        elementPositionRef = toClassify[0].GetComponent<Transform>().position;
    }

	// Update is called once per frame
	void Update () {}
}
