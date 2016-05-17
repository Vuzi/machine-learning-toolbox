using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System;
using UnityEngine.UI;
using System.Linq;
using System.Collections.Generic;

public class LinearClassificationMultiLayer : MonoBehaviour {

    public Text timerText;

    // Sphere to classify
    GameObject[] toClassify;

    // Training set
    GameObject[] reds;
    GameObject[] blues;

    public InputField dimensions;
    public InputField iterations;
    public InputField learningStep;

    public void Reset() {
        if(toClassify != null)
            foreach(GameObject gameObject in toClassify) {
                Transform t = gameObject.GetComponent<Transform>();
                Renderer r = gameObject.GetComponent<Renderer>();

                r.material.color = Color.white;
            }
        GetComponent<Transform>().position = startPosition;
        GetComponent<Transform>().rotation = startRotation;
        timerText.text = "<none> ms";

        toClassify = GameObject.FindGameObjectsWithTag("white");
        reds = GameObject.FindGameObjectsWithTag("red");
        blues = GameObject.FindGameObjectsWithTag("blue");
    }

    public void Classification() {
        Reset();

        // Get infos
        List<int> d = dimensions.text.Split(',').Select((Func<string, int>)int.Parse).ToList<int>();
        d.Insert(0, 2); // Input layer
        d.Add(1); // Output layer

        int it = int.Parse(iterations.text);
        float step = float.Parse(learningStep.text);

        Debug.Log(d);
        Debug.Log(it);
        Debug.Log(step);

        var watch = System.Diagnostics.Stopwatch.StartNew();

        // Create the perceptron
        PerceptronMultiLayer model = new PerceptronMultiLayer(d.ToArray(), PerceptronType.HEAVISIDE);

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

        model.Train(step, values, expectedValues, it);

        // Use the perceptron
        foreach(GameObject gameObject in toClassify) {
            Transform t = gameObject.GetComponent<Transform>();
            Renderer r = gameObject.GetComponent<Renderer>();

            double[] val = model.Propagate(new double[] { t.position.x, t.position.z });
            
            if(val[0] < 0)
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
    //private Vector3 elementPositionRef;

    // Use this for initialization
    void Start () {
        startPosition = GetComponent<Transform>().position;
        startRotation = GetComponent<Transform>().rotation;
        //elementPositionRef = toClassify[0].GetComponent<Transform>().position;
    }

	// Update is called once per frame
	void Update () {}
}
