using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System;
using UnityEngine.UI;

public class LinearClassificationLPA : MonoBehaviour {

    public Text timerText;

    // Renderer
    public int textureSize = 1024;
    public GameObject renderer;
    
    // Training set
    GameObject[] reds;
    GameObject[] blues;

    public void ResetClassification() {
        renderer.GetComponent<Renderer>().material.mainTexture = null;

        timerText.text = "<none> ms";

        // Update values
        reds = GameObject.FindGameObjectsWithTag("red");
        blues = GameObject.FindGameObjectsWithTag("blue");
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

        // Texture generation
        Transform tRenderer = renderer.GetComponent<Transform>();
        double xWidth = tRenderer.localScale.x;
        double yWidth = tRenderer.localScale.z;

        double xDec = tRenderer.position.x;
        double yDec = tRenderer.position.y;

        var texture = new Texture2D(textureSize, textureSize, TextureFormat.ARGB32, false);

        for (int x = 0; x < textureSize; x++) {
            for (int y = 0; y < textureSize; y++) {
                double xValue = ((((double)x / textureSize) * xWidth) - (xWidth / 2)) + xDec;
                double yValue = ((((double)y / textureSize) * yWidth) - (yWidth / 2)) + yDec;

                if (model.Classify(new double[] { xValue, yValue }) < 0)
                    texture.SetPixel(-x, -y, Color.red);
                else
                    texture.SetPixel(-x, -y, Color.blue);
            }
        }

        texture.Apply();
        renderer.GetComponent<Renderer>().material.mainTexture = texture;
        
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

        // Texture generation
        Transform tRenderer = renderer.GetComponent<Transform>();
        double xWidth = tRenderer.localScale.x;
        double yWidth = tRenderer.localScale.z;

        double xDec = tRenderer.position.x;
        double yDec = tRenderer.position.y;

        var texture = new Texture2D(textureSize, textureSize, TextureFormat.ARGB32, false);

        for (int x = 0; x < textureSize; x++) {
            for (int y = 0; y < textureSize; y++) {
                double xValue = ((((double)x / textureSize) * xWidth) - (xWidth / 2)) + xDec;
                double yValue = ((((double)y / textureSize) * yWidth) - (yWidth / 2)) + yDec;

                if (model.Classify(new double[] { xValue, yValue }) == 0)
                    texture.SetPixel(-x, -y, Color.red);
                else
                    texture.SetPixel(-x, -y, Color.blue);
            }
        }

        texture.Apply();
        renderer.GetComponent<Renderer>().material.mainTexture = texture;
        
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

        // Texture generation
        Transform tRenderer = renderer.GetComponent<Transform>();
        double xWidth = tRenderer.localScale.x;
        double yWidth = tRenderer.localScale.z;

        double xDec = tRenderer.position.x;
        double yDec = tRenderer.position.y;

        var texture = new Texture2D(textureSize, textureSize, TextureFormat.ARGB32, false);

        for (int x = 0; x < textureSize; x++) {
            for (int y = 0; y < textureSize; y++) {
                double xValue = ((((double)x / textureSize) * xWidth) - (xWidth / 2)) + xDec;
                double yValue = ((((double)y / textureSize) * yWidth) - (yWidth / 2)) + yDec;

                double val = model.Classify(new double[] { xValue, yValue });
                texture.SetPixel(-x, -y, new Color((float)val, 0f, (float)(1.0 - val)));
            }
        }

        texture.Apply();
        renderer.GetComponent<Renderer>().material.mainTexture = texture;

        watch.Stop();
        var elapsedMs = watch.ElapsedMilliseconds;
        timerText.text = "" + elapsedMs + " ms";
    }

    // Use this for initialization
    void Start () {}

	// Update is called once per frame
	void Update () {}
}
