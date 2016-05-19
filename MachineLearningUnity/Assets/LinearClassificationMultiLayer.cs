using UnityEngine;
using System.Collections;
using System;
using UnityEngine.UI;
using System.Linq;
using System.Collections.Generic;

public class LinearClassificationMultiLayer : MonoBehaviour {

    public Text timerText;
    
    // Renderer
    public int textureSize = 512;
    public GameObject renderer;

    // Training set
    GameObject[] reds;
    GameObject[] blues;
    GameObject[] greens;

    public InputField dimensions;
    public InputField iterations;
    public InputField learningStep;

    public void Reset() {
        renderer.GetComponent<Renderer>().material.mainTexture = null;

        timerText.text = "<none> ms";
        
        reds = GameObject.FindGameObjectsWithTag("red");
        blues = GameObject.FindGameObjectsWithTag("blue");
        greens = GameObject.FindGameObjectsWithTag("green");
    }
    
    public void Classification() {
        Reset();

        // Get infos
        List<int> d = dimensions.text.Length == 0 ? new List<int>() : dimensions.text.Split(',').Select((Func<string, int>)int.Parse).ToList<int>();
        d.Insert(0, 2); // Input layer
        d.Add(3); // Output layer

        int it = int.Parse(iterations.text);
        float step = float.Parse(learningStep.text);

        var watch = System.Diagnostics.Stopwatch.StartNew();

        // Create the perceptron
        PerceptronMultiLayer model = new PerceptronMultiLayer(d.ToArray(), PerceptronType.HEAVISIDE);

        // Create the training values
        double[,] values = new double[reds.Length + blues.Length + greens.Length, 2];
        double[,] expectedValues = new double[reds.Length + blues.Length + greens.Length, 3];

        for (int i = 0; i < reds.Length; i++) {
            Transform t = reds[i].GetComponent<Transform>();
            values[i, 0] = t.position.x;
            values[i, 1] = t.position.z;
            expectedValues[i, 0] = 1;
            expectedValues[i, 1] = -1;
            expectedValues[i, 2] = -1;
        }

        for (int i = 0; i < blues.Length; i++) {
            Transform t = blues[i].GetComponent<Transform>();
            values[i + reds.Length, 0] = t.position.x;
            values[i + reds.Length, 1] = t.position.z;
            expectedValues[i + reds.Length, 0] = -1;
            expectedValues[i + reds.Length, 1] = 1;
            expectedValues[i + reds.Length, 2] = -1;
        }

        for (int i = 0; i < greens.Length; i++) {
            Transform t = greens[i].GetComponent<Transform>();
            values[i + reds.Length + blues.Length, 0] = t.position.x;
            values[i + reds.Length + blues.Length, 1] = t.position.z;
            expectedValues[i + reds.Length + blues.Length, 0] = -1;
            expectedValues[i + reds.Length + blues.Length, 1] = -1;
            expectedValues[i + reds.Length + blues.Length, 2] = 1;
        }

        model.Train(step, values, expectedValues, it);

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

                double[] val = model.Propagate(new double[] { xValue, yValue });

                texture.SetPixel(-x, -y, new Color(
                    Convert.ToSingle((val[0] + 1) / 2),    // R
                    Convert.ToSingle((val[2] + 1) / 2),    // G
                    Convert.ToSingle((val[1] + 1) / 2)));  // B
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
