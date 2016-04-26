using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System;

public class MyCube : MonoBehaviour {

    // Test Unity
    Transform cubeTransform;
    
    public Material blue;
    
    public Material red;

    // Test DLL
    [DllImport("libMachineLearningToolbox")]
    public static extern IntPtr getHelloWorld();

    [DllImport("libMachineLearningToolbox")]
    public static extern int add(int a, int b);

    [DllImport("libMachineLearningToolbox")]
    public static extern int mult(int a, int b);

    // Use this for initialization
    void Start () {
        // Test DLL
        Debug.Log("Test DLL import:");

        string s = Marshal.PtrToStringAnsi(getHelloWorld());
        Debug.Log(s);

        Debug.Log("1 + 3 = " + add(1, 3));
        Debug.Log("2 * 6 = " + mult(2, 6));

        Debug.Log("All good");

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
