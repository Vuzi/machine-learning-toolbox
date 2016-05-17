using UnityEngine;
using System.Collections;
using UnityEngine.EventSystems;

public class Interactive : MonoBehaviour {

    enum ElementType {
        NONE, RED, BLUE
    }

    ElementType type = ElementType.NONE;
    GameObject elementOverlay = null;

    // Use this for initialization
    void Start () {
	
	}

    public void Clear() {
        foreach (GameObject gameObject in GameObject.FindGameObjectsWithTag("red")) {
            Destroy(gameObject);
        }
        foreach (GameObject gameObject in GameObject.FindGameObjectsWithTag("blue")) {
            Destroy(gameObject);
        }
    }

    public void SelectNone() {
        type = ElementType.NONE;

        // Destroy the overlay sphere
        DestroyOverlayElement();
    }

    public void SelectBlue() {
        type = ElementType.BLUE;

        CreateOverlayElement(Color.blue);
    }

    public void SelectRed() {
        type = ElementType.RED;

        CreateOverlayElement(Color.red);
    }

    private void DestroyOverlayElement() {
        Destroy(elementOverlay);
        elementOverlay = null;
    }

    private void CreateOverlayElement(Color color) {
        if(elementOverlay == null)
            elementOverlay = GameObject.CreatePrimitive(PrimitiveType.Sphere);

        color.a = 0.5f;
        elementOverlay.GetComponent<Renderer>().material.color = color;
        elementOverlay.GetComponent<Renderer>().material.SetFloat("_Mode", 2);
        elementOverlay.GetComponent<Renderer>().material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        elementOverlay.GetComponent<Renderer>().material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        elementOverlay.GetComponent<Renderer>().material.SetInt("_ZWrite", 0);
        elementOverlay.GetComponent<Renderer>().material.DisableKeyword("_ALPHATEST_ON");
        elementOverlay.GetComponent<Renderer>().material.EnableKeyword("_ALPHABLEND_ON");
        elementOverlay.GetComponent<Renderer>().material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        elementOverlay.GetComponent<Renderer>().material.renderQueue = 3000;
    }
	
	// Update is called once per frame
	void Update () {
        if (type != ElementType.NONE) {
            var ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            var point = ray.origin + (ray.direction * 15);
            point.y = 1.5f;

            if (Input.GetMouseButtonDown(0) && !EventSystem.current.IsPointerOverGameObject()) {
                // Create an element
                var element = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                element.transform.position = point;

                if (type == ElementType.BLUE) {
                    element.GetComponent<Renderer>().material.color = Color.blue;
                    element.tag = "blue";
                } else if (type == ElementType.RED) {
                    element.GetComponent<Renderer>().material.color = Color.red;
                    element.tag = "red";
                }
            } else {
                // Move the overlay element
                elementOverlay.transform.position = point;
            }
        }
    }
}
