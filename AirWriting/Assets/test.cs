using System.Collections;
using System.Collections.Generic;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using UnityEngine;
using Leap;
using Leap.Unity;

public class test : MonoBehaviour {

	public GameObject pre;
	Controller controller;
	int flg = 0;


	// Use this for initialization
	void Start () {
		controller = new Controller ();


	}

	

	// Update is called once per frame
	void Update () {


		if (Input.GetKey ("up")) {
			flg = 1;

		}

		if (Input.GetKey ("down")) {
			flg = 0;
		}
		Frame frame = controller.Frame (); // controller is a Controller object
		if (frame.Hands.Count > 0 && flg == 1) {
			List<Hand> hands = frame.Hands;
			Hand firstHand = hands [0];
		
			List<Finger> fingers = firstHand.Fingers;

			// fingers[1] is index
			Finger index      = fingers [1];
			Vector direction  = index.Direction;
			Vector stabilized = index.StabilizedTipPosition;
			float  lifetime   = index.TimeVisible;
			Vector3 position   = index.TipPosition.ToVector3() ;
			Vector velocity   = index.TipVelocity;
			float  speed      = velocity.Magnitude;

			print ("direction is " + direction);
			print ("stabilized is " + stabilized);
			print ("lifetime is " + lifetime);
			print ("position is " + position);
			print ("velocity is " + velocity);
			print ("speed is " + speed);

			var o = new GameObject ();

			Instantiate (pre, position / 100, Quaternion.identity);
		}
		
	}


}


namespace Leap {
	public class LeapUtil {
		public static Vector3 ToPositionVector3 ( Vector pos) {
			return new Vector3 (pos.x, pos.y, -pos.z);
		}
		public static Vector3 ToVector3 (Vector v) {
			return new Vector3 (v.x,v.y,v.z);	
		}
		public static void LookAt(Transform t, Vector n) {
			t.LookAt (t.position + ToPositionVector3(n), Vector3.forward );
		}
	}
}