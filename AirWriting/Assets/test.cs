using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Text;
using Leap;
using Leap.Unity;

public class test : MonoBehaviour {

	Controller controller;
	LeapServiceProvider leapProvider;
	int flg = 0;
	bool isInit = false;
	public GameObject prefab;

	// Use this for initialization
	void Start () {
		controller = new Controller ();

		leapProvider = FindObjectOfType<LeapServiceProvider> ();

	}



	// Update is called once per frame
	void Update () {


		if (Input.GetKey ("up")) {
			flg = 1;

		}

		if (Input.GetKey ("down")) {
			flg = 0;
		}


		Frame Lframe = controller.Frame (); // get frame for leap coordinate
		Frame frame = leapProvider.CurrentFrame; // get frame for unity coordinate
		if (frame.Hands.Count > 0 && flg == 1) {
			List<Hand> Lhands = Lframe.Hands;
			List<Hand> hands = frame.Hands;
			Hand LFHand = Lhands [0];
			Hand firstHand = hands [0];


			List<Finger> Lfingers = LFHand.Fingers;
			List<Finger> fingers = firstHand.Fingers;

			// fingers[1] is index
			Vector locus      = fingers [1].TipPosition;

			Instantiate (prefab, locus.ToVector3(), Quaternion.identity);



			Finger index = Lfingers [1];
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


			string path = @"C:\Users\ec131b\Desktop\Datas\onDesk\z\06";
			if (!File.Exists (path)) {
				string tmp = ">>\n";
				File.WriteAllText (path,tmp, Encoding.UTF8);
			}
		
			string qq = locus + "\n";
			File.AppendAllText (path,qq,Encoding.UTF8);




		}

	}


}

