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

		//key for start record
		if (Input.GetKey ("up")) {
			flg = 1;

		}

		//key for stop record
		if (Input.GetKey ("down")) {
			flg = 0;
		}


		Frame Lframe = controller.Frame ();      // get frame for leap coordinate
		Frame frame = leapProvider.CurrentFrame; // get frame for unity coordinate

		// if there is hand and "up" is pressed, then start to record
		if (frame.Hands.Count > 0 && flg == 1) {
			
			List<Hand> Lhands = Lframe.Hands; 
			List<Hand> hands = frame.Hands;   

			Hand LFHand = Lhands [0];    // fitst hand for leap motion
			Hand firstHand = hands [0];  // first hand for unity 


			List<Finger> Lfingers = LFHand.Fingers;   // finger for leap motion
			List<Finger> fingers = firstHand.Fingers; // finger for unity

			// fingers[1] is index
			// locus is the position in the unity
			Vector locus      = fingers [1].TipPosition;

			// to put prefab in the scene
			Instantiate (prefab, locus.ToVector3(), Quaternion.identity);


			// finger for leap motion
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

			// start to write data
			string path = @"C:\Users\ec131b\Desktop\Datas\onDesk\z\06";

			// check if file is existed
			if (!File.Exists (path)) {
				string tmp = ">>\n";
				File.WriteAllText (path,tmp, Encoding.UTF8);
			}

			// write data in the file end
			string qq = locus + "\n";
			File.AppendAllText (path,qq,Encoding.UTF8);




		}

	}


}

