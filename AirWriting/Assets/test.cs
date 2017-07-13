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
	int isCaps = 0;
	bool isInit = false;
	string path = "";
	string word = "";
	string name = "陳傑宇";
	int id = 3;
	public GameObject prefab;
	StringBuilder sb;

	List< string > wordList;
	int wordID = 0;

	// Use this for initialization
	void Start () {
		controller = new Controller ();

		leapProvider = FindObjectOfType<LeapServiceProvider> ();
		wordList = new List<string> ();
		wordID = -1;
		buildFiles ();
	}

	void buildFiles() {

		var path = "C:\\Users\\ec131b\\Desktop\\air_writing_datas\\" + id;

		// Create Directory for this ID
		if (!Directory.Exists (path)) {
			Directory.CreateDirectory (path);
			print (">> Create Directory " + id);
		} else {
			print ("Directory " + id + " is exist");
		}

		// Read word list
		StreamReader SR = new StreamReader( "C:\\Users\\ec131b\\Desktop\\air_writing_datas\\dataset", Encoding.Default );
		for (string tmp = SR.ReadLine (); tmp != null ; tmp = SR.ReadLine() ) {
			var filePath = path + "\\" + tmp + ".json";
			if (!File.Exists (filePath)) {
				File.Create (filePath);
				wordList.Add (tmp);
				print ("Create File " + filePath);
			} else {
				Debug.Log ("oops!!" + filePath + "is exist QQ!!");
			}

		}
	}



	// Update is called once per frame
	void FixedUpdate () {

		//Delete All thing
		if ( Input.GetKey( KeyCode.Space) ){
			var draws = GameObject.FindGameObjectsWithTag("XD");
			for( int i = 0 ; i < draws.Length ; i++ )
				Destroy(draws[i]);

		}

		//If any key put down, path will update 

		if ( Input.GetKey( KeyCode.UpArrow ) && flg == 0 ) {
			flg = 1;
			print("Is started now? " + flg);
			sb = new StringBuilder();
			sb.AppendLine("{");
			sb.AppendLine("\"word\" : " + this.word + ",");
			sb.AppendLine("\"fps\" : " + (int)Math.Ceiling( 1.0 / Time.fixedDeltaTime) + ",");
			sb.AppendLine("\"name\" : " + this.name + ",");
			sb.AppendLine("\"id\" : " + this.id + ",");
			sb.AppendLine("\"data\" : [");

		}

		if ( Input.GetKey( KeyCode.DownArrow ) && flg == 1) {
			flg = 0;
			print("Is started now? " + flg);
			sb = new StringBuilder(sb.ToString().TrimEnd('\n'));
			sb = new StringBuilder(sb.ToString().TrimEnd(','));
			sb.Append("\n");
			sb.AppendLine("\t]");
			sb.AppendLine("}");
			this.wordID = this.wordID + 1;
			getPath();
			if (!File.Exists (this.path)) {
				Debug.LogError (this.path + " is not exist");
			}
			File.AppendAllText(path, sb.ToString() , Encoding.UTF8);
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
			Vector tipPositon  = fingers [1].TipPosition;
			Vector tipDirection = fingers [1].Direction;
			Vector tipVelocity  = fingers [1].TipVelocity;

			//print(LFHand.PinchDistance);

			// to put prefab in the scene
			/*
			if ( LFHand.PinchDistance < 20.0f ) {
				Instantiate (prefab, locus.ToVector3(), Quaternion.identity);
			}
             */
			Instantiate(prefab, tipPositon.ToVector3(), Quaternion.identity);


			// finger for leap motion
			/*
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
			//string path = @"C:\Users\ec131b\Desktop\Datas\onDesk\z\06";
			//string path = @"C:\Users\ec131b\Desktop\air_writing_datas\";

			// check if file is existed
			if (!File.Exists (path)) {
				string tmp = ">>\n";
				File.WriteAllText (path,tmp, Encoding.UTF8);
			}
			*/

			var epochStart = new System.DateTime(1970, 1, 1, 8, 0, 0, System.DateTimeKind.Utc);
			var timestamp = (System.DateTime.UtcNow - epochStart).TotalSeconds;

			// write data in the file end
			string qq = "\t{ \"face\" : [" + Camera.main.transform.forward.x + ", " + Camera.main.transform.forward.y + ", " + Camera.main.transform.forward.z + "], " + 
				"\"time\" : " + timestamp +
				", \"position\" : [" + tipPositon.x + ", " + tipPositon.y + ", " + tipPositon.z + "] " + 
				", \"direction\" : [" + tipDirection.x + ", " + tipDirection.y + ", " + tipDirection.z + "] " +
				", \"velocity\" : [" + tipVelocity.x + ", " + tipVelocity.y + ", " + tipVelocity.z + "] " + 
				"},\n";

			//File.AppendAllText (path,qq,Encoding.UTF8);
			sb.Append(qq);




		}

	}

	void getPath() {

		this.path = @"C:\Users\ec131b\Desktop\air_writing_datas\" + this.id + "\\" + wordList[this.wordID] + ".json";
		print (this.path);
		/*
		if (Input.GetKey(KeyCode.KeypadPlus ) ){
			print("CapsLock " + this.isCaps);
			this.isCaps = 1;
		}
		
		if (Input.GetKey(KeyCode.KeypadMinus ) ){
			print("CapsLock " + this.isCaps);
			this.isCaps = 0;
		}
		
		foreach( char now in abc ){
			if ( Input.GetKey(now.ToString() ) ){
				
				if ( 'a' <= now && now <= 'z' && this.isCaps == 1 ){
					this.path = @"C:\Users\ec131b\Desktop\air_writing_datas\" + name + "_" + now.ToString() + ".json";
                    this.word = "\"" + now.ToString().ToUpper() + "\"";
				}
				else {
                    this.path = @"C:\Users\ec131b\Desktop\air_writing_datas\" + name + now.ToString() + ".json";
                    this.word = "\"" + now.ToString() + "\"";
                }
                
				print(this.path);
			}
		}
		*/

	}


}

