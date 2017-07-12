using UnityEngine;
using System;
using System.Threading;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Leap.Unity;
using System.Collections;
using System.Net;
using System.Net.Sockets;

namespace Leap.Unity.DetectionExamples {

public class PinchDraw : MonoBehaviour {
    [Tooltip("Each pinch detector can draw one line at a time.")]
    [SerializeField]
    private PinchDetector[] _pinchDetectors;

    [SerializeField]
    private Material _material;

    [SerializeField]
    private Color _drawColor = Color.white;

    [SerializeField]
    private float _smoothingDelay = 0.01f;

    [SerializeField]
    private float _drawRadius = 0.002f;

    [SerializeField]
    private int _drawResolution = 8;

    [SerializeField]
    private float _minSegmentLength = 0.005f;

    private DrawState[] _drawStates;

    public Color DrawColor {
        get {
        return _drawColor;
        }
        set {
        _drawColor = value;
        }
    }

    public float DrawRadius {
        get {
        return _drawRadius;
        }
        set {
        _drawRadius = value;
        }
    }
        string fffffffffffffffffff = "";
    int flg = 0, flg1 = 0;
    int wordID = 0, tag = 0;
    public int uid = 4321;
    //public string path = @"C:\Users\ec131b\Desktop\air_writing_datas";
    public string Name = "XDEX";
    public string dataset = @"";

    GameObject target;
    List<string> wordList;
    StringBuilder SB;
    LeapServiceProvider leapProvider;
    Frame frame;

    void OnValidate() {
      _drawRadius = Mathf.Max(0, _drawRadius);
      _drawResolution = Mathf.Clamp(_drawResolution, 3, 24);
      _minSegmentLength = Mathf.Max(0, _minSegmentLength);
    }

    void Awake() {
      if (_pinchDetectors.Length == 0) {
        Debug.LogWarning("No pinch detectors were specified!  PinchDraw can not draw any lines without PinchDetectors.");
      }
    }

    void Start() {
        _drawStates = new DrawState[_pinchDetectors.Length];
        for (int i = 0; i < _pinchDetectors.Length; i++) {
        _drawStates[i] = new DrawState(this);
        }

        wordList = new List<string>();
        SB = new StringBuilder();
        target = GameObject.FindGameObjectWithTag("mytext");
        //buildFiles();

            target.GetComponent<TextMesh>().text = "Welcome To Air Writing DEMO";
            leapProvider = FindObjectOfType<LeapServiceProvider>();
        this.frame = leapProvider.CurrentFrame;
    }

    void Update() {
            if (fffffffffffffffffff!="")
            {
                print(fffffffffffffffffff);
                target.GetComponent<TextMesh>().text = fffffffffffffffffff;
                fffffffffffffffffff = "";
            }
        if (this.flg == 1) {
            for (int i = 0; i < _pinchDetectors.Length; i++) {

                var detector = _pinchDetectors[i];
                var drawState = _drawStates[i];

                if (detector.DidStartHold) {
                    drawState.BeginNewLine();
                }

                if (detector.DidRelease) {
                    drawState.FinishLine();
                    this.tag = this.tag + 1;
                }

                if (detector.IsHolding) {

                    drawState.UpdateLine(detector.Position);
                    //print(detector.Position.x + " " + detector.Position.y + " " + detector.Position.z);
                }
            }   
        }
        
    }
       
        void clearall() {

            SB = new StringBuilder();
            var draws = GameObject.FindGameObjectsWithTag("XD");
        for (int i = 0; i < draws.Length; i++)
            Destroy(draws[i]);
    }

        public void myCallback(string str)
        {
            fffffffffffffffffff = str;
        }

        void FixedUpdate() {

        if ( this.flg1 < 2 && Input.GetKey(KeyCode.Space)) {
            this.flg1 = 0;
            clearall();
        }

        //if (this.flg1 == 0 && Input.GetKey(KeyCode.A)) {
        //    this.flg1 = 1;
        //    if (wordID > 0)
        //        this.wordID = this.wordID - 1;
        //    target.GetComponent<TextMesh>().text = this.wordID+ ". " + wordList[wordID];
        //    clearall();
        //    //print("In left " + wordID + " " + this.wordList[wordID]);
        //}

        //if (this.flg1 == 0 && Input.GetKey(KeyCode.D)) {
        //    this.flg1 = 1;
        //    if (this.wordID < this.wordList.Count-1)
        //        this.wordID = this.wordID + 1;
        //    target.GetComponent<TextMesh>().text = this.wordID + ". " + wordList[wordID];
        //    clearall();
        //    //print("In right " + wordID + " " + this.wordList[wordID]);
        //}

        if (this.flg == 0 && Input.GetKey(KeyCode.W)) {
            this.flg = 1;
            this.flg1 = 2;
            this.tag = 0;
            print("start to write");
            SB.Length = 0;
            SB.AppendLine("{");
            //SB.AppendLine("\"word\" : \"" + this.wordList[wordID] + "\",");

            SB.AppendLine("\"word\" : \"" + "10" + "\",");
            SB.AppendLine("\"fps\" : " + (int)Math.Ceiling(1.0 / Time.fixedDeltaTime) + ",");
            SB.AppendLine("\"name\" : \"" + this.Name + "\",");
            SB.AppendLine("\"id\" : " + this.uid + ",");
            SB.AppendLine("\"data\" : [");

        }
        
        if (this.flg == 1 && Input.GetKey(KeyCode.S)) {
            this.flg = 0;
            this.flg1 = 0;
            print("Stop to write and save data");
            SB = new StringBuilder(SB.ToString().TrimEnd('\n'));
            SB = new StringBuilder(SB.ToString().TrimEnd(','));
            SB.Append("\n");
            SB.AppendLine("\t]");
            SB.AppendLine("}");

                // TODO
            var temp = new Program();
                string value="";
            Thread socketThread = new Thread(() =>
            {
                value = temp.Modelprocess(SB.ToString(), myCallback); // Publish the return value
            });
            socketThread.Start();
            //    print(socketThread.IsAlive);
            //    socketThread.Join(10000);
            //    print(socketThread.IsAlive);
            //target.GetComponent<TextMesh>().text = value;

                //string ptr = this.path + @"\" + this.uid + @"\" + this.wordList[this.wordID] + ".json";

            //if (!File.Exists(ptr)) {
            //    Debug.LogError("Oops!! " + ptr + " is not exist!!!!!");
            //}
            //else {
            //    File.WriteAllText(ptr, SB.ToString(), Encoding.UTF8);

            //}

        }

        if (this.flg == 1) {
            
            
            for (int i = 0; i < _pinchDetectors.Length; i++) {
                var detector = _pinchDetectors[i];
                var drawState = _drawStates[i];

                
                if (detector.IsHolding) {
                    
                    
                    List<Finger> fingers = frame.Hands[0].Fingers;
                    
                    Vector3 tipPosition = fingers[1].TipPosition.ToVector3();
                    Vector3 tipDirection = fingers[1].Direction.ToVector3();
                    Vector3 tipVelocity = fingers[1].TipVelocity.ToVector3();
                    
                    var epochStart = new System.DateTime(1970, 1, 1, 8, 0, 0, System.DateTimeKind.Utc);
                    var timestamp = (System.DateTime.UtcNow - epochStart).TotalSeconds;
                    /*
                    print("detector ");
                    print(detector.Position.x + ", " + detector.Position.y + ", " + detector.Position.z);
                    print("tipPosition");
                    print(tipPosition.x + ", " + tipPosition.y + ", " + tipPosition.z);
                    */
                    string qq = "\t{ \"face\" : [" + Camera.main.transform.forward.x + ", " + Camera.main.transform.forward.y + ", " + Camera.main.transform.forward.z + "], " +
                                    "\"head\" : [" + Camera.main.transform.position.x + ", " + Camera.main.transform.position.y + ", " + Camera.main.transform.position.z +"], " +
                                    "\"time\" : " + timestamp +
                                  ", \"position\" : [" + detector.Position.x + ", " + detector.Position.y + ", " + detector.Position.z + "] " +
                                  ", \"direction\" : [" + tipDirection.x + ", " + tipDirection.y + ", " + tipDirection.z + "] " +
                                  ", \"velocity\" : [" + tipVelocity.x + ", " + tipVelocity.y + ", " + tipVelocity.z + "] " +
                                  ", \"tag\" : " + this.tag + 
                                  "},\n";
                    
                    SB.Append(qq);
                    
                }
                
            }
        }
        
        
    }

    //void buildFiles() {
    //    string dir = this.path + @"\" + this.uid;
    //    if (!Directory.Exists(dir)) {
    //        Directory.CreateDirectory(dir);
    //        print("Create directory " + dir);
    //    }
    //    else {
    //        print(dir + " is exist.");
    //    }

    //    string wordset = path + @"\" + this.dataset + @".txt";
        
    //    if (!File.Exists(wordset)) {
    //        Debug.LogError("oops!! there is no dataset ");
    //        return;
    //    }

    //    StreamReader SR = new StreamReader(wordset, Encoding.Default);
    //    string ptr = SR.ReadLine();

    //    while (ptr != null) {
    //        wordList.Add(ptr);
    //        print(ptr);
    //        ptr = SR.ReadLine();
    //    }
    //    SR.Close();

    //    for (int i = 0; i < wordList.Count; ++i) {
    //        ptr = dir + @"\" + wordList[i] + ".json";
    //        if (!File.Exists(ptr)) {
    //            File.Create(ptr);
    //        }
    //    }

    //    print("finish build files");
    //}

    private class DrawState {
      private List<Vector3> _vertices = new List<Vector3>();
      private List<int> _tris = new List<int>();
      private List<Vector2> _uvs = new List<Vector2>();
      private List<Color> _colors = new List<Color>();

      private PinchDraw _parent;

      private int _rings = 0;

      private Vector3 _prevRing0 = Vector3.zero;
      private Vector3 _prevRing1 = Vector3.zero;

      private Vector3 _prevNormal0 = Vector3.zero;

      private Mesh _mesh;
      private SmoothedVector3 _smoothedPosition;

      public DrawState(PinchDraw parent) {
        _parent = parent;

        _smoothedPosition = new SmoothedVector3();
        _smoothedPosition.delay = parent._smoothingDelay;
        _smoothedPosition.reset = true;
      }

      public GameObject BeginNewLine() {
        _rings = 0;
        _vertices.Clear();
        _tris.Clear();
        _uvs.Clear();
        _colors.Clear();

        _smoothedPosition.reset = true;

        _mesh = new Mesh();
        _mesh.name = "Line Mesh";
        _mesh.MarkDynamic();

        GameObject lineObj = new GameObject("Line Object");
        lineObj.tag = "XD";
        lineObj.transform.position = Vector3.zero;
        lineObj.transform.rotation = Quaternion.identity;
        lineObj.transform.localScale = Vector3.one;
        lineObj.AddComponent<MeshFilter>().mesh = _mesh;
        lineObj.AddComponent<MeshRenderer>().sharedMaterial = _parent._material;

        return lineObj;
      }

      public void UpdateLine(Vector3 position) {
        _smoothedPosition.Update(position, Time.deltaTime);

        bool shouldAdd = false;

        shouldAdd |= _vertices.Count == 0;
        shouldAdd |= Vector3.Distance(_prevRing0, _smoothedPosition.value) >= _parent._minSegmentLength;

        if (shouldAdd) {
          addRing(_smoothedPosition.value);
          updateMesh();
        }
      }

      public void FinishLine() {
        _mesh.UploadMeshData(true);
      }

      private void updateMesh() {
        _mesh.SetVertices(_vertices);
        _mesh.SetColors(_colors);
        _mesh.SetUVs(0, _uvs);
        _mesh.SetIndices(_tris.ToArray(), MeshTopology.Triangles, 0);
        _mesh.RecalculateBounds();
        _mesh.RecalculateNormals();
      }

      private void addRing(Vector3 ringPosition) {
        _rings++;

        if (_rings == 1) {
          addVertexRing();
          addVertexRing();
          addTriSegment();
        }

        addVertexRing();
        addTriSegment();

        Vector3 ringNormal = Vector3.zero;
        if (_rings == 2) {
          Vector3 direction = ringPosition - _prevRing0;
          float angleToUp = Vector3.Angle(direction, Vector3.up);

          if (angleToUp < 10 || angleToUp > 170) {
            ringNormal = Vector3.Cross(direction, Vector3.right);
          } else {
            ringNormal = Vector3.Cross(direction, Vector3.up);
          }

          ringNormal = ringNormal.normalized;

          _prevNormal0 = ringNormal;
        } else if (_rings > 2) {
          Vector3 prevPerp = Vector3.Cross(_prevRing0 - _prevRing1, _prevNormal0);
          ringNormal = Vector3.Cross(prevPerp, ringPosition - _prevRing0).normalized;
        }

        if (_rings == 2) {
          updateRingVerts(0,
                          _prevRing0,
                          ringPosition - _prevRing1,
                          _prevNormal0,
                          0);
        }

        if (_rings >= 2) {
          updateRingVerts(_vertices.Count - _parent._drawResolution,
                          ringPosition,
                          ringPosition - _prevRing0,
                          ringNormal,
                          0);
          updateRingVerts(_vertices.Count - _parent._drawResolution * 2,
                          ringPosition,
                          ringPosition - _prevRing0,
                          ringNormal,
                          1);
          updateRingVerts(_vertices.Count - _parent._drawResolution * 3,
                          _prevRing0,
                          ringPosition - _prevRing1,
                          _prevNormal0,
                          1);
        }

        _prevRing1 = _prevRing0;
        _prevRing0 = ringPosition;

        _prevNormal0 = ringNormal;
      }

      private void addVertexRing() {
        for (int i = 0; i < _parent._drawResolution; i++) {
          _vertices.Add(Vector3.zero);  //Dummy vertex, is updated later
          _uvs.Add(new Vector2(i / (_parent._drawResolution - 1.0f), 0));
          _colors.Add(_parent._drawColor);
        }
      }

      //Connects the most recently added vertex ring to the one before it
      private void addTriSegment() {
        for (int i = 0; i < _parent._drawResolution; i++) {
          int i0 = _vertices.Count - 1 - i;
          int i1 = _vertices.Count - 1 - ((i + 1) % _parent._drawResolution);

          _tris.Add(i0);
          _tris.Add(i1 - _parent._drawResolution);
          _tris.Add(i0 - _parent._drawResolution);

          _tris.Add(i0);
          _tris.Add(i1);
          _tris.Add(i1 - _parent._drawResolution);
        }
      }

      private void updateRingVerts(int offset, Vector3 ringPosition, Vector3 direction, Vector3 normal, float radiusScale) {
        direction = direction.normalized;
        normal = normal.normalized;

        for (int i = 0; i < _parent._drawResolution; i++) {
          float angle = 360.0f * (i / (float)(_parent._drawResolution));
          Quaternion rotator = Quaternion.AngleAxis(angle, direction);
          Vector3 ringSpoke = rotator * normal * _parent._drawRadius * radiusScale;
          _vertices[offset + i] = ringPosition + ringSpoke;
        }
      }
    }
  }
}
