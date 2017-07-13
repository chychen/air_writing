using UnityEngine;
using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.IO;
//using Newtonsoft.Json.Linq;

class Program
{
    public string Modelprocess(string json_string, Action<string> callback)
    {
        //StreamReader r = new StreamReader("I would like an icecreem.json");
        //string json = r.ReadToEnd();
        //Console.WriteLine(json);
        //Console.Read();
        System.Net.Sockets.TcpClient clientSocket = new System.Net.Sockets.TcpClient();
        clientSocket.Connect("140.113.210.19", 2001);
        //NetworkStream stream = new NetworkStream(socket);
        //StreamReader sr = new StreamReader(stream);
        //StreamWriter sw = new StreamWriter(stream);
        string hey = "bye";
        NetworkStream serverStream = clientSocket.GetStream();
        //sw.WriteLine("你好伺服器，我是客戶端。"); // 將資料寫入緩衝
        //sw.Flush(); // 刷新緩衝並將資料上傳到伺服器
        byte[] outStream = System.Text.Encoding.ASCII.GetBytes(json_string);
        serverStream.Write(outStream, 0, outStream.Length);
        byte[] heyStream = System.Text.Encoding.ASCII.GetBytes(hey);
        serverStream.Flush();
        serverStream.Write(heyStream, 0, heyStream.Length);
        serverStream.Flush();
        byte[] inStream = new byte[304];
        //serverStream.ReadAsync(inStream, 0, 154000);
        //serverStream.Read(inStream, 0, (int)clientSocket.ReceiveBufferSize);

        serverStream.Read(inStream, 0, 304);
        string _returndata = System.Text.Encoding.UTF8.GetString(inStream);
        callback(_returndata);
        Debug.Log(_returndata);
        clientSocket.Close();
        return _returndata;
    }
        

    //static void Main(string[] args)
    //{
        //StreamReader r = new StreamReader("I would like an icecreem.json");
        //string json = r.ReadToEnd();
        //string r1 = Modelprocess(json);
        //Console.WriteLine("done 1");
        //Console.ReadKey();
        //string r2 = Modelprocess(json);

        //Console.WriteLine("done 2");
        /*

        //Socket socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        StreamReader r = new StreamReader("I would like an icecreem.json");
        string json = r.ReadToEnd();
        Console.WriteLine(json);
        Console.Read();
        System.Net.Sockets.TcpClient clientSocket = new System.Net.Sockets.TcpClient();
        clientSocket.Connect("140.113.210.18", 2001);
        //NetworkStream stream = new NetworkStream(socket);
        //StreamReader sr = new StreamReader(stream);
        //StreamWriter sw = new StreamWriter(stream);
        string hey = "bye";
        NetworkStream serverStream = clientSocket.GetStream();
        //sw.WriteLine("你好伺服器，我是客戶端。"); // 將資料寫入緩衝
        //sw.Flush(); // 刷新緩衝並將資料上傳到伺服器
        byte[] outStream = System.Text.Encoding.ASCII.GetBytes(json);
        serverStream.Write(outStream, 0, outStream.Length);
        byte[] heyStream = System.Text.Encoding.ASCII.GetBytes(hey);
        serverStream.Flush();
        serverStream.Write(heyStream, 0, heyStream.Length);
        serverStream.Flush();
        byte[] inStream = new byte[304];
        //serverStream.ReadAsync(inStream, 0, 154000);
        //serverStream.Read(inStream, 0, (int)clientSocket.ReceiveBufferSize);
 
        serverStream.Read(inStream, 0, 304);
        string _returndata = System.Text.Encoding.UTF8.GetString(inStream);
            
        Console.WriteLine("從伺服器接收的資料： " + _returndata);

        Console.ReadKey();
    }
    */
    //}

}




//using UnityEngine;
//using System.Collections;
//using System;
//using System.Net;
//using System.Net.Sockets;
//using Leap.Unity.DetectionExamples;

//public class Client
//{
//    private static Socket _clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
//    private static byte[] _recieveBuffer = new byte[8142];

//    public static void SetupServerAndSend(object data)
//    {
//        try
//        {
//            _clientSocket.Connect("140.113.210.27", 2001);
//        }
//        catch (SocketException ex)
//        {
//            Debug.Log(ex.Message);
//        }

//        _clientSocket.BeginReceive(_recieveBuffer, 0, _recieveBuffer.Length, SocketFlags.None, new AsyncCallback(ReceiveCallback), null);

//        SocketAsyncEventArgs socketAsyncData = new SocketAsyncEventArgs();
//        string addBye = (string)data + "bye";
//        byte[] outStream = System.Text.Encoding.ASCII.GetBytes(addBye);
//        socketAsyncData.SetBuffer(outStream, 0, addBye.Length);
//        _clientSocket.SendAsync(socketAsyncData);
//    }

//    public static void ReceiveCallback(IAsyncResult AR)
//    {
//        //Check how much bytes are recieved and call EndRecieve to finalize handshake
//        int recieved = _clientSocket.EndReceive(AR);

//        if (recieved <= 0)
//            return;

//        //Copy the recieved data into new buffer , to avoid null bytes
//        byte[] recData = new byte[recieved];
//        Buffer.BlockCopy(_recieveBuffer, 0, recData, 0, recieved);

//        //Process data here the way you want , all your bytes will be stored in recData
//        string _returndata = System.Text.Encoding.UTF8.GetString(recData);
//        PinchDraw.setText(_returndata);

//        //Start receiving again
//        _clientSocket.BeginReceive(_recieveBuffer, 0, _recieveBuffer.Length, SocketFlags.None, new AsyncCallback(ReceiveCallback), null);
//    }
//}