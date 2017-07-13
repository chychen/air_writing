# On-Line Hand Writing Recognition using BLSTM   
## Introduction
This project is a final project of the 'Deep Learning and Practice' course hosted in NCTU 2017 SUMMER.  
The air_writing project is basically but not totally following the ideals of [A Novel Approach to On-Line Handwriting Recognition Based on Bidirectional Long Short-Term Memory Networks](http://www.cs.toronto.edu/~bonner/courses/2016s/csc321/readings/A%20novel%20approach%20to%20on-line%20handwriting%20recognition%20based%20on%20bidirectional%20long%20short-term%20memory%20networks.pdf) to implement a BLSTM model using tensorflow.




## Requirement:
    tensorflow >= 1.2
    numpy
    scipy
    python 3.5
    xml
    
## Setup
1. Go to http://www.fki.inf.unibe.ch/databases/iam-handwriting-database download the IAM On-Line Handwriting DataBase.
    And store the dataset folders 'ascii' and 'lineStrokes' under air_writing/data/

2. Generate dense tensor input data: data.npy and label.npy.  
```python
python air_writing/recognition/src UltraProcess.py
```
  
3. Generate the dense representation of label(text line): dense.npy
```python
python air_writing/recognition/src read.py
```
 

## Traning on IAM data   
```python
python air_writing/recognition/src air_writing/recognition/src train_blstm.py
```
hyper parameters:   

--data_dir  
--checkpoint_dir   
-- log_dir    
--restore_path   
--batch_size    
--total_epoches   
...(details please refer to air_writing/recognition/src/train_blstm.py)

## Testing on VR data
1. Project and normalize the 3D coordinated VR writing trajectory data and get filename.json
```python
python air_writing/ui_labeling/preprocessing sphere_fitting.py
```
2. Generate input data from filename.json and get VRdataValidation.npy and VRlabelValidation.npy
```python
python air_writing/recognition/src tagProcess.py
```
3. Test  
```python
python air_writing/recognition/src test_blstm.py
```

## Reference
[ [LiBu05-03] Liwicki, M. and Bunke, H.: IAM-OnDB - an On-Line English Sentence Database Acquired from Handwritten Text on a Whiteboard. 8th Intl. Conf. on Document Analysis and Recognition, 2005, Volume 2, pp. 956 - 961 ](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/iam-on-line-handwriting-database#LiBu05-03)   

[A Novel Approach to On-Line Handwriting Recognition Based on Bidirectional Long Short-Term Memory Networks](http://www.cs.toronto.edu/~bonner/courses/2016s/csc321/readings/A%20novel%20approach%20to%20on-line%20handwriting%20recognition%20based%20on%20bidirectional%20long%20short-term%20memory%20networks.pdf) 
## License
MIT License

Copyright (c) [2017] [Chen Chieh Yu, Wen Ze Lai]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
