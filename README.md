# Face-Recognition-with-Own-Data-Set

A Command line tool for Face Detection and Recognition with Own Data Set.

This application is appropriate for face detection with some controlled environment of a limited set of people. The Command line application can capture and store the faces in a directory with the name. Also, it will show the name of the people fetching and comparing from the known data set. It uses the open python package *‘face_recognition’* and *‘opencv-python’*.

The *‘face_recognition’* package is built using dlib's state-of-the-art face recognition built with deep learning.

The *‘opencv-python’* package is a OpenCV packages for Python and provides the features of standard desktop environments (Windows, macOS, almost any GNU/Linux distribution)

## Installation
### Requirements
* Python 3.3+ or Python 2.7
* macOS or Linux (Windows not officially supported, but might work)
### Installation Options:
#### Installing on Mac or Linux
Install the following modules using pip3 for Python 3 (or pip2 for Python 2):
```bash
pip3 install face_recognition

pip3 install opencv-python
```
#### Installing on Windows
Install dlib, face_recognition and opencv-python for OpenCV -
pip install dlib and then pip install face_recognition.
```bash
pip install dlib
pip install face_recognition
pip install opencv-python
```

## Run:
Go to the command window and change the directory in the source location -
```bash
bash$cd <source Location>
```
Run the following command for Python 3 (or python for Python 2):–
```bash
python3 ./FaceRecognition.py
```
After run the application a video window will be open. The default video channel is *'0'*. If the source of the video is deferent then change the line accordingly.
```python
video_capture = cv2.VideoCapture(0)
```
It will create a *'Data'* directory in the current location even if this directory does not exist.

To store a image you can put the image directly into the *'Data'* directory. If the face of this image matched the image name will show in the screen.  

Or

Press 'c' to capture the image and in the command prompt it will ask for the name of the face. Put the name of the image.

To quite the application press 'q'.

## Reference

[Face Recognition in Python](https://github.com/ageitgey/face_recognition/blob/master/README.md)
