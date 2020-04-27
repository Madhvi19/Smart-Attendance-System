# SWE_Project
This repo is a project for the course Software Engineering. 
<br/>
Team No: 24
<br/>
Collaborators
- Himansh Sheoran
- Maanik Arora
- Abhishek Trivedi
- Nausheen Khan
- Madhvi Panchal
- Meenu Menon

The main purpose of this project is to develop a Face Recognition Based Automated Student Attendance System.  We will be building a system that will be  able  to  detect  and  recognize  frontal  faces  of  students  in  classroom,  and  then mark their respective attendance.  The system is exclusively designed for marking attendance in various educational institutes like in colleges and schools.  The aim of this project is to eliminate the traditional system of Manual Attendance System which is usually a time consuming process with chances of human error and replace it with Automated Attendance System.

## Training/Validation Steps

- download model-weights for face detector from Code/model-weights.sh
- run python bbox_extract.py
- run python detect_face_features.py --shape-predictor shape_predictor_68_face_landmarks.dat
- run train.py
##### note: config location needs to be changed in Code/yoloface/yoloface.py
##### note: all final training/validation images are stored in final_bbox folder and features are stored in a pickle file

## Testing
- download pretrained model-weights from link https://drive.google.com/open?id=10MGSROvLdKLbCV0586qF0jesFDgnhJit
- run test.py


