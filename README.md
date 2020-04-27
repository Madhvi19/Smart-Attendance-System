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


# Face Detection
 This detects faces in all the images present in # train folder which contain all the images for training.It saves the output ,which is bounding boxes detected in the original images,in folder output_bbox and the final cropped out images of respective bounding boxes in the folder final_bbox
 
# Feature Extraction
This uses images present in output_bbox directory as an input extract the feature vectors of each image using opencv and dlib. For each image we get a feature vector of size (68,2), these are then stored in the disk after pickling as face_feature.pickle

# Face Recognition
This uses the extracted feature vectors pickled in face_feature.pickle as an input. We have used Resnet state of the art model to recognize the faces.

# Training steps
Training steps (same to be done for validation):
1) make a train folder with all training image
2) run python bbox_extract.py
3) run python detect_face_features.py --shape-predictor shape_predictor_68_face_landmarks.dat

note: all final training images are stored in final_bbox and features are stored in a pickle file

4) run train.py

# Testing
1) download pretrained model-weights from https://drive.google.com/open?id=10MGSROvLdKLbCV0586qF0jesFDgnhJit
2) run test.py
