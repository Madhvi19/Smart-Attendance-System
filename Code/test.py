import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import glob
import pickle
import numpy as np
from PIL import Image
import scipy.io
import time
from scipy import stats
from matplotlib.pyplot import imshow, figure
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet101, resnet18, resnet50, vgg16

from collections import OrderedDict
import numpy as np
import cv2
import argparse
import dlib
import imutils
import os
import pickle



os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.__version__)


class FaceRecognizerModel(nn.Module):
    def __init__(self):
        super(FaceRecognizerModel, self).__init__()
        
        self.resnet = resnet101(pretrained=True)
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.myfc1 = nn.Linear(1136, 512)
        self.myfc2 = nn.Linear(512, 256)
        self.myfc3 = nn.Linear(256, 128)
        self.myfc4 = nn.Linear(128, 64)
        self.myfc5 = nn.Linear(64, 11)

    def forward(self, x, pose):
        x = self.relu(self.resnet(x))
        x = torch.cat([x,pose], dim=1)
        x = self.relu(self.myfc1(x))
        x = self.relu(self.myfc2(x))
        x = self.relu(self.myfc3(x))
        x = self.relu(self.myfc4(x))
        x = self.myfc5(x)
        return x

        
model = FaceRecognizerModel().to(device)
model.load_state_dict(torch.load('./himansh/FaceRecognizerModel.ckpt'))
model.eval()
softmax = nn.Softmax(dim=1)
criterion = nn.CrossEntropyLoss().to(device)


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

os.system("python yoloface/yoloface.py --image "+args["image"]+" --output-dir ./output_bbox")


class_dict = {
    0: 'Brad Pitt',
    1: 'Channing Tatum',
    2: 'Ellen DeGeneres',
    3: 'Jennifer Lawerence',
    4: 'Bradley Cooper',
    5: 'Julia Roberts',
    6: 'Kevin Spacey',
    7: 'lupita nyong"o',
    8: 'Meryl Streep',
    9: 'Miley Cyus',
    10: 'Zac Effron'
}


facial_features_cordinates = {}

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 35)),
    ("Jaw", (0, 17))
])

# construct the argument parser and parse the arguments


def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts

        # check if are supposed to draw the jawline
        if name == "Jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    
    return output

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

for j in sorted(os.listdir('./final_bbox/')):
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread('./final_bbox/'+j)
    image = imutils.resize(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        print(j)
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = shape_to_numpy_array(shape)
#         print(j)
        if os.path.exists('test_face_features.pickle'):
            with open('test_face_features.pickle', 'rb') as handle:
                b = pickle.load(handle)
            b[j] = shape
            with open('test_face_features.pickle', 'wb') as handle:
                pickle.dump(b, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            dic = {}
            dic[j] = shape
            with open('test_face_features.pickle', 'wb') as handle:
                pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        output = visualize_facial_landmarks(image, shape)
        cv2.imwrite('./face_feature_visual/'+j, output)
        cv2.waitKey(0)


with open('test_face_features.pickle', 'rb') as handle:
    dic = pickle.load(handle)


tsfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))
        ])

for i in os.listdir('./final_bbox/'):
    img = Image.open('./final_bbox/'+i).convert('RGB')
    trg = torch.tensor([int(i.split('_')[0])]).to(device)
    h, w = img.size
    pil_im = Image.open('./final_bbox/'+i, 'r')
    figure()
    imshow(np.asarray(pil_im))  
    try:
        pose = dic[i]
    except:
        continue
    posex = np.reshape(stats.zscore((np.transpose(pose)[0])/h), (1,-1))
    posey = np.reshape(stats.zscore((np.transpose(pose)[1])/w), (1,-1))
    pose = np.reshape((np.transpose(np.concatenate((posex, posey)))), (1,-1))
    img = torch.unsqueeze(tsfm(img), dim=0).to(device)
    pose = torch.from_numpy(pose).float().to(device)
    output = model(img, pose)
    loss = criterion(output, trg)
    print(loss)
    soft = softmax(output)
    ind = torch.argmax(soft[0])
    label = class_dict[int(ind)]
    gt_label = class_dict[int(i.split('_')[0])]
    print(label, gt_label)

