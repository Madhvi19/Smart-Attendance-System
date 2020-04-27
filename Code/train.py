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

from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet101, resnet18, resnet50, vgg16


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

BATCH_SIZE = 10
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.__version__)

class FaceRecognizer(data.Dataset):
    def __init__(self, dstype):
        super(FaceRecognizer, self).__init__()
        
        if dstype == "train":
            self.image_path = './train_images/'
            with open(',/train_face_features.pickle', 'rb') as handle:
                self.dictionary = pickle.load(handle)
        elif dstype == "val":
            self.image_path = './val_images/'
            with open('./valid_face_features.pickle', 'rb') as handle:
                self.dictionary = pickle.load(handle)
        
        self.images = sorted(os.listdir(self.image_path))
        self.length = len(self.images)
        
        self.tsfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))
        ])
    def __getitem__(self, index):
        img_name = self.images[index]
        img_label = int(img_name.split('_')[0])
        img_path = os.path.join(self.image_path, img_name)
        img = Image.open(img_path).convert('RGB')
        h, w = img.size
        pose = self.dictionary[img_name]
        posex = np.reshape(stats.zscore((np.transpose(pose)[0])/h), (1,-1))
        posey = np.reshape(stats.zscore((np.transpose(pose)[1])/w), (1,-1))
        pose = np.reshape((np.transpose(np.concatenate((posex, posey)))), (-1))
        img = self.tsfm(img)
        pose = torch.from_numpy(pose).float()
        return img, img_label, pose

    
    def __len__(self):
        return self.length

train_dset = FaceRecognizer(dstype="train")
train_loader = DataLoader(dataset=train_dset, batch_size = BATCH_SIZE,shuffle = True, num_workers = 1) 
valid_dset = FaceRecognizer(dstype="val")
valid_loader = DataLoader(dataset=valid_dset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 1)


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

        
        return x

model = FaceRecognizerModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
softmax = nn.Softmax(dim=1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


criterion = nn.CrossEntropyLoss().to(device)


def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    correct = 0
    incorrect = torch.zeros([BATCH_SIZE, 11])
    incorrect_count = 0
    for i, batch in enumerate(iterator):
        src = batch[0].to(device)
        trg = batch[1].to(device)
        pose = batch[2].to(device)
        pose = pose.view(pose.size()[0], -1)
        optimizer.zero_grad()
        model = model.to(device)
        output = model(src, pose)
        soft = softmax(output)
        for j in range(output.size()[0]):
            ind = torch.argmax(soft[j])
            if ind == trg[j]:
                correct+=1
            else:
                incorrect_count += 1
                incorrect[j][ind] +=1
        loss = criterion(output, trg)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    acc = correct/810*100
    return epoch_loss / len(iterator), acc


def validate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    incorrect = torch.zeros([BATCH_SIZE, 11])
    incorrect_count = 0
    for i, batch in enumerate(iterator):
        src = batch[0].to(device)
        trg = batch[1].to(device)
        pose = batch[2].to(device)
        pose = pose.view(pose.size()[0], -1)
        model = model.to(device)
        output = model(src, pose)
        soft = softmax(output) 
        for j in range(output.size()[0]):
            ind = torch.argmax(soft[j])
            if ind == trg[j]:
                correct+=1
            else:
                incorrect_count += 1
                incorrect[j][ind] +=1
        loss = criterion(output, trg)
        epoch_loss += loss.item()
    acc = float(correct/110)*100
    return epoch_loss / len(iterator), acc


num_epochs = 30
best_acc = 0
for epoch in range(num_epochs):
    print('------------------------------------------------------')
    print('Epoch:', epoch)
    start_time = time.time()
    epoch_train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    epoch_val_loss, val_acc = validate(model, valid_loader, criterion, device)
    scheduler.step()
    print('Training Loss:', epoch_train_loss, 'Training Accuracy:', train_acc)
    print('Validation Loss:', epoch_val_loss, 'Validation Accuracy:', val_acc)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), './FaceRecognizerModel.ckpt')
    print('Best Validation Accuracy:', best_acc)
    

