import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layer_d0 = nn.Sequential(nn.Conv2d(3, 8, 7, padding=3, stride=1, bias=True),nn.ReLU())

        
        self.layer_d1 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1, stride=2, bias=True),
                                      nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1, stride=1, bias=True))
        self.layer_d1s = nn.Conv2d(8, 16, 1, stride=2)
        self.relu_d1 = nn.ReLU()

        
        self.layer_d2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1, stride=2, bias=True),
                                      nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1, stride=1, bias=True))
        self.layer_d2s = nn.Conv2d(16, 32, 1, stride=2)
        self.relu_d2 = nn.ReLU()


        self.layer_d3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=True),
                                      nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1, stride=1, bias=True))
        self.layer_d3s = nn.Conv2d(32, 64, 1, stride=2)
        self.relu_d3 = nn.ReLU()

        
        self.layer_d4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=2, bias=True),
                                      nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1, stride=1, bias=True))
        self.layer_d4s = nn.Conv2d(64, 128, 1, stride=2)
        self.relu_d4 = nn.ReLU()
        
        self.layer200 = nn.AvgPool2d(16,16)

        self.layer201 = nn.Sequential(nn.Linear(128, 11, bias=True))

    def forward(self, input_img):

        output_d0 = self.layer_d0(input_img)

        output_d1 = self.layer_d1(output)
        output_d1s = self.layer_d1s(output)
        output = output_d1 + output_d1s
        output = self.relu_d1(output)

        output_d2 = self.layer_d2(output)
        output_d2s = self.layer_d2s(output)
        output = output_d2 + output_d2s
        output = self.relu_d2(output)

        output_d3 = self.layer_d3(output)
        output_d3s = self.layer_d3s(output)
        output = output_d3 + output_d3s
        output = self.relu_d3(output)

        output_d4 = self.layer_d4(output)
        output_d4s = self.layer_d4s(output)
        output = output_d4 + output_d4s
        output = self.relu_d4(output)
        
        df4  = output
        
        gap_feats = self.layer200(output)
        gap_feats = gap_feats.view(gap_feats.shape[0], -1)
        probs = self.layer201(gap_feats)
        
 
        df4 = torch.cat((df4),1)


        return probs
