import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
from func.trans import TransformerLayer
from func.attention import CABlock,SEBlock,CBAM
import torch.nn.functional as F
class CNNModel_1(nn.Module):
    def __init__(self, stride=1):
        super(CNNModel_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        #x = self.fc(x)
        return x
class CNNModel_2(nn.Module):
    def __init__(self,embed_dim=64, num_heads=4, ff_dim=64, dropout_rate=0.1):
        super(CNNModel_2, self).__init__()
        self.conv5 = nn.Conv2d(64,64,kernel_size=5,padding=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, padding=1)       
        self.bn4 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(F.relu(self.bn5(self.conv5(x))))#128
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))       
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        return x
class CombinedModel(nn.Module):
    def __init__(self,attensionChannel = 128): #16
        super(CombinedModel, self).__init__()
        self.cnn_model_1 = CNNModel_1()
        self.cnn_model_2 = CNNModel_2() 
        self.cov_pool = nn.Sequential(
            nn.Conv2d(64,16,kernel_size=1,stride=2),
            nn.MaxPool2d(2, stride=2),# 32×32 → 16×16（浅层保留细节）
            nn.Conv2d(16,16,kernel_size=1,stride=2),
            nn.MaxPool2d(2, stride=2) # 32×32 → 16×16（浅层保留细节）
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16*2*2, 256),  
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 5)  # 
            )
    def forward(self, x_mfcc):
        cnn_output_1 = self.cnn_model_1(x_mfcc ) # (16,128,32,32 )

        cnn_output_1_1 = self.cov_pool(cnn_output_1)
        cnn_output_2 = self.cnn_model_2(cnn_output_1 )  #(32,16,2,2)
        #fused = torch.concat([cnn_output_1_1,cnn_output_2],dim=1)
        fused = cnn_output_1_1 + cnn_output_2 
        cnn_outputs =   fused
        cnn_outputs = self.fc(cnn_outputs)
        combined_output = cnn_outputs  
        return combined_output