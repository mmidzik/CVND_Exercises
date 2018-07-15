## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        #Initial tensor size: torch.Size([1, 224, 224]) torch.Size([68, 2])
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        #initially try architectyre from model
        # 1 input image channel (grayscale), 32 output channels/feature maps, 
        #5x5 square convolution kernel
        
        
        ## output size = (W-F)/S +1 = (224-4)/1 +1 = 221
        #output tensor = (32,221,221)
        #after maxpooling = (32,110,110)
        self.conv1 = nn.Conv2d(1, 32, 4)
        #add ReLu activation
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.1)
        
        #32 inputs, 64 output
        #output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        #output tensor = (64,108,108)
        #after maxpooling = (64,54,54)
        self.conv2 = nn.Conv2d(32,64,3)
        #RELU, Maxpool
        self.drop2 = nn.Dropout(p=0.2)
        
        #64 inputs, 128 outputs
        #output size = (W-F)/S +1 = (53-2)/1 +1 = 52
        #output tensor = (128,52,52)
        #after maxpooling = (128,26,26)
        self.conv3 = nn.Conv2d(64,128,2)
        #RELU,Maxpool
        self.drop3 = nn.Dropout(p=0.3)
        
        #128 inputs, 256 outputs
        #output size = (W-F)/S +1 = (26-1)/1 +1 = 26
        #output tensor = (256,26,26)
        #after maxpooling = (256,13,13)                    
        self.conv4 = nn.Conv2d(128,256,1)
        #RELU,Maxpool
        self.drop4 = nn.Dropout(p=0.4)
        
        #FLATTEN: flatten to 256*13*13
        
        #Note that the paper's dense layers are intiaited w/ Glorot uniform initialization weights
        self.fc1 = nn.Linear(256*13*13, 1000)
        #RELU
        self.drop5 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1000, 1000)
        #RELU
        self.drop6 = nn.Dropout(p=0.6)
        self.fc3 = nn.Linear(1000, 136)
        
        
                                
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        #flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        x = F.relu(self.fc2(x))
        x = self.drop6(x)
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
