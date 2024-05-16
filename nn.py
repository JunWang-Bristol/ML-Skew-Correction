import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers for the regression task (offset time detection)
        self.fc1_regression = nn.Linear(128 * 32 * 32 + 4, 512) # Hmax, Hmin, Bmax, Bmin + image features
        self.fc2_regression = nn.Linear(512, 1)  # Offset time detection

    def forward(self, x, extra_inputs):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Concatenate the extra inputs
        x = torch.cat((x, extra_inputs), dim=1)
        
        # Regression branch
        x_regression = F.relu(self.fc1_regression(x))
        x_regression = self.fc2_regression(x_regression)
        
        return x_regression
    

# %%
    
class CNNwithoutExtra(nn.Module):
    def __init__(self):
        super(CNNwithoutExtra, self).__init__()
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers for the regression task (offset time detection)
        self.fc1_regression = nn.Linear(128 * 32 * 32, 512) # Hmax, Hmin, Bmax, Bmin + image features
        self.fc2_regression = nn.Linear(512, 1)  # Offset time detection

    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Regression branch
        x_regression = F.relu(self.fc1_regression(x))
        x_regression = self.fc2_regression(x_regression)
        
        return x_regression