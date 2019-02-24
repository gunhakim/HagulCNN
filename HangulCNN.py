#-*- coding=utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import glob
from PIL import Image

# setting device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper Patameters
crop_size = 80
num_epochs = 1
num_class = 2350
batch_size = 1
learning_rate = 0.1

# Load dataset
class HanDB(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.files = glob.glob('/gdrive/My Drive/Colab Notebooks/Hangul/HanDB_test/HanDB_test_data/*.csv')
        self.files.sort()
        self.transforms = transforms.Compose([transforms.CenterCrop(crop_size), transforms.ToTensor(),])
        self.data_list = [Image.fromarray(pd.read_csv(f).values.astype(np.uint8),'L') for f in self.files]
        self.label_list = []
        Han = -1
        for f in self.files:
            if f[73:-4] == '1':Han +=1
            self.label_list.append(Han)

    def __getitem__(self, index):
        img = self.transforms(self.data_list[index])
        label = self.label_list[index]
        return (img,label)

    def __len__(self):
        return len(self.data_list)

data_dir = '/gdrive/My Drive/Colab Notebooks/Hangul/HanDB_test/HanDB_test_data/'
dataset = HanDB(data_dir)
dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

#CNN Network
class ConvNet(nn.Module):
    def __init__(self, num_class):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(256, num_class)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = torch.sum(out, (2,3))
        out = self.fc(out)
        return out 

model = ConvNet(num_class).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters, lr = learning_rate, momentum = 0.9, nesterov = True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Train Model
total_step = len(dataset_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataset_loader):
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if True:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#Test Model
data_dir = ''
dataset = HanDB(data_dir)
dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)

model.eval()
with torch.no_grad():
    corrrect = 0
    total = 0
    for images, labels in dataset_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = models(images)
        -, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        corrrect += (predicted == labels).sum().item()
