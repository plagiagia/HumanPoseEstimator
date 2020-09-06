# Imports
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from scipy.io import loadmat

# Load the data from the SVHN dataset
train = loadmat('train_32x32.mat')
test = loadmat('test_32x32.mat')
# Labels for number 0 have the value 10, so I have to change them to represent the actual value
train['y'][train['y'] == 10] = 0
test['y'][test['y'] == 10] = 0

# Turn data into tensors, and also change the dimensions to represent (batch, channel, height, width)
train_features_tensor = torch.from_numpy(train['X'] / 255.).permute(3, 2, 0, 1).float()
train_labels_tensor = torch.from_numpy(train['y']).squeeze().long()

test_features_tensor = torch.from_numpy(test['X'] / 255.).permute(3, 2, 0, 1).float()
test_labels_tensor = torch.from_numpy(test['y']).squeeze().long()

# Make a data loader for each dataset
train_dataset = data_utils.TensorDataset(train_features_tensor, train_labels_tensor)
test_dataset = data_utils.TensorDataset(test_features_tensor, test_labels_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=32,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=32,
                                          shuffle=True)

# Build the network
in_channels = 3

out_channels_1 = 16
out_channels_2 = 32

out1_size = 1200
out2_size = 10

kernel_size = (5, 5)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Create the first convolution layer, channel input is 3 for RGB images, the output is selected to have 16
        # filters, padding is set to 2 so we take as output the same dimensions as the original image.
        # Formula to calculate the output shape of the convolution layer [(W−K+2P)/S]+1.
        # out_shape_height = (7-3+0) + 1

        self.Conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_1,
                               padding=1, kernel_size=kernel_size)
        # Initialize the weights with the Xavier initialization method, and set bias to 0
        nn.init.xavier_normal_(self.Conv1.weight)
        nn.init.constant_(self.Conv1.bias, 0)

        self.layer1 = nn.Sequential(
            self.Conv1,
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.Conv2 = nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2,
                               kernel_size=kernel_size)
        nn.init.xavier_normal_(self.Conv2.weight)
        nn.init.constant_(self.Conv2.bias, 0)

        self.layer2 = nn.Sequential(
            self.Conv2,
            nn.BatchNorm2d(out_channels_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(32 * 5 * 5, out1_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(out1_size, out2_size)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer3(out)

        return F.log_softmax(out, dim=1)


model = ConvNet()
device = torch.device('cuda:0')
model.to(device)

learning_rate = 0.001
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
num_epochs = 100
loss_values = list()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            os.system('cls')
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            loss_values.append(loss.item())

print("Finish training")

# Plot the results
x = (range(1, num_epochs + 1))

plt.figure(figsize=(10, 5))
plt.plot(x, loss_values)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()

# Evaluate the model
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {:.4f}%)'.format(100 * correct / total))

# Test model performance and check some results
with torch.no_grad():
    images, labels = iter(test_loader).next()
    images = images.to(device)
    out_predict = model(images)
    _, predicted = torch.max(out_predict.data, 1)
    for idx, each_image in enumerate(images[:5]):
        each_image = each_image.cpu().permute(1, 2, 0).numpy()
        image_label = labels[idx].cpu().numpy().squeeze()
        print("True Label: ", image_label, " Predicted Label: ", predicted[idx].cpu().numpy().squeeze())
        plt.imshow(each_image)
        plt.show()
        print("-------------")
