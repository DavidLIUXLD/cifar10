import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import datasets
from Lenet5 import Lenet5

transform = transforms.Compose(
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
trainSets = torchvision.datasets.CIFAR10('./data', train = True, transform=transform, download=True)
trainSets = torch.utils.DataLoader(trainSets, batch_size = 32, suffle=True)
testSets = torchvision.datasets.CIFAR10('./data', False, transform=transform)
tesetSets = torch.utils.DataLoader(testSets, batch_size=32, suffle=True)  

device = torch.device('cuda')
model = Lenet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.005, momentum=0.9)

for epoch in range(1000):
    for i, (x, label) in enumerate(trainSets):
        x, label = x.to(device), label = label.to(device)
        logits = model(x)
        loss = criterion