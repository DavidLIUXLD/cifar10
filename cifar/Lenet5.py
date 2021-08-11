from torch import nn

class Lenet5(nn.Module):

    def __init__(self):
        super(Lenet5, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding = 0),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16*25, 120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

    
        


