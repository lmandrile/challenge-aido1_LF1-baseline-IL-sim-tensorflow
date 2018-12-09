import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TorchModel(nn.Module):

    def __init__(self, observation_shape, action_shape, save_path):
        super(TorchModel, self).__init__()



        self.save_path = save_path

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        #layers * entry img x * entry img y
        print(action_shape)
        self.finalLayer = nn.Linear(128*observation_shape[1] * observation_shape[2], action_shape[1])

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        print(shape(x))
        x = self.finalLayer(x)

        return x

    def save(self, path):
        torch.save(model.state_dict(), self.save_path)

    def load(self, path):
        device = torch.device('cpu')
        model.load_state_dict(torch.load(self.save_path, map_location=device))

        return self.model
