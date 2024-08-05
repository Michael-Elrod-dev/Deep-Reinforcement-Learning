import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, image_size, num_labels, seed):
        super(QNetwork, self).__init__()
        # input
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(image_size, 64)
        # advantage
        self.ad1 = nn.Linear(64, 32)
        self.ad2 = nn.Linear(32, num_labels)
        # value
        self.va1 = nn.Linear(64, 32)
        self.va2 = nn.Linear(32, 1)

    def forward(self, image):
        # input
        linear_1 = F.relu(self.fc1(image))
        # advantage
        advantage_1 = F.relu(self.ad1(linear_1))
        label_advantage = self.ad2(advantage_1)
        # value
        value_1 = F.relu(self.va1(linear_1))
        image_value = self.va2(value_1)
        # combining
        max_label_advantage = torch.max(label_advantage, dim=1)[0].unsqueeze(1)
        value_image_label = image_value + label_advantage - max_label_advantage 
        
        return value_image_label
    