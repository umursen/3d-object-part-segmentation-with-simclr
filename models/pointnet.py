import numpy as np
import torch
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self,return_point_features=False):
        super(PointNetEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, kernel_size=(1,))
        self.conv2 = nn.Conv1d(64, 128, kernel_size=(1,))
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=(1,))

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.relu = nn.ReLU()

        self.input_transform_net = TNet(k=3)
        self.feature_transform_net = TNet(k=64)

        self.return_point_features = return_point_features

    def forward(self, x):
        num_points = x.shape[2]
        input_transform = self.input_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), input_transform).transpose(2, 1)
        x = self.relu(self.bn1(self.conv1(x)))

        feature_transform = self.feature_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), feature_transform).transpose(2, 1)
        point_features = x

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Return point features for later segmentation usage.
        # Not needed when contrastive learning.
        if self.return_point_features:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x, point_features], dim=1)
        else:
            return x


class PointNetDecoder(nn.Module):
    def __init__(self, num_classes):
        super(PointNetDecoder, self).__init__()
        self.num_classes = num_classes

        self.concattenated_features = 1088
        self.conv1 = nn.Conv1d(self.concattenated_features, 512, kernel_size=(1,))
        self.conv2 = nn.Conv1d(512, 256, kernel_size=(1,))
        self.conv3 = nn.Conv1d(256, 128, kernel_size=(1,))
        self.conv4 = nn.Conv1d(128, num_classes, kernel_size=(1,))

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose(2, 1).contiguous()
        return x


class PointNetSegmentation(nn.Module):

    def __init__(self, num_classes):
        super(PointNetSegmentation, self).__init__()
        self.encoder = PointNetEncoder(return_point_features=True)
        self.decoder = PointNetDecoder(num_classes)
        
    def forward(self, x):
        x = self.encoder(x) # Concatenated features
        x = self.decoder(x)
        return x

class TNet(nn.Module):
    '''
    Submodule used to implement feature transformation part of pointnet.
    '''
    def __init__(self,k):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, kernel_size=(1,))
        self.conv2 = nn.Conv1d(64, 128, kernel_size=(1,))
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=(1,))
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.relu = nn.ReLU()

        # Segmentation class prediction
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k**2)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.register_buffer('identity', torch.from_numpy(np.eye(k).flatten().astype(np.float32)).view(1, k ** 2))
        self.k = k
    
    def forward(self,x):
        batch_size = x.shape[0]

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        identity = self.identity.repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x
