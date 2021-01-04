import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn  #For 'mean', 'sum', 'pool'

from dgl.nn import SAGEConv
class Net(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes):
        super(Net, self).__init__()
        self.layer1 = SAGEConv(in_features, hidden_size, 'max')
        self.layer2 = SAGEConv(hidden_size, num_classes, 'max')
    
    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x
