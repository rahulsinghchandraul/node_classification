import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn 

#Define messages
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, g, features):
        #with g.local_scope(): WHY is THIS USED ? NOTSURE! TEST it
        g.ndata['h'] = features
        g.update_all(gcn_msg, gcn_reduce)
        h = g.ndata['h']
        return self.linear(h)

class Net(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(in_features, hidden_size)
        self.layer2 = GCNLayer(hidden_size, num_classes)
    
    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x
