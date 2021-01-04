import torch.nn as nn
import torch.nn.functional as F
import torch as th 
import dgl.function as fn 

#Define messages
#gcn_msg = fn.copy_src(src='h', out='m')
#gcn_reduce = fn.mean(msg='m', out='h')

def gcn_msg(edges):
    return {'m' : edges.src['h']}

def gcn_reduce(nodes):
    return {'h' : nodes.mailbox['m'].mean(1)}

class SAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SAGELayer, self).__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, g, features):
        g.ndata['h'] = features
        g.update_all(gcn_msg, gcn_reduce)
        h = g.ndata['h']
        h_concat = th.cat([features, h], dim = 1)
        return self.linear(h_concat)

class Net(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes):
        super(Net, self).__init__()
        self.layer1 = SAGELayer(in_features, hidden_size)
        self.layer2 = SAGELayer(hidden_size, num_classes)  
    
    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x
