import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import torch
import torch as th 
import torch.nn.functional as F

#from gcn_mp import GCNLayer, Net
#from graph_sage import Net
#from sage_mp import SAGELayer, Net
from udf_example import SAGELayer, Net
from utils import load_cora_data, evaluate

#Load dataset
g, features, labels, train_mask, test_mask = load_cora_data()
# Add edges between each node and itself to preserve old node representations
g.add_edges(g.nodes(), g.nodes())

in_features = 1433
hidden_size = 16
num_classes = 7

net = Net(in_features, hidden_size, num_classes)
print(g.ndata)

optimizer = th.optim.Adam(net.parameters(), lr = 5e-3)
for epoch in range(500):
    t0 = time.time()

    net.train()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    t1 = time.time() - t0 # time taken for one update step
    
    acc_train = evaluate(net, g, features, labels, train_mask)
    acc_test = evaluate(net, g, features, labels, test_mask)
    print("Epoch# {:04d} | Loss {:.4f} | Train Acc {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc_train, acc_test, t1))


