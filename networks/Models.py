import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl
from dgl.nn.pytorch.conv import SAGEConv
from torch_geometric.nn import GCNConv, GATConv, SGConv, NNConv, GINConv, GatedGraphConv, CGConv, PNAConv
from torch_geometric.utils import degree

class BHSDuelingDQN(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(BHSDuelingDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_outputs # a vector of the number of actions at each diverter

        self.conv1 = nn.Conv1d(self.input_shape[-1], 128, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)
        self.conv2_3 = nn.Conv1d(128, 64, kernel_size=7, stride=1, padding=3)
        
        self.conv3 = nn.Conv1d(64*3, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.adv = nn.Linear(self.feature_size(), sum(self.num_actions)) # Might be an idea to add another fc layer here

        self.val1 = nn.Linear(self.feature_size(), 64)
        self.val2 = nn.Linear(64, 64)
        self.val3 = nn.Linear(64, 1)
        #self.val3 = nn.Linear(64, len(self.num_actions))
        

    def forward(self, x):        
        # x comes in as an N x H x C shape
        # change x to the shape N x C x H
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2_1(x))
        x2 = F.relu(self.conv2_2(x))
        x3 = F.relu(self.conv2_3(x))
        x = torch.cat((x1,x2,x3),1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        
        adv = F.relu(self.adv(x))
        adv = adv.view(adv.size(0),len(self.num_actions),-1)

        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = self.val3(val)
        
        return val.unsqueeze(-1).expand_as(adv) + adv - adv.mean(-1).unsqueeze(-1).expand_as(adv)
    
    def feature_size(self):
        return self.conv5(
            self.conv4(
                self.conv3(
                    torch.cat(
                        (self.conv2_1(
                            self.conv1(torch.zeros(1, self.input_shape[-1],self.input_shape[0]))),
                         self.conv2_2(
                             self.conv1(torch.zeros(1, self.input_shape[-1],self.input_shape[0]))),
                         self.conv2_3(
                             self.conv1(torch.zeros(1, self.input_shape[-1],self.input_shape[0])))),
                         1)
                ))).view(1, -1).size(1)
    
    def sample_noise(self):
        #ignore this for now
        pass
    
class BHS_GCN(nn.Module):
    def __init__(self, input_shape, num_outputs, edgelist, edge_weight):
        super(BHS_GCN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_outputs # a vector of the number of actions at each diverter
        self.edge = edgelist
        self.edge_weight = edge_weight

        self.conv1 = GCNConv(self.input_shape[1], 128)
        self.conv2 = GCNConv(128, 256)
        
        self.adv = nn.Linear(self.feature_size(), sum(self.num_actions)) # Might be an idea to add another fc layer here

        self.val1 = nn.Linear(self.feature_size(), 64)
        self.val2 = nn.Linear(64, 64)
        self.val3 = nn.Linear(64, 1)
        #self.val3 = nn.Linear(64, len(self.num_actions))

    def forward(self, x):        
        # x comes in as an N x H x C shape (N is batch size, H is number of elements (height), C is number of features (channels))
        x_shape = x.shape 
        x = x.view(x_shape[0]*x_shape[1],x_shape[2]) # set shape of x to [N*H, C] to get the shape of a Graph batch
        
        x = F.relu(self.conv1(x, self.edge, self.edge_weight))
        x = F.relu(self.conv2(x, self.edge, self.edge_weight))
        
        x = x.view(x_shape[0], -1) # set shape of x to [N,:] to keep the batch size
        
        adv = F.relu(self.adv(x))
        adv = adv.view(adv.size(0),len(self.num_actions),-1)
            
        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = self.val3(val)
        
        return val.unsqueeze(-1).expand_as(adv) + adv - adv.mean(-1).unsqueeze(-1).expand_as(adv)
    
    def feature_size(self):
        x = self.conv1(torch.zeros([self.input_shape[0],self.input_shape[1]],dtype=torch.float),torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = self.conv2(x,torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = x.view(1, -1)
        x = x.size(1)
        return x
    
    def sample_noise(self):
        #ignore this for now
        pass

class BHS_GAT(nn.Module):
    def __init__(self, input_shape, num_outputs, edgelist):
        super(BHS_GAT, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_outputs # a vector of the number of actions at each diverter
        self.edge = edgelist

        self.conv1 = GATConv(self.input_shape[1], 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8*8, 128, heads=8, dropout=0.6)
        
        self.adv = nn.Linear(self.feature_size(), sum(self.num_actions)) # Might be an idea to add another fc layer here

        self.val1 = nn.Linear(self.feature_size(), 64)
        self.val2 = nn.Linear(64, 64)
        self.val3 = nn.Linear(64, 1)
        #self.val3 = nn.Linear(64, len(self.num_actions))

    def forward(self, x):        
        # x comes in as an N x H x C shape (N is batch size, H is number of elements (height), C is number of features (channels))
        x_shape = x.shape 
        x = x.view(x_shape[0]*x_shape[1],x_shape[2]) # set shape of x to [N*H, C] to get the shape of a Graph batch
        
        x = F.relu(self.conv1(x, self.edge))
        x = F.relu(self.conv2(x, self.edge))
        
        x = x.view(x_shape[0], -1) # set shape of x to [N,:] to keep the batch size
        
        adv = F.relu(self.adv(x))
        adv = adv.view(adv.size(0),len(self.num_actions),-1)
            
        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = self.val3(val)
        return val.unsqueeze(-1).expand_as(adv) + adv - adv.mean(-1).unsqueeze(-1).expand_as(adv)
    
    def feature_size(self):
        x = self.conv1(torch.zeros([self.input_shape[0],self.input_shape[1]],dtype=torch.float), torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = self.conv2(x,torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = x.view(1, -1)
        x = x.size(1)
        return x
    
    def sample_noise(self):
        #ignore this for now
        pass
    
class BHS_SGN(nn.Module):
    def __init__(self, input_shape, num_outputs, edgelist, edge_weight):
        super(BHS_SGN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_outputs # a vector of the number of actions at each diverter
        self.edge = edgelist
        self.edge_weight = edge_weight

        self.conv1 = SGConv(self.input_shape[1], 128, K=2)

        self.adv = nn.Linear(self.feature_size(), sum(self.num_actions)) # Might be an idea to add another fc layer here

        self.val1 = nn.Linear(self.feature_size(), 64)
        self.val2 = nn.Linear(64, 64)
        self.val3 = nn.Linear(64, 1)
        #self.val3 = nn.Linear(64, len(self.num_actions))

    def forward(self, x):        
        # x comes in as an N x H x C shape (N is batch size, H is number of elements (height), C is number of features (channels))
        x_shape = x.shape 
        x = x.view(x_shape[0]*x_shape[1],x_shape[2]) # set shape of x to [N*H, C] to get the shape of a Graph batch
        
        x = F.relu(self.conv1(x, self.edge, self.edge_weight))
        
        x = x.view(x_shape[0], -1) # set shape of x to [N,:] to keep the batch size
        
        adv = F.relu(self.adv(x))
        adv = adv.view(adv.size(0),len(self.num_actions),-1)
            
        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = self.val3(val)
        return val.unsqueeze(-1).expand_as(adv) + adv - adv.mean(-1).unsqueeze(-1).expand_as(adv)
    
    def feature_size(self):
        x = self.conv1(torch.zeros([self.input_shape[0],self.input_shape[-1]],dtype=torch.float), torch.zeros([self.edge.shape[0],self.edge.shape[-1]], dtype=torch.long))
        x = x.view(1, -1)
        x = x.size(1)
        return x
    
    def sample_noise(self):
        #ignore this for now
        pass
    
class BHS_SAGE(nn.Module):
    def __init__(self, input_shape, num_outputs, graph):
        super(BHS_SAGE, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_outputs # a vector of the number of actions at each diverter
        self.graph = graph       

        self.conv1 = SAGEConv(self.input_shape[1], 128,"pool") #Aggregator type: "mean"/"gcn"/"pool"/"lstm"
        
        self.adv = nn.Linear(self.feature_size(), sum(self.num_actions)) # Might be an idea to add another fc layer here

        self.val1 = nn.Linear(self.feature_size(), 64)
        self.val2 = nn.Linear(64, 64)
        self.val3 = nn.Linear(64, 1)
        #self.val3 = nn.Linear(64, len(self.num_actions))

    def forward(self, x):        
        # x comes in as an N x H x C shape (N is batch size, H is number of elements (height), C is number of features (channels))
        x_shape = x.shape 
        G = dgl.batch([self.graph] * x_shape[0]) # batching DGL graphs
        x = x.view(x_shape[0]*x_shape[1],x_shape[2]) # set shape of x to [N*H, C] to get the shape of a Graph batch
        
        x = F.relu(self.conv1(G,x))
        
        x = x.view(x_shape[0], -1) # set shape of x to [N,:] to keep the batch size
        
        adv = F.relu(self.adv(x))
        adv = adv.view(adv.size(0),len(self.num_actions),-1)
            
        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = self.val3(val)
        
        return val.unsqueeze(-1).expand_as(adv) + adv - adv.mean(-1).unsqueeze(-1).expand_as(adv)
    
    def feature_size(self):
        x = self.conv1(self.graph,torch.zeros([self.input_shape[0],self.input_shape[1]],dtype=torch.float))
        x = x.view(1, -1)
        x = x.size(1)
        return x
    
    def sample_noise(self):
        #ignore this for now
        pass

class BHS_GIN(nn.Module):
    def __init__(self, input_shape, num_outputs, edgelist):
        super(BHS_GIN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_outputs # a vector of the number of actions at each diverter
        self.edge = edgelist       
        
        nn1 = nn.Sequential(nn.Linear(self.input_shape[1], 128), nn.ReLU(), nn.Linear(128,128))
        self.conv1 = GINConv(nn1, train_eps=True)
        nn2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.conv2 = GINConv(nn2, train_eps=True)
        self.conv3 = GINConv(nn2, train_eps=True)
        self.conv4 = GINConv(nn2, train_eps=True)
        self.conv5 = GINConv(nn2, train_eps=True)
        
        self.adv = nn.Linear(self.feature_size(), sum(self.num_actions)) # Might be an idea to add another fc layer here

        self.val1 = nn.Linear(self.feature_size(), 64)
        self.val2 = nn.Linear(64, 64)
        self.val3 = nn.Linear(64, 1)
        #self.val3 = nn.Linear(64, len(self.num_actions))

    def forward(self, x):        
        # x comes in as an N x H x C shape (N is batch size, H is number of elements (height), C is number of features (channels))
        x_shape = x.shape 
        x = x.view(x_shape[0]*x_shape[1],x_shape[2]) # set shape of x to [N*H, C] to get the shape of a Graph batch
        
        x = F.relu(self.conv1(x, self.edge))
        x = F.relu(self.conv2(x, self.edge))
        x = F.relu(self.conv3(x, self.edge))
        x = F.relu(self.conv4(x, self.edge))
        x = F.relu(self.conv5(x, self.edge))
        
        x = x.view(x_shape[0], -1) # set shape of x to [N,:] to keep the batch size
        
        adv = F.relu(self.adv(x))
        adv = adv.view(adv.size(0),len(self.num_actions),-1)
            
        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = self.val3(val)
        
        return val.unsqueeze(-1).expand_as(adv) + adv - adv.mean(-1).unsqueeze(-1).expand_as(adv)
    
    def feature_size(self):
        x = self.conv1(torch.zeros([self.input_shape[0],self.input_shape[1]],dtype=torch.float),torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = self.conv2(x,torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = self.conv3(x,torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = self.conv4(x,torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = self.conv5(x,torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = x.view(1, -1)
        x = x.size(1)
        return x
    
    def sample_noise(self):
        #ignore this for now
        pass

class BHS_GGNN(nn.Module):
    def __init__(self, input_shape, num_outputs, edgelist, edge_weight):
        super(BHS_GGNN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_outputs # a vector of the number of actions at each diverter
        self.edge = edgelist
        self.edge_weight = edge_weight

        self.conv1 = GatedGraphConv(self.input_shape[1], 5)
        
        self.adv = nn.Linear(self.feature_size(), sum(self.num_actions)) # Might be an idea to add another fc layer here

        self.val1 = nn.Linear(self.feature_size(), 64)
        self.val2 = nn.Linear(64, 64)
        self.val3 = nn.Linear(64, 1)
        #self.val3 = nn.Linear(64, len(self.num_actions))

    def forward(self, x):        
        # x comes in as an N x H x C shape (N is batch size, H is number of elements (height), C is number of features (channels))
        x_shape = x.shape 
        x = x.view(x_shape[0]*x_shape[1],x_shape[2]) # set shape of x to [N*H, C] to get the shape of a Graph batch
        
        x = F.relu(self.conv1(x, self.edge, self.edge_weight))
        
        x = x.view(x_shape[0], -1) # set shape of x to [N,:] to keep the batch size
        
        adv = F.relu(self.adv(x))
        adv = adv.view(adv.size(0),len(self.num_actions),-1)
            
        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = self.val3(val)
        
        return val.unsqueeze(-1).expand_as(adv) + adv - adv.mean(-1).unsqueeze(-1).expand_as(adv)
    
    def feature_size(self):
        x = self.conv1(torch.zeros([self.input_shape[0],self.input_shape[1]],dtype=torch.float), torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = x.view(1, -1)
        x = x.size(1)
        return x
    
    def sample_noise(self):
        #ignore this for now
        pass
    
class BHS_NN(nn.Module):
    def __init__(self, input_shape, num_outputs, edgelist, edge_attr):
        super(BHS_NN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_outputs # a vector of the number of actions at each diverter
        self.edge = edgelist
        self.edge_attr = edge_attr
        
        nn1 = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, self.input_shape[1]*128)) # edge attribute neural network
        self.conv1 = NNConv(self.input_shape[1], 128, nn1)
        
        self.adv = nn.Linear(self.feature_size(), sum(self.num_actions)) # Might be an idea to add another fc layer here

        self.val1 = nn.Linear(self.feature_size(), 64)
        self.val2 = nn.Linear(64, 64)
        self.val3 = nn.Linear(64, 1)
        #self.val3 = nn.Linear(64, len(self.num_actions))

    def forward(self, x):        
        # x comes in as an N x H x C shape (N is batch size, H is number of elements (height), C is number of features (channels))
        x_shape = x.shape
        x = x.view(x_shape[0]*x_shape[1],x_shape[2]) # set shape of x to [N*H, C] to get the shape of a Graph batch
        
        x = F.relu(self.conv1(x, self.edge, self.edge_attr))
        
        x = x.view(x_shape[0], -1) # set shape of x to [N,:] to keep the batch size
        
        adv = F.relu(self.adv(x))
        adv = adv.view(adv.size(0),len(self.num_actions),-1)
            
        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = self.val3(val)
        
        return val.unsqueeze(-1).expand_as(adv) + adv - adv.mean(-1).unsqueeze(-1).expand_as(adv)
    
    def feature_size(self):
        x = self.conv1(
                    torch.zeros([self.input_shape[0],self.input_shape[1]],dtype=torch.float),
                    torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long),
                    torch.zeros([self.edge_attr.shape[0],self.edge_attr.shape[1]],dtype=torch.float))
        x = x.view(1, -1)
        x = x.size(1)
        return x
    
    def sample_noise(self):
        #ignore this for now
        pass
    
class BHS_CG(nn.Module):
    def __init__(self, input_shape, num_outputs, edgelist, edge_attr):
        super(BHS_CG, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_outputs # a vector of the number of actions at each diverter
        self.edge = edgelist
        self.edge_attr = edge_attr
        
        self.conv1 = CGConv([self.input_shape[1], self.input_shape[1]], dim=self.edge_attr.shape[1], batch_norm=True)
        
        self.adv = nn.Linear(self.feature_size(), sum(self.num_actions)) # Might be an idea to add another fc layer here

        self.val1 = nn.Linear(self.feature_size(), 64)
        self.val2 = nn.Linear(64, 64)
        self.val3 = nn.Linear(64, 1)
        #self.val3 = nn.Linear(64, len(self.num_actions))

    def forward(self, x):        
        # x comes in as an N x H x C shape (N is batch size, H is number of elements (height), C is number of features (channels))
        x_shape = x.shape
        x = x.view(x_shape[0]*x_shape[1],x_shape[2]) # set shape of x to [N*H, C] to get the shape of a Graph batch
        
        x = F.relu(self.conv1(x, self.edge, self.edge_attr))
        
        x = x.view(x_shape[0], -1) # set shape of x to [N,:] to keep the batch size
        
        adv = F.relu(self.adv(x))
        adv = adv.view(adv.size(0),len(self.num_actions),-1)
            
        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = self.val3(val)
        
        return val.unsqueeze(-1).expand_as(adv) + adv - adv.mean(-1).unsqueeze(-1).expand_as(adv)
    
    def feature_size(self):
        x = self.conv1(
                    torch.zeros([self.input_shape[0],self.input_shape[1]],dtype=torch.float),
                    torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long),
                    torch.zeros([self.edge_attr.shape[0],self.edge_attr.shape[1]],dtype=torch.float))
        x = x.view(1, -1)
        x = x.size(1)
        return x
    
    def sample_noise(self):
        #ignore this for now
        pass
    
class BHS_PNA(nn.Module):
    def __init__(self, input_shape, num_outputs, edgelist, edge_attr):
        super(BHS_PNA, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_outputs # a vector of the number of actions at each diverter
        self.edge = edgelist
        self.edge_attr = edge_attr      
        
        d = degree(self.edge[1], num_nodes=self.input_shape[0], dtype=torch.long)
        deg = torch.bincount(d)
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.conv1 = PNAConv(in_channels=8, out_channels=128, aggregators=aggregators, scalers=scalers, deg=deg, edge_dim=1)
        
        self.adv = nn.Linear(self.feature_size(), sum(self.num_actions)) # Might be an idea to add another fc layer here

        self.val1 = nn.Linear(self.feature_size(), 64)
        self.val2 = nn.Linear(64, 64)
        self.val3 = nn.Linear(64, 1)
        #self.val3 = nn.Linear(64, len(self.num_actions))

    def forward(self, x):        
        # x comes in as an N x H x C shape (N is batch size, H is number of elements (height), C is number of features (channels))
        x_shape = x.shape
        x = x.view(x_shape[0]*x_shape[1],x_shape[2]) # set shape of x to [N*H, C] to get the shape of a Graph batch
        
        x = F.relu(self.conv1(x, self.edge, self.edge_attr))
        
        x = x.view(x_shape[0], -1) # set shape of x to [N,:] to keep the batch size
        
        adv = F.relu(self.adv(x))
        adv = adv.view(adv.size(0),len(self.num_actions),-1)
            
        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = self.val3(val)
        
        return val.unsqueeze(-1).expand_as(adv) + adv - adv.mean(-1).unsqueeze(-1).expand_as(adv)
    
    def feature_size(self):
        x = self.conv1(
                    torch.zeros([self.input_shape[0],self.input_shape[1]],dtype=torch.float),
                    torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long),
                    torch.zeros([self.edge_attr.shape[0],self.edge_attr.shape[1]],dtype=torch.float))
        x = x.view(1, -1)
        x = x.size(1)
        return x
    
    def sample_noise(self):
        #ignore this for now
        pass

class BHS_GCN_DQN(nn.Module):
    def __init__(self, input_shape, num_outputs, edgelist, edge_weight):
        super(BHS_GCN_DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_outputs # a vector of the number of actions at each diverter
        self.edge = edgelist
        self.edge_weight = edge_weight

        self.conv1 = GCNConv(self.input_shape[1], 128)
        self.conv2_1 = GCNConv(128, 64)
        self.conv2_2 = GCNConv(128, 64)
        self.conv2_3 = GCNConv(128, 64)

        self.conv3 = GCNConv(64*3, 128)
        self.conv4 = GCNConv(128, 128)
        self.conv5 = GCNConv(128, 128)
        
        self.adv = nn.Linear(self.feature_size(), sum(self.num_actions)) # Might be an idea to add another fc layer here

        self.val1 = nn.Linear(self.feature_size(), 64)
        self.val2 = nn.Linear(64, 64)
        self.val3 = nn.Linear(64, 1)

    def forward(self, x):        
        # x comes in as an N x H x C shape (N is batch size, H is number of elements (height), C is number of features (channels))
        x_shape = x.shape 
        x = x.view(x_shape[0]*x_shape[1],x_shape[2]) # set shape of x to [N*H, C] to get the shape of a Graph batch
        
        x = F.relu(self.conv1(x, self.edge, self.edge_weight))
        x1 = F.relu(self.conv2_1(x, self.edge, self.edge_weight))
        x2 = F.relu(self.conv2_2(x, self.edge, self.edge_weight))
        x3 = F.relu(self.conv2_3(x, self.edge, self.edge_weight))
        x = torch.cat((x1,x2,x3),1)
        x = F.relu(self.conv3(x, self.edge, self.edge_weight))
        x = F.relu(self.conv4(x, self.edge, self.edge_weight))
        x = F.relu(self.conv5(x, self.edge, self.edge_weight))
        
        x = x.view(x_shape[0], -1) # set shape of x to [N,:] to keep the batch size
        
        adv = F.relu(self.adv(x))
        adv = adv.view(adv.size(0),len(self.num_actions),-1)
            
        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = self.val3(val)
        
        return val.unsqueeze(-1).expand_as(adv) + adv - adv.mean(-1).unsqueeze(-1).expand_as(adv)
    
    def feature_size(self):
        x = self.conv1(torch.zeros([self.input_shape[0],self.input_shape[1]],dtype=torch.float),torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x1 = self.conv2_1(x,torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x2 = self.conv2_2(x,torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x3 = self.conv2_3(x,torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = torch.cat((x1,x2,x3),1)
        x = self.conv3(x,torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = self.conv4(x,torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = self.conv5(x,torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long))
        x = x.view(1, -1)
        x = x.size(1)
        return x
    
    def sample_noise(self):
        #ignore this for now
        pass
    
class BHS_TEST(nn.Module):
    def __init__(self, input_shape, num_outputs, edgelist, edge_attr, hidden):
        super(BHS_TEST, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_outputs # a vector of the number of actions at each diverter
        self.edge = edgelist
        self.edge_attr = edge_attr
        self.hidden = hidden
        
        nn1 = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, self.input_shape[1]*self.hidden.shape[2])) # edge attribute neural network
        self.conv1 = NNConv(self.input_shape[1], self.hidden.shape[2], nn1)
        self.gate = nn.GRU(self.hidden.shape[2], self.hidden.shape[2], self.hidden.shape[0])
        
        self.adv = nn.Linear(self.feature_size(), sum(self.num_actions)) # Might be an idea to add another fc layer here

        self.val1 = nn.Linear(self.feature_size(), 64)
        self.val2 = nn.Linear(64, 64)
        self.val3 = nn.Linear(64, 1)

    def forward(self, x):        
        # x comes in as an N x H x C shape (N is batch size, H is number of elements (height), C is number of features (channels))
        x_shape = x.shape
        #x = x[:x_shape[0],self.down_nodes]
        #start_down=timer()
        #x_down = torch.zeros([x_shape[0],self.input_shape[0],x_shape[2]]).to(self.device)
        #edge_occu = [0]*len(self.edge_nodes)
        # for n in range(x_shape[0]):
          #  x_down[n] = x[n][nodes]
          #  for k in range(x_shape[1]):
          #      for e in range(len(self.edge_nodes)):
          #          if k in self.edge_nodes[e]:
          #              if len(torch.nonzero(x[n][k],as_tuple=False)) != 0:
          #                  edge_occu[e] += 1
          #              break

        #edge_attr = torch.Tensor(list(np.array(self.edge_attr) - np.array(edge_occu))).to(self.device)
        #x_shape = x_down.shape
        
        #time_down = timer()-start_down
        #print('TIME DOWN = ',time_down, x_shape[0])
        x = x.view(x_shape[0]*x_shape[1],x_shape[2]) # set shape of x to [N*H, C] to get the shape of a Graph batch
        
        x = F.relu(self.conv1(x, self.edge, self.edge_attr))
        x = x.view(x_shape[0],x_shape[1],x.shape[1])
        x, self.hidden = self.gate(x, self.hidden)
        
        x = x.view(x_shape[0], -1) # set shape of x to [N,:] to keep the batch size
        
        
        adv = F.relu(self.adv(x))
        adv = adv.view(adv.size(0),len(self.num_actions),-1)
            
        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = self.val3(val)
        
        return val.unsqueeze(-1).expand_as(adv) + adv - adv.mean(-1).unsqueeze(-1).expand_as(adv)
    
    def feature_size(self):
        x = self.conv1(
                    torch.zeros([self.input_shape[0],self.input_shape[1]],dtype=torch.float),
                    torch.zeros([self.edge.shape[0],self.edge.shape[1]], dtype=torch.long),
                    torch.zeros([self.edge_attr.shape[0],self.edge_attr.shape[1]],dtype=torch.float))
        x = x.view(1,x.shape[0],x.shape[1])
        x, h = self.gate(x)
        x = x.view(1, -1)
        x = x.size(1)
        return x
    
    def sample_noise(self):
        #ignore this for now
        pass