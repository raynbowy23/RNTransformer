import torch
import torch.nn as nn
import numpy as np
from models.LocalPedsTrajNet import social_stgcnn


class LocalPedsTrajNet(social_stgcnn):
    """ Inherited class of Social STGCNN

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        pass

    def forward(self, x, A):

        v, a = social_stgcnn(x, A)

        return v, a


class GroupInteractionNet(nn.Module):
    """ Group Interaction Network

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 node_features,
                 num_nodes,
                 periods,
                 output_dim,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(GroupInteractionNet, self).__init__()
        
        # Attention Temporal Graph Convolutional Cell + Regional concat feature
        hidden_dim = 256
        out_channels = 512
        self.periods = periods
        self.num_nodes = num_nodes
        self.output_dim = output_dim

        # Can be stacked
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

        self.linear1 = torch.nn.Linear(out_channels, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        """
        x = Node features for T time steps
        A = Adjacency matrix
        """
        x = self.conv(x)
        h = torch.einsum('nctv,tvw->nctw', (x, A))
        h = self.relu(h)
        h = self.linear1(h)
        h = self.relu(h)
        h = self.linear2(h)
        return h


class PedsTrajNet(nn.Module):
    """Pedestrian Trajectory Network

    Args:
        nn (_type_): _description_
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 horizon=4,
                 bias=True):
        super(PedsTrajNet, self).__init__()

        self.local_net = LocalPedsTrajNet()
        self.group_net = GroupInteractionNet()

    def forward(self, x, gx, h_RN):
        """_summary_

        Args:
            x (_type_): pedestrian trajectory data
            gx (_type_): group interaction data
            h_RN (_type_): hidden vector of RN
        """
        # TODO: Combine into one with Multi-head-attention --> Can be temporal fusion
        h_loc = self.local_net(x)
        h_group = self.group_net(gx)
        # TemporalFusion(h_loc, h_group, h_RN)

        # return out 

