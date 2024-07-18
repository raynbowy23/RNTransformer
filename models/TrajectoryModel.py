import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from einops import rearrange

from models.SocialStgcnn import social_stgcnn
from models.SocialImplicit import SocialImplicit
from models.SocialLSTM import SocialModel
from models.RNGCN import RNTransformer
from models.LocalPedsTrajNet import *

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float=0.1, max_length: int=5000):


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=12, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head

        # self.norm_k = nn.LayerNorm(dim_head)
        # self.norm_q = nn.LayerNorm(dim_head)
        # self.norm_v = nn.LayerNorm(dim_head)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(12, inner_dim, bias = False)

        self.query_matrix = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.key_matrix = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.value_matrix = nn.Linear(self.dim_head, self.dim_head, bias=False)

        self.to_out = nn.Sequential(
            # nn.Linear(inner_dim, dim),
            nn.Linear(self.dim_head, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.norm_ini = nn.LayerNorm(12)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, value=None, key=None, query=None):
        # batch_size = vk.size(0)
        # seq_length = vk.size(2)
        # print(vk.shape)

        # value, key = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), vk)

    #     # key = key.view(batch_size, seq_length, self.heads, self.dim_head)
    #     value = value.view(batch_size, seq_length, self.heads, self.dim_head)
    #     # query = query.view(batch_size, seq_length, self.heads, self.dim_head)

        # key = self.norm_k(key)
        # query = self.norm_q(query)
        # value = self.norm_v(value)
        if value == None or key == None or query == None:
            x = self.norm_ini(x)
            # road grid matrix x flatten vectors, 64 x 96
            qkv = self.to_qkv(x).squeeze().chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=1), qkv)
        else:
            k = self.key_matrix(key)
            q = self.query_matrix(query)
            v = self.value_matrix(value)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        return self.norm(self.to_out(out))
        # out = rearrange(out, 'b h n d -> b n ()')



class FFTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., step=0):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.step = step

        self.residual = nn.Linear(12, dim)
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, value=None, key=None, query=None):
        for i, (attn, ff) in enumerate(self.layers):
            res_x = self.residual(x)
            x = attn(x, value, key, query) + res_x
            x = ff(x) + x

        return self.norm(x)


class TrajectoryModel(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes, out_list, periods, depth, mlp_dim, heads=3, dim_head=8, dropout=0., device="gpu", 
                 model_loc=None, model_rn=None, is_rn=True, model_name="trajectory_model"):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_name = model_name
        self.out_list = out_list
        self.device = device

        if is_rn:
            if model_rn != None:
                ### Pretrained & Early Fusion
                self.model_rn = model_rn
            else:
                self.model_rn = RNTransformer(node_features=7, num_nodes=num_nodes, periods=periods, output_dim_list=out_list, device=device).to(device)

        if self.model_name == "social_stgcnn":
            if model_loc != None:
                self.model_loc = model_loc
            else:
                self.model_loc = social_stgcnn(1, 5, output_feat=5, seq_len=in_channels, pred_seq_len=out_channels, num_nodes=num_nodes).to(device)
        elif self.model_name == "trajectory_model":
            self.model_loc = LocalPedsTrajNet(seq_len=out_channels, num_nodes=num_nodes, device=device).to(device)
        elif self.model_name == "social_implicit":
            self.model_loc = SocialImplicit(temporal_output=out_channels, num_nodes=num_nodes).to(device)
        elif self.model_name == "social_lstm":
            self.model_loc = SocialModel(seq_len=out_channels, is_rn=is_rn).to(device)
        # elif self.model_name == "trajnet++":
        #     self.model_loc = TrajNetPlusPlus(num_nodes=num_nodes, device=device).to(device)

        # dim_head = self.out_channels
        self.is_rn = is_rn
        # self.rn_out_dim = sum(out_list)
        # ff_in_channels = sum(out_list) * out_channels 
        # self.ff_transformer_loc = FFTransformer(dim=ff_in_channels, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout, step=0).to(device)
        # self.local_mlp_head1 = torch.nn.Linear(self.rn_out_dim * 5 * 2, self.rn_out_dim * 2) # Cross-attention
        # self.local_mlp_head2 = torch.nn.Linear(self.rn_out_dim * 2, 5) # Cross-attention
        # self.residual = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
        # self.norm = nn.LayerNorm(5)
        '''
        if self.is_rn:
            ### First argument should be the minimum size of road network
            # self.conv_rn = torch.nn.Conv1d(num_nodes, 2, kernel_size=1)
            self.conv_rn = torch.nn.Conv1d(num_nodes, 5, kernel_size=1)
            self.rn_in_dim = 8
            ff_in_channels = self.rn_out_dim * out_channels 
            # ff_in_channels = self.rn_out_dim * in_channels 
            # ff_in_channels = self.rn_out_dim * in_channels # If only use encoder from the local model
            self.ff_transformer_loc = FFTransformer(dim=ff_in_channels, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout, step=0).to(device)
            # self.ff_transformer_rn = FFTransformer(dim=ff_in_channels, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout, step=0).to(device)
            # self.ff_transformer = FFTransformer(dim=ff_in_channels, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout, step=1).to(device)

            # self.local_mlp_head = torch.nn.Linear(self.rn_out_dim * 5 * 2, 5) # Cross-attention
            # self.rn_mlp_head = torch.nn.Linear(self.rn_out_dim * 1, self.rn_out_dim * 1)
            # self.rn_mlp_head = torch.nn.Linear(self.rn_out_dim * 5 * 2, self.rn_out_dim * 1)

            self.residual = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
            self.linear_rn = torch.nn.Linear(13, out_channels)

            self.rn_norm = nn.LayerNorm(out_channels)
            self.norm = nn.LayerNorm(5)
        '''

        # self._reset_parameters()
        if self.is_rn:
            self.linear_rn = torch.nn.Linear(13, out_channels)

            self.rn_norm = nn.LayerNorm(out_channels)


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def compute_l2_loss(self, w):
        return torch.square(w).sum()

    def forward(self, V, A, x=None, rn_edge_index=None, rn_edge_attr=None, step=None, ped_list=None, h_=None, KSTEPS=20):

        h = None
        rn_pred = None
        traj_pred = None
        rn_out = [_ for _ in range(3)]

        ### Mid Fusion
        if self.is_rn:
            ### Road Network
            if h_ is not None:
                # rn_pred = self.model_rn(x, rn_edge_index, rn_edge_attr, h=h_.clone())
                rn_pred, rn_out[0], rn_out[1], rn_out[2] = self.model_rn(x, rn_edge_index, rn_edge_attr, h=h_.clone())
            else:
                # rn_pred = self.model_rn(x, rn_edge_index, rn_edge_attr)
                rn_pred, rn_out[0], rn_out[1], rn_out[2] = self.model_rn(x, rn_edge_index, rn_edge_attr)

            ## Flatten the output from RoadNetwork
            # out_rn = torch.cat((rn_pred[0], rn_pred[1], rn_pred[2]), axis=1)
            out_rn = torch.cat((rn_out[0], rn_out[1], rn_out[2]), axis=1)

            # h = F.relu(self.conv_rn(out_rn.unsqueeze(0)))
            out_rn = self.rn_norm(self.linear_rn(out_rn))
            
            h = out_rn.clone()

        ### Local Pedestrian Trajectory Network
        if h is not None:
            if self.model_name == "social_stgcnn":
                traj_pred, _, _ = self.model_loc(V, A.squeeze(), h=h)
            elif self.model_name == "social_lstm":
                traj_pred, _, _ = self.model_loc(V, ped_list, h=h)
            else:
                traj_pred = self.model_loc(V, A.squeeze(), h=h)
        else:
            if self.model_name == "social_stgcnn":
                traj_pred, mid_loc, _ = self.model_loc(V, A.squeeze())
            elif self.model_name == "social_lstm":
                traj_pred, _, _ = self.model_loc(V, ped_list)
            else:
                traj_pred = self.model_loc(V, A.squeeze())


        return rn_out, traj_pred
        # return rn_pred, traj_pred
