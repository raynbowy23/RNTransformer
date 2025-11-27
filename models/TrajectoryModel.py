import torch
import torch.nn as nn
from einops import rearrange

from models.SocialStgcnn import social_stgcnn
from models.SocialImplicit import SocialImplicit
from models.SocialLSTM import SocialModel
from models.RNGCN import RNTransformer

import logging

logger = logging.getLogger(__name__)

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

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(12, inner_dim, bias = False)

        self.query_matrix = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.key_matrix = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.value_matrix = nn.Linear(self.dim_head, self.dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.dim_head, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.norm_ini = nn.LayerNorm(12)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, value=None, key=None, query=None):

        if value == None or key == None or query == None:
            x = self.norm_ini(x)
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

    def __init__(self, opt, num_nodes, out_list, device="gpu", model_loc=None):
        super().__init__()

        self.in_channels = opt.num_timesteps_in
        self.out_channels = opt.num_timesteps_out
        self.model_name = opt.model_name
        self.periods = opt.num_timesteps_in
        self.out_list = out_list
        self.device = device
        self.grid = 8 # opt.grid
        self.is_rn = opt.is_rn

        if self.is_rn:
            self.model_rn = RNTransformer(node_features=8, num_nodes=num_nodes, periods=self.periods, output_dim_list=out_list, device=device).to(device)

        if self.model_name == "social_stgcnn":
            if model_loc != None:
                self.model_loc = model_loc
            else:
                self.model_loc = social_stgcnn(1, 5, output_feat=5, seq_len=self.in_channels, pred_seq_len=self.out_channels, num_nodes=num_nodes).to(device)
        elif self.model_name == "social_implicit":
            self.model_loc = SocialImplicit(temporal_output=self.out_channels, num_nodes=num_nodes).to(device)
        elif self.model_name == "social_lstm":
            self.model_loc = SocialModel(seq_len=self.out_channels, is_rn=self.is_rn).to(device)
        self.is_rn = self.is_rn
      
        if self.is_rn:
            # self.rn_proj = torch.nn.Linear(len(out_list) * 16, out_channels) # For rn_embed, 16 is hidden dim of RN
            self.rn_proj = torch.nn.Linear(13, self.out_channels) # For rn_pred, sum of out_list

            self.rn_norm = nn.LayerNorm(self.out_channels)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def compute_l2_loss(self, w):
        return torch.square(w).sum()

    def forward(self, V, A, ped_list=None, precomputed_rn=None):

        rn_embed = None
        rn_pred = None

        if self.is_rn:
            ### Road Network
            if precomputed_rn is not None:
                if len(precomputed_rn) == 2:
                    rn_pred, rn_embed = precomputed_rn
                else:
                    rn_pred, rn_embed, _ = precomputed_rn
                rn_embed = rn_embed.detach().to(self.device)

        ### Local Pedestrian Trajectory Network
        if self.model_name == "social_stgcnn":
            traj_pred, _, _ = self.model_loc(V, A.squeeze(), h=rn_embed)
        elif self.model_name == "social_lstm":
            traj_pred, _, _ = self.model_loc(V, ped_list, h=rn_embed)
        else:
            traj_pred, rn_fused = self.model_loc(V, A.squeeze(), rn_embed=rn_embed)
        return rn_pred, traj_pred, rn_fused
