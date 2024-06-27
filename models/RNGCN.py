import math
from icecream import ic
from einops import rearrange, repeat

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.nn.inits import glorot, zeros


class TGCN(torch.nn.Module):
    r"""An implementation of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super(TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_z = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_r = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_h = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class TemporalGraphConvNeuralNetwork(torch.nn.Module):
    r"""    
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        periods: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True
    ):
        super(TemporalGraphConvNeuralNetwork, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.num_nodes = num_nodes
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._setup_layers()

    def _setup_layers(self):
        # self.conv = GCNConv(
        #     in_channels=self.in_channels,
        #     out_channels=self.out_channels,
        #     improved=self.improved,
        #     cached=self.cached,
        #     add_self_loops=self.add_self_loops,
        # )
        # self.gru = torch.nn.GRU(
        #     input_size=self.out_channels,
        #     hidden_size=self.out_channels
        # )
        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))

    #     self._weight_att1 = torch.nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(self.out_channels, 1), requires_grad=True))
    #     self._weight_att2 = torch.nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(self.num_nodes, 1), requires_grad=True))
    #     self._bias_att1 = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(1, 1), requires_grad=True))
    #     self._bias_att2 = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(1, 1), requires_grad=True))

        # self.conv_first = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(1, 1), padding=(2, 0))


    # def attention(self, x, period) -> torch.FloatTensor:
    #     '''
    #     Attention module from original code implementation
    #     '''
    #     input_x = x
    #     x = torch.matmul(torch.reshape(x, [-1, self.out_channels]), self._weight_att1) + self._bias_att1 # [num_nodes, 1]

    #     f = torch.matmul(torch.reshape(x, [-1, self.num_nodes]), self._weight_att2) + self._bias_att2 # [1, 1]
    #     g = torch.matmul(torch.reshape(x, [-1, self.num_nodes]), self._weight_att2) + self._bias_att2
    #     h = torch.matmul(torch.reshape(x, [-1, self.num_nodes]), self._weight_att2) + self._bias_att2

    #     f1 = f.squeeze(0).expand(self.periods)
    #     h1 = h.squeeze(0).expand(self.periods)
    #     g1 = g.squeeze(0).expand(self.periods)
    #     s = g1 * f1

    #     beta = torch.nn.functional.softmax(s, dim=-1)

    #     context = beta[period] * input_x

    #     return context, beta    

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        # if H is None:
        #     H = torch.zeros(X.shape[0], 8).to(X.device)
        # else:
        #     self.weight = nn.Parameter(torch.randn(16, H.size(2) * H.size(3)), requires_grad=True).to(X.device)
        #     # print(self.conv_first(H.permute(0, 2, 1, 3)).shape)
        #     # [36, out_channels, 1, 1]
        #     H = F.conv2d(H, weight=nn.Parameter(torch.randn(6*6, 12, 1, 1), requires_grad=True).to(X.device)).reshape(1, 6*6, -1)
        #     # H = self.conv_first(H).reshape(1, 16, -1)
        #     H = F.linear(H, self.weight)
        #     # H = F.linear(H, (16, H.size(-1)))
        #     # H = H.view(1, 16, 16)
        #     # transformed_tensor = H.permute(0, 2, 1, 3).contiguous().view(1, -1, 5)
        #     # H = transformed_tensor.view(1, 12, -1)
        # weight = nn.Parameter(torch.randn(H.size(1), 24), requires_grad=True).to(H.device)

        for period in range(self.periods):
            # H_accum = H_accum + self.conv(X[:, :, period], edge_index[period], edge_weight[period])
        #     # out = self._base_tgcn(
        #     #     X[:, :, period], edge_index[period], edge_weight[period], H
        #     # )
        #     # H_accum = H_accum + self.attention(out, period)[0]

            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index[period], edge_weight[period]
            )

        #     # out = self.conv(X[:, :, period], edge_index[period], edge_weight[period]).unsqueeze(0)
        #     # out, h = self.gru(F.linear(torch.cat((self.conv(X[:, :, period], edge_index[period], edge_weight[period]), H), dim=-1), weight))
        #     # H_accum = H_accum + probs[period] * out
        #     # out, h = self.gru(self.conv(X[:, :, period], edge_index[period], edge_weight[period]))
        #     # H_accum = H_accum + probs[period] * out

        return H_accum


class RoadNetworkGCN(torch.nn.Module):
    """This is the neural network for one horizon of road network graph.

    Args:
        torch (_type_): _description_
        output_dim: time horizon

    Returns:
        _type_: _description_
    """
    def __init__(self, node_features, num_nodes, periods, output_dim):
        super(RoadNetworkGCN, self).__init__()
        # Attention Temporal Graph Convolutional Cell + Regional concat feature
        # hidden_dim = 256
        # out_channels = 512
        hidden_dim = 8
        out_channels = 16
        self.periods = periods
        self.num_nodes = num_nodes
        self.output_dim = output_dim

        self.tgnn = TemporalGraphConvNeuralNetwork(
                        in_channels=node_features, 
                        out_channels=out_channels,
                        num_nodes=self.num_nodes,
                        periods=self.periods)

        self.after_tgnn = nn.Sequential(
                    nn.Linear(out_channels, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.output_dim),
                    nn.ReLU()
                )

    def forward(self, x, edge_index, edge_attr=None, h=None):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index, edge_attr, H=h).squeeze()
        h = self.after_tgnn(h)
        return h


class FeedForward(nn.Module):
    def __init__(self, embed_dim=192, hidden_dim=128, dropout=0.):
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x.clone())


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=96, n_heads=3):
        """Multi-head attention module for different horizons of road network.

        Args:
            embed_dim (int, optional): _description_. Defaults to 512.
            n_heads (int, optional): _description_. Defaults to 3.
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads) # 192 / 3 = 64

        self.scale = self.single_head_dim ** -0.5

        inner_dim = self.n_heads * self.single_head_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.to_qkv = nn.Linear(embed_dim, inner_dim*3, bias=False)

        self.dr = 0.
        self.dropout = nn.Dropout(self.dr)

        project_out = not (self.n_heads == 1 and self.single_head_dim == embed_dim)

        # Key, query and value matrices 64 x 64
        # self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, self.embed_dim),
            nn.Dropout(self.dr)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        """Batch_size x sequence_length x embedding_dim. 32 x 10 x 192

        Args:
            key (_type_): key vector
            query (_type_): query vector
            value (_type_): value vector
            mask (_type_, optional): mask for decoder. Defaults to None.
        Returns:
            output vector from multihead attention
        """

        x = self.norm(x)
        # road grid matrix x flatten vectors, 64 x 96
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=self.n_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'h n d -> n (h d)')
        return self.out(out)
        

class RNTransformer(nn.Module):
    def __init__(self, node_features, num_nodes, periods, mlp_dim=64, dr=0., output_dim_list=[8, 16], device="cpu"):
        """ Road Network Transformer

        Args:
            node_features (int): Node features. Typically 8 for sdd
            num_nodes (int): Number of nodes
            periods (int): Input dim
            output_dim_list (List<int>): List of time horizons we consider. Should be numerical ascending order.
            mlp_dim (int): Hidden dimension for FeedForward
            n_horizon (int): The number of time horizons. Size of output_dim_list
            dr (float): Dropout rate
        """
        super(RNTransformer, self).__init__()

        hidden_dim = 256
        # self.out_channels = 512
        self.periods = periods
        self.num_nodes = num_nodes
        self.output_dim_list = output_dim_list
        self.n_horizon = len(output_dim_list)
        self.flatten = [_ for _ in range(self.n_horizon)]

        # assert isinstance(self.output_dim_list, list)
        for i in range(0, self.n_horizon):
        #     self.output_dim_list[i] = int(self.output_dim_list[i])
            # self.flatten[i] = nn.Linear(self.output_dim_list[i], 64).to(device) # Adjust features into same size
            self.flatten[i] = nn.Linear(self.output_dim_list[i], 4).to(device) # Adjust features into same size
        # self.output_dim_list.sort()
        
        self.rngcn = nn.ModuleList()
        self.mlp_head = nn.ModuleList()
        for i in range(self.n_horizon):
            self.rngcn.append(
                RoadNetworkGCN(
                    node_features=node_features,
                    num_nodes=self.num_nodes,
                    periods=self.periods,
                    output_dim=self.output_dim_list[i]
                )
            )
            # self.mlp_head.append(nn.Linear(64 * self.n_horizon, self.output_dim_list[i]))
            head_seq = nn.Sequential(
                nn.Linear(4*self.n_horizon, self.output_dim_list[i]),
                # nn.ReLU(),
                # nn.BatchNorm1d(self.output_dim_list[i])
            )
            self.mlp_head.append(head_seq)

        self.depth = 1 # Temporaly

        ### For Transformer
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                # MultiHeadAttention(embed_dim=64*self.n_horizon, n_heads=self.n_horizon).cuda(),
                MultiHeadAttention(embed_dim=4*self.n_horizon, n_heads=self.n_horizon).to(device),
                FeedForward(embed_dim=4*self.n_horizon, hidden_dim=mlp_dim, dropout=dr).to(device)
            ]))

        self.rngcn_out = [_ for _ in range(self.n_horizon)]

        self.to_latent = nn.Identity()

    def forward(self, x, edge_index, edge_attr, h=None):
        # Input should be three different timesteps (12, 24, 36) --> Different number of grids
        # Ouput also should be three different timesteps (12, 24, 36)

        # rn_out = self.rngcn[0](x[0], edge_index[0], edge_attr[0], h=h)
        # road grid (8x8) and time horizon, 64x2, 64x4, 64x8
        # rn_out = self.rngcn[0](x[0], edge_index[0], edge_attr[0], h=h)
        # rn_out = self.rngcn[0](x, edge_index, edge_attr, h=h)

        for i in range(self.n_horizon):
            # Needs to be flatten
            self.rngcn_out[i] = self.flatten[i](self.rngcn[i](x[i], edge_index[i], edge_attr[i], h=h))
            # self.rngcn_out[i] = self.flatten[i](self.rngcn[i](x, edge_index, edge_attr))
        rn_out = torch.cat(self.rngcn_out, axis=1)


        # Pos embedding?
        # Dropout?

        ### Transformer -> No needed?
        for attn, ff in self.layers:
           rn_out = attn(rn_out) + rn_out 
           rn_out = ff(rn_out) + rn_out

        # Make head to decoding to original header size
        rn_out = self.to_latent(rn_out)
        # return self.rngcn_out, rn_out

        return self.rngcn_out, self.mlp_head[0](rn_out), self.mlp_head[1](rn_out), self.mlp_head[2](rn_out)
        # return self.rngcn_out, self.mlp_head[0](rn_out), self.mlp_head[1](rn_out), self.mlp_head[2](rn_out)
       

