from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch


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

        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))

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

        for period in range(self.periods):
            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index[period], edge_weight[period]
            )

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
        # self.dropout = nn.Dropout(self.dr)

        project_out = not (self.n_heads == 1 and self.single_head_dim == embed_dim)

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
        bn, dim = x.shape[0], x.shape[1]

        x = self.norm(x)
        qkv = self.to_qkv(x) # .chunk(3, dim=-1)
        # q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=self.n_heads), qkv)
        q, k, v = torch.split(qkv, dim, dim=1)

        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # attn = F.softmax(dots, dim=-1)
        # attn = self.dropout(attn)

        # out = torch.matmul(attn, v)
        # out = rearrange(out, 'h n d -> n (h d)')
        attn_weights = torch.softmax(q @ k.transpose(0,1) / (dim**0.5), dim=-1)
        out = attn_weights @ v  # shape [bsz, embed_dim]
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

        self.periods = periods
        self.num_nodes = num_nodes
        self.output_dim_list = output_dim_list
        self.n_horizon = len(output_dim_list)
        self.device = device

        self.flatten = nn.ModuleList([
            nn.Linear(max(self.output_dim_list), 4) for _ in range(self.n_horizon)
        ])
        
        self.rngcn = RoadNetworkGCN(
            node_features=node_features,
            num_nodes=self.num_nodes,
            periods=self.periods,
            output_dim=max(self.output_dim_list)
        )
        self.mlp_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4*self.n_horizon, out_dim),
            )
            for out_dim in output_dim_list
        ])

        self.depth = 1 # Temporaly

        ### Transformer
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttention(embed_dim=4*self.n_horizon, n_heads=self.n_horizon),
                FeedForward(embed_dim=4*self.n_horizon, hidden_dim=mlp_dim, dropout=dr)
            ])
            for _ in range(self.depth)
        ])

        self.to_latent = nn.Identity()
        self.to(device)


    # def forward(self, x, edge_index, edge_attr, h=None):
        # # for i in range(self.n_horizon):
        # #     # Needs to be flatten
        # #     # self.rngcn_out[i] = self.flatten[i](self.rngcn[i](x[i], edge_index[i], edge_attr[i], h=h))
        # #     # feat = self.rngcn(x[i], edge_index[i], edge_attr[i], h=h)
        # #     self.rngcn_out[i] = self.flatten[i](feat)

        # rn_out = torch.cat(self.rngcn_out, dim=1)

        # # TODO: Add Pos embedding and Dropout

        # ## Transformer
        # for attn, ff in self.layers:
        #    rn_out = attn(rn_out) + rn_out 
        #    rn_out = ff(rn_out) + rn_out

        # # Make head to decoding to original header size
        # rn_out = self.to_latent(rn_out)

        # final_preds = []
        # for i in range(self.n_horizon):
        #     head_i = self.mlp_head[i](rn_out)
        #     final_preds.append(head_i)

        # return final_preds
       

    # def forward(self, x_list, edge_index_list, edge_attr_list, h=None):
    #     data_list =  []
    #     for i in range(self.n_horizon):
    #         data = Data(x=x_list[i], edge_index=edge_index_list[i], edge_attr=edge_attr_list[i])
    #         data_list.append(data)
    #     batched = Batch.from_data_list(data_list).to(x_list[0].device)

    #     feat = self.rngcn(batched.x, batched.edge_index, batched.edge_attr, h=h)

    #     batch = batched.batch
    #     self.rngcn_out = []
    #     for i in range(self.n_horizon):
    #         idx = (batch == i).nonzero(as_tuple=True)[0]
    #         feat_i = feat[idx]
    #         self.rngcn_out.append(self.flatten[i](feat_i))
    #     rn_out = torch.cat(self.rngcn_out, dim=1)
        
    #      # Transformer + MLP head
    #     rn_out = torch.cat(self.rngcn_out, dim=1)
    #     for attn, ff in self.layers:
    #         rn_out = attn(rn_out) + rn_out
    #         rn_out = ff(rn_out) + rn_out

    #     final_preds = []
    #     for i in range(self.n_horizon):
    #         head_i = self.mlp_head[i](rn_out)
    #         final_preds.append(head_i)
    #     return final_preds


    def forward(self, x_list, edge_index_list, edge_attr_list, h=None):
        self.rngcn_out = []
        for i in range(self.n_horizon):
            feat = self.rngcn(x_list[i], edge_index_list[i], edge_attr_list[i], h=h)
            self.rngcn_out.append(self.flatten[i](feat))

        rn_out = torch.cat(self.rngcn_out, dim=1)
        for attn, ff in self.layers:
            rn_out = attn(rn_out) + rn_out
            rn_out = ff(rn_out) + rn_out

        rn_out = self.to_latent(rn_out)
        return [head(rn_out) for head in self.mlp_head]    