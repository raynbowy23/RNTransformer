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
        H_accum = X.new_zeros(X.size(0), self.out_channels)
        probs = torch.nn.functional.softmax(self._attention, dim=0)

        for period in range(self.periods):
            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index, edge_weight
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
        h = self.tgnn(x, edge_index, edge_attr)
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

        self.out = nn.Sequential(
            nn.Linear(inner_dim, self.embed_dim),
            nn.Dropout(self.dr)
        ) if project_out else nn.Identity()

    # def forward(self, x, mask=None):
    #     """Batch_size x sequence_length x embedding_dim. 32 x 10 x 192

    #     Args:
    #         key (_type_): key vector
    #         query (_type_): query vector
    #         value (_type_): value vector
    #         mask (_type_, optional): mask for decoder. Defaults to None.
    #     Returns:
    #         output vector from multihead attention
    #     """
    #     bn, dim = x.shape[0], x.shape[1]

    #     x = self.norm(x)
    #     qkv = self.to_qkv(x) # .chunk(3, dim=-1)
    #     q, k, v = torch.split(qkv, dim, dim=1)

    #     attn_weights = torch.softmax(q @ k.transpose(0,1) / (dim**0.5), dim=-1)
    #     out = attn_weights @ v # shape [bsz, embed_dim]
    #     return self.out(out)

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

        self.periods = periods
        self.num_nodes = num_nodes
        self.output_dim_list = output_dim_list
        self.n_horizon = len(output_dim_list)
        self.hidden_dim = 16
        self.device = device

        self.flatten = nn.ModuleList([
            nn.Linear(max(self.output_dim_list), 4) for _ in range(self.n_horizon)
        ])
        
        self.rngcn = RoadNetworkGCN(
            node_features=node_features,
            num_nodes=self.num_nodes,
            periods=self.periods,
            output_dim=self.hidden_dim
        )

        self.mlp_heads = nn.ModuleList([
            nn.Sequential(
                # nn.LayerNorm(self.hidden_dim * self.n_horizon),
                # nn.Linear(self.hidden_dim * self.n_horizon, out_dim)
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, out_dim)
            ) for out_dim in self.output_dim_list
        ])

        self.depth = 1 # Temporaly

        ### Transformer
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=1, # simple since horizons are short
                dim_feedforward=mlp_dim,
                dropout=dr,
                batch_first=True
            )
        ])
        # self.layers = nn.ModuleList([])
        # for _ in range(self.depth):
        #     self.layers.append(nn.ModuleList([
        #         MultiHeadAttention(embed_dim=self.hidden_dim*self.n_horizon, n_heads=self.n_horizon).to(device),
        #         FeedForward(embed_dim=self.hidden_dim*self.n_horizon, hidden_dim=mlp_dim, dropout=dr).to(device)
        #     ]))

        self.to_latent = nn.Identity()

        self.horizon_proj = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_horizon)
        ])

    def forward(self, x_list, edge_index_list, edge_attr_list, h=None):
        rngcn_out = []

        horizon_embeddings = []
        for i in range(self.n_horizon):
            x_encoded = self.rngcn(x_list[i], edge_index_list[i], edge_attr_list[i], h=h) # [num_nodes, hidden_dim]
            h_proj = self.horizon_proj[i](x_encoded)
            # h_proj = self.flatten[i](x_encoded)
            horizon_embeddings.append(h_proj)

        rn_emb = torch.stack(horizon_embeddings, dim=1) # [num_nodes, n_horizon, hidden_dim]
        # rn_emb = torch.cat(horizon_embeddings, axis=1)

        for layer in self.layers:
            rn_emb = layer(rn_emb)

        # print(rn_out.shape)
        # for attn, ff in self.layers:
        #     rn_emb = attn(rn_emb) + rn_emb
        #     rn_emb = ff(rn_emb) + rn_emb
        rn_emb = self.to_latent(rn_emb)

        for i, head in enumerate(self.mlp_heads):
            horizon_pred = head(rn_emb[:, i, :]) # [num_nodes, out_dim]
            # horizon_pred = head(rn_emb[i]) # [num_nodes, out_dim]
            rngcn_out.append(horizon_pred)

        return rngcn_out, rn_emb


class RNAdapter(nn.Module):
    def __init__(self, hidden_dim=16, n_horizon=3, n_grids=8, out_dim=32, mode="grid"):
        super().__init__()
        self.mode = mode
        self.n_grids = n_grids
        self.n_horizon = n_horizon
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        if self.mode == "grid":
            self.proj_x = nn.Linear(n_horizon * hidden_dim, out_dim)
            self.proj_y = nn.Linear(n_horizon * hidden_dim, out_dim)
        elif self.mode == "attention":
            self.node_proj = nn.Linear(hidden_dim, out_dim)
            self.agent_proj = nn.Linear(out_dim, out_dim)
        
    def forward(self, rn_embed, agent_state=None):
        # bn = agent_state.size(0) if agent_state is not None else 1
        bn = 1
        rn_embed = rn_embed.view(bn, self.n_grids, self.n_grids, self.n_horizon, self.hidden_dim)

        if self.mode == "grid":
            # collapse nodes into 8 spatial grids of ~8 nodes each
            grid_feat = rn_embed.mean(dim=2)               # [B, 8, 3, 16]
            grid_feat = grid_feat.flatten(2)               # [B, 8, 48]

            h_x = self.proj_x(grid_feat)                       # [B, 8, 49]
            h_y = self.proj_y(grid_feat)                       # [B, 8, 49]

            h = torch.stack((h_x, h_y), dim=1)             # [B, 8, 2, 49]

            return h.view(bn, 2, self.n_grids, self.out_dim)        # [B, 2, 8, 49]
        elif self.mode == "attention":
            # Average over horizons first: [B,64,16]
            rn_nodes = rn_embed.mean(dim=2)

            # project road nodes
            rn_nodes = self.node_proj(rn_nodes)            # [B,64,49]

            # agent_state expected: [B,1,2,8,49] → flatten to Q=[B,16,49]
            agent_q = agent_state.view(bn, 16, 49)
            agent_q = self.agent_proj(agent_q)             # optional learnable transform

            # attention: [B,16,49] × [B,49,64] → [B,16,64]
            attn_logits = torch.bmm(agent_q, rn_nodes.transpose(1,2))
            attn = torch.softmax(attn_logits / (49**0.5), dim=-1)

            # context: [B,16,64] × [B,64,49] → [B,16,49]
            context = torch.bmm(attn, rn_nodes)

            return context.view(bn, 2, 8, self.out_dim)   # [B,2,8,49]


class RNBlock(nn.Module):
    def __init__(self, temporal_input=8, ped_dim=8, rn_dim=16, mode="attention"):
        super().__init__()
        
        # Spatial adaptation layer
        # self.rn_spatial_adapt = nn.Linear(8, 8)
        # self.rn_spatial_adapt = nn.Conv1d(num_nodes, spatial_output, kernel_size=1)
        # self.rn_spatial_adapt = nn.Conv1d(spatial_output, spatial_output, kernel_size=1)
        self.mode = mode
        if self.mode == "grid":
            # Spatial adaptation layer
            self.rn_proj = nn.Linear(48, temporal_input)
        elif self.mode == "attention":
            self.attn = nn.MultiheadAttention(
                embed_dim=rn_dim,
                num_heads=2,
                batch_first=True
            )
            self.ped_proj = nn.Linear(ped_dim, rn_dim)
            self.out_proj = nn.Linear(rn_dim, temporal_input)
            self.temporal_input = temporal_input

    def forward(self, rn_embed, ped_feat):
        if self.mode == "grid":
            rn_flat = rn_embed.flatten(1) # [64, 48]
            rn_pool = rn_flat.mean(dim=0) # [48]
            rn_global = self.rn_proj(rn_pool) # [temporal_input]
            return rn_global
        # """
        # rn_temporal_features = self.rn_embed_proj(rn_flat) # [64, temporal_input=8]
        
        # # Method 1: Simple averaging across all road network nodes
        # rn_scene_temporal = rn_temporal_features.mean(dim=0) # [temporal_input=8]
        
        # # Broadcast to match pedestrian batch: [batch*peds, spatial, temporal]
        # rn_scene_temporal = rn_scene_temporal.unsqueeze(0).unsqueeze(0) # [1, 1, 8]
        # rn_scene_temporal = rn_scene_temporal.expand(local_batch, 2, -1) # [batch*peds, 2, 8]

        # # Apply spatial adaptation (now with correct dimensions)
        # rn_adapted = self.rn_spatial_adapt(rn_scene_temporal) # [batch*peds, 2, 8]
        # """
        # return rn_global
        # rn_embed: [64, 3, 16] → collapse time
        # rn_embed: [64, 3, 16]
        if self.mode == "attention":
            rn_nodes = rn_embed.mean(dim=1)    # [64, 16]
            rn_nodes = rn_nodes.unsqueeze(0)   # [1, 64, 16]

            bp = ped_feat.size(0)   # batch*peds

            # ped_feat: [bp, 2, 8] → reduce spatial channels
            ped_feat_avg = ped_feat.mean(dim=1)     # [bp, 8]

            # Project pedestrians into RN dimension (8 → 16)
            q = self.ped_proj(ped_feat_avg)         # [bp, 16]
            q = q.unsqueeze(1)                      # [bp, 1, 16]

            # Expand RN nodes for each pedestrian
            k = rn_nodes.expand(bp, -1, -1)         # [bp, 64, 16]
            v = k                                    # usually same

            # Cross-attention: ped queries attend to RN nodes
            attended, _ = self.attn(q, k, v)        # [bp, 1, 16]

            attended = attended.squeeze(1)          # [bp, 16]

            # Return final fused RN feature per pedestrian
            return attended.reshape(v.size(0), 2, self.temporal_input)         # [bp, 16]
