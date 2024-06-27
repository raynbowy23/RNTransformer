import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from torch_geometric.nn.conv import GCNConv, GATConv, SAGEConv


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
        baseblock: str = "gcn",
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
        self.baseblock = baseblock

        if self.baseblock == "gcn":
            self.BASEBLOCK = GCNConv
        elif self.baseblock == "gat":
            self.BASEBLOCK = GATConv
        elif self.baseblock == "graphsage":
            self.BASEBLOCK = SAGEConv
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        if self.baseblock == "gcn":
            self.conv_z = self.BASEBLOCK(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                improved=self.improved,
                cached=self.cached,
                add_self_loops=self.add_self_loops,
            )
        else:
            self.conv_z = self.BASEBLOCK(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
            )


        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):

        if self.baseblock == "gcn":
            self.conv_r = self.BASEBLOCK(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                improved=self.improved,
                cached=self.cached,
                add_self_loops=self.add_self_loops,
            )
        else:
            self.conv_r = self.BASEBLOCK(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
            )

        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):

        if self.baseblock == "gcn":
            self.conv_h = self.BASEBLOCK(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                improved=self.improved,
                cached=self.cached,
                add_self_loops=self.add_self_loops,
            )
        else:
            self.conv_h = self.BASEBLOCK(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
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


class ConvTemporalGraphical(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            periods,
            num_nodes: int,
            improved: bool = False,
            cached: bool = False,
            add_self_loops: bool = True,
            device: str = "cpu"
        ):
        super(ConvTemporalGraphical, self).__init__()

        self.kernel_size = kernel_size
        self.periods = periods

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.device = device

        self.num_nodes = num_nodes
        self.tgcn = TGCN(
            in_channels=2,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        # self.tgcn2 = TGCN(
        #     in_channels=2,
        #     out_channels=self.out_channels,
        #     improved=self.improved,
        #     cached=self.cached,
        #     add_self_loops=self.add_self_loops,
        # )


        # self.linear = nn.Linear(-1, self.num_nodes)
        self.relu = nn.ReLU()

        # self.conv = nn.Conv2d(self.periods, 12, 3, padding=1)

        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))

        self.rn_w = torch.nn.Parameter(torch.empty(24, device=device))
        self.loc_w = torch.nn.Parameter(torch.empty(24, device=device))


    def forward(self, x, A, h=None):
        probs = torch.nn.functional.softmax(self._attention, dim=0)

        H_accum = 0

        for i in range(self.in_channels): # For each input seq

            edge_index = (A[i] > 0).nonzero().t().to(self.device)
            row, col = edge_index
            # edge_weight = A[ped][row, col].to(self.device)
            edge_weight = None
            out = self.tgcn(
                x.squeeze().permute(0, 2, 1)[:, :, i].reshape(-1, x.size(1)), edge_index, edge_weight
            )
            # out = self.tgcn2(
            #     out, edge_index, edge_weight, h
            # )

            # H_accum += self.attention(out, i)[0]
            H_accum += self.loc_w * probs[i] * out + self.rn_w * h.reshape(24)

        # for period in range(self.in_channels):
        #     H_accum = H_accum + self.conv(x[:, :, period], edge_index[period], edge_weight[period])

        # H_accum = H_accum.view(1, self.periods, -1, x.size(-1))
        # H_accum = self.relu(H_accum).view(1, -1, 12, x.size(-1))
        H_accum = H_accum.view(1, self.out_channels, -1, x.size(-1))

        H_accum = self.relu(H_accum).view(1, -1, self.out_channels, x.size(-1))
        return H_accum


class TemporalGCN(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        periods: int,
        num_nodes: int,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
        stride: int=1,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        device: str = "cpu"):
        super(TemporalGCN, self).__init__()

        padding = ((kernel_size[0] - 1) // 2, 0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.device = device

        self.gcn = ConvTemporalGraphical(
            # in_channels=self.in_channels,
            # out_channels=self.out_channels,
            in_channels=8,
            # out_channels=240,
            out_channels=24,
            kernel_size=kernel_size[1],
            periods=self.periods,
            num_nodes=num_nodes,
            device=self.device,
        )

        self.conv = nn.Conv2d(
            # int(240 / 12),
            2,
            # int(240 / 8),
            5,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )

        # self.tcn = nn.Sequential(
        #     # nn.BatchNorm2d(out_channels),
        #     nn.BatchNorm2d(4),
        #     nn.PReLU(),
        #     nn.Conv2d(
        #         # out_channels,
        #         # out_channels,
        #         4,
        #         5,
        #         (kernel_size[0], 1),
        #         (stride, 1),
        #         padding,
        #     )
        # )

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                # out_channels,
                5,
                kernel_size=1,
                stride=(stride, 1)
            ),
            # nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                in_channels=5,
                out_channels=5,
                kernel_size=(3, 3),
                # kernel_size=(1, 3),
                # kernel_size=(1, 1),
                stride=(1, 1),
                # padding=(2, 1),
                padding=(3, 1),
            ),
            nn.BatchNorm2d(5),
        )

        self.prelu = nn.PReLU()


    def forward(self, x, A, h=None):

        res = self.residual(x) # (1, 5, 12, ped_num) for Idea 2
        # res = self.residual(x) # (1, 5, 8, ped_num) for Idea 3

        x = self.gcn(x, A, h).reshape(1, -1, self.periods, x.size(-1)) 
        # x = self.gcn(x, A, h).reshape(1, -1, 8, x.size(-1)) 
        x = self.conv(x) + res # Will be replaced by einsum?
        # h = F.leaky_relu(h)
        # x = self.tcn(x) # + res

        return x, A


class LocalPedsTrajNet(nn.Module):
    def __init__(
            self,
            in_channels=2,
            out_channels=16,
            seq_len=10,
            pred_seq_len=12,
            num_nodes=16,
            kernel_size=3,
            fusion=None,
            temporal=None,
            device="cpu"):
        super(LocalPedsTrajNet, self).__init__()
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len
        self.in_channels = in_channels
        self.out_channels = out_channels

        # hidden_dim = 128

        self.n_gcn = 1

        self.tgcn = nn.ModuleList()
        for i in range(self.n_gcn):
            self.tgcn.append(
                TemporalGCN(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=(kernel_size, self.seq_len),
                    periods=self.seq_len,
                    num_nodes=num_nodes,
                    device=device,
                ))

        self.relu1 = nn.ReLU()
        # self.linear1 = nn.Linear(4 * 8, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, self.out_channels)
        self.output = nn.Conv2d(5, 5, 3, padding=1)
    

    def forward(self, v, A, h=None, C=None):

        # h = self.tgcn_seq(x, x_prev, edge_index, edge_weight)
        for k in range(self.n_gcn):
            h = self.relu1(self.tgcn[k](v, A, h)[0])

        h = self.output(h)

        # h = self.relu2(self.linear1(h.reshape(-1, 4 * 8)))
        # h = self.linear2(h)
        return h
