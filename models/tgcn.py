import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_mean
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


class TGCN_LSTM(torch.nn.Module):
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
        super(TGCN_LSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._create_parameters_and_layers()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_i = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_i = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_f = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_f = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_cell_gate_parameters_and_layers(self):

        self.conv_g = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_g = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_output_gate_parameters_and_layers(self):

        self.conv_o = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_o = torch.nn.Linear(2 * self.out_channels, self.out_channels)
        
    def _create_parameters_and_layers(self):
        self._create_cell_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_input_gate_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H):
        I = torch.cat([self.conv_i(X, edge_index, edge_weight), H], axis=1)
        I = self.linear_i(I)
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H):
        F = torch.cat([self.conv_f(X, edge_index, edge_weight), H], axis=1)
        F = self.linear_f(F)
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_gate(self, X, edge_index, edge_weight, H):
        G = torch.cat([self.conv_g(X, edge_index, edge_weight), H], axis=1)
        G = self.linear_g(G)
        G = torch.tanh(G)
        return G

    def _calculate_output_gate(self, X, edge_index, edge_weight, H):
        O = torch.cat([self.conv_o(X, edge_index, edge_weight), H], axis=1)
        O = self.linear_o(O)
        O = torch.sigmoid(O)
        return O

    def _calculate_cell_state(self, F, C, I, G):
        C = F * C + I * G
        return C

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

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
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """

        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H)
        G = self._calculate_cell_gate(X, edge_index, edge_weight, H)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H)
        C = self._calculate_cell_state(F, C, I, G)
        H = self._calculate_hidden_state(O, C)
        return O, (H, C)


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
        self.conv = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.linear = torch.nn.Linear(512*5, 512)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        # self._weight_att = torch.nn.Parameter(torch.normal(mean=0.0, std=0.1, out=self.periods, device=device))
        # self._weight_att2 = torch.nn.Parameter(torch.normal(mean=0.0, std=0.1, out=self.periods, device=device))
        # self._bias_att = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, out=self.periods, device=device))
        # self._bias_att2 = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, out=self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def self_attention(self, x):
        x = torch.matmul(torch.reshape(x, [-1, self.periods]), self._weight_att) + self._bias_att
        f = torch.matmul(torch.reshape(x, [-1, self.out_channels]), self._weight_att2) + self._bias_att2
        g = torch.matmul(torch.reshape(x, [-1, self.out_channels]), self._weight_att2) + self._bias_att2
        h = torch.matmul(torch.reshape(x, [-1, self.out_channels]), self._weight_att2) + self._bias_att2

        f1 = torch.reshape(f, [-1, self.in_channels])
        g1 = torch.reshape(g, [-1, self.in_channels])
        h1 = torch.reshape(h, [-1, self.in_channels])
        s = g1 * f1
        print('s', s)

        beta = torch.nn.Softmax(s, dim=-1) # attention map
        print('beta', beta)
        context = beta.expand(2) * torch.reshape(x, [-1, self.in_channels, self.out_channels])

        context = torch.permute(context, (0,2,1))
        print('context', context)

        return context, beta

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
        # probs, alpha = self.self_attention(X)
        for period in range(self.periods):
            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index, edge_weight, H
            )
        return H_accum


class TimeHorizonGCN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TimeHorizonGCN, self).__init__()
        # Attention Temporal Graph Convolutional Cell + Regional concat feature
        self.tgnn = TemporalGraphConvNeuralNetwork(in_channels=node_features, 
                        #    out_channels=512, 
                           out_channels=32, 
                           periods=periods)
        # Equals single-shot prediction
        hidden_dim = 16
        self.linear1 = torch.nn.Linear(32, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, periods)
        self.relu = nn.ReLU()
        # self.norm = GraphNorm(in_channels=node_features)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = self.relu(h)
        h = self.linear1(h)
        h = self.relu(h)
        h = self.linear2(h)
        return h
