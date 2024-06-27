import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple(nn.Module):
    def __init__(self, in_channels, out_channels, num_timesteps_in, is_horizontal_pred=True):
        super().__init__()
        self.is_horizontal_pred = is_horizontal_pred
        self.num_timesteps_in = num_timesteps_in
        self.hidden_channels = 200
        if self.is_horizontal_pred:
            self.in_channels = 2 # feature dim = xCenter, yCenter
            self.out_channels = 2
        else:
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.encoded_channels = 10

        # TODO: Want here to be sequential
        # self.encoder = nn.ModuleDict([
        #     ['lstm', nn.LSTM(input_size=self.in_channels, hidden_size=self.hidden_channels, batch_first=True)],
        #     ['linear1', nn.Linear(self.hidden_channels, self.hidden_channels)],
        #     ['linear2', nn.Linear(self.hidden_channels, self.encoded_channels)] 
        # ])

        # self.activations = nn.ModuleDict([
        #     ['relu', nn.ReLU()],
        # ])

        # self.decoder = nn.ModuleDict({
        #     'lstm': nn.LSTM(input_size=self.encoded_channels, hidden_size=self.hidden_channels, batch_first=True),
        #     'linear1': nn.Linear(self.hidden_channels, self.hidden_channels),
        #     'linear2': nn.Linear(self.hidden_channels, self.out_channels) 
        # })
        
        self.lstm = nn.LSTM(input_size=self.in_channels, hidden_size=self.hidden_channels, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, self.out_channels) 

        self.relu1= nn.ReLU()
        self.relu2= nn.ReLU()

    def _set_hidden_state(self, x, H):
        if H is None:
            # [batch size, track size?, out channels
            H = torch.zeros(x.shape[0], 1, self.hidden_channels).to(x.device)
        return H
        
    def _set_cell_state(self, x, C):
        if C is None:
            C = torch.zeros(x.shape[0], 1, self.hidden_channels).to(x.device)
        return C

    def forward(self, x, H=None, C=None):
        H_accum = 0
        outputs = []
        # data comes with list
        # (batch_size, lifetime, features, seqence length)
        H = self._set_hidden_state(x, H)
        C = self._set_cell_state(x, C)

        # Run with temporal divided
        if self.is_horizontal_pred:
            for period in range(self.num_timesteps_in):
                h, (H, C) = self.lstm(x[:,:,:,period], (H, C))
                h = self.relu1(h)
                h = self.fc1(h)
                h = self.relu2(h)
                h = self.fc2(h)

                outputs.append(h)
            outputs = torch.stack(outputs, dim=-1)
        else:
            # Encoder
            # h, (H, C) = self.encoder['lstm'](x, (H, C))
            # h = self.activations('relu')(h)
            # h = self.encoder('linear1')(h)
            # h = self.activations('relu')(h)
            # output = self.encoder('linear2')(h)
            h, (H, C) = self.lstm(x, (H, C))
            h = self.relu1(h)
            h = self.fc1(h)
            h = self.relu2(h)
            output = self.fc2(h)

            # Decoder
            # outputs = self.decoder(output, (H, C))
            # h, (H, C) = self.decoder['lstm'](output, (H, C))
            # h = self.activations('relu')(h)
            # h = self.decoder('linear1')(h)
            # h = self.activations('relu')(h)
            # outputs = self.decoder('linear2')(h)
            h, (H, C) = self.lstm(output, (H, C))
            h = self.relu1(h)
            h = self.fc1(h)
            h = self.relu2(h)
            outputs = self.fc2(h)
            # outputs = self.log_softmax(self.fc2(h))

        return outputs