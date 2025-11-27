import torch
import torch.nn as nn
import torch.distributions as tdist

from models.RNGCN import RNBlock


class SocialCellLocal(nn.Module):
    def __init__(self,
                 spatial_input=2,
                 spatial_output=2,
                 temporal_input=8,
                 temporal_output=12,
                 num_nodes=None):
        super(SocialCellLocal, self).__init__()

        #Spatial Section
        self.feat = nn.Conv1d(spatial_input,
                              spatial_output,
                              3,
                              padding=1,
                              padding_mode='zeros')
        self.feat_act = nn.ReLU()
        self.highway_input = nn.Conv1d(spatial_input,
                                       spatial_output,
                                       1,
                                       padding=0)

        #Temporal Section
        self.highway = nn.Conv1d(temporal_input, temporal_output, 1, padding=0)
        self.tpcnn = nn.Conv1d(temporal_input,
                               temporal_output,
                               3,
                               padding=1,
                               padding_mode='zeros')

        self.temporal_output = temporal_output

        # Road Network Fusion
        if num_nodes is not None:
            # Spatial adaptation layer
            self.rn_block = RNBlock(temporal_input)
            
            # Learnable fusion weight
            self.rn_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, v, rn_embed=None):
        v_shape = v.shape
        rn_adapted = None

        # Spatial Section
        v = v.permute(0, 3, 1, 2).reshape(v_shape[0] * v_shape[3], v_shape[1], v_shape[2])  #= PED*batch,  [x,y], TIME,
        v_res = self.highway_input(v)
        v_spatial = self.feat_act(self.feat(v)) + v_res

        # RN Fusion
        if rn_embed is not None:
            # Process RN embeddings: [num_nodes=64, n_horizon=3, hidden_dim=16]
            rn_adapted = self.rn_block(rn_embed, v_spatial) ##, v_spatial.size(0))
            rn_adapted = rn_adapted.reshape(v_spatial.size(0), 2, 8)  # [B*P, 2, 8]
            # Fusion with gating
            v_spatial = v_spatial + v_res + self.rn_gate * rn_adapted
        else:
            v_spatial = v_spatial + v_res

        # Temporal Section
        v = v_spatial.permute(0, 2, 1)
        v_res = self.highway(v)
        v = self.tpcnn(v) + v_res

        # Final Output
        v = v.permute(0, 2, 1).reshape(v_shape[0], v_shape[3], v_shape[1], self.temporal_output)
        return v.permute(0, 2, 3, 1), rn_adapted


class SocialCellGlobal(nn.Module):
    def __init__(self,
                 spatial_input=2,
                 spatial_output=2,
                 temporal_input=8,
                 temporal_output=12,
                 noise_w=None,
                 num_nodes=None):
        super(SocialCellGlobal, self).__init__()

        #Spatial Section
        self.feat = nn.Conv2d(spatial_input,
                              spatial_output,
                              3,
                              padding=1,
                              padding_mode='zeros')
        self.feat_act = nn.ReLU()
        self.highway_input = nn.Conv2d(spatial_input,
                                       spatial_output,
                                       1,
                                       padding=0)
        #Temporal Section
        self.highway = nn.Conv2d(temporal_input, temporal_output, 1, padding=0)

        self.tpcnn = nn.Conv2d(temporal_input,
                               temporal_output,
                               3,
                               padding=1,
                               padding_mode='zeros')

        #Self Learning Weights
        self.noise_w = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_weights = noise_w  # Used to scale the variance

        self.global_w = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.local_w = nn.Parameter(torch.zeros(1), requires_grad=True)

        #Local Stream
        self.ped = SocialCellLocal(spatial_input=spatial_input,
                                   spatial_output=spatial_output,
                                   temporal_input=temporal_input,
                                   temporal_output=temporal_output,
                                   num_nodes=num_nodes)

    def forward(self, v, noise, weight_select=1, rn_embed=None):
        # Combine Vectorized Noise
        v = v + self.noise_w * self.noise_weights[weight_select] * noise

        # Spatial Section
        v_ped, rn_adapted = self.ped(v, rn_embed=rn_embed)
        v_res = self.highway_input(v)
        v = self.feat_act(self.feat(v)) + v_res

        # Temporal Section
        v = v.permute(0, 2, 1, 3)
        v_res = self.highway(v)
        v = self.tpcnn(v) + v_res

        # Fuse Local and Global Streams
        v = v.permute(0, 2, 1, 3)
        v = self.global_w * v + self.local_w * v_ped
        return v, rn_adapted


class SocialImplicit(nn.Module):
    def __init__(self,
                 spatial_input=2,
                 spatial_output=2,
                 temporal_input=8,
                 temporal_output=12,
                 bins=[0, 0.01, 0.1, 1.2],
                 noise_weight=[0.05, 1, 4, 8],
                 num_nodes=None):
        super(SocialImplicit, self).__init__()

        self.bins = torch.Tensor(bins).cuda()
        self.temporal_output = temporal_output

        self.implicit_cells = nn.ModuleList([
            SocialCellGlobal(spatial_input=spatial_input,
                             spatial_output=spatial_output,
                             temporal_input=temporal_input,
                             temporal_output=temporal_output,
                             noise_w=noise_weight,
                             num_nodes=num_nodes)
            for i in range(len(self.bins))
        ])

        self.noise = tdist.multivariate_normal.MultivariateNormal(
            torch.zeros(2), torch.Tensor([[1, 0], [0, 1]]))

    def forward(self, v, obs_traj, KSTEPS=20, rn_embed=None):

        noise = self.noise.sample((KSTEPS, )).unsqueeze(-1).unsqueeze(-1).to(
            v.device).contiguous()

        # Social-Zones Section
        # Use max speed change(inf norm) to assign a zone
        norm = torch.linalg.norm(v.permute(0, 3, 1, 2)[0, :, :, 0],
                                 float('inf'),
                                 dim=1)
        displacment_indx = torch.bucketize(
            norm,
            self.bins,
            right=True,
        ) - 1 # Used to set each vector to a zone
        v_out = torch.zeros(KSTEPS, 2, self.temporal_output, v.shape[-1]).to(
            v.device).contiguous() # Stores results of each zone
        
        # Per each Social-Zone, call the proper Social-Cell
        for i in range(len(self.bins)):
            select = displacment_indx == i
            if torch.any(select):
                v_out[..., select], rn_out = self.implicit_cells[i](v[..., select].contiguous(),
                                                        noise,
                                                        weight_select=i,
                                                        rn_embed=rn_embed)

        return v_out.contiguous(), rn_out