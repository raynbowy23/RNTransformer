import itertools

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import torch.nn.functional as F


def getGridMask(frame, dimensions, num_person, neighborhood_size, grid_size, is_occupancy = False):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    num_person : number of people exist in given frame
    is_occupancy: A flag using for calculation of accupancy map

    '''
    mnp = num_person

    width, height = dimensions[0], dimensions[1]
    if is_occupancy:
        frame_mask = np.zeros((mnp, grid_size**2))
    else:
        frame_mask = np.zeros((mnp, mnp, grid_size**2))
    frame_np =  frame.cpu().data.numpy()

    #width_bound, height_bound = (neighborhood_size/(width*1.0)), (neighborhood_size/(height*1.0))
    width_bound, height_bound = (neighborhood_size/(width*1.0))*2, (neighborhood_size/(height*1.0))*2
    #print("weight_bound: ", width_bound, "height_bound: ", height_bound)

    #instead of 2 inner loop, we check all possible 2-permutations which is 2 times faster.
    list_indices = list(range(0, mnp))
    for real_frame_index, other_real_frame_index in itertools.permutations(list_indices, 2):
        current_x, current_y = frame_np[real_frame_index, 0], frame_np[real_frame_index, 1]

        width_low, width_high = current_x - width_bound/2, current_x + width_bound/2
        height_low, height_high = current_y - height_bound/2, current_y + height_bound/2

        other_x, other_y = frame_np[other_real_frame_index, 0], frame_np[other_real_frame_index, 1]
        
        if (other_x >= width_high).all() or (other_x < width_low).all() or (other_y >= height_high).all() or (other_y < height_low).all():
        # if (other_x >= width_high) or (other_x < width_low) or (other_y >= height_high) or (other_y < height_low):
                # Ped not in surrounding, so binary mask should be zero
                #print("not surrounding")
                continue
        # If in surrounding, calculate the grid cell
        cell_x = np.floor(((other_x - width_low)/width_bound) * grid_size).astype(int)
        cell_y = np.floor(((other_y - height_low)/height_bound) * grid_size).astype(int)

        if (cell_x >= grid_size).all() or (cell_x < 0).all() or (cell_y >= grid_size).all() or (cell_y < 0).all():
        # if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                continue

        if is_occupancy:
            frame_mask[real_frame_index, cell_x + cell_y*grid_size] = 1
        else:
            # Other ped is in the corresponding grid cell of current ped
            frame_mask[real_frame_index, other_real_frame_index, cell_x + cell_y*grid_size] = 1

    return frame_mask

def getSequenceGridMask(sequence, dimensions, ped_num, neighborhood_size, grid_size, is_occupancy=False):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    is_occupancy: A flag using for calculation of accupancy map
    '''
    sl = len(sequence)
    sequence_mask = []

    for i in range(sl):
        mask = torch.from_numpy(getGridMask(sequence[i], dimensions, ped_num, neighborhood_size, grid_size, is_occupancy)).float().cuda()
        sequence_mask.append(mask)

    return sequence_mask

class SocialModel(nn.Module):

    def __init__(self, seq_len, infer=False, is_rn=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(SocialModel, self).__init__()

        self.infer = infer

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = seq_len

        # Store required sizes
        self.rnn_size = 128
        self.grid_size = 4
        self.embedding_size = 64
        self.input_size = 2
        self.output_size = 5
        self.maxNumPeds = 27
        self.seq_length = seq_len
        self.neighborhood_size = 32
        self.dataset_dimensions = {'biwi':[720, 576], 'crowds':[720, 576], 'stanford':[595, 326], 'mot':[768, 576]}


        # The LSTM cell
        if is_rn:
            self.cell = nn.LSTMCell(4*self.embedding_size, self.rnn_size)
        else:
            self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)


        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size*self.rnn_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


        # Concatenate the RN feature
        if is_rn:
            self.rn_linear1 = nn.Linear(9*self.seq_length, 64)
            self.rn_linear2 = nn.Linear(4*64, 128)

    def getSocialTensor(self, grid, hidden_states):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds
        '''
        # Number of peds
        numNodes = grid.size()[0]

        # Construct the variable
        social_tensor = torch.zeros((numNodes, self.grid_size*self.grid_size, self.rnn_size))
        
        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.grid_size*self.grid_size*self.rnn_size).cuda()
        return social_tensor

    def create_lookup_table(self, x_seq, pedlist):
        #converter function to appropriate format. Instead of direcly use ped ids, we are mapping ped ids to
        #array indices using a lookup table for each sequence -> speed
        #output: seq_length (real sequence lenght+1)*max_ped_id+1 (biggest id number in the sequence)*2 (x,y)
        
        #get unique ids from sequence
        pedlist = [np.array(x.cpu()) for x in pedlist]
        unique_ids = pd.unique(np.concatenate(pedlist).ravel().tolist()).astype(int)
        # create a lookup table which maps ped ids -> array indices
        lookup_table = dict(zip(unique_ids, range(0, len(unique_ids))))

        return lookup_table
            
    def forward(self, V, PedsList, h=None):
        look_up = self.create_lookup_table(V, PedsList)
        num_ped = V.shape[3]
        num_nodes = len(look_up)
        V = V.squeeze().permute(1, 2, 0)
        grids = getSequenceGridMask(V, self.dataset_dimensions['biwi'], num_ped, self.neighborhood_size, self.grid_size)
        hidden_states = torch.zeros(num_nodes, self.rnn_size, device='cuda:0')
        cell_states = torch.zeros(num_nodes, self.rnn_size, device='cuda:0')

        '''
        Forward pass for the model
        params:
        V: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        # List of tensors each of shape args.maxNumPedsx3 corresponding to each frame in the sequence
            # frame_data = tf.split(0, args.seq_length, self.input_data, name="frame_data")
        #frame_data = [torch.squeeze(input_, [0]) for input_ in torch.split(0, self.seq_length, input_data)]
        
        #print("***************************")
        #print("input data")
        # Construct the output variable

        outputs = torch.zeros((self.seq_length * num_nodes, self.output_size))
        outputs = outputs.cuda()
        if h != None:
            # h = F.relu(self.rn_linear(h)) # (36, 12) -> (36, 128)

            out_rn = F.relu(self.rn_linear1(h.reshape(4, 9*self.seq_length)))
            out_rn = F.relu(self.rn_linear2(out_rn.reshape(1, -1))) # (36, 12) -> (ped_num, 128)
            out_rn = out_rn.expand((num_ped, -1))

        # For each frame in the sequence
        for framenum, frame in enumerate(V):

            # Peds present in the current frame

            #print("now processing: %s base frame number: %s, in-frame: %s"%(dataloader.get_test_file_name(), dataloader.frame_pointer, framenum))
            #print("list of nodes")

            #nodeIDs_boundary = num_pedlist[framenum]
            nodeIDs = [int(nodeID) for nodeID in PedsList]

            if len(nodeIDs) == 0:
                # If no peds, then go to the next frame
                continue

            # List of nodes
            list_of_nodes = [look_up[x] for x in nodeIDs]

            corr_index = torch.tensor(list_of_nodes).cuda()

            # Select the corresponding input positions
            nodes_current = frame[list_of_nodes,:]
            # Get the corresponding grid masks
            grid_current = grids[framenum]

            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)

            cell_states_current = torch.index_select(cell_states, 0, corr_index)

            # Compute the social tensor
            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)

            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            # Embed the social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))

            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            # Concatenate the RN feature
            if h != None:
                concat_embedded = torch.cat((concat_embedded, out_rn.squeeze()), 1)

            # One-step of the LSTM
            h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))

            # Compute the output
            outputs[framenum*num_nodes + corr_index.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states[corr_index.data] = h_nodes
            cell_states[corr_index.data] = c_nodes

        # Reshape outputs
        outputs_return = torch.zeros(self.seq_length, num_nodes, self.output_size, requires_grad=True)
        outputs_return = outputs_return.cuda()
        for framenum in range(self.seq_length):
            for node in range(num_nodes):
                outputs_return[framenum, node, :] = outputs[framenum*num_nodes + node, :]

        return outputs_return.unsqueeze(0), hidden_states, cell_states