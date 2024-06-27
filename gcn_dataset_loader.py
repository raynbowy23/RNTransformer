import os
import os.path as osp
import math
import pickle
import torch
from utils.static_graph_temporal_signal import StaticGraphTemporalSignal
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import polars as pl

import networkx as nx
from tqdm import tqdm

from data_import import read_from_csv, read_all_recordings_from_csv


class IdentityEncoder(object):
    '''Converts a list of floating point values into a PyTorch tensor
    '''
    def __init__(self, dtype=None):
        self.dtype = dtype
    
    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


class TrajectoryDataset(object):
    '''
    Dataset for horizontal ETA prediction on not inD-dataset
    '''
    def __init__(self, 
                data_dir, 
                sdd_loc,
                num_timesteps_in=8, 
                num_timesteps_out=8,
                skip=1,
                threshold=0.002,
                min_ped=1,
                delim='\t',
                norm_lap_matr = True,
                is_train=True,
                is_preprocessed=False,
                dataset='sdd'):
        super(TrajectoryDataset, self).__init__()

        self.is_preprocessed = is_preprocessed
        self.is_train = is_train

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.sdd_loc = sdd_loc
        self.num_timesteps_in = num_timesteps_in
        self.num_timesteps_out = num_timesteps_out
        self.skip = skip
        self.seq_len = self.num_timesteps_in + self.num_timesteps_out
        self.threshold = threshold
        self.delim = delim
        self.min_ped = min_ped
        self.norm_latp_matr = norm_lap_matr
        self.dataset = dataset

        self.sc = MinMaxScaler(feature_range=(0, 1))

        self.use_ratio = 1


        self.num_peds_in_seq = []
        self.seq_list = []
        self.seq_list_rel = []
        self.loss_mask_list = []
        self.non_linear_ped = []
        
        if self.is_train:
            all_files = os.listdir(osp.join(self.data_dir, self.sdd_loc, 'train'))
        else:
            all_files = os.listdir(osp.join(self.data_dir, self.sdd_loc, 'test'))

        if self.is_train:
            self.all_files = [os.path.join(self.data_dir, self.sdd_loc, 'train', _path) for _path in all_files]
        else:
            self.all_files = [os.path.join(self.data_dir, self.sdd_loc, 'test', _path) for _path in all_files]

        if self.is_preprocessed:
            # self.edge_index, self.edge_attr, self.node_data = torch.load(osp.join('datasets/sdd/preprocessed/', self.sdd_loc, 'road_network_in{}_out{}.pkl').format(self.num_timesteps_in, self.num_timesteps_out))
            self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, self.non_linear_ped, self.loss_mask, self.v_in, self.A_in, self.v_pred, self.A_pred, self.num_seq, self.seq_start_end = torch.load(f)
        else:
            self.create_graph()


    @property
    def processed_file_names(self):
        return osp.join(self.data_dir, 'preprocessed', self.sdd_loc, 'road_network.pkl')

    def read_file_sdd(self, _path, delim='\t'):
        data = []
        if delim == 'tab':
            delim = '\t'
        elif delim == 'space':
            delim = ' '
        with open(_path, 'r') as f:
            for line in f:
                org_line = line.strip().split(" ")
                line = [float(i) for i in org_line[:-1]]

                # Adjust to the other formats
                xcenter = (line[1] + line[3]) / 2
                ycenter = (line[2] + line[4]) / 2

                # Frame, TrackID, x center, y center, label
                line = [line[5], line[0], xcenter, ycenter, org_line[-1]]

                data.append(line)
        return np.asarray(data)

    def create_graph(self):
        '''
        Make grids from pedestrian min and max
        And count num of pedestrians in each grids
        '''
        for path in self.all_files:

            curr_ped_center_list = []
            curr_car_center_list = []
            curr_bik_center_list = []

            ped_list = []
            car_list = []
            bik_list = []

            data = self.read_file_sdd(path, self.delim)
            print(data.shape)

            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            print(f'Length of frames {len(frames)}')

            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])

            # For each road users on a frame
            for idx in range(0, int(len(frame_data) * self.use_ratio)):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.agg_frame * self.seq_len: self.agg_frame], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0

                # Pedestrian trajectory graph
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]


                for ridx in range(len(ru_in_curr_seq)):

                    # Check road user label and add number to each of them
                    ru_class = ru_in_curr_seq[ridx, -1]
                    if len(ru_in_curr_seq[ridx]) == 5:
                        ru_id = int(float(ru_in_curr_seq[ridx, 1])) # Road User's ID
                        curr_ru_center = [float(ru_in_curr_seq[ridx, 2]), float(ru_in_curr_seq[ridx, 3])]
                    else:
                        ru_id = int(float(ru_in_curr_seq[ridx, 0])) # Road User's ID
                        curr_ru_center = [float(ru_in_curr_seq[ridx, 1]), float(ru_in_curr_seq[ridx, 2])]

                    curr_ru_center = np.around(curr_ru_center, decimals=4)
                    x_list.append(curr_ru_center[0])
                    y_list.append(curr_ru_center[1])

                    if ru_class.lower() == '"pedestrian"':
                        curr_ped_center_frame_list.append(curr_ru_center)
                        ped_frame_list.append(ru_id)
                    elif ru_class.lower() == '"cars"':
                        curr_car_center_frame_list.append(curr_ru_center)
                        car_frame_list.append(ru_id)
                    elif ru_class.lower() == '"biker"':
                        curr_bik_center_frame_list.append(curr_ru_center)
                        bik_frame_list.append(ru_id)
                    else:
                        continue

                self.agg_frame = 1
                if idx % int(self.agg_frame) == 0:
                    curr_ped_center_list.append(curr_ped_center_frame_list)
                    ped_list.append(ped_frame_list)
                    curr_car_center_list.append(curr_car_center_frame_list)
                    car_list.append(car_frame_list)
                    curr_bik_center_list.append(curr_bik_center_frame_list)
                    bik_list.append(bik_frame_list)

            # Num of nodes
            x_min, y_min, x_max, y_max = min(x_list), min(y_list), max(x_list), max(y_list)
            x_seg_len = (x_max - x_min) / self.grid
            y_seg_len = (y_max - y_min) / self.grid

            x_seg = x_min
            y_seg = y_min
            x_seg_list, y_seg_list = [x_min], [y_min]
                    
            x_seg_list.append(x_seg)
            y_seg_list.append(y_seg)
            for i in range(1, self.grid):
                x_seg += x_seg_len
                y_seg += y_seg_len
                x_seg_list.append(x_seg)
                y_seg_list.append(y_seg)

            node_num = (len(x_seg_list) - 1) ** 2

            for frameId in range(0, int(int(len(frames) * self.use_ratio) / self.agg_frame)):
                ped_num_list = np.zeros((node_num))
                car_num_list = np.zeros((node_num))
                bik_num_list = np.zeros((node_num))
                ped_center = curr_ped_center_list[frameId]
                car_center = curr_car_center_list[frameId]
                bik_center = curr_bik_center_list[frameId]

                nodeId = 0
                x_center = []
                y_center = []
                # Call all sequences and count the max and min of pedestrian coordinates
                road_grid = np.zeros((node_num, node_num))
                road_grid_val_x = np.zeros((node_num, node_num, 100)) # Accumulated x center value for each ped in the grid
                road_grid_val_y = np.zeros((node_num, node_num, 100)) # Accumulated y center value for each ped in the grid

                for y in range(1, len(y_seg_list)):
                    for x in range(1, len(x_seg_list)):
                        for p in range(len(ped_list[frameId])):
                            if (ped_center[p][0] >= x_seg_list[x-1] and ped_center[p][0] <= x_seg_list[x]) and (ped_center[p][1] >= y_seg_list[y-1] and ped_center[p][1] <= y_seg_list[y]):
                                ped_num_list[nodeId] += 1
                                road_grid[y-1][x-1] += 1
                                road_grid_val_x[y-1][x-1][p] += ped_center[p][0]
                                road_grid_val_y[y-1][x-1][p] += ped_center[p][1]
                        for c in range(len(car_list[frameId])):
                            if (car_center[c][0] >= x_seg_list[x-1] and car_center[c][0] <= x_seg_list[x]) and (car_center[c][1] >= y_seg_list[y-1] and car_center[c][1] <= y_seg_list[y]):
                                car_num_list[nodeId] += 1
                        for b in range(len(bik_list[frameId])):
                            if (bik_center[b][0] >= x_seg_list[x-1] and bik_center[b][0] <= x_seg_list[x]) and (bik_center[b][1] >= y_seg_list[y-1] and bik_center[b][1] <= y_seg_list[y]):
                                bik_num_list[nodeId] += 1
                        nodeId += 1
        
                        x_center.append(x_seg_list[x-1] + (x_seg_list[x] - x_seg_list[x-1]) / 2) # Center List
                        y_center.append(y_seg_list[y-1] + (y_seg_list[y] - y_seg_list[y-1]) / 2)

                node_df = pl.DataFrame({"TIMESTEP": [frameId for _ in range(node_num)], "NODE_ID": [i for i in range(node_num)], "XCENTER": x_center, "YCENTER": y_center, "GRID": [self.grid for _ in range(node_num)],
                                        "VEHS_NUM": car_num_list, "BIKS_NUM": bik_num_list, "PEDS_NUM": ped_num_list})

                x = None
                encoders = {'TIMESTEP': IdentityEncoder(dtype=torch.long), 'NODE_ID': IdentityEncoder(dtype=torch.long), 'XCENTER': IdentityEncoder(dtype=torch.float),
                            'YCENTER': IdentityEncoder(dtype=torch.float), 'GRID': IdentityEncoder(dtype=torch.long), 'VEHS_NUM': IdentityEncoder(dtype=torch.long),
                            'BIKS_NUM': IdentityEncoder(dtype=torch.long), 'PEDS_NUM': IdentityEncoder(dtype=torch.long)}
                            
                if encoders is not None:
                    xs = [encoder(node_df[col]) for col, encoder in encoders.items()]
                    x = torch.cat(xs, dim=-1)
                    torch.from_numpy(self.sc.fit_transform(x))
                node_data_list.append(x)

        edge_index, edge_attr = self.create_edge(road_grid)

        print(self.num_timesteps_in, self.num_timesteps_out)
        torch.save((edge_index, edge_attr, node_data_list), osp.join('datasets/sdd/preprocessed/', self.sdd_loc, 'road_network_in{}_out{}.pkl').format(self.num_timesteps_in, self.num_timesteps_out))


    def create_edge(self, road_grid):
        """

        Determine source node and destination node for each road users of every timesteps
        for each time step
        for each road user
        get previous node, current node

        accumulate the number of transitions (from node A to node B)
        prob = normalization of acc transitions

        src_idx: node id from 
        dst_idx: node id to
        prob: probability to be used this node
        """
        is_ped = road_grid > 0
        node_num = len(road_grid)
        node_idx = []
        for y in range(node_num):
            for x in range(node_num):
                if is_ped[y][x]:
                    if x == 0 and y == 0:
                        node_idx.append(x)
                    elif x == 0 and y != 0: 
                        node_idx.append(node_num * y)
                    else:
                        node_idx.append(x * y)

        edge_index = np.zeros((2, node_num**2))
        # Say they're all connected
        for i in range(len(node_idx)): # src
            for j in range(len(node_idx)): # dst
                edge_index[0][len(node_idx)*i+j] = node_idx[i] # src list
                edge_index[1][len(node_idx)*i+j] = node_idx[j] # dst list
        
        edge_attr = None

        return edge_index, edge_attr

    def __len__(self):
        return len(self.processed_file_names)


    def get_dataset(self) -> StaticGraphTemporalSignal:

        features, target = [], []
        self.edge_index, self.edge_attr, self.node_data = torch.load(osp.join('datasets/sdd/preprocessed/', self.sdd_loc, 'trajectory_gcn_in{}_out{}.pkl').format(self.num_timesteps_in, self.num_timesteps_out))

        node_data = torch.stack(self.node_data, dim=1).permute(0, 2, 1)
        node_data = torch.as_tensor(node_data)
        print(node_data.shape)
        
        indices = [
            (i, i + (self.num_timesteps_in + self.num_timesteps_out))
            for i in range(node_data.shape[2] - (self.num_timesteps_in + self.num_timesteps_out) + 1)
        ]


        for i, j in indices:
            features.append((node_data[:, :-1, i:i+self.num_timesteps_in]).numpy())
            target.append((node_data[:, -1, i+self.num_timesteps_in:j]).numpy())
        
        data = StaticGraphTemporalSignal(self.edge_index, self.edge_attr, features, target)

        return data