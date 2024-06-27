import os
import os.path as osp
import math
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils.static_graph_temporal_signal import StaticGraphTemporalSignal
from torch_geometric_temporal import DynamicGraphTemporalSignal, DynamicGraphTemporalSignalBatch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import polars as pl

import networkx as nx
from tqdm import tqdm

from utils.metrics import min_max_normalize

from data_import import read_from_csv, read_all_recordings_from_csv


class IdentityEncoder(object):
    '''Converts a list of floating point values into a PyTorch tensor
    '''
    def __init__(self, dtype=None):
        self.dtype = dtype
    
    def __call__(self, df):
        # TODO: polars return series data, so AttributeError: 'Series' object has no attribute 'values'
        # return torch.from_numpy(df).view(-1, 1).to(self.dtype)
        return torch.tensor(df).view(-1, 1).to(self.dtype)


class RoadNetworkPeds(object):
    '''
    Dataset for horizontal ETA prediction on not inD-dataset
    '''
    def __init__(self, 
                data_dir, 
                sdd_loc,
                train_num, 
                test_num, 
                rn_num,
                grid=4, 
                agg_frame=1, 
                num_timesteps_in=8, 
                num_timesteps_out=8, 
                skip=1,
                threshold=0.002,
                delim='\t',
                norm_lap_matr=True,
                is_preprocessed=True):
        super(RoadNetworkPeds, self).__init__()

        self.is_preprocessed = is_preprocessed

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.sdd_loc = sdd_loc
        self.grid = int(grid)
        self.num_timesteps_in = num_timesteps_in
        self.num_timesteps_out = num_timesteps_out

        self.seq_len = self.num_timesteps_in + self.num_timesteps_out
        self.skip = skip
        self.threshold = threshold
        self.delim = delim
        self.norm_latp_matr = norm_lap_matr

        self.sc = MinMaxScaler(feature_range=(0,1))

        # self.use_ratio = 0.4 # If the computer gets OOM
        self.use_ratio = 1

        self.agg_frame = int(agg_frame)

        self.num_peds_in_seq = []
        self.seq_list = []
        self.seq_list_rel = []
        self.loss_mask_list = []
        self.non_linear_ped = []

        all_files = os.listdir(osp.join(self.data_dir, self.sdd_loc, 'train'))
        self.all_files = [osp.join(self.data_dir, self.sdd_loc, 'train', _path) for _path in all_files]

        # List of data directories where raw data resides
        self.road_network_data_dir = osp.join(self.data_dir, self.sdd_loc, 'preprocessed/road_network/')

        # Define the path in which the process data would be stored
        self.data_file_road_network = osp.join(self.road_network_data_dir, "trajectories_road_network.pkl")

        if self.is_preprocessed:
            self.edge_index, self.edge_attr, self.node_data = torch.load(osp.join('datasets/sdd/preprocessed/', self.sdd_loc, 'road_network_in{}_out{}.pkl').format(self.num_timesteps_in, self.num_timesteps_out))
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

            # if self.dataset == 'sdd':
            data = self.read_file_sdd(path, self.delim)

            frames = np.unique(data[:, 0]).tolist()
            frame_data = []

            print(f'Length of frames {len(frames)}')

            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])

            curr_ped_center_frame_list = []
            curr_car_center_frame_list = []
            curr_bik_center_frame_list = []

            # For each road users on a frame
            for idx in range(0, int(len(frame_data) * self.use_ratio)):

                x_list = []
                y_list = []
                node_data_list = []

                ped_frame_list = []
                car_frame_list = []
                bik_frame_list = []

                ru_in_curr_seq, _idx = np.unique(frame_data[idx], axis=1, return_index=True)
                ru_in_curr_seq = ru_in_curr_seq[:, np.argsort(_idx)]

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
        
        # Edge weight may be OD of each nodes
        edge_attr = None
        # if encoders is not None:
            # edge_attrs = [encoder(torch.one(0)) for col, encoder in encoders.items()]
            # edge_attr = torch.cat(edge_attrs, dim=-1)
        edge_attrs = [torch.ones(1) for _ in range(node_num**2)]
        edge_attr = torch.cat(edge_attrs, dim=-1)

        return edge_index, edge_attr

    def __len__(self):
        return len(self.processed_file_names)


    def get_dataset(self) -> StaticGraphTemporalSignal:

        features, target = [], []
        self.edge_index, self.edge_attr, self.node_data = torch.load(osp.join('datasets/sdd/preprocessed/', self.sdd_loc, 'road_network_in{}_out{}.pkl').format(self.num_timesteps_in, self.num_timesteps_out))

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


class RoadNetwork(object):
    '''
    Dataset for horizontal ETA prediction on inD-dataset
    '''
    def __init__(self, data_dir, train_num, test_num, rn_num, grid=4, agg_frame=20, num_timesteps_in=8, num_timesteps_out=8, is_preprocessed=True):

        super(RoadNetwork, self).__init__()

        self.is_preprocessed = is_preprocessed

        self.max_peds_in_frame = 0
        self.data_dir = osp.join(data_dir, 'data/')

        self.grid = int(grid)
        self.num_timesteps_in = num_timesteps_in
        self.num_timesteps_out = num_timesteps_out

        self.agg_frame = int(agg_frame) # To make roadnetwork, aggregate several frames

        self.num_peds_in_seq = []
        self.seq_list = []
        self.seq_list_rel = []
        self.loss_mask_list = []
        self.non_linear_ped = []

        # List of data directories where raw data resides
        self.road_network_data_dir = 'datasets/inD-dataset-v1.0/preprocessed/road_network/'

        # Define the path in which the process data would be stored
        self.data_file_road_network = osp.join(self.road_network_data_dir, "trajectories_road_network.pkl")

        if self.is_preprocessed:
            f = open(self.data_file_road_network, "rb")
            self.road_network_frames, self.road_network_tracks, self.road_network_static, self.road_network_meta = pickle.load(f)
        else:
            f = open(self.data_file_road_network, "wb")

            self.road_network_frames, self.road_network_tracks, self.road_network_static, self.road_network_meta = read_all_recordings_from_csv(self.data_dir, train_num, test_num, rn_num=rn_num, is_rn=True)

            pickle.dump((self.road_network_frames, self.road_network_tracks, self.road_network_static, self.road_network_meta), f, protocol=4)

            # Preprocess
            # Make road network
            self.create_graph()
            # self.edge_df = self.create_edge()

        f.close()


    def make_base_road_network(self, data_file):
        '''
        The function for set up the base road martrix
        Load osm file
        If subtype equals walkway, it returns 1
        Otherwise, it returns 0
        The value will eventually be changed by sub-goal model

        TODO: It should be better to create area for walkway, which includes crosswalk as a walkable area
        Currently it only has lanelets and a few areas for parking and building
        '''

        road_matrix = np.zeros()

        return road_matrix

    
    @property
    def processed_file_names(self):
        return osp.join('datasets/inD-dataset-v1.0/preprocessed', 'road_network.pkl')
        
        
    def create_graph(self):
        '''
        It will return road network based on base road network
        Depends on peds frequency on each grid, the value on each grid is going to be changed

        Not explicitly make road grid, but sampling candidate node (subgoals) from peds trajectories
        --> How to determine fixed points as node? --> Norm center or Mean center (explicitly)
        '''

        # To combine several recordings
        frame_set_list = []
        rn_len = len(self.road_network_frames)
        # Initialize 
        curr_ped_center_list = [[] for _ in range(rn_len)]
        curr_car_center_list = [[] for _ in range(rn_len)]
        curr_bik_center_list = [[] for _ in range(rn_len)]
        curr_truck_bus_center_list = [[] for _ in range(rn_len)]
        ped_list = [[] for _ in range(rn_len)]
        car_list = [[] for _ in range(rn_len)]
        bik_list = [[] for _ in range(rn_len)]
        truck_bus_list = [[] for _ in range(rn_len)]
        
        print(f'Length of road network frames: {len(self.road_network_frames)}')
        for recId, frame_set in enumerate(self.road_network_frames):

            x_list = []
            y_list = []
            node_data_list = []

            center_idx = np.zeros(len(self.road_network_tracks[recId])) # Index for center coordinates of each road users

            print(f'Length of frame {len(frame_set)}')

            # Initialize
            curr_ped_center_frame_list = []
            curr_car_center_frame_list = []
            curr_bik_center_frame_list = []
            curr_truck_bus_center_frame_list = []

            for idx in range(0, len(frame_set)):
                ped = 0
                ped_frame_list = []
                car_frame_list = []
                bik_frame_list = []
                truck_bus_frame_list = []

                rus_in_curr_seq = list(frame_set[idx]["trackId"]) # Get all road users ID exists in the frame

                for pidx, ru_id in enumerate(rus_in_curr_seq):
                    ru_class = self.road_network_static[recId][ru_id]["class"]
                    ru_frame = self.road_network_tracks[recId][ru_id]["frame"]
                    curr_ru_center = self.road_network_tracks[recId][ru_id]["center"][int(center_idx[ru_id])]
                    curr_ru_center = np.around(curr_ru_center, decimals=4)
                    curr_ru_seq = np.transpose(curr_ru_center)
                    x_list.append(curr_ru_center[0])
                    y_list.append(curr_ru_center[1])
                    # Filter pedestrians
                    if ru_class == "pedestrian":
                        # Make coordinates relative
                        # rel_curr_ped_seq = np.zeros(curr_ped_center.shape)
                        # rel_curr_ped_seq[:, 1:] = curr_ped_center[:, 1:] - curr_ped_center[:, :-1]
                        ped += 1
                        curr_ped_center_frame_list.append(curr_ru_center)
                        ped_frame_list.append(ru_id)
                    elif ru_class == "car":
                        curr_car_center_frame_list.append(curr_ru_center)
                        car_frame_list.append(ru_id)
                    elif ru_class == "bicycle":
                        curr_bik_center_frame_list.append(curr_ru_center)
                        bik_frame_list.append(ru_id)
                    elif ru_class == "truck_bus":
                        curr_truck_bus_center_frame_list.append(curr_ru_center)
                        truck_bus_frame_list.append(ru_id)
                    else:
                        continue

                    center_idx[ru_id] += 1

                if idx % int(self.agg_frame) == 0:
                    curr_ped_center_list[recId].append(curr_ped_center_frame_list) # For each aggregated frame
                    ped_list[recId].append(ped_frame_list)
                    curr_car_center_list[recId].append(curr_car_center_frame_list) # For each aggregated frame
                    car_list[recId].append(car_frame_list)
                    curr_bik_center_list[recId].append(curr_bik_center_frame_list) # For each aggregated frame
                    bik_list[recId].append(bik_frame_list)
                    curr_truck_bus_center_list[recId].append(curr_truck_bus_center_frame_list) # For each aggregated frame
                    truck_bus_list[recId].append(truck_bus_frame_list)

                    # Initialize
                    curr_ped_center_frame_list = []
                    curr_car_center_frame_list = []
                    curr_bik_center_frame_list = []
                    curr_truck_bus_center_frame_list = []

                print(len(ped_frame_list))
            # Num of nodes
            x_min, y_min, x_max, y_max = min(x_list), min(y_list), max(x_list), max(y_list)
            x_seg_len = (x_max - x_min) / self.grid # Segment x length
            y_seg_len = (y_max - y_min) / self.grid # Segment y length
            
            x_seg = x_min
            y_seg = y_min
            x_seg_list, y_seg_list = [x_min], [y_min]
            '''
            Left upper corner: (x_min, y_min)

            segment: the wall in below figure.
            node_num: (segment - 1) ** 2 as it's a square


            0 + x_seg_len
                |
                v
            | o | o | o | o |
            | o | o | o | o | 
            | o | o | o | o |
            | o | o | o | o |

            # Make Grid per horizon
            '''
            x_center_list = []
            y_center_list = []
            x_seg_list.append(x_seg) # Segment List. See above description
            y_seg_list.append(y_seg)
            for i in range(1, self.grid):
                x_seg += x_seg_len
                y_seg += y_seg_len
                x_seg_list.append(x_seg) # Segment List. See above description
                y_seg_list.append(y_seg)
                # x_center_list.append((x_seg_list[i] - x_seg_list[i-1]) / 2) # Center List
                # y_center_list.append((y_seg_list[i] - y_seg_list[i-1]) / 2)

            # road_graph = np.zeros((int(len(frame_set) / self.agg_frame), len(x_seg_list), len(y_seg_list)))
            
            # Re-ordered x_center and y_center
            node_num = (len(x_seg_list)-1) ** 2
            # for y in range(len(y_center_list)):
            #     for x in range(len(x_center_list)):
            #         x_center.append(x_center_list[x])
            #         y_center.append(y_center_list[y])

            frame_set_list.append(frame_set)

            min_frame_set_len = 99999999

        # for recId, frame_set in enumerate(frame_set_list):
            if len(frame_set) / self.agg_frame< min_frame_set_len:
                min_frame_set_len = int(len(frame_set) / self.agg_frame)
            # TODO: Want to combine peds num from consequent frames --> It's very hard to accumulate different data since nodeId is not aligned between recs
            # Aggregate frames and get enough trajectories in every nodes
            for frameId in range(0, int(len(frame_set) / self.agg_frame)):
                # Only fires at the first time
                # if recId == 0 and frameId == 0:
                ped_num_list = np.zeros((node_num))
                car_num_list = np.zeros((node_num))
                bik_num_list = np.zeros((node_num))
                truck_bus_num_list = np.zeros((node_num))
                ped_center = curr_ped_center_list[recId][frameId]
                car_center = curr_car_center_list[recId][frameId]
                bik_center = curr_bik_center_list[recId][frameId]
                truck_bus_center = curr_truck_bus_center_list[recId][frameId]
                nodeId = 0
                x_center = []
                y_center = []
                road_grid = np.zeros((node_num, node_num)) # Only apply for pedestrian
                road_grid_val_x = np.zeros((node_num, node_num, 100)) # Accumulated x center value for each ped in the grid
                road_grid_val_y = np.zeros((node_num, node_num, 100)) # Accumulated y center value for each ped in the grid

                for y in range(1, len(y_seg_list)):
                    for x in range(1, len(x_seg_list)):
                        for p in range(len(ped_list[recId][frameId])):
                            # Make mask to determine resides node
                            # e.g. the peds location is between node_matrix[i-1][i-1] = x[i-1] and x[i], and y[i-1] and y[i]
                            # x_center = (x_seg_list[i] - x_seg_list[i-1]) / 2

                            # If ped's coords are in the grid (between x_min to x_max, and y_min to y_max), it adds to the grid on road_graph
                            # For each node
                            if (ped_center[p][0] >= x_seg_list[x-1] and ped_center[p][0] <= x_seg_list[x]) and (ped_center[p][1] >= y_seg_list[y-1] and ped_center[p][1] <= y_seg_list[y]):
                                ped_num_list[nodeId] += 1
                                road_grid[y-1][x-1] += 1
                                road_grid_val_x[y-1][x-1][p] += ped_center[p][0]
                                road_grid_val_y[y-1][x-1][p] += ped_center[p][1]
                        for c in range(len(car_list[recId][frameId])):
                            if (car_center[c][0] >= x_seg_list[x-1] and car_center[c][0] <= x_seg_list[x]) and (car_center[c][1] >= y_seg_list[y-1] and car_center[c][1] <= y_seg_list[y]):
                                car_num_list[nodeId] += 1
                        for b in range(len(bik_list[recId][frameId])):
                            if (bik_center[b][0] >= x_seg_list[x-1] and bik_center[b][0] <= x_seg_list[x]) and (bik_center[b][1] >= y_seg_list[y-1] and bik_center[b][1] <= y_seg_list[y]):
                                bik_num_list[nodeId] += 1
                        for tb in range(len(truck_bus_list[recId][frameId])):
                            if (truck_bus_center[tb][0] >= x_seg_list[x-1] and truck_bus_center[tb][0] <= x_seg_list[x]) and (truck_bus_center[tb][1] >= y_seg_list[y-1] and truck_bus_center[tb][1] <= y_seg_list[y]):
                                truck_bus_num_list[nodeId] += 1
                        nodeId += 1

                        # Center Decision Calculation
                        # if road_grid[y-1][x-1] > 0: ## Norm Center
                        #     x_center_mean = sum(road_grid_val_x[y-1][x-1]) / road_grid[y-1][x-1]
                        #     deviation = 0
                        #     # Since the number of pedestrians in this grid is inconsistant, we need to write in wierd way
                        #     for ped in range(100):
                        #         if road_grid_val_x[y-1][x-1][ped] > 0 :
                        #             deviation += ((road_grid_val_x[y-1][x-1][ped]) - x_center_mean) ** 2
                        #     x_center_std = deviation / road_grid[y-1][x-1]
                        #     x_center.append(np.random.normal(loc=x_center_mean, scale=x_center_std))
                        #     y_center_mean = sum(road_grid_val_y[y-1][x-1]) / road_grid[y-1][x-1]

                        #     deviation = 0
                        #     for ped in range(100):
                        #         if road_grid_val_y[y-1][x-1][ped] > 0 :
                        #             deviation += ((road_grid_val_y[y-1][x-1][ped]) - y_center_mean) ** 2
                        #     y_center_std = deviation / road_grid[y-1][x-1]
                        #     y_center.append(np.random.normal(loc=y_center_mean, scale=y_center_std))
                        # else: ## Apply Mean Center
                        # TODO: Ideally, better to decide center with all recordings
                        # x_center[recId].append(x_seg_list[x-1] + (x_seg_list[x] - x_seg_list[x-1]) / 2) # Center List
                        # y_center[recId].append(y_seg_list[y-1] + (y_seg_list[y] - y_seg_list[y-1]) / 2)
                        x_center.append(x_seg_list[x-1] + (x_seg_list[x] - x_seg_list[x-1]) / 2) # Center List
                        y_center.append(y_seg_list[y-1] + (y_seg_list[y] - y_seg_list[y-1]) / 2)
                                
                # agg_road_grid = sum(sum(road_grid))
                # road_grid = road_grid / agg_road_grid # Edge attribute (trajectories frequencies should be normalized to 1)

        # For every frames with all recordings we use
        for frameId in range(min_frame_set_len):

            # Write node data to each timestep and for each recordings (This should be one recording for every locations)
            node_df = pl.DataFrame({"TIMESTEP": [frameId for _ in range(node_num)], "NODE_ID": [i for i in range(node_num)], "XCENTER": x_center, "YCENTER": y_center, "GRID": [self.grid for _ in range(node_num)],
                                "VEHS_NUM": car_num_list, "BIKS_NUM": bik_num_list, "TRUCK_BUS_NUM": truck_bus_num_list, "OBSTACLES_NUM": np.zeros((node_num)), "PEDS_NUM": ped_num_list})

            x = None
            encoders = {'TIMESTEP': IdentityEncoder(dtype=torch.long), 'NODE_ID': IdentityEncoder(dtype=torch.long), 'XCENTER': IdentityEncoder(dtype=torch.float),
                        'YCENTER': IdentityEncoder(dtype=torch.float), 'GRID': IdentityEncoder(dtype=torch.long), 'VEHS_NUM': IdentityEncoder(dtype=torch.long),
                        'BIKS_NUM': IdentityEncoder(dtype=torch.long), 'OBSTACLES_NUM': IdentityEncoder(dtype=torch.long), 'PEDS_NUM': IdentityEncoder(dtype=torch.long)}

            # Normalized
            if encoders is not None:
                xs = [encoder(node_df[col]) for col, encoder in encoders.items()]
                x = torch.cat(xs, dim=-1)

            # mapping = {index: i for i, index in enumerate(node_df.index.unique())}
            # data = Data(x=node_df[:, :-1].type(torch.FloatTensor), requires_grad_=True)
            # data.num_node_features = 7
            # data.num_nodes = len(mapping)
            # data.y = node_df[:, -1].type(torch.FloatTensor)
            # df.write_csv("road_node_{}.csv".format(frameId))
            node_data_list.append(x)

        edge_index, edge_attr = self.create_edge(road_grid)

        torch.save((edge_index, edge_attr, node_data_list), 'datasets/sdd/preprocessed/road_network_grid{}_in{}_out{}.pkl'.format(self.grid, self.num_timesteps_in, self.num_timesteps_out))


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


    def get_dataset(self):
        
        features, target = [], []
        edge_index, edge_attr, node_data = torch.load('datasets/sdd/preprocessed/road_network_grid{}_in{}_out{}.pkl'.format(self.grid, self.num_timesteps_in, self.num_timesteps_out))

        node_data = torch.stack(node_data, dim=1).permute(0, 2, 1)
        node_data = torch.as_tensor(node_data)
        print(node_data.shape)
        
        indices = [
            (i, i + (self.num_timesteps_in + self.num_timesteps_out))
            for i in range(node_data.shape[2] - (self.num_timesteps_in + self.num_timesteps_out) + 1)
        ]

        for i, j in indices:
            features.append((node_data[:,:-1,i:i+self.num_timesteps_in]).numpy())
            target.append((node_data[:,-1,i+self.num_timesteps_out:j]).numpy())
        
        data = StaticGraphTemporalSignal(edge_index, edge_attr, features, target)

        return data


class inDDatasetGraph(Dataset):
    """
    Dataset preparation for inD Dataset for graph
    Output is per frame data and graph data (V, A),then iterate through learning and prediction timesteps
    Referred @abduallahmohamed github repo (https://github.com/abduallahmohamed/Social-STGCNN/blob/333d3a57b4d2705e129b21aefefa09c79b2b9ae1/utils.py)
    :param 
        data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        input_len: Number of time-steps in input trajectories
        pred_len: Number of time-steps in output trajectories
        skip: Number of frames to skip while making the dataset
        threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        min_ped: Minimum number of pedestrians that should be in a seqeunce
        delim: Delimiter in the dataset files
    :return: in_traj[start:end, :], pred_traj[start:end, :], in_traj_rel[start:end, :], pred_traj_rel[start:end, :], non_linear_ped[start:end], loss_mask[start:end, :], v_in[index], A_in[index], v_pred[index], A_pred[index]
        where start = number of pedestrian, end = num of peds + 1, iterate from learning step:learning + prediction step.
        non_linear_ped: non linear trajectory to complement the rest of linear traj
        loss_mask: Only consider the time peds actually traverse
        v_in: current vertices matrix
        A_in: current adjacency matrix
    """

    # To replace sequential data as node in graph, the data should be same length with padding before and after data.
    def __init__(
        self, data_dir, num_timesteps_in=8, num_timesteps_out=8, train_num=24, test_num=6, rn_num=1,
        is_train=True, skip=1, min_ped=1, norm_lap_matr=True, horizon=4, is_preprocessed=True):

        super(inDDatasetGraph, self).__init__()

        self.is_preprocessed = is_preprocessed

        # Should be preprocessed
        data_file_train = osp.join('datasets/inD-dataset-v1.0/preprocessed/all_train.pkl')
        data_file_test = osp.join('datasets/inD-dataset-v1.0/preprocessed/all_test.pkl')

        self.max_peds_in_frame = 0
        self.data_dir = osp.join(data_dir, 'data/')
        self.num_timesteps_in = num_timesteps_in
        self.num_timesteps_out = num_timesteps_out
        # 25 FPS for inD --> 0.24 sec per frame
        # 0.25 FPS for inD --> 4 sec per frame
        self.skip = skip
        self.seq_len = self.num_timesteps_in + self.num_timesteps_out
        self.is_train = is_train
        self.min_ped = min_ped
        self.norm_lap_matr = norm_lap_matr
        self.horizon = horizon

        self.num_peds_in_seq = []
        self.seq_list = []
        self.seq_list_rel = []
        self.loss_mask_list = []
        self.non_linear_ped = []

        # List of data directories where raw data resides
        self.train_data_dir = 'datasets/inD-dataset-v1.0/preprocessed/train/'
        self.test_data_dir = 'datasets/inD-dataset-v1.0/preprocessed/test/'
        self.val_data_dir = 'datasets/inD-dataset-v1.0/preprocessed/val/'

        # Define the path in which the process data would be stored
        self.data_file_tr = osp.join(self.train_data_dir, "trajectories_train_in{}_out{}.pkl".format(num_timesteps_in, num_timesteps_out))
        self.data_file_te = osp.join(self.test_data_dir, "trajectories_test_in{}_out{}.pkl".format(num_timesteps_in, num_timesteps_out))
        self.data_file_vl = osp.join(self.val_data_dir, "trajectories_val_in{}_out{}.pkl".format(num_timesteps_in, num_timesteps_out))

        if self.is_preprocessed:
            if is_train:
                f = open(data_file_train, "rb")
            else:
                f = open(data_file_test, "rb")
            self.all_frames, self.all_tracks, self.all_static, self.all_meta = pickle.load(f)

            if self.is_train:
                f = open(self.data_file_tr, "rb")
            else:
                f = open(self.data_file_te, "rb")
            self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, self.non_linear_ped, self.loss_mask, self.v_in, self.A_in, self.v_pred, self.A_pred, self.num_seq, self.seq_start_end = pickle.load(f)
        else:
            if is_train:
                f = open(data_file_train, "wb")
                data_file = self.data_file_tr
                # self.data_ratio = 0.5
                self.all_frames, self.all_tracks, self.all_static, self.all_meta = read_all_recordings_from_csv(self.data_dir, train_num=train_num, test_num=test_num, rn_num=rn_num, is_train=is_train, is_rn=False)

                pickle.dump((self.all_frames, self.all_tracks, self.all_static, self.all_meta), f, protocol=4)

                print(f'Length of all frames {len(self.all_frames)}')

                # Due to the less number of pedestrian except scene 3, we intentionally choose one of the recordings from scene 3.
                self.train_frames = self.all_frames[0:1] # Use first recordings of training set. Out of 33

                self.data_frames = self.train_frames
            else:
                f = open(data_file_test, "wb")
                data_file = self.data_file_te
                # self.data_ratio = 0.2
                self.all_frames, self.all_tracks, self.all_static, self.all_meta = read_all_recordings_from_csv(self.data_dir, train_num=train_num, test_num=test_num, is_train=is_train, is_rn=False)

                pickle.dump((self.all_frames, self.all_tracks, self.all_static, self.all_meta), f, protocol=4)

                print(len(self.all_frames))

                self.test_frames = self.all_frames[0:1] # Out of 33 # No need test? --> Transductive learning
            
                self.data_frames = self.test_frames

            # TODO: Load these datasets
            # self.train_frames = self.all_frames[0:train_num] # Out of 33
            # self.test_frames = self.all_frames[train_num:train_num+test_num]
            # self.validation_frames = self.all_frames[train_num+test_num:-1]

            self.set_graph_data(data_file)

        f.close()


    def set_graph_data(self, data_file):
        # Recording
        for self.recId, self.frame_set in enumerate(self.data_frames):
            print(f'Num of recording we use {len(self.frame_set)}')
            # recordingId, trackId, frame, trackLifetime, xCenter, yCenter, heading, 
            # width, length, xVelocity, yVelocity, xAcceleration, yAcceleration, lonVelocity, 
            # latVelocity, lonAcceleration, latAcceleration, center

            # for frame_id, frame in enumerate(self.frame_set):
                # print(frame)
                # print(self.frame_set[frame == self.frame_set[:][2]])
                # frame_data.append(self.frame_set[frame == self.frame_set[:][:][2]][:])
            num_sequences = int(
                # math.ceil((int(len(self.frame_set)*self.data_ratio) - self.seq_len + 1) / self.skip)
                math.ceil((len(self.frame_set) - self.seq_len + 1) / self.skip)
            ) # Whole sequences(frames) for model input

            for idx in range(0, num_sequences * self.skip, self.skip):
                # curr_seq_data = np.concatenate(
                #     new_frame_set, axis=0
                # )

                # curr_seq_data = [list(self.frame_set[i].values()) for i in range(idx, idx+self.seq_len)]

                # Extract unique arrays
                peds_in_curr_seq = [list(self.frame_set[i]["trackId"]) for i in range(idx, idx+self.seq_len)]
                if isinstance(peds_in_curr_seq[0], list):
                    peds_in_curr_seq = self.unique(peds_in_curr_seq)
                else:
                    peds_in_curr_seq = np.unique(peds_in_curr_seq)

                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                _non_linear_ped = []

                for pidx, ped_id in enumerate(peds_in_curr_seq):
                    # Filter pedestrians
                    ped_class = self.all_static[self.recId][ped_id]["class"]
                    ped_frame = self.all_tracks[self.recId][ped_id]["frame"]
                    # starts from 0 in list, but value should start from ped_start_frame
                    if len(np.where(ped_frame == idx)[0]) != 0:
                        # When start frame is same as idx
                        ped_start_frame = np.where(ped_frame == idx)[0][0]
                    else:
                        # When start frame is ahead of current frame
                        ped_start_frame = 0
                    if ped_class == "pedestrian":
                        curr_ped_seq = self.all_tracks[self.recId][ped_id]["center"][ped_start_frame:ped_start_frame+self.seq_len]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        # print(self.all_tracks[self.frame_set_id][ped_id]["frame"][ped_start_frame - idx:ped_start_frame - idx+self.seq_len])
                        curr_ped_frame = self.all_tracks[self.recId][ped_id]["frame"][ped_start_frame:ped_start_frame+self.seq_len][0]
                        last_ped_frame = self.all_tracks[self.recId][ped_id]["frame"][ped_start_frame:ped_start_frame+self.seq_len][-1]
                        pad_front = curr_ped_frame - idx + 1
                        pad_end = last_ped_frame - idx + 1
                        assert pad_end - pad_front <= self.seq_len
                        if pad_end - pad_front != self.seq_len:
                            continue
                        # Make coordinates relative
                        curr_ped_seq = np.transpose(curr_ped_seq)
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                        curr_seq[pidx, :, pad_front:pad_end] = curr_ped_seq # Make sequence with mask of 0
                        curr_seq_rel[pidx, :, pad_front:pad_end] = rel_curr_ped_seq

                        # TODO: Linear vs Non-Linear Trajectory
                        #
                        #

                        curr_loss_mask[pidx, pad_front:pad_end] = 1
                    else:
                        continue

                if pidx > self.min_ped:
                    self.num_peds_in_seq.append(pidx)
                    self.seq_list.append(curr_seq[:pidx])
                    self.seq_list_rel.append(curr_seq_rel[:pidx])
                    self.loss_mask_list.append(curr_loss_mask[:pidx])

            self.num_seq = len(self.seq_list)
            self.seq_list = np.concatenate(self.seq_list, axis=0)
            self.seq_list_rel = np.concatenate(self.seq_list_rel, axis=0)
            self.loss_mask_list = np.concatenate(self.loss_mask_list, axis=0)
            self.non_linear_ped = np.asarray(self.non_linear_ped)

            # Convert numpy -> Torch tensor
            self.in_traj = torch.from_numpy(
                self.seq_list[:, :, :self.num_timesteps_in]).type(torch.float)
            self.pred_traj = torch.from_numpy(
                self.seq_list[:, :, self.num_timesteps_in:]).type(torch.float)
            self.in_traj_rel = torch.from_numpy(
                self.seq_list_rel[:, :, :self.num_timesteps_in]).type(torch.float)
            self.pred_traj_rel = torch.from_numpy(
                self.seq_list_rel[:, :, self.num_timesteps_in:]).type(torch.float)
            self.loss_mask = torch.from_numpy(self.loss_mask_list).type(torch.float)
            self.non_linear_ped = torch.from_numpy(self.non_linear_ped).type(torch.float)
            cum_start_idx = [0] + np.cumsum(self.num_peds_in_seq).tolist()
            self.seq_start_end = [
                (start, end)
                for start, end in zip(cum_start_idx, cum_start_idx[1:])
            ]
            # Convert to Graphs --> # Would be better to use Pytorch geometric for constancy?
            self.v_in = []
            self.A_in = []
            self.v_pred = []
            self.A_pred = []
            print("Processing Data .....")
            pbar = tqdm(total=len(self.seq_start_end))
            for ss in range(len(self.seq_start_end)):
                pbar.update(1)

                start, end = self.seq_start_end[ss]

                v_, a_ = self.seq_to_graph(self.in_traj[start:end, :], self.in_traj_rel[start:end, :], self.norm_lap_matr)
                self.v_in.append(v_.clone())
                self.A_in.append(a_.clone())
                v_, a_ = self.seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], self.norm_lap_matr)
                self.v_pred.append(v_.clone())
                self.A_pred.append(a_.clone())
            pbar.close()

        # Save the arrays in the pickle file
        f = open(data_file, "wb")
        pickle.dump((self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, self.non_linear_ped, self.loss_mask, self.v_in, self.A_in, self.v_pred, self.A_pred, self.num_seq, self.seq_start_end), f, protocol=4)
        f.close()

    def unique(self, ped_list):
        val = []
        for ped in ped_list:
            for p in ped:
                val.append(p)
        unique_ped_list = np.unique(val)
        return unique_ped_list

    def anorm(self, p1, p2): 
        NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
        if NORM == 0:
            return 0
        return 1 / (NORM)

    def seq_to_graph(self, seq_, seq_rel, norm_lap_matr=True):
        """
        At = St^(-1/2)AtSt^(-1/2)
        Graph is consisted of adjacency matrix
        """
        seq_ = seq_.squeeze()
        seq_rel = seq_rel.squeeze()
        seq_len = seq_.shape[2]
        max_nodes = seq_.shape[0]
        
        V = np.zeros((seq_len, max_nodes, 2))
        A = np.zeros((seq_len, max_nodes, max_nodes))
        for s in range(seq_len):
            step_ = seq_[:, :, s]
            step_rel = seq_rel[:,:,s]
            for h in range(len(step_)): 
                V[s, h, :] = step_rel[h]
                A[s, h, h] = 1
                for k in range(h+1, len(step_)):
                    l2_norm = self.anorm(step_rel[h], step_rel[k])
                    A[s, h, k] = l2_norm
                    A[s, k, h] = l2_norm
            if norm_lap_matr: 
                G = nx.from_numpy_array(A[s, :, :])
                A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()
                
        return torch.from_numpy(V).type(torch.float),\
            torch.from_numpy(A).type(torch.float)

    def __len__(self):
        if self.is_preprocessed:
            if self.is_train:
                f = open(self.data_file_tr, "rb")
            else:
                f = open(self.data_file_te, "rb")
            self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, self.non_linear_ped, self.loss_mask, self.v_in, self.A_in, self.v_pred, self.A_pred, self.num_seq, self.seq_start_end = pickle.load(f)
            f.close()
        return self.num_seq
    
    def __getitem__(self, index):

        if self.is_preprocessed:
            if self.is_train:
                f = open(self.data_file_tr, "rb")
            else:
                f = open(self.data_file_te, "rb")
            self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, self.non_linear_ped, self.loss_mask, self.v_in, self.A_in, self.v_pred, self.A_pred, self.num_seq, self.seq_start_end = pickle.load(f)
            f.close()

        start, end = self.seq_start_end[index]

        out = [
            self.in_traj[start:end, :], self.pred_traj[start:end, :],
            self.in_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_in[index], self.A_in[index],
            self.v_pred[index], self.A_pred[index]
        ]
        return out


class inDDatasetSocialLSTM(Dataset):
    """
    Dataset preparation for inD Dataset for Social LSTM (and other social force models, I guess)
    Output is frame level data
    :param
    :return:
        xbatch: learning frames sequence
        ybatch: prediction frames sequence
        d: dataset pointer?
        numPedsList: number of pedestrians in each frames
        target_ids: pedestrians ids
    """
    def __init__(self,
                path,
                train_num,
                num_timesteps_in,
                num_timesteps_out,
                is_horizontal_pred=False,
                is_train=True):

        super(inDDatasetSocialLSTM, self).__init__()
        # recordings = read_all_recordings_from_csv(path)
        # self.all_tracks, self.all_static, self.all_meta = recordings["tracks"], recordings["tracks_meta"], recordings["recording_meta"]
        self.all_tracks, self.all_static, self.all_meta = read_all_recordings_from_csv(path)

        self.train_tracks = self.all_tracks[0:train_num] # Out of 33
        self.test_tracks = self.all_tracks[train_num:train_num+6]
        self.validation_tracks = self.all_tracks[train_num+6:-1]

        # self.data_file_train = osp.join(self.train_data_dir, "trajectories_train.pkl")
        # self.data_file_test = osp.join(self.test_data_dir, "trajectories_test.pkl")
        # self.data_file_validation = osp.join(self.validation_data_dir, "trajectories_validation.pkl")

        self.num_timesteps_in = num_timesteps_in
        self.num_timesteps_out = num_timesteps_out
        self.is_horizontal_pred = is_horizontal_pred
        self.num_tracks = 0
        self.track_list = [] # For each track (object)
        self.x_min = 10000
        self.x_max = -10000
        self.y_min = 10000
        self.y_max = -10000
        lifetime = 200

        # Need to split into each tracks?
        if is_train:
            # For each recordings
            for self.track_set_id, self.track_set in enumerate(self.train_tracks):
                self.num_tracks = self.num_tracks + len(self.track_set)
                # For each tracks
                for track_id, track in enumerate(self.track_set):
                    if track['trackLifetime'][-1] < lifetime: # why 200? --> It only considers 200 tracks (the whole dataset is A LOT!)
                        self.num_tracks = self.num_tracks - 1
                        continue
                    if track['xCenter'].min() < self.x_min:
                        self.x_min = track['xCenter'].min()
                    if track['yCenter'].min() < self.y_min:
                        self.y_min = track['yCenter'].min()                
                    if track['xCenter'].max() > self.x_max:
                        self.x_max = track['xCenter'].max()
                    if track['yCenter'].max() > self.y_max:
                        self.y_max = track['yCenter'].max()
                    # self.track_list.append(torch.FloatTensor(np.stack([track['xCenter'][0:200:10],track['yCenter'][0:200:10],track['xVelocity'][0:200:10],track['yVelocity'][0:200:10]], axis=-1)))
                    self.track_list.append(torch.FloatTensor(np.stack([track['xCenter'],track['yCenter']], axis=-1)).transpose(1, 0))
        else:
            for self.track_set_id, self.track_set in enumerate(self.test_tracks):
                self.num_tracks = self.num_tracks + len(self.track_set)
                for track_id, track in enumerate(self.track_set):
                    if track['trackLifetime'][-1] < 200: # why 200?
                        self.num_tracks = self.num_tracks - 1
                        continue
                    if track['xCenter'].min() < self.x_min:
                        self.x_min = track['xCenter'].min()
                    if track['yCenter'].min() < self.y_min:
                        self.y_min = track['yCenter'].min()                
                    if track['xCenter'].max() > self.x_max:
                        self.x_max = track['xCenter'].max()
                    if track['yCenter'].max() > self.y_max:
                        self.y_max = track['yCenter'].max()

                    # self.track_list.append(torch.FloatTensor(np.stack([track['xCenter'][0:200:10],track['yCenter'][0:200:10],track['xVelocity'][0:200:10],track['yVelocity'][0:200:10]], axis=-1)))
                    self.track_list.append(torch.FloatTensor(np.stack([track['xCenter'],track['yCenter']], axis=-1)).transpose(1, 0))

        print(len(self.track_list)) # 5373 # Different shape of trajectories --> Padding?

        # # Make time horizon input and output --> Should be frame level?
        # # (batch size, track id, num of tracks, (x,y))
        # track_data = torch.stack(self.track_list, dim=1).permute(0, 2, 1).unsqueeze(dim=0)
        # track_data = torch.as_tensor(track_data)
        # print(track_data.shape)

    def frame_preprocess(self, data_dirs, data_files, validation_set = False):
        '''Function that will pre-process the files of each dataset into
        data with occupancy grid that can be used
        :params:
        data_dirs: List of directories where raw data resides
        data_file: The file into which all the pre-processed data needs to be stored
        validation_set: true when a dataset is in validation set
        '''
        # all_frame_data would be a list of numpy arrays corresponding to each dataset
        # Each numpy array will correspond to a frame and would be of size (numPeds, 3) each row
        # containing pedID, x, y
        all_frame_data = []
        # Validation frame data
        valid_frame_data = []
        # frameList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset 
        frameList_data = []
        valid_numPeds_data = []
        # numPeds_data would be a List of lists corresponding to each dataset
        # Each list would contain the number of pedestrians in each frame in the dataset
        numPeds_data = []

        # Each list includes ped ids of the frame
        pedsList_data = []
        valid_pedsList_data = []
        # target ped ids for each sequence, which is used for lookup table (and occupancy map)
        target_ids = []
        orig_data = []

        # Index of the current dataset
        dataset_index = 0

        # For each recordings
        for directory in data_dirs:

            # Load the data from the csv file
            print("Now processing: ", directory)

            self.tracks, self.tracks_meta, self.recording_meta = read_from_csv(directory)

            # convert pl -> numpy array
            data = np.array(self.tracks)

            # keep original copy of file
            orig_data.append(data)

            # get frame numbers
            frameList = data[0, :].tolist()

            # Number of frames
            numFrames = len(frameList)

            # Add the list of frameIDs to the frameList_data
            frameList_data.append(frameList)

            # Initialize the list of numPeds for the current dataset
            numPeds_data.append([])
            valid_numPeds_data.append([])

            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            valid_pedsList_data.append([])

            target_ids.append(self.target_ids)


        # Save the arrays in the pickle file
        f = open(data_files, "wb")
        pickle.dump(())
        f.close()
        

    def __len__(self):
        return int(self.num_tracks)

    def __getitem__(self, idx):
        if self.is_horizontal_pred:
            if self.track_list[idx].shape[1] - (self.num_timesteps_in + self.num_timesteps_out) + 1 >= 0:
                indices = [
                    (i, i + (self.num_timesteps_in + self.num_timesteps_out))
                    for i in range(self.track_list[idx].shape[1] - (self.num_timesteps_in + self.num_timesteps_out) + 1)
                ]
            else:
                indices = [
                    (i, i + (self.num_timesteps_in + self.num_timesteps_out))
                    for i in range(self.track_list[idx].shape[1] - (50 + 50) + 1)
                ]

            self.features = []
            self.pred_gt = []

            for i, j in indices:
                self.features.append(self.track_list[idx][:,i:i+self.num_timesteps_in].type(torch.float))
                self.pred_gt.append(self.track_list[idx][:,i+self.num_timesteps_in:j].type(torch.float))

            self.features = torch.stack(self.features, dim=0) # .unsqueeze(dim=0)
            self.pred_gt = torch.stack(self.pred_gt, dim=0) #.unsqueeze(dim=0)

            # Return all
        return x, y, d, numPedsList, PedsList, target_ids


def anorm(p1, p2): 
    NORM = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    if NORM == 0:
        return 0
    return 1 / (NORM)
                

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """
        Dataloder for the Trajectory datasets and Road Network dataset
        Except inD-dataset
    """
    def __init__(self,
                data_dir,
                sdd_loc: str='',
                in_channels: int=8,
                out_channels: int=8,
                rn_out_channels: int=8,
                grid: int=4,
                threshold: float=0.002,
                min_ped: int=1,
                delim: str='\t',
                norm_lap_matr: bool=True,
                is_preprocessed: bool=False,
                agg_frame: int=20,
                skip: int=1,
                dataset: str='sdd',
                dataset_iter: int=0,
                train_mode: str='train',
                is_rn: bool=False,
                is_normalize: bool=False):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        - sdd_loc: Location in SDD
        <frame_id> <ped_id> <x> <y>
        - in_channels: Number of time-steps in input trajectories, observation
        - out_channels: Number of time-steps in output trajectories, prediction
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        - dataset_iter: Load local peds trajectory only at the first time
        - train_mode: train / test / val
        """
        super(TrajectoryDataset, self).__init__()

        self.is_preprocessed = is_preprocessed

        self.sc = MinMaxScaler(feature_range=(0, 1))

        # Should be preprocessed
        # data_file_train = osp.join('datasets', dataset, 'preprocessed', 'all_train.pkl')
        # data_file_test = osp.join('datasets', dataset, 'preprocessed', 'all_test.pkl')

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.sdd_loc = sdd_loc
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rn_out_channels = rn_out_channels
        self.seq_len = self.in_channels + self.out_channels
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.threshold = threshold
        self.min_ped = min_ped
        self.dataset = dataset
        self.grid = int(grid)
        self.agg_frame = int(agg_frame)
        self.dataset_iter = dataset_iter
        # Ratio that you can use it for this dataset. Reduce it when you meet OOM.
        self.use_ratio = 1.0
        self.is_rn = is_rn
        self.skip = skip
        self.train_mode = train_mode
        self.is_normalize = is_normalize

        all_files = os.listdir(osp.join(self.data_dir, self.sdd_loc, self.train_mode))
        self.all_files = [osp.join(self.data_dir, self.sdd_loc, self.train_mode, _path) for _path in all_files]

        self.num_peds_in_seq = []
        self.seq_list = []
        self.seq_list_rel = []
        self.loss_mask_list = []
        self.non_linear_ped = []

        # List of data directories where raw data resides
        self.load_data_dir = osp.join('datasets', self.dataset, 'preprocessed', self.sdd_loc, self.train_mode)

        # Define the path in which the process data would be stored
        if self.is_normalize:
            self.data_file = osp.join(self.load_data_dir, "trajnet_normalize_in{}_out{}_aggframe{}.pkl".format(self.in_channels, self.out_channels, self.agg_frame))
        else:
            self.data_file = osp.join(self.load_data_dir, "trajnet_unnormalize_in{}_out{}_aggframe{}.pkl".format(self.in_channels, self.out_channels, self.agg_frame))
        f = self.data_file

        if self.is_preprocessed:
            self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, self.non_linear_ped, \
                self.loss_mask, self.v_in, self.A_in, self.v_pred, self.A_pred, self.num_seq, self.seq_start_end, self.min_rel_pred, self.max_rel_pred = torch.load(f)
            self.edge_index_list, self.edge_attr_list, self.node_data_list = torch.load(osp.join('datasets', self.dataset, 'preprocessed', self.sdd_loc, self.train_mode, 'roadnet_in{}_out{}_aggframe{}.pkl').format(self.in_channels, self.out_channels, self.agg_frame))
        else:
            if dataset_iter == 0:
                self.set_trajnet_graph()
            if is_rn:
                self.set_roadnet_graph()
                _, _, _, _, _, \
                    _, _, _, _, _, self.num_seq, _, _, _ = torch.load(f)


    @property
    def processed_file_names(self):
        return osp.join(self.data_dir, 'preprocessed', self.sdd_loc, self.train_mode, 'trajnet_in{}_out{}_aggframe{}.pt'.format(self.in_channels, self.out_channels, self.agg_frame))


    def read_file(self, _path, delim='\t', is_rn=False):
        data = []
        if delim == 'tab':
            delim = '\t'
        elif delim == 'space':
            delim = ' '
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                if is_rn:
                    line.append('pedestrian')
                data.append(line)

        return np.asarray(data)


    def read_file_sdd(self, _path, delim='\t'):
        data = []
        data_ped = []
        data_bik = []
        if delim == 'tab':
            delim = '\t'
        elif delim == 'space':
            delim = ' '
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(" ")
                line = [float(i) for i in line[:-1]]

                # Adjust to the other formats
                xcenter = (line[1] + line[3]) / 2
                ycenter = (line[2] + line[4]) / 2

                line = [line[5], line[0], xcenter, ycenter]
                line = [float(i) for i in line]

                data.append(line)
        return np.asarray(data)


    def read_file_rn(self, _path, delim='\t'):
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


    def set_roadnet_graph(self):
        self.node_data_list = []
        self.edge_index_list = []
        self.edge_attr_list = []

        for path in self.all_files:
            if self.dataset == 'sdd':
                data_rn = self.read_file_rn(path, self.delim)
            else:
                data_rn = self.read_file(path, self.delim, True)

            frames_rn = np.unique(data_rn[:, 0]).tolist()
            frame_data_rn = []
            frame_data_rn_peds = np.array(1.0)

            for frame_ in frames_rn:
                frame_data_rn.append(data_rn[frame_ == data_rn[:, 0], :])
                frame_data_rn_peds = np.append(frame_data_rn_peds, data_rn[frame_ == data_rn[:, 0], 1])
                
            curr_ped_center_list = []
            curr_car_center_list = []
            curr_bik_center_list = []

            ped_list = []
            car_list = []
            bik_list = []


            # Location for each pedestrian at every timeframe (num of peds, total frame len)
            self.ru_in_seq = np.unique(frame_data_rn_peds)
            # ped_od = np.full((len(self.ru_in_seq), int(len(frame_data_rn) / self.agg_frame)), -1)
            ped_od = [[] for _ in range(len(self.ru_in_seq))]
            # print(len(self.ru_in_seq))

            for idx in range(0, int(len(frame_data_rn) * self.use_ratio)):

                x_list = []
                y_list = []

                ped_frame_list = []
                car_frame_list = []
                bik_frame_list = []
                curr_ped_center_frame_list = []
                curr_car_center_frame_list = []
                curr_bik_center_frame_list = []

                ru_in_curr_seq, _idx = np.unique(frame_data_rn[idx], axis=1, return_index=True)
                ru_in_curr_seq = ru_in_curr_seq[:, np.argsort(_idx)]

                # Road Network
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

                    if ru_class.lower() == '"pedestrian"' or ru_class.lower() == 'pedestrian':
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

                if idx % int(self.agg_frame) == 0:
                    curr_ped_center_list.append(curr_ped_center_frame_list)
                    ped_list.append(ped_frame_list)
                    curr_car_center_list.append(curr_car_center_frame_list)
                    car_list.append(car_frame_list)
                    curr_bik_center_list.append(curr_bik_center_frame_list)
                    bik_list.append(bik_frame_list)

            # Num of nodes
            x_min, y_min, x_max, y_max = min(x_list), min(y_list), max(x_list), max(y_list)
            x_seg_len = (x_max - x_min) / float(self.grid)
            y_seg_len = (y_max - y_min) / float(self.grid)

            x_seg = x_min
            y_seg = y_min
            x_seg_list, y_seg_list = [x_min], [y_min]
                    
            for _ in range(0, self.grid):
                x_seg += x_seg_len
                y_seg += y_seg_len
                x_seg_list.append(x_seg)
                y_seg_list.append(y_seg)

            node_num = (len(x_seg_list) - 1) ** 2

            for frameId in tqdm(range(0, int(int(len(frames_rn) * self.use_ratio) / self.agg_frame))):
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
                road_grid = np.zeros((self.grid, self.grid))
                road_grid_val_x = np.zeros((self.grid, self.grid, 100)) # Accumulated x center value for each ped in the grid
                road_grid_val_y = np.zeros((self.grid, self.grid, 100)) # Accumulated y center value for each ped in the grid

                # Assign Pedestrians to each road grid
                for y in range(1, len(y_seg_list)):
                    for x in range(1, len(x_seg_list)):
                        for p in range(len(np.unique(ped_list[frameId]))):
                            if (ped_center[p][0] >= x_seg_list[x-1] and ped_center[p][0] <= x_seg_list[x]) and (ped_center[p][1] >= y_seg_list[y-1] and ped_center[p][1] <= y_seg_list[y]):
                                ped_num_list[nodeId] += 1
                                road_grid[y-1][x-1] += 1
                                road_grid_val_x[y-1][x-1][p] += ped_center[p][0]
                                road_grid_val_y[y-1][x-1][p] += ped_center[p][1]
                                ped_od[p].append(nodeId)
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
                    x = torch.from_numpy(self.sc.fit_transform(x))
                self.node_data_list.append(x)

                self.edge_index, self.edge_attr = self.create_edge(road_grid, ped_od)
                self.edge_index_list.append(self.edge_index)
                self.edge_attr_list.append(self.edge_attr)

        torch.save((self.edge_index_list, self.edge_attr_list, self.node_data_list),
                    osp.join('datasets', self.dataset, 'preprocessed', self.sdd_loc, self.train_mode, 'roadnet_in{}_out{}_aggframe{}.pkl').format(self.in_channels, self.out_channels, self.agg_frame))


    def set_trajnet_graph(self):
        """
        Data format for SDD:
            Track ID, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label
            Format description from here: https://github.com/flclain/StanfordDroneDataset

        Data format for others:
            Frame number, Track ID, center x coord, center y coord

        """
        frames_loc = [0 for _ in range(len(self.all_files))]
        for p_id, path in enumerate(self.all_files):
            if self.dataset == 'sdd':
                data_loc = self.read_file_sdd(path, self.delim)
            else:
                data_loc = self.read_file(path, self.delim, False)

            frames_loc[p_id] = np.unique(data_loc[:, 0]).tolist()
            frame_data_loc = []
            print(f'Length of frames: {len(frames_loc[p_id])}')
            for frame in frames_loc[p_id]:
                frame_data_loc.append(data_loc[frame == data_loc[:, 0], :])

            num_sequences = int(
                math.ceil((len(frames_loc[p_id]) - self.seq_len + 1) / self.skip))

            print("Preparing Data .....")
            # pbar = tqdm(total=int(len(frame_data_loc) * self.use_ratio))
            # pbar = tqdm(total=num_sequences * self.skip + 1)
            # for idx in pbar:
            # for idx in range(int(len(frame_data_loc) * self.use_ratio)):

            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                # pbar.update(self.skip)
                # pbar.update(1)

                curr_seq_data = np.concatenate(
                    frame_data_loc[idx:idx + self.skip * self.seq_len:self.skip], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []

                # Pedestrian trajectory graph
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames_loc[p_id].index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames_loc[p_id].index(curr_ped_seq[-1, 0]) - idx + self.skip
                    if pad_end - pad_front != self.seq_len * self.skip:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered

                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, self.out_channels, self.threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > 1:
                    self.non_linear_ped += _non_linear_ped
                    self.num_peds_in_seq.append(num_peds_considered)
                    self.loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    self.seq_list.append(curr_seq[:num_peds_considered])
                    self.seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(self.seq_list)
        self.seq_list = np.concatenate(self.seq_list, axis=0)
        self.seq_list_rel = np.concatenate(self.seq_list_rel, axis=0)
        self.loss_mask_list = np.concatenate(self.loss_mask_list, axis=0)
        self.non_linear_ped = np.asarray(self.non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.in_traj = torch.from_numpy(
            self.seq_list[:, :, :self.in_channels]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            self.seq_list[:, :, self.in_channels:]).type(torch.float)
        self.in_traj_rel = torch.from_numpy(
            self.seq_list_rel[:, :, :self.in_channels]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            self.seq_list_rel[:, :, self.in_channels:]).type(torch.float)
        self.loss_mask = torch.from_numpy(self.loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(self.non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(self.num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        # sum_of_lists = [len(frames_loc[i]) for i in range(len(frames_loc))]
        
        # self.seq_start_end = [
        #     (i, i + (self.in_channels + self.out_channels))
        #     for i in range(math.ceil((sum(sum_of_lists) - self.seq_len + 1) / self.skip))
        # ]


        # Convert to Graphs 
        self.v_in = [] 
        self.A_in = [] 
        self.v_pred = [] 
        self.A_pred = [] 
        self.min_rel_pred = []
        self.max_rel_pred = []

        print("Processing Data .....")
        ### This should be timesteps differences?
        pbar = tqdm(total=len(self.seq_start_end) - len(self.seq_start_end) % self.skip) 
        # pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end) - len(self.seq_start_end) % self.skip):
        # for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            
            start, end = self.seq_start_end[ss]

            v_, a_, _, _ = self.seq_to_graph(self.in_traj[start:end, :], self.in_traj_rel[start:end, :], is_sc=self.is_normalize)
            self.v_in.append(v_.clone())
            self.A_in.append(a_.clone())

            v_, a_, min_rel, max_rel = self.seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], is_sc=self.is_normalize)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())

            self.min_rel_pred.append(min_rel)
            self.max_rel_pred.append(max_rel)
        pbar.close()

        # Save the arrays in the pickle file
        if self.is_normalize:
            torch.save((self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, 
                        self.non_linear_ped, self.loss_mask, self.v_in, self.A_in, self.v_pred, 
                        self.A_pred, self.num_seq, self.seq_start_end, self.min_rel_pred, self.max_rel_pred),  
                        osp.join('datasets', self.dataset, 'preprocessed', self.sdd_loc, self.train_mode, 'trajnet_normalize_in{}_out{}_aggframe{}.pkl').format(self.in_channels, self.out_channels, self.agg_frame))
        else:
            torch.save((self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, 
                        self.non_linear_ped, self.loss_mask, self.v_in, self.A_in, self.v_pred, 
                        self.A_pred, self.num_seq, self.seq_start_end, self.min_rel_pred, self.max_rel_pred),  
                        osp.join('datasets', self.dataset, 'preprocessed', self.sdd_loc, self.train_mode, 'trajnet_unnormalize_in{}_out{}_aggframe{}.pkl').format(self.in_channels, self.out_channels, self.agg_frame))


    def seq_to_graph(self, seq_, seq_rel, is_sc=False):
        seq_ = seq_.squeeze()
        seq_rel = seq_rel.squeeze()

        seq_len = seq_.shape[2]
        max_nodes = seq_.shape[0]
        min_rel = torch.tensor([0, 0])
        max_rel = torch.tensor([1, 1])
        
        V = np.zeros((seq_len, max_nodes, 2))
        A = np.zeros((seq_len, max_nodes, max_nodes))
        for s in range(seq_len):
            step_ = seq_[:, :, s]
            if is_sc and seq_rel.shape[0] != 0:
                step_rel, min_rel, max_rel = min_max_normalize(seq_rel[:, :, s])
            else:
                step_rel = seq_rel[:, :, s]
            for h in range(len(step_)): 
                V[s, h, :] = step_rel[h]
                A[s, h, h] = 1
                for k in range(h+1, len(step_)):
                    l2_norm = anorm(step_rel[h], step_rel[k])
                    A[s, h, k] = l2_norm
                    A[s, k, h] = l2_norm
            if seq_rel.shape[0] != 0:
                if self.norm_lap_matr: 
                    G = nx.from_numpy_array(A[s, :, :])
                    A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

        return torch.from_numpy(V).type(torch.float),\
            torch.from_numpy(A).type(torch.float), min_rel, max_rel


    def create_edge(self, road_grid, ped_od):
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
        ### OD matrix
        od_grid = torch.zeros((self.grid**2, self.grid**2))

        for p in range(len(self.ru_in_seq)):
            if len(ped_od[p]) > 1:
                # print(ped_od[p][-2], ped_od[p][-1])
                od_grid[ped_od[p][-2]][ped_od[p][-1]] += 1
                
        is_ped = road_grid > 0
        node_num = len(road_grid) ** 2
        node_idx = []
        for y in range(self.grid):
            for x in range(self.grid):
                if is_ped[y][x]:
                    if x == 0 and y == 0:
                        node_idx.append(x)
                    elif x == 0 and y != 0: 
                        node_idx.append(self.grid * y)
                    else:
                        node_idx.append(x * y)

        edge_index = np.zeros((2, node_num**2))
        # Say they're all connected if there is more than one people have ever existed
        for i in range(len(node_idx)): # src
            for j in range(len(node_idx)): # dst
                edge_index[0][len(node_idx)*i+j] = node_idx[i] # src list
                edge_index[1][len(node_idx)*i+j] = node_idx[j] # dst list

        # edge_attr = None
        # edge_attrs = [torch.ones(1) for _ in range(node_num**2)]
        # edge_attr = torch.cat(edge_attrs, dim=-1)
        # edge_attr = torch.cat(od_grid, dim=-1)

        return edge_index, od_grid.view(-1)


    def __len__(self):
        # return len(self.processed_file_names)
        return math.floor(self.num_seq / self.skip)


    def get(self, num_timesteps_in, num_timesteps_out) -> DynamicGraphTemporalSignal:
        features = []
        target = []
        edge_index_out = []
        edge_attr_out = []

        # edge_index, edge_attr, node_data = torch.load(osp.join('datasets', self.dataset, 'preprocessed', self.sdd_loc, self.train_mode, 'roadnet_in{}_out{}_aggframe{}.pkl').format(self.in_channels, self.out_channels, self.agg_frame))
        
        node_data = torch.stack(self.node_data_list, dim=1).permute(0, 2, 1)
        node_data = torch.as_tensor(node_data)


        # self.edge_index_list = torch.stack(self.edge_index_list, dim=1)
        self.edge_attr_list = torch.stack(self.edge_attr_list, dim=1).permute(1, 0)

        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(node_data.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]
        
        for i, j in indices:
            features.append((node_data[:, :-1, i:i+num_timesteps_in]).numpy())
            target.append((node_data[:, -1, i+num_timesteps_in:j]).numpy())
            edge_index_out.append(np.asarray(self.edge_index_list[i:i+num_timesteps_in]))
            edge_attr_out.append(np.asarray(self.edge_attr_list[i:i+num_timesteps_in]))

        

        data = DynamicGraphTemporalSignal(edge_index_out, edge_attr_out, features, target)
        # data = Data(x=features, edge_index=edge_index_out, edge_attr=edge_attr_out, y=target)
        # data = StaticGraphTemporalSignal(self.edge_index, self.edge_attr, features, target)

        return data


    def __getitem__(self, index):


        start, end = self.seq_start_end[index]
        out = [
            self.in_traj[start:end, :], self.pred_traj[start:end, :],
            self.in_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_in[index], self.A_in[index],
            self.v_pred[index], self.A_pred[index],
            torch.stack(self.min_rel_pred)[index], torch.stack(self.max_rel_pred)[index],
        ]
        return out



class GroupTrajectoryDataset(Dataset):
    """
        Dataloder for the Group Trajectory datasets
        Except inD-dataset
    """
    def __init__(
        self, data_dir, sdd_loc, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True, is_train=True, is_preprocessed=False,
        dataset='eth'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        - sdd_loc: Location in SDD
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(GroupTrajectoryDataset, self).__init__()

        self.is_preprocessed = is_preprocessed

        self.is_train = is_train
        self.sc = MinMaxScaler(feature_range=(0, 1))

        # Should be preprocessed
        # data_file_train = osp.join('datasets', dataset, 'preprocessed', 'all_train.pkl')
        # data_file_test = osp.join('datasets', dataset, 'preprocessed', 'all_test.pkl')

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.sdd_loc = sdd_loc
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.threshold = threshold
        self.min_ped = min_ped
        self.dataset = dataset

        if self.is_train:
            all_files = os.listdir(osp.join(self.data_dir, self.sdd_loc, 'train'))
        else:
            all_files = os.listdir(osp.join(self.data_dir, self.sdd_loc, 'test'))

        if self.is_train:
            self.all_files = [os.path.join(self.data_dir, self.sdd_loc, 'train', _path) for _path in all_files]
        else:
            self.all_files = [os.path.join(self.data_dir, self.sdd_loc, 'test', _path) for _path in all_files]
        self.num_peds_in_seq = []
        self.seq_list = []
        self.seq_list_rel = []
        self.loss_mask_list = []
        self.non_linear_ped = []

        # List of data directories where raw data resides
        self.train_data_dir = osp.join('datasets', dataset, 'preprocessed', self.sdd_loc, 'train/')
        self.test_data_dir = osp.join('datasets', dataset, 'preprocessed', self.sdd_loc, 'test/')
        self.val_data_dir = osp.join('datasets', dataset, 'preprocessed', self.sdd_loc, 'val/')

        # Define the path in which the process data would be stored
        self.data_file_tr = osp.join(self.train_data_dir, "trajectories_train_in{}_out{}.pkl".format(obs_len, pred_len))
        self.data_file_te = osp.join(self.test_data_dir, "trajectories_test_in{}_out{}.pkl".format(obs_len, pred_len))
        self.data_file_vl = osp.join(self.val_data_dir, "trajectories_val_in{}_out{}.pkl".format(obs_len, pred_len))

        if self.is_preprocessed:
            # if is_train:
            #     f = open(data_file_train, "rb")
            # else:
            #     f = open(data_file_test, "rb")
            # self.all_frames, self.all_tracks, self.all_static, self.all_meta = pickle.load(f)

            if self.is_train:
                # f = open(self.data_file_tr, "rb")
                f = self.data_file_tr
            else:
                # f = open(self.data_file_te, "rb")
                f = self.data_file_te
            self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, self.non_linear_ped, \
                self.loss_mask, self.v_in, self.A_in, self.v_pred, self.A_pred, self.num_seq, self.seq_start_end = torch.load(f)

            # f.close()
        else:
            if self.is_train:
                data_file = self.data_file_tr
            else:
                data_file = self.data_file_te
            
            self.set_graph_data(data_file)


    def read_file(self, _path, delim='\t'):
        data = []
        if delim == 'tab':
            delim = '\t'
        elif delim == 'space':
            delim = ' '
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                data.append(line)
        return np.asarray(data)


    def read_file_sdd(self, _path, delim='\t'):
        data = []
        if delim == 'tab':
            delim = '\t'
        elif delim == 'space':
            delim = ' '
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(" ")
                line = [float(i) for i in line[:-1]]

                # Adjust to the other formats
                xcenter = (line[1] + line[3]) / 2
                ycenter = (line[2] + line[4]) / 2

                line = [line[5], line[0], xcenter, ycenter]
                line = [float(i) for i in line]

                data.append(line)
        return np.asarray(data)


    def set_graph_data(self, data_file):
        """
        Data format for SDD:
            Track ID, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label
            Format description from here: https://github.com/flclain/StanfordDroneDataset

        Data format for others:
            Frame number, Track ID, center x coord, center y coord

        """
        for path in self.all_files:
            if self.dataset == 'sdd':
                # data_ped, data_bik = self.read_file_sdd(path, self.delim)
                data = self.read_file_sdd(path, self.delim)
            else:
                data = self.read_file(path, self.delim)

            print(data.shape)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            print(f'Length of frames: {len(frames)}')
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])

            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / self.skip))

            # For each road users on a frame
            for idx in range(0, int(len(frame_data) * self.use_ratio)):

                x_list = []
                y_list = []
                node_data_list = []

                ped_frame_list = []
                car_frame_list = []
                bik_frame_list = []

                ru_in_curr_seq, _idx = np.unique(frame_data[idx], axis=1, return_index=True)
                ru_in_curr_seq = ru_in_curr_seq[:, np.argsort(_idx)]

                for ridx in range(len(ru_in_curr_seq)):

                    # Need to get their id, location, location_rel (direction)
                    print(ridx)
            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.skip * self.seq_len:self.skip], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                # Going to be the scene for each timestep, which will be separated into train and prediction sequences.
                for _, t in enumerate(frame_data[idx:idx + self.skip * self.seq_len:self.skip]):
                    # Will need pedestrian locations, pedestrian id
                    # Connecting between them.
                    unique_frames = np.unique(curr_seq_data[:, 0])
                    for _, f_id in enumerate(unique_frames):
                        curr_frame_data = curr_seq_data[curr_seq_data[:, 0] == f_id, :]
                        curr_frame_seq = np.around(curr_frame_data, decimals=4)
                        pad_front = frames.index(curr_frame_seq[0, 0]) - idx
                        pad_end = frames.index(curr_frame_seq[-1, 0]) - idx + self.skip
                        if pad_end - pad_front != self.seq_len * self.skip:
                            continue
                        curr_frame_seq = np.transpose(curr_frame_seq[:, 2:])

                        print("called")
                        print(curr_frame_seq)

                        # Assign pedestrians to node, (frame ID, pedestrian ID, (coord_x, coord_y))
                        # Velocity or angle, compare with previous location if the pedestrian was there.

                        # Assign their location differences with other pedestrians to edge, sqrt((coord_x1, coord_y1) - (coord_x2, coord_y2))^2

                        # Relative coordinates
                        # rel_curr_frame_seq = np.zeros(curr_frame_seq.shape)
                        # rel_curr_ped_seq[:, 1:] = \
                        #     curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]


                    break


                # for _, ped_id in enumerate(peds_in_curr_seq):
                #     curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                #     curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                #     pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                #     pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + self.skip
                #     if pad_end - pad_front != self.seq_len * self.skip:
                #         continue
                #     curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                #     # Make coordinates relative
                #     rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                #     rel_curr_ped_seq[:, 1:] = \
                #         curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                #     _idx = num_peds_considered

                #     curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                #     curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                #     # Linear vs Non-Linear Trajectory
                #     _non_linear_ped.append(
                #         poly_fit(curr_ped_seq, self.pred_len, self.threshold))
                #     curr_loss_mask[_idx, pad_front:pad_end] = 1
                #     num_peds_considered += 1
                #     # print(num_peds_considered)

                # if num_peds_considered > self.min_ped:
                #     self.non_linear_ped += _non_linear_ped
                #     self.num_peds_in_seq.append(num_peds_considered)
                #     self.loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                #     self.seq_list.append(curr_seq[:num_peds_considered])
                #     self.seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(self.seq_list)
        self.seq_list = np.concatenate(self.seq_list, axis=0)
        self.seq_list_rel = np.concatenate(self.seq_list_rel, axis=0)
        self.loss_mask_list = np.concatenate(self.loss_mask_list, axis=0)
        self.non_linear_ped = np.asarray(self.non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.in_traj = torch.from_numpy(
            self.seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            self.seq_list[:, :, self.obs_len:]).type(torch.float)
        self.in_traj_rel = torch.from_numpy(
            self.seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            self.seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(self.loss_mask_list).type(torch.float)
        # self.non_linear_ped = torch.from_numpy(self.non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(self.num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        # Convert to Graphs 
        self.v_in = [] 
        self.A_in = [] 
        self.v_pred = [] 
        self.A_pred = [] 

        # Normalization
        # self.in_traj = self.sc.fit_transform(self.in_traj)
        # self.in_traj_rel = self.sc.fit_transform(self.in_traj_rel)
            
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_, a_ = seq_to_graph(self.in_traj[start:end,:], self.in_traj_rel[start:end, :], self.norm_lap_matr, sc=self.sc)

            self.v_in.append(v_.clone())
            self.A_in.append(a_.clone())
            v_, a_ = seq_to_graph(self.pred_traj[start:end,:], self.pred_traj_rel[start:end, :], self.norm_lap_matr, sc=self.sc)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

        # Save the arrays in the pickle file
        # f = open(data_file, "wb")
        # pickle.dump((self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, self.non_linear_ped, self.loss_mask, self.v_in, self.A_in, self.v_pred, self.A_pred, self.num_seq, self.seq_start_end), f, protocol=2)
        torch.save((self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, self.non_linear_ped, self.loss_mask, self.v_in, self.A_in, self.v_pred, self.A_pred, self.num_seq, self.seq_start_end), data_file)
        # f.close()

    def __len__(self):
        if self.is_preprocessed:
            if self.is_train:
                # f = open(self.data_file_tr, "rb")
                self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, self.non_linear_ped, self.loss_mask, self.v_in, self.A_in, self.v_pred, self.A_pred, self.num_seq, self.seq_start_end = torch.load(self.data_file_tr)
            else:
                self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, self.non_linear_ped, self.loss_mask, self.v_in, self.A_in, self.v_pred, self.A_pred, self.num_seq, self.seq_start_end = torch.load(self.data_file_te)
                # f = open(self.data_file_te, "rb")
            # self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, self.non_linear_ped, self.loss_mask, self.v_in, self.A_in, self.v_pred, self.A_pred, self.num_seq, self.seq_start_end = pickle.load(f)
            # self.in_traj, self.pred_traj, self.in_traj_rel, self.pred_traj_rel, self.non_linear_ped, self.loss_mask, self.v_in, self.A_in, self.v_pred, self.A_pred, self.num_seq, self.seq_start_end = torch.load(self.data_file_tr)
            # f.close()
        return self.num_seq

    def __getitem__(self, index):
        
        start, end = self.seq_start_end[index]

        out = [
            self.in_traj[start:end, :], self.pred_traj[start:end, :],
            self.in_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            # self.loss_mask[start:end, :], self.loss_mask[start:end, :],
            self.v_in[index], self.A_in[index],
            self.v_pred[index], self.A_pred[index]
        ]
        return out