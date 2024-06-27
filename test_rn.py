from os import path as osp
import torch
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils.static_graph_temporal_signal import temporal_signal_split
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from utils.metrics import * 
from models.RNGCN import RNTransformer
from data_loader import RoadNetwork, RoadNetworkPeds, TrajectoryDataset


parser = argparse.ArgumentParser(description="Parameter Settings for Training")
# dataset options
parser.add_argument('--dataset_dir', default="./datasets/",
                help="Path to directory that contains the dataset csv files.", type=str)
parser.add_argument('--sdd_loc', default="",
                help="Location of SDD", type=str)
parser.add_argument('--dataset', default="inD-dataset-v1.0",
                help="Name of the dataset. Needed to apply dataset specific visualization adjustments.",
                type=str)
parser.add_argument('--pretrained_dir', default="./pretrained",
                help="Path to directory that contains the pretrained model.", type=str)

# Test options
parser.add_argument('--seed', default=42,
                help="seed number", type=int)
parser.add_argument('--epochs', default="30",
                help="Epochs for train, validation, and test suite", type=int)
parser.add_argument('--pretrained_file', default="val_29.pt", type=str,
                help="Path to pretrained file which wants to be validated")
parser.add_argument('--is_preprocessed', action="store_true",
                help="If preprocessed file exists")
parser.add_argument('--train_num', default="1",
                help="Number of training dataset the model splits", type=int)
parser.add_argument('--test_num', default="1",
                help="Number of testing dataset the model splits", type=int)
parser.add_argument('--rn_num', default="1",
                help="Number of road network dataset the model splits", type=int)

parser.add_argument('--pretrained_model', default="model_0.pt",
                help="Name of pretrained model", type=str)
parser.add_argument('--rn_num_timesteps_in', default="8",
                help="Number of timesteps for road network input", type=int)
parser.add_argument('--rn_num_timesteps_out', default="12",
                help="Number of timesteps for road network output", type=int)
parser.add_argument('--rn_num_timesteps_out_list', default="8",
                help="List of the number of timesteps for road network output", nargs='+', type=list)
parser.add_argument('--is_horizontal_pred', default=False,
                help="If the model is trained time horizontaly, it is true", type=bool)
parser.add_argument('--is_rn', action="store_true", default=False,
                help="If road network is taken in the learning phase")
parser.add_argument('--grid', default="2",
                help="Number of grid on one side")
parser.add_argument('--use_lrschd', action="store_true", default=False,
                help='Use lr rate scheduler')
parser.add_argument('--lr_sh_rate', type=int, default=150,
                help='number of steps to drop the lr')
parser.add_argument('--clip_grad', type=float, default=None,
                help='gradient clipping')
parser.add_argument('--agg_frame', default="20",
                help="Aggregated number of frames")
parser.add_argument('--is_rn_preprocessed', action="store_true", default=False,
                help="If preprocessed file exists")
parser.add_argument('--tr', '--train_ratio', default=0.8, type=float,
                help="Train ratio")
parser.add_argument('--skip', default=12, type=int,
                help="Frame adjustable parameter. 12 is 2.5 FPS for SDD.")
parser.add_argument('--is_normalize', action="store_true", default=False,
                help="")

opt = parser.parse_args()

# gpu settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


def load_data(opt):
    dataset_dir = osp.join(opt.dataset_dir, opt.dataset)
    global out_list

    if opt.dataset == 'inD-dataset-v1.0':
        print("Load Road Network")
        road_network = RoadNetwork(dataset_dir, opt.train_num, opt.test_num, opt.rn_num, 
                                   opt.grid, opt.agg_frame, num_timesteps_in=opt.rn_num_timesteps_in, 
                                   num_timesteps_out=opt.rn_num_timesteps_out, is_preprocessed=opt.is_rn_preprocessed)
        dataset = road_network.get_dataset() 
        train_rn_dataset, test_rn_dataset = temporal_signal_split(dataset, train_ratio=opt.tr)
    else:
        print("Load Road Network")
        # out_list = [8, 16, 24, 36]
        # out_list = [8, 16, 24]
        out_list = [12]
        # out_list = [4, 8, 16]
        road_network = [object for _ in range(len(out_list))]
        dataset = [Dataset for _ in range(len(out_list))]
        # train_rn_dataset, test_rn_dataset = [Dataset for _ in range(len(out_list))], [Dataset for _ in range(len(out_list))]
        test_rn_loader_list = []
        for i in range(len(out_list)):            

            test_dataset = TrajectoryDataset(
                    dataset_dir,
                    sdd_loc=opt.sdd_loc,
                    in_channels=opt.rn_num_timesteps_in,
                    out_channels=opt.rn_num_timesteps_out,
                    rn_out_channels=out_list[i],
                    agg_frame=opt.agg_frame,
                    skip=opt.skip,
                    norm_lap_matr=True,
                    is_preprocessed=opt.is_rn_preprocessed,
                    dataset_iter=i,
                    dataset=opt.dataset,
                    train_mode='test',
                    is_rn=opt.is_rn,
                    is_normalize=opt.is_normalize)
            # road_network[i] = RoadNetworkPeds(dataset_dir, opt.sdd_loc, opt.train_num, opt.test_num, 
            #                             opt.rn_num, opt.grid, opt.agg_frame, num_timesteps_in=opt.rn_num_timesteps_in, 
            #                             num_timesteps_out=out_list[i], skip=opt.skip, is_preprocessed=opt.is_rn_preprocessed)
            # dataset[i] = road_network[i].get_dataset() 
            # train_rn_dataset[i], test_rn_dataset[i] = temporal_signal_split(dataset[i], train_ratio=opt.tr)
            test_rn_dataset = test_dataset.get(opt.rn_num_timesteps_in, out_list[i])
            test_rn_loader_list.append(test_rn_dataset)

    return test_rn_loader_list


# Test phase
@torch.no_grad()
def test_rn(test_rn_dataset):
    mse8, mse16, mse24 = [], [], []
    mse = []

    model_rn.eval()

    for i, batch in tqdm(enumerate(test_rn_dataset[0])):
    # for i, batch in tqdm(enumerate(zip(test_rn_dataset[0], test_rn_dataset[1], test_rn_dataset[2]))):
        # rn8, rn16, rn24 = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        # x = [batch[i].x for i in range(len(batch))]
        # edge_index = [batch[i].edge_index for i in range(len(batch))]
        # edge_attr = [batch[i].edge_attr for i in range(len(batch))]
        x = [batch.x.to(device)]
        y = [batch.y.to(device)]
        edge_index = [batch.edge_index.to(device)]
        edge_attr = [batch.edge_attr.to(device)]

        out = model_rn(x, edge_index, edge_attr)
        mse.append(((out - y[0])**2).cpu())

        # out8, out16, out24 = model_rn(x, edge_index, edge_attr)
        # mse8.append(((out8 - rn8.y)**2).cpu()) # + ((out16 - rn16.y)**2).cpu() # + ((out24 - rn24.y)**2).cpu() + ((out32 - rn32.y)**2).cpu())
        # mse16.append(((out16 - rn16.y)**2).cpu()) 
        # mse24.append(((out24 - rn24.y)**2).cpu()) 
    # return float(torch.cat(mse8, dim=0).mean().sqrt()), float(torch.cat(mse8, dim=0).mean()), \
    #     float(torch.cat(mse16, dim=0).mean().sqrt()), float(torch.cat(mse16, dim=0).mean()), \
    #     float(torch.cat(mse24, dim=0).mean().sqrt()), float(torch.cat(mse24, dim=0).mean())
    return float(torch.cat(mse, dim=0).mean().sqrt()), float(torch.cat(mse, dim=0).mean())


def main():

    test_rn_dataset = load_data(opt)

    ### Model saving directory
    print("===> Save model to %s" % opt.pretrained_dir)

    print("===== Initializing model for road network =====")

    # if opt.dataset == "sdd":
    #     num_node_features = 7
    # else:
    num_node_features = 7

    num_nodes = len(next(iter(test_rn_dataset[0])).x)
    print(out_list)

    # model_path = osp.join(opt.pretrained_dir, 'road_network', 'model_grid{}_outlist{}_{}_{}_epoch{}.pt'.format(opt.grid, out_list[0], out_list[1], out_list[2], opt.epochs))
    model_path = osp.join(opt.pretrained_dir, 'road_network', 'model_grid{}_outlist{}_epoch{}.pt'.format(opt.grid, out_list[0], opt.epochs))
    # model_path = osp.join(opt.pretrained_dir, 'social_stgcnn', opt.dataset, 'model_grid{}_epoch{}.pt'.format(opt.grid, opt.epochs))

    global model_rn
    model_rn = RNTransformer(node_features=num_node_features, num_nodes=num_nodes, periods=opt.rn_num_timesteps_in, output_dim_list=out_list, device=device).to(device)
    model_rn.load_state_dict(torch.load(model_path))

    # rmse, mse, rmse16, mse16, rmse24, mse24 = test_rn(test_rn_dataset)

    # print(f'Accuracy. RMSE: {rmse}, MSE: {mse}, RMSE 16: {rmse16}, MSE 16: {mse16}, RMSE 24: {rmse24}, MSE 24: {mse24}')

    rmse, mse = test_rn(test_rn_dataset)

    print(f'Accuracy. RMSE: {rmse}, MSE: {mse}')


if __name__ == '__main__':
    main()