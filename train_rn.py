import sys
import os
from os import path as osp
import argparse
import datetime
import pytz
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from utils.static_graph_temporal_signal import temporal_signal_split
from tqdm import tqdm
import logging
import pickle
import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from utils.metrics import *
from models.RNGCN import RNTransformer
from data_loader import inDDatasetGraph, RoadNetwork, RoadNetworkPeds, TrajectoryDataset

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

log_path = './logs'
os.makedirs(log_path, exist_ok=True)

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description="Parameter Settings for Training")
# --- Input ---
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

# training options
parser.add_argument('--seed', default=42,
                help="seed number", type=int)
parser.add_argument('--epochs', default="10",
                help="Epochs for train, validation, and test suite", type=int)
parser.add_argument('--train_num', default="1",
                help="Number of training dataset the model splits", type=int)
parser.add_argument('--test_num', default="1",
                help="Number of testing dataset the model splits", type=int)
parser.add_argument('--rn_num', default="1",
                help="Number of road network dataset the model splits", type=int)
parser.add_argument('--optimizer', default="Adam",
                help="Name of the optimizer we use for train", type=str)
parser.add_argument('--model_name', default="social_stgcnn",
                help="Model name", type=str)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--is_pretrained', action="store_true", default=False,
                help="Use pretrained model")
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
parser.add_argument('--grid', default="4",
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
parser.add_argument('--uid', default=0, type=int,
                help="Unique ID")

opt = parser.parse_args()

# gpu settings
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
random.seed(opt.seed)
np.random.seed(opt.seed)

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
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
        out_list = [1, 4, 8]
        # out_list = [12]
        road_network = [object for _ in range(len(out_list))]
        # train_rn_dataset, test_rn_dataset = [Dataset for _ in range(len(out_list))], [Dataset for _ in range(len(out_list))]
        train_rn_loader_list, test_rn_loader_list = [], []
        for i in range(len(out_list)):
            train_dataset = TrajectoryDataset(
                    dataset_dir,
                    sdd_loc=opt.sdd_loc,
                    in_channels=opt.rn_num_timesteps_in,
                    out_channels=opt.rn_num_timesteps_out,
                    rn_out_channels=out_list[i],
                    agg_frame=opt.agg_frame,
                    skip=opt.skip,
                    grid=opt.grid,
                    norm_lap_matr=True,
                    is_preprocessed=opt.is_rn_preprocessed,
                    dataset_iter=i,
                    dataset=opt.dataset,
                    train_mode='train',
                    is_rn=opt.is_rn,
                    is_normalize=opt.is_normalize)

            test_dataset = TrajectoryDataset(
                    dataset_dir,
                    sdd_loc=opt.sdd_loc,
                    in_channels=opt.rn_num_timesteps_in,
                    out_channels=opt.rn_num_timesteps_out,
                    rn_out_channels=out_list[i],
                    agg_frame=opt.agg_frame,
                    skip=opt.skip,
                    grid=opt.grid,
                    norm_lap_matr=True,
                    is_preprocessed=opt.is_rn_preprocessed,
                    dataset_iter=i,
                    dataset=opt.dataset,
                    train_mode='val',
                    is_rn=opt.is_rn,
                    is_normalize=opt.is_normalize)
            # road_network[i] = RoadNetworkPeds(
            #         dataset_dir, opt.sdd_loc, opt.train_num, opt.test_num, 
            #                             opt.rn_num, opt.grid, opt.agg_frame, num_timesteps_in=opt.rn_num_timesteps_in, 
            #                             num_timesteps_out=out_list[i], skip=opt.skip, is_preprocessed=opt.is_rn_preprocessed)
            # dataset[i] = road_network[i].get_dataset() 
            # train_rn_dataset[i], test_rn_dataset[i] = temporal_signal_split(dataset[i], train_ratio=opt.tr)
            train_rn_dataset = train_dataset.get(opt.rn_num_timesteps_in, out_list[i])
            test_rn_dataset = test_dataset.get(opt.rn_num_timesteps_in, out_list[i])
            train_rn_loader_list.append(train_rn_dataset)
            test_rn_loader_list.append(test_rn_dataset)

            # print(next(iter(train_rn_dataset[i])))
    return train_rn_loader_list, test_rn_loader_list


def train_rn(train_rn_dataset):
    """
    Train model for time horizon prediction with the road network
    Currently using A3TGCN
    """
    # Simple temporal Graph Neural Network
    
    loss = 0
    rn_out_list = [_ for _ in range(3)]

    model_rn.train()
    step = 0

    if len(train_rn_dataset) == 1:
        pbar = tqdm(enumerate(train_rn_dataset[0]))
    else:
        pbar = tqdm(enumerate(zip(train_rn_dataset[0], train_rn_dataset[1], train_rn_dataset[2])))
        
    for i, batch in pbar:
        optimizer_rn.zero_grad()

        if len(train_rn_dataset) == 1:
            x = [batch.x.to(device)]
            y = [batch.y.to(device)]
            edge_index = [batch.edge_index.to(device)]
            edge_attr = [batch.edge_attr.to(device)]
        else:
            x = [batch[i].x.to(device) for i in range(len(batch))]
            y = [batch[i].y.to(device) for i in range(len(batch))]

            edge_index = [batch[i].edge_index.to(device) for i in range(len(batch))]
            edge_attr = [batch[i].edge_attr.to(device) for i in range(len(batch))]

        _, rn_out_list[0], rn_out_list[1], rn_out_list[2] = model_rn(x, edge_index, edge_attr)

        if len(rn_out_list) > 4:
            loss += torch.mean((rn_out_list[0] - y[0])**2).cpu()
        else:
            for i in range(len(rn_out_list)):
                loss += torch.mean((rn_out_list[i] - y[i])**2).cpu()

        step += 1

    loss = loss / (step+1)
    loss.backward()
    optimizer_rn.step()

    out_loss = loss.detach()

    return out_loss


# Test phase
@torch.no_grad()
def test_rn(test_rn_dataset):
    mse8, mse16, mse24 = [], [], []
    mse_list = []
    rn_out_list = [_ for _ in range(3)]

    model_rn.eval()

    # for i, batch in tqdm(enumerate(test_rn_dataset[0])):
    if len(test_rn_dataset) == 1:
        pbar = tqdm(enumerate(test_rn_dataset[0]))
    else:
        pbar = tqdm(enumerate(zip(test_rn_dataset[0], test_rn_dataset[1], test_rn_dataset[2])))
        
    for i, batch in pbar:

        if len(test_rn_dataset) == 1:
            x = [batch.x.to(device)]
            y = [batch.y.to(device)]
            edge_index = [batch.edge_index.to(device)]
            edge_attr = [batch.edge_attr.to(device)]
        else:
            x = [batch[i].x.to(device) for i in range(len(batch))]
            y = [batch[i].y.to(device) for i in range(len(batch))]
            edge_index = [batch[i].edge_index.to(device) for i in range(len(batch))]
            edge_attr = [batch[i].edge_attr.to(device) for i in range(len(batch))]
        
        _, out8, out16, out24 = model_rn(x, edge_index, edge_attr)
        # rn_out_list = model_rn(x, edge_index, edge_attr)

        # mse_list.append(((rn_out_list - y[0])**2).cpu())
        # for i in range(len(rn_out_list)):
        #     loss += torch.mean((rn_out_list[i] - y[i])**2).cpu()

        mse8.append(((out8 - y[0])**2).cpu()) # + ((out16 - rn16.y)**2).cpu() # + ((out24 - rn24.y)**2).cpu() + ((out32 - rn32.y)**2).cpu())
        mse16.append(((out16 - y[1])**2).cpu()) 
        mse24.append(((out24 - y[2])**2).cpu()) 
    # return float(torch.cat(mse_list, dim=0).mean().sqrt()), float(torch.cat(mse_list, dim=0).mean())
    return float(torch.cat(mse8, dim=0).mean().sqrt()), float(torch.cat(mse8, dim=0).mean()), \
        float(torch.cat(mse16, dim=0).mean().sqrt()), float(torch.cat(mse16, dim=0).mean()), \
        float(torch.cat(mse24, dim=0).mean().sqrt()), float(torch.cat(mse24, dim=0).mean())


def main():
    h = None
    loss_list = []

    train_rn_dataset, test_rn_dataset = load_data(opt)

    ### Model saving directory
    print("===> Save model to %s" % opt.pretrained_dir)

    os.makedirs(opt.pretrained_dir, exist_ok=True)

    print("===== Initializing model for road network =====")

    num_node_features = 7

    num_nodes = len(next(iter(train_rn_dataset[0])).x)
    print(out_list)

    global model_rn, optimizer_rn
    model_rn = RNTransformer(node_features=7, num_nodes=num_nodes, periods=opt.rn_num_timesteps_in, output_dim_list=out_list, device=device).to(device)

    # Training settings for road network
    # optimizer_rn = optim.RMSprop(model_rn.parameters(), lr=1e-2, weight_decay=5e-5)
    # optimizer_rn = optim.RMSprop(model_rn.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer_rn = torch.optim.SGD(model_rn.parameters(), lr=1e-2, weight_decay=1e-3)

    if opt.use_lrschd:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_rn, step_size=opt.lr_sh_rate, gamma=0.2)

    total_param = 0
    for param_tensor in model_rn.state_dict():
        print(param_tensor, '\t', model_rn.state_dict()[param_tensor].size())
        total_param += np.prod(model_rn.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    for epoch in tqdm(range(opt.epochs + 1)):

        train_loss = train_rn(train_rn_dataset)

        rmse, mse, rmse16, mse16, rmse24, mse24 = test_rn(test_rn_dataset)
        # rmse, mse = test_rn(test_rn_dataset)

        print("Road Network - Epoch: {}, Train Loss: {:.4f}, Test RMSE 8: {:.4f}, Test RMSE 16: {:4f}, Test RMSE 24: {:4f}".format(epoch, train_loss, rmse, rmse16, rmse24))
        # print("Road Network - Epoch: {}, Train Loss: {:.4f}, Test RMSE 8: {:.4f}".format(epoch, train_loss, rmse))

        # logger.info("Road Network - Train Loss: {:.4f}, Test RMSE: {:.4f}".format(train_loss, rmse))
        loss_list.append(train_loss)
        writer.add_scalar('Long term: Loss/train', train_loss, epoch)
        writer.add_scalar('Long term: Accuracy/test', rmse, epoch)

        if epoch % 10 == 0:
            # torch.save(model_rn.state_dict(), osp.join(opt.pretrained_dir, 'road_network', 'model_grid{}_outlist{}_{}_{}_epoch{}.pt'.format(opt.grid, out_list[0], out_list[1], out_list[2], epoch)))
            torch.save(model_rn.state_dict(), osp.join(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_out_list{}_{}_{}_epoch{}.pt'.format(opt.uid, opt.grid, out_list[0], out_list[1], out_list[2], epoch)))
            # torch.save(model_rn.state_dict(), osp.join(opt.pretrained_dir, 'road_network', 'model_grid{}_outlist{}_epoch{}.pt'.format(opt.grid, out_list[0], epoch)))

        if opt.use_lrschd:
            scheduler.step()

    writer.close()

if __name__ == '__main__':
    main()