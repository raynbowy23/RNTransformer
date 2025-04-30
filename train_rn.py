import sys
import os
from os import path as osp
import argparse
import datetime
import pytz
import torch
from torch.profiler import ProfilerActivity
from torch import nn
from utils.static_graph_temporal_signal import temporal_signal_split
from tqdm import tqdm
import logging
import pickle
import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from utils.metrics import *
from models.RNGCN import RNTransformer
# from data_loader import RoadNetwork, TrajectoryDataset
from data_loader import TrajectoryDataset

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
        out_list = [1, 4, 8]
        road_network = [object for _ in range(len(out_list))]
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
            train_rn_dataset = train_dataset.get(opt.rn_num_timesteps_in, out_list[i])
            test_rn_dataset = test_dataset.get(opt.rn_num_timesteps_in, out_list[i])
            train_rn_loader_list.append(train_rn_dataset)
            test_rn_loader_list.append(test_rn_dataset)

    return train_rn_loader_list, test_rn_loader_list


train_rn_dataset, test_rn_dataset = load_data(opt)

### Model saving directory
print("===> Save model to %s" % opt.pretrained_dir)

os.makedirs(opt.pretrained_dir, exist_ok=True)

print("===== Initializing model for road network =====")

num_nodes = len(next(iter(train_rn_dataset[0])).x)
print(out_list)

model_rn = RNTransformer(node_features=7, num_nodes=num_nodes, periods=opt.rn_num_timesteps_in, output_dim_list=out_list, device=device).to(device)

# Training settings for road network
# optimizer_rn = torch.optim.SGD(model_rn.parameters(), lr=1e-2, weight_decay=1e-3)
optimizer_rn = torch.optim.Adam(model_rn.parameters(), lr=1e-3, weight_decay=1e-4)

if opt.use_lrschd:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_rn, step_size=opt.lr_sh_rate, gamma=0.2)

total_param = 0
for param_tensor in model_rn.state_dict():
    print(param_tensor, '\t', model_rn.state_dict()[param_tensor].size())
    total_param += np.prod(model_rn.state_dict()[param_tensor].size())
print('Net\'s total params:', total_param)

def huber_loss(rn_pred, rn_gt):
    h_loss = torch.nn.HuberLoss('mean', delta=0.1)
    return h_loss(rn_pred, rn_gt)

def train_rn(train_rn_dataset):
    """
    Train model for time horizon prediction with the road network
    """
    model_rn.train()
    total_loss = 0
    step = 0

    train_rn_dataset = [[b.to(device) for b in batch] for batch in zip(*train_rn_dataset)]

    for batch in tqdm(train_rn_dataset):
    # for i, batch in tqdm(enumerate(zip(*train_rn_dataset))):
        x, y, edge_index, edge_attr = zip(*[(b.x, b.y, b.edge_index, b.edge_attr) for b in batch])
        # x = [b.x.to(device) for b in batch]
        # y = [b.y.to(device) for b in batch]

        # edge_index = [b.edge_index.to(device) for b in batch]
        # edge_attr = [b.edge_attr.to(device) for b in batch]

        optimizer_rn.zero_grad()

        predictions = model_rn(x, edge_index, edge_attr)

        # batch_loss = 0.0
        # for i in range(len(predictions)):
        #     batch_loss += huber_loss(predictions[i], y[i])
        batch_loss = sum(huber_loss(p, y_) for p, y_ in zip(predictions, y))

        batch_loss.backward()
        optimizer_rn.step()

        total_loss += batch_loss.item()
        step += 1

    return total_loss / max(step, 1)


@torch.no_grad()
def test_rn(test_rn_dataset):
    n_horizon = len(test_rn_dataset)
    horizon_losses = [[] for _ in range(n_horizon)]

    model_rn.eval()

    test_rn_dataset = [[b.to(device) for b in batch] for batch in zip(*test_rn_dataset)]
    horizon_losses = [[] for _ in range(n_horizon)]

    # for i, batches in tqdm(enumerate(zip(*test_rn_dataset))):
    for batches in tqdm(test_rn_dataset):

        x, y, edge_index, edge_attr = zip(*[(b.x, b.y, b.edge_index, b.edge_attr) for b in batches])

        # x_list = []
        # y_list = []
        # edge_index_list = []
        # edge_attr_list = []
        # for i in range(n_horizon):
        #     batch = batches[i]
        #     x_list.append(batch.x.to(device))
        #     y_list.append(batch.y.to(device))
        #     edge_index_list.append(batch.edge_index.to(device))
        #     edge_attr_list.append(batch.edge_attr.to(device))

        # predictions = model_rn(x_list, edge_index_list, edge_attr_list)
        predictions = model_rn(x, edge_index, edge_attr)

        # Compute MSE for each horizon
        for i in range(n_horizon):
            mse = ((predictions[i] - y[i]) ** 2).mean()
            # horizon_losses[i].append(mse.cpu())
            horizon_losses[i].append(mse)

        # results = []
        # for i in range(n_horizon):
        #     # cat all MSE values for horizon i
        #     all_mse = torch.stack(horizon_losses[i], dim=0)  # shape [num_batches]
        #     mean_mse = float(all_mse.mean())                # average MSE across batches
        #     rmse = mean_mse ** 0.5                          # sqrt of MSE
        #     results.append((rmse, mean_mse))
    results = [(torch.stack(h).mean().sqrt().item(), torch.stack(h).mean().item()) for h in horizon_losses]

    return results


def main():
    loss_list = []

    _loss = [0 for _ in range(len(out_list))]

    # with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #                             record_shapes=True) as prof:

    for epoch in tqdm(range(opt.epochs + 1)):

        train_loss = train_rn(train_rn_dataset)

        # if len(out_list) == 1:
        #     _loss[0], _ = test_rn(test_rn_dataset)
        # elif len(out_list) == 2:
        #     _loss[0], _, _loss[1], _ = test_rn(test_rn_dataset)
        # elif len(out_list) == 3:
        #     _loss[0], _, _loss[1], _, _loss[2], _ = test_rn(test_rn_dataset)

        # print("Road Network - Epoch: {}, Train Loss: {:.4f}, Test Loss 1: {:.4f}, Test Loss 2: {:4f}, Test Loss 3: {:4f}".format(epoch, train_loss, _loss[0], _loss[1], _loss[2]))

        test_results = test_rn(test_rn_dataset)  

        loss_strings = []
        for i, (rmse, mse) in enumerate(test_results, 1):
            loss_strings.append(f"[H{i} RMSE: {rmse:.4f}, MSE: {mse:.4f}]")

        # Join them for printing:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, " + " ".join(loss_strings))


        loss_list.append(train_loss)
        writer.add_scalar('Long term: Loss/train', train_loss, epoch)
        writer.add_scalar('Long term: Accuracy/test', _loss[0], epoch)

        if epoch % 10 == 0:
            if len(out_list) == 1:
                torch.save(model_rn.state_dict(), osp.join(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_outlist{}_epoch{}.pt'.format(opt.uid, opt.grid, out_list[0], epoch)))
            elif len(out_list) == 2:
                torch.save(model_rn.state_dict(), osp.join(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_outlist{}_{}_epoch{}.pt'.format(opt.uid, opt.grid, out_list[0], out_list[1], epoch)))
            elif len(out_list) == 3:
                torch.save(model_rn.state_dict(), osp.join(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_outlist{}_{}_{}_epoch{}.pt'.format(opt.uid, opt.grid, out_list[0], out_list[1], out_list[2], epoch)))

        if opt.use_lrschd:
            scheduler.step()
    # print(prof.key_averages().table(sort_by="cpu_time_total"))

    writer.close()

if __name__ == '__main__':
    main()