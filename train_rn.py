import os
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
import random
import numpy as np

import torch
from torch_geometric.loader import DataLoader

import mlflow

from utils.metrics import *
from models.RNGCN import RNTransformer
from data_loader import RoadNetworkDataset

log_path = './logs'
os.makedirs(log_path, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    dataset_dir = Path(opt.dataset_dir, opt.dataset)
    global out_list

    print("Load Road Network")
    out_list = [1, 4, 8]
    train_rn_loader_list, test_rn_loader_list = [], []
    for i in range(len(out_list)):
        train_dataset = RoadNetworkDataset(
                dataset_dir,
                sdd_loc=opt.sdd_loc,
                in_channels=opt.rn_num_timesteps_in,
                out_channels=opt.rn_num_timesteps_out,
                rn_out_channels=out_list[i],
                agg_frame=opt.agg_frame,
                skip=opt.skip,
                grid=opt.grid,
                is_preprocessed=opt.is_rn_preprocessed,
                dataset=opt.dataset,
                train_mode='train')

        test_dataset = RoadNetworkDataset(
                dataset_dir,
                sdd_loc=opt.sdd_loc,
                in_channels=opt.rn_num_timesteps_in,
                out_channels=opt.rn_num_timesteps_out,
                rn_out_channels=out_list[i],
                agg_frame=opt.agg_frame,
                skip=opt.skip,
                grid=opt.grid,
                is_preprocessed=opt.is_rn_preprocessed,
                dataset=opt.dataset,
                train_mode='val')
        train_rn_loader_list.append(train_dataset)
        test_rn_loader_list.append(test_dataset)

    return train_rn_loader_list, test_rn_loader_list


train_rn_dataset, test_rn_dataset = load_data(opt)

train_loaders = [
    DataLoader(ds, batch_size=16, shuffle=True)
    for ds in train_rn_dataset
]

test_loaders = [
    DataLoader(ds, batch_size=16, shuffle=False)
    for ds in test_rn_dataset
]

### Model saving directory
print("===> Save model to %s" % opt.pretrained_dir)

os.makedirs(opt.pretrained_dir, exist_ok=True)

print("===== Initializing model for road network =====")

num_nodes = len(next(iter(train_rn_dataset[0])).x)
print(out_list)

model_rn = RNTransformer(node_features=8, num_nodes=num_nodes, periods=opt.rn_num_timesteps_in, output_dim_list=out_list, device=device).to(device)

# Training settings for road network
# optimizer_rn = torch.optim.SGD(model_rn.parameters(), lr=1e-2, weight_decay=1e-3)
optimizer_rn = torch.optim.Adam(model_rn.parameters(), lr=1e-3, weight_decay=1e-4)

if opt.use_lrschd:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_rn, mode='min', factor=0.5, patience=3, verbose=True
    )

total_param = 0
for param_tensor in model_rn.state_dict():
    print(param_tensor, '\t', model_rn.state_dict()[param_tensor].size())
    total_param += np.prod(model_rn.state_dict()[param_tensor].size())
print('Net\'s total params:', total_param)

def multi_horizon_loss(predictions, targets, delta=0.1):
    h_loss = torch.nn.HuberLoss('mean', delta=delta)
    losses = []
    for pred, tgt in zip(predictions, targets):
        losses.append(h_loss(pred, tgt))
    return sum(losses)

def train_rn(train_loader):

    total_loss = 0
    step = 0
    for batches in tqdm(zip(*train_loader)):
        batch_loss = 0
        batches = [b.to(device) for b in batches]

        x_list = [batch.x for batch in batches]
        y_list = [batch.y for batch in batches]
        edge_index_list = [batch.edge_index for batch in batches]
        edge_attr_list = [batch.edge_attr for batch in batches]

        optimizer_rn.zero_grad()
        predictions = model_rn(x_list, edge_index_list, edge_attr_list)

        batch_loss = multi_horizon_loss(predictions, y_list, delta=0.1)
        batch_loss.backward()
        optimizer_rn.step()

        total_loss += batch_loss.item()
        step += 1

    return total_loss / max(step, 1)

@torch.no_grad()
def test_rn(test_loader):
    model_rn.eval()

    n_horizon = len(out_list)
    horizon_losses = [[] for _ in range(n_horizon)]

    for batches in tqdm(zip(*test_loader)):
        batches = [b.to(device) for b in batches]

        x_list = [batch.x for batch in batches]
        y_list = [batch.y for batch in batches]
        edge_index_list = [batch.edge_index for batch in batches]
        edge_attr_list = [batch.edge_attr for batch in batches]

        predictions = model_rn(x_list, edge_index_list, edge_attr_list)

        for i in range(n_horizon):
            mse = ((predictions[i] - y_list[i]) ** 2).mean()
            horizon_losses[i].append(mse)

    results = [(torch.stack(h).mean().sqrt().item(), torch.stack(h).mean().item()) for h in horizon_losses]

    return results


def main():

    print("===== Start training =====")

    for epoch in tqdm(range(opt.epochs + 1)):

        train_loss = train_rn(train_loaders)

        test_results = test_rn(test_loaders)

        loss_strings = []
        for i, (rmse, mse) in enumerate(test_results, 1):
            loss_strings.append(f"[H{i} RMSE: {rmse:.4f}, MSE: {mse:.4f}]")
            mlflow.log_metric(f"test_rmse/horizontal {i}", rmse, step=epoch)
            mlflow.log_metric(f"test_mse/horizontal {i}", mse, step=epoch)

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, " + " ".join(loss_strings))

        mlflow.log_metric("train_loss", train_loss, step=epoch)

        if epoch % 10 == 0:
            if len(out_list) == 1:
                torch.save(model_rn.state_dict(), Path(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_outlist{}_epoch{}.pt'.format(opt.uid, opt.grid, out_list[0], epoch)))
            elif len(out_list) == 2:
                torch.save(model_rn.state_dict(), Path(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_outlist{}_{}_epoch{}.pt'.format(opt.uid, opt.grid, out_list[0], out_list[1], epoch)))
            elif len(out_list) == 3:
                torch.save(model_rn.state_dict(), Path(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_outlist{}_{}_{}_epoch{}.pt'.format(opt.uid, opt.grid, out_list[0], out_list[1], out_list[2], epoch)))
        if opt.use_lrschd:
            scheduler.step()

if __name__ == '__main__':
    main()