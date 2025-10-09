import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch_geometric.loader import DataLoader
from utils.static_graph_temporal_signal import temporal_signal_split

from models.RNGCN import RNTransformer
from data_loader import TrajectoryDataset #, RoadNetwork
from utils.metrics import * 


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
parser.add_argument('--pretrained_epoch', default="0",
                help="Pretrained epoch for test", type=int)
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
parser.add_argument('--uid', type=int, help='Unique ID for the experiment', default=0)

opt = parser.parse_args()

# gpu settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


def load_data(opt):
    dataset_dir = Path(opt.dataset_dir, opt.dataset)
    global out_list

    if opt.dataset == 'inD-dataset-v1.0':
        print("Load Road Network")
        road_network = RoadNetwork(dataset_dir, opt.train_num, opt.test_num, opt.rn_num, 
                                   opt.grid, opt.agg_frame, num_timesteps_in=opt.rn_num_timesteps_in, 
                                   num_timesteps_out=opt.rn_num_timesteps_out, is_preprocessed=opt.is_rn_preprocessed)
        dataset = road_network.get_dataset() 
        _, test_rn_dataset = temporal_signal_split(dataset, train_ratio=opt.tr)
    else:
        print("Load Road Network")
        out_list = [1, 4, 8]
        road_network = [object for _ in range(len(out_list))]
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
            test_rn_loader_list.append(test_dataset)

    return test_rn_loader_list


test_rn_dataset = load_data(opt)

test_loaders = [
    DataLoader(ds, batch_size=16, shuffle=False)
    for ds in test_rn_dataset
]

num_node_features = 8

num_nodes = len(next(iter(test_rn_dataset[0])).x)
print(out_list)

if len(out_list) == 1:
    model_path = Path(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_outlist{}_epoch{}.pt'.format(opt.uid, opt.grid, out_list[0], opt.pretrained_epoch))
elif len(out_list) == 2:
    model_path = Path(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_outlist{}_{}_epoch{}.pt'.format(opt.uid, opt.grid, out_list[0], out_list[1], opt.pretrained_epoch))
elif len(out_list) == 3:
    model_path = Path(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_outlist{}_{}_{}_epoch{}.pt'.format(opt.uid, opt.grid, out_list[0], out_list[1], out_list[2], opt.pretrained_epoch))

model_rn = RNTransformer(node_features=num_node_features, num_nodes=num_nodes, periods=opt.rn_num_timesteps_in, output_dim_list=out_list, device=device).to(device)
model_rn.load_state_dict(torch.load(model_path, weights_only=True))

@torch.no_grad()
def test_rn():
    n_horizon = len(out_list)
    horizon_losses = [[] for _ in range(n_horizon)]

    model_rn.eval()

    for batches in tqdm(zip(*test_loaders)):
        batches = [b.to(device) for b in batches]

        x_list = [batch.x for batch in batches]
        y_list = [batch.y for batch in batches]
        edge_index_list = [batch.edge_index for batch in batches]
        edge_attr_list = [batch.edge_attr for batch in batches]

        predictions = model_rn(x_list, edge_index_list, edge_attr_list)

        for i in range(n_horizon):
            horizon_losses[i].append(((predictions[i] - y_list[i])**2).mean())

    results = [(torch.stack(h).mean().sqrt().item(), torch.stack(h).mean().item()) for h in horizon_losses]
    return results


def main():

    ### Model saving directory
    print("===> Save model to %s" % opt.pretrained_dir)

    print("===== Initializing model for road network =====")
    test_results = test_rn()

    loss_strings = []
    for i, (rmse, mse) in enumerate(test_results, 1):
        loss_strings.append(f"[H{i} RMSE: {rmse:.4f}, MSE: {mse:.4f}]")

    print("Road Network - Accuracy. " + ', '.join(loss_strings))


if __name__ == '__main__':
    main()