import pickle
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import mlflow
import logging

from utils.metrics import *
from models.TrajectoryModel import *
from data_loader import inDDatasetGraph, TrajectoryDataset, RoadNetworkDataset

log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

handlers = []

file_handler = logging.FileHandler(Path(log_dir, "training.log"), mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
handlers.append(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))
handlers.append(console_handler)

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=handlers,
    force=True # Override any existing configuration
)

# Quiet some noisy loggers
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Parameter Settings for Training")
# --- Input ---
# dataset options
parser.add_argument('--dataset_dir', default="./datasets/",
                help="Path to directory that contains the dataset csv files.", type=str)
parser.add_argument('--dataset', default="inD-dataset-v1.0",
                help="Name of the dataset. Needed to apply dataset specific visualization adjustments.",
                type=str)
parser.add_argument('--sdd_loc', default="",
                help="Location of SDD", type=str)
parser.add_argument('--pretrained_dir', default="./pretrained",
                help="Path to directory that contains the pretrained model.", type=str)
parser.add_argument('--recording', default="26",
                help="Name of the recording given by a number with a leading zero.", type=str)

# training options
parser.add_argument('--seed', default=42,
                help="seed number", type=int)
parser.add_argument('--epochs', default="10",
                help="Epochs for train, validation, and test suite", type=int)
parser.add_argument('--train_num', default="24",
                help="Number of training dataset the model splits", type=int)
parser.add_argument('--test_num', default="8",
                help="Number of testing dataset the model splits", type=int)
parser.add_argument('--rn_num', default="1",
                help="Number of road network dataset the model splits", type=int)
parser.add_argument('--bs', default="4",
                help="Batch size", type=int)
parser.add_argument('--optimizer', default="Adam",
                help="Name of the optimizer we use for train", type=str)
parser.add_argument('--model_name', default="social_stgcnn",
                help="Model name", type=str)
parser.add_argument('--is_pretrained', action="store_true", default=False,
                help="Use pretrained model")
parser.add_argument('--pretrained_model', default="model_0.pt",
                help="Name of pretrained model", type=str)
parser.add_argument('--num_timesteps_in', default="8",
                help="Number of timesteps for input", type=int)
parser.add_argument('--num_timesteps_out', default="12",
                help="Number of timesteps for output", type=int)
parser.add_argument('--rn_num_timesteps_in', default="8",
                help="Number of timesteps for road network input", type=int)
parser.add_argument('--rn_num_timesteps_out', default="8",
                help="Number of timesteps for road network output", type=int)
parser.add_argument('--is_horizontal_pred', default=False,
                help="If the model is trained time horizontaly, it is true", type=bool)
parser.add_argument('--use_lrschd', action="store_true", default=False,
                help='Use lr rate scheduler')
parser.add_argument('--lr_sh_rate', type=int, default=150,
                help='number of steps to drop the lr')
parser.add_argument('--clip_grad', type=float, default=None,
                help='gradient clipping')
parser.add_argument('--horizon', default="4",
                help="Number of horizon")
parser.add_argument('--agg_frame', default="20",
                help="Aggregated number of frames")
parser.add_argument('--is_rn', action="store_true", default=False,
                help="If road network is taken in the learning phase")
parser.add_argument('--grid', default="4",
                help="Number of grid on one side")
parser.add_argument('--is_preprocessed', action="store_true", default=False,
                help="If preprocessed file exists")
parser.add_argument('--is_rn_preprocessed', action="store_true", default=False,
                help="If RoadNetwork preprocessed file exists")
parser.add_argument('--load_preprocessed', action="store_true", default=False,
                help="If RoadNetwork preprocessed file exists")
parser.add_argument('--tr', '--train_ratio', default=0.8, type=float,
                help="Train ratio")
parser.add_argument('--fusion', default=None, type=str,
                help="Feature fusion method")
parser.add_argument('--temporal', default=None, type=str,
                help="Temporal model name")
parser.add_argument('--skip', default=1, type=int,
                help="Frame adjustable parameter. 12 is 2.5 FPS for SDD.")
parser.add_argument('--is_normalize', action="store_true", default=False,
                help="If you want to normalize")
parser.add_argument('--uid', default=0, type=int,
                help="Unique ID")
parser.add_argument('--rn_uid', default=0, type=int,
                help="Unique ID for Road Network model")

opt = parser.parse_args()

# gpu settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(opt.seed)

def load_data(opt):
    dataset_dir = Path(opt.dataset_dir, opt.dataset)

    # logger.info("Loading dataset")
    print("Loading dataset")

    if opt.dataset == "inD_dataset-v1.0":
        train_dataset = inDDatasetGraph(dataset_dir, num_timesteps_in=opt.num_timesteps_in, num_timesteps_out=opt.num_timesteps_out, train_num=opt.train_num, test_num=opt.test_num, is_train=True, skip=10, is_preprocessed=opt.is_preprocessed)
        test_dataset = inDDatasetGraph(dataset_dir, num_timesteps_in=opt.num_timesteps_in, num_timesteps_out=opt.num_timesteps_out, train_num=opt.train_num, test_num=opt.test_num, is_train=False, skip=10, is_preprocessed=opt.is_preprocessed)
        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    else:
        if opt.is_rn:
            out_list = [1, 4, 8]
        else:
            out_list = [opt.num_timesteps_out]

        train_dataset = TrajectoryDataset(
                dataset_dir,
                sdd_loc=opt.sdd_loc,
                in_channels=opt.num_timesteps_in,
                out_channels=opt.num_timesteps_out,
                agg_frame=opt.agg_frame,
                skip=opt.skip,
                norm_lap_matr=True,
                grid=opt.grid,
                is_preprocessed=opt.is_preprocessed,
                dataset=opt.dataset,
                train_mode='train',
                is_rn=opt.is_rn,
                is_normalize=opt.is_normalize)

        val_dataset = TrajectoryDataset(
                dataset_dir,
                sdd_loc=opt.sdd_loc,
                in_channels=opt.num_timesteps_in,
                out_channels=opt.num_timesteps_out,
                agg_frame=opt.agg_frame,
                skip=opt.skip,
                norm_lap_matr=True,
                grid=opt.grid,
                is_preprocessed=opt.is_preprocessed,
                dataset=opt.dataset,
                train_mode='val',
                is_rn=opt.is_rn,
                is_normalize=opt.is_normalize)

        train_loader = DataLoader(
                train_dataset,
                batch_size=1, # Peds tensor are always different
                shuffle=True,
                num_workers=0)

        val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1)

        train_rn_loader_list, test_rn_loader_list = [], []
        for i in range(len(out_list)):
            train_rn_dataset = RoadNetworkDataset(
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

            test_rn_dataset = RoadNetworkDataset(
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


            if opt.is_rn:
                # train_rn_dataset = train_dataset.get(opt.rn_num_timesteps_in, out_list[i])
                # val_rn_dataset = val_dataset.get(opt.rn_num_timesteps_in, out_list[i])

                train_rn_loader_list.append(train_rn_dataset)
                test_rn_loader_list.append(test_rn_dataset)

    if opt.is_rn:
        return train_loader, val_loader, train_rn_loader_list, test_rn_loader_list
    else:
        return train_loader, val_loader

num_nodes = 0

out_list = [1, 4, 8]

metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}

if opt.is_rn:
    train_loader, val_loader, train_rn_dataset, val_rn_dataset = load_data(opt)

    train_rn_loaders = [
        DataLoader(ds, batch_size=1, shuffle=True)
        for ds in train_rn_dataset
    ]

    val_rn_loaders = [
        DataLoader(ds, batch_size=1, shuffle=False)
        for ds in val_rn_dataset
    ]
else:
    train_loader, val_loader = load_data(opt)
print(len(train_loader))
print(len(val_loader))

### Model saving directory
print("===> Save model to %s" % opt.pretrained_dir)

os.makedirs(opt.pretrained_dir, exist_ok=True)

if opt.is_rn:
    print("===== Initializing model for Road Network =====")
    num_nodes = len(next(iter(train_rn_dataset[0])).x)
    print(f'Number of Nodes: {num_nodes}')

if opt.is_rn:
    out_string = ''
    for out in out_list:
        out_string += str(out) + '_'
    rn_checkpoint_path = Path(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_outlist{}epoch{}.pt'.format(opt.rn_uid, opt.grid, out_string, 10))

    rn_config = {
        'node_features': 8,
        'num_nodes': len(next(iter(train_rn_dataset[0])).x),
        'periods': opt.num_timesteps_in,
        'output_dim_list': [1, 4, 8],
        'device': device
    }
    
    model_rn = RNTransformer(**rn_config).to(device)
    model_rn.load_state_dict(torch.load(rn_checkpoint_path, weights_only=True))

    ### Freeze the parameters
    for param in model_rn.parameters():
        param.requires_grad = False
else:
    model_rn = None

### Model setup
print("===== Initializing model for trajectory prediction =====")
model = TrajectoryModel(opt, num_nodes=num_nodes, out_list=out_list, device=device).to(device)

### Optimization
if opt.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
elif opt.optimizer == "SGD":
    if opt.model_name == "trajectory_model":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    elif opt.model_name == "social_stgcnn":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)
    elif opt.model_name == "social_implicit":
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
elif opt.optimizer == "RMSProps":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-3)
elif opt.optimizer == "AdaGrad":
    optimizer = torch.optim.Adagrad(model.parameters(), weight_decay=5e-4)

if opt.is_rn:
    optimizer_rn = torch.optim.RMSprop(model.model_rn.parameters(), lr=1e-3, weight_decay=1e-3)

if opt.use_lrschd:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_sh_rate, gamma=0.2)

total_param = 0
for param_tensor in model.state_dict():
    # logger.info("{}, {}".format(param_tensor, model.state_dict()[param_tensor].size()))
#     print("{}, {}".format(param_tensor, model.state_dict()[param_tensor].size()))
    total_param += np.prod(model.state_dict()[param_tensor].size())
#     print(model.state_dict()[param_tensor].size())
# logger.info('Net\'s total params: {}' % total_param)
print('Net\'s total param-s:', total_param)

# Global feature storage (automatically filled by hooks)
feature_maps = {}
def save_feature(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach().cpu()
    return hook

def graph_loss(V_pred, V_target):
    if opt.model_name == 'social_stgcnn' or opt.model_name == 'social_lstm':
        return bivariate_loss(V_pred, V_target)
    elif opt.model_name == 'social_implicit':
        return implicit_likelihood_estimation_fast_with_trip_geo(V_pred, V_target)

def direction_loss(traj_pred, rn_fused):
    # traj_pred: [B,2,12] but actually missing P dimension
    # rn_fused:  [B*P,2,8]

    B = traj_pred.size(0)
    P = rn_fused.size(0) // B

    # --- reshape traj_pred into per-pedestrian form ---
    traj_pred = traj_pred.reshape(B, P, 2, 12)
    traj_pred = traj_pred.permute(0,1,3,2)   # [B,P,T=12,C=2]

    # final displacement direction
    final_disp = traj_pred[:,:, -1, :] - traj_pred[:,:, -2, :]
    final_dir = F.normalize(final_disp, dim=-1)   # [B,P,2]

    # --- reshape rn_fused the same way ---
    rn_fused = rn_fused.reshape(B, P, 2, 8)          # [B,P,2,8]
    rn_vec = rn_fused.mean(dim=-1)                # reduce 8 → [B,P,2]
    rn_dir = F.normalize(rn_vec, dim=-1)

    # cosine distance
    cos_sim = (final_dir * rn_dir).sum(dim=-1)    # [B,P]
    loss = (1 - cos_sim).mean()

    return loss

def mse_loss(V_pred, V_target):
    return torch.mean((V_pred - V_target)**2)

def huber_loss(rn_pred, rn_gt):
    h_loss = torch.nn.HuberLoss('mean', delta=0.1)
    return h_loss(rn_pred, rn_gt)

@torch.no_grad()
def precompute_rn_features(model_rn, rn_loaders, mode="train"):
    """Pre-compute RN embeddings for all trajectory data"""
    feature_cache = {}
    logger.info("Pre-computing RN features for all trajectory data")
    
    if opt.load_preprocessed:
        with open(Path(opt.pretrained_dir, 'road_network', opt.dataset, f'feature_cache_{mode}.pkl'), 'rb') as f:
            feature_cache = pickle.load(f)
    else:
        # Create a lightweight dataloader just for RN computation
        for idx, batches in tqdm(enumerate(zip(*rn_loaders)), desc="Precomputing RN features"):
            x_list = [batch.x.to(device) for batch in batches]
            edge_attr_list = [batch.edge_attr.to(device) for batch in batches]
            edge_index_list = [batch.edge_index.to(device) for batch in batches]

            # x_list, edge_index_list, edge_attr_list = batch.x, batch.edge_index, batch.edge_attr
            rn_pred, rn_embed = model_rn(x_list, edge_index_list, edge_attr_list)
            # logger.info(rn_pred)
            # feature_cache[idx] = (rn_pred, rn_embed.detach().cpu())
            feature_cache[idx] = (
                rn_pred,
                rn_embed.to(device, non_blocking=True)
            )

        with open(Path(opt.pretrained_dir, 'road_network', opt.dataset, f'feature_cache_{mode}.pkl'), 'wb') as f:
            pickle.dump(feature_cache, f)

    return feature_cache

def train_peds(epoch, train_loader, rn_feature_cache=None):
    """
    Train function for pedestrian trajectory prediction graph network
    h: features from road network
    """

    model.train()

    loss_batch = 0 
    loss = 0
    traj_pred = None

    for idx, batch in enumerate(tqdm(train_loader)): 
        loc_data = []

        ### Train Local Peds Trajectory
        for tensor in batch:
            if not isinstance(tensor, list):
                loc_data.append(tensor.to(device))
            else:
                loc_data.append(tensor)

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask, V_obs, A_obs, V_tr, A_tr, _, _, ped_list = loc_data

        # Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        if opt.is_rn:
            precomputed_rn = rn_feature_cache[idx]
            _, traj_pred, rn_fused = model(V_obs_tmp, A_obs, ped_list=ped_list, precomputed_rn=precomputed_rn)

        else:
            _, traj_pred = model(V_obs_tmp, A_obs, ped_list=ped_list)
            traj_pred = traj_pred
        
        if opt.model_name != "social_lstm":
            traj_pred = traj_pred.permute(0, 2, 3, 1)

        traj_loss = graph_loss(traj_pred, V_tr)
        loc_loss = 0
        if opt.is_rn:
            # lambda_l1 = 1e-5
            # lambda_l2 = 1e-5

            # l1_loss = sum(model.compute_l1_loss(param) for param in model.parameters())
            # l2_loss = sum(model.compute_l2_loss(param) for param in model.parameters())
            # final direction (simple, fast)
            lambda_dir = 0.005

            dir_loss = direction_loss(traj_pred, rn_fused)
            loss += traj_loss #+ lambda_dir * dir_loss #+ lambda_l1 * l1_loss + lambda_l2 * l2_loss #+ mse_loss(traj_pred[..., :2], V_tr)
        else:
            loss += traj_loss

        if (idx+1) % opt.bs == 0 and (idx+1) != 0:
            l = loss / opt.bs
            l.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            loss_batch += l.item()
            loss = 0

        # logger.info('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / idx, '\t Traj Loss:', traj_loss.item(), '\t L1 Loss:', l1_loss.item(), '\t L2 Loss:', l2_loss.item())
    logger.info("Epoch: {}, Valid Loss - Total Loss: {:4f}, Trajectory Loss: {:.4f}, Direction Loss: {:.4f}".format(epoch, loss_batch / idx, traj_loss, dir_loss))
    return traj_loss, loc_loss, loss


@torch.no_grad()
def valid_peds(epoch, val_loader, metrics=None, constant_metrics=None, rn_feature_cache=None):
    model.eval()
    batch_count = 0
    traj_pred = None
    loss = 0
    loss_batch = 0

    for idx, batch in enumerate(tqdm(val_loader)): 
        loc_data = []

        batch_count += 1

        ### Train Local Peds Trajectory
        for tensor in batch:
            if not isinstance(tensor, list):
                loc_data.append(tensor.to(device))
            else:
                loc_data.append(tensor)

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask, V_obs, A_obs, V_tr, A_tr, _, _, ped_list = loc_data

        # Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        if (V_obs_tmp.size(-1) == 0):
            break
        if opt.is_rn:
            if idx in rn_feature_cache:
                precomputed_rn = rn_feature_cache[idx]
                _, traj_pred, rn_fused = model(V_obs_tmp, A_obs, ped_list=ped_list, precomputed_rn=precomputed_rn)
        else:
            _, traj_pred = model(V_obs_tmp, A_obs, ped_list=ped_list)
            traj_pred = traj_pred + 0.001 * torch.randn_like(traj_pred)

        if opt.model_name != "social_lstm":
            traj_pred = traj_pred.permute(0, 2, 3, 1)

        traj_loss = graph_loss(traj_pred, V_tr)
        loc_loss = 0
        if opt.is_rn:
            # lambda_l1 = 1e-5
            # lambda_l2 = 1e-5
            # l1_loss = sum(model.compute_l1_loss(param) for param in model.parameters())
            # l2_loss = sum(model.compute_l2_loss(param) for param in model.parameters())
            lambda_dir = 0.005

            dir_loss = direction_loss(traj_pred, rn_fused)
            # loc_loss = graph_loss(out_loc, V_tr)
            loss += traj_loss #+ lambda_dir * dir_loss #+ lambda_l1 * l1_loss + lambda_l2 * l2_loss #+ mse_loss(traj_pred[..., :2], V_tr)
        else:
            loss += traj_loss

        if (idx+1) % opt.bs == 0 and (idx+1) != 0:
            loss = loss / opt.bs
            loss_batch += loss.item()

            # logger.info('VALD: Epoch: {}, Loss: {}'.format(epoch, loss_batch / batch_count))
            metrics['val_loss'].append(loss)
            loss = 0

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] =  metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        if opt.is_rn:
            torch.save(model.state_dict(), Path(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_val_rn_best.pt'.format(opt.uid)))
        else:
            torch.save(model.state_dict(), Path(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_val_best.pt'.format(opt.uid)))

    logger.info("Epoch: {}, Valid Loss - Total Loss: {:4f}, Trajectory Loss: {:.4f}, Direction Loss: {:.4f}".format(epoch, loss, loss_batch / batch_count, dir_loss))
    return traj_loss, dir_loss, loss


def main():
    # Pre-compute RN features for the entire dataset
    if opt.is_rn:
        train_rn_feature_cache = precompute_rn_features(model_rn, train_rn_loaders, mode="train")
        val_rn_feature_cache = precompute_rn_features(model_rn, val_rn_loaders, mode="val")
    else:
        train_rn_feature_cache = None
        val_rn_feature_cache = None

    for epoch in range(opt.epochs + 1):

        if opt.is_rn:
            traj_loss, dir_loss, train_loss = train_peds(epoch, train_loader, rn_feature_cache=train_rn_feature_cache)
            traj_val_loss, dir_val_loss, val_loss = valid_peds(epoch, val_loader, metrics=metrics, constant_metrics=constant_metrics, rn_feature_cache=val_rn_feature_cache)
        else:
            traj_loss, dir_loss, train_loss = train_peds(epoch, train_loader)
            traj_val_loss, dir_val_loss, val_loss = valid_peds(epoch, val_loader, metrics=metrics, constant_metrics=constant_metrics)

        torch.save(feature_maps, "{}_feature_maps_epoch{}.pt".format(opt.uid, epoch))

        mlflow.log_metric("Train Loss/Direction Loss", dir_loss, step=epoch)
        mlflow.log_metric("Train Loss/Trajectory Network", traj_loss, step=epoch)
        mlflow.log_metric("Train Loss/Whole Network", train_loss, step=epoch)

        mlflow.log_metric("Val Loss/Direction Loss", dir_val_loss, step=epoch)
        mlflow.log_metric("Val Loss/Trajectory Network", traj_val_loss, step=epoch)
        mlflow.log_metric("Val Loss/Whole Network", val_loss, step=epoch)

        logger.info('Epoch {}'.format(epoch))

        logger.info("Train Loss/Local Network: {}".format(traj_loss))
        logger.info("Train Loss/Whole Network: {}".format(train_loss))

        logger.info("Val Loss/Local Network: {}".format(traj_val_loss))
        logger.info("Val Loss/Whole Network: {}".format(val_loss))

        # Save the model
        if epoch % 5 == 0:
            if opt.is_rn:
                torch.save(model.state_dict(), Path(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_model_grid{}_epoch{}.pt'.format(opt.uid, opt.grid, epoch)))
            else:
                torch.save(model.state_dict(), Path(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_model_{}.pt'.format(opt.uid, epoch)))
            with open(Path(opt.pretrained_dir, opt.model_name, opt.dataset, 'constant_metrics.pkl'), 'wb') as fp:
                pickle.dump(constant_metrics, fp)
            with open(Path(opt.pretrained_dir, opt.model_name, opt.dataset, 'args.pkl'), 'wb') as fp:
                pickle.dump(opt, fp)


        if opt.use_lrschd:
            scheduler.step()

if __name__ == '__main__':
    main()