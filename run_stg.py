import pickle
import os
from os import path as osp
import argparse
import datetime
import pytz
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging


from utils.metrics import *
from models.TrajectoryModel import *
from models.LocalPedsTrajNet import *
from models.SocialStgcnn import social_stgcnn
from data_loader import inDDatasetGraph, TrajectoryDataset

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

opt = parser.parse_args()

# gpu settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(opt.seed)

def load_data(opt):
    dataset_dir = osp.join(opt.dataset_dir, opt.dataset)

    print("Loading dataset")

    if opt.dataset == "inD_dataset-v1.0":
        train_dataset = inDDatasetGraph(dataset_dir, num_timesteps_in=opt.num_timesteps_in, num_timesteps_out=opt.num_timesteps_out, train_num=opt.train_num, test_num=opt.test_num, is_train=True, skip=10, is_preprocessed=opt.is_preprocessed)
        test_dataset = inDDatasetGraph(dataset_dir, num_timesteps_in=opt.num_timesteps_in, num_timesteps_out=opt.num_timesteps_out, train_num=opt.train_num, test_num=opt.test_num, is_train=False, skip=10, is_preprocessed=opt.is_preprocessed)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    else:
        if opt.is_rn:
            out_list = [1, 4, 8]
            # out_list = [4, 8, 16]
            # out_list = [12]
        else:
            out_list = [opt.num_timesteps_out]
        for i in range(len(out_list)):
            train_dataset = TrajectoryDataset(
                    dataset_dir,
                    sdd_loc=opt.sdd_loc,
                    in_channels=opt.num_timesteps_in,
                    out_channels=opt.num_timesteps_out,
                    rn_out_channels=out_list[i],
                    agg_frame=opt.agg_frame,
                    skip=opt.skip,
                    norm_lap_matr=True,
                    grid=opt.grid,
                    is_preprocessed=opt.is_preprocessed,
                    dataset_iter=i,
                    dataset=opt.dataset,
                    train_mode='train',
                    is_rn=opt.is_rn,
                    is_normalize=opt.is_normalize)

            val_dataset = TrajectoryDataset(
                    dataset_dir,
                    sdd_loc=opt.sdd_loc,
                    in_channels=opt.num_timesteps_in,
                    out_channels=opt.num_timesteps_out,
                    rn_out_channels=out_list[i],
                    agg_frame=opt.agg_frame,
                    skip=opt.skip,
                    norm_lap_matr=True,
                    grid=opt.grid,
                    is_preprocessed=opt.is_preprocessed,
                    dataset_iter=i,
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

    return train_loader, val_loader


def graph_loss(V_pred, V_target, num_of_objs=None):
    return bivariate_loss(V_pred, V_target)

def huber_loss(rn_pred, rn_gt):
    h_loss = torch.nn.HuberLoss('mean', delta=0.1)
    return h_loss(rn_pred, rn_gt)


def train_peds(epoch, train_loader):
    """
    Train function for pedestrian trajectory prediction graph network
    h: features from road network
    """

    model.train()

    loss_batch = 0 
    batch_count = 0
    temp_batch_count = 0
    is_fst_loss = True
    loss = 0
    loader_len = len(train_loader)
    rn_loss = 0
    turn_point = int(loader_len / opt.bs) * opt.bs + loader_len % opt.bs - 1
    traj_pred = None

    pbar = tqdm(enumerate(train_loader))

    for i, batch in pbar: 
        pbar.update(1)
        batch_count += 1
        rn_loss = 0
        loc_data = []
        optimizer.zero_grad()

        ### Train Local Peds Trajectory
        for tensor in batch:
            if not isinstance(tensor, list):
                loc_data.append(tensor.to(device))
            else:
                loc_data.append(tensor)

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask, V_obs, A_obs, V_tr, A_tr, _, _ = loc_data

        # Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        traj_pred, _, _ = model(V_obs_tmp, A_obs.squeeze())
        traj_pred = traj_pred.permute(0, 2, 3, 1)

        if batch_count % opt.bs != 0 and i != turn_point:
            temp_batch_count += 1
            traj_loss = graph_loss(traj_pred, V_tr, traj_pred.size(2))

            l = traj_loss 

            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l
        else:
            loss = loss / opt.bs
            is_fst_loss = True

            loss.backward()

            if opt.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad)

            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    print("Epoch: {}, Train Loss - Total Loss: {:4f}, Road Loss: {:.4f}, Local Trajectory Loss: {:.4f}".format(epoch, loss_batch / batch_count, rn_loss, traj_loss))
    
    return traj_loss, loss



@torch.no_grad()
def valid_peds(epoch, val_loader, metrics={}, constant_metrics={}):
    model.eval()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(val_loader)
    turn_point =int(loader_len / opt.bs) * opt.bs + loader_len % opt.bs - 1
    rn_loss = 0
    traj_pred = None
    tot_traj_loss = 0
    tot_rn_loss = 0
    # out_list = [4, 8, 16]

    pbar = tqdm(enumerate(val_loader))

    for i, batch in pbar: 
        pbar.update(1)
        rn_loss = 0
        loc_data = []
        # pbar.set_description("Processing Validation %s" % i)

        batch_count += 1

        ### Train Local Peds Trajectory
        # Get data
        # loc_data = [tensor.to(device) for tensor in batch]
        for tensor in batch:
            if not isinstance(tensor, list):
                loc_data.append(tensor.to(device))
            else:
                loc_data.append(tensor)

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask, V_obs, A_obs, V_tr, A_tr, _, _ = loc_data

        # Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        if (V_obs_tmp.size(-1) == 0):
            break
        traj_pred, _, _ = model(V_obs_tmp, A_obs.squeeze())
        traj_pred = traj_pred.permute(0, 2, 3, 1)

        if batch_count % opt.bs != 0 and i != turn_point :
            traj_loss = graph_loss(traj_pred, V_tr, traj_pred.size(2))
            if is_fst_loss :
                loss = traj_loss 
                is_fst_loss = False
            else:
                loss += traj_loss
            tot_traj_loss += traj_loss
            tot_rn_loss += rn_loss
        else:
            loss = loss / opt.bs
            is_fst_loss = True
            # Metrics
            loss_batch += loss.item()
            print('VALD: Epoch: {}, Loss: {}'.format(epoch, loss_batch / batch_count))


    metrics['val_loss'].append(loss_batch / batch_count)

    
    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] =  metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_val_best.pt'.format(opt.uid)))

    print("Epoch: {}, Valid Loss - Total Loss: {:4f}, Local Trajectory Loss: {:.4f}".format(epoch, loss / batch_count, traj_loss))
    return traj_loss, loss
      

def main():
    num_nodes = 0

    metrics = {'train_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}

    train_loader, val_loader = load_data(opt)


    ### Model saving directory
    print("===> Save model to %s" % opt.pretrained_dir)

    os.makedirs(opt.pretrained_dir, exist_ok=True)

    global model, optimizer

    ### Model setup
    print("===== Initializing model for trajectory prediction =====")
    model = social_stgcnn(1, 5, output_feat=5, seq_len=opt.num_timesteps_in, pred_seq_len=opt.num_timesteps_out, num_nodes=num_nodes).to(device)

    # Check if pretrained model exists
    if opt.is_pretrained:
        model.load_state_dict(torch.load(osp.join(opt.pretrained_dir, opt.model_name, opt.pretrained_model)))

    ### Optimization
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)

    total_param = 0
    for param_tensor in model.state_dict():
    #     print("{}, {}".format(param_tensor, model.state_dict()[param_tensor].size()))
        total_param += np.prod(model.state_dict()[param_tensor].size())
    #     print(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    # for epoch in tqdm(range(opt.epochs + 1)):
    for epoch in range(opt.epochs + 1):

        traj_loss, train_loss = train_peds(epoch=epoch, train_loader=train_loader)
        traj_val_loss, val_loss = valid_peds(epoch, val_loader, metrics=metrics, constant_metrics=constant_metrics)

        # Save the model
        if epoch % 5 == 0:
            torch.save(model.state_dict(), osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_model_{}.pt'.format(opt.uid, epoch)))
            with open(osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, 'constant_metrics.pkl'), 'wb') as fp:
                pickle.dump(constant_metrics, fp)


if __name__ == '__main__':
    main()