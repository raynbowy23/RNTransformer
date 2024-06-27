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

from torch.utils.tensorboard import SummaryWriter

from utils.metrics import *
from models.TrajectoryModel import *
from models.LocalPedsTrajNet import *
from data_loader import inDDatasetGraph, TrajectoryDataset

# Writer will output to ./runs/ directory by default
writer_flg = False
if writer_flg:
    writer = SummaryWriter()

# log_path = './logs'
# os.makedirs(log_path, exist_ok=True)

# # Log
# logging.basicConfig(filename=osp.join(log_path, datetime.datetime.now().astimezone(pytz.timezone('US/Central')).strftime('%Y-%m-%d_%H-%M') + ".log"),
#                                        format='%(asctime)s %(levelname)-8s %(message)s',
#                                        filemode='a')
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

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
parser.add_argument('--pretrained_epoch', default=0,
                help="The number of pretrained epoch", type=int)
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
                help="If t            self.local_mlp_head1 = torch.nn.Linear(self.rn_out_dim * 5, self.rn_out_dim * 2) # Cross-attentionhe model is trained time horizontaly, it is true", type=bool)
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
                help="Feat            self.local_mlp_head1 = torch.nn.Linear(self.rn_out_dim * 5, self.rn_out_dim * 2) # Cross-attentionure fusion method")
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

    # logger.info("Loading dataset")
    print("Loading dataset")

    if opt.is_rn:
        out_list = [1, 4, 8]
        # out_list = [4, 8, 16]
        # out_list = [12]
    else:
        out_list = [opt.num_timesteps_out]
    val_rn_loader_list = []
    for i in range(len(out_list)):

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
                train_mode='train',
                is_rn=opt.is_rn,               
                is_normalize=opt.is_normalize)

        if opt.is_rn:
            val_rn_dataset = val_dataset.get(opt.rn_num_timesteps_in, out_list[i])
            val_rn_loader_list.append(val_rn_dataset)

        if i == 0:
            val_loader = DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=1)

    if opt.is_rn:
        return val_loader, val_rn_loader_list
    else:
        return val_loader


def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)
    # return multivariate_loss(V_pred, V_target)

def mse_loss(V_pred, V_target):
    return torch.mean((V_pred - V_target)**2)


def huber_loss(rn_pred, rn_gt):
    h_loss = torch.nn.HuberLoss('mean', delta=0.1)
    return h_loss(rn_pred, rn_gt)


def valid_peds(epoch, val_loader, val_rn_dataset=None):
    model.train()

    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(val_loader)
    turn_point =int(loader_len / opt.bs) * opt.bs + loader_len % opt.bs - 1
    rn_loss = 0
    prev_rn_loss = 0
    traj_pred = None
    weight_rn_init = 1
    weight_rn_final = 0.01


    if opt.is_rn:
        if len(val_rn_dataset) == 1:
            pbar = tqdm(enumerate(zip(val_loader, val_rn_dataset[0])))
        else:
            pbar = tqdm(enumerate(zip(val_loader, val_rn_dataset[0], val_rn_dataset[1], val_rn_dataset[2])))
    else:
        pbar = tqdm(enumerate(val_loader))

    for i, batch in pbar: 
        pbar.update(1)

        # weight_rn = weight_rn_init - (weight_rn_init - weight_rn_final) * (epoch / opt.epochs)
        weight_rn = 1
        rn_loss = 0
        loc_data = []

        optimizer.zero_grad()
        optimizer_rn.zero_grad()

        batch_count += 1

        ### Train Local Peds Trajectory
        # Get data
        if opt.is_rn:
            for tensor in batch[0]:
                if not isinstance(tensor, list):
                    loc_data.append(tensor.to(device))
                else:
                    loc_data.append(tensor.to(device))
            x = [batch[i].x.to(device) for i in range(1, len(batch))]
            y = [batch[i].y.to(device) for i in range(1, len(batch))]
        else:
            for tensor in batch:
                if not isinstance(tensor, list):
                    loc_data.append(tensor.to(device))
                else:
                    loc_data.append(tensor.to(device))

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask, V_obs, A_obs, V_tr, A_tr, _, _ = loc_data


        # Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

            
        if opt.is_rn:

            rn_edge_index = [batch[i].edge_index.to(device) for i in range(1, len(batch))]
            rn_edge_attr = [batch[i].edge_attr.to(device) for i in range(1, len(batch))]

            rn_pred, out_loc, traj_pred = model(V_obs_tmp, A_obs, x, rn_edge_index, rn_edge_attr, i, h_=traj_pred)
            # rn_pred, out_loc, traj_pred = model(V_obs_tmp, A_obs, x, rn_edge_index, rn_edge_attr, i)
            out_loc = out_loc.permute(0, 2, 3, 1)
            
            for j in range(len(rn_pred)):
                # rn_loss += torch.mean((rn_pred[j] - y[j])**2).cpu()
                rn_loss += huber_loss(rn_pred[j], y[j]).cpu()
        else:
            rn_pred, traj_pred, _ = model(V_obs_tmp, A_obs.squeeze())
            traj_pred = traj_pred.permute(0, 2, 3, 1)


        if batch_count % opt.bs != 0 and i != turn_point:
            traj_loss = graph_loss(traj_pred, V_tr)
            if opt.is_rn:
                # loc_loss = graph_loss(out_loc, V_tr)
                loc_loss = 0
                l = loc_loss + traj_loss + weight_rn * rn_loss + mse_loss(traj_pred[..., :2], V_tr)
            else:
                loc_loss = 0
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.clip_grad)

            optimizer.step()
            optimizer_rn.step()
            # Metrics
            loss_batch += loss.item()
            # print('VALD: Epoch: {}, Loss: {}'.format(epoch, loss_batch / batch_count))


        # if prev_rn_loss - rn_loss < 0.0001:
        #     for param in model.model_rn.parameters():
        #         param.requires_grad = False

        # prev_rn_loss = rn_loss


    print("Epoch: {}, Valid Loss - Total Loss: {:4f}, Road Loss: {:.4f}, Trajectory Loss: {:.4f}, Local Loss: {:.4f}".format(epoch, loss_batch / batch_count, rn_loss, traj_loss, loc_loss))
    return rn_loss, traj_loss, loc_loss, loss
      

def main():
    num_nodes = 0
    out_list = [1, 4, 8]

    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}

    if opt.is_rn:
        val_loader, val_rn_loader = load_data(opt)
    else:
        val_loader = load_data(opt)
    print(len(val_loader))

    ### Model saving directory
    print("===> Save model to %s" % opt.pretrained_dir)

    os.makedirs(opt.pretrained_dir, exist_ok=True)

    global model, model_rn, optimizer, optimizer_rn
    if opt.is_rn:
        print("===== Initializing model for Road Network =====")
        num_nodes = len(next(iter(val_rn_loader[0])).x)
        print(f'Number of Nodes: {num_nodes}')

    ### Model setup
    print("===== Initializing model for trajectory prediction =====")
    if opt.model_name == "trajectory_model" or opt.model_name == "social_stgcnn" or opt.model_name == "social_implicit":
        model_rn = None
        # model_rn = RNTransformer(node_features=7, num_nodes=num_nodes, periods=opt.num_timesteps_in, output_dim_list=out_list, device=device).to(device)
        # ### Pretrained
        # model_path = osp.join(opt.pretrained_dir, 'road_network', 'model_grid{}_outlist{}_epoch{}.pt'.format(opt.grid, out_list[0], 10))
        # model_path = osp.join(opt.pretrained_dir, 'road_network', opt.dataset, '{}_model_grid{}_out_list{}_{}_{}_epoch{}.pt'.format(32, opt.grid, out_list[0], out_list[1], out_list[2], 50))
        # model_rn.load_state_dict(torch.load(model_path))
        model_loc = None
        # model_loc = social_stgcnn(1, 5, output_feat=5, seq_len=opt.num_timesteps_in, pred_seq_len=opt.num_timesteps_out, num_nodes=num_nodes).to(device)
        # model_loc.load_state_dict(torch.load(osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_model_{}.pt'.format(200, 100))))

        model = TrajectoryModel(in_channels=opt.num_timesteps_in, out_channels=opt.num_timesteps_out,
                                 num_nodes=num_nodes, out_list=out_list, periods=opt.num_timesteps_in, 
                                 depth=1, mlp_dim=128, device=device, is_rn=opt.is_rn, model_name=opt.model_name, 
                                 model_loc=model_loc, model_rn=model_rn).to(device)
        # for param in model_rn.rngcn.parameters():
        #     param.requires_grad = False
        # for param in model_loc.parameters():
        #     param.requires_grad = False
        # for param in model_loc.tpcnn_output.parameters():
        #     param.requires_grad = True

    # Check if pretrained model exists
    if opt.is_pretrained:
        if opt.is_rn:
            model.load_state_dict(torch.load(osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_model_grid{}_epoch{}.pt'.format(opt.uid, opt.grid, opt.pretrained_epoch))))
        else:
            model.load_state_dict(torch.load(osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_model_{}.pt'.format(opt.uid, opt.pretrained_epoch))))

    ### Optimization
    if opt.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    elif opt.optimizer == "SGD":
        if opt.model_name == "trajectory_model":
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        elif opt.model_name == "social_stgcnn":
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)
            # optimizer = torch.optim.SGD([
            #     {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': 1e-2, 'weight_decay': 1e-3},
            #     # {'params': model.model_rn.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3}
            # ])
        elif opt.model_name == "social_implicit":
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    elif opt.optimizer == "RMSProps":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-3)

    if opt.is_rn:
        optimizer_rn = torch.optim.RMSprop(model.model_rn.parameters(), lr=1e-3, weight_decay=1e-3)

    total_param = 0
    for param_tensor in model.state_dict():
        # logger.info("{}, {}".format(param_tensor, model.state_dict()[param_tensor].size()))
    #     print("{}, {}".format(param_tensor, model.state_dict()[param_tensor].size()))
        total_param += np.prod(model.state_dict()[param_tensor].size())
    #     print(model.state_dict()[param_tensor].size())
    # logger.info('Net\'s total params: {}' % total_param)
    print('Net\'s total params:', total_param)

    for epoch in range(opt.epochs + 1):

        if opt.is_rn:
            rn_val_loss, traj_val_loss, val_loss, loc_loss = valid_peds(epoch, val_loader, val_rn_loader)
        else:
            rn_val_loss, traj_val_loss, val_loss, loc_loss = valid_peds(epoch, val_loader)

        if writer_flg:
            writer.add_scalar("Valid Loss/Road Network", rn_val_loss, epoch)
            writer.add_scalar("Valid Loss/Local Network", loc_loss, epoch)
            writer.add_scalar("Valid Loss/Trajectory Network", traj_val_loss, epoch)
            writer.add_scalar("Valid Loss/Whole Network", val_loss, epoch)

        # logger.info('Epoch {}'.format(epoch))

        # logger.info("Valid Loss/Road Network: {}".format(rn_val_loss))
        # logger.info("Valid Loss/Local Network: {}".format(traj_val_loss))
        # logger.info("Valid Loss/Whole Network: {}".format(val_loss))

        # Save the model
        if epoch % 5 == 0:
            if opt.is_rn:
                # torch.save(model_rn.state_dict(), osp.join(opt.pretrained_dir, 'road_network', 'model_grid{}_outlist{}_epoch{}.pt'.format(opt.grid, out_list[0], epoch)))
                torch.save(model.state_dict(), osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_model_grid{}_epoch{}.pt'.format(opt.uid, opt.grid, opt.pretrained_epoch + epoch)))
            else:
                torch.save(model.state_dict(), osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, '{}_model_{}.pt'.format(opt.uid, opt.pretrained_epoch + epoch)))
            with open(osp.join(opt.pretrained_dir, opt.model_name, opt.dataset, 'constant_metrics.pkl'), 'wb') as fp:
                pickle.dump(constant_metrics, fp)


if __name__ == '__main__':
    main()